"""Train residual SAC with real MetaDrive TD targets and optional TD-MPC2 guidance."""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path


INTEGRATION_DIR = Path(__file__).resolve().parent
REPO_ROOT = INTEGRATION_DIR.parent
UPSTREAM_ROOT = REPO_ROOT / "tdmpc2"
for path in (REPO_ROOT, UPSTREAM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from common.parser import parse_cfg
from env import make_env
from lattice.frenet_metadrive import FRENET_DEFAULT_CONFIG, MetaDriveFrenetController
from sac.model import SACAgent
from sac.replay_buffer import ReplayBuffer
try:
    from tdmpc2 import TDMPC2
except ImportError:  # Package-style import from the repository root.
    from tdmpc2.tdmpc2 import TDMPC2

from lattice_tdmpc2.planner import ResidualSACLatticePlanner, WORLD_MODEL_MODES
from lattice_tdmpc2.replay_adapter import OnlineWorldModelReplayAdapter
from lattice_tdmpc2.residual_action import SUPPORTED_ACTION_MAPPINGS
from lattice_tdmpc2.versions import V4, sac_agent_class, training_semantics, version_from_cfg


VALID_TRAINING_MODES = {"sac_residual_frozen_wm", "sac_residual_online_wm"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint_has_cost(path: Path) -> bool:
    state = torch.load(path, map_location="cpu", weights_only=False)
    state = state.get("model", state)
    return any(key.startswith("_cost.") for key in state)


def checkpoint_has_cost_q(path: Path) -> bool:
    state = torch.load(path, map_location="cpu", weights_only=False)
    state = state.get("model", state)
    return any(key.startswith("_cost_Qs.") for key in state)


def load_tdmpc2(cfg, planner_mode: str):
    if planner_mode not in WORLD_MODEL_MODES:
        return None
    if not torch.cuda.is_available():
        raise RuntimeError("TD-MPC2 residual modes require CUDA; use planner.type=lattice_sac for CPU tests.")
    value = cfg.residual_rl.get("tdmpc2_checkpoint") or cfg.checkpoint
    if not value:
        raise ValueError("Set residual_rl.tdmpc2_checkpoint=/absolute/path/to/final.pt.")
    checkpoint = Path(value).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"TD-MPC2 checkpoint not found: {checkpoint}")
    if bool(cfg.residual_rl.get("use_world_model_cost", False)) and not checkpoint_has_cost(checkpoint):
        raise RuntimeError("use_world_model_cost=true, but the TD-MPC2 checkpoint has no trained cost head.")
    if bool(cfg.residual_rl.get("use_terminal_cost_q", False)) and not checkpoint_has_cost_q(checkpoint):
        raise RuntimeError(
            "use_terminal_cost_q=true, but the TD-MPC2 checkpoint has no trained cost-Q head."
        )
    agent = TDMPC2(cfg)
    agent.load(str(checkpoint))
    agent.eval()
    return agent


def make_sac(cfg, state_dim: int, device, planner_mode: str, hidden_sizes=None) -> SACAgent:
    del planner_mode
    residual = cfg.residual_rl
    agent_class = sac_agent_class(cfg)
    kwargs = {}
    if agent_class.algorithm_version == "lattice_tdmpc2_v2":
        kwargs.update(
            lagrangian_pid=tuple(float(value) for value in residual.get("lagrangian_pid", (0.05, 0.0005, 0.1))),
            trust_loss_weight=float(residual.get("trust_loss_weight", 0.01)),
            residual_sigma=(float(residual["sigma_d"]), float(residual["sigma_v"])),
        )
    else:
        kwargs["lagrangian_lr"] = float(residual.get("lagrangian_lr", 0.01))
    return agent_class(
        state_dim, 2, device=device,
        hidden_sizes=residual["hidden_sizes"] if hidden_sizes is None else hidden_sizes,
        actor_lr=float(residual["actor_lr"]), critic_lr=float(residual["critic_lr"]),
        alpha_lr=float(residual["alpha_lr"]), alpha=float(residual["alpha"]),
        auto_alpha=bool(residual["auto_alpha"]), tau=float(residual["tau"]),
        use_lagrangian=bool(residual["use_lagrangian"]),
        cost_limit=float(residual["cost_limit"]),
        lagrangian_rescaling=bool(residual["lagrangian_rescaling"]),
        **kwargs,
    )


def validate_sac_action_mapping(
    checkpoint_state: dict, checkpoint: Path, expected_mapping: str | None = None
) -> None:
    saved_config = checkpoint_state.get("config") or {}
    saved_mapping = saved_config.get("residual_rl", {}).get("action_mapping")
    if saved_mapping not in SUPPORTED_ACTION_MAPPINGS:
        raise ValueError(
            f"Residual SAC checkpoint {checkpoint} uses the old or unknown action mapping. "
            f"Expected one of {sorted(SUPPORTED_ACTION_MAPPINGS)}; retrain SAC."
        )
    if expected_mapping is not None and saved_mapping != str(expected_mapping):
        raise ValueError(
            f"Residual SAC checkpoint {checkpoint} uses action mapping {saved_mapping!r}, "
            f"but this run requests {str(expected_mapping)!r}."
        )


def make_replay(cfg, state_dim: int) -> ReplayBuffer:
    residual = cfg.residual_rl
    semantics = training_semantics(cfg)
    return ReplayBuffer(
        int(residual["replay_size"]), state_dim, 2,
        gamma=float(residual["gamma"]), n_step=int(residual["n_step"]),
        extra_specs=semantics.replay_extra_specs,
    )


def wandb_run(raw_cfg):
    if not bool(raw_cfg.enable_wandb):
        return None
    import wandb

    name = raw_cfg.wandb_run_name or f"{raw_cfg.planner.type}_seed{raw_cfg.seed}"
    return wandb.init(
        project=str(raw_cfg.wandb_project), entity=raw_cfg.wandb_entity, name=str(name),
        config=OmegaConf.to_container(raw_cfg, resolve=True),
    )


def save_sac(agent, path, cfg, env_step: int) -> None:
    agent.save(
        path, OmegaConf.to_container(cfg, resolve=True), training_step=agent.update_steps,
        environment_step=env_step,
        tdmpc2_checkpoint_path=cfg.residual_rl.get("tdmpc2_checkpoint"),
    )


@torch.no_grad()
def evaluate_policy(env, planner, sac_agent, cfg) -> dict[str, float]:
    returns, costs, successes, completions, lengths = [], [], [], [], []
    start = int(cfg.metadrive["simulator"]["start_seed"])
    count = int(cfg.metadrive["simulator"]["num_scenarios"])
    for episode in range(int(cfg.residual_rl["eval_episodes"])):
        observation = env.reset(seed=start + episode % count)
        planner.reset()
        context = planner.prepare(observation, env.unwrapped_metadrive.agent)
        done = False
        episode_return = 0.0
        episode_cost = 0.0
        length = 0
        info = {}
        while not done:
            normalized_action = sac_agent.act(context.residual_state.numpy(), deterministic=True)
            result = planner.refine(context, normalized_action, env.unwrapped_metadrive.agent)
            observation, reward, done, info = env.step(result.low_level_action)
            episode_return += float(reward)
            episode_cost += float(info.get("cost", 0.0))
            length += 1
            if not done:
                context = planner.prepare(observation, env.unwrapped_metadrive.agent)
        returns.append(episode_return)
        costs.append(episode_cost)
        successes.append(float(info.get("success", 0.0)))
        completions.append(float(info.get("route_completion", 0.0)))
        lengths.append(length)
    return {
        "eval/return": float(np.mean(returns)), "eval/cost": float(np.mean(costs)),
        "eval/success_rate": float(np.mean(successes)),
        "eval/route_completion": float(np.mean(completions)), "eval/episode_length": float(np.mean(lengths)),
    }


def run_training(raw_cfg: DictConfig) -> None:
    planner_mode = str(raw_cfg.planner.type)
    if planner_mode == "lattice":
        raise ValueError("The lattice baseline has no SAC parameters to train; run lattice_tdmpc2/evaluate.py.")
    training_mode = str(raw_cfg.residual_rl.training_mode)
    if training_mode not in VALID_TRAINING_MODES:
        raise ValueError(f"Unknown residual_rl.training_mode={training_mode!r}.")
    if training_mode == "sac_residual_online_wm" and planner_mode not in WORLD_MODEL_MODES:
        raise ValueError("sac_residual_online_wm requires a planner mode that uses TD-MPC2.")
    if training_mode == "sac_residual_online_wm" and bool(raw_cfg.residual_rl.freeze_tdmpc2):
        raise ValueError("Online world-model training requires residual_rl.freeze_tdmpc2=false.")
    raw_cfg.compile = False
    cfg = parse_cfg(raw_cfg)
    seed_everything(int(cfg.seed))
    semantics = training_semantics(cfg)
    algorithm_version = version_from_cfg(cfg)
    if algorithm_version == V4:
        if planner_mode not in WORLD_MODEL_MODES:
            raise ValueError("V4 requires planner.type=lattice_tdmpc_sac.")
        if training_mode != "sac_residual_frozen_wm" or not bool(cfg.residual_rl["freeze_tdmpc2"]):
            raise ValueError("V4 requires a frozen world model.")
        if not bool(cfg.residual_rl.get("use_world_model_cost", False)):
            raise ValueError("V4 requires residual_rl.use_world_model_cost=true.")
        if not bool(cfg.residual_rl.get("use_terminal_cost_q", False)):
            raise ValueError("V4 requires residual_rl.use_terminal_cost_q=true.")

    requested_device = str(cfg.residual_rl["device"])
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Residual SAC device=cuda, but CUDA is unavailable. Use device=cpu for lattice_sac tests.")
    sac_device = torch.device(requested_device)
    env = make_env(cfg)
    run = None
    try:
        tdmpc_agent = load_tdmpc2(cfg, planner_mode)
        lattice_config = dict(FRENET_DEFAULT_CONFIG)
        lattice_config.update(dict(cfg.lattice))
        lattice_config["_tracking_target_speed"] = semantics.tracking_target_speed
        controller = MetaDriveFrenetController(lattice_config)
        planner = ResidualSACLatticePlanner(
            cfg, controller, tdmpc_agent=tdmpc_agent, action_space=env.action_space
        )
        if training_mode == "sac_residual_online_wm":
            planner.unfreeze_world_model()

        observation = env.reset(seed=int(cfg.seed))
        planner.reset()
        context = planner.prepare(observation, env.unwrapped_metadrive.agent)
        state_dim = int(context.residual_state.numel())
        checkpoint = cfg.residual_rl.get("sac_checkpoint")
        hidden_sizes = None
        if checkpoint:
            checkpoint = Path(checkpoint).expanduser().resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Residual SAC checkpoint not found: {checkpoint}")
            checkpoint_state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            validate_sac_action_mapping(
                checkpoint_state, checkpoint,
                expected_mapping=str(cfg.residual_rl["action_mapping"]),
            )
            checkpoint_cfg = checkpoint_state.get("config") or {}
            hidden_sizes = checkpoint_cfg.get("residual_rl", {}).get("hidden_sizes")
        sac_agent = make_sac(cfg, state_dim, sac_device, planner_mode, hidden_sizes=hidden_sizes)
        if checkpoint:
            sac_agent.load(checkpoint, load_optimizers=True)
        replay = make_replay(cfg, state_dim)
        online_wm = (
            OnlineWorldModelReplayAdapter(cfg, tdmpc_agent)
            if training_mode == "sac_residual_online_wm" else None
        )
        if online_wm is not None:
            online_wm.start_episode(observation)
        run = wandb_run(raw_cfg)
        name = raw_cfg.wandb_run_name or f"{planner_mode}_seed{cfg.seed}"
        output_dir = Path(hydra.utils.get_original_cwd()) / str(cfg.residual_rl["output_dir"]) / str(name)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"Algorithm={version_from_cfg(cfg)}, planner={planner_mode}, "
            f"residual_state_dim={state_dim}, SAC action_dim=2"
        )

        episode_return = episode_cost = episode_constraint_cost = 0.0
        episode_length = episode_index = 0
        update_budget = 0.0
        last_sac_metrics = {}
        fallback_count = 0
        best_return = -float("inf")

        for env_step in range(1, int(cfg.steps) + 1):
            sac_start = time.perf_counter()
            if env_step <= int(cfg.residual_rl["seed_steps"]):
                normalized_action = np.random.uniform(-1.0, 1.0, size=2).astype(np.float32)
            else:
                normalized_action = sac_agent.act(context.residual_state.numpy(), deterministic=False)
            sac_ms = 1000.0 * (time.perf_counter() - sac_start)
            result = planner.refine(context, normalized_action, env.unwrapped_metadrive.agent)
            result.metrics["time/sac_ms"] = sac_ms
            result.metrics["time/total_ms"] += sac_ms
            next_observation, reward, done, info = env.step(result.low_level_action)
            info.update(result.info)
            real_cost = float(info.get("cost", 0.0))
            replay_cost, replay_extras = semantics.transition(real_cost, result)
            fallback_count += int(not result.residual_valid)
            if done:
                next_state = np.zeros(state_dim, dtype=np.float32)
                next_context = None
            else:
                next_context = planner.prepare(next_observation, env.unwrapped_metadrive.agent)
                next_state = next_context.residual_state.numpy()
            replay.add(
                context.residual_state.numpy(), result.normalized_residual_action, float(reward),
                replay_cost, next_state, done, extras=replay_extras,
            )
            if online_wm is not None:
                online_wm.add(
                    next_observation, result.low_level_action, reward,
                    bool(np.asarray(info.get("terminated", done)).item()), real_cost, done,
                )

            episode_return += float(reward)
            episode_cost += real_cost
            episode_constraint_cost += replay_cost
            episode_length += 1
            step_metrics = {
                **result.metrics,
                "env/reward": float(reward), "env/cost": real_cost,
                "env/constraint_cost": replay_cost,
                "env/world_model_constraint_cost": (
                    replay_cost if algorithm_version == V4 else 0.0
                ),
                "env/safety_risk_cost": float(info.get("safety_risk_cost", 0.0)),
                "env/safety_event_cost": float(info.get("safety_event_cost", 0.0)),
                "env/success": float(info.get("success", 0.0)),
                "env/route_completion": float(info.get("route_completion", 0.0)),
                "env/collision": float(info.get("crash", 0.0)),
                "env/offroad": float(info.get("out_of_road", 0.0)),
                "residual/unsafe": float(result.unsafe_residual),
                "control/steering": float(result.low_level_action[0]),
                "control/throttle_brake": float(result.low_level_action[1]),
                "residual/fallback_reason": result.fallback_reason or "none",
                "residual/fallback_rate_running": fallback_count / env_step,
            }

            if env_step >= int(cfg.residual_rl["learning_starts"]) and len(replay) >= int(cfg.residual_rl["batch_size"]):
                update_budget += float(cfg.residual_rl["update_per_step"])
                while update_budget >= 1.0:
                    last_sac_metrics = sac_agent.update(
                        replay.sample(int(cfg.residual_rl["batch_size"]), sac_device)
                    )
                    update_budget -= 1.0
            wm_metrics = {}
            if online_wm is not None and env_step % int(cfg.residual_rl["online_wm_update_every"]) == 0:
                wm_metrics = online_wm.update()

            if done:
                lagrangian_episode_cost = semantics.lagrangian_episode_cost(
                    episode_cost, episode_constraint_cost, episode_length
                )
                lagrangian_value = sac_agent.update_lagrangian(lagrangian_episode_cost)
                episode_metrics = {
                    "env/episode_return": episode_return, "env/episode_cost": episode_cost,
                    "env/episode_constraint_cost_sum": episode_constraint_cost,
                    "lagrangian/episode_cost": lagrangian_episode_cost,
                    "lagrangian/value": lagrangian_value,
                    "lagrangian/cost_limit": sac_agent.cost_limit,
                    "env/episode_length": episode_length, "env/episode": episode_index,
                    "env/episode_success": float(info.get("success", 0.0)),
                }
                if hasattr(sac_agent.lagrangian, "learning_rate"):
                    episode_metrics["lagrangian/lr"] = sac_agent.lagrangian.learning_rate
                if hasattr(sac_agent.lagrangian, "pid"):
                    episode_metrics["lagrangian/pid_kp"] = sac_agent.lagrangian.pid[0]
                    episode_metrics["lagrangian/pid_ki"] = sac_agent.lagrangian.pid[1]
                    episode_metrics["lagrangian/pid_kd"] = sac_agent.lagrangian.pid[2]
                print(
                    f"train E={episode_index} step={env_step} R={episode_return:.2f} "
                    f"C={episode_cost:.2f} C_lag={lagrangian_episode_cost:.2f} "
                    f"lambda={lagrangian_value:.4f} L={episode_length} "
                    f"fallback={fallback_count / env_step:.3f}"
                )
                if run is not None:
                    run.log(episode_metrics, step=env_step)
                observation = env.reset()
                planner.reset()
                context = planner.prepare(observation, env.unwrapped_metadrive.agent)
                if online_wm is not None:
                    online_wm.start_episode(observation)
                episode_return = episode_cost = episode_constraint_cost = 0.0
                episode_length = 0
                episode_index += 1
            else:
                observation = next_observation
                context = next_context

            if env_step % int(cfg.residual_rl["log_every"]) == 0 and run is not None:
                run.log({**step_metrics, **last_sac_metrics, **wm_metrics}, step=env_step)
            if env_step % int(cfg.residual_rl["checkpoint_every"]) == 0:
                save_sac(sac_agent, output_dir / f"step_{env_step}.pt", raw_cfg, env_step)
                if online_wm is not None:
                    tdmpc_agent.save(str(output_dir / f"world_model_step_{env_step}.pt"))
            if env_step % int(cfg.residual_rl["eval_every"]) == 0:
                replay.flush()
                eval_metrics = evaluate_policy(env, planner, sac_agent, cfg)
                print(
                    f"eval step={env_step} R={eval_metrics['eval/return']:.2f} "
                    f"C={eval_metrics['eval/cost']:.2f} S={eval_metrics['eval/success_rate']:.3f}"
                )
                if run is not None:
                    run.log(eval_metrics, step=env_step)
                save_sac(sac_agent, output_dir / "latest.pt", raw_cfg, env_step)
                if online_wm is not None:
                    tdmpc_agent.save(str(output_dir / "world_model_latest.pt"))
                if eval_metrics["eval/return"] > best_return:
                    best_return = eval_metrics["eval/return"]
                    save_sac(sac_agent, output_dir / "best.pt", raw_cfg, env_step)
                    if online_wm is not None:
                        tdmpc_agent.save(str(output_dir / "world_model_best.pt"))
                observation = env.reset()
                planner.reset()
                context = planner.prepare(observation, env.unwrapped_metadrive.agent)
                episode_return = episode_cost = episode_constraint_cost = 0.0
                episode_length = 0
                if online_wm is not None:
                    online_wm.start_episode(observation)

        replay.flush()
        save_sac(sac_agent, output_dir / "final.pt", raw_cfg, int(cfg.steps))
        if online_wm is not None:
            tdmpc_agent.save(str(output_dir / "world_model_final.pt"))
        print(f"Residual SAC training complete: {output_dir / 'final.pt'}")
    finally:
        env.close()
        if run is not None:
            run.finish()


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(raw_cfg: DictConfig) -> None:
    """Backward-compatible direct entry point."""
    run_training(raw_cfg)


if __name__ == "__main__":
    main()
