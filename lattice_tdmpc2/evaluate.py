"""Evaluate Lattice and residual-SAC ablations in closed-loop MetaDrive."""

from __future__ import annotations

import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
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
from lattice_tdmpc2.planner import VALID_MODES, ResidualSACLatticePlanner
from lattice_tdmpc2.trainer import (
    load_tdmpc2,
    make_sac,
    seed_everything,
    validate_sac_action_mapping,
)
from lattice_tdmpc2.versions import training_semantics
from lattice_tdmpc2.visualization import draw_refinement_overlay


def _sample_scenarios(cfg) -> np.ndarray:
    simulator = cfg.metadrive["simulator"]
    start = int(simulator["start_seed"])
    count = int(simulator["num_scenarios"])
    episodes = int(cfg.eval_episodes)
    if episodes <= 0 or count <= 0:
        raise ValueError("eval_episodes and num_scenarios must be positive.")
    rng = np.random.default_rng(int(cfg.seed))
    return rng.choice(np.arange(start, start + count), size=episodes, replace=episodes > count)


def _wandb_run(raw_cfg, mode: str):
    if not bool(raw_cfg.enable_wandb):
        return None
    import wandb

    name = raw_cfg.wandb_run_name or f"eval_{mode}_seed{raw_cfg.seed}"
    return wandb.init(
        project=str(raw_cfg.wandb_project), entity=raw_cfg.wandb_entity, name=str(name),
        config=OmegaConf.to_container(raw_cfg, resolve=True),
    )


def _render(env, result, episode, scenario, step, episode_return, episode_cost) -> None:
    raw_env = env.unwrapped_metadrive
    text = {
        "episode": episode, "scenario": scenario, "step": step,
        "return": f"{episode_return:.2f}", "cost": f"{episode_cost:.2f}",
        "coarse d/v": f"{result.coarse_d:+.2f} / {result.coarse_v:.2f}",
        "delta d/v": f"{result.delta_d:+.2f} / {result.delta_v:+.2f}",
        "refined d/v": f"{result.refined_d:+.2f} / {result.refined_v:.2f}",
    }
    raw_env.render(
        text=text, mode="topdown", window=True, screen_record=False,
        screen_size=(720, 720), film_size=(1200, 1200), scaling=5,
        target_agent_heading_up=True, draw_target_vehicle_trajectory=True,
    )
    draw_refinement_overlay(raw_env, result)


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(raw_cfg: DictConfig) -> None:
    mode = str(raw_cfg.planner.type)
    if mode not in VALID_MODES:
        raise ValueError(f"Expected planner.type in {sorted(VALID_MODES)}, got {mode!r}.")
    raw_cfg.compile = False
    raw_cfg.flow.enabled = False
    raw_cfg.self_improve.enabled = False
    raw_cfg.save_agent = False
    raw_cfg.save_csv = False
    raw_cfg.save_video = False
    cfg = parse_cfg(raw_cfg)
    seed_everything(int(cfg.seed))

    requested_device = str(cfg.residual_rl["device"])
    if mode != "lattice" and requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Residual SAC device=cuda, but CUDA is unavailable.")
    env = make_env(cfg)
    run = None
    summaries = []
    all_metrics: list[dict[str, float]] = []
    scenarios = _sample_scenarios(cfg)
    try:
        tdmpc_agent = load_tdmpc2(cfg, mode)
        semantics = training_semantics(cfg)
        lattice_config = dict(FRENET_DEFAULT_CONFIG)
        lattice_config.update(dict(cfg.lattice))
        lattice_config["_tracking_target_speed"] = semantics.tracking_target_speed
        planner = ResidualSACLatticePlanner(
            cfg, MetaDriveFrenetController(lattice_config),
            tdmpc_agent=tdmpc_agent, action_space=env.action_space,
        )
        first_observation = env.reset(seed=int(scenarios[0]))
        planner.reset()
        first_context = planner.prepare(first_observation, env.unwrapped_metadrive.agent)
        sac_agent = None
        if planner.uses_sac:
            checkpoint_value = cfg.residual_rl.get("sac_checkpoint")
            if not checkpoint_value:
                raise ValueError("Set residual_rl.sac_checkpoint=/absolute/path/to/residual_sac.pt.")
            checkpoint = Path(checkpoint_value).expanduser().resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Residual SAC checkpoint not found: {checkpoint}")
            checkpoint_state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            validate_sac_action_mapping(checkpoint_state, checkpoint)
            checkpoint_cfg = checkpoint_state.get("config") or {}
            saved_residual_cfg = checkpoint_cfg.get("residual_rl", {})
            sac_agent = make_sac(
                cfg, int(first_context.residual_state.numel()), torch.device(requested_device), mode,
                hidden_sizes=saved_residual_cfg.get("hidden_sizes", cfg.residual_rl["hidden_sizes"]),
            )
            sac_agent.load(checkpoint, load_optimizers=False)
            print(f"Residual SAC checkpoint: {checkpoint}")
        run = _wandb_run(raw_cfg, mode)
        print(f"Planner: {mode}; random scenarios: {scenarios.tolist()}")

        for episode_index, scenario in enumerate(scenarios, start=1):
            observation = env.reset(seed=int(scenario))
            planner.reset()
            context = planner.prepare(observation, env.unwrapped_metadrive.agent)
            done = False
            episode_return = episode_cost = 0.0
            step = 0
            info = {}
            while not done:
                start = time.perf_counter()
                normalized_action = (
                    sac_agent.act(context.residual_state.numpy(), deterministic=True)
                    if sac_agent is not None else np.zeros(2, dtype=np.float32)
                )
                sac_ms = 1000.0 * (time.perf_counter() - start)
                result = planner.refine(context, normalized_action, env.unwrapped_metadrive.agent)
                result.metrics["time/sac_ms"] = sac_ms
                result.metrics["time/total_ms"] += sac_ms
                observation, reward, done, info = env.step(result.low_level_action)
                step += 1
                episode_return += float(reward)
                episode_cost += float(info.get("cost", 0.0))
                all_metrics.append(result.metrics)
                if step == 1:
                    print(
                        f"[plan] d0={result.coarse_d:+.3f} V0={result.coarse_v:.3f} "
                        f"delta=({result.delta_d:+.3f},{result.delta_v:+.3f}) "
                        f"refined=({result.refined_d:+.3f},{result.refined_v:.3f}) "
                        f"fallback={result.fallback_reason or 'none'}"
                    )
                if run is not None:
                    run.log({
                        **result.metrics, "eval/reward": float(reward),
                        "eval/cost": float(info.get("cost", 0.0)),
                        "residual/fallback_reason": result.fallback_reason or "none",
                    })
                if bool(cfg.metadrive["simulator"].get("use_render", False)):
                    _render(
                        env, result, episode_index, int(scenario), step,
                        episode_return, episode_cost,
                    )
                if not done:
                    context = planner.prepare(observation, env.unwrapped_metadrive.agent)

            row = {
                "scenario": int(scenario), "return": episode_return, "cost": episode_cost,
                "success": float(info.get("success", 0.0)),
                "route_completion": float(info.get("route_completion", 0.0)), "length": step,
                "reason": str(info.get("end_reason", "unknown")),
            }
            summaries.append(row)
            print(
                f"[{episode_index:02d}/{len(scenarios):02d}] seed={int(scenario)} "
                f"R={episode_return:.2f} C={episode_cost:.2f} S={row['success']:.0f} "
                f"L={step} reason={row['reason']}"
            )
    finally:
        env.close()
        if run is not None:
            run.finish()

    print("Evaluation summary")
    print(
        f"episodes={len(summaries)} return={np.mean([x['return'] for x in summaries]):.3f} "
        f"cost={np.mean([x['cost'] for x in summaries]):.3f} "
        f"success_rate={np.mean([x['success'] for x in summaries]):.3f} "
        f"route_completion={np.mean([x['route_completion'] for x in summaries]):.3f}"
    )
    print(
        f"fallback_rate={np.mean([x['residual/fallback_rate'] for x in all_metrics]):.3f} "
        f"wm_improvement={np.mean([x['residual/wm_improvement'] for x in all_metrics]):.6f} "
        f"planning_ms={np.mean([x['time/total_ms'] for x in all_metrics]):.3f}"
    )


if __name__ == "__main__":
    main()
