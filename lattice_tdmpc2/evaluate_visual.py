"""Visual evaluation of Lattice + TD-MPC2 + residual SAC in MetaDrive."""

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
from omegaconf import DictConfig, OmegaConf, open_dict

from common.parser import parse_cfg
from env import make_env
from lattice.frenet_metadrive import FRENET_DEFAULT_CONFIG, MetaDriveFrenetController
from lattice_tdmpc2.planner import ResidualSACLatticePlanner, WORLD_MODEL_MODES
from lattice_tdmpc2.trainer import (
    load_tdmpc2,
    make_sac,
    seed_everything,
    validate_sac_action_mapping,
)
from lattice_tdmpc2.versions import training_semantics
from lattice_tdmpc2.visualization import (
    MainCameraTrajectoryOverlay,
    draw_refinement_overlay,
)


RANDOM_SCENARIO_COUNT = 20
RANDOM_SCENARIO_SEED_LIMIT = 10_000


def _sample_scenarios(cfg) -> np.ndarray:
    simulator = cfg.metadrive["simulator"]
    training_start = int(simulator["start_seed"])
    training_count = int(simulator["num_scenarios"])
    training_end = training_start + training_count
    candidates = np.arange(RANDOM_SCENARIO_SEED_LIMIT, dtype=np.int64)
    candidates = candidates[(candidates < training_start) | (candidates >= training_end)]
    if len(candidates) < RANDOM_SCENARIO_COUNT:
        raise ValueError("The random scenario pool is too small after excluding training seeds.")
    # Deliberately use OS entropy so every visual evaluation samples new roads.
    return np.random.default_rng().choice(
        candidates, size=RANDOM_SCENARIO_COUNT, replace=False
    )


def _resolve_checkpoints(raw_cfg: DictConfig):
    sac_value = raw_cfg.residual_rl.get("sac_checkpoint")
    if not sac_value:
        raise ValueError(
            "Set residual_rl.sac_checkpoint to a SAC-Lagrangian checkpoint trained "
            "with the symmetric_coarse_speed_v2 action mapping."
        )
    sac_checkpoint = Path(sac_value).expanduser().resolve()
    if not sac_checkpoint.is_file():
        raise FileNotFoundError(f"Residual SAC checkpoint not found: {sac_checkpoint}")

    state = torch.load(sac_checkpoint, map_location="cpu", weights_only=False)
    validate_sac_action_mapping(state, sac_checkpoint)
    saved_config = state.get("config")
    if not saved_config:
        raise ValueError(f"SAC checkpoint has no saved training config: {sac_checkpoint}")

    explicit_wm = raw_cfg.residual_rl.get("tdmpc2_checkpoint")
    paired_wm = sac_checkpoint.with_name(f"world_model_{sac_checkpoint.name}")
    wm_value = explicit_wm or (paired_wm if paired_wm.is_file() else state.get("tdmpc2_checkpoint_path"))
    if not wm_value:
        raise ValueError("No paired TD-MPC2 checkpoint was found; set residual_rl.tdmpc2_checkpoint.")
    wm_checkpoint = Path(wm_value).expanduser().resolve()
    if not wm_checkpoint.is_file():
        raise FileNotFoundError(f"TD-MPC2 checkpoint not found: {wm_checkpoint}")
    return sac_checkpoint, wm_checkpoint, state, OmegaConf.create(saved_config)


def _visual_config(raw_cfg: DictConfig, saved_cfg: DictConfig, sac_checkpoint: Path, wm_checkpoint: Path):
    cfg = OmegaConf.create(OmegaConf.to_container(saved_cfg, resolve=True))
    with open_dict(cfg):
        saved_residual = cfg.get("residual_rl", {})
        cfg.algorithm_version = cfg.get("algorithm_version") or (
            "lattice_tdmpc2_v2"
            if "lagrangian_pid" in saved_residual and "lagrangian_lr" not in saved_residual
            else "lattice_tdmpc2_v3"
        )
        # Evaluation controls come from this invocation; model/environment settings
        # otherwise stay exactly as recorded by the training checkpoint.
        cfg.seed = int(raw_cfg.seed)
        cfg.eval_episodes = RANDOM_SCENARIO_COUNT
        cfg.residual_rl.device = str(raw_cfg.residual_rl.device)
        cfg.residual_rl.sac_checkpoint = str(sac_checkpoint)
        cfg.residual_rl.tdmpc2_checkpoint = str(wm_checkpoint)
        cfg.residual_rl.freeze_tdmpc2 = True
        cfg.residual_rl.safe_fallback = bool(raw_cfg.residual_rl.safe_fallback)

        cfg.compile = False
        cfg.flow.enabled = False
        cfg.self_improve.enabled = False
        cfg.enable_wandb = False
        cfg.save_agent = False
        cfg.save_csv = False
        cfg.save_video = False

        simulator = cfg.metadrive.simulator
        simulator.use_render = True
        simulator.image_observation = False
        simulator.manual_control = False
        simulator.show_interface = True
        simulator.force_render_fps = 30
        simulator.traffic_density = float(raw_cfg.metadrive.simulator.traffic_density)
        simulator.accident_prob = float(raw_cfg.metadrive.simulator.accident_prob)
    return parse_cfg(cfg)


def _render_views(
    env,
    overlay: MainCameraTrajectoryOverlay,
    result,
    *,
    episode: int,
    total_episodes: int,
    scenario: int,
    step: int,
    episode_return: float,
    episode_cost: float,
) -> None:
    raw_env = env.unwrapped_metadrive
    overlay.update(raw_env, result)
    text = {
        "episode": f"{episode}/{total_episodes}",
        "scenario": scenario,
        "step": step,
        "trajectory": "YELLOW=Lattice  RED=SAC refined",
        "coarse d/v": f"{result.coarse_d:+.2f} m / {result.coarse_v:.2f} m/s",
        "delta d/v": f"{result.delta_d:+.2f} m / {result.delta_v:+.2f} m/s",
        "output d/v": f"{result.refined_d:+.2f} m / {result.refined_v:.2f} m/s",
        "return": f"{episode_return:.2f}",
        "cost": f"{episode_cost:.2f}",
        "fallback": result.fallback_reason or "none",
    }
    raw_env.render(text=text)
    raw_env.render(
        text=text,
        mode="topdown",
        window=True,
        screen_record=False,
        screen_size=(720, 720),
        film_size=(1200, 1200),
        scaling=5,
        num_stack=15,
        history_smooth=1,
        target_agent_heading_up=True,
        draw_target_vehicle_trajectory=True,
    )
    draw_refinement_overlay(raw_env, result)


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(raw_cfg: DictConfig) -> None:
    sac_checkpoint, wm_checkpoint, sac_state, saved_cfg = _resolve_checkpoints(raw_cfg)
    cfg = _visual_config(raw_cfg, saved_cfg, sac_checkpoint, wm_checkpoint)
    mode = str(cfg.planner["type"])
    if mode not in WORLD_MODEL_MODES:
        raise ValueError(f"Visual residual evaluation requires a TD-MPC2 planner mode, got {mode!r}.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TD-MPC2 visual evaluation.")
    seed_everything(int(cfg.seed))
    scenarios = _sample_scenarios(cfg)
    cfg.metadrive["simulator"]["start_seed"] = 0
    cfg.metadrive["simulator"]["num_scenarios"] = RANDOM_SCENARIO_SEED_LIMIT

    print(f"Residual SAC checkpoint: {sac_checkpoint}")
    print(f"TD-MPC2 checkpoint: {wm_checkpoint}")
    print(f"Training environment step: {sac_state.get('environment_step', 'unknown')}")
    print(f"Random scenarios outside the training seed range: {scenarios.tolist()}")
    print(
        "Test difficulty: "
        f"traffic_density={cfg.metadrive['simulator']['traffic_density']}, "
        f"accident_prob={cfg.metadrive['simulator']['accident_prob']}"
    )
    print("Two windows will open; nothing is recorded or saved.")
    print("Trajectory colors: YELLOW=Lattice nominal, RED=SAC-refined.")
    print(
        "Refined-trajectory safety validation: "
        f"{'enabled' if cfg.residual_rl['safe_fallback'] else 'disabled (execute directly)'}"
    )

    env = make_env(cfg)
    overlay = MainCameraTrajectoryOverlay()
    summaries = []
    try:
        tdmpc_agent = load_tdmpc2(cfg, mode)
        semantics = training_semantics(cfg)
        lattice_config = dict(FRENET_DEFAULT_CONFIG)
        lattice_config.update(dict(cfg.lattice))
        lattice_config["_tracking_target_speed"] = semantics.tracking_target_speed
        planner = ResidualSACLatticePlanner(
            cfg,
            MetaDriveFrenetController(lattice_config),
            tdmpc_agent=tdmpc_agent,
            action_space=env.action_space,
        )
        planner.freeze_world_model()

        first_observation = env.reset(seed=int(scenarios[0]))
        planner.reset()
        first_context = planner.prepare(first_observation, env.unwrapped_metadrive.agent)
        saved_residual_cfg = (sac_state.get("config") or {}).get("residual_rl", {})
        sac_agent = make_sac(
            cfg,
            int(first_context.residual_state.numel()),
            torch.device(str(cfg.residual_rl["device"])),
            mode,
            hidden_sizes=saved_residual_cfg.get("hidden_sizes", cfg.residual_rl["hidden_sizes"]),
        )
        sac_agent.load(sac_checkpoint, load_optimizers=False)

        for episode_index, scenario in enumerate(scenarios, start=1):
            observation = env.reset(seed=int(scenario))
            planner.reset()
            context = planner.prepare(observation, env.unwrapped_metadrive.agent)
            done = False
            step = 0
            episode_return = 0.0
            episode_cost = 0.0
            info = {}

            while not done:
                started = time.perf_counter()
                normalized_action = sac_agent.act(
                    context.residual_state.numpy(), deterministic=True
                )
                sac_ms = 1000.0 * (time.perf_counter() - started)
                result = planner.refine(context, normalized_action, env.unwrapped_metadrive.agent)
                result.metrics["time/sac_ms"] = sac_ms
                result.metrics["time/total_ms"] += sac_ms

                _render_views(
                    env,
                    overlay,
                    result,
                    episode=episode_index,
                    total_episodes=len(scenarios),
                    scenario=int(scenario),
                    step=step,
                    episode_return=episode_return,
                    episode_cost=episode_cost,
                )
                observation, reward, done, info = env.step(result.low_level_action)
                step += 1
                episode_return += float(reward)
                episode_cost += float(info.get("cost", 0.0))
                if not done:
                    context = planner.prepare(observation, env.unwrapped_metadrive.agent)

            summary = {
                "scenario": int(scenario),
                "return": episode_return,
                "cost": episode_cost,
                "success": float(info.get("success", 0.0)),
                "completion": float(info.get("route_completion", 0.0)),
                "length": step,
                "reason": str(info.get("end_reason", "unknown")),
            }
            summaries.append(summary)
            print(
                f"[{episode_index:02d}/{len(scenarios):02d}] seed={summary['scenario']} "
                f"R={summary['return']:.2f} C={summary['cost']:.2f} "
                f"S={summary['success']:.0f} RC={summary['completion']:.3f} "
                f"L={summary['length']} reason={summary['reason']}"
            )
    except (KeyboardInterrupt, SystemExit):
        print("Visual evaluation stopped by user.")
    finally:
        overlay.clear()
        env.close()

    if summaries:
        print(
            "Summary: "
            f"episodes={len(summaries)} "
            f"return={np.mean([row['return'] for row in summaries]):.3f} "
            f"cost={np.mean([row['cost'] for row in summaries]):.3f} "
            f"success_rate={np.mean([row['success'] for row in summaries]):.3f} "
            f"route_completion={np.mean([row['completion'] for row in summaries]):.3f}"
        )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
