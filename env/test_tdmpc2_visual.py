"""Interactively evaluate a TD-MPC2 checkpoint with main-camera and BEV views."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
UPSTREAM_CODE_ROOT = REPO_ROOT / "tdmpc2"
for path in (REPO_ROOT, UPSTREAM_CODE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, open_dict
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from env import make_env as make_metadrive_env
from tdmpc2 import TDMPC2


DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "logs/metadrive-risk/1/tdmpc2_metadrive_risk/models/final.pt"
)


def _configure_visual_evaluation(raw_cfg: DictConfig) -> None:
    """Force interactive evaluation settings without enabling any recording."""
    with open_dict(raw_cfg):
        raw_cfg.compile = False
        raw_cfg.flow.enabled = False
        raw_cfg.planner.type = "mppi"
        raw_cfg.planner.final_cost_lambda = 0.0
        raw_cfg.cost.enabled = False
        raw_cfg.guidance.use_cost = False
        raw_cfg.self_improve.enabled = False
        raw_cfg.enable_wandb = False
        raw_cfg.save_agent = False
        raw_cfg.save_csv = False
        raw_cfg.save_video = False

        simulator = raw_cfg.metadrive.simulator
        simulator.use_render = True
        simulator.image_observation = False
        simulator.show_interface = True
        simulator.force_render_fps = 30


def _sample_scenarios(cfg) -> np.ndarray:
    start_seed = int(cfg.metadrive["simulator"]["start_seed"])
    num_scenarios = int(cfg.metadrive["simulator"]["num_scenarios"])
    num_episodes = int(cfg.eval_episodes)
    if num_episodes <= 0:
        raise ValueError("eval_episodes must be positive.")
    if num_episodes > num_scenarios:
        raise ValueError(
            "eval_episodes cannot exceed metadrive.simulator.num_scenarios "
            "when sampling without replacement."
        )
    rng = np.random.default_rng(int(cfg.seed))
    candidates = np.arange(start_seed, start_seed + num_scenarios)
    return rng.choice(candidates, size=num_episodes, replace=False)


def _render_views(
    env,
    *,
    episode: int,
    total_episodes: int,
    scenario: int,
    step: int,
    episode_return: float,
    episode_cost: float,
    model_action: tuple[float, float] | None = None,
    applied_action: tuple[float, float] | None = None,
) -> None:
    raw_env = env.unwrapped_metadrive
    model_text = "--" if model_action is None else f"{model_action[0]:+.3f}, {model_action[1]:+.3f}"
    applied_text = "--" if applied_action is None else f"{applied_action[0]:+.3f}, {applied_action[1]:+.3f}"
    text = {
        "episode": f"{episode}/{total_episodes}",
        "scenario": scenario,
        "step": step,
        "model [steer,pedal]": model_text,
        "applied [steer,pedal]": applied_text,
        "return": f"{episode_return:.2f}",
        "cost": f"{episode_cost:.2f}",
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


def _control_values(
    env, action: torch.Tensor
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], float]:
    model_array = action.detach().cpu().numpy().astype(np.float64).reshape(-1)
    if model_array.size != 2:
        raise ValueError(f"Expected a two-dimensional action, got shape {tuple(action.shape)}.")
    model_action = (float(model_array[0]), float(model_array[1]))

    vehicle = env.unwrapped_metadrive.agent
    current_action = np.asarray(vehicle.current_action, dtype=np.float64).reshape(-1)
    if current_action.size == 2:
        applied_action = (float(current_action[0]), float(current_action[1]))
    else:
        applied_action = (
            float(getattr(vehicle, "steering", np.nan)),
            float(getattr(vehicle, "throttle_brake", np.nan)),
        )
    vehicle_control = (
        float(getattr(vehicle, "steering", np.nan)),
        float(getattr(vehicle, "throttle_brake", np.nan)),
    )
    speed_km_h = float(getattr(vehicle, "speed_km_h", np.nan))
    return model_action, applied_action, vehicle_control, speed_km_h


@hydra.main(version_base="1.3", config_path="../tdmpc2_fm", config_name="config")
def main(raw_cfg: DictConfig) -> None:
    _configure_visual_evaluation(raw_cfg)
    if not raw_cfg.checkpoint:
        raw_cfg.checkpoint = str(DEFAULT_CHECKPOINT)
    cfg = parse_cfg(raw_cfg)
    set_seed(int(cfg.seed))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run the TD-MPC2 agent.")

    checkpoint = Path(cfg.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"TD-MPC2 checkpoint not found: {checkpoint}")

    scenarios = _sample_scenarios(cfg)
    print(colored(f"Checkpoint: {checkpoint}", "cyan", attrs=["bold"]))
    print(colored(f"Random scenarios: {scenarios.tolist()}", "cyan"))
    print("MetaDrive main-camera and BEV windows will open; no output will be saved.")
    print("Action convention: steering in [-1, 1]; pedal > 0 is throttle, pedal < 0 is brake.")

    env = make_metadrive_env(cfg)
    results: list[dict[str, float | int | str]] = []
    try:
        agent = TDMPC2(cfg)
        agent.load(str(checkpoint))
        agent.eval()

        for episode_index, scenario_seed in enumerate(scenarios, start=1):
            obs = env.reset(seed=int(scenario_seed))
            done = False
            step = 0
            episode_return = 0.0
            episode_cost = 0.0
            info = {}
            _render_views(
                env,
                episode=episode_index,
                total_episodes=len(scenarios),
                scenario=int(scenario_seed),
                step=step,
                episode_return=episode_return,
                episode_cost=episode_cost,
            )

            while not done:
                action = agent.act(obs, t0=(step == 0), eval_mode=True)
                obs, reward, done, info = env.step(action)
                step += 1
                episode_return += float(reward)
                episode_cost += float(info.get("cost", 0.0))
                model_action, applied_action, vehicle_control, speed_km_h = _control_values(env, action)
                print(
                    f"[control] ep={episode_index:02d} seed={int(scenario_seed)} step={step:04d} "
                    f"model_steer={model_action[0]:+.6f} "
                    f"model_pedal={model_action[1]:+.6f} "
                    f"applied_steer={applied_action[0]:+.6f} "
                    f"applied_pedal={applied_action[1]:+.6f} "
                    f"vehicle_steer={vehicle_control[0]:+.6f} "
                    f"vehicle_pedal={vehicle_control[1]:+.6f} "
                    f"speed_km_h={speed_km_h:.3f}",
                    flush=True,
                )
                _render_views(
                    env,
                    episode=episode_index,
                    total_episodes=len(scenarios),
                    scenario=int(scenario_seed),
                    step=step,
                    episode_return=episode_return,
                    episode_cost=episode_cost,
                    model_action=model_action,
                    applied_action=applied_action,
                )

            result = {
                "scenario": int(scenario_seed),
                "return": episode_return,
                "cost": episode_cost,
                "success": float(info.get("success", 0.0)),
                "length": step,
                "reason": str(info.get("end_reason", "unknown")),
            }
            results.append(result)
            print(
                f"[{episode_index:02d}/{len(scenarios):02d}] "
                f"seed={result['scenario']} R={result['return']:.2f} "
                f"C={result['cost']:.2f} S={result['success']:.0f} "
                f"L={result['length']} reason={result['reason']}"
            )
    except (KeyboardInterrupt, SystemExit):
        print(colored("Evaluation stopped by user.", "yellow"))
    finally:
        env.close()

    if results:
        print(colored("Evaluation summary", "green", attrs=["bold"]))
        print(f"episodes: {len(results)}")
        print(f"mean return: {np.mean([r['return'] for r in results]):.2f}")
        print(f"mean cost: {np.mean([r['cost'] for r in results]):.2f}")
        print(f"success rate: {np.mean([r['success'] for r in results]):.3f}")
        print(f"mean length: {np.mean([r['length'] for r in results]):.1f}")


if __name__ == "__main__":
    main()
