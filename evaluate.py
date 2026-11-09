"""Evaluate original TD-MPC2 or MPR-MPC on reproducible MetaDrive scenes."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from mpr_mpc._bootstrap import LOG_ROOT, bootstrap

bootstrap()

from common.parser import parse_cfg  # noqa: E402
from common.seed import set_seed  # noqa: E402
from env import make_env  # noqa: E402

from mpr_mpc.agent import MPRMPCAgent  # noqa: E402
from mpr_mpc.lattice.frenet_metadrive import (  # noqa: E402
    FRENET_DEFAULT_CONFIG,
    MetaDriveFrenetController,
)
from mpr_mpc.planning.coordinator import MPRMPCPlanner  # noqa: E402
from mpr_mpc.planning.visualization import draw_refinement_overlay  # noqa: E402
from mpr_mpc.tdmpc2.core import MPRTDMPC2  # noqa: E402
from mpr_mpc.evaluation_diagnostics import EpisodeDiagnostics, fixed_scenarios, write_report
from mpr_mpc.evaluate_model_suite import _apply_checkpoint_runtime_config


def _wandb_run(raw_cfg, cfg):
    if not bool(raw_cfg.enable_wandb):
        return None
    import wandb

    name = raw_cfg.wandb_run_name or f"eval_mpr_mpc_seed{raw_cfg.seed}"
    return wandb.init(
        project=str(raw_cfg.wandb_project),
        entity=raw_cfg.wandb_entity,
        name=str(name),
        config=OmegaConf.to_container(raw_cfg, resolve=True),
        job_type="evaluation",
        dir=str(cfg.work_dir),
    )


def _scenarios(cfg) -> np.ndarray:
    return fixed_scenarios(cfg)


def _render(env, result, episode: int, scenario: int, step: int) -> None:
    text = {
        "episode": episode,
        "scenario": scenario,
        "step": step,
        "paths": "yellow=Lattice red=Residual cyan=MPPI final",
    }
    if result is not None:
        text.update(
            {
                "coarse d/v/T": (
                    f"{result.coarse_path.target_d:+.2f}/"
                    f"{result.coarse_path.target_speed:.2f}/"
                    f"{result.coarse_path.horizon:.2f}"
                ),
                "residual d/v": "/".join(
                    f"{float(value):+.2f}" for value in result.residual.cpu()
                ),
                "WM baseline/selected": (
                    f"{result.metrics['mpr/baseline_wm_value']:.2f}/"
                    f"{result.metrics['mpr/selected_wm_value']:.2f}"
                ),
                "score baseline/selected": (
                    f"{result.metrics['mpr/baseline_score']:.2f}/"
                    f"{result.metrics['mpr/selected_score']:.2f}"
                ),
            }
        )
    env.unwrapped_metadrive.render(text=text)
    env.unwrapped_metadrive.render(
        text=text,
        mode="topdown",
        window=True,
        screen_record=False,
        screen_size=(720, 720),
        film_size=(1200, 1200),
        scaling=5,
        target_agent_heading_up=True,
        draw_target_vehicle_trajectory=True,
    )
    if result is not None:
        draw_refinement_overlay(env.unwrapped_metadrive, result)


def _apply_evaluation_stack(planner, requested: str, checkpoint_state: dict | None) -> dict:
    requested = str(requested or "checkpoint")
    milestone_type = None
    if isinstance(checkpoint_state, dict):
        milestone_type = checkpoint_state.get("milestone_type")
    if requested == "checkpoint":
        stack = (
            str(milestone_type)
            if milestone_type in {"world_model", "world_model_residual_rl", "full_mpr_mpc"}
            else "stage"
        )
    else:
        stack = requested
    components = checkpoint_state.get("components", {}) if isinstance(checkpoint_state, dict) else {}
    if stack in {"world_model_residual_rl", "full_mpr_mpc"} and components and not components.get("residual_rl", False):
        raise ValueError("Requested SAC stack but checkpoint has no trained residual policy.")
    if stack == "world_model":
        planner.allow_residual = False
        planner.allow_mppi = False
    elif stack == "world_model_residual_rl":
        planner.allow_residual = True
        planner.allow_mppi = False
    elif stack == "lattice_mppi":
        planner.allow_residual = False
        planner.allow_mppi = True
    elif stack == "full_mpr_mpc":
        planner.allow_residual = True
        planner.allow_mppi = True
    elif stack != "stage":
        raise ValueError(
            "evaluation_stack must be one of checkpoint, world_model, "
            "world_model_residual_rl, lattice_mppi, full_mpr_mpc."
        )
    return {
        "milestone_type": milestone_type or "n/a",
        "stack": stack,
        "residual": bool(planner.allow_residual) if stack != "stage" else None,
        "mppi": bool(planner.allow_mppi) if stack != "stage" else None,
    }


def _print_evaluation_stack(stack_info: dict) -> None:
    stack = stack_info["stack"]
    residual = stack_info["residual"]
    mppi = stack_info["mppi"]
    print(
        "Evaluation checkpoint:\n"
        f"  milestone_type: {stack_info['milestone_type']}\n"
        "Evaluation stack:\n"
        "  Lattice: ON\n"
        f"  Residual SAC: {'stage-based' if residual is None else ('ON' if residual else 'OFF')}\n"
        f"  Local MPPI: {'stage-based' if mppi is None else ('ON' if mppi else 'OFF')}"
    )


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(raw_cfg: DictConfig) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires CUDA for TD-MPC2.")
    raw_cfg.compile = False
    raw_cfg.save_agent = False
    raw_cfg.save_csv = False
    if raw_cfg.get("log_root") is None:
        raw_cfg.log_root = str(LOG_ROOT)
    if raw_cfg.checkpoint:
        _apply_checkpoint_runtime_config(raw_cfg, Path(str(raw_cfg.checkpoint)))
    cfg = parse_cfg(raw_cfg)
    set_seed(int(cfg.seed))
    checkpoint = Path(str(cfg.checkpoint)).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    env = make_env(cfg)
    run = None
    summaries = []
    try:
        run = _wandb_run(raw_cfg, cfg)
        tdmpc_agent = MPRTDMPC2(cfg)
        lattice_config = dict(FRENET_DEFAULT_CONFIG)
        lattice_config.update(dict(cfg.lattice))
        planner = MPRMPCPlanner(
            cfg,
            tdmpc_agent,
            MetaDriveFrenetController(lattice_config),
            action_space=env.action_space,
        )
        agent = MPRMPCAgent(cfg, tdmpc_agent, planner, env)
        checkpoint_state = agent.load(checkpoint)
        stack_info = _apply_evaluation_stack(
            planner,
            str(raw_cfg.get("evaluation_stack", "checkpoint")),
            checkpoint_state,
        )
        _print_evaluation_stack(stack_info)
        if stack_info["stack"] != "stage":
            agent.set_global_step(max(agent.global_env_step, planner.mppi_start_step))
        agent.eval()
        scenes = _scenarios(cfg)
        validation_episodes = []
        print(f"MPR-MPC enabled: {planner.enabled}; scenarios: {scenes.tolist()}")
        for episode, scenario in enumerate(scenes, start=1):
            diagnostics = EpisodeDiagnostics()
            obs = env.reset(seed=int(scenario))
            assert int(env.unwrapped_metadrive.current_seed) == int(scenario)
            done = False
            step = 0
            episode_return = 0.0
            episode_cost = 0.0
            episode_base_reward = 0.0
            episode_risk_cost = 0.0
            episode_risk_penalty = 0.0
            info = {}
            while not done:
                action = agent.act(obs, t0=step == 0, eval_mode=True)
                previous_obs = obs
                obs, reward, done, info = env.step(action)
                diagnostics.observe(previous_obs, action, info, agent.last_plan, env.unwrapped_metadrive.agent)
                step += 1
                if run is not None and agent.last_plan is not None:
                    run.log({"planner/scenario": int(scenario), "planner/episode_step": step,
                             **agent.last_plan.metrics})
                episode_return += float(reward)
                episode_cost += float(info.get("cost", 0.0))
                episode_base_reward += float(info.get("metadrive_reward", 0.0))
                episode_risk_cost += float(info.get("risk_field_cost", 0.0))
                episode_risk_penalty += float(
                    info.get("risk_field_reward_penalty", 0.0)
                )
                if bool(cfg.metadrive["simulator"].get("use_render", False)):
                    _render(env, agent.last_plan, episode, int(scenario), step)
            row = {
                "episode": episode, "scenario": int(scenario),
                "return": episode_return,
                "cost": episode_cost,
                "success": float(info.get("success", 0.0)),
                "length": step,
                "metadrive_reward": episode_base_reward,
                "risk_field_cost": episode_risk_cost,
                "risk_penalty": episode_risk_penalty,
                "collision": float(info.get("crash", 0.0)),
                "offroad": float(info.get("out_of_road", 0.0)),
                "route_completion": float(info.get("route_completion", 0.0)),
            }
            row.update(diagnostics.summary(info, agent))
            if len(validation_episodes) < 20:
                validation_episodes.append(diagnostics.trajectory())
            summaries.append(row)
            write_report(raw_cfg, scenes, summaries)
            if run is not None:
                run.log(
                    {f"eval/{key}": value for key, value in row.items()},
                )
            print(
                f"[{episode:02d}/{len(scenes):02d}] seed={int(scenario)} "
                f"R={episode_return:.2f} C={episode_cost:.2f} "
                f"S={row['success']:.0f} L={step}"
            )
        out = write_report(raw_cfg, scenes, summaries, agent.validate_cost_model(validation_episodes))
        print(f"Results: {out}")
    finally:
        env.close()
        if run is not None:
            if summaries:
                for key in summaries[0]:
                    if key == "end_reason":
                        continue
                    run.summary[f"eval/mean_{key}"] = float(
                        np.mean([row[key] for row in summaries])
                    )
            run.finish()
    if summaries:
        print(
            f"Summary: return={np.mean([x['return'] for x in summaries]):.3f} "
            f"cost={np.mean([x['cost'] for x in summaries]):.3f} "
            f"success={np.mean([x['success'] for x in summaries]):.3f}"
        )


if __name__ == "__main__":
    main()
