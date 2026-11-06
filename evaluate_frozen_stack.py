"""Evaluate a composite frozen MPR-MPC stack.

Composite evaluation stack:
    20K milestone -> World Model only, frozen
    50K milestone -> Residual SAC only, frozen
    1M milestone  -> Planner / MPPI state and configuration

The 1M checkpoint does NOT overwrite the 20K world model or
the 50K residual policy.

This script is intended for evaluation only:
    - no model training
    - no optimizer loading
    - no checkpoint saving
    - no CSV saving
    - optional real-time MetaDrive rendering
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


# ---------------------------------------------------------------------
# Repository bootstrap
# ---------------------------------------------------------------------

SCRIPT_ROOT = Path(__file__).resolve().parent

if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from mpr_mpc._bootstrap import LOG_ROOT, bootstrap

bootstrap()

from common.parser import parse_cfg
from common.seed import set_seed
from env import make_env

from mpr_mpc.agent import MPRMPCAgent
from mpr_mpc.lattice.frenet_metadrive import (
    FRENET_DEFAULT_CONFIG,
    MetaDriveFrenetController,
)
from mpr_mpc.planning.coordinator import MPRMPCPlanner
from mpr_mpc.planning.visualization import draw_refinement_overlay
from mpr_mpc.tdmpc2.core import MPRTDMPC2
from mpr_mpc.evaluation_diagnostics import EpisodeDiagnostics, fixed_scenarios, write_report
from mpr_mpc.evaluate_model_suite import _apply_checkpoint_runtime_config, _merge_section


# ---------------------------------------------------------------------
# W&B
# ---------------------------------------------------------------------

def _wandb_run(raw_cfg, cfg):
    """Create an optional W&B evaluation run."""

    if not bool(raw_cfg.enable_wandb):
        return None

    import wandb

    name = raw_cfg.wandb_run_name or f"eval_frozen_stack_seed{raw_cfg.seed}"

    return wandb.init(
        project=str(raw_cfg.wandb_project),
        entity=raw_cfg.wandb_entity,
        name=str(name),
        config=OmegaConf.to_container(raw_cfg, resolve=True),
        job_type="evaluation",
        dir=str(cfg.work_dir),
    )


# ---------------------------------------------------------------------
# Scenario sampling
# ---------------------------------------------------------------------

def _scenarios(cfg) -> np.ndarray:
    """Sample reproducible MetaDrive scenarios."""

    return fixed_scenarios(cfg)


# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------

def _render(
    env,
    result,
    episode: int,
    scenario: int,
    step: int,
) -> None:
    """Render MetaDrive together with MPR-MPC planning trajectories."""

    text = {
        "episode": episode,
        "scenario": scenario,
        "step": step,
        "stack": "20K WM + 50K Residual + MPPI",
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
                    f"{float(value):+.2f}"
                    for value in result.residual.cpu()
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

    # Normal MetaDrive rendering
    env.unwrapped_metadrive.render(text=text)

    # Top-down planning visualization
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
        draw_refinement_overlay(
            env.unwrapped_metadrive,
            result,
        )


# ---------------------------------------------------------------------
# Freeze helpers
# ---------------------------------------------------------------------

def _freeze_world_model(agent: MPRMPCAgent) -> None:
    """Freeze the loaded 20K TD-MPC2 world model."""

    agent.tdmpc_agent.model.eval()

    for parameter in agent.tdmpc_agent.model.parameters():
        parameter.requires_grad_(False)

    agent._tdmpc_frozen = True


def _freeze_residual_sac(agent: MPRMPCAgent) -> None:
    """Freeze the loaded 50K Residual SAC agent."""

    residual = agent.residual_sac

    residual.eval()

    networks = (
        residual.actor,
        residual.q1,
        residual.q2,
        residual.target_q1,
        residual.target_q2,
        residual.cost_q1,
        residual.cost_q2,
        residual.target_cost_q1,
        residual.target_cost_q2,
    )

    for network in networks:
        network.eval()

        for parameter in network.parameters():
            parameter.requires_grad_(False)

    # Temperature parameter is also fixed during evaluation.
    if residual.log_alpha.requires_grad:
        residual.log_alpha.requires_grad_(False)


# ---------------------------------------------------------------------
# Composite checkpoint loading
# ---------------------------------------------------------------------

def _load_composite_stack(
    agent: MPRMPCAgent,
    planner: MPRMPCPlanner,
    world_model_checkpoint: Path,
    residual_checkpoint: Path,
    mppi_checkpoint: Path | None,
) -> None:
    """Load different components from different milestone checkpoints."""

    print()
    print("=" * 72)
    print("Loading composite frozen MPR-MPC stack")
    print("=" * 72)

    print(f"20K World Model : {world_model_checkpoint}")
    print(f"50K Residual SAC: {residual_checkpoint}")
    print(f"1M Planner/MPPI : {mppi_checkpoint}")
    print()

    # -------------------------------------------------------------
    # 1. Load ONLY the 20K World Model.
    # -------------------------------------------------------------

    wm_state = agent.load_milestone(
        world_model_checkpoint,
        components=["world_model"],
        load_optimizer=False,
    )

    wm_type = str(wm_state.get("milestone_type", ""))

    if wm_type != "world_model":
        raise ValueError(
            "Expected the 20K checkpoint to have "
            f"milestone_type='world_model', got {wm_type!r}."
        )

    print("[OK] Loaded 20K World Model only.")

    # -------------------------------------------------------------
    # 2. Load ONLY the Residual SAC from the 50K checkpoint.
    #
    # IMPORTANT:
    # This does NOT load the 50K World Model.
    # Therefore the previously loaded 20K WM remains intact.
    # -------------------------------------------------------------

    residual_state = agent.load_milestone(
        residual_checkpoint,
        components=["residual_rl"],
        load_optimizer=False,
    )

    residual_type = str(
        residual_state.get("milestone_type", "")
    )

    if residual_type != "world_model_residual_rl":
        raise ValueError(
            "Expected the 50K checkpoint to have "
            "milestone_type='world_model_residual_rl', "
            f"got {residual_type!r}."
        )

    print("[OK] Loaded 50K Residual SAC only.")

    # -------------------------------------------------------------
    # 3. Load ONLY the planner state from the 1M checkpoint.
    #
    # This performs strict compatibility checking against the
    # full Stage-C checkpoint, including MPPI configuration.
    #
    # It does NOT overwrite:
    #   - 20K World Model
    #   - 50K Residual SAC
    # -------------------------------------------------------------

    if mppi_checkpoint is not None:
        planner_state = agent.load_milestone(mppi_checkpoint, components=["planner"], load_optimizer=False)
        if planner_state.get("milestone_type") != "full_mpr_mpc":
            raise ValueError("mppi_checkpoint must contain a full_mpr_mpc planner milestone.")
        print("[OK] Loaded planner counters; evaluation MPPI configuration remains explicit.")
    else:
        print("[OK] MPPI has no learned weights; using configured optimizer without a planner checkpoint.")

    # -------------------------------------------------------------
    # Explicitly freeze learned modules.
    # -------------------------------------------------------------

    _freeze_world_model(agent)
    _freeze_residual_sac(agent)

    # -------------------------------------------------------------
    # Force full Stage-C runtime.
    # -------------------------------------------------------------

    planner.allow_residual = True
    planner.allow_mppi = True

    # Stage C is active for steps >= mppi_start_step.
    # 1,000,000 represents the final deployment stage.
    agent.set_global_step(1_000_000)

    # Evaluation mode.
    agent.eval()

    print()
    print("=" * 72)
    print("Composite evaluation stack")
    print("=" * 72)
    print("Lattice       : ON")
    print("World Model   : 20K checkpoint - FROZEN")
    print("Residual SAC  : 50K checkpoint - FROZEN")
    print("Local MPPI    : ON")
    print("Planner stage : Stage C")
    print("Training      : OFF")
    print("=" * 72)
    print()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

@hydra.main(
    version_base="1.3",
    config_path=".",
    config_name="config",
)
def main(raw_cfg: DictConfig) -> None:

    # -----------------------------------------------------------------
    # CUDA check
    # -----------------------------------------------------------------

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Evaluation requires CUDA for TD-MPC2."
        )

    # -----------------------------------------------------------------
    # Evaluation must not save or train anything.
    # -----------------------------------------------------------------

    raw_cfg.compile = False
    raw_cfg.save_agent = False
    raw_cfg.save_csv = False
    raw_cfg.save_video = False

    if raw_cfg.get("log_root") is None:
        raw_cfg.log_root = str(LOG_ROOT)

    # -----------------------------------------------------------------
    # Composite checkpoint arguments
    # -----------------------------------------------------------------

    world_checkpoint_arg = raw_cfg.get(
        "world_model_checkpoint"
    )

    residual_checkpoint_arg = raw_cfg.get(
        "residual_checkpoint"
    )

    mppi_checkpoint_arg = raw_cfg.get(
        "mppi_checkpoint"
    )

    if not world_checkpoint_arg:
        raise ValueError(
            "world_model_checkpoint is required.\n"
            "Example:\n"
            "+world_model_checkpoint=/path/to/"
            "milestone_20k_world_model.pt"
        )

    if not residual_checkpoint_arg:
        raise ValueError(
            "residual_checkpoint is required.\n"
            "Example:\n"
            "+residual_checkpoint=/path/to/"
            "milestone_50k_world_model_residual_rl.pt"
        )

    world_model_checkpoint = Path(
        str(world_checkpoint_arg)
    ).expanduser().resolve()

    residual_checkpoint = Path(
        str(residual_checkpoint_arg)
    ).expanduser().resolve()

    mppi_checkpoint = Path(str(mppi_checkpoint_arg)).expanduser().resolve() if mppi_checkpoint_arg else None

    # -----------------------------------------------------------------
    # File checks
    # -----------------------------------------------------------------

    if not world_model_checkpoint.is_file():
        raise FileNotFoundError(
            "20K World Model checkpoint not found:\n"
            f"{world_model_checkpoint}"
        )

    if not residual_checkpoint.is_file():
        raise FileNotFoundError(
            "50K Residual checkpoint not found:\n"
            f"{residual_checkpoint}"
        )

    if mppi_checkpoint is not None and not mppi_checkpoint.is_file():
        raise FileNotFoundError(
            "1M MPPI/planner checkpoint not found:\n"
            f"{mppi_checkpoint}"
        )

    # -----------------------------------------------------------------
    # Parse normal project configuration
    # -----------------------------------------------------------------

    _apply_checkpoint_runtime_config(raw_cfg, residual_checkpoint)
    wm_state = torch.load(world_model_checkpoint, map_location="cpu", weights_only=False)
    encoder = wm_state.get("metadata", {}).get("encoder_config")
    if isinstance(encoder, dict):
        _merge_section(raw_cfg.state_encoder, encoder)
    else:
        raw_cfg.state_encoder.grouped = any("ego_encoder" in key for key in wm_state.get("model", {}))
    del wm_state
    cfg = parse_cfg(raw_cfg)

    set_seed(int(cfg.seed))

    # -----------------------------------------------------------------
    # Environment
    # -----------------------------------------------------------------

    env = make_env(cfg)

    run = None
    summaries: list[dict[str, float]] = []

    try:

        # -------------------------------------------------------------
        # Optional W&B
        # -------------------------------------------------------------

        run = _wandb_run(
            raw_cfg,
            cfg,
        )

        # -------------------------------------------------------------
        # TD-MPC2
        # -------------------------------------------------------------

        tdmpc_agent = MPRTDMPC2(cfg)

        # -------------------------------------------------------------
        # Frenet Lattice
        # -------------------------------------------------------------

        lattice_config = dict(
            FRENET_DEFAULT_CONFIG
        )

        lattice_config.update(
            dict(cfg.lattice)
        )

        lattice_controller = MetaDriveFrenetController(
            lattice_config
        )

        # -------------------------------------------------------------
        # MPR-MPC planner
        # -------------------------------------------------------------

        planner = MPRMPCPlanner(
            cfg,
            tdmpc_agent,
            lattice_controller,
            action_space=env.action_space,
        )

        # -------------------------------------------------------------
        # Complete MPR-MPC agent
        # -------------------------------------------------------------

        agent = MPRMPCAgent(
            cfg,
            tdmpc_agent,
            planner,
            env,
        )

        # -------------------------------------------------------------
        # Load:
        #
        # 20K WM
        # +
        # 50K Residual SAC
        # +
        # 1M MPPI / Planner
        # -------------------------------------------------------------

        _load_composite_stack(
            agent,
            planner,
            world_model_checkpoint,
            residual_checkpoint,
            mppi_checkpoint,
        )

        # -------------------------------------------------------------
        # Random evaluation scenes
        # -------------------------------------------------------------

        scenes = _scenarios(cfg)
        validation_episodes = []

        print(
            "Evaluation scenarios:",
            scenes.tolist(),
        )

        print(
            f"Number of scenarios: {len(scenes)}"
        )

        print()

        # -------------------------------------------------------------
        # Evaluation loop
        # -------------------------------------------------------------

        for episode, scenario in enumerate(
            scenes,
            start=1,
        ):
            diagnostics = EpisodeDiagnostics()
            obs = env.reset(
                seed=int(scenario)
            )
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

                # -----------------------------------------------------
                # Composite frozen stack inference:
                #
                # Lattice
                #   ->
                # 20K frozen World Model
                #   ->
                # 50K frozen Residual SAC
                #   ->
                # MPPI
                # -----------------------------------------------------

                with torch.no_grad():
                    action = agent.act(
                        obs,
                        t0=(step == 0),
                        eval_mode=True,
                    )

                previous_obs = obs
                obs, reward, done, info = env.step(
                    action
                )
                diagnostics.observe(previous_obs, action, info, agent.last_plan, env.unwrapped_metadrive.agent)

                step += 1

                episode_return += float(reward)

                episode_cost += float(
                    info.get(
                        "cost",
                        0.0,
                    )
                )

                episode_base_reward += float(
                    info.get(
                        "metadrive_reward",
                        0.0,
                    )
                )

                episode_risk_cost += float(
                    info.get(
                        "risk_field_cost",
                        0.0,
                    )
                )

                episode_risk_penalty += float(
                    info.get(
                        "risk_field_reward_penalty",
                        0.0,
                    )
                )

                # -----------------------------------------------------
                # Real-time visualization
                # -----------------------------------------------------

                if bool(
                    cfg.metadrive[
                        "simulator"
                    ].get(
                        "use_render",
                        False,
                    )
                ):
                    _render(
                        env,
                        agent.last_plan,
                        episode,
                        int(scenario),
                        step,
                    )

            # ---------------------------------------------------------
            # Episode statistics
            # ---------------------------------------------------------

            row = {
                "episode": episode, "scenario": int(scenario),
                "return": episode_return,
                "cost": episode_cost,
                "success": float(
                    info.get(
                        "success",
                        0.0,
                    )
                ),
                "length": float(step),
                "metadrive_reward": episode_base_reward,
                "risk_field_cost": episode_risk_cost,
                "risk_penalty": episode_risk_penalty,
                "collision": float(
                    info.get(
                        "crash",
                        0.0,
                    )
                ),
                "offroad": float(
                    info.get(
                        "out_of_road",
                        0.0,
                    )
                ),
                "route_completion": float(
                    info.get(
                        "route_completion",
                        0.0,
                    )
                ),
            }
            row.update(diagnostics.summary(info, agent))
            if len(validation_episodes) < 20:
                validation_episodes.append(diagnostics.trajectory())
            summaries.append(row)
            write_report(raw_cfg, scenes, summaries)

            # ---------------------------------------------------------
            # Optional W&B
            # ---------------------------------------------------------

            if run is not None:

                run.log(
                    {
                        f"eval/{key}": value
                        for key, value
                        in row.items()
                    },
                    step=episode,
                )

            # ---------------------------------------------------------
            # Terminal output
            # ---------------------------------------------------------

            print(
                f"[{episode:02d}/{len(scenes):02d}] "
                f"seed={int(scenario)} "
                f"R={episode_return:.2f} "
                f"C={episode_cost:.2f} "
                f"S={row['success']:.0f} "
                f"RC={row['route_completion']:.3f} "
                f"Crash={row['collision']:.0f} "
                f"Offroad={row['offroad']:.0f} "
                f"L={step}"
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

                    run.summary[
                        f"eval/mean_{key}"
                    ] = float(
                        np.mean(
                            [
                                row[key]
                                for row
                                in summaries
                            ]
                        )
                    )

            run.finish()

    # -----------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------

    if summaries:

        mean_return = float(
            np.mean(
                [
                    x["return"]
                    for x
                    in summaries
                ]
            )
        )

        mean_cost = float(
            np.mean(
                [
                    x["cost"]
                    for x
                    in summaries
                ]
            )
        )

        mean_success = float(
            np.mean(
                [
                    x["success"]
                    for x
                    in summaries
                ]
            )
        )

        mean_route_completion = float(
            np.mean(
                [
                    x["route_completion"]
                    for x
                    in summaries
                ]
            )
        )

        collision_rate = float(
            np.mean(
                [
                    x["collision"]
                    for x
                    in summaries
                ]
            )
        )

        offroad_rate = float(
            np.mean(
                [
                    x["offroad"]
                    for x
                    in summaries
                ]
            )
        )

        print()
        print("=" * 72)
        print("Composite frozen-stack evaluation summary")
        print("=" * 72)

        print(
            f"Episodes         : {len(summaries)}"
        )

        print(
            f"Mean return      : {mean_return:.3f}"
        )

        print(
            f"Mean cost        : {mean_cost:.3f}"
        )

        print(
            f"Success rate     : {mean_success:.3f}"
        )

        print(
            f"Route completion : {mean_route_completion:.3f}"
        )

        print(
            f"Collision rate   : {collision_rate:.3f}"
        )

        print(
            f"Off-road rate    : {offroad_rate:.3f}"
        )

        print("=" * 72)


if __name__ == "__main__":
    main()
