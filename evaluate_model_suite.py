"""Benchmark all MPR-MPC model versions on the same MetaDrive scenarios.

The script is evaluation-only:
    - no training
    - no model saving
    - no rendering
    - one fixed scenario list shared by every checkpoint
    - per-episode CSV plus aggregate CSV/JSON outputs
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict


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
from mpr_mpc.planning.native_config import native_mppi_config
from mpr_mpc.tdmpc2.core import MPRTDMPC2
from mpr_mpc.evaluation_diagnostics import EpisodeDiagnostics, StepDiagnosticsLogger


DEFAULT_MODELS_ROOT = (
    SCRIPT_ROOT / "logs" / "metadrive-risk" / "1"
)


PREFERRED_CHECKPOINT_NAMES = (
    "milestone_1m_full_mpr_mpc.pt",
    "milestone_500k_full_mpr_mpc.pt",
    "milestone_500k_world_model_residual_rl.pt",
    "milestone_200k_world_model.pt",
    "milestone_50k_world_model_residual_rl.pt",
    "milestone_20k_world_model.pt",
    "final.pt",
    "latest.pt",
    "best.pt",
)


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
            "world_model_residual_rl, lattice_mppi, full_mpr_mpc, stage."
        )
    return {
        "milestone_type": milestone_type or "n/a",
        "stack": stack,
        "residual": bool(planner.allow_residual) if stack != "stage" else None,
        "mppi": bool(planner.allow_mppi) if stack != "stage" else None,
    }


def _print_evaluation_stack(stack_info: dict) -> None:
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


STEP_SUM_KEYS = (
    "cost",
    "metadrive_reward",
    "risk_field_cost",
    "risk_field_normalized_cost",
    "risk_field_reward_penalty",
    "risk_field_boundary_cost",
    "risk_field_lane_cost",
    "risk_field_offroad_cost",
    "risk_field_vehicle_cost",
    "risk_field_object_cost",
    "safety_risk_cost",
    "safety_event_cost",
)


FINAL_INFO_KEYS = (
    "success",
    "arrive_dest",
    "crash",
    "crash_vehicle",
    "crash_object",
    "out_of_road",
    "out_of_road_warning",
    "route_completion",
    "max_step",
)


PLAN_MEAN_KEYS = (
    "mpr/stage",
    "mpr/candidate_count",
    "mpr/feasible_count",
    "mpr/feasible_ratio",
    "mpr/used_lattice_fallback",
    "mpr/coarse_value",
    "mpr/baseline_wm_value",
    "mpr/selected_wm_value",
    "mpr/wm_value_gain",
    "mpr/baseline_score",
    "mpr/selected_score",
    "mpr/planner_score_gain",
    "mpr/residual_d",
    "mpr/residual_v",
    "mpr/residual_valid",
    "mpr/corridor_reject_rate",
    "mpr/baseline_selected",
    "mpr/planning_ms",
) + tuple(f"mpr/{stage}_{key}" for stage in ("lattice", "sac", "mppi") for key in
          ("predicted_cost", "value", "safe", "min_ttc", "min_clearance", "boundary_margin", "soft_risk")) + (
    "mpr/risk_reduction_sac", "mpr/risk_reduction_mppi", "mpr/mppi_correction_magnitude",
    "mpr/safe_candidate_ratio", "mpr/no_safe_candidate", "mpr/selected_safe", "mpr/baseline_safe",
    "mpr/collision_reject_count", "mpr/offroad_reject_count", "mpr/kinematic_reject_count", "mpr/expanded_search",
    "mpr/collision_reject_rate", "mpr/offroad_reject_rate", "mpr/cost_validated",
)


def _plain_name(path: Path) -> str:
    return path.name.replace("/", "_")


def _metric_name(prefix: str, key: str) -> str:
    return f"{prefix}_{key.replace('/', '_')}"


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _nanmean(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def _discover_checkpoints(models_root: Path) -> list[tuple[str, Path]]:
    versions: list[tuple[str, Path]] = []
    for version_dir in sorted(path for path in models_root.iterdir() if path.is_dir()):
        model_dir = version_dir / "models"
        if not model_dir.is_dir():
            continue
        checkpoints = {path.name: path for path in model_dir.glob("*.pt")}
        selected = None
        for name in PREFERRED_CHECKPOINT_NAMES:
            if name in checkpoints:
                selected = checkpoints[name]
                break
        if selected is None and checkpoints:
            selected = max(checkpoints.values(), key=lambda path: path.stat().st_mtime)
        if selected is not None:
            versions.append((_plain_name(version_dir), selected.resolve()))
    if not versions:
        raise FileNotFoundError(f"No model checkpoints found under {models_root}")
    return versions


def _scenarios(raw_cfg: DictConfig) -> list[int]:
    benchmark = raw_cfg.get("benchmark", {})
    simulator = raw_cfg.metadrive["simulator"]
    episodes = int(benchmark.get("episodes", raw_cfg.eval_episodes))
    start = int(benchmark.get("scenario_start", simulator["start_seed"]))
    pool_count = int(
        benchmark.get(
            "scenario_pool_count",
            max(int(simulator["num_scenarios"]), episodes),
        )
    )
    if episodes <= 0:
        raise ValueError("benchmark.episodes must be positive.")
    scenario_seed = int(benchmark.get("scenario_seed", raw_cfg.seed))
    fixed = list(raw_cfg.get("evaluation_scenarios", []))
    scenario_file = benchmark.get("scenarios_file") or raw_cfg.get("evaluation_scenarios_file")
    if scenario_file:
        payload = json.loads(Path(str(scenario_file)).expanduser().read_text(encoding="utf-8"))
        fixed = payload.get("scenarios") if isinstance(payload, dict) else payload
        if not isinstance(fixed, list) or not fixed:
            raise ValueError("scenarios_file must contain a nonempty scenario list.")
    if fixed:
        if any(type(value) is not int or value < 0 for value in fixed):
            raise ValueError("Scenario IDs must be nonnegative integers.")
        if len(set(fixed)) != len(fixed):
            raise ValueError("scenarios_file contains duplicate scenario IDs.")
        if len(fixed) == episodes:
            return fixed
        if episodes < len(fixed) or not bool(benchmark.get("extend_scenarios", False)):
            raise ValueError("Scenario file length differs from benchmark.episodes; enable extend_scenarios to append IDs.")
    if pool_count < episodes:
        raise ValueError(
            "scenario_pool_count must be at least benchmark.episodes to avoid duplicates."
        )
    rng = np.random.default_rng(scenario_seed)
    candidates = np.arange(start, start + pool_count)
    if fixed:
        remaining = np.setdiff1d(candidates, np.asarray(fixed))
        extra_count = episodes - len(fixed)
        if len(remaining) < extra_count:
            raise ValueError("Scenario pool cannot supply enough additional unique IDs.")
        return fixed + [int(value) for value in rng.choice(remaining, size=extra_count, replace=False)]
    return [int(value) for value in rng.choice(candidates, size=episodes, replace=False)]


def _configure_checkpoint_architecture(cfg_raw: DictConfig, version: str) -> None:
    """Match historical checkpoints that predate the grouped state encoder."""

    is_grouped_latent_tanh = "grouped_latent_tanh" in version
    cfg_raw.state_encoder.grouped = bool(is_grouped_latent_tanh)
    cfg_raw.residual_rl.residual_mapping = (
        "latent_tanh" if is_grouped_latent_tanh else "linear"
    )


def _merge_section(target, values: dict[str, Any]) -> None:
    for key, value in values.items():
        if isinstance(value, dict) and key in target and hasattr(target[key], "items"):
            _merge_section(target[key], value)
        else:
            target[key] = value


def _apply_checkpoint_runtime_config(cfg_raw: DictConfig, checkpoint: Path) -> None:
    """Apply planner/runtime config saved in checkpoints before model creation."""

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        return

    planner_state = state.get("planner") or state.get("mpr_mpc") or {}
    metadata = state.get("metadata") or {}
    weights = state.get("model", {})
    encoder = metadata.get("encoder_config")
    if isinstance(encoder, dict):
        _merge_section(cfg_raw.state_encoder, encoder)
    else:
        cfg_raw.state_encoder.grouped = any("ego_encoder" in key for key in weights)
    residual_config = planner_state.get("residual_config") or metadata.get("residual_config") or {}
    for key in ("hidden_dims", "latent_tanh_scale", "latent_tanh_eps", "wm_return_scale", "wm_cost_scale",
                "path_feature_clip", "path_feature_scales", "risk_feature_scales"):
        if key in residual_config:
            cfg_raw.residual_rl[key] = residual_config[key]
    cfg_raw.residual_rl.residual_mapping = metadata.get("residual_mapping", residual_config.get("residual_mapping", "linear"))
    features = metadata.get("risk_features", residual_config.get("risk_features", []))
    cfg_raw.residual_rl.risk_features = features
    cfg_raw.residual_rl.include_wm_cost_in_state = bool(residual_config.get("include_wm_cost_in_state", False))
    cfg_raw.residual_rl.coarse_cost_feature = bool(metadata.get("coarse_cost_feature", residual_config.get("coarse_cost_feature", False)))
    if not bool(cfg_raw.get("restore_checkpoint_runtime", True)):
        return

    stage_config = planner_state.get("stage_config") or metadata.get("stage_config")
    if isinstance(stage_config, dict):
        _merge_section(cfg_raw.mpr_mpc.stages, stage_config)

    mppi_config = planner_state.get("mppi_config") or metadata.get("mppi_config")
    if isinstance(mppi_config, dict) and bool(cfg_raw.get("restore_mppi_runtime", False)):
        _merge_section(cfg_raw.mpr_mpc.mppi, native_mppi_config(mppi_config))

    residual_config = planner_state.get("residual_config")
    if isinstance(residual_config, dict):
        _merge_section(cfg_raw.residual_rl, residual_config)
    # Only old SAC feature geometry uses these legacy settings, never MPPI.
    if features and isinstance(mppi_config, dict):
        geometry = {key: mppi_config[key] for key in (
            "substeps", "model_margin", "road_margin", "max_acceleration", "max_deceleration",
            "ttc_lookahead", "swept_max_depth", "swept_angle_tolerance", "swept_position_tolerance",
        ) if key in mppi_config}
        cfg_raw.residual_rl.risk_geometry = geometry


def _make_cfg(raw_cfg: DictConfig, version: str, checkpoint: Path, scenarios: list[int]):
    cfg_raw = OmegaConf.create(OmegaConf.to_container(raw_cfg, resolve=True))
    with open_dict(cfg_raw):
        _configure_checkpoint_architecture(cfg_raw, version)
        _apply_checkpoint_runtime_config(cfg_raw, checkpoint)
        benchmark = cfg_raw.get("benchmark", {})
        cfg_raw.checkpoint = str(checkpoint)
        cfg_raw.eval_episodes = len(scenarios)
        cfg_raw.compile = False
        cfg_raw.enable_wandb = False
        cfg_raw.save_agent = False
        cfg_raw.save_csv = False
        cfg_raw.save_video = False
        cfg_raw.metadrive.simulator.use_render = False
        cfg_raw.metadrive.simulator.image_observation = False
        cfg_raw.metadrive.simulator.manual_control = False
        cfg_raw.metadrive.simulator.num_scenarios = max(
            int(cfg_raw.metadrive.simulator.num_scenarios),
            max(scenarios) - min(scenarios) + 1,
        )
        cfg_raw.metadrive.simulator.start_seed = min(scenarios)
        if cfg_raw.get("log_root") is None:
            cfg_raw.log_root = str(LOG_ROOT)
    return cfg_raw, parse_cfg(cfg_raw)


def _make_agent(cfg, env) -> MPRMPCAgent:
    tdmpc_agent = MPRTDMPC2(cfg)
    lattice_config = dict(FRENET_DEFAULT_CONFIG)
    lattice_config.update(dict(cfg.lattice))
    planner = MPRMPCPlanner(
        cfg,
        tdmpc_agent,
        MetaDriveFrenetController(lattice_config),
        action_space=env.action_space,
    )
    return MPRMPCAgent(cfg, tdmpc_agent, planner, env)


def _evaluate_checkpoint(
    raw_cfg: DictConfig,
    version: str,
    checkpoint: Path,
    scenarios: list[int],
    *,
    on_episode: Callable[[dict[str, Any]], None] | None = None,
    on_step: Callable[[int, int, Any, Any, bool], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg_raw, cfg = _make_cfg(raw_cfg, version, checkpoint, scenarios)
    set_seed(int(cfg.seed))

    env = make_env(cfg)
    rows: list[dict[str, Any]] = []
    validation_episodes = []
    benchmark = cfg_raw.get("benchmark", {})
    save_steps = bool(benchmark.get("save_step_diagnostics", False))
    step_dir = (Path(str(benchmark.get("output_dir", SCRIPT_ROOT / "logs/benchmark"))) / "steps"
                if save_steps or benchmark.get("enable_wandb", False) else None)
    step_log = StepDiagnosticsLogger(step_dir, f"{version}_{cfg_raw.get('evaluation_stack', 'checkpoint')}_seed{cfg.seed}",
        wandb_cfg=cfg_raw if benchmark.get("enable_wandb", False) else None,
        compact=bool(benchmark.get("compact_step_logs", True)), stride=int(benchmark.get("step_log_stride", 10)))
    try:
        agent = _make_agent(cfg, env)
        checkpoint_state = agent.load(checkpoint)
        stack_info = _apply_evaluation_stack(
            agent.planner,
            str(cfg_raw.get("evaluation_stack", "checkpoint")),
            checkpoint_state,
        )
        if stack_info["stack"] != "stage":
            # Explicit evaluation switches must not be suppressed by a training threshold.
            agent.set_global_step(max(agent.global_env_step, agent.planner.mppi_start_step))
            stage = agent.planner.stage_for_step(agent.global_env_step)
            if (stage.use_residual, stage.use_mppi) != (
                stack_info["residual"], stack_info["mppi"]
            ):
                raise RuntimeError("Requested evaluation stack is not active.")
        _print_evaluation_stack(stack_info)
        agent.eval()
        benchmark = cfg_raw.get("benchmark", {})
        paired_seeds = bool(benchmark.get("paired_episode_seeds", False))
        if benchmark.get("mppi_eval_seed") is not None:
            agent.planner.local_mppi.eval_seed = int(benchmark["mppi_eval_seed"])
        cost_limit = _as_float(cfg_raw.get("residual_rl", {}).get("cost_limit"), float("nan"))

        for episode_index, scenario in enumerate(scenarios, start=1):
            print(f"[{episode_index:03d}/{len(scenarios):03d}] seed={scenario} reset", flush=True)
            diagnostics = EpisodeDiagnostics()
            episode_seed = (int(cfg.seed) + int(scenario)) % 2**32
            agent.planner.reset(seed=episode_seed)
            if paired_seeds:
                set_seed(episode_seed)
            obs = env.reset(seed=int(scenario))
            if int(env.unwrapped_metadrive.current_seed) != int(scenario):
                raise RuntimeError(f"Environment remapped scenario seed {scenario}.")
            done = False
            step = 0
            episode_return = 0.0
            info: dict[str, Any] = {}
            step_sums = defaultdict(float)
            plan_values = {key: [] for key in PLAN_MEAN_KEYS}
            episode_crash = False
            episode_out_of_road = False
            first_crash_step = 0
            first_out_of_road_step = 0
            ttc_values = []
            mppi_baselines, mppi_gains = [], []

            while not done:
                with torch.no_grad():
                    action = agent.act(obs, t0=(step == 0), eval_mode=True)
                previous_obs = obs
                obs, reward, done, info = env.step(action)
                diagnostics.observe(previous_obs, action, info, agent.last_plan, getattr(env.unwrapped_metadrive, "agent", None))
                episode_crash = episode_crash or any(
                    _as_float(info.get(key, 0.0)) > 0.0
                    for key in (
                        "crash", "crash_vehicle", "crash_object", "crash_building",
                        "crash_human", "crash_sidewalk",
                    )
                )
                episode_out_of_road = episode_out_of_road or (
                    _as_float(info.get("out_of_road", 0.0)) > 0.0
                )
                step += 1
                step_log.log(scenario, step, action, agent.last_plan, info=info)
                if on_step is not None:
                    on_step(int(scenario), step, env, agent.last_plan, bool(done))
                if step == 1 or step % 100 == 0:
                    print(f"[progress] seed={scenario} step={step} "
                          f"route={float(info.get('route_completion', 0)):.3f}", flush=True)
                if episode_crash and not first_crash_step:
                    first_crash_step = step
                if episode_out_of_road and not first_out_of_road_step:
                    first_out_of_road_step = step
                ttc = _as_float(info.get("risk_field_min_ttc"), float("nan"))
                if math.isfinite(ttc) and ttc >= 0.0:
                    ttc_values.append(ttc)
                episode_return += float(reward)
                for key in STEP_SUM_KEYS:
                    step_sums[key] += _as_float(info.get(key, 0.0))
                if agent.last_plan is not None:
                    for key in PLAN_MEAN_KEYS:
                        value = agent.last_plan.metrics.get(key)
                        if value is not None:
                            plan_values[key].append(float(value))
                    if agent.last_plan.stage.use_mppi:
                        mppi_baselines.append(float(agent.last_plan.metrics["mpr/baseline_selected"]))
                        mppi_gains.append(float(agent.last_plan.metrics["mpr/planner_score_gain"]))

            row: dict[str, Any] = {
                "version": version,
                "checkpoint": str(checkpoint),
                "milestone_type": stack_info["milestone_type"],
                "stack": stack_info["stack"],
                "residual_enabled": stack_info["residual"],
                "mppi_enabled": stack_info["mppi"],
                "episode": episode_index,
                "scenario": int(scenario),
                "return": episode_return,
                "length": int(step),
                "end_reason": str(info.get("end_reason", "")),
                "episode_crash": float(episode_crash),
                "episode_out_of_road": float(episode_out_of_road),
                "collision_and_success": float(episode_crash and _as_float(info.get("success")) > 0.0),
                "collision_and_offroad": float(episode_crash and episode_out_of_road),
                "first_crash_step": first_crash_step,
                "first_out_of_road_step": first_out_of_road_step,
                "mean_ttc": _nanmean(ttc_values),
                "minimum_ttc": min(ttc_values) if ttc_values else float("nan"),
                "ttc_valid_steps": len(ttc_values),
                "ttc_valid_step_fraction": len(ttc_values) / max(step, 1),
                "cost_limit": cost_limit,
                "episode_cost_over_limit": (
                    float(step_sums["cost"] > cost_limit) if math.isfinite(cost_limit) else float("nan")
                ),
                "mppi_steps": len(mppi_baselines),
                "mppi_baseline_selected": _nanmean(mppi_baselines),
                "mppi_score_gain": _nanmean(mppi_gains),
            }
            if paired_seeds:
                row["episode_seed"] = episode_seed
            row.update(diagnostics.summary(info, agent))
            if len(validation_episodes) < 20:
                validation_episodes.append(diagnostics.trajectory())
            for key in STEP_SUM_KEYS:
                row[_metric_name("sum", key)] = float(step_sums[key])
            for key in FINAL_INFO_KEYS:
                row[_metric_name("final", key)] = _as_float(info.get(key, 0.0))
            for key in PLAN_MEAN_KEYS:
                row[_metric_name("mean", key)] = _nanmean(plan_values[key])

            rows.append(row)
            if on_episode is not None:
                on_episode(row)
            print(
                f"[{version}] [{episode_index:03d}/{len(scenarios):03d}] "
                f"seed={int(scenario)} R={episode_return:.2f} "
                f"C={row['sum_cost']:.2f} S={row['final_success']:.0f} "
                f"RC={row['final_route_completion']:.3f} L={step}"
            )
    finally:
        env.close()
        step_log.close()

    summary = _summarize(version, checkpoint, rows)
    summary.update(agent.validate_cost_model(validation_episodes))
    return rows, summary


def _summarize(version: str, checkpoint: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "version": version,
        "checkpoint": str(checkpoint),
        "episodes": len(rows),
    }
    numeric_keys = dict.fromkeys(
        key
        for row in rows
        for key, value in row.items()
        if isinstance(value, (int, float, np.integer, np.floating))
        and key not in {"episode", "scenario", "episode_seed", "test_seed"}
    )
    for key in numeric_keys:
        values = [float(row.get(key, float("nan"))) for row in rows]
        summary[f"mean_{key}"] = _nanmean(values)
        finite = [value for value in values if math.isfinite(value)]
        summary[f"std_{key}"] = float(np.std(finite)) if finite else float("nan")
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _output_dir(raw_cfg: DictConfig) -> Path:
    benchmark = raw_cfg.get("benchmark", {})
    value = benchmark.get("output_dir")
    if value:
        return Path(str(value)).expanduser().resolve()
    return (
        SCRIPT_ROOT
        / "logs"
        / "benchmark"
        / f"mpr_mpc_suite_100_seed{int(raw_cfg.seed)}"
    ).resolve()


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(raw_cfg: DictConfig) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires CUDA for TD-MPC2.")

    with open_dict(raw_cfg):
        raw_cfg.eval_episodes = int(raw_cfg.get("benchmark", {}).get("episodes", 100))
        raw_cfg.enable_wandb = False
        raw_cfg.save_agent = False
        raw_cfg.save_csv = False
        raw_cfg.save_video = False
        raw_cfg.compile = False
        raw_cfg.metadrive.simulator.use_render = False
        if raw_cfg.get("log_root") is None:
            raw_cfg.log_root = str(LOG_ROOT)

    benchmark = raw_cfg.get("benchmark", {})
    models_root = Path(str(benchmark.get("models_root", DEFAULT_MODELS_ROOT))).expanduser().resolve()
    checkpoints = _discover_checkpoints(models_root)
    scenarios = _scenarios(raw_cfg)
    output_dir = _output_dir(raw_cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Models root: {models_root}")
    print(f"Output dir: {output_dir}")
    print(f"Scenarios ({len(scenarios)}): {scenarios}")
    print("Selected checkpoints:")
    for version, checkpoint in checkpoints:
        print(f"  {version}: {checkpoint}")

    (output_dir / "scenarios.json").write_text(
        json.dumps({"scenarios": scenarios}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "checkpoints.json").write_text(
        json.dumps(
            [
                {"version": version, "checkpoint": str(checkpoint)}
                for version, checkpoint in checkpoints
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for version, checkpoint in checkpoints:
        rows, summary = _evaluate_checkpoint(raw_cfg, version, checkpoint, scenarios)
        all_rows.extend(rows)
        summaries.append(summary)
        _write_csv(output_dir / f"episodes_{version}.csv", rows)
        (output_dir / f"summary_{version}.json").write_text(
            json.dumps(summary, indent=2, allow_nan=True),
            encoding="utf-8",
        )

    _write_csv(output_dir / "episodes.csv", all_rows)
    _write_csv(output_dir / "summary.csv", summaries)
    (output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    print()
    print("Benchmark complete.")
    print(f"Episodes CSV: {output_dir / 'episodes.csv'}")
    print(f"Summary CSV : {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
