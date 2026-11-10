"""Shared whole-episode safety diagnostics for visual and frozen evaluations."""

from collections import defaultdict
from pathlib import Path
import csv
import json

import numpy as np
import torch
from omegaconf import OmegaConf
from mpr_mpc.evaluation_statistics import cluster_intervals
from mpr_mpc.planning.local_mppi import MPPI_DIAGNOSTIC_KEYS, MPPI_EPISODE_METRICS


PLAN_ALIASES = {
    "sac_predicted_cost": "mpr/sac_predicted_cost",
    "mppi_predicted_cost": "mpr/mppi_predicted_cost",
    "safe_candidate_ratio": "mpr/safe_candidate_ratio",
    "collision_reject_rate": "mpr/collision_reject_rate",
    "offroad_reject_rate": "mpr/offroad_reject_rate",
    "baseline_selected_rate": "mpr/baseline_selected",
    "planner_score_gain": "mpr/planner_score_gain",
    "mppi_correction_magnitude": "mpr/mppi_correction_magnitude",
    "no_safe_candidate_rate": "mpr/no_safe_candidate",
}


def prediction_outcomes(info, result):
    """Compare the executed transition to the first-control prediction only.

    A collision anywhere in the H-step forecast is a different statistic and
    must not be used as the predicted label for the next environment step.
    """
    if result is None or not result.stage.use_mppi:
        return {}
    crash = any(bool(info.get(k, False)) for k in
                ("crash", "crash_vehicle", "crash_object", "crash_building", "crash_human", "crash_sidewalk"))
    offroad = bool(info.get("out_of_road", False))
    metrics = {"mppi/actual_collision": float(crash), "mppi/actual_offroad": float(offroad)}
    for name, actual in (("collision", crash), ("offroad", offroad)):
        prediction = result.metrics.get(f"mppi/predicted_first_{name}", float("nan"))
        known = prediction is not None and np.isfinite(prediction)
        metrics[f"mppi/predicted_safe_actual_{name}"] = float(not prediction and actual) if known else float("nan")
        metrics[f"mppi/prediction_{name}_known"] = float(known)
    return metrics


class StepDiagnosticsLogger:
    """Optional per-step evidence for the existing evaluation entry points."""

    def __init__(self, directory, name, *, wandb_cfg=None, compact=False, stride=1):
        self.stream = self.run = None
        self.upload_failed = False
        self.compact, self.stride = bool(compact), max(1, int(stride))
        self._last_state = None
        if directory is not None:
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            self.stream = (directory / f"{name}.jsonl").open("a", encoding="utf-8")
        if wandb_cfg is not None:
            try:
                import wandb
                self.run = wandb.init(project=str(wandb_cfg.wandb_project), entity=wandb_cfg.wandb_entity,
                    name=name, job_type="evaluation", dir=str(directory),
                    config=OmegaConf.to_container(wandb_cfg, resolve=True),
                    settings=wandb.Settings(console="off", init_timeout=60))
            except Exception as error:
                print(f"[evaluation] W&B unavailable: {error}; local step evidence continues.", flush=True)

    def log(self, scenario, step, action, result, info=None):
        if self.stream is None and self.run is None:
            return
        state = (int(scenario), getattr(result, "planning_mode", "unknown"))
        incident = info is not None and any(bool(info.get(k, False)) for k in
            ("crash", "crash_vehicle", "crash_object", "out_of_road", "success", "truncated"))
        changed = state != self._last_state
        self._last_state = state
        if self.compact and not (changed or incident or step == 1 or step % self.stride == 0):
            return
        row = dict(scenario=int(scenario), episode_step=int(step),
                   executed_action=np.asarray(action.detach().cpu() if torch.is_tensor(action) else action).tolist())
        if result is not None:
            row.update({k: v for k, v in result.metrics.items() if not self.compact or
                (k.startswith("mppi/") and not k.endswith("cpu_wall_ms")) or k in
                {"mpr/actual_speed_mps", "mpr/coarse_target_speed_mps", "mpr/refined_target_speed_mps", "mpr/total_planning_ms"}})
            row.update(prediction_outcomes(info, result) if info is not None else {})
            row.update(residual_enabled=bool(result.stage.use_residual), mppi_enabled=bool(result.stage.use_mppi))
            row["planning_mode"] = getattr(result, "planning_mode", "unknown")
            row["recovery_reason"] = getattr(result, "recovery_reason", "unknown")
        if info is not None:
            row["route_completion"] = float(info.get("route_completion", 0.))
        row = {key: None if isinstance(value, (float, np.floating)) and not np.isfinite(value) else value
               for key, value in row.items()}
        if self.stream is not None:
            self.stream.write(json.dumps(row, allow_nan=False) + "\n")
            self.stream.flush()
        if self.run is not None and not self.upload_failed:
            try:
                self.run.log(row)
            except Exception as error:
                print(f"[evaluation] W&B step upload failed: {error}; local step evidence continues.", flush=True)
                self.upload_failed = True

    def close(self):
        if self.stream is not None:
            self.stream.close()
        if self.run is not None:
            try:
                self.run.finish()
            except Exception as error:
                print(f"[evaluation] W&B final sync failed: {error}", flush=True)


def fixed_scenarios(cfg):
    simulator = cfg.metadrive["simulator"]
    start, count = int(simulator["start_seed"]), int(simulator["num_scenarios"])
    episodes = int(cfg.eval_episodes)
    fixed = list(cfg.get("evaluation_scenarios", []))
    if cfg.get("evaluation_scenarios_file"):
        payload = json.loads(Path(str(cfg.evaluation_scenarios_file)).expanduser().read_text())
        fixed = payload.get("scenarios", []) if isinstance(payload, dict) else payload
    if fixed:
        if len(fixed) != episodes or len(set(fixed)) != episodes:
            raise ValueError("evaluation_scenarios must contain exactly eval_episodes unique IDs.")
        if any(type(seed) is not int or not start <= seed < start + count for seed in fixed):
            raise ValueError("Scenario IDs must be integers inside the configured simulator seed range.")
        return np.asarray(fixed, dtype=int)
    if not 0 < episodes <= count:
        raise ValueError("Require 0 < eval_episodes <= num_scenarios; sampling with replacement is disabled.")
    return np.random.default_rng(int(cfg.seed)).choice(np.arange(start, start + count), episodes, replace=False)


class EpisodeDiagnostics:
    def __init__(self):
        self.crash = self.offroad = False
        self.ttcs, self.clearances = [], []
        self.plans = defaultdict(list, {f"mpr/{key}": [] for key in MPPI_DIAGNOSTIC_KEYS})
        self.plans.update({key: [] for key in MPPI_EPISODE_METRICS.values()})
        self.sums = defaultdict(float)
        self.observations, self.actions, self.costs = [], [], []
        self.speeds = []
        self.distance = 0.0

    def observe(self, observation, action, info, result, vehicle=None):
        self.observations.append(np.asarray(observation.detach().cpu() if torch.is_tensor(observation) else observation, dtype=np.float32).copy())
        self.actions.append(np.asarray(action.detach().cpu() if torch.is_tensor(action) else action, dtype=np.float32).copy())
        self.costs.append(float(info.get("cost", 0)))
        if vehicle is not None:
            self.speeds.append(float(np.linalg.norm(np.asarray(vehicle.velocity)[:2])))
            if hasattr(vehicle, "last_position"):
                self.distance += float(np.linalg.norm(np.asarray(vehicle.position)[:2] - np.asarray(vehicle.last_position)[:2]))
        self.crash |= any(bool(info.get(key, False)) for key in
            ("crash", "crash_vehicle", "crash_object", "crash_building", "crash_human", "crash_sidewalk"))
        self.offroad |= bool(info.get("out_of_road", False))
        ttc = float(info.get("risk_field_min_ttc", float("nan")))
        if np.isfinite(ttc):
            self.ttcs.append(ttc)
        for key in ("risk_field_boundary_cost", "risk_field_vehicle_cost", "risk_field_object_cost"):
            self.sums[key] += float(info.get(key, 0))
        for key in ("dynamic_collision", "static_collision", "road_event", "other_event", "unallocated_event"):
            self.sums[f"safety_{key}_cost"] += float(info.get(f"safety_{key}_cost", 0))
        if result is not None:
            result.metrics.update(prediction_outcomes(info, result))
            code = result.metrics.get("mppi/planning_mode", float("nan"))
            if result.stage.use_mppi and np.isfinite(code):
                self.sums[f"mode{int(code)}_steps"] += 1
                self.sums[f"mode{int(code)}_collision_steps"] += result.metrics["mppi/actual_collision"]
                self.sums[f"mode{int(code)}_offroad_steps"] += result.metrics["mppi/actual_offroad"]
            predicted_step = float(result.metrics.get("mppi/predicted_step_cost", float("nan")))
            if result.stage.use_mppi and np.isfinite(predicted_step):
                self.sums["predicted_step_cost"] += predicted_step
                self.sums["predicted_cost_steps"] += 1
            for key, value in result.metrics.items():
                self.plans.setdefault(key, [])
                if key.startswith("mppi/") and not result.stage.use_mppi:
                    continue
                if key == "mpr/sac_predicted_cost" and not result.stage.use_residual:
                    continue
                if key in set(PLAN_ALIASES.values()) - {"mpr/sac_predicted_cost"} and not result.stage.use_mppi:
                    continue
                if np.isfinite(value):
                    self.plans[key].append(float(value))
            clearance = result.metrics.get("mpr/mppi_min_clearance", float("nan"))
            if np.isfinite(clearance):
                self.clearances.append(clearance)

    def trajectory(self):
        return (np.asarray(self.observations), np.asarray(self.actions), np.asarray(self.costs))

    def summary(self, info, agent=None):
        mean = lambda values: float(np.mean(values)) if values else float("nan")
        sac = getattr(agent, "residual_sac", None)
        planning = self.plans.get("mpr/total_planning_ms", [])
        legacy = {"mppi_baseline_selected": "mpr/baseline_selected", "mppi_score_gain": "mpr/planner_score_gain"}
        return {
            **{f"planning_p{q}_ms": float(np.percentile(planning, q)) if planning else float("nan")
               for q in (50, 95, 99)},
            **{f"{name}_p{q}": float(np.percentile(self.plans[f"mppi/{name}"], q))
                if self.plans[f"mppi/{name}"] else float("nan")
               for name in ("physics_position_error_m", "physics_speed_error_mps", "physics_heading_error_rad",
                    "physics_acceleration_bias_mps2", "physics_braking_response_bias_mps2", "execution_action_error")
               for q in (50, 95, 99)},
            **{f"mean_{key.replace('/', '_')}": mean(values) for key, values in self.plans.items()},
            **{name: mean(self.plans[key]) for name, key in PLAN_ALIASES.items()},
            **{name: mean(self.plans[key] or self.plans.get(legacy.get(name, ""), []))
               for name, key in MPPI_EPISODE_METRICS.items()},
            **{f"mppi_{name}": mean(self.plans[f"mppi/{name}"]) for name in
                ("actual_collision", "actual_offroad", "predicted_safe_actual_collision", "predicted_safe_actual_offroad",
                 "prediction_collision_known", "prediction_offroad_known")},
            **{f"mppi_mode_{name}_rate": mean([float(v == code) for v in self.plans["mppi/planning_mode"]])
                for code, name in enumerate(("baseline", "normal", "recovery_lattice", "recovery_controlled",
                                             "lattice_fallback", "emergency_model_checked", "emergency_model_check_failed"))},
            **{f"mppi_mode_{name}_{event}_steps": (self.sums[f"mode{code}_{event}_steps"]
                if self.plans["mppi/planning_mode"] else float("nan"))
               for code, name in enumerate(("baseline", "normal", "recovery_lattice", "recovery_controlled",
                    "lattice_fallback", "emergency_model_checked", "emergency_model_check_failed"))
               for event in ("collision", "offroad")},
            "collision": float(self.crash), "offroad": float(self.offroad),
            "collision_and_success_rate": float(self.crash and bool(info.get("success", False))),
            "collision_and_offroad_rate": float(self.crash and self.offroad),
            "mean_ttc": mean(self.ttcs), "min_ttc": min(self.ttcs) if self.ttcs else float("nan"),
            "ttc_valid_steps": len(self.ttcs),
            "min_predicted_obstacle_clearance": min(self.clearances) if self.clearances else float("nan"),
            "min_obstacle_clearance": min(self.clearances) if self.clearances else float("nan"),
            "road_boundary_cost": self.sums["risk_field_boundary_cost"],
            "dynamic_risk_field_cost": self.sums["risk_field_vehicle_cost"],
            "static_risk_field_cost": self.sums["risk_field_object_cost"],
            "dynamic_collision_cost": self.sums["safety_dynamic_collision_cost"],
            "static_collision_cost": self.sums["safety_static_collision_cost"],
            "road_event_cost": self.sums["safety_road_event_cost"],
            "other_event_cost": self.sums["safety_other_event_cost"],
            "unallocated_event_cost": self.sums["safety_unallocated_event_cost"],
            "episode_cost": float(sum(self.costs)),
            "success": float(info.get("success", 0)),
            "terminated": float(info.get("terminated", 0)),
            "truncated": float(info.get("truncated", 0)),
            "end_reason": str(info.get("end_reason", "")),
            "episode_length": len(self.actions),
            "route_completion": float(info.get("route_completion", 0)),
            # Sum only first-step predictions along executed controls. Summing
            # overlapping H-step forecasts would count future cost repeatedly.
            "total_predicted_cost": self.sums["predicted_step_cost"] if self.sums["predicted_cost_steps"] == len(self.actions) and self.actions else float("nan"),
            "total_actual_cost": float(sum(self.costs)),
            "timeout": float(bool(info.get("truncated", False)) or bool(info.get("max_step", False))),
            "timeout_rate": float(bool(info.get("truncated", False)) or bool(info.get("max_step", False))),
            "mean_speed_mps": mean(self.speeds), "distance_travelled_m": self.distance,
            "lagrangian_value": float(sac.lagrangian_multiplier) if sac is not None else float("nan"),
            "cost_limit": float(sac.cost_limit) if sac is not None else float("nan"),
            # No optimizer runs at evaluation; training-only losses are unavailable.
            "actor_safety_loss": float("nan"), "cost_critic_loss": float("nan"),
        }


def write_report(raw_cfg, scenarios, rows, cost_metrics=None):
    out = Path(str(raw_cfg.get("evaluation_output_dir") or (Path(raw_cfg.work_dir) / "evaluation")))
    out.mkdir(parents=True, exist_ok=True)
    (out / "scenarios.json").write_text(json.dumps([int(s) for s in scenarios], indent=2))
    OmegaConf.save(raw_cfg, out / "config.yaml")
    if not rows:
        return out
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (out / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    summary = {"episodes": len(rows), **(cost_metrics or {})}
    for key in keys:
        if key in {"episode", "scenario", "end_reason"}:
            continue
        values = [float(row.get(key, float("nan"))) for row in rows]
        values = [value for value in values if np.isfinite(value)]
        summary[key] = float(np.mean(values)) if values else None
    for source, target in (("success", "success_rate"), ("offroad", "out_of_road_rate"),
                           ("collision", "collision_rate"), ("cost", "average_safety_cost")):
        summary[target] = summary.get(source)
    summary.update(cluster_intervals(rows, {
        "success": ("success", None), "crash": ("collision", None),
        "out_of_road": ("offroad", None), "route_completion": ("route_completion", None),
        "collision_and_success": ("collision_and_success_rate", None),
        "collision_and_offroad": ("collision_and_offroad_rate", None),
    }, samples=2000, seed=int(raw_cfg.seed)))
    finite_ttc = [float(row["min_ttc"]) for row in rows if np.isfinite(row.get("min_ttc", float("nan")))]
    summary["mean_episode_min_ttc"] = summary.get("min_ttc")
    summary["min_ttc"] = min(finite_ttc) if finite_ttc else None
    clearances = [float(row["min_obstacle_clearance"]) for row in rows
                  if np.isfinite(row.get("min_obstacle_clearance", float("nan")))]
    summary["min_obstacle_clearance"] = min(clearances) if clearances else None
    valid_steps = sum(int(row.get("ttc_valid_steps", 0)) for row in rows)
    summary["ttc_valid_step_fraction"] = valid_steps / max(sum(int(row["length"]) for row in rows), 1)
    summary["mean_episode_mean_ttc"] = summary.get("mean_ttc")
    summary["mean_ttc"] = (sum(float(row["mean_ttc"]) * int(row["ttc_valid_steps"]) for row in rows
                                if int(row.get("ttc_valid_steps", 0))) / valid_steps) if valid_steps else None
    clean = {key: None if isinstance(value, (float, np.floating)) and not np.isfinite(value) else value
             for key, value in summary.items()}
    (out / "summary.json").write_text(json.dumps(clean, indent=2, allow_nan=False))
    return out
