"""Shared whole-episode safety diagnostics for visual and frozen evaluations."""

from collections import defaultdict
from pathlib import Path
import csv
import json

import numpy as np
import torch
from omegaconf import OmegaConf
from mpr_mpc.evaluation_statistics import cluster_intervals
from mpr_mpc.planning.local_mppi import MPPI_DIAGNOSTIC_KEYS


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
            for key, value in result.metrics.items():
                self.plans.setdefault(key, [])
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
        return {
            **{f"mean_{key.replace('/', '_')}": mean(values) for key, values in self.plans.items()},
            **{name: mean(self.plans[key]) for name, key in PLAN_ALIASES.items()},
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
        if key in {"episode", "scenario"}:
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
