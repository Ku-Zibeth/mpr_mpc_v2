"""Scenario-clustered summaries for repeated, paired evaluation runs."""

from __future__ import annotations

import math

import numpy as np


METRICS = {
    "out_of_road": ("episode_out_of_road", None),
    "crash": ("episode_crash", None),
    "route_completion": ("final_route_completion", None),
    "success": ("final_success", None),
    "collision_and_success": ("collision_and_success", None),
    "collision_and_offroad": ("collision_and_offroad", None),
    "collision_given_success": ("collision_and_success", "final_success"),
    "offroad_given_collision": ("collision_and_offroad", "episode_crash"),
    "mean_ttc": ("mean_ttc", None),
    "mean_episode_minimum_ttc": ("minimum_ttc", None),
    "cost_per_episode": ("sum_cost", None),
    "cost_limit_exceedance": ("episode_cost_over_limit", None),
    "mppi_baseline_selected": ("mppi_baseline_selected", None),
    "mppi_score_gain": ("mppi_score_gain", None),
    "safe_candidate_ratio": ("safe_candidate_ratio", None),
    "collision_reject_rate": ("collision_reject_rate", None),
    "offroad_reject_rate": ("offroad_reject_rate", None),
    "baseline_selected_rate": ("baseline_selected_rate", None),
    "planner_score_gain": ("planner_score_gain", None),
    "mppi_correction_magnitude": ("mppi_correction_magnitude", None),
    "no_safe_candidate_rate": ("no_safe_candidate_rate", None),
    "sac_predicted_cost": ("sac_predicted_cost", None),
    "mppi_predicted_cost": ("mppi_predicted_cost", None),
    "dynamic_collision_cost": ("dynamic_collision_cost", None),
    "static_collision_cost": ("static_collision_cost", None),
    "mean_speed_mps": ("mean_speed_mps", None),
    "timeout_rate": ("timeout", None),
    "planning_time_ms": ("mean_mpr_total_planning_ms", None),
    "predicted_cost": ("total_predicted_cost", None),
    "actual_cost": ("total_actual_cost", None),
    "safety_failure_rate": ("safety_failure", None),
    "unverified_action_executions": ("unverified_action_executions", None),
}
BINARY_RATES = {
    "out_of_road", "crash", "success", "collision_and_success",
    "collision_and_offroad", "cost_limit_exceedance", "timeout_rate", "safety_failure_rate",
}


def _number(row: dict, key: str) -> float:
    value = row.get(key)
    return float(value) if value is not None else float("nan")


def _nullable(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _metric_definitions(rows):
    diagnostics = {key: (key, None) for row in rows for key, value in row.items()
                   if key.startswith("mppi_") and isinstance(value, (int, float, np.number))}
    return {**METRICS, **diagnostics}


def cluster_intervals(rows: list[dict], metrics: dict, *, samples: int, seed: int) -> dict:
    if not rows or samples <= 0:
        raise ValueError("Confidence intervals require observations and positive bootstrap_samples.")
    scenarios = sorted({int(row["scenario"]) for row in rows})
    index = {scenario: i for i, scenario in enumerate(scenarios)}
    totals = np.zeros((len(scenarios), len(metrics)))
    denominators = np.zeros_like(totals)
    for row in rows:
        i = index[int(row["scenario"])]
        for j, (numerator, denominator) in enumerate(metrics.values()):
            value = _number(row, numerator)
            weight = _number(row, denominator) if denominator else 1.0
            if math.isfinite(value) and math.isfinite(weight):
                totals[i, j] += value
                denominators[i, j] += weight
    point = np.divide(
        totals.sum(axis=0), denominators.sum(axis=0),
        out=np.full(len(metrics), np.nan), where=denominators.sum(axis=0) > 0,
    )
    rng = np.random.default_rng(seed)
    estimates = np.full((samples, len(metrics)), np.nan)
    # All repeats of a scenario travel together; repeated trials are not new maps.
    for start in range(0, samples, 200):
        count = min(200, samples - start)
        draw = rng.integers(len(scenarios), size=(count, len(scenarios)))
        numerator = totals[draw].sum(axis=1)
        denominator = denominators[draw].sum(axis=1)
        estimates[start:start + count] = np.divide(
            numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0,
        )
    result = {}
    for j, name in enumerate(metrics):
        finite = estimates[:, j][np.isfinite(estimates[:, j])]
        bounds = np.percentile(finite, [2.5, 97.5]) if finite.size else [np.nan, np.nan]
        if name in BINARY_RATES and point[j] in (0.0, 1.0):
            # Bound the chance of a nonzero cluster, conservatively using unique maps.
            count = np.count_nonzero(denominators[:, j])
            endpoint = 0.025 ** (1.0 / count)
            bounds = [0.0, 1.0 - endpoint] if point[j] == 0.0 else [endpoint, 1.0]
        result[name] = _nullable(point[j])
        result[f"{name}_ci95_low"] = _nullable(bounds[0])
        result[f"{name}_ci95_high"] = _nullable(bounds[1])
    return result


def summarize_runs(label: str, rows: list[dict], *, samples: int, seed: int) -> dict:
    keys = [(int(r["scenario"]), int(r["test_seed"])) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate (scenario, test_seed) observations in one configuration.")
    result = {
        "configuration": label,
        "episodes": len(rows),
        "unique_scenarios": len({k[0] for k in keys}),
        "test_seeds": ",".join(str(s) for s in sorted({k[1] for k in keys})),
        **cluster_intervals(rows, _metric_definitions(rows), samples=samples, seed=seed),
    }
    for name in ("collision_and_success", "collision_and_offroad"):
        result[f"{name}_count"] = int(sum(_number(r, name) for r in rows))
        result[f"{name}_unique_scenarios"] = len({
            int(r["scenario"]) for r in rows if _number(r, name) > 0
        })
    ttc = [_number(r, "minimum_ttc") for r in rows]
    finite = [value for value in ttc if math.isfinite(value)]
    result["minimum_ttc"] = min(finite) if finite else None
    result["ttc_valid_episodes"] = len(finite)
    result["ttc_valid_step_fraction"] = (
        sum(_number(r, "ttc_valid_steps") for r in rows)
        / max(sum(_number(r, "length") for r in rows), 1)
    )
    limits = [_number(r, "cost_limit") for r in rows]
    result["cost_limit"] = next((v for v in limits if math.isfinite(v)), None)
    for source, target in (("success", "success_rate"), ("out_of_road", "out_of_road_rate"),
                           ("crash", "collision_rate"), ("cost_per_episode", "average_safety_cost"),
                           ("collision_and_success", "collision_and_success_rate"),
                           ("collision_and_offroad", "collision_and_offroad_rate")):
        result[target] = result[source]
    clearance = [_number(row, "min_obstacle_clearance") for row in rows]
    finite_clearance = [number for number in clearance if math.isfinite(number)]
    result["min_obstacle_clearance"] = min(finite_clearance) if finite_clearance else None
    return result


def paired_comparisons(rows: list[dict], *, samples: int, seed: int, pairs=None) -> list[dict]:
    groups = {}
    for row in rows:
        groups.setdefault(row["configuration"], {})[
            (int(row["scenario"]), int(row["test_seed"]))
        ] = row
    pairs = pairs if pairs is not None else (
        ("A_lattice", "B_lattice_sac"),
        ("A_lattice", "C_lattice_mppi"),
        ("B_lattice_sac", "D_lattice_sac_mppi"),
    )
    metrics = {name: source for name, (source, denominator) in _metric_definitions(rows).items() if denominator is None}
    results = []
    for baseline, candidate in pairs:
        a, b = groups.get(baseline, {}), groups.get(candidate, {})
        common = sorted(a.keys() & b.keys())
        if not common:
            continue
        differences = [
            {"scenario": scenario, **{
                name: _number(b[scenario, test_seed], source) - _number(a[scenario, test_seed], source)
                for name, source in metrics.items()
            }}
            for scenario, test_seed in common
        ]
        result = {
            "comparison": f"{candidate} - {baseline}",
            "paired_episodes": len(common),
            "unique_scenarios": len({k[0] for k in common}),
            **cluster_intervals(
                differences, {f"{name}_delta": (name, None) for name in metrics},
                samples=samples, seed=seed,
            ),
        }
        result["success_rescued_count"] = sum(
            _number(a[key], "final_success") == 0 and _number(b[key], "final_success") == 1
            for key in common
        )
        result["success_regressed_count"] = sum(
            _number(a[key], "final_success") == 1 and _number(b[key], "final_success") == 0
            for key in common
        )
        results.append(result)
    return results
