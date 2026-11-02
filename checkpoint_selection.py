"""Versioned, finite-metric safety-first selection with progress feasibility."""

import math

DEFAULT_PRIORITY = ("collision_rate", "out_of_road_rate", "average_safety_cost",
                    "success_rate", "route_completion", "episode_reward")
MINIMIZE = set(DEFAULT_PRIORITY[:3])


def select_checkpoint(metrics, previous, cfg):
    priority = tuple(cfg.get("priority", DEFAULT_PRIORITY))
    if len(priority) != len(DEFAULT_PRIORITY) or set(priority) != set(DEFAULT_PRIORITY):
        raise ValueError("checkpoint_selection.priority must contain each of the six metrics exactly once.")
    floors = dict(min_success_rate=float(cfg.get("min_success_rate", .05)),
                  min_route_completion=float(cfg.get("min_route_completion", .1)),
                  max_success_drop=float(cfg.get("max_success_drop", .1)),
                  max_route_completion_drop=float(cfg.get("max_route_completion_drop", .1)))
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in floors.values()):
        raise ValueError("Checkpoint feasibility thresholds must be finite fractions in [0,1].")
    schema = dict(version=2, priority=list(priority), **floors)
    for key in priority:
        if key not in metrics or not math.isfinite(float(metrics[key])):
            return False, previous, f"ineligible: missing/nonfinite {key}"
    if metrics["success_rate"] < floors["min_success_rate"] or metrics["route_completion"] < floors["min_route_completion"]:
        return False, previous, "ineligible: success/route feasibility floor"
    values = {k: float(metrics[k]) for k in priority}
    progress_keys = ("success_rate", "route_completion")
    candidate = dict(schema=schema, metrics=values,
                     progress_reference={k: values[k] for k in progress_keys})
    if not previous or previous.get("schema") != schema:
        return True, candidate, "first eligible result under current selection schema"
    for key, limit in zip(progress_keys, (floors["max_success_drop"], floors["max_route_completion_drop"])):
        reference = previous.get("progress_reference", previous.get("metrics", {})).get(key)
        if reference is None or not math.isfinite(float(reference)):
            return True, candidate, "replaces invalid historical progress metadata"
        if values[key] < float(reference) - limit:
            return False, previous, f"ineligible: excessive {key} drop from accepted-best high-water mark"
        candidate["progress_reference"][key] = max(float(reference), values[key])
    for key in priority:
        old = previous.get("metrics", {}).get(key)
        if old is None or not math.isfinite(float(old)):
            return True, candidate, "replaces invalid historical selection metadata"
        if values[key] == old:
            continue
        better = values[key] < old if key in MINIMIZE else values[key] > old
        return better, candidate if better else previous, ("improved " if better else "worse ") + key
    return False, previous, "tie"
