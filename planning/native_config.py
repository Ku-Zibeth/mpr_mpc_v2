"""Restore native planner settings without resurrecting retired checkpoint modes."""

MPPI_SAFETY_KEYS = frozenset({
    "enabled", "substeps", "model_margin", "road_margin", "max_acceleration", "max_deceleration",
    "max_steering_rate", "max_lateral_acceleration", "swept_max_depth", "swept_angle_tolerance",
    "swept_position_tolerance", "hard_clearance", "hard_ttc", "ttc_lookahead", "emergency_throttle",
})
NATIVE_MPPI_KEYS = frozenset({
    "enabled", "cost_enabled", "cost_coef", "num_samples", "num_elites", "iterations",
    "initial_std", "min_std", "max_std", "max_delta", "temperature", "proposal_init",
    "blend_alpha", "deviation_coef", "smoothness_coef", "throttle_deviation_weight",
    "min_improvement_abs", "min_improvement_ratio", "eval_seed", "lateral_corridor_ratio",
    "lateral_safety_margin", "kinematic_vehicle_width", "kinematic_wheelbase",
    "kinematic_max_steering_angle", "kinematic_max_acceleration", "kinematic_max_deceleration",
    "safety",
})


def native_mppi_config(saved):
    result = {key: value for key, value in saved.items() if key in NATIVE_MPPI_KEYS}
    if "safety" in result:
        result["safety"] = {key: value for key, value in result["safety"].items() if key in MPPI_SAFETY_KEYS}
    return result
