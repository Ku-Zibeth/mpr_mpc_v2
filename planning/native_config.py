"""Restore native planner settings without resurrecting retired checkpoint modes."""

NATIVE_MPPI_KEYS = frozenset({
    "enabled", "cost_enabled", "cost_coef", "num_samples", "num_elites", "iterations",
    "initial_std", "min_std", "max_std", "max_delta", "temperature", "proposal_init",
    "blend_alpha", "deviation_coef", "smoothness_coef", "throttle_deviation_weight",
    "min_improvement_abs", "min_improvement_ratio", "eval_seed", "lateral_corridor_ratio",
    "lateral_safety_margin", "kinematic_vehicle_width", "kinematic_wheelbase",
    "kinematic_max_steering_angle", "kinematic_max_acceleration", "kinematic_max_deceleration",
})


def native_mppi_config(saved):
    return {key: value for key, value in saved.items() if key in NATIVE_MPPI_KEYS}
