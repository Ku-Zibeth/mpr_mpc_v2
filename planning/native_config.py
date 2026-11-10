"""Restore native planner settings without resurrecting retired checkpoint modes."""

MPPI_SAFETY_KEYS = frozenset({
    "enabled", "substeps", "model_margin", "road_margin", "max_acceleration", "max_deceleration",
    "max_steering_rate", "max_lateral_acceleration", "swept_max_depth", "swept_angle_tolerance",
    "swept_position_tolerance", "hard_clearance", "hard_ttc", "ttc_lookahead", "emergency_throttle",
    "obstacle_acceleration_bound", "obstacle_velocity_error", "braking_distance_deceleration",
    "two_layer", "action_delay_s", "long_risk_coef", "staged", "center_bicycle", "center_model_guard", "motion_validation_path",
})
MPPI_RECOVERY_KEYS = frozenset({"enabled", "num_samples", "iterations", "initial_std", "max_delta",
                                "straighten_fractions", "throttle_levels", "lattice_guided", "positive_throttle",
                                "anti_stall", "stall_speed_mps", "stall_duration_s", "stall_progress_m", "stall_exit_s",
                                "progressive", "positive_throttles", "transition_fractions", "controlled_deceleration"})
MPPI_SELECTION_KEYS = frozenset({"enabled", "record_candidates", "min_progress_m", "normal_progress_ratio",
                                 "recovery_progress_ratio", "max_throttle_step", "brake_release_enabled",
                                 "brake_release_step", "throttle_increase_step", "throttle_decrease_step"})
NATIVE_MPPI_KEYS = frozenset({
    "enabled", "cost_enabled", "cost_coef", "num_samples", "num_elites", "iterations",
    "initial_std", "min_std", "max_std", "max_delta", "temperature", "proposal_init",
    "blend_alpha", "deviation_coef", "smoothness_coef", "throttle_deviation_weight",
    "min_improvement_abs", "min_improvement_ratio", "eval_seed", "lateral_corridor_ratio",
    "lateral_safety_margin", "kinematic_vehicle_width", "kinematic_wheelbase",
    "kinematic_max_steering_angle", "kinematic_max_acceleration", "kinematic_max_deceleration",
    "safety", "recovery", "allow_unvalidated_cost", "trajectory_selection",
})


def native_mppi_config(saved):
    result = {key: value for key, value in saved.items() if key in NATIVE_MPPI_KEYS}
    if "safety" in result:
        result["safety"] = {key: value for key, value in result["safety"].items() if key in MPPI_SAFETY_KEYS}
    if "recovery" in result:
        result["recovery"] = {key: value for key, value in result["recovery"].items() if key in MPPI_RECOVERY_KEYS}
    if "trajectory_selection" in result:
        result["trajectory_selection"] = {key: value for key, value in result["trajectory_selection"].items() if key in MPPI_SELECTION_KEYS}
    return result
