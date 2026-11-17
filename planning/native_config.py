"""Migrate legacy checkpoint planner metadata to the single MPPI contract."""

NATIVE_MPPI_KEYS = frozenset({
    "enabled", "num_samples", "num_elites", "iterations", "initial_std", "min_std", "max_std",
    "temperature", "eval_seed", "kinematic_vehicle_width", "kinematic_wheelbase",
    "kinematic_max_steering_angle",
    "noise_rho",
    "temperature_definition", "terminal_action", "q_aggregation",
    "terminal_q_weight",
    "terminal_value", "sac_value_checkpoint", "audit_candidates", "record_cost", "cost_weight",
})


def native_mppi_config(saved):
    # Retired recovery, safety and scoring flags cannot resurrect old planners.
    return {key: value for key, value in saved.items() if key in NATIVE_MPPI_KEYS}
