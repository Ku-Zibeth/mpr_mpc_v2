"""Migrate legacy checkpoint planner metadata to the single MPPI contract."""

NATIVE_MPPI_KEYS = frozenset({
    "enabled", "num_samples", "num_elites", "iterations", "initial_std", "min_std", "max_std",
    "max_delta", "temperature", "eval_seed", "kinematic_vehicle_width", "kinematic_wheelbase",
    "kinematic_max_steering_angle",
    "noise_rho", "allow_unvalidated_cost", "safety",
})


def native_mppi_config(saved):
    # Retired recovery, safety and scoring flags cannot resurrect old planners.
    return {key: value for key, value in saved.items() if key in NATIVE_MPPI_KEYS}
