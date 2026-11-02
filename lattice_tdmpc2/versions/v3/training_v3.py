"""V3 replay and episodic constraint-cost semantics."""

from __future__ import annotations


class V3TrainingSemantics:
    version = "lattice_tdmpc2_v3"
    replay_extra_specs = {}

    def __init__(self, residual_cfg):
        del residual_cfg

    @staticmethod
    def transition(real_cost: float, result) -> tuple[float, dict]:
        del result
        return real_cost, {}

    @staticmethod
    def lagrangian_episode_cost(
        real_episode_cost: float, constraint_episode_cost: float, episode_length: int
    ) -> float:
        del constraint_episode_cost
        del episode_length
        return real_episode_cost

    @staticmethod
    def tracking_target_speed(path_target_speed: float, planner_target_speed: float) -> float:
        del planner_target_speed
        return max(float(path_target_speed), 0.0)
