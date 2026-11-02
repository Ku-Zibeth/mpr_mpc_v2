"""V4 replay semantics using frozen world-model cost-to-go supervision."""

from __future__ import annotations

import math


class V4TrainingSemantics:
    version = "lattice_tdmpc2_v4"
    replay_extra_specs = {}

    def __init__(self, residual_cfg):
        source = str(residual_cfg.get("constraint_cost_source", "world_model_cost_to_go"))
        if source != "world_model_cost_to_go":
            raise ValueError("V4 requires constraint_cost_source=world_model_cost_to_go.")

    @staticmethod
    def transition(real_cost: float, result) -> tuple[float, dict]:
        del real_cost
        predicted_cost = float(result.refined_cost)
        if not math.isfinite(predicted_cost):
            raise RuntimeError("V4 received a non-finite refined world-model cost.")
        return max(predicted_cost, 0.0), {}

    @staticmethod
    def lagrangian_episode_cost(
        real_episode_cost: float, constraint_episode_cost: float, episode_length: int
    ) -> float:
        del real_episode_cost
        return constraint_episode_cost / max(int(episode_length), 1)

    @staticmethod
    def tracking_target_speed(path_target_speed: float, planner_target_speed: float) -> float:
        del planner_target_speed
        return max(float(path_target_speed), 0.0)
