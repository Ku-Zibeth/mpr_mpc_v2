"""V2 replay and episodic constraint-cost semantics."""

from __future__ import annotations

import numpy as np


class V2TrainingSemantics:
    version = "lattice_tdmpc2_v2"
    replay_extra_specs = {
        "residual_low": (2,),
        "residual_high": (2,),
        "unsafe_residual": (1,),
    }

    def __init__(self, residual_cfg):
        self.unsafe_residual_cost = float(residual_cfg.get("unsafe_residual_cost", 1.0))

    def transition(self, real_cost: float, result) -> tuple[float, dict]:
        replay_cost = real_cost + self.unsafe_residual_cost * float(result.unsafe_residual)
        extras = {
            "residual_low": np.asarray(result.bounds.low, dtype=np.float32),
            "residual_high": np.asarray(result.bounds.high, dtype=np.float32),
            "unsafe_residual": np.asarray([result.unsafe_residual], dtype=np.float32),
        }
        return replay_cost, extras

    @staticmethod
    def lagrangian_episode_cost(
        real_episode_cost: float, constraint_episode_cost: float, episode_length: int
    ) -> float:
        del real_episode_cost
        del episode_length
        return constraint_episode_cost

    @staticmethod
    def tracking_target_speed(path_target_speed: float, planner_target_speed: float) -> float:
        return float(path_target_speed if path_target_speed > 0.0 else planner_target_speed)
