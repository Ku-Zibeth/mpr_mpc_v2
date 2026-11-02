"""Map normalized SAC actions linearly into physical trajectory residuals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ACTION_MAPPING_VERSION = "symmetric_coarse_speed_v2"
LANE_WIDTH_ACTION_MAPPING_VERSION = "lane_width_coarse_speed_v4"
SUPPORTED_ACTION_MAPPINGS = {
    ACTION_MAPPING_VERSION,
    LANE_WIDTH_ACTION_MAPPING_VERSION,
}


@dataclass(frozen=True)
class ResidualBounds:
    low: np.ndarray
    high: np.ndarray

    def __post_init__(self):
        if self.low.shape != (2,) or self.high.shape != (2,):
            raise ValueError("Residual bounds must each have shape [2].")
        if np.any(self.low > self.high):
            raise ValueError("Residual lower bounds exceed upper bounds.")


class ResidualActionAdapter:
    def __init__(self, config):
        mapping = str(config.get("action_mapping", ACTION_MAPPING_VERSION))
        if mapping not in SUPPORTED_ACTION_MAPPINGS:
            raise ValueError(
                f"Unsupported residual action mapping {mapping!r}; expected one of "
                f"{sorted(SUPPORTED_ACTION_MAPPINGS)}."
            )
        self.mapping = mapping
        self.delta_d_max = abs(float(config["delta_d_max"]))

    def bounds(self, coarse_path, lattice_planner) -> ResidualBounds:
        coarse_v = max(float(coarse_path.target_speed), 0.0)
        if self.mapping == LANE_WIDTH_ACTION_MAPPING_VERSION:
            lateral_limit = abs(float(lattice_planner.last_lane_width))
            if not np.isfinite(lateral_limit) or lateral_limit <= 0.0:
                raise RuntimeError("V4 residual mapping requires a positive Lattice lane width.")
        else:
            lateral_limit = self.delta_d_max
        high = np.asarray([lateral_limit, coarse_v], dtype=np.float32)
        low = -high
        return ResidualBounds(low=low, high=high)

    @staticmethod
    def to_physical(normalized_action, bounds: ResidualBounds) -> np.ndarray:
        action = np.asarray(normalized_action, dtype=np.float32)
        if action.shape[-1] != 2:
            raise ValueError(f"Residual SAC action must end in dimension 2, got {action.shape}.")
        clipped = np.clip(action, -1.0, 1.0)
        return clipped * bounds.high
