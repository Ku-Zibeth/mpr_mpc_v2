"""Convert geometric Frenet paths to TD-MPC2-compatible action sequences."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Sequence

import numpy as np
import torch

from lattice.frenet_metadrive import FrenetPath, MetaDriveFrenetController


def _interp_angle(times: np.ndarray, source_times: np.ndarray, angles: np.ndarray) -> np.ndarray:
    return np.interp(times, source_times, np.unwrap(angles))


class LatticeActionAdapter:
    """Reference-action approximation using the existing Lattice tracker."""

    def __init__(self, controller: MetaDriveFrenetController, action_dim: int = 2):
        if int(action_dim) != 2:
            raise ValueError(
                "Lattice tracker emits [steering, throttle_brake], so TD-MPC2 action_dim must be 2."
            )
        self.controller = controller
        self.action_dim = 2

    @staticmethod
    def validate_action_space(action_space) -> None:
        if tuple(action_space.shape) != (2,):
            raise ValueError(f"Expected MetaDrive Box action shape (2,), got {action_space.shape}.")
        low = np.asarray(action_space.low, dtype=float)
        high = np.asarray(action_space.high, dtype=float)
        if np.any(low > -1.0) or np.any(high < 1.0):
            raise ValueError(f"Expected normalized action range covering [-1, 1], got {low}..{high}.")

    def path_to_actions(
        self,
        path: FrenetPath,
        vehicle,
        *,
        horizon: int,
        control_dt: float,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return one [H, 2] sequence; only k=0 uses the real vehicle state."""
        if horizon <= 0:
            raise ValueError("TD-MPC2 horizon must be positive.")
        if control_dt <= 0:
            raise ValueError("control_dt must be positive.")
        points = np.asarray(path.xy, dtype=float)
        source_times = np.asarray(path.t, dtype=float)
        source_speeds = np.asarray(path.s_d, dtype=float)
        if len(points) < 2 or len(source_times) != len(points):
            raise ValueError("Frenet path must contain at least two time-aligned XY points.")

        segments = np.diff(points, axis=0)
        segment_yaws = np.arctan2(segments[:, 1], segments[:, 0])
        path_yaws = np.concatenate([segment_yaws, segment_yaws[-1:]])
        query_times = np.minimum(np.arange(horizon, dtype=float) * control_dt, source_times[-1])
        positions = np.column_stack(
            [np.interp(query_times, source_times, points[:, axis]) for axis in range(2)]
        )
        headings = _interp_angle(query_times, source_times, path_yaws)
        speeds = np.interp(query_times, source_times, source_speeds)

        actions = []
        for step in range(horizon):
            if step == 0:
                tracking_state = vehicle
            else:
                tracking_state = SimpleNamespace(
                    position=positions[step],
                    heading_theta=float(headings[step]),
                    speed=max(0.0, float(speeds[step])),
                )
            action, _ = self.controller._track_path(tracking_state, path)
            actions.append(action)
        tensor = torch.as_tensor(np.asarray(actions), dtype=dtype, device=device)
        if tensor.shape != (horizon, self.action_dim):
            raise RuntimeError(f"Action adapter produced {tuple(tensor.shape)}, expected {(horizon, 2)}.")
        if not torch.isfinite(tensor).all():
            raise RuntimeError("Action adapter produced non-finite controls.")
        return tensor.clamp(-1.0, 1.0)

    def paths_to_actions(
        self,
        paths: Sequence[FrenetPath],
        vehicle,
        *,
        horizon: int,
        control_dt: float,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return batch actions with canonical shape [N, H, 2]."""
        if not paths:
            return torch.empty((0, horizon, self.action_dim), dtype=dtype, device=device)
        return torch.stack(
            [
                self.path_to_actions(
                    path,
                    vehicle,
                    horizon=horizon,
                    control_dt=control_dt,
                    device=device,
                    dtype=dtype,
                )
                for path in paths
            ],
            dim=0,
        )
