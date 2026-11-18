"""Time alignment between Frenet trajectories and MetaDrive control steps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AlignedTrajectory:
    times: np.ndarray
    positions: np.ndarray
    headings: np.ndarray
    speeds: np.ndarray


def _interp_angle(query: np.ndarray, source: np.ndarray, angles: np.ndarray) -> np.ndarray:
    return np.interp(query, source, np.unwrap(angles))


def align_frenet_path(path, *, required_steps: int, control_dt: float,
                      cartesian_speed: bool = False) -> AlignedTrajectory:
    if required_steps <= 0:
        raise ValueError("required_steps must be positive.")
    if control_dt <= 0:
        raise ValueError("control_dt must be positive.")
    points = np.asarray(path.xy, dtype=np.float64)
    source_times = np.asarray(path.t, dtype=np.float64)
    speeds = np.asarray(path.s_d, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("Frenet path must contain at least two XY points.")
    if len(source_times) != len(points) or len(speeds) != len(points):
        raise ValueError("Frenet positions, times, and speeds must be aligned.")
    if np.any(np.diff(source_times) <= 0.0):
        raise ValueError("Frenet path times must be strictly increasing.")

    segment_yaws = np.arctan2(np.diff(points[:, 1]), np.diff(points[:, 0]))
    point_yaws = np.concatenate([segment_yaws, segment_yaws[-1:]])
    query_times = np.arange(required_steps, dtype=np.float64) * float(control_dt)
    clipped_times = np.minimum(query_times, source_times[-1])
    positions = np.column_stack(
        [np.interp(clipped_times, source_times, points[:, axis]) for axis in range(2)]
    )
    headings = _interp_angle(clipped_times, source_times, point_yaws)
    aligned_speeds = np.interp(clipped_times, source_times, speeds)
    if cartesian_speed:
        # s_d is longitudinal Frenet speed, while the tracker consumes the
        # magnitude of world velocity. Derive that magnitude from the SAME
        # position/time trajectory; keep legacy interpolation as the default.
        edge_order = 2 if len(points) > 2 else 1
        velocity = np.column_stack([
            np.gradient(points[:, axis], source_times, edge_order=edge_order)
            for axis in range(2)
        ])
        ground_speed = np.linalg.norm(velocity, axis=1)
        aligned_speeds = np.interp(clipped_times, source_times, ground_speed)
        # Beyond the endpoint the existing position policy holds position.
        # Its velocity is zero, not the last nonzero longitudinal speed.
        aligned_speeds[query_times > source_times[-1]] = 0.0
    return AlignedTrajectory(query_times, positions, headings, aligned_speeds)


__all__ = ["AlignedTrajectory", "align_frenet_path"]
