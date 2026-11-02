"""Fixed-size normalized feature encoding for a coarse Frenet trajectory."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch


FEATURE_NAMES = (
    "target_d", "target_speed", "horizon", "min_obstacle_clearance", "obstacle_cost",
    "max_abs_d_d", "max_abs_d_dd", "max_abs_d_ddd", "max_abs_s_dd", "max_abs_s_ddd",
    "curvature_max", "path_length",
)

DEFAULT_FEATURE_SCALES = np.asarray(
    [4.0, 20.0, 5.0, 20.0, 100.0, 5.0, 5.0, 10.0, 5.0, 10.0, 0.5, 100.0],
    dtype=np.float32,
)


def _max_abs(values) -> float:
    array = np.asarray(values, dtype=np.float32)
    return float(np.max(np.abs(array))) if array.size else 0.0


def extract_path_features(path) -> dict[str, float]:
    """Return raw diagnostic features while replacing all non-finite values."""
    points = np.asarray(path.xy, dtype=np.float32)
    path_length = 0.0
    curvature = 0.0
    if len(points) >= 2:
        segments = np.diff(points, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        path_length = float(lengths.sum())
        valid = lengths > 1e-6
        if np.count_nonzero(valid) >= 2:
            yaws = np.unwrap(np.arctan2(segments[valid, 1], segments[valid, 0]))
            curvature = float(np.max(np.abs(np.diff(yaws) / lengths[valid][1:])))
    clearance = float(path.min_obstacle_clearance)
    if not np.isfinite(clearance):
        clearance = float(DEFAULT_FEATURE_SCALES[3])
    values = {
        "target_d": float(path.target_d), "target_speed": float(path.target_speed),
        "horizon": float(path.horizon), "min_obstacle_clearance": clearance,
        "obstacle_cost": float(path.obstacle_cost), "max_abs_d_d": _max_abs(path.d_d),
        "max_abs_d_dd": _max_abs(path.d_dd), "max_abs_d_ddd": _max_abs(path.d_ddd),
        "max_abs_s_dd": _max_abs(path.s_dd), "max_abs_s_ddd": _max_abs(path.s_ddd),
        "curvature_max": curvature, "path_length": path_length,
    }
    return {
        key: float(np.nan_to_num(value, nan=0.0, posinf=1e3, neginf=-1e3))
        for key, value in values.items()
    }


def extract_coarse_path_features(
    path, *, scales=None, clip: float = 5.0, device: str | torch.device = "cpu"
) -> torch.Tensor:
    """Return normalized trajectory features with fixed shape ``[F]``."""
    raw = extract_path_features(path)
    values = np.asarray([raw[name] for name in FEATURE_NAMES], dtype=np.float32)
    scale_values = DEFAULT_FEATURE_SCALES if scales is None else np.asarray(scales, dtype=np.float32)
    if scale_values.shape != values.shape or np.any(scale_values <= 0.0):
        raise ValueError(f"feature scales must contain {len(FEATURE_NAMES)} positive values.")
    normalized = np.clip(values / scale_values, -abs(float(clip)), abs(float(clip)))
    if not np.isfinite(normalized).all():
        raise RuntimeError("Coarse path feature encoding produced non-finite values.")
    return torch.as_tensor(normalized, dtype=torch.float32, device=device)


def feature_scales_from_config(config: Mapping[str, object] | None):
    if not config or config.get("path_feature_scales") is None:
        return DEFAULT_FEATURE_SCALES
    return config["path_feature_scales"]
