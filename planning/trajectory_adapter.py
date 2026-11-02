"""Convert a selected Frenet trajectory into H+1 MetaDrive actions."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Sequence

import numpy as np
import torch

from .time_alignment import align_frenet_path


class LatticeActionAdapter:
    """Use the existing Lattice tracker on real and virtual future states."""

    def __init__(self, controller, action_dim: int = 2):
        if int(action_dim) != 2:
            raise ValueError("MetaDrive Lattice actions must be [steering, throttle_brake].")
        self.controller = controller
        self.action_dim = 2

    @staticmethod
    def validate_action_space(action_space) -> None:
        if tuple(action_space.shape) != (2,):
            raise ValueError(f"Expected action shape (2,), got {action_space.shape}.")
        low = np.asarray(action_space.low, dtype=np.float64)
        high = np.asarray(action_space.high, dtype=np.float64)
        if np.any(low > -1.0) or np.any(high < 1.0):
            raise ValueError(f"Expected normalized action bounds covering [-1,1], got {low}..{high}.")

    def path_to_actions(
        self,
        path,
        vehicle,
        *,
        required_steps: int | None = None,
        horizon: int | None = None,
        control_dt: float,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return ``[required_steps,2]``; ``horizon`` is a legacy alias."""
        if required_steps is None:
            if horizon is None:
                raise ValueError("Set required_steps (or the legacy horizon argument).")
            required_steps = int(horizon)
        if horizon is not None and int(horizon) != int(required_steps):
            raise ValueError("required_steps and horizon disagree.")
        aligned = align_frenet_path(
            path, required_steps=int(required_steps), control_dt=float(control_dt)
        )
        actions = []
        for step in range(int(required_steps)):
            state = vehicle if step == 0 else SimpleNamespace(
                position=aligned.positions[step],
                heading_theta=float(aligned.headings[step]),
                speed=max(0.0, float(aligned.speeds[step])),
            )
            action, _ = self.controller._track_path(state, path)
            actions.append(action)
        tensor = torch.as_tensor(np.asarray(actions), dtype=dtype, device=device)
        expected = (int(required_steps), self.action_dim)
        if tensor.shape != expected:
            raise RuntimeError(f"Action adapter produced {tuple(tensor.shape)}, expected {expected}.")
        if not torch.isfinite(tensor).all():
            raise RuntimeError("Action adapter produced non-finite controls.")
        return tensor.clamp(-1.0, 1.0)

    def paths_to_actions(
        self,
        paths: Sequence,
        vehicle,
        *,
        required_steps: int,
        control_dt: float,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if not paths:
            return torch.empty(
                (0, required_steps, self.action_dim), dtype=dtype, device=device
            )
        return torch.stack(
            [
                self.path_to_actions(
                    path,
                    vehicle,
                    required_steps=required_steps,
                    control_dt=control_dt,
                    device=device,
                    dtype=dtype,
                )
                for path in paths
            ],
            dim=0,
        )


__all__ = ["LatticeActionAdapter"]
