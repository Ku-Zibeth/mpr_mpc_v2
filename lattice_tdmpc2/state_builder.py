"""Residual SAC state assembly from TD-MPC2 and coarse Lattice outputs."""

from __future__ import annotations

import torch

from .evaluator import WorldModelEvaluation


class ResidualStateBuilder:
    def __init__(
        self, *, include_latent: bool, include_wm_return: bool, include_wm_cost: bool,
        include_wm_uncertainty: bool, wm_return_scale: float = 10.0,
        wm_cost_scale: float = 10.0, wm_uncertainty_scale: float = 10.0,
    ):
        self.include_latent = bool(include_latent)
        self.include_wm_return = bool(include_wm_return)
        self.include_wm_cost = bool(include_wm_cost)
        self.include_wm_uncertainty = bool(include_wm_uncertainty)
        self.scales = (
            max(abs(float(wm_return_scale)), 1e-6), max(abs(float(wm_cost_scale)), 1e-6),
            max(abs(float(wm_uncertainty_scale)), 1e-6),
        )
        self.state_dim: int | None = None

    def build(
        self, latent: torch.Tensor | None, path_features: torch.Tensor,
        evaluation: WorldModelEvaluation | None,
    ) -> torch.Tensor:
        if path_features.ndim != 1:
            raise ValueError(f"path_features must have shape [F], got {tuple(path_features.shape)}.")
        pieces = []
        if self.include_latent:
            if latent is None:
                raise ValueError("Residual state requires TD-MPC2 latent, but none was provided.")
            pieces.append(latent.reshape(-1).detach().to(device="cpu", dtype=torch.float32))
        pieces.append(path_features.detach().to(device="cpu", dtype=torch.float32))
        if self.include_wm_return:
            pieces.append(self._scalar(evaluation, "predicted_return", self.scales[0]))
        if self.include_wm_cost:
            pieces.append(self._scalar(evaluation, "predicted_cost", self.scales[1]))
        if self.include_wm_uncertainty:
            pieces.append(self._scalar(evaluation, "uncertainty", self.scales[2]))
        state = torch.cat(pieces, dim=0)
        if not torch.isfinite(state).all():
            raise RuntimeError("Residual SAC state contains non-finite values.")
        if self.state_dim is None:
            self.state_dim = int(state.numel())
        elif state.numel() != self.state_dim:
            raise RuntimeError(f"Residual state changed from {self.state_dim} to {state.numel()} dimensions.")
        return state

    @staticmethod
    def _scalar(evaluation, field: str, scale: float) -> torch.Tensor:
        if evaluation is None:
            raise ValueError(f"Residual state requests {field}, but world-model evaluation is unavailable.")
        value = getattr(evaluation, field)
        if value.numel() != 1:
            raise ValueError(f"Coarse {field} must contain one value, got shape {tuple(value.shape)}.")
        return value.detach().reshape(1).to(device="cpu", dtype=torch.float32) / scale
