"""Map normalized SAC actions to MPR-MPC parameter refinements."""

from __future__ import annotations

import torch


class ResidualActionAdapter:
    """Convert normalized ``[-1, 1]^2`` residual actions."""

    action_dim = 2

    @staticmethod
    def to_physical(normalized_action: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
        action = torch.as_tensor(
            normalized_action,
            dtype=bounds.dtype,
            device=bounds.device,
        )
        if action.ndim == 1:
            action = action.unsqueeze(0)
        if action.shape[-1] != 2:
            raise ValueError(f"Residual SAC action must end in dim 2, got {tuple(action.shape)}.")
        if bounds.ndim == 1:
            bounds = bounds.unsqueeze(0)
        if bounds.shape[-1] != 2 or bounds.shape[0] not in {1, action.shape[0]}:
            raise ValueError("Residual bounds must be [2] or [B,2].")
        if not torch.isfinite(bounds).all() or torch.any(bounds < 0):
            raise ValueError("Residual bounds must be finite and non-negative.")
        return action.clamp(-1.0, 1.0) * bounds.expand(action.shape[0], -1)

    @staticmethod
    def refine_parameters(
        normalized_action: torch.Tensor,
        coarse_parameters: torch.Tensor,
        lower_bounds: torch.Tensor,
        upper_bounds: torch.Tensor,
        scale: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coarse = torch.as_tensor(coarse_parameters, dtype=torch.float32)
        device = coarse.device
        action = torch.as_tensor(normalized_action, dtype=coarse.dtype, device=device)
        lower = torch.as_tensor(lower_bounds, dtype=coarse.dtype, device=device)
        upper = torch.as_tensor(upper_bounds, dtype=coarse.dtype, device=device)
        scale = torch.as_tensor(scale, dtype=coarse.dtype, device=device)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        if coarse.ndim == 1:
            coarse = coarse.unsqueeze(0)
        if lower.ndim == 1:
            lower = lower.unsqueeze(0)
        if upper.ndim == 1:
            upper = upper.unsqueeze(0)
        if action.shape[-1] != 2 or coarse.shape[-1] != 2:
            raise ValueError("Residual action and coarse parameters must end in dim 2.")
        if lower.shape[-1] != 2 or upper.shape[-1] != 2:
            raise ValueError("Residual parameter bounds must end in dim 2.")
        if scale.shape != (2,):
            raise ValueError(f"latent_tanh scale must have shape [2], got {tuple(scale.shape)}.")
        batch = action.shape[0]
        for name, tensor in (("coarse", coarse), ("lower", lower), ("upper", upper)):
            if tensor.shape[0] not in {1, batch}:
                raise ValueError(f"{name} must be [2] or [B,2].")
        coarse = coarse.expand(batch, -1)
        lower = lower.expand(batch, -1)
        upper = upper.expand(batch, -1)
        if not (0.0 < float(eps) < 1.0):
            raise ValueError("latent_tanh eps must be in (0, 1).")
        span = upper - lower
        if not torch.isfinite(coarse).all() or not torch.isfinite(lower).all() or not torch.isfinite(upper).all():
            raise ValueError("Residual refinement inputs must be finite.")
        if torch.any(span <= 0):
            raise ValueError("Residual parameter upper bounds must be greater than lower bounds.")
        action = action.clamp(-1.0, 1.0)
        normalized = 2.0 * (coarse - lower) / span - 1.0
        normalized = normalized.clamp(-1.0 + float(eps), 1.0 - float(eps))
        latent = torch.atanh(normalized)
        refined_normalized = torch.tanh(latent + scale.reshape(1, 2) * action)
        refined = lower + 0.5 * (refined_normalized + 1.0) * span
        refined = torch.where(action == 0.0, coarse, refined)
        residual = refined - coarse
        return refined, residual
