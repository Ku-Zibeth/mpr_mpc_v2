"""Candidate-free Gaussian residual policy for one selected coarse trajectory."""

from __future__ import annotations

import torch
from torch import nn


def _mlp(input_dim: int, hidden_dims, output_dim: int) -> nn.Sequential:
    modules: list[nn.Module] = []
    previous = int(input_dim)
    for width in hidden_dims:
        width = int(width)
        modules.extend([nn.Linear(previous, width), nn.LayerNorm(width), nn.Mish()])
        previous = width
    modules.append(nn.Linear(previous, int(output_dim)))
    return nn.Sequential(*modules)


class ResidualTrajectoryPrior(nn.Module):
    """Map one coarse-trajectory context to bounded ``[Δd,Δv]``."""

    residual_dim = 2

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dims=(256, 256),
        log_std_min: float = -5.0,
        log_std_max: float = 1.0,
        initial_log_std: float = -3.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.network = _mlp(self.input_dim, hidden_dims, 2 * self.residual_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        if self.log_std_min >= self.log_std_max:
            raise ValueError("log_std_min must be less than log_std_max.")
        final = self.network[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        with torch.no_grad():
            final.bias[self.residual_dim :].fill_(
                float(max(self.log_std_min, min(self.log_std_max, initial_log_std)))
            )

    def distribution(self, residual_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if residual_input.ndim != 2 or residual_input.shape[-1] != self.input_dim:
            raise ValueError(
                f"residual_input must be [B,{self.input_dim}], got {tuple(residual_input.shape)}."
            )
        mean, raw_log_std = self.network(residual_input).chunk(2, dim=-1)
        log_std = raw_log_std.clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def _bounds_for(self, residual_input: torch.Tensor, delta_bounds) -> torch.Tensor:
        bounds = torch.as_tensor(
            delta_bounds, dtype=residual_input.dtype, device=residual_input.device
        )
        if bounds.ndim == 1:
            bounds = bounds.unsqueeze(0)
        if bounds.shape[-1] != self.residual_dim or bounds.shape[0] not in {
            1,
            residual_input.shape[0],
        }:
            raise ValueError(
                "delta_bounds must be [2] or [B,2] and match residual_input batch size."
            )
        if torch.any(bounds < 0):
            raise ValueError("delta_bounds must be non-negative.")
        return bounds.expand(residual_input.shape[0], -1)

    def forward(
        self,
        residual_input: torch.Tensor,
        *,
        delta_bounds,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mean, log_std = self.distribution(residual_input)
        bounds = self._bounds_for(residual_input, delta_bounds)
        raw = mean if deterministic else mean + log_std.exp() * torch.randn_like(mean)
        residual = torch.tanh(raw) * bounds
        return residual, {
            "mean": mean,
            "log_std": log_std,
            "bounded_mean": torch.tanh(mean) * bounds,
            "delta_bounds": bounds,
        }

    def supervised_loss(
        self,
        residual_input: torch.Tensor,
        target_residual: torch.Tensor,
        *,
        delta_bounds,
        residual_reg_coef: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mean, log_std = self.distribution(residual_input)
        bounds = self._bounds_for(residual_input, delta_bounds)
        if target_residual.shape != mean.shape:
            raise ValueError(
                f"target_residual must be {tuple(mean.shape)}, got {tuple(target_residual.shape)}."
            )
        safe_bounds = bounds.clamp_min(torch.finfo(bounds.dtype).eps)
        normalized = (target_residual / safe_bounds).clamp(-0.999999, 0.999999)
        normalized = torch.where(bounds > 0, normalized, torch.zeros_like(normalized))
        raw_target = torch.atanh(normalized)
        inverse_variance = torch.exp(-2.0 * log_std)
        nll = 0.5 * ((raw_target - mean).square() * inverse_variance + 2.0 * log_std)
        nll = nll.sum(dim=-1).mean()
        bounded_mean = torch.tanh(mean) * bounds
        normalized_prediction = torch.where(
            bounds > 0, bounded_mean / safe_bounds, torch.zeros_like(bounded_mean)
        )
        regularization = normalized_prediction.square().sum(dim=-1).mean()
        loss = nll + float(residual_reg_coef) * regularization
        return loss, {
            "residual_nll": nll.detach(),
            "residual_reg": regularization.detach(),
            "bounded_mean": bounded_mean,
            "regularization_loss": regularization,
        }


__all__ = ["ResidualTrajectoryPrior"]
