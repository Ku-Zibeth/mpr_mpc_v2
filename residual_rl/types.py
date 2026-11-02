"""Small records shared by residual SAC planning and training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ResidualBounds:
    """Symmetric physical residual bounds for normalized SAC actions."""

    high: torch.Tensor

    def __post_init__(self):
        if self.high.shape[-1] != 2:
            raise ValueError(f"Residual bounds must end in dim 2, got {tuple(self.high.shape)}.")
        if not torch.isfinite(self.high).all() or torch.any(self.high < 0):
            raise ValueError("Residual bounds must be finite and non-negative.")

    @property
    def low(self) -> torch.Tensor:
        return -self.high


@dataclass(frozen=True)
class ResidualPolicyOutput:
    """Normalized actor output plus optional diagnostics."""

    action: torch.Tensor
    mean: torch.Tensor
    log_std: torch.Tensor
    warmup: bool = False


@dataclass(frozen=True)
class ResidualActionTrace:
    """Action values needed by replay and diagnostics."""

    normalized: torch.Tensor
    requested_physical: np.ndarray
    effective_physical: np.ndarray
    mean: torch.Tensor
    log_std: torch.Tensor
    valid: bool
    fallback_reason: str
    warmup: bool = False
