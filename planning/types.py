"""Shared immutable planning records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class StructuredSelection:
    candidates: tuple[Any, ...]
    feasible: tuple[Any, ...]
    coarse_path: Any
    selected_index: int
    used_fallback: bool = False


@dataclass(frozen=True)
class PlanningStage:
    name: str
    use_residual: bool
    use_mppi: bool


@dataclass(frozen=True)
class TrajectoryConsequence:
    reward_sequence: torch.Tensor
    terminal_q: torch.Tensor
    terminal_q_std: torch.Tensor
    total_value: torch.Tensor
    terminal_latent: torch.Tensor
    predicted_cost: torch.Tensor | None = None
    cumulative_cost: torch.Tensor | None = None
    cost_validated: bool = False
    cost_ranking_validated: bool = False
    cost_validated_mask: torch.Tensor | None = None
    cost_unavailable_reason: str = "unvalidated"
    risk_summary: dict[str, Any] = field(default_factory=dict)
    discounted_reward: torch.Tensor | None = None
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def reward_sum(self) -> torch.Tensor:
        """Undiscounted sum of the H one-step reward predictions."""
        return self.reward_sequence.sum(dim=-1)

    @property
    def features(self) -> torch.Tensor:
        """Compact residual-policy features: [sum(r_0...r_H-1), Q_H, J]."""
        return torch.cat(
            [
                self.reward_sum.unsqueeze(-1),
                self.terminal_q.unsqueeze(-1),
                self.total_value.unsqueeze(-1),
            ],
            dim=-1,
        )


@dataclass
class MPRPlanContext:
    observation: torch.Tensor
    latent: torch.Tensor | None
    selection: StructuredSelection
    coarse_actions: torch.Tensor
    coarse_consequence: TrajectoryConsequence | None
    wm_features: torch.Tensor | None
    coarse_parameters: torch.Tensor
    residual_bounds: torch.Tensor
    residual_state: torch.Tensor | None
    safety_context: Any = None
    mppi_safety_context: Any = None


@dataclass
class MPRPlanResult:
    action: torch.Tensor
    coarse_path: Any
    refined_path: Any
    coarse_actions: torch.Tensor
    refined_actions: torch.Tensor
    requested_normalized_residual: torch.Tensor
    requested_physical_residual: torch.Tensor
    residual: torch.Tensor
    residual_mean: torch.Tensor
    residual_log_std: torch.Tensor
    residual_valid: bool
    fallback_reason: str
    residual_warmup: bool
    coarse_consequence: TrajectoryConsequence | None
    stage: PlanningStage
    metrics: dict[str, float] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)
    planning_mode: str = "baseline"
    safety_failure: bool = False

    @property
    def coarse_parameters(self) -> np.ndarray:
        return np.asarray(
            [self.coarse_path.target_d, self.coarse_path.target_speed, self.coarse_path.horizon],
            dtype=np.float32,
        )

    @property
    def refined_parameters(self) -> np.ndarray:
        return np.asarray(
            [self.refined_path.target_d, self.refined_path.target_speed, self.refined_path.horizon],
            dtype=np.float32,
        )
