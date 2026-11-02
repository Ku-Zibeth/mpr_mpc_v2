"""Vectorized TD-MPC2 world-model evaluation for Lattice action sequences."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from common import math as td_math


@dataclass(frozen=True)
class WorldModelEvaluation:
    predicted_return: torch.Tensor
    terminal_value: torch.Tensor
    predicted_cost: torch.Tensor
    uncertainty: torch.Tensor
    score: torch.Tensor


class WorldModelEvaluator:
    def __init__(
        self,
        agent,
        *,
        use_cost: bool = False,
        cost_weight: float = 0.0,
        uncertainty_weight: float = 0.0,
        use_q_uncertainty: bool = True,
        gamma_cost: float | None = None,
        use_terminal_cost_q: bool = False,
    ):
        self.agent = agent
        self.use_cost = bool(use_cost)
        self.cost_weight = float(cost_weight)
        self.uncertainty_weight = float(uncertainty_weight)
        self.use_q_uncertainty = bool(use_q_uncertainty)
        self.gamma_cost = gamma_cost
        self.use_terminal_cost_q = bool(use_terminal_cost_q)

    def encode(self, observation: torch.Tensor, task=None) -> torch.Tensor:
        observation = observation.to(self.agent.device, non_blocking=True)
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        with torch.no_grad():
            self.agent.model.eval()
            return self.agent.model.encode(observation, task)

    @torch.no_grad()
    def evaluate_action_sequences(
        self, z_t: torch.Tensor, actions: torch.Tensor, task=None
    ) -> WorldModelEvaluation:
        """Evaluate [N,H,A] in one candidate batch and loop only over H."""
        if actions.ndim != 3:
            raise ValueError(f"actions must be [N,H,A], got {tuple(actions.shape)}.")
        num_candidates, horizon, action_dim = actions.shape
        if num_candidates == 0:
            raise ValueError("At least one action sequence is required.")
        if action_dim != int(self.agent.cfg.action_dim):
            raise ValueError(f"Action dim {action_dim} != TD-MPC2 dim {self.agent.cfg.action_dim}.")

        model = self.agent.model
        model.eval()
        actions = actions.to(self.agent.device)
        z = z_t.to(self.agent.device)
        if z.ndim == 1:
            z = z.unsqueeze(0)
        if z.shape[-1] != int(self.agent.cfg.latent_dim):
            raise ValueError(f"Latent dim {z.shape[-1]} != configured {self.agent.cfg.latent_dim}.")
        if z.shape[0] == 1:
            z = z.expand(num_candidates, -1)
        elif z.shape[0] != num_candidates:
            raise ValueError(f"Latent batch {z.shape[0]} != candidate count {num_candidates}.")

        gamma = self.agent.discount
        if not torch.is_tensor(gamma):
            gamma = actions.new_tensor(float(gamma))
        else:
            gamma = gamma.to(device=actions.device, dtype=actions.dtype)
        gamma_cost = float(self.gamma_cost) if self.gamma_cost is not None else float(gamma.mean())
        predicted_return = actions.new_zeros((num_candidates, 1))
        predicted_cost = actions.new_zeros((num_candidates, 1))
        discount = actions.new_ones((num_candidates, 1))
        cost_discount = actions.new_ones((num_candidates, 1))
        terminated = actions.new_zeros((num_candidates, 1))

        for step in range(horizon):
            action = actions[:, step].clamp(-1.0, 1.0)
            reward = td_math.two_hot_inv(model.reward(z, action, task), self.agent.cfg)
            predicted_return += discount * (1.0 - terminated) * reward
            if self.use_cost:
                if not hasattr(model, "cost"):
                    raise RuntimeError("use_cost=true but the TD-MPC2 world model has no cost predictor.")
                predicted_cost += cost_discount * model.cost(z, action, task)
                cost_discount *= gamma_cost
            z = model.next(z, action, task)
            discount *= gamma
            if bool(self.agent.cfg.episodic) and getattr(model, "_termination", None) is not None:
                terminated = torch.clamp(
                    terminated + (model.termination(z, task) > 0.5).to(actions.dtype), max=1.0
                )

        terminal_action, _ = model.pi(z, task)
        terminal_value = model.Q(z, terminal_action, task, return_type="avg")
        predicted_return += discount * (1.0 - terminated) * terminal_value
        if self.use_cost and self.use_terminal_cost_q:
            if not bool(getattr(model, "cost_q_enabled", False)):
                raise RuntimeError(
                    "use_terminal_cost_q=true but the TD-MPC2 world model has no trained cost-Q head."
                )
            terminal_cost = model.cost_Q(z, terminal_action, task, return_type="avg")
            predicted_cost += cost_discount * (1.0 - terminated) * terminal_cost

        if self.use_q_uncertainty:
            q_logits = model.Q(z, terminal_action, task, return_type="all")
            q_values = td_math.two_hot_inv(q_logits, self.agent.cfg)
            uncertainty = q_values.std(dim=0, unbiased=False)
        else:
            uncertainty = torch.zeros_like(terminal_value)
        score = (
            predicted_return
            - self.cost_weight * predicted_cost
            - self.uncertainty_weight * uncertainty
        )
        outputs = [predicted_return, terminal_value, predicted_cost, uncertainty, score]
        if not all(torch.isfinite(value).all() for value in outputs):
            raise RuntimeError("TD-MPC2 world-model evaluation produced non-finite values.")
        return WorldModelEvaluation(*(value.squeeze(-1).detach() for value in outputs))
