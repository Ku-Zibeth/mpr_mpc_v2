"""Trajectory-consistent TD-MPC2 World Model evaluation."""

from __future__ import annotations

import torch

from mpr_mpc._bootstrap import bootstrap
from .types import TrajectoryConsequence


bootstrap()
from common import math as td_math  # noqa: E402


class TrajectoryWorldModelEvaluator:
    """Evaluate H-step rollouts with either policy or trajectory terminal action."""

    def __init__(self, agent, horizon: int):
        self.agent = agent
        self.horizon = int(horizon)
        if self.horizon <= 0:
            raise ValueError("World Model horizon must be positive.")

    @torch.no_grad()
    def encode(self, observation: torch.Tensor, task=None) -> torch.Tensor:
        observation = observation.to(self.agent.device, non_blocking=True)
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        self.agent.model.eval()
        return self.agent.model.encode(observation, task)

    def _discount(self, task, actions: torch.Tensor):
        discount = self.agent.discount
        if self.agent.cfg.multitask:
            return discount[torch.as_tensor(task, device=actions.device)].reshape(-1, 1)
        return actions.new_tensor(float(discount))

    @torch.no_grad()
    def evaluate(
        self,
        z_t: torch.Tensor,
        actions: torch.Tensor,
        *,
        terminal_action_mode: str,
        task=None,
        mode_ids=None,
    ) -> TrajectoryConsequence:
        if actions.ndim != 3:
            raise ValueError(f"actions must be [B,S,A], got {tuple(actions.shape)}.")
        batch, steps, action_dim = actions.shape
        if action_dim != int(self.agent.cfg.action_dim):
            raise ValueError(f"Action dim {action_dim} != configured {self.agent.cfg.action_dim}.")
        expected_steps = self.horizon + (terminal_action_mode == "trajectory")
        if terminal_action_mode not in {"policy", "trajectory"}:
            raise ValueError("terminal_action_mode must be 'policy' or 'trajectory'.")
        if steps != expected_steps:
            raise ValueError(
                f"{terminal_action_mode} mode expects {expected_steps} actions, got {steps}."
            )

        actions = actions.to(self.agent.device).clamp(-1.0, 1.0)
        z = z_t.to(self.agent.device)
        if z.ndim == 1:
            z = z.unsqueeze(0)
        if z.shape[0] == 1 and batch > 1:
            z = z.expand(batch, -1)
        if z.shape != (batch, int(self.agent.cfg.latent_dim)):
            raise ValueError(
                f"Latent shape {tuple(z.shape)} does not match {(batch, int(self.agent.cfg.latent_dim))}."
            )

        model = self.agent.model
        model.eval()
        discount_step = self._discount(task, actions)
        discount = actions.new_ones((batch, 1))
        terminated = actions.new_zeros((batch, 1))
        rewards = []
        total = actions.new_zeros((batch, 1))
        cost_ready = bool(getattr(model, "cost_ready", False))
        cost_cfg = getattr(self.agent.cfg, "cost", {}) or {}
        cost_ready = cost_ready and bool(cost_cfg.get("enabled", False))
        cost_mask = torch.zeros(batch, dtype=torch.bool, device=actions.device)
        if cost_ready and mode_ids is not None and self.horizon < 63:
            ids = torch.as_tensor(mode_ids, device=actions.device, dtype=torch.long)
            if ids.shape != (batch,):
                raise ValueError("mode_ids must have shape [B].")
            known = (ids >= 0) & (ids < 6)
            cost_mask[known] = model._cost_validated_by_mode[ids[known], self.horizon]
        cost_ready = bool(cost_mask.any())
        cost_gamma = float(cost_cfg.get("gamma", float(discount_step.flatten()[0])))
        cost_total = actions.new_zeros((batch, 1))
        costs = []
        for step in range(self.horizon):
            action = actions[:, step]
            reward = td_math.two_hot_inv(model.reward(z, action, task), self.agent.cfg)
            rewards.append(reward.squeeze(-1))
            total += discount * (1.0 - terminated) * reward
            if cost_ready:
                cost = actions.new_full((batch, 1), float("nan"))
                cost_task = task
                if torch.is_tensor(task) and task.ndim and task.shape[0] == batch:
                    cost_task = task[cost_mask]
                cost[cost_mask] = model.cost(z[cost_mask], action[cost_mask], cost_task)
                costs.append(cost.squeeze(-1))
                # A learned early termination must not hide later predicted danger.
                cost_total += cost_gamma**step * cost
            z = model.next(z, action, task)
            discount = discount * discount_step
            if bool(self.agent.cfg.episodic) and getattr(model, "_termination", None) is not None:
                terminated = torch.clamp(
                    terminated + (model.termination(z, task) > 0.5).to(actions.dtype), max=1.0
                )

        if terminal_action_mode == "trajectory":
            terminal_action = actions[:, self.horizon]
        else:
            terminal_action, _ = model.pi(z, task)
        # Use the full ensemble instead of TD-MPC2's random two-Q subsample. This
        # makes evaluation deterministic and exposes epistemic spread to MPPI.
        terminal_q_all = td_math.two_hot_inv(
            model.Q(z, terminal_action, task, return_type="all"), self.agent.cfg
        )
        terminal_q = terminal_q_all.mean(dim=0)
        terminal_q_std = terminal_q_all.std(dim=0, unbiased=False)
        total += discount * (1.0 - terminated) * terminal_q
        if costs:
            cost_mask &= torch.isfinite(cost_total.squeeze(-1))
        return TrajectoryConsequence(
            reward_sequence=torch.stack(rewards, dim=1).detach(),
            terminal_q=terminal_q.squeeze(-1).detach(),
            terminal_q_std=terminal_q_std.squeeze(-1).detach(),
            total_value=total.squeeze(-1).detach(),
            terminal_latent=z.detach(),
            predicted_cost=torch.stack(costs, dim=1).detach() if costs else None,
            cumulative_cost=cost_total.squeeze(-1).detach() if costs else None,
            cost_validated=bool(cost_mask.all()) and cost_ready,
            cost_validated_mask=cost_mask,
        )

    def evaluate_trajectory_consequence(
        self, z_t: torch.Tensor, actions: torch.Tensor, task=None, mode_ids=None
    ) -> TrajectoryConsequence:
        return self.evaluate(
            z_t, actions, terminal_action_mode="trajectory", task=task, mode_ids=mode_ids
        )

    def evaluate_policy_terminal(
        self, z_t: torch.Tensor, actions: torch.Tensor, task=None
    ) -> TrajectoryConsequence:
        return self.evaluate(z_t, actions, terminal_action_mode="policy", task=task)

    def evaluate_mppi_candidates(
        self, z_t: torch.Tensor, actions: torch.Tensor, task=None, mode_ids=None
    ) -> TrajectoryConsequence:
        return self.evaluate_trajectory_consequence(z_t, actions, task=task, mode_ids=mode_ids)


__all__ = ["TrajectoryWorldModelEvaluator"]
