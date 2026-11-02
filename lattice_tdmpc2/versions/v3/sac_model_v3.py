"""Frozen V3 SAC-Lagrangian network update.

V3 uses projected dual-gradient multiplier updates, the mean of twin cost
critics, Huber cost regression, and no residual trust penalty.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from sac.model import SACAgent


class ProjectedLagrangianV3:
    """Non-negative dual variable updated once per completed episode."""

    def __init__(self, learning_rate: float = 0.01, initial_value: float = 0.0):
        self.learning_rate = float(learning_rate)
        if self.learning_rate <= 0.0:
            raise ValueError("lagrangian_lr must be positive.")
        self.value = max(float(initial_value), 0.0)

    def step(self, episode_cost: float, cost_limit: float) -> float:
        error = float(episode_cost) - float(cost_limit)
        self.value = max(self.value + self.learning_rate * error, 0.0)
        return self.value

    def state_dict(self) -> dict:
        return {"learning_rate": self.learning_rate, "value": self.value}

    def load_state_dict(self, state: dict) -> None:
        if "learning_rate" in state:
            learning_rate = float(state["learning_rate"])
            if learning_rate <= 0.0:
                raise ValueError("Checkpoint lagrangian learning rate must be positive.")
            self.learning_rate = learning_rate
        self.value = max(float(state["value"]), 0.0)


class SACAgentV3(SACAgent):
    """Checkpoint-compatible V3 SAC with its update rule fixed in this file."""

    algorithm_version = "lattice_tdmpc2_v3"

    def __init__(self, *args, lagrangian_lr: float = 0.01, **kwargs):
        super().__init__(*args, lagrangian_lr=lagrangian_lr, **kwargs)
        self.lagrangian = ProjectedLagrangianV3(lagrangian_lr)

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        observation = batch["obs"]
        action = batch["action"]
        reward = batch["reward"]
        cost = batch["cost"]
        next_observation = batch["next_obs"]
        done = batch["done"]
        discount = batch["discount"]

        with torch.no_grad():
            next_action, next_log_probability = self.actor(next_observation)
            next_q = torch.minimum(
                self.target_q1(next_observation, next_action),
                self.target_q2(next_observation, next_action),
            )
            target = reward + discount * (1.0 - done) * (
                next_q - self.alpha.detach() * next_log_probability
            )
            if self.use_lagrangian:
                next_cost_q = 0.5 * (
                    self.target_cost_q1(next_observation, next_action)
                    + self.target_cost_q2(next_observation, next_action)
                )
                cost_target = cost + discount * (1.0 - done) * next_cost_q
            else:
                cost_target = torch.zeros_like(cost)

        q1 = self.q1(observation, action)
        q2 = self.q2(observation, action)
        critic_loss = (q1 - target).square().mean() + (q2 - target).square().mean()
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.use_lagrangian:
            cost_q1 = self.cost_q1(observation, action)
            cost_q2 = self.cost_q2(observation, action)
            cost_critic_loss = (
                F.smooth_l1_loss(cost_q1, cost_target)
                + F.smooth_l1_loss(cost_q2, cost_target)
            )
            self.cost_critic_optimizer.zero_grad(set_to_none=True)
            cost_critic_loss.backward()
            self.cost_critic_optimizer.step()
        else:
            cost_q1 = cost_q2 = torch.zeros_like(q1)
            cost_critic_loss = torch.zeros((), device=self.device)

        actor_critics = [self.q1, self.q2]
        if self.use_lagrangian:
            actor_critics.extend([self.cost_q1, self.cost_q2])
        critic_parameters = [p for critic in actor_critics for p in critic.parameters()]
        for parameter in critic_parameters:
            parameter.requires_grad_(False)
        try:
            policy_action, log_probability = self.actor(observation)
            policy_q = torch.minimum(
                self.q1(observation, policy_action),
                self.q2(observation, policy_action),
            )
            actor_sac_loss = (self.alpha.detach() * log_probability - policy_q).mean()
            if self.use_lagrangian:
                policy_cost = 0.5 * (
                    self.cost_q1(observation, policy_action)
                    + self.cost_q2(observation, policy_action)
                ).mean()
            else:
                policy_cost = torch.zeros((), device=self.device)
            lagrangian = self.lagrangian_multiplier
            actor_safety_loss = lagrangian * policy_cost
            rescaling = 1.0 / (1.0 + lagrangian) if self.lagrangian_rescaling else 1.0
            actor_loss = rescaling * (actor_sac_loss + actor_safety_loss)
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
        finally:
            for parameter in critic_parameters:
                parameter.requires_grad_(True)

        alpha_loss = torch.zeros((), device=self.device)
        if self.auto_alpha:
            alpha_loss = -(
                self.log_alpha * (log_probability.detach() + self.target_entropy)
            ).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()

        with torch.no_grad():
            self._soft_update(self.target_q1, self.q1)
            self._soft_update(self.target_q2, self.q2)
            if self.use_lagrangian:
                self._soft_update(self.target_cost_q1, self.cost_q1)
                self._soft_update(self.target_cost_q2, self.cost_q2)
        self.update_steps += 1
        return {
            "loss/critic": float(critic_loss.detach()),
            "loss/cost_critic": float(cost_critic_loss.detach()),
            "loss/actor": float(actor_loss.detach()),
            "loss/actor_sac": float(actor_sac_loss.detach()),
            "loss/actor_safety": float(actor_safety_loss.detach()),
            "loss/alpha": float(alpha_loss.detach()),
            "train/alpha": float(self.alpha.detach()),
            "train/entropy": float((-log_probability).detach().mean()),
            "train/q1": float(q1.detach().mean()),
            "train/q2": float(q2.detach().mean()),
            "train/target_q": float(target.detach().mean()),
            "train/cost_q": float((0.5 * (cost_q1 + cost_q2)).detach().mean()),
            "train/cost_target_q": float(cost_target.detach().mean()),
            "lagrangian/value": float(lagrangian),
            "lagrangian/cost_limit": self.cost_limit,
            "lagrangian/lr": self.lagrangian.learning_rate,
            "lagrangian/rescaling": float(rescaling),
            "sac/critic_loss": float(critic_loss.detach()),
            "sac/cost_critic_loss": float(cost_critic_loss.detach()),
            "sac/actor_loss": float(actor_loss.detach()),
            "sac/actor_sac_loss": float(actor_sac_loss.detach()),
            "sac/alpha": float(self.alpha.detach()),
            "sac/entropy": float((-log_probability).detach().mean()),
            "sac/q1": float(q1.detach().mean()),
            "sac/q2": float(q2.detach().mean()),
        }

    def _soft_update(self, target, source) -> None:
        for target_parameter, parameter in zip(target.parameters(), source.parameters()):
            target_parameter.lerp_(parameter, self.tau)
