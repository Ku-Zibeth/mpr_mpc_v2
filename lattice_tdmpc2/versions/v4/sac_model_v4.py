"""V4 SAC-Lagrangian update with world-model cost-to-go supervision.

Reward critics retain the V3 SAC TD target. The twin cost critics directly
regress the frozen world model's finite-rollout plus terminal cost-Q estimate;
bootstrapping that estimate again would count future cost twice.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from lattice_tdmpc2.versions.v3.sac_model_v3 import (
    ProjectedLagrangianV3,
    SACAgentV3,
)


class ProjectedLagrangianV4(ProjectedLagrangianV3):
    """Projected dual update applied to mean per-state WM cost-to-go."""


class SACAgentV4(SACAgentV3):
    """V3 SAC with direct frozen-world-model targets for both cost critics."""

    algorithm_version = "lattice_tdmpc2_v4"

    def __init__(self, *args, lagrangian_lr: float = 0.01, **kwargs):
        super().__init__(*args, lagrangian_lr=lagrangian_lr, **kwargs)
        self.lagrangian = ProjectedLagrangianV4(lagrangian_lr)

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        observation = batch["obs"]
        action = batch["action"]
        reward = batch["reward"]
        world_model_cost_to_go = batch["cost"]
        next_observation = batch["next_obs"]
        done = batch["done"]
        discount = batch["discount"]

        with torch.no_grad():
            next_action, next_log_probability = self.actor(next_observation)
            next_q = torch.minimum(
                self.target_q1(next_observation, next_action),
                self.target_q2(next_observation, next_action),
            )
            reward_target = reward + discount * (1.0 - done) * (
                next_q - self.alpha.detach() * next_log_probability
            )
            cost_target = (
                world_model_cost_to_go
                if self.use_lagrangian else torch.zeros_like(world_model_cost_to_go)
            )

        q1 = self.q1(observation, action)
        q2 = self.q2(observation, action)
        critic_loss = (
            (q1 - reward_target).square().mean()
            + (q2 - reward_target).square().mean()
        )
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
        critic_parameters = [parameter for critic in actor_critics for parameter in critic.parameters()]
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
            "train/target_q": float(reward_target.detach().mean()),
            "train/cost_q": float((0.5 * (cost_q1 + cost_q2)).detach().mean()),
            "train/cost_target_q": float(cost_target.detach().mean()),
            "train/world_model_cost_target": float(world_model_cost_to_go.detach().mean()),
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
