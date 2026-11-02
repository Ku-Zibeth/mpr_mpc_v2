"""V2 SAC-Lagrangian implementation reconstructed from the recorded source diff.

V2 uses PID multiplier updates, max aggregation for twin cost critics, MSE cost
regression, and a physical-residual trust penalty in the actor objective.
"""

from __future__ import annotations

from typing import Sequence

import torch

from sac.model import SACAgent


class PIDLagrangianV2:
    """Historical non-negative PID multiplier used by the V2 run."""

    def __init__(self, pid: Sequence[float] = (0.05, 0.0005, 0.1)):
        if len(pid) != 3:
            raise ValueError("lagrangian_pid must contain [kp, ki, kd].")
        self.pid = tuple(float(value) for value in pid)
        self.error_old = 0.0
        self.error_integral = 0.0
        self.value = 0.0

    def step(self, episode_cost: float, cost_limit: float) -> float:
        error = float(episode_cost) - float(cost_limit)
        error_diff = max(error - self.error_old, 0.0)
        self.error_integral = max(self.error_integral + error, 0.0)
        self.error_old = error
        kp, ki, kd = self.pid
        self.value = max(
            kp * error + ki * self.error_integral + kd * error_diff,
            0.0,
        )
        return self.value

    def state_dict(self) -> dict:
        return {
            "pid": self.pid,
            "error_old": self.error_old,
            "error_integral": self.error_integral,
            "value": self.value,
        }

    def load_state_dict(self, state: dict) -> None:
        self.pid = tuple(float(value) for value in state["pid"])
        self.error_old = float(state["error_old"])
        self.error_integral = float(state["error_integral"])
        self.value = float(state.get("value", state.get("lagrangian", 0.0)))


class SACAgentV2(SACAgent):
    """Historical V2 SAC network update with checkpoint-compatible modules."""

    algorithm_version = "lattice_tdmpc2_v2"

    def __init__(
        self,
        *args,
        lagrangian_pid: Sequence[float] = (0.05, 0.0005, 0.1),
        trust_loss_weight: float = 0.01,
        residual_sigma: Sequence[float] = (0.4, 2.0),
        **kwargs,
    ):
        kwargs.pop("lagrangian_lr", None)
        super().__init__(*args, lagrangian_lr=0.01, **kwargs)
        self.lagrangian = PIDLagrangianV2(lagrangian_pid)
        self.trust_loss_weight = float(trust_loss_weight)
        if len(residual_sigma) != self.action_dim:
            raise ValueError("residual_sigma must have one value per SAC action dimension.")
        self.residual_sigma = torch.as_tensor(
            residual_sigma, dtype=torch.float32, device=self.device
        ).clamp_min(1e-6)

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
                next_cost_q = torch.maximum(
                    self.target_cost_q1(next_observation, next_action),
                    self.target_cost_q2(next_observation, next_action),
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
                (cost_q1 - cost_target).square().mean()
                + (cost_q2 - cost_target).square().mean()
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
                policy_cost = torch.maximum(
                    self.cost_q1(observation, policy_action),
                    self.cost_q2(observation, policy_action),
                ).mean()
            else:
                policy_cost = torch.zeros((), device=self.device)
            lagrangian = self.lagrangian_multiplier
            actor_safety_loss = lagrangian * policy_cost
            rescaling = 1.0 / (1.0 + lagrangian) if self.lagrangian_rescaling else 1.0
            trust_loss = self._trust_loss(batch, policy_action)
            actor_loss = (
                rescaling * (actor_sac_loss + actor_safety_loss)
                + self.trust_loss_weight * trust_loss
            )
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
            "loss/trust": float(trust_loss.detach()),
            "loss/alpha": float(alpha_loss.detach()),
            "train/alpha": float(self.alpha.detach()),
            "train/entropy": float((-log_probability).detach().mean()),
            "train/q1": float(q1.detach().mean()),
            "train/q2": float(q2.detach().mean()),
            "train/target_q": float(target.detach().mean()),
            "train/cost_q": float(torch.maximum(cost_q1, cost_q2).detach().mean()),
            "train/cost_target_q": float(cost_target.detach().mean()),
            "lagrangian/value": float(lagrangian),
            "lagrangian/cost_limit": self.cost_limit,
            "lagrangian/rescaling": float(rescaling),
            "sac/critic_loss": float(critic_loss.detach()),
            "sac/cost_critic_loss": float(cost_critic_loss.detach()),
            "sac/actor_loss": float(actor_loss.detach()),
            "sac/actor_sac_loss": float(actor_sac_loss.detach()),
            "sac/trust_loss": float(trust_loss.detach()),
            "sac/alpha": float(self.alpha.detach()),
            "sac/entropy": float((-log_probability).detach().mean()),
            "sac/q1": float(q1.detach().mean()),
            "sac/q2": float(q2.detach().mean()),
        }

    def _trust_loss(
        self, batch: dict[str, torch.Tensor], normalized_action: torch.Tensor
    ) -> torch.Tensor:
        if self.trust_loss_weight <= 0.0 or "residual_low" not in batch:
            return normalized_action.new_zeros(())
        low = batch["residual_low"]
        high = batch["residual_high"]
        if low.shape != normalized_action.shape or high.shape != normalized_action.shape:
            raise ValueError("residual_low/high must have shape [B,A].")
        physical_delta = low + 0.5 * (normalized_action + 1.0) * (high - low)
        return ((physical_delta / self.residual_sigma).square().sum(dim=-1)).mean()

    def _soft_update(self, target, source) -> None:
        for target_parameter, parameter in zip(target.parameters(), source.parameters()):
            target_parameter.lerp_(parameter, self.tau)
