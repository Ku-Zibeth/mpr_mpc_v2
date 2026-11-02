"""Residual SAC-Lagrangian agent for normalized ``[delta_d, delta_v]`` actions."""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .lagrangian import ProjectedLagrangian
from .networks import Critic, GaussianActor
from .types import ResidualPolicyOutput


class ResidualSACAgent:
    action_dim = 2

    def __init__(
        self,
        state_dim: int,
        cfg: dict,
        *,
        device: str | torch.device,
    ):
        self.state_dim = int(state_dim)
        self.device = torch.device(device)
        hidden_dims: Sequence[int] = cfg.get("hidden_dims", [256, 256])
        self.gamma = float(cfg.get("gamma", 0.99))
        self.tau = float(cfg.get("tau", 0.05))
        self.auto_alpha = bool(cfg.get("auto_alpha", True))
        self.target_entropy = float(cfg.get("target_entropy", -self.action_dim))
        self.use_lagrangian = bool(cfg.get("use_lagrangian", True))
        self.cost_limit = float(cfg.get("cost_limit", 5.0))
        self.lagrangian_rescaling = bool(cfg.get("lagrangian_rescaling", True))
        self.constraint_alignment = bool(cfg.get("constraint_alignment", False))
        self.constraint_mode = str(cfg.get("constraint_mode", "episode"))
        if self.constraint_mode not in {"episode", "discounted_episode"}:
            raise ValueError("constraint_mode must be episode or discounted_episode.")
        self.cost_gamma = (1.0 if self.constraint_mode == "episode" else self.gamma)
        self.conservative_cost = bool(cfg.get("conservative_cost", False))
        self.nonnegative_cost = bool(cfg.get("nonnegative_cost", False))
        self.risk_auxiliary_weight = float(cfg.get("risk_auxiliary_weight", 0.0))

        self.actor = GaussianActor(
            self.state_dim,
            self.action_dim,
            hidden_dims,
            log_std_min=float(cfg.get("log_std_min", -20.0)),
            log_std_max=float(cfg.get("log_std_max", 2.0)),
        ).to(self.device)
        self.q1 = Critic(self.state_dim, self.action_dim, hidden_dims).to(self.device)
        self.q2 = Critic(self.state_dim, self.action_dim, hidden_dims).to(self.device)
        self.target_q1 = deepcopy(self.q1).to(self.device).requires_grad_(False)
        self.target_q2 = deepcopy(self.q2).to(self.device).requires_grad_(False)
        self.cost_q1 = Critic(self.state_dim, self.action_dim, hidden_dims).to(self.device)
        self.cost_q2 = Critic(self.state_dim, self.action_dim, hidden_dims).to(self.device)
        self.target_cost_q1 = deepcopy(self.cost_q1).to(self.device).requires_grad_(False)
        self.target_cost_q2 = deepcopy(self.cost_q2).to(self.device).requires_grad_(False)

        actor_lr = float(cfg.get("actor_lr", cfg.get("lr", 5e-4)))
        critic_lr = float(cfg.get("critic_lr", cfg.get("lr", 1e-3)))
        alpha_lr = float(cfg.get("alpha_lr", cfg.get("lr", 3e-4)))
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=critic_lr
        )
        self.cost_critic_optimizer = torch.optim.Adam(
            list(self.cost_q1.parameters()) + list(self.cost_q2.parameters()), lr=critic_lr
        )
        alpha = max(float(cfg.get("alpha", 0.005)), 1e-8)
        self.log_alpha = torch.tensor(
            np.log(alpha),
            device=self.device,
            dtype=torch.float32,
            requires_grad=self.auto_alpha,
        )
        self.alpha_optimizer = (
            torch.optim.Adam([self.log_alpha], lr=alpha_lr) if self.auto_alpha else None
        )
        self.lagrangian = ProjectedLagrangian(
            learning_rate=float(cfg.get("lagrangian_lr", 0.01)),
            initial_value=float(cfg.get("lagrangian_initial", 0.0)),
        )
        self.update_steps = 0
        self.risk_updates = 0
        self.risk_warmup_updates = max(1, int(cfg.get("risk_auxiliary_warmup", 1000)))
        self.risk_predictor = None
        if self.risk_auxiliary_weight > 0:
            self.risk_predictor = Critic(self.state_dim, self.action_dim, hidden_dims).to(self.device)
            self.risk_optimizer = torch.optim.Adam(self.risk_predictor.parameters(), lr=critic_lr)

    def _cost_value(self, critic, state, action):
        value = critic(state, action)
        return F.softplus(value) if self.nonnegative_cost else value

    def _combined_cost(self, first, second):
        return torch.maximum(first, second) if self.conservative_cost else 0.5 * (first + second)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @property
    def lagrangian_multiplier(self) -> float:
        return self.lagrangian.value if self.use_lagrangian else 0.0

    @torch.no_grad()
    def select_action(
        self,
        state: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> ResidualPolicyOutput:
        state = state.to(self.device, dtype=torch.float32)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.shape[-1] != self.state_dim:
            raise ValueError(f"Residual state dim {state.shape[-1]} != {self.state_dim}.")
        action, _, mean, log_std = self.actor(state, deterministic=deterministic)
        return ResidualPolicyOutput(
            action=action.clamp(-1.0, 1.0).detach(),
            mean=torch.tanh(mean).detach(),
            log_std=log_std.detach(),
            warmup=False,
        )

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        state = batch["state"]
        action = batch["action"]
        reward = batch["reward"]
        cost = batch["cost"]
        next_state = batch["next_state"]
        done = batch["done"]
        discount = batch["discount"]

        with torch.no_grad():
            next_action, next_log_prob, _, _ = self.actor(next_state)
            next_q = torch.minimum(
                self.target_q1(next_state, next_action),
                self.target_q2(next_state, next_action),
            )
            reward_target = reward + discount * (1.0 - done) * (
                next_q - self.alpha.detach() * next_log_prob
            )
            if self.use_lagrangian:
                next_cost_q = self._combined_cost(
                    self._cost_value(self.target_cost_q1, next_state, next_action),
                    self._cost_value(self.target_cost_q2, next_state, next_action),
                )
                cost_discount = self.cost_gamma if self.constraint_alignment else discount
                cost_target = cost + cost_discount * (1.0 - done) * next_cost_q
            else:
                cost_target = torch.zeros_like(cost)

        q1 = self.q1(state, action)
        q2 = self.q2(state, action)
        critic_loss = (q1 - reward_target).square().mean() + (q2 - reward_target).square().mean()
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        reward_critic_grad_norm = self._check_gradients(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            "reward critic",
        )
        self.critic_optimizer.step()

        if self.use_lagrangian:
            cost_q1 = self._cost_value(self.cost_q1, state, action)
            cost_q2 = self._cost_value(self.cost_q2, state, action)
            cost_critic_loss = (
                F.smooth_l1_loss(cost_q1, cost_target)
                + F.smooth_l1_loss(cost_q2, cost_target)
            )
            self.cost_critic_optimizer.zero_grad(set_to_none=True)
            cost_critic_loss.backward()
            cost_critic_grad_norm = self._check_gradients(
                list(self.cost_q1.parameters()) + list(self.cost_q2.parameters()),
                "cost critic",
            )
            self.cost_critic_optimizer.step()
        else:
            cost_q1 = cost_q2 = torch.zeros_like(q1)
            cost_critic_loss = torch.zeros((), device=self.device)
            cost_critic_grad_norm = torch.zeros((), device=self.device)

        critic_parameters = [
            parameter
            for critic in (self.q1, self.q2, self.cost_q1, self.cost_q2)
            for parameter in critic.parameters()
        ]
        for parameter in critic_parameters:
            parameter.requires_grad_(False)
        risk_loss = torch.zeros((), device=self.device)
        risk_valid = batch.get("sac_risk_valid")
        if self.risk_predictor is not None and risk_valid is not None and risk_valid.bool().any():
            mask = risk_valid.squeeze(-1).bool() & torch.isfinite(batch["sac_predicted_cost"].squeeze(-1))
            if not mask.any():
                raise ValueError("Validated SAC risk labels must be finite.")
            prediction = F.softplus(self.risk_predictor(state[mask], action[mask]))
            risk_loss = F.smooth_l1_loss(prediction, batch["sac_predicted_cost"][mask])
            self.risk_optimizer.zero_grad(set_to_none=True)
            risk_loss.backward()
            self.risk_optimizer.step()
            self.risk_updates += 1
        try:
            policy_action, log_prob, _, _ = self.actor(state)
            policy_q = torch.minimum(
                self.q1(state, policy_action),
                self.q2(state, policy_action),
            )
            actor_sac_loss = (self.alpha.detach() * log_prob - policy_q).mean()
            lagrangian = self.lagrangian_multiplier
            if self.use_lagrangian:
                policy_cost = self._combined_cost(
                    self._cost_value(self.cost_q1, state, policy_action),
                    self._cost_value(self.cost_q2, state, policy_action),
                ).mean()
            else:
                policy_cost = torch.zeros((), device=self.device)
            actor_safety_loss = lagrangian * policy_cost
            auxiliary = torch.zeros((), device=self.device)
            if self.risk_predictor is not None and self.risk_updates >= self.risk_warmup_updates:
                self.risk_predictor.requires_grad_(False)
                auxiliary = F.softplus(self.risk_predictor(state, policy_action)).mean()
            rescaling = 1.0 / (1.0 + lagrangian) if self.lagrangian_rescaling else 1.0
            actor_loss = rescaling * (actor_sac_loss + actor_safety_loss + self.risk_auxiliary_weight * auxiliary)
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_grad_norm = self._check_gradients(self.actor.parameters(), "actor")
            self.actor_optimizer.step()
        finally:
            for parameter in critic_parameters:
                parameter.requires_grad_(True)
            if self.risk_predictor is not None:
                self.risk_predictor.requires_grad_(True)

        alpha_loss = torch.zeros((), device=self.device)
        if self.auto_alpha:
            alpha_loss = -(
                self.log_alpha * (log_prob.detach() + self.target_entropy)
            ).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            if self.log_alpha.grad is None or not torch.isfinite(self.log_alpha.grad).all():
                raise RuntimeError("Residual SAC alpha gradient contains non-finite values.")
            self.alpha_optimizer.step()

        with torch.no_grad():
            self._soft_update(self.target_q1, self.q1)
            self._soft_update(self.target_q2, self.q2)
            if self.use_lagrangian:
                self._soft_update(self.target_cost_q1, self.cost_q1)
                self._soft_update(self.target_cost_q2, self.cost_q2)
        self.update_steps += 1
        info = {
            "sac/loss_actor": actor_loss.detach(),
            "sac/loss_actor_sac": actor_sac_loss.detach(),
            "sac/loss_actor_safety": actor_safety_loss.detach(),
            "sac/loss_reward_critic": critic_loss.detach(),
            "sac/loss_cost_critic": cost_critic_loss.detach(),
            "sac/predicted_cost": policy_cost.detach(),
            "sac/risk_auxiliary_loss": risk_loss.detach(),
            "sac/risk_auxiliary_actor": auxiliary.detach(),
            "sac/cost_gamma": torch.tensor(self.cost_gamma if self.constraint_alignment else self.gamma, device=self.device),
            "sac/loss_alpha": alpha_loss.detach(),
            "sac/alpha": self.alpha.detach(),
            "sac/log_alpha": self.log_alpha.detach(),
            "sac/entropy": (-log_prob).detach().mean(),
            "sac/target_entropy": torch.tensor(self.target_entropy, device=self.device),
            "sac/q1": q1.detach().mean(),
            "sac/q2": q2.detach().mean(),
            "sac/q_mean": (0.5 * (q1 + q2)).detach().mean(),
            "sac/target_q_mean": reward_target.detach().mean(),
            "sac/target_q_max": reward_target.detach().max(),
            "sac/cost_q1": cost_q1.detach().mean(),
            "sac/cost_q2": cost_q2.detach().mean(),
            "sac/cost_q_mean": (0.5 * (cost_q1 + cost_q2)).detach().mean(),
            "sac/cost_target_mean": cost_target.detach().mean(),
            "sac/cost_target_max": cost_target.detach().max(),
            "lagrangian/value": torch.tensor(float(lagrangian), device=self.device),
            "lagrangian/cost_limit": torch.tensor(self.cost_limit, device=self.device),
            "lagrangian/lr": torch.tensor(self.lagrangian.learning_rate, device=self.device),
            "lagrangian/rescaling": torch.tensor(float(rescaling), device=self.device),
        }
        if (self.update_steps - 1) % 1000 == 0:
            info.update(
                {
                    "sac/actor_grad_norm": actor_grad_norm.detach(),
                    "sac/reward_critic_grad_norm": reward_critic_grad_norm.detach(),
                    "sac/cost_critic_grad_norm": cost_critic_grad_norm.detach(),
                }
            )
        return info

    def update_lagrangian(self, episode_cost: float, episode_steps: int = 1) -> dict[str, float]:
        if not self.use_lagrangian:
            return {
                "lagrangian/value": 0.0,
                "lagrangian/cost_limit": self.cost_limit,
                "lagrangian/episode_cost": float(episode_cost),
                "lagrangian/episode_cost_minus_limit": float(episode_cost) - self.cost_limit,
                "lagrangian/episode_cost_per_step": float(episode_cost) / max(int(episode_steps), 1),
                "lagrangian/update_delta": 0.0,
                "lagrangian/old_value": 0.0,
                "lagrangian/new_value": 0.0,
            }
        old_value = self.lagrangian.value
        new_value = self.lagrangian.step(episode_cost, self.cost_limit)
        return {
            "lagrangian/value": new_value,
            "lagrangian/cost_limit": self.cost_limit,
            "lagrangian/episode_cost": float(episode_cost),
            "lagrangian/constraint_violation": float(episode_cost) - self.cost_limit,
            "lagrangian/episode_cost_minus_limit": float(episode_cost) - self.cost_limit,
            "lagrangian/episode_cost_per_step": float(episode_cost) / max(int(episode_steps), 1),
            "lagrangian/update_delta": new_value - old_value,
            "lagrangian/old_value": old_value,
            "lagrangian/new_value": new_value,
        }

    def _soft_update(self, target, source) -> None:
        for target_parameter, parameter in zip(target.parameters(), source.parameters()):
            target_parameter.lerp_(parameter, self.tau)

    def _check_gradients(self, parameters, label: str) -> torch.Tensor:
        total = torch.zeros((), device=self.device)
        for parameter in parameters:
            if parameter.grad is None:
                continue
            grad = parameter.grad.detach()
            if not torch.isfinite(grad).all():
                raise RuntimeError(f"Residual SAC {label} gradient contains non-finite values.")
            total = total + grad.square().sum()
        return total.sqrt()

    def train(self) -> None:
        self.actor.train()
        self.q1.train()
        self.q2.train()
        self.cost_q1.train()
        self.cost_q2.train()

    def eval(self) -> None:
        self.actor.eval()
        self.q1.eval()
        self.q2.eval()
        self.cost_q1.eval()
        self.cost_q2.eval()

    def state_dict(self) -> dict:
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "use_lagrangian": self.use_lagrangian,
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "target_q1": self.target_q1.state_dict(),
            "target_q2": self.target_q2.state_dict(),
            "cost_q1": self.cost_q1.state_dict(),
            "cost_q2": self.cost_q2.state_dict(),
            "target_cost_q1": self.target_cost_q1.state_dict(),
            "target_cost_q2": self.target_cost_q2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "cost_critic_optimizer": self.cost_critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_optimizer": self.alpha_optimizer.state_dict() if self.alpha_optimizer else None,
            "lagrangian": self.lagrangian.state_dict(),
            "update_steps": self.update_steps,
            "constraint_alignment": self.constraint_alignment,
            "constraint_mode": self.constraint_mode,
            "nonnegative_cost": self.nonnegative_cost,
            "risk_predictor": self.risk_predictor.state_dict() if self.risk_predictor is not None else None,
            "risk_optimizer": self.risk_optimizer.state_dict() if self.risk_predictor is not None else None,
            "risk_updates": self.risk_updates,
        }

    def load_state_dict(self, state: dict, *, load_optimizer: bool = False) -> None:
        if int(state["state_dim"]) != self.state_dim or int(state["action_dim"]) != self.action_dim:
            raise ValueError("Residual SAC checkpoint dimensions do not match this run.")
        if load_optimizer:
            for key, default in (("constraint_alignment", False), ("constraint_mode", "episode"), ("nonnegative_cost", False)):
                if state.get(key, default) != getattr(self, key):
                    raise ValueError(f"Cannot resume SAC with changed {key}; use evaluation-only loading or retrain.")
        for name in (
            "actor",
            "q1",
            "q2",
            "target_q1",
            "target_q2",
            "cost_q1",
            "cost_q2",
            "target_cost_q1",
            "target_cost_q2",
        ):
            getattr(self, name).load_state_dict(state[name])
        self.log_alpha.data.copy_(state["log_alpha"].to(self.device))
        if "lagrangian" in state:
            self.lagrangian.load_state_dict(state["lagrangian"])
        self.update_steps = int(state.get("update_steps", 0))
        if self.risk_predictor is not None and state.get("risk_predictor") is not None:
            self.risk_predictor.load_state_dict(state["risk_predictor"])
            self.risk_updates = int(state.get("risk_updates", 0))
            if load_optimizer and state.get("risk_optimizer") is not None:
                self.risk_optimizer.load_state_dict(state["risk_optimizer"])
        if load_optimizer:
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
            self.critic_optimizer.load_state_dict(state["critic_optimizer"])
            self.cost_critic_optimizer.load_state_dict(state["cost_critic_optimizer"])
            if self.alpha_optimizer is not None and state.get("alpha_optimizer") is not None:
                self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
