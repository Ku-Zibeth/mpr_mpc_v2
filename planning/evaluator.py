"""Batched H-step reward, terminal-Q and one-step-cost model rollouts."""

from __future__ import annotations

import math
import torch

from mpr_mpc._bootstrap import bootstrap
from .timing import PhaseTimings
from .types import TrajectoryConsequence
from .cost_validation import candidate_ranking_ready, cost_definition, _weight_version, CostModelValidator

bootstrap()
from common import math as td_math  # noqa: E402


class TrajectoryWorldModelEvaluator:
    def validate_value_ranking(self, candidates):
        """Reuse matched-origin ranking statistics; never approve truncated proxies.

        Labels must be independent, executed discounted returns through termination,
        from identical restored simulator states. An H-step reward/correlation alone
        is insufficient to validate the terminal-Q ranking.
        """
        import numpy as np
        self._value_ranking_evidence = None
        rows = list(candidates)
        if not rows:
            return {"validated": False, "reason": "no_matched_executed_returns"}
        validator = CostModelValidator(dict(getattr(self.agent.cfg, "cost", {}) or {}),
                                       label_definition="executed_discounted_task_return")
        validator.set_candidates(rows)  # validates same observations/state IDs/actions
        qualified = all(r.get("episode_complete", False) and r.get("heldout", False)
                        and r.get("horizon") == self.horizon for r in rows)
        # Recompute predictions with these weights; caller-supplied old scores
        # must not certify a different checkpoint. No network or weight changes.
        prediction = []
        for row in rows:
            obs = torch.as_tensor(row["observation"], device=self.agent.device, dtype=torch.float32)
            sequence = torch.as_tensor(row["planned_actions"], device=self.agent.device, dtype=torch.float32)
            prediction.append(-float(self.evaluate_mppi_candidates(
                self.encode(obs), sequence[None], compute_cost=False).total_value[0]))
        prediction = np.asarray(prediction, dtype=float)
        actual = -np.asarray([r["actual_return"] for r in rows], dtype=float)
        stats = {}
        passed = bool(qualified and np.isfinite([prediction, actual]).all()
                      and validator._ranking_passed(prediction, actual, rows, stats))
        self._value_ranking_evidence = (self.horizon, _weight_version(self.agent.model)) if passed else None
        return dict(stats, validated=passed,
                    reason="matched_holdout_passed" if passed else "insufficient_matched_holdout_evidence")

    def value_ranking_ready(self):
        evidence = getattr(self, "_value_ranking_evidence", None)
        return evidence is not None and evidence == (self.horizon, _weight_version(self.agent.model))

    def __init__(self, agent, horizon):
        self.agent = agent
        self.horizon = int(horizon)
        if self.horizon <= 0:
            raise ValueError("World Model horizon must be positive.")

    @torch.no_grad()
    def encode(self, observation, task=None):
        observation = observation.to(self.agent.device, non_blocking=True)
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        self.agent.model.eval()
        return self.agent.model.encode(observation, task)

    def _discount(self, task, actions):
        if self.agent.cfg.multitask:
            return self.agent.discount[torch.as_tensor(task, device=actions.device)].reshape(-1, 1)
        return actions.new_tensor(float(self.agent.discount))

    @torch.no_grad()
    def evaluate(self, z_t, actions, *, terminal_action_mode, task=None, compute_cost=None):
        if terminal_action_mode not in {"policy", "trajectory"}:
            raise ValueError("terminal_action_mode must be 'policy' or 'trajectory'.")
        if actions.ndim != 3 or actions.shape[-1] != int(self.agent.cfg.action_dim):
            raise ValueError("Actions must have shape [B,S,action_dim].")
        batch, steps, _ = actions.shape
        if steps != self.horizon + (terminal_action_mode == "trajectory"):
            raise ValueError(f"{terminal_action_mode} mode has the wrong number of actions.")
        if not torch.isfinite(actions).all():
            raise ValueError("World Model actions must be finite.")
        actions = actions.to(self.agent.device).clamp(-1, 1)
        z = z_t.to(self.agent.device).reshape(-1, int(self.agent.cfg.latent_dim))
        if z.shape[0] == 1:
            z = z.expand(batch, -1)
        if z.shape[0] != batch:
            raise ValueError("Latent batch does not match candidate batch.")
        model = self.agent.model
        model.eval()
        cfg = getattr(self.agent.cfg, "cost", {}) or {}
        enabled = bool(cfg.get("enabled", False))
        compute_cost = enabled if compute_cost is None else bool(compute_cost)
        if compute_cost and (not enabled or not callable(getattr(model, "cost", None))):
            raise RuntimeError("MPPI WM Cost requires cost.enabled=true and a Cost Head.")
        gamma_c = float(cfg.get("gamma", .99))
        if not math.isfinite(gamma_c) or not 0 <= gamma_c <= 1:
            raise ValueError("cost.gamma must be finite and in [0,1].")
        discount_step = self._discount(task, actions)
        discount = actions.new_ones((batch, 1))
        terminated = actions.new_zeros((batch, 1))
        total = actions.new_zeros((batch, 1))
        cost_total = torch.zeros_like(total)
        rewards, cost_latents = [], []
        timing = PhaseTimings(actions.device)
        with timing.measure("world_model_rollout"):
            for step in range(self.horizon):
                action = actions[:, step]
                # Cost is instantaneous; discount and accumulate exactly once.
                reward = td_math.two_hot_inv(model.reward(z, action, task), self.agent.cfg)
                rewards.append(reward.squeeze(-1))
                total += discount * (1-terminated) * reward
                if compute_cost:
                    cost_latents.append(z)
                z = model.next(z, action, task)
                discount *= discount_step
                if bool(self.agent.cfg.episodic) and getattr(model, "_termination", None) is not None:
                    terminated = (terminated + (model.termination(z, task) > .5)).clamp_max(1)
            terminal_action = actions[:, self.horizon] if terminal_action_mode == "trajectory" else model.pi(z, task)[0]
            # Preserve the historical evaluator's deterministic full-ensemble Q mean.
            qs = td_math.two_hot_inv(model.Q(z, terminal_action, task, return_type="all"), self.agent.cfg)
            terminal_q = qs.mean(0)
            discounted_reward = total.clone()
            total += discount * (1-terminated) * terminal_q
            prediction = None
            if compute_cost:
                # Cost has no recurrent dependency. One H*B batch preserves
                # the same latent/action pairs and discount semantics while
                # avoiding H small MLP launches and CUDA timing event pairs.
                with timing.measure("cost_prediction"):
                    latent_batch = torch.stack(cost_latents, 1).reshape(batch*self.horizon, -1)
                    action_batch = actions[:, :self.horizon].reshape(batch*self.horizon, -1)
                    cost_task = task
                    if task is not None and torch.is_tensor(task) and task.numel() == batch:
                        cost_task = task.reshape(-1).repeat_interleave(self.horizon)
                    raw_cost = model.cost(latent_batch, action_batch, cost_task)
                    if raw_cost.shape != (batch*self.horizon, 1):
                        raise RuntimeError("WM Cost Head returned the wrong batched prediction shape.")
                    prediction = raw_cost.reshape(batch, self.horizon)
                    weights = gamma_c**torch.arange(self.horizon, device=actions.device, dtype=actions.dtype)
                    cost_total = (prediction*weights).sum(1, keepdim=True)
        timing_values = timing.finish()
        timing_values.setdefault("cost_prediction_ms", 0.0)
        if prediction is not None and (not torch.isfinite(prediction).all() or (prediction < 0).any()):
            raise RuntimeError("WM Cost Head produced non-finite or negative step costs; check weights and label scale.")
        if prediction is not None and not torch.isfinite(cost_total).all():
            raise RuntimeError("WM cumulative Cost overflowed; check horizon and prediction scale.")
        validated = bool(compute_cost and getattr(model, "cost_ready", False)
                         and self.horizon < 63
                         and int(model._cost_validated_horizons) & (1 << self.horizon))
        return TrajectoryConsequence(
            reward_sequence=torch.stack(rewards, 1), terminal_q=terminal_q.squeeze(-1),
            terminal_q_std=qs.std(0, unbiased=False).squeeze(-1), total_value=total.squeeze(-1),
            terminal_latent=z, predicted_cost=prediction,
            cumulative_cost=cost_total.squeeze(-1) if compute_cost else None,
            cost_validated=validated,
            cost_ranking_validated=bool(compute_cost and candidate_ranking_ready(
                model, self.horizon, cost_definition(self.agent.cfg))),
            cost_validated_mask=torch.full((batch,), validated, device=actions.device, dtype=torch.bool),
            cost_unavailable_reason="disabled" if not compute_cost else "available" if validated else "unvalidated",
            discounted_reward=discounted_reward.squeeze(-1), timings=timing_values,
        )

    def evaluate_trajectory_consequence(self, z_t, actions, task=None, *, compute_cost=None):
        return self.evaluate(z_t, actions, terminal_action_mode="trajectory", task=task, compute_cost=compute_cost)

    def evaluate_policy_terminal(self, z_t, actions, task=None):
        return self.evaluate(z_t, actions, terminal_action_mode="policy", task=task)

    def evaluate_mppi_candidates(self, z_t, actions, task=None, *, compute_cost=None):
        return self.evaluate_trajectory_consequence(z_t, actions, task=task, compute_cost=compute_cost)


__all__ = ["TrajectoryWorldModelEvaluator"]
