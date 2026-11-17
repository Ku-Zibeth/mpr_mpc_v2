"""Batched H-step reward, terminal-Q and one-step-cost model rollouts."""

from __future__ import annotations

import math
import torch

from mpr_mpc._bootstrap import bootstrap
from .timing import PhaseTimings
from .types import TrajectoryConsequence
from .cost_validation import candidate_ranking_ready, cost_definition, _weight_version

bootstrap()
from common import math as td_math  # noqa: E402


class TrajectoryWorldModelEvaluator:
    def __init__(self, agent, horizon):
        self.agent = agent
        self.horizon = int(horizon)
        self.sac_value = None
        self.cost_validation_model = None
        self._cost_source_versions = None
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
    def _terminal_prediction(self,z,actions,task,terminal_action_mode,q_aggregation,q_indices,capture_ensemble):
        model = self.agent.model
        if terminal_action_mode == "state_value":
            if self.sac_value is None:
                raise RuntimeError("Attach a validated SAC value, or explicitly attach it for offline experiments.")
            value = self.sac_value(z).reshape(-1,1)
            return value,torch.full_like(value,float("nan")),None,None
        if terminal_action_mode == "trajectory":
            terminal_action = actions[:, self.horizon]
        elif terminal_action_mode == "policy_mean":
            # pi() exposes its squashed mean. Restore its unused sample's
            # RNG draw, so D2 does not inadvertently change random Q pairs.
            device = torch.device(self.agent.device)
            devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                _, info = model.pi(z, task)
            terminal_action = info["mean"]
        else:
            terminal_action = model.pi(z, task)[0]
        qs = None
        if q_aggregation == "tdmpc2_avg":
            # Official Q(avg) already inverse-transforms its random pair.
            # ONE call shares the pair across every candidate in this batch.
            terminal_q = model.Q(z, terminal_action, task, return_type="avg")
            q_std = torch.full_like(terminal_q, float("nan"))
        elif q_aggregation in {"ensemble_mean", "fixed_pair"}:
            # Frozen SAC feature semantics MUST retain the full-Q reducer.
            qs = td_math.two_hot_inv(model.Q(z, terminal_action, task, return_type="all"), self.agent.cfg)
            if q_aggregation == "fixed_pair":
                if (q_indices is None or len(q_indices) != 2 or len(set(q_indices)) != 2
                        or min(q_indices) < 0 or max(q_indices) >= len(qs)):
                    raise ValueError("Fixed Q pair requires two distinct valid member indices.")
                terminal_q = qs[list(q_indices)].mean(0)
            else:
                terminal_q = qs.mean(0)
            q_std = qs.std(0, unbiased=False)
        else:
            raise ValueError("Unknown terminal Q aggregation.")
        if capture_ensemble and qs is None:
            # Offline diagnostics only; never an extra online score or gate.
            qs = td_math.two_hot_inv(model.Q(z, terminal_action, task, return_type="all"), self.agent.cfg)
            q_std = qs.std(0, unbiased=False)
        return terminal_q,q_std,terminal_action,qs

    @torch.no_grad()
    def evaluate(self, z_t, actions, *, terminal_action_mode, task=None, compute_cost=None,
                 q_aggregation="ensemble_mean", q_indices=None, capture_ensemble=False,
                 terminal_q_weight=1.0, cost_weight=0.0, mppi_scoring=False):
        cost_weight = float(cost_weight)
        if not math.isfinite(cost_weight) or cost_weight < 0:
            raise ValueError("cost_weight must be finite and nonnegative.")
        terminal_q_weight = float(terminal_q_weight)
        if not math.isfinite(terminal_q_weight) or terminal_q_weight < 0:
            raise ValueError("terminal_q_weight must be finite and nonnegative.")
        if terminal_action_mode not in {"policy", "policy_mean", "trajectory", "state_value"}:
            raise ValueError("Unknown terminal action definition.")
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
            terminal_q,q_std,terminal_action,qs = self._terminal_prediction(
                z,actions,task,terminal_action_mode,q_aggregation,q_indices,capture_ensemble)
            discounted_reward = total.clone()
            terminal_value = discount * (1-terminated) * terminal_q
            if terminal_q_weight != 1.0:
                terminal_value = terminal_q_weight * terminal_value
            total += terminal_value
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
        validation_model = self.cost_validation_model if mppi_scoring and self.cost_validation_model is not None else model
        validated = bool(compute_cost and getattr(validation_model, "cost_ready", False)
                         and self.horizon < 63
                         and int(validation_model._cost_validated_horizons) & (1 << self.horizon))
        penalty, effective_weight = self._cost_penalty(cost_total.squeeze(-1) if compute_cost else None, cost_weight)
        total -= penalty[:, None] if penalty is not None else 0.0
        return TrajectoryConsequence(
            reward_sequence=torch.stack(rewards, 1), terminal_q=terminal_q.squeeze(-1),
            terminal_q_std=q_std.squeeze(-1), total_value=total.squeeze(-1),
            terminal_latent=z, predicted_cost=prediction,
            cumulative_cost=cost_total.squeeze(-1) if compute_cost else None,
            cost_validated=validated,
            cost_ranking_validated=bool(compute_cost and (self.cost_ranking_ready() if mppi_scoring else candidate_ranking_ready(
                model, self.horizon, cost_definition(self.agent.cfg)))),
            cost_validated_mask=torch.full((batch,), validated, device=actions.device, dtype=torch.bool),
            cost_unavailable_reason="disabled" if not compute_cost else "available" if validated else "unvalidated",
            discounted_reward=discounted_reward.squeeze(-1), timings=timing_values,
            terminal_actions=terminal_action,
            terminal_discount=(discount*(1-terminated)).squeeze(-1),
            q_member_values=qs.squeeze(-1) if qs is not None else None,
            cost_penalty=penalty, effective_cost_weight=effective_weight,
        )

    def evaluate_trajectory_consequence(self, z_t, actions, task=None, *, compute_cost=None):
        return self.evaluate(z_t, actions, terminal_action_mode="trajectory", task=task, compute_cost=compute_cost)

    def evaluate_policy_terminal(self, z_t, actions, task=None):
        return self.evaluate(z_t, actions, terminal_action_mode="policy", task=task)

    def new_mppi_q_pair(self):
        """Exactly one random-two draw for a whole plan, not for each batch."""
        count = int(self.agent.cfg.num_q)
        if count < 2:
            raise ValueError("Need at least two Q members.")
        return tuple(torch.randperm(count, device=self.agent.device)[:2].cpu().tolist())

    def attach_offline_sac_value(self, head):
        """For explicit offline counterfactuals only; config loading requires validation."""
        self.sac_value = head.to(self.agent.device).eval().requires_grad_(False)

    def load_sac_value(self, path, wm_checkpoint):
        from .sac_value import load_sac_value, policy_context_sha256
        self.sac_value, _ = load_sac_value(path, wm_checkpoint=wm_checkpoint,
            horizon=self.horizon, gamma=float(self.agent.discount), device=self.agent.device,
            policy_context=policy_context_sha256(self.agent.cfg))

    def _cost_penalty(self, cost, weight):
        # A continuous learned objective. No candidate is rejected by this term.
        ready = (weight > 0 and cost is not None
                 and self.cost_ranking_ready())
        effective = float(weight) if ready else 0.0
        return (cost * effective if cost is not None else None), effective

    def cost_ranking_ready(self):
        model = self.cost_validation_model if self.cost_validation_model is not None else self.agent.model
        if self.cost_validation_model is not None and self._cost_source_versions != _weight_version(self.agent.model):
            return False
        return candidate_ranking_ready(model,self.horizon,cost_definition(self.agent.cfg))

    def validate_mppi_cost(self, candidates):
        # Validation changes readiness buffers. Keep those changes off the
        # original WM, whose readiness flags are part of frozen SAC features.
        import copy
        from .cost_validation import CostModelValidator
        self.cost_validation_model = copy.deepcopy(self.agent.model).eval().requires_grad_(False)
        self._cost_source_versions = _weight_version(self.agent.model)
        validator = CostModelValidator(self.agent.cfg.cost,label_definition=cost_definition(self.agent.cfg))
        validator.set_candidates(candidates)
        return validator.validate(self.cost_validation_model,self.agent.device,horizon=self.horizon)

    @torch.no_grad()
    def rescore_cached_candidates(self, results, *, terminal_value="tdmpc2_q", terminal_action="candidate",
                                  q_aggregation="random_pair", q_indices=None, terminal_q_weight=1.,
                                  cost_weight=0., task=None):
        """One common terminal batch; no repeated reward/dynamics evaluation.

        Callers pass (consequence, indices) pairs in their action-pool order.
        In random-two mode, ONE new pair scores the entire common pool.
        """
        from dataclasses import replace
        def cat(name):
            values = [getattr(r,name) for r,_ in results]
            return None if any(v is None for v in values) else torch.cat([v[i] for v,(_,i) in zip(values,results)],0)
        mode = "state_value" if terminal_value == "sac_value" else "trajectory" if terminal_action == "candidate" else "policy_mean"
        z, prefix, discount = cat("terminal_latent"), cat("discounted_reward"), cat("terminal_discount")
        actions = z.new_zeros((len(z),self.horizon+1,int(self.agent.cfg.action_dim)))
        if mode == "trajectory":
            actions[:,-1] = cat("terminal_actions")
        reducers = {"random_pair":"tdmpc2_avg","fixed_pair":"fixed_pair","ensemble_mean":"ensemble_mean"}
        q,std,a,qs = self._terminal_prediction(z,actions,task,mode,reducers[q_aggregation],q_indices,False)
        penalty,effective = self._cost_penalty(cat("cumulative_cost"),cost_weight)
        value = prefix + float(terminal_q_weight)*discount*q.squeeze(-1)
        if penalty is not None:
            value -= penalty
        return replace(results[0][0],reward_sequence=cat("reward_sequence"),terminal_q=q.squeeze(-1),
            terminal_q_std=std.squeeze(-1),total_value=value,terminal_latent=z,discounted_reward=prefix,
            terminal_discount=discount,terminal_actions=a,q_member_values=qs.squeeze(-1) if qs is not None else None,
            cumulative_cost=cat("cumulative_cost"),predicted_cost=cat("predicted_cost"),cost_penalty=penalty,
            cost_validated_mask=cat("cost_validated_mask"),effective_cost_weight=effective,
            cost_ranking_validated=bool(cat("cumulative_cost") is not None and self.cost_ranking_ready()))

    def evaluate_mppi_candidates(self, z_t, actions, task=None, *, compute_cost=None,
                                 terminal_action="candidate", q_aggregation="random_pair",
                                 q_indices=None, capture_ensemble=False, terminal_q_weight=1.0,
                                 terminal_value="tdmpc2_q", cost_weight=0.0):
        if actions.ndim != 3 or actions.shape[1] != self.horizon+1:
            raise ValueError("MPPI keeps the Coordinator H+1 interface.")
        if terminal_action not in {"candidate", "policy_prior"}:
            raise ValueError("Unknown MPPI terminal action.")
        reducers = {"random_pair": "tdmpc2_avg", "fixed_pair": "fixed_pair", "ensemble_mean": "ensemble_mean"}
        if q_aggregation not in reducers:
            raise ValueError("Unknown MPPI Q reducer.")
        if terminal_value not in {"tdmpc2_q", "sac_value"}:
            raise ValueError("Unknown MPPI terminal value model.")
        if terminal_value == "sac_value" and terminal_q_weight != 1.:
            raise ValueError("SAC continuation value uses unit terminal weight, not Q-weight calibration.")
        mode = "state_value" if terminal_value == "sac_value" else "trajectory" if terminal_action == "candidate" else "policy_mean"
        controls = actions if mode == "trajectory" else actions[:, :self.horizon]
        return self.evaluate(z_t, controls, terminal_action_mode=mode, task=task,
            compute_cost=compute_cost, q_aggregation=reducers[q_aggregation],
            q_indices=q_indices, capture_ensemble=capture_ensemble, terminal_q_weight=terminal_q_weight,
            cost_weight=cost_weight, mppi_scoring=True)


__all__ = ["TrajectoryWorldModelEvaluator"]
