"""Safety-bounded H+1 local MPPI around one refined Lattice action sequence."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .kinematic_filter import BatchedKinematicCorridor, CorridorContext
from .multimodal import MultiModalProposals


MPPI_DIAGNOSTIC_KEYS = (
    "no_safe_candidate", "selected_safe", "baseline_safe", "expanded_search",
    "collision_reject_count", "offroad_reject_count", "collision_reject_rate",
    "offroad_reject_rate", "kinematic_reject_count", "baseline_soft_risk",
    "selected_soft_risk", "risk_reduction", "selected_min_clearance",
    "selected_min_ttc", "selected_boundary_margin", "cost_validated",
) + tuple(f"mode_{mode}_{metric}" for mode in ("nominal", "braking", "lateral", "joint")
          for metric in ("samples", "safe_ratio"))


def _action_vector(value, action_dim: int, name: str, device) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.ndim == 0:
        tensor = tensor.repeat(action_dim)
    if tensor.shape != (action_dim,):
        raise ValueError(f"{name} must be scalar or [{action_dim}], got {tuple(tensor.shape)}.")
    if not torch.isfinite(tensor).all() or torch.any(tensor < 0):
        raise ValueError(f"{name} values must be finite and non-negative.")
    return tensor


@dataclass(frozen=True)
class LocalMPPIResult:
    action: torch.Tensor
    selected_sequence: torch.Tensor
    initial_mean: torch.Tensor
    final_mean: torch.Tensor
    initial_std: torch.Tensor
    final_std: torch.Tensor
    baseline_wm_value: torch.Tensor
    selected_wm_value: torch.Tensor
    wm_value_gain: torch.Tensor
    baseline_score: torch.Tensor
    selected_score: torch.Tensor
    planner_score_gain: torch.Tensor
    # Compatibility aliases; these are all planner-score quantities.
    baseline_value: torch.Tensor
    final_value: torch.Tensor
    planner_gain: torch.Tensor
    elite_values: torch.Tensor
    baseline_selected: bool
    corridor_reject_count: int
    corridor_reject_rate: float
    max_lateral_deviation: float
    max_action_delta: torch.Tensor
    safety_metrics: dict = field(default_factory=dict)


class LocalMPPI:
    def __init__(self, evaluator, cfg, *, horizon: int, action_dim: int, device, control_dt=0.1):
        self.evaluator = evaluator
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        self.num_samples = int(cfg.get("num_samples", 512))
        self.num_elites = int(cfg.get("num_elites", 64))
        self.iterations = int(cfg.get("iterations", 6))
        self.min_std_vector = _action_vector(cfg.get("min_std", [0.005, 0.02]), self.action_dim, "min_std", self.device)
        self.max_std_vector = _action_vector(cfg.get("max_std", [0.05, 0.20]), self.action_dim, "max_std", self.device)
        self.initial_std_vector = _action_vector(cfg.get("initial_std", [0.025, 0.10]), self.action_dim, "initial_std", self.device)
        self.max_delta = _action_vector(cfg.get("max_delta", [0.05, 0.20]), self.action_dim, "max_delta", self.device)
        self.temperature = float(cfg.get("temperature", 0.5))
        self.proposal_init = str(cfg.get("proposal_init", "structured_baseline"))
        self.blend_alpha = float(cfg.get("blend_alpha", 0.8))
        self.deviation_coef = float(cfg.get("deviation_coef", 5.0))
        self.smoothness_coef = float(cfg.get("smoothness_coef", 1.0))
        self.throttle_deviation_weight = float(cfg.get("throttle_deviation_weight", 0.25))
        self.min_improvement_abs = float(cfg.get("min_improvement_abs", 1.0))
        self.min_improvement_ratio = float(cfg.get("min_improvement_ratio", 0.01))
        self.uncertainty_coef = float(cfg.get("uncertainty_coef", 0.0))
        self.eval_seed = int(cfg.get("eval_seed", 0))
        self.multimodal = bool(cfg.get("multimodal", False))
        self._sampling_cfg = dict(cfg)
        self._mode_bank = None
        self.mode_weights = torch.tensor(cfg.get("mode_weights", [1, 1, 1, 1]), dtype=torch.float32)
        if self.mode_weights.shape != (4,) or not torch.isfinite(self.mode_weights).all() or (self.mode_weights < 0).any() or self.mode_weights.sum() <= 0:
            raise ValueError("mode_weights must contain four nonnegative weights with a positive sum.")
        self.braking_center = float(cfg.get("braking_center", -0.6))
        self.braking_std = float(cfg.get("braking_std", 0.15))
        self.lateral_shift = float(cfg.get("lateral_shift", 0.12))
        self.lateral_delta = float(cfg.get("lateral_delta", 0.25))
        self.braking_delta = float(cfg.get("braking_delta", 0.4))
        self.mode_score_delta = _action_vector(cfg.get("mode_score_delta", [0.25, 1.5]), self.action_dim, "mode_score_delta", self.device)
        if (self.mode_score_delta <= 0).any():
            raise ValueError("mode_score_delta must be positive.")
        self.absolute_smoothness = bool(cfg.get("absolute_smoothness", False))
        self.safety_evaluator = None
        self.cost_coef = float(cfg.get("cost_coef", 1.0))
        self.geometry_coef = float(cfg.get("geometry_coef", 10.0))
        self.expanded_delta = _action_vector(cfg.get("expanded_delta", [0.3, 1.5]), self.action_dim, "expanded_delta", self.device)
        if self.num_samples < 2 or not 0 < self.num_elites <= self.num_samples:
            raise ValueError("Require 2 <= num_samples and 0 < num_elites <= num_samples.")
        if self.iterations <= 0:
            raise ValueError("MPPI iterations must be positive.")
        if torch.any(self.min_std_vector > self.max_std_vector):
            raise ValueError("min_std cannot exceed max_std.")
        if torch.any(self.initial_std_vector < self.min_std_vector) or torch.any(self.initial_std_vector > self.max_std_vector):
            raise ValueError("initial_std must be within [min_std,max_std].")
        if torch.any(self.max_delta <= 0):
            raise ValueError("max_delta must be positive.")
        if self.proposal_init not in {"structured_baseline", "previous_mean", "blend"}:
            raise ValueError("Unknown proposal_init mode.")
        self.corridor = BatchedKinematicCorridor(cfg, control_dt=control_dt, device=device)
        self._prev_mean: torch.Tensor | None = None

    def reset(self) -> None:
        self._prev_mean = None
        self._mode_bank = None

    def _shift_previous(self, baseline: torch.Tensor) -> torch.Tensor:
        if self._prev_mean is None:
            return baseline.clone()
        shifted = baseline.clone()
        shifted[:-1] = self._prev_mean[1:]
        shifted[-1] = baseline[-1]
        return self._project(shifted, baseline)

    def _project(self, actions: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
        delta = actions - baseline.reshape(*([1] * (actions.ndim - 2)), *baseline.shape)
        max_delta = self.max_delta.reshape(*([1] * (actions.ndim - 1)), self.action_dim)
        delta = torch.maximum(torch.minimum(delta, max_delta), -max_delta)
        return (baseline.reshape(*([1] * (actions.ndim - 2)), *baseline.shape) + delta).clamp(-1.0, 1.0)

    def initial_mean(self, baseline: torch.Tensor) -> torch.Tensor:
        expected = (self.horizon + 1, self.action_dim)
        if baseline.shape != expected:
            raise ValueError(f"Baseline shape {tuple(baseline.shape)} != {expected}.")
        baseline = baseline.to(self.device).clamp(-1.0, 1.0)
        if self.proposal_init == "structured_baseline":
            return baseline.clone()
        shifted = self._shift_previous(baseline)
        if self.proposal_init == "previous_mean":
            return shifted
        return self._project(
            self.blend_alpha * baseline + (1.0 - self.blend_alpha) * shifted,
            baseline,
        )

    def _penalties(self, actions: torch.Tensor, baseline: torch.Tensor):
        delta = actions - baseline.unsqueeze(0)
        score_delta = self.mode_score_delta if self.multimodal else self.max_delta
        normalized = delta / score_delta.view(1, 1, -1)
        deviation = normalized[..., 0].square()
        if self.action_dim > 1:
            deviation = deviation + self.throttle_deviation_weight * normalized[..., 1].square()
        deviation = deviation.mean(dim=1)
        if delta.shape[1] > 1:
            smooth_source = actions if self.absolute_smoothness else delta
            smoothness = (smooth_source[:, 1:] - smooth_source[:, :-1]).square().sum(dim=-1).mean(dim=1)
        else:
            smoothness = delta.new_zeros(delta.shape[0])
        return deviation, smoothness

    def _candidate_batch(self, mean, std, baseline, generator=None):
        if self.multimodal:
            if self._mode_bank is None:
                self._mode_bank = MultiModalProposals(self._sampling_cfg, baseline, mean, std,
                    num_samples=self.num_samples, num_elites=self.num_elites, temperature=self.temperature,
                    min_std=self.min_std_vector, max_std=self.max_std_vector, max_delta=self.max_delta)
            return self._mode_bank.sample(generator)
        noise = torch.randn(
            self.num_samples,
            self.horizon + 1,
            self.action_dim,
            device=self.device,
            generator=generator,
        )
        actions = self._project(mean.unsqueeze(0) + std.unsqueeze(0) * noise, baseline)
        actions[0] = baseline
        actions[1] = self._project(mean, baseline)
        return actions

    @torch.no_grad()
    def plan(
        self,
        z_t: torch.Tensor,
        baseline: torch.Tensor,
        *,
        eval_mode: bool,
        corridor_context: CorridorContext | None = None,
        safety_context=None,
        alternatives=None,
        task=None,
    ):
        if not torch.isfinite(baseline).all():
            raise ValueError("MPPI baseline must be finite.")
        if self.safety_evaluator is not None and safety_context is None:
            raise ValueError("Dynamic safety is enabled but live safety context is missing.")
        self._mode_bank = None
        baseline = baseline.to(self.device).clamp(-1.0, 1.0)
        mean = self.initial_mean(baseline)
        initial_mean = mean.clone()
        std = self.initial_std_vector.view(1, -1).expand_as(mean).clone()
        initial_std = std.clone()
        generator = None
        if eval_mode:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.eval_seed)

        total_rejected = 0
        total_checked = 0
        safety_metrics = {}
        expanded = False
        actions = elite_actions = elite_values = elite_wm_values = weights = None
        baseline_wm_value = baseline_score = None
        previous_best = None
        previous_best_mode = -1
        for _ in range(self.iterations):
            actions = self._candidate_batch(mean, std, baseline, generator)
            mode_ids = self._mode_bank.ids.clone() if self._mode_bank is not None else None
            if previous_best is not None:
                actions[1] = previous_best
                if mode_ids is not None:
                    mode_ids[1] = previous_best_mode
            safety = None
            if self.safety_evaluator is not None and safety_context is not None:
                safety = self.safety_evaluator.check(actions, safety_context)
                valid = safety.valid.clone()
                if not valid.any():
                    expanded = True
                    expanded_actions = (baseline + torch.randn(actions.shape, device=self.device, generator=generator) * self.expanded_delta).clamp(-1, 1)
                    braking = baseline.expand(7, -1, -1).clone()
                    braking[:, :, 1] = -1
                    braking[:5, :, 0] = torch.linspace(-0.3, 0.3, 5, device=self.device)[:, None]
                    braking[6, :, 0] = float(safety_context.previous_action[0])
                    extra = [actions, expanded_actions, braking]
                    alternative_actions = alternatives() if callable(alternatives) else alternatives
                    if alternative_actions is not None and len(alternative_actions):
                        extra.append(alternative_actions)
                    actions = torch.cat(extra)
                    if mode_ids is not None:
                        mode_ids = torch.cat((mode_ids, mode_ids.new_full((len(actions) - len(mode_ids),), -1)))
                    safety = self.safety_evaluator.check(actions, safety_context)
                    valid = safety.valid.clone()
            elif corridor_context is None:
                valid = torch.ones(len(actions), dtype=torch.bool, device=self.device)
            else:
                corridor = self.corridor.check(actions, baseline, corridor_context)
                valid = corridor.valid.clone()
            total_rejected += int((~valid).sum().item())
            total_checked += len(actions)
            # Hard-invalid trajectories never reach the learned value/cost ranking.
            wm_values = torch.full((len(actions),), float("nan"), device=self.device)
            uncertainty_penalty = torch.zeros_like(wm_values)
            predicted_cost = torch.zeros_like(wm_values)
            costs_validated = False
            if valid.any():
                consequence = self.evaluator.evaluate_mppi_candidates(z_t, actions[valid], task=task)
                wm_values[valid] = consequence.total_value
                uncertainty_penalty[valid] = self.uncertainty_coef * consequence.terminal_q_std
                if getattr(consequence, "cost_validated", False):
                    predicted_cost[valid] = consequence.cumulative_cost
                    costs_validated = True
            deviation, smoothness = self._penalties(actions, baseline)
            scores = (
                wm_values
                - uncertainty_penalty
                - self.deviation_coef * deviation
                - self.smoothness_coef * smoothness
                - self.cost_coef * predicted_cost
            )
            if safety is not None:
                scores -= self.geometry_coef * safety.soft_risk
            valid &= torch.isfinite(scores)
            scores = scores.masked_fill(~valid, -torch.inf)
            baseline_wm_value = wm_values[0]
            baseline_score = scores[0]
            if not valid.any():
                if self._mode_bank is not None:
                    self._mode_bank.update(actions, scores, valid, mode_ids)
                # This is a minimum-risk emergency action, NOT a certified safe fallback.
                emergency = safety.emergency_cost if safety is not None else (actions[:, :, 1].mean(-1) + 1)
                elite_indices = emergency.argmin().reshape(1)
                elite_values = torch.zeros(1, device=self.device)
                elite_wm_values = wm_values[elite_indices]
                elite_actions = actions[elite_indices]
                weights = torch.ones(1, device=self.device)
                break
            elite_indices = torch.topk(scores, min(self.num_elites, int(valid.sum())), dim=0).indices
            elite_values = scores[elite_indices]
            elite_wm_values = wm_values[elite_indices]
            elite_actions = actions[elite_indices]
            previous_best = elite_actions[0].clone()
            if mode_ids is not None:
                previous_best_mode = int(mode_ids[elite_indices[0]])
                self._mode_bank.update(actions, scores, valid, mode_ids)
            finite = torch.isfinite(elite_values)
            logits = self.temperature * (elite_values - elite_values[finite].max())
            weights = torch.where(finite, torch.exp(logits), torch.zeros_like(logits))
            weights = weights / weights.sum().clamp_min(1e-9)
            mean = (weights.view(-1, 1, 1) * elite_actions).sum(dim=0)
            mean = self._project(mean, baseline)
            variance = (
                weights.view(-1, 1, 1) * (elite_actions - mean.unsqueeze(0)).square()
            ).sum(dim=0)
            std = torch.maximum(
                torch.minimum(variance.sqrt(), self.max_std_vector.view(1, -1)),
                self.min_std_vector.view(1, -1),
            )
            if self._mode_bank is not None:
                mean, std = self._mode_bank.means[0].clone(), self._mode_bank.stds[0].clone()
        if eval_mode:
            selected_index = int(torch.argmax(elite_values).item())
        else:
            selected_index = int(torch.multinomial(weights, 1).item())
        sequence = elite_actions[selected_index]
        selected_wm_value = elite_wm_values[selected_index]
        selected_score = elite_values[selected_index]
        baseline_valid = bool(valid[0])
        wm_value_gain = selected_wm_value - baseline_wm_value
        planner_score_gain = selected_score - baseline_score if baseline_valid else torch.tensor(float("nan"), device=self.device)
        threshold = max(
            self.min_improvement_abs,
            self.min_improvement_ratio * abs(float(baseline_score.cpu())),
        )
        safety_improved = safety is not None and bool(safety.soft_risk[0] - safety.soft_risk[elite_indices[selected_index]] > 0.05)
        baseline_selected = baseline_valid and not safety_improved and bool(float(planner_score_gain.cpu()) <= threshold)
        if baseline_selected:
            sequence = baseline
            selected_wm_value = baseline_wm_value
            selected_score = baseline_score
            wm_value_gain = selected_wm_value - baseline_wm_value
            planner_score_gain = selected_score - baseline_score

        selected_batch_index = 0 if baseline_selected else int(elite_indices[selected_index])
        if safety is not None:
            safety_metrics = {
                "no_safe_candidate": float(not valid.any()),
                "selected_safe": float(valid[selected_batch_index]),
                "baseline_safe": float(valid[0]),
                "expanded_search": float(expanded),
                "collision_reject_count": float(safety.collision.sum()),
                "offroad_reject_count": float(safety.offroad.sum()),
                "collision_reject_rate": float(safety.collision.float().mean()),
                "offroad_reject_rate": float(safety.offroad.float().mean()),
                "kinematic_reject_count": float(safety.kinematic_violation.sum()),
                "baseline_soft_risk": float(safety.soft_risk[0]),
                "selected_soft_risk": float(safety.soft_risk[selected_batch_index]),
                "risk_reduction": float(safety.soft_risk[0] - safety.soft_risk[selected_batch_index]),
                "selected_min_clearance": float(safety.min_clearance[selected_batch_index]),
                "selected_min_ttc": float(safety.min_ttc[selected_batch_index]),
                "selected_boundary_margin": float(safety.boundary_margin[selected_batch_index]),
                "cost_validated": float(costs_validated),
            }
        if self._mode_bank is not None:
            safety_metrics.update(self._mode_bank.metrics())

        self._prev_mean = mean.detach().clone()
        action_delta = (sequence - baseline).abs().amax(dim=0)
        selected_lateral = 0.0
        if corridor_context is not None:
            selected_lateral = float(
                self.corridor.check(sequence.unsqueeze(0), baseline, corridor_context)
                .max_lateral_deviation[0]
                .cpu()
            )
        return LocalMPPIResult(
            action=sequence[0].clamp(-1.0, 1.0),
            selected_sequence=sequence,
            initial_mean=initial_mean,
            final_mean=mean,
            initial_std=initial_std,
            final_std=std,
            baseline_wm_value=baseline_wm_value,
            selected_wm_value=selected_wm_value,
            wm_value_gain=wm_value_gain,
            baseline_score=baseline_score,
            selected_score=selected_score,
            planner_score_gain=planner_score_gain,
            baseline_value=baseline_score,
            final_value=selected_score,
            planner_gain=planner_score_gain,
            elite_values=elite_values,
            baseline_selected=baseline_selected,
            corridor_reject_count=total_rejected,
            corridor_reject_rate=float(total_rejected / max(1, total_checked)),
            max_lateral_deviation=selected_lateral,
            max_action_delta=action_delta,
            safety_metrics=safety_metrics,
        )


__all__ = ["LocalMPPI", "LocalMPPIResult", "_action_vector"]
