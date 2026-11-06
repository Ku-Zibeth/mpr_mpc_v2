"""Historical single-Gaussian H+1 MPPI, extended only with WM Cost scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import warnings

import torch

from .kinematic_filter import BatchedKinematicCorridor, CorridorContext
from .timing import PhaseTimings


MPPI_DIAGNOSTIC_KEYS = (
    "predicted_reward", "terminal_q", "predicted_cost", "cost_penalty", "cost_coef",
    "cost_enabled", "cost_validated", "deviation_penalty", "smoothness_penalty",
    "final_score", "selected_candidate_cost", "baseline_candidate_cost",
    "predicted_step_cost",
    "wm_evaluated_candidates", "wm_cache_hits", "baseline_corridor_valid",
    "mppi_sampling_ms", "world_model_rollout_ms", "cost_prediction_ms", "elite_update_ms",
)


def _action_vector(value, action_dim, name, device):
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.ndim == 0:
        tensor = tensor.repeat(action_dim)
    if tensor.shape != (action_dim,):
        raise ValueError(f"{name} must be scalar or [{action_dim}].")
    if not torch.isfinite(tensor).all() or (tensor < 0).any():
        raise ValueError(f"{name} must be finite and nonnegative.")
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
    baseline_value: torch.Tensor
    final_value: torch.Tensor
    planner_gain: torch.Tensor
    elite_values: torch.Tensor
    baseline_selected: bool
    corridor_reject_count: int
    corridor_reject_rate: float
    max_lateral_deviation: float
    max_action_delta: torch.Tensor
    metrics: dict = field(default_factory=dict)
    baseline_consequence: dict = field(default_factory=dict)
    selected_consequence: dict = field(default_factory=dict)


class LocalMPPI:
    def __init__(self, evaluator, cfg, *, horizon, action_dim, device, control_dt=.1):
        self.evaluator = evaluator
        self.horizon, self.action_dim = int(horizon), int(action_dim)
        self.device = torch.device(device)
        # Original residual-SAC MPPI config: mpr_mpc commit f09b64f.
        self.num_samples = int(cfg.get("num_samples", 64))
        self.num_elites = int(cfg.get("num_elites", 8))
        self.iterations = int(cfg.get("iterations", 4))
        self.min_std_vector = _action_vector(cfg.get("min_std", [.005, .02]), action_dim, "min_std", device)
        self.max_std_vector = _action_vector(cfg.get("max_std", [.05, .2]), action_dim, "max_std", device)
        self.initial_std_vector = _action_vector(cfg.get("initial_std", [.025, .1]), action_dim, "initial_std", device)
        self.max_delta = _action_vector(cfg.get("max_delta", [.05, .2]), action_dim, "max_delta", device)
        self.temperature = float(cfg.get("temperature", .5))
        self.proposal_init = str(cfg.get("proposal_init", "structured_baseline"))
        self.blend_alpha = float(cfg.get("blend_alpha", .8))
        self.deviation_coef = float(cfg.get("deviation_coef", 5.))
        self.smoothness_coef = float(cfg.get("smoothness_coef", 1.))
        self.throttle_deviation_weight = float(cfg.get("throttle_deviation_weight", .25))
        self.min_improvement_abs = float(cfg.get("min_improvement_abs", 1.))
        self.min_improvement_ratio = float(cfg.get("min_improvement_ratio", .01))
        self.cost_enabled = bool(cfg.get("cost_enabled", True))
        self.cost_coef = float(cfg.get("cost_coef", 1.))
        self.eval_seed = int(cfg.get("eval_seed", 0))
        if self.horizon <= 0 or self.num_samples < 2 or not 0 < self.num_elites <= self.num_samples or self.iterations <= 0:
            raise ValueError("Require positive horizon/iterations, >=2 samples and 0<elites<=samples.")
        if ((self.min_std_vector > self.max_std_vector).any()
                or (self.initial_std_vector < self.min_std_vector).any()
                or (self.initial_std_vector > self.max_std_vector).any() or (self.max_delta <= 0).any()):
            raise ValueError("Invalid MPPI standard deviations or trust region.")
        for name in ("temperature", "deviation_coef", "smoothness_coef", "throttle_deviation_weight",
                     "min_improvement_abs", "min_improvement_ratio", "cost_coef"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if not math.isfinite(self.blend_alpha) or not 0 <= self.blend_alpha <= 1:
            raise ValueError("blend_alpha must be in [0,1].")
        if self.proposal_init not in {"structured_baseline", "previous_mean", "blend"}:
            raise ValueError("Unknown proposal_init mode.")
        self.corridor = BatchedKinematicCorridor(cfg, control_dt=control_dt, device=device)
        self._prev_mean = None
        self._warned_unvalidated = False

    def reset(self):
        self._prev_mean = None

    def _project(self, actions, baseline):
        return (baseline + (actions-baseline).clamp(-self.max_delta, self.max_delta)).clamp(-1, 1)

    def _shift_previous(self, baseline):
        if self._prev_mean is None:
            return baseline.clone()
        shifted = baseline.clone()
        shifted[:-1] = self._prev_mean[1:]
        return self._project(shifted, baseline)

    def initial_mean(self, baseline):
        if baseline.shape != (self.horizon+1, self.action_dim):
            raise ValueError("Baseline must have shape [H+1, action_dim].")
        if self.proposal_init == "structured_baseline":
            return baseline.clone()
        shifted = self._shift_previous(baseline)
        if self.proposal_init == "previous_mean":
            return shifted
        return self._project(self.blend_alpha*baseline+(1-self.blend_alpha)*shifted, baseline)

    def _penalties(self, actions, baseline):
        delta = actions-baseline
        normalized = delta/self.max_delta
        deviation = normalized[..., 0].square()
        if self.action_dim > 1:
            deviation = deviation+self.throttle_deviation_weight*normalized[..., 1].square()
        smoothness = delta.diff(dim=1).square().sum(-1).mean(1)
        return deviation.mean(1), smoothness

    def _candidate_batch(self, mean, std, baseline, generator=None):
        noise = torch.randn(self.num_samples, self.horizon+1, self.action_dim,
                            device=self.device, generator=generator)
        actions = self._project(mean+std*noise, baseline)
        actions[0], actions[1] = baseline, self._project(mean, baseline)
        return actions

    def _evaluate_cached(self, z, actions, cache, task, timing):
        # Exact action bytes, scoped to one observation/plan. Never reuse values
        # across steps, and evaluate duplicate baseline/mean trajectories only once.
        keys = [row.tobytes() for row in actions.detach().cpu().numpy()]
        missing = {}
        for index, key in enumerate(keys):
            if key not in cache and key not in missing:
                missing[key] = index
        enabled = self.cost_enabled
        if missing:
            result = self.evaluator.evaluate_mppi_candidates(
                z, actions[list(missing.values())], task=task, compute_cost=enabled)
            for name, value in getattr(result, "timings", {}).items():
                timing[name] = timing.get(name, 0.)+value
            cost = getattr(result, "cumulative_cost", None)
            if enabled and cost is None:
                raise RuntimeError("MPPI Cost is enabled but the World Model returned no predicted cost.")
            if cost is not None and (not torch.isfinite(cost).all() or (cost < 0).any()):
                raise RuntimeError("MPPI received non-finite or negative cumulative WM Cost.")
            for i, key in enumerate(missing):
                cache[key] = {
                    "value": result.total_value[i],
                    "cost": cost[i] if cost is not None else actions.new_tensor(float("nan")),
                    "step_cost": (result.predicted_cost[i, 0] if getattr(result, "predicted_cost", None) is not None else actions.new_tensor(float("nan"))),
                    "reward": (result.discounted_reward[i] if getattr(result, "discounted_reward", None) is not None else actions.new_tensor(float("nan"))),
                    "terminal_q": (result.terminal_q[i] if getattr(result, "terminal_q", None) is not None else actions.new_tensor(float("nan"))),
                    "validated": bool(getattr(result, "cost_validated", False)),
                }
        rows = [cache[key] for key in keys]
        packed = {name: torch.stack([r[name] for r in rows]) for name in ("value", "cost", "reward", "terminal_q")}
        return rows, packed, len(missing)

    @torch.no_grad()
    def plan(self, z_t, baseline, *, eval_mode, corridor_context: CorridorContext | None = None, task=None):
        if not torch.isfinite(baseline).all():
            raise ValueError("MPPI baseline must be finite.")
        baseline = baseline.to(self.device).clamp(-1, 1)
        mean = self.initial_mean(baseline)
        initial_mean = mean.clone()
        std = self.initial_std_vector.expand_as(mean).clone()
        initial_std = std.clone()
        generator = None
        if eval_mode:
            generator = torch.Generator(device=self.device).manual_seed(self.eval_seed)
        phase = PhaseTimings(self.device)
        timing, cache = {}, {}
        rejected = checked = evaluated = 0
        enabled = self.cost_enabled and self.cost_coef > 0
        for _ in range(self.iterations):
            with phase.measure("mppi_sampling"):
                actions = self._candidate_batch(mean, std, baseline, generator)
            corridor = self.corridor.check(actions, baseline, corridor_context) if corridor_context is not None else None
            valid = corridor.valid.clone() if corridor is not None else torch.ones(len(actions), device=self.device, dtype=torch.bool)
            baseline_corridor_valid = bool(valid[0])
            # Historical Lattice fallback, not an independent safety certificate.
            valid[0] = True
            rejected += int((~valid).sum())
            checked += len(actions)
            rows, values, count = self._evaluate_cached(z_t, actions, cache, task, timing)
            evaluated += count
            deviation, smoothness = self._penalties(actions, baseline)
            cost_penalty = self.cost_coef*values["cost"] if enabled else torch.zeros_like(values["value"])
            scores = values["value"]-cost_penalty-self.deviation_coef*deviation-self.smoothness_coef*smoothness
            valid &= torch.isfinite(scores)
            if not valid[0]:
                raise RuntimeError("World Model returned a non-finite baseline score; no emergency controller is substituted.")
            scores = scores.masked_fill(~valid, -torch.inf)
            with phase.measure("elite_update"):
                indices = scores.topk(min(self.num_elites, int(valid.sum()))).indices
                elite_values, elites = scores[indices], actions[indices]
                weights = (self.temperature*(elite_values-elite_values.max())).exp()
                weights /= weights.sum().clamp_min(1e-9)
                mean = self._project((weights[:, None, None]*elites).sum(0), baseline)
                variance = (weights[:, None, None]*(elites-mean).square()).sum(0)
                std = variance.sqrt().clamp(self.min_std_vector, self.max_std_vector)
        index = int(indices[0]) if eval_mode else int(indices[torch.multinomial(weights, 1)])
        gain = scores[index]-scores[0]
        threshold = max(self.min_improvement_abs, self.min_improvement_ratio*abs(float(scores[0])))
        baseline_selected = bool(float(gain) <= threshold)
        if baseline_selected:
            index = 0
        sequence = actions[index]
        selected, base = rows[index], rows[0]
        if enabled and not selected["validated"] and not self._warned_unvalidated:
            warnings.warn("WM Cost is used in MPPI ranking but has NOT passed basic held-out validation; this is not a safety guarantee.",
                          RuntimeWarning, stacklevel=2)
            self._warned_unvalidated = True
        timing.update(phase.finish())
        for name in ("mppi_sampling_ms", "world_model_rollout_ms", "cost_prediction_ms", "elite_update_ms"):
            timing.setdefault(name, 0.)
        metrics = {
            **timing,
            "predicted_reward": float(selected["reward"]), "terminal_q": float(selected["terminal_q"]),
            "predicted_cost": float(selected["cost"]), "cost_penalty": float(cost_penalty[index]),
            "predicted_step_cost": float(selected["step_cost"]),
            "cost_coef": self.cost_coef, "cost_enabled": float(enabled),
            "cost_validated": float(selected["validated"]),
            "deviation_penalty": float(self.deviation_coef*deviation[index]),
            "smoothness_penalty": float(self.smoothness_coef*smoothness[index]),
            "final_score": float(scores[index]), "selected_candidate_cost": float(selected["cost"]),
            "baseline_candidate_cost": float(base["cost"]), "wm_evaluated_candidates": float(evaluated),
            "wm_cache_hits": float(checked-evaluated), "baseline_corridor_valid": float(baseline_corridor_valid),
        }
        self._prev_mean = mean.detach().clone()
        lateral = float(corridor.max_lateral_deviation[index]) if corridor is not None else 0.
        value_gain, score_gain = selected["value"]-base["value"], scores[index]-scores[0]
        return LocalMPPIResult(
            action=sequence[0], selected_sequence=sequence, initial_mean=initial_mean, final_mean=mean,
            initial_std=initial_std, final_std=std, baseline_wm_value=base["value"],
            selected_wm_value=selected["value"], wm_value_gain=value_gain, baseline_score=scores[0],
            selected_score=scores[index], planner_score_gain=score_gain, baseline_value=scores[0],
            final_value=scores[index], planner_gain=score_gain, elite_values=elite_values,
            baseline_selected=baseline_selected, corridor_reject_count=rejected,
            corridor_reject_rate=rejected/max(checked, 1), max_lateral_deviation=lateral,
            max_action_delta=(sequence-baseline).abs().amax(0), metrics=metrics,
            baseline_consequence=base, selected_consequence=selected,
        )


__all__ = ["LocalMPPI", "LocalMPPIResult", "_action_vector", "MPPI_DIAGNOSTIC_KEYS"]
