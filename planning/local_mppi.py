"""SAC-anchored TD-MPC2 Gaussian trajectory optimisation (H+1 convention)."""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import torch
from .timing import PhaseTimings


MPPI_DIAGNOSTIC_KEYS = (
    "predicted_reward", "terminal_q", "baseline_score", "selected_score", "final_score",
    "score_gain", "baseline_selected", "invalid_prediction", "warm_start_used", "iterations_completed",
    "wm_evaluated_candidates", "mppi_sampling_ms", "world_model_rollout_ms", "elite_update_ms",
    "final_value_evaluation_ms", "first_steer_delta", "first_throttle_delta", "first_action_delta_norm",
    "sac_to_selected_delta", "initial_std_steer", "initial_std_throttle",
    "final_std_steer", "final_std_throttle", "max_delta_steer", "max_delta_throttle", "planning_mode",
)
MPPI_EPISODE_METRICS = {f"mppi_{k}": f"mppi/{k}" for k in (*MPPI_DIAGNOSTIC_KEYS, "total_planning_ms")}


def _action_vector(value, action_dim, name, device):
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.ndim == 0:
        tensor = tensor.repeat(action_dim)
    if tensor.shape != (action_dim,) or not torch.isfinite(tensor).all() or (tensor < 0).any():
        raise ValueError(f"{name} must be finite, nonnegative and scalar or [{action_dim}].")
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
    elite_values: torch.Tensor
    baseline_selected: bool
    max_action_delta: torch.Tensor
    metrics: dict = field(default_factory=dict)
    baseline_consequence: dict = field(default_factory=dict)
    selected_consequence: dict = field(default_factory=dict)
    planning_mode: str = "baseline"

    # Public Coordinator aliases all denote the same unpenalised WM objective.
    baseline_score = property(lambda self: self.baseline_wm_value)
    baseline_value = property(lambda self: self.baseline_wm_value)
    selected_score = property(lambda self: self.selected_wm_value)
    final_value = property(lambda self: self.selected_wm_value)
    planner_score_gain = property(lambda self: self.wm_value_gain)
    planner_gain = property(lambda self: self.wm_value_gain)


class LocalMPPI:
    def __init__(self, evaluator, cfg, *, horizon, action_dim, device):
        self.evaluator, self.device = evaluator, torch.device(device)
        self.horizon, self.action_dim = int(horizon), int(action_dim)
        self.num_samples = int(cfg.get("num_samples", 64))
        self.num_elites = int(cfg.get("num_elites", 8))
        self.iterations = int(cfg.get("iterations", 4))
        self.temperature = float(cfg.get("temperature", .5))
        self.eval_seed = int(cfg.get("eval_seed", 0))
        self.initial_std_vector = _action_vector(cfg.get("initial_std", [.04, .12]), action_dim, "initial_std", device)
        self.min_std_vector = _action_vector(cfg.get("min_std", [.01, .03]), action_dim, "min_std", device)
        self.max_std_vector = _action_vector(cfg.get("max_std", [.08, .25]), action_dim, "max_std", device)
        self.max_delta = _action_vector(cfg.get("max_delta", [.10, .25]), action_dim, "max_delta", device)
        if (self.horizon <= 0 or self.action_dim <= 0 or self.num_samples < 2
                or not 0 < self.num_elites <= self.num_samples or self.iterations <= 0):
            raise ValueError("Require positive horizon/iterations and 0 < elites <= samples, samples >= 2.")
        if (not math.isfinite(self.temperature) or self.temperature < 0
                or (self.max_delta <= 0).any() or (self.min_std_vector > self.initial_std_vector).any()
                or (self.initial_std_vector > self.max_std_vector).any()):
            raise ValueError("Invalid temperature, Gaussian bounds or SAC trust region.")
        self.generator = torch.Generator(device=self.device).manual_seed(self.eval_seed)
        self._prev_mean = None

    def reset(self, seed=None):
        self._prev_mean = None
        self.generator.manual_seed(self.eval_seed if seed is None else int(seed) % 2**32)

    def _project(self, actions, baseline):
        return (baseline+(actions-baseline).clamp(-self.max_delta, self.max_delta)).clamp(-1., 1.)

    def initial_mean(self, baseline):
        mean = baseline.clone()
        if self._prev_mean is not None:
            mean[:-1] = self._prev_mean[1:]
        return self._project(mean, baseline)

    def _candidate_batch(self, mean, std, baseline):
        noise = torch.randn(self.num_samples, self.horizon+1, self.action_dim,
                            device=self.device, generator=self.generator)
        actions = self._project(mean.unsqueeze(0)+std.unsqueeze(0)*noise, baseline)
        actions[0] = baseline
        return actions

    def _update_distribution(self, actions, values):
        elite_indices = torch.topk(values, self.num_elites, dim=0).indices
        elite_values, elite_actions = values[elite_indices], actions[elite_indices]
        weights = torch.exp(self.temperature*(elite_values-elite_values.max()))
        weights = weights/(weights.sum()+1e-9)
        mean = (weights[:, None, None]*elite_actions).sum(0)
        variance = (weights[:, None, None]*(elite_actions-mean.unsqueeze(0)).square()).sum(0)
        return mean, variance.sqrt().clamp(self.min_std_vector, self.max_std_vector), elite_values

    @torch.no_grad()
    def plan(self, z_t, baseline, *, eval_mode, task=None):
        # Training already excludes MPPI at the Coordinator; both call modes use
        # the same mean output, with no post-selection noise or auxiliary driver.
        baseline = baseline.to(self.device)
        if (baseline.shape != (self.horizon+1, self.action_dim) or not torch.isfinite(baseline).all()
                or (baseline.abs() > 1).any()):
            raise ValueError("SAC baseline must be finite, in [-1,1], and [H+1, action_dim].")
        warm = self._prev_mean is not None
        mean = self.initial_mean(baseline)
        std = self.initial_std_vector.expand_as(mean).clone()
        initial_mean, initial_std = mean.clone(), std.clone()
        phase, timing = PhaseTimings(self.device), {}
        evaluated = completed = 0
        invalid = False
        nan = baseline.new_tensor(float("nan"))
        baseline_value, selected_value = nan, nan
        base = selected = {"value": nan, "cost": nan, "validated": False,
                           "reward": nan, "terminal_q": nan}
        elite_values = baseline.new_empty(0)

        def evaluate(actions):
            nonlocal evaluated
            result = self.evaluator.evaluate_mppi_candidates(z_t, actions, task=task, compute_cost=False)
            evaluated += len(actions)
            for key, value in getattr(result, "timings", {}).items():
                timing[key] = timing.get(key, 0.)+value
            values = result.total_value.reshape(-1)
            if values.shape != (len(actions),):
                raise ValueError("WM must return one value per action sequence.")
            return result, values

        def row(result, values, index):
            def item(name):
                value = getattr(result, name, None)
                return nan if value is None else value[index]
            return {"value": values[index], "cost": nan, "validated": False,
                    "reward": item("discounted_reward"), "terminal_q": item("terminal_q")}

        for _ in range(self.iterations):
            with phase.measure("mppi_sampling"):
                candidates = self._candidate_batch(mean, std, baseline)
            result, values = evaluate(candidates)
            baseline_value = values[0]
            base = row(result, values, 0)
            if not bool(torch.isfinite(values).all()):
                invalid = True
                break
            with phase.measure("elite_update"):
                mean, std, elite_values = self._update_distribution(candidates, values)
            completed += 1
            if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all()):
                invalid = True
                break

        mean = self._project(mean, baseline) if not invalid else baseline.clone()
        if not invalid:
            with phase.measure("final_value_evaluation"):
                # Same batch/Q reducer/termination mask for BOTH alternatives.
                result, values = evaluate(torch.stack((baseline, mean)))
            baseline_value = values[0]
            base = row(result, values, 0)
            invalid = not bool(torch.isfinite(values).all())
            if not invalid and bool(values[1] > values[0]):
                sequence, selected_value, selected = mean, values[1], row(result, values, 1)
            else:
                sequence, selected_value, selected = baseline, baseline_value, base
        else:
            sequence, selected_value, selected = baseline, baseline_value, base
        baseline_selected = invalid or torch.equal(sequence, baseline)
        if baseline_selected:
            sequence, selected_value, selected = baseline, baseline_value, base
        self._prev_mean = (baseline if invalid else mean).detach().clone()
        delta, maximum = sequence[0]-baseline[0], (sequence-baseline).abs().amax(0)
        gain = selected_value-baseline_value
        timing.update(phase.finish())
        metrics = {**timing, "predicted_reward": float(selected["reward"]), "terminal_q": float(selected["terminal_q"]),
            "baseline_score": float(baseline_value), "selected_score": float(selected_value), "final_score": float(selected_value),
            "score_gain": float(gain), "baseline_selected": float(baseline_selected), "invalid_prediction": float(invalid),
            "warm_start_used": float(warm), "iterations_completed": float(completed), "wm_evaluated_candidates": float(evaluated),
            "first_steer_delta": float(delta[0]), "first_throttle_delta": float(delta[1]) if self.action_dim>1 else 0.,
            "first_action_delta_norm": float(delta.norm()), "sac_to_selected_delta": float(delta.norm()),
            "initial_std_steer": float(initial_std[:, 0].mean()), "initial_std_throttle": float(initial_std[:, -1].mean()),
            "final_std_steer": float(std[:, 0].mean()), "final_std_throttle": float(std[:, -1].mean()),
            "max_delta_steer": float(maximum[0]), "max_delta_throttle": float(maximum[-1]),
            "planning_mode": float(not baseline_selected)}
        return LocalMPPIResult(action=sequence[0], selected_sequence=sequence,
            initial_mean=initial_mean, final_mean=mean, initial_std=initial_std, final_std=std,
            baseline_wm_value=baseline_value, selected_wm_value=selected_value, wm_value_gain=gain,
            elite_values=elite_values, baseline_selected=bool(baseline_selected), max_action_delta=maximum,
            metrics=metrics, baseline_consequence=base, selected_consequence=selected,
            planning_mode="baseline" if baseline_selected else "normal")


__all__ = ["LocalMPPI", "LocalMPPIResult", "_action_vector", "MPPI_DIAGNOSTIC_KEYS", "MPPI_EPISODE_METRICS"]
