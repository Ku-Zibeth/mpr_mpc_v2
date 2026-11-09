"""Single-Gaussian H+1 MPPI with independent pre-WM candidate safety."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import warnings

import torch

from .kinematic_filter import BatchedKinematicCorridor, CorridorContext
from .timing import PhaseTimings
from .mppi_safety import MPPISafetyFilter


MPPI_DIAGNOSTIC_KEYS = (
    "predicted_reward", "terminal_q", "predicted_cost", "cost_penalty", "cost_coef",
    "cost_enabled", "cost_validated", "deviation_penalty", "smoothness_penalty",
    "final_score", "selected_candidate_cost", "baseline_candidate_cost",
    "predicted_step_cost",
    "wm_evaluated_candidates", "wm_cache_hits", "baseline_corridor_valid",
    "mppi_sampling_ms", "world_model_rollout_ms", "cost_prediction_ms", "elite_update_ms",
    "safety_check_ms", "first_steer_delta", "first_throttle_delta", "first_action_delta_norm",
    "baseline_selected", "score_gain", "gate_threshold", "initial_std_steer", "initial_std_throttle",
    "final_std_steer", "final_std_throttle", "max_delta_steer", "max_delta_throttle",
    "unique_candidate_count", "safety_valid_ratio", "safety_reject_count", "collision_reject_count",
    "road_boundary_reject_count", "kinematic_reject_count", "baseline_safety_valid",
    "emergency_fallback", "lattice_fallback", "safety_fallback_unverified", "safety_evaluated_candidates",
    "selected_safety_valid",
    "normal_search_delta", "normal_search_steer_delta", "normal_search_throttle_delta",
    "recovery_search_delta", "recovery_search_steer_delta", "recovery_search_throttle_delta",
    "emergency_fallback_delta", "emergency_fallback_steer_delta", "emergency_fallback_throttle_delta",
    "normal_safety_valid_ratio", "recovery_safety_valid_ratio", "lattice_baseline_safety_valid",
    "recovery_search", "recovery_wm_evaluated_candidates", "recovery_search_ms",
    "planning_mode", "recovery_reason", "cost_horizon_validated", "cost_scoring_requested",
    "allow_unvalidated_cost", "predicted_collision", "predicted_offroad",
    "predicted_first_collision", "predicted_first_offroad", "stopping_distance_m",
    "stopping_time_s", "stopping_horizon_insufficient", "obstacle_uncertainty_radius_m",
    "selected_full_braking",
    "lattice_baseline_checked",
    "lattice_fallback_delta", "lattice_fallback_steer_delta", "lattice_fallback_throttle_delta",
)
MPPI_EPISODE_METRICS = {
    (key if key.startswith("mppi_") else f"mppi_{key}"): f"mppi/{key}"
    for key in (*MPPI_DIAGNOSTIC_KEYS, "total_planning_ms")
}


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
    selected_positions: torch.Tensor | None = None
    planning_mode: str = "baseline"
    recovery_reason: str = "none"


class LocalMPPI:
    def __init__(self, evaluator, cfg, *, horizon, action_dim, device, control_dt=.1):
        self.evaluator = evaluator
        self.horizon, self.action_dim = int(horizon), int(action_dim)
        self.device = torch.device(device)
        # Original residual-SAC MPPI config: mpr_mpc commit f09b64f.
        self.num_samples = int(cfg.get("num_samples", 64))
        self.num_elites = int(cfg.get("num_elites", 8))
        self.iterations = int(cfg.get("iterations", 4))
        self.min_std_vector = _action_vector(cfg.get("min_std", [.01, .03]), action_dim, "min_std", device)
        self.max_std_vector = _action_vector(cfg.get("max_std", [.08, .25]), action_dim, "max_std", device)
        self.initial_std_vector = _action_vector(cfg.get("initial_std", [.04, .12]), action_dim, "initial_std", device)
        self.max_delta = _action_vector(cfg.get("max_delta", [.10, .25]), action_dim, "max_delta", device)
        self.temperature = float(cfg.get("temperature", .5))
        self.proposal_init = str(cfg.get("proposal_init", "structured_baseline"))
        self.blend_alpha = float(cfg.get("blend_alpha", .8))
        self.deviation_coef = float(cfg.get("deviation_coef", 1.))
        self.smoothness_coef = float(cfg.get("smoothness_coef", .3))
        self.throttle_deviation_weight = float(cfg.get("throttle_deviation_weight", .25))
        self.min_improvement_abs = float(cfg.get("min_improvement_abs", .05))
        self.min_improvement_ratio = float(cfg.get("min_improvement_ratio", .001))
        self.cost_enabled = bool(cfg.get("cost_enabled", True))
        self.cost_coef = float(cfg.get("cost_coef", 1.))
        self.allow_unvalidated_cost = bool(cfg.get("allow_unvalidated_cost", False))
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
        safety_cfg = dict(cfg.get("safety", {}))
        self.safety_enabled = bool(safety_cfg.get("enabled", True))
        self.safety_corridor = BatchedKinematicCorridor(cfg, control_dt=control_dt, device="cpu")
        self.candidate_safety = MPPISafetyFilter(safety_cfg, control_dt=control_dt)
        recovery = dict(cfg.get("recovery", {}))
        self.recovery_enabled = bool(recovery.get("enabled", self.safety_enabled))
        if self.recovery_enabled and not self.safety_enabled:
            raise ValueError("Recovery Search requires independent safety to be enabled.")
        self.recovery_samples = int(recovery.get("num_samples", min(32, self.num_samples)))
        self.recovery_iterations = int(recovery.get("iterations", min(2, self.iterations)))
        if not 2 <= self.recovery_samples <= self.num_samples or not 1 <= self.recovery_iterations <= self.iterations:
            raise ValueError("Recovery budget must be positive and no larger than the normal search budget.")
        self.recovery_delta = _action_vector(recovery.get("max_delta", [.20, .50]), action_dim, "recovery.max_delta", device)
        self.recovery_std = _action_vector(recovery.get("initial_std", [.06, .20]), action_dim, "recovery.initial_std", device)
        if (self.recovery_delta <= 0).any() or (self.recovery_std <= 0).any():
            raise ValueError("Recovery range/std must be positive.")
        if (self.recovery_std < self.min_std_vector).any():
            raise ValueError("Recovery initial_std must be >= min_std.")
        self.straighten_fractions = tuple(float(v) for v in recovery.get("straighten_fractions", [0., .5, 1.]))
        self.throttle_levels = tuple(float(v) for v in recovery.get("throttle_levels", [0., -.5, -1.]))
        if (not self.straighten_fractions or not self.throttle_levels
                or len(self.straighten_fractions)*len(self.throttle_levels) > 16
                or any(not math.isfinite(v) or not 0 <= v <= 1 for v in self.straighten_fractions)
                or any(not math.isfinite(v) or not -1 <= v <= 0 for v in self.throttle_levels)):
            raise ValueError("Controlled recovery requires <=16 bounded hold/straighten/decelerate candidates.")
        self.generator = torch.Generator(device=self.device).manual_seed(self.eval_seed)
        self._prev_mean = None
        self._warned_unvalidated = False

    def reset(self, seed=None):
        self._prev_mean = None
        self.generator.manual_seed(self.eval_seed if seed is None else int(seed) % 2**32)
        self.candidate_safety.reset()

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

    def _penalties(self, actions, baseline, max_delta=None):
        delta = actions-baseline
        normalized = delta/(self.max_delta if max_delta is None else max_delta)
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

    def _evaluate_cached(self, z, actions, cache, task, timing, keys=None):
        # Exact action bytes, scoped to one observation/plan. Never reuse values
        # across steps, and evaluate duplicate baseline/mean trajectories only once.
        keys = keys if keys is not None else [row.tobytes() for row in actions.detach().cpu().numpy()]
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
            if cost is not None:
                bad_cost = ~torch.isfinite(cost) | (cost < 0)
                if bool(bad_cost.all()):
                    raise RuntimeError("MPPI received non-finite or negative cumulative WM Cost for every evaluated candidate.")
                cost = cost.masked_fill(bad_cost, float("nan"))
            for i, key in enumerate(missing):
                cache[key] = {
                    "value": result.total_value[i],
                    "cost": cost[i] if cost is not None else actions.new_tensor(float("nan")),
                    "step_cost": (result.predicted_cost[i, 0] if getattr(result, "predicted_cost", None) is not None else actions.new_tensor(float("nan"))),
                    "reward": (result.discounted_reward[i] if getattr(result, "discounted_reward", None) is not None else actions.new_tensor(float("nan"))),
                    "terminal_q": (result.terminal_q[i] if getattr(result, "terminal_q", None) is not None else actions.new_tensor(float("nan"))),
                    "validated": bool(getattr(result, "cost_validated", False)),
                    "ranking_validated": bool(getattr(result, "cost_ranking_validated", False)),
                }
        rows = [cache[key] for key in keys]
        packed = {name: torch.stack([r[name] for r in rows]) for name in ("value", "cost", "reward", "terminal_q")}
        packed["ranking_validated"] = torch.tensor([r["ranking_validated"] for r in rows], device=actions.device)
        return rows, packed, len(missing)

    @torch.no_grad()
    def plan(self, z_t, baseline, *, eval_mode, corridor_context: CorridorContext | None = None,
             safety_context=None, lattice_actions=None, lattice_corridor_context=None, task=None):
        if not torch.isfinite(baseline).all():
            raise ValueError("MPPI baseline must be finite.")
        if self.safety_enabled and safety_context is None:
            raise RuntimeError("MPPI independent safety requires safety_context; SAC state is not a substitute.")
        baseline = baseline.to(self.device).clamp(-1, 1)
        mean = self.initial_mean(baseline)
        initial_mean = mean.clone()
        std = self.initial_std_vector.expand_as(mean).clone()
        initial_std = std.clone()
        generator = self.generator if eval_mode else None
        phase = PhaseTimings(self.device)
        timing, cache, safety_cache, unique, corridor_snapshots = {}, {}, {}, set(), {}
        baseline_host = baseline.detach().cpu()
        rejected = checked = evaluated = wm_requested = 0
        enabled = self.cost_enabled and self.cost_coef > 0
        safety_rejected = collision_rejected = road_rejected = kinematic_rejected = 0
        baseline_safe, baseline_corridor_valid = False, False
        best = None
        elite_values = baseline.new_empty(0)

        def blank():
            return {**{key: baseline.new_tensor(float("nan"))
                       for key in ("value", "cost", "step_cost", "reward", "terminal_q")},
                    "validated": False, "ranking_validated": False}

        def check_actions(actions, reference, corridor_ctx):
            nonlocal checked, rejected, safety_rejected, collision_rejected, road_rejected, kinematic_rejected
            with phase.measure("safety_check"):
                host = actions.detach().cpu().contiguous()
                keys = [row.tobytes() for row in host.numpy()]
                unique.update(keys)
                if self.safety_enabled:
                    if corridor_ctx is not None and id(corridor_ctx) not in corridor_snapshots:
                        corridor_snapshots[id(corridor_ctx)] = replace(corridor_ctx, refined_lateral=corridor_ctx.refined_lateral.cpu())
                    cpu_context = corridor_snapshots.get(id(corridor_ctx))
                    ref = baseline_host if reference is baseline else reference.detach().cpu()
                    corridor = self.safety_corridor.check(host, ref, cpu_context) if cpu_context is not None else None
                    fast = torch.isfinite(host).all((1, 2)) & (host.abs() <= 1).all((1, 2))
                else:
                    corridor = self.corridor.check(actions, reference, corridor_ctx) if corridor_ctx is not None else None
                    fast = torch.isfinite(actions).all((1, 2)) & (actions.abs() <= 1).all((1, 2))
                if corridor is not None:
                    fast &= corridor.valid
                fast_list = fast.cpu().tolist()
                if self.safety_enabled:
                    missing = {}
                    for i, key in enumerate(keys):
                        if fast_list[i] and key not in safety_cache:
                            missing.setdefault(key, i)
                    if missing:
                        result = self.candidate_safety.check(host[list(missing.values())], safety_context)
                        columns = {name: getattr(result, name).tolist() for name in
                                   ("valid", "collision", "offroad", "kinematic_violation")}
                        for name in ("first_collision", "first_offroad"):
                            value = getattr(result, name, None)
                            if value is not None:
                                columns[name] = value.tolist()
                        for i, key in enumerate(missing):
                            safety_cache[key] = {name: values[i] for name, values in columns.items()}
                            if hasattr(result, "positions"):
                                safety_cache[key]["positions"] = result.positions[i]
                    flags = [safety_cache[key] if fast_list[i] else
                             dict(valid=False, collision=False, offroad=False, kinematic_violation=True)
                             for i, key in enumerate(keys)]
                    valid = torch.tensor([fast_list[i] and row["valid"] for i, row in enumerate(flags)],
                                         dtype=torch.bool, device=self.device)
                    collision_rejected += sum(bool(row["collision"]) for row in flags)
                    road_rejected += sum(bool(row["offroad"]) for row in flags)
                    kinematic_rejected += sum(bool(row["kinematic_violation"]) for row in flags)
                else:
                    valid = fast
                checked += len(actions)
                rejected += int((~fast).sum())
                safety_rejected += int((~valid).sum())
            return valid, keys, corridor

        def evaluate(actions, keys, reference=baseline, bounds=None):
            nonlocal evaluated, wm_requested
            wm_requested += len(actions)
            rows, values, count = self._evaluate_cached(z_t, actions, cache, task, timing, keys)
            evaluated += count
            deviation, smoothness = self._penalties(actions, reference, bounds)
            cost_penalty = (torch.where(values["ranking_validated"] | self.allow_unvalidated_cost,
                self.cost_coef*values["cost"], 0.) if enabled else torch.zeros_like(values["value"]))
            scores = values["value"]-cost_penalty-self.deviation_coef*deviation-self.smoothness_coef*smoothness
            return rows, scores, deviation, smoothness, cost_penalty

        baseline_key = baseline.detach().cpu().numpy().tobytes()
        normal_checked_start = checked
        for _ in range(self.iterations):
            with phase.measure("mppi_sampling"):
                actions = self._candidate_batch(mean, std, baseline, generator)
            valid, keys, corridor = check_actions(actions, baseline, corridor_context)
            baseline_safe = bool(valid[0])
            baseline_corridor_valid = bool(corridor.valid[0]) if corridor is not None else True
            original = valid.nonzero().flatten()
            if not len(original):
                if self.recovery_enabled and not baseline_safe:
                    break
                continue
            candidate_actions = actions[original]
            candidate_keys = [keys[i] for i in original.cpu().tolist()]
            rows, scores, deviation, smoothness, cost_penalty = evaluate(candidate_actions, candidate_keys)
            finite = torch.isfinite(scores).nonzero().flatten()
            if not len(finite):
                continue
            with phase.measure("elite_update"):
                indices = finite[scores[finite].topk(min(self.num_elites, len(finite))).indices]
                elite_values, elites = scores[indices], candidate_actions[indices]
                weights = (self.temperature*(elite_values-elite_values.max())).exp()
                weights /= weights.sum().clamp_min(1e-9)
                mean = self._project((weights[:, None, None]*elites).sum(0), baseline)
                variance = (weights[:, None, None]*(elites-mean).square()).sum(0)
                std = variance.sqrt().clamp(self.min_std_vector, self.max_std_vector)
            index = int(indices[0]) if eval_mode else int(indices[torch.multinomial(weights, 1)])
            if best is None or float(scores[index]) > best[0]:
                best = (float(scores[index]), candidate_actions[index].clone(), rows[index], candidate_keys[index])
            if self.recovery_enabled and not baseline_safe:
                break

        normal_checked, normal_rejects = checked-normal_checked_start, safety_rejected
        base = cache.get(baseline_key, blank())
        baseline_cost_scored = enabled and (base["ranking_validated"] or self.allow_unvalidated_cost)
        baseline_score = base["value"]-(self.cost_coef*base["cost"] if baseline_cost_scored else 0.)
        baseline_finite = baseline_safe and math.isfinite(float(baseline_score))
        threshold = (max(self.min_improvement_abs, self.min_improvement_ratio*abs(float(baseline_score)))
                     if baseline_finite else float("nan"))
        lattice_fallback = emergency_fallback = unverified = False
        recovery_used, lattice_safe = False, float("nan")
        if lattice_actions is not None and torch.equal(lattice_actions.to(self.device), baseline):
            lattice_safe = baseline_safe
        recovery_reason = "none"
        score_reference, score_bounds = baseline, self.max_delta
        mode = "baseline"
        recovery_checked = recovery_rejects = recovery_evaluated = 0
        recovery_best = None
        if self.recovery_enabled and (not baseline_safe or best is None):
            recovery_used = True
            recovery_reason = "unsafe_sac_baseline" if not baseline_safe else "no_usable_normal_candidate"
            start_checked, start_rejects, start_evaluated = checked, safety_rejected, evaluated
            with phase.measure("recovery_search"):
                lattice = lattice_actions.to(self.device).clamp(-1, 1) if lattice_actions is not None else None
                if lattice is not None:
                    valid, keys, _ = check_actions(lattice[None], lattice, lattice_corridor_context)
                    lattice_safe = bool(valid[0])
                if lattice is not None and lattice_safe:
                    # Same Gaussian/elite/WM scorer, recentered on checked
                    # Lattice. Neither SAC delta bounds nor SAC corridor apply.
                    rmean, rstd = lattice.clone(), self.recovery_std.expand_as(lattice).clone()
                    project = lambda a: (lattice+(a-lattice).clamp(-self.recovery_delta, self.recovery_delta)).clamp(-1, 1)
                    for _ in range(self.recovery_iterations):
                        with phase.measure("mppi_sampling"):
                            noise = torch.randn(self.recovery_samples, *lattice.shape,
                                device=self.device, generator=generator)
                            actions = project(rmean+rstd*noise)
                            actions[0], actions[1] = lattice, project(rmean)
                        valid, keys, _ = check_actions(actions, lattice, lattice_corridor_context)
                        indices = valid.nonzero().flatten()
                        if not len(indices):
                            continue
                        choices, candidate_keys = actions[indices], [keys[i] for i in indices.cpu().tolist()]
                        rows, scores, *_ = evaluate(choices, candidate_keys, lattice, self.recovery_delta)
                        finite = torch.isfinite(scores).nonzero().flatten()
                        if not len(finite):
                            continue
                        with phase.measure("elite_update"):
                            elite = finite[scores[finite].topk(min(self.num_elites, len(finite))).indices]
                            weights = (self.temperature*(scores[elite]-scores[elite].max())).exp()
                            weights /= weights.sum().clamp_min(1e-9)
                            rmean = project((weights[:, None, None]*choices[elite]).sum(0))
                            rstd = (weights[:, None, None]*(choices[elite]-rmean).square()).sum(0).sqrt()
                            rstd = rstd.clamp(self.min_std_vector, self.recovery_std)
                        i = int(elite[0])
                        if recovery_best is None or float(scores[i]) > recovery_best[0]:
                            recovery_best = (float(scores[i]), choices[i].clone(), rows[i], candidate_keys[i])
                    if recovery_best is not None:
                        score_reference, score_bounds, mode = lattice, self.recovery_delta, "recovery_lattice"
                else:
                    # Small deterministic controls, not another search mode
                    # bank/controller. Rate-limited first step uses measured
                    # previous steering; every trajectory is independently checked.
                    previous = baseline.new_tensor(safety_context.previous_action)
                    progress = torch.linspace(1/(self.horizon+1), 1., self.horizon+1, device=self.device)
                    controlled = []
                    step_limit = float(self.candidate_safety.cfg.get("max_steering_rate", 7.))*self.candidate_safety.geometry.dt/safety_context.max_steering
                    for fraction in self.straighten_fractions:
                        for throttle in self.throttle_levels:
                            steer = previous[0]*(1-fraction*progress)
                            last = previous[0]
                            for i in range(len(steer)):
                                steer[i] = last+(steer[i]-last).clamp(-step_limit, step_limit)
                                last = steer[i].clone()
                            controlled.append(torch.stack((steer, torch.full_like(steer, throttle)), -1))
                    choices = torch.stack(controlled).clamp(-1, 1)
                    valid, keys, _ = check_actions(choices, baseline, None)
                    indices = valid.nonzero().flatten()
                    if len(indices):
                        choices = choices[indices]
                        candidate_keys = [keys[i] for i in indices.cpu().tolist()]
                        reference = self.candidate_safety.braking_sequence(safety_context,
                            horizon=self.horizon, device=self.device, dtype=baseline.dtype)
                        rows, scores, *_ = evaluate(choices, candidate_keys, reference, self.recovery_delta)
                        finite = torch.isfinite(scores).nonzero().flatten()
                        if len(finite):
                            i = int(finite[scores[finite].argmax()])
                            recovery_best = (float(scores[i]), choices[i].clone(), rows[i], candidate_keys[i])
                            score_reference, score_bounds, mode = reference, self.recovery_delta, "recovery_controlled"
            recovery_checked, recovery_rejects = checked-start_checked, safety_rejected-start_rejects
            recovery_evaluated = evaluated-start_evaluated
        baseline_selected = baseline_safe and (best is None or
            (baseline_finite and best[0]-float(baseline_score) <= threshold))
        if recovery_best is not None:
            _, sequence, selected, selected_key = recovery_best
            baseline_selected = torch.equal(sequence, baseline)
            mean, std = ((rmean, rstd) if mode == "recovery_lattice" else
                         (sequence.clone(), torch.zeros_like(sequence)))
        elif baseline_selected:
            sequence, selected = baseline, base
            selected_key = baseline_key
        elif best is not None:
            _, sequence, selected, selected_key = best
            baseline_selected = torch.equal(sequence, baseline)
            mode = "baseline" if baseline_selected else "normal"
        else:
            sequence, selected = None, blank()
            if lattice_actions is not None:
                lattice = lattice_actions.to(self.device).clamp(-1, 1)
                valid, keys, _ = check_actions(lattice.unsqueeze(0), lattice, lattice_corridor_context)
                if bool(valid[0]):
                    sequence, lattice_fallback = lattice, True
                    mode = "lattice_fallback"
                    selected_key = keys[0]
                    rows, *_ = evaluate(lattice.unsqueeze(0), keys)
                    selected = rows[0]
            if sequence is None:
                sequence = self.candidate_safety.braking_sequence(safety_context, horizon=self.horizon,
                                                                 device=self.device, dtype=baseline.dtype)
                # Emergency braking is not tied to an unsafe SAC reference or
                # its trust region, but must pass the same physical safety check.
                valid, keys, _ = check_actions(sequence.unsqueeze(0), sequence, None)
                emergency_fallback, unverified = True, not bool(valid[0])
                mode = "emergency_unverified" if unverified else "emergency_verified"
                mean, std = sequence.clone(), torch.zeros_like(sequence)
                selected_key = keys[0]
                if not unverified:
                    rows, *_ = evaluate(sequence.unsqueeze(0), keys)
                    selected = rows[0]

        full_braking = bool((sequence[:self.horizon, 1] <= self.candidate_safety.emergency_throttle).all())
        if mode == "recovery_controlled" and full_braking:
            # Full braking discovered among recovery candidates is still a
            # braking action, not evidence of improved search corrections.
            mode, emergency_fallback = "emergency_verified", True
        deviation, smoothness = self._penalties(sequence.unsqueeze(0), score_reference, score_bounds)
        cost_scored = enabled and (selected["ranking_validated"] or self.allow_unvalidated_cost)
        cost_penalty = self.cost_coef*selected["cost"] if cost_scored else baseline.new_tensor(0.)
        selected_score = selected["value"]-cost_penalty-self.deviation_coef*deviation[0]-self.smoothness_coef*smoothness[0]
        score_gain = (selected_score-baseline_score if baseline_finite and mode in {"baseline", "normal"} and math.isfinite(float(selected_score))
                      else baseline.new_tensor(float("nan")))
        value_gain = (selected["value"]-base["value"] if baseline_finite else baseline.new_tensor(float("nan")))
        if cost_scored and math.isfinite(float(selected["cost"])) and not selected["ranking_validated"] and not self._warned_unvalidated:
            warnings.warn("Experimental WM Cost ranking has NOT passed same-state candidate validation; this is not a safety guarantee.",
                          RuntimeWarning, stacklevel=2)
            self._warned_unvalidated = True
        timing.update(phase.finish())
        for name in ("mppi_sampling_ms", "safety_check_ms", "world_model_rollout_ms", "cost_prediction_ms", "elite_update_ms", "recovery_search_ms"):
            timing.setdefault(name, 0.)
        delta = sequence[0]-baseline[0]
        max_delta = (sequence-baseline).abs().amax(0)
        metrics = {
            **timing,
            "predicted_reward": float(selected["reward"]), "terminal_q": float(selected["terminal_q"]),
            "predicted_cost": float(selected["cost"]), "cost_penalty": float(cost_penalty),
            "predicted_step_cost": float(selected["step_cost"]),
            "cost_coef": self.cost_coef, "cost_enabled": float(cost_scored),
            "cost_scoring_requested": float(enabled), "allow_unvalidated_cost": float(self.allow_unvalidated_cost),
            "cost_validated": float(selected["ranking_validated"]), "cost_horizon_validated": float(selected["validated"]),
            "deviation_penalty": float(self.deviation_coef*deviation[0]),
            "smoothness_penalty": float(self.smoothness_coef*smoothness[0]),
            "final_score": float(selected_score), "selected_candidate_cost": float(selected["cost"]),
            "baseline_candidate_cost": float(base["cost"]), "wm_evaluated_candidates": float(evaluated),
            "wm_cache_hits": float(wm_requested-evaluated), "baseline_corridor_valid": float(baseline_corridor_valid),
            "first_steer_delta": float(delta[0]), "first_throttle_delta": float(delta[1]),
            "first_action_delta_norm": float(delta.norm()), "baseline_selected": float(baseline_selected),
            "score_gain": float(score_gain), "gate_threshold": threshold,
            "initial_std_steer": float(initial_std[:, 0].mean()), "initial_std_throttle": float(initial_std[:, 1].mean()),
            "final_std_steer": float(std[:, 0].mean()), "final_std_throttle": float(std[:, 1].mean()),
            "max_delta_steer": float(max_delta[0]), "max_delta_throttle": float(max_delta[1]),
            "unique_candidate_count": float(len(unique)), "safety_evaluated_candidates": float(len(safety_cache)),
            "safety_valid_ratio": 1-safety_rejected/max(checked, 1) if self.safety_enabled else float("nan"),
            "safety_reject_count": float(safety_rejected), "collision_reject_count": float(collision_rejected),
            "road_boundary_reject_count": float(road_rejected), "kinematic_reject_count": float(kinematic_rejected),
            "baseline_safety_valid": float(baseline_safe) if self.safety_enabled else float("nan"),
            "emergency_fallback": float(emergency_fallback), "lattice_fallback": float(lattice_fallback),
            "safety_fallback_unverified": float(unverified),
            "selected_safety_valid": float(not unverified) if self.safety_enabled else float("nan"),
            "normal_safety_valid_ratio": 1-normal_rejects/max(normal_checked, 1) if self.safety_enabled else float("nan"),
            "recovery_safety_valid_ratio": 1-recovery_rejects/max(recovery_checked, 1) if recovery_checked else float("nan"),
            "lattice_baseline_safety_valid": float(lattice_safe),
            "lattice_baseline_checked": float(math.isfinite(lattice_safe)),
            "recovery_search": float(recovery_used), "recovery_wm_evaluated_candidates": float(recovery_evaluated),
            "planning_mode": float({"baseline": 0, "normal": 1, "recovery_lattice": 2, "recovery_controlled": 3,
                                     "lattice_fallback": 4, "emergency_verified": 5, "emergency_unverified": 6}[mode]),
            "recovery_reason": float({"none": 0, "unsafe_sac_baseline": 1, "no_usable_normal_candidate": 2}[recovery_reason]),
            "selected_full_braking": float(full_braking),
        }
        for label, active in (("normal_search", mode == "normal"),
                              ("recovery_search", mode.startswith("recovery_")),
                              ("lattice_fallback", mode == "lattice_fallback"),
                              ("emergency_fallback", emergency_fallback)):
            part = delta if active else torch.zeros_like(delta)
            metrics.update({f"{label}_delta": float(part.norm()), f"{label}_steer_delta": float(part[0]),
                            f"{label}_throttle_delta": float(part[1])})
        flags = safety_cache.get(selected_key, {})
        for target, source in (("predicted_collision", "collision"), ("predicted_offroad", "offroad"),
                               ("predicted_first_collision", "first_collision"), ("predicted_first_offroad", "first_offroad")):
            metrics[target] = float(flags.get(source, float("nan")))
        if self.safety_enabled:
            metrics.update(self.candidate_safety.stopping_diagnostics(safety_context, self.horizon))
        self._prev_mean = mean.detach().clone()
        if corridor_context is None:
            lateral = 0.
        elif self.safety_enabled:
            lateral = float(self.safety_corridor.check(sequence[None].cpu(), baseline_host,
                corridor_snapshots[id(corridor_context)]).max_lateral_deviation[0])
        else:
            lateral = float(self.corridor.check(sequence[None], baseline, corridor_context).max_lateral_deviation[0])
        return LocalMPPIResult(
            action=sequence[0], selected_sequence=sequence, initial_mean=initial_mean, final_mean=mean,
            initial_std=initial_std, final_std=std, baseline_wm_value=base["value"],
            selected_wm_value=selected["value"], wm_value_gain=value_gain, baseline_score=baseline_score,
            selected_score=selected_score, planner_score_gain=score_gain, baseline_value=baseline_score,
            final_value=selected_score, planner_gain=score_gain, elite_values=elite_values,
            baseline_selected=baseline_selected, corridor_reject_count=rejected,
            corridor_reject_rate=rejected/max(checked, 1), max_lateral_deviation=lateral,
            max_action_delta=(sequence-baseline).abs().amax(0), metrics=metrics,
            baseline_consequence=base, selected_consequence=selected,
            selected_positions=(safety_cache[selected_key]["positions"]
                                [::self.candidate_safety.geometry.substeps].clone()
                                if self.safety_enabled and "positions" in safety_cache.get(selected_key, {})
                                else None),
            planning_mode=mode, recovery_reason=recovery_reason,
        )


__all__ = ["LocalMPPI", "LocalMPPIResult", "_action_vector", "MPPI_DIAGNOSTIC_KEYS"]
