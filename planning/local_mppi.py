"""Single-Gaussian H+1 MPPI with independent pre-WM candidate safety."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from collections import deque
import math
import time
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
    "sac_to_selected_delta", "sac_to_selected_steer_delta", "sac_to_selected_throttle_delta",
    "recovery_lattice_baseline_selected", "emergency_horizon_limited",
    "near_hard_valid_ratio", "baseline_near_valid", "lattice_baseline_near_valid",
    "baseline_full_valid", "lattice_baseline_full_valid", "long_risk_score", "long_risk_penalty",
    "long_road_risk", "long_collision_risk", "long_clearance_risk", "long_speed_risk", "long_kinematic_risk",
    "future_unrecoverable_reject_count", "normal_near_valid_ratio", "recovery_lattice_valid_ratio",
    "committed_horizon_s", "emergency_streak", "braking_streak", "recovery_streak", "mode_switch_count",
    "emergency_reason", "scoring_unavailable_fallback", "target_speed_mps", "actual_speed_mps",
    "baseline_future_hard_reject", "baseline_near_collision", "baseline_near_offroad",
    "lattice_baseline_future_hard_reject", "lattice_baseline_near_collision", "lattice_baseline_near_offroad",
    "selected_near_valid", "selected_full_valid",
    "long_lateral_risk",
    "exact_checked_candidates", "exact_failed_candidates", "selected_exact_verified", "final_safety_verification_ms",
    "recovery_stall_count", "stall_active", "can_accelerate", "accelerate_below_target", "accelerate_no_near_collision",
    "positive_throttle_candidate_count", "positive_throttle_rejected_count", "positive_throttle_exact_rejected_count",
    "positive_throttle_near_collision_rejected", "positive_throttle_near_offroad_rejected", "positive_throttle_kinematic_rejected",
    "positive_throttle_wm_rejected_count", "positive_throttle_feasible_not_selected_count",
    "positive_throttle_unverified_count", "positive_throttle_score_not_selected_count",
    "selected_recovery_steer", "selected_recovery_throttle", "progress_delta", "progress_distance_m", "recovery_success",
    "mode_transition_reason", "geometric_steer", "geometric_lateral_error", "geometric_heading_error", "geometric_body_margin",
    "near_collision", "near_offroad", "future_hard", "progress_reward", "value_p50", "risk_p50", "risk_p95",
    "physics_position_error_m", "physics_speed_error_mps", "physics_heading_error_rad", "physics_acceleration_bias_mps2",
    "physics_braking_response_bias_mps2", "execution_action_error",
    "predicted_progress_m", "preview_candidate_count", "exact_candidate_count",
    "positive_throttle_future_hard_rejected", "positive_throttle_exact_offroad_rejected",
    "positive_throttle_exact_collision_rejected", "positive_throttle_exact_kinematic_rejected", "candidate_evaluation_ms",
    "accelerate_road_clear", "accelerate_kinematic_valid", "positive_throttle_near_safe_count",
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
        self.long_risk_coef = float(safety_cfg.get("long_risk_coef", 1.)) if self.candidate_safety.two_layer else 0.
        if not math.isfinite(self.long_risk_coef) or self.long_risk_coef < 0:
            raise ValueError("Long-horizon risk coefficient must be finite and nonnegative.")
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
        self.lattice_guided = bool(recovery.get("lattice_guided", False))
        self.positive_throttle = float(recovery.get("positive_throttle", .1))
        self.anti_stall = bool(recovery.get("anti_stall", False))
        self.stall_speed = float(recovery.get("stall_speed_mps", .3))
        self.stall_duration = float(recovery.get("stall_duration_s", 2.))
        self.stall_progress = float(recovery.get("stall_progress_m", .2))
        self.stall_exit = float(recovery.get("stall_exit_s", 1.))
        if not all(math.isfinite(v) and v > 0 for v in (self.stall_speed, self.stall_duration, self.stall_progress, self.stall_exit)):
            raise ValueError("Stall thresholds must be finite and positive.")
        self._motion = deque(maxlen=max(2, math.ceil(self.stall_duration/control_dt)))
        self._stall_active = False
        self._stall_clear = 0
        self._stall_count = 0
        self._last_position = self._last_progress = None
        self._last_measurement_step = None
        if not math.isfinite(self.positive_throttle) or not 0 < self.positive_throttle <= 1:
            raise ValueError("Recovery positive_throttle must be in (0,1].")
        if (not self.straighten_fractions or not self.throttle_levels
                or len(self.straighten_fractions)*len(self.throttle_levels) > 16
                or any(not math.isfinite(v) or not 0 <= v <= 1 for v in self.straighten_fractions)
                or any(not math.isfinite(v) or not -1 <= v <= 0 for v in self.throttle_levels)):
            raise ValueError("Controlled recovery requires <=16 bounded hold/straighten/decelerate candidates.")
        self.generator = torch.Generator(device=self.device).manual_seed(self.eval_seed)
        self._prev_mean = None
        self._warned_unvalidated = False
        self._emergency_streak = self._braking_streak = self._recovery_streak = self._mode_switches = 0
        self._last_mode = None

    def reset(self, seed=None):
        self._prev_mean = None
        self.generator.manual_seed(self.eval_seed if seed is None else int(seed) % 2**32)
        self.candidate_safety.reset()
        self._emergency_streak = self._braking_streak = self._recovery_streak = self._mode_switches = 0
        self._last_mode = None
        self._motion.clear()
        self._stall_active = False
        self._stall_clear = self._stall_count = 0
        self._last_position = self._last_progress = None
        self._last_measurement_step = None

    def _observe_motion(self, context):
        """Distance/speed hysteresis uses measured motion, never requested throttle."""
        if context is None:
            return 0., 0., False
        token = getattr(context, "measurement_step", None)
        if token is not None and token == self._last_measurement_step:
            return 0., 0., False  # repeated planning is not elapsed driving time
        self._last_measurement_step = token
        position = torch.as_tensor(context.position).double()
        distance = 0.
        if self._last_position is not None:
            direction = position.new_tensor([math.cos(context.heading), math.sin(context.heading)])
            distance = float((position-self._last_position)@direction)
        progress = getattr(context, "route_progress", None)
        delta = float(progress-self._last_progress) if progress is not None and self._last_progress is not None else 0.
        self._last_position, self._last_progress = position.clone(), progress
        self._motion.append((float(context.speed), distance))
        success = False
        if self.anti_stall:
            blocked = (len(self._motion) == self._motion.maxlen and
                max(v for v, _ in self._motion) < self.stall_speed and
                sum(max(0., d) for _, d in self._motion) < self.stall_progress)
            if blocked and not self._stall_active:
                self._stall_active = True
                self._stall_count += 1
            moving = context.speed >= 2*self.stall_speed and distance > 0
            self._stall_clear = self._stall_clear+1 if moving else 0
            if self._stall_active and self._stall_clear*self.candidate_safety.geometry.dt >= self.stall_exit:
                self._stall_active, success = False, True
        return delta, distance, success

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
        progress_delta, progress_distance, recovery_success = self._observe_motion(safety_context)
        positive_count = positive_rejected = positive_wm_rejected = 0
        positive_reasons = {name: 0 for name in ("collision", "offroad", "kinematic")}
        positive_keys = []
        can_accelerate = accelerate_below_target = accelerate_no_collision = False
        baseline_host = baseline.detach().cpu()
        rejected = checked = evaluated = wm_requested = 0
        enabled = self.cost_enabled and self.cost_coef > 0
        safety_rejected = collision_rejected = road_rejected = kinematic_rejected = 0
        baseline_safe, baseline_corridor_valid = False, False
        best = None
        safe_fallback = None
        fallback_candidates = {}
        exact_checked = exact_failed = 0
        near_checked = future_hard_rejected = normal_near_checked = 0
        elite_values = baseline.new_empty(0)

        def blank():
            return {**{key: baseline.new_tensor(float("nan"))
                       for key in ("value", "cost", "step_cost", "reward", "terminal_q")},
                    "validated": False, "ranking_validated": False, "long_risk": 0.}

        def check_actions(actions, reference, corridor_ctx, *, exact=False):
            nonlocal checked, rejected, safety_rejected, collision_rejected, road_rejected, kinematic_rejected
            nonlocal safe_fallback, near_checked, future_hard_rejected
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
                # Frenet corridor is a heuristic tied to the nominal path, not
                # an independent world-frame safety certificate. In two-layer
                # mode retain its diagnostics; the swept world-frame prefix
                # and checked recovery continuation decide admissibility.
                if corridor is not None and not self.candidate_safety.two_layer:
                    fast &= corridor.valid
                fast_list = fast.cpu().tolist()
                if self.safety_enabled:
                    missing = {}
                    for i, key in enumerate(keys):
                        if fast_list[i] and (key not in safety_cache or (exact and not safety_cache[key].get("exact", False))):
                            missing.setdefault(key, i)
                    if missing:
                        checker = (self.candidate_safety.check if exact or not self.candidate_safety.staged
                                   else self.candidate_safety.preview)
                        result = checker(host[list(missing.values())], safety_context)
                        columns = {name: getattr(result, name).tolist() for name in
                                   ("valid", "collision", "offroad", "kinematic_violation")}
                        for name in ("first_collision", "first_offroad"):
                            value = getattr(result, name, None)
                            if value is not None:
                                columns[name] = value.tolist()
                        for name in ("near_valid", "near_collision", "near_offroad", "near_kinematic", "full_valid", "risk_score", "road_risk", "collision_risk",
                                     "clearance_risk", "speed_risk", "lateral_risk", "kinematic_risk", "unrecoverable", "progress_estimate"):
                            value = getattr(result, name, None)
                            if value is not None:
                                columns[name] = value.tolist()
                        for i, key in enumerate(missing):
                            safety_cache[key] = {name: values[i] for name, values in columns.items()}
                            safety_cache[key]["exact"] = exact or not self.candidate_safety.staged
                            if hasattr(result, "positions"):
                                safety_cache[key]["positions"] = result.positions[i]
                    flags = [safety_cache[key] if fast_list[i] else
                             dict(valid=False, collision=False, offroad=False, kinematic_violation=True)
                             for i, key in enumerate(keys)]
                    valid = torch.tensor([fast_list[i] and row["valid"] for i, row in enumerate(flags)],
                                         dtype=torch.bool, device=self.device)
                    collision_rejected += sum(not row["valid"] and bool(row["collision"]) for row in flags)
                    road_rejected += sum(not row["valid"] and bool(row["offroad"]) for row in flags)
                    kinematic_rejected += sum(not row["valid"] and bool(row["kinematic_violation"]) for row in flags)
                    near_checked += sum(bool(row.get("near_valid", row["valid"])) for row in flags)
                    future_hard_rejected += sum(bool(row.get("unrecoverable", False)) for row in flags)
                    for i in valid.nonzero().flatten().cpu().tolist():
                        risk = float(flags[i].get("risk_score", 0.))
                        brake = bool(self.lattice_guided and
                            (host[i, :self.horizon, 1] <= self.candidate_safety.emergency_throttle).all())
                        if safe_fallback is None or (brake, risk) < safe_fallback[:2]:
                            safe_fallback = (brake, risk, host[i].clone(), keys[i])
                        fallback_candidates[keys[i]] = (brake, risk, host[i].clone(), keys[i])
                else:
                    valid = fast
                checked += len(actions)
                rejected += int((~fast).sum())
                safety_rejected += int((~valid).sum())
            return valid, keys, corridor

        def score(actions, values, reference=baseline, bounds=None):
            deviation, smoothness = self._penalties(actions, reference, bounds)
            cost_penalty = (torch.where(values["ranking_validated"] | self.allow_unvalidated_cost,
                self.cost_coef*values["cost"], 0.) if enabled else torch.zeros_like(values["value"]))
            scores = (values["value"]-cost_penalty-self.deviation_coef*deviation-self.smoothness_coef*smoothness
                      -self.long_risk_coef*values["long_risk"])
            return scores, deviation, smoothness, cost_penalty

        def evaluate(actions, keys, reference=baseline, bounds=None):
            nonlocal evaluated, wm_requested
            started_evaluation = time.perf_counter()
            wm_requested += len(actions)
            rows, values, count = self._evaluate_cached(z_t, actions, cache, task, timing, keys)
            for key, row in zip(keys, rows):
                row["long_risk"] = float(safety_cache.get(key, {}).get("risk_score", 0.))
            values["long_risk"] = actions.new_tensor([row["long_risk"] for row in rows])
            evaluated += count
            scored = score(actions, values, reference, bounds)
            timing["candidate_evaluation_ms"] = timing.get("candidate_evaluation_ms", 0.)+1000*(time.perf_counter()-started_evaluation)
            return rows, *scored, values

        def verified_choice(actions, keys, rows, scores, indices):
            """Promote candidates by priority; preview flags never authorize execution."""
            nonlocal exact_checked, exact_failed
            if not self.candidate_safety.staged:
                i = int(indices[0])
                return (float(scores[i]), actions[i].clone(), rows[i], keys[i])
            for i in indices.cpu().tolist():
                old_risk = rows[i].get("long_risk", 0.)
                with phase.measure("final_safety_verification"):
                    was_exact = safety_cache[keys[i]].get("exact", False)
                    valid, _, _ = check_actions(actions[i:i+1], baseline, None, exact=True)
                    exact_checked += int(not was_exact)
                    if not bool(valid[0]):
                        exact_failed += int(not was_exact)
                        continue
                # Recompute the common objective with exact risk, the same SAC
                # reference and the same penalty bounds for BOTH search modes.
                new_risk = float(safety_cache[keys[i]].get("risk_score", 0.))
                checked_score = float(scores[i])+self.long_risk_coef*(old_risk-new_risk)
                rows[i]["long_risk"] = new_risk
                if math.isfinite(checked_score):
                    return (checked_score, actions[i].clone(), rows[i], keys[i])
            return None

        baseline_key = baseline.detach().cpu().numpy().tobytes()
        if self.safety_enabled and self.candidate_safety.staged:
            check_actions(baseline[None], baseline, corridor_context, exact=True)
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
            rows, scores, deviation, smoothness, cost_penalty, _ = evaluate(candidate_actions, candidate_keys)
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
            priority = scores.argsort(descending=True)
            priority = priority[torch.isfinite(scores[priority])]
            if not eval_mode:
                first = indices[torch.multinomial(weights, 1)]
                priority = torch.cat((first, priority[priority != first]))
            choice = verified_choice(candidate_actions, candidate_keys, rows, scores, priority)
            if choice is not None and (best is None or choice[0] > best[0]):
                best = choice
            if self.recovery_enabled and not baseline_safe:
                break

        normal_checked, normal_rejects = checked-normal_checked_start, safety_rejected
        normal_near_checked = near_checked
        base = cache.get(baseline_key, blank())
        baseline_cost_scored = enabled and (base["ranking_validated"] or self.allow_unvalidated_cost)
        baseline_score = (base["value"]-(self.cost_coef*base["cost"] if baseline_cost_scored else 0.)
                          -self.long_risk_coef*base.get("long_risk", 0.))
        baseline_finite = baseline_safe and math.isfinite(float(baseline_score))
        threshold = (max(self.min_improvement_abs, self.min_improvement_ratio*abs(float(baseline_score)))
                     if baseline_finite else float("nan"))
        lattice_fallback = emergency_fallback = unverified = False
        recovery_used, lattice_safe = False, float("nan")
        if lattice_actions is not None and torch.equal(lattice_actions.to(self.device), baseline):
            lattice_safe = baseline_safe
        recovery_reason = "none"
        score_reference, score_bounds = baseline, self.max_delta
        recovery_reference = None
        mode = "baseline"
        recovery_checked = recovery_rejects = recovery_evaluated = 0
        lattice_checked = lattice_passed = 0
        recovery_best = None
        if self.recovery_enabled and (not baseline_safe or best is None or self._stall_active):
            recovery_used = True
            recovery_reason = ("low_speed_stall" if self._stall_active else
                "unsafe_sac_baseline" if not baseline_safe else "no_usable_normal_candidate")
            start_checked, start_rejects, start_evaluated = checked, safety_rejected, evaluated
            with phase.measure("recovery_search"):
                lattice = lattice_actions.to(self.device).clamp(-1, 1) if lattice_actions is not None else None
                if lattice is not None:
                    valid, keys, _ = check_actions(lattice[None], lattice, lattice_corridor_context, exact=True)
                    lattice_safe = bool(valid[0])
                if lattice is not None and lattice_safe and not self._stall_active:
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
                        lattice_checked += len(actions)
                        lattice_passed += int(valid.sum())
                        indices = valid.nonzero().flatten()
                        if not len(indices):
                            continue
                        choices, candidate_keys = actions[indices], [keys[i] for i in indices.cpu().tolist()]
                        rows, scores, _, _, _, values = evaluate(choices, candidate_keys, lattice, self.recovery_delta)
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
                        # Distribution updates keep their local objective. Final
                        # selection ranks ALL safe recovery candidates against
                        # the same SAC reference/bounds as normal search, using
                        # cached WM values; do not compare unlike local scores.
                        common_scores, *_ = score(choices, values)
                        common_finite = torch.isfinite(common_scores).nonzero().flatten()
                        if len(common_finite):
                            priority = common_finite[common_scores[common_finite].argsort(descending=True)]
                            choice = verified_choice(choices, candidate_keys, rows, common_scores, priority)
                            if choice is not None and (recovery_best is None or choice[0] > recovery_best[0]):
                                recovery_best = choice
                    if recovery_best is not None:
                        recovery_reference, mode = lattice, "recovery_lattice"
                else:
                    # Small deterministic controls, not another search mode
                    # bank/controller. Rate-limited first step uses measured
                    # previous steering; every trajectory is independently checked.
                    previous = baseline.new_tensor(safety_context.previous_action)
                    progress = torch.linspace(1/(self.horizon+1), 1., self.horizon+1, device=self.device)
                    controlled = []
                    step_limit = float(self.candidate_safety.cfg.get("max_steering_rate", 7.))*self.candidate_safety.geometry.dt/safety_context.max_steering
                    levels = self.throttle_levels
                    guidance = lattice if self.lattice_guided and lattice is not None else None
                    if guidance is not None:
                        flags = safety_cache.get(keys[0], {}) if keys else {}
                        target = getattr(safety_context, "target_speed", None)
                        accelerate_below_target = target is not None and safety_context.speed < target
                        accelerate_no_collision = not flags.get("near_collision", flags.get("first_collision", False))
                        # Lattice's collision is a diagnostic, not a veto on a
                        # DIFFERENT geometric steering proposal. Each positive
                        # candidate receives the same prefix and exact checks.
                        can_accelerate = accelerate_below_target
                        if can_accelerate:
                            # Replace one old coasting/braking level, not the
                            # sample budget. Small positive actions must pass
                            # the SAME imminent and viability checks below.
                            requested = min(self.positive_throttle,
                                (target-safety_context.speed)/(self.horizon*self.candidate_safety.geometry.dt
                                    *float(self.candidate_safety.cfg.get("max_acceleration", 4.))))
                            levels = (requested, 0., self.candidate_safety.emergency_throttle)
                    for fraction in self.straighten_fractions:
                        for throttle in levels:
                            steer = (previous[0]+fraction*(guidance[:, 0]-previous[0]) if guidance is not None
                                     else previous[0]*(1-fraction*progress))
                            geometric = getattr(safety_context, "geometric_steer", None)
                            if self.anti_stall and geometric is not None and fraction == self.straighten_fractions[len(self.straighten_fractions)//2]:
                                steer = torch.full_like(steer, geometric)
                            tangent = getattr(safety_context, "road_tangent_steer", None)
                            if self.anti_stall and tangent is not None and fraction == self.straighten_fractions[0]:
                                steer = torch.full_like(steer, tangent)
                            last = previous[0]
                            for i in range(len(steer)):
                                steer[i] = last+(steer[i]-last).clamp(-step_limit, step_limit)
                                last = steer[i].clone()
                            controlled.append(torch.stack((steer, torch.full_like(steer, throttle)), -1))
                    choices = torch.stack(controlled).clamp(-1, 1)
                    valid, keys, _ = check_actions(choices, baseline, None)
                    positive = choices[:, 0, 1] > 0
                    positive_count = int(positive.sum())
                    positive_keys = [key for key, flag in zip(keys, positive.cpu().tolist()) if flag]
                    positive_rejected = int((positive & ~valid).sum())
                    for key in positive_keys:
                        flags = safety_cache.get(key, {})
                        for name in positive_reasons:
                            positive_reasons[name] += int(not flags.get("valid", False) and
                                flags.get("near_kinematic" if name == "kinematic" else f"near_{name}", False))
                    indices = valid.nonzero().flatten()
                    if len(indices):
                        choices = choices[indices]
                        candidate_keys = [keys[i] for i in indices.cpu().tolist()]
                        reference = self.candidate_safety.braking_sequence(safety_context,
                            horizon=self.horizon, device=self.device, dtype=baseline.dtype)
                        rows, _, _, _, _, values = evaluate(choices, candidate_keys, reference, self.recovery_delta)
                        common_scores, *_ = score(choices, values)
                        finite = torch.isfinite(common_scores).nonzero().flatten()
                        positive_wm_rejected = int(((choices[:, 0, 1] > 0) & ~torch.isfinite(common_scores)).sum())
                        if self.lattice_guided and len(finite):
                            # Full emergency braking must not win an ordinary
                            # WM ranking contest while a checked non-emergency
                            # continuation exists. Coasting/deceleration remain
                            # eligible; this never forces a positive action.
                            non_emergency = ~(choices[:, :self.horizon, 1] <= self.candidate_safety.emergency_throttle).all(1)
                            if bool(non_emergency.any()):
                                finite = finite[non_emergency[finite]]
                        if len(finite):
                            priority = finite[common_scores[finite].argsort(descending=True)]
                            recovery_best = verified_choice(choices, candidate_keys, rows, common_scores, priority)
                            recovery_reference, mode = reference, "recovery_controlled"
            recovery_checked, recovery_rejects = checked-start_checked, safety_rejected-start_rejects
            recovery_evaluated = evaluated-start_evaluated
        choose_recovery = recovery_best is not None and (best is None or recovery_best[0] > best[0])
        winner = recovery_best if choose_recovery else best
        baseline_selected = baseline_safe and (winner is None or
            (baseline_finite and winner[0]-float(baseline_score) <= threshold))
        scoring_unavailable = False
        if baseline_selected:
            sequence, selected, selected_key, mode = baseline, base, baseline_key, "baseline"
        elif choose_recovery:
            _, sequence, selected, selected_key = recovery_best
            baseline_selected = torch.equal(sequence, baseline)
            mean, std = ((rmean, rstd) if mode == "recovery_lattice" else
                         (sequence.clone(), torch.zeros_like(sequence)))
        elif best is not None:
            _, sequence, selected, selected_key = best
            baseline_selected = torch.equal(sequence, baseline)
            mode = "baseline" if baseline_selected else "normal"
        else:
            sequence, selected = None, blank()
            if lattice_actions is not None:
                lattice = lattice_actions.to(self.device).clamp(-1, 1)
                valid, keys, _ = check_actions(lattice.unsqueeze(0), lattice, lattice_corridor_context, exact=True)
                if bool(valid[0]):
                    sequence, lattice_fallback = lattice, True
                    mode = "lattice_fallback"
                    selected_key = keys[0]
                    rows, *_ = evaluate(lattice.unsqueeze(0), keys)
                    selected = rows[0]
            if sequence is None and safe_fallback is not None:
                # WM scores can be unavailable while geometric candidates are
                # feasible. This is not an emergency-braking condition.
                for _, _, fallback, key in sorted(fallback_candidates.values(), key=lambda row: row[:2]):
                    fallback = fallback.to(self.device)
                    valid, _, _ = check_actions(fallback[None], baseline, None, exact=True)
                    if bool(valid[0]):
                        sequence, selected_key = fallback, key
                        selected = cache.get(key, blank())
                        mode, scoring_unavailable = "normal", True
                        break
            if sequence is None:
                sequence = self.candidate_safety.braking_sequence(safety_context, horizon=self.horizon,
                                                                 device=self.device, dtype=baseline.dtype)
                # Emergency braking is not tied to an unsafe SAC reference or
                # its trust region, but must pass the same physical safety check.
                valid, keys, _ = check_actions(sequence.unsqueeze(0), sequence, None, exact=True)
                emergency_fallback, unverified = True, not bool(valid[0])
                mode = "emergency_model_check_failed" if unverified else "emergency_model_checked"
                mean, std = sequence.clone(), torch.zeros_like(sequence)
                selected_key = keys[0]
                if not unverified:
                    rows, *_ = evaluate(sequence.unsqueeze(0), keys)
                    selected = rows[0]

        if self.safety_enabled and not unverified:
            certificate = safety_cache[selected_key]
            if not certificate.get("exact", False) or not certificate["valid"]:
                raise RuntimeError("Selected MPPI action has no exact safety certificate.")
        full_braking = bool((sequence[:self.horizon, 1] <= self.candidate_safety.emergency_throttle).all())
        if full_braking and (mode == "recovery_controlled" or scoring_unavailable):
            # Full braking discovered among recovery candidates is still a
            # braking action, not evidence of improved search corrections.
            mode, emergency_fallback = "emergency_model_checked", True
        deviation, smoothness = self._penalties(sequence.unsqueeze(0), score_reference, score_bounds)
        cost_scored = enabled and (selected["ranking_validated"] or self.allow_unvalidated_cost)
        cost_penalty = self.cost_coef*selected["cost"] if cost_scored else baseline.new_tensor(0.)
        long_penalty = self.long_risk_coef*selected.get("long_risk", 0.)
        selected_score = selected["value"]-cost_penalty-self.deviation_coef*deviation[0]-self.smoothness_coef*smoothness[0]-long_penalty
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
            "exact_checked_candidates": float(exact_checked), "exact_failed_candidates": float(exact_failed),
            "selected_exact_verified": float(self.safety_enabled and not unverified),
            "recovery_stall_count": float(self._stall_count), "stall_active": float(self._stall_active),
            # Availability refers to the positive proposals' OWN committed
            # prefix, not the possibly different Lattice steering direction.
            "can_accelerate": float(any(safety_cache[k].get("near_valid", safety_cache[k]["valid"]) for k in positive_keys)),
            "accelerate_below_target": float(accelerate_below_target),
            "accelerate_no_near_collision": float(any(not safety_cache[k].get("near_collision", False) for k in positive_keys)),
            "accelerate_road_clear": float(any(not safety_cache[k].get("near_offroad", False) for k in positive_keys)),
            "accelerate_kinematic_valid": float(any(not safety_cache[k].get("near_kinematic", False) for k in positive_keys)),
            "positive_throttle_near_safe_count": float(sum(safety_cache[k].get("near_valid", safety_cache[k]["valid"]) for k in positive_keys)),
            "positive_throttle_candidate_count": float(positive_count), "positive_throttle_rejected_count": float(positive_rejected),
            "positive_throttle_exact_rejected_count": float(sum(safety_cache[k].get("exact", False) and not safety_cache[k]["valid"] for k in positive_keys)),
            "positive_throttle_wm_rejected_count": float(positive_wm_rejected),
            "positive_throttle_feasible_not_selected_count": float(sum(safety_cache[k]["valid"] and safety_cache[k].get("exact", False) and k != selected_key for k in positive_keys)),
            "positive_throttle_unverified_count": float(sum(not safety_cache[k].get("exact", False) and safety_cache[k]["valid"] for k in positive_keys)),
            "positive_throttle_score_not_selected_count": float(sum(k != selected_key and k in cache and math.isfinite(float(cache[k]["value"]))
                and safety_cache[k]["valid"] for k in positive_keys)),
            "progress_delta": progress_delta, "progress_distance_m": progress_distance, "recovery_success": float(recovery_success),
            "selected_recovery_steer": float(sequence[0, 0]) if mode.startswith("recovery_") else float("nan"),
            "selected_recovery_throttle": float(sequence[0, 1]) if mode.startswith("recovery_") else float("nan"),
            "mode_transition_reason": float(0 if mode == self._last_mode else 3 if recovery_success else 2 if self._stall_active else 1),
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
            "sac_to_selected_delta": float(delta.norm()), "sac_to_selected_steer_delta": float(delta[0]),
            "sac_to_selected_throttle_delta": float(delta[1]),
            "recovery_lattice_baseline_selected": float(mode == "recovery_lattice" and
                torch.equal(sequence, recovery_reference)),
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
                                     "lattice_fallback": 4, "emergency_model_checked": 5, "emergency_model_check_failed": 6}[mode]),
            "recovery_reason": float({"none": 0, "unsafe_sac_baseline": 1, "no_usable_normal_candidate": 2, "low_speed_stall": 3}[recovery_reason]),
            "selected_full_braking": float(full_braking),
            "near_hard_valid_ratio": near_checked/max(checked, 1) if self.safety_enabled else float("nan"),
            "normal_near_valid_ratio": normal_near_checked/max(normal_checked, 1) if self.safety_enabled else float("nan"),
            "recovery_lattice_valid_ratio": lattice_passed/lattice_checked if lattice_checked else float("nan"),
            "future_unrecoverable_reject_count": float(future_hard_rejected),
            "long_risk_score": float(selected.get("long_risk", 0.)), "long_risk_penalty": float(long_penalty),
            "committed_horizon_s": self.candidate_safety.committed_controls(self.horizon)*self.candidate_safety.geometry.dt,
            "scoring_unavailable_fallback": float(scoring_unavailable),
            "target_speed_mps": float(safety_context.target_speed) if getattr(safety_context, "target_speed", None) is not None else float("nan"),
            "actual_speed_mps": float(getattr(safety_context, "speed", float("nan"))),
        }
        for label, active in (("normal_search", mode == "normal"),
                              ("recovery_search", mode.startswith("recovery_")),
                              ("lattice_fallback", mode == "lattice_fallback"),
                              ("emergency_fallback", emergency_fallback)):
            part = (sequence[0]-recovery_reference[0] if label == "recovery_search" else delta) if active else torch.zeros_like(delta)
            metrics.update({f"{label}_delta": float(part.norm()), f"{label}_steer_delta": float(part[0]),
                            f"{label}_throttle_delta": float(part[1])})
        flags = safety_cache.get(selected_key, {})
        for name, field in (("future_hard", "unrecoverable"), ("exact_offroad", "offroad"),
                            ("exact_collision", "collision"), ("exact_kinematic", "kinematic_violation")):
            metrics[f"positive_throttle_{name}_rejected"] = float(sum(safety_cache[k].get("exact", False)
                and not safety_cache[k]["valid"] and safety_cache[k].get(field, False) for k in positive_keys))
        metrics["predicted_progress_m"] = float(flags.get("progress_estimate", float("nan")))
        metrics["preview_candidate_count"] = float(sum(not row.get("exact", False) for row in safety_cache.values()))
        metrics["exact_candidate_count"] = float(sum(row.get("exact", False) for row in safety_cache.values()))
        for name, count in positive_reasons.items():
            metrics[f"positive_throttle_near_{name}_rejected" if name != "kinematic" else "positive_throttle_kinematic_rejected"] = float(count)
        for name in ("geometric_steer", "geometric_lateral_error", "geometric_heading_error", "geometric_body_margin"):
            metrics[name] = float(getattr(safety_context, name, float("nan")) or 0.)
        metrics.update(near_collision=float(flags.get("near_collision", False)), near_offroad=float(flags.get("near_offroad", False)),
            future_hard=float(flags.get("unrecoverable", False)))
        # Efficiency is diagnostic-only until its distribution/ablation has
        # been reviewed; WM reward remains the trained progress objective.
        metrics["progress_reward"] = progress_distance/max(self.candidate_safety.geometry.dt*float(getattr(safety_context, "target_speed", None) or 1.), 1e-6)
        host_values = torch.stack([row["value"] for row in cache.values()]).detach().cpu() if cache else torch.empty(0)
        finite_values = host_values[torch.isfinite(host_values)]
        risks = [float(row["risk_score"]) for row in safety_cache.values() if math.isfinite(float(row.get("risk_score", float("nan"))))]
        metrics["value_p50"] = float(finite_values.median()) if len(finite_values) else float("nan")
        metrics["risk_p50"] = float(torch.tensor(risks).median()) if risks else 0.
        metrics["risk_p95"] = float(torch.tensor(risks).quantile(.95)) if risks else 0.
        metrics["selected_near_valid"] = float(flags.get("near_valid", flags.get("valid", float("nan"))))
        metrics["selected_full_valid"] = float(flags.get("full_valid", flags.get("valid", float("nan"))))
        baseline_flags = safety_cache.get(baseline_key, {})
        lattice_key = lattice_actions.detach().cpu().contiguous().numpy().tobytes() if lattice_actions is not None else None
        lattice_flags = safety_cache.get(lattice_key, {})
        for prefix, row in (("baseline", baseline_flags), ("lattice_baseline", lattice_flags)):
            metrics[f"{prefix}_near_valid"] = float(row.get("near_valid", row.get("valid", float("nan"))))
            metrics[f"{prefix}_full_valid"] = float(row.get("full_valid", row.get("valid", float("nan"))))
            metrics[f"{prefix}_future_hard_reject"] = float(row.get("unrecoverable", False))
            metrics[f"{prefix}_near_collision"] = float(row.get("near_collision", row.get("first_collision", False)))
            metrics[f"{prefix}_near_offroad"] = float(row.get("near_offroad", row.get("first_offroad", False)))
        for name in ("road", "collision", "clearance", "speed", "lateral", "kinematic"):
            metrics[f"long_{name}_risk"] = float(flags.get(f"{name}_risk", 0.))
        for target, source in (("predicted_collision", "collision"), ("predicted_offroad", "offroad"),
                               ("predicted_first_collision", "first_collision"), ("predicted_first_offroad", "first_offroad")):
            metrics[target] = float(flags.get(source, float("nan")))
        if self.safety_enabled:
            metrics.update(self.candidate_safety.stopping_diagnostics(safety_context, self.horizon))
        # A passing finite-horizon model check neither verifies a complete
        # stop nor guarantees safety in the real simulator/environment.
        metrics["emergency_horizon_limited"] = float(emergency_fallback and
            bool(metrics.get("stopping_horizon_insufficient", False)))
        self._emergency_streak = self._emergency_streak+1 if emergency_fallback else 0
        self._braking_streak = self._braking_streak+1 if float(sequence[0, 1]) < 0 else 0
        self._recovery_streak = self._recovery_streak+1 if mode.startswith("recovery_") else 0
        self._mode_switches += int(self._last_mode is not None and mode != self._last_mode)
        self._last_mode = mode
        metrics.update(emergency_streak=float(self._emergency_streak), braking_streak=float(self._braking_streak),
            recovery_streak=float(self._recovery_streak), mode_switch_count=float(self._mode_switches),
            emergency_reason=float((3 if recovery_best is not None and full_braking else
                                    1 if near_checked == 0 else 2) if emergency_fallback else 0))
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
