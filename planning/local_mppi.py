"""Single-Gaussian H+1 MPPI with independent pre-WM candidate safety."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from collections import deque
import math
import time
import warnings
import hashlib

import torch

from .kinematic_filter import BatchedKinematicCorridor, CorridorContext
from .timing import PhaseTimings
from .mppi_safety import MPPISafetyFilter


class InvalidCandidatePrediction(RuntimeError):
    """Invalid WM output: reject its score while retaining geometric fallbacks."""


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
    "predicted_route_progress", "predicted_terminal_speed", "forward_verified_candidates", "forward_candidate_selected",
    "progress_guard_active", "progress_rejected_candidates", "recovery_latched", "continuity_changed", "continuity_rejected",
    "center_model_active", "model_guard_active",
    "positive_sequence_candidate_count", "positive_sequence_verified_count",
    "safety_failure", "execution_authorized", "emergency_verified", "emergency_unverified", "safety_failure_streak",
    "brake_released", "longitudinal_mode", "lattice_speed_limit", "sac_target_speed", "mppi_target_speed",
    "baseline_future_recoverable", "recovery_forced_near", "recovery_forced_unrecoverable",
    "primary_calibrated", "legacy_shadow",
    "center_near_collision", "legacy_near_collision", "center_near_offroad", "legacy_near_offroad",
    *(f"motion_{model}_{metric}" for model in ("center", "legacy") for metric in
      ("position_error_m", "heading_error_rad", "lateral_error_m", "speed_error_mps", "collision_false_positive", "collision_false_negative", "offroad_false_positive", "offroad_false_negative")),
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
    candidate_diagnostics: list = field(default_factory=list)
    safety_failure: bool = False
    safety_certificate: dict = field(default_factory=dict)


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
        selection = dict(cfg.get("trajectory_selection", {}))
        self.trajectory_selection = bool(selection.get("enabled", False))
        self.record_candidates = bool(selection.get("record_candidates", False))
        self.min_progress = float(selection.get("min_progress_m", .05))
        self.progress_ratio = float(selection.get("recovery_progress_ratio", .8))
        self.normal_progress_ratio = float(selection.get("normal_progress_ratio", .9))
        self.throttle_step = float(selection.get("max_throttle_step", .25))
        self.brake_release_enabled = bool(selection.get("brake_release_enabled", True))
        self.brake_release_step = float(selection.get("brake_release_step", 1.))
        self.throttle_increase_step = float(selection.get("throttle_increase_step", self.throttle_step))
        self.throttle_decrease_step = float(selection.get("throttle_decrease_step", self.throttle_step))
        if any(not math.isfinite(v) or not 0 < v <= 2 for v in
               (self.brake_release_step, self.throttle_increase_step, self.throttle_decrease_step)):
            raise ValueError("Longitudinal command steps must be finite in (0,2].")
        if (not math.isfinite(self.min_progress) or self.min_progress <= 0 or
                not 0 < self.progress_ratio <= 1 or not 0 < self.normal_progress_ratio <= 1 or
                not math.isfinite(self.throttle_step) or not 0 < self.throttle_step <= 2):
            raise ValueError("Invalid trajectory progress/continuity settings.")
        if self.trajectory_selection and not self.safety_enabled:
            raise ValueError("Trajectory selection requires independent safety.")
        self.progressive_recovery = bool(recovery.get("progressive", False))
        self.positive_levels = tuple(float(v) for v in recovery.get("positive_throttles", [.1, .25, .4]))
        self.transition_fractions = tuple(float(v) for v in recovery.get("transition_fractions", [.4, .9]))
        self.controlled_deceleration = float(recovery.get("controlled_deceleration", -.3))
        if (len(self.positive_levels) != 3 or any(not math.isfinite(v) or not 0 < v <= 1 for v in self.positive_levels)
                or tuple(sorted(self.positive_levels)) != self.positive_levels
                or len(self.transition_fractions) != 2 or any(not math.isfinite(v) or v <= 0 for v in self.transition_fractions)
                or not -1 < self.controlled_deceleration < 0):
            raise ValueError("Progressive recovery needs 3 positive throttle levels, 2 transition times and controlled deceleration.")
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
        self._recovery_latched = False
        self._recovery_exit_count = 0
        self._last_selected_sequence = None
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
        self._safety_failure_streak = 0

    def reset(self, seed=None):
        self._prev_mean = None
        self.generator.manual_seed(self.eval_seed if seed is None else int(seed) % 2**32)
        self.candidate_safety.reset()
        self._emergency_streak = self._braking_streak = self._recovery_streak = self._mode_switches = 0
        self._last_mode = None
        self._safety_failure_streak = 0
        self._motion.clear()
        self._stall_active = False
        self._stall_clear = self._stall_count = 0
        self._last_position = self._last_progress = None
        self._last_measurement_step = None
        self._recovery_latched = False
        self._recovery_exit_count = 0
        self._last_selected_sequence = None

    def _continuous(self, actions, context):
        """Slew-limit a proposal; caller must exact-check the resulting sequence."""
        result = actions.clone()
        previous = result.new_tensor(context.previous_action)
        steer_step = float(self.candidate_safety.cfg.get("max_steering_rate", 7.))*self.candidate_safety.geometry.dt/context.max_steering
        for i in range(len(result)):
            steering = previous[0]+(result[i, 0]-previous[0]).clamp(-steer_step, steer_step)
            requested = result[i, 1]
            fast_release = (i == 0 and self.brake_release_enabled and
                (self._recovery_latched or self._stall_active) and context.speed < 2*self.stall_speed and
                getattr(context, "brake_action_mapping_verified", False) and previous[1] < 0 and requested >= 0)
            if fast_release:
                released = (previous[1]+self.brake_release_step).clamp_max(0.)
                longitudinal = torch.minimum(requested, released+self.throttle_increase_step) if released == 0 else released
            else:
                step = self.throttle_increase_step if requested >= previous[1] else self.throttle_decrease_step
                longitudinal = previous[1]+(requested-previous[1]).clamp(-step, step)
            result[i] = torch.stack((steering, longitudinal))
            previous = result[i].clone()
        return result.clamp(-1, 1)

    def _controlled_candidates(self, context, lattice, baseline):
        """At most 16 time-varying references, not another sampling algorithm."""
        previous = baseline.new_tensor(context.previous_action)
        horizon_s = self.horizon*self.candidate_safety.geometry.dt
        t = torch.arange(1, self.horizon+2, device=self.device)*self.candidate_safety.geometry.dt
        tangent = getattr(context, "road_tangent_steer", None)
        tangent = float(previous[0] if tangent is None else tangent)
        geometric = getattr(context, "geometric_steer", None)
        geometric = float(previous[0] if geometric is None else geometric)
        guide = lattice[:, 0] if lattice is not None else torch.full_like(t, tangent)
        def ramp(target, duration, amplitude=1.):
            ratio = (t/max(duration*horizon_s, self.candidate_safety.geometry.dt)).clamp(0, 1)
            blend = ratio.square()*(3-2*ratio)
            return previous[0]+amplitude*blend*(target-previous[0])
        fast, slow = self.transition_fractions
        steers = [torch.full_like(t, previous[0]), ramp(tangent, fast), ramp(tangent, slow),
            ramp(geometric, slow, .25), ramp(geometric, slow, .5), ramp(geometric, slow), ramp(guide, slow), ramp(geometric, fast)]
        target = getattr(context, "target_speed", None)
        cap = (max(0., (target-context.speed)/(horizon_s*float(self.candidate_safety.cfg.get("max_acceleration", 4.))))
               if target is not None and math.isfinite(target) else 0.)
        low, middle, high = [min(v, cap) for v in self.positive_levels]
        # Preserve all direction families without a Cartesian product explosion.
        specs = [(0, low, "hold"), (0, 0., "coast"), (0, self.controlled_deceleration, "decelerate"),
            (1, low, "tangent_fast"), (1, middle, "tangent_fast"), (1, high, "tangent_fast"),
            (7, middle, "center_full_fast"), (2, high, "tangent_slow"),
            (3, middle, "center_quarter"), (3, high, "center_quarter"),
            (4, low, "center_half"), (4, high, "center_half"), (5, middle, "center_full"),
            (6, middle, "lattice_ramp"), (6, high, "lattice_ramp")]
        choices, types = [], []
        for index, throttle, label in specs:
            proposal = torch.stack((steers[index], torch.full_like(t, throttle)), -1)
            choices.append(self._continuous(proposal, context))
            types.append("recovery_"+label)
        if self._recovery_latched and self._last_selected_sequence is not None:
            last = self._last_selected_sequence
            proposal = torch.cat((last[1:], last[-1:]), 0)
            label = "recovery_continuation"
        else:
            proposal = torch.stack((ramp(geometric, fast, .5), torch.full_like(t, middle)), -1)
            label = "recovery_center_half_fast"
        choices.append(self._continuous(proposal, context)); types.append(label)
        return torch.stack(choices), types

    def _progress_eligible(self, progress, safe, *, recovery, baseline_progress=float("nan"), recovery_target=None):
        """Dimensionless retention constraints; never a weighted safety tradeoff."""
        eligible = safe & torch.isfinite(progress)
        if not bool(eligible.any()):
            return eligible
        if recovery:
            maximum = float(progress[eligible].max())
            if maximum >= self.min_progress:
                target = self.progress_ratio*maximum
                if recovery_target is not None:
                    target = min(target, recovery_target)
                eligible &= progress >= max(self.min_progress, target)
        elif math.isfinite(baseline_progress) and baseline_progress >= self.min_progress:
            eligible &= progress >= self.normal_progress_ratio*baseline_progress
        return eligible

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
                raise InvalidCandidatePrediction("MPPI Cost is enabled but the World Model returned no predicted cost.")
            if cost is not None:
                bad_cost = ~torch.isfinite(cost) | (cost < 0)
                if bool(bad_cost.all()):
                    raise InvalidCandidatePrediction("MPPI received non-finite or negative cumulative WM Cost for every evaluated candidate.")
                cost = cost.masked_fill(bad_cost, float("nan"))
            for i, key in enumerate(missing):
                cache[key] = {
                    "value": (result.total_value[i] if cost is None or torch.isfinite(cost[i]) else actions.new_tensor(float("nan"))),
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
        candidate_data = {}
        candidate_kinds = {}
        candidate_aliases = {}
        decision_reasons = {}
        progress_active = self.trajectory_selection and getattr(safety_context, "road_reference", None) is not None
        previous_measurement = self._last_measurement_step
        progress_delta, progress_distance, recovery_success = self._observe_motion(safety_context)
        fresh_motion = getattr(safety_context, "measurement_step", None) is None or previous_measurement != safety_context.measurement_step
        seconds = self.horizon*self.candidate_safety.geometry.dt
        recovery_progress_target = (max(self.min_progress, min(float(getattr(safety_context, "target_speed", None) or 0.)*seconds,
            float(getattr(safety_context, "speed", 0.))*seconds+.5*float(self.candidate_safety.cfg.get("max_acceleration", 4.))*self.positive_levels[1]*seconds**2))
            if progress_active else None)
        positive_count = positive_rejected = positive_wm_rejected = 0
        positive_sequence_count = positive_sequence_verified = 0
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

        def check_actions(actions, reference, corridor_ctx, *, exact=False, kind="normal"):
            nonlocal checked, rejected, safety_rejected, collision_rejected, road_rejected, kinematic_rejected
            nonlocal safe_fallback, near_checked, future_hard_rejected
            with phase.measure("safety_check"):
                host = actions.detach().cpu().contiguous()
                keys = [row.tobytes() for row in host.numpy()]
                unique.update(keys)
                for i, key in enumerate(keys):
                    label = "sac_baseline" if key == baseline_key else kind[i] if isinstance(kind, list) else kind
                    candidate_kinds.setdefault(key, label)
                    aliases = candidate_aliases.setdefault(key, [])
                    if label not in aliases:
                        aliases.append(label)
                if self.record_candidates:
                    bounds = self.max_delta.detach().cpu()
                    deviation, smooth = self._penalties(host, baseline_host, bounds)
                    for i, key in enumerate(keys):
                        if key not in candidate_data:
                            candidate_data[key] = dict(candidate_id=f"c{len(candidate_data):04d}",
                                candidate_type="sac_baseline" if key == baseline_key else kind[i] if isinstance(kind, list) else kind,
                                first_action=host[i, 0].tolist(), steering_change=float(host[i, :, 0].diff().abs().sum()),
                                deviation_penalty=float(self.deviation_coef*deviation[i]), smoothness_penalty=float(self.smoothness_coef*smooth[i]))
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
                                     "clearance_risk", "speed_risk", "lateral_risk", "kinematic_risk", "unrecoverable", "progress_estimate",
                                     "route_progress_estimate", "terminal_speed", "terminal_lateral_error", "committed_lateral_error", "boundary_margin", "min_clearance",
                                     "center_model_valid", "legacy_model_valid", "primary_calibrated", "legacy_shadow",
                                     "center_near_collision", "legacy_near_collision", "center_near_offroad", "legacy_near_offroad"):
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
            try:
                rows, values, count = self._evaluate_cached(z_t, actions, cache, task, timing, keys)
            except InvalidCandidatePrediction:
                # An invalid value must never become a normal score, nor
                # bypass final exact verification of the geometric fallback.
                for key in keys:
                    if key in cache:
                        continue
                    cache[key] = {**{name: actions.new_tensor(float("nan")) for name in
                        ("value", "cost", "step_cost", "reward", "terminal_q")},
                        "validated": False, "ranking_validated": False}
                rows, values, count = self._evaluate_cached(z_t, actions, cache, task, timing, keys)
            for key, row in zip(keys, rows):
                row["long_risk"] = float(safety_cache.get(key, {}).get("risk_score", 0.))
            values["long_risk"] = actions.new_tensor([row["long_risk"] for row in rows])
            evaluated += count
            scored = score(actions, values, reference, bounds)
            timing["candidate_evaluation_ms"] = timing.get("candidate_evaluation_ms", 0.)+1000*(time.perf_counter()-started_evaluation)
            return rows, *scored, values

        def verified_choice(actions, keys, rows, scores, indices, *, recovery=False):
            """Promote candidates by priority; preview flags never authorize execution."""
            nonlocal exact_checked, exact_failed
            if progress_active:
                if recovery:
                    # Small recovery pools receive complete evidence before
                    # their progress tiers compete; preview-only cannot win.
                    was_exact = [safety_cache[k].get("exact", False) for k in keys]
                    valid, _, _ = check_actions(actions, baseline, None, exact=True)
                    exact_checked += sum(not v for v in was_exact)
                    exact_failed += sum(not old and not bool(v) for old, v in zip(was_exact, valid.cpu().tolist()))
                    adjusted = []
                    for key, row, score_value in zip(keys, rows, scores):
                        risk = float(safety_cache[key].get("risk_score", 0.))
                        adjusted.append(float(score_value)+self.long_risk_coef*(row.get("long_risk", 0.)-risk))
                        row["long_risk"] = risk
                    scores = actions.new_tensor(adjusted)
                progress = actions.new_tensor([safety_cache[k].get("route_progress_estimate", float("nan")) for k in keys])
                admissible = torch.tensor([safety_cache[k]["valid"] for k in keys], device=self.device)
                eligible = self._progress_eligible(progress, admissible & torch.isfinite(scores), recovery=recovery,
                    baseline_progress=safety_cache.get(baseline_key, {}).get("route_progress_estimate", float("nan")))
                if recovery:
                    # Full-safe and same-prefix-recoverable trajectories BOTH
                    # passed the required exact certificate. Preferring even a
                    # crawling full-safe tail here would recreate the deadlock.
                    feasible = admissible & torch.isfinite(scores)
                    eligible = self._progress_eligible(progress, feasible, recovery=True, recovery_target=recovery_progress_target)
                    # Progress is an adequacy constraint, not a contest for
                    # maximum predicted speed. Within that set prefer fewer
                    # future violating intervals, then use the shared WM score.
                    future_risk = actions.new_tensor([tail_risk(k) for k in keys])
                    if bool(eligible.any()):
                        eligible &= future_risk <= future_risk[eligible].min()+1/(self.horizon*self.candidate_safety.geometry.substeps)
                indices = eligible.nonzero().flatten()
                indices = indices[scores[indices].argsort(descending=True)]
                if recovery and self._recovery_latched and len(indices):
                    best_score = float(scores[indices[0]])
                    gate = max(self.min_improvement_abs, self.min_improvement_ratio*abs(best_score))
                    for i in indices.cpu().tolist():
                        if "recovery_continuation" in candidate_aliases.get(keys[i], ()) and best_score-float(scores[i]) <= gate:
                            indices = torch.cat((indices.new_tensor([i]), indices[indices != i]))
                            break
            if not self.candidate_safety.staged and not progress_active:
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

        def prefer(new, old):
            if new is None:
                return False
            if old is None:
                return True
            if progress_active:
                progress = baseline.new_tensor([safety_cache[c[3]].get("route_progress_estimate", float("nan")) for c in (old, new)])
                eligible = self._progress_eligible(progress, torch.ones(2, dtype=torch.bool, device=self.device), recovery=True,
                    recovery_target=recovery_progress_target if self._recovery_latched else None)
                if bool(eligible[1]) != bool(eligible[0]):
                    return bool(eligible[1])
                if self._recovery_latched and bool(eligible.all()):
                    old_risk, new_risk = tail_risk(old[3]), tail_risk(new[3])
                    interval = 1/(self.horizon*self.candidate_safety.geometry.substeps)
                    if abs(new_risk-old_risk) > interval:
                        return new_risk < old_risk
            return new[0] > old[0]

        def tail_risk(key):
            row = safety_cache[key]
            return sum(float(row.get(f"{name}_risk", 0.)) for name in ("road", "collision", "kinematic"))

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
            if prefer(choice, best):
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
        baseline_flags = safety_cache.get(baseline_key, {})
        near_failure = not bool(baseline_flags.get("near_valid", baseline_safe))
        unrecoverable_failure = bool(baseline_flags.get("unrecoverable", False)) or (not baseline_safe and not near_failure)
        if progress_active:
            if self._stall_active:
                self._recovery_latched = True
            if near_failure or unrecoverable_failure:
                self._recovery_latched, self._recovery_exit_count = True, 0
            elif self._recovery_latched and fresh_motion:
                actual_progress = progress_delta if math.isfinite(progress_delta) else progress_distance
                moving = safety_context.speed >= 2*self.stall_speed and actual_progress > 0
                self._recovery_exit_count = self._recovery_exit_count+1 if moving else 0
                if not self._stall_active and self._recovery_exit_count*self.candidate_safety.geometry.dt >= self.stall_exit:
                    self._recovery_latched = False
        if self.recovery_enabled and (not baseline_safe or best is None or self._stall_active or self._recovery_latched):
            recovery_used = True
            recovery_reason = ("low_speed_stall" if self._stall_active else
                "recovery_hysteresis" if self._recovery_latched and baseline_safe and best is not None else
                "unsafe_sac_baseline" if not baseline_safe else "no_usable_normal_candidate")
            start_checked, start_rejects, start_evaluated = checked, safety_rejected, evaluated
            with phase.measure("recovery_search"):
                lattice = lattice_actions.to(self.device).clamp(-1, 1) if lattice_actions is not None else None
                if lattice is not None:
                    valid, keys, _ = check_actions(lattice[None], lattice, lattice_corridor_context, exact=True, kind="lattice_baseline")
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
                        valid, keys, _ = check_actions(actions, lattice, lattice_corridor_context, kind="recovery_lattice")
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
                            choice = verified_choice(choices, candidate_keys, rows, common_scores, priority, recovery=True)
                            if prefer(choice, recovery_best):
                                recovery_best = choice
                    if recovery_best is not None:
                        recovery_reference, mode = lattice, "recovery_lattice"
                if self.progressive_recovery or not (lattice is not None and lattice_safe and not self._stall_active):
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
                    for fraction in (() if self.progressive_recovery else self.straighten_fractions):
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
                    if self.progressive_recovery:
                        choices, types = self._controlled_candidates(safety_context, lattice, baseline)
                    else:
                        choices, types = torch.stack(controlled).clamp(-1, 1), "recovery_controlled"
                    controlled_reference = choices[0].clone()
                    valid, keys, _ = check_actions(choices, baseline, None, exact=self.progressive_recovery, kind=types)
                    positive = choices[:, 0, 1] > 0  # preserve the existing first-action metric
                    positive_sequence = (choices[:, :self.horizon, 1] > 0).any(1)
                    positive_sequence_count = int(positive_sequence.sum())
                    positive_sequence_verified = int((positive_sequence & valid).sum()) if self.progressive_recovery else 0
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
                        reference = (controlled_reference if self.progressive_recovery else
                            self.candidate_safety.braking_sequence(safety_context,
                                horizon=self.horizon, device=self.device, dtype=baseline.dtype))
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
                            choice = verified_choice(choices, candidate_keys, rows, common_scores, priority, recovery=True)
                            if prefer(choice, recovery_best):
                                recovery_best = choice
                                recovery_reference, mode = reference, "recovery_controlled"
            recovery_checked, recovery_rejects = checked-start_checked, safety_rejected-start_rejects
            recovery_evaluated = evaluated-start_evaluated
        choose_recovery = prefer(recovery_best, best)
        winner = recovery_best if choose_recovery else best
        baseline_selected = baseline_safe and (winner is None or
            (baseline_finite and winner[0]-float(baseline_score) <= threshold))
        if progress_active and baseline_safe and winner is not None:
            base_progress = safety_cache[baseline_key].get("route_progress_estimate", float("nan"))
            winner_progress = safety_cache[winner[3]].get("route_progress_estimate", float("nan"))
            if winner_progress >= self.min_progress and base_progress < self.min_progress:
                baseline_selected = False
            elif base_progress >= self.min_progress and winner_progress < self.normal_progress_ratio*base_progress:
                baseline_selected = True
        scoring_unavailable = False
        if baseline_selected:
            sequence, selected, selected_key, mode = baseline, base, baseline_key, "baseline"
            scoring_unavailable = not baseline_finite
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
            if lattice_actions is not None and not progress_active:
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
                def fallback_priority(row):
                    progress = safety_cache[row[3]].get("route_progress_estimate", float("nan"))
                    return (-progress if math.isfinite(progress) else float("inf"), *row[:2]) if progress_active else row[:2]
                for _, _, fallback, key in sorted(fallback_candidates.values(), key=fallback_priority):
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
                valid, keys, _ = check_actions(sequence.unsqueeze(0), sequence, None, exact=True, kind="emergency_braking")
                emergency_fallback, unverified = True, not bool(valid[0])
                mode = "emergency_model_check_failed" if unverified else "emergency_model_checked"
                mean, std = sequence.clone(), torch.zeros_like(sequence)
                selected_key = keys[0]
                if not unverified:
                    rows, *_ = evaluate(sequence.unsqueeze(0), keys)
                    selected = rows[0]

        continuity_changed = continuity_rejected = False
        if progress_active and not emergency_fallback and not baseline_selected:
            adjusted = self._continuous(sequence, safety_context)
            if not torch.equal(adjusted, sequence):
                # The handoff is a new trajectory. It has no inherited safety
                # certificate or WM score until both are computed for it.
                valid, adjusted_keys, _ = check_actions(adjusted[None], baseline, None, exact=True, kind="continuity_adjusted")
                adjusted_progress = safety_cache[adjusted_keys[0]].get("route_progress_estimate", float("nan"))
                old_progress = safety_cache[selected_key].get("route_progress_estimate", float("nan"))
                retained_progress = old_progress < self.min_progress or adjusted_progress >= self.normal_progress_ratio*old_progress
                if bool(valid[0]) and retained_progress:
                    sequence, selected_key, continuity_changed = adjusted, adjusted_keys[0], True
                    rows, *_ = evaluate(adjusted[None], adjusted_keys)
                    selected = rows[0]
                else:
                    continuity_rejected = True  # retain the original exact-checked action
                    if bool(valid[0]) and not retained_progress:
                        decision_reasons[adjusted_keys[0]] = "continuity_progress_loss"
        if self.safety_enabled and not unverified:
            certificate = safety_cache[selected_key]
            if not certificate.get("exact", False) or not certificate["valid"]:
                raise RuntimeError("Selected MPPI action has no exact safety certificate.")
        sequence_key = sequence.detach().cpu().contiguous().numpy().tobytes()
        certificate_valid = bool(self.safety_enabled and not unverified and sequence_key == selected_key
            and safety_cache.get(selected_key, {}).get("exact", False) and safety_cache[selected_key]["valid"])
        safety_failure = self.safety_enabled and not certificate_valid
        self._safety_failure_streak = self._safety_failure_streak+1 if safety_failure else 0
        safety_certificate = dict(valid=certificate_valid, scope="finite_horizon_model_check",
            sequence_sha256=hashlib.sha256(sequence_key).hexdigest(),
            horizon_controls=self.horizon, action_executable=not safety_failure)
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
            "brake_released": float(safety_context is not None and float(safety_context.previous_action[1]) < 0 and float(sequence[0, 1]) >= 0),
            "longitudinal_mode": float(1 if safety_context is not None and float(safety_context.previous_action[1]) < 0 and float(sequence[0, 1]) >= 0
                else 0 if float(sequence[0, 1]) < 0 else 2 if safety_context is not None and float(sequence[0, 1]) > float(safety_context.previous_action[1]) else 3),
            "baseline_future_recoverable": float(baseline_safe and not baseline_flags.get("full_valid", baseline_safe)),
            "recovery_forced_near": float(near_failure), "recovery_forced_unrecoverable": float(unrecoverable_failure),
            **{name: float(getattr(safety_context, name, None) if getattr(safety_context, name, None) is not None else float("nan"))
                for name in ("lattice_speed_limit", "sac_target_speed", "mppi_target_speed")},
            "safety_failure": float(safety_failure), "execution_authorized": float(not safety_failure),
            "emergency_verified": float(emergency_fallback and certificate_valid),
            "emergency_unverified": float(emergency_fallback and not certificate_valid),
            "safety_failure_streak": float(self._safety_failure_streak),
            "progress_guard_active": float(progress_active), "recovery_latched": float(self._recovery_latched),
            "continuity_changed": float(continuity_changed), "continuity_rejected": float(continuity_rejected),
            "center_model_active": float(self.candidate_safety.center_bicycle and getattr(safety_context, "rear_axle_to_center", None) is not None),
            "model_guard_active": float(self.candidate_safety.center_bicycle and self.candidate_safety.center_model_guard
                and getattr(safety_context, "rear_axle_to_center", None) is not None),
            "exact_checked_candidates": float(exact_checked), "exact_failed_candidates": float(exact_failed),
            "selected_exact_verified": float(certificate_valid),
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
            "positive_sequence_candidate_count": float(positive_sequence_count),
            "positive_sequence_verified_count": float(positive_sequence_verified),
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
            "selected_safety_valid": float(certificate_valid) if self.safety_enabled else float("nan"),
            "normal_safety_valid_ratio": 1-normal_rejects/max(normal_checked, 1) if self.safety_enabled else float("nan"),
            "recovery_safety_valid_ratio": 1-recovery_rejects/max(recovery_checked, 1) if recovery_checked else float("nan"),
            "lattice_baseline_safety_valid": float(lattice_safe),
            "lattice_baseline_checked": float(math.isfinite(lattice_safe)),
            "recovery_search": float(recovery_used), "recovery_wm_evaluated_candidates": float(recovery_evaluated),
            "planning_mode": float({"baseline": 0, "normal": 1, "recovery_lattice": 2, "recovery_controlled": 3,
                                     "lattice_fallback": 4, "emergency_model_checked": 5, "emergency_model_check_failed": 6}[mode]),
            "recovery_reason": float({"none": 0, "unsafe_sac_baseline": 1, "no_usable_normal_candidate": 2, "low_speed_stall": 3, "recovery_hysteresis": 4, "future_sac_risk": 5}[recovery_reason]),
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
        for name in ("primary_calibrated", "legacy_shadow", "center_near_collision", "legacy_near_collision", "center_near_offroad", "legacy_near_offroad"):
            metrics[name] = float(flags.get(name, float("nan")))
        metrics["predicted_route_progress"] = float(flags.get("route_progress_estimate", float("nan")))
        metrics["predicted_terminal_speed"] = float(flags.get("terminal_speed", float("nan")))
        metrics["forward_verified_candidates"] = float(sum(row.get("exact", False) and row["valid"]
            and row.get("route_progress_estimate", float("nan")) >= self.min_progress for row in safety_cache.values()))
        metrics["forward_candidate_selected"] = float(not unverified and not emergency_fallback and
            flags.get("route_progress_estimate", float("nan")) >= self.min_progress)
        metrics["progress_rejected_candidates"] = float(sum(row.get("exact", False) and row["valid"]
            and row.get("route_progress_estimate", float("nan")) < self.min_progress for row in safety_cache.values()))
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
        self._last_selected_sequence = sequence.detach().clone() if not emergency_fallback and not unverified else None
        candidate_records = []
        if self.record_candidates:
            def number(value):
                value = float(value)
                return value if math.isfinite(value) else None
            selected_progress = flags.get("route_progress_estimate", float("nan"))
            wm_names = ("value", "cost", "reward", "terminal_q")
            wm_keys = list(cache)
            packed_wm = (torch.stack([torch.stack([cache[k][name] for name in wm_names]) for k in wm_keys]).detach().cpu().tolist()
                         if wm_keys else [])
            host_wm = {key: {**cache[key], **dict(zip(wm_names, row))} for key, row in zip(wm_keys, packed_wm)}
            selected_value = float(selected["value"])
            selected_deviation = float(self.deviation_coef*deviation[0])
            for key, item in candidate_data.items():
                state, wm = safety_cache.get(key, {}), host_wm.get(key, {})
                exact = bool(state.get("exact", False))
                valid = bool(state.get("valid", False))
                progress = state.get("route_progress_estimate", float("nan"))
                cost_scored_here = enabled and (wm.get("ranking_validated", False) or self.allow_unvalidated_cost)
                value = float(wm.get("value", float("nan")))
                cost = self.cost_coef*float(wm.get("cost", 0.)) if cost_scored_here else 0.
                risk = float(state.get("risk_score", 0.))
                final = value-cost-item["deviation_penalty"]-item["smoothness_penalty"]-self.long_risk_coef*risk
                if key == selected_key:
                    reason = "selected_unverified_emergency" if unverified else "selected"
                elif not state.get("near_valid", valid):
                    reason = ("near_collision" if state.get("near_collision", state.get("collision", False)) else
                        "near_offroad" if state.get("near_offroad", state.get("offroad", False)) else "near_kinematic")
                elif exact and not valid:
                    reason = ("future_recoverability_failed_offroad" if state.get("offroad", False) else
                        "future_recoverability_failed_collision" if state.get("collision", False) else "future_recoverability_failed_kinematic")
                elif not exact:
                    reason = "not_finally_verified"
                elif key in decision_reasons:
                    reason = decision_reasons[key]
                elif progress_active and selected_progress >= self.min_progress and progress < max(self.min_progress,
                        min(self.progress_ratio*selected_progress, recovery_progress_target) if self._recovery_latched else self.progress_ratio*selected_progress):
                    reason = "insufficient_progress"
                elif progress_active and self._recovery_latched and tail_risk(key) > tail_risk(selected_key)+1/(self.horizon*self.candidate_safety.geometry.substeps):
                    reason = "future_risk_tier"
                elif not math.isfinite(final):
                    reason = "wm_score_unavailable"
                elif baseline_selected:
                    reason = "sac_baseline_retained"
                elif value > selected_value and item["deviation_penalty"] > selected_deviation:
                    reason = "deviation_penalty"
                else:
                    reason = "wm_score_lower" if value < selected_value else "trajectory_quality_score_lower"
                candidate_records.append(dict(**item, exact_verified=exact and valid,
                    candidate_types=candidate_aliases.get(key, [item["candidate_type"]]),
                    center_model_valid=bool(state["center_model_valid"]) if "center_model_valid" in state else None,
                    legacy_model_valid=bool(state["legacy_model_valid"]) if "legacy_model_valid" in state else None,
                    near_safety_valid=bool(state.get("near_valid", valid)),
                    full_safety_valid=bool(state.get("full_valid", valid)) if exact else None,
                    future_recoverable=valid if exact else None,
                    predicted_route_progress=number(progress), predicted_terminal_speed=number(state.get("terminal_speed", float("nan"))),
                    terminal_lateral_error=number(state.get("terminal_lateral_error", float("nan"))),
                    committed_lateral_error=number(state.get("committed_lateral_error", float("nan"))),
                    min_boundary_margin=number(state.get("boundary_margin", float("nan"))),
                    min_obstacle_clearance=number(state.get("min_clearance", float("nan"))),
                    boundary_margin_basis="body_samples" if exact else "body_hull_preview",
                    obstacle_clearance_basis="bounding_circle_lower_bound",
                    predicted_collision=bool(state.get("collision", False)), predicted_offroad=bool(state.get("offroad", False)),
                    predicted_reward=number(wm.get("reward", float("nan"))), terminal_q=number(wm.get("terminal_q", float("nan"))),
                    wm_value=number(value), long_risk=number(risk), cost_penalty=number(cost), final_score=number(final),
                    score_basis="exact" if exact else "preview", rejection_reason=reason, selected=key == selected_key, selected_rank=None))
            ranked = sorted([r for r in candidate_records if r["exact_verified"] and r["final_score"] is not None],
                key=lambda r: (not r["selected"], r["rejection_reason"] == "insufficient_progress", -r["final_score"]))
            for rank, row in enumerate(ranked, 1):
                row["selected_rank"] = rank
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
            candidate_diagnostics=candidate_records,
            safety_failure=safety_failure, safety_certificate=safety_certificate,
        )


__all__ = ["LocalMPPI", "LocalMPPIResult", "_action_vector", "MPPI_DIAGNOSTIC_KEYS"]
