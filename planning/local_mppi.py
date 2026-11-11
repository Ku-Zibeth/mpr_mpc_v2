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
    "boundary_trend_delta", "boundary_decline_mps", "active_boundary_recovery", "continuation_effective", "recovery_scoring_active",
    "recovery_progress_unavailable",
    "decision_state", "safety_override", "model_uncertainty", "baseline_model_disagreement",
    "baseline_score", "selected_score", "improvement", "performance_modification", "safety_modification",
    "sac_first_steer", "sac_first_throttle", "selected_first_steer", "selected_first_throttle",
    "value_ranking_validated", "elite_certified_count", "elite_update_skipped", "traffic_waiting",
    "selection_reason", "search_skipped_unvalidated_value",
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
        self._route_history = deque(maxlen=self._motion.maxlen)
        self._stall_active = False
        self._stall_clear = 0
        self._stall_count = 0
        self._last_position = self._last_progress = None
        self._last_measurement_step = None
        self._recovery_latched = False
        self._recovery_exit_count = 0
        self._last_selected_sequence = None
        self._boundary_history = deque(maxlen=3)
        self._boundary_active = False
        self._continuation_effective = False
        self._last_recovery_observation = None
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
        self._route_history.clear()
        self._stall_active = False
        self._stall_clear = self._stall_count = 0
        self._last_position = self._last_progress = None
        self._last_measurement_step = None
        self._recovery_latched = False
        self._recovery_exit_count = 0
        self._last_selected_sequence = None
        self._boundary_history.clear()
        self._boundary_active = self._continuation_effective = False
        self._last_recovery_observation = None

    def _observe_boundary(self, context, distance, fresh, route_delta=float("nan")):
        """Measured trend only. No predicted tail risk can alone trigger this."""
        self._continuation_effective = False
        if context is None or not fresh:
            return {}
        if math.isfinite(route_delta) and route_delta <= 0:
            distance = 0.  # sideways motion is not forward route progress
        margin = float(getattr(context, "geometric_body_margin", float("nan")))
        lateral = abs(float(getattr(context, "geometric_lateral_error", 0.))-
            min(getattr(context, "lane_centers", ()) or (0.,), key=lambda d: abs(d-getattr(context, "geometric_lateral_error", 0.))))
        lane_width = float(getattr(context, "lane_width", None) or context.width)
        tolerance = float(self.candidate_safety.geometry.swept_position_tolerance)
        road_token = (id(context.road), round(lane_width, 4))
        old = self._last_recovery_observation
        if old is not None and old["step"] is not None:
            self._continuation_effective = bool(old["verified"] and
                old.get("road_token", road_token) == road_token and
                getattr(context, "measurement_step", None) == old["step"]+1 and
                torch.allclose(torch.as_tensor(context.previous_action).float(), old["action"].cpu(), atol=1e-6, rtol=0) and
                margin >= old["margin"]-tolerance and lateral <= old["lateral"]+tolerance and
                (margin > old["margin"]+tolerance or lateral < old["lateral"]-tolerance) and
                distance >= self.min_progress*self.candidate_safety.geometry.dt/(self.horizon*self.candidate_safety.geometry.dt))
        if not math.isfinite(margin):
            return {}
        if self._boundary_history and self._boundary_history[-1][3] != road_token:
            self._boundary_history.clear()  # road scope / width change is not vehicle motion
        self._boundary_history.append((margin, lateral, distance, road_token))
        delta = self._boundary_history[-1][0]-self._boundary_history[0][0]
        duration = max(1, len(self._boundary_history)-1)*self.candidate_safety.geometry.dt
        free_half_width = max(tolerance, (lane_width-context.width)/2)
        if len(self._boundary_history) == self._boundary_history.maxlen:
            margins = [r[0] for r in self._boundary_history]
            inward = lateral < self._boundary_history[0][1]-tolerance
            declining = all(a-b > tolerance for a, b in zip(margins, margins[1:]))
            insufficient = sum(r[2] for r in list(self._boundary_history)[1:]) < self.stall_speed*duration
            if margin < free_half_width and lateral > free_half_width and declining and insufficient and not inward:
                self._boundary_active = True
            elif self._boundary_active and delta > tolerance and inward and not self._recovery_latched:
                self._boundary_active = False
        return dict(boundary_trend_delta=delta, boundary_decline_mps=delta/duration,
            active_boundary_recovery=float(self._boundary_active), continuation_effective=float(self._continuation_effective))

    def _recovery_penalties(self, actions, baseline, reference, context):
        """Bounded action quality, shared across ALL recovery candidate families."""
        squared = ((actions-reference)/self.recovery_delta).square()
        to_reference = (squared/(1+squared)).mean((1, 2))
        sac_squared = ((actions-baseline)/self.max_delta).square()
        to_sac = (sac_squared/(1+sac_squared)).mean((1, 2))
        smoothness = actions.diff(dim=1).square().sum(-1).mean(1)
        previous = actions.new_tensor(context.previous_action)
        step = actions.new_tensor([float(self.candidate_safety.cfg.get("max_steering_rate", 7.))*self.candidate_safety.geometry.dt/context.max_steering,
            self.throttle_increase_step])
        first = ((actions[:, 0]-previous)/step).square()
        continuity = (first/(1+first)).mean(1)
        return to_reference, to_sac, smoothness, continuity

    def _recovery_order(self, keys, scores, states, context):
        """Hard safety/progress precede epsilon-Pareto recovery quality, then WM.

        No geometric objective can admit an unsafe action. Near-equal changes
        below the existing body model margin do not create a privileged tier.
        """
        width = max(float(getattr(context, "lane_width", None) or context.width), context.width)
        scale = max(context.width, width-context.width)
        epsilon = max(self.candidate_safety.geometry.model_margin,
            self.candidate_safety.geometry.swept_position_tolerance)/scale
        def value(k, name, default):
            x = float(states[k].get(name, default))
            return x if math.isfinite(x) else default
        objectives = []
        for key in keys:
            row = states[key]
            clearance = value(key, "min_clearance", 1e6)
            objectives.append([
                value(key, "terminal_lateral_error", 0.)/scale,
                value(key, "committed_lateral_error", 0.)/scale,
                -value(key, "near_boundary_margin", 0.)/scale,
                -value(key, "terminal_boundary_margin", 0.)/scale,
                -value(key, "boundary_margin_change", 0.)/scale,
                1/(1+max(0., clearance)/context.width),
                sum(float(row.get(f"{n}_risk", 0.)) for n in ("road", "collision", "kinematic")),
                -value(key, "boundary_margin", 0.)/scale])
        matrix = torch.tensor(objectives, dtype=torch.float64)
        score_values = scores.detach().cpu().tolist()
        # Epsilon dominance: at least one material improvement, no material
        # deterioration. Nondominated choices keep WM/quality ordering.
        primary = matrix[:, [0, 3]]  # terminal lateral error and terminal body margin
        no_worse = (primary[:, None] <= primary[None]+epsilon).all(-1)
        better = (primary[:, None] < primary[None]-epsilon).any(-1)
        dominates = no_worse & better
        remaining = list(range(len(keys))); ordered = []; fronts = {}
        while remaining:
            ids = torch.tensor(remaining, dtype=torch.long)
            front = ids[~dominates[ids][:, ids].any(0)].tolist()
            if not front: front = remaining[:]  # epsilon ties can form cycles
            for i in front: fronts[keys[i]] = len(ordered)
            secondary = matrix[front][:, [1, 2, 4, 5, 6, 7]]
            dominance = ((secondary[:, None] <= secondary[None]+epsilon).all(-1)
                & (secondary[:, None] < secondary[None]-epsilon).any(-1))
            counts = dominance.sum(0).tolist()
            ordered.extend(sorted(front, key=lambda i: (counts[front.index(i)], -score_values[i])))
            remaining = [i for i in remaining if i not in front]
        return ordered, fronts

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
        # Replace two slow inward families with bounded inward-then-tangent
        # shapes. Persistent full lock can keep swinging the rear body outward.
        # Reuse the existing transition times and the SAME 16-slot budget.
        return_ratio = ((t-fast*horizon_s)/max((slow-fast)*horizon_s, self.candidate_safety.geometry.dt)).clamp(0, 1)
        blend_back = return_ratio.square()*(3-2*return_ratio)
        for index, amplitude in ((3, .25), (4, .5)):
            inward = ramp(geometric, fast, amplitude)
            steers[index] = inward+blend_back*(tangent-inward)
        target = getattr(context, "target_speed", None)
        cap = (max(0., (target-context.speed)/(horizon_s*float(self.candidate_safety.cfg.get("max_acceleration", 4.))))
               if target is not None and math.isfinite(target) else 0.)
        low, middle, high = [min(v, cap) for v in self.positive_levels]
        # Preserve all direction families without a Cartesian product explosion.
        specs = [(0, low, "hold"), (0, 0., "coast"), (0, self.controlled_deceleration, "decelerate"),
            (1, low, "tangent_fast"), (1, 0., "tangent_coast"), (1, high, "tangent_fast"),
            (7, middle, "center_full_fast"), (2, high, "tangent_slow"),
            (3, middle, "center_quarter"), (3, self.controlled_deceleration, "center_quarter_decelerate"),
            (4, low, "center_half"), (4, 0., "center_half_coast"), (5, middle, "center_full"),
            (6, middle, "lattice_ramp"), (6, high, "lattice_ramp")]
        choices, types = [], []
        for index, throttle, label in specs:
            proposal = torch.stack((steers[index], torch.full_like(t, throttle)), -1)
            choices.append(self._continuous(proposal, context))
            types.append("recovery_"+label)
        if self._recovery_latched and self._continuation_effective and self._last_selected_sequence is not None:
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
                    # Turning recovery need only retain the SAME configured
                    # progress fraction of a straight-line reachable estimate.
                    target = min(target, self.progress_ratio*recovery_target)
                eligible &= progress >= max(self.min_progress, target)
            else:
                eligible &= progress >= self.min_progress  # stationary plans are fallbacks, not successful recovery
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
        self._route_history.append(float(progress) if progress is not None else float("nan"))
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

    def _trapped_state(self, context, progress_delta):
        """Low speed alone (including traffic waiting) is never a takeover cause."""
        if context is None:
            return False, False
        import numpy as np
        heading = np.array([math.cos(context.heading), math.sin(context.heading)])
        waiting = False
        for obstacle in context.obstacles:
            offset = np.asarray(obstacle.position)[:2]-np.asarray(context.position)[:2]
            along = float(offset@heading)
            lateral = abs(float(offset@np.array([-heading[1], heading[0]])))
            # Existing model clearance + braking distance, not a relaxed collision threshold.
            extent = .5*(context.length+float(obstacle.length))
            distance = max(context.speed*self.stall_duration,
                           context.speed**2/(2*float(self.candidate_safety.cfg.get("max_deceleration", 6.))))
            waiting |= (0 < along < extent+distance+context.length and
                        lateral < .5*(context.width+float(obstacle.width)))
        margin = float(getattr(context, "geometric_body_margin", float("nan")))
        lateral = float(getattr(context, "geometric_lateral_error", float("nan")))
        centres = getattr(context, "lane_centers", ()) or ()
        width = float(getattr(context, "lane_width", 0.) or 0.)
        boundary_trapped = (math.isfinite(margin) and margin <= context.width/2 and
            math.isfinite(lateral) and width > 0 and bool(centres) and
            min(abs(lateral-c) for c in centres) > (width-context.width)/2)
        known_progress = (getattr(context, "route_progress", None) is not None and
                          len(self._route_history) == self._route_history.maxlen and
                          all(math.isfinite(v) for v in self._route_history) and
                          self._route_history[-1] <= self._route_history[0] and
                          math.isfinite(progress_delta) and progress_delta <= 0)
        return bool(self._stall_active and known_progress and boundary_trapped and not waiting), bool(waiting)

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
        boundary_diagnostics = self._observe_boundary(safety_context, progress_distance, fresh_motion, progress_delta) if progress_active else {}
        true_stuck, traffic_waiting = self._trapped_state(safety_context, progress_delta)
        # Prediction-tail risk, unavailable projection and previous search failure
        # do not latch an independent driving strategy.
        self._recovery_latched = False
        recovery_scoring = False
        active_recovery_reference = baseline
        recovery_reference_kind = "none"
        recovery_fronts = {}
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
                                deviation_penalty=float(self.deviation_coef*deviation[i]), smoothness_penalty=float(self.smoothness_coef*smooth[i]),
                                deviation_to_sac=float(deviation[i]), deviation_to_recovery_reference=None,
                                smoothness=float(smooth[i]), first_action_continuity=None, recovery_reference="none")
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
                                     "center_near_collision", "legacy_near_collision", "center_near_offroad", "legacy_near_offroad",
                                     "near_boundary_margin", "terminal_boundary_margin", "boundary_margin_change", "committed_margin_change",
                                     "center_first_failure_substep", "legacy_first_failure_substep", "braking_continuation_valid"):
                            value = getattr(result, name, None)
                            if value is not None:
                                columns[name] = value.tolist()
                        for model in ("center", "legacy"):
                            for field in ("collision", "offroad", "kinematic_violation", "braking_continuation_valid"):
                                value = getattr(result, model+"_"+field, None)
                                if value is not None: columns[model+"_"+field] = value.tolist()
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
            deviation, smoothness = self._penalties(actions, baseline)
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

        def relative_progress_ok(key):
            if not progress_active:
                return True
            base_progress = float(safety_cache.get(baseline_key, {}).get("route_progress_estimate", float("nan")))
            progress = float(safety_cache.get(key, {}).get("route_progress_estimate", float("nan")))
            # Unknown projections are not zero progress and do not trigger recovery.
            if not math.isfinite(base_progress) or not math.isfinite(progress):
                return True
            return progress+1e-6 >= base_progress-(1-self.normal_progress_ratio)*abs(base_progress)

        def certified_shortlist(actions, keys, rows, scores, *, intervention=False):
            """Bounded Top-K; exact rejection never supplies an Elite sample."""
            nonlocal exact_checked, exact_failed
            ids = torch.isfinite(scores).nonzero().flatten().cpu().tolist()
            ids.sort(key=lambda i: float(scores[i]), reverse=True)
            shortlist = ids[:min(len(ids), 2*self.num_elites)]
            certified = []
            if not shortlist:
                return certified
            with phase.measure("final_safety_verification"):
                was_exact = [safety_cache.get(keys[i], {}).get("exact", not self.safety_enabled) for i in shortlist]
                valid, _, _ = check_actions(actions[shortlist], baseline, None, exact=True)
                exact_checked += sum(not old for old in was_exact)
                exact_failed += sum(not old and not ok for old, ok in zip(was_exact, valid.cpu().tolist()))
            for i, ok in zip(shortlist, valid.cpu().tolist()):
                key = keys[i]
                progress_loss = baseline_safe and not intervention and not relative_progress_ok(key)
                if not ok or progress_loss:
                    if ok: decision_reasons[key] = "relative_progress_loss"
                    continue
                old_risk = rows[i].get("long_risk", 0.)
                risk = float(safety_cache.get(key, {}).get("risk_score", 0.))
                scores[i] += self.long_risk_coef*(old_risk-risk)
                rows[i]["long_risk"] = risk
                if math.isfinite(float(scores[i])):
                    certified.append(i)
            certified.sort(key=lambda i: float(scores[i]), reverse=True)
            return certified

        def prefer(new, old):
            return new is not None and (old is None or new[0] > old[0])

        def tail_risk(key):
            return float(safety_cache[key].get("risk_score", 0.))

        baseline_key = baseline.detach().cpu().numpy().tobytes()
        base_valid, _, _ = check_actions(baseline[None], baseline, corridor_context, exact=True)
        baseline_safe = bool(base_valid[0])
        baseline_flags = safety_cache.get(baseline_key, {})
        baseline_near_safe = bool(baseline_flags.get("near_valid", baseline_safe))
        near_failure = not baseline_near_safe
        baseline_disagreement = baseline_flags.get("center_model_valid", baseline_safe) != baseline_flags.get("legacy_model_valid", baseline_safe)
        near_disagreement = any(baseline_flags.get("center_near_"+k, False) != baseline_flags.get("legacy_near_"+k, False)
                                for k in ("collision", "offroad"))
        # Preserve conservative certification, but do not activate geometric recovery
        # merely because uncalibrated prediction hypotheses disagree.
        intervention_required = near_failure and not near_disagreement
        decision_state = "TRUE_STUCK_RECOVERY" if true_stuck else "SAFETY_INTERVENTION" if intervention_required else "NORMAL_REFINEMENT"
        elite_certified_count = elite_update_skipped = 0
        value_ranking_ready = bool(getattr(self.evaluator, "value_ranking_ready", lambda: True)())
        normal_checked_start = checked
        # No performance proposal is admissible while Value ranking is unqualified
        # and SAC already has a complete execution certificate. Return that exact
        # policy rather than sample proposals we are forbidden to accept. Active
        # search keeps the configured 64/8/4 budget; safety intervention never
        # takes this shortcut. Measured true-stuck recovery remains below.
        skip_unqualified_search = baseline_safe and not value_ranking_ready and not true_stuck
        for _ in range(0 if skip_unqualified_search else self.iterations):
            with phase.measure("mppi_sampling"):
                actions = self._candidate_batch(mean, std, baseline, generator)
            valid, keys, corridor = check_actions(actions, baseline, corridor_context)
            baseline_corridor_valid = bool(corridor.valid[0]) if corridor is not None else True
            original = valid.nonzero().flatten()
            if not len(original):
                elite_update_skipped += 1
                continue
            candidate_actions = actions[original]
            candidate_keys = [keys[i] for i in original.cpu().tolist()]
            rows, scores, deviation, smoothness, cost_penalty, _ = evaluate(candidate_actions, candidate_keys)
            finite = torch.isfinite(scores).nonzero().flatten()
            if not len(finite):
                continue
            # Without ranking evidence no performance candidate can be accepted
            # over an available baseline. Do not exact-check unused tail proposals;
            # leave the distribution unchanged when certified elites are insufficient.
            certified = (certified_shortlist(candidate_actions, candidate_keys, rows, scores,
                                             intervention=near_failure)
                         if value_ranking_ready or not baseline_safe else [])
            elite_certified_count += len(certified)
            if len(certified) >= self.num_elites:
                with phase.measure("elite_update"):
                    indices = torch.tensor(certified[:self.num_elites], device=self.device)
                    elite_values, elites = scores[indices], candidate_actions[indices]
                    weights = (self.temperature*(elite_values-elite_values.max())).exp()
                    weights /= weights.sum().clamp_min(1e-9)
                    mean = self._project((weights[:, None, None]*elites).sum(0), baseline)
                    variance = (weights[:, None, None]*(elites-mean).square()).sum(0)
                    std = variance.sqrt().clamp(self.min_std_vector, self.max_std_vector)
            else:
                elite_update_skipped += 1
            if certified:
                i = certified[0]
                choice = (float(scores[i]), candidate_actions[i].clone(), rows[i], candidate_keys[i])
                if prefer(choice, best): best = choice

        normal_checked, normal_rejects = checked-normal_checked_start, safety_rejected
        normal_near_checked = near_checked
        if baseline_key not in cache and baseline_safe:
            evaluate(baseline[None], [baseline_key])
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
        if self.recovery_enabled and best is None and (intervention_required or true_stuck):
            recovery_used = True
            recovery_reason = "low_speed_stall" if true_stuck else "unsafe_sac_baseline"
            start_checked, start_rejects, start_evaluated = checked, safety_rejected, evaluated
            with phase.measure("recovery_search"):
                lattice = lattice_actions.to(self.device).clamp(-1, 1) if lattice_actions is not None else None
                if true_stuck:
                    choices, kinds = self._controlled_candidates(safety_context, lattice, baseline)
                else:
                    # Minimal bounded SAC corrections come first, not an independent
                    # Lattice Gaussian or geometric controller.
                    choices = torch.stack([baseline.clone() for _ in range(4)])
                    choices[0, :, 1] -= self.max_delta[1]
                    choices[1, :, 0] -= self.max_delta[0]
                    choices[2, :, 0] += self.max_delta[0]
                    choices[3, :, 1] -= self.max_delta[1]
                    choices[3, :, 0] = baseline[:, 0]*.9
                    choices = self._project(choices, baseline)
                    kinds = ["safety_override_local"]*len(choices)
                valid, keys, _ = check_actions(choices, baseline, None, exact=True, kind=kinds)
                positive_keys = [k for k, a in zip(keys, choices) if a[0, 1] > 0]
                positive_count = len(positive_keys)
                positive_rejected = sum(not safety_cache[k]["valid"] for k in positive_keys)
                ids = valid.nonzero().flatten()
                if len(ids):
                    choices = choices[ids]
                    candidate_keys = [keys[i] for i in ids.cpu().tolist()]
                    rows, scores, *_ = evaluate(choices, candidate_keys)
                    finite = torch.isfinite(scores).nonzero().flatten().cpu().tolist()
                    if finite:
                        # Same J for all families; safety is a feasibility constraint.
                        i = max(finite, key=lambda i: float(scores[i]))
                        recovery_best = (float(scores[i]), choices[i].clone(), rows[i], candidate_keys[i])
                        # This deterministic control is its own reference, not
                        # a Gaussian refinement of SAC. Keep its switch separate
                        # from an MPPI optimisation delta (SAC delta is retained).
                        recovery_reference, mode = choices[i].clone(), "recovery_controlled"
            recovery_checked, recovery_rejects = checked-start_checked, safety_rejected-start_rejects
            recovery_evaluated = evaluated-start_evaluated
        choose_recovery = prefer(recovery_best, best)
        winner = recovery_best if choose_recovery else best
        baseline_selected = baseline_safe and (winner is None or
            (baseline_finite and winner[0]-float(baseline_score) <= threshold))
        # No trustworthy score means no performance override of an available SAC.
        if baseline_safe and not baseline_finite:
            baseline_selected = True
        if baseline_safe and winner is not None and not near_failure:
            if not relative_progress_ok(winner[3]): baseline_selected = True
        if baseline_safe and winner is not None and decision_state == "NORMAL_REFINEMENT":
            in_trust = bool(((winner[1]-baseline).abs() <= self.max_delta+1e-7).all())
            if not in_trust or not value_ranking_ready:
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
            if sequence is None and safe_fallback is not None:
                # WM scores can be unavailable while geometric candidates are
                # feasible. This is not an emergency-braking condition.
                def fallback_priority(row):
                    return float((row[2]-baseline_host).square().mean()), *row[:2]
                for _, _, fallback, key in sorted(fallback_candidates.values(), key=fallback_priority):
                    fallback = fallback.to(self.device)
                    valid, _, _ = check_actions(fallback[None], baseline, None, exact=True)
                    if bool(valid[0]):
                        sequence, selected_key = fallback, key
                        selected = cache.get(key, blank())
                        mode, scoring_unavailable = ("recovery_controlled" if recovery_scoring else "normal"), True
                        break
            if sequence is None and lattice_actions is not None and intervention_required:
                lattice = lattice_actions.to(self.device).clamp(-1, 1)
                valid, keys, _ = check_actions(lattice.unsqueeze(0), lattice, lattice_corridor_context, exact=True)
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
                valid, keys, _ = check_actions(sequence.unsqueeze(0), sequence, None, exact=True, kind="emergency_braking")
                emergency_fallback, unverified = True, not bool(valid[0])
                mode = "emergency_model_check_failed" if unverified else "emergency_model_checked"
                mean, std = sequence.clone(), torch.zeros_like(sequence)
                selected_key = keys[0]
                if not unverified:
                    rows, *_ = evaluate(sequence.unsqueeze(0), keys)
                    selected = rows[0]

        continuity_changed = continuity_rejected = False
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
        if recovery_scoring:
            primary, secondary, smoothness, continuity = self._recovery_penalties(sequence[None], baseline, active_recovery_reference, safety_context)
            deviation, smoothness = primary+secondary, smoothness+continuity
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
            "value_ranking_validated": float(value_ranking_ready),
            "search_skipped_unvalidated_value": float(skip_unqualified_search),
            "decision_state": float({"NORMAL_REFINEMENT": 0, "SAFETY_INTERVENTION": 1, "TRUE_STUCK_RECOVERY": 2}[decision_state]),
            "safety_override": float(intervention_required and not baseline_selected),
            "traffic_waiting": float(traffic_waiting),
            "elite_certified_count": float(elite_certified_count), "elite_update_skipped": float(elite_update_skipped),
            "baseline_score": float(baseline_score), "selected_score": float(selected_score), "improvement": float(score_gain),
            "performance_modification": float(mode == "normal" and baseline_safe and not baseline_selected and not true_stuck),
            "safety_modification": float(not baseline_safe and not baseline_selected),
            "sac_first_steer": float(baseline[0, 0]), "sac_first_throttle": float(baseline[0, 1]),
            "selected_first_steer": float(sequence[0, 0]), "selected_first_throttle": float(sequence[0, 1]),
            "model_uncertainty": float(baseline_disagreement or near_disagreement),
            "baseline_model_disagreement": float(baseline_flags.get("center_model_valid", baseline_safe) != baseline_flags.get("legacy_model_valid", baseline_safe)),
            "selection_reason": float(4 if baseline_selected and not value_ranking_ready else 0 if baseline_selected else 2 if intervention_required else 3 if true_stuck else 5 if not baseline_safe else 1),
            **boundary_diagnostics,
            "recovery_scoring_active": float(recovery_scoring),
            "recovery_progress_unavailable": float(recovery_scoring and not any(r.get("exact", False) and r["valid"]
                and r.get("route_progress_estimate", -1) >= self.min_progress for r in safety_cache.values())),
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
        for label, active in (("normal_search", mode == "normal" and baseline_safe and decision_state == "NORMAL_REFINEMENT"),
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
        metrics["progress_rejected_candidates"] = float(sum(
            reason == "relative_progress_loss" for reason in decision_reasons.values()))
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
        self._last_recovery_observation = (dict(step=getattr(safety_context, "measurement_step", -2),
            road_token=(id(safety_context.road), round(float(getattr(safety_context, "lane_width", None) or safety_context.width), 4)),
            action=sequence[0].detach().clone(), verified=certificate_valid,
            margin=float(getattr(safety_context, "geometric_body_margin", float("nan"))),
            lateral=abs(float(getattr(safety_context, "geometric_lateral_error", 0.))-
                min(getattr(safety_context, "lane_centers", ()) or (0.,), key=lambda d: abs(d-getattr(safety_context, "geometric_lateral_error", 0.)))))
            if recovery_scoring and not emergency_fallback and certificate_valid else None)
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
            recovery_components = {}
            if recovery_scoring and candidate_data:
                import numpy as np
                archive_keys = list(candidate_data)
                archive_actions = torch.stack([torch.from_numpy(np.frombuffer(k, dtype=baseline_host.numpy().dtype).copy()).reshape_as(baseline_host)
                    for k in archive_keys]).to(self.device)
                parts = torch.stack(self._recovery_penalties(archive_actions, baseline, active_recovery_reference, safety_context), 1).cpu().tolist()
                recovery_components = dict(zip(archive_keys, parts))
            for key, item in candidate_data.items():
                state, wm = safety_cache.get(key, {}), host_wm.get(key, {})
                exact = bool(state.get("exact", False))
                valid = bool(state.get("valid", False))
                progress = state.get("route_progress_estimate", float("nan"))
                cost_scored_here = enabled and (wm.get("ranking_validated", False) or self.allow_unvalidated_cost)
                value = float(wm.get("value", float("nan")))
                cost = self.cost_coef*float(wm.get("cost", 0.)) if cost_scored_here else 0.
                risk = float(state.get("risk_score", 0.))
                if key in recovery_components:
                    ref_deviation, sac_deviation, raw_smoothness, first_continuity = recovery_components[key]
                    item["deviation_penalty"] = self.deviation_coef*(ref_deviation+sac_deviation)
                    item["smoothness_penalty"] = self.smoothness_coef*(raw_smoothness+first_continuity)
                    item.update(deviation_to_sac=sac_deviation, deviation_to_recovery_reference=ref_deviation,
                        smoothness=raw_smoothness, first_action_continuity=first_continuity,
                        recovery_reference=recovery_reference_kind)
                final = value-cost-item["deviation_penalty"]-item["smoothness_penalty"]-self.long_risk_coef*risk
                if key == selected_key:
                    reason = ("selected_unverified_emergency" if unverified else
                        "selected_verified_no_progress_fallback" if metrics["recovery_progress_unavailable"] and not emergency_fallback else "selected")
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
                elif recovery_scoring and progress < self.min_progress:
                    reason = "insufficient_progress"
                elif recovery_scoring and key in recovery_fronts and selected_key in recovery_fronts and recovery_fronts[key] > recovery_fronts[selected_key]:
                    reason = "recovery_geometry_front"
                elif key in decision_reasons:
                    reason = decision_reasons[key]
                elif not math.isfinite(final):
                    reason = "wm_score_unavailable"
                elif baseline_selected:
                    reason = "sac_baseline_retained"
                elif value > selected_value and item["deviation_penalty"] > selected_deviation:
                    reason = "deviation_penalty"
                else:
                    reason = "wm_score_lower" if value < selected_value else "trajectory_quality_score_lower"
                candidate_records.append(dict(**item, exact_verified=exact and valid,
                    scoring_mode="recovery" if recovery_scoring else "normal",
                    reward=number(wm.get("reward", float("nan"))), route_progress=number(progress),
                    boundary_risk=number(1/(1+max(0., float(state.get("near_boundary_margin", 0.)))/safety_context.width)) if safety_context is not None else None,
                    lateral_improvement=number(min(abs(float(getattr(safety_context, "geometric_lateral_error", 0.))-c)
                        for c in (getattr(safety_context, "lane_centers", ()) or (0.,)))-float(state.get("terminal_lateral_error", float("nan")))),
                    near_boundary_margin=number(state.get("near_boundary_margin", float("nan"))),
                    terminal_boundary_margin=number(state.get("terminal_boundary_margin", float("nan"))),
                    boundary_margin_change=number(state.get("boundary_margin_change", float("nan"))),
                    center_first_failure_substep=number(state.get("center_first_failure_substep", -1)),
                    legacy_first_failure_substep=number(state.get("legacy_first_failure_substep", -1)),
                    braking_continuation_valid=bool(state.get("braking_continuation_valid", False)),
                    model_rejections={model: {field: state.get(model+"_"+field) for field in
                        ("collision", "offroad", "kinematic_violation", "braking_continuation_valid")} for model in ("center", "legacy")},
                    first_failure_time_upper_s={model: (state[model+"_first_failure_substep"]*self.candidate_safety.geometry.dt/self.candidate_safety.geometry.substeps
                        if state.get(model+"_first_failure_substep", -1) > 0 else None) for model in ("center", "legacy")},
                    first_failure_physics_tick_upper_bound={model: (math.ceil(state[model+"_first_failure_substep"]*self.candidate_safety.geometry.dt/
                        self.candidate_safety.geometry.substeps/float((getattr(safety_context, "motion_signature", None) or {}).get("physics_dt") or
                            self.candidate_safety.geometry.dt/self.candidate_safety.geometry.substeps))
                        if state.get(model+"_first_failure_substep", -1) > 0 else None) for model in ("center", "legacy")},
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
            if recovery_scoring:
                for key, item in zip(candidate_data, candidate_records):
                    item["selected_rank"] = (1 if key == selected_key and certificate_valid else
                        recovery_fronts.get(key) if selected_key in recovery_fronts else
                        recovery_fronts[key]+1 if key in recovery_fronts else None)
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
