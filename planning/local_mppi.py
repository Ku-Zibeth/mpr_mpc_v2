"""SAC-centred refinement with independent same-prefix stopping certification."""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import hashlib
from collections import deque
import numpy as np
from .mppi_safety import MPPISafetyFilter
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
MPPI_DIAGNOSTIC_KEYS += (
    "near_collision_reject_count", "near_offroad_reject_count", "full_collision_reject_count", "full_offroad_reject_count",
    "kinematic_reject_count", "braking_continuation_reject_count", "hold_braking_pass_rate", "guided_braking_pass_rate",
    "tail_viability_repair_calls", "tail_viability_repair_success", "model_uncertainty", "model_unknown", "safety_horizon_s",
    "exact_checked_candidates", "exact_failed_candidates", "elite_update_skipped", "value_ranking_validated",
    "safety_failure", "execution_authorized", "selected_safety_valid", "decision_state", "baseline_near_valid",
    "baseline_full_valid", "normal_search_delta", "recovery_search_delta", "stopping_time_s", "cost_penalty", "cost_validated",
    "preview_ms", "safety_check_ms", "candidate_evaluation_ms",
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
    safety_certificate: dict = field(default_factory=dict)
    safety_failure: bool = False
    decision_metadata: dict = field(default_factory=dict)

    # Public Coordinator aliases all denote the same unpenalised WM objective.
    baseline_score = property(lambda self: self.baseline_wm_value)
    baseline_value = property(lambda self: self.baseline_wm_value)
    selected_score = property(lambda self: self.selected_wm_value)
    final_value = property(lambda self: self.selected_wm_value)
    planner_score_gain = property(lambda self: self.wm_value_gain)
    planner_gain = property(lambda self: self.wm_value_gain)


class LocalMPPI:
    def __init__(self, evaluator, cfg, *, horizon, action_dim, device, control_dt=.1):
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
        self.control_dt = float(control_dt)
        self.noise_rho = float(cfg.get("noise_rho", .8))
        if not 0 <= self.noise_rho < 1:
            raise ValueError("noise_rho must be in [0,1).")
        safety = dict(cfg.get("safety", {}))
        safety.update(two_layer=True, staged=True, center_bicycle=True, center_model_guard=True)
        self.candidate_safety = MPPISafetyFilter(safety, control_dt=control_dt)
        if cfg.get("allow_unvalidated_cost", False):
            raise ValueError("Unvalidated Cost is forbidden in certified SAC refinement.")
        self.generator = torch.Generator(device=self.device).manual_seed(self.eval_seed)
        self._prev_mean = None
        self._previous_certified = None
        self._motion_history = deque(maxlen=max(2, math.ceil(2./control_dt)))

    def reset(self, seed=None):
        self._prev_mean = self._previous_certified = None
        self._motion_history.clear()
        self.candidate_safety.reset()
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
        # Independent steering/throttle innovations; stationary AR(1) in time.
        for step in range(1, self.horizon+1):
            noise[:, step] = self.noise_rho*noise[:, step-1]+math.sqrt(1-self.noise_rho**2)*noise[:, step]
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

    def _structured(self, baseline, context):
        proposals = [("sac", baseline)]
        lattice = getattr(context, "lattice_actions", None)
        if lattice is not None:
            proposals.append(("lattice", self._project(lattice.to(baseline), baseline)))
        if self._previous_certified is not None:
            shifted = torch.cat((self._previous_certified[1:], self._previous_certified[-1:]))
            if len(shifted) < len(baseline):
                shifted = torch.cat((shifted, shifted[-1:].expand(len(baseline)-len(shifted), -1)))
            proposals.append(("certified_shift", self._project(shifted[:len(baseline)].to(baseline), baseline)))
        tangent = getattr(context, "road_tangent_steer", None)
        if tangent is not None and math.isfinite(tangent):
            reference = baseline.clone()
            blend = torch.linspace(0., 1., len(reference), device=self.device)
            reference[:, 0] += blend*(tangent-reference[:, 0])
            proposals.append(("road_tangent", self._project(reference, baseline)))
            proposals.append(("road_tangent_half", self._project(baseline+.5*(reference-baseline), baseline)))
        # Local, bounded alternatives; no independent Lattice Gaussian or takeover.
        for name, throttle in (("toward_deceleration", -.5), ("toward_coast", 0.), ("toward_acceleration", .25)):
            reference = proposals[-1][1].clone()
            reference[:, 1] = throttle
            proposals.append((name, self._project(reference, baseline)))
        return proposals

    @staticmethod
    def _geometric_key(result, index, actions, baseline, context):
        def number(name, missing):
            value = getattr(result, name, None)
            if value is None:
                return missing
            value = float(value[index])
            return value if math.isfinite(value) else missing
        margin = number("boundary_margin", -math.inf)
        gap = number("min_clearance", math.inf if not context.obstacles else -math.inf)
        progress = number("route_progress_estimate", -math.inf)
        speed = min(number("terminal_speed", -math.inf), getattr(context, "target_speed", None) or math.inf)
        deviation = float((actions-baseline).square().mean())
        prior = actions.new_tensor(context.previous_action)[None]
        smooth = float(torch.diff(torch.cat((prior, actions)), dim=0).square().mean())
        return (min(margin, gap), margin, gap, progress, speed, -deviation, -smooth)

    @torch.no_grad()
    def plan(self, z_t, baseline, *, eval_mode, task=None, safety_context=None, lattice_actions=None):
        baseline = baseline.to(self.device)
        if (baseline.shape != (self.horizon+1, self.action_dim) or not torch.isfinite(baseline).all()
                or (baseline.abs() > 1).any()):
            raise ValueError("SAC baseline must be finite, in [-1,1], and [H+1, action_dim].")
        phase = PhaseTimings(self.device)
        context = safety_context
        if context is None:
            raise RuntimeError("Certified MPPI requires a current independent safety context.")
        if lattice_actions is not None:
            context.lattice_actions = lattice_actions
        metrics = dict(exact_checked_candidates=0., exact_failed_candidates=0.,
                       elite_update_skipped=0., iterations_completed=0., wm_evaluated_candidates=0.,
                       near_collision_reject_count=0., near_offroad_reject_count=0.,
                       full_collision_reject_count=0., full_offroad_reject_count=0.,
                       kinematic_reject_count=0., braking_continuation_reject_count=0.,
                       model_uncertainty=0., tail_viability_repair_calls=0., tail_viability_repair_success=0.)
        def certify(actions):
            with phase.measure("safety_check"):
                result = self.candidate_safety.certify(actions, context)
            metrics["exact_checked_candidates"] += len(actions)
            metrics["exact_failed_candidates"] += float((~result.valid).sum())
            for key, field in (("near_collision", "near_collision"), ("near_offroad", "near_offroad"),
                               ("full_collision", "collision"), ("full_offroad", "offroad"),
                               ("kinematic", "kinematic_violation")):
                metrics[key+"_reject_count"] += float(getattr(result, field).sum())
            metrics["braking_continuation_reject_count"] += float((result.near_valid & ~result.valid & ~result.safety_unknown).sum())
            center, legacy = result.center_model_valid, result.legacy_model_valid
            if center is not None and legacy is not None:
                metrics["model_uncertainty"] = max(metrics["model_uncertainty"], float((center != legacy).any()))
            return result
        base_check = certify(baseline[None])
        baseline_safe = bool(base_check.valid[0])
        near_safe = bool(base_check.near_valid[0])
        # REAL measured progress only. Unknown route projection cannot trigger stuck.
        progress = getattr(context, "route_progress", None)
        self._motion_history.append((context.speed, progress))
        history = list(self._motion_history)
        valid_history = len(history) == self._motion_history.maxlen and all(p is not None and math.isfinite(p) for _, p in history)
        direction = np.array([math.cos(context.heading), math.sin(context.heading)])
        normal = np.array([-direction[1], direction[0]])
        blocked = any(float(o.velocity @ direction) < context.speed
                      and abs(float((o.position-context.position) @ normal)) < (context.width+o.width)/2
                      and 0 < float((o.position-context.position) @ direction) < context.length*3
                      for o in context.obstacles)
        stuck = bool(valid_history and max(v for v, _ in history) < .3 and abs(history[-1][1]-history[0][1]) < .001
                     and math.isfinite(getattr(context, "geometric_body_margin", math.nan))
                     and context.geometric_body_margin < context.width/2 and not blocked)
        state = ("NORMAL_REFINEMENT" if baseline_safe else "TRUE_STUCK_RECOVERY" if stuck
                 else "TAIL_VIABILITY_REPAIR" if near_safe else "SAFETY_INTERVENTION")
        qualified = bool(getattr(self.evaluator, "value_ranking_ready", lambda: False)())
        metrics.update(value_ranking_validated=float(qualified), baseline_safety_valid=float(baseline_safe),
            baseline_near_valid=float(near_safe), baseline_full_valid=float(base_check.full_valid[0]),
            traffic_waiting=float(blocked), tail_viability_repair_calls=float(state == "TAIL_VIABILITY_REPAIR"),
            center_first_failure_substep=float(base_check.center_first_failure_substep[0]) if base_check.center_first_failure_substep is not None else -1.,
            legacy_first_failure_substep=float(base_check.legacy_first_failure_substep[0]) if base_check.legacy_first_failure_substep is not None else -1.,
            road_seams_repaired=float(context.road_seams_repaired))
        mean = self.initial_mean(baseline)
        std = self.initial_std_vector.expand_as(mean).clone()
        initial_mean, initial_std = mean.clone(), std.clone()
        best, best_key, selected_check, source = None, None, None, "none"
        nan = baseline.new_tensor(float("nan")); base_value = selected_value = nan
        base_row = chosen_row = {"value": nan, "cost": nan, "validated": False, "reward": nan, "terminal_q": nan}
        elite_values = baseline.new_empty(0)
        last_prediction = None
        def evaluate(actions):
            nonlocal last_prediction
            result = self.evaluator.evaluate_mppi_candidates(z_t, actions, task=task, compute_cost=False)
            last_prediction = result
            metrics["wm_evaluated_candidates"] += len(actions)
            for key, value in getattr(result, "timings", {}).items():
                metrics[key] = metrics.get(key, 0.)+value
            return result.total_value.reshape(-1)
        if baseline_safe:
            best, selected_check, source = baseline, base_check, "sac"
            # A high unvalidated Q must NEVER approve a normal performance edit.
            if not qualified:
                best_key = (0.,)
        skip = baseline_safe and not qualified
        proposals = self._structured(baseline, context)
        if not skip:
            for iteration in range(self.iterations):
                with phase.measure("mppi_sampling"):
                    actions = self._candidate_batch(mean, std, baseline)
                    sources = ["gaussian"]*len(actions); sources[0] = "sac"
                    if state != "NORMAL_REFINEMENT":
                        for i, (name, proposal) in enumerate(proposals[:len(actions)]):
                            actions[i], sources[i] = proposal, name
                with phase.measure("preview"):
                    preview = self.candidate_safety.preview(actions, context)
                # Geometry/ranking live on CPU; transfer each batch once, not
                # two scalar synchronizations for every lexicographic candidate.
                actions_cpu, baseline_cpu = actions.detach().cpu(), baseline.detach().cpu()
                live = preview.near_valid.nonzero().flatten().to(self.device)
                if not len(live):
                    metrics["elite_update_skipped"] += 1
                    continue
                with phase.measure("candidate_evaluation"):
                    values = evaluate(actions[live])
                values_cpu = values.detach().cpu()
                if state == "NORMAL_REFINEMENT":
                    ordering = sorted(range(len(live)), key=lambda i: float(values_cpu[i]) if math.isfinite(float(values_cpu[i])) else -math.inf, reverse=True)
                else:
                    live_cpu = live.cpu().tolist()
                    ordering = sorted(range(len(live)), key=lambda i: self._geometric_key(preview, live_cpu[i], actions_cpu[live_cpu[i]], baseline_cpu, context), reverse=True)
                # Bounded Top-K exact promotion. Failed promotions never update
                # the Gaussian; another bounded block is tried next iteration.
                start = (iteration*2*self.num_elites) % max(1, len(ordering))
                shortlist = ordering if state == "NORMAL_REFINEMENT" else ordering[start:]+ordering[:start]
                shortlist = shortlist[:2*self.num_elites]
                indices = live[shortlist]
                exact = certify(actions[indices])
                accepted = []
                for j, idx in enumerate(indices.tolist()):
                    if not bool(exact.valid[j]):
                        continue
                    value = values[shortlist[j]]
                    numeric = float(values_cpu[shortlist[j]])
                    if state == "NORMAL_REFINEMENT" and not math.isfinite(numeric):
                        continue
                    key = ((numeric,) if state == "NORMAL_REFINEMENT" else
                           self._geometric_key(exact, j, actions_cpu[idx], baseline_cpu, context))
                    accepted.append((key, idx, value))
                    if best_key is None or key > best_key:
                        best, best_key, selected_check, source = actions[idx].clone(), key, (exact, j), sources[idx]
                if len(accepted) >= self.num_elites:
                    accepted.sort(key=lambda row: row[0], reverse=True)
                    ids = [r[1] for r in accepted[:self.num_elites]]
                    # Geometric intervention has ordinal lexicographic ranks,
                    # NOT unvalidated Q numbers mixed into safety selection.
                    weights = (torch.stack([r[2] for r in accepted[:self.num_elites]]) if state == "NORMAL_REFINEMENT"
                               else baseline.new_tensor(list(range(self.num_elites, 0, -1))))
                    with phase.measure("elite_update"):
                        mean, std, elite_values = self._update_distribution(actions[ids], weights)
                        ratio = len(accepted)/max(1, len(indices))
                        std = (std*(.5+.5*ratio)).clamp(self.min_std_vector, self.max_std_vector)
                else:
                    metrics["elite_update_skipped"] += 1
                metrics["iterations_completed"] += 1
            mean = self._project(mean, baseline)
            # A weighted average of certified trajectories is NOT certified.
            checked_mean = certify(mean[None])
            if bool(checked_mean.valid[0]):
                value = evaluate(mean[None])[0]
                key = ((float(value),) if state == "NORMAL_REFINEMENT" and torch.isfinite(value) else
                       self._geometric_key(checked_mean, 0, mean.cpu(), baseline.cpu(), context) if state != "NORMAL_REFINEMENT" else None)
                if key is not None and (best_key is None or key > best_key):
                    best, best_key, selected_check, source = mean, key, checked_mean, "gaussian_mean"
        # Baseline and selected WM values are diagnostics; never intervention approval.
        if best is not None:
            values = evaluate(torch.stack((baseline, best)))
            base_value, selected_value = values[0], values[1]
            def prediction_row(index, value):
                return dict(value=value, cost=nan, validated=False,
                    reward=getattr(last_prediction, "discounted_reward", baseline.new_full((2,), float("nan")))[index],
                    terminal_q=getattr(last_prediction, "terminal_q", baseline.new_full((2,), float("nan")))[index])
            base_row, chosen_row = prediction_row(0, base_value), prediction_row(1, selected_value)
            if state == "NORMAL_REFINEMENT" and (not qualified or not torch.isfinite(values).all() or not values[1] > values[0]):
                best, selected_check, source, selected_value = baseline, base_check, "sac", base_value
                chosen_row = base_row
        failed = best is None
        sequence = baseline.clone() if failed else best
        certificate = {"valid": False, "reason": "no_certified_action"}
        if not failed:
            result, row = selected_check if isinstance(selected_check, tuple) else (selected_check, 0)
            witness = result.stopping_witnesses[row]
            certificate = dict(valid=True, sequence_sha256=hashlib.sha256(sequence.contiguous().cpu().numpy().tobytes()).hexdigest(),
                observation_context_id=id(context), evidence_sha256=result.evidence_sha256,
                witness_sha256=hashlib.sha256(witness.contiguous().numpy().tobytes()).hexdigest(),
                safety_horizon_steps=result.safety_horizon_steps, committed_controls=self.candidate_safety.committed_controls(self.horizon),
                scope="same-prefix model-checked stopping continuation; not a real-world safety guarantee")
            # Only the stopping witness was certified beyond the committed
            # prefix; do not call an arbitrary WM tail a certified continuation.
            self._previous_certified = witness.detach().to(self.device).clone()
            metrics.update(safety_horizon_s=result.safety_horizon_steps*self.control_dt,
                hold_braking_pass_rate=float(result.hold_braking_valid.float().mean()),
                guided_braking_pass_rate=float(result.guided_braking_valid.float().mean()),
                predicted_first_collision=float(result.near_collision[row]),
                predicted_first_offroad=float(result.near_offroad[row]),
                predicted_collision=float(result.collision[row]), predicted_offroad=float(result.offroad[row]))
        else:
            self._previous_certified = None
        self._prev_mean = (sequence if failed else mean).detach().clone()
        delta = sequence[0]-baseline[0]
        chosen = torch.equal(sequence, baseline) and not failed
        mode = ("safety_failure" if failed else "baseline" if chosen else "normal" if state == "NORMAL_REFINEMENT"
                else "tail_repair" if state == "TAIL_VIABILITY_REPAIR" else "true_stuck_recovery" if stuck else "safety_intervention")
        metrics.update(phase.finish())
        metrics.update(score_gain=float(selected_value-base_value), baseline_selected=float(chosen),
            predicted_reward=float(chosen_row["reward"]), terminal_q=float(chosen_row["terminal_q"]),
            first_steer_delta=float(delta[0]), first_throttle_delta=float(delta[1]), first_action_delta_norm=float(delta.norm()),
            sac_to_selected_delta=float(delta.norm()), safety_failure=float(failed), execution_authorized=float(not failed),
            selected_safety_valid=float(not failed), model_unknown=float(base_check.safety_unknown.any()),
            baseline_certification_reason=base_check.certification_reason, decision_state=("NORMAL_REFINEMENT", "SAFETY_INTERVENTION", "TRUE_STUCK_RECOVERY", "TAIL_VIABILITY_REPAIR").index(state),
            diagnostic_state="MODEL_UNCERTAINTY" if metrics["model_uncertainty"] or base_check.safety_unknown.any() else state,
            decision_state_name=state, candidate_source=source, selection_reason="unvalidated_value_keep_sac" if skip else "no_certified_action" if failed else "certified_lexicographic" if state != "NORMAL_REFINEMENT" else "validated_value",
            tail_viability_repair_success=float(state == "TAIL_VIABILITY_REPAIR" and not failed),
            normal_search_delta=float(delta.norm()) if state == "NORMAL_REFINEMENT" and not chosen else 0.,
            recovery_search_delta=float(delta.norm()) if source in ("gaussian", "gaussian_mean") and state != "NORMAL_REFINEMENT" else 0.,
            planning_mode=float(("baseline", "normal", "tail_repair", "safety_intervention", "true_stuck_recovery", "safety_failure").index(mode)),
            cost_penalty=0., allow_unvalidated_cost=0., cost_validated=0.,
            road_rejection_kind=base_check.road_rejection_kind, stopping_time_s=base_check.stopping_time_s,
            nominal_stopping_time_s=self.candidate_safety.stopping_diagnostics(context, self.horizon)["stopping_time_s"])
        base_row = dict(base_row, value=base_value); chosen_row = dict(chosen_row, value=selected_value)
        metadata = {key: metrics.pop(key) for key in ("baseline_certification_reason", "diagnostic_state", "decision_state_name", "candidate_source", "selection_reason", "road_rejection_kind")}
        return LocalMPPIResult(action=sequence[0], selected_sequence=sequence, initial_mean=initial_mean, final_mean=mean,
            initial_std=initial_std, final_std=std, baseline_wm_value=base_value, selected_wm_value=selected_value,
            wm_value_gain=selected_value-base_value, elite_values=elite_values, baseline_selected=chosen,
            max_action_delta=(sequence-baseline).abs().amax(0), metrics=metrics, baseline_consequence=base_row,
            selected_consequence=chosen_row, planning_mode=mode,
            safety_certificate=certificate, safety_failure=failed, decision_metadata=metadata)


__all__ = ["LocalMPPI", "LocalMPPIResult", "_action_vector", "MPPI_DIAGNOSTIC_KEYS", "MPPI_EPISODE_METRICS"]
