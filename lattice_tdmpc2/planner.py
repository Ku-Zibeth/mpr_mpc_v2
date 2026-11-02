"""Lattice coarse planning with continuous SAC residual refinement."""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import torch

from lattice.frenet_metadrive import FrenetPath, MetaDriveFrenetController

from .action_adapter import LatticeActionAdapter
from .evaluator import WorldModelEvaluation, WorldModelEvaluator
from .features import extract_coarse_path_features, extract_path_features, feature_scales_from_config
from .residual_action import ResidualActionAdapter, ResidualBounds
from .state_builder import ResidualStateBuilder


VALID_MODES = {
    "lattice", "lattice_sac", "lattice_tdmpc_sac",
}
WORLD_MODEL_MODES = {"lattice_tdmpc_sac"}


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    return dict(value)


def _value(tensor: torch.Tensor | None) -> float:
    if tensor is None or tensor.numel() == 0:
        return 0.0
    return float(tensor.detach().reshape(-1)[0].cpu())


def _zero_evaluation(device="cpu") -> WorldModelEvaluation:
    zero = torch.zeros(1, dtype=torch.float32, device=device)
    return WorldModelEvaluation(zero, zero.clone(), zero.clone(), zero.clone(), zero.clone())


@dataclass
class ResidualPlanContext:
    coarse_path: FrenetPath
    latent: torch.Tensor | None
    coarse_evaluation: WorldModelEvaluation | None
    path_features: torch.Tensor
    residual_state: torch.Tensor | None
    bounds: ResidualBounds | None
    coarse_candidate_count: int
    coarse_safe_count: int
    fallback_reason: str = ""
    timings_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class ResidualPlanResult:
    coarse_path: FrenetPath
    refined_path: FrenetPath
    low_level_action: np.ndarray
    normalized_residual_action: np.ndarray
    delta_d: float
    delta_v: float
    coarse_return: float
    refined_return: float
    coarse_cost: float
    refined_cost: float
    coarse_uncertainty: float
    refined_uncertainty: float
    wm_improvement: float
    residual_valid: bool
    unsafe_residual: bool
    fallback_reason: str
    residual_state: torch.Tensor | None
    bounds: ResidualBounds | None
    metrics: dict[str, float] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def coarse_d(self) -> float:
        return float(self.coarse_path.target_d)

    @property
    def coarse_v(self) -> float:
        return float(self.coarse_path.target_speed)

    @property
    def coarse_horizon(self) -> float:
        return float(self.coarse_path.horizon)

    @property
    def refined_d(self) -> float:
        return float(self.refined_path.target_d)

    @property
    def refined_v(self) -> float:
        return float(self.refined_path.target_speed)


class ResidualSACLatticePlanner:
    """Receding-horizon bridge between Lattice, TD-MPC2 and residual SAC."""

    def __init__(self, cfg, controller: MetaDriveFrenetController, *, tdmpc_agent=None, action_space=None):
        self.cfg = cfg
        self.controller = controller
        self.lattice = controller.planner
        self.mode = str(_mapping(getattr(cfg, "planner", {})).get("type", "lattice"))
        if self.mode not in VALID_MODES:
            raise ValueError(f"Unsupported planner.type={self.mode!r}; expected {sorted(VALID_MODES)}.")
        self.residual_cfg = _mapping(getattr(cfg, "residual_rl", {}))
        self.safe_fallback = bool(self.residual_cfg.get("safe_fallback", True))
        self.uses_sac = self.mode != "lattice"
        self.uses_world_model = self.mode in WORLD_MODEL_MODES
        self.tdmpc_agent = tdmpc_agent
        if self.uses_world_model and tdmpc_agent is None:
            raise ValueError(f"planner.type={self.mode} requires a pretrained TD-MPC2 agent.")

        self.horizon = int(getattr(cfg, "horizon", 3))
        simulator = _mapping(_mapping(getattr(cfg, "metadrive", {})).get("simulator", {}))
        self.control_dt = float(simulator.get("decision_repeat", 5)) * float(
            simulator.get("physics_world_step_size", 0.02)
        )
        action_dim = int(getattr(cfg, "action_dim", 2) or 2)
        self.path_action_adapter = LatticeActionAdapter(controller, action_dim=action_dim)
        if action_space is not None:
            self.path_action_adapter.validate_action_space(action_space)
        self.residual_action_adapter = ResidualActionAdapter(self.residual_cfg)

        include_wm = self.uses_world_model
        self.state_builder = ResidualStateBuilder(
            include_latent=include_wm,
            include_wm_return=include_wm and bool(self.residual_cfg.get("include_wm_return_in_state", True)),
            include_wm_cost=include_wm and bool(self.residual_cfg.get("include_wm_cost_in_state", False)),
            include_wm_uncertainty=include_wm and bool(self.residual_cfg.get("include_wm_uncertainty_in_state", False)),
            wm_return_scale=float(self.residual_cfg.get("wm_return_state_scale", 10.0)),
            wm_cost_scale=float(self.residual_cfg.get("wm_cost_state_scale", 10.0)),
            wm_uncertainty_scale=float(self.residual_cfg.get("wm_uncertainty_state_scale", 10.0)),
        )
        self.evaluator = None
        if self.uses_world_model:
            self.evaluator = WorldModelEvaluator(
                tdmpc_agent,
                use_cost=bool(self.residual_cfg.get("use_world_model_cost", False)),
                cost_weight=0.0,
                uncertainty_weight=0.0,
                use_q_uncertainty=bool(self.residual_cfg.get("use_uncertainty", True)),
                gamma_cost=self.residual_cfg.get("wm_cost_gamma"),
                use_terminal_cost_q=bool(
                    self.residual_cfg.get("use_terminal_cost_q", False)
                ),
            )
            if bool(self.residual_cfg.get("freeze_tdmpc2", True)):
                self.freeze_world_model()
        self.last_context: ResidualPlanContext | None = None
        self.last_result: ResidualPlanResult | None = None

    def freeze_world_model(self) -> None:
        if self.tdmpc_agent is not None:
            self.tdmpc_agent.model.eval()
            for parameter in self.tdmpc_agent.model.parameters():
                parameter.requires_grad_(False)

    def unfreeze_world_model(self) -> None:
        if self.tdmpc_agent is not None:
            for parameter in self.tdmpc_agent.model.parameters():
                parameter.requires_grad_(True)
            self.tdmpc_agent.model.eval()

    def reset(self) -> None:
        self.controller.reset()
        self.last_context = None
        self.last_result = None

    @torch.no_grad()
    def prepare(self, observation, vehicle) -> ResidualPlanContext:
        """Build the actor state using only information available before its decision."""
        start = time.perf_counter()
        candidates = self.lattice.generate_candidates(vehicle)
        selected_index = self.lattice.select_nominal_path(candidates)
        self.lattice.last_selected_index = selected_index
        coarse_path = candidates[selected_index]
        safe_candidates = self.lattice.filter_feasible_paths(candidates)
        lattice_done = time.perf_counter()

        latent = None
        coarse_evaluation = None
        fallback_reason = ""
        wm_start = time.perf_counter()
        if self.uses_world_model:
            try:
                tensor = observation if torch.is_tensor(observation) else torch.as_tensor(observation, dtype=torch.float32)
                latent = self.evaluator.encode(tensor)
                coarse_evaluation = self.evaluate_paths(latent, [coarse_path], vehicle)
            except (RuntimeError, ValueError, IndexError) as exc:
                fallback_reason = f"wm_coarse_error:{type(exc).__name__}"
                warnings.warn(f"TD-MPC2 coarse evaluation failed: {exc}", RuntimeWarning)
                latent = torch.zeros(
                    (1, int(self.tdmpc_agent.cfg.latent_dim)), dtype=torch.float32,
                    device=self.tdmpc_agent.device,
                )
                coarse_evaluation = _zero_evaluation(self.tdmpc_agent.device)
        wm_done = time.perf_counter()

        path_features = extract_coarse_path_features(
            coarse_path, scales=feature_scales_from_config(self.residual_cfg),
            clip=float(self.residual_cfg.get("path_feature_clip", 5.0)),
        )
        residual_state = None
        bounds = None
        if self.uses_sac:
            residual_state = self.state_builder.build(latent, path_features, coarse_evaluation)
            bounds = self.residual_action_adapter.bounds(coarse_path, self.lattice)
        context = ResidualPlanContext(
            coarse_path=coarse_path, latent=latent, coarse_evaluation=coarse_evaluation,
            path_features=path_features, residual_state=residual_state, bounds=bounds,
            coarse_candidate_count=len(candidates), coarse_safe_count=len(safe_candidates),
            fallback_reason=fallback_reason,
            timings_ms={
                "lattice_coarse": 1000.0 * (lattice_done - start),
                "wm_coarse": 1000.0 * (wm_done - wm_start),
            },
        )
        self.last_context = context
        return context

    @torch.no_grad()
    def refine(self, context: ResidualPlanContext, normalized_action, vehicle) -> ResidualPlanResult:
        start = time.perf_counter()
        coarse = context.coarse_path
        requested_action = np.zeros(2, dtype=np.float32)
        delta = np.zeros(2, dtype=np.float32)
        refined = coarse
        valid = True
        unsafe = False
        fallback_reason = context.fallback_reason
        sac_done = start
        lattice_refine_start = start

        if self.uses_sac:
            requested_action = np.clip(np.asarray(normalized_action, np.float32), -1.0, 1.0)
            if requested_action.shape != (2,):
                raise ValueError(f"Residual SAC action must have shape [2], got {requested_action.shape}.")
            delta = self.residual_action_adapter.to_physical(requested_action, context.bounds)
            sac_done = time.perf_counter()
            lattice_refine_start = sac_done
            if fallback_reason:
                valid = False
                delta[:] = 0.0
            elif np.any(np.abs(delta) > 1e-8):
                try:
                    generated = self.lattice.generate_parameterized_paths(
                        [(float(coarse.target_d + delta[0]), float(coarse.target_speed + delta[1]))],
                        horizon=float(coarse.horizon),
                    )[0]
                    if not self.safe_fallback:
                        refined = generated
                    elif self.lattice.filter_feasible_paths([generated]):
                        refined = generated
                    else:
                        valid = False
                        unsafe = True
                        fallback_reason = "unsafe_refined_path"
                        delta[:] = 0.0
                except (RuntimeError, ValueError, IndexError) as exc:
                    valid = False
                    fallback_reason = f"lattice_refine_error:{type(exc).__name__}"
                    warnings.warn(f"Lattice residual regeneration failed: {exc}", RuntimeWarning)
                    delta[:] = 0.0
        lattice_refine_done = time.perf_counter()

        refined_evaluation = context.coarse_evaluation
        wm_refine_start = time.perf_counter()
        if self.uses_world_model and valid:
            try:
                refined_evaluation = self.evaluate_paths(context.latent, [refined], vehicle)
            except (RuntimeError, ValueError, IndexError) as exc:
                valid = False
                refined = coarse
                delta[:] = 0.0
                refined_evaluation = context.coarse_evaluation
                fallback_reason = f"wm_refine_error:{type(exc).__name__}"
                warnings.warn(f"TD-MPC2 refined evaluation failed: {exc}", RuntimeWarning)
        wm_refine_done = time.perf_counter()

        action, target_index = self.controller._track_path(vehicle, refined)
        low_level_action = np.clip(np.asarray(action, np.float32), -1.0, 1.0)
        if low_level_action.shape != (2,) or not np.isfinite(low_level_action).all():
            raise RuntimeError("Lattice tracker must return a finite [steering, throttle_brake] action.")
        self._set_visualized_paths(coarse, refined)
        end = time.perf_counter()
        coarse_return = _value(context.coarse_evaluation.predicted_return) if context.coarse_evaluation else 0.0
        coarse_cost = _value(context.coarse_evaluation.predicted_cost) if context.coarse_evaluation else 0.0
        coarse_uncertainty = _value(context.coarse_evaluation.uncertainty) if context.coarse_evaluation else 0.0
        refined_return = _value(refined_evaluation.predicted_return) if refined_evaluation else 0.0
        refined_cost = _value(refined_evaluation.predicted_cost) if refined_evaluation else 0.0
        refined_uncertainty = _value(refined_evaluation.uncertainty) if refined_evaluation else 0.0
        norm = self.residual_norm(delta)
        improvement = (
            refined_return - coarse_return
            - float(self.residual_cfg.get("wm_cost_weight", 1.0)) * (refined_cost - coarse_cost)
            - float(self.residual_cfg.get("wm_uncertainty_weight", 0.0)) * (refined_uncertainty - coarse_uncertainty)
            - float(self.residual_cfg.get("wm_delta_weight", 0.1)) * norm
        )
        metrics = self._metrics(
            context, refined, delta, norm, valid, coarse_return, refined_return,
            coarse_cost, refined_cost, coarse_uncertainty, refined_uncertainty,
            improvement, sac_done, start, lattice_refine_start, lattice_refine_done,
            wm_refine_start, wm_refine_done, end, vehicle,
        )
        info = {
            "residual_coarse_d": float(coarse.target_d), "residual_coarse_v": float(coarse.target_speed),
            "residual_delta_d": float(delta[0]), "residual_delta_v": float(delta[1]),
            "residual_refined_d": float(refined.target_d), "residual_refined_v": float(refined.target_speed),
            "residual_valid": bool(valid), "unsafe_residual": bool(unsafe),
            "residual_fallback_reason": fallback_reason, "wm_coarse_return": coarse_return,
            "wm_refined_return": refined_return, "wm_improvement": improvement,
            "residual_target_waypoint": tuple(float(value) for value in refined.xy[target_index]),
            **metrics,
        }
        result = ResidualPlanResult(
            coarse_path=coarse, refined_path=refined, low_level_action=low_level_action,
            normalized_residual_action=requested_action, delta_d=float(delta[0]), delta_v=float(delta[1]),
            coarse_return=coarse_return, refined_return=refined_return,
            coarse_cost=coarse_cost, refined_cost=refined_cost,
            coarse_uncertainty=coarse_uncertainty, refined_uncertainty=refined_uncertainty,
            wm_improvement=improvement, residual_valid=valid, unsafe_residual=unsafe,
            fallback_reason=fallback_reason, residual_state=context.residual_state,
            bounds=context.bounds, metrics=metrics, info=info,
        )
        self.last_result = result
        return result

    def plan(self, observation, vehicle, normalized_action=None) -> ResidualPlanResult:
        context = self.prepare(observation, vehicle)
        if normalized_action is None:
            normalized_action = np.zeros(2, dtype=np.float32)
        return self.refine(context, normalized_action, vehicle)

    @torch.no_grad()
    def evaluate_paths(self, latent, paths, vehicle) -> WorldModelEvaluation:
        actions = self.path_action_adapter.paths_to_actions(
            paths, vehicle, horizon=self.horizon, control_dt=self.control_dt,
            device=self.tdmpc_agent.device,
        )
        return self.evaluator.evaluate_action_sequences(latent, actions)

    def residual_norm(self, delta) -> float:
        sigma_d = max(abs(float(self.residual_cfg.get("sigma_d", 0.4))), 1e-6)
        sigma_v = max(abs(float(self.residual_cfg.get("sigma_v", 2.0))), 1e-6)
        return float((float(delta[0]) / sigma_d) ** 2 + (float(delta[1]) / sigma_v) ** 2)

    def _set_visualized_paths(self, coarse, refined) -> None:
        paths = list(self.lattice.last_candidates)
        if not any(path is coarse for path in paths):
            paths.append(coarse)
        if not any(path is refined for path in paths):
            paths.append(refined)
        self.lattice.last_candidates = paths
        self.lattice.last_selected_index = next(index for index, path in enumerate(paths) if path is refined)

    def _metrics(
        self, context, refined, delta, norm, valid, j0, jr, c0, cr, u0, ur,
        improvement, sac_done, start, lattice_start, lattice_done, wm_start, wm_done,
        end, vehicle,
    ) -> dict[str, float]:
        coarse = context.coarse_path
        points = refined.xy
        position = np.asarray(vehicle.position, dtype=float)[:2]
        tracking_lateral_error = float(np.min(np.linalg.norm(points - position, axis=1))) if len(points) else 0.0
        speed_error = float(refined.target_speed - float(vehicle.speed))
        heading_error = 0.0
        if len(points) >= 2:
            nearest = int(np.argmin(np.linalg.norm(points - position, axis=1)))
            other = min(nearest + 1, len(points) - 1)
            if other == nearest:
                other = max(0, nearest - 1)
            tangent = points[other] - points[nearest]
            if np.linalg.norm(tangent) > 1e-6:
                path_heading = math.atan2(tangent[1], tangent[0])
                heading_error = (path_heading - float(vehicle.heading_theta) + math.pi) % (2 * math.pi) - math.pi
        return {
            "coarse/d": float(coarse.target_d), "coarse/v": float(coarse.target_speed),
            "coarse/horizon": float(coarse.horizon),
            "coarse/clearance": float(extract_path_features(coarse)["min_obstacle_clearance"]),
            "coarse/obstacle_cost": float(coarse.obstacle_cost),
            "wm/coarse_return": j0, "wm/refined_return": jr, "wm/delta_return": jr - j0,
            "wm/coarse_cost": c0, "wm/refined_cost": cr, "wm/delta_cost": cr - c0,
            "wm/coarse_uncertainty": u0, "wm/refined_uncertainty": ur,
            "residual/delta_d": float(delta[0]), "residual/delta_v": float(delta[1]),
            "residual/final_d": float(refined.target_d), "residual/final_v": float(refined.target_speed),
            "residual/norm": norm, "residual/valid": float(valid),
            "residual/fallback_rate": float(not valid), "residual/wm_improvement": improvement,
            "diagnostic/tracking_lateral_error": tracking_lateral_error,
            "diagnostic/tracking_speed_error": speed_error,
            "diagnostic/heading_error": float(heading_error),
            "diagnostic/min_clearance": float(extract_path_features(refined)["min_obstacle_clearance"]),
            "time/lattice_coarse_ms": context.timings_ms["lattice_coarse"],
            "time/wm_coarse_ms": context.timings_ms["wm_coarse"],
            "time/sac_ms": 1000.0 * (sac_done - start),
            "time/lattice_refine_ms": 1000.0 * (lattice_done - lattice_start),
            "time/wm_refine_ms": 1000.0 * (wm_done - wm_start),
            "time/total_ms": 1000.0 * (end - start) + context.timings_ms["lattice_coarse"] + context.timings_ms["wm_coarse"],
            "lattice/coarse_candidates": float(context.coarse_candidate_count),
            "lattice/coarse_safe_candidates": float(context.coarse_safe_count),
        }
