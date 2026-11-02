"""Single-coarse-trajectory MPR-MPC planning coordinator."""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import torch

from .config import mapping, section
from .evaluator import TrajectoryWorldModelEvaluator
from .kinematic_filter import CorridorContext
from .local_mppi import LocalMPPI
from .safety import TrajectorySafetyEvaluator
from mpr_mpc.residual_rl import ResidualActionAdapter, ResidualStateBuilder
from mpr_mpc.residual_rl.types import ResidualPolicyOutput
from .structured_proposal import StructuredProposalGenerator
from .time_alignment import align_frenet_path
from .trajectory_adapter import LatticeActionAdapter
from .types import MPRPlanContext, MPRPlanResult, PlanningStage


class MPRMPCPlanner:
    """Global Lattice selection, optional 2-D residual, then bounded local MPPI."""

    def __init__(self, cfg, tdmpc_agent, controller, *, action_space=None):
        self.cfg = cfg
        self.agent = tdmpc_agent
        self.device = tdmpc_agent.device
        self.horizon = int(cfg.horizon)
        self.action_dim = int(cfg.action_dim)
        self.mpr_cfg = section(cfg, "mpr_mpc")
        self.enabled = bool(self.mpr_cfg.get("enabled", True))
        self.allow_residual = bool(self.mpr_cfg.get("use_residual", True))
        self.allow_mppi = bool(self.mpr_cfg.get("use_mppi", True))
        stages = mapping(self.mpr_cfg.get("stages"))
        self.residual_start_step = int(stages.get("residual_start_step", 20000))
        self.mppi_start_step = int(stages.get("mppi_start_step", 50000))
        if not 0 <= self.residual_start_step <= self.mppi_start_step:
            raise ValueError("Require 0 <= residual_start_step <= mppi_start_step.")

        simulator = mapping(section(cfg, "metadrive").get("simulator"))
        self.control_dt = float(simulator.get("physics_world_step_size", 0.02)) * float(
            simulator.get("decision_repeat", 5)
        )
        self.structured = StructuredProposalGenerator(controller)
        self.action_adapter = LatticeActionAdapter(controller, action_dim=self.action_dim)
        if action_space is not None:
            self.action_adapter.validate_action_space(action_space)
        self.evaluator = TrajectoryWorldModelEvaluator(tdmpc_agent, self.horizon)

        self.residual_rl_cfg = mapping(section(cfg, "residual_rl"))
        self.residual_state_builder = ResidualStateBuilder(
            self.residual_rl_cfg,
            latent_dim=int(cfg.latent_dim),
            device=self.device,
        )
        self.residual_state_dim = self.residual_state_builder.state_dim
        self.residual_action_adapter = ResidualActionAdapter()
        self.residual_action_provider = None
        self.residual_mapping = str(
            self.residual_rl_cfg.get("residual_mapping", "linear")
        ).lower()
        if self.residual_mapping not in {"linear", "latent_tanh"}:
            raise ValueError("residual_rl.residual_mapping must be 'linear' or 'latent_tanh'.")
        scale = self.residual_rl_cfg.get("latent_tanh_scale", [1.0, 1.0])
        if len(scale) != 2:
            raise ValueError("residual_rl.latent_tanh_scale must contain two values.")
        self.latent_tanh_scale = torch.tensor(
            [float(scale[0]), float(scale[1])],
            dtype=torch.float32,
            device=self.device,
        )
        self.latent_tanh_eps = float(self.residual_rl_cfg.get("latent_tanh_eps", 1e-6))

        refinement = mapping(self.mpr_cfg.get("refinement"))
        self.min_speed = float(refinement.get("min_speed", 0.5))
        self.max_speed = float(refinement.get("max_speed", 20.0))
        self.min_horizon = float(refinement.get("min_horizon", 1.5))
        self.max_horizon = float(refinement.get("max_horizon", 3.5))
        mppi_cfg = mapping(self.mpr_cfg.get("mppi"))
        self.mppi_cfg = mppi_cfg
        self.local_mppi = LocalMPPI(
            self.evaluator,
            mppi_cfg,
            horizon=self.horizon,
            action_dim=self.action_dim,
            device=self.device,
            control_dt=self.control_dt,
        )
        self.safety = (TrajectorySafetyEvaluator(mppi_cfg, control_dt=self.control_dt, device=self.device)
                       if mppi_cfg.get("dynamic_safety", False) else None)
        self.local_mppi.safety_evaluator = self.safety
        self._print_startup_diagnostics(mppi_cfg)
        self.last_context: MPRPlanContext | None = None
        self.last_result: MPRPlanResult | None = None
        self.residual_invalid_count = 0
        self.residual_attempt_count = 0
        self.mppi_call_count = 0
        self.baseline_selected_count = 0


    def set_residual_action_provider(self, provider) -> None:
        self.residual_action_provider = provider

    def _print_startup_diagnostics(self, mppi_cfg) -> None:
        latent_dim = int(self.cfg.latent_dim)
        path_features = 12
        wm_return = 1
        feasibility = 2
        wm_cost = self.residual_state_builder.wm_cost_dim
        print(
            "MPR-MPC configuration\n"
            "---------------------\n"
            f"Stage A: [0, {self.residual_start_step}) Lattice + TD-MPC2\n"
            f"Stage B: [{self.residual_start_step}, {self.mppi_start_step}) "
            "Lattice + Residual SAC, MPPI OFF\n"
            f"Stage C: [{self.mppi_start_step}, ...) "
            "Lattice + Residual SAC + Local MPPI\n\n"
            "Residual algorithm: SAC-Lagrangian\n"
            f"Residual SAC state_dim={self.residual_state_dim}\n"
            f"  latent={latent_dim} path_features={path_features} "
            f"wm_return={wm_return} feasibility={feasibility} wm_cost={wm_cost}\n"
            "Residual action dim: 2\n"
            f"Residual mapping: {self.residual_mapping}\n"
            "Physical delta_d bound: dynamic lane width\n"
            "Physical delta_v bound: 0.5 * coarse speed\n\n"
            "Local MPPI:\n"
            f"samples={int(mppi_cfg.get('num_samples', 64))} "
            f"elites={int(mppi_cfg.get('num_elites', 8))} "
            f"iterations={int(mppi_cfg.get('iterations', 4))}"
        )

    def stage_for_step(self, global_step: int) -> PlanningStage:
        step = max(0, int(global_step))
        if step < self.residual_start_step:
            return PlanningStage("A", False, False)
        if step < self.mppi_start_step:
            return PlanningStage("B", self.allow_residual, False)
        return PlanningStage("C", self.allow_residual, self.allow_mppi)

    def reset(self) -> None:
        self.structured.reset()
        self.local_mppi.reset()
        if self.safety is not None:
            self.safety.reset()
        self.last_context = None
        self.last_result = None

    def clamp_parameters(self, parameters) -> np.ndarray:
        values = np.asarray(parameters, dtype=np.float32).copy()
        lateral_min, lateral_max = self.structured.planner.last_lateral_bounds
        values[0] = np.clip(values[0], lateral_min, lateral_max)
        values[1] = np.clip(values[1], self.min_speed, self.max_speed)
        values[2] = np.clip(values[2], self.min_horizon, self.max_horizon)
        return values

    def residual_bounds(self, coarse_path) -> torch.Tensor:
        lane_width = float(self.structured.planner.last_lane_width)
        if not np.isfinite(lane_width) or lane_width <= 0:
            raise RuntimeError("Lattice planner did not provide a positive current lane width.")
        target_speed = max(0.0, float(coarse_path.target_speed))
        return torch.tensor(
            [[lane_width, 0.5 * target_speed]], dtype=torch.float32, device=self.device
        )

    def residual_parameter_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        d_min, d_max = self.structured.planner.last_lateral_bounds
        lower = torch.tensor(
            [[float(d_min), self.min_speed]], dtype=torch.float32, device=self.device
        )
        upper = torch.tensor(
            [[float(d_max), self.max_speed]], dtype=torch.float32, device=self.device
        )
        return lower, upper

    def apply_residual_parameters(self, coarse_parameters, residual) -> np.ndarray:
        coarse = np.asarray(coarse_parameters, dtype=np.float32)
        correction = np.asarray(residual, dtype=np.float32).reshape(2)
        requested = coarse.copy()
        requested[:2] += correction
        refined = self.clamp_parameters(requested)
        refined[2] = coarse[2]
        return refined

    @torch.no_grad()
    def prepare(
        self, observation, vehicle, *, need_world_model_features: bool = True
    ) -> MPRPlanContext:
        observation = observation if torch.is_tensor(observation) else torch.as_tensor(
            observation, dtype=torch.float32
        )
        selection = self.structured.select_coarse(vehicle)
        coarse_path = selection.coarse_path
        coarse_actions = self.action_adapter.path_to_actions(
            coarse_path,
            vehicle,
            required_steps=self.horizon + 1,
            control_dt=self.control_dt,
            device=self.device,
        )
        parameters = torch.tensor(
            [[coarse_path.target_d, coarse_path.target_speed, coarse_path.horizon]],
            dtype=torch.float32,
            device=self.device,
        )
        residual_bounds = self.residual_bounds(coarse_path)
        safety_context = None
        if self.safety is not None:
            width, wheelbase, steering = self._vehicle_geometry(vehicle)
            safety_context = self.safety.context(self.structured.planner, vehicle,
                                                  width=width, wheelbase=wheelbase, max_steering=steering)
        latent = consequence = wm_features = residual_state = None
        if need_world_model_features:
            latent = self.evaluator.encode(observation).detach()
            consequence = self.evaluator.evaluate_trajectory_consequence(
                latent, coarse_actions.unsqueeze(0)
            )
            if safety_context is not None:
                consequence = replace(consequence, risk_summary=self.safety.check(coarse_actions.unsqueeze(0), safety_context).summary())
            wm_features = consequence.features.detach()
            residual_state = self.residual_state_builder.build(
                latent=latent,
                coarse_path=coarse_path,
                consequence=consequence,
                feasible_ratio=len(selection.feasible) / max(len(selection.candidates), 1),
                used_fallback=selection.used_fallback,
            )
        context = MPRPlanContext(
            observation=observation,
            latent=latent,
            selection=selection,
            coarse_actions=coarse_actions,
            coarse_consequence=consequence,
            wm_features=wm_features,
            coarse_parameters=parameters,
            residual_bounds=residual_bounds,
            residual_state=residual_state,
            safety_context=safety_context,
        )
        self.last_context = context
        return context

    def _vehicle_geometry(self, vehicle) -> tuple[float, float, float]:
        def positive(value, fallback):
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = float("nan")
            return value if np.isfinite(value) and value > 0.0 else float(fallback)

        fallback_width = float(self.mppi_cfg.get("kinematic_vehicle_width", 1.852))
        fallback_wheelbase = float(self.mppi_cfg.get("kinematic_wheelbase", 2.46894))
        fallback_steering = float(
            self.mppi_cfg.get("kinematic_max_steering_angle", 0.6981317)
        )
        vehicle_width = positive(getattr(vehicle, "WIDTH", None), fallback_width)
        front = getattr(vehicle, "FRONT_WHEELBASE", None)
        rear = getattr(vehicle, "REAR_WHEELBASE", None)
        try:
            wheelbase = float(front) + float(rear)
        except (TypeError, ValueError):
            wheelbase = fallback_wheelbase
        wheelbase = positive(wheelbase, fallback_wheelbase)
        try:
            max_steering_angle = np.deg2rad(float(vehicle.max_steering))
        except (AttributeError, TypeError, ValueError):
            max_steering_angle = fallback_steering
        max_steering_angle = positive(max_steering_angle, fallback_steering)
        return vehicle_width, wheelbase, max_steering_angle

    def _corridor_context(self, refined, vehicle) -> CorridorContext:
        # H controls a_t...a_{t+H-1} produce H post-transition states. The
        # H+1-th control belongs only to terminal Q and is not integrated here.
        query = (np.arange(self.horizon, dtype=np.float64) + 1.0) * self.control_dt
        source_t = np.asarray(refined.t, dtype=np.float64)
        source_d = np.asarray(refined.d, dtype=np.float64)
        refined_d = np.interp(np.minimum(query, source_t[-1]), source_t, source_d)
        try:
            speed = float(np.linalg.norm(np.asarray(vehicle.velocity, dtype=float)[:2]))
        except Exception:
            speed = max(0.0, float(getattr(vehicle, "speed", 0.0)))
        planner = self.structured.planner
        vehicle_width, wheelbase, max_steering_angle = self._vehicle_geometry(vehicle)
        safe_min, safe_max = planner.last_lateral_bounds
        lattice_width = float(planner.last_vehicle_width or vehicle_width)
        lattice_margin = float(planner.config.get("frenet_lane_margin", 0.0))
        # last_lateral_bounds is a target/vehicle-center range, already inset by
        # half the Lattice vehicle width and frenet_lane_margin. Undo that inset
        # to recover raw road edges; the kinematic filter then applies the current
        # vehicle width and its own safety margin exactly once.
        road_min = float(safe_min) - 0.5 * lattice_width - lattice_margin
        road_max = float(safe_max) + 0.5 * lattice_width + lattice_margin
        return CorridorContext(
            refined_lateral=torch.as_tensor(refined_d, dtype=torch.float32, device=self.device),
            lane_width=float(self.structured.planner.last_lane_width),
            road_min=float(road_min),
            road_max=float(road_max),
            initial_speed=speed,
            vehicle_width=vehicle_width,
            wheelbase=wheelbase,
            max_steering_angle=max_steering_angle,
        )

    def _selected_sequence_xy(
        self,
        refined,
        baseline_actions: torch.Tensor,
        selected_actions: torch.Tensor,
        corridor_context: CorridorContext,
    ) -> np.ndarray:
        """Map the selected MPPI sequence to world XY using its corridor rollout."""
        explicit_steps = self.horizon
        aligned = align_frenet_path(
            refined,
            required_steps=explicit_steps + 1,
            control_dt=self.control_dt,
        )
        rollout_kwargs = {
            "wheelbase": corridor_context.wheelbase,
            "max_steering_angle": corridor_context.max_steering_angle,
        }
        selected_lateral = self.local_mppi.corridor.rollout(
            selected_actions[:explicit_steps].unsqueeze(0),
            corridor_context.initial_speed,
            **rollout_kwargs,
        )[0]
        baseline_lateral = self.local_mppi.corridor.rollout(
            baseline_actions[:explicit_steps].unsqueeze(0),
            corridor_context.initial_speed,
            **rollout_kwargs,
        )[0]
        lateral_delta = (selected_lateral - baseline_lateral).detach().cpu().numpy()
        headings = aligned.headings[1:]
        normals = np.column_stack((-np.sin(headings), np.cos(headings)))
        predicted = aligned.positions[1:] + lateral_delta[:, None] * normals
        return np.vstack((aligned.positions[0], predicted))

    @torch.no_grad()
    def _verified_alternatives(self, context, vehicle):
        paths = list(context.selection.feasible)[:4]
        actions = [context.coarse_actions]
        for path in paths:
            if path is not context.selection.coarse_path:
                actions.append(self.action_adapter.path_to_actions(path, vehicle,
                    required_steps=self.horizon + 1, control_dt=self.control_dt, device=self.device))
        return torch.stack(actions)

    @torch.no_grad()
    def refine_and_plan(
        self,
        context: MPRPlanContext,
        vehicle,
        *,
        eval_mode: bool,
        global_step: int,
        force_zero_residual: bool = False,
    ) -> MPRPlanResult:
        started = time.perf_counter()
        stage = self.stage_for_step(global_step)
        coarse = context.selection.coarse_path
        residual_warmup = False
        coarse_parameters = np.asarray(
            [coarse.target_d, coarse.target_speed, coarse.horizon], dtype=np.float32
        )
        latent_refined_parameters = None
        if force_zero_residual or not stage.use_residual:
            normalized_residual = torch.zeros((1, 2), device=self.device)
            physical_residual = normalized_residual.clone()
            mean = normalized_residual.clone()
            log_std = normalized_residual.clone()
        else:
            if context.residual_state is None:
                raise RuntimeError("Stage C requires current residual SAC state.")
            if self.residual_action_provider is None:
                raise RuntimeError("Stage C requires a residual SAC action provider.")
            policy_output: ResidualPolicyOutput = self.residual_action_provider(
                context.residual_state,
                eval_mode=eval_mode,
            )
            normalized_residual = policy_output.action.to(self.device, dtype=torch.float32)
            if normalized_residual.ndim == 1:
                normalized_residual = normalized_residual.unsqueeze(0)
            if self.residual_mapping == "linear":
                physical_residual = self.residual_action_adapter.to_physical(
                    normalized_residual,
                    context.residual_bounds,
                )
            else:
                lower_bounds, upper_bounds = self.residual_parameter_bounds()
                coarse_dv = torch.as_tensor(
                    coarse_parameters[:2].reshape(1, 2),
                    dtype=torch.float32,
                    device=self.device,
                )
                refined_dv, physical_residual = self.residual_action_adapter.refine_parameters(
                    normalized_residual,
                    coarse_dv,
                    lower_bounds,
                    upper_bounds,
                    self.latent_tanh_scale,
                    self.latent_tanh_eps,
                )
                latent_refined_parameters = coarse_parameters.copy()
                latent_refined_parameters[:2] = refined_dv.squeeze(0).detach().cpu().numpy()
            mean = policy_output.mean.to(self.device, dtype=torch.float32)
            log_std = policy_output.log_std.to(self.device, dtype=torch.float32)
            if mean.ndim == 1:
                mean = mean.unsqueeze(0)
            if log_std.ndim == 1:
                log_std = log_std.unsqueeze(0)
            residual_warmup = bool(policy_output.warmup)
            self.residual_attempt_count += 1

        requested = physical_residual.squeeze(0).detach().cpu().numpy()
        if latent_refined_parameters is None:
            refined_parameters = self.apply_residual_parameters(coarse_parameters, requested)
        else:
            refined_parameters = self.clamp_parameters(latent_refined_parameters)
            refined_parameters[2] = coarse_parameters[2]
        effective_residual = refined_parameters[:2] - coarse_parameters[:2]
        refined = coarse
        residual_valid = True
        fallback_reason = ""
        if np.any(np.abs(effective_residual) > 1e-8):
            if self.safety is not None and stage.use_mppi:
                generated, fallback_reason = self.structured.regenerate(refined_parameters, defer_collision_to_mppi=True)
            else:
                generated, fallback_reason = self.structured.regenerate(refined_parameters)
            if generated is None:
                residual_valid = False
                self.residual_invalid_count += 1
                effective_residual[:] = 0.0
            else:
                refined = generated
        refined_actions = self.action_adapter.path_to_actions(
            refined,
            vehicle,
            required_steps=self.horizon + 1,
            control_dt=self.control_dt,
            device=self.device,
        )

        corridor_context = self._corridor_context(refined, vehicle)
        if stage.use_mppi:
            if context.latent is None:
                raise RuntimeError("Stage B/C MPPI requires an encoded current observation.")
            mppi = self.local_mppi.plan(
                context.latent,
                refined_actions,
                eval_mode=eval_mode,
                corridor_context=corridor_context,
                safety_context=context.safety_context,
                alternatives=(lambda: self._verified_alternatives(context, vehicle)),
            )
            self.mppi_call_count += 1
            self.baseline_selected_count += int(mppi.baseline_selected)
            action = mppi.action
            initial_mean, final_mean = mppi.initial_mean, mppi.final_mean
            initial_std, final_std = mppi.initial_std, mppi.final_std
            baseline_wm_value = float(mppi.baseline_wm_value.cpu())
            selected_wm_value = float(mppi.selected_wm_value.cpu())
            wm_value_gain = float(mppi.wm_value_gain.cpu())
            baseline_score = float(mppi.baseline_score.cpu())
            selected_score = float(mppi.selected_score.cpu())
            planner_score_gain = float(mppi.planner_score_gain.cpu())
            baseline_selected = float(mppi.baseline_selected)
            reject_count = float(mppi.corridor_reject_count)
            reject_rate = float(mppi.corridor_reject_rate)
            max_lateral_deviation = float(mppi.max_lateral_deviation)
            max_delta = mppi.max_action_delta
            selected_actions = mppi.selected_sequence
        else:
            action = refined_actions[0]
            initial_mean = final_mean = refined_actions
            initial_std = final_std = torch.zeros_like(refined_actions)
            if context.latent is None:
                # Stage A is Lattice-only: no planning-time encoder/WM rollout.
                baseline_wm_value = selected_wm_value = float("nan")
            else:
                refined_value = self.evaluator.evaluate_trajectory_consequence(
                    context.latent, refined_actions.unsqueeze(0)
                ).total_value[0]
                baseline_wm_value = selected_wm_value = float(refined_value.cpu())
            baseline_score = selected_score = baseline_wm_value
            wm_value_gain = planner_score_gain = 0.0
            baseline_selected = 1.0
            reject_count = reject_rate = max_lateral_deviation = 0.0
            max_delta = torch.zeros(self.action_dim, device=self.device)
            selected_actions = refined_actions

        final_mppi_xy = self._selected_sequence_xy(
            refined,
            refined_actions,
            selected_actions,
            corridor_context,
        )
        diagnostics = {}
        stage_actions = torch.stack((context.coarse_actions, refined_actions, selected_actions))
        geometry = None
        if self.safety is not None and context.safety_context is not None:
            geometry = self.safety.check(stage_actions, context.safety_context)
            final_mppi_xy = geometry.positions[2].detach().cpu().numpy()
        stage_consequence = None
        if context.latent is not None:
            stage_consequence = self.evaluator.evaluate_trajectory_consequence(context.latent, stage_actions)
        for index, name in enumerate(("lattice", "sac", "mppi")):
            cost = (float(stage_consequence.cumulative_cost[index])
                    if stage_consequence is not None and stage_consequence.cost_validated else float("nan"))
            diagnostics[f"mpr/{name}_predicted_cost"] = cost
            diagnostics[f"mpr/{name}_value"] = (float(stage_consequence.total_value[index])
                                                      if stage_consequence is not None else float("nan"))
            if geometry is not None:
                diagnostics[f"mpr/{name}_safe"] = float(geometry.valid[index])
                for key, value in geometry.summary(index).items():
                    diagnostics[f"mpr/{name}_{key}"] = value
        diagnostics["mpr/risk_reduction_sac"] = diagnostics["mpr/lattice_predicted_cost"] - diagnostics["mpr/sac_predicted_cost"]
        diagnostics["mpr/risk_reduction_mppi"] = diagnostics["mpr/sac_predicted_cost"] - diagnostics["mpr/mppi_predicted_cost"]
        diagnostics["mpr/mppi_correction_magnitude"] = float((selected_actions - refined_actions).square().mean().sqrt())
        diagnostics["mpr/sac_risk_valid"] = float(residual_valid and stage_consequence is not None and stage_consequence.cost_validated)
        diagnostics["mpr/sac_trajectory_is_fallback"] = float(not residual_valid)
        if not residual_valid:
            diagnostics["mpr/sac_predicted_cost"] = float("nan")
            diagnostics["mpr/risk_reduction_sac"] = diagnostics["mpr/risk_reduction_mppi"] = float("nan")
        if stage.use_mppi:
            diagnostics.update({f"mpr/{key}": value for key, value in mppi.safety_metrics.items()})
        diagnostics["mpr/safe_candidate_ratio"] = 1 - reject_rate
        if geometry is not None and not bool(geometry.valid[2]) and stage.use_mppi:
            fallback_reason = "no_verified_safe_candidate:minimum_risk_emergency"

        paths = list(context.selection.candidates)
        if not any(path is refined for path in paths):
            paths.append(refined)
        self.structured.planner.last_candidates = paths
        self.structured.planner.last_selected_index = next(
            index for index, path in enumerate(paths) if path is refined
        )
        invalid_rate = self.residual_invalid_count / max(1, self.residual_attempt_count)
        baseline_rate = self.baseline_selected_count / max(1, self.mppi_call_count)
        elapsed_ms = 1000.0 * (time.perf_counter() - started)
        coarse_value = (
            float(context.coarse_consequence.total_value[0].cpu())
            if context.coarse_consequence is not None
            else float("nan")
        )
        result = MPRPlanResult(
            action=action.detach().cpu(),
            coarse_path=coarse,
            refined_path=refined,
            coarse_actions=context.coarse_actions.detach(),
            refined_actions=refined_actions.detach(),
            requested_normalized_residual=normalized_residual.squeeze(0).detach(),
            requested_physical_residual=physical_residual.squeeze(0).detach(),
            residual=torch.as_tensor(effective_residual, device=self.device),
            residual_mean=mean.squeeze(0).detach(),
            residual_log_std=log_std.squeeze(0).detach(),
            residual_valid=residual_valid,
            fallback_reason=fallback_reason,
            residual_warmup=residual_warmup,
            coarse_consequence=context.coarse_consequence,
            stage=stage,
            metrics={
                **diagnostics,
                "mpr/stage": float(ord(stage.name) - ord("A")),
                "mpr/candidate_count": float(len(context.selection.candidates)),
                "mpr/feasible_count": float(len(context.selection.feasible)),
                "mpr/feasible_ratio": float(len(context.selection.feasible) / max(len(context.selection.candidates), 1)),
                "mpr/used_lattice_fallback": float(context.selection.used_fallback),
                "mpr/selected_index": float(context.selection.selected_index),
                "mpr/coarse_value": coarse_value,
                "mpr/baseline_wm_value": baseline_wm_value,
                "mpr/selected_wm_value": selected_wm_value,
                "mpr/wm_value_gain": wm_value_gain,
                "mpr/baseline_score": baseline_score,
                "mpr/selected_score": selected_score,
                "mpr/planner_score_gain": planner_score_gain,
                # Compatibility aliases: all three are planner-score quantities.
                "mpr/baseline_value": baseline_score,
                "mpr/final_value": selected_score,
                "mpr/planner_gain": planner_score_gain,
                "mpr/residual_d": float(effective_residual[0]),
                "mpr/residual_v": float(effective_residual[1]),
                "mpr/requested_residual_d": float(requested[0]),
                "mpr/requested_residual_v": float(requested[1]),
                "residual_rl/action_d": float(normalized_residual[0, 0].detach().cpu()),
                "residual_rl/action_v": float(normalized_residual[0, 1].detach().cpu()),
                "residual_rl/warmup_action": float(residual_warmup),
                "mpr/residual_bound_d": float(context.residual_bounds[0, 0].cpu()),
                "mpr/residual_bound_v": float(context.residual_bounds[0, 1].cpu()),
                "mpr/residual_valid": float(residual_valid),
                "mpr/residual_invalid_count": float(self.residual_invalid_count),
                "mpr/residual_invalid_rate": float(invalid_rate),
                "mpr/initial_std_steer": float(initial_std[:, 0].mean().cpu()),
                "mpr/initial_std_throttle": float(initial_std[:, 1].mean().cpu()),
                "mpr/final_std_steer": float(final_std[:, 0].mean().cpu()),
                "mpr/final_std_throttle": float(final_std[:, 1].mean().cpu()),
                "mpr/max_delta_steer": float(max_delta[0].cpu()),
                "mpr/max_delta_throttle": float(max_delta[1].cpu()),
                "mpr/max_lateral_deviation": max_lateral_deviation,
                "mpr/corridor_reject_count": reject_count,
                "mpr/corridor_reject_rate": reject_rate,
                "mpr/baseline_selected": baseline_selected,
                "mpr/baseline_selected_rate": float(baseline_rate),
                "mpr/planning_ms": elapsed_ms,
            },
            debug={
                "selected_mppi_actions": selected_actions.detach().cpu(),
                "final_mppi_xy": final_mppi_xy,
                "initial_mppi_mean": initial_mean.detach().cpu(),
                "final_mppi_mean": final_mean.detach().cpu(),
                "initial_mppi_std": initial_std.detach().cpu(),
                "final_mppi_std": final_std.detach().cpu(),
            },
        )
        self.last_result = result
        return result

    def plan(
        self,
        observation,
        vehicle,
        *,
        global_step: int,
        t0=False,
        eval_mode=False,
        force_zero_residual=False,
    ):
        if t0:
            self.reset()
        stage = self.stage_for_step(global_step)
        context = self.prepare(
            observation,
            vehicle,
            need_world_model_features=stage.use_residual or stage.use_mppi,
        )
        return self.refine_and_plan(
            context,
            vehicle,
            eval_mode=eval_mode,
            global_step=global_step,
            force_zero_residual=force_zero_residual,
        )


__all__ = ["MPRMPCPlanner"]
