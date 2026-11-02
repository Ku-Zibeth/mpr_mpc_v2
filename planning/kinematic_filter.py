"""Batched short-horizon geometry filter for local MPPI candidates."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class CorridorContext:
    """Geometry at planning time; road bounds are raw lane/road edges."""

    refined_lateral: torch.Tensor
    lane_width: float
    road_min: float
    road_max: float
    initial_speed: float
    vehicle_width: float
    wheelbase: float
    max_steering_angle: float


@dataclass(frozen=True)
class CorridorResult:
    valid: torch.Tensor
    max_lateral_deviation: torch.Tensor
    lateral: torch.Tensor


class BatchedKinematicCorridor:
    """Use a bicycle approximation only to reject geometrically unsafe actions."""

    def __init__(self, cfg, *, control_dt: float, device):
        self.device = torch.device(device)
        self.dt = float(control_dt)
        self.ratio = float(cfg.get("lateral_corridor_ratio", 0.5))
        self.safety_margin = float(cfg.get("lateral_safety_margin", 0.15))
        self.default_wheelbase = float(cfg.get("kinematic_wheelbase", 2.46894))
        self.default_max_steering_angle = float(
            cfg.get("kinematic_max_steering_angle", 0.6981317)
        )
        self.default_vehicle_width = float(cfg.get("kinematic_vehicle_width", 1.852))
        self.max_acceleration = float(cfg.get("kinematic_max_acceleration", 4.0))
        self.max_deceleration = float(cfg.get("kinematic_max_deceleration", 6.0))
        if min(
            self.dt,
            self.ratio,
            self.default_wheelbase,
            self.default_max_steering_angle,
            self.default_vehicle_width,
        ) <= 0:
            raise ValueError("Kinematic corridor positive parameters must be > 0.")

    @staticmethod
    def _valid_or(value: float, fallback: float) -> float:
        value = float(value)
        return value if math.isfinite(value) and value > 0.0 else float(fallback)

    def rollout(
        self,
        actions: torch.Tensor,
        initial_speed: float,
        *,
        wheelbase: float,
        max_steering_angle: float,
    ) -> torch.Tensor:
        if actions.ndim != 3 or actions.shape[-1] != 2:
            raise ValueError("actions must have shape [N,H,2].")
        count, steps, _ = actions.shape
        wheelbase = self._valid_or(wheelbase, self.default_wheelbase)
        max_steering_angle = self._valid_or(
            max_steering_angle, self.default_max_steering_angle
        )
        speed = actions.new_full((count,), max(0.0, float(initial_speed)))
        yaw = actions.new_zeros(count)
        lateral_position = actions.new_zeros(count)
        traces = []
        for step in range(steps):
            steering = actions[:, step, 0] * max_steering_angle
            throttle = actions[:, step, 1]
            acceleration = torch.where(
                throttle >= 0.0,
                throttle * self.max_acceleration,
                throttle * self.max_deceleration,
            )
            yaw = yaw + self.dt * speed / wheelbase * torch.tan(steering)
            lateral_position = lateral_position + self.dt * speed * torch.sin(yaw)
            speed = (speed + self.dt * acceleration).clamp_min(0.0)
            traces.append(lateral_position)
        return torch.stack(traces, dim=1)

    def check(
        self,
        actions: torch.Tensor,
        baseline: torch.Tensor,
        context: CorridorContext,
    ) -> CorridorResult:
        actions = actions.to(self.device)
        baseline = baseline.to(self.device)
        refined_d = context.refined_lateral.to(self.device, dtype=actions.dtype)
        explicit_steps = actions.shape[1] - 1
        if explicit_steps <= 0 or refined_d.shape != (explicit_steps,):
            raise ValueError(
                "refined_lateral must contain H post-transition states for [H+1] actions."
            )
        rollout_kwargs = {
            "wheelbase": context.wheelbase,
            "max_steering_angle": context.max_steering_angle,
        }
        # actions[:, H] is only the terminal-Q action. Geometry integrates a_t
        # through a_{t+H-1}, producing states at t+1 through t+H.
        candidate_y = self.rollout(
            actions[:, :explicit_steps], context.initial_speed, **rollout_kwargs
        )
        baseline_y = self.rollout(
            baseline[:explicit_steps].unsqueeze(0),
            context.initial_speed,
            **rollout_kwargs,
        )[0]
        lateral_deviation = candidate_y - baseline_y.unsqueeze(0)
        predicted_d = refined_d.unsqueeze(0) + lateral_deviation

        corridor_radius = self.ratio * float(context.lane_width)
        vehicle_width = self._valid_or(
            context.vehicle_width, self.default_vehicle_width
        )
        half_vehicle_width = 0.5 * vehicle_width
        road_lower = float(context.road_min) + half_vehicle_width + self.safety_margin
        road_upper = float(context.road_max) - half_vehicle_width - self.safety_margin
        lower = torch.maximum(
            refined_d - corridor_radius,
            refined_d.new_full(refined_d.shape, road_lower),
        )
        upper = torch.minimum(
            refined_d + corridor_radius,
            refined_d.new_full(refined_d.shape, road_upper),
        )
        valid = ((predicted_d >= lower) & (predicted_d <= upper)).all(dim=1)
        return CorridorResult(
            valid=valid,
            max_lateral_deviation=lateral_deviation.abs().amax(dim=1),
            lateral=predicted_d,
        )


__all__ = ["BatchedKinematicCorridor", "CorridorContext", "CorridorResult"]
