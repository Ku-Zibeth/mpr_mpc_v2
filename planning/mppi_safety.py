"""Independent candidate certification under a short-horizon kinematic/CV model.

This never constructs SAC inputs. It reuses the legacy swept-OBB primitive,
but owns its settings, road cache and per-observation traffic snapshot.
"""

from dataclasses import dataclass, fields
import math

import numpy as np
import shapely
import torch

from .safety import SafetyContext, SafetyResult, TrajectorySafetyEvaluator, _corners
from .swept_collision import _position_bound
from .swept_collision import swept_collisions


@dataclass
class MPPISafetyContext(SafetyContext):
    brake_action_mapping_verified: bool = False


class MPPISafetyFilter:
    def __init__(self, cfg, *, control_dt):
        self.cfg = dict(cfg)
        # GEOS lives on CPU. One batched copy per cache miss also avoids dozens
        # of small CUDA launches and CPU synchronizations in swept geometry.
        self.geometry = TrajectorySafetyEvaluator(cfg, control_dt=control_dt, device="cpu")
        self.safe_roads = {}
        for name, default in (("max_acceleration", 4.), ("max_deceleration", 6.),
                              ("max_steering_rate", 7.0), ("max_lateral_acceleration", 6.)):
            value = float(cfg.get(name, default))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"MPPI safety {name} must be positive and finite.")
        for name in ("hard_ttc", "hard_clearance"):
            value = float(cfg.get(name, 0.))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"MPPI safety {name} must be finite and nonnegative.")
        for name in ("obstacle_acceleration_bound", "obstacle_velocity_error", "braking_distance_deceleration"):
            value = float(cfg.get(name, 0.))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"MPPI safety {name} must be finite and nonnegative.")
        self.emergency_throttle = float(cfg.get("emergency_throttle", -1.))
        if not math.isfinite(self.emergency_throttle) or not -1 <= self.emergency_throttle < 0:
            raise ValueError("Emergency throttle must be in [-1,0) for verified non-reversing MetaDrive vehicles.")

    def reset(self):
        self.geometry.reset()
        self.safe_roads.clear()

    def context(self, planner, vehicle, *, width, wheelbase, max_steering, snapshot=None):
        config = getattr(vehicle, "config", {})
        if bool(getattr(vehicle, "enable_reverse", config.get("enable_reverse", True))):
            raise RuntimeError("Cannot certify braking when MetaDrive reverse is enabled or unknown.")
        source = snapshot or self.geometry.context(planner, vehicle, width=width,
                                                    wheelbase=wheelbase, max_steering=max_steering)
        values = {field.name: getattr(source, field.name) for field in fields(SafetyContext)}
        values["cache"] = {}
        # BaseVehicle._apply_throttle_brake: negative action[1] applies setBrake
        # (and zero engine force) exactly when enable_reverse is false.
        return MPPISafetyContext(**values, brake_action_mapping_verified=True)

    @torch.no_grad()
    def check(self, actions, context):
        if context is None or context.road is None:
            raise RuntimeError("MPPI independent safety requires a live safety_context with road geometry.")
        if actions.ndim != 3 or actions.shape[-1] != 2 or actions.shape[1] < 2 or not len(actions):
            raise ValueError("MPPI safety actions must be nonempty [N,H+1,2].")
        controls = actions.detach().to("cpu")
        scalars = [context.heading, context.speed, context.length, context.width,
                   context.wheelbase, context.max_steering]
        if (not np.isfinite(scalars).all() or not np.isfinite(context.position).all()
                or not np.isfinite(context.previous_action).all() or context.speed < 0
                or min(context.length, context.width, context.wheelbase) <= 0
                or not 0 < context.max_steering < math.pi/2):
            raise ValueError("MPPI safety requires finite, physically valid measured vehicle state.")
        for obj in context.obstacles:
            if (not np.isfinite([*obj.position, *obj.velocity, obj.heading, obj.length, obj.width]).all()
                    or min(obj.length, obj.width) <= 0):
                raise RuntimeError("MPPI traffic snapshot contains invalid measured geometry/velocity.")
        positions, headings, speeds = self.geometry.rollout(controls, context)
        finite = torch.isfinite(controls).all((1, 2)) & (controls.abs() <= 1).all((1, 2))
        previous = controls.new_tensor(context.previous_action).view(1, 1, 2).expand(len(controls), -1, -1)
        explicit = controls[:, :-1]
        rate = torch.diff(torch.cat((previous, explicit), 1)[:, :, 0], dim=1).abs()*context.max_steering/self.geometry.dt
        curvature = (explicit[:, :, 0]*context.max_steering).tan().abs()/context.wheelbase
        acceleration = torch.maximum(speeds[:, :-1], speeds[:, 1:]).square()*curvature.repeat_interleave(self.geometry.substeps, 1)
        kinematic = (~finite | (rate > float(self.cfg.get("max_steering_rate", 7.0))).any(1)
                     | (acceleration > float(self.cfg.get("max_lateral_acceleration", 6.))).any(1)
                     | (headings.diff(dim=1).abs() >= math.pi).any(1))
        # Enclose the full translation/rotation between adjacent substeps, not
        # just the center or endpoint corners. The same arc-error bound is used
        # by the continuous collision primitive.
        middle = (headings[:, :-1] + headings[:, 1:]) / 2
        length = context.length + 2*self.geometry.model_margin
        width = context.width + 2*self.geometry.model_margin
        vertices = torch.cat((_corners(positions[:, :-1], middle, length, width),
                              _corners(positions[:, 1:], middle, length, width)), dim=-2)
        turn = headings[:, 1:] - headings[:, :-1]
        pad = _position_bound((positions[:, 1:]-positions[:, :-1]).norm(dim=-1), turn,
                              torch.zeros_like(turn), torch.ones_like(turn),
                              speeds[:, :-1], speeds[:, 1:], turn.abs() < math.pi)
        pad += math.hypot(length, width) * (turn.abs().clamp_max(2*math.pi)/4).sin()
        key = (id(context.road), float(self.cfg.get("road_margin", .15)))
        if key not in self.safe_roads:
            safe_road = context.road.buffer(-key[1])
            shapely.prepare(safe_road)
            self.safe_roads[key] = (context.road, safe_road)
        safe_road = self.safe_roads[key][1]
        lower, upper = vertices.amin(-2)-pad[..., None], vertices.amax(-2)+pad[..., None]
        # A whole-trajectory enclosing box certifies easy interior cases with
        # N GEOS queries, not N*substeps. Ambiguous paths still receive every
        # interval's exact conservative swept-envelope test below.
        whole = torch.cat((lower.amin(1), upper.amax(1)), -1).numpy()
        whole_inside = np.asarray(shapely.covers(safe_road, shapely.box(*whole.T)))
        first_offroad = np.zeros(len(actions), dtype=bool)
        paths = np.flatnonzero(~whole_inside)
        if len(paths):
            bounds = torch.cat((lower[paths], upper[paths]), -1).numpy().reshape(-1, 4)
            inside = np.asarray(shapely.covers(safe_road, shapely.box(*bounds.T)))
            ambiguous = ~inside
            if ambiguous.any():
                hulls = shapely.convex_hull(shapely.multipoints(vertices[paths].numpy().reshape(-1, 8, 2)[ambiguous]))
                # GEOS buffers are inscribed polygons; expand to enclose the
                # required uncertainty disk, never shave a safety bound off.
                swept = shapely.buffer(hulls, pad[paths].numpy().reshape(-1)[ambiguous]/math.cos(math.pi/32)+1e-6, quad_segs=8)
                inside[ambiguous] = shapely.covers(safe_road, swept)
            whole_inside[paths] = inside.reshape(len(paths), -1).all(1)
            first_offroad[paths] = ~inside.reshape(len(paths), -1)[:, :self.geometry.substeps].all(1)
        offroad = torch.from_numpy(~whole_inside)
        interval = controls.new_full((len(actions), positions.shape[1]-1), torch.inf)
        if context.obstacles:
            tensor_key = ("mppi_measured_traffic", controls.dtype)
            if tensor_key not in context.cache:
                context.cache[tensor_key] = tuple(controls.new_tensor(values) for values in (
                    np.asarray([o.position for o in context.obstacles]), np.asarray([o.velocity for o in context.obstacles]),
                    [o.heading for o in context.obstacles], [o.length for o in context.obstacles],
                    [o.width for o in context.obstacles]))
            other_xy, other_v, other_heading, lengths, widths = context.cache[tensor_key]
            acceleration_bound = float(self.cfg.get("obstacle_acceleration_bound", 0.))
            velocity_error = float(self.cfg.get("obstacle_velocity_error", 0.))
            if acceleration_bound or velocity_error:
                interval = swept_collisions(positions, headings, other_xy, other_v, other_heading, lengths, widths,
                    ego_length=length, ego_width=width, dt=self.geometry.dt/self.geometry.substeps,
                    max_depth=self.geometry.swept_max_depth, angle_tolerance=self.geometry.swept_angle_tolerance,
                    position_tolerance=self.geometry.swept_position_tolerance, curved=True, speeds=speeds,
                    obstacle_acceleration_bound=acceleration_bound, obstacle_velocity_error=velocity_error)
            else:
                interval = self.geometry._swept_check(positions, headings, speeds, other_xy, other_v,
                    other_heading, lengths, widths, length, width, skip=None)
        collision = torch.isfinite(interval).any(1)
        times = torch.arange(interval.shape[1])*self.geometry.dt/self.geometry.substeps
        ttc = (times[None]+interval*self.geometry.dt/self.geometry.substeps).amin(1)
        valid = ~offroad & ~kinematic & ~collision
        # Optional additional same-lane longitudinal stopping envelope. A
        # positive value must be a calibrated LOWER braking bound, not the
        # kinematic point estimate. Default stays diagnostic-only.
        deceleration = float(self.cfg.get("braking_distance_deceleration", 0.))
        if deceleration and context.obstacles:
            direction = positions.new_tensor([math.cos(context.heading), math.sin(context.heading)])
            normal = torch.stack((-direction[1], direction[0]))
            relative = other_xy-context.position
            gap = relative@direction-(length+lengths)/2
            lateral = (relative@normal).abs()
            closing = (context.speed-other_v@direction).clamp_min(0)
            required = closing*self.geometry.dt+closing.square()/(2*deceleration)
            blocked = (gap >= 0) & (gap < required) & (lateral < (width+widths)/2)
            # Reject unbraked sequences; full braking still goes through all
            # road/collision checks and may be explicitly unverified.
            valid &= ~blocked.any() | (explicit[:, 0, 1] <= self.emergency_throttle)
        unavailable = controls.new_full((len(controls),), float("nan"))
        clearance = unavailable.masked_fill(collision, 0.)
        if float(self.cfg.get("hard_ttc", 0.)) > 0 or float(self.cfg.get("hard_clearance", 0.)) > 0:
            details = self.geometry.check(controls, context)
            valid &= details.valid
            clearance, ttc = details.min_clearance, details.min_ttc
        return SafetyResult(valid, collision, offroad, kinematic, clearance, ttc,
                            unavailable, torch.zeros_like(ttc), positions, unavailable, torch.isfinite(interval),
                            torch.isfinite(interval[:, :self.geometry.substeps]).any(1), torch.from_numpy(first_offroad))

    def stopping_diagnostics(self, context, horizon):
        # Model estimate only: MetaDrive Bullet brake force/mass/tire friction
        # are not a measured deceleration bound. Report uncovered stop horizon.
        deceleration = float(self.cfg.get("braking_distance_deceleration", 0.) or self.cfg.get("max_deceleration", 6.))
        stop_time = context.speed/deceleration
        t = horizon*self.geometry.dt
        return dict(stopping_distance_m=context.speed*self.geometry.dt+context.speed**2/(2*deceleration),
                    stopping_time_s=stop_time, stopping_horizon_insufficient=float(stop_time > t),
                    obstacle_uncertainty_radius_m=float(self.cfg.get("obstacle_velocity_error", 0.))*t
                        +.5*float(self.cfg.get("obstacle_acceleration_bound", 0.))*t*t)

    def braking_sequence(self, context, *, horizon, device, dtype):
        # This verifies the actuator convention only, not collision avoidance
        # or the ability to stop within the model's finite prediction horizon.
        if not getattr(context, "brake_action_mapping_verified", False):
            raise RuntimeError("Emergency braking action mapping has not been verified from the live MetaDrive vehicle.")
        action = torch.tensor([float(np.clip(context.previous_action[0], -1, 1)), self.emergency_throttle],
                              dtype=dtype, device=device)
        return action.expand(horizon+1, -1).clone()
