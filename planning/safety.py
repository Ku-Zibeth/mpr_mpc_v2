"""Legacy SAC geometric input features, not a native-MPPI candidate filter.

Ego motion uses a configurable kinematic bicycle approximation; surrounding
objects use their measured constant velocities and headings. This is a model
check, not a guarantee against unobserved motion or Bullet integration error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import shapely
from shapely.errors import GEOSException
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from shapely.prepared import prep
from shapely.validation import make_valid

from .swept_collision import swept_collisions


@dataclass
class SafetyContext:
    position: np.ndarray
    heading: float
    speed: float
    length: float
    width: float
    wheelbase: float
    max_steering: float
    previous_action: np.ndarray
    road: Any
    obstacles: tuple
    cache: dict = field(default_factory=dict, repr=False)


@dataclass
class SafetyResult:
    valid: torch.Tensor
    collision: torch.Tensor
    offroad: torch.Tensor
    kinematic_violation: torch.Tensor
    min_clearance: torch.Tensor
    min_ttc: torch.Tensor
    boundary_margin: torch.Tensor
    soft_risk: torch.Tensor
    positions: torch.Tensor
    boundary_risk: torch.Tensor | None = None
    interval_collision: torch.Tensor | None = None
    first_collision: torch.Tensor | None = None
    first_offroad: torch.Tensor | None = None

    def summary(self, index=0):
        return {key: float(getattr(self, key)[index].cpu())
                for key in ("min_clearance", "min_ttc", "boundary_margin", "soft_risk", "boundary_risk")}


def _corners(positions, headings, length, width):
    forward = torch.stack((headings.cos(), headings.sin()), -1)
    normal = torch.stack((-headings.sin(), headings.cos()), -1)
    return torch.stack([positions + x * length * forward / 2 + y * width * normal / 2
                        for x, y in ((1, 1), (1, -1), (-1, -1), (-1, 1))], -2)


def _vertex_edge_distance(vertices, edges):
    start = edges.unsqueeze(-3)
    direction = (torch.roll(edges, -1, -2) - edges).unsqueeze(-3)
    delta = vertices.unsqueeze(-2) - start
    ratio = (delta * direction).sum(-1) / direction.square().sum(-1).clamp_min(1e-12)
    error = delta - ratio.clamp(0, 1).unsqueeze(-1) * direction
    return error.square().sum(-1).amin(dim=(-1, -2)).sqrt()


def _road_polygon_parts(geometry):
    if geometry is None or geometry.is_empty:
        raise RuntimeError("MetaDrive supplied an empty or missing lane road geometry.")
    if not np.isfinite(shapely.get_coordinates(geometry)).all():
        raise RuntimeError("MetaDrive supplied non-finite lane road coordinates.")
    try:
        repaired = geometry if geometry.is_valid else make_valid(geometry)
    except GEOSException as error:
        raise RuntimeError("Unable to repair MetaDrive lane road geometry.") from error
    # make_valid can return polygon lobes together with collapsed line artifacts.
    # Retain every area component, but never treat those lines as drivable road.
    pending, polygons = [repaired], []
    while pending:
        part = pending.pop()
        if part.geom_type == "Polygon" and not part.is_empty:
            polygons.append(part)
        elif part.geom_type in ("MultiPolygon", "GeometryCollection"):
            pending.extend(part.geoms)
    if not polygons:
        raise RuntimeError("MetaDrive lane road geometry has no drivable polygon area.")
    return polygons


class TrajectorySafetyEvaluator:
    def __init__(self, cfg, *, control_dt: float, device):
        self.cfg = dict(cfg)
        self.dt = float(control_dt)
        self.device = torch.device(device)
        self.substeps = int(cfg.get("substeps", 5))
        self.model_margin = float(cfg.get("model_margin", 0.0))
        self.swept_max_depth = int(cfg.get("swept_max_depth", 10))
        self.swept_angle_tolerance = float(cfg.get("swept_angle_tolerance", 1e-4))
        self.swept_position_tolerance = float(cfg.get("swept_position_tolerance", 1e-4))
        if self.swept_max_depth < 0 or not np.isfinite(self.swept_angle_tolerance) or self.swept_angle_tolerance <= 0:
            raise ValueError("Swept collision depth must be nonnegative and angular tolerance positive/finite.")
        if not np.isfinite(self.swept_position_tolerance) or self.swept_position_tolerance <= 0:
            raise ValueError("Swept position tolerance must be positive and finite.")
        for name in ("soft_boundary_margin", "boundary_risk_weight", "clearance_risk_weight", "ttc_risk_weight", "road_margin"):
            value = float(cfg.get(name, .5 if name == "soft_boundary_margin" else 1. if name.endswith("weight") else .15))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if not np.isfinite([self.dt, self.model_margin]).all() or self.dt <= 0 or self.model_margin < 0 or self.substeps <= 0:
            raise ValueError("Safety timestep/substeps must be positive and model_margin finite/nonnegative.")
        self._road_cache = {}
        self._road_geometry_cache = {}

    def reset(self):
        self._road_cache.clear()
        self._road_geometry_cache.clear()

    def context(self, planner, vehicle, *, width, wheelbase, max_steering):
        if bool(getattr(vehicle, "enable_reverse", False)):
            raise ValueError("Safety braking requires MetaDrive enable_reverse=false; negative throttle would reverse otherwise.")
        navigation = getattr(vehicle, "navigation", None)
        groups = planner._navigation_lane_groups(navigation)[:4]
        lanes = list(getattr(navigation, "current_ref_lanes", None) or [])
        lanes += list(getattr(navigation, "next_ref_lanes", None) or [])
        lanes += [lane for group in groups for lane in group]
        lanes += planner._reference_lane_sequence(vehicle)
        current_lane = getattr(vehicle, "lane", None)
        if current_lane is not None:
            lanes.append(current_lane)
        # Navigation can advance to the next road while the rear footprint is
        # still on its predecessor (also true at MetaDrive's spawn-road join).
        network = getattr(getattr(navigation, "map", None), "road_network", None)
        graph = getattr(network, "graph", {})
        current_lanes = list(getattr(navigation, "current_ref_lanes", None) or [])
        if current_lane is not None:
            current_lanes.append(current_lane)
        starts = {lane.index[0] for lane in current_lanes if getattr(lane, "index", None)}
        for roads in graph.values():
            for node in starts:
                lanes.extend(roads.get(node, []))
        lanes = list({id(lane): lane for lane in lanes if lane is not None}.values())
        key = tuple(sorted(id(lane) for lane in lanes))
        if key not in self._road_cache:
            polygons = [polygon for lane in lanes
                        for polygon in _road_polygon_parts(lane.shapely_polygon)]
            if not polygons:
                raise RuntimeError("Dynamic safety requires MetaDrive navigation lane polygons.")
            road = unary_union(polygons)
            if not road.is_valid:
                road = unary_union(_road_polygon_parts(road))
            if road.is_empty or not road.is_valid or road.geom_type not in ("Polygon", "MultiPolygon"):
                raise RuntimeError("MetaDrive supplied an invalid or empty drivable road polygon.")
            self._road_cache[key] = road
        previous = np.asarray(getattr(vehicle, "current_action", [0.0, 0.0]), dtype=float).reshape(2)
        return SafetyContext(
            position=np.asarray(vehicle.position, dtype=float)[:2], heading=float(vehicle.heading_theta),
            speed=float(np.linalg.norm(np.asarray(vehicle.velocity, dtype=float)[:2])),
            length=float(getattr(vehicle, "LENGTH", 4.5)), width=width, wheelbase=wheelbase,
            max_steering=max_steering, previous_action=previous,
            road=self._road_cache[key], obstacles=tuple(planner._collect_obstacles(vehicle)),
        )

    def rollout(self, actions, context):
        controls = actions[:, :-1].nan_to_num().clamp(-1, 1)
        xy = torch.as_tensor(context.position, device=actions.device, dtype=actions.dtype).expand(len(actions), -1).clone()
        yaw = actions.new_full((len(actions),), context.heading)
        speed = actions.new_full((len(actions),), max(0.0, context.speed))
        positions, headings, speeds = [xy.clone()], [yaw.clone()], [speed.clone()]
        dt = self.dt / self.substeps
        for control in controls.unbind(1):
            steering = control[:, 0] * context.max_steering
            acceleration = control[:, 1] * torch.where(
                control[:, 1] >= 0,
                control.new_tensor(float(self.cfg.get("max_acceleration", 4.0))),
                control.new_tensor(float(self.cfg.get("max_deceleration", 6.0))),
            )
            for _ in range(self.substeps):
                next_speed = (speed + dt * acceleration).clamp_min(0)
                middle_speed = (speed + next_speed) / 2
                turn = dt * middle_speed * steering.tan() / context.wheelbase
                middle_yaw = yaw + turn / 2
                xy = xy + dt * middle_speed[:, None] * torch.stack((middle_yaw.cos(), middle_yaw.sin()), -1)
                yaw, speed = yaw + turn, next_speed
                positions.append(xy.clone())
                headings.append(yaw.clone())
                speeds.append(speed.clone())
        return torch.stack(positions, 1), torch.stack(headings, 1), torch.stack(speeds, 1)

    def _road_metrics(self, ego, road):
        road_key = (id(road), float(self.cfg.get("road_margin", .15)))
        if road_key not in self._road_geometry_cache:
            safe_road = road.buffer(-road_key[1])
            self._road_geometry_cache[road_key] = (road, safe_road, prep(safe_road), road.boundary, prep(road))
        _, safe_road, prepared, boundary, prepared_road = self._road_geometry_cache[road_key]
        corners = ego.cpu().numpy().reshape(-1, 4, 2)
        if hasattr(shapely, "polygons"):
            footprints = shapely.polygons(corners)
            road_valid = shapely.covers(safe_road, footprints)
            # Every safe-inset footprint is already inside the original road.
            inside = road_valid.copy()
            inside[~inside] = shapely.covers(road, footprints[~inside])
            margins = np.empty(len(footprints))
            margins[inside] = shapely.distance(boundary, footprints[inside])
            if (~inside).any():
                outside = shapely.distance(shapely.points(corners[~inside]), road).max(-1)
                margins[~inside] = -np.maximum(outside, 1e-6)
        else:
            road_valid, margins = [], []
            for vertices in corners:
                footprint = Polygon(vertices)
                road_valid.append(prepared.covers(footprint))
                margins.append(boundary.distance(footprint) if prepared_road.covers(footprint)
                    else -max(1e-6, max(Point(point).distance(road) for point in vertices)))
        n, samples = ego.shape[:2]
        return (torch.as_tensor(np.asarray(road_valid), device=ego.device).view(n, samples),
                ego.new_tensor(margins).view(n, samples).amin(1))

    def _swept_check(self, positions, headings, speeds, other_xy, other_v, other_heading, lengths, widths, ego_length, ego_width, skip):
        return swept_collisions(positions, headings, other_xy, other_v, other_heading, lengths, widths,
            ego_length=ego_length, ego_width=ego_width, dt=self.dt/self.substeps,
            max_depth=self.swept_max_depth, angle_tolerance=self.swept_angle_tolerance,
            position_tolerance=self.swept_position_tolerance, skip=skip, curved=True, speeds=speeds)

    @torch.no_grad()
    def check(self, actions, context: SafetyContext):
        if actions.ndim != 3 or actions.shape[-1] != 2 or actions.shape[1] < 2:
            raise ValueError("Safety actions must be [N,H+1,2].")
        scalars = np.asarray([context.heading, context.speed, context.length, context.width,
                              context.wheelbase, context.max_steering], dtype=float)
        if (not np.isfinite(scalars).all() or not np.isfinite(context.position).all()
                or not np.isfinite(context.previous_action).all() or context.speed < 0
                or min(context.length, context.width, context.wheelbase) <= 0
                or not 0 < context.max_steering < np.pi / 2):
            raise ValueError("Live safety state and vehicle geometry must be finite and physically valid.")
        positions, headings, speeds = self.rollout(actions, context)
        n, steps, _ = positions.shape
        finite = torch.isfinite(actions).all(dim=(1, 2)) & (actions.abs() <= 1).all(dim=(1, 2))
        controls = actions[:, :-1]
        previous = actions.new_tensor(context.previous_action).view(1, 1, 2).expand(n, -1, -1)
        rate = torch.diff(torch.cat((previous, controls), 1)[:, :, 0], dim=1).abs() * context.max_steering / self.dt
        curvature = (controls[:, :, 0] * context.max_steering).tan().abs() / context.wheelbase
        peak_speed = torch.maximum(speeds[:, :-1], speeds[:, 1:])
        lateral_acceleration = peak_speed.square() * curvature.repeat_interleave(self.substeps, dim=1)
        kinematic = (~finite | (rate > float(self.cfg.get("max_steering_rate", 2.5))).any(1)
                     | (lateral_acceleration > float(self.cfg.get("max_lateral_acceleration", 6.0))).any(1))
        kinematic |= (torch.diff(headings, dim=1).abs() >= np.pi).any(1)
        ego_length, ego_width = context.length + 2*self.model_margin, context.width + 2*self.model_margin
        ego = _corners(positions, headings, ego_length, ego_width)
        road_valid, margin = self._road_metrics(ego, context.road)
        offroad = ~road_valid.all(1)
        collision = torch.zeros((n, steps), device=actions.device, dtype=torch.bool)
        interval_collision = torch.zeros((n, steps-1), device=actions.device, dtype=torch.bool)
        clearance = actions.new_full((n,), torch.inf)
        ttc = clearance.clone()
        obstacles = context.obstacles
        if obstacles:
            times = actions.new_tensor(np.arange(steps) * self.dt / self.substeps)
            tensor_key = (str(actions.device), actions.dtype)
            if tensor_key not in context.cache:
                context.cache[tensor_key] = tuple(actions.new_tensor(values) for values in (
                    np.asarray([o.position for o in obstacles]), np.asarray([o.velocity for o in obstacles]),
                    [o.heading for o in obstacles], [o.length for o in obstacles], [o.width for o in obstacles]))
            other_xy, other_v, other_heading, lengths, widths = context.cache[tensor_key]
            snapshot_key = ("predicted", *tensor_key, steps, self.dt, self.substeps)
            if snapshot_key not in context.cache:
                predicted = other_xy[None] + times[:, None, None] * other_v[None]
                other = _corners(predicted, other_heading[None].expand(steps, -1), lengths[None, :, None], widths[None, :, None])
                context.cache[snapshot_key] = (predicted, other,
                    torch.stack((other_heading.cos(), other_heading.sin()), -1),
                    torch.stack((-other_heading.sin(), other_heading.cos()), -1))
            predicted, other, other_forward, other_normal = context.cache[snapshot_key]
            ego_forward = torch.stack((headings.cos(), headings.sin()), -1)
            ego_normal = torch.stack((-headings.sin(), headings.cos()), -1)
            axes = torch.stack((ego_forward[:, :, None].expand(-1, -1, len(obstacles), -1),
                                ego_normal[:, :, None].expand(-1, -1, len(obstacles), -1),
                                other_forward[None, None].expand(n, steps, -1, -1),
                                other_normal[None, None].expand(n, steps, -1, -1)), -2)
            distance = ((predicted[None] - positions[:, :, None])[:, :, :, None] * axes).sum(-1)
            radius = (ego_length/2 * (ego_forward[:, :, None, None] * axes).sum(-1).abs()
                      + ego_width/2 * (ego_normal[:, :, None, None] * axes).sum(-1).abs()
                      + lengths[None, None, :, None]/2 * (other_forward[None, None, :, None] * axes).sum(-1).abs()
                      + widths[None, None, :, None]/2 * (other_normal[None, None, :, None] * axes).sum(-1).abs())
            contact = (distance.abs() <= radius).all(-1)
            collision = contact.any(-1)
            gap = torch.minimum(_vertex_edge_distance(ego[:, :, None], other[None]),
                                _vertex_edge_distance(other[None], ego[:, :, None]))
            gap = gap.masked_fill(contact, 0)
            clearance = gap.amin(dim=(1, 2))
            relative_v = other_v[None, None] - speeds[:, :, None, None] * ego_forward[:, :, None]
            velocity = (relative_v[:, :, :, None] * axes).sum(-1)
            moving = velocity.abs() > 1e-6
            divisor = torch.where(moving, velocity, torch.ones_like(velocity))
            first, last = (-radius-distance)/divisor, (radius-distance)/divisor
            entry = torch.minimum(first, last)
            exit = torch.maximum(first, last)
            entry = torch.where(moving, entry, torch.where(distance.abs() <= radius, -torch.inf, torch.inf))
            exit = torch.where(moving, exit, torch.where(distance.abs() <= radius, torch.inf, -torch.inf))
            entry, exit = entry.amax(-1).clamp_min(0), exit.amin(-1)
            absolute_entry = times[None, :, None] + entry
            contact_time = torch.where((exit >= entry) & (absolute_entry <= float(self.cfg.get("ttc_lookahead", 5.0))), absolute_entry, torch.inf)
            ttc = contact_time.amin(dim=(1, 2))
            fraction = self._swept_check(positions, headings, speeds, other_xy, other_v, other_heading, lengths, widths,
                                         ego_length, ego_width, collision.any(1))
            interval_collision = torch.isfinite(fraction)
            absolute = times[:-1][None] + fraction * (self.dt/self.substeps)
            swept_ttc = absolute.masked_fill(absolute > float(self.cfg.get("ttc_lookahead", 5.0)), torch.inf).amin(1)
            ttc = torch.minimum(ttc, swept_ttc)
            collision[:, 1:] |= interval_collision
            clearance = clearance.masked_fill(interval_collision.any(1), 0)
        hit = collision.any(1)
        distance_margin = float(self.cfg.get("soft_clearance", 1.0))
        ttc_margin = float(self.cfg.get("soft_ttc", 3.0))
        boundary_safe = float(self.cfg.get("soft_boundary_margin", .5))
        boundary_risk = (boundary_safe-margin).clamp_min(0)/max(boundary_safe, 1e-6)
        soft = float(self.cfg.get("clearance_risk_weight", 1.0)) * (distance_margin-clearance).clamp_min(0)/max(distance_margin, 1e-6)
        soft += float(self.cfg.get("ttc_risk_weight", 1.0)) * (ttc_margin-ttc).clamp_min(0)/max(ttc_margin, 1e-6)
        soft += float(self.cfg.get("boundary_risk_weight", 1.0)) * boundary_risk
        valid = (~hit & ~offroad & ~kinematic
                 & (clearance >= float(self.cfg.get("hard_clearance", 0.0)))
                 & (ttc >= float(self.cfg.get("hard_ttc", 0.0))))
        return SafetyResult(valid, hit, offroad, kinematic, clearance, ttc, margin, soft, positions,
                            boundary_risk, interval_collision)
