"""Time-aligned footprint checks using MetaDrive road polygons and live obstacles.

Ego motion uses a configurable kinematic bicycle approximation; surrounding
objects use their measured constant velocities and headings. This is a model
check, not a guarantee against unobserved motion or Bullet integration error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import shapely
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from shapely.prepared import prep


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
    emergency_cost: torch.Tensor
    positions: torch.Tensor

    def summary(self, index=0):
        return {key: float(getattr(self, key)[index].cpu())
                for key in ("min_clearance", "min_ttc", "boundary_margin", "soft_risk")}


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


class TrajectorySafetyEvaluator:
    def __init__(self, cfg, *, control_dt: float, device):
        self.cfg = dict(cfg)
        self.dt = float(control_dt)
        self.device = torch.device(device)
        self.substeps = max(1, int(cfg.get("substeps", 5)))
        self.model_margin = max(0., float(cfg.get("model_margin", 0.0)))
        self._road_cache = {}

    def reset(self):
        self._road_cache.clear()

    def context(self, planner, vehicle, *, width, wheelbase, max_steering):
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
            polygons = [lane.shapely_polygon for lane in lanes]
            if not polygons:
                raise RuntimeError("Dynamic safety requires MetaDrive navigation lane polygons.")
            road = unary_union(polygons)
            if not road.is_valid:
                road = road.buffer(0)
            if road.is_empty:
                raise RuntimeError("MetaDrive supplied an empty drivable road polygon.")
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

    @torch.no_grad()
    def check(self, actions, context: SafetyContext):
        if actions.ndim != 3 or actions.shape[-1] != 2 or actions.shape[1] < 2:
            raise ValueError("Safety actions must be [N,H+1,2].")
        positions, headings, speeds = self.rollout(actions, context)
        n, steps, _ = positions.shape
        finite = torch.isfinite(actions).all(dim=(1, 2)) & (actions.abs() <= 1).all(dim=(1, 2))
        controls = actions[:, :-1]
        previous = actions.new_tensor(context.previous_action).view(1, 1, 2).expand(n, -1, -1)
        rate = torch.diff(torch.cat((previous, controls), 1)[:, :, 0], dim=1).abs() * context.max_steering / self.dt
        lateral_acceleration = speeds[:, :-1:self.substeps].square() * (controls[:, :, 0] * context.max_steering).tan().abs() / context.wheelbase
        kinematic = (~finite | (rate > float(self.cfg.get("max_steering_rate", 2.5))).any(1)
                     | (lateral_acceleration > float(self.cfg.get("max_lateral_acceleration", 6.0))).any(1))
        ego_length, ego_width = context.length + 2*self.model_margin, context.width + 2*self.model_margin
        ego = _corners(positions, headings, ego_length, ego_width)
        road = context.road
        safe_road = road.buffer(-float(self.cfg.get("road_margin", 0.15)))
        prepared = prep(safe_road)
        corners_array = ego.cpu().numpy().reshape(-1, 4, 2)
        if hasattr(shapely, "polygons"):
            footprints = shapely.polygons(corners_array)
            road_valid = shapely.covers(safe_road, footprints)
            inside = shapely.covers(road, footprints)
            inside_margin = shapely.distance(road.boundary, footprints)
            outside_margin = shapely.distance(shapely.points(corners_array), road).max(-1)
            margins = np.where(inside, inside_margin, -outside_margin)
        else:
            road_valid, margins = [], []
            for corners in corners_array:
                footprint = Polygon(corners)
                road_valid.append(prepared.covers(footprint))
                margins.append(road.boundary.distance(footprint) if road.covers(footprint)
                               else -max(Point(point).distance(road) for point in corners))
        road_valid = torch.tensor(road_valid, device=actions.device).view(n, steps)
        offroad = ~road_valid.all(1)
        margin = actions.new_tensor(margins).view(n, steps).amin(1)
        collision = torch.zeros((n, steps), device=actions.device, dtype=torch.bool)
        clearance = actions.new_full((n,), torch.inf)
        ttc = clearance.clone()
        obstacles = context.obstacles
        if obstacles:
            times = actions.new_tensor(np.arange(steps) * self.dt / self.substeps)
            other_xy = actions.new_tensor(np.asarray([o.position for o in obstacles]))
            other_v = actions.new_tensor(np.asarray([o.velocity for o in obstacles]))
            other_heading = actions.new_tensor([o.heading for o in obstacles])
            lengths = actions.new_tensor([o.length for o in obstacles])
            widths = actions.new_tensor([o.width for o in obstacles])
            predicted = other_xy[None] + times[:, None, None] * other_v[None]
            other = _corners(predicted, other_heading[None].expand(steps, -1), lengths[None, :, None], widths[None, :, None])
            ego_forward = torch.stack((headings.cos(), headings.sin()), -1)
            ego_normal = torch.stack((-headings.sin(), headings.cos()), -1)
            other_forward = torch.stack((other_heading.cos(), other_heading.sin()), -1)
            other_normal = torch.stack((-other_heading.sin(), other_heading.cos()), -1)
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
            contact_time = torch.where((exit >= entry) & (entry <= float(self.cfg.get("ttc_lookahead", 5.0))), entry, torch.inf)
            ttc = contact_time.amin(dim=(1, 2))
        hit = collision.any(1)
        distance_margin = float(self.cfg.get("soft_clearance", 1.0))
        ttc_margin = float(self.cfg.get("soft_ttc", 3.0))
        soft = (distance_margin-clearance).clamp_min(0)/max(distance_margin, 1e-6)
        soft += (ttc_margin-ttc).clamp_min(0)/max(ttc_margin, 1e-6)
        valid = (~hit & ~offroad & ~kinematic
                 & (clearance >= float(self.cfg.get("hard_clearance", 0.0)))
                 & (ttc >= float(self.cfg.get("hard_ttc", 0.0))))
        emergency = (collision.float().sum(1)*10000 + (~road_valid).float().sum(1)*10000
                     + kinematic.float()*1000 + soft*100 + speeds[:, -1].square())
        return SafetyResult(valid, hit, offroad, kinematic, clearance, ttc, margin, soft, emergency, positions)
