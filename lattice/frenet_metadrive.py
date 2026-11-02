import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from metadrive.obs.top_down_obs_impl import WorldSurface
from metadrive.utils.utils import import_pygame, is_map_related_instance


pygame = import_pygame()


FRENET_DEFAULT_CONFIG = {
    "frenet_planner": True,
    "frenet_strict": False,
    "frenet_debug_info": False,
    "frenet_target_speed": 13.89,  # m/s, same default as RL-frenet
    "frenet_rl_action": False,
    "frenet_rl_first_action": [0.0, -1.0],
    "frenet_rl_action_horizon": 3.0,
    "frenet_rl_min_speed": 5.0,
    "frenet_rl_max_speed": 18.0,
    "frenet_append_reference_obs": True,
    "frenet_reference_obs_dim": 6,
    "frenet_reference_reward_weight": 0.05,
    "frenet_dt": 0.1,
    "frenet_time_horizon": 3.0,
    "frenet_reference_length": 45.0,
    "frenet_min_reference_length": 24.0,
    "frenet_reference_speed_gain": 1.6,
    "frenet_reference_sample_step": 2.0,
    "frenet_follow_lookahead": 4.5,
    "frenet_follow_speed_gain": 0.16,
    "frenet_min_follow_lookahead": 3.0,
    "frenet_max_follow_lookahead": 7.0,
    "frenet_curve_lookahead_gain": 8.0,
    "frenet_curve_speed_limit": True,
    "frenet_max_lateral_accel": 2.8,
    "frenet_min_curve_speed": 4.0,
    # Build the local reference from route-lane centerlines instead of a straight checkpoint segment.
    "frenet_use_middle_lane": False,
    # Legacy option: lane-width multipliers. None enables in-lane lattice sampling.
    "frenet_candidate_lane_offsets": None,
    "frenet_lateral_sample_count": 5,
    "frenet_time_horizons": [2.0, 3.0],
    "frenet_speed_offsets": [0.0, -4.0, -8.0],
    # Clearance from the ego vehicle side to a lane/road boundary when sampling d.
    "frenet_lane_margin": 0.30,
    "frenet_sample_lane_center_band": True,
    "frenet_lane_center_sample_ratio": 0.45,
    "frenet_use_multi_lane_lattice": True,
    "frenet_filter_off_lane_paths": True,
    "frenet_obstacle_check": True,
    "frenet_obstacle_detection_range": 70.0,
    "frenet_obstacle_safety_margin": 0.35,
    "frenet_obstacle_ego_buffer_scale": 0.45,
    "frenet_obstacle_cost_weight": 18.0,
    "frenet_blocked_lane_cost_weight": 14.0,
    "frenet_collision_penalty": 100000.0,
    "frenet_max_draw_candidates": 18,
    "frenet_show_in_window": True,
    "frenet_show_reference": True,
    "frenet_show_obstacle_safety": True,
}


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _norm2(vec: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(vec, dtype=float)[:2]))


class QuinticPolynomial:
    def __init__(self, xs, vxs, axs, xe, vxe, axe, horizon):
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0
        a = np.array(
            [
                [horizon**3, horizon**4, horizon**5],
                [3 * horizon**2, 4 * horizon**3, 5 * horizon**4],
                [6 * horizon, 12 * horizon**2, 20 * horizon**3],
            ],
            dtype=float,
        )
        b = np.array(
            [
                xe - self.a0 - self.a1 * horizon - self.a2 * horizon**2,
                vxe - self.a1 - 2 * self.a2 * horizon,
                axe - 2 * self.a2,
            ],
            dtype=float,
        )
        self.a3, self.a4, self.a5 = np.linalg.solve(a, b)

    def calc_point(self, t):
        return self.a0 + self.a1 * t + self.a2 * t**2 + self.a3 * t**3 + self.a4 * t**4 + self.a5 * t**5

    def calc_first_derivative(self, t):
        return self.a1 + 2 * self.a2 * t + 3 * self.a3 * t**2 + 4 * self.a4 * t**3 + 5 * self.a5 * t**4

    def calc_second_derivative(self, t):
        return 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2 + 20 * self.a5 * t**3

    def calc_third_derivative(self, t):
        return 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t**2


class QuarticPolynomial:
    def __init__(self, xs, vxs, axs, vxe, axe, horizon):
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0
        a = np.array([[3 * horizon**2, 4 * horizon**3], [6 * horizon, 12 * horizon**2]], dtype=float)
        b = np.array([vxe - self.a1 - 2 * self.a2 * horizon, axe - 2 * self.a2], dtype=float)
        self.a3, self.a4 = np.linalg.solve(a, b)

    def calc_point(self, t):
        return self.a0 + self.a1 * t + self.a2 * t**2 + self.a3 * t**3 + self.a4 * t**4

    def calc_first_derivative(self, t):
        return self.a1 + 2 * self.a2 * t + 3 * self.a3 * t**2 + 4 * self.a4 * t**3

    def calc_second_derivative(self, t):
        return 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2

    def calc_third_derivative(self, t):
        return 6 * self.a3 + 24 * self.a4 * t


@dataclass
class FrenetPath:
    t: List[float]
    d: List[float]
    d_d: List[float]
    d_dd: List[float]
    d_ddd: List[float]
    s: List[float]
    s_d: List[float]
    s_dd: List[float]
    s_ddd: List[float]
    x: List[float]
    y: List[float]
    yaw: List[float]
    target_d: float = 0.0
    target_speed: float = 0.0
    horizon: float = 0.0
    cost: float = 0.0
    obstacle_cost: float = 0.0
    min_obstacle_clearance: float = float("inf")
    collision: bool = False
    _xy_cache: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    @property
    def xy(self) -> np.ndarray:
        if self._xy_cache is None:
            self._xy_cache = np.column_stack([self.x, self.y]).astype(float) if self.x else np.zeros((0, 2), dtype=float)
        return self._xy_cache


@dataclass
class FrenetObstacle:
    name: str
    position: np.ndarray
    velocity: np.ndarray
    heading: float
    radius: float
    half_length: float
    half_width: float
    length: float
    width: float


class PolylineReference:
    def __init__(self, points: Sequence[Sequence[float]]):
        cleaned = []
        for point in points:
            point = np.asarray(point, dtype=float)[:2]
            if not cleaned or _norm2(point - cleaned[-1]) > 1e-3:
                cleaned.append(point)
        if len(cleaned) < 2:
            raise ValueError("Frenet reference line needs at least two distinct points.")

        self.points = np.asarray(cleaned, dtype=float)
        self.segments = self.points[1:] - self.points[:-1]
        self.segment_lengths = np.linalg.norm(self.segments, axis=1)
        valid = self.segment_lengths > 1e-6
        self.segments = self.segments[valid]
        self.segment_lengths = self.segment_lengths[valid]
        self.points = np.vstack([self.points[:-1][valid], self.points[-1]])
        self.s = np.concatenate([[0.0], np.cumsum(self.segment_lengths)])
        self.length = float(self.s[-1])
        if self.length <= 1e-6:
            raise ValueError("Frenet reference line length is zero.")

    def _segment_index(self, s_value: float) -> int:
        s_value = float(np.clip(s_value, 0.0, self.length))
        idx = int(np.searchsorted(self.s, s_value, side="right") - 1)
        return min(max(idx, 0), len(self.segment_lengths) - 1)

    def calc_position(self, s_value: float) -> np.ndarray:
        s_value = float(np.clip(s_value, 0.0, self.length))
        idx = self._segment_index(s_value)
        ratio = (s_value - self.s[idx]) / self.segment_lengths[idx]
        return self.points[idx] + ratio * self.segments[idx]

    def calc_yaw(self, s_value: float) -> float:
        idx = self._segment_index(s_value)
        seg = self.segments[idx]
        return math.atan2(seg[1], seg[0])

    def calc_positions(self, s_values: Sequence[float]) -> np.ndarray:
        s_values = np.clip(np.asarray(s_values, dtype=float).reshape(-1), 0.0, self.length)
        indices = np.searchsorted(self.s, s_values, side="right") - 1
        indices = np.clip(indices, 0, len(self.segment_lengths) - 1)
        ratios = (s_values - self.s[indices]) / self.segment_lengths[indices]
        return self.points[indices] + ratios[:, None] * self.segments[indices]

    def calc_yaws(self, s_values: Sequence[float]) -> np.ndarray:
        s_values = np.clip(np.asarray(s_values, dtype=float).reshape(-1), 0.0, self.length)
        indices = np.searchsorted(self.s, s_values, side="right") - 1
        indices = np.clip(indices, 0, len(self.segment_lengths) - 1)
        segments = self.segments[indices]
        return np.arctan2(segments[:, 1], segments[:, 0])

    def project(self, position: Sequence[float]) -> Tuple[float, float, float]:
        pos = np.asarray(position, dtype=float)[:2]
        best_s, best_d, best_dist = 0.0, 0.0, float("inf")
        for idx, seg in enumerate(self.segments):
            seg_len = self.segment_lengths[idx]
            rel = pos - self.points[idx]
            ratio = float(np.clip(np.dot(rel, seg) / (seg_len * seg_len), 0.0, 1.0))
            proj = self.points[idx] + ratio * seg
            diff = pos - proj
            dist = _norm2(diff)
            if dist < best_dist:
                yaw = math.atan2(seg[1], seg[0])
                left_normal = np.array([-math.sin(yaw), math.cos(yaw)])
                best_s = float(self.s[idx] + ratio * seg_len)
                best_d = float(np.dot(diff, left_normal))
                best_dist = dist
        return best_s, best_d, best_dist


class MetaDriveFrenetPlanner:
    def __init__(self, config):
        self.config = config
        self.dt = float(config["frenet_dt"])
        self.target_speed = float(config["frenet_target_speed"])
        self.time_horizon = float(config["frenet_time_horizon"])
        self.reference_length = float(config["frenet_reference_length"])
        self.sample_step = float(config["frenet_reference_sample_step"])
        self.last_reference_points = np.zeros((0, 2), dtype=float)
        self.last_candidates: List[FrenetPath] = []
        self.last_obstacles: List[FrenetObstacle] = []
        self.last_selected_index = 0
        self.last_valid_candidate_count = 0
        self.last_rl_action = None
        self.last_rl_target_d = 0.0
        self.last_lateral_bounds = (0.0, 0.0)
        self.last_lane_centers: List[float] = []
        self.last_lane_safe_half_width = 0.0
        self.last_vehicle_width = 0.0
        self.last_reference: Optional[PolylineReference] = None
        self.last_f_state: Optional[List[float]] = None
        self.last_lane_width = 0.0
        self._lane_centerline_cache: Dict[Tuple[int, float], Tuple[np.ndarray, np.ndarray]] = {}

    def reset(self):
        self.last_reference_points = np.zeros((0, 2), dtype=float)
        self.last_candidates = []
        self.last_obstacles = []
        self.last_selected_index = 0
        self.last_valid_candidate_count = 0
        self.last_rl_action = None
        self.last_rl_target_d = 0.0
        self.last_lateral_bounds = (0.0, 0.0)
        self.last_lane_centers = []
        self.last_lane_safe_half_width = 0.0
        self.last_vehicle_width = 0.0
        self.last_reference = None
        self.last_f_state = None
        self.last_lane_width = 0.0
        self._lane_centerline_cache = {}

    def plan(self, vehicle, rl_action=None) -> FrenetPath:
        """Preserve the original nominal Lattice planning behavior."""
        candidates = self.generate_candidates(vehicle, rl_action=rl_action)
        selected = self.select_nominal_path(candidates)
        self.last_selected_index = selected
        return candidates[selected]

    def generate_candidates(self, vehicle, rl_action=None) -> List[FrenetPath]:
        """Generate and safety-annotate all Frenet candidates without selecting one."""
        reference = PolylineReference(self._build_reference_points(vehicle))
        s, d, _ = reference.project(vehicle.position)
        yaw = reference.calc_yaw(s)
        velocity = np.asarray(vehicle.velocity, dtype=float)[:2]
        tangent = np.array([math.cos(yaw), math.sin(yaw)])
        left_normal = np.array([-math.sin(yaw), math.cos(yaw)])
        f_state = [s, max(0.0, float(np.dot(velocity, tangent))), 0.0, d, float(np.dot(velocity, left_normal)), 0.0]

        lane_width = self._lane_width(vehicle)
        lateral_min, lateral_max, lane_centers = self._drivable_lateral_bounds(reference, vehicle, lane_width)
        self.last_lateral_bounds = (float(lateral_min), float(lateral_max))
        self.last_lane_centers = [float(center) for center in lane_centers]
        if self.config["frenet_rl_action"] and rl_action is not None:
            candidates = [
                self._make_rl_path(reference, f_state, rl_action, lateral_min, lateral_max, lane_width, lane_centers)
            ]
        else:
            candidates = self._make_lattice(reference, f_state, lateral_min, lateral_max, lane_width, lane_centers)
        if not candidates:
            candidates = [
                self._generate_single_path(
                    reference, f_state, df=0.0, time_horizon=self.time_horizon, target_speed=self.target_speed
                )
            ]

        obstacles = self._collect_obstacles(vehicle)
        self._evaluate_obstacle_risk(candidates, obstacles, reference)
        self.last_reference = reference
        self.last_f_state = [float(value) for value in f_state]
        self.last_lane_width = float(lane_width)
        self.last_reference_points = reference.points
        self.last_candidates = candidates
        self.last_obstacles = obstacles
        self.last_selected_index = 0
        self.last_valid_candidate_count = len(self.filter_feasible_paths(candidates))
        return candidates

    def select_nominal_path(self, candidates: List[FrenetPath]) -> int:
        """Return the original Lattice nominal selector result."""
        if not candidates:
            raise ValueError("Cannot select from an empty Frenet candidate set.")
        return self._select_nominal_path(candidates)

    def filter_feasible_paths(self, paths: Sequence[FrenetPath], *, check_obstacles=True) -> List[FrenetPath]:
        """Apply Lattice hard geometry and obstacle constraints."""
        lateral_min, lateral_max = self.last_lateral_bounds
        feasible = []
        for path in paths:
            arrays = (path.t, path.d, path.s, path.x, path.y, path.yaw)
            finite = all(
                len(values) > 0 and np.isfinite(np.asarray(values, dtype=float)).all()
                for values in arrays
            )
            lateral_ok = self._is_path_within_lateral_limit(path, lateral_min, lateral_max)
            clearance_ok = not np.isfinite(path.min_obstacle_clearance) or path.min_obstacle_clearance > 0.0
            if finite and lateral_ok and (not check_obstacles or (not path.collision and clearance_ok)):
                feasible.append(path)
        return feasible

    def generate_parameterized_paths(
        self,
        parameters: Sequence[Tuple[float, float]],
        *,
        horizon: float,
    ) -> List[FrenetPath]:
        """Regenerate paths with existing Frenet polynomials at explicit (d, speed)."""
        if self.last_reference is None or self.last_f_state is None:
            raise RuntimeError("generate_candidates() must run before parameterized path generation.")
        paths = [
            self._generate_single_path(
                self.last_reference,
                self.last_f_state,
                df=float(target_d),
                time_horizon=float(horizon),
                target_speed=float(target_speed),
            )
            for target_d, target_speed in parameters
        ]
        self._evaluate_obstacle_risk(paths, self.last_obstacles, self.last_reference)
        return paths

    def _make_rl_path(self, reference, f_state, rl_action, lateral_min, lateral_max, lane_width, lane_centers):
        target_d, target_speed, time_horizon = self._decode_rl_action(
            rl_action, f_state[3], lateral_min, lateral_max, lane_width, lane_centers
        )
        return self._generate_single_path(
            reference,
            f_state,
            df=target_d,
            time_horizon=time_horizon,
            target_speed=target_speed,
        )

    def _decode_rl_action(self, rl_action, current_d, lateral_min, lateral_max, lane_width, lane_centers):
        action = np.asarray(rl_action, dtype=float).reshape(-1)
        if len(action) == 0:
            action = np.array([0.0, -1.0], dtype=float)
        if len(action) == 1:
            action = np.array([action[0], 0.0], dtype=float)
        action = np.clip(action[:2], -1.0, 1.0)

        lateral_direction = -1 if action[0] < -0.33 else 1 if action[0] > 0.33 else 0
        raw_target_d = lateral_direction * lane_width
        target_d = self._nearest_lateral_target(raw_target_d, lateral_min, lateral_max, lane_centers)

        min_speed = float(self.config["frenet_rl_min_speed"])
        max_speed = float(self.config["frenet_rl_max_speed"])
        speed_center = 0.5 * (min_speed + max_speed)
        speed_radius = 0.5 * (max_speed - min_speed)
        target_speed = float(np.clip(speed_center + speed_radius * action[1], min_speed, max_speed))
        time_horizon = float(self.config["frenet_rl_action_horizon"])
        self.last_rl_action = tuple(float(v) for v in action)
        self.last_rl_target_d = float(target_d)
        return target_d, target_speed, time_horizon

    def reference_observation(self, vehicle) -> np.ndarray:
        obs_dim = int(self.config["frenet_reference_obs_dim"])
        features = np.zeros(obs_dim, dtype=np.float32)

        reference_points = self._build_reference_points(vehicle)
        if len(reference_points) < 2:
            return features

        try:
            reference = PolylineReference(reference_points)
            s, d, yaw = reference.project(vehicle.position)
        except Exception:
            return features

        lane_width = max(0.1, self._lane_width(vehicle))
        max_speed = max(1.0, float(self.config["frenet_rl_max_speed"]))
        velocity = self._object_velocity(vehicle)
        tangent = np.array([math.cos(yaw), math.sin(yaw)])
        left_normal = np.array([-math.sin(yaw), math.cos(yaw)])

        try:
            heading = float(vehicle.heading_theta)
        except Exception:
            heading = yaw

        lookahead_s = min(reference.length, float(s) + 10.0)
        lookahead_yaw = reference.calc_yaw(lookahead_s)
        progress = float(s) / max(reference.length, 1e-6)
        raw_features = [
            float(np.clip(d / lane_width, -1.0, 1.0)),
            float(np.clip(_wrap_to_pi(heading - yaw) / math.pi, -1.0, 1.0)),
            float(np.clip(np.dot(velocity, tangent) / max_speed, -1.0, 1.0)),
            float(np.clip(np.dot(velocity, left_normal) / max_speed, -1.0, 1.0)),
            float(np.clip(_wrap_to_pi(lookahead_yaw - yaw) / math.pi, -1.0, 1.0)),
            float(np.clip(2.0 * progress - 1.0, -1.0, 1.0)),
        ]
        features[:min(obs_dim, len(raw_features))] = raw_features[:obs_dim]
        return features

    @staticmethod
    def _nearest_lateral_target(raw_target_d, lateral_min, lateral_max, lane_centers):
        clipped = float(np.clip(raw_target_d, lateral_min, lateral_max))
        if lane_centers:
            return min((float(center) for center in lane_centers), key=lambda center: abs(center - clipped))
        return clipped

    def _make_lattice(self, reference, f_state, lateral_min, lateral_max, lane_width, lane_centers):
        candidates = []
        lateral_targets = self._candidate_lateral_targets(lane_width, lateral_min, lateral_max, lane_centers)
        time_horizons = [float(t) for t in self.config["frenet_time_horizons"]]
        speed_targets = [max(0.5, self.target_speed + float(offset)) for offset in self.config["frenet_speed_offsets"]]

        for df in lateral_targets:
            for horizon in time_horizons:
                for speed in speed_targets:
                    path = self._generate_single_path(reference, f_state, df=df, time_horizon=horizon, target_speed=speed)
                    if self._is_path_within_lateral_limit(path, lateral_min, lateral_max):
                        candidates.append(path)
        return candidates

    def _generate_single_path(
        self,
        reference: PolylineReference,
        f_state,
        df: float,
        time_horizon: Optional[float] = None,
        target_speed: Optional[float] = None,
    ) -> FrenetPath:
        time_horizon = self.time_horizon if time_horizon is None else float(time_horizon)
        target_speed = self.target_speed if target_speed is None else float(target_speed)
        s, s_d, s_dd, d, d_d, d_dd = f_state
        lat_qp = QuinticPolynomial(d, d_d, d_dd, df, 0.0, 0.0, time_horizon)
        lon_qp = QuarticPolynomial(s, s_d, s_dd, target_speed, 0.0, time_horizon)
        times = np.arange(0.0, time_horizon + 0.5 * self.dt, self.dt, dtype=float)
        s_values = np.minimum(lon_qp.calc_point(times), reference.length)
        reached_end = np.flatnonzero((s_values >= reference.length) & (np.arange(len(times)) > 2))
        if len(reached_end):
            keep = int(reached_end[0]) + 1
            times = times[:keep]
            s_values = s_values[:keep]

        d_values = lat_qp.calc_point(times)
        d_d_values = lat_qp.calc_first_derivative(times)
        d_dd_values = lat_qp.calc_second_derivative(times)
        d_ddd_values = lat_qp.calc_third_derivative(times)
        s_d_values = lon_qp.calc_first_derivative(times)
        s_dd_values = lon_qp.calc_second_derivative(times)
        s_ddd_values = lon_qp.calc_third_derivative(times)

        ref_xy = reference.calc_positions(s_values)
        ref_yaw = reference.calc_yaws(s_values)
        left_normals = np.column_stack([-np.sin(ref_yaw), np.cos(ref_yaw)])
        xy = ref_xy + d_values[:, None] * left_normals

        path = FrenetPath(
            times.tolist(),
            d_values.tolist(),
            d_d_values.tolist(),
            d_dd_values.tolist(),
            d_ddd_values.tolist(),
            s_values.tolist(),
            s_d_values.tolist(),
            s_dd_values.tolist(),
            s_ddd_values.tolist(),
            xy[:, 0].tolist(),
            xy[:, 1].tolist(),
            ref_yaw.tolist(),
            df,
            target_speed,
            time_horizon,
        )
        path._xy_cache = xy.astype(float, copy=False)
        return path

    def _candidate_lateral_targets(
        self, lane_width: float, lateral_min: float, lateral_max: float, lane_centers: Sequence[float]
    ) -> List[float]:
        lane_offsets = self.config["frenet_candidate_lane_offsets"]
        if lane_offsets is not None:
            targets = [float(offset) * lane_width for offset in lane_offsets]
            targets = [float(np.clip(target, lateral_min, lateral_max)) for target in targets]
        elif self.config["frenet_sample_lane_center_band"] and lane_centers:
            sample_count = max(1, int(self.config["frenet_lateral_sample_count"]))
            lane_safe_half_width = max(0.0, float(self.last_lane_safe_half_width))
            sample_radius = lane_safe_half_width * float(self.config["frenet_lane_center_sample_ratio"])
            if sample_count <= 3 or sample_radius <= 1e-6:
                offsets = [0.0]
            else:
                offsets = [-sample_radius, 0.0, sample_radius]
            targets = []
            for center in lane_centers:
                targets.extend(float(center) + offset for offset in offsets)
            targets = [float(np.clip(target, lateral_min, lateral_max)) for target in targets]
        else:
            sample_count = max(3, int(self.config["frenet_lateral_sample_count"]))
            targets = np.linspace(lateral_min, lateral_max, sample_count).tolist()
        targets.extend(float(center) for center in lane_centers)
        targets.append(0.0)
        return sorted({round(float(target), 4) for target in targets})

    def _is_path_within_lateral_limit(self, path: FrenetPath, lateral_min: float, lateral_max: float) -> bool:
        if not self.config["frenet_filter_off_lane_paths"]:
            return True
        return bool(path.d) and min(path.d) >= lateral_min - 1e-6 and max(path.d) <= lateral_max + 1e-6

    def _select_nominal_path(self, candidates: List[FrenetPath]) -> int:
        def score(item):
            idx, path = item
            collision_cost = float(self.config["frenet_collision_penalty"]) if path.collision else 0.0
            speed_cost = 0.25 * abs(path.target_speed - self.target_speed)
            horizon_cost = 0.15 * abs(path.horizon - self.time_horizon)
            center_cost = 1.00 * abs(path.target_d)
            smooth_cost = 0.02 * (sum(abs(v) for v in path.d_ddd) + 0.2 * sum(abs(v) for v in path.s_ddd))
            path.cost = collision_cost + path.obstacle_cost + center_cost + horizon_cost + speed_cost + smooth_cost
            return (path.collision, path.cost, idx)

        return min(enumerate(candidates), key=score)[0]

    def _drivable_lateral_bounds(self, reference, vehicle, lane_width):
        margin = float(self.config["frenet_lane_margin"])
        vehicle_width = self._object_extent(vehicle, "WIDTH", "top_down_width", 1.9)
        vehicle_half_width = 0.5 * vehicle_width
        default_limit = max(0.0, lane_width / 2.0 - vehicle_half_width - margin)
        lane_centers = [0.0]
        self.last_vehicle_width = float(vehicle_width)
        self.last_lane_safe_half_width = float(default_limit)

        lanes = getattr(vehicle.navigation, "current_ref_lanes", None) or [getattr(vehicle, "lane", None)]
        lanes = [lane for lane in lanes if lane is not None]
        if self.config["frenet_use_multi_lane_lattice"] and len(lanes) > 1:
            for lane in lanes:
                try:
                    current_long, _ = lane.local_coordinates(vehicle.position)
                    center_pos = lane.position(float(np.clip(current_long, 0.0, lane.length)), 0.0)
                    _, center_d, _ = reference.project(center_pos)
                    lane_centers.append(float(center_d))
                except Exception:
                    continue

        lane_centers = sorted({round(float(center), 4) for center in lane_centers})
        lateral_min = min(lane_centers) - default_limit
        lateral_max = max(lane_centers) + default_limit
        return lateral_min, lateral_max, lane_centers

    def _collect_obstacles(self, vehicle) -> List[FrenetObstacle]:
        if not self.config["frenet_obstacle_check"]:
            return []

        engine = getattr(vehicle, "engine", None)
        if engine is None:
            return []

        ego_pos = np.asarray(vehicle.position, dtype=float)[:2]
        ego_length = self._object_extent(vehicle, "LENGTH", "top_down_length", 4.5)
        ego_width = self._object_extent(vehicle, "WIDTH", "top_down_width", 1.9)
        detection_range = float(self.config["frenet_obstacle_detection_range"])
        safety_margin = float(self.config["frenet_obstacle_safety_margin"])
        ego_buffer_scale = float(self.config["frenet_obstacle_ego_buffer_scale"])

        obstacles = []
        detection_range_sq = detection_range * detection_range
        for name, obj in engine.get_objects().items():
            if obj is vehicle or is_map_related_instance(obj) or not hasattr(obj, "position"):
                continue
            try:
                pos = np.asarray(obj.position, dtype=float)[:2]
            except Exception:
                continue
            diff = pos - ego_pos
            if float(np.dot(diff, diff)) > detection_range_sq:
                continue

            length = self._object_extent(obj, "LENGTH", "top_down_length", 1.0)
            width = self._object_extent(obj, "WIDTH", "top_down_width", 1.0)
            half_length = 0.5 * length + 0.5 * ego_length * ego_buffer_scale + safety_margin
            half_width = 0.5 * width + 0.5 * ego_width * ego_buffer_scale + safety_margin
            obstacles.append(
                FrenetObstacle(
                    name=str(name),
                    position=pos,
                    velocity=self._object_velocity(obj),
                    heading=self._object_heading(obj),
                    radius=math.hypot(half_length, half_width),
                    half_length=half_length,
                    half_width=half_width,
                    length=length,
                    width=width,
                )
            )
        return obstacles

    def _evaluate_obstacle_risk(
        self, candidates: List[FrenetPath], obstacles: Sequence[FrenetObstacle], reference: PolylineReference
    ):
        if not obstacles:
            for path in candidates:
                path.collision = False
                path.obstacle_cost = 0.0
                path.min_obstacle_clearance = float("inf")
            return

        obstacle_weight = float(self.config["frenet_obstacle_cost_weight"])
        blocked_lane_weight = float(self.config["frenet_blocked_lane_cost_weight"])
        obstacle_positions = np.asarray([obstacle.position for obstacle in obstacles], dtype=float)
        obstacle_velocities = np.asarray([obstacle.velocity for obstacle in obstacles], dtype=float)
        headings = np.asarray([obstacle.heading for obstacle in obstacles], dtype=float)
        forward = np.column_stack([np.cos(headings), np.sin(headings)])
        left = np.column_stack([-np.sin(headings), np.cos(headings)])
        half_lengths = np.asarray([obstacle.half_length for obstacle in obstacles], dtype=float)
        half_widths = np.asarray([obstacle.half_width for obstacle in obstacles], dtype=float)
        obstacle_sd = np.asarray([reference.project(obstacle.position)[:2] for obstacle in obstacles], dtype=float)

        for path in candidates:
            points = path.xy
            if len(points) == 0:
                path.collision = False
                path.min_obstacle_clearance = float("inf")
                path.obstacle_cost = 0.0
                continue

            times = np.asarray(path.t, dtype=float)
            predicted = obstacle_positions[None, :, :] + times[:, None, None] * obstacle_velocities[None, :, :]
            rel = points[:, None, :] - predicted
            longitudinal_over = np.abs(np.sum(rel * forward[None, :, :], axis=2)) - half_lengths[None, :]
            lateral_over = np.abs(np.sum(rel * left[None, :, :], axis=2)) - half_widths[None, :]
            inside = (longitudinal_over <= 0.0) & (lateral_over <= 0.0)
            clearances = np.where(
                inside,
                np.maximum(longitudinal_over, lateral_over),
                np.hypot(np.maximum(longitudinal_over, 0.0), np.maximum(lateral_over, 0.0)),
            )
            min_clearance = float(np.min(clearances))
            collision = bool(np.any(clearances <= 0.0))
            path.collision = collision
            path.min_obstacle_clearance = min_clearance
            path.obstacle_cost = obstacle_weight / max(min_clearance + 1.0, 0.1)
            path.obstacle_cost += self._blocked_lane_cost(path, obstacle_sd, half_widths, blocked_lane_weight)
            if collision:
                path.obstacle_cost += obstacle_weight * 10.0

    def _blocked_lane_cost(
        self,
        path: FrenetPath,
        obstacle_sd: np.ndarray,
        obstacle_half_widths: np.ndarray,
        weight: float,
    ) -> float:
        if not path.s or len(obstacle_sd) == 0:
            return 0.0

        start_s = float(path.s[0])
        detection_range = float(self.config["frenet_obstacle_detection_range"])
        ahead_distance = obstacle_sd[:, 0] - start_s
        in_range = (ahead_distance >= -2.0) & (ahead_distance <= detection_range)
        if not np.any(in_range):
            return 0.0

        lateral_gap = np.abs(float(path.target_d) - obstacle_sd[:, 1])
        lateral_influence = np.maximum(0.0, 1.0 - lateral_gap / np.maximum(obstacle_half_widths + 0.5, 0.1))
        valid = in_range & (lateral_influence > 0.0)
        if not np.any(valid):
            return 0.0

        longitudinal_influence = 1.0 - 0.5 * np.minimum(ahead_distance / detection_range, 1.0)
        return float(np.sum(weight * lateral_influence[valid] * longitudinal_influence[valid]))

    @staticmethod
    def _obstacle_clearance(point, obstacle_position, obstacle: FrenetObstacle) -> float:
        rel = np.asarray(point, dtype=float)[:2] - np.asarray(obstacle_position, dtype=float)[:2]
        heading = float(obstacle.heading)
        forward = np.array([math.cos(heading), math.sin(heading)])
        left = np.array([-math.sin(heading), math.cos(heading)])
        longitudinal_over = abs(float(np.dot(rel, forward))) - obstacle.half_length
        lateral_over = abs(float(np.dot(rel, left))) - obstacle.half_width
        if longitudinal_over <= 0.0 and lateral_over <= 0.0:
            return max(longitudinal_over, lateral_over)
        return math.hypot(max(longitudinal_over, 0.0), max(lateral_over, 0.0))

    @staticmethod
    def _object_extent(obj, primary_attr, fallback_attr, default):
        for attr in (primary_attr, fallback_attr):
            try:
                value = getattr(obj, attr)
            except Exception:
                continue
            try:
                return max(0.1, float(value))
            except (TypeError, ValueError):
                continue
        return float(default)

    @staticmethod
    def _object_velocity(obj):
        try:
            velocity = np.asarray(obj.velocity, dtype=float)[:2]
        except Exception:
            velocity = np.zeros(2, dtype=float)
        if len(velocity) < 2 or not np.all(np.isfinite(velocity)):
            return np.zeros(2, dtype=float)
        return velocity

    @staticmethod
    def _object_heading(obj):
        try:
            return float(obj.heading_theta)
        except Exception:
            return 0.0

    def _build_reference_points(self, vehicle) -> List[np.ndarray]:
        lane_points = self._build_lane_reference_points(vehicle)
        if len(lane_points) >= 2:
            return lane_points

        start = np.asarray(vehicle.position, dtype=float)[:2]
        checkpoint = self._nearest_navigation_point(vehicle, start)
        if checkpoint is not None and _norm2(checkpoint - start) > 1e-3:
            return self._sample_segment(start, checkpoint)

        heading_vec = self._heading_vector(vehicle)
        return self._sample_segment(start, start + heading_vec * max(10.0, self.sample_step))

    def _build_lane_reference_points(self, vehicle) -> List[np.ndarray]:
        lanes = self._reference_lane_sequence(vehicle)
        if not lanes:
            return []

        reference_length = self._local_reference_length(vehicle)
        points: List[np.ndarray] = []
        current_length = 0.0
        for idx, lane in enumerate(lanes):
            if lane is None:
                continue
            remaining = reference_length - current_length
            if remaining <= self.sample_step * 0.25:
                break
            try:
                start_long = float(lane.local_coordinates(vehicle.position)[0]) if idx == 0 else 0.0
                lane_points = self._sample_lane(lane, start_long, remaining)
            except Exception:
                continue
            old_count = len(points)
            self._append_unique_points(points, lane_points)
            if len(points) > max(old_count, 1):
                new_points = points[max(old_count - 1, 0):]
                current_length += self.polyline_length(new_points)
            if current_length >= reference_length:
                break

        return points

    def _local_reference_length(self, vehicle) -> float:
        min_length = float(self.config["frenet_min_reference_length"])
        max_length = float(self.config["frenet_reference_length"])
        speed_gain = float(self.config["frenet_reference_speed_gain"])
        speed = max(0.0, float(getattr(vehicle, "speed", 0.0)))
        return float(np.clip(min_length + speed_gain * speed, min_length, max_length))

    def _reference_lane_sequence(self, vehicle) -> List:
        navigation = getattr(vehicle, "navigation", None)
        current_ref_lanes = self._clean_lanes(getattr(navigation, "current_ref_lanes", None)) if navigation else []
        next_ref_lanes = self._clean_lanes(getattr(navigation, "next_ref_lanes", None)) if navigation else []

        lane_groups = self._navigation_lane_groups(navigation)
        first_lane = self._select_next_navigation_lane(vehicle, lane_groups)
        if first_lane is None:
            first_lane = self._select_reference_lane(vehicle)
        if first_lane is None:
            return []

        route = [first_lane]
        start_group_idx = None
        for group_idx, group in enumerate(lane_groups):
            if first_lane in group:
                start_group_idx = group_idx
                break

        if start_group_idx is None:
            lane_groups = [group for group in (current_ref_lanes, next_ref_lanes) if group]
            for group_idx, group in enumerate(lane_groups):
                if first_lane in group:
                    start_group_idx = group_idx
                    break

        if start_group_idx is not None:
            future_groups = lane_groups[start_group_idx + 1:]
        else:
            future_groups = [group for group in (next_ref_lanes,) if group]

        for group in future_groups:
            next_lane = self._select_next_lane(route[-1], group)
            if next_lane is not None and next_lane is not route[-1]:
                route.append(next_lane)
            if len(route) >= 4:
                break
        return route

    def _select_next_navigation_lane(self, vehicle, lane_groups: Sequence[Sequence]) -> Optional[object]:
        navigation = getattr(vehicle, "navigation", None)
        if navigation is None:
            return None

        current_ref_lanes = self._clean_lanes(getattr(navigation, "current_ref_lanes", None))
        candidate_lanes = current_ref_lanes or (self._clean_lanes(lane_groups[0]) if lane_groups else [])
        if not candidate_lanes:
            return None

        next_point = self._navigation_checkpoint(navigation, 0)
        if next_point is not None:
            lane = self._lane_containing_point(candidate_lanes, next_point)
            if lane is not None:
                return lane

        if self.config["frenet_use_middle_lane"]:
            return candidate_lanes[min(len(candidate_lanes) // 2, len(candidate_lanes) - 1)]
        return self._lane_closest_to_road_center(candidate_lanes)

    def _navigation_checkpoint(self, navigation, index: int) -> Optional[np.ndarray]:
        if navigation is None or not hasattr(navigation, "get_checkpoints"):
            return None
        try:
            checkpoints = navigation.get_checkpoints()
            point = np.asarray(checkpoints[index], dtype=float)[:2]
        except Exception:
            return None
        if len(point) != 2 or not np.all(np.isfinite(point)):
            return None
        return point

    def _lane_containing_point(self, lanes: Sequence, point: np.ndarray):
        best_lane = None
        best_score = float("inf")
        point = np.asarray(point, dtype=float)[:2]
        for lane in self._clean_lanes(lanes):
            try:
                longitudinal, lateral = lane.local_coordinates(point)
                lane_length = max(0.0, float(getattr(lane, "length", 0.0)))
                lane_width = max(0.1, float(getattr(lane, "width", 3.5)))
            except Exception:
                continue

            longitudinal_over = max(0.0, -float(longitudinal), float(longitudinal) - lane_length)
            lateral_abs = abs(float(lateral))
            inside_bonus = 0.0 if lateral_abs <= 0.5 * lane_width + 1e-3 else lane_width
            score = lateral_abs + 2.0 * longitudinal_over + inside_bonus
            if score < best_score:
                best_score = score
                best_lane = lane
        return best_lane

    @staticmethod
    def _lane_closest_to_road_center(lanes: Sequence):
        lanes = [lane for lane in lanes if lane is not None]
        if not lanes:
            return None
        return lanes[min(len(lanes) // 2, len(lanes) - 1)]

    def _navigation_lane_groups(self, navigation) -> List[List]:
        if navigation is None:
            return []
        checkpoints = getattr(navigation, "checkpoints", None)
        target_indices = getattr(navigation, "_target_checkpoints_index", None)
        road_network = getattr(getattr(navigation, "map", None), "road_network", None)
        if not checkpoints or target_indices is None or road_network is None:
            return []

        try:
            start_idx = int(target_indices[0])
        except (TypeError, ValueError, IndexError):
            start_idx = 0

        groups = []
        graph = getattr(road_network, "graph", {})
        for idx in range(max(0, start_idx), len(checkpoints) - 1):
            try:
                lanes = graph[checkpoints[idx]][checkpoints[idx + 1]]
            except Exception:
                continue
            lanes = self._clean_lanes(lanes)
            if lanes:
                groups.append(lanes)
        return groups

    @staticmethod
    def _clean_lanes(lanes) -> List:
        if lanes is None:
            return []
        try:
            return [lane for lane in lanes if lane is not None]
        except TypeError:
            return [lanes] if lanes is not None else []

    @staticmethod
    def _append_unique_points(target: List[np.ndarray], source: Sequence[np.ndarray]):
        for point in source:
            point = np.asarray(point, dtype=float)[:2]
            if len(point) != 2 or not np.all(np.isfinite(point)):
                continue
            if target and _norm2(point - target[-1]) <= 1e-3:
                continue
            target.append(point)

    @staticmethod
    def _heading_vector(vehicle) -> np.ndarray:
        heading = np.asarray(vehicle.heading, dtype=float)[:2]
        if len(heading) == 2 and np.all(np.isfinite(heading)) and _norm2(heading) > 1e-6:
            return heading / _norm2(heading)
        heading_theta = float(getattr(vehicle, "heading_theta", 0.0))
        return np.array([math.cos(heading_theta), math.sin(heading_theta)], dtype=float)

    def _nearest_navigation_point(self, vehicle, start: np.ndarray) -> Optional[np.ndarray]:
        navigation = getattr(vehicle, "navigation", None)
        if navigation is None or not hasattr(navigation, "get_checkpoints"):
            return None

        try:
            checkpoints = navigation.get_checkpoints()
        except Exception:
            return None

        points = []
        for checkpoint in checkpoints:
            point = np.asarray(checkpoint, dtype=float)[:2]
            if len(point) == 2 and np.all(np.isfinite(point)):
                points.append(point)
        if not points:
            return None

        min_distance = max(1.0, self.sample_step)
        usable_points = [point for point in points if _norm2(point - start) >= min_distance]
        candidates = usable_points or points
        return min(candidates, key=lambda point: _norm2(point - start))

    def _sample_segment(self, start: np.ndarray, end: np.ndarray) -> List[np.ndarray]:
        distance = _norm2(end - start)
        if distance <= 1e-6:
            return [start, end]
        segment_count = max(1, int(math.ceil(distance / self.sample_step)))
        return [start + (end - start) * (idx / segment_count) for idx in range(segment_count + 1)]

    def _sample_lane(self, lane, start_long: float, max_length: float) -> List[np.ndarray]:
        lane_length = max(0.0, float(getattr(lane, "length", 0.0)))
        start_long = float(np.clip(start_long, 0.0, lane_length))
        end_long = float(np.clip(start_long + max(0.0, float(max_length)), 0.0, lane_length))
        if end_long <= start_long + 1e-6:
            return [np.asarray(lane.position(start_long, 0.0), dtype=float)[:2]]

        cached_longs, cached_points = self._cached_lane_centerline(lane)
        points = [np.asarray(lane.position(start_long, 0.0), dtype=float)[:2]]
        mask = (cached_longs > start_long + 1e-3) & (cached_longs < end_long - 1e-3)
        points.extend(cached_points[mask])
        points.append(np.asarray(lane.position(end_long, 0.0), dtype=float)[:2])
        return points

    def _cached_lane_centerline(self, lane) -> Tuple[np.ndarray, np.ndarray]:
        lane_length = max(0.0, float(getattr(lane, "length", 0.0)))
        key = (id(lane), round(self.sample_step, 4))
        cached = self._lane_centerline_cache.get(key)
        if cached is not None:
            return cached

        longs = np.arange(0.0, lane_length, self.sample_step, dtype=float)
        if len(longs) == 0 or abs(float(longs[-1]) - lane_length) > 1e-3:
            longs = np.append(longs, lane_length)
        points = np.asarray(
            [np.asarray(lane.position(float(longitudinal), 0.0), dtype=float)[:2] for longitudinal in longs]
        )
        self._lane_centerline_cache[key] = (longs, points)
        return longs, points

    def _select_reference_lane(self, vehicle):
        navigation = getattr(vehicle, "navigation", None)
        lanes = self._clean_lanes(getattr(navigation, "current_ref_lanes", None)) if navigation else []
        if not lanes:
            lanes = self._clean_lanes([getattr(vehicle, "lane", None)])
        if not lanes:
            return None
        if self.config["frenet_use_middle_lane"]:
            return lanes[min(len(lanes) // 2, len(lanes) - 1)]
        ego_lane = getattr(vehicle, "lane", None)
        if ego_lane in lanes:
            return ego_lane
        return lanes[min(len(lanes) // 2, len(lanes) - 1)]

    def _select_next_lane(self, current_lane, next_lanes):
        next_lanes = self._clean_lanes(next_lanes)
        if not next_lanes:
            return None
        if self.config["frenet_use_middle_lane"]:
            return next_lanes[min(len(next_lanes) // 2, len(next_lanes) - 1)]
        try:
            lane_id = int(current_lane.index[-1])
        except (TypeError, ValueError, AttributeError):
            lane_id = len(next_lanes) // 2
        return next_lanes[min(max(lane_id, 0), len(next_lanes) - 1)]

    def _lane_width(self, vehicle) -> float:
        try:
            return float(vehicle.navigation.get_current_lane_width())
        except Exception:
            return float(getattr(vehicle.lane, "width", 3.5))

    @staticmethod
    def polyline_length(points: Sequence[np.ndarray]) -> float:
        if len(points) < 2:
            return 0.0
        return float(sum(_norm2(np.asarray(p2) - np.asarray(p1)) for p1, p2 in zip(points[:-1], points[1:])))


class MetaDriveFrenetController:
    def __init__(self, config):
        self.config = config
        self.planner = MetaDriveFrenetPlanner(config)
        self.last_info = {}

    def reset(self):
        self.planner.reset()
        self.last_info = {}

    def reference_observation(self, vehicle) -> np.ndarray:
        return self.planner.reference_observation(vehicle)

    def act(self, vehicle, rl_action=None) -> List[float]:
        path = self.planner.plan(vehicle, rl_action=rl_action)
        action, target_idx = self._track_path(vehicle, path)
        selected_path = self.planner.last_candidates[self.planner.last_selected_index]
        self.last_info = {
            "frenet_action": tuple(float(v) for v in action),
            "frenet_rl_action": self.planner.last_rl_action,
            "frenet_rl_target_d": float(self.planner.last_rl_target_d),
            "frenet_target_speed": float(selected_path.target_speed),
            "frenet_candidate_count": len(self.planner.last_candidates),
            "frenet_valid_candidate_count": int(self.planner.last_valid_candidate_count),
            "frenet_obstacle_count": len(self.planner.last_obstacles),
            "frenet_selected_path_index": int(self.planner.last_selected_index),
            "frenet_selected_target_d": float(selected_path.target_d),
            "frenet_selected_horizon": float(selected_path.horizon),
            "frenet_selected_speed": float(selected_path.target_speed),
            "frenet_selected_collision": bool(selected_path.collision),
            "frenet_selected_clearance": float(selected_path.min_obstacle_clearance),
            "frenet_target_waypoint": tuple(float(v) for v in path.xy[target_idx]) if len(path.x) else None,
            "frenet_reference_length": float(self.planner.polyline_length(self.planner.last_reference_points)),
            "frenet_lateral_bounds": tuple(float(v) for v in self.planner.last_lateral_bounds),
            "frenet_lane_centers": tuple(float(v) for v in self.planner.last_lane_centers),
            "frenet_vehicle_width": float(self.planner.last_vehicle_width),
        }
        if self.config["frenet_debug_info"]:
            self.last_info["frenet_path"] = path.xy.astype(np.float32)
            self.last_info["frenet_reference"] = self.planner.last_reference_points.astype(np.float32)
        return action

    def _track_path(self, vehicle, path: FrenetPath) -> Tuple[List[float], int]:
        points = path.xy
        if len(points) < 2:
            return [0.0, 0.0], 0

        ego_pos = np.asarray(vehicle.position, dtype=float)[:2]
        distances = np.linalg.norm(points - ego_pos, axis=1)
        closest_idx = int(np.argmin(distances))
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(points[1:] - points[:-1], axis=1))])
        lookahead = self._follow_lookahead(vehicle, points, arc, closest_idx)
        target_idx = min(int(np.searchsorted(arc, arc[closest_idx] + lookahead, side="left")), len(points) - 1)

        target_vec = points[target_idx] - ego_pos
        if _norm2(target_vec) < 1e-6:
            target_vec = points[min(target_idx + 1, len(points) - 1)] - ego_pos
        bearing = math.atan2(target_vec[1], target_vec[0])

        prev_idx = max(target_idx - 1, 0)
        next_idx = min(target_idx + 1, len(points) - 1)
        tangent = points[next_idx] - points[prev_idx]
        path_yaw = math.atan2(tangent[1], tangent[0]) if _norm2(tangent) > 1e-6 else bearing

        closest_next = min(closest_idx + 1, len(points) - 1)
        closest_prev = max(closest_idx - 1, 0)
        closest_tangent = points[closest_next] - points[closest_prev]
        closest_yaw = math.atan2(closest_tangent[1], closest_tangent[0]) if _norm2(closest_tangent) > 1e-6 else path_yaw
        left_normal = np.array([-math.sin(closest_yaw), math.cos(closest_yaw)])
        lateral_error = float(np.dot(ego_pos - points[closest_idx], left_normal))

        heading_error = _wrap_to_pi(path_yaw - float(vehicle.heading_theta))
        bearing_error = _wrap_to_pi(bearing - float(vehicle.heading_theta))
        steering = float(np.clip(0.60 * bearing_error + 0.35 * heading_error - 0.08 * lateral_error, -1.0, 1.0))
        target_speed_resolver = self.config.get("_tracking_target_speed")
        if target_speed_resolver is None:
            target_speed = max(float(path.target_speed), 0.0)
        else:
            target_speed = float(
                target_speed_resolver(path.target_speed, self.planner.target_speed)
            )
        target_speed = self._curve_limited_speed(target_speed, points, arc, closest_idx)
        throttle_brake = float(np.clip(0.35 * (target_speed - float(vehicle.speed)), -1.0, 1.0))
        return [steering, throttle_brake], target_idx

    def _follow_lookahead(self, vehicle, points: np.ndarray, arc: np.ndarray, closest_idx: int) -> float:
        speed = max(0.0, float(getattr(vehicle, "speed", 0.0)))
        lookahead = float(self.config["frenet_follow_lookahead"]) + speed * float(self.config["frenet_follow_speed_gain"])
        lookahead = float(
            np.clip(
                lookahead,
                float(self.config["frenet_min_follow_lookahead"]),
                float(self.config["frenet_max_follow_lookahead"]),
            )
        )

        curvature = self._local_path_curvature(points, arc, closest_idx)
        if curvature > 1e-6:
            lookahead /= 1.0 + float(self.config["frenet_curve_lookahead_gain"]) * curvature
        return max(float(self.config["frenet_min_follow_lookahead"]), lookahead)

    def _curve_limited_speed(self, target_speed: float, points: np.ndarray, arc: np.ndarray, closest_idx: int) -> float:
        if not self.config["frenet_curve_speed_limit"]:
            return target_speed

        curvature = self._local_path_curvature(points, arc, closest_idx)
        if curvature <= 1e-6:
            return target_speed

        max_lateral_accel = max(0.1, float(self.config["frenet_max_lateral_accel"]))
        curve_speed = math.sqrt(max_lateral_accel / curvature)
        curve_speed = max(float(self.config["frenet_min_curve_speed"]), curve_speed)
        return min(target_speed, curve_speed)

    @staticmethod
    def _local_path_curvature(points: np.ndarray, arc: np.ndarray, closest_idx: int) -> float:
        if len(points) < 4:
            return 0.0

        end_s = arc[closest_idx] + 14.0
        end_idx = min(int(np.searchsorted(arc, end_s, side="right")), len(points) - 1)
        if end_idx - closest_idx < 3:
            return 0.0

        local_points = points[closest_idx:end_idx + 1]
        segments = local_points[1:] - local_points[:-1]
        seg_lengths = np.linalg.norm(segments, axis=1)
        valid = seg_lengths > 1e-6
        if np.count_nonzero(valid) < 2:
            return 0.0

        yaws = np.arctan2(segments[valid, 1], segments[valid, 0])
        yaw_deltas = np.abs(np.diff(np.unwrap(yaws)))
        distance = max(float(np.sum(seg_lengths[valid])), 1e-6)
        return float(np.sum(yaw_deltas) / distance)


def draw_frenet_overlay(renderer, planner, config):
    if renderer is None or planner is None:
        return

    surface = renderer.screen_canvas
    if config["frenet_show_reference"] and len(planner.last_reference_points) >= 2:
        _draw_world_polyline(surface, renderer, planner.last_reference_points, color=(45, 45, 45), width=2)

    if config["frenet_show_obstacle_safety"]:
        for obstacle in planner.last_obstacles:
            _draw_world_rotated_box(
                surface,
                renderer,
                obstacle.position,
                obstacle.heading,
                obstacle.half_length,
                obstacle.half_width,
                color=(240, 140, 30),
                width=1,
            )

    draw_indices = _candidate_draw_indices(planner.last_candidates, planner.last_selected_index, config)
    for idx in draw_indices:
        path = planner.last_candidates[idx]
        if idx != planner.last_selected_index:
            color = (160, 160, 160) if path.collision else (80, 170, 255)
            _draw_world_polyline(surface, renderer, path.xy, color=color, width=2)

    if planner.last_candidates:
        selected_points = planner.last_candidates[planner.last_selected_index].xy
        _draw_world_polyline(surface, renderer, selected_points, color=(230, 45, 35), width=4)
        if len(selected_points) > 0:
            pygame.draw.circle(surface, (30, 220, 80), _world_to_topdown_screen(renderer, selected_points[-1]), 5)


def topdown_to_image(renderer):
    return WorldSurface.to_cv2_image(renderer.screen_canvas)


def _draw_world_polyline(surface, renderer, points, color, width):
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return
    pix_points = [_world_to_topdown_screen(renderer, point) for point in points]
    for start, end in zip(pix_points[:-1], pix_points[1:]):
        pygame.draw.line(surface, color, start, end, width)


def _draw_world_rotated_box(surface, renderer, center, heading, half_length, half_width, color, width):
    center = np.asarray(center, dtype=float)[:2]
    forward = np.array([math.cos(heading), math.sin(heading)])
    left = np.array([-math.sin(heading), math.cos(heading)])
    corners = [
        center + forward * half_length + left * half_width,
        center + forward * half_length - left * half_width,
        center - forward * half_length - left * half_width,
        center - forward * half_length + left * half_width,
    ]
    pix_points = [_world_to_topdown_screen(renderer, corner) for corner in corners]
    pygame.draw.polygon(surface, color, pix_points, width)


def _candidate_draw_indices(candidates: Sequence[FrenetPath], selected_index: int, config) -> List[int]:
    if not candidates:
        return []
    max_draw = int(config["frenet_max_draw_candidates"])
    if max_draw <= 0 or len(candidates) <= max_draw:
        return list(range(len(candidates)))

    ranked = sorted(
        range(len(candidates)),
        key=lambda idx: (
            idx != selected_index,
            candidates[idx].collision,
            candidates[idx].cost,
            abs(candidates[idx].target_d),
            idx,
        ),
    )
    return sorted(ranked[:max_draw])


def _world_to_topdown_screen(renderer, point) -> Tuple[int, int]:
    point = np.asarray(point, dtype=float)[:2]
    screen_w, screen_h = renderer.screen_canvas.get_size()
    agent = renderer.current_track_agent

    if renderer.target_agent_heading_up and agent is not None:
        diff = point - np.asarray(agent.position, dtype=float)[:2]
        heading = float(agent.heading_theta)
        forward = diff[0] * math.cos(heading) + diff[1] * math.sin(heading)
        right = diff[0] * math.sin(heading) - diff[1] * math.cos(heading)
        return int(screen_w / 2 + right * renderer.scaling), int(screen_h / 2 - forward * renderer.scaling)

    frame_canvas = renderer._frame_canvas
    frame_point = frame_canvas.pos2pix(float(point[0]), float(point[1]))
    if renderer.position is not None or agent is not None:
        if renderer.center_on_map:
            frame_w, frame_h = frame_canvas.get_size()
            center = (frame_w / 2, frame_h / 2)
        else:
            camera_position = renderer.position or agent.position
            center = frame_canvas.pos2pix(float(camera_position[0]), float(camera_position[1]))
    else:
        center = (screen_w / 2, screen_h / 2)

    offset = (center[0] - screen_w / 2, center[1] - screen_h / 2)
    return int(frame_point[0] - offset[0]), int(frame_point[1] - offset[1])
