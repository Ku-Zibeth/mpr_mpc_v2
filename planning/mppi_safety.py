"""Independent candidate certification under a short-horizon kinematic/CV model.

This never constructs SAC inputs. It reuses the legacy swept-OBB primitive,
but owns its settings, road cache and per-observation traffic snapshot.
"""

from dataclasses import dataclass, fields, replace
import math
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import shapely
import torch

from .safety import SafetyContext, SafetyResult, TrajectorySafetyEvaluator, _corners
from .swept_collision import _position_bound
from .swept_collision import swept_collisions


def repair_proven_lane_seams(road, lanes):
    """Remove ONLY holes wholly covered by an actual analytic circular lane.

    MetaDrive samples CircularLane polygons at equal longitudinal distances;
    neighboring radii therefore use different angles on their shared edge.
    Their union can contain millimeter slivers along an INTERNAL lane line.
    No buffer/area cutoff is used here. The convex hull must be contained in
    a single analytic annular sector; medians, exterior edges and unknown lane
    types are retained. This is independent of frozen SAC feature geometry.
    """
    from metadrive.component.lane.circular_lane import CircularLane
    circles = [lane for lane in lanes if type(lane) is CircularLane and not lane.forbidden]
    seams = []
    for lane in circles:
        shared = lane.radius+lane.width/2
        for other in circles:
            if (lane is other or lane.direction != other.direction or
                not np.array_equal(np.asarray(lane.center), np.asarray(other.center)) or
                abs(shared-(other.radius-other.width/2)) > 1e-10 or
                abs(math.atan2(math.sin(lane.start_phase-other.start_phase), math.cos(lane.start_phase-other.start_phase))) > 1e-10 or
                abs(lane.angle-other.angle) > 1e-10):
                continue
            # Error bound from MetaDrive's ACTUAL longitudinal tessellation,
            # not a configurable hole-size or road-clearance relaxation.
            step = max(lane.POLYGON_SAMPLE_RATE/lane.radius, other.POLYGON_SAMPLE_RATE/other.radius)
            sagitta = shared*(1-math.cos(step/2))
            seams.append((lane, shared, sagitta))
    def covered(hole):
        hull = hole.convex_hull
        vertices = np.asarray(hull.exterior.coords)[:-1]
        for lane, shared, sagitta in seams:
            center = np.asarray(lane.center, dtype=float)
            inner, outer = lane.radius-lane.width/2, lane.radius+lane.width/2
            # Only a floating-point error bound (64 float64 ULPs at this map
            # coordinate scale), typically ~1e-12 m. This is NOT a millimeter
            # road tolerance; the full physical road_margin is still applied.
            roundoff = 64*float(np.spacing(max(1., float(np.abs(center).max()), outer)))
            if inner <= 0 or hull.distance(shapely.Point(center)) < inner-roundoff:
                continue
            delta = vertices-center
            radial_max = np.linalg.norm(delta, axis=1).max()
            if radial_max > outer+roundoff or radial_max > shared+roundoff or hull.distance(shapely.Point(center)) < shared-sagitta-roundoff:
                continue
            mid = lane.start_phase+lane.direction*lane.angle/2
            angle = np.arctan2(delta[:, 1], delta[:, 0])-mid
            angle = np.arctan2(np.sin(angle), np.cos(angle))
            # A hull excluding the center and spanning <pi has no hidden
            # angular extrema between its vertices. Restrict to supported arcs.
            if lane.angle <= math.pi and angle.max()-angle.min() < math.pi and np.abs(angle).max() <= lane.angle/2+roundoff/inner:
                return True
        return False
    repaired, removed = [], 0
    for polygon in ([road] if road.geom_type == "Polygon" else road.geoms):
        holes = []
        for ring in polygon.interiors:
            if covered(shapely.Polygon(ring)):
                removed += 1
            else:
                holes.append(ring)
        repaired.append(shapely.Polygon(polygon.exterior, holes))
    if not removed:
        return road, 0
    result = shapely.union_all(repaired)
    if not result.is_valid or result.is_empty:
        raise RuntimeError("Analytically proven lane-seam repair produced invalid geometry.")
    return result, removed


@dataclass
class MPPISafetyContext(SafetyContext):
    brake_action_mapping_verified: bool = False
    target_speed: float | None = None
    lattice_actions: torch.Tensor | None = None
    lattice_positions: np.ndarray | torch.Tensor | None = None
    lane_width: float | None = None
    road_reference: object | None = None
    route_progress: float | None = None
    longitudinal_parameter_source: str = "configured_unvalidated_point_estimate"
    geometric_steer: float | None = None
    road_tangent_steer: float | None = None
    geometric_lateral_error: float = 0.
    geometric_heading_error: float = 0.
    geometric_body_margin: float = float("nan")
    dynamics_parameters: dict | None = None
    measurement_step: int | None = None
    lane_centers: tuple = ()
    rear_axle_to_center: float | None = None
    lattice_speed_limit: float | None = None
    sac_target_speed: float | None = None
    mppi_target_speed: float | None = None
    motion_signature: dict | None = None
    road_seams_repaired: int = 0


def geometric_recovery_reference(context, reference, lane_centers, control_dt):
    """One pure-pursuit proposal toward the nearest lane, never a safety override.

    Lookahead clears the front footprint plus one wheelbase, or one planning
    horizon of measured travel. At a low-speed bend, a rearward target can
    incorrectly aim across the lane just before the front body reaches a turn.
    Nearest-lane and Lattice directions are offered separately; obstacle and
    body clearance are resolved by the existing exact candidate checker.
    """
    s, lateral, _ = reference.project(context.position)
    yaw = reference.calc_yaw(s)
    context.geometric_lateral_error = float(lateral)
    context.geometric_heading_error = math.atan2(math.sin(yaw-context.heading), math.cos(yaw-context.heading))
    lane = min(lane_centers, key=lambda d: abs(d-lateral)) if lane_centers else lateral
    lookahead = max(context.length/2+context.wheelbase, context.speed*(len(context.lattice_actions)-1)*control_dt)
    target_s = min(reference.length, s+lookahead)
    target = reference.calc_position(target_s)+lane*np.array([-math.sin(reference.calc_yaw(target_s)), math.cos(reference.calc_yaw(target_s))])
    delta = target-context.position
    bearing = math.atan2(delta[1], delta[0])-context.heading
    steering = math.atan2(2*context.wheelbase*math.sin(bearing), max(float(np.linalg.norm(delta)), 1e-6))
    context.geometric_steer = float(np.clip(steering/context.max_steering, -1, 1))
    # A second, lane-tangent proposal avoids the rear-corner swing caused by
    # immediately aiming across the lane from a nearly parallel boundary pose.
    future_yaw = reference.calc_yaw(target_s)
    future_error = math.atan2(math.sin(future_yaw-context.heading), math.cos(future_yaw-context.heading))
    context.road_tangent_steer = float(np.clip(math.atan2(2*context.wheelbase*math.sin(future_error),
        lookahead)/context.max_steering, -1, 1))
    body = shapely.polygons(_corners(torch.tensor(context.position)[None, None], torch.tensor([[context.heading]]),
        context.length, context.width)[0, 0].numpy())
    context.geometric_body_margin = float(shapely.distance(body, context.road.boundary)) if context.road.covers(body) else -float(shapely.distance(body, context.road))


@dataclass
class MPPISafetyResult(SafetyResult):
    near_valid: torch.Tensor | None = None
    near_collision: torch.Tensor | None = None
    near_offroad: torch.Tensor | None = None
    near_kinematic: torch.Tensor | None = None
    full_valid: torch.Tensor | None = None
    risk_score: torch.Tensor | None = None
    road_risk: torch.Tensor | None = None
    collision_risk: torch.Tensor | None = None
    clearance_risk: torch.Tensor | None = None
    speed_risk: torch.Tensor | None = None
    lateral_risk: torch.Tensor | None = None
    kinematic_risk: torch.Tensor | None = None
    unrecoverable: torch.Tensor | None = None
    interval_offroad: torch.Tensor | None = None
    interval_kinematic: torch.Tensor | None = None
    speeds: torch.Tensor | None = None
    headings: torch.Tensor | None = None
    progress_estimate: torch.Tensor | None = None
    route_progress_estimate: torch.Tensor | None = None
    terminal_speed: torch.Tensor | None = None
    terminal_lateral_error: torch.Tensor | None = None
    committed_lateral_error: torch.Tensor | None = None
    center_model_valid: torch.Tensor | None = None
    legacy_model_valid: torch.Tensor | None = None
    primary_calibrated: torch.Tensor | None = None
    legacy_shadow: torch.Tensor | None = None
    center_near_collision: torch.Tensor | None = None
    legacy_near_collision: torch.Tensor | None = None
    center_near_offroad: torch.Tensor | None = None
    legacy_near_offroad: torch.Tensor | None = None
    near_boundary_margin: torch.Tensor | None = None
    terminal_boundary_margin: torch.Tensor | None = None
    boundary_margin_change: torch.Tensor | None = None
    committed_margin_change: torch.Tensor | None = None
    center_first_failure_substep: torch.Tensor | None = None
    legacy_first_failure_substep: torch.Tensor | None = None
    braking_continuation_valid: torch.Tensor | None = None


def projected_route_progress(positions, context, *, with_lateral=False):
    """World XY -> the SAME road polyline's longitudinal coordinate.

    Vectorized PolylineReference.project for endpoints; generic references use
    their project() directly. Heading displacement is never a substitute.
    """
    reference = getattr(context, "road_reference", None)
    if reference is None:
        missing = positions.new_full((len(positions),), float("nan"))
        return (missing, missing.clone()) if with_lateral else missing
    start = float(reference.project(context.position)[0])
    endpoints = positions[:, -1].detach().cpu().numpy()
    if all(hasattr(reference, name) for name in ("segments", "segment_lengths", "points", "s")):
        segments, lengths = reference.segments, reference.segment_lengths
        delta = endpoints[:, None]-reference.points[None, :-1]
        ratio = np.clip((delta*segments).sum(-1)/(lengths*lengths), 0., 1.)
        distances = ((delta-ratio[..., None]*segments)**2).sum(-1)
        index = distances.argmin(1)
        values = reference.s[index]+ratio[np.arange(len(index)), index]*lengths[index]
        offset = delta[np.arange(len(index)), index]-ratio[np.arange(len(index)), index, None]*segments[index]
        normal = np.stack((-segments[index, 1], segments[index, 0]), -1)/lengths[index, None]
        lateral = (offset*normal).sum(-1)
    else:
        projections = [reference.project(point) for point in endpoints]
        values = np.array([p[0] for p in projections])
        lateral = np.array([p[1] for p in projections])
    progress = positions.new_tensor(values-start)
    centers = np.asarray(getattr(context, "lane_centers", ()) or (0.,))
    error = positions.new_tensor(np.abs(lateral[:, None]-centers[None]).min(1))
    return (progress, error) if with_lateral else progress


class MPPISafetyFilter:
    def __init__(self, cfg, *, control_dt):
        self.cfg = dict(cfg)
        # GEOS lives on CPU. One batched copy per cache miss also avoids dozens
        # of small CUDA launches and CPU synchronizations in swept geometry.
        self.geometry = TrajectorySafetyEvaluator(cfg, control_dt=control_dt, device="cpu")
        self.safe_roads = {}
        self._lane_seam_cache = {}
        self._response_prediction = None
        self.two_layer = bool(cfg.get("two_layer", False))
        self.staged = self.two_layer and bool(cfg.get("staged", False))
        self.center_bicycle = bool(cfg.get("center_bicycle", False))
        self.center_model_guard = bool(cfg.get("center_model_guard", False))
        self.motion_evidence = None
        self.motion_evidence_sha = None
        evidence_path = cfg.get("motion_validation_path")
        self.motion_evidence_path = Path(str(evidence_path)).expanduser() if evidence_path else None
        if self.motion_evidence_path is not None and self.motion_evidence_path.is_file():
            try:
                payload = self.motion_evidence_path.read_bytes()
                evidence = json.loads(payload)
                data = Path(evidence.get("dataset", ""))
                if (self.valid_motion_evidence(evidence) and data.is_file()
                        and hashlib.sha256(data.read_bytes()).hexdigest() == evidence.get("dataset_sha256")):
                    self.motion_evidence, self.motion_evidence_sha = evidence, hashlib.sha256(payload).hexdigest()
                    self._evidence_stamp = (self.motion_evidence_path.stat().st_mtime_ns, data.stat().st_mtime_ns)
                    self._evidence_data_path = data
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                pass  # Malformed/stale evidence retains the conservative dual check.
        self.action_delay_s = float(cfg.get("action_delay_s", 0.))
        if not math.isfinite(self.action_delay_s) or self.action_delay_s < 0:
            raise ValueError("Action delay must be finite and nonnegative.")
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
        self._brake_filters = {}
        self.braking_evidence_path = Path(str(cfg.get("braking_validation_path", "")))

    def reset(self):
        self.geometry.reset()
        self.safe_roads.clear()
        self._lane_seam_cache.clear()
        self._response_prediction = None

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
        result = MPPISafetyContext(**values, brake_action_mapping_verified=True)
        token = id(source.road)
        if token not in self._lane_seam_cache:
            road, count = repair_proven_lane_seams(source.road, source.source_lanes)
            # Holding source prevents id reuse while this observation cache lives.
            self._lane_seam_cache[token] = (source.road, road, count)
        _, result.road, result.road_seams_repaired = self._lane_seam_cache[token]
        result.measurement_step = getattr(getattr(vehicle, "engine", None), "episode_step", None)
        rear = getattr(vehicle, "REAR_WHEELBASE", None)
        if rear is not None and math.isfinite(float(rear)) and 0 < float(rear) < wheelbase:
            result.rear_axle_to_center = float(rear)
        # Bullet engine/brake forces are NOT longitudinal acceleration bounds:
        # load transfer, tire traction, drag and speed dependence intervene.
        # MetaDrive exposes geometry but no certified acceleration/deceleration
        # envelope. Keep the existing point model and report its provenance.
        result.dynamics_parameters = {key: config.get(key) for key in ("mass", "max_engine_force", "max_brake_force")}
        result.dynamics_parameters.update(model_acceleration=float(self.cfg.get("max_acceleration", 4.)),
            model_deceleration=float(self.cfg.get("max_deceleration", 6.)), longitudinal_bounds_calibrated=False)
        result.motion_signature = dict(length=result.length, width=result.width, wheelbase=result.wheelbase,
            rear_axle=result.rear_axle_to_center, steering=result.max_steering,
            mass=float(config.get("mass") or getattr(vehicle, "MASS", 0.)),
            engine=float(config.get("max_engine_force", 0.)), brake=float(config.get("max_brake_force", 0.)),
            friction=float(config.get("wheel_friction", 0.)))
        physics = getattr(getattr(vehicle, "engine", None), "global_config", {})
        result.motion_signature.update(physics_dt=physics.get("physics_world_step_size"), decision_repeat=physics.get("decision_repeat"))
        return result

    @staticmethod
    def valid_motion_evidence(e):
        try:
            if not (e["schema"] == 1 and e["predictor_schema"] == "center_bicycle_v1"
                    and e["protocol"] == "disjoint_scene_physics_tick_v1"
                    and e["fit_windows"] >= 20 and e["heldout_windows"] >= 20
                    and e["fit_scenes"] and e["heldout_scenes"]
                    and set(e["fit_scenes"]).isdisjoint(e["heldout_scenes"])):
                return False
            if e.get("validated", False) and not e.get("event_validation_supported", False):
                return False
            for horizon in e["valid_horizons"]:
                if type(horizon) is not int or horizon <= 0:
                    return False
                bound = e["bounds"][str(horizon)]
                if not all(math.isfinite(bound[k]) and bound[k] >= 0 for k in
                           ("position_m", "heading_rad", "lateral_m", "speed_mps")):
                    return False
            for name, value in e.get("parameters", {}).items():
                if name not in ("max_acceleration", "max_deceleration", "beta_gain", "yaw_gain") or not math.isfinite(value) or value <= 0:
                    return False
            domain = e["domain"]
            return (len(domain["speed"]) == 2 and 0 <= domain["speed"][0] <= domain["speed"][1]
                    and all(math.isfinite(v) for v in domain["speed"])
                    and len(domain["action_min"]) == len(domain["action_max"]) == 2
                    and all(-1 <= lo <= hi <= 1 for lo, hi in zip(domain["action_min"], domain["action_max"])))
        except (ValueError, TypeError, KeyError):
            return False

    def calibrated_mask(self, actions, context):
        """Evidence-gated operating domain; absent/stale data retains both checks."""
        no = torch.zeros(len(actions), dtype=torch.bool)
        e = self.motion_evidence
        if not e or not e.get("validated", False) or context.rear_axle_to_center is None:
            return no
        if (not self.motion_evidence_path.is_file() or not self._evidence_data_path.is_file() or
            (self.motion_evidence_path.stat().st_mtime_ns, self._evidence_data_path.stat().st_mtime_ns) != self._evidence_stamp):
            return no
        if (abs(e.get("control_dt", -1)-self.geometry.dt) > 1e-9 or
                actions.shape[1]-1 not in e.get("valid_horizons", []) or
                e.get("nominal_config") != self.motion_nominal_config() or
                e.get("predictor_sha256") != self.motion_predictor_sha()):
            return no
        if not any(context.motion_signature == signature for signature in e.get("vehicle_signatures", [])):
            return no
        domain = e["domain"]
        if not domain["speed"][0] <= context.speed <= domain["speed"][1]:
            return no
        controls = actions.detach().cpu()[:, :-1]
        return (torch.isfinite(controls).all((1, 2)) &
            (controls >= controls.new_tensor(domain["action_min"])).all((1, 2)) &
            (controls <= controls.new_tensor(domain["action_max"])).all((1, 2)))

    def motion_nominal_config(self):
        return {"max_acceleration": float(self.cfg.get("max_acceleration", 4.)),
                "max_deceleration": float(self.cfg.get("max_deceleration", 6.)), "substeps": self.geometry.substeps}

    @staticmethod
    def motion_predictor_sha():
        return hashlib.sha256((inspect.getsource(MPPISafetyFilter._center_rollout)
            +inspect.getsource(TrajectorySafetyEvaluator.rollout)).encode()).hexdigest()

    def calibrated_extra_margin(self, actions, context):
        ready = self.calibrated_mask(actions, context)
        if not bool(ready.any()):
            return 0.
        bound = self.motion_evidence["bounds"][str(actions.shape[1]-1)]
        # Convert orientation error into a body-corner displacement. Keep the
        # original margin as the minimum; never subtract a road/collision bound.
        pose = bound["position_m"]+math.hypot(context.length, context.width)*math.sin(min(math.pi, bound["heading_rad"])/2)
        return max(0., pose-self.geometry.model_margin)

    def response_errors(self, vehicle):
        """Compare a stored executed-action model prediction after ONE env step.

        Repeated fixed-state planning is excluded by the simulator step token.
        These are kinematic/actuation residuals, not latent WM errors.
        """
        pending = self._response_prediction
        step = getattr(getattr(vehicle, "engine", None), "episode_step", None)
        if pending is None or step is None or step != pending["step"]+1:
            return {}
        actual_speed = float(np.linalg.norm(vehicle.velocity))
        angle = float(vehicle.heading_theta)-pending["heading"]
        actual_action = np.asarray(getattr(vehicle, "current_action", pending["action"]))
        speed_residual = actual_speed-pending["speed"]
        return {"physics_position_error_m": float(np.linalg.norm(vehicle.position-pending["position"])),
            "physics_speed_error_mps": abs(speed_residual),
            "physics_heading_error_rad": abs(math.atan2(math.sin(angle), math.cos(angle))),
            "physics_acceleration_bias_mps2": speed_residual/self.geometry.dt,
            "physics_braking_response_bias_mps2": speed_residual/self.geometry.dt if pending["action"][1] < 0 else float("nan"),
            "execution_action_error": float(np.linalg.norm(actual_action-pending["action"]))}

    def predict_response(self, actions, context, vehicle):
        # Use the same horizon/domain as the certified model, then take its
        # first control. A prefix-only call could use a different evidence gate.
        positions, headings, speeds = (value[:, :self.geometry.substeps+1]
            for value in self._rollout(actions[None].detach().cpu(), context))
        legacy_xy, legacy_yaw, legacy_speed = self.geometry.rollout(actions[None, :2].detach().cpu(), context)
        step = getattr(getattr(vehicle, "engine", None), "episode_step", None)
        if step is not None:
            self._response_prediction = dict(step=step, position=positions[0, -1].numpy(), heading=float(headings[0, -1]),
                speed=float(speeds[0, -1]), action=actions[0].detach().cpu().numpy())
        return dict(control_dt=self.geometry.dt, measured_speed=context.speed,
            reference_points=getattr(context.road_reference, "points", np.empty((0, 2))).tolist(),
            center=dict(position=positions[0, -1].tolist(), heading=float(headings[0, -1]), speed=float(speeds[0, -1])),
            legacy=dict(position=legacy_xy[0, -1].tolist(), heading=float(legacy_yaw[0, -1]), speed=float(legacy_speed[0, -1])))



    def _rollout(self, controls, context):
        # Observation-local cache: exact promotion can reuse a preview rollout
        # without re-integrating it. Prefix and full-horizon keys are distinct.
        signature = (tuple(np.asarray(context.position)), context.heading, context.speed,
            context.wheelbase, context.max_steering, str(controls.dtype), self.geometry.dt,
            self.geometry.substeps, self.cfg.get("max_acceleration", 4.), self.cfg.get("max_deceleration", 6.),
            self.center_bicycle, getattr(context, "rear_axle_to_center", None))
        signature += (self.motion_evidence_sha,)
        cache = context.cache.setdefault(("mppi_rollouts", signature), {})
        keys = [row.tobytes() for row in controls.contiguous().numpy()]
        missing = {key: i for i, key in enumerate(keys) if key not in cache}
        if missing:
            requested = controls[list(missing.values())]
            packed = (self._center_rollout(requested, context) if self.center_bicycle and context.rear_axle_to_center is not None
                      else self.geometry.rollout(requested, context))
            for i, key in enumerate(missing):
                cache[key] = tuple(tensor[i] for tensor in packed)
        return tuple(torch.stack([cache[key][j] for key in keys]) for j in range(3))

    def _center_rollout(self, actions, context, *, parameters=None):
        """Kinematic bicycle at the measured chassis CENTER, not rear axle.

        Bullet wheel locations are +FRONT_WHEELBASE/-REAR_WHEELBASE relative
        to this origin. beta rotates CENTER velocity relative to chassis yaw.
        The swept checker uses endpoints/chord and body-yaw increments, so its
        continuous arc enclosure remains applicable without changing thresholds.
        Legacy geometry.rollout still constructs the frozen SAC features.
        """
        controls = actions[:, :-1].nan_to_num().clamp(-1, 1)
        xy = actions.new_tensor(context.position).expand(len(actions), -1).clone()
        yaw = actions.new_full((len(actions),), context.heading)
        speed = actions.new_full((len(actions),), context.speed)
        positions, headings, speeds = [xy.clone()], [yaw.clone()], [speed.clone()]
        dt = self.geometry.dt/self.geometry.substeps
        if parameters is not None:
            ready = torch.ones(len(actions), dtype=torch.bool)
            fit = parameters
        else:
            ready = self.calibrated_mask(actions, context)
            fit = self.motion_evidence.get("parameters", {}) if self.motion_evidence else {}
        ready = ready.to(actions.device)
        parameter = lambda name, nominal: torch.where(ready, actions.new_tensor(float(fit.get(name, nominal))), actions.new_tensor(nominal))
        positive_acceleration = parameter("max_acceleration", float(self.cfg.get("max_acceleration", 4.)))
        braking_acceleration = parameter("max_deceleration", float(self.cfg.get("max_deceleration", 6.)))
        beta_gain, yaw_gain = parameter("beta_gain", 1.), parameter("yaw_gain", 1.)
        for action in controls.unbind(1):
            delta = action[:, 0]*context.max_steering
            beta = (context.rear_axle_to_center/context.wheelbase*delta.tan()).atan()*beta_gain
            acceleration = action[:, 1]*torch.where(action[:, 1] >= 0,
                positive_acceleration, braking_acceleration)
            for _ in range(self.geometry.substeps):
                next_speed = (speed+dt*acceleration).clamp_min(0)
                middle = (speed+next_speed)/2
                turn = dt*middle*beta.cos()*delta.tan()/context.wheelbase*yaw_gain
                direction = yaw+turn/2+beta
                distance = dt*middle*torch.sinc(turn/(2*math.pi))
                xy = xy+distance[:, None]*torch.stack((direction.cos(), direction.sin()), -1)
                yaw, speed = yaw+turn, next_speed
                positions.append(xy.clone()); headings.append(yaw.clone()); speeds.append(speed.clone())
        return torch.stack(positions, 1), torch.stack(headings, 1), torch.stack(speeds, 1)

    @torch.no_grad()
    def _strict_check(self, actions, context):
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
        positions, headings, speeds = self._rollout(controls, context)
        finite = torch.isfinite(controls).all((1, 2)) & (controls.abs() <= 1).all((1, 2))
        previous = controls.new_tensor(context.previous_action).view(1, 1, 2).expand(len(controls), -1, -1)
        explicit = controls[:, :-1]
        rate = torch.diff(torch.cat((previous, explicit), 1)[:, :, 0], dim=1).abs()*context.max_steering/self.geometry.dt
        curvature = (explicit[:, :, 0]*context.max_steering).tan().abs()/context.wheelbase
        acceleration = torch.maximum(speeds[:, :-1], speeds[:, 1:]).square()*curvature.repeat_interleave(self.geometry.substeps, 1)
        interval_kinematic = ((rate > float(self.cfg.get("max_steering_rate", 7.0))).repeat_interleave(self.geometry.substeps, 1)
            | (acceleration > float(self.cfg.get("max_lateral_acceleration", 6.)))
            | (headings.diff(dim=1).abs() >= math.pi))
        speed_limit = getattr(context, "lattice_speed_limit", None)
        if speed_limit is not None:
            # An already overspeed state cannot brake instantaneously. Do not
            # permit its speed to increase further; target/risk still demand
            # deceleration toward the same physical ceiling.
            interval_kinematic |= speeds[:, 1:] > max(float(speed_limit), context.speed)+1e-6
        kinematic = ~finite | interval_kinematic.any(1)
        # Enclose the full translation/rotation between adjacent substeps, not
        # just the center or endpoint corners. The same arc-error bound is used
        # by the continuous collision primitive.
        middle = (headings[:, :-1] + headings[:, 1:]) / 2
        extra_margin = self.calibrated_extra_margin(controls, context) if self.center_bicycle and context.rear_axle_to_center is not None else 0.
        length = context.length + 2*(self.geometry.model_margin+extra_margin)
        width = context.width + 2*(self.geometry.model_margin+extra_margin)
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
        # Rotated cars in a narrow road often fail the axis-aligned box test
        # although their complete swept footprint fits. A conservative hull
        # of ALL interval vertices, inflated by the largest interval error,
        # can certify the same set with N queries instead of N*substeps.
        uncertain = np.flatnonzero(~whole_inside)
        if len(uncertain):
            hull = shapely.convex_hull(shapely.multipoints(vertices[uncertain].numpy().reshape(len(uncertain), -1, 2)))
            inflated = shapely.buffer(hull, pad[uncertain].amax(1).numpy()/math.cos(math.pi/32)+1e-6, quad_segs=8)
            whole_inside[uncertain] = shapely.covers(safe_road, inflated)
        road_inside = np.ones((len(actions), positions.shape[1]-1), dtype=bool)
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
            road_inside[paths] = inside.reshape(len(paths), -1)
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
        return MPPISafetyResult(valid, collision, offroad, kinematic, clearance, ttc,
                            unavailable, torch.zeros_like(ttc), positions, unavailable, torch.isfinite(interval),
                            torch.isfinite(interval[:, :self.geometry.substeps]).any(1), torch.from_numpy(first_offroad),
                            full_valid=valid.clone(), interval_offroad=torch.from_numpy(~road_inside),
                            interval_kinematic=interval_kinematic, speeds=speeds, headings=headings)

    def committed_controls(self, horizon):
        # MetaDrive applies the new command in before_step, then holds it for
        # decision_repeat physics ticks. There is no queued actuator delay.
        # Additional deployment delay must be explicitly configured, not inferred
        # from wall time while this synchronous simulator is paused.
        return min(horizon, 1+math.ceil(self.action_delay_s/self.geometry.dt))

    def _braking_continuations(self, actions, context, committed):
        hold = actions.clone()
        hold[:, committed:, 0] = actions[:, committed-1:committed, 0]
        hold[:, committed:, 1] = self.emergency_throttle
        guidance = getattr(context, "lattice_actions", None)
        if guidance is None:
            return hold
        guided = hold.clone()
        limit = float(self.cfg.get("max_steering_rate", 7.))*self.geometry.dt/context.max_steering
        previous = actions[:, committed-1, 0]
        for step in range(committed, actions.shape[1]):
            previous = previous+(float(guidance[min(step, len(guidance)-1), 0])-previous).clamp(-limit, limit)
            guided[:, step, 0] = previous
        return torch.cat((hold, guided), 0)

    def braking_domain(self, context, actions):
        """Read evidence on every observation; stale/out-of-domain is UNKNOWN.

        A brake-force/mass quotient or the nominal 6m/s² is not evidence.
        Data and predictor identities, disjoint scenes, and heldout coverage
        are required. This does not qualify the Center-only motion policy.
        """
        try:
            payload = self.braking_evidence_path.read_bytes()
            evidence = json.loads(payload)
            data = Path(evidence["dataset"])
            if (not evidence.get("validated") or evidence.get("schema") != "brake_envelope_v1"
                    or evidence.get("good_fit", 0) < 8 or evidence.get("good_heldout", 0) < 8
                    or evidence.get("heldout_bound_violations", 1) != 0
                    or not evidence["fit_scenes"] or not evidence["heldout_scenes"]
                    or not set(evidence["fit_scenes"]).isdisjoint(evidence["heldout_scenes"])
                    or hashlib.sha256(data.read_bytes()).hexdigest() != evidence["dataset_sha256"]
                    or evidence["predictor_sha256"] != self.motion_predictor_sha()
                    or abs(evidence["control_dt"]-self.geometry.dt) > 1e-9):
                return None, "unvalidated_or_stale_braking_model"
            bound = float(evidence["deceleration_lower_mps2"])
            if not math.isfinite(bound) or bound <= 0:
                return None, "invalid_braking_bound"
            domain = evidence["domain"]
            scalars = [evidence["acceleration_upper_mps2"], evidence["validated_prediction_time_s"],
                       domain["max_speed_mps"], domain["max_curvature_inv_m"]]
            if not all(math.isfinite(float(value)) and float(value) >= 0 for value in scalars):
                return None, "invalid_braking_domain"
            curvature = (actions.detach().cpu()[..., 0]*context.max_steering).tan().abs().max()/context.wheelbase
            if not (0 <= context.speed <= domain["max_speed_mps"] and curvature <= domain["max_curvature_inv_m"]):
                return None, "outside_braking_domain"
            if context.motion_signature not in evidence["vehicle_signatures"]:
                return None, "unknown_vehicle_signature"
            # CV envelopes need their own disjoint measured evidence. Even a
            # currently stationary vehicle may accelerate; zero speed alone
            # must not turn an unidentified actor into a known static cone.
            if context.obstacles and not evidence.get("traffic_validated", False):
                return None, "unknown_traffic_prediction_envelope"
            return (evidence, hashlib.sha256(payload).hexdigest()), "validated_domain"
        except (OSError, KeyError, ValueError, TypeError, AttributeError):
            return None, "missing_braking_evidence"

    @torch.no_grad()
    def certify(self, actions, context):
        """Exact candidate + SAME-prefix stopping witness, independent of WM H.

        An observation-local certificate is a model/evidence check, not a
        physical-world guarantee. No unknown range is silently extrapolated.
        Center AND Legacy must approve the very SAME witness.
        """
        controls = actions.detach().cpu()
        cache = context.cache.setdefault("adaptive_certificates", {})
        key = (controls.numpy().tobytes(), tuple(controls.shape))
        # Evidence may have been revoked since a preceding call.
        domain, reason = self.braking_domain(context, controls)
        stamp = domain[1] if domain else reason
        key += (stamp,)
        if key in cache:
            return cache[key]
        result = self.check(controls, context)
        if result.near_valid is None:
            count = self.committed_controls(controls.shape[1]-1)
            near = self._strict_check(controls[:, :count+1], context)
            guard = self._strict_check(controls[:, :count+1], replace(context, rear_axle_to_center=None))
            result.near_valid = near.valid & guard.valid
            result.near_collision = near.collision | guard.collision
            result.near_offroad = near.offroad | guard.offroad
        result.nominal_valid = result.valid.clone()
        result.safety_unknown = torch.zeros(len(controls), dtype=torch.bool)
        result.certification_reason = reason
        result.hold_braking_valid = torch.zeros(len(controls), dtype=torch.bool)
        result.guided_braking_valid = torch.zeros(len(controls), dtype=torch.bool)
        result.safety_horizon_steps = 0
        result.stopping_witnesses = [None]*len(controls)
        result.evidence_sha256 = None
        result.stopping_time_s = float("nan")
        # Rejection classification only; this NEVER edits the road geometry.
        # A hole is not assumed to be an internal seam or a drivable median.
        has_holes = any(len(poly.interiors) for poly in ([context.road] if context.road.geom_type == "Polygon" else context.road.geoms))
        result.road_rejection_kind = ("swept_outer_boundary_or_road_end" if result.offroad.any() and not has_holes
                                      else "unproven_hole_or_outer_boundary" if result.offroad.any() else "none")
        if domain is None:
            result.valid[:] = False
            result.safety_unknown = result.near_valid.clone()
            cache[key] = result
            return result
        evidence, digest = domain
        committed = self.committed_controls(controls.shape[1]-1)
        deceleration = float(evidence["deceleration_lower_mps2"])
        # Include worst observed positive-prefix acceleration before braking.
        prefix_speed = context.speed+float(evidence["acceleration_upper_mps2"])*committed*self.geometry.dt
        if prefix_speed > evidence["domain"]["max_speed_mps"]:
            result.valid[:] = False
            result.safety_unknown = result.near_valid.clone()
            result.certification_reason = "prefix_exits_braking_speed_domain"
            cache[key] = result
            return result
        steps = committed+math.ceil(prefix_speed/deceleration/self.geometry.dt)+1
        result.stopping_time_s = prefix_speed/deceleration
        maximum = int(self.cfg.get("max_safety_steps", 100))
        if (steps > maximum or steps*self.geometry.dt > evidence["validated_prediction_time_s"]):
            result.valid[:] = False
            result.safety_unknown[:] = True
            result.certification_reason = "outside_validated_safety_horizon"
            cache[key] = result
            return result
        extended = controls[:, -1:].expand(-1, steps+1, -1).clone()
        extended[:, :committed] = controls[:, :committed]
        witnesses = self._braking_continuations(extended, context, committed)
        # Lattice guidance may bend beyond the evidence support even when the
        # submitted prefix was in-domain. Never certify that extrapolation.
        witness_curvature = (witnesses[..., 0]*context.max_steering).tan().abs().amax(1)/context.wheelbase
        in_domain = witness_curvature <= evidence["domain"]["max_curvature_inv_m"]
        # Isolated candidate checker: NEVER mutate the frozen SAC geometry
        # service or change the nominal predictor's training/input semantics.
        if digest not in self._brake_filters:
            cfg = dict(self.cfg, two_layer=False, staged=False, max_deceleration=deceleration,
                       max_acceleration=evidence["acceleration_upper_mps2"], motion_validation_path=None,
                       obstacle_velocity_error=evidence.get("obstacle_velocity_error_mps", 0.),
                       obstacle_acceleration_bound=evidence.get("obstacle_acceleration_error_mps2", 0.))
            self._brake_filters[digest] = MPPISafetyFilter(cfg, control_dt=self.geometry.dt)
            self._brake_filters[digest].safe_roads = self.safe_roads
        checker = self._brake_filters[digest]
        center = checker._strict_check(witnesses, context)
        legacy = checker._strict_check(witnesses, replace(context, rear_axle_to_center=None))
        passing = (in_domain & center.valid & legacy.valid & (center.speeds[:, -1] <= 1e-6)
                   & (legacy.speeds[:, -1] <= 1e-6)).reshape(-1, len(controls))
        result.hold_braking_valid = passing[0]
        if passing.shape[0] > 1:
            result.guided_braking_valid = passing[1]
        result.valid = result.near_valid & passing.any(0)
        result.safety_unknown = result.near_valid & ~in_domain.reshape(-1, len(controls)).any(0)
        for i in result.valid.nonzero().flatten().tolist():
            branch = int(passing[:, i].nonzero()[0])
            result.stopping_witnesses[i] = witnesses[branch*len(controls)+i].clone()
        result.certification_reason = "stopping_witness_checked"
        result.safety_horizon_steps = steps
        result.evidence_sha256 = digest
        cache[key] = result
        return result

    @torch.no_grad()
    def preview(self, actions, context):
        """Exact committed prefix, approximate tail. NEVER an execution certificate.

        Tail footprint/circle samples are priority estimates only. check() still
        uses swept OBBs and, when needed, the same-prefix braking witnesses.
        """
        if not self.staged:
            return self.check(actions, context)
        controls = actions.detach().cpu()
        horizon = controls.shape[1]-1
        committed = self.committed_controls(horizon)
        result = self._strict_check(controls[:, :committed+1], context)
        if self.center_bicycle and self.center_model_guard and context.rear_axle_to_center is not None:
            guard = self._strict_check(controls[:, :committed+1], replace(context, rear_axle_to_center=None))
            shadow = self.calibrated_mask(controls, context)
            result.valid &= shadow | guard.valid
            result.collision |= ~shadow & guard.collision
            result.offroad |= ~shadow & guard.offroad
            result.kinematic_violation |= ~shadow & guard.kinematic_violation
        result.near_valid = result.valid.clone()
        result.near_collision = result.collision.clone()
        result.near_offroad = result.offroad.clone()
        result.near_kinematic = result.kinematic_violation.clone()
        result.near_valid &= torch.isfinite(controls).all((1, 2)) & (controls.abs() <= 1).all((1, 2))
        if self.action_delay_s+self.geometry.dt > horizon*self.geometry.dt+1e-9:
            result.near_valid[:] = False
        # Do not run expensive tail geometry for candidates already rejected.
        live = result.near_valid.nonzero().flatten()
        count = len(controls)
        result.risk_score = torch.full((count,), float("inf"))
        result.full_valid = torch.zeros(count, dtype=torch.bool)
        result.unrecoverable = torch.zeros(count, dtype=torch.bool)
        result.progress_estimate = torch.zeros(count)
        result.route_progress_estimate = torch.full((count,), float("nan"))
        result.terminal_speed = torch.full((count,), float("nan"))
        result.terminal_lateral_error = torch.full((count,), float("nan"))
        result.committed_lateral_error = torch.full((count,), float("nan"))
        result.boundary_margin = torch.full((count,), float("nan"))
        for name in ("road_risk", "collision_risk", "clearance_risk", "speed_risk", "lateral_risk", "kinematic_risk"):
            setattr(result, name, torch.zeros(count))
        if len(live):
            positions, headings, speeds = self._rollout(controls[live], context)
            direction = positions.new_tensor([math.cos(context.heading), math.sin(context.heading)])
            result.progress_estimate[live] = (positions[:, -1]-positions[:, 0])@direction
            progress, lateral_error = projected_route_progress(positions, context, with_lateral=True)
            result.route_progress_estimate[live], result.terminal_lateral_error[live] = progress, lateral_error
            _, result.committed_lateral_error[live] = projected_route_progress(
                positions[:, :committed*self.geometry.substeps+1], context, with_lateral=True)
            result.terminal_speed[live] = speeds[:, -1]
            # Discrete body footprints cannot certify between-sample safety.
            corners = _corners(positions[:, ::self.geometry.substeps], headings[:, ::self.geometry.substeps],
                context.length+2*self.geometry.model_margin, context.width+2*self.geometry.model_margin)
            # One hull query per candidate replaces H+1 polygon queries. This
            # over-approximates discrete body samples for PRIORITY only; curved
            # swept intervals are still checked exactly at promotion.
            hulls = shapely.convex_hull(shapely.multipoints(corners.numpy().reshape(len(live), -1, 2)))
            key = (id(context.road), float(self.cfg.get("road_margin", .15)))
            safe_road = self.safe_roads[key][1]  # already built by prefix check
            road = torch.from_numpy(np.asarray(shapely.covers(safe_road, hulls)))
            boundary = context.cache.setdefault("mppi_road_boundary", context.road.boundary)
            margin = torch.as_tensor(shapely.distance(boundary, hulls), dtype=positions.dtype)
            margin = margin.masked_fill(~road, 0.)
            result.boundary_margin[live] = margin
            result.road_risk[live] = (~road).float()
            result.offroad[live] |= ~road
            result.collision_risk[live] = 0.
            if context.obstacles:
                xy = positions.new_tensor(np.asarray([o.position for o in context.obstacles]))
                velocity = positions.new_tensor(np.asarray([o.velocity for o in context.obstacles]))
                times = torch.arange(positions.shape[1])*self.geometry.dt/self.geometry.substeps
                other = xy[None]+times[:, None, None]*velocity[None]
                radii = positions.new_tensor([math.hypot(o.length, o.width)/2 for o in context.obstacles])
                gap = (positions[:, :, None]-other[None]).norm(dim=-1)-radii-math.hypot(context.length, context.width)/2
                clearance = gap.clamp_min(0).amin((1, 2))
                result.min_clearance[live] = clearance
                result.clearance_risk[live] = torch.exp(-clearance/context.width)
                result.collision_risk[live] = (gap <= 0).any(2).float().mean(1)
            target = getattr(context, "target_speed", None)
            if target is not None and math.isfinite(target) and target >= 0:
                result.speed_risk[live] = ((speeds[:, -1]-target).clamp_min(0)/max(target, 1.)).square()
            rate = torch.diff(torch.cat((controls.new_tensor(context.previous_action)[None, None].expand(len(live), -1, -1),
                controls[live, :-1]), 1)[:, :, 0], dim=1).abs()*context.max_steering/self.geometry.dt
            lateral = speeds[:, 1:].square()*(controls[live, :-1, 0]*context.max_steering).tan().abs().repeat_interleave(self.geometry.substeps, 1)/context.wheelbase
            result.kinematic_risk[live] = (((rate > float(self.cfg.get("max_steering_rate", 7.))).float().mean(1))
                +(lateral > float(self.cfg.get("max_lateral_acceleration", 6.))).float().mean(1))
            result.risk_score[live] = (result.road_risk[live]+torch.exp(-margin.clamp_min(0)/context.width)
                +result.clearance_risk[live]+result.collision_risk[live]+result.speed_risk[live]+result.kinematic_risk[live])
        result.valid = result.near_valid.clone()  # provisional; callers MUST check() before execution
        return result

    @torch.no_grad()
    def check(self, actions, context):
        result = self._check_nominal(actions, context)
        result.center_model_valid = result.valid.clone() if self.center_bicycle and context.rear_axle_to_center is not None else None
        result.center_near_collision, result.center_near_offroad = result.near_collision, result.near_offroad
        for name in ("collision", "offroad", "kinematic_violation", "braking_continuation_valid"):
            value = getattr(result, name)
            setattr(result, "center_"+name, value.clone() if value is not None else None)
        if self.center_bicycle and self.center_model_guard and context.rear_axle_to_center is not None:
            # Preserve the original geometric standard as a second hypothesis.
            # Bullet tire transients need not instantly reach steady beta. Both
            # hypotheses receive FULL swept/body checks and braking witnesses;
            # neither a failed check nor missing evidence can be averaged away.
            guard = self._check_nominal(actions, replace(context, rear_axle_to_center=None))
            result.legacy_model_valid = guard.valid.clone()
            for name in ("collision", "offroad", "kinematic_violation", "braking_continuation_valid"):
                value = getattr(guard, name)
                setattr(result, "legacy_"+name, value.clone() if value is not None else None)
            result.legacy_near_collision, result.legacy_near_offroad = guard.near_collision, guard.near_offroad
            shadow = result.primary_calibrated
            result.legacy_shadow = shadow.clone()
            for name in ("valid", "near_valid", "full_valid"):
                a, b = getattr(result, name), getattr(guard, name)
                if a is not None and b is not None:
                    setattr(result, name, a & (shadow | b))
            for name in ("collision", "offroad", "kinematic_violation", "near_collision", "near_offroad", "near_kinematic", "unrecoverable"):
                a, b = getattr(result, name), getattr(guard, name)
                if a is not None and b is not None:
                    setattr(result, name, a | (~shadow & b))
            for name in ("risk_score", "road_risk", "collision_risk", "clearance_risk", "speed_risk", "lateral_risk", "kinematic_risk",
                         "terminal_lateral_error", "committed_lateral_error"):
                a, b = getattr(result, name), getattr(guard, name)
                if a is not None and b is not None:
                    setattr(result, name, torch.where(shadow, a, torch.maximum(a, b)))
            for name in ("boundary_margin", "min_clearance", "min_ttc"):
                a, b = getattr(result, name), getattr(guard, name)
                if a is not None and b is not None:
                    setattr(result, name, torch.where(shadow, a, torch.minimum(a, b)))
            result.legacy_first_failure_substep = guard.center_first_failure_substep
            for name in ("near_boundary_margin", "terminal_boundary_margin", "boundary_margin_change", "committed_margin_change"):
                if getattr(result, name) is not None and getattr(guard, name) is not None:
                    setattr(result, name, torch.minimum(getattr(result, name), getattr(guard, name)))
            result.braking_continuation_valid &= guard.braking_continuation_valid
        return result

    @torch.no_grad()
    def _check_nominal(self, actions, context):
        result = self._strict_check(actions, context)
        result.primary_calibrated = self.calibrated_mask(actions, context)
        failures = result.interval_offroad | result.interval_collision | result.interval_kinematic
        result.center_first_failure_substep = torch.where(failures.any(1), failures.int().argmax(1)+1, -1)
        result.braking_continuation_valid = result.full_valid.clone()
        result.route_progress_estimate, result.terminal_lateral_error = projected_route_progress(result.positions, context, with_lateral=True)
        result.terminal_speed = result.speeds[:, -1]
        committed = self.committed_controls(actions.shape[1]-1)
        _, result.committed_lateral_error = projected_route_progress(
            result.positions[:, :committed*self.geometry.substeps+1], context, with_lateral=True)
        if not self.two_layer:
            return result
        horizon = actions.shape[1]-1
        committed = self.committed_controls(horizon)
        near = committed*self.geometry.substeps
        finite = torch.isfinite(actions).all((1, 2)) & (actions.abs() <= 1).all((1, 2))
        result.near_offroad = result.interval_offroad[:, :near].any(1)
        result.near_collision = result.interval_collision[:, :near].any(1)
        result.near_kinematic = result.interval_kinematic[:, :near].any(1)
        near_valid = finite.cpu() & ~result.near_offroad & ~result.near_collision & ~result.near_kinematic
        if self.action_delay_s+self.geometry.dt > horizon*self.geometry.dt+1e-9:
            near_valid[:] = False  # cannot certify an uncovered committed interval
        if float(self.cfg.get("hard_ttc", 0.)) > 0 or float(self.cfg.get("hard_clearance", 0.)) > 0:
            near_valid &= self._strict_check(actions[:, :committed+1], context).valid
        # Future conflicts remain checked. A risky tail is admissible only when
        # at least one rate-limited braking continuation from the SAME committed
        # prefix passes the unchanged full swept/body/kinematic checks. Failure
        # is a model viability rejection, not a proof of physical inevitability.
        unrecoverable = torch.zeros_like(near_valid)
        risky = (near_valid & ~result.full_valid).nonzero().flatten()
        if len(risky) and not getattr(context, "brake_action_mapping_verified", False):
            unrecoverable[risky] = True
        elif len(risky):
            witnesses = self._braking_continuations(actions[risky].cpu(), context, committed)
            checked = self._strict_check(witnesses, context)
            viable = checked.valid.reshape(-1, len(risky)).any(0)
            result.braking_continuation_valid[risky] = viable
            unrecoverable[risky] = ~viable
            calibrated_viable = (checked.valid & self.calibrated_mask(witnesses, context)).reshape(-1, len(risky)).any(0)
            result.primary_calibrated[risky] &= calibrated_viable
        result.near_valid, result.unrecoverable = near_valid, unrecoverable
        result.valid = near_valid & ~unrecoverable
        # Only the uncommitted tail contributes soft risk. Immediate violations
        # are filtered, rather than charged repeatedly through soft penalties.
        tail = slice(near, None)
        zero = torch.zeros(len(actions))
        mean_tail = lambda tensor: tensor[:, tail].float().mean(1) if near < tensor.shape[1] else zero.clone()
        result.road_risk = mean_tail(result.interval_offroad)
        result.kinematic_risk = mean_tail(result.interval_kinematic)
        result.collision_risk = mean_tail(result.interval_collision)
        # Continuous swept intervals above remain authoritative. Soft boundary
        # distance only needs control-rate samples, not repeated GEOS queries
        # at every physics substep.
        corners = _corners(result.positions[:, near::self.geometry.substeps], result.headings[:, near::self.geometry.substeps],
                            context.length+2*self.geometry.model_margin, context.width+2*self.geometry.model_margin)
        _, margin = self.geometry._road_metrics(corners, context.road)
        result.boundary_margin = margin
        # Continuous signed BODY clearance is a ranking diagnostic, never a
        # substitute for the unchanged swept-interval certificate above.
        body_corners = _corners(result.positions[:, ::self.geometry.substeps], result.headings[:, ::self.geometry.substeps],
            context.length, context.width).numpy()
        bodies = shapely.polygons(body_corners)
        inside = shapely.covers(context.road, bodies)
        distances = shapely.distance(bodies, context.road.boundary)
        point_inside = shapely.covers(context.road, shapely.points(body_corners))
        penetration = np.where(point_inside, 0., shapely.distance(shapely.points(body_corners), context.road.boundary)).max(-1)
        signed_margin = torch.from_numpy(np.where(inside, distances, -np.maximum(penetration, 1e-6))).float()
        result.boundary_margin = signed_margin.amin(1)
        result.near_boundary_margin = signed_margin[:, :committed+1].amin(1)
        result.terminal_boundary_margin = signed_margin[:, -1]
        result.boundary_margin_change = signed_margin[:, -1]-signed_margin[:, 0]
        result.committed_margin_change = signed_margin[:, committed]-signed_margin[:, 0]
        boundary_risk = torch.exp(-margin.clamp_min(0)/context.width)
        # Bounding-circle separation is a conservative lower bound on clearance;
        # continuous contact/TTC still comes from swept_collisions above.
        result.clearance_risk = zero.clone()
        if context.obstacles:
            times = torch.arange(result.positions.shape[1])*self.geometry.dt/self.geometry.substeps
            xy = result.positions.new_tensor(np.asarray([o.position for o in context.obstacles]))
            velocity = result.positions.new_tensor(np.asarray([o.velocity for o in context.obstacles]))
            other = xy[None]+times[:, None, None]*velocity[None]
            distance = (result.positions[:, :, None]-other[None]).norm(dim=-1)
            radii = result.positions.new_tensor([math.hypot(o.length, o.width)/2 for o in context.obstacles])
            clearance = (distance-radii-math.hypot(context.length, context.width)/2).clamp_min(0).amin(2)
            result.min_clearance = clearance.amin(1)
            result.clearance_risk = (torch.exp(-clearance[:, near+1:]/context.width).mean(1)
                                    if near+1 < clearance.shape[1] else zero.clone())
        target = getattr(context, "target_speed", None)
        result.speed_risk = zero.clone()
        if target is not None and math.isfinite(target) and target >= 0:
            reachable = min(target, context.speed+float(self.cfg.get("max_acceleration", 4.))*horizon*self.geometry.dt)
            result.speed_risk = ((result.speeds[:, -1]-reachable).clamp_min(0)/max(reachable, 1.)).square()
        result.lateral_risk = zero.clone()
        reference = getattr(context, "lattice_positions", None)
        if reference is not None and len(reference) >= 2:
            # Both trajectories are in world XY. Closest segment distance,
            # rather than equal-time distance, avoids treating slower progress
            # as lateral error on the nominal Frenet path.
            reference = result.positions.new_tensor(np.asarray(reference))
            segment = reference[1:]-reference[:-1]
            delta = result.positions[:, near:, None]-reference[None, None, :-1]
            u = (delta*segment).sum(-1)/segment.square().sum(-1).clamp_min(1e-12)
            distance = (delta-u.clamp(0, 1)[..., None]*segment).square().sum(-1).amin(-1)
            scale = float(getattr(context, "lane_width", None) or context.width)
            result.lateral_risk = distance.mean(1)/(scale*scale)
        ttc_risk = torch.where(torch.isfinite(result.min_ttc), 1/(1+result.min_ttc/self.geometry.dt), 0.)
        result.risk_score = (result.road_risk+result.collision_risk+result.kinematic_risk
                             +boundary_risk+result.clearance_risk+ttc_risk+result.speed_risk+result.lateral_risk)
        direction = result.positions.new_tensor([math.cos(context.heading), math.sin(context.heading)])
        result.progress_estimate = (result.positions[:, -1]-result.positions[:, 0])@direction
        return result

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


def motion_response_errors(prediction, vehicle, info):
    """Compare both forecasts against ONE actual control interval, never an abort."""
    if not prediction or not bool(info.get("action_executed", True)):
        return {}
    actual_xy = np.asarray(vehicle.position)
    speed, heading = float(np.linalg.norm(vehicle.velocity)), float(vehicle.heading_theta)
    points = np.asarray(prediction.get("reference_points", []))
    def lateral(xy):
        if len(points) < 2:
            return float("nan")
        segments = points[1:]-points[:-1]
        length2 = np.einsum("ij,ij->i", segments, segments).clip(1e-12)
        delta = xy-points[:-1]
        u = np.clip((delta*segments).sum(-1)/length2, 0, 1)
        error = delta-u[:, None]*segments
        index = (error*error).sum(-1).argmin()
        normal = np.array([-segments[index, 1], segments[index, 0]])/np.sqrt(length2[index])
        return float(error[index]@normal)
    collision = any(bool(info.get(k, False)) for k in ("crash", "crash_vehicle", "crash_object", "crash_building", "crash_human", "crash_sidewalk"))
    offroad = bool(info.get("out_of_road", False))
    result = {}
    for name in ("center", "legacy"):
        model = prediction[name]
        angle = heading-model["heading"]
        prefix = f"motion_{name}_"
        result.update({prefix+"position_error_m": float(np.linalg.norm(actual_xy-model["position"])),
            prefix+"heading_error_rad": abs(math.atan2(math.sin(angle), math.cos(angle))),
            prefix+"lateral_error_m": abs(lateral(actual_xy)-lateral(np.asarray(model["position"]))),
            prefix+"speed_error_mps": abs(speed-model["speed"])})
        for event, actual in (("collision", collision), ("offroad", offroad)):
            predicted = model.get(event, float("nan"))
            known = isinstance(predicted, (bool, int, float)) and math.isfinite(float(predicted))
            result[prefix+event+"_false_positive"] = float(bool(predicted) and not actual) if known else float("nan")
            result[prefix+event+"_false_negative"] = float(not bool(predicted) and actual) if known else float("nan")
    return result
