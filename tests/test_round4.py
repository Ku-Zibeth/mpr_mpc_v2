from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import shapely
from shapely.geometry import Polygon, box
import torch

from mpr_mpc.planning.cost_validation import CostModelValidator
from mpr_mpc.planning.local_mppi import LocalMPPI
from mpr_mpc.planning.safety import TrajectorySafetyEvaluator, _corners
from mpr_mpc.planning.swept_collision import swept_collisions
from mpr_mpc.residual_rl.state_builder import ResidualStateBuilder
from tests.test_mpr_mpc import mppi_cfg, RecordingEvaluator, make_path
from tests.test_v2_safety import context, obstacle


















@pytest.mark.parametrize("dt,substeps", [(1., 1), (.5, 1), (.1, 1), (.1, 5)])
@pytest.mark.parametrize("kind", ["crossing", "head_on", "high_relative_speed", "static"])
def test_swept_contact_between_safe_endpoint_footprints(dt, substeps, kind):
    # Tiny cars let contact fit strictly between old discrete poses.
    ctx = context(speed=20, road=box(-100, -100, 100, 100))
    ctx.length, ctx.width = .2, .2
    if kind == "crossing":
        obj = obstacle(11, 11, 0, -20, np.pi/2)
    elif kind == "head_on":
        obj = obstacle(12, 10, -20, 0)
    elif kind == "high_relative_speed":
        obj = obstacle(12, 10, -60, 0)
    else:
        obj = obstacle(11, 10)
    obj.length = obj.width = .2
    ctx.obstacles = (obj,)
    safety = TrajectorySafetyEvaluator({}, control_dt=dt, device="cpu")
    safety.substeps = substeps
    result = safety.check(torch.zeros(1, int(1/dt)+1, 2), ctx)
    assert result.collision.item() and not result.valid.item()
    assert 0 < result.min_ttc.item() < .1
    if dt == 1:
        assert result.interval_collision.any()


@pytest.mark.parametrize("dyaw", [np.pi/2, -np.pi/2])
def test_rotation_sweep_hits_static_obstacle_without_endpoint_overlap(dyaw):
    # A long narrow rectangle rotates around its center. Neither endpoint
    # occupies (1.4, +/-1.4), but the intermediate 45-degree footprint does.
    positions = torch.tensor([[[0., 0.], [0., 0.]]])
    headings = torch.tensor([[0., dyaw]])
    sign = 1 if dyaw > 0 else -1
    args = (torch.tensor([[1.4, sign*1.4]]), torch.zeros(1, 2), torch.zeros(1), torch.tensor([.15]), torch.tensor([.15]))
    contact = swept_collisions(positions, headings, *args, ego_length=4., ego_width=.2, dt=1.)
    assert torch.isfinite(contact).all()
    assert 0 < contact.item() < 1


@pytest.mark.parametrize("yaw", [0., .3, 1.57])
def test_near_miss_remains_safe_after_rotating_sweep(yaw):
    positions = torch.tensor([[[0., 0.], [5., 0.]]])
    headings = torch.tensor([[0., yaw]])
    # Beyond the ego circumradius plus obstacle radius at every instant.
    contact = swept_collisions(positions, headings, torch.tensor([[2.5, 3.1]]), torch.zeros(1, 2),
        torch.zeros(1), torch.tensor([.2]), torch.tensor([.2]), ego_length=4., ego_width=2., dt=1.)
    assert torch.isinf(contact).all()


def test_adaptive_rotation_rejects_bounding_box_false_positive():
    # The inflated midpoint OBB overlaps, but adaptive refinement proves a gap.
    positions = torch.zeros(1, 2, 2)
    headings = torch.tensor([[0., np.pi/2]])
    contact = swept_collisions(positions, headings, torch.tensor([[2., 2.]]), torch.zeros(1, 2),
        torch.zeros(1), torch.tensor([.1]), torch.tensor([.1]), ego_length=4., ego_width=.2, dt=1.)
    assert torch.isinf(contact).all()


def test_curved_center_sweep_catches_contact_missed_by_the_endpoint_chord():
    positions = torch.tensor([[[0., 0.], [2., 2.]]])
    headings = torch.tensor([[0., np.pi/2]])
    args = (torch.tensor([[np.sqrt(2), 2-np.sqrt(2)]]), torch.zeros(1, 2), torch.zeros(1),
            torch.tensor([.1]), torch.tensor([.1]))
    linear = swept_collisions(positions, headings, *args, ego_length=.1, ego_width=.1, dt=1.)
    curved = swept_collisions(positions, headings, *args, ego_length=.1, ego_width=.1, dt=1., curved=True)
    assert torch.isinf(linear).all()
    assert torch.isfinite(curved).all() and 0 < curved.item() < 1


def test_accelerating_sweep_uses_time_not_linear_chord_progress():
    positions = torch.tensor([[[0., 0.], [10., 0.]]])
    headings = torch.zeros(1, 2)
    speeds = torch.tensor([[0., 20.]])
    static = swept_collisions(positions, headings, torch.tensor([[2.5, 0.]]), torch.zeros(1, 2), torch.zeros(1),
        torch.tensor([.1]), torch.tensor([.1]), ego_length=.1, ego_width=.1, dt=1., curved=True, speeds=speeds)
    assert static.item() == pytest.approx(np.sqrt(.24), abs=2e-4)
    crossing = swept_collisions(positions, headings, torch.tensor([[5., -5.]]), torch.tensor([[0., 10.]]),
        torch.tensor([np.pi/2]), torch.tensor([.1]), torch.tensor([.1]), ego_length=.1, ego_width=.1,
        dt=1., curved=True, speeds=speeds)
    assert torch.isinf(crossing).all()


@pytest.mark.parametrize("curved,accelerating", [(False, False), (True, False), (True, True)])
def test_swept_contact_matches_dense_independent_shapely_oracle(curved, accelerating):
    rng = np.random.default_rng(71)
    count = 40
    starts, ends = rng.uniform(-3, 3, (count, 2)), rng.uniform(-3, 3, (count, 2))
    yaw0, yaw1 = rng.uniform(-2, 2, count), rng.uniform(-2, 2, count)
    positions = torch.tensor(np.stack((starts, ends), 1), dtype=torch.float64)
    headings = torch.tensor(np.stack((yaw0, yaw1), 1), dtype=torch.float64)
    velocity = torch.tensor(rng.uniform(-2, 2, (1, 2)), dtype=torch.float64)
    speeds = torch.tensor(rng.uniform(0, 20, (count, 2)), dtype=torch.float64) if accelerating else None
    predicted = torch.isfinite(swept_collisions(positions, headings, torch.zeros(1, 2, dtype=torch.float64),
        velocity, torch.zeros(1, dtype=torch.float64), torch.ones(1, dtype=torch.float64),
        torch.ones(1, dtype=torch.float64), ego_length=2., ego_width=1., dt=1., curved=curved, speeds=speeds)).flatten().numpy()
    reference = []
    target = box(-.5, -.5, .5, .5)
    for i, (start, end, a, b) in enumerate(zip(starts, ends, yaw0, yaw1)):
        u = torch.linspace(0, 1, 2049, dtype=torch.float64)
        progress = (u*(2*speeds[i, 0]+(speeds[i, 1]-speeds[i, 0])*u)/speeds[i].sum()) if accelerating else u
        if curved and 1e-6 < abs(b-a) < np.pi:
            delta = end-start
            circle = (start+end)/2 + np.array([-delta[1], delta[0]])/(2*np.tan((b-a)/2))
            radial = start-circle
            angle = progress*(b-a)
            centers = torch.as_tensor(circle)[None] + torch.stack((radial[0]*angle.cos()-radial[1]*angle.sin(),
                radial[0]*angle.sin()+radial[1]*angle.cos()), -1)
        else:
            centers = torch.as_tensor(start)[None] + progress[:, None]*torch.as_tensor(end-start)[None]
        centers = centers - u[:, None]*velocity
        corners = _corners(centers, a+progress*(b-a), 2., 1.).numpy()
        reference.append(bool(shapely.intersects(shapely.polygons(corners), target).any()))
    # Dense sampling is a regression oracle, not the deployed safety method.
    assert not (np.asarray(reference) & ~predicted).any()
    assert (predicted != reference).sum() <= 1


def test_optimized_road_metrics_equal_original_geometry_formula():
    rng = np.random.default_rng(16)
    road = box(-8, -5, 8, 5).difference(box(-2, -1, 2, 1))
    centers = torch.tensor(rng.uniform(-10, 10, (8, 17, 2)), dtype=torch.float64)
    headings = torch.tensor(rng.uniform(-3, 3, (8, 17)), dtype=torch.float64)
    ego = _corners(centers, headings, 2., 1.)
    corners = ego.numpy().reshape(-1, 4, 2)
    footprints = shapely.polygons(corners)
    valid = shapely.covers(road.buffer(-.15), footprints).reshape(8, 17)
    inside = shapely.covers(road, footprints)
    margin = np.where(inside, shapely.distance(road.boundary, footprints),
        -np.maximum(shapely.distance(shapely.points(corners), road).max(-1), 1e-6)).reshape(8, 17).min(1)
    actual_valid, actual_margin = TrajectorySafetyEvaluator({}, control_dt=.1, device="cpu")._road_metrics(ego, road)
    np.testing.assert_array_equal(actual_valid, valid)
    np.testing.assert_allclose(actual_margin, margin, atol=1e-12)








def test_safety_context_rejects_reverse_action_mapping():
    safety = TrajectorySafetyEvaluator({}, control_dt=.1, device="cpu")
    with pytest.raises(ValueError, match="enable_reverse=false"):
        safety.context(None, SimpleNamespace(enable_reverse=True), width=2, wheelbase=2.5, max_steering=.6)


def test_cost_feature_availability_is_reported_without_unmasking():
    builder = ResidualStateBuilder({"risk_features": ["cumulative_cost", "min_ttc", "min_clearance", "boundary_margin"]},
        latent_dim=512, device="cpu")
    value = SimpleNamespace(total_value=torch.ones(1), cumulative_cost=torch.tensor([.4]), cost_validated=False,
        cost_unavailable_reason="calibration_not_ready", risk_summary=dict(min_ttc=float("inf"), min_clearance=3., boundary_margin=.5))
    state = builder.build(latent=torch.zeros(512), coarse_path=make_path(), consequence=value, feasible_ratio=1, used_fallback=False)
    assert state.shape == (535,)
    assert state[527:529].tolist() == [0, 0]
    assert builder.last_risk_status["cumulative_cost"] == dict(valid=False, reason="calibration_not_ready")
    assert all(builder.last_risk_status[k]["valid"] for k in ["min_ttc", "min_clearance", "boundary_margin"])
    value.cost_validated = True
    state = builder.build(latent=torch.zeros(512), coarse_path=make_path(), consequence=value, feasible_ratio=1, used_fallback=False)
    assert state[527].item() == pytest.approx(.04)
    assert state[528] == 1






def test_calibrated_cost_profile_preserves_535_layout_but_is_explicitly_separate():
    from hydra import compose, initialize_config_dir
    from mpr_mpc._bootstrap import PACKAGE_ROOT
    for profile in ["E5", "E5_calibrated_cost"]:
        with initialize_config_dir(version_base="1.3", config_dir=str(PACKAGE_ROOT)):
            cfg = compose(config_name="config", overrides=[f"+experiment={profile}"])
        builder = ResidualStateBuilder(dict(cfg.residual_rl), latent_dim=512, device="cpu")
        assert builder.state_dim == 535
        assert cfg.steps == 50000 and cfg.residual_rl.freeze_tdmpc2_on_start
        assert cfg.residual_rl.coarse_cost_feature == (profile == "E5_calibrated_cost")


def test_checkpoint_cannot_silently_change_cost_feature_semantics():
    from tests.test_residual_rl import _checkpoint_agent
    agent = _checkpoint_agent()
    state = {"metadata": agent._checkpoint_metadata()}
    agent.residual_cfg["coarse_cost_feature"] = True
    with pytest.raises(ValueError, match="cost-feature semantics"):
        agent._check_checkpoint_compatibility(state, strict_planner_config=False)


def test_kernel_trace_counts_real_kernels_not_gpu_annotation_spans():
    from tools.planning_profiler import kernel_trace_summary
    events = [dict(cat="user_annotation", name="total", ts=0, dur=100, pid=1, tid=1),
        dict(cat="user_annotation", name="safety", ts=10, dur=20, pid=1, tid=1),
        dict(cat="gpu_user_annotation", name="safety", ts=10, dur=9000, pid=0, tid=7),
        dict(cat="cpu_op", ts=12, pid=1, tid=1, args={"External id": 42}),
        dict(cat="kernel", ts=200, dur=3, args={"External id": 42}),
        dict(cat="cuda_runtime", ts=60, pid=1, tid=1, args={"correlation": 9}),
        dict(cat="kernel", ts=300, dur=4, args={"correlation": 9}),
        dict(cat="kernel", ts=400, dur=5, args={"External id": 404})]
    summary, audit = kernel_trace_summary(dict(traceEvents=events), ["safety", "total", "unused"])
    assert summary["safety"] == dict(cpu_range_ms=.02, cuda_kernel_ms=.003)
    assert summary["total"]["cuda_kernel_ms"] == .007
    assert summary["unused"]["cuda_kernel_ms"] == 0
    assert audit == dict(all_cuda_kernel_ms=.012, unattributed_kernels=1)
