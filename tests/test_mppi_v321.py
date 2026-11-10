"""Targeted staged-certification, recovery and physical-response regressions."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from shapely.geometry import box

from mpr_mpc.planning.mppi_safety import MPPISafetyFilter, geometric_recovery_reference
from mpr_mpc.lattice.frenet_metadrive import PolylineReference
from tests.test_mppi_v3 import planner, braking_context, safety_cfg, fixed_batch, fake_filter
from tests.test_mppi_v31 import RECOVERY
from tests.test_v2_safety import obstacle


def staged(**kwargs):
    return MPPISafetyFilter(safety_cfg(two_layer=True, staged=True, **kwargs), control_dt=.1)


def test_preview_only_checks_exact_prefix_and_cannot_certify_unsafe_tail():
    f = staged()
    ctx = braking_context(speed=5, obstacles=(obstacle(15, 10),))
    actions = torch.zeros(2, 11, 2)
    with patch.object(f, "_strict_check", wraps=f._strict_check) as strict:
        p = f.preview(actions, ctx)
        assert p.near_valid.all() and p.valid.all()  # provisional ONLY
        assert [c.args[0].shape[1] for c in strict.call_args_list] == [2]
    exact = f.check(actions, ctx)
    assert exact.unrecoverable.all() and not exact.valid.any()


def test_near_collision_rejected_before_tail_rollout_even_during_recovery():
    f = staged()
    ctx = braking_context(speed=5, obstacles=(obstacle(11, 10),))
    actions = torch.zeros(2, 11, 2); actions[:, :, 1] = .1
    with patch.object(f.geometry, "rollout", wraps=f.geometry.rollout) as rollout:
        r = f.preview(actions, ctx)
        assert not r.valid.any() and r.near_collision.all()
        assert len(rollout.call_args_list) == 1  # no tail work for rejected batch


@pytest.mark.parametrize("mapping,expected", [(True, True), (False, False)])
def test_same_prefix_braking_witness_is_required_for_future_road_risk(mapping, expected):
    f = staged()
    ctx = braking_context(speed=5, road=box(0, 5, 14.7, 15))
    ctx.brake_action_mapping_verified = mapping
    a = torch.zeros(1, 11, 2)
    assert f.preview(a, ctx).valid[0]
    r = f.check(a, ctx)
    assert not r.full_valid[0]
    assert bool(r.valid[0]) == expected


def test_final_selection_promotes_next_candidate_after_exact_failure():
    p = planner()
    p.candidate_safety.two_layer = p.candidate_safety.staged = True
    base = torch.zeros(4, 2); high = base.clone(); high[:, 0] = .04
    second = base.clone(); second[:, 0] = .02
    p._candidate_batch = lambda *args: torch.stack((base, high, second, high))
    def result(valid):
        return SimpleNamespace(valid=valid, near_valid=valid.clone(), collision=~valid,
            offroad=torch.zeros_like(valid), kinematic_violation=torch.zeros_like(valid))
    p.candidate_safety.preview = lambda actions, ctx: result(torch.ones(len(actions), dtype=torch.bool))
    certified = []
    def exact(actions, ctx):
        certified.extend(actions.clone())
        return result((actions[:, 0, 0] > 0) & (actions[:, 0, 0] < .03))
    p.candidate_safety.check = exact
    p.recovery_enabled = False
    r = p.plan(torch.zeros(1, 2), base, eval_mode=True, safety_context=braking_context(speed=0))
    torch.testing.assert_close(r.selected_sequence, second)
    assert r.metrics["selected_exact_verified"] == 1 and r.metrics["exact_failed_candidates"] >= 1
    assert any(torch.equal(a, second) for a in certified)


def test_stall_latches_then_exits_only_after_sustained_measured_progress():
    p = planner(recovery={**RECOVERY, "anti_stall": True, "stall_duration_s": .3, "stall_exit_s": .2})
    ctx = braking_context(speed=0)
    for _ in range(3): p._observe_motion(ctx)
    assert p._stall_active and p._stall_count == 1
    ctx.speed = 1.; ctx.position[0] += .1
    assert not p._observe_motion(ctx)[2] and p._stall_active
    ctx.speed = 0
    p._observe_motion(ctx)
    assert p._stall_active  # intermittent motion cannot clear the latch
    ctx.speed = 1.
    ctx.position[0] += .1; p._observe_motion(ctx)
    ctx.position[0] += .1
    assert p._observe_motion(ctx)[2] and not p._stall_active
    p.reset(); assert p._stall_count == 0 and not p._motion
    ctx.measurement_step = 1; ctx.speed = 0
    for _ in range(10): p._observe_motion(ctx)
    assert len(p._motion) == 1 and not p._stall_active


def test_geometric_direction_differs_from_lattice_without_extra_candidate_budget():
    ctx = braking_context(speed=0)
    ctx.lattice_actions = torch.zeros(4, 2)
    ref = PolylineReference([[0., 9.], [30., 9.]])
    geometric_recovery_reference(ctx, ref, [0.], .1)
    assert ctx.geometric_steer < 0 and ctx.road_tangent_steer == pytest.approx(0.)
    p = planner(recovery={**RECOVERY, "lattice_guided": True, "anti_stall": True})
    base = torch.ones(4, 2); fixed_batch(p, base, base)
    calls = fake_filter(p, lambda a: (a[:, 0, 0] < 0) & (a[:, 0, 1] > 0))
    ctx.target_speed = 5.
    r = p.plan(torch.zeros(1, 2), base, eval_mode=True, safety_context=ctx, lattice_actions=ctx.lattice_actions)
    assert r.action[0] < 0 and r.action[1] > 0
    assert max(len(a) for a in calls) <= 9
    # Same measured state / same deterministic recovery feasibility must not
    # alternate between emergency and recovery merely because RNG advanced.
    for _ in range(3):
        repeated = p.plan(torch.zeros(1, 2), base, eval_mode=True, safety_context=ctx, lattice_actions=ctx.lattice_actions)
        assert repeated.planning_mode == r.planning_mode
        torch.testing.assert_close(repeated.action, r.action)
    # The same positive candidate is rejected if its safety predicate changes.
    fake_filter(p, lambda a: a[:, 0, 1] <= 0)
    r = p.plan(torch.zeros(1, 2), base, eval_mode=True, safety_context=ctx, lattice_actions=ctx.lattice_actions)
    assert r.action[1] <= 0


def test_underspeed_is_efficiency_not_safety_risk():
    f = staged(); ctx = braking_context(speed=0); ctx.target_speed = 5.
    a = torch.zeros(1, 11, 2)
    assert f.check(a, ctx).speed_risk[0] == 0
    ctx.speed = 10.
    assert f.check(a, ctx).speed_risk[0] > 0


def test_response_errors_require_one_real_step_and_wrap_heading():
    f = staged(); ctx = braking_context(speed=2)
    vehicle = SimpleNamespace(engine=SimpleNamespace(episode_step=5), position=ctx.position.copy(),
        heading_theta=ctx.heading, velocity=np.array([2., 0.]), current_action=np.array([0., -1.]))
    a = torch.zeros(11, 2); a[:, 1] = -1.
    f.predict_response(a, ctx, vehicle)
    assert not f.response_errors(vehicle)
    vehicle.engine.episode_step += 1
    r = f.response_errors(vehicle)
    assert r["physics_speed_error_mps"] > 0 and r["execution_action_error"] == 0
    assert np.isfinite(r["physics_braking_response_bias_mps2"])
    vehicle.engine.episode_step += 1
    assert not f.response_errors(vehicle)


def test_preview_rollout_is_reused_only_for_the_same_measured_state():
    f = staged(); ctx = braking_context(speed=2)
    a = torch.zeros(1, 11, 2)
    with patch.object(f.geometry, "rollout", wraps=f.geometry.rollout) as rollout:
        f.preview(a, ctx)
        calls = rollout.call_count
        f.check(a, ctx)
        assert rollout.call_count == calls
        ctx.speed = 3.
        f.check(a, ctx)
        assert rollout.call_count > calls
