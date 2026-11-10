"""Small regressions for committed-prefix safety and guided recovery."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from shapely.geometry import box

from mpr_mpc.evaluation_diagnostics import EpisodeDiagnostics, StepDiagnosticsLogger
from mpr_mpc.planning.mppi_safety import MPPISafetyFilter
from tests.test_mppi_v3 import planner, braking_context, fake_filter, fixed_batch, safety_cfg, RankedEvaluator
from tests.test_mppi_v31 import RECOVERY
from tests.test_v2_safety import obstacle


@pytest.mark.parametrize("road_end,near_safe", [(14.7, True), (12.3, False)])
def test_full_road_check_is_retained_but_recoverable_tail_is_soft(road_end, near_safe):
    f = MPPISafetyFilter(safety_cfg(two_layer=True), control_dt=.1)
    ctx = braking_context(speed=5, road=box(0, 5, road_end, 15))
    actions = torch.zeros(1, 11, 2)
    r = f.check(actions, ctx)
    assert not r.full_valid[0] and r.offroad[0]
    assert bool(r.near_valid[0]) == near_safe
    assert bool(r.valid[0]) == near_safe
    assert r.risk_score[0] > 0
    if near_safe:
        ctx.lattice_positions = np.array([[10., 10.], [15., 10.]])
        ctx.lane_width = 3.5
        assert f.check(actions, ctx).lateral_risk[0] == 0
        ctx.lattice_positions[:, 1] += 1
        assert f.check(actions, ctx).lateral_risk[0] > 0
        ctx.brake_action_mapping_verified = False
        assert not f.check(actions, ctx).valid[0]


def test_committed_window_is_derived_from_control_period_and_actuator_delay():
    ctx = braking_context(speed=0, road=box(0, 5, 12.1, 15))
    actions = torch.zeros(1, 11, 2); actions[:, :, 1] = 1
    immediate = MPPISafetyFilter(safety_cfg(two_layer=True), control_dt=.1)
    delayed = MPPISafetyFilter(safety_cfg(two_layer=True, action_delay_s=.2), control_dt=.1)
    assert immediate.committed_controls(10) == 1
    assert delayed.committed_controls(10) == 3
    assert immediate.check(actions, ctx).valid[0]
    assert not delayed.check(actions, ctx).near_valid[0]


def test_future_collision_without_a_checked_braking_continuation_remains_hard():
    f = MPPISafetyFilter(safety_cfg(two_layer=True), control_dt=.1)
    ctx = braking_context(speed=5, obstacles=(obstacle(15, 10),))
    r = f.check(torch.zeros(1, 11, 2), ctx)
    assert r.near_valid[0] and r.collision[0]
    assert r.unrecoverable[0] and not r.valid[0]


def test_guided_recovery_contains_safe_positive_actions_only_below_target_speed():
    p = planner(recovery={**RECOVERY, "lattice_guided": True})
    baseline = torch.ones(4, 2)
    lattice = baseline.clone(); lattice[:, 0] = -.2
    fixed_batch(p, baseline, baseline)
    checks = fake_filter(p, lambda a: (a[:, 0, 1] == .1) & (a[:, 0, 0] < 0))
    ctx = braking_context(speed=0); ctx.target_speed = 5.
    ctx.previous_action = np.array([.4, 0.])
    r = p.plan(torch.zeros(1, 2), baseline, eval_mode=True, safety_context=ctx, lattice_actions=lattice)
    assert r.planning_mode == "recovery_controlled"
    assert r.action[0] < 0 and r.action[1] == pytest.approx(.1)
    assert not r.metrics["emergency_fallback"]
    # Once the target is reached, positive recovery candidates disappear.
    ctx.target_speed = 0.
    checks.clear()
    r = p.plan(torch.zeros(1, 2), baseline, eval_mode=True, safety_context=ctx, lattice_actions=lattice)
    assert r.action[1] <= 0
    assert not any(bool((a[:, 0, 1] == .1).any()) for a in checks)


def test_missing_wm_scores_do_not_turn_safe_candidates_into_emergency_braking():
    p = planner()
    baseline, candidate = torch.zeros(4, 2), torch.zeros(4, 2); candidate[:, 0] = .04
    fixed_batch(p, baseline, candidate)
    fake_filter(p, lambda a: a[:, 0, 0] > 0)
    p.evaluator = RankedEvaluator(lambda a: torch.full((len(a),), float("nan")))
    r = p.plan(torch.zeros(1, 2), baseline, eval_mode=True, safety_context=braking_context(speed=0))
    torch.testing.assert_close(r.action, candidate[0])
    assert r.metrics["scoring_unavailable_fallback"] == 1 and not r.metrics["emergency_fallback"]


def test_checked_coasting_is_not_overridden_by_a_higher_wm_braking_score():
    p = planner(recovery={**RECOVERY, "lattice_guided": True})
    baseline = torch.ones(4, 2); lattice = baseline.clone(); lattice[:, 0] = .2
    fixed_batch(p, baseline, baseline)
    fake_filter(p, lambda a: a[:, 0, 1] <= 0)
    p.evaluator = RankedEvaluator(lambda a: -10*a[:, 0, 1])
    ctx = braking_context(speed=0); ctx.target_speed = 5.
    r = p.plan(torch.zeros(1, 2), baseline, eval_mode=True, safety_context=ctx, lattice_actions=lattice)
    assert r.planning_mode == "recovery_controlled" and r.action[1] == 0
    assert r.metrics["emergency_fallback"] == r.metrics["emergency_reason"] == 0
    p.evaluator = RankedEvaluator(lambda a: torch.where(a[:, 0, 1] < 0, 10., float("nan")))
    r = p.plan(torch.zeros(1, 2), baseline, eval_mode=True, safety_context=ctx, lattice_actions=lattice)
    assert r.action[1] == 0 and not r.metrics["emergency_fallback"]
    assert r.metrics["scoring_unavailable_fallback"] == 1


def test_emergency_counters_clear_as_soon_as_a_safe_candidate_returns():
    p = planner(); baseline = torch.zeros(4, 2)
    fixed_batch(p, baseline, baseline)
    fake_filter(p, lambda a: torch.zeros(len(a), dtype=torch.bool))
    for count in [1, 2]:
        r = p.plan(torch.zeros(1, 2), baseline, eval_mode=True, safety_context=braking_context())
        assert r.metrics["emergency_streak"] == r.metrics["braking_streak"] == count
    fake_filter(p, lambda a: torch.ones(len(a), dtype=torch.bool))
    r = p.plan(torch.zeros(1, 2), baseline, eval_mode=True, safety_context=braking_context(speed=0))
    assert r.metrics["emergency_streak"] == r.metrics["braking_streak"] == 0
    p.reset(); assert p._mode_switches == p._emergency_streak == 0


def test_compact_step_evidence_keeps_switches_and_accidents_and_mode_outcomes(tmp_path):
    plan = SimpleNamespace(stage=SimpleNamespace(use_mppi=True, use_residual=True),
        planning_mode="normal", recovery_reason="none", metrics={"mppi/planning_mode": 1.})
    logger = StepDiagnosticsLogger(tmp_path, "compact", compact=True, stride=10)
    logger.log(1714, 1, np.zeros(2), plan)
    logger.log(1714, 2, np.zeros(2), plan)
    plan.planning_mode = "recovery_controlled"; plan.metrics["mppi/planning_mode"] = 3.
    logger.log(1714, 3, np.zeros(2), plan)
    logger.log(1714, 4, np.zeros(2), plan, info={"crash_vehicle": True})
    logger.close()
    rows = [json.loads(s) for s in (tmp_path/"compact.jsonl").read_text().splitlines()]
    assert [r["episode_step"] for r in rows] == [1, 3, 4]
    d = EpisodeDiagnostics(); d.observe(np.zeros(3), np.zeros(2), {"crash_vehicle": True}, plan)
    assert d.summary({})["mppi_mode_recovery_controlled_collision_steps"] == 1
