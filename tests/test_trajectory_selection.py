"""Recovery generation, road progress and final certification regressions."""
from unittest.mock import patch

import numpy as np
import pytest
import torch
from shapely.geometry import box

from mpr_mpc.lattice.frenet_metadrive import PolylineReference
from mpr_mpc.planning.local_mppi import LocalMPPI
from mpr_mpc.planning.mppi_safety import projected_route_progress
from tests.test_mpr_mpc import mppi_cfg
from tests.test_mppi_v3 import RankedEvaluator, braking_context, safety_cfg
from mpr_mpc.planning.mppi_safety import MPPISafetyFilter


def make_planner(**selection):
    cfg = mppi_cfg(num_samples=4, num_elites=2, iterations=1, cost_enabled=True,
        safety=safety_cfg(two_layer=True, staged=True),
        trajectory_selection=dict(enabled=True, record_candidates=True, **selection),
        recovery=dict(enabled=True, progressive=True, lattice_guided=True, anti_stall=True,
            num_samples=4, iterations=1, stall_exit_s=.3))
    p = LocalMPPI(RankedEvaluator(), cfg, horizon=10, action_dim=2, device="cpu")
    original = p.evaluator.evaluate_mppi_candidates
    def evaluate(*args, **kwargs):
        result = original(*args, **kwargs)
        result.cost_ranking_validated = False
        return result
    p.evaluator.evaluate_mppi_candidates = evaluate
    return p


def road_context(**kwargs):
    c = braking_context(speed=kwargs.pop("speed", 0), **kwargs)
    c.road_reference = PolylineReference([[0., 10.], [100., 10.]])
    c.lane_centers = (0.,)
    c.target_speed, c.geometric_steer, c.road_tangent_steer = 5., -.4, 0.
    c.route_progress = 0.
    return c


def stuck_baseline(p):
    baseline = torch.ones(11, 2)
    # Reject the original abrupt steering but keep slew-limited proposals.
    p.candidate_safety.cfg["max_steering_rate"] = 1.
    p._candidate_batch = lambda *args: baseline[None].expand(4, -1, -1).clone()
    p._stall_active = True
    return baseline


def test_progressive_steering_and_positive_levels_use_at_most_16_candidates():
    p = make_planner(); c = road_context()
    c.previous_action = np.array([-.2, 0.])
    c.geometric_steer, c.road_tangent_steer = .8, .15
    actions, types = p._controlled_candidates(c, torch.zeros(11, 2), torch.zeros(11, 2))
    assert len(actions) == len(types) == 16
    geometry = actions[types.index("recovery_center_full"), :, 0]
    assert geometry[0] != geometry[-1] and geometry[0] != .8
    assert "recovery_hold" in types and "recovery_lattice_ramp" in types
    assert all(any(torch.isclose(a[:, 1], torch.tensor(level)).any() for a in actions) for level in [.1, .25, .4])
    previous = torch.tensor(c.previous_action).float()[None, None].expand(len(actions), -1, -1)
    increments = torch.cat((previous, actions), 1).diff(dim=1)
    assert increments[:, :, 1].abs().max() <= p.throttle_step+1e-6
    assert (actions.abs() <= 1).all() and not (actions[:, :, 1] <= -1).all(1).any()
    c.speed = c.target_speed
    capped, _ = p._controlled_candidates(c, None, torch.zeros(11, 2))
    assert (capped[:, :, 1] <= 0).all()


def test_route_progress_follows_curved_road_instead_of_vehicle_heading():
    c = road_context(); c.position = np.array([4., 0.]); c.heading = 0.
    c.road_reference = PolylineReference([[0., 0.], [5., 0.], [5., 20.]])
    positions = torch.tensor([[[4., 0.], [5., 4.]]])
    assert projected_route_progress(positions, c)[0] == pytest.approx(5.)
    assert positions[0, -1, 0]-positions[0, 0, 0] == 1.


def test_safe_forward_recovery_beats_high_wm_coasting_and_keeps_cost_gated():
    p = make_planner(); baseline = stuck_baseline(p); c = road_context()
    # A deliberate anti-progress critic: selecting forward must be a real
    # hierarchy decision, not merely the evaluator echoing progress.
    p.evaluator.value = lambda a: -100*a[:, 0, 1]
    result = p.plan(torch.zeros(1, 2), baseline, eval_mode=True, safety_context=c, lattice_actions=torch.zeros(11, 2))
    assert result.action[1] > 0
    assert result.metrics["predicted_route_progress"] >= p.min_progress
    assert result.metrics["selected_exact_verified"] == 1 and result.metrics["cost_penalty"] == 0
    selected = next(row for row in result.candidate_diagnostics if row["selected"])
    assert selected["exact_verified"] and selected["selected_rank"] == 1
    assert any(row["rejection_reason"] == "insufficient_progress" for row in result.candidate_diagnostics)


def test_unsafe_positive_controls_are_not_admitted_by_recovery():
    p = make_planner(); baseline = stuck_baseline(p)
    c = road_context(road=box(0, 5, 12.0001, 15))
    result = p.plan(torch.zeros(1, 2), baseline, eval_mode=True, safety_context=c, lattice_actions=torch.zeros(11, 2))
    assert result.action[1] <= 0
    positive = [r for r in result.candidate_diagnostics if r["candidate_type"].startswith("recovery_") and r["first_action"][1] > 0]
    assert positive and all(not r["near_safety_valid"] and not r["exact_verified"] for r in positive)


def test_continuity_adjustment_gets_a_new_exact_certificate():
    p = make_planner(max_throttle_step=.6); p.recovery_enabled = False
    base = torch.zeros(11, 2); candidate = base.clone(); candidate[:, 0] = .04; candidate[:, 1] = 1
    p._candidate_batch = lambda *args: torch.stack((base, candidate, base, candidate))
    p.evaluator.value = lambda a: a[:, 0, 1]*100
    c = road_context()
    with patch.object(p.candidate_safety, "check", wraps=p.candidate_safety.check) as exact:
        result = p.plan(torch.zeros(1, 2), base, eval_mode=True, safety_context=c)
    assert result.action[1] == pytest.approx(.6)
    assert result.metrics["continuity_changed"] == 1
    assert any(torch.isclose(call.args[0][0, 0, 1], torch.tensor(.6)) for call in exact.call_args_list)
    assert next(r for r in result.candidate_diagnostics if r["selected"])["candidate_type"] == "continuity_adjusted"


def test_healthy_sac_is_kept_when_no_meaningful_improvement_exists():
    p = make_planner(); p.min_improvement_abs = 100.
    base = torch.zeros(11, 2); base[:, 1] = .25
    other = base.clone(); other[:, 0] = .04
    p._candidate_batch = lambda *args: torch.stack((base, other, base, other))
    result = p.plan(torch.zeros(1, 2), base, eval_mode=True, safety_context=road_context(speed=2))
    assert result.baseline_selected and not result.metrics["recovery_search"]
    torch.testing.assert_close(result.selected_sequence, base)


def test_recovery_latch_uses_actual_progress_for_exit():
    p = make_planner(); base = stuck_baseline(p); c = road_context()
    c.measurement_step = 1
    p.plan(torch.zeros(1, 2), base, eval_mode=True, safety_context=c)
    assert p._recovery_latched
    safe = torch.zeros(11, 2); safe[:, 1] = .25
    p._candidate_batch = lambda *args: safe[None].expand(4, -1, -1).clone()
    c.measurement_step += 1
    p.plan(torch.zeros(1, 2), safe, eval_mode=True, safety_context=c)
    assert p._recovery_latched  # requesting forward is not actual recovery
    c.speed = 1.
    for i in range(4):
        c.position[0] += .1; c.route_progress += .01; c.measurement_step += 1
        p.plan(torch.zeros(1, 2), safe, eval_mode=True, safety_context=c)
    assert not p._recovery_latched


def test_center_prediction_uses_measured_rear_axle_without_changing_sac_geometry():
    f = MPPISafetyFilter(safety_cfg(center_bicycle=True), control_dt=.1)
    c = road_context(speed=5); c.rear_axle_to_center = c.wheelbase*.57
    actions = torch.zeros(1, 2, 2); actions[:, :, 0] = .3
    legacy = f.geometry.rollout(actions, c)
    center = f._rollout(actions, c)
    assert center[0][0, -1, 1] > legacy[0][0, -1, 1]
    # SAC still gets the original rear-axle-style legacy rollout, unchanged.
    torch.testing.assert_close(f.geometry.rollout(actions, c)[0], legacy[0])
    actions[:, :, 0] = 0
    torch.testing.assert_close(f._rollout(actions, c)[0], f.geometry.rollout(actions, c)[0])


def test_center_prediction_keeps_continuous_collision_rejection():
    from tests.test_v2_safety import obstacle
    f = MPPISafetyFilter(safety_cfg(center_bicycle=True, two_layer=True), control_dt=.1)
    c = road_context(speed=5, obstacles=(obstacle(11., 10.),))
    c.rear_axle_to_center = c.wheelbase*.57
    with patch.object(f.geometry, "_swept_check", wraps=f.geometry._swept_check) as swept:
        r = f.check(torch.zeros(1, 11, 2), c)
    assert swept.called and r.near_collision[0] and not r.valid[0]


def test_paired_cli_can_select_only_b_and_d_without_changing_default_suite():
    from evaluate_ablation import _select_variants, VARIANTS
    assert _select_variants({}) == VARIANTS
    assert [name for name, _ in _select_variants(dict(variants=["B_lattice_sac", "D_lattice_sac_mppi"]))] == ["B_lattice_sac", "D_lattice_sac_mppi"]
    with pytest.raises(ValueError):
        _select_variants(dict(variants=["unknown"]))


def test_guard_never_accepts_a_path_rejected_by_original_geometry():
    f = MPPISafetyFilter(safety_cfg(center_bicycle=True, center_model_guard=True, two_layer=True), control_dt=.1)
    c = road_context(speed=5); c.rear_axle_to_center = c.wheelbase*.57
    actions = torch.zeros(5, 11, 2)
    actions[:, :, 0] = torch.linspace(-.5, .5, 5)[:, None]
    result = f.check(actions, c)
    assert torch.equal(result.valid, result.center_model_valid & result.legacy_model_valid)
    assert not (result.valid & ~result.legacy_model_valid).any()


def test_recovery_progress_is_adequacy_not_always_maximum_speed():
    p = make_planner()
    progress = torch.tensor([0., .2, .7, 2.])
    eligible = p._progress_eligible(progress, torch.ones(4, dtype=torch.bool), recovery=True, recovery_target=.5)
    assert eligible.tolist() == [False, False, True, True]


def test_candidate_logging_switch_does_not_change_actions():
    outputs = []
    for enabled in [True, False]:
        p = make_planner(); p.record_candidates = enabled
        baseline = stuck_baseline(p)
        outputs.append(p.plan(torch.zeros(1, 2), baseline, eval_mode=True, safety_context=road_context()))
    torch.testing.assert_close(outputs[0].selected_sequence, outputs[1].selected_sequence)
    assert outputs[0].candidate_diagnostics and not outputs[1].candidate_diagnostics


def test_future_acceleration_is_not_counted_as_an_immediate_positive_action():
    p = make_planner(); baseline = stuck_baseline(p); c = road_context()
    c.previous_action = np.array([0., -1.])
    result = p.plan(torch.zeros(1, 2), baseline, eval_mode=True, safety_context=c)
    assert result.metrics["positive_throttle_candidate_count"] == 0
    assert result.metrics["positive_sequence_candidate_count"] > 0
    assert not result.metrics["can_accelerate"]


def test_step_logger_keeps_scalar_candidate_archive_separate_from_metrics(tmp_path):
    import gzip, json
    from types import SimpleNamespace
    from mpr_mpc.evaluation_diagnostics import StepDiagnosticsLogger
    candidate = dict(candidate_id="c0000", first_action=[0., .1], final_score=None, exact_verified=True)
    result = SimpleNamespace(metrics={}, planning_mode="recovery_controlled", recovery_reason="low_speed_stall",
        stage=SimpleNamespace(use_residual=True, use_mppi=True), debug=dict(mppi_candidates=[candidate]))
    logger = StepDiagnosticsLogger(tmp_path, "evidence", compact=True)
    logger.log(559, 1, [0., .1], result)
    logger.close()
    row = json.loads((tmp_path/'evidence.jsonl').read_text())
    assert row['candidate_records_count'] == 1 and 'candidates' not in row
    with gzip.open(tmp_path/row['candidate_records_file'], 'rt') as stream:
        assert json.loads(stream.readline())['candidates'][0] == candidate
