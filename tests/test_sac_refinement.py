"""Six regressions for SAC-centred selection, not another planner test suite."""
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from tests.test_mppi_v3 import planner, braking_context, fake_filter
from tests.test_v2_safety import obstacle


def setup(value=lambda a: torch.zeros(len(a))):
    p = planner(); p.min_improvement_abs = .05
    p.evaluator.value = value
    p.candidate_safety.staged = True
    base = torch.zeros(4, 2)
    def flags(a, ctx):
        valid = torch.ones(len(a), dtype=torch.bool)
        return SimpleNamespace(valid=valid, near_valid=valid.clone(), collision=~valid,
            offroad=~valid, kinematic_violation=~valid)
    p.candidate_safety.check = p.candidate_safety.preview = flags
    p.trajectory_selection = True
    ctx = braking_context(speed=0)
    ctx.road_reference = object()  # filtering is mocked; only availability is used
    return p, base, ctx


def test_A_exact_baseline_retained_without_meaningful_gain():
    p, base, ctx = setup()
    r = p.plan(torch.zeros(1, 2), base, eval_mode=True, safety_context=ctx)
    assert torch.equal(r.selected_sequence, base) and r.baseline_selected
    assert r.metrics['normal_search_delta'] == 0 and r.planning_mode == 'baseline'


def test_B_certified_improvement_within_sac_trust_region():
    p, base, ctx = setup(lambda a: 100*a[:, 0, 0])
    candidate = base.clone(); candidate[:, 0] = .02
    p._candidate_batch = lambda *a: torch.stack((base, candidate, candidate, base))
    r = p.plan(torch.zeros(1, 2), base, eval_mode=True, safety_context=ctx)
    assert torch.equal(r.selected_sequence, candidate) and not r.baseline_selected
    assert ((r.selected_sequence-base).abs() <= p.max_delta).all()
    assert r.planner_score_gain > .05 and r.metrics['selected_exact_verified'] == 1
    original = p.candidate_safety.check
    def regressing_progress(a, ctx):
        flags = original(a, ctx)
        flags.route_progress_estimate = torch.where(a[:, 0, 0] > 0, .2, 1.)
        return flags
    p.candidate_safety.check = regressing_progress
    p.reset()
    rejected = p.plan(torch.zeros(1, 2), base, eval_mode=True, safety_context=ctx)
    assert rejected.baseline_selected and torch.equal(rejected.selected_sequence, base)
    def unsafe_reference(a, ctx):
        flags = regressing_progress(a, ctx)
        flags.valid = flags.near_valid = a[:, 0, 0] > 0
        return flags
    p.candidate_safety.check = unsafe_reference
    p.reset()
    intervention = p.plan(torch.zeros(1, 2), base, eval_mode=True, safety_context=ctx)
    assert not intervention.baseline_selected and torch.equal(intervention.selected_sequence, candidate)
    assert intervention.metrics['safety_modification'] == 1


def test_C_search_failure_or_unknown_projection_never_triggers_recovery():
    p, base, ctx = setup(lambda a: torch.full((len(a),), float('nan')))
    p.recovery_enabled = True
    original = p.candidate_safety.check
    def unknown_progress(a, c):
        flags = original(a, c)
        flags.route_progress_estimate = torch.full((len(a),), float('nan'))
        return flags
    p.candidate_safety.check = unknown_progress
    with patch.object(p, '_controlled_candidates', side_effect=AssertionError('spurious recovery')):
        r = p.plan(torch.zeros(1, 2), base, eval_mode=True, safety_context=ctx)
    assert r.baseline_selected and torch.equal(r.selected_sequence, base)
    assert r.metrics['decision_state'] == r.metrics['recovery_search'] == r.metrics['emergency_fallback'] == 0
    # Missing WM scores do not justify a large Lattice takeover when a
    # bounded exact-checked local correction is already available.
    local = base.clone(); local[:, 0] = .02
    lattice = base.clone(); lattice[:, 0] = .9
    p._candidate_batch = lambda *a: torch.stack((base,local,local,base))
    def safe_local(a, c):
        flags = original(a,c)
        flags.valid = flags.near_valid = a[:, 0, 0] > 0
        return flags
    p.candidate_safety.check = safe_local
    p.recovery_enabled = False; p.reset()
    fallback = p.plan(torch.zeros(1, 2),base,eval_mode=True,safety_context=ctx,lattice_actions=lattice)
    assert torch.equal(fallback.selected_sequence,local) and not fallback.metrics['lattice_fallback']
    p.evaluator.value = lambda a: 100*a[:, 0, 0]
    p._candidate_batch = lambda *a: base[None].expand(4, -1, -1).clone()
    p.recovery_enabled = True; p.reset()
    controlled = p.plan(torch.zeros(1, 2),base,eval_mode=True,safety_context=ctx)
    assert controlled.planning_mode == 'recovery_controlled'
    assert controlled.metrics['recovery_search_delta'] == 0
    assert controlled.metrics['sac_to_selected_delta'] > 0


def test_D_model_disagreement_is_uncertainty_not_geometric_takeover():
    p, base, ctx = setup()
    original = p.candidate_safety.check
    def disagreement(a, c):
        r = original(a, c)
        r.center_model_valid = torch.ones(len(a), dtype=torch.bool)
        r.legacy_model_valid = torch.zeros(len(a), dtype=torch.bool)
        # A qualified primary may shadow Legacy; disagreement is still logged.
        r.primary_calibrated = torch.ones(len(a), dtype=torch.bool)
        r.legacy_shadow = torch.ones(len(a), dtype=torch.bool)
        r.full_valid = torch.zeros(len(a), dtype=torch.bool)
        r.braking_continuation_valid = torch.ones(len(a), dtype=torch.bool)
        return r
    p.candidate_safety.check = disagreement
    r = p.plan(torch.zeros(1, 2), base, eval_mode=True, safety_context=ctx)
    assert r.baseline_selected and r.metrics['model_uncertainty'] == 1
    assert r.metrics['recovery_search'] == 0 and r.safety_certificate['valid']
    # Without primary calibration the real filter must retain the AND guard.
    from tests.test_trajectory_selection import test_guard_never_accepts_a_path_rejected_by_original_geometry
    test_guard_never_accepts_a_path_rejected_by_original_geometry()


def test_E_preview_only_high_score_cannot_update_elites_or_win():
    p, base, ctx = setup(lambda a: 100*a[:, 0, 0])
    high = base.clone(); high[:, 0] = .04
    p._candidate_batch = lambda *a: torch.stack((base, high, high, high))
    def exact(a, c):
        valid = a[:, 0, 0] == 0
        return SimpleNamespace(valid=valid, near_valid=torch.ones_like(valid),
            collision=~valid, offroad=torch.zeros_like(valid), kinematic_violation=torch.zeros_like(valid))
    p.candidate_safety.check = exact
    r = p.plan(torch.zeros(1, 2), base, eval_mode=True, safety_context=ctx)
    assert torch.equal(r.final_mean, r.initial_mean) and torch.equal(r.final_std, r.initial_std)
    assert r.baseline_selected and r.metrics['exact_failed_candidates'] > 0
    assert r.metrics['elite_update_skipped'] == p.iterations


def test_F_traffic_waiting_and_unverified_value_keep_sac_priority():
    p, base, ctx = setup(lambda a: 100*a[:, 0, 0])
    p._stall_active = True
    ctx.route_progress = 0.; ctx.geometric_body_margin = .1
    ctx.geometric_lateral_error = 5.; ctx.lane_width = 4.; ctx.lane_centers = (0.,)
    ctx.obstacles = (obstacle(14, 10),)
    trapped, waiting = p._trapped_state(ctx, 0.)
    assert waiting and not trapped
    p.evaluator.value_ranking_ready = lambda: False
    candidate = base.clone(); candidate[:, 0] = .02
    p._candidate_batch = lambda *a: torch.stack((base, candidate, candidate, base))
    with patch.object(p, '_controlled_candidates', side_effect=AssertionError('accelerating into traffic')):
        r = p.plan(torch.zeros(1, 2), base, eval_mode=True, safety_context=ctx)
    assert r.baseline_selected and r.metrics['value_ranking_validated'] == 0
    assert r.metrics['traffic_waiting'] == 1 and r.metrics['recovery_search'] == 0
