"""Focused V3.1 regressions; reuse the existing planner/model fixtures."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mpr_mpc.evaluation_diagnostics import EpisodeDiagnostics, StepDiagnosticsLogger
from mpr_mpc.evaluation_statistics import paired_comparisons, summarize_runs
from mpr_mpc.planning.cost_validation import CostModelValidator, candidate_ranking_ready, cost_definition
from mpr_mpc.planning.mppi_safety import MPPISafetyFilter
from tests.test_mppi_v3 import planner, braking_context, fake_filter, fixed_batch, safety_cfg
from tests.test_native_mppi_cost import fixed_planner, rollout_evaluator
from tests.test_v2_safety import ExactCostModel, obstacle
from tests.test_evaluation_ablation import diagnostic_row


RECOVERY = dict(enabled=True, num_samples=4, iterations=2,
                initial_std=[.06, .2], max_delta=[.2, .5])


@pytest.mark.parametrize("recovery_value,expected_mode", [(3., "normal"), (20., "recovery_lattice")])
def test_normal_and_recovery_use_common_final_scores_and_direct_lattice_is_not_optimization(
        monkeypatch, recovery_value, expected_mode):
    p = planner(recovery=RECOVERY, cost_enabled=True, cost_coef=2.,
                deviation_coef=1., smoothness_coef=.3)
    baseline, normal, lattice = torch.zeros(4, 2), torch.zeros(4, 2), torch.zeros(4, 2)
    normal[:] = torch.tensor([.04, .05])
    lattice[:, 0], lattice[:, 1] = torch.tensor([.2, .1, .2, .1]), .2
    fixed_batch(p, baseline, normal)
    fake_filter(p, lambda a: a[:, 0, 0] > 0)
    p.evaluator.value = lambda a: torch.where(a[:, 0, 0] < .1, 2., recovery_value)
    # Recovery proposals stay at Lattice; its local deviation/smoothness are
    # zero, but its SAC-referenced final penalties are much larger. A raw
    # comparison would incorrectly choose recovery in the low-value case.
    monkeypatch.setattr(torch, "randn", lambda *shape, **kw: torch.zeros(*shape, device=kw.get("device")))
    result = p.plan(torch.zeros(1, 2), baseline, eval_mode=True,
                    safety_context=braking_context(speed=0), lattice_actions=lattice)
    assert not p.allow_unvalidated_cost
    assert result.planning_mode == expected_mode
    assert result.metrics["recovery_search"] == 1
    expected = normal if expected_mode == "normal" else lattice
    torch.testing.assert_close(result.selected_sequence, expected)
    deviation, smoothness = p._penalties(expected[None], baseline, p.max_delta)
    value = 2. if expected_mode == "normal" else recovery_value
    expected_score = value-2*float(expected[0, 1])-float(deviation[0])-.3*float(smoothness[0])
    assert result.selected_score == pytest.approx(expected_score)
    assert result.metrics["final_score"] == pytest.approx(expected_score)
    assert result.metrics["recovery_search_delta"] == 0
    assert result.metrics["recovery_lattice_baseline_selected"] == float(expected_mode == "recovery_lattice")
    assert result.metrics["sac_to_selected_delta"] == pytest.approx(float((expected[0]-baseline[0]).norm()))
    assert result.metrics["first_action_delta_norm"] == result.metrics["sac_to_selected_delta"]


def test_recovery_recenters_on_checked_lattice_outside_sac_trust_region():
    p = planner(recovery=RECOVERY)
    baseline, lattice = torch.full((4, 2), .6), torch.zeros(4, 2)
    fixed_batch(p, baseline, baseline)
    checks = fake_filter(p, lambda a: a[:, 0, 0].abs() < .15)
    result = p.plan(torch.zeros(1, 2), baseline, eval_mode=True,
                    safety_context=braking_context(speed=0), lattice_actions=lattice)
    assert result.planning_mode == "recovery_lattice"
    assert result.recovery_reason == "unsafe_sac_baseline"
    assert result.metrics["lattice_baseline_safety_valid"] == 1
    assert result.metrics["normal_search_delta"] == result.metrics["emergency_fallback_delta"] == 0
    assert result.metrics["recovery_search_delta"] == pytest.approx(float((result.action-lattice[0]).norm()))
    assert result.metrics["sac_to_selected_delta"] == pytest.approx(float((result.action-baseline[0]).norm()))
    assert (result.selected_sequence-baseline).abs().max() > p.max_delta.max()
    assert ((result.selected_sequence-lattice).abs() <= p.recovery_delta+1e-6).all()
    assert ((result.final_mean-lattice).abs() <= p.recovery_delta+1e-6).all()
    torch.testing.assert_close(p._prev_mean, result.final_mean)
    assert sum(len(a) for a in checks) <= 4+1+4*2
    assert result.metrics["recovery_wm_evaluated_candidates"] <= 1+4*2
    assert result.metrics["emergency_fallback"] == 0


def test_controlled_recovery_and_unverified_brake_have_distinct_attribution():
    p = planner(recovery=RECOVERY)
    baseline = torch.full((4, 2), .6)
    fixed_batch(p, baseline, baseline)
    fake_filter(p, lambda a: (a[:, 0, 1] == -.5) & (a[:, 0, 0] <= .351))
    ctx = braking_context()
    ctx.previous_action = np.array([.4, .6])
    result = p.plan(torch.zeros(1, 2), baseline, eval_mode=True,
        safety_context=ctx, lattice_actions=baseline)
    assert result.planning_mode == "recovery_controlled"
    assert result.metrics["lattice_baseline_safety_valid"] == 0
    assert result.metrics["emergency_fallback_delta"] == 0
    assert result.metrics["recovery_search_delta"] > 0
    reference = p.candidate_safety.braking_sequence(ctx, horizon=3, device="cpu", dtype=baseline.dtype)
    assert result.metrics["recovery_search_delta"] == pytest.approx(float((result.action-reference[0]).norm()))
    assert result.metrics["sac_to_selected_delta"] == pytest.approx(float((result.action-baseline[0]).norm()))
    fake_filter(p, lambda a: torch.zeros(len(a), dtype=torch.bool))
    result = p.plan(torch.zeros(1, 2), baseline, eval_mode=True,
                    safety_context=braking_context(), lattice_actions=baseline)
    assert result.planning_mode == "emergency_model_check_failed"
    assert result.metrics["safety_fallback_unverified"] == 1
    assert result.metrics["normal_search_delta"] == result.metrics["recovery_search_delta"] == 0
    assert result.metrics["emergency_fallback_delta"] > 0


def test_healthy_baseline_never_consumes_recovery_budget_and_full_brake_is_not_search_gain():
    p = planner(recovery=RECOVERY)
    baseline = torch.zeros(4, 2)
    fixed_batch(p, baseline, baseline)
    result = p.plan(torch.zeros(1, 2), baseline, eval_mode=True, safety_context=braking_context(speed=0))
    assert result.metrics["recovery_search"] == result.metrics["recovery_search_ms"] == 0
    assert result.metrics["recovery_wm_evaluated_candidates"] == 0
    fake_filter(p, lambda a: a[:, 0, 1] <= -1)
    result = p.plan(torch.zeros(1, 2), baseline, eval_mode=True, safety_context=braking_context())
    assert result.planning_mode == "emergency_model_checked"
    assert result.metrics["emergency_fallback_delta"] > 0
    assert result.metrics["recovery_search_delta"] == 0


def test_unvalidated_cost_is_diagnostic_only_unless_explicitly_enabled():
    p, evaluator, baseline = fixed_planner()
    p.allow_unvalidated_cost = False
    result = p.plan(torch.zeros(1, 2), baseline, eval_mode=True)
    assert result.baseline_selected
    assert result.metrics["cost_scoring_requested"] == 1 and result.metrics["cost_enabled"] == 0
    assert result.metrics["baseline_candidate_cost"] == .5 and result.metrics["cost_penalty"] == 0
    p.allow_unvalidated_cost = True
    with pytest.warns(RuntimeWarning, match="same-state"):
        result = p.plan(torch.zeros(1, 2), baseline, eval_mode=True)
    assert not result.baseline_selected and result.metrics["cost_enabled"] == 1


def test_horizon_validation_cannot_open_ranking_gate_and_evidence_is_revoked():
    cfg = dict(min_samples=1, min_updates=1, horizons=[1, 3], ranking_min_origins=2,
               ranking_min_pairs=2, ranking_min_origin_ci95=.01, max_mae=.001)
    definition = cost_definition({"cost": cfg})
    validator = CostModelValidator(cfg)
    model = ExactCostModel()
    # A real parameter lets us verify optimizer/load mutation revocation too.
    model.register_parameter("version_probe", torch.nn.Parameter(torch.ones(1)))
    actions = np.tile([.2, 0.], (12, 1)).astype("float32")
    validator.set_episodes([(np.zeros((12, 3)), actions, actions[:, 0])])
    validator.validate(model, "cpu", horizon=3)
    assert model.cost_ready and not candidate_ranking_ready(model, 3, definition)
    rows = [dict(executed=True, origin_id=origin, initial_state_id=f"snapshot-{origin}",
        cost_definition=definition, observation=np.zeros(3), actions=np.tile([value, 0.], (3, 1)),
        costs=np.full(3, value)) for origin in [1, 2] for value in [.1, .4, .8]]
    validator.set_candidates(rows)
    metrics = validator.validate(model, "cpu", horizon=3)
    assert metrics["wm/cost_ranking_ready"] == 1
    assert candidate_ranking_ready(model, 3, definition)
    assert not candidate_ranking_ready(model, 3, definition+"changed-cost-definition")
    validator.validate(model, "cpu", horizon=3)
    with torch.no_grad():
        model.version_probe.add_(1)
    assert not candidate_ranking_ready(model, 3, definition)
    validator.validate(model, "cpu", horizon=3)
    model._cost_updates += 1
    assert not candidate_ranking_ready(model, 3, definition)
    validator.validate(model, "cpu", horizon=3)
    validator.invalidate_data()
    assert not candidate_ranking_ready(model, 3, definition)
    validator.set_candidates(rows)
    validator._candidate_data_time -= 100000
    validator.validate(model, "cpu", horizon=3)
    assert not candidate_ranking_ready(model, 3, definition)
    # Good unpaired prediction errors cannot compensate for reversed ordering.
    reversed_rows = [{**row, "costs": 1-np.asarray(row["costs"])} for row in rows]
    validator.cfg.update(max_mae=2., max_rmse=2.)
    validator.set_candidates(reversed_rows)
    reversed_metrics = validator.validate(model, "cpu", horizon=3)
    assert reversed_metrics["wm/cost_ready"] == 1 and reversed_metrics["risk_ranking_accuracy"] == 0
    assert not candidate_ranking_ready(model, 3, definition)
    # Observation-only grouping is still diagnostic, never proof that the
    # simulator hidden state/RNG/other actors were restored.
    for row in rows:
        row.pop("initial_state_id")
    validator.set_candidates(rows)
    validator.validate(model, "cpu", horizon=3)
    assert not candidate_ranking_ready(model, 3, definition)


def test_batched_cost_prediction_preserves_latent_time_alignment():
    evaluator = rollout_evaluator(horizon=3)
    evaluator.agent.model.cost = lambda z, a, task: z[:, :1]+a[:, :1].abs()
    actions = torch.full((2, 4, 2), .4)
    result = evaluator.evaluate_mppi_candidates(torch.zeros(1, 3), actions)
    expected = torch.tensor([[.4, 1.4, 2.4]]).expand(2, -1)
    torch.testing.assert_close(result.predicted_cost, expected)
    torch.testing.assert_close(result.cumulative_cost, (expected*torch.tensor([1., .5, .25])).sum(1))


def test_uncertainty_envelope_never_opens_collision_gate_and_first_transition_is_separate():
    ctx = braking_context(speed=0, obstacles=(obstacle(18., 10., vx=-3.),))
    controls = torch.zeros(1, 11, 2)
    cv = MPPISafetyFilter(safety_cfg(), control_dt=.1).check(controls, ctx)
    inflated = MPPISafetyFilter(safety_cfg(obstacle_acceleration_bound=8.), control_dt=.1).check(controls, ctx)
    assert not cv.collision[0] and inflated.collision[0]
    assert not inflated.first_collision[0]  # the collision is later in H
    assert not inflated.valid[0]


def test_recovery_and_prediction_outcomes_reach_step_episode_and_paired_outputs(tmp_path):
    p = planner(recovery=RECOVERY)
    baseline = torch.zeros(4, 2)
    result = p.plan(torch.zeros(1, 2), baseline, eval_mode=True, safety_context=braking_context(speed=0))
    plan = SimpleNamespace(metrics={f"mppi/{k}": v for k, v in result.metrics.items()},
        stage=SimpleNamespace(use_mppi=True, use_residual=True), planning_mode=result.planning_mode,
        recovery_reason=result.recovery_reason)
    info = dict(crash_vehicle=True, out_of_road=True)
    diagnostics = EpisodeDiagnostics()
    diagnostics.observe(np.zeros(3), result.action, info, plan)
    summary = diagnostics.summary(info)
    assert summary["mppi_predicted_safe_actual_collision"] == 1
    assert summary["mppi_predicted_safe_actual_offroad"] == 1
    assert "mppi_normal_search_delta" in summary and "mppi_recovery_search_delta" in summary
    for name in ("sac_to_selected_delta", "recovery_lattice_baseline_selected", "emergency_horizon_limited"):
        assert f"mppi_{name}" in summary
    logger = StepDiagnosticsLogger(tmp_path, "v31")
    logger.log(539, 1, result.action, plan, info=info)
    logger.close()
    row = json.loads((tmp_path/"v31.jsonl").read_text())
    assert row["mppi/actual_collision"] == 1 and row["planning_mode"] == result.planning_mode
    assert row["mppi/sac_to_selected_delta"] == result.metrics["sac_to_selected_delta"]
    a = diagnostic_row(539, 1, 1, configuration="B_lattice_sac")
    b = diagnostic_row(539, 1, 1, configuration="D_lattice_sac_mppi")
    a.update(summary); b.update(summary)
    assert "mppi_recovery_search_delta" in summarize_runs("A_lattice", [a], samples=8, seed=1)
    assert any("mppi_recovery_search_delta" in key for row in paired_comparisons([a, b], samples=8, seed=1) for key in row)
    assert any("mppi_emergency_horizon_limited" in key for row in paired_comparisons([a, b], samples=8, seed=1) for key in row)
