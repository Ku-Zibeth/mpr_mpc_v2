import json
import os
import shlex
import subprocess
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
import numpy as np
import pytest
from shapely.geometry import box
from shapely.ops import unary_union
import torch

import evaluate_model_suite as suite
from mpr_mpc._bootstrap import PACKAGE_ROOT
from mpr_mpc.evaluation_diagnostics import EpisodeDiagnostics, StepDiagnosticsLogger
from mpr_mpc.evaluation_statistics import paired_comparisons, summarize_runs
from mpr_mpc.planning.local_mppi import LocalMPPI, MPPI_EPISODE_METRICS
from mpr_mpc.planning.kinematic_filter import CorridorContext, CorridorResult
from mpr_mpc.planning.native_config import native_mppi_config
from mpr_mpc.planning.coordinator import MPRMPCPlanner
from mpr_mpc.planning.types import StructuredSelection
from mpr_mpc.planning.mppi_safety import MPPISafetyContext, MPPISafetyFilter
from tests.test_evaluation_ablation import diagnostic_row
from tests.test_mpr_mpc import mppi_cfg
from tests.test_residual_rl import _checkpoint_agent
from tests.test_v2_safety import context, obstacle


class RankedEvaluator:
    def __init__(self, value=None):
        self.calls = []
        self.value = value or (lambda actions: actions[:, 0, 0] * 10)

    def evaluate_mppi_candidates(self, z, actions, *, task=None, compute_cost=None):
        self.calls.append(actions.clone())
        values = self.value(actions)
        cost = actions[:, 0, 1].abs()
        return SimpleNamespace(total_value=values, discounted_reward=values, terminal_q=values*0,
            cumulative_cost=cost if compute_cost else None,
            predicted_cost=cost[:, None].expand(-1, actions.shape[1]-1) if compute_cost else None,
            cost_validated=True, cost_ranking_validated=True, timings={})


def safety_cfg(**changes):
    return dict(enabled=True, substeps=2, model_margin=0., road_margin=0.,
                max_steering_rate=20., max_lateral_acceleration=10., **changes)


def planner(**changes):
    cfg = mppi_cfg(safety=safety_cfg(), num_samples=4, num_elites=2, iterations=2,
                   min_improvement_abs=0., min_improvement_ratio=0., **changes)
    return LocalMPPI(RankedEvaluator(), cfg, horizon=3, action_dim=2, device="cpu")


def braking_context(**kwargs):
    original = context(**kwargs)
    return MPPISafetyContext(**{key: value for key, value in vars(original).items()}, brake_action_mapping_verified=True)


def fake_filter(planner, rule):
    calls = []

    def check(actions, ctx):
        calls.append(actions.clone())
        valid = rule(actions)
        return SimpleNamespace(valid=valid, collision=~valid, offroad=torch.zeros_like(valid),
                               kinematic_violation=torch.zeros_like(valid))

    planner.candidate_safety.check = check
    return calls


def fixed_batch(planner, baseline, candidate):
    bad = baseline.clone()
    bad[:, 0] = -.04
    planner._candidate_batch = lambda *args: torch.stack((baseline, candidate, bad, baseline))


def test_unsafe_baseline_never_overrides_safe_candidate_or_reaches_world_model():
    p = planner()
    p.min_improvement_abs = 1000
    baseline, candidate = torch.zeros(4, 2), torch.zeros(4, 2)
    candidate[:, 0] = .04
    fixed_batch(p, baseline, candidate)
    checks = fake_filter(p, lambda actions: actions[:, 0, 0] > 0)
    result = p.plan(torch.zeros(1, 2), baseline, eval_mode=True, safety_context=braking_context())
    torch.testing.assert_close(result.selected_sequence, candidate)
    assert not result.baseline_selected and result.metrics["baseline_safety_valid"] == 0
    assert np.isnan(result.metrics["gate_threshold"]) and torch.isnan(result.planner_score_gain)
    assert len(p.evaluator.calls) == 1 and len(p.evaluator.calls[0]) == 1
    assert len(checks) == 1 and len(checks[0]) == 3
    assert result.metrics["safety_reject_count"] == result.metrics["collision_reject_count"] == 6
    assert result.metrics["wm_cache_hits"] == 1
    assert result.metrics["unique_candidate_count"] == 3
    assert result.metrics["first_steer_delta"] == pytest.approx(.04)
    assert result.metrics["first_action_delta_norm"] == pytest.approx(float((result.action-baseline[0]).norm()))


def test_v3_expanded_search_uses_new_range_without_escaping_trust_region():
    with initialize_config_dir(version_base="1.3", config_dir=str(PACKAGE_ROOT)):
        cfg = compose(config_name="config", overrides=["+experiment=E5"])
    p = LocalMPPI(RankedEvaluator(), cfg.mpr_mpc.mppi, horizon=3, action_dim=2, device="cpu")
    baseline = torch.zeros(4, 2)
    p.reset(seed=1)
    actions = p._candidate_batch(baseline, p.initial_std_vector.expand_as(baseline), baseline, p.generator)
    assert (actions[..., 0].abs() > .05).any()
    assert (actions.abs() <= p.max_delta).all()
    result = p.plan(torch.zeros(1, 2), baseline, eval_mode=True, safety_context=braking_context(speed=0))
    assert result.metrics["first_action_delta_norm"] > 0
    assert ((result.selected_sequence-baseline).abs() <= p.max_delta+1e-6).all()


def test_safe_baseline_can_be_kept_by_improvement_gate():
    p = planner()
    p.min_improvement_abs = 1.
    baseline, candidate = torch.zeros(4, 2), torch.zeros(4, 2)
    candidate[:, 0] = .04
    fixed_batch(p, baseline, candidate)
    result = p.plan(torch.zeros(1, 2), baseline, eval_mode=True, safety_context=braking_context(speed=0))
    assert result.baseline_selected and result.metrics["baseline_safety_valid"] == 1
    assert result.metrics["gate_threshold"] == 1
    assert result.metrics["first_action_delta_norm"] == 0


def test_fast_corridor_rejects_before_independent_safety_and_wm():
    p = planner()
    baseline, candidate = torch.zeros(4, 2), torch.zeros(4, 2)
    candidate[:, 0] = .04
    fixed_batch(p, baseline, candidate)
    p.safety_corridor.check = lambda *args: CorridorResult(
        torch.tensor([False, True, False, False]), torch.zeros(4), torch.zeros(4, 3))
    checks = fake_filter(p, lambda a: torch.ones(len(a), dtype=torch.bool))
    ctx = CorridorContext(torch.zeros(3), 3.5, -3.5, 3.5, 0., 2., 2.5, .6)
    result = p.plan(torch.zeros(1, 2), baseline, eval_mode=True,
                    safety_context=braking_context(), corridor_context=ctx)
    assert sum(len(batch) for batch in checks) == 1
    assert sum(len(batch) for batch in p.evaluator.calls) == 1
    torch.testing.assert_close(result.action, candidate[0])


def test_mixed_nonfinite_baseline_cost_does_not_discard_other_usable_candidates():
    p = planner()
    p.cost_enabled = True
    original = p.evaluator.evaluate_mppi_candidates

    def evaluate(z, actions, **kwargs):
        result = original(z, actions, **kwargs)
        result.cumulative_cost = torch.where(actions[:, 0, 0] == 0, float("nan"), 0.)
        return result

    p.evaluator.evaluate_mppi_candidates = evaluate
    baseline, candidate = torch.zeros(4, 2), torch.zeros(4, 2)
    candidate[:, 0] = .04
    fixed_batch(p, baseline, candidate)
    result = p.plan(torch.zeros(1, 2), baseline, eval_mode=True, safety_context=braking_context(speed=0))
    torch.testing.assert_close(result.action, candidate[0])
    assert torch.isnan(result.planner_score_gain)


def test_selected_safety_positions_match_h_transitions_and_global_vehicle_pose():
    p = planner()
    result = p.plan(torch.zeros(1, 2), torch.zeros(4, 2), eval_mode=True, safety_context=braking_context())
    assert result.selected_positions.shape == (4, 2)
    np.testing.assert_allclose(result.selected_positions[0].numpy(), [10., 10.])
    assert result.selected_positions[-1, 0] > result.selected_positions[0, 0]


def test_mppi_context_preparation_does_not_require_sac_features_or_residual_state():
    from unittest.mock import Mock

    p = object.__new__(MPRMPCPlanner)
    coarse = SimpleNamespace(target_d=0., target_speed=5., horizon=3.)
    selection = StructuredSelection((coarse,), (coarse,), coarse, 0)
    p.device, p.horizon, p.control_dt = "cpu", 3, .1
    p.safety = None
    p.structured = SimpleNamespace(select_coarse=lambda vehicle: selection, planner=object())
    p.action_adapter = SimpleNamespace(path_to_actions=lambda *args, **kwargs: torch.zeros(4, 2))
    p.residual_bounds = lambda path: torch.ones(1, 2)
    p._vehicle_geometry = lambda vehicle: (2., 2.5, .6)
    physical = braking_context()
    candidate_filter = SimpleNamespace(context=Mock(return_value=physical))
    p.local_mppi = SimpleNamespace(safety_enabled=True, candidate_safety=candidate_filter)
    result = p.prepare(torch.zeros(259), SimpleNamespace(enable_reverse=False),
                       need_world_model_features=False, need_mppi_safety=True)
    assert result.residual_state is result.safety_context is None
    assert result.mppi_safety_context is physical
    assert candidate_filter.context.call_args.kwargs["snapshot"] is None
    p.safety = SimpleNamespace(context=Mock(), check=Mock())
    p.evaluator = SimpleNamespace(encode=Mock(return_value=torch.zeros(1, 512)),
                                 evaluate_trajectory_consequence=Mock())
    result = p.prepare(torch.zeros(259), SimpleNamespace(enable_reverse=False),
                       need_world_model_features=True, need_mppi_safety=True, need_residual_state=False)
    assert result.residual_state is result.coarse_consequence is None
    assert result.mppi_safety_context is physical
    p.safety.context.assert_not_called()
    p.evaluator.evaluate_trajectory_consequence.assert_not_called()


def test_native_config_preserves_new_safety_keys_but_not_retired_mode_gates():
    result = native_mppi_config({"num_samples": 64, "six_modes": True,
                                "safety": {"enabled": True, "substeps": 2, "six_mode_approval": True}})
    assert result == {"num_samples": 64, "safety": {"enabled": True, "substeps": 2}}


@pytest.mark.parametrize("invalid", [float("nan"), -float("inf")])
def test_nonfinite_baseline_score_does_not_poison_safe_candidate_selection(invalid):
    p = planner()
    p.evaluator = RankedEvaluator(lambda a: torch.where(a[:, 0, 0] == 0, invalid, a[:, 0, 0]*10))
    baseline, candidate = torch.zeros(4, 2), torch.zeros(4, 2)
    candidate[:, 0] = .04
    fixed_batch(p, baseline, candidate)
    result = p.plan(torch.zeros(1, 2), baseline, eval_mode=True, safety_context=braking_context(speed=0))
    torch.testing.assert_close(result.action, candidate[0])
    assert np.isnan(result.metrics["gate_threshold"])
    assert torch.isnan(result.planner_score_gain)


def test_lattice_fallback_uses_same_independent_safety_filter():
    p = planner()
    baseline, lattice = torch.zeros(4, 2), torch.zeros(4, 2)
    lattice[:, 0] = .04
    fixed_batch(p, baseline, baseline)
    checks = fake_filter(p, lambda a: a[:, 0, 0] > 0)
    result = p.plan(torch.zeros(1, 2), baseline, eval_mode=True,
                    safety_context=braking_context(), lattice_actions=lattice)
    assert len(checks) == 2
    assert result.metrics["lattice_fallback"] == 1
    assert result.metrics["recovery_search_delta"] == 0
    assert result.metrics["lattice_fallback_delta"] > 0
    assert result.metrics["emergency_fallback"] == 0
    torch.testing.assert_close(result.action, lattice[0])


@pytest.mark.parametrize("brake_safe", [True, False])
def test_unsafe_lattice_and_baseline_use_verified_brake_mapping_and_honest_fallback(brake_safe):
    p = planner()
    baseline = torch.zeros(4, 2)
    fixed_batch(p, baseline, baseline)
    fake_filter(p, lambda a: (a[:, 0, 1] < 0) & brake_safe)
    result = p.plan(torch.zeros(1, 2), baseline, eval_mode=True,
                    safety_context=braking_context(), lattice_actions=baseline)
    assert result.action.tolist() == [0, -1]
    assert result.metrics["emergency_fallback"] == 1
    assert result.metrics["lattice_fallback"] == 0
    assert result.metrics["safety_fallback_unverified"] == float(not brake_safe)
    assert result.metrics["wm_evaluated_candidates"] == int(brake_safe)
    assert result.planning_mode == ("emergency_model_checked" if brake_safe else "emergency_model_check_failed")
    assert result.metrics["stopping_time_s"] > p.horizon*p.candidate_safety.geometry.dt
    assert result.metrics["emergency_horizon_limited"] == 1
    if brake_safe:
        result = p.plan(torch.zeros(1, 2), baseline, eval_mode=True,
                        safety_context=braking_context(speed=0), lattice_actions=baseline)
        assert result.planning_mode == "emergency_model_checked"
        assert result.metrics["emergency_horizon_limited"] == 0


def test_missing_safety_context_is_an_error_and_unknown_brake_mapping_is_not_assumed():
    p = planner()
    with pytest.raises(RuntimeError, match="safety_context"):
        p.plan(torch.zeros(1, 2), torch.zeros(4, 2), eval_mode=True)
    assert not p.evaluator.calls
    with pytest.raises(RuntimeError, match="not been verified"):
        p.candidate_safety.braking_sequence(context(), horizon=3, device="cpu", dtype=torch.float32)


def test_brake_action_uses_metadrives_real_non_reversing_mapping():
    from unittest.mock import Mock
    from metadrive.component.vehicle.base_vehicle import BaseVehicle

    system = Mock()
    vehicle = SimpleNamespace(config={"max_engine_force": 1000, "max_brake_force": 100},
                              system=system, enable_reverse=False)
    BaseVehicle._apply_throttle_brake(vehicle, -1.)
    assert [call.args for call in system.applyEngineForce.call_args_list] == [(0., wheel) for wheel in range(4)]
    assert [call.args for call in system.setBrake.call_args_list] == [(100., wheel) for wheel in range(4)]


def test_continuous_road_check_catches_gap_between_safe_endpoint_footprints():
    safety = MPPISafetyFilter(dict(substeps=1, road_margin=0., model_margin=0.), control_dt=1.)
    road = unary_union([box(0, 5, 15, 15), box(19, 5, 50, 15)])
    result = safety.check(torch.zeros(1, 2, 2), braking_context(speed=20, road=road))
    assert result.offroad.item() and not result.valid.item()


def test_continuous_dynamic_collision_check_rejects_contact_between_endpoints():
    safety = MPPISafetyFilter(dict(substeps=1, road_margin=0., model_margin=0.), control_dt=1.)
    result = safety.check(torch.zeros(1, 2, 2), braking_context(speed=20, obstacles=(obstacle(20, 10),)))
    assert result.collision.item() and not result.valid.item()


def test_body_not_just_center_is_checked_and_terminal_q_action_has_no_motion():
    safety = MPPISafetyFilter(safety_cfg(), control_dt=.1)
    actions = torch.zeros(2, 4, 2)
    actions[1, -1] = 1
    narrow = braking_context(speed=0, road=box(0, 9.5, 50, 10.5))
    result = safety.check(actions, narrow)
    assert result.offroad.all()
    torch.testing.assert_close(result.positions[0], result.positions[1])
    assert result.positions.shape == (2, 7, 2)


def test_episode_rng_is_reproducible_but_advances_without_touching_sac_rng():
    p = planner()
    p.reset(seed=123)
    global_rng = torch.random.get_rng_state().clone()
    p.plan(torch.zeros(1, 2), torch.zeros(4, 2), eval_mode=True, safety_context=braking_context(speed=0))
    first = p.evaluator.calls[0].clone()
    assert torch.equal(global_rng, torch.random.get_rng_state())
    p.evaluator.calls.clear()
    p.plan(torch.zeros(1, 2), torch.zeros(4, 2), eval_mode=True, safety_context=braking_context(speed=0))
    assert not torch.equal(first, p.evaluator.calls[0])
    p.reset(seed=123)
    p.evaluator.calls.clear()
    p.plan(torch.zeros(1, 2), torch.zeros(4, 2), eval_mode=True, safety_context=braking_context(speed=0))
    torch.testing.assert_close(first, p.evaluator.calls[0])
    p.reset(seed=124)
    p.evaluator.calls.clear()
    p.plan(torch.zeros(1, 2), torch.zeros(4, 2), eval_mode=True, safety_context=braking_context(speed=0))
    assert not torch.equal(first, p.evaluator.calls[0])


def test_c_and_d_consume_same_episode_noise_rule_even_with_different_baselines():
    a, b = planner(), planner()
    for p, baseline in ((a, torch.zeros(4, 2)), (b, torch.full((4, 2), .02))):
        p.reset(seed=1715)
        p.plan(torch.zeros(1, 2), baseline, eval_mode=True, safety_context=braking_context(speed=0))
    assert torch.equal(a.generator.get_state(), b.generator.get_state())


@pytest.mark.parametrize("restore", [False, True])
def test_checkpoint_restores_sac_semantics_but_mppi_only_when_authorized(tmp_path, restore):
    with initialize_config_dir(version_base="1.3", config_dir=str(PACKAGE_ROOT)):
        cfg = compose(config_name="config", overrides=["+experiment=E5"])
    cfg.restore_mppi_runtime = restore
    checkpoint = tmp_path / "model.pt"
    features = ["cumulative_cost", "min_ttc", "min_clearance", "boundary_margin"]
    torch.save(dict(metadata=dict(encoder_config=dict(grouped=True), risk_features=features,
                                  residual_mapping="latent_tanh"),
                    planner=dict(mppi_config=dict(max_delta=[.02, .03], multimodal=True),
                                 stage_config=dict(residual_start_step=200000, mppi_start_step=500000))), checkpoint)
    suite._apply_checkpoint_runtime_config(cfg, checkpoint)
    assert cfg.restore_checkpoint_runtime
    assert cfg.mpr_mpc.stages.residual_start_step == 200000
    assert list(cfg.residual_rl.risk_features) == features
    assert cfg.residual_rl.residual_mapping == "latent_tanh"
    assert list(cfg.mpr_mpc.mppi.max_delta) == ([.02, .03] if restore else [.10, .25])
    assert cfg.mpr_mpc.mppi.safety.enabled is not restore
    assert "multimodal" not in cfg.mpr_mpc.mppi


@pytest.mark.parametrize("field,bad", [("latent_dim", 16), ("residual_state_dim", 1),
                                       ("residual_mapping", "latent_tanh"), ("risk_features", ["changed"]),
                                       ("model_size", 7)])
def test_mppi_override_does_not_relax_checkpoint_architecture_checks(field, bad):
    agent = _checkpoint_agent()
    state = agent.milestone_state("full_mpr_mpc", global_env_step=1000000)
    agent.cfg.restore_mppi_runtime = False
    agent.planner.mppi_cfg = {"num_samples": 128}
    agent._check_checkpoint_compatibility(state, strict_planner_config=True)
    state["metadata"][field] = bad
    with pytest.raises(ValueError):
        agent._check_checkpoint_compatibility(state, strict_planner_config=True)


def test_new_metrics_reach_episode_summary_pairs_and_step_evidence(tmp_path):
    p = planner()
    base, changed = torch.zeros(4, 2), torch.zeros(4, 2)
    changed[:, 0] = .04
    fixed_batch(p, base, changed)
    result = p.plan(torch.zeros(1, 2), base, eval_mode=True, safety_context=braking_context(speed=0))
    metrics = {f"mppi/{key}": value for key, value in result.metrics.items()}
    metrics["mppi/total_planning_ms"] = 10.
    plan = SimpleNamespace(stage=SimpleNamespace(use_residual=True, use_mppi=True), metrics=metrics)
    diagnostic = EpisodeDiagnostics()
    diagnostic.observe(np.zeros(3), result.action, {}, plan)
    summary = diagnostic.summary({})
    assert summary["mppi_first_steer_delta"] == pytest.approx(.04)
    for key in MPPI_EPISODE_METRICS:
        assert key in summary
    rows = [{**diagnostic_row(1714, 1, 1.), **summary, "configuration": label}
            for label in ("C_lattice_mppi", "D_lattice_sac_mppi")]
    stats = summarize_runs("D", rows[1:], samples=30, seed=1)
    assert stats["mppi_first_steer_delta"] == pytest.approx(.04)
    pairs = paired_comparisons(rows, samples=30, seed=1, pairs=[("C_lattice_mppi", "D_lattice_sac_mppi")])
    assert pairs[0]["mppi_first_steer_delta_delta"] == 0
    logger = StepDiagnosticsLogger(tmp_path, "case")
    logger.log(1714, 1, result.action, plan)
    logger.close()
    saved = json.loads((tmp_path / "case.jsonl").read_text())
    assert saved["executed_action"][0] == pytest.approx(.04)
    assert saved["mppi/first_steer_delta"] == pytest.approx(.04)


def test_formal_ablation_command_selects_frozen_model_fixed100_and_v3_runtime(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    result = subprocess.run(["bash", str(PACKAGE_ROOT / "tools/run_v3_ablation.sh"), "--dry-run"],
        env={**os.environ, "CHECKPOINT": str(checkpoint), "EXP_NAME": "test_v3_formal_command"},
        check=True, capture_output=True, text=True, timeout=10)
    command = shlex.split(result.stdout)
    with initialize_config_dir(version_base="1.3", config_dir=str(PACKAGE_ROOT)):
        cfg = compose(config_name="config", overrides=command[3:])
    assert cfg.checkpoint == str(checkpoint)
    assert cfg.restore_checkpoint_runtime and not cfg.restore_mppi_runtime
    assert cfg.mpr_mpc.mppi.safety.enabled
    assert cfg.benchmark.episodes == cfg.eval_episodes == 100
    assert not cfg.benchmark.cost_ablation
    assert list(cfg.benchmark.repeat_seeds) == [1]
    assert cfg.benchmark.save_step_diagnostics
    scenarios = json.loads(open(cfg.benchmark.scenarios_file).read())["scenarios"]
    assert len(scenarios) == len(set(scenarios)) == 100
