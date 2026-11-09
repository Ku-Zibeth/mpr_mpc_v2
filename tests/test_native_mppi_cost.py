from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mpr_mpc.planning.cost_validation import CostModelValidator, cost_statistics
from mpr_mpc.planning.evaluator import TrajectoryWorldModelEvaluator
from mpr_mpc.planning.local_mppi import LocalMPPI
from mpr_mpc.planning.native_config import native_mppi_config
from tests.test_mpr_mpc import mppi_cfg
from tests.test_v2_safety import ExactCostModel


class CostEvaluator:
    def __init__(self, *, validated=False, bad_cost=None):
        self.calls = []
        self.validated = validated
        self.bad_cost = bad_cost

    def evaluate_mppi_candidates(self, z, actions, *, task=None, compute_cost=None):
        self.calls.append(actions.clone())
        value = actions.new_full((len(actions),), 10.)
        cost = actions[:, 0, 1].abs()
        if self.bad_cost is not None:
            cost[:] = self.bad_cost
        return SimpleNamespace(total_value=value, terminal_q=value*0,
            discounted_reward=value, cumulative_cost=cost if compute_cost else None,
            predicted_cost=cost[:, None].expand(-1, actions.shape[1]-1) if compute_cost else None,
            cost_validated=self.validated, cost_ranking_validated=self.validated, timings={})


def fixed_planner(coef=1., *, cost_enabled=True, **kwargs):
    evaluator = CostEvaluator(**kwargs)
    planner = LocalMPPI(evaluator, mppi_cfg(cost_enabled=cost_enabled, cost_coef=coef, allow_unvalidated_cost=True,
        num_samples=4, num_elites=2, iterations=3), horizon=3, action_dim=2, device="cpu")
    baseline = torch.zeros(4, 2)
    baseline[:, 1] = .5
    candidates = torch.stack((baseline, baseline.clone(), torch.zeros_like(baseline), torch.zeros_like(baseline)))
    planner._candidate_batch = lambda *args: candidates.clone()
    return planner, evaluator, baseline


def test_explicit_experiment_allows_unvalidated_cost_without_safety_claim():
    planner, _, baseline = fixed_planner()
    with pytest.warns(RuntimeWarning, match="NOT passed"):
        result = planner.plan(torch.zeros(1, 2), baseline, eval_mode=True)
    assert result.action[1] == 0 and not result.baseline_selected
    assert result.metrics["cost_validated"] == 0
    assert result.metrics["cost_penalty"] == result.metrics["selected_candidate_cost"] == 0
    assert result.metrics["baseline_candidate_cost"] == .5
    assert result.planner_score_gain == pytest.approx(.5)


@pytest.mark.parametrize("coefficient", [0., .25, 1., 2.])
def test_cost_is_deducted_exactly_once(coefficient):
    planner, _, baseline = fixed_planner(coefficient, validated=True)
    result = planner.plan(torch.zeros(1, 2), baseline, eval_mode=True)
    assert result.baseline_score == pytest.approx(10-coefficient*.5)
    m = result.metrics
    assert result.selected_score == pytest.approx(
        result.selected_wm_value-m["cost_penalty"]-m["deviation_penalty"]-m["smoothness_penalty"])
    if coefficient == 0:
        assert result.baseline_selected and result.baseline_score == 10


def test_cost_disabled_does_not_require_or_call_cost_head():
    planner, evaluator, baseline = fixed_planner(cost_enabled=False)
    result = planner.plan(torch.zeros(1, 2), baseline, eval_mode=True)
    assert result.metrics["cost_penalty"] == 0
    assert np.isnan(result.metrics["predicted_cost"])
    assert result.baseline_selected


def test_exact_candidates_are_never_rolled_out_twice_within_plan():
    planner, evaluator, baseline = fixed_planner(validated=True)
    result = planner.plan(torch.zeros(1, 2), baseline, eval_mode=True)
    assert len(evaluator.calls) == 1 and len(evaluator.calls[0]) == 2
    assert result.metrics["wm_evaluated_candidates"] == 2
    assert result.metrics["wm_cache_hits"] == 10
    planner.plan(torch.ones(1, 2), baseline, eval_mode=True)
    assert len(evaluator.calls) == 2  # No reuse at a different current state.


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -.1])
def test_invalid_predicted_cost_is_explicit_error(bad):
    planner, _, baseline = fixed_planner(bad_cost=bad)
    with pytest.raises(RuntimeError, match="cumulative WM Cost"):
        planner.plan(torch.zeros(1, 2), baseline, eval_mode=True)


@pytest.mark.parametrize("parameter", ["cost_coef", "deviation_coef", "smoothness_coef", "temperature"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.])
def test_native_score_weights_must_be_finite_nonnegative(parameter, bad):
    with pytest.raises(ValueError, match=parameter):
        LocalMPPI(CostEvaluator(), mppi_cfg(**{parameter: bad}), horizon=3, action_dim=2, device="cpu")


def test_native_planner_has_no_multimodal_or_emergency_state():
    planner, _, _ = fixed_planner()
    for name in ("_mode_bank", "mode_weights", "expanded_delta", "emergency_count", "safety_evaluator"):
        assert not hasattr(planner, name)


def test_historical_native_settings_and_checkpoint_migration():
    from hydra import compose, initialize_config_dir
    from mpr_mpc._bootstrap import PACKAGE_ROOT
    with initialize_config_dir(version_base="1.3", config_dir=str(PACKAGE_ROOT)):
        cfg = compose(config_name="config", overrides=["+experiment=E5"])
    mppi = cfg.mpr_mpc.mppi
    assert (mppi.num_samples, mppi.num_elites, mppi.iterations) == (64, 8, 4)
    assert list(mppi.initial_std) == [.04, .12]
    assert list(mppi.max_delta) == [.10, .25]
    assert mppi.temperature == .5 and mppi.deviation_coef == 1 and mppi.smoothness_coef == .3
    assert mppi.safety.enabled
    assert cfg.cost.enabled and cfg.metadrive.cost.enabled
    assert cfg.steps == 50000 and cfg.mpr_mpc.stages.residual_start_step == 20000
    saved = dict(mppi, multimodal=True, dynamic_safety=True, uncertainty_coef=1000., mode_weights=[1, 2, 3, 4])
    restored = native_mppi_config(saved)
    assert restored == dict(mppi)


class RolloutModel(ExactCostModel):
    _termination = None

    def reward(self, z, action, task):
        return torch.ones_like(action[:, :1])*float(np.log1p(2))

    def Q(self, z, action, task, return_type="all"):
        return torch.ones(3, len(action), 1)*float(np.log1p(4))


def rollout_evaluator(horizon=3):
    model = RolloutModel()
    cfg = SimpleNamespace(action_dim=2, latent_dim=3, multitask=False, episodic=True,
                          num_bins=1, cost={"enabled": True, "gamma": .5})
    agent = SimpleNamespace(model=model, cfg=cfg, discount=.9, device="cpu")
    return TrajectoryWorldModelEvaluator(agent, horizon)


def test_cost_discount_h_plus_one_reward_and_terminal_q_semantics():
    evaluator = rollout_evaluator()
    actions = torch.full((2, 4, 2), .4)
    result = evaluator.evaluate_mppi_candidates(torch.zeros(1, 3), actions)
    torch.testing.assert_close(result.cumulative_cost, torch.full((2,), .4*(1+.5+.25)))
    torch.testing.assert_close(result.discounted_reward, torch.full((2,), 2*(1+.9+.81)))
    torch.testing.assert_close(result.total_value, result.discounted_reward+(.9**3)*4)
    assert result.predicted_cost.shape == (2, 3) and not result.cost_validated
    assert "world_model_rollout_ms" in result.timings and "cost_prediction_ms" in result.timings
    changed_terminal = actions.clone()
    changed_terminal[:, -1] = 1
    changed = evaluator.evaluate_mppi_candidates(torch.zeros(1, 3), changed_terminal)
    torch.testing.assert_close(changed.cumulative_cost, result.cumulative_cost)


def test_predicted_termination_does_not_hide_future_cost():
    evaluator = rollout_evaluator()
    evaluator.agent.model._termination = True
    evaluator.agent.model.termination = lambda z, task: torch.ones(len(z), 1)
    result = evaluator.evaluate_mppi_candidates(torch.zeros(1, 3), torch.full((1, 4, 2), .4))
    assert result.total_value.item() == 2
    assert result.cumulative_cost.item() == pytest.approx(.7)


@pytest.mark.parametrize("horizon", [1, 5, 10, 15, 20])
def test_cost_diagnostics_validate_each_horizon_on_real_labels(horizon):
    validator = CostModelValidator(dict(min_samples=1, min_updates=1, horizons=[1, horizon]))
    controls = np.stack((np.linspace(0, 1, 40), np.zeros(40)), 1).astype("float32")
    validator.set_episodes([(np.zeros((40, 3)), controls, controls[:, 0])])
    model = ExactCostModel()
    metrics = validator.validate(model, "cpu", horizon=horizon)
    assert metrics[f"wm/cost_mae_h{horizon}"] < 1e-6
    assert metrics[f"wm/cost_correlation_h{horizon}"] > .99
    assert np.isnan(metrics["risk_ranking_accuracy"])
    assert model._cost_validated_horizons & (1 << horizon)
    model._cost_updates += 1
    assert not model.cost_ready


def test_candidate_ranking_uses_shared_origin_only_not_between_states():
    stats = cost_statistics([0, 1, 1, 0], [0, 1, 0, 1], groups=["a", "a", "b", "b"])
    assert stats["ranking_pairs"] == 2 and stats["ranking_accuracy"] == .5
    assert np.isnan(cost_statistics([0, 1], [0, 1])["ranking_accuracy"])
    validator = CostModelValidator({"min_samples": 1, "min_updates": 1, "horizons": [1]})
    validator.set_candidates([dict(executed=True, origin_id=1, observation=np.zeros(3),
        actions=np.tile([x, 0], (3, 1)), costs=np.full(3, x)) for x in [.1, .4, .8]])
    result = validator.validate(ExactCostModel(), "cpu", horizon=3)
    assert result["risk_ranking_accuracy"] == 1 and result["wm/cost_ranking_pairs"] == 3
    with pytest.raises(ValueError, match="same observation"):
        validator.set_candidates([dict(executed=True, origin_id=1, observation=np.zeros(3), actions=[], costs=[]),
                                  dict(executed=True, origin_id=1, observation=np.ones(3), actions=[], costs=[])])
    with pytest.raises(ValueError, match="executed"):
        validator.set_candidates([dict(origin_id=1, observation=np.zeros(3))])


@pytest.mark.parametrize("feature_enabled", [False, True])
def test_mppi_cost_does_not_unmask_existing_sac_coarse_cost_input(feature_enabled):
    from mpr_mpc.planning.coordinator import MPRMPCPlanner
    from mpr_mpc.planning.types import TrajectoryConsequence
    from mpr_mpc.residual_rl import ResidualStateBuilder
    from tests.test_mpr_mpc import make_path
    planner = object.__new__(MPRMPCPlanner)
    planner.device, planner.horizon, planner.control_dt, planner.safety = torch.device("cpu"), 3, .1, None
    path = make_path()
    selection = SimpleNamespace(coarse_path=path, feasible=[path], candidates=[path], used_fallback=False)
    planner.structured = SimpleNamespace(select_coarse=lambda vehicle: selection,
                                        planner=SimpleNamespace(last_lane_width=3.6))
    planner.action_adapter = SimpleNamespace(path_to_actions=lambda *args, **kwargs: torch.zeros(4, 2))
    consequence = TrajectoryConsequence(torch.zeros(1, 3), torch.zeros(1), torch.zeros(1),
        torch.zeros(1), torch.zeros(1, 512), cumulative_cost=torch.tensor([2.]), cost_validated=True)
    planner.evaluator = SimpleNamespace(encode=lambda obs: torch.zeros(1, 512),
        evaluate_trajectory_consequence=lambda *args: consequence)
    planner.residual_rl_cfg = {"coarse_cost_feature": feature_enabled}
    planner.residual_state_builder = ResidualStateBuilder({"risk_features": [
        "cumulative_cost", "min_ttc", "min_clearance", "boundary_margin"]}, latent_dim=512, device="cpu")
    ctx = planner.prepare(torch.zeros(259), SimpleNamespace(enable_reverse=False))
    assert ctx.residual_state.shape == (535,)
    assert ctx.residual_state[528].item() == float(feature_enabled)
    assert ctx.residual_state[527].item() == pytest.approx(.2 if feature_enabled else 0.)
    assert ctx.coarse_consequence.cost_validated  # MPPI diagnostics keep the actual validation status.
    planner.safety = SimpleNamespace(context=lambda *args, **kwargs: pytest.fail("Stage A must not construct SAC geometry"))
    cold = planner.prepare(torch.zeros(259), SimpleNamespace(enable_reverse=False), need_world_model_features=False)
    assert cold.coarse_consequence is None and cold.residual_state is None


def test_episode_cost_and_timeout_diagnostics_do_not_sum_overlapping_forecasts():
    from mpr_mpc.evaluation_diagnostics import EpisodeDiagnostics
    diagnostic = EpisodeDiagnostics()
    result = SimpleNamespace(stage=SimpleNamespace(use_mppi=True, use_residual=True), metrics={
        "mppi/predicted_step_cost": .2, "mppi/predicted_cost": 5., "mpr/mppi_predicted_cost": 5.})
    for _ in range(3):
        diagnostic.observe(np.zeros(3), np.zeros(2), {"cost": .3}, result)
    summary = diagnostic.summary(dict(truncated=True, terminated=False, end_reason="max_step"))
    assert summary["total_predicted_cost"] == pytest.approx(.6)
    assert summary["total_actual_cost"] == pytest.approx(.9)
    assert summary["timeout_rate"] == 1 and summary["terminated"] == 0 and summary["episode_length"] == 3
