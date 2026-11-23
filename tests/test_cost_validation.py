"""Retained WM-cost and535-D compatibility tests; old planner variants removed."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from mpr_mpc.planning.cost_validation import CostModelValidator,cost_statistics
from mpr_mpc.planning.evaluator import TrajectoryWorldModelEvaluator
from tests.mppi_fixtures import ExactCostModel,make_path

class RolloutModel(ExactCostModel):
    _termination = None

    def reward(self, z, action, task):
        return torch.ones_like(action[:, :1])*float(np.log1p(2))

    def Q(self, z, action, task, return_type="all"):
        if return_type == "avg":
            return torch.full((len(action), 1), 4.)
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
    from tests.mppi_fixtures import make_path
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
