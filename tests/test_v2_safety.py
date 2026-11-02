from types import SimpleNamespace

import numpy as np
import pytest
import torch
from shapely.geometry import box
from shapely.ops import unary_union

from mpr_mpc.planning.safety import SafetyContext, TrajectorySafetyEvaluator
from mpr_mpc.planning.cost_validation import CostModelValidator
from mpr_mpc.planning.local_mppi import LocalMPPI
from mpr_mpc.residual_rl import ResidualReplayBuffer, ResidualSACAgent, ResidualStateBuilder
from tests.test_mpr_mpc import make_path, mppi_cfg, RecordingEvaluator


def context(*, obstacles=(), speed=5, road=None):
    return SafetyContext(np.array([10., 10.]), 0., speed, 4., 2., 2.5, 0.6,
                         np.zeros(2), box(0, 5, 50, 15) if road is None else road, obstacles)


def obstacle(x, y, vx=0., vy=0., heading=0.):
    return SimpleNamespace(position=np.array([x, y]), velocity=np.array([vx, vy]),
                           heading=heading, length=4., width=2.)


def test_live_pose_and_terminal_action_alignment():
    evaluator = TrajectorySafetyEvaluator({}, control_dt=.1, device="cpu")
    actions = torch.zeros(2, 11, 2)
    actions[1, -1] = 1
    result = evaluator.check(actions, context())
    assert result.valid.all()
    torch.testing.assert_close(result.positions[0], result.positions[1])
    torch.testing.assert_close(result.positions[0, 0], torch.tensor([10., 10.]))
    assert result.positions[0, -1, 0] == pytest.approx(15, abs=1e-4)


def test_static_footprint_collision_and_clearance():
    evaluator = TrajectorySafetyEvaluator({}, control_dt=.1, device="cpu")
    actions = torch.zeros(2, 11, 2)
    result = evaluator.check(actions, context(obstacles=(obstacle(17, 10),)))
    assert result.collision.all()
    assert not result.valid.any()
    assert (result.min_clearance == 0).all()
    far = evaluator.check(actions, context(obstacles=(obstacle(17, 14),)))
    assert not far.collision.any()
    torch.testing.assert_close(far.min_clearance, torch.full((2,), 2.))


def test_time_aligned_crossing_and_ttc():
    evaluator = TrajectorySafetyEvaluator({}, control_dt=.1, device="cpu")
    crossing = context(obstacles=(obstacle(15, 15, 0, -5, np.pi/2),))
    result = evaluator.check(torch.zeros(3, 11, 2), crossing)
    assert result.collision.all()
    torch.testing.assert_close(result.min_ttc, torch.full((3,), .4))
    stopped = evaluator.check(torch.zeros(1, 11, 2), context(speed=0, obstacles=crossing.obstacles))
    assert not stopped.collision.any()


def test_lane_union_has_no_internal_boundary():
    evaluator = TrajectorySafetyEvaluator({}, control_dt=.1, device="cpu")
    road = unary_union([box(0, 5, 50, 10), box(0, 10, 50, 15)])
    result = evaluator.check(torch.zeros(1, 11, 2), context(road=road))
    assert result.valid.item()
    assert result.boundary_margin.item() == pytest.approx(4)


def test_navigation_road_join_includes_predecessor_for_rear_footprint():
    evaluator = TrajectorySafetyEvaluator({}, control_dt=.1, device="cpu")
    previous = SimpleNamespace(index=("spawn", "a", 0), shapely_polygon=box(0, 5, 10, 15))
    current = SimpleNamespace(index=("a", "b", 0), shapely_polygon=box(10, 5, 40, 15))
    navigation = SimpleNamespace(current_ref_lanes=[current], next_ref_lanes=[],
        map=SimpleNamespace(road_network=SimpleNamespace(graph={"spawn": {"a": [previous]}, "a": {"b": [current]}})))
    vehicle = SimpleNamespace(navigation=navigation, lane=current, position=np.array([10.5, 10.]),
        heading_theta=0., velocity=np.array([5., 0.]), current_action=[0., 0.], LENGTH=4.)
    planner = SimpleNamespace(_navigation_lane_groups=lambda nav: [[current]],
                              _reference_lane_sequence=lambda v: [current], _collect_obstacles=lambda v: [])
    ctx = evaluator.context(planner, vehicle, width=2., wheelbase=2.5, max_steering=.6)
    assert evaluator.check(torch.zeros(1, 11, 2), ctx).valid.item()


def test_only_validated_cost_changes_soft_ranking():
    class Evaluator:
        ready = False

        def evaluate_mppi_candidates(self, z, actions, task=None, mode_ids=None):
            value = actions[:, :, 1].mean(-1)
            return SimpleNamespace(total_value=value, terminal_q_std=torch.zeros_like(value),
                cumulative_cost=100 * value.clamp_min(0), cost_validated=self.ready,
                cost_validated_mask=torch.full_like(value, self.ready, dtype=torch.bool))

    evaluator = Evaluator()
    planner = LocalMPPI(evaluator, mppi_cfg(num_samples=4, iterations=1, num_elites=2, cost_coef=10),
                        horizon=10, action_dim=2, device="cpu")
    candidates = torch.zeros(4, 11, 2)
    candidates[1, :, 1] = 1
    candidates[2, :, 1] = .5
    candidates[3, :, 1] = -1
    planner._candidate_batch = lambda *args: candidates.clone()
    result = planner.plan(torch.zeros(1, 8), candidates[0], eval_mode=True)
    assert result.action[1] == 1
    evaluator.ready = True
    result = planner.plan(torch.zeros(1, 8), candidates[0], eval_mode=True)
    assert result.action[1] == 0


def test_high_value_unsafe_baseline_is_not_evaluated_or_selected():
    evaluator = RecordingEvaluator()
    planner = LocalMPPI(evaluator, mppi_cfg(multimodal=True, num_samples=64),
                        horizon=10, action_dim=2, device="cpu")
    planner.safety_evaluator = TrajectorySafetyEvaluator({}, control_dt=.1, device="cpu")
    baseline = torch.zeros(11, 2)
    baseline[:, 1] = 1
    ctx = context(obstacles=(obstacle(20, 10),))
    assert not planner.safety_evaluator.check(baseline[None], ctx).valid.item()
    result = planner.plan(torch.zeros(1, 8), baseline, eval_mode=True, safety_context=ctx)
    assert not result.baseline_selected
    assert result.safety_metrics["selected_safe"] == 1
    assert result.action[1] < 0
    assert all(planner.safety_evaluator.check(batch, ctx).valid.all() for batch in evaluator.calls)


def test_no_safe_candidate_is_explicit_and_never_certified():
    evaluator = RecordingEvaluator()
    planner = LocalMPPI(evaluator, mppi_cfg(num_samples=8), horizon=10, action_dim=2, device="cpu")
    planner.safety_evaluator = TrajectorySafetyEvaluator({}, control_dt=.1, device="cpu")
    result = planner.plan(torch.zeros(1, 8), torch.zeros(11, 2), eval_mode=True,
                          safety_context=context(obstacles=(obstacle(10, 10),)))
    assert result.safety_metrics["no_safe_candidate"] == 1
    assert result.safety_metrics["selected_safe"] == 0
    assert not result.baseline_selected
    assert torch.isfinite(result.action).all()
    assert evaluator.calls == []


def test_risk_features_are_masked_until_validated():
    builder = ResidualStateBuilder({"risk_features": ["cumulative_cost", "min_ttc", "min_clearance", "boundary_margin"]},
                                   latent_dim=512, device="cpu")
    consequence = SimpleNamespace(total_value=torch.tensor([10.]), cumulative_cost=torch.tensor([999.]),
                                  cost_validated=False, risk_summary={"boundary_margin": 2.})
    state = builder.build(latent=torch.zeros(512), coarse_path=make_path(), consequence=consequence,
                          feasible_ratio=1, used_fallback=False)
    assert builder.state_dim == 535
    assert state[527:529].tolist() == [0, 0]
    assert state[-2:].tolist() == [.5, 1]


def test_requested_and_executed_actions_are_distinct_in_replay():
    replay = ResidualReplayBuffer(4, 3)
    replay.add(np.zeros(3), [.2, .3], 1., .4, np.ones(3), False,
               diagnostics={"executed_action": [-.1, -1.], "sac_predicted_cost": .8, "sac_risk_valid": 1})
    batch = replay.sample(1, device="cpu", gamma=.99)
    torch.testing.assert_close(batch["action"], torch.tensor([[.2, .3]]))
    torch.testing.assert_close(batch["executed_action"], torch.tensor([[-.1, -1.]]))
    assert batch["cost"].item() == pytest.approx(.4)


class ExactCostModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("_cost_updates", torch.tensor(2000))
        self.register_buffer("_cost_validated", torch.tensor(False))
        self.register_buffer("_cost_validated_horizons", torch.tensor(0))

    def encode(self, x, task):
        return x

    def next(self, z, action, task):
        return z + 1

    def cost(self, z, action, task):
        return action[:, :1].abs()


def test_cost_validation_uses_actual_executed_controls_and_horizon_mask():
    model = ExactCostModel()
    actions = np.stack([np.linspace(0, 1, 80), np.zeros(80)], 1).astype("float32")
    validator = CostModelValidator({"min_samples": 8, "stride": 1, "horizons": [1, 5, 10], "max_mae": .001})
    validator.set_episodes([(np.zeros((80, 3)), actions, actions[:, 0])])
    metrics = validator.validate(model, "cpu", horizon=10)
    # Unlabelled executed sequences cannot authorize any counterfactual mode.
    assert metrics["wm/cost_ready"] == 0
    assert not model._cost_validated_horizons.item() & (1 << 10)
    assert not model._cost_validated_horizons.item() & (1 << 20)
    assert metrics["wm/cost_mae_h10"] < 1e-5
    model._cost_updates.zero_()
    assert validator.validate(model, "cpu", horizon=10)["wm/cost_ready"] == 0


def test_cost_constraint_discount_alignment_and_auxiliary_checkpoint():
    agent = ResidualSACAgent(3, {"hidden_dims": [8], "constraint_alignment": True,
        "constraint_mode": "episode", "risk_auxiliary_weight": .1}, device="cpu")
    assert agent.cost_gamma == 1
    other = ResidualSACAgent(3, {"hidden_dims": [8], "constraint_alignment": True,
        "constraint_mode": "discounted_episode", "risk_auxiliary_weight": .1}, device="cpu")
    assert other.cost_gamma == .99
    other.load_state_dict(agent.state_dict())
    with pytest.raises(ValueError, match="constraint_mode"):
        other.load_state_dict(agent.state_dict(), load_optimizer=True)
