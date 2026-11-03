from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import shapely
from shapely.geometry import Polygon, box
import torch

from mpr_mpc.planning.calibration_anchor import make_anchor, verify_anchor
from mpr_mpc.planning.cost_validation import CostModelValidator, calibration_statistics
from mpr_mpc.planning.emergency import emergency_braking
from mpr_mpc.planning.local_mppi import LocalMPPI
from mpr_mpc.planning.safety import TrajectorySafetyEvaluator, _corners
from mpr_mpc.planning.swept_collision import swept_collisions
from mpr_mpc.residual_rl.state_builder import ResidualStateBuilder
from tests.test_mpr_mpc import mppi_cfg, RecordingEvaluator, make_path
from tests.test_round3 import ModeCostModel
from tests.test_v2_safety import context, obstacle


def paired_records(*, anchors=True, coarse=False):
    rows = []
    for scene in range(20):
        for origin in range(4):
            for mode in ([-1] if coarse else []) + list(range(6)):
                for candidate in range(2):
                    value = .02 + (mode+1)*.12 + candidate*.06 + origin*.005 + (scene%3)*.001
                    rows.append(dict(source="metadrive_mode_replay_v2", replay_verified=True, scene_id=scene,
                        anchor_id=f"s{scene}:t{origin}" if anchors else "", mode_id=mode,
                        trajectory_kind="coarse_lattice" if mode == -1 else "mppi_mode", observation=np.zeros(3),
                        actions=np.tile([value, 0.], (10, 1)).astype("float32"), costs=np.full(10, value)))
    return rows


def test_matched_anchor_cross_mode_calibration_passes_real_thresholds():
    validator = CostModelValidator({"horizons": [1, 5, 10, 20]})
    validator.set_mode_episodes(paired_records(coarse=True))
    model = ModeCostModel()
    validator.validate(model, "cpu", horizon=10)
    assert model._cost_validated_by_mode[:, [1, 5, 10]].all()
    assert model._cost_validated_coarse[[1, 5, 10]].all()
    assert not model._cost_validated_by_mode[:, 20].any()
    for name in validator.mode_report:
        stats = validator.mode_report[name][10]
        assert stats["within_mode_ranking"]["accuracy"] == 1
        assert stats["cross_mode_ranking"]["accuracy"] == 1
        assert stats["overall_ranking"]["ci_low"] == 1
        assert stats["cross_mode_ranking"]["anchors"] == 80
        assert stats["samples"] == 160 and stats["scenes"] == 20
    assert model._cost_validation_schema == 2


def test_between_scene_offset_predictor_cannot_pass_action_ranking():
    class SceneOnly(ModeCostModel):
        def cost(self, z, action, task):
            return z[:, :1]

        def next(self, z, action, task):
            return z

    rows = paired_records()
    for row in rows:
        offset = row["scene_id"] / 20
        row["observation"] = np.array([offset, 0., 0.])
        row["costs"] = np.full(10, offset + .06 * (row["mode_id"] % 2))
    validator = CostModelValidator({"horizons": [1, 10]})
    validator.set_mode_episodes(rows)
    model = SceneOnly()
    validator.validate(model, "cpu", horizon=10)
    assert not model.cost_ready
    assert validator.mode_report["nominal"][1]["mae"] < .1
    assert validator.mode_report["nominal"][1]["correlation"] > .99
    assert validator.mode_report["nominal"][1]["cross_mode_ranking"]["accuracy"] == 0


def test_missing_anchors_cannot_retain_or_gain_permissions():
    validator, model = CostModelValidator({}), ModeCostModel()
    validator.set_mode_episodes(paired_records())
    validator.validate(model, "cpu", horizon=10)
    assert model.cost_ready
    validator.set_mode_episodes(paired_records(anchors=False))
    validator.validate(model, "cpu", horizon=10)
    assert not model.cost_ready
    assert validator.mode_report["nominal"][10]["cross_mode_ranking"]["pairs"] == 0


def test_pairing_never_compares_different_prediction_origins():
    p, y, scenes, modes = [0., 1., 1., 0.], [0., 1., 0., 1.], [1]*4, [0, 1, 0, 1]
    stats = calibration_statistics(p, y, scenes, {}, anchors=["a", "a", "b", "b"], modes=modes)
    assert stats["overall_ranking"]["pairs"] == 2
    assert stats["cross_mode_ranking"]["accuracy"] == .5
    assert stats["within_mode_ranking"]["pairs"] == 0


def test_bootstrap_resamples_scenes_not_candidate_rows():
    scenes = np.repeat(np.arange(20), 4)
    anchors = [f"s{s}" for s in scenes]
    modes = np.tile([0, 1, 2, 3], 20)
    y = np.tile([.1, .3, .5, .7], 20)
    p = y.copy()
    p[scenes < 4] = 1 - p[scenes < 4]
    first = calibration_statistics(p, y, scenes, {}, anchors=anchors, modes=modes)
    duplicate = calibration_statistics(np.tile(p, 2), np.tile(y, 2), np.tile(scenes, 2), {},
        anchors=anchors*2, modes=np.tile(modes, 2))
    assert first["cross_mode_ranking"]["ci_low"] == duplicate["cross_mode_ranking"]["ci_low"]
    assert first["cross_mode_ranking"]["ci_high"] == duplicate["cross_mode_ranking"]["ci_high"]


def test_anchor_content_checks_pose_traffic_history_and_prediction_start():
    args = [539, 540, [[0., 1.]], [1., 2.], [10., 0., 5., 1.], .1]
    anchor, payload = make_anchor(*args)
    verify_anchor(anchor, payload)
    for key in ("scene_id", "prediction_step", "fingerprint", "prefix_actions"):
        modified = deepcopy(payload)
        modified[key] = 999 if isinstance(modified[key], int) else []
        with pytest.raises(ValueError):
            verify_anchor(anchor, modified)


def test_same_anchor_cannot_mix_scenes_or_start_observations():
    rows = paired_records()[:2]
    rows[1]["observation"] = np.ones(3)
    with pytest.raises(ValueError, match="anchor_id"):
        CostModelValidator({}).set_mode_episodes(rows)


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


def test_unavoidable_emergency_is_fixed_braking_not_an_ordinary_candidate():
    planner = LocalMPPI(RecordingEvaluator(), mppi_cfg(num_samples=8), horizon=10, action_dim=2, device="cpu")
    planner.safety_evaluator = TrajectorySafetyEvaluator({}, control_dt=.1, device="cpu")
    ctx = context(obstacles=(obstacle(10, 10),))
    baseline = torch.ones(11, 2)*.9
    first = planner.plan(torch.zeros(1, 2), baseline, eval_mode=True, safety_context=ctx)
    second = planner.plan(torch.zeros(1, 2), baseline, eval_mode=True, safety_context=ctx)
    assert first.emergency_fallback and not first.emergency_verified
    assert first.action.tolist() == [0., -1.]
    assert first.safety_metrics["safe_candidate_selected"] == 0
    assert first.safety_metrics["selected_safe"] == 0
    assert first.safety_metrics["emergency_new_event"] == 1
    assert second.safety_metrics["emergency_new_event"] == 0
    assert second.safety_metrics["emergency_duration_s"] == .2
    planner.reset()
    assert planner.emergency_streak == 0


def test_independent_braking_can_be_verified_and_respects_steering_rate():
    safety = TrajectorySafetyEvaluator({}, control_dt=.1, device="cpu")
    ctx = context(speed=2)
    ctx.previous_action[0] = .8
    result = emergency_braking(torch.ones(11, 2), ctx, safety, control_dt=.1)
    assert result.verified
    assert (result.sequence[:, 1] == -1).all()
    assert abs(result.sequence[0, 0]-.8) * ctx.max_steering/.1 <= 2.5


def test_missing_global_geometry_never_certifies_emergency_control():
    result = emergency_braking(torch.ones(11, 2), None, None, control_dt=.1)
    assert not result.verified
    assert result.sequence[0].tolist() == [0., -1.]


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


def test_horizons_cannot_share_counterfactual_labels():
    rows = paired_records()
    for row in rows:
        if row["mode_id"] != 0:
            row["actions"], row["costs"] = row["actions"][:1], row["costs"][:1]
    validator, model = CostModelValidator({"horizons": [1, 10]}), ModeCostModel()
    validator.set_mode_episodes(rows)
    validator.validate(model, "cpu", horizon=10)
    assert model._cost_validated_by_mode[:, 1].all()
    assert not model._cost_validated_by_mode[:, 10].any()


def test_verified_emergency_is_distinct_from_ordinary_selection_and_logs_events():
    class BadValue:
        fail = True

        def evaluate_mppi_candidates(self, z, actions, task=None, mode_ids=None):
            return SimpleNamespace(total_value=torch.full((len(actions),), float("nan") if self.fail else 0.),
                                   terminal_q_std=torch.zeros(len(actions)))

    evaluator = BadValue()
    planner = LocalMPPI(evaluator, mppi_cfg(num_samples=8), horizon=10, action_dim=2, device="cpu")
    planner.safety_evaluator = TrajectorySafetyEvaluator({}, control_dt=.1, device="cpu")
    first = planner.plan(torch.zeros(1, 2), torch.zeros(11, 2), eval_mode=True, safety_context=context())
    assert first.emergency_fallback and first.emergency_verified
    assert first.safety_metrics["selected_safe"] == 1 and first.safety_metrics["safe_candidate_selected"] == 0
    assert first.emergency_reason == "nonfinite_scores" and first.action[1] == -1
    from mpr_mpc.evaluation_diagnostics import EpisodeDiagnostics
    diagnostics = EpisodeDiagnostics()
    result = SimpleNamespace(metrics={f"mpr/{k}": v for k, v in first.safety_metrics.items()},
                             stage=SimpleNamespace(use_residual=True, use_mppi=True))
    diagnostics.observe(torch.zeros(3), first.action, {"cost": .5}, result)
    summary = diagnostics.summary({})
    assert summary["emergency_events"] == 1 and summary["emergency_steps"] == 1
    assert summary["emergency_duration_s"] == .1 and summary["emergency_reason_4_steps"] == 1
    evaluator.fail = False
    normal = planner.plan(torch.zeros(1, 2), torch.zeros(11, 2), eval_mode=True, safety_context=context())
    assert not normal.emergency_fallback and planner.emergency_streak == 0
    evaluator.fail = True
    again = planner.plan(torch.zeros(1, 2), torch.zeros(11, 2), eval_mode=True, safety_context=context())
    assert again.safety_metrics["emergency_new_event"] == 1 and planner.emergency_count == 2


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
