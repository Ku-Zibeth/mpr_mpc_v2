from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from shapely.geometry import box

from mpr_mpc.checkpoint_selection import select_checkpoint
from mpr_mpc.planning.cost_validation import CostModelValidator, calibration_statistics
from mpr_mpc.planning.local_mppi import LocalMPPI
from mpr_mpc.planning.safety import TrajectorySafetyEvaluator
from tests.test_v2_safety import context, obstacle, ExactCostModel
from tests.test_mpr_mpc import mppi_cfg


@pytest.mark.parametrize("dt,substeps", [(.1, 1), (.1, 5), (.2, 5), (.05, 2)])
@pytest.mark.parametrize("objects,expected", [
    ((obstacle(10, 10),), 0.),
    ((obstacle(15, 15, 0, -5, np.pi/2),), .4),
    ((obstacle(20, 10, -5, 0),), .6),
    ((obstacle(20, 10, 10, 0),), np.inf),
    ((obstacle(20, 10),), 1.2),
    ((), np.inf),
    ((obstacle(20, 10), obstacle(15, 15, 0, -5, np.pi/2)), .4),
])
def test_absolute_ttc(dt, substeps, objects, expected):
    safety = TrajectorySafetyEvaluator({"substeps": substeps}, control_dt=dt, device="cpu")
    actual = safety.check(torch.zeros(1, int(1/dt)+1, 2), context(obstacles=objects)).min_ttc.item()
    assert actual == pytest.approx(expected, abs=1e-5)


def test_ttc_absolute_lookahead_not_local_horizon():
    safety = TrajectorySafetyEvaluator({"ttc_lookahead": .39}, control_dt=.1, device="cpu")
    actual = safety.check(torch.zeros(1, 11, 2), context(obstacles=(obstacle(15, 15, 0, -5, np.pi/2),))).min_ttc.item()
    assert np.isinf(actual)


def test_boundary_risk_continuity_and_signed_margin():
    evaluator = TrajectorySafetyEvaluator({"soft_boundary_margin": .5}, control_dt=.1, device="cpu")
    risks = []
    for y in [10, 13.6, 13.7, 13.8, 6.3, 14.2]:
        ctx = context(speed=0)
        ctx.position[1] = y
        result = evaluator.check(torch.zeros(1, 11, 2), ctx)
        risks.append(result.boundary_risk.item())
        assert result.offroad.item() == (y == 14.2)
        assert (result.boundary_margin.item() < 0) == (y == 14.2)
    assert risks[:4] == pytest.approx([0, .2, .4, .6], abs=1e-5)
    assert risks[4] == pytest.approx(risks[2], abs=1e-5)
    assert risks[5] > 1


class ZeroValue:
    def evaluate_mppi_candidates(self, z, actions, task=None, mode_ids=None):
        value = torch.zeros(len(actions))
        return SimpleNamespace(total_value=value, terminal_q_std=value)


def test_boundary_penalty_really_changes_mppi_selection():
    candidates = torch.zeros(4, 11, 2)
    candidates[0, :, 0] = .15
    candidates[1:, :, 0] = -.1
    ctx = context()
    ctx.position[1] = 12.5
    selections = []
    for weight in [0, 1]:
        planner = LocalMPPI(ZeroValue(), mppi_cfg(num_samples=4, num_elites=2, iterations=1,
            deviation_coef=0, smoothness_coef=0, geometry_coef=100), horizon=10, action_dim=2, device="cpu")
        planner._candidate_batch = lambda *args: candidates.clone()
        planner.safety_evaluator = TrajectorySafetyEvaluator({"soft_boundary_margin": 2., "boundary_risk_weight": weight}, control_dt=.1, device="cpu")
        result = planner.plan(torch.zeros(1, 2), candidates[0], eval_mode=True, safety_context=ctx)
        selections.append(float(result.action[0]))
        assert result.safety_metrics["selected_safe"] == 1
    assert selections == pytest.approx([.15, -.1])


class ModeCostModel(ExactCostModel):
    def __init__(self):
        super().__init__()
        self.register_buffer("_cost_validated_by_mode", torch.zeros(6, 63, dtype=torch.bool))
        self.register_buffer("_cost_validated_by_horizon", torch.zeros(63, dtype=torch.bool))
        self.register_buffer("_cost_validation_samples_by_mode", torch.zeros(6, 63, dtype=torch.long))
        self.register_buffer("_cost_validation_scenes_by_mode", torch.zeros(6, 63, dtype=torch.long))
        self.register_buffer("_cost_mode_validated_at", torch.tensor(0))
        self.register_buffer("_cost_validation_schema", torch.tensor(0))
        self.register_buffer("_cost_validated_coarse", torch.zeros(63, dtype=torch.bool))

    @property
    def cost_ready(self):
        return (bool(self._cost_validated_by_mode.any() | self._cost_validated_coarse.any())
                and self._cost_mode_validated_at == self._cost_updates and self._cost_validation_schema == 2)

    def invalidate_cost_validation(self):
        for name, value in self.named_buffers():
            if name != "_cost_updates":
                value.zero_()


def mode_records():
    records = []
    for mode in [0, 1, 2]:
        for i in range(160):
            value = (.01, .4, .8)[mode] + (i%2)*.06 + ((i%8)//2)*.005 + ((i//8)%3)*.001
            actions = np.tile([value, 0.], (10, 1)).astype("float32")
            records.append(dict(source="metadrive_mode_replay_v2", replay_verified=True,
                scene_id=i//8, anchor_id=f"{i//8}:{(i%8)//2}", mode_id=mode, observation=np.zeros(3), actions=actions,
                costs=np.full(10, value if mode == 0 else (1-value if mode == 1 else .2))))
    return records


def test_mode_calibration_independence_scene_bootstrap_and_horizons():
    validator = CostModelValidator({"horizons": [1, 5, 10, 20]})
    validator.set_mode_episodes(mode_records())
    model = ModeCostModel()
    metrics = validator.validate(model, "cpu", horizon=10)
    assert model._cost_validated_by_mode[0, [1, 5, 10]].all()
    assert not model._cost_validated_by_mode[0, 20]
    assert not model._cost_validated_by_mode[1:].any()
    assert metrics["wm/cost_nominal_scenes_h10"] == 20
    assert metrics["wm/cost_nominal_samples_h10"] == 160
    assert validator.mode_report["nominal"][10]["ranking_ci_low"] == 1
    assert not validator.mode_report["right_lateral"][10]["identifiable"]
    clone = ModeCostModel()
    clone.load_state_dict(deepcopy(model.state_dict()))
    assert torch.equal(clone._cost_validated_by_mode, model._cost_validated_by_mode)
    validator.set_mode_episodes(mode_records()[:40])
    validator.validate(model, "cpu", horizon=10)
    assert not model._cost_validated_by_mode.any()


def test_mode_labels_must_come_from_verified_replays():
    validator = CostModelValidator({})
    rows = mode_records()[:1]
    rows[0]["replay_verified"] = False
    with pytest.raises(ValueError, match="verified"):
        validator.set_mode_episodes(rows)


def test_cluster_bootstrap_rejects_constant_or_single_scene_ranking():
    for labels, scenes in [(np.ones(128), np.arange(128)), (np.arange(128), np.zeros(128))]:
        result = calibration_statistics(np.arange(128), labels, scenes, {})
        assert not result["identifiable"]
        assert not np.isfinite(result["ranking_ci_low"])


def test_safety_first_and_progress_feasibility():
    base = dict(collision_rate=.1, out_of_road_rate=.1, average_safety_cost=1., success_rate=.8,
                route_completion=.9, episode_reward=100.)
    yes, previous, _ = select_checkpoint(base, None, {})
    assert yes
    for field in ["collision_rate", "out_of_road_rate", "average_safety_cost"]:
        worse = dict(base, episode_reward=10000)
        worse[field] += .01
        assert not select_checkpoint(worse, previous, {})[0]
    assert select_checkpoint(dict(base, collision_rate=.09, episode_reward=-100), previous, {})[0]
    assert not select_checkpoint(dict(base, success_rate=0, route_completion=0, collision_rate=0), previous, {})[0]
    assert not select_checkpoint(dict(base, success_rate=.5, collision_rate=0), previous, {})[0]
    better, intermediate, _ = select_checkpoint(dict(base, success_rate=.71, collision_rate=.09), previous, {})
    assert better
    # A sequence of small safety improvements must not gradually erase throughput.
    assert not select_checkpoint(dict(base, success_rate=.62, collision_rate=.08), intermediate, {})[0]
    for value in [float("nan"), float("inf")]:
        assert not select_checkpoint(dict(base, collision_rate=value), previous, {})[0]
    del base["collision_rate"]
    assert not select_checkpoint(base, previous, {})[0]


def test_best_and_latest_checkpoint_save_contract():
    from mpr_mpc.trainer import MPROnlineTrainer
    trainer = object.__new__(MPROnlineTrainer)
    base = dict(collision_rate=.1, out_of_road_rate=.1, average_safety_cost=1, success_rate=.8,
                route_completion=.9, episode_reward=100)
    trainer.eval = lambda: dict(base)
    trainer.common_metrics = lambda: {}
    trainer.logger = SimpleNamespace(log=lambda *a: None)
    trainer.cfg = SimpleNamespace(checkpoint_selection={})
    trainer.agent = SimpleNamespace(best_eval_selection=None)
    saved = []
    trainer._save_checkpoint = saved.append
    trainer._handle_eval()
    assert saved == ["best", "latest"]
    saved.clear()
    trainer._handle_eval()
    assert saved == ["latest"]


def test_default_training_schedule_and_six_profiles():
    from hydra import compose, initialize_config_dir
    from mpr_mpc._bootstrap import PACKAGE_ROOT
    for profile in ["E0", "E1", "E2", "E3", "E4", "E5"]:
        with initialize_config_dir(version_base="1.3", config_dir=str(PACKAGE_ROOT)):
            cfg = compose(config_name="config", overrides=[f"+experiment={profile}"])
        assert cfg.steps == 50000
        assert cfg.mpr_mpc.stages.residual_start_step == 20000
        assert cfg.mpr_mpc.stages.mppi_start_step == 50000
        assert cfg.residual_rl.freeze_tdmpc2_on_start


def test_collector_rejects_training_overlap_or_unproven_old_checkpoints():
    from tools.calibrate_cost_modes import assert_heldout
    with pytest.raises(ValueError, match="provenance"):
        assert_heldout([101], {})
    metadata = dict(training_scene_range=dict(start_seed=100, num_scenarios=50), validation_scene_ids=[149])
    with pytest.raises(ValueError, match="overlap"):
        assert_heldout([101], metadata)
    assert_heldout([149, 200], metadata)


def test_mppi_real_component_ids_gate_cost_without_rejecting_other_modes():
    class Evaluator:
        approved = False
        seen = None

        def evaluate_mppi_candidates(self, z, actions, task=None, mode_ids=None):
            self.seen = mode_ids.clone()
            brake = mode_ids == 1
            mask = brake & self.approved
            return SimpleNamespace(total_value=brake.float(), terminal_q_std=torch.zeros(len(actions)),
                cost_validated_mask=mask, cumulative_cost=torch.where(mask, 100., float("nan")))

    evaluator = Evaluator()
    planner = LocalMPPI(evaluator, mppi_cfg(multimodal=True, iterations=1, num_samples=64),
                        horizon=10, action_dim=2, device="cpu")
    baseline = torch.zeros(11, 2)
    before = planner.plan(torch.zeros(1, 3), baseline, eval_mode=True)
    assert before.selected_mode_id == 1
    assert set(evaluator.seen.tolist()) == set(range(6))
    evaluator.approved = True
    after = planner.plan(torch.zeros(1, 3), baseline, eval_mode=True)
    assert after.selected_mode_id != 1
    assert 0 < after.safety_metrics["cost_validated_ratio"] < 1
    assert torch.isfinite(after.selected_score)


def test_dataset_checksum_cost_definition_and_heldout_provenance(tmp_path):
    import hashlib
    import json
    path = tmp_path / "mode_replays.npz"
    np.savez_compressed(path, source="metadrive_mode_replay_v1", observations=np.zeros((1, 3)),
        actions=np.zeros((1, 10, 2)), costs=np.zeros((1, 10)), lengths=[10], modes=[0], scenes=[539], verified=[True])
    manifest = dict(dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), cost_definition={"gamma": .99},
                    scenes=[539], training_provenance=dict(start_seed=100, num_scenarios=3, reserved=[]))
    metadata = tmp_path / "calibration.json"
    metadata.write_text(json.dumps(manifest))
    validator = CostModelValidator({})
    validator.load_dataset(path, cost_definition={"gamma": .99})
    assert len(validator.mode_episodes) == 1
    with pytest.raises(ValueError, match="cost definition"):
        validator.load_dataset(path, cost_definition={"gamma": .9})
    manifest["dataset_sha256"] = "wrong"
    metadata.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checksum"):
        validator.load_dataset(path)


def test_safety_cache_reuses_only_static_road_and_current_context_snapshot():
    safety = TrajectorySafetyEvaluator({}, control_dt=.1, device="cpu")
    road = box(0, 5, 50, 15)
    first = context(road=road, obstacles=(obstacle(17, 10),))
    actions = torch.zeros(2, 11, 2)
    original = safety.check(actions, first)
    repeated = safety.check(actions, first)
    for key in ["valid", "min_clearance", "min_ttc", "boundary_margin", "soft_risk"]:
        torch.testing.assert_close(getattr(original, key), getattr(repeated, key))
    next_context = context(road=road, obstacles=(obstacle(40, 10),))
    updated = safety.check(actions, next_context)
    assert original.collision.all() and not updated.collision.any()
    assert first.cache is not next_context.cache
    assert len(safety._road_geometry_cache) == 1
