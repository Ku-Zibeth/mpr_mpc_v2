from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from shapely.geometry import box

from mpr_mpc.checkpoint_selection import select_checkpoint
from mpr_mpc.planning.cost_validation import CostModelValidator
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
