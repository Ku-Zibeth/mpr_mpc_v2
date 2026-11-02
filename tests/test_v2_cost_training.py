from pathlib import Path
from copy import deepcopy

import pytest
import torch
from omegaconf import OmegaConf

from mpr_mpc._bootstrap import PACKAGE_ROOT, bootstrap
from mpr_mpc.planning.evaluator import TrajectoryWorldModelEvaluator
from mpr_mpc.planning.cost_validation import CostModelValidator

bootstrap()
from common.parser import parse_cfg
from mpr_mpc.tdmpc2.core import MPRTDMPC2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Vendored TD-MPC2 learner requires CUDA")
def test_environment_cost_supervision_and_legacy_readiness(tmp_path: Path):
    raw = OmegaConf.load(PACKAGE_ROOT / "config.yaml")
    raw.log_root = str(tmp_path)
    raw.batch_size = 4
    raw.horizon = 3
    raw.obs_shape = {"state": [259]}
    raw.action_dim = 2
    raw.episode_length = 100
    raw.enable_wandb = False
    cfg = parse_cfg(raw)
    learner = MPRTDMPC2(cfg)
    device, h, b = learner.device, cfg.horizon, cfg.batch_size
    observations = torch.randn(h + 1, b, 259, device=device)
    actions = torch.rand(h, b, 2, device=device) * 2 - 1
    costs = torch.linspace(0, 1, h * b, device=device).reshape(h, b, 1)
    before = [p.detach().clone() for p in learner.model._cost.parameters()]
    info = learner._update(observations, actions, torch.ones_like(costs), torch.zeros_like(costs), cost=costs)
    assert torch.isfinite(info["cost_loss"]) and info["cost_loss"] > 0
    assert any(not torch.equal(a, b) for a, b in zip(before, learner.model._cost.parameters()))
    assert learner.model._cost_updates.item() == 1
    assert not learner.model.cost_ready
    validator = CostModelValidator({"horizons": [1, h], "stride": 1})
    validator.set_episodes([(observations[:-1, 0].cpu().numpy(), actions[:, 0].cpu().numpy(), costs[:, 0, 0].cpu().numpy())])
    metrics = validator.validate(learner.model, device, horizon=h)
    assert metrics[f"wm/cost_samples_h{h}"] == 1
    assert metrics["wm/cost_ready"] == 0
    evaluator = TrajectoryWorldModelEvaluator(learner, h)
    z = evaluator.encode(observations[0, :1])
    controls = torch.zeros(2, h + 1, 2, device=device)
    assert evaluator.evaluate_trajectory_consequence(z, controls).cumulative_cost is None
    learner.model._cost_validated.fill_(True)
    learner.model._cost_validated_horizons.fill_(1 << h)
    result = evaluator.evaluate_trajectory_consequence(z, controls)
    assert result.cost_validated
    expected = sum(cfg.cost["gamma"] ** t * result.predicted_cost[:, t] for t in range(h))
    torch.testing.assert_close(result.cumulative_cost, expected)
    assert (result.predicted_cost >= 0).all()
    evaluator.horizon = h + 1
    assert not evaluator.evaluate_trajectory_consequence(z, torch.zeros(1, h + 2, 2, device=device)).cost_validated
    legacy = {key: deepcopy(value) for key, value in learner.model.state_dict().items()
              if key not in {"_cost_updates", "_cost_validated", "_cost_validated_horizons"}}
    learner.load({"model": legacy})
    assert not learner.model.cost_ready
    assert learner.model._cost_updates.item() == 0
