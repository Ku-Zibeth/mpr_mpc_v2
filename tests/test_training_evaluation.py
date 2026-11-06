from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mpr_mpc.trainer import MPROnlineTrainer
from tests.test_evaluation_ablation import FakeAgent, FakeEnv


@pytest.mark.parametrize("step,use_residual,use_mppi", [
    (0, False, False),
    (500000, True, False),
    (1000000, True, True),
])
def test_training_eval_aggregates_numeric_diagnostics_not_end_reason(
    monkeypatch, step, use_residual, use_mppi
):
    class EvaluationEnv(FakeEnv):
        agent = None

        def reset(self, *, seed):
            seeds.append(seed)
            return super().reset(seed=seed)

        def step(self, action):
            obs, reward, done, info = super().step(action)
            info.update(end_reason="arrive_dest" if done else "", terminated=done,
                        truncated=False)
            return obs, reward, done, info

    class EvaluationAgent(FakeAgent):
        def act(self, obs, **kwargs):
            assert kwargs["global_step"] == step
            action = super().act(obs, **kwargs)
            self.last_plan = SimpleNamespace(
                stage=SimpleNamespace(use_residual=use_residual, use_mppi=use_mppi),
                metrics={"mpr/total_planning_ms": 3.0},
            )
            if use_residual:
                self.last_plan.metrics["mpr/sac_inference_cpu_wall_ms"] = 1.25
            if use_mppi:
                self.last_plan.metrics["mppi/predicted_step_cost"] = 0.5
            if len(self.executed) > 2:
                self.last_plan.metrics["mpr/late_metric"] = 7.0
            return action

        def validate_cost_model(self, episodes):
            assert len(episodes) == 2
            assert all(len(costs) == 2 for _, _, costs in episodes)
            return {"cost_validation_samples": 4.0}

    seeds = []
    trainer = object.__new__(MPROnlineTrainer)
    trainer.cfg = SimpleNamespace(eval_episodes=2, save_video=False)
    trainer.env = EvaluationEnv()
    trainer.agent = EvaluationAgent()
    trainer._step = step
    trainer._eval_scenarios = [11998, 11999]
    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", lambda: None)

    metrics = trainer.eval()

    assert seeds == [11998, 11999]
    assert "end_reason" not in metrics
    assert all(isinstance(value, (int, float, np.number)) for value in metrics.values())
    assert metrics["episode_reward"] == 2.0
    assert metrics["episode_cost"] == metrics["average_safety_cost"] == 6.0
    assert metrics["success_rate"] == metrics["collision_rate"] == 1.0
    assert metrics["out_of_road_rate"] == 1.0
    assert metrics["route_completion"] == pytest.approx(0.98)
    assert metrics["mean_ttc"] == 3.0
    assert metrics["min_ttc"] == 2.0
    assert metrics["terminated"] == 1.0
    assert metrics["truncated"] == metrics["timeout_rate"] == 0.0
    assert metrics["mean_mpr_total_planning_ms"] == 3.0
    assert metrics["mean_mpr_late_metric"] == 7.0
    assert np.isnan(metrics["min_obstacle_clearance"])
    assert metrics["cost_validation_samples"] == 4.0
    if use_residual:
        assert metrics["mean_mpr_sac_inference_cpu_wall_ms"] == 1.25
    else:
        assert "mean_mpr_sac_inference_cpu_wall_ms" not in metrics
    if use_mppi:
        assert metrics["total_predicted_cost"] == 1.0
    else:
        assert np.isnan(metrics["total_predicted_cost"])
