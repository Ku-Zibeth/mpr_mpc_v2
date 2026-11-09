from types import SimpleNamespace
from unittest.mock import Mock
import json

import numpy as np
import pytest
import torch

from mpr_mpc.trainer import MPROnlineTrainer
from common.logger import Logger


def make_logger(tmp_path, *, artifacts=True):
    logger = object.__new__(Logger)
    logger._model_dir = tmp_path
    logger._metrics_path = tmp_path / "metrics.jsonl"
    logger._wandb_metrics_enabled = True
    logger._video = None
    logger._save_agent = True
    logger._artifact_logging_enabled = artifacts
    logger._group, logger._seed = "test", 1
    logger._wandb = Mock()
    logger._save_csv = False
    logger._print = Mock()
    agent = SimpleNamespace(save=lambda path: torch.save({"global_env_step": 510325}, path))
    return logger, agent


@pytest.mark.parametrize("failure", ["Artifact", "add_file", "log_artifact"])
def test_artifact_failure_keeps_local_checkpoints_and_metric_logging(tmp_path, capsys, failure):
    logger, agent = make_logger(tmp_path)
    if failure == "Artifact":
        target = logger._wandb.Artifact
    elif failure == "add_file":
        target = logger._wandb.Artifact.return_value.add_file
    else:
        target = logger._wandb.log_artifact

    def fail(*args, **kwargs):
        assert (tmp_path / "latest.pt").is_file()
        raise RuntimeError("403 Forbidden")

    target.side_effect = fail
    logger.save_agent(agent, "latest")
    assert torch.load(tmp_path / "latest.pt", weights_only=False)["global_env_step"] == 510325
    assert not logger._artifact_logging_enabled
    logger.save_agent(agent, "final")
    assert (tmp_path / "final.pt").is_file()
    assert target.call_count == 1
    logger.log({"step": 510326, "episode_reward": 1.0})
    logger._wandb.log.assert_called_once()
    assert "Local checkpoint is saved" in capsys.readouterr().out


def test_artifacts_can_be_disabled_without_disabling_local_saves(tmp_path):
    logger, agent = make_logger(tmp_path, artifacts=False)
    logger.save_agent(agent, "latest")
    assert (tmp_path / "latest.pt").is_file()
    logger._wandb.Artifact.assert_not_called()
    logger._wandb.log_artifact.assert_not_called()


def test_local_checkpoint_write_failure_is_not_hidden(tmp_path):
    logger, agent = make_logger(tmp_path)
    agent.save = Mock(side_effect=OSError("disk full"))
    with pytest.raises(OSError, match="disk full"):
        logger.save_agent(agent, "latest")
    logger._wandb.Artifact.assert_not_called()


def test_failed_checkpoint_write_preserves_previous_checkpoint(tmp_path):
    logger, agent = make_logger(tmp_path)
    torch.save({"global_env_step": 500000}, tmp_path / "latest.pt")

    def fail(path):
        path.write_bytes(b"incomplete checkpoint")
        raise OSError("disk full")

    agent.save = fail
    with pytest.raises(OSError, match="disk full"):
        logger.save_agent(agent, "latest")
    assert torch.load(tmp_path / "latest.pt", weights_only=False)["global_env_step"] == 500000


def test_metric_upload_failure_does_not_stop_training_or_local_metrics(tmp_path):
    logger, _ = make_logger(tmp_path)
    logger._wandb.log.side_effect = RuntimeError("403 Forbidden")
    for step in (500001, 500002):
        logger.log({"step": step, "reward": torch.tensor(2.), "cost": np.float32(.5),
                    "unavailable": float("nan"), "Reason": "arrive_dest"})
    assert logger._wandb.log.call_count == 1
    rows = [json.loads(line) for line in logger._metrics_path.read_text().splitlines()]
    assert [row["step"] for row in rows] == [500001, 500002]
    assert all(row["reward"] == 2 and row["cost"] == .5 and row["unavailable"] is None for row in rows)


def test_final_cloud_sync_failure_does_not_lose_final_model(tmp_path):
    logger, agent = make_logger(tmp_path, artifacts=False)
    logger._wandb.finish.side_effect = RuntimeError("network down")
    logger.finish(agent)
    assert (tmp_path / "final.pt").is_file()


@pytest.mark.parametrize("start", [0, 510325])
def test_training_starts_a_new_episode_after_resume_before_replay_add(start):
    trainer = object.__new__(MPROnlineTrainer)
    trainer.cfg = SimpleNamespace(steps=start + 3, eval_freq=10000,
                                  batch_size=1, horizon=1, episodic=True)
    trainer._step = start
    trainer._checkpoint_freq = 50000
    trainer._replay_steps = 0
    trainer._training_scenarios = np.array([10001])
    trainer._scenario_rng = np.random.default_rng(1)
    trainer._tdmpc_update_gate = lambda *args: (False, 0.0, 0)
    trainer._maybe_save_milestones = lambda *args: {}
    trainer._handle_eval = Mock()
    trainer.common_metrics = lambda: {"step": trainer._step}
    trainer.logger = SimpleNamespace(log=Mock(), finish=Mock())
    trainer.buffer = SimpleNamespace(add=Mock(return_value=1))
    resets, action_calls = [], []
    episode_step = 0

    def reset(*, seed):
        nonlocal episode_step
        episode_step = 0
        resets.append(seed)
        return torch.zeros(3)

    def step(action):
        nonlocal episode_step
        episode_step += 1
        done = episode_step == 2
        return torch.zeros(3), torch.tensor(1.), done, {
            "terminated": torch.tensor(done), "success": float(done), "cost": .1,
            "end_reason": "arrive_dest" if done else "",
        }

    def act(obs, **kwargs):
        action_calls.append((kwargs["global_step"], kwargs["t0"]))
        return torch.zeros(2)

    trainer.env = SimpleNamespace(reset=reset, step=step, rand_act=lambda: torch.zeros(2))
    trainer.agent = SimpleNamespace(
        act=act, set_global_step=Mock(), update=lambda *args, **kwargs: {},
        planner=SimpleNamespace(stage_for_step=lambda step: SimpleNamespace(name="B", use_residual=True)),
        last_plan=SimpleNamespace(stage=SimpleNamespace(use_mppi=False)),
    )
    trainer.train()
    assert trainer._step == start + 3
    assert action_calls == [(start, True), (start + 1, False), (start + 2, True)]
    assert resets == [10001, 10001]
    trainer.buffer.add.assert_called_once()
    assert trainer._replay_steps == 2
    assert trainer._handle_eval.call_count == (2 if start == 0 else 1)
    trainer.logger.finish.assert_called_once_with(trainer.agent)
