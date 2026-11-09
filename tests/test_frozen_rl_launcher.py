import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import shutil
import time
from unittest.mock import Mock

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest

from mpr_mpc._bootstrap import PACKAGE_ROOT
from common.logger import Logger
from common.parser import parse_cfg


def logger_config(tmp_path):
    raw = OmegaConf.load(PACKAGE_ROOT / "config.yaml")
    raw.work_dir = str(tmp_path)
    raw.enable_wandb = True
    raw.wandb_entity = "test-entity"
    raw.obs_shape = {"state": (259,)}
    raw.action_dim = 2
    return parse_cfg(raw)


@pytest.mark.parametrize("failures", [0, 1, 2])
def test_cloud_initialization_can_fall_back_without_disabling_local_saves(tmp_path, monkeypatch, failures):
    wandb = Mock()
    wandb.init.side_effect = [RuntimeError("403 Forbidden")] * failures + [Mock()]
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    cfg = logger_config(tmp_path)
    logger = Logger(cfg)
    wandb.Settings.assert_called_once_with(console="off", init_timeout=60)
    assert wandb.init.call_count == min(failures + 1, 2)
    if failures:
        assert wandb.init.call_args_list[1].kwargs["mode"] == "offline"
    assert logger._save_agent and cfg.save_agent
    logger.log({"step": 500001, "episode_reward": 1.}, "eval")
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert rows[0]["step"] == 500001
    if failures == 2:
        assert logger._wandb is None


def test_launcher_dry_run_selects_world_model_only_and_correct_schedule(tmp_path):
    checkpoint = tmp_path / "milestone_500k_world_model.pt"
    checkpoint.touch()
    script = PACKAGE_ROOT / "tools/launch_frozen_wm500k_rl500k.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    env = {**os.environ, "PYTHON": sys.executable, "WM_CHECKPOINT": str(checkpoint),
           "EXP_NAME": "test_frozen_wm500k_rl500k", "MPR_PROXY": ""}
    result = subprocess.run(["bash", str(script), "--dry-run"], env=env,
                            check=True, capture_output=True, text=True, timeout=20)
    assert "WANDB_CONSOLE=off" in result.stdout
    command = shlex.split(result.stdout.splitlines()[1])
    assert command[:3] == [sys.executable, "-u", "train.py"]
    with initialize_config_dir(version_base="1.3", config_dir=str(PACKAGE_ROOT)):
        cfg = compose(config_name="config", overrides=command[3:])
    assert cfg.resume_checkpoint == str(checkpoint)
    assert cfg.steps == cfg.mpr_mpc.stages.mppi_start_step == 1000000
    assert cfg.mpr_mpc.stages.residual_start_step == 500000
    assert cfg.residual_rl.freeze_tdmpc2_on_start
    assert cfg.tdmpc_stage_training.stage_b_update_ratio == 0
    assert cfg.tdmpc_stage_training.stage_c_update_ratio == 0
    assert cfg.enable_wandb and not cfg.wandb_log_artifacts
    assert cfg.wandb_run_name == cfg.exp_name


def test_launcher_refuses_duplicate_training_without_starting_another_job(tmp_path):
    checkpoint = tmp_path / "milestone_500k_world_model.pt"
    checkpoint.touch()
    env = {**os.environ, "PYTHON": sys.executable, "WM_CHECKPOINT": str(checkpoint),
           "EXP_NAME": "test_frozen_rl_duplicate_guard"}
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)", "train.py"],
                               cwd=PACKAGE_ROOT)
    try:
        result = subprocess.run(["bash", str(PACKAGE_ROOT / "tools/launch_frozen_wm500k_rl500k.sh")],
                                env=env, capture_output=True, text=True, timeout=10)
        assert result.returncode == 1
        assert "Another training process is active" in result.stderr
        assert not (PACKAGE_ROOT / "logs/launch/test_frozen_rl_duplicate_guard").exists()
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_background_launcher_writes_past_pipe_capacity_to_a_regular_file(tmp_path):
    name = f"test_direct_log_{os.getpid()}_{time.time_ns()}"
    output = PACKAGE_ROOT / "logs/launch" / name
    model_run = PACKAGE_ROOT / "logs/metadrive-risk/1" / name
    checkpoint = tmp_path / "milestone_500k_world_model.pt"
    checkpoint.touch()
    finished = tmp_path / "finished"
    python = tmp_path / "fake_python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import os, stat, time\n"
        "assert stat.S_ISREG(os.fstat(1).st_mode)\n"
        "assert os.environ['WANDB_CONSOLE'] == 'off'\n"
        "print('x' * 200000, flush=True)\n"
        "time.sleep(4)\n"
        f"open({str(finished)!r}, 'w').close()\n"
    )
    python.chmod(0o700)
    # This positive launch test must not depend on a user's real training job.
    # The preceding test independently exercises the real duplicate-job guard.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    pgrep = bin_dir / "pgrep"
    pgrep.write_text("#!/bin/sh\nexit 1\n")
    pgrep.chmod(0o700)
    env = {**os.environ, "PYTHON": str(python), "WM_CHECKPOINT": str(checkpoint), "EXP_NAME": name,
           "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"]}
    try:
        result = subprocess.run(["bash", str(PACKAGE_ROOT / "tools/launch_frozen_wm500k_rl500k.sh")],
                                env=env, capture_output=True, text=True, check=True, timeout=20)
        assert "Started PID:" in result.stdout
        assert (output / "run.log").stat().st_size > 65536
        assert (model_run / "models/milestone_500k_world_model.pt").is_file()
        deadline = time.monotonic() + 10
        while not finished.exists() and time.monotonic() < deadline:
            time.sleep(.1)
        assert finished.exists()
    finally:
        if not finished.exists() and (output / "train.pid").exists():
            pid = int((output / "train.pid").read_text())
            try:
                os.kill(pid, 15)
            except ProcessLookupError:
                pass
        if output.exists():
            shutil.rmtree(output)
        if model_run.exists():
            shutil.rmtree(model_run)
