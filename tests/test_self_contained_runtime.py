from __future__ import annotations

import importlib
import sys
from pathlib import Path

from omegaconf import OmegaConf

from mpr_mpc._bootstrap import LOG_ROOT, PACKAGE_ROOT, TDMPC2_RUNTIME, bootstrap


bootstrap()

from common.logger import Logger  # noqa: E402
from common.parser import parse_cfg  # noqa: E402


def _cfg(tmp_path: Path | None = None):
    cfg = OmegaConf.load(PACKAGE_ROOT / "config.yaml")
    cfg.enable_wandb = False
    cfg.save_agent = False
    cfg.save_csv = False
    cfg.save_video = False
    cfg.steps = 1
    cfg.obs_shape = {"state": (259,)}
    cfg.action_dim = 2
    if tmp_path is not None:
        cfg.log_root = str(tmp_path / "logs")
    elif cfg.get("log_root") is None:
        cfg.log_root = str(LOG_ROOT)
    return cfg


def test_default_log_root_is_repository_local():
    cfg = _cfg()
    parsed = parse_cfg(cfg)
    assert Path(parsed.log_root).resolve() == LOG_ROOT.resolve()


def test_work_dir_is_repository_local():
    cfg = _cfg()
    parsed = parse_cfg(cfg)
    expected = LOG_ROOT / parsed.task / str(parsed.seed) / parsed.exp_name
    assert Path(parsed.work_dir).resolve() == expected.resolve()


def test_custom_log_root_override(tmp_path):
    cfg = _cfg(tmp_path)
    parsed = parse_cfg(cfg)
    expected = tmp_path / "logs" / parsed.task / str(parsed.seed) / parsed.exp_name
    assert Path(parsed.work_dir).resolve() == expected.resolve()


def test_custom_work_dir_with_null_log_root(tmp_path):
    cfg = _cfg()
    work_dir = tmp_path / "runs" / "metadrive-risk" / "7" / "custom_run"
    cfg.work_dir = str(work_dir)
    cfg.log_root = None
    parsed = parse_cfg(cfg)
    assert Path(parsed.work_dir).resolve() == work_dir.resolve()
    assert str(parsed.log_root) != "None"
    assert Path(parsed.log_root).resolve() == (tmp_path / "runs").resolve()


def test_model_dir_is_under_work_dir(tmp_path):
    cfg = parse_cfg(_cfg(tmp_path))
    logger = Logger(cfg)
    assert Path(logger.model_dir).resolve() == (Path(cfg.work_dir) / "models").resolve()


def test_runtime_modules_are_repository_local():
    modules = {
        "env": PACKAGE_ROOT,
        "lattice": PACKAGE_ROOT,
        "lattice.frenet_metadrive": PACKAGE_ROOT,
        "lattice_tdmpc2": PACKAGE_ROOT,
        "common.buffer": TDMPC2_RUNTIME,
        "tdmpc2": TDMPC2_RUNTIME,
    }
    for name, root in modules.items():
        for loaded in list(sys.modules):
            if loaded == name or loaded.startswith(f"{name}."):
                sys.modules.pop(loaded, None)
        bootstrap()
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        path.relative_to(root.resolve())
