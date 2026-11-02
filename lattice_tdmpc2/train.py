"""Launch residual Lattice-SAC training with the shared MetaDrive config."""

from __future__ import annotations

import os
import sys
from pathlib import Path


# Match the runtime setup used by tdmpc2_fm/train.py.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("LAZY_LEGACY_OP", "0")
os.environ.setdefault("TORCHDYNAMO_INLINE_INBUILT_NN_MODULES", "1")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
UPSTREAM_CODE_ROOT = REPO_ROOT / "tdmpc2"
for path in (REPO_ROOT, UPSTREAM_CODE_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch
import hydra
from omegaconf import DictConfig

from lattice_tdmpc2.trainer import run_training


torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(raw_cfg: DictConfig) -> None:
    """Use the local config and forward Hydra overrides to the residual trainer."""
    run_training(raw_cfg)


if __name__ == "__main__":
    main()
