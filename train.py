"""Train TD-MPC2 with a single-coarse-trajectory MPR-MPC planner."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("LAZY_LEGACY_OP", "0")
os.environ.setdefault("TORCHDYNAMO_INLINE_INBUILT_NN_MODULES", "1")

import hydra
import torch
from omegaconf import DictConfig

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from mpr_mpc._bootstrap import LOG_ROOT, bootstrap

bootstrap()

from common.buffer import Buffer  # noqa: E402
from common.logger import Logger  # noqa: E402
from common.parser import parse_cfg  # noqa: E402
from common.seed import set_seed  # noqa: E402
from env import make_env  # noqa: E402
from mpr_mpc.agent import MPRMPCAgent  # noqa: E402
from mpr_mpc.lattice.frenet_metadrive import (  # noqa: E402
    FRENET_DEFAULT_CONFIG,
    MetaDriveFrenetController,
)
from mpr_mpc.planning.coordinator import MPRMPCPlanner  # noqa: E402
from mpr_mpc.tdmpc2.core import MPRTDMPC2  # noqa: E402
from mpr_mpc.trainer import MPROnlineTrainer  # noqa: E402


torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(raw_cfg: DictConfig) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("MPR-MPC uses the CUDA-only TD-MPC2 implementation.")
    raw_cfg.compile = False
    if raw_cfg.get("log_root") is None:
        raw_cfg.log_root = str(LOG_ROOT)
    if raw_cfg.get("checkpoint") is not None:
        raise ValueError(
            "mpr_mpc checkpoint must remain null for a fresh run. Use "
            "resume_checkpoint only for an mpr_mpc_v2_residual_sac training checkpoint."
        )
    cfg = parse_cfg(raw_cfg)
    print(f"MPR-MPC run directory:\n  {cfg.work_dir}")
    print(f"Model directory:\n  {Path(cfg.work_dir) / 'models'}")
    set_seed(int(cfg.seed))
    env = make_env(cfg)
    # The shared environment normally expands seed_steps for random exploration.
    # MPR-MPC owns its cold start and is Lattice-controlled from step zero.
    cfg.seed_steps = 0
    try:
        tdmpc_agent = MPRTDMPC2(cfg)
        lattice_config = dict(FRENET_DEFAULT_CONFIG)
        lattice_config.update(dict(cfg.lattice))
        controller = MetaDriveFrenetController(lattice_config)
        planner = MPRMPCPlanner(
            cfg, tdmpc_agent, controller, action_space=env.action_space
        )
        agent = MPRMPCAgent(cfg, tdmpc_agent, planner, env)
        if cfg.resume_checkpoint:
            agent.load(cfg.resume_checkpoint, load_optimizer=True)
            print(f"Resumed MPR-MPC checkpoint: {cfg.resume_checkpoint}")
        trainer = MPROnlineTrainer(
            cfg=cfg,
            env=env,
            agent=agent,
            buffer=Buffer(cfg),
            logger=Logger(cfg),
        )
        trainer.train()
    finally:
        env.close()


if __name__ == "__main__":
    main()
