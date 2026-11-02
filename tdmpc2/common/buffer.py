"""MPR-MPC namespace for the unchanged TD-MPC2 replay buffer."""

from mpr_mpc._bootstrap import bootstrap

bootstrap()

from common.buffer import Buffer  # noqa: E402,F401

__all__ = ["Buffer"]
