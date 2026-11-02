"""Compatibility exports for the copied TD-MPC2 tool namespace."""

from mpr_mpc._bootstrap import bootstrap

bootstrap()

from common.buffer import Buffer  # noqa: E402
from common.world_model import WorldModel  # noqa: E402

__all__ = ["Buffer", "WorldModel"]
