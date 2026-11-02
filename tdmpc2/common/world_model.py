"""MPR-MPC namespace for the unchanged TD-MPC2 world model."""

from mpr_mpc._bootstrap import bootstrap

bootstrap()

from common.world_model import WorldModel  # noqa: E402,F401

__all__ = ["WorldModel"]
