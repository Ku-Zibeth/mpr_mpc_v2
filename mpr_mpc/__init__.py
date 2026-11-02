"""Package alias for running this repository from its clone root.

The repository keeps the historical source files at the project root
(`agent.py`, `planning/`, `residual_rl/`, ...), while the code imports them via
the stable ``mpr_mpc`` package name. Point this package's search path at the
repository root so ``import mpr_mpc.agent`` works after cloning to any directory
name, including ``MPR_MetaDrive``.
"""

from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
__path__ = [str(PACKAGE_ROOT)]

from .agent import MPRMPCAgent  # noqa: E402,F401
from .planning.coordinator import MPRMPCPlanner  # noqa: E402,F401

__all__ = ["MPRMPCAgent", "MPRMPCPlanner"]
