"""MPR-MPC: structured trajectory priors with TD-MPC2 local refinement."""

from pathlib import Path
import sys

_root = Path(__file__).resolve().parent
_existing = sys.modules.get("mpr_mpc")
if _existing is None:
    sys.modules["mpr_mpc"] = sys.modules[__name__]
elif str(_root) not in list(getattr(_existing, "__path__", [])):
    raise ImportError("A different mpr_mpc package is already loaded; run v2 in a fresh process.")

from .agent import MPRMPCAgent
from .planning.coordinator import MPRMPCPlanner

__all__ = ["MPRMPCAgent", "MPRMPCPlanner"]
