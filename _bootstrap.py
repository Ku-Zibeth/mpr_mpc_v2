"""Import helpers for the self-contained MPR-MPC repository."""

from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
VENDOR_ROOT = PACKAGE_ROOT / "vendor"
TDMPC2_RUNTIME = VENDOR_ROOT / "tdmpc2_runtime"
LOG_ROOT = PACKAGE_ROOT / "logs"


def bootstrap() -> None:
    """Expose only repository-local runtime paths.

    The vendored TD-MPC2 runtime must precede PACKAGE_ROOT so imports such as
    ``common.buffer`` and ``tdmpc2`` cannot fall through to an outer workspace.
    """
    required = (TDMPC2_RUNTIME, PACKAGE_ROOT)
    missing = [path for path in required if not path.exists()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise RuntimeError(f"Missing vendored MPR-MPC runtime path(s): {joined}")
    for path in reversed(required):
        value = str(path)
        while value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)


__all__ = ["PACKAGE_ROOT", "VENDOR_ROOT", "TDMPC2_RUNTIME", "LOG_ROOT", "bootstrap"]
