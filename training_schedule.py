"""Small dependency-free helpers for staged online training schedules."""

from __future__ import annotations


def tdmpc_update_gate(
    *,
    stage_name: str,
    replay_ready: bool,
    budget: float,
    ratios: dict[str, float],
) -> tuple[bool, float, float, int]:
    """Return whether to update TD-MPC2 and the updated fractional budget."""

    ratio = max(0.0, float(ratios.get(str(stage_name), 1.0)))
    if not replay_ready or ratio <= 0.0:
        return False, float(budget), ratio, 0
    budget = float(budget) + ratio
    if budget >= 1.0:
        return True, budget - 1.0, ratio, 1
    return False, budget, ratio, 0


__all__ = ["tdmpc_update_gate"]
