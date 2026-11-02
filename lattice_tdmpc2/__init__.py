"""Vendored Lattice / TD-MPC2 integration helpers.

Importing the package itself must not mutate ``sys.path`` or eagerly import
legacy training modules. MPR-MPC's default runtime imports only the safety-cost
submodules, while the historical helper classes remain available lazily.
"""

__all__ = [
    "LatticeActionAdapter",
    "ResidualPlanResult",
    "ResidualActionAdapter",
    "ResidualSACLatticePlanner",
    "ResidualStateBuilder",
    "WorldModelEvaluator",
]


def __getattr__(name):
    if name == "LatticeActionAdapter":
        from .action_adapter import LatticeActionAdapter

        return LatticeActionAdapter
    if name == "WorldModelEvaluator":
        from .evaluator import WorldModelEvaluator

        return WorldModelEvaluator
    if name in {"ResidualPlanResult", "ResidualSACLatticePlanner"}:
        from .planner import ResidualPlanResult, ResidualSACLatticePlanner

        return {
            "ResidualPlanResult": ResidualPlanResult,
            "ResidualSACLatticePlanner": ResidualSACLatticePlanner,
        }[name]
    if name == "ResidualActionAdapter":
        from .residual_action import ResidualActionAdapter

        return ResidualActionAdapter
    if name == "ResidualStateBuilder":
        from .state_builder import ResidualStateBuilder

        return ResidualStateBuilder
    raise AttributeError(name)
