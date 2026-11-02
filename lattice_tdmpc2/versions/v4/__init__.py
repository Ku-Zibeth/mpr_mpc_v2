"""Executable V4 algorithm components."""

__all__ = ["ProjectedLagrangianV4", "SACAgentV4", "V4TrainingSemantics"]


def __getattr__(name):
    if name in {"ProjectedLagrangianV4", "SACAgentV4"}:
        from .sac_model_v4 import ProjectedLagrangianV4, SACAgentV4

        return {
            "ProjectedLagrangianV4": ProjectedLagrangianV4,
            "SACAgentV4": SACAgentV4,
        }[name]
    if name == "V4TrainingSemantics":
        from .training_v4 import V4TrainingSemantics

        return V4TrainingSemantics
    raise AttributeError(name)
