"""Executable V3 algorithm components."""

__all__ = ["ProjectedLagrangianV3", "SACAgentV3", "V3TrainingSemantics"]


def __getattr__(name):
    if name in {"ProjectedLagrangianV3", "SACAgentV3"}:
        from .sac_model_v3 import ProjectedLagrangianV3, SACAgentV3

        return {
            "ProjectedLagrangianV3": ProjectedLagrangianV3,
            "SACAgentV3": SACAgentV3,
        }[name]
    if name == "V3TrainingSemantics":
        from .training_v3 import V3TrainingSemantics

        return V3TrainingSemantics
    raise AttributeError(name)
