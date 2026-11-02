"""Executable V2 algorithm components."""

__all__ = ["PIDLagrangianV2", "SACAgentV2", "V2TrainingSemantics"]


def __getattr__(name):
    if name in {"PIDLagrangianV2", "SACAgentV2"}:
        from .sac_model_v2 import PIDLagrangianV2, SACAgentV2

        return {"PIDLagrangianV2": PIDLagrangianV2, "SACAgentV2": SACAgentV2}[name]
    if name == "V2TrainingSemantics":
        from .training_v2 import V2TrainingSemantics

        return V2TrainingSemantics
    raise AttributeError(name)
