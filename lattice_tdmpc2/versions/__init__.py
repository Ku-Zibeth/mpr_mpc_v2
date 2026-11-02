"""Executable algorithm-version registry for Lattice + TD-MPC2 + SAC."""

from __future__ import annotations


V2 = "lattice_tdmpc2_v2"
V3 = "lattice_tdmpc2_v3"
V4 = "lattice_tdmpc2_v4"
SUPPORTED_VERSIONS = {V2, V3, V4}


def normalize_version(value) -> str:
    aliases = {
        None: V3,
        "v2": V2,
        "v3": V3,
        "v4": V4,
        V2: V2,
        V3: V3,
        V4: V4,
    }
    version = aliases.get(None if value is None else str(value).lower())
    if version is None:
        raise ValueError(
            f"Unsupported algorithm_version={value!r}; expected one of "
            f"{sorted(SUPPORTED_VERSIONS)}."
        )
    return version


def version_from_cfg(cfg) -> str:
    return normalize_version(getattr(cfg, "algorithm_version", None))


def training_semantics(cfg):
    version = version_from_cfg(cfg)
    if version == V2:
        from .v2.training_v2 import V2TrainingSemantics

        return V2TrainingSemantics(cfg.residual_rl)
    if version == V4:
        from .v4.training_v4 import V4TrainingSemantics

        return V4TrainingSemantics(cfg.residual_rl)
    from .v3.training_v3 import V3TrainingSemantics

    return V3TrainingSemantics(cfg.residual_rl)


def sac_agent_class(cfg):
    version = version_from_cfg(cfg)
    if version == V2:
        from .v2.sac_model_v2 import SACAgentV2

        return SACAgentV2
    if version == V4:
        from .v4.sac_model_v4 import SACAgentV4

        return SACAgentV4
    from .v3.sac_model_v3 import SACAgentV3

    return SACAgentV3


__all__ = [
    "SUPPORTED_VERSIONS",
    "V2",
    "V3",
    "V4",
    "normalize_version",
    "sac_agent_class",
    "training_semantics",
    "version_from_cfg",
]
