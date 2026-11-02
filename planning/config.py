"""Small helpers for nested DictConfig/dataclass/mapping access."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    try:
        return dict(value)
    except (TypeError, ValueError):
        return {}


def section(cfg: Any, name: str) -> dict[str, Any]:
    if isinstance(cfg, Mapping):
        return mapping(cfg.get(name))
    return mapping(getattr(cfg, name, None))


__all__ = ["mapping", "section"]
