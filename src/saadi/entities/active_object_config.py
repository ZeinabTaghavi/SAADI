"""Deprecated compatibility helpers for the former active-object API."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .active_entity_config import (
    DEFAULT_MAX_ALIASES,
    DEFAULT_MAX_DESC,
    DEFAULT_MAX_ENTITIES,
    DEFAULT_MIN_RELEVANCE_SCORE,
    normalize_active_entity_limit,
    normalize_active_entity_min_relevance,
    resolve_active_entity_limits,
)

DEFAULT_MAX_OBJECTS = DEFAULT_MAX_ENTITIES


def normalize_active_object_limit(value: Any, *, default: int, field_name: str) -> int:
    return normalize_active_entity_limit(value, default=default, field_name=field_name)


def normalize_active_object_min_relevance(
    value: Any,
    *,
    default: float = DEFAULT_MIN_RELEVANCE_SCORE,
    field_name: str = "ingest.strategy_kwargs.active_object_min_relevance",
) -> float:
    return normalize_active_entity_min_relevance(
        value,
        default=default,
        field_name=field_name,
    )


def resolve_active_object_limits(raw_kwargs: Mapping[str, Any] | None) -> tuple[int, int, int]:
    return resolve_active_entity_limits(raw_kwargs)


def active_object_config_label(max_objects: int, max_aliases: int, max_desc: int) -> str:
    return f"AO_{int(max_objects)}_{int(max_aliases)}_{int(max_desc)}"
