"""Canonical configuration helpers for active-entity selection."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

DEFAULT_MAX_ENTITIES = 5
DEFAULT_MAX_ALIASES = 2
DEFAULT_MAX_DESC = 1
DEFAULT_MIN_RELEVANCE_SCORE = 0.0


def normalize_active_entity_limit(value: Any, *, default: int, field_name: str) -> int:
    if value is None or (isinstance(value, str) and not value.strip()):
        return int(default)
    try:
        out = int(value)
    except Exception as exc:
        raise ValueError(f"{field_name} must be an integer (got: {value!r})") from exc
    if out < 0:
        raise ValueError(f"{field_name} must be >= 0 (got: {out})")
    return out


def normalize_active_entity_min_relevance(
    value: Any,
    *,
    default: float = DEFAULT_MIN_RELEVANCE_SCORE,
    field_name: str = "ingest.strategy_kwargs.active_entity_min_relevance",
) -> float:
    if value is None or (isinstance(value, str) and not value.strip()):
        return float(default)
    try:
        out = float(value)
    except Exception as exc:
        raise ValueError(
            f"{field_name} must be a finite non-negative number (got: {value!r})"
        ) from exc
    if not math.isfinite(out) or out < 0:
        raise ValueError(
            f"{field_name} must be a finite non-negative number (got: {value!r})"
        )
    return out


def resolve_active_entity_limits(raw_kwargs: Mapping[str, Any] | None) -> tuple[int, int, int]:
    """Resolve canonical keys while accepting the legacy ``max_objects`` key."""
    kwargs = raw_kwargs if isinstance(raw_kwargs, Mapping) else {}
    max_entities = kwargs.get("max_entities")
    if max_entities is None:
        max_entities = kwargs.get("max_objects")
    return (
        normalize_active_entity_limit(
            max_entities,
            default=DEFAULT_MAX_ENTITIES,
            field_name="ingest.strategy_kwargs.max_entities",
        ),
        normalize_active_entity_limit(
            kwargs.get("max_aliases"),
            default=DEFAULT_MAX_ALIASES,
            field_name="ingest.strategy_kwargs.max_aliases",
        ),
        normalize_active_entity_limit(
            kwargs.get("max_desc"),
            default=DEFAULT_MAX_DESC,
            field_name="ingest.strategy_kwargs.max_desc",
        ),
    )


def active_entity_config_label(max_entities: int, max_aliases: int, max_desc: int) -> str:
    """Return the canonical cache/config label for new runs."""
    return f"AE_{int(max_entities)}_{int(max_aliases)}_{int(max_desc)}"


def legacy_active_object_config_label(max_entities: int, max_aliases: int, max_desc: int) -> str:
    """Return the historical label used when looking up an existing cache."""
    return f"AO_{int(max_entities)}_{int(max_aliases)}_{int(max_desc)}"
