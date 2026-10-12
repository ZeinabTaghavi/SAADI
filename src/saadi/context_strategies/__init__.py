"""Registers and resolves available SAADI context strategies."""

from .base import ContextStrategy
from .hierarchical_summary import (
    MEMORY_UPDATE_MODE_HIS_MEM,
    HierarchicalSummaryStrategy,
    LimitedHierarchicalSummaryStrategy,
    normalize_memory_update_mode,
)


def get_strategy(name: str, **kwargs):
    # Accept old experiment files without carrying the former terminology into
    # the strategy API. Conflicting old/new keys are rejected deterministically.
    aliases = {
        "max_objects": "max_entities",
        "active_object_selector": "active_entity_selector",
        "active_object_min_relevance": "active_entity_min_relevance",
    }
    kwargs = dict(kwargs)
    for legacy_key, canonical_key in aliases.items():
        if legacy_key not in kwargs:
            continue
        if canonical_key in kwargs and kwargs[canonical_key] != kwargs[legacy_key]:
            raise ValueError(
                f"Conflicting strategy settings: {canonical_key!r} and deprecated {legacy_key!r}."
            )
        kwargs.setdefault(canonical_key, kwargs[legacy_key])
        kwargs.pop(legacy_key, None)
    n = name.lower()
    if n in {"hierarchical", "saadi_hierarchical", "accumulative"}:
        return HierarchicalSummaryStrategy(**kwargs)
    if n in {"limited_hierarchical", "limited_hierarchical_summary", "saadi_limited_hierarchical"}:
        return LimitedHierarchicalSummaryStrategy(**kwargs)
    raise ValueError(f"Unknown strategy: {name}")


def strategy_cache_label(name: str, strategy_kwargs: dict | None = None) -> str:
    """Return the legacy label by default and isolate opt-in memory ablations."""
    kwargs = strategy_kwargs or {}
    mode = normalize_memory_update_mode(kwargs.get("memory_update_mode"))
    return str(name) if mode == MEMORY_UPDATE_MODE_HIS_MEM else f"{name}_{mode}"
