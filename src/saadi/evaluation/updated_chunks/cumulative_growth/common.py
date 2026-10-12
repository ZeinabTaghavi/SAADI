"""Shared helpers for cumulative-growth analysis over updated chunks."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..common import resolve_updated_chunks_output_dir
from ..static_statistics.common import active_entities_from_record


_WS_RE = re.compile(r"\s+")


def normalize_growth_text(value: Any) -> str:
    """
    Deterministic text normalization for novelty tracking.

    We merge strings that differ only by case or whitespace.
    Punctuation differences are intentionally preserved to keep matching simple.
    """
    text = str(value or "").strip().lower()
    if not text:
        return ""
    return _WS_RE.sub(" ", text)


def normalize_entity_name(value: Any) -> str:
    return normalize_growth_text(value)


def normalize_alias_key(*, entity_name: Any, alias: Any) -> tuple[str, str] | None:
    entity_key = normalize_entity_name(entity_name)
    alias_key = normalize_growth_text(alias)
    if not entity_key or not alias_key:
        return None
    return (entity_key, alias_key)


def normalize_description_key(*, entity_name: Any, description: Any) -> tuple[str, str] | None:
    entity_key = normalize_entity_name(entity_name)
    description_key = normalize_growth_text(description)
    if not entity_key or not description_key:
        return None
    return (entity_key, description_key)


def resolve_per_document_growth_output_dir(
    experiment_config: dict[str, Any],
    *,
    run_name: str = "default",
) -> Path:
    return resolve_updated_chunks_output_dir(
        experiment_config,
        subphase_id="3_1_Per_Document_Cumulative_Growth_Logging",
        run_name=run_name,
    )


def resolve_aggregated_growth_output_dir(
    experiment_config: dict[str, Any],
    *,
    run_name: str = "default",
) -> Path:
    return resolve_updated_chunks_output_dir(
        experiment_config,
        subphase_id="3_2_Dataset_Level_Aggregated_Cumulative_Curves",
        run_name=run_name,
    )


__all__ = [
    "active_entities_from_record",
    "normalize_alias_key",
    "normalize_description_key",
    "normalize_entity_name",
    "normalize_growth_text",
    "resolve_aggregated_growth_output_dir",
    "resolve_per_document_growth_output_dir",
]
