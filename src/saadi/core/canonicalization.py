"""Converts raw LLM responses into validated canonicalization outputs."""

from __future__ import annotations

from ..entities.active_entity_config import (
    DEFAULT_MAX_ALIASES,
    DEFAULT_MAX_DESC,
    DEFAULT_MAX_ENTITIES,
)
from .parsers import safe_json_from_text
from .validation import build_canonicalization_output


def parse_canonicalization_response(
    raw_model_text: str,
    *,
    fallback_chunk_text: str,
    allow_linking_sentence: bool = True,
    max_entities: int = DEFAULT_MAX_ENTITIES,
    max_aliases: int = DEFAULT_MAX_ALIASES,
    max_desc: int = DEFAULT_MAX_DESC,
    max_objects: int | None = None,
    anchor_matching: str = "strict",
):
    if max_objects is not None:
        max_entities = int(max_objects)
    parsed, status = safe_json_from_text(raw_model_text)
    return build_canonicalization_output(
        parsed=parsed,
        raw_model_text=raw_model_text,
        parser_status=status,
        fallback_chunk_text=fallback_chunk_text,
        allow_linking_sentence=allow_linking_sentence,
        max_entities=max_entities,
        max_aliases=max_aliases,
        max_desc=max_desc,
        anchor_matching=anchor_matching,
    )
