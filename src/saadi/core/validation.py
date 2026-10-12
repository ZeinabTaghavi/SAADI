"""Validates and coerces parsed LLM output fields into typed objects."""

from __future__ import annotations

import json
from typing import Any

from ..entities.active_entity_config import (
    DEFAULT_MAX_ALIASES,
    DEFAULT_MAX_DESC,
    DEFAULT_MAX_ENTITIES,
)
from .text_edits import materialize_chunk_text
from .types import CanonicalizationOutput, EntityRecord, TextInsertion

UNPARSED_CHUNK_PREFIX = "[UNPARSED_OUTPUT] "


def _sanitize_parsed_for_cache(parsed: dict[str, Any], *, allow_linking_sentence: bool) -> tuple[dict[str, Any], bool]:
    scrubbed = dict(parsed)
    changed = False

    raw_insertions = scrubbed.get("insertions")
    if isinstance(raw_insertions, list):
        allowed_insertion_keys = {"name", "anchor_text", "left_context", "right_context", "occurrence"}
        clean_insertions: list[Any] = []
        for item in raw_insertions:
            if not isinstance(item, dict):
                clean_insertions.append(item)
                continue
            cleaned = {key: value for key, value in item.items() if key in allowed_insertion_keys}
            if cleaned != item:
                changed = True
            clean_insertions.append(cleaned)
        scrubbed["insertions"] = clean_insertions

    if not allow_linking_sentence and "linking_sentence" in scrubbed:
        scrubbed.pop("linking_sentence", None)
        changed = True

    return scrubbed, changed


def _stored_model_text_for_cache(*, raw_model_text: str, parsed: dict[str, Any], allow_linking_sentence: bool) -> str:
    scrubbed, changed = _sanitize_parsed_for_cache(parsed, allow_linking_sentence=allow_linking_sentence)
    if not changed:
        return raw_model_text
    return json.dumps(scrubbed, ensure_ascii=False) + "\n[END_JSON]"


def coerce_entity_records(
    items: Any,
    *,
    max_entities: int = DEFAULT_MAX_ENTITIES,
    max_aliases: int = DEFAULT_MAX_ALIASES,
    max_desc: int = DEFAULT_MAX_DESC,
) -> list[EntityRecord]:
    if not isinstance(items, list):
        return []
    out: list[EntityRecord] = []
    for item in items[:max_entities]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        aliases_raw = item.get("aliases", [])
        desc_raw = item.get("description", [])
        aliases = [str(x).strip() for x in aliases_raw if isinstance(x, str) and str(x).strip()][:max_aliases if max_aliases >= 0 else None]
        desc = [str(x).strip() for x in desc_raw if isinstance(x, str) and str(x).strip()][:max_desc if max_desc >= 0 else None]
        out.append(EntityRecord(name=name, aliases=aliases, description=desc))
    return out


def coerce_text_insertions(items: Any, *, max_insertions: int = 12) -> list[TextInsertion]:
    if not isinstance(items, list):
        return []
    out: list[TextInsertion] = []
    for item in items[:max_insertions]:
        if not isinstance(item, dict):
            continue
        name = str(
            item.get("name", item.get("object_name", item.get("entity_name", ""))) or ""
        ).strip()
        anchor_text = str(item.get("anchor_text", "") or "").strip()
        if not name:
            continue
        left_context = str(item.get("left_context", "") or "")
        right_context = str(item.get("right_context", "") or "")
        try:
            occurrence = int(item.get("occurrence", 1) or 1)
        except Exception:
            occurrence = 1
        out.append(
            TextInsertion(
                anchor_text=anchor_text,
                left_context=left_context,
                right_context=right_context,
                occurrence=max(1, occurrence),
                name=name,
            )
        )
    return out


def _entity_key(name: str) -> str:
    return " ".join(str(name or "").strip().casefold().split())


def _unique_exact_name_insertion(raw_text: str, name: str) -> TextInsertion | None:
    raw = str(raw_text or "")
    target = str(name or "").strip()
    if not raw or not target:
        return None
    first = raw.find(target)
    if first < 0:
        return None
    if raw.find(target, first + len(target)) >= 0:
        return None
    return TextInsertion(anchor_text=target, name=target)


def ensure_active_entity_insertions(
    *,
    raw_text: str,
    active_entities: list[EntityRecord],
    insertions: list[TextInsertion],
) -> tuple[list[TextInsertion], list[str]]:
    """Ensure each active entity has one insertion carrying its canonical name."""
    if not active_entities:
        return insertions, []

    active_by_key = {_entity_key(entity.name): entity for entity in active_entities if _entity_key(entity.name)}
    covered: set[str] = set()
    normalized: list[TextInsertion] = []

    for insertion in insertions:
        candidate_key = _entity_key(insertion.name)
        entity = active_by_key.get(candidate_key)
        if entity is None:
            normalized.append(insertion)
            continue
        covered.add(_entity_key(entity.name))
        normalized.append(
            TextInsertion(
                anchor_text=insertion.anchor_text,
                left_context=insertion.left_context,
                right_context=insertion.right_context,
                occurrence=insertion.occurrence,
                name=entity.name,
            )
        )

    missing: list[str] = []
    for entity in active_entities:
        key = _entity_key(entity.name)
        if not key or key in covered:
            continue
        auto = _unique_exact_name_insertion(raw_text, entity.name)
        if auto is None:
            missing.append(entity.name)
            continue
        covered.add(key)
        normalized.append(auto)

    return normalized, missing


def build_canonicalization_output(
    *,
    parsed: dict[str, Any] | None,
    raw_model_text: str,
    parser_status: str,
    fallback_chunk_text: str,
    allow_linking_sentence: bool = True,
    max_entities: int = DEFAULT_MAX_ENTITIES,
    max_aliases: int = DEFAULT_MAX_ALIASES,
    max_desc: int = DEFAULT_MAX_DESC,
    max_objects: int | None = None,
) -> CanonicalizationOutput:
    if max_objects is not None:
        max_entities = int(max_objects)
    parsed = parsed or {}
    model_chunk = parsed.get("updated_current_chunk") or parsed.get("chunk_resolved")
    raw_active_entities = parsed.get("active_entities")
    if raw_active_entities is None:
        raw_active_entities = parsed.get("active_objects", parsed.get("objects_active", []))
    active_entities = coerce_entity_records(
        raw_active_entities,
        max_entities=max_entities,
        max_aliases=max_aliases,
        max_desc=max_desc,
    )
    linking_sentence = str(parsed.get("linking_sentence", "") or "").strip()
    insertions = coerce_text_insertions(parsed.get("insertions", []))
    insertions, missing_active_insertions = ensure_active_entity_insertions(
        raw_text=fallback_chunk_text,
        active_entities=active_entities,
        insertions=insertions,
    )

    status = parser_status if parser_status in {"ok", "fallback", "failed"} else "failed"
    uses_edit_ops = ("linking_sentence" in parsed) or ("insertions" in parsed)
    has_active_entities = any(
        key in parsed for key in ("active_entities", "active_objects", "objects_active")
    )
    if active_entities:
        uses_edit_ops = True
    if not allow_linking_sentence and linking_sentence:
        linking_sentence = ""
        if status == "ok":
            status = "fallback"
    stored_model_text = _stored_model_text_for_cache(
        raw_model_text=raw_model_text,
        parsed=parsed,
        allow_linking_sentence=allow_linking_sentence,
    )
    rejected_entities: list[EntityRecord] = []

    if uses_edit_ops:
        materialized = materialize_chunk_text(
            fallback_chunk_text,
            linking_sentence=linking_sentence,
            insertions=insertions,
        )
        updated_chunk = materialized.text
        linking_sentence = materialized.linking_sentence
        insertions = materialized.insertions
        if not allow_linking_sentence and linking_sentence:
            linking_sentence = ""
        if status == "ok" and (materialized.dropped_insertions or materialized.linking_sentence_dropped):
            status = "fallback"
        if active_entities:
            accepted_keys = {_entity_key(item.name) for item in insertions}
            rejected_entities = [
                entity
                for entity in active_entities
                if _entity_key(entity.name) not in accepted_keys
            ]
            # An entity is grounded only when at least one of its insertion
            # anchors survived exact-span materialization. Rejected entities are
            # kept in diagnostics but must never reach discourse state/memory.
            if rejected_entities:
                rejected_keys = {_entity_key(entity.name) for entity in rejected_entities}
                active_entities = [
                    entity
                    for entity in active_entities
                    if _entity_key(entity.name) not in rejected_keys
                ]
            if status == "ok" and (missing_active_insertions or rejected_entities):
                status = "fallback"
    elif model_chunk:
        updated_chunk = str(model_chunk)
    elif has_active_entities:
        updated_chunk = str(fallback_chunk_text or "")
    else:
        fallback_text = str(fallback_chunk_text or "")
        updated_chunk = (
            fallback_text
            if fallback_text.startswith(UNPARSED_CHUNK_PREFIX)
            else f"{UNPARSED_CHUNK_PREFIX}{fallback_text}"
        )
    out = CanonicalizationOutput(
        updated_chunk_text=updated_chunk,
        active_entities=active_entities,
        raw_model_text=stored_model_text,
        parser_status=status,
        linking_sentence=linking_sentence,
        insertions=insertions,
        rejected_entities=rejected_entities,
    )
    validate_canonicalization_output(out)
    return out


def validate_canonicalization_output(output: CanonicalizationOutput) -> None:
    if not isinstance(output.updated_chunk_text, str) or output.updated_chunk_text == "":
        raise ValueError("updated_chunk_text must be a non-empty string")
    if output.parser_status not in {"ok", "fallback", "failed"}:
        raise ValueError(f"invalid parser_status: {output.parser_status}")


def coerce_entity_objects(
    items: Any,
    *,
    max_objects: int = DEFAULT_MAX_ENTITIES,
    max_aliases: int = DEFAULT_MAX_ALIASES,
    max_desc: int = DEFAULT_MAX_DESC,
) -> list[EntityRecord]:
    """Deprecated wrapper for callers using the former object terminology."""
    return coerce_entity_records(
        items,
        max_entities=max_objects,
        max_aliases=max_aliases,
        max_desc=max_desc,
    )


ensure_active_object_insertions = ensure_active_entity_insertions
