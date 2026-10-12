"""Memory-Based Re-scoring (MBR) helpers and retrieval implementation."""

from __future__ import annotations

from dataclasses import asdict
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...core.types import CanonicalizedChunkRecord, RetrievalExample, RetrievalResult
from ..retriever_adapters import (
    BM25Retriever,
    build_encoded_text_pool,
    compute_similarity,
    encode_query,
    get_retriever,
)
from ...io.hierarchy_cache import (
    _load_v2_doc_records,
    _resolve_cache_file_path,
    load_canonicalized_chunk_records_from_cache_index,
)
from ...io.jsonl import read_jsonl


MBR_VARIANT_MBR1 = "MBR1"
MBR_VARIANT_MBR2 = "MBR2"
MBR_VARIANT_LINEAR = "MBR_LINEAR"
MBR_VARIANT_POOL = "MBR_POOL"
MBR_VARIANTS = {
    MBR_VARIANT_MBR1,
    MBR_VARIANT_MBR2,
    MBR_VARIANT_LINEAR,
    MBR_VARIANT_POOL,
}

DEFAULT_MBR_LAMBDA_T = 1.0
DEFAULT_MBR_LAMBDA_M = 0.5
DEFAULT_MBR_LAMBDA_H = 0.3
DEFAULT_MBR_MEMORY_NAME_WEIGHT = 1.0
DEFAULT_MBR_MEMORY_ALIAS_WEIGHT = 0.0
DEFAULT_MBR_MEMORY_DESCRIPTION_WEIGHT = 0.0
DEFAULT_MBR2_MEMORY_NAME_WEIGHT = 1.0
DEFAULT_MBR2_MEMORY_ALIAS_WEIGHT = 1.0
DEFAULT_MBR2_MEMORY_DESCRIPTION_WEIGHT = 1.0
DEFAULT_MBR_MEMORY_AGGREGATION = "max"
MBR_NORMALIZATION_METHOD_BERT_RESCORING = "bert_rescoring"

# MBR variant memory formulas:
# - MBR1: SM(c) = agg_{e in E(c)} sim(q, name(e)).
# - MBR2: SM(c) = average of available name, alias, and description channels.
# - MBR_LINEAR: SM(c) = agg_{e in E(c)} sim(q, concat(name, aliases, descriptions)).
# - MBR_POOL: SM(c) = agg over the flat union of name, alias, and description items.


@dataclass(slots=True, frozen=True)
class MBRChunkPoolRecord:
    """Document chunk used as a future MBR retrieval unit.

    `text` is intentionally the original raw chunk text. MBR text scoring must
    not use canonicalized/L0 text.
    """

    chunk_id: str
    text: str
    raw_text: str = ""
    step: int | None = None


@dataclass(slots=True, frozen=True)
class MBRSummaryPoolRecord:
    """Level-1 hierarchical summary available for future MBR matching."""

    summary_id: str
    text: str
    step: int


@dataclass(slots=True, frozen=True)
class MBRLinearMemoryPoolRecord:
    """Legacy per-chunk concatenated memory text kept for old diagnostics."""

    chunk_id: str
    text: str


@dataclass(slots=True)
class MBRMemoryEntityRecord:
    """Final document-memory entity used by structured MBR memory scoring."""

    name: str
    aliases: list[str] = field(default_factory=list)
    descriptions: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MBRDocumentMemoryIndex:
    """Single-document pools and cross-links needed by a future MBR retriever.

    `chunk_pool` contains the main retrieval units.
    `name_pool` contains unique active-entity names only.
    `alias_pool` contains unique active-entity aliases only.
    `description_pool` contains unique active-entity descriptions only.
    `summary_pool` contains only level-1 summaries.

    `chunk_to_summary` includes every chunk in the document. Chunks without an
    available level-1 summary are mapped to `None`.
    """

    doc_id: str
    chunk_pool: list[MBRChunkPoolRecord] = field(default_factory=list)
    linear_memory_pool: list[MBRLinearMemoryPoolRecord] = field(default_factory=list)
    name_pool: list[str] = field(default_factory=list)
    alias_pool: list[str] = field(default_factory=list)
    description_pool: list[str] = field(default_factory=list)
    memory_entities: list[MBRMemoryEntityRecord] = field(default_factory=list)
    summary_pool: list[MBRSummaryPoolRecord] = field(default_factory=list)
    chunk_to_linear_memory: dict[str, str] = field(default_factory=dict)
    chunk_to_names: dict[str, list[str]] = field(default_factory=dict)
    name_to_chunks: dict[str, list[str]] = field(default_factory=dict)
    entity_to_aliases: dict[str, list[str]] = field(default_factory=dict)
    entity_to_descriptions: dict[str, list[str]] = field(default_factory=dict)
    chunk_to_aliases: dict[str, list[str]] = field(default_factory=dict)
    alias_to_chunks: dict[str, list[str]] = field(default_factory=dict)
    chunk_to_descriptions: dict[str, list[str]] = field(default_factory=dict)
    description_to_chunks: dict[str, list[str]] = field(default_factory=dict)
    chunk_to_summary: dict[str, str | None] = field(default_factory=dict)
    summary_to_chunks: dict[str, list[str]] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class MBRChunkScoreRecord:
    """Chunk-aligned score streams before any final MBR combination step."""

    text_score: float
    linear_memory_score: float | None = None
    name_score: float | None = None
    alias_score: float | None = None
    description_score: float | None = None
    memory_score: float | None = None
    summary_score: float | None = None
    memory_effective_weights: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class MBRChunkScoreStreams:
    """Query-time score views over one document's chunk, name, and summary pools."""

    doc_id: str
    query: str
    retriever_name: str = ""
    text_scores: dict[str, float] = field(default_factory=dict)
    linear_memory_scores: dict[str, float] = field(default_factory=dict)
    name_scores: dict[str, float] = field(default_factory=dict)
    alias_scores: dict[str, float] = field(default_factory=dict)
    description_scores: dict[str, float] = field(default_factory=dict)
    summary_scores: dict[str, float] = field(default_factory=dict)
    chunk_scores: dict[str, MBRChunkScoreRecord] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class MBRWeights:
    """Static MBR combination weights before per-chunk renormalization."""

    lambda_t: float = DEFAULT_MBR_LAMBDA_T
    lambda_n: float = DEFAULT_MBR_LAMBDA_M
    lambda_h: float = DEFAULT_MBR_LAMBDA_H


@dataclass(slots=True, frozen=True)
class MBRMemoryWeights:
    """Level-B memory weights for name, alias, and description signals."""

    memory_name_weight: float = DEFAULT_MBR_MEMORY_NAME_WEIGHT
    memory_alias_weight: float = DEFAULT_MBR_MEMORY_ALIAS_WEIGHT
    memory_description_weight: float = DEFAULT_MBR_MEMORY_DESCRIPTION_WEIGHT


@dataclass(slots=True, frozen=True)
class MBRRankedChunkRecord:
    """Final per-chunk MBR score after normalization and weight renormalization."""

    chunk_id: str
    chunk_index: int
    final_score: float
    text_score: float
    linear_memory_score: float | None
    name_score: float | None
    alias_score: float | None
    description_score: float | None
    memory_score: float | None
    summary_score: float | None
    normalized_text_score: float
    normalized_linear_memory_score: float | None
    normalized_name_score: float | None
    normalized_alias_score: float | None
    normalized_description_score: float | None
    normalized_memory_score: float | None
    normalized_summary_score: float | None
    effective_weights: dict[str, float]
    memory_effective_weights: dict[str, float]


@dataclass(slots=True)
class MBRRetrievalResult:
    """Final ranked MBR result for one query over one document."""

    doc_id: str
    query: str
    method_label: str
    normalization_method: str
    weights: MBRWeights
    memory_weights: MBRMemoryWeights = field(default_factory=MBRMemoryWeights)
    ranked_chunks: list[MBRRankedChunkRecord] = field(default_factory=list)


def _read_index_payload(index_path: Path) -> dict[str, Any]:
    with index_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in cache index: {index_path}")
    return payload


def _find_doc_entry(index_payload: dict[str, Any], *, doc_id: str) -> dict[str, Any] | None:
    docs = index_payload.get("docs", []) or []
    if not isinstance(docs, list):
        return None
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        if str(doc.get("doc_id", "") or "") == doc_id:
            return doc
    return None


def _active_entities(record: CanonicalizedChunkRecord) -> list[dict[str, Any]]:
    raw = (record.state_snapshot or {}).get("active_entities", []) or []
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _normalize_memory_text_values(raw: Any) -> list[str]:
    values: list[str] = []
    if raw is None:
        return values
    if isinstance(raw, str):
        text = str(raw).strip()
        return [text] if text else []
    if isinstance(raw, dict):
        for key in ("text", "value", "name"):
            if key in raw:
                return _normalize_memory_text_values(raw.get(key))
        return values
    if isinstance(raw, (list, tuple)):
        for item in raw:
            values.extend(_normalize_memory_text_values(item))
        return values
    text = str(raw or "").strip()
    if text:
        values.append(text)
    return values


def _dedupe_texts(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _memory_entity_from_any(obj: Any) -> MBRMemoryEntityRecord | None:
    if isinstance(obj, MBRMemoryEntityRecord):
        return MBRMemoryEntityRecord(
            name=str(obj.name or "").strip(),
            aliases=_dedupe_texts(list(obj.aliases)),
            descriptions=_dedupe_texts(list(obj.descriptions)),
        )
    if not isinstance(obj, dict):
        return None
    name = str(obj.get("name", "") or "").strip()
    if not name:
        return None
    description_raw = obj.get("description", obj.get("descriptions", []))
    return MBRMemoryEntityRecord(
        name=name,
        aliases=_dedupe_texts(_normalize_memory_text_values(obj.get("aliases", []))),
        descriptions=_dedupe_texts(_normalize_memory_text_values(description_raw)),
    )


def _merge_memory_entities(entities: list[MBRMemoryEntityRecord]) -> list[MBRMemoryEntityRecord]:
    by_name: dict[str, MBRMemoryEntityRecord] = {}
    ordered_names: list[str] = []
    for entity in entities:
        name = str(entity.name or "").strip()
        if not name:
            continue
        if name not in by_name:
            by_name[name] = MBRMemoryEntityRecord(
                name=name,
                aliases=_dedupe_texts(list(entity.aliases)),
                descriptions=_dedupe_texts(list(entity.descriptions)),
            )
            ordered_names.append(name)
            continue
        current = by_name[name]
        current.aliases = _dedupe_texts([*current.aliases, *entity.aliases])
        current.descriptions = _dedupe_texts([*current.descriptions, *entity.descriptions])
    return [by_name[name] for name in ordered_names]


def _memory_entities_from_payload(payload: Any) -> list[MBRMemoryEntityRecord]:
    if isinstance(payload, dict):
        raw = payload.get("entities")
        if raw is None:
            raw = payload.get("entity_memory")
        if raw is None:
            raw = payload.get("memory")
    else:
        raw = payload
    if not isinstance(raw, list):
        return []
    return _merge_memory_entities([e for e in (_memory_entity_from_any(item) for item in raw) if e is not None])


def _final_memory_entities_from_records(records: list[CanonicalizedChunkRecord]) -> list[MBRMemoryEntityRecord]:
    if not records:
        return []
    last_record = max(records, key=lambda row: (int(row.chunk_index), str(row.chunk_id)))
    snapshot_entities = _memory_entities_from_payload((last_record.state_snapshot or {}).get("entity_memory", []))
    if snapshot_entities:
        return snapshot_entities

    # Fallback for older/minimal caches that kept only active entities.
    active_entities: list[MBRMemoryEntityRecord] = []
    for record in sorted(records, key=lambda row: (int(row.chunk_index), str(row.chunk_id))):
        active_entities.extend(
            e
            for e in (_memory_entity_from_any(item) for item in _active_entities(record))
            if e is not None
        )
    return _merge_memory_entities(active_entities)


def _load_final_memory_entities(
    file_map: dict[str, Any],
    *,
    records: list[CanonicalizedChunkRecord],
    base_dir: Path,
) -> list[MBRMemoryEntityRecord]:
    memory_ref = file_map.get("memory") or file_map.get("entity_memory") or file_map.get("memory_json")
    if memory_ref:
        memory_path = _resolve_cache_file_path(memory_ref, base_dir=base_dir)
        if memory_path.exists():
            try:
                with memory_path.open("r", encoding="utf-8") as handle:
                    entities = _memory_entities_from_payload(json.load(handle))
                if entities:
                    return entities
            except Exception:
                pass
    return _final_memory_entities_from_records(records)


def _extract_active_names(record: CanonicalizedChunkRecord) -> list[str]:
    raw = _active_entities(record)
    names: list[str] = []
    seen: set[str] = set()
    for item in raw:
        name = str(item.get("name", "") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def _extract_active_aliases(record: CanonicalizedChunkRecord) -> list[str]:
    aliases: list[str] = []
    seen: set[str] = set()
    for item in _active_entities(record):
        for alias in _normalize_memory_text_values(item.get("aliases")):
            if alias in seen:
                continue
            seen.add(alias)
            aliases.append(alias)
    return aliases


def _extract_active_descriptions(record: CanonicalizedChunkRecord) -> list[str]:
    descriptions: list[str] = []
    seen: set[str] = set()
    for item in _active_entities(record):
        for description in _normalize_memory_text_values(item.get("description")):
            if description in seen:
                continue
            seen.add(description)
            descriptions.append(description)
    return descriptions


def _build_linear_memory_text(
    names: list[str],
    entity_to_aliases: dict[str, list[str]],
    entity_to_descriptions: dict[str, list[str]],
) -> str:
    """Build deterministic legacy per-chunk memory text.

    Current MBR_LINEAR scoring builds one concatenated text per entity at
    retrieval time; this helper is retained only for existing cache diagnostics.
    """

    ordered_values: list[str] = []
    for name in names:
        ordered_values.extend(
            [
                name,
                *entity_to_aliases.get(name, []),
                *entity_to_descriptions.get(name, []),
            ]
        )
    return "\n".join(_dedupe_texts(ordered_values))


def _read_level_rows(path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    for row in read_jsonl(path):
        if not isinstance(row, dict):
            continue
        try:
            step = int(row.get("step"))
        except Exception:
            continue
        chunk = row.get("chunk")
        if chunk is None:
            continue
        out[int(step)] = str(chunk)
    return out


def _infer_level1_cycle_size(index_payload: dict[str, Any], records: list[CanonicalizedChunkRecord]) -> int | None:
    raw_b = index_payload.get(
        "branching_factor",
        index_payload.get("hierarchy_branching_factor", index_payload.get("hierarchy_k", index_payload.get("k"))),
    )
    try:
        b = int(raw_b)
    except Exception:
        b = None
    if isinstance(b, int) and b > 1:
        return int(b)

    cycle_sizes: set[int] = set()
    for record in records:
        raw = (record.state_snapshot or {}).get("hierarchy_summaries", {}) or {}
        if not isinstance(raw, dict):
            continue
        for cycle_key in raw:
            try:
                cycle = int(cycle_key)
            except Exception:
                continue
            if cycle > 1:
                cycle_sizes.add(cycle)
    return min(cycle_sizes) if cycle_sizes else None


def _infer_cycle_size_from_summary_steps(chunk_steps: list[int], summary_steps: list[int]) -> int | None:
    if not chunk_steps or not summary_steps:
        return None
    first_summary_step = min(int(step) for step in summary_steps)
    inferred = len([step for step in chunk_steps if int(step) <= first_summary_step])
    return inferred if inferred > 1 else None


def _level1_source_steps(summary_step: int, cycle_size: int) -> list[int]:
    """Return the exact chunk indices covered by one level-1 summary.

    A level-1 summary produced at `summary_step` with branching factor `b`
    covers the chunk-index interval:

    [summary_step - b + 1, ..., summary_step]
    """

    if cycle_size <= 1:
        return []
    start = int(summary_step) - int(cycle_size) + 1
    if start < 0:
        return []
    return list(range(start, int(summary_step) + 1))


def _load_doc_records(
    *,
    index_path: Path,
    index_payload: dict[str, Any],
    doc_entry: dict[str, Any],
    doc_id: str,
) -> list[CanonicalizedChunkRecord]:
    file_map = doc_entry.get("files", {}) or {}
    if isinstance(file_map, dict):
        records = _load_v2_doc_records(
            doc_id=doc_id,
            file_map=file_map,
            hierarchy_branching_factor=_infer_level1_cycle_size(index_payload, []),
            strategy_name_fallback=str(index_payload.get("strategy_name", "") or ""),
            base_dir=index_path.parent,
        )
        if records:
            records.sort(key=lambda row: (int(row.chunk_index), str(row.chunk_id)))
            return records

    fallback = [
        row
        for row in load_canonicalized_chunk_records_from_cache_index(index_path)
        if str(row.doc_id) == doc_id
    ]
    fallback.sort(key=lambda row: (int(row.chunk_index), str(row.chunk_id)))
    return fallback


def build_mbr_document_memory_index(index_path: str | Path, *, doc_id: str) -> MBRDocumentMemoryIndex:
    """Build chunk/name/summary pools plus mappings for one cached document.

    Parameters
    ----------
    index_path:
        Path to the existing cache `index.json`.
    doc_id:
        The processed document identifier to restore.
    """

    path = Path(index_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Cache index not found: {path}")

    index_payload = _read_index_payload(path)
    doc_entry = _find_doc_entry(index_payload, doc_id=doc_id)
    if doc_entry is None:
        raise KeyError(f"Document {doc_id!r} not found in cache index: {path}")

    records = _load_doc_records(index_path=path, index_payload=index_payload, doc_entry=doc_entry, doc_id=doc_id)
    if not records:
        raise ValueError(f"No canonicalized chunk records could be restored for document {doc_id!r}")

    chunk_pool = [
        MBRChunkPoolRecord(
            chunk_id=str(record.chunk_id),
            text=str(record.raw_chunk_text or ""),
            raw_text=str(record.raw_chunk_text or ""),
            step=int(record.chunk_index),
        )
        for record in records
    ]

    chunk_to_names: dict[str, list[str]] = {}
    chunk_to_linear_memory: dict[str, str] = {}
    name_to_chunks_lists: defaultdict[str, list[str]] = defaultdict(list)
    chunk_to_aliases: dict[str, list[str]] = {}
    alias_to_chunks_lists: defaultdict[str, list[str]] = defaultdict(list)
    chunk_to_descriptions: dict[str, list[str]] = {}
    description_to_chunks_lists: defaultdict[str, list[str]] = defaultdict(list)
    file_map = doc_entry.get("files", {}) or {}
    if not isinstance(file_map, dict):
        file_map = {}

    memory_entities = _load_final_memory_entities(file_map, records=records, base_dir=path.parent)
    memory_by_name = {entity.name: entity for entity in memory_entities if str(entity.name or "").strip()}
    entity_to_aliases = {
        entity.name: _dedupe_texts(list(entity.aliases))
        for entity in memory_entities
        if str(entity.name or "").strip()
    }
    entity_to_descriptions = {
        entity.name: _dedupe_texts(list(entity.descriptions))
        for entity in memory_entities
        if str(entity.name or "").strip()
    }

    name_pool = [entity.name for entity in memory_entities if str(entity.name or "").strip()]
    alias_pool = _dedupe_texts(
        [
            alias
            for entity in memory_entities
            for alias in entity_to_aliases.get(entity.name, [])
        ]
    )
    description_pool = _dedupe_texts(
        [
            description
            for entity in memory_entities
            for description in entity_to_descriptions.get(entity.name, [])
        ]
    )

    for record in records:
        chunk_id = str(record.chunk_id)
        active_names = _extract_active_names(record)
        names = [name for name in active_names if name in memory_by_name]
        aliases = _dedupe_texts(
            [
                alias
                for name in names
                for alias in entity_to_aliases.get(name, [])
            ]
        )
        descriptions = _dedupe_texts(
            [
                description
                for name in names
                for description in entity_to_descriptions.get(name, [])
            ]
        )
        chunk_to_linear_memory[chunk_id] = _build_linear_memory_text(
            names,
            entity_to_aliases,
            entity_to_descriptions,
        )
        chunk_to_names[chunk_id] = names
        chunk_to_aliases[chunk_id] = aliases
        chunk_to_descriptions[chunk_id] = descriptions
        for name in names:
            name_to_chunks_lists[name].append(chunk_id)
        for alias in aliases:
            alias_to_chunks_lists[alias].append(chunk_id)
        for description in descriptions:
            description_to_chunks_lists[description].append(chunk_id)

    chunk_to_summary: dict[str, str | None] = {str(record.chunk_id): None for record in records}
    summary_to_chunks: dict[str, list[str]] = {}
    summary_pool: list[MBRSummaryPoolRecord] = []

    cycle_size = _infer_level1_cycle_size(index_payload, records)
    if isinstance(file_map, dict):
        l1_ref = file_map.get("L_1")
        if l1_ref:
            l1_path = _resolve_cache_file_path(l1_ref, base_dir=path.parent)
            if l1_path.exists():
                summaries_by_step = _read_level_rows(l1_path)
                chunk_id_by_step = {int(record.chunk_index): str(record.chunk_id) for record in records}
                if not cycle_size:
                    cycle_size = _infer_cycle_size_from_summary_steps(sorted(chunk_id_by_step), sorted(summaries_by_step))
                if cycle_size and cycle_size > 1:
                    for step, summary_text in sorted(summaries_by_step.items()):
                        # A level-1 summary at `step` covers exactly the chunk-index
                        # interval [step - k + 1, ..., step], where k is the hierarchy
                        # branching factor. This makes the chunk<->summary mapping
                        # deterministic and independent of incidental row ordering.
                        source_steps = _level1_source_steps(int(step), int(cycle_size))
                        if len(source_steps) != cycle_size:
                            continue
                        if any(source_step not in chunk_id_by_step for source_step in source_steps):
                            continue
                        summary_id = f"{doc_id}:H1:{int(step)}"
                        source_chunk_ids: list[str] = []
                        for source_step in source_steps:
                            chunk_id = chunk_id_by_step[source_step]
                            previous = chunk_to_summary.get(chunk_id)
                            if previous is not None and previous != summary_id:
                                raise ValueError(
                                    f"Chunk {chunk_id!r} maps to multiple level-1 summaries: "
                                    f"{previous!r} and {summary_id!r}"
                                )
                            chunk_to_summary[chunk_id] = summary_id
                            source_chunk_ids.append(chunk_id)
                        summary_pool.append(
                            MBRSummaryPoolRecord(
                                summary_id=summary_id,
                                text=str(summary_text or ""),
                                step=int(step),
                            )
                        )
                        summary_to_chunks[summary_id] = source_chunk_ids

    linear_memory_pool = [
        MBRLinearMemoryPoolRecord(
            chunk_id=chunk.chunk_id,
            text=str(chunk_to_linear_memory.get(chunk.chunk_id, "") or ""),
        )
        for chunk in chunk_pool
    ]

    return MBRDocumentMemoryIndex(
        doc_id=doc_id,
        chunk_pool=chunk_pool,
        linear_memory_pool=linear_memory_pool,
        name_pool=name_pool,
        alias_pool=alias_pool,
        description_pool=description_pool,
        memory_entities=memory_entities,
        summary_pool=summary_pool,
        chunk_to_linear_memory=chunk_to_linear_memory,
        chunk_to_names=chunk_to_names,
        name_to_chunks=dict(name_to_chunks_lists),
        entity_to_aliases=entity_to_aliases,
        entity_to_descriptions=entity_to_descriptions,
        chunk_to_aliases=chunk_to_aliases,
        alias_to_chunks=dict(alias_to_chunks_lists),
        chunk_to_descriptions=chunk_to_descriptions,
        description_to_chunks=dict(description_to_chunks_lists),
        chunk_to_summary=chunk_to_summary,
        summary_to_chunks=summary_to_chunks,
    )


def _score_pool(
    texts: list[str],
    *,
    ids: list[str],
    query: str,
    retriever_name: str,
    retriever_kwargs: dict[str, Any] | None = None,
) -> dict[str, float]:
    if not texts:
        return {}
    encoded_pool = build_encoded_text_pool(
        texts,
        retriever_name=retriever_name,
        chunk_ids=ids,
        retriever_kwargs=retriever_kwargs,
    )
    query_representation = encode_query(query, encoded_pool.retriever)
    scores = compute_similarity(
        query_representation,
        encoded_pool.document_representations,
        encoded_pool.retriever,
    )
    return {
        str(item_id): float(score)
        for item_id, score in zip(encoded_pool.chunk_ids, scores, strict=False)
    }


def _normalize_float(value: Any, *, field_name: str) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise ValueError(f"{field_name} must be a float (got: {value!r})") from exc
    if out < 0:
        raise ValueError(f"{field_name} must be >= 0 (got: {out})")
    return out


def normalize_mbr_weights(
    *,
    lambda_t: float = DEFAULT_MBR_LAMBDA_T,
    lambda_n: float = DEFAULT_MBR_LAMBDA_M,
    lambda_h: float = DEFAULT_MBR_LAMBDA_H,
) -> MBRWeights:
    return MBRWeights(
        lambda_t=_normalize_float(lambda_t, field_name="lambda_t"),
        lambda_n=_normalize_float(lambda_n, field_name="lambda_n"),
        lambda_h=_normalize_float(lambda_h, field_name="lambda_h"),
    )


def normalize_mbr_memory_weights(
    *,
    memory_name_weight: float = DEFAULT_MBR_MEMORY_NAME_WEIGHT,
    memory_alias_weight: float = DEFAULT_MBR_MEMORY_ALIAS_WEIGHT,
    memory_description_weight: float = DEFAULT_MBR_MEMORY_DESCRIPTION_WEIGHT,
) -> MBRMemoryWeights:
    return MBRMemoryWeights(
        memory_name_weight=_normalize_float(memory_name_weight, field_name="memory_name_weight"),
        memory_alias_weight=_normalize_float(memory_alias_weight, field_name="memory_alias_weight"),
        memory_description_weight=_normalize_float(memory_description_weight, field_name="memory_description_weight"),
    )


def normalize_mbr_variant(value: str | None = None) -> str:
    raw = str(value or MBR_VARIANT_MBR1).strip()
    key = raw.upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "MBR": MBR_VARIANT_MBR1,
        "MBR1": MBR_VARIANT_MBR1,
        "MBR_1": MBR_VARIANT_MBR1,
        "MBR2": MBR_VARIANT_MBR2,
        "MBR_2": MBR_VARIANT_MBR2,
        "MBR_LINEAR": MBR_VARIANT_LINEAR,
        "LINEAR": MBR_VARIANT_LINEAR,
        "MBRLINEAR": MBR_VARIANT_LINEAR,
        "MBR_POOL": MBR_VARIANT_POOL,
        "MBR_ONEPOOL": MBR_VARIANT_POOL,
        "MBR_ONE_POOL": MBR_VARIANT_POOL,
        "ONEPOOL": MBR_VARIANT_POOL,
        "ONE_POOL": MBR_VARIANT_POOL,
        "POOL": MBR_VARIANT_POOL,
    }
    variant = aliases.get(key, key)
    if variant not in MBR_VARIANTS:
        raise ValueError(
            f"Invalid MBR variant: {value!r}. Expected one of: "
            f"{', '.join(sorted(MBR_VARIANTS))}"
        )
    return variant


def format_mbr_method_label(
    *,
    lambda_t: float = DEFAULT_MBR_LAMBDA_T,
    lambda_n: float = DEFAULT_MBR_LAMBDA_M,
    lambda_h: float = DEFAULT_MBR_LAMBDA_H,
    normalization_method: str = "none",
    memory_name_weight: float = DEFAULT_MBR_MEMORY_NAME_WEIGHT,
    memory_alias_weight: float = DEFAULT_MBR_MEMORY_ALIAS_WEIGHT,
    memory_description_weight: float = DEFAULT_MBR_MEMORY_DESCRIPTION_WEIGHT,
) -> str:
    weights = normalize_mbr_weights(lambda_t=lambda_t, lambda_n=lambda_n, lambda_h=lambda_h)
    memory_weights = normalize_mbr_memory_weights(
        memory_name_weight=memory_name_weight,
        memory_alias_weight=memory_alias_weight,
        memory_description_weight=memory_description_weight,
    )
    norm = normalize_mbr_normalization_method(normalization_method)
    label = (
        f"mbr_"
        f"lt{weights.lambda_t:g}_"
        f"ln{weights.lambda_n:g}_"
        f"lh{weights.lambda_h:g}_"
        f"norm_{norm}"
    )
    if memory_weights != MBRMemoryWeights():
        label += (
            f"_mnw{memory_weights.memory_name_weight:g}"
            f"_maw{memory_weights.memory_alias_weight:g}"
            f"_mdw{memory_weights.memory_description_weight:g}"
        )
    return label


def format_mbr_linear_method_label(
    *,
    lambda_t: float = DEFAULT_MBR_LAMBDA_T,
    lambda_n: float = DEFAULT_MBR_LAMBDA_M,
    lambda_h: float = DEFAULT_MBR_LAMBDA_H,
    normalization_method: str = "none",
) -> str:
    weights = normalize_mbr_weights(lambda_t=lambda_t, lambda_n=lambda_n, lambda_h=lambda_h)
    norm = normalize_mbr_normalization_method(normalization_method)
    return (
        f"mbr_linear_"
        f"lt{weights.lambda_t:g}_"
        f"ln{weights.lambda_n:g}_"
        f"lh{weights.lambda_h:g}_"
        f"norm_{norm}"
    )


def format_mbr_pool_method_label(
    *,
    lambda_t: float = DEFAULT_MBR_LAMBDA_T,
    lambda_n: float = DEFAULT_MBR_LAMBDA_M,
    lambda_h: float = DEFAULT_MBR_LAMBDA_H,
    normalization_method: str = "none",
) -> str:
    weights = normalize_mbr_weights(lambda_t=lambda_t, lambda_n=lambda_n, lambda_h=lambda_h)
    norm = normalize_mbr_normalization_method(normalization_method)
    return (
        f"mbr_pool_"
        f"lt{weights.lambda_t:g}_"
        f"ln{weights.lambda_n:g}_"
        f"lh{weights.lambda_h:g}_"
        f"norm_{norm}"
    )


def format_mbr_variant_method_label(
    *,
    variant: str,
    lambda_t: float = DEFAULT_MBR_LAMBDA_T,
    lambda_n: float = DEFAULT_MBR_LAMBDA_M,
    lambda_h: float = DEFAULT_MBR_LAMBDA_H,
    normalization_method: str = "none",
    memory_name_weight: float = DEFAULT_MBR2_MEMORY_NAME_WEIGHT,
    memory_alias_weight: float = DEFAULT_MBR2_MEMORY_ALIAS_WEIGHT,
    memory_description_weight: float = DEFAULT_MBR2_MEMORY_DESCRIPTION_WEIGHT,
) -> str:
    normalized_variant = normalize_mbr_variant(variant)
    if normalized_variant == MBR_VARIANT_LINEAR:
        return format_mbr_linear_method_label(
            lambda_t=lambda_t,
            lambda_n=lambda_n,
            lambda_h=lambda_h,
            normalization_method=normalization_method,
        )
    if normalized_variant == MBR_VARIANT_POOL:
        return format_mbr_pool_method_label(
            lambda_t=lambda_t,
            lambda_n=lambda_n,
            lambda_h=lambda_h,
            normalization_method=normalization_method,
        )
    if normalized_variant == MBR_VARIANT_MBR1:
        return format_mbr_method_label(
            lambda_t=lambda_t,
            lambda_n=lambda_n,
            lambda_h=lambda_h,
            normalization_method=normalization_method,
            memory_name_weight=DEFAULT_MBR_MEMORY_NAME_WEIGHT,
            memory_alias_weight=DEFAULT_MBR_MEMORY_ALIAS_WEIGHT,
            memory_description_weight=DEFAULT_MBR_MEMORY_DESCRIPTION_WEIGHT,
        )
    return format_mbr_method_label(
        lambda_t=lambda_t,
        lambda_n=lambda_n,
        lambda_h=lambda_h,
        normalization_method=normalization_method,
        memory_name_weight=memory_name_weight,
        memory_alias_weight=memory_alias_weight,
        memory_description_weight=memory_description_weight,
    )


def normalize_mbr_normalization_method(value: str | None = None) -> str:
    """Return the canonical MBR normalization method key."""

    key = str(value or "none").strip().lower().replace("-", "_") or "none"
    aliases = {
        "min_max": "minmax",
        "z_score": "zscore",
        "bert_rescore": MBR_NORMALIZATION_METHOD_BERT_RESCORING,
        "bertrescore": MBR_NORMALIZATION_METHOD_BERT_RESCORING,
        "bert_reranking": MBR_NORMALIZATION_METHOD_BERT_RESCORING,
        "bert_rescoring": MBR_NORMALIZATION_METHOD_BERT_RESCORING,
    }
    return aliases.get(key, key)


def _uses_empirical_bert_rescoring_max(retriever_name: str | None) -> bool:
    normalized = str(retriever_name or "").strip()
    if not normalized:
        raise ValueError("bert_rescoring normalization requires retriever_name so M can be selected.")
    Retriever = get_retriever(normalized)
    return issubclass(Retriever, BM25Retriever)


def normalize_signal_scores(
    scores: dict[str, float],
    *,
    method: str = "none",
    retriever_name: str | None = None,
) -> dict[str, float]:
    """Normalize present signal scores across the full candidate set."""

    normalized_method = normalize_mbr_normalization_method(method)
    if not scores:
        return {}
    if normalized_method in {"none", "identity", "raw"}:
        return {
            str(key): float(value)
            for key, value in scores.items()
        }
    values = [float(value) for value in scores.values()]
    if normalized_method == "minmax":
        min_value = min(values)
        max_value = max(values)
        spread = max_value - min_value
        if spread == 0:
            return {str(key): 0.0 for key in scores}
        return {
            str(key): float((float(value) - min_value) / spread)
            for key, value in scores.items()
        }
    if normalized_method == "zscore":
        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        std_value = variance ** 0.5
        if std_value == 0:
            return {str(key): 0.0 for key in scores}
        return {
            str(key): float((float(value) - mean_value) / std_value)
            for key, value in scores.items()
        }
    if normalized_method == MBR_NORMALIZATION_METHOD_BERT_RESCORING:
        mean_value = sum(values) / len(values)
        max_value = max(values) if _uses_empirical_bert_rescoring_max(retriever_name) else 1.0
        denominator = max_value - mean_value
        if denominator == 0:
            return {str(key): 0.0 for key in scores}
        return {
            str(key): float((float(value) - mean_value) / denominator)
            for key, value in scores.items()
        }
    if normalized_method not in {"none", "identity", "raw", "minmax", "zscore", MBR_NORMALIZATION_METHOD_BERT_RESCORING}:
        raise ValueError(f"Unsupported MBR normalization method: {method!r}")
    return {}


def _aggregate_memory_values(
    values: list[float],
    *,
    aggregation: str = DEFAULT_MBR_MEMORY_AGGREGATION,
) -> float | None:
    numbers = [float(value) for value in values]
    if not numbers:
        return None
    method = str(aggregation or DEFAULT_MBR_MEMORY_AGGREGATION).strip().lower()
    if method not in {"max", "mean"}:
        raise ValueError(f"Unsupported MBR memory aggregation: {aggregation!r}")
    return max(numbers) if method == "max" else sum(numbers) / len(numbers)


def _chunk_aliases_for_names(
    document_index: MBRDocumentMemoryIndex,
    *,
    chunk_id: str,
    linked_names: list[str],
) -> list[str]:
    if document_index.entity_to_aliases:
        return _dedupe_texts(
            [
                alias
                for name in linked_names
                for alias in document_index.entity_to_aliases.get(name, [])
            ]
        )
    return list(document_index.chunk_to_aliases.get(chunk_id, []) or [])


def _chunk_descriptions_for_names(
    document_index: MBRDocumentMemoryIndex,
    *,
    chunk_id: str,
    linked_names: list[str],
) -> list[str]:
    if document_index.entity_to_descriptions:
        return _dedupe_texts(
            [
                description
                for name in linked_names
                for description in document_index.entity_to_descriptions.get(name, [])
            ]
        )
    return list(document_index.chunk_to_descriptions.get(chunk_id, []) or [])


def _entity_concat_texts(document_index: MBRDocumentMemoryIndex) -> dict[str, str]:
    names = [
        entity.name
        for entity in document_index.memory_entities
        if str(entity.name or "").strip()
    ]
    if not names:
        names = [name for name in document_index.name_pool if str(name or "").strip()]

    texts: dict[str, str] = {}
    for name in names:
        text = "; ".join(
            _dedupe_texts(
                [
                    str(name),
                    *document_index.entity_to_aliases.get(name, []),
                    *document_index.entity_to_descriptions.get(name, []),
                ]
            )
        )
        if text.strip():
            texts[str(name)] = text
    return texts


def _variant_memory_weights(
    *,
    variant: str,
    memory_name_weight: float,
    memory_alias_weight: float,
    memory_description_weight: float,
) -> MBRMemoryWeights:
    normalized_variant = normalize_mbr_variant(variant)
    if normalized_variant == MBR_VARIANT_MBR1:
        return MBRMemoryWeights()
    if normalized_variant == MBR_VARIANT_MBR2:
        return normalize_mbr_memory_weights(
            memory_name_weight=memory_name_weight,
            memory_alias_weight=memory_alias_weight,
            memory_description_weight=memory_description_weight,
        )
    return MBRMemoryWeights(
        memory_name_weight=0.0,
        memory_alias_weight=0.0,
        memory_description_weight=0.0,
    )


def _variant_from_memory_weights(
    *,
    memory_name_weight: float,
    memory_alias_weight: float,
    memory_description_weight: float,
) -> str:
    memory_weights = normalize_mbr_memory_weights(
        memory_name_weight=memory_name_weight,
        memory_alias_weight=memory_alias_weight,
        memory_description_weight=memory_description_weight,
    )
    if (
        memory_weights.memory_name_weight == DEFAULT_MBR_MEMORY_NAME_WEIGHT
        and memory_weights.memory_alias_weight == DEFAULT_MBR_MEMORY_ALIAS_WEIGHT
        and memory_weights.memory_description_weight == DEFAULT_MBR_MEMORY_DESCRIPTION_WEIGHT
    ):
        return MBR_VARIANT_MBR1
    return MBR_VARIANT_MBR2


def score_mbr_variant_document_pools(
    document_index: MBRDocumentMemoryIndex,
    *,
    query: str,
    retriever_name: str,
    variant: str = MBR_VARIANT_MBR1,
    retriever_kwargs: dict[str, Any] | None = None,
    memory_name_weight: float = DEFAULT_MBR2_MEMORY_NAME_WEIGHT,
    memory_alias_weight: float = DEFAULT_MBR2_MEMORY_ALIAS_WEIGHT,
    memory_description_weight: float = DEFAULT_MBR2_MEMORY_DESCRIPTION_WEIGHT,
    memory_aggregation: str = DEFAULT_MBR_MEMORY_AGGREGATION,
) -> MBRChunkScoreStreams:
    """Score query/pools once and build chunk-aligned streams for one MBR variant."""

    normalized_variant = normalize_mbr_variant(variant)
    memory_weights = _variant_memory_weights(
        variant=normalized_variant,
        memory_name_weight=memory_name_weight,
        memory_alias_weight=memory_alias_weight,
        memory_description_weight=memory_description_weight,
    )
    chunk_ids = [row.chunk_id for row in document_index.chunk_pool]
    chunk_texts = [row.text for row in document_index.chunk_pool]
    text_scores = _score_pool(
        chunk_texts,
        ids=chunk_ids,
        query=query,
        retriever_name=retriever_name,
        retriever_kwargs=retriever_kwargs,
    )

    name_scores: dict[str, float] = {}
    alias_scores: dict[str, float] = {}
    description_scores: dict[str, float] = {}
    linear_memory_scores: dict[str, float] = {}
    entity_concat_scores: dict[str, float] = {}
    if normalized_variant in {MBR_VARIANT_MBR1, MBR_VARIANT_MBR2, MBR_VARIANT_POOL}:
        name_scores = _score_pool(
            list(document_index.name_pool),
            ids=list(document_index.name_pool),
            query=query,
            retriever_name=retriever_name,
            retriever_kwargs=retriever_kwargs,
        )
        alias_scores = _score_pool(
            list(document_index.alias_pool),
            ids=list(document_index.alias_pool),
            query=query,
            retriever_name=retriever_name,
            retriever_kwargs=retriever_kwargs,
        )
        description_scores = _score_pool(
            list(document_index.description_pool),
            ids=list(document_index.description_pool),
            query=query,
            retriever_name=retriever_name,
            retriever_kwargs=retriever_kwargs,
        )
    elif normalized_variant == MBR_VARIANT_LINEAR:
        entity_concat_texts = _entity_concat_texts(document_index)
        entity_concat_scores = _score_pool(
            list(entity_concat_texts.values()),
            ids=list(entity_concat_texts.keys()),
            query=query,
            retriever_name=retriever_name,
            retriever_kwargs=retriever_kwargs,
        )

    summary_ids = [row.summary_id for row in document_index.summary_pool]
    summary_texts = [row.text for row in document_index.summary_pool]
    summary_scores = _score_pool(
        summary_texts,
        ids=summary_ids,
        query=query,
        retriever_name=retriever_name,
        retriever_kwargs=retriever_kwargs,
    )

    chunk_scores: dict[str, MBRChunkScoreRecord] = {}
    for chunk in document_index.chunk_pool:
        chunk_id = str(chunk.chunk_id)
        linked_names = document_index.chunk_to_names.get(chunk_id, []) or []
        linked_name_scores = [
            float(name_scores[name])
            for name in linked_names
            if name in name_scores
        ]
        linked_aliases = _chunk_aliases_for_names(document_index, chunk_id=chunk_id, linked_names=linked_names)
        linked_alias_scores = [
            float(alias_scores[alias])
            for alias in linked_aliases
            if alias in alias_scores
        ]
        linked_descriptions = _chunk_descriptions_for_names(document_index, chunk_id=chunk_id, linked_names=linked_names)
        linked_description_scores = [
            float(description_scores[description])
            for description in linked_descriptions
            if description in description_scores
        ]
        name_score = _aggregate_memory_values(linked_name_scores, aggregation=memory_aggregation)
        alias_score = _aggregate_memory_values(linked_alias_scores, aggregation=memory_aggregation)
        description_score = _aggregate_memory_values(linked_description_scores, aggregation=memory_aggregation)

        linear_memory_score: float | None = None
        memory_effective_weights: dict[str, float] = {}
        memory_score: float | None = None
        if normalized_variant == MBR_VARIANT_MBR1:
            memory_score = name_score
            memory_effective_weights = {"name": 1.0} if memory_score is not None else {}
        elif normalized_variant == MBR_VARIANT_MBR2:
            memory_components: list[tuple[str, float, float]] = []
            if name_score is not None and float(memory_weights.memory_name_weight) > 0:
                memory_components.append(("name", float(memory_weights.memory_name_weight), float(name_score)))
            if alias_score is not None and float(memory_weights.memory_alias_weight) > 0:
                memory_components.append(("alias", float(memory_weights.memory_alias_weight), float(alias_score)))
            if description_score is not None and float(memory_weights.memory_description_weight) > 0:
                memory_components.append(("description", float(memory_weights.memory_description_weight), float(description_score)))
            memory_weight_sum = sum(weight for _, weight, _ in memory_components)
            if memory_components and memory_weight_sum > 0:
                memory_effective_weights = {
                    component_name: float(weight / memory_weight_sum)
                    for component_name, weight, _ in memory_components
                }
                memory_score = sum(
                    float(memory_effective_weights[component_name]) * float(component_score)
                    for component_name, _, component_score in memory_components
                )
        elif normalized_variant == MBR_VARIANT_POOL:
            memory_score = _aggregate_memory_values(
                [*linked_name_scores, *linked_alias_scores, *linked_description_scores],
                aggregation=memory_aggregation,
            )
            memory_effective_weights = {"pool": 1.0} if memory_score is not None else {}
        elif normalized_variant == MBR_VARIANT_LINEAR:
            linked_concat_scores = [
                float(entity_concat_scores[name])
                for name in linked_names
                if name in entity_concat_scores
            ]
            linear_memory_score = _aggregate_memory_values(linked_concat_scores, aggregation=memory_aggregation)
            if linear_memory_score is not None:
                linear_memory_scores[chunk_id] = float(linear_memory_score)
                memory_score = float(linear_memory_score)
                memory_effective_weights = {"entity_concat": 1.0}

        summary_id = document_index.chunk_to_summary.get(chunk_id)
        summary_score = float(summary_scores[summary_id]) if summary_id in summary_scores else None

        chunk_scores[chunk_id] = MBRChunkScoreRecord(
            text_score=float(text_scores.get(chunk_id, 0.0)),
            linear_memory_score=linear_memory_score,
            name_score=name_score,
            alias_score=alias_score,
            description_score=description_score,
            memory_score=memory_score,
            summary_score=summary_score,
            memory_effective_weights=memory_effective_weights,
        )

    return MBRChunkScoreStreams(
        doc_id=document_index.doc_id,
        query=str(query or ""),
        retriever_name=str(retriever_name or ""),
        text_scores=text_scores,
        linear_memory_scores=linear_memory_scores,
        name_scores=name_scores,
        alias_scores=alias_scores,
        description_scores=description_scores,
        summary_scores=summary_scores,
        chunk_scores=chunk_scores,
    )


def combine_mbr_variant_chunk_scores(
    document_index: MBRDocumentMemoryIndex,
    chunk_scores: MBRChunkScoreStreams,
    *,
    variant: str = MBR_VARIANT_MBR1,
    lambda_t: float = DEFAULT_MBR_LAMBDA_T,
    lambda_n: float = DEFAULT_MBR_LAMBDA_M,
    lambda_h: float = DEFAULT_MBR_LAMBDA_H,
    normalization_method: str = "none",
    memory_name_weight: float = DEFAULT_MBR2_MEMORY_NAME_WEIGHT,
    memory_alias_weight: float = DEFAULT_MBR2_MEMORY_ALIAS_WEIGHT,
    memory_description_weight: float = DEFAULT_MBR2_MEMORY_DESCRIPTION_WEIGHT,
) -> MBRRetrievalResult:
    """Combine text, memory, and summary streams with one shared outer fusion path.

    Order of operations for normalized variants:
    1. Compute raw ST/SN/SA/SD/SH or ST/SM_raw/SH upstream.
    2. Normalize each present base channel across all candidate chunks:
       RAW: X, MINMAX: (X-min)/(max-min), ZSCORE: (X-mean)/std,
       BERT_RESCORING: (X-mean)/(M-mean), where M is 1.0 for embedding
       retrievers and the empirical max for BM25.
    3. Build SM from normalized channels for MBR2, or normalized SM_raw for Linear.
    4. Apply the shared outer fusion with per-chunk missing-signal renormalization.
    """

    normalized_variant = normalize_mbr_variant(variant)
    weights = normalize_mbr_weights(lambda_t=lambda_t, lambda_n=lambda_n, lambda_h=lambda_h)
    memory_weights = _variant_memory_weights(
        variant=normalized_variant,
        memory_name_weight=memory_name_weight,
        memory_alias_weight=memory_alias_weight,
        memory_description_weight=memory_description_weight,
    )
    applied_normalization_method = (
        normalize_mbr_normalization_method(normalization_method)
        if normalized_variant in {MBR_VARIANT_MBR2, MBR_VARIANT_LINEAR}
        else "none"
    )
    normalization_retriever_name = str(chunk_scores.retriever_name or "")
    normalized_text = normalize_signal_scores(
        chunk_scores.text_scores,
        method=applied_normalization_method,
        retriever_name=normalization_retriever_name,
    )
    normalized_linear_memory = normalize_signal_scores(
        {
            chunk_id: row.linear_memory_score
            for chunk_id, row in chunk_scores.chunk_scores.items()
            if row.linear_memory_score is not None
        },
        method=applied_normalization_method,
        retriever_name=normalization_retriever_name,
    )
    normalized_name = normalize_signal_scores(
        {chunk_id: row.name_score for chunk_id, row in chunk_scores.chunk_scores.items() if row.name_score is not None},
        method=applied_normalization_method,
        retriever_name=normalization_retriever_name,
    )
    normalized_alias = normalize_signal_scores(
        {chunk_id: row.alias_score for chunk_id, row in chunk_scores.chunk_scores.items() if row.alias_score is not None},
        method=applied_normalization_method,
        retriever_name=normalization_retriever_name,
    )
    normalized_description = normalize_signal_scores(
        {
            chunk_id: row.description_score
            for chunk_id, row in chunk_scores.chunk_scores.items()
            if row.description_score is not None
        },
        method=applied_normalization_method,
        retriever_name=normalization_retriever_name,
    )
    normalized_memory = normalize_signal_scores(
        {chunk_id: row.memory_score for chunk_id, row in chunk_scores.chunk_scores.items() if row.memory_score is not None},
        method="none",
        retriever_name=normalization_retriever_name,
    )
    normalized_summary = normalize_signal_scores(
        {chunk_id: row.summary_score for chunk_id, row in chunk_scores.chunk_scores.items() if row.summary_score is not None},
        method=applied_normalization_method,
        retriever_name=normalization_retriever_name,
    )

    chunk_order = {row.chunk_id: idx for idx, row in enumerate(document_index.chunk_pool)}
    ranked_rows: list[MBRRankedChunkRecord] = []
    for chunk in document_index.chunk_pool:
        raw_scores = chunk_scores.chunk_scores[str(chunk.chunk_id)]
        normalized_memory_score = normalized_memory.get(chunk.chunk_id)
        normalized_memory_effective_weights = dict(raw_scores.memory_effective_weights)
        if normalized_variant == MBR_VARIANT_MBR2:
            memory_components: list[tuple[str, float, float]] = []
            if raw_scores.name_score is not None and chunk.chunk_id in normalized_name and float(memory_weights.memory_name_weight) > 0:
                memory_components.append(("name", float(memory_weights.memory_name_weight), float(normalized_name[chunk.chunk_id])))
            if raw_scores.alias_score is not None and chunk.chunk_id in normalized_alias and float(memory_weights.memory_alias_weight) > 0:
                memory_components.append(("alias", float(memory_weights.memory_alias_weight), float(normalized_alias[chunk.chunk_id])))
            if (
                raw_scores.description_score is not None
                and chunk.chunk_id in normalized_description
                and float(memory_weights.memory_description_weight) > 0
            ):
                memory_components.append(("description", float(memory_weights.memory_description_weight), float(normalized_description[chunk.chunk_id])))
            memory_weight_sum = sum(weight for _, weight, _ in memory_components)
            if memory_components and memory_weight_sum > 0:
                normalized_memory_effective_weights = {
                    name: float(weight / memory_weight_sum)
                    for name, weight, _ in memory_components
                }
                normalized_memory_score = sum(
                    float(normalized_memory_effective_weights[name]) * float(score)
                    for name, _, score in memory_components
                )
            else:
                normalized_memory_score = None
                normalized_memory_effective_weights = {}
        elif normalized_variant == MBR_VARIANT_LINEAR:
            normalized_memory_score = normalized_linear_memory.get(chunk.chunk_id)
        elif normalized_variant == MBR_VARIANT_MBR1:
            normalized_memory_score = normalized_name.get(chunk.chunk_id)

        available_components: list[tuple[str, float, float]] = [
            ("text", float(weights.lambda_t), float(normalized_text.get(chunk.chunk_id, 0.0))),
        ]
        if raw_scores.memory_score is not None and normalized_memory_score is not None:
            available_components.append(("memory", float(weights.lambda_n), float(normalized_memory_score)))
        if raw_scores.summary_score is not None and chunk.chunk_id in normalized_summary:
            available_components.append(("summary", float(weights.lambda_h), float(normalized_summary[chunk.chunk_id])))

        weight_sum = sum(weight for _, weight, _ in available_components)
        if weight_sum <= 0:
            available_components = [("text", 1.0, float(normalized_text.get(chunk.chunk_id, 0.0)))]
            weight_sum = 1.0

        effective_weights = {
            name: float(weight / weight_sum)
            for name, weight, _ in available_components
        }
        final_score = sum(
            float(effective_weights[name]) * float(score)
            for name, _, score in available_components
        )
        ranked_rows.append(
            MBRRankedChunkRecord(
                chunk_id=str(chunk.chunk_id),
                chunk_index=int(chunk.step if chunk.step is not None else chunk_order[str(chunk.chunk_id)]),
                final_score=float(final_score),
                text_score=float(raw_scores.text_score),
                linear_memory_score=raw_scores.linear_memory_score,
                name_score=raw_scores.name_score,
                alias_score=raw_scores.alias_score,
                description_score=raw_scores.description_score,
                memory_score=raw_scores.memory_score,
                summary_score=raw_scores.summary_score,
                normalized_text_score=float(normalized_text.get(chunk.chunk_id, 0.0)),
                normalized_linear_memory_score=(
                    float(normalized_linear_memory[chunk.chunk_id])
                    if chunk.chunk_id in normalized_linear_memory
                    else None
                ),
                normalized_name_score=(
                    float(normalized_name[chunk.chunk_id]) if chunk.chunk_id in normalized_name else None
                ),
                normalized_alias_score=(
                    float(normalized_alias[chunk.chunk_id]) if chunk.chunk_id in normalized_alias else None
                ),
                normalized_description_score=(
                    float(normalized_description[chunk.chunk_id]) if chunk.chunk_id in normalized_description else None
                ),
                normalized_memory_score=(
                    float(normalized_memory_score) if normalized_memory_score is not None else None
                ),
                normalized_summary_score=(
                    float(normalized_summary[chunk.chunk_id]) if chunk.chunk_id in normalized_summary else None
                ),
                effective_weights=effective_weights,
                memory_effective_weights=normalized_memory_effective_weights,
            )
        )

    ranked_rows.sort(
        key=lambda row: (
            -float(row.final_score),
            -float(row.text_score),
            int(row.chunk_index),
            str(row.chunk_id),
        )
    )
    return MBRRetrievalResult(
        doc_id=document_index.doc_id,
        query=str(chunk_scores.query or ""),
        method_label=format_mbr_variant_method_label(
            variant=normalized_variant,
            lambda_t=weights.lambda_t,
            lambda_n=weights.lambda_n,
            lambda_h=weights.lambda_h,
            normalization_method=applied_normalization_method,
            memory_name_weight=memory_weights.memory_name_weight,
            memory_alias_weight=memory_weights.memory_alias_weight,
            memory_description_weight=memory_weights.memory_description_weight,
        ),
        normalization_method=str(applied_normalization_method or "none").strip().lower() or "none",
        weights=weights,
        memory_weights=memory_weights,
        ranked_chunks=ranked_rows,
    )


def score_mbr_document_pools(
    document_index: MBRDocumentMemoryIndex,
    *,
    query: str,
    retriever_name: str,
    retriever_kwargs: dict[str, Any] | None = None,
    memory_name_weight: float = DEFAULT_MBR_MEMORY_NAME_WEIGHT,
    memory_alias_weight: float = DEFAULT_MBR_MEMORY_ALIAS_WEIGHT,
    memory_description_weight: float = DEFAULT_MBR_MEMORY_DESCRIPTION_WEIGHT,
    memory_aggregation: str = DEFAULT_MBR_MEMORY_AGGREGATION,
) -> MBRChunkScoreStreams:
    """Backward-compatible structured-MBR scorer for MBR1/MBR2."""

    variant = _variant_from_memory_weights(
        memory_name_weight=memory_name_weight,
        memory_alias_weight=memory_alias_weight,
        memory_description_weight=memory_description_weight,
    )
    return score_mbr_variant_document_pools(
        document_index,
        query=query,
        retriever_name=retriever_name,
        variant=variant,
        retriever_kwargs=retriever_kwargs,
        memory_name_weight=memory_name_weight,
        memory_alias_weight=memory_alias_weight,
        memory_description_weight=memory_description_weight,
        memory_aggregation=memory_aggregation,
    )


def score_mbr_linear_document_pools(
    document_index: MBRDocumentMemoryIndex,
    *,
    query: str,
    retriever_name: str,
    retriever_kwargs: dict[str, Any] | None = None,
    memory_aggregation: str = DEFAULT_MBR_MEMORY_AGGREGATION,
) -> MBRChunkScoreStreams:
    """Score one query against chunks, per-entity concatenated memory, and summaries."""

    return score_mbr_variant_document_pools(
        document_index,
        query=query,
        retriever_name=retriever_name,
        variant=MBR_VARIANT_LINEAR,
        retriever_kwargs=retriever_kwargs,
        memory_aggregation=memory_aggregation,
    )


def score_mbr_pool_document_pools(
    document_index: MBRDocumentMemoryIndex,
    *,
    query: str,
    retriever_name: str,
    retriever_kwargs: dict[str, Any] | None = None,
    memory_aggregation: str = DEFAULT_MBR_MEMORY_AGGREGATION,
) -> MBRChunkScoreStreams:
    """Score one query against chunks, one pooled memory channel, and summaries."""

    return score_mbr_variant_document_pools(
        document_index,
        query=query,
        retriever_name=retriever_name,
        variant=MBR_VARIANT_POOL,
        retriever_kwargs=retriever_kwargs,
        memory_aggregation=memory_aggregation,
    )


def combine_mbr_chunk_scores(
    document_index: MBRDocumentMemoryIndex,
    chunk_scores: MBRChunkScoreStreams,
    *,
    lambda_t: float = DEFAULT_MBR_LAMBDA_T,
    lambda_n: float = DEFAULT_MBR_LAMBDA_M,
    lambda_h: float = DEFAULT_MBR_LAMBDA_H,
    normalization_method: str = "none",
    memory_name_weight: float = DEFAULT_MBR_MEMORY_NAME_WEIGHT,
    memory_alias_weight: float = DEFAULT_MBR_MEMORY_ALIAS_WEIGHT,
    memory_description_weight: float = DEFAULT_MBR_MEMORY_DESCRIPTION_WEIGHT,
) -> MBRRetrievalResult:
    """Backward-compatible structured-MBR combiner for MBR1/MBR2."""

    variant = _variant_from_memory_weights(
        memory_name_weight=memory_name_weight,
        memory_alias_weight=memory_alias_weight,
        memory_description_weight=memory_description_weight,
    )
    return combine_mbr_variant_chunk_scores(
        document_index,
        chunk_scores,
        variant=variant,
        lambda_t=lambda_t,
        lambda_n=lambda_n,
        lambda_h=lambda_h,
        normalization_method=normalization_method,
        memory_name_weight=memory_name_weight,
        memory_alias_weight=memory_alias_weight,
        memory_description_weight=memory_description_weight,
    )


def combine_mbr_linear_chunk_scores(
    document_index: MBRDocumentMemoryIndex,
    chunk_scores: MBRChunkScoreStreams,
    *,
    lambda_t: float = DEFAULT_MBR_LAMBDA_T,
    lambda_n: float = DEFAULT_MBR_LAMBDA_M,
    lambda_h: float = DEFAULT_MBR_LAMBDA_H,
    normalization_method: str = "none",
) -> MBRRetrievalResult:
    """Backward-compatible linear-MBR combiner."""

    return combine_mbr_variant_chunk_scores(
        document_index,
        chunk_scores,
        variant=MBR_VARIANT_LINEAR,
        lambda_t=lambda_t,
        lambda_n=lambda_n,
        lambda_h=lambda_h,
        normalization_method=normalization_method,
    )


def combine_mbr_pool_chunk_scores(
    document_index: MBRDocumentMemoryIndex,
    chunk_scores: MBRChunkScoreStreams,
    *,
    lambda_t: float = DEFAULT_MBR_LAMBDA_T,
    lambda_n: float = DEFAULT_MBR_LAMBDA_M,
    lambda_h: float = DEFAULT_MBR_LAMBDA_H,
    normalization_method: str = "none",
) -> MBRRetrievalResult:
    """Combine pooled-memory MBR streams."""

    return combine_mbr_variant_chunk_scores(
        document_index,
        chunk_scores,
        variant=MBR_VARIANT_POOL,
        lambda_t=lambda_t,
        lambda_n=lambda_n,
        lambda_h=lambda_h,
        normalization_method=normalization_method,
    )


def retrieve_mbr_variant_for_query(
    document_index: MBRDocumentMemoryIndex,
    *,
    query: str,
    retriever_name: str,
    retrieve_k: int,
    variant: str = MBR_VARIANT_MBR1,
    retriever_kwargs: dict[str, Any] | None = None,
    lambda_t: float = DEFAULT_MBR_LAMBDA_T,
    lambda_n: float = DEFAULT_MBR_LAMBDA_M,
    lambda_h: float = DEFAULT_MBR_LAMBDA_H,
    normalization_method: str = "none",
    memory_name_weight: float = DEFAULT_MBR2_MEMORY_NAME_WEIGHT,
    memory_alias_weight: float = DEFAULT_MBR2_MEMORY_ALIAS_WEIGHT,
    memory_description_weight: float = DEFAULT_MBR2_MEMORY_DESCRIPTION_WEIGHT,
    memory_aggregation: str = DEFAULT_MBR_MEMORY_AGGREGATION,
) -> tuple[MBRRetrievalResult, list[str], list[float], list[int]]:
    """Run one complete MBR variant and return top-k ranked chunk ids/scores."""

    normalized_variant = normalize_mbr_variant(variant)
    chunk_scores = score_mbr_variant_document_pools(
        document_index,
        query=query,
        retriever_name=retriever_name,
        variant=normalized_variant,
        retriever_kwargs=retriever_kwargs,
        memory_name_weight=memory_name_weight,
        memory_alias_weight=memory_alias_weight,
        memory_description_weight=memory_description_weight,
        memory_aggregation=memory_aggregation,
    )
    final_result = combine_mbr_variant_chunk_scores(
        document_index,
        chunk_scores,
        variant=normalized_variant,
        lambda_t=lambda_t,
        lambda_n=lambda_n,
        lambda_h=lambda_h,
        normalization_method=normalization_method,
        memory_name_weight=memory_name_weight,
        memory_alias_weight=memory_alias_weight,
        memory_description_weight=memory_description_weight,
    )
    top_rows = final_result.ranked_chunks[: max(0, int(retrieve_k))]
    return (
        final_result,
        [row.chunk_id for row in top_rows],
        [float(row.final_score) for row in top_rows],
        [int(row.chunk_index) for row in top_rows],
    )


def retrieve_mbr_for_query(
    document_index: MBRDocumentMemoryIndex,
    *,
    query: str,
    retriever_name: str,
    retrieve_k: int,
    retriever_kwargs: dict[str, Any] | None = None,
    lambda_t: float = DEFAULT_MBR_LAMBDA_T,
    lambda_n: float = DEFAULT_MBR_LAMBDA_M,
    lambda_h: float = DEFAULT_MBR_LAMBDA_H,
    normalization_method: str = "none",
    memory_name_weight: float = DEFAULT_MBR_MEMORY_NAME_WEIGHT,
    memory_alias_weight: float = DEFAULT_MBR_MEMORY_ALIAS_WEIGHT,
    memory_description_weight: float = DEFAULT_MBR_MEMORY_DESCRIPTION_WEIGHT,
    memory_aggregation: str = DEFAULT_MBR_MEMORY_AGGREGATION,
) -> tuple[MBRRetrievalResult, list[str], list[float], list[int]]:
    """Backward-compatible structured MBR query helper."""

    variant = _variant_from_memory_weights(
        memory_name_weight=memory_name_weight,
        memory_alias_weight=memory_alias_weight,
        memory_description_weight=memory_description_weight,
    )
    return retrieve_mbr_variant_for_query(
        document_index,
        query=query,
        retriever_name=retriever_name,
        retrieve_k=retrieve_k,
        variant=variant,
        retriever_kwargs=retriever_kwargs,
        lambda_t=lambda_t,
        lambda_n=lambda_n,
        lambda_h=lambda_h,
        normalization_method=normalization_method,
        memory_name_weight=memory_name_weight,
        memory_alias_weight=memory_alias_weight,
        memory_description_weight=memory_description_weight,
        memory_aggregation=memory_aggregation,
    )


def run_mbr_variant_retrieval_pipeline(
    *,
    examples: list[RetrievalExample],
    document_indices: dict[str, MBRDocumentMemoryIndex],
    retriever_name: str,
    retrieve_k: int,
    variant: str = MBR_VARIANT_MBR1,
    retriever_kwargs: dict[str, Any] | None = None,
    lambda_t: float = DEFAULT_MBR_LAMBDA_T,
    lambda_n: float = DEFAULT_MBR_LAMBDA_M,
    lambda_h: float = DEFAULT_MBR_LAMBDA_H,
    normalization_method: str = "none",
    memory_name_weight: float = DEFAULT_MBR2_MEMORY_NAME_WEIGHT,
    memory_alias_weight: float = DEFAULT_MBR2_MEMORY_ALIAS_WEIGHT,
    memory_description_weight: float = DEFAULT_MBR2_MEMORY_DESCRIPTION_WEIGHT,
    memory_aggregation: str = DEFAULT_MBR_MEMORY_AGGREGATION,
) -> tuple[list[RetrievalResult], dict[str, MBRRetrievalResult], str]:
    """Run one MBR variant over examples using per-document pools."""

    normalized_variant = normalize_mbr_variant(variant)
    method_label = format_mbr_variant_method_label(
        variant=normalized_variant,
        lambda_t=lambda_t,
        lambda_n=lambda_n,
        lambda_h=lambda_h,
        normalization_method=normalization_method,
        memory_name_weight=memory_name_weight,
        memory_alias_weight=memory_alias_weight,
        memory_description_weight=memory_description_weight,
    )
    results: list[RetrievalResult] = []
    detailed: dict[str, MBRRetrievalResult] = {}
    for ex in examples:
        document_index = document_indices.get(str(ex.doc_id))
        if document_index is None:
            retrieved_ids: list[str] = []
            scores: list[float] = []
            indices: list[int] = []
        else:
            final_result, retrieved_ids, scores, indices = retrieve_mbr_variant_for_query(
                document_index,
                query=ex.question,
                retriever_name=retriever_name,
                retrieve_k=retrieve_k,
                variant=normalized_variant,
                retriever_kwargs=retriever_kwargs,
                lambda_t=lambda_t,
                lambda_n=lambda_n,
                lambda_h=lambda_h,
                normalization_method=normalization_method,
                memory_name_weight=memory_name_weight,
                memory_alias_weight=memory_alias_weight,
                memory_description_weight=memory_description_weight,
                memory_aggregation=memory_aggregation,
            )
            detailed[str(ex.query_id)] = final_result
        results.append(
            RetrievalResult(
                query_id=ex.query_id,
                doc_id=ex.doc_id,
                question=ex.question,
                retrieved_chunk_ids=retrieved_ids,
                retrieved_indices=indices,
                scores=scores,
                gold_chunk_ids=list(ex.gold_chunk_ids),
                silver_chunk_ids=list(ex.silver_chunk_ids),
                silver_chunk_groups=[list(group) for group in ex.silver_chunk_groups],
            )
        )
    return results, detailed, method_label


def run_mbr_retrieval_pipeline(
    *,
    examples: list[RetrievalExample],
    document_indices: dict[str, MBRDocumentMemoryIndex],
    retriever_name: str,
    retrieve_k: int,
    retriever_kwargs: dict[str, Any] | None = None,
    lambda_t: float = DEFAULT_MBR_LAMBDA_T,
    lambda_n: float = DEFAULT_MBR_LAMBDA_M,
    lambda_h: float = DEFAULT_MBR_LAMBDA_H,
    normalization_method: str = "none",
    memory_name_weight: float = DEFAULT_MBR_MEMORY_NAME_WEIGHT,
    memory_alias_weight: float = DEFAULT_MBR_MEMORY_ALIAS_WEIGHT,
    memory_description_weight: float = DEFAULT_MBR_MEMORY_DESCRIPTION_WEIGHT,
    memory_aggregation: str = DEFAULT_MBR_MEMORY_AGGREGATION,
) -> tuple[list[RetrievalResult], dict[str, MBRRetrievalResult], str]:
    """Backward-compatible structured MBR runner."""

    variant = _variant_from_memory_weights(
        memory_name_weight=memory_name_weight,
        memory_alias_weight=memory_alias_weight,
        memory_description_weight=memory_description_weight,
    )
    return run_mbr_variant_retrieval_pipeline(
        examples=examples,
        document_indices=document_indices,
        retriever_name=retriever_name,
        retrieve_k=retrieve_k,
        variant=variant,
        retriever_kwargs=retriever_kwargs,
        lambda_t=lambda_t,
        lambda_n=lambda_n,
        lambda_h=lambda_h,
        normalization_method=normalization_method,
        memory_name_weight=memory_name_weight,
        memory_alias_weight=memory_alias_weight,
        memory_description_weight=memory_description_weight,
        memory_aggregation=memory_aggregation,
    )


def retrieve_mbr_linear_for_query(
    document_index: MBRDocumentMemoryIndex,
    *,
    query: str,
    retriever_name: str,
    retrieve_k: int,
    retriever_kwargs: dict[str, Any] | None = None,
    lambda_t: float = DEFAULT_MBR_LAMBDA_T,
    lambda_n: float = DEFAULT_MBR_LAMBDA_M,
    lambda_h: float = DEFAULT_MBR_LAMBDA_H,
    normalization_method: str = "none",
    memory_aggregation: str = DEFAULT_MBR_MEMORY_AGGREGATION,
) -> tuple[MBRRetrievalResult, list[str], list[float], list[int]]:
    """Run complete per-entity linear-memory MBR scoring and return top-k chunks."""

    return retrieve_mbr_variant_for_query(
        document_index,
        query=query,
        retriever_name=retriever_name,
        retrieve_k=retrieve_k,
        variant=MBR_VARIANT_LINEAR,
        retriever_kwargs=retriever_kwargs,
        lambda_t=lambda_t,
        lambda_n=lambda_n,
        lambda_h=lambda_h,
        normalization_method=normalization_method,
        memory_aggregation=memory_aggregation,
    )


def retrieve_mbr_pool_for_query(
    document_index: MBRDocumentMemoryIndex,
    *,
    query: str,
    retriever_name: str,
    retrieve_k: int,
    retriever_kwargs: dict[str, Any] | None = None,
    lambda_t: float = DEFAULT_MBR_LAMBDA_T,
    lambda_n: float = DEFAULT_MBR_LAMBDA_M,
    lambda_h: float = DEFAULT_MBR_LAMBDA_H,
    normalization_method: str = "none",
    memory_aggregation: str = DEFAULT_MBR_MEMORY_AGGREGATION,
) -> tuple[MBRRetrievalResult, list[str], list[float], list[int]]:
    """Run complete one-pool MBR scoring and return top-k chunks."""

    return retrieve_mbr_variant_for_query(
        document_index,
        query=query,
        retriever_name=retriever_name,
        retrieve_k=retrieve_k,
        variant=MBR_VARIANT_POOL,
        retriever_kwargs=retriever_kwargs,
        lambda_t=lambda_t,
        lambda_n=lambda_n,
        lambda_h=lambda_h,
        normalization_method=normalization_method,
        memory_aggregation=memory_aggregation,
    )


def run_mbr_linear_retrieval_pipeline(
    *,
    examples: list[RetrievalExample],
    document_indices: dict[str, MBRDocumentMemoryIndex],
    retriever_name: str,
    retrieve_k: int,
    retriever_kwargs: dict[str, Any] | None = None,
    lambda_t: float = DEFAULT_MBR_LAMBDA_T,
    lambda_n: float = DEFAULT_MBR_LAMBDA_M,
    lambda_h: float = DEFAULT_MBR_LAMBDA_H,
    normalization_method: str = "none",
    memory_aggregation: str = DEFAULT_MBR_MEMORY_AGGREGATION,
) -> tuple[list[RetrievalResult], dict[str, MBRRetrievalResult], str]:
    """Run `mbr_linear` over examples using per-entity concatenated memory."""

    return run_mbr_variant_retrieval_pipeline(
        examples=examples,
        document_indices=document_indices,
        retriever_name=retriever_name,
        retrieve_k=retrieve_k,
        variant=MBR_VARIANT_LINEAR,
        retriever_kwargs=retriever_kwargs,
        lambda_t=lambda_t,
        lambda_n=lambda_n,
        lambda_h=lambda_h,
        normalization_method=normalization_method,
        memory_aggregation=memory_aggregation,
    )


def run_mbr_pool_retrieval_pipeline(
    *,
    examples: list[RetrievalExample],
    document_indices: dict[str, MBRDocumentMemoryIndex],
    retriever_name: str,
    retrieve_k: int,
    retriever_kwargs: dict[str, Any] | None = None,
    lambda_t: float = DEFAULT_MBR_LAMBDA_T,
    lambda_n: float = DEFAULT_MBR_LAMBDA_M,
    lambda_h: float = DEFAULT_MBR_LAMBDA_H,
    normalization_method: str = "none",
    memory_aggregation: str = DEFAULT_MBR_MEMORY_AGGREGATION,
) -> tuple[list[RetrievalResult], dict[str, MBRRetrievalResult], str]:
    """Run `mbr_pool` over examples using one flat memory pool per chunk."""

    return run_mbr_variant_retrieval_pipeline(
        examples=examples,
        document_indices=document_indices,
        retriever_name=retriever_name,
        retrieve_k=retrieve_k,
        variant=MBR_VARIANT_POOL,
        retriever_kwargs=retriever_kwargs,
        lambda_t=lambda_t,
        lambda_n=lambda_n,
        lambda_h=lambda_h,
        normalization_method=normalization_method,
        memory_aggregation=memory_aggregation,
    )


def build_mbr_diagnostic_rows(
    *,
    examples: list[RetrievalExample],
    detailed_results: dict[str, MBRRetrievalResult],
) -> list[dict[str, Any]]:
    """Flatten per-query MBR scoring details for JSONL diagnostics output."""

    rows: list[dict[str, Any]] = []
    examples_by_query_id = {str(example.query_id): example for example in examples}
    for query_id, detail in detailed_results.items():
        example = examples_by_query_id.get(str(query_id))
        for ranked_chunk in detail.ranked_chunks:
            missing_signals: list[str] = []
            if ranked_chunk.memory_score is None:
                missing_signals.append("memory")
            if ranked_chunk.name_score is None:
                missing_signals.append("name")
            if ranked_chunk.alias_score is None:
                missing_signals.append("alias")
            if ranked_chunk.description_score is None:
                missing_signals.append("description")
            if ranked_chunk.summary_score is None:
                missing_signals.append("summary")
            row = {
                "query_id": str(query_id),
                "doc_id": str(detail.doc_id),
                "question": str(example.question if example is not None else detail.query),
                "method_label": str(detail.method_label),
                "normalization_method": str(detail.normalization_method),
                "weights": asdict(detail.weights),
                "memory_weights": asdict(detail.memory_weights),
                "chunk_id": str(ranked_chunk.chunk_id),
                "chunk_index": int(ranked_chunk.chunk_index),
                "text_score": float(ranked_chunk.text_score),
                "name_score": (
                    float(ranked_chunk.name_score) if ranked_chunk.name_score is not None else None
                ),
                "alias_score": (
                    float(ranked_chunk.alias_score) if ranked_chunk.alias_score is not None else None
                ),
                "description_score": (
                    float(ranked_chunk.description_score) if ranked_chunk.description_score is not None else None
                ),
                "memory_score": (
                    float(ranked_chunk.memory_score) if ranked_chunk.memory_score is not None else None
                ),
                "summary_score": (
                    float(ranked_chunk.summary_score) if ranked_chunk.summary_score is not None else None
                ),
                "normalized_text_score": float(ranked_chunk.normalized_text_score),
                "normalized_name_score": (
                    float(ranked_chunk.normalized_name_score)
                    if ranked_chunk.normalized_name_score is not None
                    else None
                ),
                "normalized_alias_score": (
                    float(ranked_chunk.normalized_alias_score)
                    if ranked_chunk.normalized_alias_score is not None
                    else None
                ),
                "normalized_description_score": (
                    float(ranked_chunk.normalized_description_score)
                    if ranked_chunk.normalized_description_score is not None
                    else None
                ),
                "normalized_memory_score": (
                    float(ranked_chunk.normalized_memory_score)
                    if ranked_chunk.normalized_memory_score is not None
                    else None
                ),
                "normalized_summary_score": (
                    float(ranked_chunk.normalized_summary_score)
                    if ranked_chunk.normalized_summary_score is not None
                    else None
                ),
                "final_mbr_score": float(ranked_chunk.final_score),
                "missing_signals": missing_signals,
                "effective_weights": dict(ranked_chunk.effective_weights),
                "memory_effective_weights": dict(ranked_chunk.memory_effective_weights),
            }
            if str(detail.method_label).startswith("mbr_linear_"):
                row["linear_memory_score"] = (
                    float(ranked_chunk.linear_memory_score)
                    if ranked_chunk.linear_memory_score is not None
                    else None
                )
                row["normalized_linear_memory_score"] = (
                    float(ranked_chunk.normalized_linear_memory_score)
                    if ranked_chunk.normalized_linear_memory_score is not None
                    else None
                )
            rows.append(row)
    return rows


__all__ = [
    "DEFAULT_MBR_LAMBDA_H",
    "DEFAULT_MBR_LAMBDA_M",
    "DEFAULT_MBR_LAMBDA_T",
    "DEFAULT_MBR_MEMORY_AGGREGATION",
    "MBRChunkPoolRecord",
    "MBRChunkScoreRecord",
    "MBRChunkScoreStreams",
    "MBRDocumentMemoryIndex",
    "MBRLinearMemoryPoolRecord",
    "MBRMemoryEntityRecord",
    "MBRMemoryWeights",
    "MBRRankedChunkRecord",
    "MBRRetrievalResult",
    "MBR_VARIANT_LINEAR",
    "MBR_VARIANT_MBR1",
    "MBR_VARIANT_MBR2",
    "MBR_VARIANT_POOL",
    "MBR_NORMALIZATION_METHOD_BERT_RESCORING",
    "MBRWeights",
    "MBRSummaryPoolRecord",
    "build_mbr_document_memory_index",
    "build_mbr_diagnostic_rows",
    "combine_mbr_chunk_scores",
    "combine_mbr_linear_chunk_scores",
    "combine_mbr_pool_chunk_scores",
    "combine_mbr_variant_chunk_scores",
    "format_mbr_method_label",
    "format_mbr_linear_method_label",
    "format_mbr_pool_method_label",
    "format_mbr_variant_method_label",
    "normalize_mbr_normalization_method",
    "normalize_mbr_variant",
    "normalize_mbr_weights",
    "normalize_mbr_memory_weights",
    "normalize_signal_scores",
    "retrieve_mbr_for_query",
    "retrieve_mbr_linear_for_query",
    "retrieve_mbr_pool_for_query",
    "retrieve_mbr_variant_for_query",
    "run_mbr_retrieval_pipeline",
    "run_mbr_linear_retrieval_pipeline",
    "run_mbr_pool_retrieval_pipeline",
    "run_mbr_variant_retrieval_pipeline",
    "score_mbr_linear_document_pools",
    "score_mbr_pool_document_pools",
    "score_mbr_document_pools",
    "score_mbr_variant_document_pools",
]
