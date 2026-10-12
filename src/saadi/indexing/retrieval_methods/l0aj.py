"""Builds retrieval-time L0+active-entity corpora from cached canonicalized chunks."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Iterable

from ...core.types import CanonicalizedChunkRecord, CorpusArtifact
from ...entities.active_selection import select_active_entities
from ...entities.merge import merge_active_entities
from ...runtime.progress import new_progress_bar

ENTITY_CONTEXT_HEADER = "[Entity context]"
logger = logging.getLogger(__name__)


def format_l0aj_method_label(augmentation_j: int) -> str:
    return f"l0aj_j{int(augmentation_j)}"


def _normalize_entity_dict(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": str(raw.get("name", "") or ""),
        "aliases": [str(x) for x in (raw.get("aliases", []) or []) if str(x).strip()],
        "description": [str(x) for x in (raw.get("description", []) or []) if str(x).strip()],
    }


def _extract_entity_memory(snapshot: dict[str, Any] | None) -> list[dict[str, Any]]:
    raw_items = (snapshot or {}).get("entity_memory", []) or []
    if not isinstance(raw_items, list):
        return []
    return [_normalize_entity_dict(item) for item in raw_items if isinstance(item, dict)]


def _extract_active_entities(snapshot: dict[str, Any] | None) -> list[dict[str, Any]]:
    raw_items = (snapshot or {}).get("active_entities", []) or []
    if not isinstance(raw_items, list):
        return []
    return [_normalize_entity_dict(item) for item in raw_items if isinstance(item, dict)]


def build_document_full_memory(chunks: Iterable[CanonicalizedChunkRecord]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[CanonicalizedChunkRecord]] = defaultdict(list)
    for chunk in chunks:
        grouped[str(chunk.doc_id)].append(chunk)

    out: dict[str, list[dict[str, Any]]] = {}
    for doc_id, doc_chunks in grouped.items():
        ordered = sorted(doc_chunks, key=lambda c: (int(c.chunk_index), str(c.chunk_id)))
        memory = _extract_entity_memory(ordered[-1].state_snapshot if ordered else {})
        if not memory:
            merged: list[dict[str, Any]] = []
            for chunk in ordered:
                merged = merge_active_entities(merged, _extract_active_entities(chunk.state_snapshot))
            memory = [_normalize_entity_dict(item) for item in merged]
        out[doc_id] = memory
    return out


def _format_entity_line(entity: dict[str, Any]) -> str:
    name = str(entity.get("name", "") or "").strip()
    aliases = ", ".join(str(x).strip() for x in (entity.get("aliases", []) or []) if str(x).strip())
    descriptions = " / ".join(str(x).strip() for x in (entity.get("description", []) or []) if str(x).strip())
    return f"{name} | {aliases} | {descriptions}"


def format_entity_context_block(entities: list[dict[str, Any]], *, header: str = ENTITY_CONTEXT_HEADER) -> str:
    lines = [_format_entity_line(entity) for entity in entities if str(entity.get("name", "") or "").strip()]
    if not lines:
        return ""
    return "\n".join([header, *lines])


def build_l0aj_text(
    chunk: CanonicalizedChunkRecord,
    *,
    document_memory: list[dict[str, Any]],
    augmentation_j: int,
    max_aliases: int = 2,
    max_desc: int = 1,
    active_entity_selector: str = "bm25",
    header: str = ENTITY_CONTEXT_HEADER,
) -> tuple[str, list[dict[str, Any]]]:
    selected = select_active_entities(
        document_memory,
        chunk.raw_chunk_text,
        max_entities=max(0, int(augmentation_j)),
        max_aliases=max_aliases,
        max_desc=max_desc,
        selector=active_entity_selector,
    )
    if not selected:
        return chunk.canonicalized_chunk_text, []
    entity_block = format_entity_context_block(selected, header=header)
    if not entity_block:
        return chunk.canonicalized_chunk_text, []
    base = str(chunk.canonicalized_chunk_text or "")
    if base.strip():
        return f"{base.rstrip()}\n\n{entity_block}", selected
    return entity_block, selected


def build_l0aj_rows(
    chunks: Iterable[CanonicalizedChunkRecord],
    *,
    augmentation_j: int,
    max_aliases: int = 2,
    max_desc: int = 1,
    active_entity_selector: str = "bm25",
    header: str = ENTITY_CONTEXT_HEADER,
) -> list[dict[str, Any]]:
    ordered = sorted(chunks, key=lambda c: (str(c.doc_id), int(c.chunk_index), str(c.chunk_id)))
    memory_by_doc = build_document_full_memory(ordered)
    method_label = format_l0aj_method_label(augmentation_j)
    logger.info("Building %s rows chunks=%d docs=%d", method_label, len(ordered), len(memory_by_doc))
    progress = new_progress_bar(total=len(ordered), desc=f"Building {method_label} corpus", unit="chunk", leave=False)
    logged_interval = max(250, len(ordered) // 20) if ordered else 250

    rows: list[dict[str, Any]] = []
    for idx, chunk in enumerate(ordered, start=1):
        retrieval_text, selected = build_l0aj_text(
            chunk,
            document_memory=memory_by_doc.get(str(chunk.doc_id), []),
            augmentation_j=augmentation_j,
            max_aliases=max_aliases,
            max_desc=max_desc,
            active_entity_selector=active_entity_selector,
            header=header,
        )
        rows.append(
            {
                "doc_id": str(chunk.doc_id),
                "chunk_id": str(chunk.chunk_id),
                "chunk_index": int(chunk.chunk_index),
                "raw_chunk_text": str(chunk.raw_chunk_text or ""),
                "canonicalized_chunk_text": str(chunk.canonicalized_chunk_text or ""),
                "retrieval_text": retrieval_text,
                "selected_entities": selected,
                "document_memory_size": len(memory_by_doc.get(str(chunk.doc_id), [])),
                "retrieval_method": method_label,
                "augmentation_j": int(augmentation_j),
            }
        )
        if progress is not None:
            progress.update(1)
        elif idx % logged_interval == 0 or idx == len(ordered):
            logger.info("Building %s corpus progress=%d/%d", method_label, idx, len(ordered))
    if progress is not None:
        progress.close()
    logger.info("Built %s rows count=%d", method_label, len(rows))
    return rows


def build_l0aj_corpus(
    chunks: Iterable[CanonicalizedChunkRecord],
    *,
    augmentation_j: int,
    max_aliases: int = 2,
    max_desc: int = 1,
    active_entity_selector: str = "bm25",
    header: str = ENTITY_CONTEXT_HEADER,
    metadata: dict[str, Any] | None = None,
) -> tuple[CorpusArtifact, list[dict[str, Any]]]:
    rows = build_l0aj_rows(
        chunks,
        augmentation_j=augmentation_j,
        max_aliases=max_aliases,
        max_desc=max_desc,
        active_entity_selector=active_entity_selector,
        header=header,
    )
    method_label = format_l0aj_method_label(augmentation_j)
    corpus = CorpusArtifact(
        chunk_ids=[str(row["chunk_id"]) for row in rows],
        chunk_texts=[str(row["retrieval_text"]) for row in rows],
        metadata=metadata or {
            "source": "canonicalized",
            "retrieval_mode": method_label,
            "augmentation_j": int(augmentation_j),
        },
    )
    return corpus, rows


def build_l0aj_corpus_by_doc(
    chunks: Iterable[CanonicalizedChunkRecord],
    *,
    augmentation_j: int,
    max_aliases: int = 2,
    max_desc: int = 1,
    active_entity_selector: str = "bm25",
    header: str = ENTITY_CONTEXT_HEADER,
) -> tuple[dict[str, CorpusArtifact], list[dict[str, Any]]]:
    rows = build_l0aj_rows(
        chunks,
        augmentation_j=augmentation_j,
        max_aliases=max_aliases,
        max_desc=max_desc,
        active_entity_selector=active_entity_selector,
        header=header,
    )
    method_label = format_l0aj_method_label(augmentation_j)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["doc_id"])].append(row)

    out: dict[str, CorpusArtifact] = {}
    for doc_id, doc_rows in grouped.items():
        out[doc_id] = CorpusArtifact(
            chunk_ids=[str(row["chunk_id"]) for row in doc_rows],
            chunk_texts=[str(row["retrieval_text"]) for row in doc_rows],
            metadata={
                "source": "canonicalized",
                "retrieval_mode": method_label,
                "augmentation_j": int(augmentation_j),
                "doc_id": doc_id,
            },
        )
    return out, rows


__all__ = [
    "ENTITY_CONTEXT_HEADER",
    "build_document_full_memory",
    "build_l0aj_corpus",
    "build_l0aj_corpus_by_doc",
    "build_l0aj_rows",
    "build_l0aj_text",
    "format_entity_context_block",
    "format_l0aj_method_label",
]
