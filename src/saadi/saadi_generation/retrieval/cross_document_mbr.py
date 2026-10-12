"""Cross-document MBR retrieval for context expansion experiments.

When SAADI is run with context expansion, the expanded document group contains
multiple documents.  This module scores all chunks from all documents against
a query using per-document MBR scoring (text + memory + summary signals), then
selects the global top-k chunks across the entire group.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...indexing.retrieval_methods.mbr import (
    MBRDocumentMemoryIndex,
    MBRRankedChunkRecord,
    MBRRetrievalResult,
    build_mbr_document_memory_index,
    combine_mbr_chunk_scores,
    score_mbr_document_pools,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CrossDocumentMBRConfig:
    """Static configuration for cross-document MBR retrieval."""

    retriever_name: str
    top_k: int
    lambda_t: float = 1.0
    lambda_n: float = 0.5
    lambda_h: float = 0.3
    normalization_method: str = "none"
    memory_name_weight: float = 1.0
    memory_alias_weight: float = 1.0
    memory_description_weight: float = 1.0
    target_token_budget: int | None = None
    tolerance_tokens: int = 0


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CrossDocumentMBRResult:
    """Global top-k MBR result across multiple expanded documents."""

    query: str
    doc_ids: list[str]
    ranked_chunks: list[MBRRankedChunkRecord]
    chunk_doc_ids: dict[str, str]  # chunk_id -> source doc_id
    per_document_results: dict[str, MBRRetrievalResult]
    token_budget_applied: bool
    truncated_doc_ids: list[str]
    per_document_chunk_counts: dict[str, int]
    debug_metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Token counting helpers
# ---------------------------------------------------------------------------


def _count_tokens(text: str, tokenizer: Any | None) -> int:
    """Count tokens using the provided tokenizer, or fall back to word-split."""
    if tokenizer is not None:
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if ids is not None:
                return len(ids)
        except Exception:
            pass
    return len(str(text or "").split())


# ---------------------------------------------------------------------------
# Build memory indices with token budget enforcement
# ---------------------------------------------------------------------------


def build_cross_document_memory_indices(
    index_path: str | Path,
    doc_ids: list[str],
    *,
    query_doc_id: str | None = None,
    tokenizer: Any | None = None,
    target_token_budget: int | None = None,
    tolerance_tokens: int = 0,
) -> tuple[dict[str, MBRDocumentMemoryIndex], dict[str, Any]]:
    """Build MBR memory indices for multiple documents, enforcing token budget.

    The query's own document is always loaded first and never truncated.
    Remaining filler documents are loaded in their original order.  When a
    filler document causes the cumulative chunk tokens to exceed
    ``target_token_budget + tolerance_tokens``:

    * **Non-last documents** are truncated from the **end** (trailing chunks
      removed, leading chunks kept).
    * The **last document** in the processing order is truncated from the
      **beginning** (leading chunks removed, trailing chunks kept).

    No further documents are loaded after truncation.

    Returns
    -------
    indices : dict[str, MBRDocumentMemoryIndex]
        Keyed by doc_id.
    metadata : dict
        Diagnostic info about budget enforcement.
    """
    path = Path(index_path).resolve()
    budget = None if target_token_budget is None else int(target_token_budget) + int(tolerance_tokens)

    # Ensure the query document is loaded first (always fully loaded)
    ordered_doc_ids = list(doc_ids)
    if query_doc_id and str(query_doc_id) in ordered_doc_ids:
        ordered_doc_ids.remove(str(query_doc_id))
        ordered_doc_ids.insert(0, str(query_doc_id))

    indices: dict[str, MBRDocumentMemoryIndex] = {}
    cumulative_tokens = 0
    truncated_doc_ids: list[str] = []
    per_doc_tokens: dict[str, int] = {}
    skipped_doc_ids: list[str] = []
    truncation_sides: dict[str, str] = {}

    for doc_idx, doc_id in enumerate(ordered_doc_ids):
        try:
            doc_index = build_mbr_document_memory_index(path, doc_id=doc_id)
        except (KeyError, FileNotFoundError, ValueError) as exc:
            logger.warning("Skipping document %s for cross-document MBR: %s", doc_id, exc)
            skipped_doc_ids.append(doc_id)
            continue

        # Count tokens in the chunk pool
        doc_tokens = sum(
            _count_tokens(chunk.text, tokenizer)
            for chunk in doc_index.chunk_pool
        )

        # Query document is never truncated
        is_query_doc = query_doc_id is not None and str(doc_id) == str(query_doc_id)

        if not is_query_doc and budget is not None and cumulative_tokens + doc_tokens > budget:
            remaining_budget = max(0, budget - cumulative_tokens)
            is_last_doc = (doc_idx == len(ordered_doc_ids) - 1)

            if is_last_doc:
                # Last document: truncate from beginning (keep trailing chunks)
                truncated_pool = []
                running = 0
                for chunk in reversed(doc_index.chunk_pool):
                    chunk_tok = _count_tokens(chunk.text, tokenizer)
                    if running + chunk_tok > remaining_budget:
                        break
                    truncated_pool.append(chunk)
                    running += chunk_tok
                truncated_pool.reverse()  # restore original chunk order
                truncation_side = "beginning"
            else:
                # Non-last document: truncate from end (keep leading chunks)
                truncated_pool = []
                running = 0
                for chunk in doc_index.chunk_pool:
                    chunk_tok = _count_tokens(chunk.text, tokenizer)
                    if running + chunk_tok > remaining_budget:
                        break
                    truncated_pool.append(chunk)
                    running += chunk_tok
                truncation_side = "end"

            if not truncated_pool:
                skipped_doc_ids.append(doc_id)
                break

            kept_chunk_ids = {c.chunk_id for c in truncated_pool}
            doc_index = _truncate_document_index(doc_index, kept_chunk_ids)
            doc_tokens = running
            truncated_doc_ids.append(doc_id)
            truncation_sides[doc_id] = truncation_side
            logger.info(
                "Truncated document %s from %s to %d chunks (%d tokens) to fit token budget %d",
                doc_id, truncation_side, len(truncated_pool), doc_tokens, budget,
            )

            indices[doc_id] = doc_index
            per_doc_tokens[doc_id] = doc_tokens
            cumulative_tokens += doc_tokens
            # Stop processing further documents
            break

        indices[doc_id] = doc_index
        per_doc_tokens[doc_id] = doc_tokens
        cumulative_tokens += doc_tokens

    metadata = {
        "total_documents_requested": len(doc_ids),
        "total_documents_loaded": len(indices),
        "total_chunk_tokens": cumulative_tokens,
        "token_budget": budget,
        "token_budget_applied": budget is not None,
        "query_doc_id": query_doc_id,
        "processing_order": ordered_doc_ids,
        "truncated_doc_ids": truncated_doc_ids,
        "truncation_sides": truncation_sides,
        "skipped_doc_ids": skipped_doc_ids,
        "per_document_tokens": per_doc_tokens,
        "per_document_chunk_counts": {
            doc_id: len(idx.chunk_pool)
            for doc_id, idx in indices.items()
        },
    }
    return indices, metadata


def _truncate_document_index(
    original: MBRDocumentMemoryIndex,
    kept_chunk_ids: set[str],
) -> MBRDocumentMemoryIndex:
    """Return a copy of *original* with only the chunks in *kept_chunk_ids*."""
    chunk_pool = [c for c in original.chunk_pool if c.chunk_id in kept_chunk_ids]
    linear_memory_pool = [m for m in original.linear_memory_pool if m.chunk_id in kept_chunk_ids]

    # Filter cross-link maps
    chunk_to_names = {cid: v for cid, v in original.chunk_to_names.items() if cid in kept_chunk_ids}
    chunk_to_aliases = {cid: v for cid, v in original.chunk_to_aliases.items() if cid in kept_chunk_ids}
    chunk_to_descriptions = {cid: v for cid, v in original.chunk_to_descriptions.items() if cid in kept_chunk_ids}
    chunk_to_linear_memory = {cid: v for cid, v in original.chunk_to_linear_memory.items() if cid in kept_chunk_ids}
    chunk_to_summary = {cid: v for cid, v in original.chunk_to_summary.items() if cid in kept_chunk_ids}

    # Keep only summaries that still cover at least one kept chunk
    kept_summary_ids = {sid for sid in chunk_to_summary.values() if sid is not None}
    summary_pool = [s for s in original.summary_pool if s.summary_id in kept_summary_ids]
    summary_to_chunks = {
        sid: [cid for cid in cids if cid in kept_chunk_ids]
        for sid, cids in original.summary_to_chunks.items()
        if sid in kept_summary_ids
    }

    # Keep name/alias/description pools that are still referenced
    kept_names = {n for names in chunk_to_names.values() for n in names}
    kept_aliases = {a for aliases in chunk_to_aliases.values() for a in aliases}
    kept_descriptions = {d for descs in chunk_to_descriptions.values() for d in descs}

    name_to_chunks = {
        n: [cid for cid in cids if cid in kept_chunk_ids]
        for n, cids in original.name_to_chunks.items()
        if n in kept_names
    }
    memory_entities = [entity for entity in original.memory_entities if entity.name in kept_names]
    entity_to_aliases = {
        name: list(aliases)
        for name, aliases in original.entity_to_aliases.items()
        if name in kept_names
    }
    entity_to_descriptions = {
        name: list(descriptions)
        for name, descriptions in original.entity_to_descriptions.items()
        if name in kept_names
    }
    alias_to_chunks = {
        a: [cid for cid in cids if cid in kept_chunk_ids]
        for a, cids in original.alias_to_chunks.items()
        if a in kept_aliases
    }
    description_to_chunks = {
        d: [cid for cid in cids if cid in kept_chunk_ids]
        for d, cids in original.description_to_chunks.items()
        if d in kept_descriptions
    }

    return MBRDocumentMemoryIndex(
        doc_id=original.doc_id,
        chunk_pool=chunk_pool,
        linear_memory_pool=linear_memory_pool,
        name_pool=[n for n in original.name_pool if n in kept_names],
        alias_pool=[a for a in original.alias_pool if a in kept_aliases],
        description_pool=[d for d in original.description_pool if d in kept_descriptions],
        memory_entities=memory_entities,
        summary_pool=summary_pool,
        chunk_to_linear_memory=chunk_to_linear_memory,
        chunk_to_names=chunk_to_names,
        name_to_chunks=name_to_chunks,
        entity_to_aliases=entity_to_aliases,
        entity_to_descriptions=entity_to_descriptions,
        chunk_to_aliases=chunk_to_aliases,
        alias_to_chunks=alias_to_chunks,
        chunk_to_descriptions=chunk_to_descriptions,
        description_to_chunks=description_to_chunks,
        chunk_to_summary=chunk_to_summary,
        summary_to_chunks=summary_to_chunks,
    )


# ---------------------------------------------------------------------------
# Score & rank across all documents
# ---------------------------------------------------------------------------


def score_and_rank_cross_document(
    document_indices: dict[str, MBRDocumentMemoryIndex],
    *,
    query: str,
    config: CrossDocumentMBRConfig,
    retriever_kwargs: dict[str, Any] | None = None,
) -> CrossDocumentMBRResult:
    """Score all chunks from all documents against *query*, return global top-k.

    Each document's chunks are scored using their own document's memory and
    summary signals (per-document MBR).  The resulting scored chunks are merged
    and the global top-k are selected.
    """
    all_ranked: list[tuple[str, MBRRankedChunkRecord]] = []  # (doc_id, record)
    per_document_results: dict[str, MBRRetrievalResult] = {}
    chunk_doc_ids: dict[str, str] = {}

    for doc_id, doc_index in document_indices.items():
        # Build the chunk_id -> doc_id mapping
        for chunk in doc_index.chunk_pool:
            chunk_doc_ids[str(chunk.chunk_id)] = doc_id

        # Score all pools for this document
        chunk_scores = score_mbr_document_pools(
            doc_index,
            query=query,
            retriever_name=config.retriever_name,
            retriever_kwargs=retriever_kwargs,
            memory_name_weight=config.memory_name_weight,
            memory_alias_weight=config.memory_alias_weight,
            memory_description_weight=config.memory_description_weight,
        )
        # Combine into ranked result
        result = combine_mbr_chunk_scores(
            doc_index,
            chunk_scores,
            lambda_t=config.lambda_t,
            lambda_n=config.lambda_n,
            lambda_h=config.lambda_h,
            normalization_method=config.normalization_method,
            memory_name_weight=config.memory_name_weight,
            memory_alias_weight=config.memory_alias_weight,
            memory_description_weight=config.memory_description_weight,
        )
        per_document_results[doc_id] = result
        for record in result.ranked_chunks:
            all_ranked.append((doc_id, record))

    # Global sort by descending final score
    all_ranked.sort(
        key=lambda item: (
            -float(item[1].final_score),
            -float(item[1].text_score),
            int(item[1].chunk_index),
            str(item[1].chunk_id),
        )
    )

    top_k = max(0, int(config.top_k))
    top_records = [record for _, record in all_ranked[:top_k]]
    per_document_chunk_counts = {
        doc_id: len(idx.chunk_pool)
        for doc_id, idx in document_indices.items()
    }

    # Per-document top-k counts
    top_k_doc_counts: dict[str, int] = {}
    for doc_id, record in all_ranked[:top_k]:
        top_k_doc_counts[doc_id] = top_k_doc_counts.get(doc_id, 0) + 1

    return CrossDocumentMBRResult(
        query=query,
        doc_ids=list(document_indices.keys()),
        ranked_chunks=top_records,
        chunk_doc_ids=chunk_doc_ids,
        per_document_results=per_document_results,
        token_budget_applied=config.target_token_budget is not None,
        truncated_doc_ids=[],  # filled by caller from build metadata
        per_document_chunk_counts=per_document_chunk_counts,
        debug_metadata={
            "total_chunks_scored": len(all_ranked),
            "top_k": top_k,
            "per_document_top_k_counts": top_k_doc_counts,
            "global_top_k_chunk_ids": [
                {"chunk_id": r.chunk_id, "doc_id": chunk_doc_ids.get(r.chunk_id, ""), "score": r.final_score}
                for r in top_records
            ],
        },
    )
