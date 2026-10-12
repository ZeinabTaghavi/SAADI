"""Cross-document folded context builder for SAADI context expansion.

When the global top-k chunks span multiple documents, this builder creates a
folded context that groups chunks by source document, uses per-document
summaries, generates bridges only within the same document, and inserts
cross-document boundary markers between document groups.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import BuiltContext, RetrievedContextItem
from .folded_context import (
    SummaryBlock,
    apply_bridge_policy,
    build_generated_bridge_context,
    select_summary_ranges,
)

logger = logging.getLogger(__name__)


def _retrieved_chunk_index(item: RetrievedContextItem) -> int | None:
    try:
        return int(item.metadata.get("chunk_index"))
    except Exception:
        return None


def _chunk_doc_id(item: RetrievedContextItem) -> str:
    return str(item.metadata.get("doc_id", "") or "")


def _chunk_text(item: RetrievedContextItem) -> str:
    return str(item.raw_text or "").strip()


def _chunk_display_id(item: RetrievedContextItem) -> object:
    display_id = item.metadata.get("chunk_index")
    if display_id is None or str(display_id).strip() == "":
        return item.chunk_id
    return display_id


def _format_retrieved_chunk(item: RetrievedContextItem) -> str:
    doc_id = _chunk_doc_id(item)
    display = _chunk_display_id(item)
    label = f"[Retrieved Chunk {display}"
    if doc_id:
        label += f" (doc: {doc_id})"
    label += f"]\n{_chunk_text(item)}"
    return label


def _format_summary_block(summary: SummaryBlock) -> str:
    return f"[Summary L{summary.level} covering chunks {summary.range_label}]\n{summary.text}"


def _group_items_by_doc(
    items: list[RetrievedContextItem],
) -> list[tuple[str, list[RetrievedContextItem]]]:
    """Group items by source document, preserving first-appearance order.

    Returns a list of ``(doc_id, items)`` tuples.  Documents are ordered by
    the rank of their highest-scoring chunk (the order in which the items
    appear).  Within each document group, chunks are sorted by chunk index
    (original document order).
    """
    doc_order: list[str] = []
    groups: dict[str, list[RetrievedContextItem]] = {}
    for item in items:
        doc_id = _chunk_doc_id(item)
        if doc_id not in groups:
            doc_order.append(doc_id)
            groups[doc_id] = []
        groups[doc_id].append(item)

    # Sort each group by chunk index within the document
    for doc_items in groups.values():
        doc_items.sort(key=lambda it: (
            _retrieved_chunk_index(it) if _retrieved_chunk_index(it) is not None else 10**12,
            it.rank,
            str(it.chunk_id),
        ))

    return [(doc_id, groups[doc_id]) for doc_id in doc_order]


def build_cross_document_folded_context(
    *,
    retrieved_items: list[RetrievedContextItem],
    summaries_by_doc: dict[str, list[SummaryBlock]],
    chunk_doc_ids: dict[str, str],
    summary_k: int,
    use_bridge: bool = True,
    bridge_mode: str = "generated",
    query: str = "",
    example_id: str = "",
    bridge_generator: Any | None = None,
) -> BuiltContext:
    """Build folded context from cross-document global top-k chunks.

    Chunks are grouped by source document and presented in document order.
    Bridges are generated only between consecutive chunks from the *same*
    document.  A ``[Cross-Document Boundary]`` marker separates groups from
    different documents.

    Parameters
    ----------
    retrieved_items :
        Global top-k chunks, already ordered by descending MBR score.
    summaries_by_doc :
        Pre-loaded summary blocks, keyed by doc_id.
    chunk_doc_ids :
        chunk_id -> source doc_id mapping.
    summary_k :
        Hierarchy branching factor for summary selection.
    use_bridge :
        Whether to generate bridges between consecutive same-document chunks.
    bridge_mode :
        Bridge policy mode (``"generated"``, ``"local"``, ``"prefix"``).
    query :
        The user question.
    example_id :
        Unique identifier for this query example.
    bridge_generator :
        The active :class:`SaadiAnswerGenerator` for bridge generation.

    Returns
    -------
    BuiltContext
        Context text and debug payload.
    """
    if not retrieved_items:
        return BuiltContext(context_text="", debug_payload={"cross_document": True, "empty": True})

    # Attach doc_id to item metadata if not already present
    for item in retrieved_items:
        if not _chunk_doc_id(item) and item.chunk_id in chunk_doc_ids:
            item.metadata["doc_id"] = chunk_doc_ids[item.chunk_id]

    doc_groups = _group_items_by_doc(retrieved_items)
    context_blocks: list[str] = []
    all_bridge_rows: list[dict[str, object]] = []
    debug_doc_groups: list[dict[str, Any]] = []
    cross_document_boundaries: list[int] = []

    for group_idx, (doc_id, doc_items) in enumerate(doc_groups):
        if group_idx > 0:
            cross_document_boundaries.append(len(context_blocks))
            context_blocks.append("[Cross-Document Boundary]")

        doc_summaries = summaries_by_doc.get(doc_id, [])

        # Determine which chunk indices we have for this document
        retrieved_indices = [
            idx for idx in (
                _retrieved_chunk_index(item) for item in doc_items
            )
            if idx is not None
        ]

        # Apply bridge policy for this document's chunks
        bridge_policy = {}
        selected_summaries: list[SummaryBlock] = []
        if retrieved_indices and doc_summaries and summary_k > 0:
            bridge_policy = apply_bridge_policy(
                retrieved_indices=retrieved_indices,
                summary_k=summary_k,
                bridge_mode=bridge_mode if bridge_mode != "generated" else "generated",
                use_bridge=use_bridge,
            )
            target_start = bridge_policy.get("target_summary_start")
            target_end = bridge_policy.get("target_summary_end")
            if target_start is not None and target_end is not None:
                selected_summaries, _ = select_summary_ranges(
                    doc_summaries,
                    target_start=int(target_start),
                    target_end=int(target_end),
                    retrieved_indices=set(retrieved_indices),
                )

        # Build context for this document group
        if bridge_mode == "generated" and use_bridge and bridge_generator is not None and len(doc_items) > 1:
            doc_context_text, bridge_rows = build_generated_bridge_context(
                retrieved_items=doc_items,
                summaries=selected_summaries,
                chunk_texts=None,
                query=query,
                example_id=f"{example_id}__{doc_id}",
                bridge_generator=bridge_generator,
            )
            all_bridge_rows.extend(bridge_rows)
        else:
            # Block-style: summaries first, then retrieved chunks
            parts: list[str] = []
            if selected_summaries:
                parts.append(f"[Document {doc_id} Summaries]")
                for summary in selected_summaries:
                    parts.append(_format_summary_block(summary))
            for item in doc_items:
                parts.append(_format_retrieved_chunk(item))
            doc_context_text = "\n\n".join(parts).strip()

        context_blocks.append(doc_context_text)

        debug_doc_groups.append({
            "doc_id": doc_id,
            "chunk_count": len(doc_items),
            "chunk_ids": [str(item.chunk_id) for item in doc_items],
            "chunk_indices": [_retrieved_chunk_index(item) for item in doc_items],
            "summary_count": len(selected_summaries),
            "bridge_policy": bridge_policy,
        })

    context_text = "\n\n".join(context_blocks).strip()

    debug_payload: dict[str, Any] = {
        "context_builder": "cross_document_folded",
        "cross_document": True,
        "document_count": len(doc_groups),
        "document_groups": debug_doc_groups,
        "cross_document_boundaries": cross_document_boundaries,
        "total_retrieved_chunks": len(retrieved_items),
        "bridge_mode": bridge_mode,
        "uses_bridge": use_bridge,
        "summary_k": summary_k,
    }
    if all_bridge_rows:
        debug_payload["generated_bridges"] = all_bridge_rows

    return BuiltContext(context_text=context_text, debug_payload=debug_payload)
