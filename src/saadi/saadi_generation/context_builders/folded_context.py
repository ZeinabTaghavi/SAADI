from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...io.hierarchy_cache import _resolve_cache_file_path
from ...io.jsonl import read_jsonl
from .base import BuiltContext, ContextBuilderConfig, QueryContextRequest, RetrievedContextItem


@dataclass(frozen=True, slots=True)
class SummaryBlock:
    doc_id: str
    summary_id: str
    level: int
    span_size: int
    start_chunk_index: int
    end_chunk_index: int
    text: str

    @property
    def range_label(self) -> str:
        return f"{self.start_chunk_index}-{self.end_chunk_index}"


def _load_cache_index(index_path: Path) -> dict[str, Any]:
    with index_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected cache index JSON object: {index_path}")
    return payload


def load_summary_blocks_by_doc(
    llm_cache_index_json: str | Path,
    *,
    summary_k: int,
) -> dict[str, list[SummaryBlock]]:
    """Load all available hierarchical summary blocks from cache `L_*.jsonl` files."""

    k = int(summary_k)
    if k <= 0:
        raise ValueError("summary_k must be > 0 for folded_context_block")

    index_path = Path(llm_cache_index_json).resolve()
    payload = _load_cache_index(index_path)
    docs = payload.get("docs", []) or []
    if not isinstance(docs, list):
        return {}

    out: dict[str, list[SummaryBlock]] = {}
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        doc_id = str(doc.get("doc_id", "") or "")
        files = doc.get("files", {}) or {}
        if not doc_id or not isinstance(files, dict):
            continue
        blocks: list[SummaryBlock] = []
        seen_ranges: set[tuple[int, int, int]] = set()
        level_paths: list[tuple[int, Any]] = []
        for key, path_value in files.items():
            if not isinstance(key, str) or not key.startswith("L_"):
                continue
            try:
                level = int(key.split("_", 1)[1])
            except Exception:
                continue
            if level <= 0:
                continue
            level_paths.append((level, path_value))
        for level, path_value in sorted(level_paths):
            path = _resolve_cache_file_path(path_value, base_dir=index_path.parent)
            if not path.exists():
                continue
            span_size = k**level
            for row in read_jsonl(path):
                if not isinstance(row, dict):
                    continue
                try:
                    end = int(row.get("step"))
                except Exception:
                    continue
                text = str(row.get("chunk", "") or "").strip()
                if not text:
                    continue
                start = max(0, end - span_size + 1)
                range_key = (level, start, end)
                if range_key in seen_ranges:
                    continue
                seen_ranges.add(range_key)
                blocks.append(
                    SummaryBlock(
                        doc_id=str(row.get("doc_id", doc_id) or doc_id),
                        summary_id=str(row.get("summary_id", f"{doc_id}:L{level}:{start}-{end}") or f"{doc_id}:L{level}:{start}-{end}"),
                        level=level,
                        span_size=span_size,
                        start_chunk_index=start,
                        end_chunk_index=end,
                        text=text,
                    )
                )
        if blocks:
            blocks.sort(key=lambda item: (item.start_chunk_index, item.end_chunk_index, item.level, item.summary_id))
            out[doc_id] = blocks
    return out


def _span_contains_retrieved(block: SummaryBlock, retrieved_indices: set[int]) -> bool:
    return any(block.start_chunk_index <= idx <= block.end_chunk_index for idx in retrieved_indices)


def select_summary_ranges(
    summaries: list[SummaryBlock],
    *,
    target_start: int,
    target_end: int,
    retrieved_indices: set[int],
) -> tuple[list[SummaryBlock], list[dict[str, int]]]:
    """Greedily cover target_start..target_end with the largest usable blocks.

    A high-level summary is usable only when its span contains no retrieved
    chunk. When a retrieved chunk falls inside a high-level span, we decompose
    that region down to L1 summaries so local evidence is not hidden inside a
    coarse summary block.
    """

    if target_start < 0 or target_end < target_start:
        return [], []

    by_start: dict[int, list[SummaryBlock]] = {}
    for block in summaries:
        if block.start_chunk_index < target_start or block.end_chunk_index > target_end:
            continue
        by_start.setdefault(block.start_chunk_index, []).append(block)
    for blocks in by_start.values():
        blocks.sort(key=lambda block: (block.span_size, block.level, block.end_chunk_index), reverse=True)

    selected: list[SummaryBlock] = []
    missing_ranges: list[dict[str, int]] = []
    cursor = target_start
    while cursor <= target_end:
        candidates = by_start.get(cursor, [])
        usable = [
            block
            for block in candidates
            if block.level <= 1 or not _span_contains_retrieved(block, retrieved_indices)
        ]
        if usable:
            chosen = usable[0]
            selected.append(chosen)
            cursor = chosen.end_chunk_index + 1
            continue

        next_starts = [start for start in by_start if start > cursor]
        if not next_starts:
            missing_ranges.append({"start": cursor, "end": target_end})
            break
        next_start = min(next_starts)
        missing_ranges.append({"start": cursor, "end": next_start - 1})
        cursor = next_start

    return selected, missing_ranges


def _summary_range_start(chunk_index: int, summary_k: int) -> int:
    return (chunk_index // summary_k) * summary_k


def _summary_range_end(chunk_index: int, summary_k: int) -> int:
    return ((chunk_index // summary_k) + 1) * summary_k - 1


def apply_bridge_policy(
    *,
    retrieved_indices: list[int],
    summary_k: int,
    bridge_mode: str,
    use_bridge: bool,
) -> dict[str, int | str | None]:
    """Resolve which document summary interval the folded builder should cover."""

    if not retrieved_indices:
        return {
            "bridge_mode": bridge_mode,
            "first_retrieved_chunk": None,
            "last_retrieved_chunk": None,
            "target_summary_start": None,
            "target_summary_end": None,
        }
    if not use_bridge:
        raise NotImplementedError("This context builder requires bridge summaries.")

    first_retrieved_chunk = min(retrieved_indices)
    last_retrieved_chunk = max(retrieved_indices)
    normalized_mode = str(bridge_mode or "prefix").strip().lower()
    if normalized_mode == "prefix":
        target_summary_start = 0
    elif normalized_mode in {"local", "generated"}:
        target_summary_start = _summary_range_start(first_retrieved_chunk, summary_k)
    else:
        raise ValueError(f"Unsupported folded-context bridge_mode: {bridge_mode!r}")

    return {
        "bridge_mode": normalized_mode,
        "first_retrieved_chunk": first_retrieved_chunk,
        "last_retrieved_chunk": last_retrieved_chunk,
        "target_summary_start": target_summary_start,
        "target_summary_end": _summary_range_end(last_retrieved_chunk, summary_k),
    }


def _retrieved_chunk_index(item: RetrievedContextItem) -> int | None:
    try:
        return int(item.metadata.get("chunk_index"))
    except Exception:
        return None


def _coerce_retrieved_item(raw_item: Any, *, default_rank: int) -> RetrievedContextItem:
    if isinstance(raw_item, RetrievedContextItem):
        return raw_item
    if isinstance(raw_item, dict):
        metadata = raw_item.get("metadata", {}) or {}
        if not isinstance(metadata, dict):
            metadata = {}
        metadata = dict(metadata)
        chunk_index = raw_item.get("chunk_index", raw_item.get("index"))
        if chunk_index is not None and str(chunk_index).strip():
            metadata.setdefault("chunk_index", chunk_index)
        chunk_id = str(
            raw_item.get("chunk_id")
            or raw_item.get("id")
            or raw_item.get("chunk_index")
            or raw_item.get("index")
            or default_rank
        )
        try:
            rank = int(raw_item.get("rank", default_rank))
        except Exception:
            rank = default_rank
        score = raw_item.get("score")
        try:
            score = float(score) if score is not None else None
        except Exception:
            score = None
        return RetrievedContextItem(
            chunk_id=chunk_id,
            rank=rank,
            score=score,
            raw_text=str(raw_item.get("raw_text") or raw_item.get("text") or raw_item.get("chunk_text") or ""),
            metadata=metadata,
        )

    chunk_id = str(raw_item)
    metadata: dict[str, object] = {}
    try:
        metadata["chunk_index"] = int(raw_item)
    except Exception:
        pass
    return RetrievedContextItem(chunk_id=chunk_id, rank=default_rank, raw_text="", metadata=metadata)


def _retrieved_sort_key(item: RetrievedContextItem) -> tuple[int, int, str]:
    chunk_index = _retrieved_chunk_index(item)
    return (
        chunk_index if chunk_index is not None else 10**12,
        int(item.rank),
        str(item.chunk_id),
    )


def _chunk_display_id(item: RetrievedContextItem) -> object:
    display_id = item.metadata.get("chunk_index")
    if display_id is None or str(display_id).strip() == "":
        return item.chunk_id
    return display_id


def _chunk_text(item: RetrievedContextItem, chunk_texts: dict[object, str] | None) -> str:
    raw_text = str(item.raw_text or "").strip()
    if raw_text or not chunk_texts:
        return raw_text

    candidates: list[object] = [item.chunk_id, str(item.chunk_id)]
    chunk_index = _retrieved_chunk_index(item)
    if chunk_index is not None:
        candidates.extend([chunk_index, str(chunk_index)])
    for key in candidates:
        if key in chunk_texts:
            return str(chunk_texts[key] or "").strip()
    return ""


def _format_summary_block(summary: SummaryBlock) -> str:
    return f"[Summary L{summary.level} covering chunks {summary.range_label}]\n{summary.text}"


def _format_retrieved_chunk(
    item: RetrievedContextItem,
    chunk_texts: dict[object, str] | None,
    *,
    nested: bool,
) -> str:
    label = "Retrieved Chunk" if nested else "Chunk"
    return f"[{label} {_chunk_display_id(item)}]\n{_chunk_text(item, chunk_texts)}"


def _retrieved_chunk_debug(item: RetrievedContextItem, chunk_texts: dict[object, str] | None) -> dict[str, object]:
    text = _chunk_text(item, chunk_texts)
    return {
        "rank": int(item.rank),
        "chunk_id": str(item.chunk_id),
        "chunk_index": item.metadata.get("chunk_index"),
        "score": item.score,
        "text_chars": len(text),
    }


def _summary_block_debug(summary: SummaryBlock) -> dict[str, object]:
    return {
        "summary_id": summary.summary_id,
        "level": summary.level,
        "span_size": summary.span_size,
        "range": summary.range_label,
        "start_chunk_index": summary.start_chunk_index,
        "end_chunk_index": summary.end_chunk_index,
    }


def _summary_for_chunk(summaries: list[SummaryBlock], chunk_index: int) -> SummaryBlock | None:
    candidates = [
        summary
        for summary in summaries
        if summary.start_chunk_index <= chunk_index <= summary.end_chunk_index
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda summary: (summary.span_size, summary.level, summary.start_chunk_index))
    return candidates[0]


def _build_nested_chunk_groups(
    *,
    summaries: list[SummaryBlock],
    retrieved_items: list[RetrievedContextItem],
) -> tuple[dict[str, list[RetrievedContextItem]], list[RetrievedContextItem], dict[str, dict[str, object]]]:
    groups: dict[str, list[RetrievedContextItem]] = {summary.range_label: [] for summary in summaries}
    uncovered: list[RetrievedContextItem] = []

    for item in retrieved_items:
        chunk_index = _retrieved_chunk_index(item)
        if chunk_index is None:
            uncovered.append(item)
            continue
        summary = _summary_for_chunk(summaries, chunk_index)
        if summary is None:
            uncovered.append(item)
            continue
        groups.setdefault(summary.range_label, []).append(item)

    debug_groups: dict[str, dict[str, object]] = {}
    for summary in summaries:
        items = sorted(groups.get(summary.range_label, []), key=_retrieved_sort_key)
        groups[summary.range_label] = items
        debug_groups[summary.range_label] = {
            "summary_id": summary.summary_id,
            "chunk_ids": [str(item.chunk_id) for item in items],
            "chunk_indices": [item.metadata.get("chunk_index") for item in items],
        }
    if uncovered:
        items = sorted(uncovered, key=_retrieved_sort_key)
        debug_groups["__uncovered__"] = {
            "summary_id": None,
            "chunk_ids": [str(item.chunk_id) for item in items],
            "chunk_indices": [item.metadata.get("chunk_index") for item in items],
        }
    return groups, uncovered, debug_groups


def build_block_context(
    *,
    summaries: list[SummaryBlock],
    retrieved_items: list[RetrievedContextItem],
    chunk_texts: dict[object, str] | None,
) -> str:
    blocks: list[str] = ["[Document Summaries]"]
    for summary in summaries:
        blocks.append(_format_summary_block(summary))
    blocks.append("[Retrieved Evidence Chunks]")
    for item in retrieved_items:
        blocks.append(_format_retrieved_chunk(item, chunk_texts, nested=False))
    return "\n\n".join(blocks).strip()


def build_nested_context(
    *,
    summaries: list[SummaryBlock],
    retrieved_items: list[RetrievedContextItem],
    chunk_texts: dict[object, str] | None,
) -> tuple[str, dict[str, dict[str, object]]]:
    groups, uncovered_items, debug_groups = _build_nested_chunk_groups(
        summaries=summaries,
        retrieved_items=retrieved_items,
    )
    blocks = ["[Folded Context]"]
    for summary in summaries:
        blocks.append(_format_summary_block(summary))
        for item in groups.get(summary.range_label, []):
            blocks.append(_format_retrieved_chunk(item, chunk_texts, nested=True))
    if uncovered_items:
        blocks.append("[Retrieved Evidence Chunks Without Summary]")
        for item in sorted(uncovered_items, key=_retrieved_sort_key):
            blocks.append(_format_retrieved_chunk(item, chunk_texts, nested=True))
    return "\n\n".join(blocks).strip(), debug_groups


def _summary_overlaps_interval(summary: SummaryBlock, *, start: int, end: int) -> bool:
    return summary.start_chunk_index <= end and summary.end_chunk_index >= start


def _summary_text_for_bridge(summaries: list[SummaryBlock]) -> str:
    blocks: list[str] = []
    for summary in summaries:
        blocks.append(f"[Summary covering chunks {summary.range_label}]\n{summary.text}")
    return "\n\n".join(blocks).strip()


def generate_bridge(
    *,
    bridge_generator: Any,
    query: str,
    chunk_a: RetrievedContextItem,
    chunk_b: RetrievedContextItem,
    intermediate_summaries: list[SummaryBlock],
    chunk_texts: dict[object, str] | None,
    example_id: str,
) -> dict[str, object]:
    if bridge_generator is None or not hasattr(bridge_generator, "generate_bridge"):
        raise RuntimeError(
            "folded_context_generated_bridge requires the active SAADI answer generator "
            "so bridge generation uses the same model as final answer generation."
        )
    summary_text = _summary_text_for_bridge(intermediate_summaries)
    generated = bridge_generator.generate_bridge(
        query=query,
        chunk_a=_chunk_text(chunk_a, chunk_texts),
        summaries=summary_text,
        chunk_b=_chunk_text(chunk_b, chunk_texts),
    )
    return {
        "example_id": str(example_id),
        "query": str(query or ""),
        "chunk_a_id": str(chunk_a.chunk_id),
        "chunk_b_id": str(chunk_b.chunk_id),
        "chunk_a_index": _retrieved_chunk_index(chunk_a),
        "chunk_b_index": _retrieved_chunk_index(chunk_b),
        "intermediate_summary_ranges": [summary.range_label for summary in intermediate_summaries],
        "intermediate_summary_ids": [summary.summary_id for summary in intermediate_summaries],
        "intermediate_summary_text": summary_text,
        "bridge_prompt": str(generated.get("bridge_prompt", "") or ""),
        "generated_bridge": str(generated.get("generated_bridge", "") or "").strip(),
        "generated_bridge_token_length": generated.get("generated_bridge_token_length"),
        "bridge_generation_latency_ms": generated.get("bridge_generation_latency_ms"),
    }


def build_generated_bridge_context(
    *,
    retrieved_items: list[RetrievedContextItem],
    summaries: list[SummaryBlock],
    chunk_texts: dict[object, str] | None,
    query: str,
    example_id: str,
    bridge_generator: Any,
) -> tuple[str, list[dict[str, object]]]:
    if not retrieved_items:
        return "", []

    blocks: list[str] = []
    bridge_rows: list[dict[str, object]] = []
    for index, item in enumerate(retrieved_items):
        blocks.append(f"[Retrieved Chunk {_chunk_display_id(item)}]\n{_chunk_text(item, chunk_texts)}")
        if index >= len(retrieved_items) - 1:
            continue

        next_item = retrieved_items[index + 1]
        start = _retrieved_chunk_index(item)
        end = _retrieved_chunk_index(next_item)
        if start is None or end is None:
            intermediate_summaries: list[SummaryBlock] = []
        else:
            left = min(start, end)
            right = max(start, end)
            intermediate_summaries = [
                summary
                for summary in summaries
                if _summary_overlaps_interval(summary, start=left, end=right)
            ]

        bridge_row = generate_bridge(
            bridge_generator=bridge_generator,
            query=query,
            chunk_a=item,
            chunk_b=next_item,
            intermediate_summaries=intermediate_summaries,
            chunk_texts=chunk_texts,
            example_id=example_id,
        )
        bridge_rows.append(bridge_row)
        bridge_text = str(bridge_row.get("generated_bridge", "") or "").strip()
        blocks.append(f"[Generated Bridge]\n{bridge_text}")

    return "\n\n".join(blocks).strip(), bridge_rows


def build_folded_context(
    retrieved_chunks: list[Any],
    chunk_texts: dict[object, str] | None,
    summaries: list[SummaryBlock],
    summary_k: int,
    ordering: str = "nested",
    use_bridge: bool = True,
    bridge_mode: str = "prefix",
    *,
    context_builder_name: str = "folded_context",
    experiment_name: str = "folded_context",
    query: str = "",
    example_id: str = "",
    bridge_generator: Any | None = None,
) -> BuiltContext:
    """Build folded context using shared summary selection and pluggable ordering."""

    summary_k_int = int(summary_k)
    if summary_k_int <= 0:
        raise ValueError("folded-context generation requires --summary_k > 0")

    normalized_ordering = str(ordering or "block").strip().lower()
    if normalized_ordering not in {"block", "nested"}:
        raise ValueError(f"Unsupported folded-context ordering: {ordering!r}")

    retrieved_items = sorted(
        [
            _coerce_retrieved_item(raw_item, default_rank=rank)
            for rank, raw_item in enumerate(list(retrieved_chunks), start=1)
        ],
        key=_retrieved_sort_key,
    )
    retrieved_indices = [
        chunk_index
        for chunk_index in (_retrieved_chunk_index(item) for item in retrieved_items)
        if chunk_index is not None
    ]
    bridge_policy = apply_bridge_policy(
        retrieved_indices=retrieved_indices,
        summary_k=summary_k_int,
        bridge_mode=bridge_mode,
        use_bridge=use_bridge,
    )
    target_summary_start = bridge_policy["target_summary_start"]
    target_summary_end = bridge_policy["target_summary_end"]

    if target_summary_start is None or target_summary_end is None:
        selected_summaries, missing_summary_ranges = [], []
    else:
        selected_summaries, missing_summary_ranges = select_summary_ranges(
            summaries,
            target_start=int(target_summary_start),
            target_end=int(target_summary_end),
            retrieved_indices=set(retrieved_indices),
        )
    selected_summaries.sort(
        key=lambda block: (block.start_chunk_index, block.end_chunk_index, block.level, block.summary_id)
    )

    if bridge_policy["bridge_mode"] == "generated":
        if normalized_ordering != "nested":
            raise ValueError("folded_context_generated_bridge requires nested ordering.")
        context_text, generated_bridge_rows = build_generated_bridge_context(
            retrieved_items=retrieved_items,
            summaries=selected_summaries,
            chunk_texts=chunk_texts,
            query=query,
            example_id=example_id,
            bridge_generator=bridge_generator,
        )
        debug_groups = None
    elif normalized_ordering == "block":
        context_text = build_block_context(
            summaries=selected_summaries,
            retrieved_items=retrieved_items,
            chunk_texts=chunk_texts,
        )
        debug_groups = None
        generated_bridge_rows = []
    else:
        context_text, debug_groups = build_nested_context(
            summaries=selected_summaries,
            retrieved_items=retrieved_items,
            chunk_texts=chunk_texts,
        )
        generated_bridge_rows = []

    selected_summary_ranges = [summary.range_label for summary in selected_summaries]
    selected_summary_ids = [summary.summary_id for summary in selected_summaries]
    selected_summary_levels = [summary.level for summary in selected_summaries]

    debug_payload: dict[str, object] = {
        "context_builder": context_builder_name,
        "experiment_name": experiment_name,
        "uses_summaries": True,
        "uses_entity_memory": False,
        "uses_bridge": bool(use_bridge),
        "bridge_mode": bridge_policy["bridge_mode"],
        "summary_k": summary_k_int,
        "first_retrieved_chunk": bridge_policy["first_retrieved_chunk"],
        "last_retrieved_chunk": bridge_policy["last_retrieved_chunk"],
        "latest_retrieved_chunk_index": (
            bridge_policy["last_retrieved_chunk"]
            if bridge_policy["last_retrieved_chunk"] is not None
            else -1
        ),
        "target_summary_start": bridge_policy["target_summary_start"],
        "target_summary_end": bridge_policy["target_summary_end"],
        "latest_selected_summary_end": (
            bridge_policy["target_summary_end"]
            if bridge_policy["target_summary_end"] is not None
            else -1
        ),
        "summary_selection_strategy": "largest_nonretrieved_hierarchy_blocks_with_l1_fallback",
        "selected_summary_ranges": selected_summary_ranges,
        "selected_summary_ids": selected_summary_ids,
        "selected_summary_levels": selected_summary_levels,
        "selected_summary_blocks": [_summary_block_debug(summary) for summary in selected_summaries],
        "missing_summary_ranges": missing_summary_ranges,
        "retrieved_chunks": [_retrieved_chunk_debug(item, chunk_texts) for item in retrieved_items],
    }
    if normalized_ordering == "nested":
        debug_payload["ordering"] = "nested"
        debug_payload["chunks_grouped_by_summary_range"] = debug_groups or {}
    if bridge_policy["bridge_mode"] == "generated":
        debug_payload["generated_bridges"] = generated_bridge_rows

    return BuiltContext(context_text=context_text, debug_payload=debug_payload)


@dataclass(slots=True)
class FoldedContextBuilder:

    config: ContextBuilderConfig
    summaries_by_doc: dict[str, list[SummaryBlock]]
    bridge_generator: Any | None = None
    name: str = "folded_context"

    def build(self, request: QueryContextRequest) -> BuiltContext:
        if self.config.experiment_name not in {
            "folded_context_block",
            "folded_context_nested",
            "folded_context_local_bridge",
            "folded_context_generated_bridge",
        }:
            raise NotImplementedError(
                "FoldedContextBuilder currently implements folded_context_block, "
                "folded_context_nested, folded_context_local_bridge, "
                "and folded_context_generated_bridge. "
            )
        summary_k = int(self.config.summary_k)
        if summary_k <= 0:
            raise ValueError(f"{self.config.experiment_name} requires --summary_k > 0")

        doc_id = str(request.retrieval_result.doc_id)
        ordering = str(
            self.config.extra.get("ordering")
            or ("block" if self.config.experiment_name == "folded_context_block" else "nested")
        )
        bridge_mode = str(self.config.extra.get("bridge_mode") or "prefix")
        return build_folded_context(
            retrieved_chunks=list(request.retrieved_items)[: max(1, int(self.config.top_k))],
            chunk_texts=None,
            summaries=self.summaries_by_doc.get(doc_id, []),
            summary_k=summary_k,
            ordering=ordering,
            use_bridge=bool(self.config.include_bridge),
            bridge_mode=bridge_mode,
            context_builder_name=self.name,
            experiment_name=self.config.experiment_name,
            query=request.question,
            example_id=str(request.retrieval_result.query_id),
            bridge_generator=self.bridge_generator,
        )
