"""Runs chunk-by-chunk canonicalization and writes ingestion/cache artifacts."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

from ..context_strategies import get_strategy
from ..core.chunking import chunk_documents_grouped_records
from ..core.state import init_discourse_state, state_from_snapshot
from ..core.types import CanonicalizedChunkRecord, ChunkRecord
from ..entities import active_entity_config_label
from ..io.artifact_paths import default_artifact_paths
from ..io.hierarchy_cache import (
    export_llm_history_cache,
    load_canonicalized_chunk_records_cache,
    load_llm_history_cache_index,
)
from ..io.jsonl import write_jsonl
from ..llm.base import GenerateParams, LLMBackend
from ..llm.factory import resolve_backend

logger = logging.getLogger(__name__)

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None


def _progress_enabled() -> bool:
    raw = str(os.getenv("SAADI_PROGRESS", "1")).strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    return sys.stderr.isatty()


def _new_progress_bar(*, total: int, desc: str, unit: str, leave: bool) -> object | None:
    if _tqdm is None or not _progress_enabled():
        return None
    return _tqdm(total=total, desc=desc, unit=unit, leave=leave, dynamic_ncols=True)


def _snapshot_has_required_hierarchy_for_step(snapshot: dict | None, *, step: int, strategy: Any) -> bool:
    """
    Validate that a cached snapshot is structurally complete for this step.
    - L0 must include this step.
    - For hierarchical runs (branching_factor > 1), each expected boundary summary
      at cycles b, b^2, ... up to this step must be present.
    """
    stores_hierarchy = bool(getattr(strategy, "stores_hierarchy", True))
    raw = (snapshot or {}).get("hierarchy_summaries", {}) or {}
    if not stores_hierarchy:
        # No-history arms intentionally have no hierarchy in their state.
        return not raw
    if not isinstance(raw, dict):
        return False

    def _steps_for_cycle(cycle: int) -> set[int]:
        vals = raw.get(cycle, raw.get(str(cycle)))
        if isinstance(vals, dict):
            out: set[int] = set()
            for key in vals.keys():
                try:
                    out.add(int(key))
                except Exception:
                    continue
            return out
        if isinstance(vals, list):
            # Legacy layout: positional summaries.
            return {(idx + 1) * cycle - 1 for idx, _ in enumerate(vals)}
        return set()

    if int(step) not in _steps_for_cycle(1):
        return False

    b = getattr(strategy, "branching_factor", getattr(strategy, "k", None))
    if not isinstance(b, int) or b <= 1:
        return True

    max_level = getattr(strategy, "max_hierarchy_level", None)
    level = 1
    cycle = b
    # For current step s, expected boundaries for cycle c are:
    # c-1, 2c-1, ..., floor((s+1)/c)*c - 1
    while cycle <= (int(step) + 1):
        if isinstance(max_level, int) and max_level >= 0 and level > int(max_level):
            break
        present = _steps_for_cycle(cycle)
        last_expected = ((int(step) + 1) // cycle) * cycle - 1
        boundary = cycle - 1
        while boundary <= last_expected:
            if boundary not in present:
                return False
            boundary += cycle
        level += 1
        cycle *= b
    return True


def _resolve_valid_cached_record(
    *,
    chunk: ChunkRecord,
    strategy: Any,
    cached_records_by_chunk_id: dict[str, CanonicalizedChunkRecord] | None,
) -> CanonicalizedChunkRecord | None:
    if not cached_records_by_chunk_id:
        return None
    cached = cached_records_by_chunk_id.get(chunk.chunk_id)
    if cached is None:
        return None
    if cached.doc_id != chunk.doc_id:
        return None
    if int(cached.chunk_index) != int(chunk.chunk_index):
        return None
    if cached.raw_chunk_text != chunk.raw_text:
        return None
    if not _snapshot_has_required_hierarchy_for_step(
        cached.state_snapshot,
        step=int(chunk.chunk_index),
        strategy=strategy,
    ):
        return None
    return cached


def _cached_prefix_records_for_doc(
    *,
    raw_chunks: list[ChunkRecord],
    strategy: Any,
    cached_records_by_chunk_id: dict[str, CanonicalizedChunkRecord] | None,
) -> list[CanonicalizedChunkRecord]:
    prefix_records: list[CanonicalizedChunkRecord] = []
    if not raw_chunks or not cached_records_by_chunk_id:
        return prefix_records
    for chunk in raw_chunks:
        cached = _resolve_valid_cached_record(
            chunk=chunk,
            strategy=strategy,
            cached_records_by_chunk_id=cached_records_by_chunk_id,
        )
        if cached is None:
            break
        prefix_records.append(cached)
    return prefix_records


def _build_doc_cache_status(
    *,
    doc_id: str,
    raw_chunks: list[ChunkRecord],
    strategy: Any,
    use_cache: bool,
    cached_records_by_chunk_id: dict[str, CanonicalizedChunkRecord] | None,
) -> dict[str, Any]:
    prefix_records = (
        _cached_prefix_records_for_doc(
            raw_chunks=raw_chunks,
            strategy=strategy,
            cached_records_by_chunk_id=cached_records_by_chunk_id,
        )
        if use_cache
        else []
    )
    total_chunks = len(raw_chunks)
    cached_prefix_chunks = len(prefix_records)
    remaining_chunks = max(0, total_chunks - cached_prefix_chunks)
    fully_cached = total_chunks > 0 and remaining_chunks == 0
    partially_cached = total_chunks > 0 and 0 < cached_prefix_chunks < total_chunks
    return {
        "doc_id": str(doc_id),
        "total_chunks": int(total_chunks),
        "cached_prefix_chunks": int(cached_prefix_chunks),
        "remaining_chunks": int(remaining_chunks),
        "fully_cached": bool(fully_cached),
        "partially_cached": bool(partially_cached),
        "prefix_records": prefix_records,
        "cache_hits": 0,
        "llm_updates": 0,
        "completed_from_cache": False,
    }


def _log_cache_readiness_summary(*, schedule_label: str, doc_statuses: list[dict[str, Any]]) -> None:
    if not doc_statuses:
        return
    docs_total = len(doc_statuses)
    docs_with_chunks = sum(1 for status in doc_statuses if int(status.get("total_chunks", 0)) > 0)
    docs_empty = docs_total - docs_with_chunks
    docs_with_any_cache = sum(1 for status in doc_statuses if int(status.get("cached_prefix_chunks", 0)) > 0)
    docs_fully_cached = sum(1 for status in doc_statuses if bool(status.get("fully_cached", False)))
    docs_partially_cached = sum(1 for status in doc_statuses if bool(status.get("partially_cached", False)))
    docs_without_cache = sum(
        1
        for status in doc_statuses
        if int(status.get("total_chunks", 0)) > 0 and int(status.get("cached_prefix_chunks", 0)) == 0
    )
    docs_remaining = sum(1 for status in doc_statuses if int(status.get("remaining_chunks", 0)) > 0)
    total_chunks = sum(int(status.get("total_chunks", 0)) for status in doc_statuses)
    cached_prefix_chunks = sum(int(status.get("cached_prefix_chunks", 0)) for status in doc_statuses)
    remaining_chunks = sum(int(status.get("remaining_chunks", 0)) for status in doc_statuses)
    logger.info(
        "Cache readiness [%s]: docs_total=%d docs_with_chunks=%d docs_empty=%d docs_with_any_cache=%d "
        "docs_fully_cached=%d docs_partially_cached=%d docs_without_cache=%d docs_remaining_to_process=%d "
        "total_chunks=%d cached_prefix_chunks=%d remaining_chunks=%d",
        schedule_label,
        docs_total,
        docs_with_chunks,
        docs_empty,
        docs_with_any_cache,
        docs_fully_cached,
        docs_partially_cached,
        docs_without_cache,
        docs_remaining,
        total_chunks,
        cached_prefix_chunks,
        remaining_chunks,
    )
    remaining_doc_ids = [str(status.get("doc_id", "")) for status in doc_statuses if int(status.get("remaining_chunks", 0)) > 0]
    if remaining_doc_ids:
        sample = ", ".join(remaining_doc_ids[:10])
        suffix = " ..." if len(remaining_doc_ids) > 10 else ""
        logger.info(
            "Cache readiness [%s]: documents still needing generation=%s%s",
            schedule_label,
            sample,
            suffix,
        )


def _log_cache_result_summary(*, schedule_label: str, doc_statuses: list[dict[str, Any]]) -> None:
    if not doc_statuses:
        return
    docs_total = len(doc_statuses)
    docs_with_chunks = sum(1 for status in doc_statuses if int(status.get("total_chunks", 0)) > 0)
    docs_completed_from_cache = sum(1 for status in doc_statuses if bool(status.get("completed_from_cache", False)))
    docs_with_llm_updates = sum(1 for status in doc_statuses if int(status.get("llm_updates", 0)) > 0)
    total_chunks = sum(int(status.get("total_chunks", 0)) for status in doc_statuses)
    total_cache_hits = sum(int(status.get("cache_hits", 0)) for status in doc_statuses)
    total_llm_updates = sum(int(status.get("llm_updates", 0)) for status in doc_statuses)
    logger.info(
        "Cache result [%s]: docs_total=%d docs_with_chunks=%d docs_completed_from_cache=%d docs_with_llm_updates=%d "
        "total_chunks=%d cache_hits=%d llm_updates=%d",
        schedule_label,
        docs_total,
        docs_with_chunks,
        docs_completed_from_cache,
        docs_with_llm_updates,
        total_chunks,
        total_cache_hits,
        total_llm_updates,
    )


def ingest_document(
    *,
    doc_id: str,
    document: str,
    backend: LLMBackend,
    strategy,
    chunk_size: int = 500,
    chunk_overlap: int = 0,
    tokenizer=None,
    generate: GenerateParams | None = None,
    use_cache: bool = False,
    cached_records_by_chunk_id: dict[str, CanonicalizedChunkRecord] | None = None,
    on_chunk_processed: Callable[[CanonicalizedChunkRecord, bool], None] | None = None,
    precomputed_raw_chunks: list[ChunkRecord] | None = None,
) -> tuple[list[ChunkRecord], list[CanonicalizedChunkRecord], dict[str, Any]]:
    backend = resolve_backend(backend)
    generate = generate or GenerateParams()
    if precomputed_raw_chunks is not None:
        raw_chunks = precomputed_raw_chunks
    else:
        grouped = chunk_documents_grouped_records([document], doc_ids=[doc_id], tokenizer=tokenizer, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        raw_chunks = grouped[0]
    logger.info(
        "Ingest doc=%s chunked into %d chunks (chunk_size=%d overlap=%d)",
        doc_id,
        len(raw_chunks),
        chunk_size,
        chunk_overlap,
    )
    total_chunks = len(raw_chunks)
    doc_cache_status = _build_doc_cache_status(
        doc_id=doc_id,
        raw_chunks=raw_chunks,
        strategy=strategy,
        use_cache=use_cache,
        cached_records_by_chunk_id=cached_records_by_chunk_id,
    )
    logger.info(
        "Cache readiness [document]: doc=%s total_chunks=%d cached_prefix_chunks=%d remaining_chunks=%d fully_cached=%s",
        doc_id,
        total_chunks,
        int(doc_cache_status["cached_prefix_chunks"]),
        int(doc_cache_status["remaining_chunks"]),
        str(doc_cache_status["fully_cached"]),
    )
    state = init_discourse_state(hierarchy_levels=[1])
    out_records: list[CanonicalizedChunkRecord] = []
    # We only trust cache hits on a contiguous prefix.
    # After the first miss, regenerate all subsequent chunks for consistency.
    cache_prefix_valid = True
    cache_hits = 0
    llm_updates = 0
    bar = _new_progress_bar(
        total=total_chunks,
        desc=f"Ingest chunks doc={str(doc_id)[:8]}",
        unit="chunk",
        leave=False,
    )
    if bool(doc_cache_status["fully_cached"]) and total_chunks > 0:
        prefix_records = list(doc_cache_status["prefix_records"])
        try:
            state_from_snapshot(prefix_records[-1].state_snapshot)
            out_records = prefix_records
            cache_hits = total_chunks
            if bar is not None:
                bar.update(total_chunks)
                bar.set_postfix_str(f"step={total_chunks}/{total_chunks} source=cache")
                bar.close()
            doc_cache_status["cache_hits"] = int(cache_hits)
            doc_cache_status["llm_updates"] = 0
            doc_cache_status["completed_from_cache"] = True
            logger.info(
                "Ingest doc=%s completed chunks=%d cache_hits=%d llm_updates=%d fully_cached=%s",
                doc_id,
                total_chunks,
                cache_hits,
                0,
                "True",
            )
            return raw_chunks, out_records, doc_cache_status
        except Exception:
            logger.warning(
                "Full-document cache restore precheck failed doc=%s; falling back to chunk-level replay",
                doc_id,
            )
    for chunk in raw_chunks:
        step_idx = int(chunk.chunk_index) + 1
        source_label = "llm"
        if cache_prefix_valid and use_cache and cached_records_by_chunk_id:
            cached = _resolve_valid_cached_record(
                chunk=chunk,
                strategy=strategy,
                cached_records_by_chunk_id=cached_records_by_chunk_id,
            )
            if cached is not None:
                out_records.append(cached)
                try:
                    state = state_from_snapshot(cached.state_snapshot)
                    cache_hits += 1
                    source_label = "cache"
                    if on_chunk_processed:
                        on_chunk_processed(cached, True)
                    if bar is not None:
                        bar.update(1)
                        bar.set_postfix_str(f"step={step_idx}/{total_chunks} source={source_label}")
                    continue
                except Exception:
                    # Fall back to regeneration if snapshot cannot be restored.
                    logger.warning(
                        "Cache snapshot restore failed doc=%s chunk_id=%s; regenerating from this step",
                        doc_id,
                        chunk.chunk_id,
                    )
                    out_records.pop()
        # From first miss onward, force regeneration to avoid stale downstream state.
        cache_prefix_valid = False

        prompt = strategy.build_prompt(chunk, state)
        raw_model_text = backend.generate_chat_batch(
            [prompt],
            max_tokens=generate.max_tokens,
            temperature=generate.temperature,
            stop=generate.stop,
        )[0]
        parsed_out, state = strategy.apply_output(
            chunk,
            state,
            raw_model_text,
            backend=backend,
            generate=generate,
        )
        out_records.append(
            CanonicalizedChunkRecord(
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                chunk_index=chunk.chunk_index,
                raw_chunk_text=chunk.raw_text,
                canonicalized_chunk_text=parsed_out.updated_chunk_text,
                state_snapshot=state.snapshot(),
                strategy_name=strategy.name,
                parser_status=parsed_out.parser_status,
                raw_model_text=raw_model_text,
                linking_sentence=parsed_out.linking_sentence,
                insertions=list(parsed_out.insertions),
                rejected_entities=list(parsed_out.rejected_entities),
            )
        )
        llm_updates += 1
        if on_chunk_processed:
            on_chunk_processed(out_records[-1], False)
        if bar is not None:
            bar.update(1)
            bar.set_postfix_str(f"step={step_idx}/{total_chunks} source={source_label}")
    if bar is not None:
        bar.close()
    logger.info(
        "Ingest doc=%s completed chunks=%d cache_hits=%d llm_updates=%d fully_cached=%s",
        doc_id,
        total_chunks,
        cache_hits,
        llm_updates,
        "True" if llm_updates == 0 and cache_hits == total_chunks and total_chunks > 0 else "False",
    )
    doc_cache_status["cache_hits"] = int(cache_hits)
    doc_cache_status["llm_updates"] = int(llm_updates)
    doc_cache_status["completed_from_cache"] = bool(total_chunks > 0 and llm_updates == 0 and cache_hits == total_chunks)
    return raw_chunks, out_records, doc_cache_status


def _iter_job_batches(items: list[tuple[str, ChunkRecord, str]], batch_size: int | None) -> Iterable[list[tuple[str, ChunkRecord, str]]]:
    if not items:
        return
    if batch_size is None or int(batch_size) <= 0:
        yield items
        return
    step = int(batch_size)
    for i in range(0, len(items), step):
        yield items[i : i + step]


def _drive_rollups_batched(
    strategy: Any,
    gens: list[Any],
    *,
    backend: LLMBackend,
    generate: GenerateParams | None,
) -> None:
    """Advance many deferred-rollup generators in lockstep.

    All generators are at the same wavefront step, so the rollup requests they yield
    for a given cascade level are independent across documents. We collect one request
    per active generator, issue the rollup prompts as a single ``generate_chat_batch`` call,
    then send the summaries back and repeat for the next (higher) cascade level. This
    replaces the previous behaviour of one single-prompt LLM call per rollup per doc.
    """
    if not gens:
        return
    active: list[tuple[Any, tuple]] = []
    for gen in gens:
        try:
            active.append((gen, next(gen)))
        except StopIteration:
            continue
    while active:
        prompts = [
            strategy.build_rollup_prompt(items=req[0], lower_cycle=req[1], next_cycle=req[2], step=req[3])
            for _, req in active
        ]
        summaries = strategy.run_rollup_prompts(prompts=prompts, backend=backend, generate=generate)
        next_active: list[tuple[Any, tuple]] = []
        for (gen, req), raw in zip(active, summaries):
            summary = raw if (isinstance(raw, str) and raw.strip()) else strategy._fallback_rollup(req[0])
            try:
                next_active.append((gen, gen.send(summary)))
            except StopIteration:
                continue
        active = next_active


def _ingest_documents_wavefront(
    *,
    doc_items: list[tuple[str, str]],
    backend: LLMBackend,
    strategy: Any,
    chunk_size: int,
    chunk_overlap: int,
    tokenizer: Any,
    generate: GenerateParams | None,
    use_cache: bool,
    cached_records_by_chunk_id: dict[str, CanonicalizedChunkRecord] | None,
    on_chunk_processed: Callable[[CanonicalizedChunkRecord, bool], None] | None,
    on_document_completed: Callable[[str], None] | None,
    wavefront_batch_size: int | None,
    precomputed_raw_chunks_by_doc_id: dict[str, list[ChunkRecord]] | None = None,
) -> tuple[list[ChunkRecord], list[CanonicalizedChunkRecord]]:
    generate = generate or GenerateParams()
    debug_batching = str(os.getenv("SAADI_DEBUG_BATCHING", "0")).strip().lower() in {"1", "true", "yes", "on"}

    if not doc_items:
        return [], []

    doc_ids = [str(doc_id) for doc_id, _ in doc_items]
    docs = [text for _, text in doc_items]
    precomp = precomputed_raw_chunks_by_doc_id or {}
    missing_ids = [d for d in doc_ids if d not in precomp]
    if missing_ids:
        missing_docs = [text for doc_id, text in doc_items if str(doc_id) not in precomp]
        grouped = chunk_documents_grouped_records(
            missing_docs,
            doc_ids=missing_ids,
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        missing_chunked = {mid: grouped[i] for i, mid in enumerate(missing_ids)}
    else:
        missing_chunked = {}
    raw_chunks_by_doc: dict[str, list[ChunkRecord]] = {
        doc_id: precomp[doc_id] if doc_id in precomp else missing_chunked.get(doc_id, [])
        for doc_id in doc_ids
    }

    for doc_id in doc_ids:
        logger.info(
            "Ingest doc=%s chunked into %d chunks (chunk_size=%d overlap=%d) [wavefront]",
            doc_id,
            len(raw_chunks_by_doc.get(doc_id, [])),
            chunk_size,
            chunk_overlap,
        )

    states: dict[str, Any] = {
        doc_id: init_discourse_state(hierarchy_levels=[1]) for doc_id in doc_ids
    }
    out_records_by_doc: dict[str, list[CanonicalizedChunkRecord]] = {doc_id: [] for doc_id in doc_ids}
    cache_prefix_valid: dict[str, bool] = {doc_id: True for doc_id in doc_ids}
    cache_hits_by_doc: dict[str, int] = {doc_id: 0 for doc_id in doc_ids}
    llm_updates_by_doc: dict[str, int] = {doc_id: 0 for doc_id in doc_ids}
    doc_cache_statuses = [
        _build_doc_cache_status(
            doc_id=doc_id,
            raw_chunks=raw_chunks_by_doc.get(doc_id, []),
            strategy=strategy,
            use_cache=use_cache,
            cached_records_by_chunk_id=cached_records_by_chunk_id,
        )
        for doc_id in doc_ids
    ]
    doc_cache_status_by_doc = {
        str(status["doc_id"]): status for status in doc_cache_statuses
    }
    _log_cache_readiness_summary(schedule_label="wavefront", doc_statuses=doc_cache_statuses)

    completed_docs: set[str] = set()
    if on_document_completed is not None:
        for doc_id in doc_ids:
            if len(raw_chunks_by_doc.get(doc_id, [])) == 0:
                completed_docs.add(doc_id)
                on_document_completed(doc_id)

    total_chunks = sum(len(chunks) for chunks in raw_chunks_by_doc.values())
    max_chunks_all = max((len(chunks) for chunks in raw_chunks_by_doc.values()), default=0)
    docs_with_chunks = sum(1 for doc_id in doc_ids if len(raw_chunks_by_doc.get(doc_id, [])) > 0)
    if debug_batching:
        logger.info(
            "Wavefront debug init docs_total=%d docs_with_chunks=%d total_chunks=%d max_chunks=%d batch_size=%s use_cache=%s",
            len(doc_ids),
            docs_with_chunks,
            total_chunks,
            max_chunks_all,
            str(wavefront_batch_size),
            str(use_cache),
        )
    bar = _new_progress_bar(
        total=total_chunks,
        desc="Ingest chunks (wavefront)",
        unit="chunk",
        leave=False,
    )
    remaining_doc_ids: list[str] = []
    for doc_id in doc_ids:
        status = doc_cache_status_by_doc.get(doc_id, {})
        raw_chunks = raw_chunks_by_doc.get(doc_id, [])
        if not raw_chunks:
            continue
        if bool(status.get("fully_cached", False)):
            prefix_records = list(status.get("prefix_records", []))
            try:
                state_from_snapshot(prefix_records[-1].state_snapshot)
                out_records_by_doc[doc_id] = prefix_records
                cache_hits_by_doc[doc_id] = len(prefix_records)
                status["cache_hits"] = len(prefix_records)
                status["llm_updates"] = 0
                status["completed_from_cache"] = True
                completed_docs.add(doc_id)
                if bar is not None:
                    bar.update(len(prefix_records))
                    bar.set_postfix_str("source=cache")
                if on_document_completed is not None:
                    on_document_completed(doc_id)
                continue
            except Exception:
                logger.warning(
                    "Full-document cache restore precheck failed doc=%s [wavefront]; falling back to chunk-level replay",
                    doc_id,
                )
                status["fully_cached"] = False
                status["partially_cached"] = int(status.get("cached_prefix_chunks", 0)) > 0
                status["remaining_chunks"] = int(status.get("total_chunks", 0))
        remaining_doc_ids.append(doc_id)

    max_chunks = max((len(raw_chunks_by_doc.get(doc_id, [])) for doc_id in remaining_doc_ids), default=0)

    def _mark_doc_completed_if_last_chunk(doc_id: str, chunk_idx: int) -> None:
        if doc_id in completed_docs:
            return
        chunks = raw_chunks_by_doc.get(doc_id, [])
        if chunk_idx == len(chunks) - 1:
            completed_docs.add(doc_id)
            if on_document_completed is not None:
                on_document_completed(doc_id)

    for step in range(max_chunks):
        pending_jobs: list[tuple[str, ChunkRecord, str]] = []
        cache_hits_this_step = 0

        for doc_id in remaining_doc_ids:
            raw_chunks = raw_chunks_by_doc.get(doc_id, [])
            if step >= len(raw_chunks):
                continue
            chunk = raw_chunks[step]

            if cache_prefix_valid.get(doc_id, True) and use_cache and cached_records_by_chunk_id:
                cached = _resolve_valid_cached_record(
                    chunk=chunk,
                    strategy=strategy,
                    cached_records_by_chunk_id=cached_records_by_chunk_id,
                )
                if cached is not None:
                    out_records_by_doc[doc_id].append(cached)
                    try:
                        states[doc_id] = state_from_snapshot(cached.state_snapshot)
                        cache_hits_by_doc[doc_id] += 1
                        cache_hits_this_step += 1
                        if on_chunk_processed:
                            on_chunk_processed(cached, True)
                        if bar is not None:
                            bar.update(1)
                            bar.set_postfix_str(f"step={step + 1}/{max_chunks} source=cache")
                        _mark_doc_completed_if_last_chunk(doc_id, int(chunk.chunk_index))
                        continue
                    except Exception:
                        logger.warning(
                            "Cache snapshot restore failed doc=%s chunk_id=%s; regenerating from this step",
                            doc_id,
                            chunk.chunk_id,
                        )
                        out_records_by_doc[doc_id].pop()

            # From first miss onward, force regeneration for this document.
            cache_prefix_valid[doc_id] = False
            prompt = strategy.build_prompt(chunk, states[doc_id])
            pending_jobs.append((doc_id, chunk, prompt))

        active_docs_this_step = sum(1 for doc_id in remaining_doc_ids if step < len(raw_chunks_by_doc.get(doc_id, [])))
        if debug_batching:
            logger.info(
                "Wavefront debug step=%d/%d active_docs=%d cache_hits=%d pending_jobs=%d",
                step + 1,
                max_chunks,
                active_docs_this_step,
                cache_hits_this_step,
                len(pending_jobs),
            )
        if step == 0 and not use_cache and docs_with_chunks > 1 and len(pending_jobs) <= 1:
            logger.warning(
                "Wavefront step=1 has pending_jobs=%d despite docs_with_chunks=%d and use_cache=false. "
                "This usually indicates the run is not using wavefront ingestion path or input docs include <=1 non-empty document.",
                len(pending_jobs),
                docs_with_chunks,
            )

        for batch_idx, jobs in enumerate(_iter_job_batches(pending_jobs, wavefront_batch_size), start=1):
            prompts = [prompt for _, _, prompt in jobs]
            if debug_batching:
                logger.info(
                    "Wavefront debug step=%d/%d batch=%d batch_size=%d",
                    step + 1,
                    max_chunks,
                    batch_idx,
                    len(prompts),
                )
            raw_model_texts = backend.generate_chat_batch(
                prompts,
                max_tokens=generate.max_tokens,
                temperature=generate.temperature,
                stop=generate.stop,
            )
            # Parse each document's chunk output first, deferring hierarchy rollups so
            # the rollup LLM calls can be batched across documents (see
            # _drive_rollups_batched). Fall back to the per-document synchronous path
            # for strategies that don't support deferred rollups.
            use_deferred = hasattr(strategy, "apply_output_deferred")
            committed: list[tuple[str, ChunkRecord, str, Any, Any]] = []
            if use_deferred:
                pending_gens: list[Any] = []
                for (doc_id, chunk, _), raw_model_text in zip(jobs, raw_model_texts):
                    parsed_out, new_state, rollup_gen = strategy.apply_output_deferred(
                        chunk, states[doc_id], raw_model_text
                    )
                    if rollup_gen is not None:
                        pending_gens.append(rollup_gen)
                    committed.append((doc_id, chunk, raw_model_text, parsed_out, new_state))
                _drive_rollups_batched(strategy, pending_gens, backend=backend, generate=generate)
            else:
                for (doc_id, chunk, _), raw_model_text in zip(jobs, raw_model_texts):
                    parsed_out, new_state = strategy.apply_output(
                        chunk,
                        states[doc_id],
                        raw_model_text,
                        backend=backend,
                        generate=generate,
                    )
                    committed.append((doc_id, chunk, raw_model_text, parsed_out, new_state))

            for doc_id, chunk, raw_model_text, parsed_out, new_state in committed:
                states[doc_id] = new_state
                rec = CanonicalizedChunkRecord(
                    doc_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    chunk_index=chunk.chunk_index,
                    raw_chunk_text=chunk.raw_text,
                    canonicalized_chunk_text=parsed_out.updated_chunk_text,
                    state_snapshot=new_state.snapshot(),
                    strategy_name=strategy.name,
                    parser_status=parsed_out.parser_status,
                    raw_model_text=raw_model_text,
                    linking_sentence=parsed_out.linking_sentence,
                    insertions=list(parsed_out.insertions),
                    rejected_entities=list(parsed_out.rejected_entities),
                )
                out_records_by_doc[doc_id].append(rec)
                llm_updates_by_doc[doc_id] += 1
                if on_chunk_processed:
                    on_chunk_processed(rec, False)
                if bar is not None:
                    bar.update(1)
                    bar.set_postfix_str(f"step={step + 1}/{max_chunks} source=llm")
                _mark_doc_completed_if_last_chunk(doc_id, int(chunk.chunk_index))

    if bar is not None:
        bar.close()

    all_raw: list[ChunkRecord] = []
    all_canon: list[CanonicalizedChunkRecord] = []
    for doc_id in doc_ids:
        raw = raw_chunks_by_doc.get(doc_id, [])
        canon = out_records_by_doc.get(doc_id, [])
        all_raw.extend(raw)
        all_canon.extend(canon)
        status = doc_cache_status_by_doc.get(doc_id)
        if status is not None:
            status["cache_hits"] = int(cache_hits_by_doc.get(doc_id, 0))
            status["llm_updates"] = int(llm_updates_by_doc.get(doc_id, 0))
            status["completed_from_cache"] = bool(
                len(raw) > 0 and llm_updates_by_doc.get(doc_id, 0) == 0 and cache_hits_by_doc.get(doc_id, 0) == len(raw)
            )
        logger.info(
            "Ingest doc=%s completed chunks=%d cache_hits=%d llm_updates=%d fully_cached=%s [wavefront]",
            doc_id,
            len(raw),
            cache_hits_by_doc.get(doc_id, 0),
            llm_updates_by_doc.get(doc_id, 0),
            "True" if len(raw) > 0 and llm_updates_by_doc.get(doc_id, 0) == 0 and cache_hits_by_doc.get(doc_id, 0) == len(raw) else "False",
        )
    _log_cache_result_summary(schedule_label="wavefront", doc_statuses=doc_cache_statuses)
    return all_raw, all_canon


def ingest_documents(
    *,
    documents_by_id: dict[str, str],
    backend: LLMBackend,
    strategy_name: str = "hierarchical",
    strategy_kwargs: dict | None = None,
    cache_namespace: dict[str, Any] | None = None,
    use_cache: bool = False,
    cache_save_every_steps: int = 10,
    cache_save_each_document: bool = False,
    chunk_size: int = 500,
    chunk_overlap: int = 0,
    tokenizer=None,
    output_dir: str | None = None,
    generate: GenerateParams | None = None,
    write_raw_chunks: bool = False,
    schedule: str = "document",
    wavefront_batch_size: int | None = None,
    precomputed_raw_chunks_by_doc_id: dict[str, list[ChunkRecord]] | None = None,
):
    backend = resolve_backend(backend)
    logger.info(
        "Starting ingest for %d documents strategy=%s schedule=%s use_cache=%s cache_save_each_document=%s",
        len(documents_by_id),
        strategy_name,
        str(schedule),
        use_cache,
        cache_save_each_document,
    )
    strategy = get_strategy(strategy_name, **(strategy_kwargs or {}))
    all_raw: list[ChunkRecord] = []
    all_canon: list[CanonicalizedChunkRecord] = []

    cache_namespace = cache_namespace or {}
    if not cache_namespace.get("active_entity_config"):
        cache_namespace["active_entity_config"] = cache_namespace.get("active_object_config")
    if not cache_namespace.get("active_entity_config"):
        cache_namespace["active_entity_config"] = active_entity_config_label(
            getattr(strategy, "max_entities", getattr(strategy, "max_objects", 5)),
            getattr(strategy, "max_aliases", 2),
            getattr(strategy, "max_desc", 1),
        )
    if not cache_namespace.get("active_entity_selector"):
        cache_namespace["active_entity_selector"] = cache_namespace.get("active_object_selector")
    cache_store: dict[str, CanonicalizedChunkRecord] = {}
    processed_steps = 0
    dirty_steps_since_flush = 0

    if output_dir and use_cache:
        namespace_kwargs = dict(
            output_dir=output_dir,
            dataset_label=cache_namespace.get("dataset"),
            method_label=cache_namespace.get("method", strategy_name),
            hierarchy_branching_factor=getattr(strategy, "branching_factor", getattr(strategy, "k", None)),
            chunk_size=int(chunk_size),
            active_entity_config_label=cache_namespace.get("active_entity_config"),
            llm_label=cache_namespace.get("llm"),
            active_entity_selector_label=cache_namespace.get("active_entity_selector"),
        )
        cached_records = load_canonicalized_chunk_records_cache(**namespace_kwargs)
        cache_store = {r.chunk_id: r for r in cached_records}
        logger.info("Loaded canonicalization cache records=%d", len(cache_store))
        # A cache records which text the model saw as L0 history. Records built
        # under the other setting would silently change the context of every
        # later chunk, so they are not reused. Indexes written before the field
        # existed were built with the anchored copies.
        strategy_history_mode = getattr(strategy, "history_l0_text", None)
        if cache_store and strategy_history_mode is not None and getattr(strategy, "stores_hierarchy", True):
            cache_index_payload = load_llm_history_cache_index(**namespace_kwargs) or {}
            cached_history_mode = str(cache_index_payload.get("history_l0_text") or "anchored")
            if cached_history_mode != str(strategy_history_mode):
                logger.warning(
                    "Ignoring canonicalization cache records=%d: cache history_l0_text=%s differs from the "
                    "configured history_l0_text=%s",
                    len(cache_store),
                    cached_history_mode,
                    strategy_history_mode,
                )
                cache_store = {}
        # Records built under an explicitly different anchor-matching setting
        # carry a different memory, so they are not reused either. Indexes
        # without the field predate it and are reused as they are.
        strategy_anchor_matching = getattr(strategy, "anchor_matching", None)
        if cache_store and strategy_anchor_matching is not None:
            cache_index_payload = load_llm_history_cache_index(**namespace_kwargs) or {}
            cached_anchor_matching = cache_index_payload.get("anchor_matching")
            if cached_anchor_matching and str(cached_anchor_matching) != str(strategy_anchor_matching):
                logger.warning(
                    "Ignoring canonicalization cache records=%d: cache anchor_matching=%s differs from the "
                    "configured anchor_matching=%s",
                    len(cache_store),
                    cached_anchor_matching,
                    strategy_anchor_matching,
                )
                cache_store = {}

    def flush_namespace_cache() -> dict | None:
        nonlocal dirty_steps_since_flush
        if not output_dir:
            return None
        records = list(cache_store.values()) if cache_store else list(all_canon)
        out = export_llm_history_cache(
            records,
            output_dir=output_dir,
            dataset_label=cache_namespace.get("dataset"),
            method_label=cache_namespace.get("method", strategy_name),
            strategy_name=getattr(strategy, "name", None),
            hierarchy_branching_factor=getattr(strategy, "branching_factor", getattr(strategy, "k", None)),
            chunk_size=int(chunk_size),
            active_entity_config_label=cache_namespace.get("active_entity_config"),
            llm_label=cache_namespace.get("llm"),
            active_entity_selector_label=cache_namespace.get("active_entity_selector"),
            active_entity_min_relevance=float(
                getattr(
                    strategy,
                    "active_entity_min_relevance",
                    getattr(strategy, "active_entity_min_relevance", 0.0),
                )
            ),
            memory_update_mode=getattr(strategy, "memory_update_mode", None),
            history_l0_text=getattr(strategy, "history_l0_text", None),
            anchor_matching=getattr(strategy, "anchor_matching", None),
            stores_hierarchy=bool(getattr(strategy, "stores_hierarchy", True)),
        )
        dirty_steps_since_flush = 0
        return out

    def on_chunk_processed(rec: CanonicalizedChunkRecord, used_cache: bool) -> None:
        nonlocal processed_steps, dirty_steps_since_flush
        processed_steps += 1
        if not output_dir or used_cache:
            return
        cache_store[rec.chunk_id] = rec
        dirty_steps_since_flush += 1
        if cache_save_every_steps > 0 and dirty_steps_since_flush >= cache_save_every_steps:
            logger.info(
                "Checkpointing ingest cache at processed_steps=%d",
                processed_steps,
            )
            flush_namespace_cache()

    doc_items = list(documents_by_id.items())
    schedule_norm = str(schedule or "document").strip().lower()
    if schedule_norm == "wavefront" and len(doc_items) <= 1:
        logger.warning(
            "Wavefront schedule requested with %d document(s). Cross-document batching cannot occur; "
            "each generation call will have at most one prompt.",
            len(doc_items),
        )
    if schedule_norm in {"document", "sequential"}:
        docs_bar = _new_progress_bar(
            total=len(doc_items),
            desc="Ingest documents",
            unit="doc",
            leave=True,
        )
        doc_cache_statuses: list[dict[str, Any]] = []
        for doc_idx, (doc_id, text) in enumerate(doc_items, start=1):
            logger.info("Processing document doc_id=%s", doc_id)
            raw, canon, doc_cache_status = ingest_document(
                doc_id=str(doc_id),
                document=text,
                backend=backend,
                strategy=strategy,
                use_cache=use_cache,
                cached_records_by_chunk_id=cache_store if (output_dir and use_cache) else None,
                on_chunk_processed=on_chunk_processed if output_dir else None,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                tokenizer=tokenizer,
                generate=generate,
                precomputed_raw_chunks=(precomputed_raw_chunks_by_doc_id or {}).get(str(doc_id)),
            )
            all_raw.extend(raw)
            all_canon.extend(canon)
            doc_cache_statuses.append(doc_cache_status)
            logger.info("Document done doc_id=%s raw_chunks=%d canonicalized=%d", doc_id, len(raw), len(canon))
            if output_dir and cache_save_each_document and dirty_steps_since_flush > 0:
                logger.info(
                    "Checkpointing ingest cache at document boundary doc=%d/%d doc_id=%s processed_steps=%d",
                    doc_idx,
                    len(doc_items),
                    doc_id,
                    processed_steps,
                )
                flush_namespace_cache()
            if docs_bar is not None:
                docs_bar.update(1)
                docs_bar.set_postfix_str(f"doc={doc_idx}/{len(doc_items)}")
        if docs_bar is not None:
            docs_bar.close()
        _log_cache_result_summary(schedule_label=schedule_norm, doc_statuses=doc_cache_statuses)
    elif schedule_norm == "wavefront":
        doc_positions = {str(doc_id): idx + 1 for idx, (doc_id, _) in enumerate(doc_items)}

        def on_document_completed(doc_id: str) -> None:
            if output_dir and cache_save_each_document and dirty_steps_since_flush > 0:
                logger.info(
                    "Checkpointing ingest cache at document boundary doc=%d/%d doc_id=%s processed_steps=%d",
                    doc_positions.get(str(doc_id), 0),
                    len(doc_items),
                    str(doc_id),
                    processed_steps,
                )
                flush_namespace_cache()

        raw, canon = _ingest_documents_wavefront(
            doc_items=[(str(doc_id), text) for doc_id, text in doc_items],
            backend=backend,
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tokenizer=tokenizer,
            generate=generate,
            use_cache=use_cache,
            cached_records_by_chunk_id=cache_store if (output_dir and use_cache) else None,
            on_chunk_processed=on_chunk_processed if output_dir else None,
            on_document_completed=on_document_completed if (output_dir and cache_save_each_document) else None,
            wavefront_batch_size=wavefront_batch_size,
            precomputed_raw_chunks_by_doc_id=precomputed_raw_chunks_by_doc_id,
        )
        all_raw.extend(raw)
        all_canon.extend(canon)
    else:
        raise ValueError(
            "Invalid ingest schedule. Expected one of: document, sequential, wavefront "
            f"(got: {schedule!r})."
        )

    artifacts = None
    if output_dir:
        # When a canonical prepared dataset grows or is corrected in place,
        # old cache records can reference document/chunk IDs that are no
        # longer present. Valid active records are already content-checked by
        # _resolve_valid_cached_record; remove inactive leftovers before the
        # final index export so the cache exactly matches the current corpus.
        active_chunk_ids = {chunk.chunk_id for chunk in all_raw}
        stale_chunk_ids = set(cache_store) - active_chunk_ids
        if stale_chunk_ids:
            logger.info(
                "Pruning stale ingest cache records=%d before final export",
                len(stale_chunk_ids),
            )
            cache_store = {
                chunk_id: record
                for chunk_id, record in cache_store.items()
                if chunk_id in active_chunk_ids
            }
        paths = default_artifact_paths(output_dir)
        cache_index = flush_namespace_cache() or {}
        if "cache_namespace_dir" in cache_index:
            paths.llm_cache_dir = Path(str(cache_index["cache_namespace_dir"]))
        if "index_path" in cache_index:
            paths.llm_cache_index_json = Path(str(cache_index["index_path"]))
        # Keep ingest artifacts with the namespace cache so each experiment folder is self-contained.
        artifact_dir = paths.llm_cache_dir if "cache_namespace_dir" in cache_index else paths.root
        paths.raw_chunks_jsonl = artifact_dir / "raw_chunks.jsonl"
        if write_raw_chunks:
            write_jsonl(paths.raw_chunks_jsonl, [c.to_dict() for c in all_raw])
        else:
            logger.info("Skipping raw chunk artifact write (write_raw_chunks=false)")
        artifacts = paths
        logger.info(
            "Ingest artifacts written raw=%s cache_dir=%s",
            paths.raw_chunks_jsonl,
            paths.llm_cache_dir,
        )
    logger.info("Ingest pipeline completed documents=%d raw_chunks=%d canonicalized_chunks=%d", len(documents_by_id), len(all_raw), len(all_canon))
    return all_raw, all_canon, artifacts
