"""Generic retrieval-only pipeline for Saadi and its retrieval ablations."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Iterable

from ..core.types import CanonicalizedChunkRecord, ChunkRecord, CorpusArtifact
from ..evaluation.retrieval_metrics import evaluate_retrieval_results, write_eval_report_json
from ..indexing.corpus_builder import build_corpus_from_canonicalized, build_corpus_from_raw_chunks
from ..indexing.labeling import build_retrieval_examples
from ..indexing.retrieval_methods.l0aj import (
    build_l0aj_corpus,
    build_l0aj_corpus_by_doc,
    format_l0aj_method_label,
)
from ..indexing.retrieval_methods.mbr import (
    DEFAULT_MBR_LAMBDA_H,
    DEFAULT_MBR_LAMBDA_M,
    DEFAULT_MBR_LAMBDA_T,
    DEFAULT_MBR_MEMORY_AGGREGATION,
    build_mbr_diagnostic_rows,
    build_mbr_document_memory_index,
    format_mbr_method_label,
    format_mbr_linear_method_label,
    format_mbr_pool_method_label,
    run_mbr_retrieval_pipeline,
    run_mbr_linear_retrieval_pipeline,
    run_mbr_pool_retrieval_pipeline,
)
from ..io.hierarchy_cache import (
    load_doc_ids_from_cache_index,
    load_canonicalized_chunk_records_from_cache_index,
    load_raw_chunk_records_from_cache_index,
)
from ..io.manifests import GIT_PROVENANCE_FIELDS, git_provenance
from .retrieval_pipeline import run_retrieval_pipeline, write_retrieval_results_json

logger = logging.getLogger(__name__)

def normalize_cached_retrieval_method(value: str | None) -> str:
    raw = str(value or "mbr").strip().lower()
    alias = {
        "canonicalized": "l0",
        "stateful": "l0",
        "contextualized": "l0",
        "canonical": "l0",
        "l0aj": "l0aj",
        "l0_augmented": "l0aj",
        "l0+ae": "l0aj",
        "l0+ao": "l0aj",
        "mbr": "mbr",
        "mbr_linear": "mbr_linear",
        "mbr-linear": "mbr_linear",
        "mbr_pool": "mbr_pool",
        "mbr-pool": "mbr_pool",
        "mbr_onepool": "mbr_pool",
        "mbr-onepool": "mbr_pool",
    }
    out = alias.get(raw, raw)
    if out not in {"l0", "l0aj", "mbr", "mbr_linear", "mbr_pool"}:
        raise ValueError(f"Invalid method: {value!r}. Expected one of: l0, l0aj, mbr, mbr_linear, mbr_pool")
    return out


def _safe_output_component(value: str, *, default: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "")).strip("._")
    return out or default


def normalize_cached_retrieval_scope(value: str | None) -> str:
    raw = str(value or "global").strip().lower()
    if raw not in {"global", "per_document"}:
        raise ValueError(f"Invalid scope: {value!r}. Expected global or per_document.")
    return raw


def normalize_positive_int(value: Any, *, field_name: str) -> int:
    try:
        out = int(value)
    except Exception as exc:
        raise ValueError(f"{field_name} must be a positive integer (got: {value!r})") from exc
    if out <= 0:
        raise ValueError(f"{field_name} must be > 0 (got: {out})")
    return out


def resolve_cached_report_root(llm_cache_index_json: str | Path, output_dir: str | None = None) -> Path:
    if output_dir:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        return root
    return Path(llm_cache_index_json).resolve().parent


def read_qa_entries(qa_json: str | Path) -> list[dict[str, Any]]:
    with Path(qa_json).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"QA JSON must be a list: {qa_json}")
    return [row for row in payload if isinstance(row, dict)]


def _sample_doc_ids(values: Iterable[str], *, limit: int = 5) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        doc_id = str(value).strip()
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        out.append(doc_id)
        if len(out) >= limit:
            break
    return out


def _align_qa_entries_to_cached_docs(
    qa_entries: list[dict[str, Any]],
    chunks_by_doc: dict[str, list[ChunkRecord]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cached_doc_ids = sorted(doc_id for doc_id in chunks_by_doc.keys() if str(doc_id).strip())
    cached_doc_id_set = set(cached_doc_ids)

    matched_entries: list[dict[str, Any]] = []
    qa_doc_ids_all: list[str] = []
    matched_doc_ids: list[str] = []
    for entry in qa_entries:
        if not isinstance(entry, dict):
            continue
        doc_id = str(entry.get("document_id", "")).strip()
        if not doc_id:
            continue
        qa_doc_ids_all.append(doc_id)
        if doc_id not in cached_doc_id_set:
            continue
        matched_entries.append(entry)
        matched_doc_ids.append(doc_id)

    metadata = {
        "cached_documents": len(cached_doc_ids),
        "qa_entries_loaded": len(qa_entries),
        "qa_entries_matched_cached_docs": len(matched_entries),
        "qa_entries_unmatched_cached_docs": max(0, len(qa_entries) - len(matched_entries)),
        "qa_document_ids_sample": _sample_doc_ids(qa_doc_ids_all),
        "cached_document_ids_sample": _sample_doc_ids(cached_doc_ids),
        "matched_document_ids_sample": _sample_doc_ids(matched_doc_ids),
    }
    return matched_entries, metadata


def merge_retrieval_report_index(report_dir: Path, run_rows: list[dict[str, Any]], *, metadata: dict[str, Any]) -> Path:
    index_path = report_dir / "report_index.json"
    payload: dict[str, Any] = {}
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            existing = json.load(f)
        if isinstance(existing, dict):
            payload = existing

    existing_runs = payload.get("runs", []) or []
    if not isinstance(existing_runs, list):
        existing_runs = []

    def _same_run(a: dict[str, Any], b: dict[str, Any]) -> bool:
        return (
            str(a.get("retriever", "")) == str(b.get("retriever", ""))
            and str(a.get("method", "")) == str(b.get("method", ""))
            and str(a.get("scope", "")) == str(b.get("scope", ""))
        )

    merged = [row for row in existing_runs if isinstance(row, dict)]
    for run in run_rows:
        merged = [row for row in merged if not _same_run(row, run)]
        merged.append(run)

    payload.update(metadata)
    payload["runs"] = merged
    payload["methods_executed"] = sorted({str(row.get("method", "")) for row in merged if str(row.get("method", "")).strip()})
    payload["scopes_executed"] = sorted({str(row.get("scope", "")) for row in merged if str(row.get("scope", "")).strip()})

    with index_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return index_path


def _build_raw_chunk_rows(chunks: Iterable[ChunkRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for chunk in sorted(chunks, key=lambda c: (str(c.doc_id), int(c.chunk_index), str(c.chunk_id))):
        rows.append(
            {
                "doc_id": str(chunk.doc_id),
                "chunk_id": str(chunk.chunk_id),
                "chunk_index": int(chunk.chunk_index),
                "raw_chunk_text": str(chunk.raw_text or ""),
                "canonicalized_chunk_text": "",
                "retrieval_text": str(chunk.raw_text or ""),
                "selected_entities": [],
                "document_memory_size": 0,
                "retrieval_method": "raw_chunks",
            }
        )
    return rows


def _build_l0_rows(chunks: Iterable[CanonicalizedChunkRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for chunk in sorted(chunks, key=lambda c: (str(c.doc_id), int(c.chunk_index), str(c.chunk_id))):
        rows.append(
            {
                "doc_id": str(chunk.doc_id),
                "chunk_id": str(chunk.chunk_id),
                "chunk_index": int(chunk.chunk_index),
                "raw_chunk_text": str(chunk.raw_chunk_text or ""),
                "canonicalized_chunk_text": str(chunk.canonicalized_chunk_text or ""),
                "retrieval_text": str(chunk.canonicalized_chunk_text or ""),
                "selected_entities": [],
                "document_memory_size": 0,
                "retrieval_method": "l0",
            }
        )
    return rows


def _build_corpus_by_doc_from_raw(chunks: Iterable[ChunkRecord]) -> dict[str, CorpusArtifact]:
    grouped: dict[str, list[ChunkRecord]] = {}
    for chunk in chunks:
        grouped.setdefault(str(chunk.doc_id), []).append(chunk)
    return {
        doc_id: build_corpus_from_raw_chunks(doc_chunks, metadata={"source": "raw", "retrieval_mode": "raw_chunks", "doc_id": doc_id})
        for doc_id, doc_chunks in grouped.items()
    }


def _build_corpus_by_doc_from_canonicalized(chunks: Iterable[CanonicalizedChunkRecord]) -> dict[str, CorpusArtifact]:
    grouped: dict[str, list[CanonicalizedChunkRecord]] = {}
    for chunk in chunks:
        grouped.setdefault(str(chunk.doc_id), []).append(chunk)
    return {
        doc_id: build_corpus_from_canonicalized(doc_chunks, metadata={"source": "canonicalized", "retrieval_mode": "l0", "doc_id": doc_id})
        for doc_id, doc_chunks in grouped.items()
    }


def build_cached_retrieval_artifacts(
    *,
    method: str,
    raw_chunks: list[ChunkRecord],
    canonicalized_chunks: list[CanonicalizedChunkRecord] | None,
    augmentation_j: int,
    max_aliases: int,
    max_desc: int,
    active_entity_selector: str = "bm25",
    lambda_t: float = DEFAULT_MBR_LAMBDA_T,
    lambda_n: float = DEFAULT_MBR_LAMBDA_M,
    lambda_h: float = DEFAULT_MBR_LAMBDA_H,
    normalization_method: str = "none",
    memory_name_weight: float = 1.0,
    memory_alias_weight: float = 0.0,
    memory_description_weight: float = 0.0,
) -> tuple[str, CorpusArtifact, dict[str, CorpusArtifact] | None, list[dict[str, Any]]]:
    normalized = normalize_cached_retrieval_method(method)
    if canonicalized_chunks is None or not canonicalized_chunks:
        raise ValueError(f"Method {normalized!r} requires canonicalized chunks from the cache index")
    if normalized == "l0":
        return (
            "l0",
            build_corpus_from_canonicalized(canonicalized_chunks, metadata={"source": "canonicalized", "retrieval_mode": "l0"}),
            _build_corpus_by_doc_from_canonicalized(canonicalized_chunks),
            _build_l0_rows(canonicalized_chunks),
        )
    if normalized == "mbr":
        method_label = format_mbr_method_label(
            lambda_t=float(lambda_t),
            lambda_n=float(lambda_n),
            lambda_h=float(lambda_h),
            normalization_method=str(normalization_method or "none"),
            memory_name_weight=float(memory_name_weight),
            memory_alias_weight=float(memory_alias_weight),
            memory_description_weight=float(memory_description_weight),
        )
        return (
            method_label,
            build_corpus_from_raw_chunks(
                raw_chunks,
                metadata={"source": "raw", "retrieval_mode": method_label},
            ),
            _build_corpus_by_doc_from_raw(raw_chunks),
            _build_raw_chunk_rows(raw_chunks),
        )
    if normalized == "mbr_linear":
        method_label = format_mbr_linear_method_label(
            lambda_t=float(lambda_t),
            lambda_n=float(lambda_n),
            lambda_h=float(lambda_h),
            normalization_method=str(normalization_method or "none"),
        )
        return (
            method_label,
            build_corpus_from_raw_chunks(
                raw_chunks,
                metadata={"source": "raw", "retrieval_mode": method_label},
            ),
            _build_corpus_by_doc_from_raw(raw_chunks),
            _build_raw_chunk_rows(raw_chunks),
        )
    if normalized == "mbr_pool":
        method_label = format_mbr_pool_method_label(
            lambda_t=float(lambda_t),
            lambda_n=float(lambda_n),
            lambda_h=float(lambda_h),
            normalization_method=str(normalization_method or "none"),
        )
        return (
            method_label,
            build_corpus_from_raw_chunks(
                raw_chunks,
                metadata={"source": "raw", "retrieval_mode": method_label},
            ),
            _build_corpus_by_doc_from_raw(raw_chunks),
            _build_raw_chunk_rows(raw_chunks),
        )
    method_label = format_l0aj_method_label(augmentation_j)
    corpus, rows = build_l0aj_corpus(
        canonicalized_chunks,
        augmentation_j=augmentation_j,
        max_aliases=max_aliases,
        max_desc=max_desc,
        active_entity_selector=active_entity_selector,
    )
    corpus_by_doc, _ = build_l0aj_corpus_by_doc(
        canonicalized_chunks,
        augmentation_j=augmentation_j,
        max_aliases=max_aliases,
        max_desc=max_desc,
        active_entity_selector=active_entity_selector,
    )
    return method_label, corpus, corpus_by_doc, rows


def run_cached_retrieval_eval(
    *,
    llm_cache_index_json: str | Path,
    qa_json: str | Path | None = None,
    qa_entries: list[dict[str, Any]] | None = None,
    max_qa_entries: int | None = None,
    method: str = "mbr",
    retriever_name: str = "bm25",
    retrieve_k: int = 10,
    eval_k: int = 5,
    retrieval_scope: str = "global",
    augmentation_j: int = 3,
    max_aliases: int = 2,
    max_desc: int = 1,
    active_entity_selector: str = "bm25",
    lambda_t: float = DEFAULT_MBR_LAMBDA_T,
    lambda_n: float = DEFAULT_MBR_LAMBDA_M,
    lambda_h: float = DEFAULT_MBR_LAMBDA_H,
    normalization_method: str = "none",
    memory_name_weight: float = 1.0,
    memory_alias_weight: float = 0.0,
    memory_description_weight: float = 0.0,
    memory_aggregation: str = DEFAULT_MBR_MEMORY_AGGREGATION,
    output_dir: str | None = None,
) -> dict[str, Any]:
    retrieval_provenance = git_provenance(
        cwd=str(Path(__file__).resolve().parents[3])
    )
    retrieve_k = normalize_positive_int(retrieve_k, field_name="retrieve_k")
    if max_qa_entries is not None:
        max_qa_entries = normalize_positive_int(max_qa_entries, field_name="max_qa_entries")
    eval_k = normalize_positive_int(eval_k, field_name="eval_k")
    if eval_k > retrieve_k:
        raise ValueError(f"eval_k={eval_k} exceeds retrieve_k={retrieve_k}; increase retrieval depth")
    augmentation_j = normalize_positive_int(augmentation_j, field_name="augmentation_j")
    if max_aliases < 0:
        raise ValueError(f"max_aliases must be >= 0 (got: {max_aliases})")
    if max_desc < 0:
        raise ValueError(f"max_desc must be >= 0 (got: {max_desc})")

    normalized_method = normalize_cached_retrieval_method(method)
    scope = normalize_cached_retrieval_scope(retrieval_scope)
    if normalized_method == "mbr" and scope != "per_document":
        raise ValueError(
            "MBR is currently implemented only for retrieval_scope='per_document'. "
            "Global scope is not implemented."
        )
    if normalized_method == "mbr_linear" and scope != "per_document":
        raise ValueError(
            "MBR Linear is currently implemented only for retrieval_scope='per_document'. "
            "Global scope is not implemented."
        )
    if normalized_method == "mbr_pool" and scope != "per_document":
        raise ValueError(
            "MBR Pool is currently implemented only for retrieval_scope='per_document'. "
            "Global scope is not implemented."
        )
    llm_cache_index_json = str(llm_cache_index_json)
    qa_json_str = str(qa_json) if qa_json is not None else ""

    raw_chunks = load_raw_chunk_records_from_cache_index(llm_cache_index_json)
    if not raw_chunks:
        raise ValueError(f"No raw chunks could be loaded from cache index: {llm_cache_index_json}")
    canonicalized_chunks = None
    if normalized_method in {"l0", "l0aj", "mbr", "mbr_linear", "mbr_pool"}:
        canonicalized_chunks = load_canonicalized_chunk_records_from_cache_index(llm_cache_index_json)
        if not canonicalized_chunks:
            raise ValueError(f"No canonicalized chunks could be loaded from cache index: {llm_cache_index_json}")

    chunks_by_doc: dict[str, list[ChunkRecord]] = {}
    for chunk in raw_chunks:
        chunks_by_doc.setdefault(str(chunk.doc_id), []).append(chunk)
    resolved_qa_entries = list(qa_entries) if qa_entries is not None else read_qa_entries(qa_json_str)
    aligned_qa_entries, qa_alignment = _align_qa_entries_to_cached_docs(resolved_qa_entries, chunks_by_doc)
    if max_qa_entries is not None:
        aligned_qa_entries = aligned_qa_entries[:max_qa_entries]
    qa_alignment["qa_entries_used"] = len(aligned_qa_entries)
    logger.info(
        "Cached retrieval QA alignment cached_docs=%d loaded=%d matched=%d unmatched=%d",
        int(qa_alignment["cached_documents"]),
        int(qa_alignment["qa_entries_loaded"]),
        int(qa_alignment["qa_entries_matched_cached_docs"]),
        int(qa_alignment["qa_entries_unmatched_cached_docs"]),
    )
    if resolved_qa_entries and not aligned_qa_entries:
        raise RuntimeError(
            "Loaded QA entries but none matched the cached document set. "
            "This usually means the cache namespace was built for a different document sample "
            "or before a document_id alignment fix. "
            f"Example QA document_ids: {qa_alignment['qa_document_ids_sample']}. "
            f"Example cached doc ids: {qa_alignment['cached_document_ids_sample']}."
        )

    label_start = time.perf_counter()
    examples = build_retrieval_examples(aligned_qa_entries, chunks_by_doc)
    if aligned_qa_entries and not examples:
        raise RuntimeError(
            "Matched QA entries to cached documents, but none produced retrieval examples. "
            "This usually means the QA answers/evidence could not be aligned to the cached chunks. "
            f"Example matched document_ids: {qa_alignment['matched_document_ids_sample']}."
        )
    qa_alignment["retrieval_examples_built"] = len(examples)
    logger.info(
        "Cached retrieval labels built examples=%d elapsed=%.1fs",
        len(examples),
        time.perf_counter() - label_start,
    )

    corpus_start = time.perf_counter()
    method_label, corpus, corpus_by_doc_all, _retrieval_rows = build_cached_retrieval_artifacts(
        method=normalized_method,
        raw_chunks=raw_chunks,
        canonicalized_chunks=canonicalized_chunks,
        augmentation_j=augmentation_j,
        max_aliases=int(max_aliases),
        max_desc=int(max_desc),
        active_entity_selector=active_entity_selector,
        lambda_t=float(lambda_t),
        lambda_n=float(lambda_n),
        lambda_h=float(lambda_h),
        normalization_method=str(normalization_method or "none"),
        memory_name_weight=float(memory_name_weight),
        memory_alias_weight=float(memory_alias_weight),
        memory_description_weight=float(memory_description_weight),
    )
    logger.info(
        "Cached retrieval corpus built method=%s chunks=%d elapsed=%.1fs",
        method_label,
        len(corpus.chunk_ids),
        time.perf_counter() - corpus_start,
    )
    corpus_by_doc = corpus_by_doc_all if scope == "per_document" else None

    mbr_details: dict[str, Any] | None = None
    if normalized_method in {"mbr", "mbr_linear", "mbr_pool"}:
        if canonicalized_chunks is None:
            raise ValueError(f"{normalized_method} retrieval requires canonicalized chunks from the cache index")
        document_indices = {
            str(doc_id): build_mbr_document_memory_index(llm_cache_index_json, doc_id=str(doc_id))
            for doc_id in sorted({str(chunk.doc_id) for chunk in canonicalized_chunks})
        }
        if normalized_method == "mbr":
            results, mbr_details, _mbr_method_label = run_mbr_retrieval_pipeline(
                examples=examples,
                document_indices=document_indices,
                retriever_name=retriever_name,
                retrieve_k=retrieve_k,
                lambda_t=float(lambda_t),
                lambda_n=float(lambda_n),
                lambda_h=float(lambda_h),
                normalization_method=str(normalization_method or "none"),
                memory_name_weight=float(memory_name_weight),
                memory_alias_weight=float(memory_alias_weight),
                memory_description_weight=float(memory_description_weight),
                memory_aggregation=str(memory_aggregation or DEFAULT_MBR_MEMORY_AGGREGATION),
            )
        elif normalized_method == "mbr_linear":
            results, mbr_details, _mbr_method_label = run_mbr_linear_retrieval_pipeline(
                examples=examples,
                document_indices=document_indices,
                retriever_name=retriever_name,
                retrieve_k=retrieve_k,
                lambda_t=float(lambda_t),
                lambda_n=float(lambda_n),
                lambda_h=float(lambda_h),
                normalization_method=str(normalization_method or "none"),
                memory_aggregation=str(memory_aggregation or DEFAULT_MBR_MEMORY_AGGREGATION),
            )
        else:
            results, mbr_details, _mbr_method_label = run_mbr_pool_retrieval_pipeline(
                examples=examples,
                document_indices=document_indices,
                retriever_name=retriever_name,
                retrieve_k=retrieve_k,
                lambda_t=float(lambda_t),
                lambda_n=float(lambda_n),
                lambda_h=float(lambda_h),
                normalization_method=str(normalization_method or "none"),
                memory_aggregation=str(memory_aggregation or DEFAULT_MBR_MEMORY_AGGREGATION),
            )
        if _mbr_method_label != method_label:
            raise RuntimeError(
                f"{normalized_method} method label mismatch during one run: artifacts={method_label!r}, scoring={_mbr_method_label!r}"
            )
    else:
        results = run_retrieval_pipeline(
            examples=examples,
            corpus=corpus,
            corpus_by_doc=corpus_by_doc,
            retriever_name=retriever_name,
            retrieve_k=retrieve_k,
            retrieval_scope=scope,
        )


    report_metadata = dict(qa_alignment)
    report_metadata.update(retrieval_provenance)
    report = evaluate_retrieval_results(results, eval_k=eval_k, retrieve_k=retrieve_k, metadata=report_metadata)

    report_root = resolve_cached_report_root(llm_cache_index_json, output_dir=output_dir)
    report_dir = report_root / "retrieval_reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    retriever_label = _safe_output_component(str(retriever_name), default="retriever")
    method_component = _safe_output_component(method_label, default="method")
    scope_component = _safe_output_component(scope, default="scope")
    results_path = report_dir / f"retrieval_results__{retriever_label}__{method_component}__{scope_component}.json"
    eval_path = report_dir / f"eval_report__{retriever_label}__{method_component}__{scope_component}.json"

    write_retrieval_results_json(
        results_path,
        results,
        git_provenance_metadata=retrieval_provenance,
    )
    write_eval_report_json(eval_path, report)

    diagnostics_path: Path | None = None
    if normalized_method in {"mbr", "mbr_linear", "mbr_pool"} and mbr_details is not None:
        diagnostics_dir = report_dir / "mbr_diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_path = diagnostics_dir / f"mbr_details__{retriever_label}__{method_component}__{scope_component}.jsonl"
        diagnostic_rows = build_mbr_diagnostic_rows(examples=examples, detailed_results=mbr_details)
        with diagnostics_path.open("w", encoding="utf-8") as handle:
            for row in diagnostic_rows:
                diagnostic_row = dict(row)
                for field in GIT_PROVENANCE_FIELDS:
                    diagnostic_row[field] = retrieval_provenance.get(field)
                handle.write(json.dumps(diagnostic_row, ensure_ascii=False) + "\n")

    run_row = {
        **retrieval_provenance,
        "retriever": str(retriever_name),
        "method": method_label,
        "scope": scope,
        "retrieve_k": int(retrieve_k),
        "eval_k": eval_k,
        "metric": "HR",
        "hit_rate_at_k": report.hit_rate_at_k,
        "results_json": str(results_path),
        "eval_json": str(eval_path),
    }
    if normalized_method in {"mbr", "mbr_linear", "mbr_pool"}:
        run_row["lambda_t"] = float(lambda_t)
        run_row["lambda_n"] = float(lambda_n)
        run_row["lambda_h"] = float(lambda_h)
        run_row["normalization_method"] = str(normalization_method or "none")
        run_row["memory_aggregation"] = str(memory_aggregation or DEFAULT_MBR_MEMORY_AGGREGATION)
        if normalized_method == "mbr":
            run_row["memory_name_weight"] = float(memory_name_weight)
            run_row["memory_alias_weight"] = float(memory_alias_weight)
            run_row["memory_description_weight"] = float(memory_description_weight)
        run_row["mbr_diagnostics_jsonl"] = str(diagnostics_path) if diagnostics_path is not None else None

    report_index_path = merge_retrieval_report_index(
        report_dir,
        [run_row],
        metadata={
            **retrieval_provenance,
            "retrieval_method": method_label,
            "retrieval_scope": scope,
            "retrieval_runner": "saadi.pipelines.retrieval_cached_pipeline",
            "active_entity_selector": str(active_entity_selector),
            "lambda_t": (float(lambda_t) if normalized_method in {"mbr", "mbr_linear", "mbr_pool"} else None),
            "lambda_n": (float(lambda_n) if normalized_method in {"mbr", "mbr_linear", "mbr_pool"} else None),
            "lambda_h": (float(lambda_h) if normalized_method in {"mbr", "mbr_linear", "mbr_pool"} else None),
            "normalization_method": (str(normalization_method or "none") if normalized_method in {"mbr", "mbr_linear", "mbr_pool"} else None),
            "memory_aggregation": (
                str(memory_aggregation or DEFAULT_MBR_MEMORY_AGGREGATION)
                if normalized_method in {"mbr", "mbr_linear", "mbr_pool"}
                else None
            ),
            "memory_name_weight": (float(memory_name_weight) if normalized_method == "mbr" else None),
            "memory_alias_weight": (float(memory_alias_weight) if normalized_method == "mbr" else None),
            "memory_description_weight": (float(memory_description_weight) if normalized_method == "mbr" else None),
            "cached_documents": int(qa_alignment["cached_documents"]),
            "qa_entries_loaded": int(qa_alignment["qa_entries_loaded"]),
            "qa_entries_matched_cached_docs": int(qa_alignment["qa_entries_matched_cached_docs"]),
            "qa_entries_unmatched_cached_docs": int(qa_alignment["qa_entries_unmatched_cached_docs"]),
            "retrieval_examples_built": int(qa_alignment["retrieval_examples_built"]),
        },
    )

    return {
        "results": results,
        "report": report,
        "corpus": corpus,
        "examples": examples,
        "qa_alignment": qa_alignment,
        "artifacts": {
            "report_dir": report_dir,
            "retrieval_results_json": results_path,
            "eval_report_json": eval_path,
            "report_index_json": report_index_path,
            "mbr_diagnostics_jsonl": diagnostics_path,
        },
        "method_label": method_label,
        "active_entity_selector": str(active_entity_selector),
        "mbr": mbr_details,
    }
