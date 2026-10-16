"""Builds the Saadi document index and entity memory."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..context_strategies import strategy_cache_label
from ..entities import (
    active_entity_config_label,
    normalize_active_entity_min_relevance,
    resolve_active_entity_limits,
)
from ..entities.active_selection import active_entity_selector_label, normalize_active_entity_selector
from ..io.artifact_paths import default_artifact_paths
from ..io.manifests import build_manifest, git_provenance, write_manifest
from ..llm.base import GenerateParams
from ..llm.base import describe_llm_provenance
from ..llm.factory import build_backend_from_config
from .ingest_pipeline import ingest_documents

logger = logging.getLogger(__name__)


def _infer_dataset_label(config: dict[str, Any]) -> str:
    ds = config.get("dataset", {}) or {}
    if isinstance(ds, dict):
        for key in ("name", "dataset", "id"):
            value = ds.get(key)
            if value:
                return str(value)
        hf_dataset = ds.get("hf_dataset")
        if hf_dataset:
            parts = [p for p in str(hf_dataset).split("/") if p]
            if parts:
                return parts[-1]
    value = config.get("dataset_name")
    return str(value) if value else "unknown_dataset"


def _infer_llm_label(config: dict[str, Any]) -> str:
    model_cfg = config.get("model", {}) or {}
    if isinstance(model_cfg, dict):
        for key in ("name", "alias", "llm"):
            value = model_cfg.get(key)
            if value:
                return str(value)
        model_id = model_cfg.get("model_id")
        if model_id:
            parts = [p for p in str(model_id).split("/") if p]
            if len(parts) >= 2:
                return parts[0]
            if parts:
                return parts[-1]
        backend = model_cfg.get("backend")
        if backend:
            return str(backend)
    return "unknown_llm"


def _configured_active_entity_selector(config: dict[str, Any]) -> str | None:
    ingest_cfg = config.get("ingest", {}) or {}
    if not isinstance(ingest_cfg, dict):
        return None
    raw_kwargs = ingest_cfg.get("strategy_kwargs", {}) or {}
    if not isinstance(raw_kwargs, dict):
        return None
    value = raw_kwargs.get("active_entity_selector")
    if value is None:
        value = raw_kwargs.get("active_object_selector")
    if value is None:
        return None
    return normalize_active_entity_selector(str(value))


def _configured_active_entity_limits(config: dict[str, Any]) -> tuple[int, int, int]:
    ingest_cfg = config.get("ingest", {}) or {}
    if not isinstance(ingest_cfg, dict):
        return resolve_active_entity_limits(None)
    raw_kwargs = ingest_cfg.get("strategy_kwargs", {}) or {}
    return resolve_active_entity_limits(raw_kwargs if isinstance(raw_kwargs, dict) else None)


def _configured_active_entity_min_relevance(config: dict[str, Any]) -> float:
    ingest_cfg = config.get("ingest", {}) or {}
    raw_kwargs = ingest_cfg.get("strategy_kwargs", {}) if isinstance(ingest_cfg, dict) else {}
    value = raw_kwargs.get("active_entity_min_relevance") if isinstance(raw_kwargs, dict) else None
    if value is None and isinstance(raw_kwargs, dict):
        value = raw_kwargs.get("active_object_min_relevance")
    return normalize_active_entity_min_relevance(value)


def _resolve_active_entity_selector(config: dict[str, Any]) -> str:
    configured = _configured_active_entity_selector(config)
    return configured if configured is not None else normalize_active_entity_selector("bm25")


def _infer_active_entity_selector_cache_label(config: dict[str, Any]) -> str | None:
    return active_entity_selector_label(_resolve_active_entity_selector(config))


def _infer_active_entity_config_cache_label(config: dict[str, Any]) -> str:
    max_entities, max_aliases, max_desc = _configured_active_entity_limits(config)
    return active_entity_config_label(max_entities, max_aliases, max_desc)


def _normalize_retrieval_scope(value: Any) -> str:
    raw = str(value or "global").strip().lower()
    alias = {
        "default": "global",
        "all_docs": "global",
        "all_documents": "global",
        "all_scopes": "both",
        "multi_scope": "both",
        "benchmark": "per_document",
        "gold_document": "per_document",
        "doc": "per_document",
    }
    normalized = alias.get(raw, raw)
    if normalized not in {"global", "per_document", "both"}:
        raise ValueError(
            "Invalid retrieval.scope. Expected one of: global, per_document, both "
            f"(got: {value!r})."
        )
    return normalized


def _normalize_optional_positive_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        out = int(value)
    except Exception as exc:
        raise ValueError(f"{field_name} must be an integer or null (got: {value!r})") from exc
    if out <= 0:
        raise ValueError(f"{field_name} must be > 0 when provided (got: {out})")
    return out


def _build_generate_params(model_cfg: dict[str, Any]) -> GenerateParams:
    raw = model_cfg.get("generate", {}) or {}
    if not isinstance(raw, dict):
        raw = {}
    max_tokens = int(raw.get("max_tokens", 1024))
    temperature = float(raw.get("temperature", 0.0))
    stop_present = "stop" in raw
    stop_raw = raw.get("stop")
    if not stop_present:
        stop = ["[END_JSON]", "<|im_end|>", "<|endoftext|>"]
    elif stop_raw is None:

        stop = []
    elif isinstance(stop_raw, list):
        stop = [str(x) for x in stop_raw if str(x).strip()]
    else:
        txt = str(stop_raw).strip()
        stop = [txt] if txt else []
    return GenerateParams(
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop,
    )


def _apply_sample_limits(
    *,
    documents_by_id: dict[str, str],
    qa_entries: list[dict[str, Any]],
    max_documents: int | None,
    max_qa_entries: int | None,
) -> tuple[dict[str, str], list[dict[str, Any]], dict[str, int]]:
    doc_items_all = [(str(doc_id), text) for doc_id, text in documents_by_id.items()]
    original_doc_count = len(doc_items_all)
    original_qa_count = len(qa_entries)

    doc_lookup = {doc_id: text for doc_id, text in doc_items_all}
    doc_items = list(doc_items_all)
    if max_documents is not None:


        ordered_qa_doc_ids: list[str] = []
        seen_qa_doc_ids: set[str] = set()
        for row in qa_entries:
            doc_id = str(row.get("document_id", ""))
            if not doc_id or doc_id in seen_qa_doc_ids or doc_id not in doc_lookup:
                continue
            seen_qa_doc_ids.add(doc_id)
            ordered_qa_doc_ids.append(doc_id)

        selected_doc_ids_ordered: list[str] = []
        selected_set: set[str] = set()
        for doc_id in ordered_qa_doc_ids:
            selected_doc_ids_ordered.append(doc_id)
            selected_set.add(doc_id)
            if len(selected_doc_ids_ordered) >= max_documents:
                break
        if len(selected_doc_ids_ordered) < max_documents:
            for doc_id, _ in doc_items_all:
                if doc_id in selected_set:
                    continue
                selected_doc_ids_ordered.append(doc_id)
                selected_set.add(doc_id)
                if len(selected_doc_ids_ordered) >= max_documents:
                    break

        doc_items = [(doc_id, doc_lookup[doc_id]) for doc_id in selected_doc_ids_ordered]

    doc_subset = {doc_id: text for doc_id, text in doc_items}
    selected_doc_ids = set(doc_subset.keys())

    if max_documents is not None:
        qa_subset = [row for row in qa_entries if str(row.get("document_id", "")) in selected_doc_ids]
    else:
        qa_subset = list(qa_entries)

    if max_qa_entries is not None:
        qa_subset = qa_subset[:max_qa_entries]

    return (
        doc_subset,
        qa_subset,
        {
            "original_documents": original_doc_count,
            "used_documents": len(doc_subset),
            "original_qa_entries": original_qa_count,
            "used_qa_entries": len(qa_subset),
        },
    )


def run_experiment(
    config: dict[str, Any],
    *,
    documents_by_id: dict[str, str],
    qa_entries: list[dict[str, Any]],
    backend=None,
):
    repository_root = str(Path(__file__).resolve().parents[3])
    run_provenance = git_provenance(cwd=repository_root)
    loaded_documents_by_id = {str(doc_id): text for doc_id, text in documents_by_id.items()}
    loaded_qa_entries = list(qa_entries)
    sample_cfg = config.get("sample", {}) or {}
    max_documents = _normalize_optional_positive_int(sample_cfg.get("max_documents"), "sample.max_documents")
    max_qa_entries = _normalize_optional_positive_int(sample_cfg.get("max_qa_entries"), "sample.max_qa_entries")
    documents_by_id, qa_entries, sample_stats = _apply_sample_limits(
        documents_by_id=documents_by_id,
        qa_entries=qa_entries,
        max_documents=max_documents,
        max_qa_entries=max_qa_entries,
    )
    logger.info(
        "Sampling applied docs=%d/%d qa_entries=%d/%d",
        sample_stats["used_documents"],
        sample_stats["original_documents"],
        sample_stats["used_qa_entries"],
        sample_stats["original_qa_entries"],
    )
    if sample_stats["original_qa_entries"] > 0 and sample_stats["used_qa_entries"] == 0:
        qa_doc_ids = sorted(
            {
                str(row.get("document_id", "")).strip()
                for row in loaded_qa_entries
                if str(row.get("document_id", "")).strip()
            }
        )
        doc_ids = sorted(str(doc_id).strip() for doc_id in loaded_documents_by_id.keys() if str(doc_id).strip())
        raise RuntimeError(
            "Loaded QA entries but none matched the selected document set after sampling. "
            "This usually means the QA loader and document loader are using different document_id values. "
            f"Example QA document_ids: {qa_doc_ids[:5]}. "
            f"Example document ids: {doc_ids[:5]}."
        )

    output_dir = config.get("output_dir", "outputs")
    paths = default_artifact_paths(output_dir)

    ingest_cfg = config.get("ingest", {}) or {}
    retrieval_cfg = config.get("retrieval", {}) or {}
    retrieval_scope = _normalize_retrieval_scope(retrieval_cfg.get("scope", "global"))
    if ingest_cfg.get("enabled") is False:
        raise ValueError("Saadi construction requires ingest.enabled=true")
    write_raw_chunks = bool(ingest_cfg.get("write_raw_chunks", False))
    logger.info(
        "Saadi construction scope=%s write_raw_chunks=%s output_dir=%s",
        retrieval_scope,
        write_raw_chunks,
        output_dir,
    )

    model_cfg = config.get("model", {}) or {}
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    generate = _build_generate_params(model_cfg)
    active_selector = _resolve_active_entity_selector(config)
    active_entity_min_relevance = _configured_active_entity_min_relevance(config)
    active_entity_limits = _configured_active_entity_limits(config)
    active_entity_config = _infer_active_entity_config_cache_label(config)

    if backend is None:
        backend = build_backend_from_config(model_cfg)
        logger.info("Initialized LLM backend=%s", model_cfg.get("backend", "vllm"))

    logger.info("Starting ingest pipeline")
    strategy_name = str(ingest_cfg.get("strategy", "hierarchical"))
    strategy_kwargs = ingest_cfg.get("strategy_kwargs", {}) or {}
    cache_method_label = strategy_cache_label(strategy_name, strategy_kwargs)
    raw_chunks, canonicalized_chunks, ingest_artifacts = ingest_documents(
        documents_by_id=documents_by_id,
        backend=backend,
        strategy_name=strategy_name,
        strategy_kwargs=strategy_kwargs,
        cache_namespace={
            "dataset": _infer_dataset_label(config),
            "method": cache_method_label,
            "llm": _infer_llm_label(config),
            "active_entity_config": active_entity_config,
            "active_entity_selector": _infer_active_entity_selector_cache_label(config),
        },
        use_cache=bool(ingest_cfg.get("use_cache", True)),
        cache_save_every_steps=int(ingest_cfg.get("cache_save_every_steps", 10)),
        cache_save_each_document=bool(ingest_cfg.get("cache_save_each_document", False)),
        chunk_size=int(ingest_cfg.get("chunk_size", 500)),
        chunk_overlap=int(ingest_cfg.get("chunk_overlap", 0)),
        output_dir=output_dir,
        generate=generate,
        write_raw_chunks=write_raw_chunks,
        schedule=str(ingest_cfg.get("schedule", "document")),
        wavefront_batch_size=_normalize_optional_positive_int(
            ingest_cfg.get("wavefront_batch_size"),
            "ingest.wavefront_batch_size",
        ),
    )
    if ingest_artifacts is not None:
        paths = ingest_artifacts
    logger.info(
        "Ingest completed raw_chunks=%d canonicalized_chunks=%d",
        len(raw_chunks),
        len(canonicalized_chunks),
    )
    rejected_entity_counts_by_document: dict[str, int] = {}
    for record in canonicalized_chunks:
        doc_id = str(record.doc_id)
        rejected_entity_counts_by_document[doc_id] = (
            rejected_entity_counts_by_document.get(doc_id, 0)
            + len(record.rejected_entities)
        )
    rejected_entity_count = sum(rejected_entity_counts_by_document.values())

    paths.run_manifest_json = Path(paths.llm_cache_dir) / "run_manifest.json"
    artifacts = {
        "llm_cache_dir": str(paths.llm_cache_dir),
        "llm_cache_index": str(paths.llm_cache_index_json),
    }
    if write_raw_chunks and paths.raw_chunks_jsonl.exists():
        artifacts["raw_chunks"] = str(paths.raw_chunks_jsonl)
    manifest = build_manifest(
        config=config,
        artifacts=artifacts,
        metadata={
            "runner": "saadi.pipelines.experiment_runner",
            "retrieval_enabled": False,
            "ingest_enabled": True,
            "sample_stats": sample_stats,
            "active_entity_config": active_entity_config,
            "active_entity_selector": active_selector,
            "active_entity_min_relevance": active_entity_min_relevance,
            "rejected_entity_count": rejected_entity_count,
            "rejected_entity_counts_by_document": rejected_entity_counts_by_document,
            "active_entity_limits": {
                "max_entities": int(active_entity_limits[0]),
                "max_aliases": int(active_entity_limits[1]),
                "max_desc": int(active_entity_limits[2]),
            },
        },
        cwd=str(Path(__file__).resolve().parents[3]),
        backend=backend,
        generate=generate,
        provenance=run_provenance,
    )
    manifest.metadata["reproducibility"] = describe_llm_provenance(
        backend, manifest.metadata.get("reproducibility", {})
    )
    write_manifest(paths.run_manifest_json, manifest)
    logger.info("Ingest-only experiment complete; retrieval was not started")
    return {
        "raw_chunks": raw_chunks,
        "canonicalized_chunks": canonicalized_chunks,
        "stateful_report": None,
        "stateful_reports_by_scope": {},
        "retrieval_mode": "disabled",
        "retrieval_scope": retrieval_scope,
        "active_entity_config": active_entity_config,
        "active_entity_selector": active_selector,
        "active_entity_min_relevance": active_entity_min_relevance,
        "rejected_entity_count": rejected_entity_count,
        "active_entity_limits": {
            "max_entities": int(active_entity_limits[0]),
            "max_aliases": int(active_entity_limits[1]),
            "max_desc": int(active_entity_limits[2]),
        },
        "sample_stats": sample_stats,
        "manifest": manifest,
        "artifacts": paths,
        "examples": [],
    }
