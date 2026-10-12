"""Unified CLI entrypoint for SAADI generation experiments."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from saadi.config.loader import load_config
    from saadi.context_expansion import (
        ContextExpansionBundle,
        apply_context_expansion_overrides,
        build_document_context_expansion,
        build_example_metadata_rows,
        context_expansion_cli_overrides,
        context_expansion_config_from_config,
        expansion_output_name,
        trim_text_tokens,
        write_context_expansion_metadata,
    )
    from saadi.evaluation.rag_answer_metrics import write_generation_evaluation
    from saadi.evaluation.updated_chunks.common import infer_dataset_label, resolve_cache_index_from_experiment_config
    from saadi.indexing.dataset_loaders import load_dataset_from_config, load_qa_entries_from_config
    from saadi.io.hierarchy_cache import load_doc_ids_from_cache_index
    from saadi.pipelines.retrieval_cached_pipeline import run_cached_retrieval_eval
    from saadi.saadi_generation.configs.experiments import experiment_names, get_experiment_spec
    from saadi.saadi_generation.context_builders.base import ContextBuilderConfig, QueryContextRequest, RetrievedContextItem
    from saadi.saadi_generation.context_builders.folded_context import (
        FoldedContextBuilder,
        load_summary_blocks_by_doc,
    )
    from saadi.saadi_generation.context_builders.cross_document_folded import (
        build_cross_document_folded_context,
    )
    from saadi.saadi_generation.generation.generator import GenerationRequest, SaadiAnswerGenerator
    from saadi.saadi_generation.retrieval.mbr_loader import (
        MBRRetrievalLoader,
        build_retrieved_context_items,
        load_raw_chunk_items_by_id,
        resolve_mbr_results_json,
    )
    from saadi.saadi_generation.retrieval.cross_document_mbr import (
        CrossDocumentMBRConfig,
        build_cross_document_memory_indices,
        score_and_rank_cross_document,
    )
    from saadi.saadi_generation.utils.io import read_jsonl, write_json, write_jsonl
    from saadi.saadi_generation.utils.logging import configure_logging
    from saadi.saadi_generation.utils.paths import resolve_saadi_generation_paths
else:
    from ..config.loader import load_config
    from ..context_expansion import (
        ContextExpansionBundle,
        apply_context_expansion_overrides,
        build_document_context_expansion,
        build_example_metadata_rows,
        context_expansion_cli_overrides,
        context_expansion_config_from_config,
        expansion_output_name,
        trim_text_tokens,
        write_context_expansion_metadata,
    )
    from ..evaluation.rag_answer_metrics import write_generation_evaluation
    from ..evaluation.updated_chunks.common import infer_dataset_label, resolve_cache_index_from_experiment_config
    from ..indexing.dataset_loaders import load_dataset_from_config, load_qa_entries_from_config
    from ..io.hierarchy_cache import load_doc_ids_from_cache_index
    from ..pipelines.retrieval_cached_pipeline import run_cached_retrieval_eval
    from .configs.experiments import experiment_names, get_experiment_spec
    from .context_builders.base import ContextBuilderConfig, QueryContextRequest, RetrievedContextItem
    from .context_builders.folded_context import FoldedContextBuilder, load_summary_blocks_by_doc
    from .context_builders.cross_document_folded import build_cross_document_folded_context
    from .generation.generator import GenerationRequest, SaadiAnswerGenerator
    from .retrieval.mbr_loader import (
        MBRRetrievalLoader,
        build_retrieved_context_items,
        load_raw_chunk_items_by_id,
        resolve_mbr_results_json,
    )
    from .retrieval.cross_document_mbr import (
        CrossDocumentMBRConfig,
        build_cross_document_memory_indices,
        score_and_rank_cross_document,
    )
    from .utils.io import read_jsonl, write_json, write_jsonl
    from .utils.logging import configure_logging
    from .utils.paths import resolve_saadi_generation_paths


logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a SAADI generation experiment")
    parser.add_argument("--experiment", required=True, choices=experiment_names())
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--retriever", required=True)
    parser.add_argument("--top_k", type=int, required=True)
    parser.add_argument("--summary_k", type=int, default=0, help="Summary block size for folded-context variants.")
    parser.add_argument("--config", default=None, help="Optional experiment YAML/JSON for model settings and cache inference.")
    parser.add_argument(
        "--dataset_experiment_dir",
        default=None,
        help="Cache namespace directory that owns index.json, retrieval_reports/, and generation_pipeline/.",
    )
    parser.add_argument("--llm_cache_index_json", default=None, help="Optional explicit cache index.json for raw chunks.")
    parser.add_argument(
        "--mbr_results_json",
        default=None,
        help="Optional explicit Step-B default MBR retrieval_results JSON.",
    )
    parser.add_argument(
        "--no_auto_mbr",
        action="store_true",
        help="Do not auto-generate missing default MBRB retrieval outputs before generation.",
    )
    parser.add_argument("--scope", default="per_document", choices=["per_document"])
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--max_context_tokens", type=int, default=None)
    parser.add_argument("--max_output_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--no_eval", action="store_true", help="Skip generation metrics/BERTScore artifacts.")
    parser.add_argument("--bert_score", default=None, choices=["auto", "disabled", "required"])
    parser.add_argument("--bert_score_device", default=None)
    parser.add_argument("--bert_score_batch_size", type=int, default=None)
    parser.add_argument("--expand_context", nargs="?", const="true", default=None, help="Opt into document-level context expansion.")
    parser.add_argument("--target_context_tokens", type=int, default=None)
    parser.add_argument("--tolerance_tokens", type=int, default=None)
    parser.add_argument("--expansion_seed", type=int, default=None)
    parser.add_argument("--expansion_mode", default=None)
    parser.add_argument(
        "--output_name",
        default=None,
        help=(
            "Optional explicit folder name under generation_pipeline/saadi_generation/outputs/. "
            "Defaults to the standard experiment/top-k path, or the expansion-specific path for expanded runs."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate all selected examples and overwrite existing SAADI generation artifacts.",
    )
    parser.add_argument(
        "--no_write",
        action="store_true",
        help="Dry-run path/config resolution only; do not generate or write artifacts.",
    )
    return parser


def _positive(value: int, *, name: str) -> int:
    if int(value) <= 0:
        raise ValueError(f"{name} must be > 0")
    return int(value)


def _load_optional_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    return load_config(Path(path).resolve())


def _resolve_dataset_experiment_dir(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    if args.dataset_experiment_dir:
        return Path(args.dataset_experiment_dir).resolve()
    if args.llm_cache_index_json:
        return Path(args.llm_cache_index_json).resolve().parent
    if config:
        index_path = resolve_cache_index_from_experiment_config(config)
        if index_path:
            return Path(index_path).resolve().parent
    return _discover_dataset_experiment_dir(
        dataset=str(args.dataset),
        retriever=str(args.retriever),
        top_k=int(args.top_k),
        scope=str(args.scope),
    )


def _discover_dataset_experiment_dir(*, dataset: str, retriever: str, top_k: int, scope: str) -> Path:
    search_root = Path("outputs") / f"{dataset}_ablation" / "llm_cache" / dataset
    if not search_root.exists():
        raise ValueError(
            "Could not infer --dataset_experiment_dir. "
            "Pass --dataset_experiment_dir, --llm_cache_index_json, or --config."
        )
    matches: list[Path] = []
    method = MBRRetrievalLoader.default_step_b_method_label()
    for index_path in search_root.rglob("index.json"):
        namespace = index_path.parent
        try:
            resolve_mbr_results_json(
                dataset_experiment_dir=namespace,
                retriever=retriever,
                method=method,
                scope=scope,
                min_retrieve_k=top_k,
            )
        except Exception:
            continue
        matches.append(namespace)
    if not matches:
        raise ValueError(
            "Could not auto-discover a cache namespace with Step-B default MBR outputs. "
            "Pass --dataset_experiment_dir or --mbr_results_json explicitly."
        )
    matches.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0].resolve()


def _resolve_llm_cache_index(args: argparse.Namespace, dataset_experiment_dir: Path, config: dict[str, Any]) -> Path:
    if args.llm_cache_index_json:
        path = Path(args.llm_cache_index_json).resolve()
    elif (dataset_experiment_dir / "index.json").exists():
        path = dataset_experiment_dir / "index.json"
    elif config and resolve_cache_index_from_experiment_config(config):
        path = Path(str(resolve_cache_index_from_experiment_config(config))).resolve()
    else:
        raise ValueError("Could not resolve raw chunk cache index. Pass --llm_cache_index_json or --config.")
    if not path.exists():
        raise FileNotFoundError(f"Raw chunk cache index does not exist: {path}")
    return path


def _generation_cfg(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    raw_generation = config.get("generation", {}) or {}
    if not isinstance(raw_generation, dict):
        raw_generation = {}
    cfg = raw_generation.get("saadi_generation", {}) or {}
    if not isinstance(cfg, dict):
        cfg = {}
    cfg = dict(cfg)
    if args.batch_size is not None:
        cfg["batch_size"] = int(args.batch_size)
    if args.max_context_tokens is not None:
        cfg["max_context_tokens"] = int(args.max_context_tokens)
    if args.max_output_tokens is not None:
        cfg["max_output_tokens"] = int(args.max_output_tokens)
    if args.temperature is not None:
        cfg["temperature"] = float(args.temperature)
    return cfg


def _dataset_split(config: dict[str, Any]) -> str:
    dataset_cfg = config.get("dataset", {}) or {}
    if isinstance(dataset_cfg, dict):
        return str(dataset_cfg.get("split", "test") or "test")
    return "test"


def _evaluation_cfg(config: dict[str, Any]) -> dict[str, Any]:
    generation_cfg = config.get("generation", {}) or {}
    if not isinstance(generation_cfg, dict):
        return {}
    raw = generation_cfg.get("evaluation", {}) or {}
    return raw if isinstance(raw, dict) else {}


def _forced_bert_score_mode() -> str:
    return "required"


def _answers_from_entry(entry: dict[str, Any] | None) -> list[str]:
    raw = (entry or {}).get("answers")
    out: list[str] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                text = str(item.get("text", "") or "").strip()
            else:
                text = str(item or "").strip()
            if text:
                out.append(text)
    else:
        text = str(raw or "").strip()
        if text:
            out.append(text)
    return out


def _qa_id_candidates(entry: dict[str, Any], *, index: int) -> list[str]:
    doc_id = str(entry.get("document_id", "") or "")
    candidates = [
        str(entry.get("id", "") or ""),
        f"q{index}",
        str(index),
        f"{doc_id}::qa::{index}",
    ]
    out: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in out:
            out.append(candidate)
    return out


def _load_qa_entry_lookup(
    config: dict[str, Any],
    *,
    config_path: str | None,
    required: bool,
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    if not config:
        if required:
            raise ValueError("Evaluation requires --config so QA answer references can be loaded.")
        return {}, {}, {"dataset_source": "missing_config", "qa_entries_loaded": 0}
    base_dir = Path(config_path).resolve().parent if config_path else Path.cwd()
    try:
        qa_entries, meta = load_qa_entries_from_config(config.get("dataset", {}), base_dir=base_dir)
    except Exception as exc:
        if required:
            raise RuntimeError("Evaluation requires QA answer references, but loading QA entries failed.") from exc
        return {}, {}, {
            "dataset_source": "qa_load_failed",
            "qa_entries_loaded": 0,
            "error": str(exc),
        }
    by_id: dict[str, dict[str, Any]] = {}
    by_doc_question: dict[tuple[str, str], dict[str, Any]] = {}
    for index, entry in enumerate(qa_entries):
        if not isinstance(entry, dict):
            continue
        for candidate in _qa_id_candidates(entry, index=index):
            by_id.setdefault(candidate, entry)
        doc_id = str(entry.get("document_id", "") or "")
        question = str(entry.get("question", "") or "").strip()
        if doc_id and question:
            by_doc_question.setdefault((doc_id, question), entry)
    return by_id, by_doc_question, {
        **meta,
        "qa_entries_loaded": len(qa_entries),
        "qa_lookup_ids": len(by_id),
    }


def _load_qa_entries_for_retrieval_generation(
    config: dict[str, Any],
    *,
    config_path: str | None,
    llm_cache_index_json: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not config:
        raise ValueError("Automatic MBRB generation requires --config so QA entries can be loaded.")

    dataset_cfg = config.get("dataset", {}) or {}
    if not isinstance(dataset_cfg, dict):
        raise ValueError("Automatic MBRB generation requires a dataset config section.")

    base_dir = Path(config_path).resolve().parent if config_path else Path.cwd()
    allowed_doc_ids: set[str] | None = None
    dataset_name = str(dataset_cfg.get("name", "") or "").strip().lower()
    if dataset_name in {"novelhopqa", "abhaygupta1266/novelhopqa"}:
        cached_doc_ids = load_doc_ids_from_cache_index(llm_cache_index_json)
        if cached_doc_ids:
            allowed_doc_ids = set(cached_doc_ids)

    return load_qa_entries_from_config(dataset_cfg, base_dir=base_dir, allowed_doc_ids=allowed_doc_ids)


def _configured_retrieve_k(config: dict[str, Any], *, top_k: int) -> int:
    retrieval_cfg = config.get("retrieval", {}) or {}
    raw = retrieval_cfg.get("retrieve_k", 10) if isinstance(retrieval_cfg, dict) else 10
    try:
        retrieve_k = int(raw)
    except Exception:
        retrieve_k = 10
    return max(int(top_k), retrieve_k)


def _configured_eval_k(config: dict[str, Any], *, retrieve_k: int) -> int:
    retrieval_cfg = config.get("retrieval", {}) or {}
    raw = retrieval_cfg.get("eval_k", 5) if isinstance(retrieval_cfg, dict) else 5
    eval_k = 5 if raw is None else int(raw)
    if not 0 < eval_k <= retrieve_k:
        raise ValueError(f"eval_k must be positive and at most retrieve_k={retrieve_k}; got {eval_k}")
    return eval_k


def _auto_generate_default_mbrb_results(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    dataset_experiment_dir: Path,
    llm_cache_index_json: Path,
) -> Path:
    weights = MBRRetrievalLoader.default_step_b_weights()
    retrieve_k = _configured_retrieve_k(config, top_k=int(args.top_k))
    eval_k = _configured_eval_k(config, retrieve_k=retrieve_k)
    output_dir = MBRRetrievalLoader.default_step_b_output_dir(
        dataset_experiment_dir,
        retriever=str(args.retriever),
    )
    qa_entries, qa_meta = _load_qa_entries_for_retrieval_generation(
        config,
        config_path=args.config,
        llm_cache_index_json=llm_cache_index_json,
    )
    logger.warning(
        "Default MBRB retrieval results were not found; generating them now. "
        "Using the standard Step-B directory generated by the retrieval scripts: %s",
        output_dir,
    )
    logger.warning(
        "MBRB default config: lambda_t=%s lambda_n=%s lambda_h=%s "
        "memory_name=%s memory_alias=%s memory_description=%s retrieve_k=%s eval_k=%s qa_entries=%s",
        weights["lambda_t"],
        weights["lambda_n"],
        weights["lambda_h"],
        weights["memory_name_weight"],
        weights["memory_alias_weight"],
        weights["memory_description_weight"],
        retrieve_k,
        eval_k,
        qa_meta.get("qa_entries_loaded", len(qa_entries)) if isinstance(qa_meta, dict) else len(qa_entries),
    )
    out = run_cached_retrieval_eval(
        llm_cache_index_json=str(llm_cache_index_json),
        qa_entries=qa_entries,
        method="mbr",
        output_dir=str(output_dir),
        retriever_name=str(args.retriever),
        retrieve_k=retrieve_k,
        eval_k=eval_k,
        retrieval_scope=str(args.scope),
        lambda_t=float(weights["lambda_t"]),
        lambda_n=float(weights["lambda_n"]),
        lambda_h=float(weights["lambda_h"]),
        normalization_method="none",
        memory_name_weight=float(weights["memory_name_weight"]),
        memory_alias_weight=float(weights["memory_alias_weight"]),
        memory_description_weight=float(weights["memory_description_weight"]),
    )
    generated = Path(out["artifacts"]["retrieval_results_json"]).resolve()
    logger.warning("Generated default MBRB retrieval results: %s", generated)
    return generated


def _resolve_mbr_loader(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    dataset_experiment_dir: Path,
    llm_cache_index_json: Path,
) -> MBRRetrievalLoader:
    def standard_default_report_dir() -> Path:
        return (
            MBRRetrievalLoader.default_step_b_output_dir(
                dataset_experiment_dir,
                retriever=str(args.retriever),
            )
            / "retrieval_reports"
        ).resolve()

    def resolved_exact_standard_default(loader: MBRRetrievalLoader) -> bool:
        if loader.selection.results_json is None:
            return False
        return (
            Path(loader.selection.results_json).resolve().parent == standard_default_report_dir()
            and str(loader.selection.method) == MBRRetrievalLoader.default_step_b_method_label()
        )

    try:
        loader = MBRRetrievalLoader.for_step_b_default(
            dataset=str(args.dataset),
            retriever=str(args.retriever),
            top_k=int(args.top_k),
            dataset_experiment_dir=dataset_experiment_dir,
            results_json=args.mbr_results_json,
            scope=str(args.scope),
        )
        if args.mbr_results_json or args.no_auto_mbr or resolved_exact_standard_default(loader):
            return loader
        logger.warning(
            "Resolved a fallback MBR artifact at %s, but the standard default MBRB directory is missing or empty: %s. "
            "Generating the default MBRB outputs before SAADI generation.",
            loader.selection.results_json,
            standard_default_report_dir().parent,
        )
    except FileNotFoundError:
        if args.mbr_results_json or args.no_auto_mbr:
            raise
    generated_path = _auto_generate_default_mbrb_results(
        args=args,
        config=config,
        dataset_experiment_dir=dataset_experiment_dir,
        llm_cache_index_json=llm_cache_index_json,
    )
    return MBRRetrievalLoader.for_step_b_default(
        dataset=str(args.dataset),
        retriever=str(args.retriever),
        top_k=int(args.top_k),
        dataset_experiment_dir=dataset_experiment_dir,
        results_json=generated_path,
        scope=str(args.scope),
    )


def _match_qa_entry(
    *,
    result,
    qa_by_id: dict[str, dict[str, Any]],
    qa_by_doc_question: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    direct = qa_by_id.get(str(result.query_id))
    if direct is not None:
        return direct
    return qa_by_doc_question.get((str(result.doc_id), str(result.question or "").strip()), {})


def _row_example_id(row: dict[str, Any]) -> str:
    return str(row.get("example_id") or row.get("query_id") or "").strip()


def _is_reusable_generation_row(row: dict[str, Any]) -> bool:
    """Return True when an existing row should not call the LLM again."""

    status = str(row.get("status", "ok") or "ok").strip().lower()
    if status in {"failed", "failure", "error"} or status.startswith("failed"):
        return False
    generated = str(row.get("generated_answer") or row.get("raw_output") or "").strip()
    if generated:
        return True
    if status.startswith("skipped") or str(row.get("skip_reason", "") or "").strip():
        return True
    return False


def _rows_by_example_id(
    rows: list[dict[str, Any]],
    *,
    reusable_generation_only: bool = False,
) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        example_id = _row_example_id(row)
        if not example_id:
            continue
        if reusable_generation_only and not _is_reusable_generation_row(row):
            continue
        by_id[example_id] = dict(row)
    return by_id


def _bridge_rows_by_example_id(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_id: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        example_id = _row_example_id(row)
        if not example_id:
            continue
        by_id.setdefault(example_id, []).append(dict(row))
    return by_id


def _generated_bridge_rows_from_debug(row: dict[str, Any]) -> list[dict[str, Any]]:
    raw = row.get("generated_bridges", [])
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, dict)]


def _enrich_existing_generation_row(
    row: dict[str, Any],
    *,
    example_id: str,
    result,
    qa_entry: dict[str, Any],
    args: argparse.Namespace,
    loader: MBRRetrievalLoader,
) -> dict[str, Any]:
    enriched = dict(row)
    answers = _answers_from_entry(qa_entry)
    enriched["example_id"] = str(enriched.get("example_id") or example_id)
    enriched["query_id"] = str(enriched.get("query_id") or example_id)
    enriched.setdefault("doc_id", str(result.doc_id))
    enriched.setdefault("question", str(result.question or ""))
    if answers:
        enriched["answers"] = answers
    if isinstance(qa_entry, dict) and qa_entry.get("choices") and not enriched.get("choices"):
        enriched["choices"] = qa_entry.get("choices", [])
    enriched.setdefault("retriever", str(args.retriever))
    enriched.setdefault("retrieval_method", str(loader.selection.method))
    enriched.setdefault("retrieval_scope", str(args.scope))
    enriched.setdefault("top_k", int(args.top_k))
    enriched.setdefault("summary_k", int(args.summary_k or 0))
    if enriched.get("context_tokens") in (None, "") and enriched.get("context_token_length") not in (None, ""):
        enriched["context_tokens"] = enriched.get("context_token_length")
    if enriched.get("context_token_length") in (None, "") and enriched.get("context_tokens") not in (None, ""):
        enriched["context_token_length"] = enriched.get("context_tokens")
    enriched["resumed_from_existing"] = True
    return enriched


def _enrich_existing_debug_row(
    row: dict[str, Any],
    *,
    example_id: str,
    result,
    qa_entry: dict[str, Any],
    generation_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    enriched = dict(row)
    answers = _answers_from_entry(qa_entry)
    enriched["example_id"] = str(enriched.get("example_id") or example_id)
    enriched["query_id"] = str(enriched.get("query_id") or example_id)
    enriched.setdefault("doc_id", str(result.doc_id))
    enriched.setdefault("question", str(result.question or ""))
    enriched.setdefault("query", str(result.question or ""))
    if answers:
        enriched["answers"] = answers
    if generation_row:
        for key in ("context_token_length", "context_tokens"):
            if enriched.get(key) in (None, "") and generation_row.get(key) not in (None, ""):
                enriched[key] = generation_row.get(key)
    if enriched.get("context_tokens") in (None, "") and enriched.get("context_token_length") not in (None, ""):
        enriched["context_tokens"] = enriched.get("context_token_length")
    if enriched.get("context_token_length") in (None, "") and enriched.get("context_tokens") not in (None, ""):
        enriched["context_token_length"] = enriched.get("context_tokens")
    enriched["resumed_from_existing"] = True
    return enriched


def _fallback_token_count(text: str) -> int:
    return len(str(text or "").split())


def _load_documents_for_context_expansion(
    config: dict[str, Any],
    *,
    config_path: str | None,
    llm_cache_index_json: Path,
    retrieval_results: list[Any],
) -> tuple[dict[str, str], dict[str, Any]]:
    if not config:
        raise ValueError("Context expansion requires --config so source documents can be loaded.")
    base_dir = Path(config_path).resolve().parent if config_path else Path.cwd()
    documents, _qa_entries, meta = load_dataset_from_config(config.get("dataset", {}), base_dir=base_dir)
    cached_doc_ids = set(load_doc_ids_from_cache_index(llm_cache_index_json))
    result_doc_ids = {str(result.doc_id) for result in retrieval_results if str(result.doc_id).strip()}
    allowed_doc_ids = cached_doc_ids or result_doc_ids
    if allowed_doc_ids:
        documents = {doc_id: text for doc_id, text in documents.items() if str(doc_id) in allowed_doc_ids}
    missing = sorted(result_doc_ids.difference(documents.keys()))
    if missing:
        logger.warning(
            "Context expansion source documents are missing %d retrieval doc ids; sample=%s",
            len(missing),
            missing[:10],
        )
    return documents, {
        **meta,
        "context_expansion_documents_loaded": len(documents),
        "context_expansion_allowed_doc_ids": len(allowed_doc_ids),
        "context_expansion_missing_doc_ids_sample": missing[:20],
    }


def _compose_expanded_saadi_context(
    *,
    method_context: str,
    expanded_context: str,
    expansion_meta: dict[str, Any],
    generator: SaadiAnswerGenerator,
) -> tuple[str, dict[str, Any]]:
    header = "[SAADI Generated Context]"
    expanded_header = "[Expanded Document Group]"
    prefix = f"{header}\n{str(method_context or '').strip()}\n\n{expanded_header}\n"
    prefix_tokens = generator.count_tokens(prefix)
    target = int(expansion_meta.get("target_context_tokens") or 0)
    tolerance = int(expansion_meta.get("tolerance_tokens") or 0)
    upper_bound = target + tolerance
    if upper_bound <= 0:
        combined = f"{prefix}{expanded_context}".strip()
        return combined, {
            "saadi_method_context_tokens": generator.count_tokens(method_context),
            "expanded_context_tokens_before_saadi_trim": generator.count_tokens(expanded_context),
            "saadi_expansion_trimmed_to_fit_method_context": False,
        }

    budget_for_expanded = max(0, upper_bound - prefix_tokens)
    trim_side = str(expansion_meta.get("trim_side") or "end")
    if trim_side == "none":
        target_position = int(expansion_meta.get("target_doc_position") or 1)
        group_order = expansion_meta.get("group_doc_order") or []
        if isinstance(group_order, list) and target_position >= len(group_order):
            trim_side = "beginning"
        else:
            trim_side = "end"
    trimmed_expanded, trimmed_expanded_tokens, was_trimmed_for_method = trim_text_tokens(
        expanded_context,
        tokenizer=generator.tokenizer,
        max_tokens=budget_for_expanded,
        trim_side=trim_side,
    )
    combined = f"{prefix}{trimmed_expanded}".strip()
    final_tokens = generator.count_tokens(combined)
    meta = {
        "saadi_method_context_tokens": generator.count_tokens(method_context),
        "expanded_context_tokens_before_saadi_trim": int(expansion_meta.get("final_context_tokens", generator.count_tokens(expanded_context)) or 0),
        "expanded_context_tokens_after_saadi_trim": int(trimmed_expanded_tokens),
        "saadi_expansion_trimmed_to_fit_method_context": bool(was_trimmed_for_method),
        "saadi_expansion_trim_side": trim_side if was_trimmed_for_method else "none",
        "final_context_tokens": int(final_tokens),
        "context_tokens": int(final_tokens),
        "context_token_length": int(final_tokens),
    }
    return combined, meta


def _context_builder_extra(spec) -> dict[str, str]:
    extra: dict[str, str] = {}
    if getattr(spec, "ordering", None):
        extra["ordering"] = str(spec.ordering)
    if getattr(spec, "bridge_mode", None):
        extra["bridge_mode"] = str(spec.bridge_mode)
    return extra


def build_run_config(args: argparse.Namespace, *, config: dict[str, Any], dataset_experiment_dir: Path) -> dict[str, Any]:
    top_k = _positive(args.top_k, name="--top_k")
    summary_k = int(args.summary_k or 0)
    spec = get_experiment_spec(args.experiment)
    expansion_cfg = context_expansion_config_from_config(config, dataset_name=str(args.dataset))
    expansion_cfg.validate()
    output_name = str(args.output_name).strip() if args.output_name else None
    if output_name is None and expansion_cfg.enabled:
        method_name = spec.name
        if spec.name == "folded_context_generated_bridge":
            method_name = f"saadi_llm_guided_bridge_top{top_k}"
        else:
            method_name = f"{spec.name}_top{top_k}"
        output_name = expansion_output_name(method_name, int(expansion_cfg.target_context_tokens or 0))
    paths = resolve_saadi_generation_paths(
        dataset_experiment_dir=dataset_experiment_dir,
        experiment_name=spec.name,
        dataset=args.dataset,
        retriever=args.retriever,
        top_k=top_k,
        summary_k=(summary_k if summary_k > 0 else None),
        output_name=output_name,
    )
    builder_config = ContextBuilderConfig(
        experiment_name=spec.name,
        top_k=top_k,
        summary_k=summary_k,
        include_bridge=spec.include_bridge,
        use_entity_memory=spec.uses_entity_memory,
        extra=_context_builder_extra(spec),
    )
    run_config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner": "saadi.saadi_generation.run_saadi_generation",
        "status": "configured",
        "experiment": spec.to_dict(),
        "dataset": {
            "requested": str(args.dataset),
            "config_label": infer_dataset_label(config) if config else str(args.dataset),
        },
        "retrieval": {
            "dataset": str(args.dataset),
            "retriever": str(args.retriever),
            "top_k": top_k,
            "top_k_meaning": "retrieval_top_k",
            "summary_k": summary_k,
            "method": MBRRetrievalLoader.default_step_b_method_label(),
            "scope": str(args.scope),
            "mbr_results_json": str(Path(args.mbr_results_json).resolve()) if args.mbr_results_json else None,
        },
        "context_builder": {
            "family": spec.context_builder_family,
            "config": {
                "experiment_name": builder_config.experiment_name,
                "top_k": builder_config.top_k,
                "summary_k": builder_config.summary_k,
                "include_bridge": builder_config.include_bridge,
                "use_entity_memory": builder_config.use_entity_memory,
                "extra": builder_config.extra,
            },
        },
        "artifacts": paths.to_dict(),
    }
    if expansion_cfg.enabled:
        run_config["context_expansion"] = expansion_cfg.to_dict()
    return run_config


def run_generation_experiment(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_optional_config(args.config)
    apply_context_expansion_overrides(config, context_expansion_cli_overrides(args))
    dataset_experiment_dir = _resolve_dataset_experiment_dir(args, config)
    llm_cache_index_json = _resolve_llm_cache_index(args, dataset_experiment_dir, config)
    run_config = build_run_config(args, config=config, dataset_experiment_dir=dataset_experiment_dir)
    if args.no_write:
        return {**run_config, "status": "dry_run"}

    paths = run_config["artifacts"]
    loader = _resolve_mbr_loader(
        args=args,
        config=config,
        dataset_experiment_dir=dataset_experiment_dir,
        llm_cache_index_json=llm_cache_index_json,
    )
    retrieval_results = loader.load_results()
    if args.max_examples is not None:
        retrieval_results = retrieval_results[: int(args.max_examples)]
    chunk_items_by_id = load_raw_chunk_items_by_id(llm_cache_index_json)
    qa_by_id, qa_by_doc_question, qa_meta = _load_qa_entry_lookup(
        config,
        config_path=args.config,
        required=not args.no_eval,
    )

    spec = get_experiment_spec(args.experiment)
    summaries_by_doc = {}
    if args.experiment in {
        "folded_context_block",
        "folded_context_nested",
        "folded_context_local_bridge",
        "folded_context_generated_bridge",
    }:
        summary_k = _positive(int(args.summary_k), name="--summary_k")
        summaries_by_doc = load_summary_blocks_by_doc(llm_cache_index_json, summary_k=summary_k)
        builder = FoldedContextBuilder(
            ContextBuilderConfig(
                experiment_name=spec.name,
                top_k=int(args.top_k),
                summary_k=summary_k,
                include_bridge=spec.include_bridge,
                use_entity_memory=False,
                extra=_context_builder_extra(spec),
            ),
            summaries_by_doc=summaries_by_doc,
        )
        context_source = f"mbr_{args.experiment}"
    else:
        raise NotImplementedError(
            f"{args.experiment} is registered but not implemented yet. "
        )
    uses_generated_bridges = str(args.experiment) == "folded_context_generated_bridge"
    generation_cfg = _generation_cfg(config, args)
    generation_cfg.setdefault("prompt_version", f"saadi_{args.experiment}_v1")
    model_cfg = config.get("model", {}) if isinstance(config.get("model", {}), dict) else {}
    batch_size = int(generation_cfg.get("batch_size", 1) or 1)
    if batch_size <= 0:
        raise ValueError("SAADI generation batch_size must be > 0")

    generator: SaadiAnswerGenerator | None = None

    def ensure_generator() -> SaadiAnswerGenerator:
        nonlocal generator, batch_size
        if generator is None:
            generator = SaadiAnswerGenerator.from_model_config(model_cfg, generation_cfg=generation_cfg)
            batch_size = int(generator.config.batch_size)
            logger.info("Initialized SAADI generation backend for %s", args.experiment)
        return generator

    expansion_cfg = context_expansion_config_from_config(config, dataset_name=str(args.dataset))
    expansion_cfg.validate()
    expansion_bundle: ContextExpansionBundle | None = None
    expansion_artifact_paths: dict[str, str] = {}
    example_expansion_metadata_by_id: dict[str, dict[str, Any]] = {}
    expansion_doc_meta: dict[str, Any] = {}
    if expansion_cfg.enabled:
        expansion_documents, expansion_doc_meta = _load_documents_for_context_expansion(
            config,
            config_path=args.config,
            llm_cache_index_json=llm_cache_index_json,
            retrieval_results=retrieval_results,
        )
        active_generator = ensure_generator()
        expansion_bundle = build_document_context_expansion(
            expansion_documents,
            expansion_cfg,
            tokenizer=active_generator.tokenizer,
        )
        selected_qa_entries = [
            {
                "id": str(result.query_id),
                "document_id": str(result.doc_id),
                "question": str(result.question or ""),
            }
            for result in retrieval_results
        ]
        example_metadata_rows = build_example_metadata_rows(
            selected_qa_entries,
            expansion_bundle,
            query_id_fn=lambda entry, _index: str(entry.get("id", "")),
        )
        example_expansion_metadata_by_id = {
            str(row.get("example_id")): dict(row) for row in example_metadata_rows if str(row.get("example_id", "")).strip()
        }
        expansion_artifact_paths = write_context_expansion_metadata(
            paths["experiment_output_dir"],
            expansion_bundle,
            example_metadata_rows,
        )

    existing_generation_rows = [] if args.force else read_jsonl(paths["generations_jsonl"])
    existing_debug_rows = [] if args.force else read_jsonl(paths["context_debug_jsonl"])
    existing_bridge_rows = [] if args.force or not uses_generated_bridges else read_jsonl(paths["generated_bridges_jsonl"])
    existing_generation_by_id = _rows_by_example_id(existing_generation_rows, reusable_generation_only=True)
    existing_debug_by_id = _rows_by_example_id(existing_debug_rows)
    existing_bridge_by_id = _bridge_rows_by_example_id(existing_bridge_rows)
    if args.force:
        logger.info("Force mode enabled; existing SAADI generation artifacts will be replaced.")
    elif existing_generation_rows or existing_debug_rows:
        logger.info(
            "Resume mode loaded %d existing generation rows (%d reusable), %d debug rows, and %d generated bridge rows.",
            len(existing_generation_rows),
            len(existing_generation_by_id),
            len(existing_debug_rows),
            len(existing_bridge_rows),
        )

    generation_by_id: dict[str, dict[str, Any]] = {}
    debug_by_id: dict[str, dict[str, Any]] = {}
    bridge_rows_by_id: dict[str, list[dict[str, Any]]] = {}
    selected_example_ids: list[str] = []
    pending: list[GenerationRequest] = []
    pending_debug: list[dict[str, Any]] = []
    generated_count = 0
    reused_count = 0
    rebuilt_debug_count = 0
    checkpoint_count = 0
    started_at = datetime.now(timezone.utc).isoformat()

    def ordered_rows(rows_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        return [rows_by_id[example_id] for example_id in selected_example_ids if example_id in rows_by_id]

    def ordered_bridge_rows() -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for example_id in selected_example_ids:
            rows.extend(bridge_rows_by_id.get(example_id, []))
        return rows

    def checkpoint_outputs() -> None:
        nonlocal checkpoint_count
        write_jsonl(paths["generations_jsonl"], ordered_rows(generation_by_id))
        write_jsonl(paths["context_debug_jsonl"], ordered_rows(debug_by_id))
        if uses_generated_bridges:
            write_jsonl(paths["generated_bridges_jsonl"], ordered_bridge_rows())
        checkpoint_count += 1

    def flush_pending() -> None:
        nonlocal pending, pending_debug, generated_count
        if not pending:
            return
        rows = ensure_generator().generate_batch(pending)
        if len(rows) != len(pending_debug):
            raise RuntimeError(f"Generated {len(rows)} rows but had {len(pending_debug)} debug rows.")
        for row, debug_row in zip(rows, pending_debug):
            row["resumed_from_existing"] = False
            example_id = _row_example_id(row)
            if not example_id:
                raise RuntimeError("Generated row is missing example_id/query_id.")
            generation_by_id[example_id] = row
            debug_by_id[example_id] = debug_row
        generated_count += len(rows)
        logger.info(
            "SAADI generation progress generated_this_run=%d reused_existing=%d selected=%d",
            generated_count,
            reused_count,
            len(selected_example_ids),
        )
        checkpoint_outputs()
        pending = []
        pending_debug = []

    def build_context(
        *,
        result,
        retrieved_items,
        qa_entry: dict[str, Any],
    ):
        if uses_generated_bridges and isinstance(builder, FoldedContextBuilder):
            builder.bridge_generator = ensure_generator()
        return builder.build(
            QueryContextRequest(
                retrieval_result=result,
                retrieved_items=retrieved_items,
                question=str(result.question or ""),
                qa_entry=qa_entry,
            )
        )

    for result in retrieval_results:
        example_id = str(result.query_id)
        selected_example_ids.append(example_id)
        qa_entry = _match_qa_entry(
            result=result,
            qa_by_id=qa_by_id,
            qa_by_doc_question=qa_by_doc_question,
        )
        existing_generation = existing_generation_by_id.get(example_id)
        if existing_generation is not None:
            generation_row = _enrich_existing_generation_row(
                existing_generation,
                example_id=example_id,
                result=result,
                qa_entry=qa_entry,
                args=args,
                loader=loader,
            )
            generation_by_id[example_id] = generation_row
            existing_debug = existing_debug_by_id.get(example_id)
            if existing_debug is not None:
                debug_by_id[example_id] = _enrich_existing_debug_row(
                    existing_debug,
                    example_id=example_id,
                    result=result,
                    qa_entry=qa_entry,
                    generation_row=generation_row,
                )
            else:
                retrieved_items = build_retrieved_context_items(
                    result=result,
                    chunk_items_by_id=chunk_items_by_id,
                    top_k=int(args.top_k),
                )
                built = build_context(result=result, retrieved_items=retrieved_items, qa_entry=qa_entry)
                context_token_length = generation_row.get("context_token_length", generation_row.get("context_tokens"))
                if context_token_length in (None, ""):
                    context_token_length = _fallback_token_count(built.context_text)
                debug_by_id[example_id] = {
                    "example_id": example_id,
                    "query_id": example_id,
                    "doc_id": str(result.doc_id),
                    "question": str(result.question or ""),
                    "query": str(result.question or ""),
                    "answers": _answers_from_entry(qa_entry),
                    "retrieved_chunk_ids": [item.chunk_id for item in retrieved_items],
                    "retrieved_scores": [item.score for item in retrieved_items],
                    "final_context": built.context_text,
                    "context_token_length": context_token_length,
                    "context_tokens": context_token_length,
                    "resumed_from_existing": True,
                    "rebuilt_debug_from_retrieval": True,
                    **built.debug_payload,
                }
                if uses_generated_bridges:
                    bridge_rows_by_id[example_id] = _generated_bridge_rows_from_debug(debug_by_id[example_id])
                rebuilt_debug_count += 1
            reused_count += 1
            if uses_generated_bridges and example_id not in bridge_rows_by_id:
                bridge_rows = existing_bridge_by_id.get(example_id)
                if bridge_rows:
                    bridge_rows_by_id[example_id] = bridge_rows
                elif example_id in debug_by_id:
                    bridge_rows_by_id[example_id] = _generated_bridge_rows_from_debug(debug_by_id[example_id])
            continue

        answers = _answers_from_entry(qa_entry)
        active_generator = ensure_generator()
        expansion_meta = dict(example_expansion_metadata_by_id.get(example_id, {}) or {})


        if expansion_bundle is not None and expansion_meta:
            expanded_doc_context = expansion_bundle.context_for_doc(str(result.doc_id))
            if expanded_doc_context is not None:
                group_doc_ids = list(expanded_doc_context.group_doc_order)
                target_tokens = int(expanded_doc_context.target_context_tokens or 0)
                tolerance = int(expanded_doc_context.tolerance_tokens or 0)

                cross_doc_config = MBRRetrievalLoader.default_cross_document_config(
                    retriever_name=str(args.retriever),
                    top_k=int(args.top_k),
                    target_token_budget=target_tokens if target_tokens > 0 else None,
                    tolerance_tokens=tolerance,
                )
                cross_doc_indices, cross_doc_build_meta = build_cross_document_memory_indices(
                    llm_cache_index_json,
                    group_doc_ids,
                    query_doc_id=str(result.doc_id),
                    tokenizer=active_generator.tokenizer,
                    target_token_budget=cross_doc_config.target_token_budget,
                    tolerance_tokens=cross_doc_config.tolerance_tokens,
                )
                cross_doc_result = score_and_rank_cross_document(
                    cross_doc_indices,
                    query=str(result.question or ""),
                    config=cross_doc_config,
                )
                cross_doc_result.truncated_doc_ids = cross_doc_build_meta.get("truncated_doc_ids", [])


                cross_retrieved_items: list = []
                for rank, ranked in enumerate(cross_doc_result.ranked_chunks, start=1):
                    source_doc_id = cross_doc_result.chunk_doc_ids.get(ranked.chunk_id, "")
                    base = chunk_items_by_id.get(str(ranked.chunk_id))
                    raw_text = base.raw_text if base is not None else ""
                    metadata = dict(base.metadata) if base is not None else {}
                    metadata["doc_id"] = source_doc_id
                    if "chunk_index" not in metadata:
                        metadata["chunk_index"] = ranked.chunk_index
                    cross_retrieved_items.append(RetrievedContextItem(
                        chunk_id=ranked.chunk_id,
                        rank=rank,
                        score=ranked.final_score,
                        raw_text=raw_text,
                        metadata=metadata,
                    ))


                cross_built = build_cross_document_folded_context(
                    retrieved_items=cross_retrieved_items,
                    summaries_by_doc=summaries_by_doc,
                    chunk_doc_ids=cross_doc_result.chunk_doc_ids,
                    summary_k=int(args.summary_k or 0),
                    use_bridge=uses_generated_bridges,
                    bridge_mode="generated" if uses_generated_bridges else "prefix",
                    query=str(result.question or ""),
                    example_id=example_id,
                    bridge_generator=active_generator if uses_generated_bridges else None,
                )

                context_text = cross_built.context_text
                retrieved_items = cross_retrieved_items
                retrieved_scores = [item.score for item in cross_retrieved_items]
                context_token_length = active_generator.count_tokens(context_text)
                expansion_meta.update({
                    "cross_document_retrieval": True,
                    "cross_document_build_meta": cross_doc_build_meta,
                    "cross_document_debug": cross_doc_result.debug_metadata,
                    "saadi_method_context_tokens": context_token_length,
                    "context_tokens": context_token_length,
                    "context_token_length": context_token_length,
                })

                debug_row = {
                    "example_id": example_id,
                    "query_id": example_id,
                    "doc_id": str(result.doc_id),
                    "question": str(result.question or ""),
                    "query": str(result.question or ""),
                    "answers": answers,
                    "retrieved_chunk_ids": [item.chunk_id for item in cross_retrieved_items],
                    "retrieved_scores": retrieved_scores,
                    "final_context": context_text,
                    "context_token_length": context_token_length,
                    "context_tokens": context_token_length,
                    "resumed_from_existing": False,
                    **expansion_meta,
                    **cross_built.debug_payload,
                }
                if uses_generated_bridges:
                    bridge_rows_by_id[example_id] = _generated_bridge_rows_from_debug(debug_row)
                pending.append(
                    GenerationRequest(
                        example_id=example_id,
                        question=str(result.question or ""),
                        qa_entry=qa_entry,
                        context=context_text,
                        metadata={
                            "doc_id": str(result.doc_id),
                            "answers": answers,
                            "choices": qa_entry.get("choices", []) if isinstance(qa_entry, dict) else [],
                            "retriever": str(args.retriever),
                            "retrieval_method": "cross_document_mbr",
                            "retrieval_scope": "cross_document",
                            "top_k": int(args.top_k),
                            "summary_k": int(args.summary_k or 0),
                            "retrieved_chunk_ids": [item.chunk_id for item in cross_retrieved_items],
                            "retrieved_scores": retrieved_scores,
                            "context_source": f"{context_source}_cross_document_mbr",
                            **expansion_meta,
                        },
                    )
                )
                pending_debug.append(debug_row)
                if len(pending) >= batch_size:
                    flush_pending()
                continue


        retrieved_items = build_retrieved_context_items(
            result=result,
            chunk_items_by_id=chunk_items_by_id,
            top_k=int(args.top_k),
        )
        built = build_context(result=result, retrieved_items=retrieved_items, qa_entry=qa_entry)
        retrieved_scores = [item.score for item in retrieved_items]
        context_text = built.context_text
        context_token_length = active_generator.count_tokens(context_text)
        debug_row = {
            "example_id": example_id,
            "query_id": example_id,
            "doc_id": str(result.doc_id),
            "question": str(result.question or ""),
            "query": str(result.question or ""),
            "answers": answers,
            "retrieved_chunk_ids": [item.chunk_id for item in retrieved_items],
            "retrieved_scores": retrieved_scores,
            "final_context": context_text,
            "context_token_length": context_token_length,
            "context_tokens": context_token_length,
            "resumed_from_existing": False,
            **built.debug_payload,
        }
        if uses_generated_bridges:
            bridge_rows_by_id[example_id] = _generated_bridge_rows_from_debug(debug_row)
        pending.append(
            GenerationRequest(
                example_id=example_id,
                question=str(result.question or ""),
                qa_entry=qa_entry,
                context=context_text,
                metadata={
                    "doc_id": str(result.doc_id),
                    "answers": answers,
                    "choices": qa_entry.get("choices", []) if isinstance(qa_entry, dict) else [],
                    "retriever": str(args.retriever),
                    "retrieval_method": loader.selection.method,
                    "retrieval_scope": str(args.scope),
                    "top_k": int(args.top_k),
                    "summary_k": int(args.summary_k or 0),
                    "retrieved_chunk_ids": [item.chunk_id for item in retrieved_items],
                    "retrieved_scores": retrieved_scores,
                    "context_source": context_source,
                },
            )
        )
        pending_debug.append(debug_row)
        if len(pending) >= batch_size:
            flush_pending()
    flush_pending()

    generation_rows = ordered_rows(generation_by_id)
    debug_rows = ordered_rows(debug_by_id)

    write_jsonl(paths["generations_jsonl"], generation_rows)
    write_jsonl(paths["context_debug_jsonl"], debug_rows)
    generated_bridge_rows = ordered_bridge_rows()
    if uses_generated_bridges:
        write_jsonl(paths["generated_bridges_jsonl"], generated_bridge_rows)
    if expansion_cfg.enabled and expansion_bundle is not None:
        final_example_metadata_rows: list[dict[str, Any]] = []
        for example_id in selected_example_ids:
            row = dict(example_expansion_metadata_by_id.get(example_id, {}) or {})
            debug_row = debug_by_id.get(example_id, {})
            if row and debug_row:
                final_tokens = debug_row.get("final_context_tokens", debug_row.get("context_tokens"))
                if final_tokens not in (None, ""):
                    row["final_context_tokens"] = final_tokens
                if debug_row.get("saadi_expansion_trimmed_to_fit_method_context") is not None:
                    row["saadi_expansion_trimmed_to_fit_method_context"] = debug_row.get(
                        "saadi_expansion_trimmed_to_fit_method_context"
                    )
                    row["saadi_expansion_trim_side"] = debug_row.get("saadi_expansion_trim_side")
            if row:
                final_example_metadata_rows.append(row)
        expansion_artifact_paths = write_context_expansion_metadata(
            paths["experiment_output_dir"],
            expansion_bundle,
            final_example_metadata_rows,
        )
    eval_paths: dict[str, str] = {}
    if not args.no_eval:
        if not any(row.get("answers") for row in generation_rows):
            raise RuntimeError(
                "Evaluation requires answer references, but no generated rows had `answers`. "
                "Check that retrieval query_ids match the QA entries loaded from --config."
            )
        eval_cfg = _evaluation_cfg(config)
        bert_score_mode = str(args.bert_score or _forced_bert_score_mode())
        bert_score_device = args.bert_score_device
        if bert_score_device is None and eval_cfg.get("bert_score_device") not in (None, ""):
            bert_score_device = str(eval_cfg.get("bert_score_device"))
        bert_score_batch_size = int(args.bert_score_batch_size or eval_cfg.get("bert_score_batch_size", 16) or 16)
        eval_paths = write_generation_evaluation(
            paths["experiment_output_dir"],
            generation_rows,
            dataset_name=infer_dataset_label(config) if config else str(args.dataset),
            split=_dataset_split(config),
            method_name=f"saadi_{args.experiment}",
            run_name=str(args.output_name).strip() if args.output_name else f"{args.experiment}/top_{int(args.top_k)}",
            generation_top_k=int(args.top_k),
            bert_score_mode=bert_score_mode,
            bert_score_model=str(eval_cfg.get("bert_score_model", "roberta-large") or "roberta-large"),
            bert_score_lang=str(eval_cfg.get("bert_score_lang", "en") or "en"),
            bert_score_batch_size=bert_score_batch_size,
            bert_score_device=bert_score_device,
            bert_score_rescale_with_baseline=bool(eval_cfg.get("bert_score_rescale_with_baseline", False)),
            input_files_used={
                "generations_file": str(paths["generations_jsonl"]),
                "context_debug_file": str(paths["context_debug_jsonl"]),
                "generated_bridges_file": str(paths["generated_bridges_jsonl"]) if uses_generated_bridges else "",
                "mbr_results_json": str(loader.selection.results_json),
                "llm_cache_index_json": str(llm_cache_index_json),
                "config_file": str(Path(args.config).resolve()) if args.config else "",
            },
            command_used=" ".join(sys.argv),
        )
    run_config["status"] = "completed"
    run_config["started_at_utc"] = started_at
    run_config["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    run_config["retrieval"]["method"] = str(loader.selection.method)
    run_config["retrieval"]["mbr_results_json"] = str(loader.selection.results_json)
    run_config["retrieval"]["n_results_loaded"] = len(retrieval_results)
    run_config["raw_chunks"] = {
        "llm_cache_index_json": str(llm_cache_index_json),
        "n_raw_chunks_loaded": len(chunk_items_by_id),
    }
    if summaries_by_doc:
        summary_levels = sorted({int(block.level) for blocks in summaries_by_doc.values() for block in blocks})
        run_config["summaries"] = {
            "summary_level": "all_available_L_ge_1",
            "summary_levels_loaded": [f"L_{level}" for level in summary_levels],
            "summary_k": int(args.summary_k),
            "n_docs_with_summaries": len(summaries_by_doc),
            "n_summary_blocks_loaded": sum(len(blocks) for blocks in summaries_by_doc.values()),
        }
    run_config["qa"] = qa_meta
    if expansion_cfg.enabled:
        run_config["context_expansion"] = {
            **expansion_cfg.to_dict(),
            **expansion_doc_meta,
            "n_groups": len(expansion_bundle.group_metadata_rows) if expansion_bundle is not None else 0,
            "n_example_metadata_rows": len(example_expansion_metadata_by_id),
        }
    if generator is not None:
        prompt_version = generator.config.prompt_version
        max_context_tokens = generator.config.max_context_tokens
        overflow_policy = generator.config.overflow_policy
        max_output_tokens = generator.config.max_output_tokens
        temperature = generator.config.temperature
        stop = list(generator.config.stop or [])
    else:
        model_generate = model_cfg.get("generate", {}) if isinstance(model_cfg.get("generate", {}), dict) else {}
        prompt_version = str(generation_cfg.get("prompt_version", f"saadi_{args.experiment}_v1") or f"saadi_{args.experiment}_v1")
        max_context_tokens = (
            None
            if generation_cfg.get("max_context_tokens") in (None, "", "null")
            else int(generation_cfg.get("max_context_tokens"))
        )
        overflow_policy = str(generation_cfg.get("overflow_policy", "fail") or "fail")
        max_output_tokens = int(
            generation_cfg.get(
                "max_output_tokens",
                model_generate.get("max_tokens", 1024) if isinstance(model_generate, dict) else 1024,
            )
            or 1024
        )
        temperature = float(
            generation_cfg.get(
                "temperature",
                model_generate.get("temperature", 0.0) if isinstance(model_generate, dict) else 0.0,
            )
            or 0.0
        )
        stop_raw = generation_cfg.get("stop", model_generate.get("stop", ["<|im_end|>", "<|endoftext|>"]))
        if stop_raw is None:
            stop = []
        elif isinstance(stop_raw, list):
            stop = [str(item) for item in stop_raw if str(item).strip()]
        else:
            stop = [str(stop_raw)] if str(stop_raw).strip() else []
    run_config["generation"] = {
        "batch_size": batch_size,
        "prompt_version": prompt_version,
        "max_context_tokens": max_context_tokens,
        "overflow_policy": overflow_policy,
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
        "stop": stop,
        "n_generations": len(generation_rows),
        "n_selected_examples": len(selected_example_ids),
        "generated_count_this_run": generated_count,
        "reused_existing_count": reused_count,
        "rebuilt_debug_count": rebuilt_debug_count,
        "checkpoint_count": checkpoint_count,
        "generated_bridge_count": len(generated_bridge_rows),
        "existing_generation_rows_loaded": len(existing_generation_rows),
        "existing_generation_rows_reusable": len(existing_generation_by_id),
        "existing_context_debug_rows_loaded": len(existing_debug_rows),
        "existing_generated_bridge_rows_loaded": len(existing_bridge_rows),
        "resume_existing": not bool(args.force),
        "force": bool(args.force),
        "evaluation_enabled": not args.no_eval,
    }
    run_config["artifacts"].update(eval_paths)
    if expansion_artifact_paths:
        run_config["artifacts"].update(expansion_artifact_paths)
    if eval_paths:
        run_config["evaluation"] = eval_paths
    write_json(paths["run_config_json"], run_config)
    return run_config


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)
    out = run_generation_experiment(args)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
