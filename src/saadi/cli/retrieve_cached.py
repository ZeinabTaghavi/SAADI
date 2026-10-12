"""CLI command that runs retrieval-only cached experiments for Saadi and its retrieval ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..config.loader import load_config
from ..config.validation import validate_experiment
from ..context_strategies import strategy_cache_label
from ..entities import active_entity_config_label, resolve_active_entity_limits
from ..entities.active_selection import active_entity_selector_label, normalize_active_entity_selector
from ..indexing.dataset_loaders import load_qa_entries_from_config
from ..indexing.retrieval_methods.mbr import (
    DEFAULT_MBR_LAMBDA_H,
    DEFAULT_MBR_LAMBDA_M,
    DEFAULT_MBR_LAMBDA_T,
    DEFAULT_MBR_MEMORY_AGGREGATION,
)
from ..io.hierarchy_cache import (
    load_doc_ids_from_cache_index,
    load_llm_history_cache_index,
    resolve_llm_cache_namespace_dir,
)
from ..pipelines.retrieval_cached_pipeline import run_cached_retrieval_eval


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SAADI cached retrieval pipeline")
    p.add_argument("--config", required=False, help="Optional YAML/JSON config for cached retrieval")
    p.add_argument("--method", required=False, choices=["l0", "l0aj", "mbr", "mbr_linear", "mbr_pool"], help="Retrieval method to use")
    p.add_argument("--llm_cache_index_json", required=False, help="Path to llm_cache namespace index.json")
    p.add_argument("--qa_json", required=False, help="QA entries JSON list")
    p.add_argument("--output_dir", required=False, help="Optional output root; defaults to the llm_cache namespace dir")
    p.add_argument("--retrieve_k", type=int, default=None)
    p.add_argument("--eval_k", type=int, default=None, help="Primary HR cutoff (configuration value or 5)")
    p.add_argument("--retriever", default=None)
    p.add_argument("--scope", choices=["global", "per_document", "both"], default=None)
    p.add_argument("--augmentation_j", type=int, default=None)
    p.add_argument("--max_aliases", type=int, default=None)
    p.add_argument("--max_desc", type=int, default=None)
    p.add_argument("--lambda_t", type=float, default=None)
    p.add_argument("--lambda_n", type=float, default=None)
    p.add_argument("--lambda_h", type=float, default=None)
    p.add_argument("--memory_name_weight", type=float, default=None)
    p.add_argument("--memory_alias_weight", type=float, default=None)
    p.add_argument("--memory_description_weight", type=float, default=None)
    p.add_argument("--memory_aggregation", choices=["max", "mean"], default=None)
    p.add_argument("--normalization_method", required=False)
    return p


def _pick(args: argparse.Namespace, cfg: dict[str, Any], key: str, default: Any = None) -> Any:
    value = getattr(args, key, None)
    if value is not None:
        return value
    if key in cfg:
        return cfg.get(key)
    return default


def _pick_from_sections(
    args: argparse.Namespace,
    *,
    key: str,
    default: Any = None,
    sections: list[dict[str, Any] | None],
) -> Any:
    value = getattr(args, key, None)
    if value is not None:
        return value
    for section in sections:
        if isinstance(section, dict) and key in section and section.get(key) is not None:
            return section.get(key)
    return default


def _float_from_sections(
    args: argparse.Namespace,
    *,
    key: str,
    default: float,
    sections: list[dict[str, Any] | None],
) -> float:
    value = _pick_from_sections(args, key=key, default=default, sections=sections)
    if value is None or (isinstance(value, str) and not value.strip()):
        value = default
    return float(value)


def _resolve_path(value: Any, *, base_dir: Path | None) -> Any:
    if base_dir is None or value is None:
        return value
    text = str(value).strip()
    if not text:
        return value
    path = Path(text)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


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


def _infer_hierarchy_branching_factor(config: dict[str, Any]) -> int | None:
    ingest_cfg = config.get("ingest", {}) or {}
    if not isinstance(ingest_cfg, dict):
        return None
    raw_kwargs = ingest_cfg.get("strategy_kwargs", {}) or {}
    if isinstance(raw_kwargs, dict):
        for key in ("branching_factor", "hierarchy_branching_factor", "k"):
            if raw_kwargs.get(key) is not None:
                try:
                    return int(raw_kwargs.get(key))
                except Exception:
                    return None
    strategy_name = str(ingest_cfg.get("strategy", "hierarchical") or "hierarchical").strip().lower()
    if strategy_name in {"hierarchical", "saadi_hierarchical", "accumulative", "limited_hierarchical", "limited_hierarchical_summary", "saadi_limited_hierarchical"}:

        return 5
    return None


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


def _resolve_active_entity_selector(config: dict[str, Any]) -> str:
    configured = _configured_active_entity_selector(config)
    return configured if configured is not None else normalize_active_entity_selector("bm25")


def _infer_active_entity_selector_cache_label(config: dict[str, Any]) -> str | None:
    return active_entity_selector_label(_resolve_active_entity_selector(config))


def _infer_active_entity_config_cache_label(config: dict[str, Any]) -> str:
    max_entities, max_aliases, max_desc = _configured_active_entity_limits(config)
    return active_entity_config_label(max_entities, max_aliases, max_desc)


def _resolve_cache_index_from_config(config: dict[str, Any]) -> str | None:
    output_dir = config.get("output_dir")
    if not output_dir:
        return None
    ingest_cfg = config.get("ingest", {}) or {}
    strategy_name = str((ingest_cfg or {}).get("strategy", "hierarchical") or "hierarchical")
    strategy_kwargs = (ingest_cfg or {}).get("strategy_kwargs", {}) or {}
    method_label = strategy_cache_label(
        strategy_name,
        strategy_kwargs if isinstance(strategy_kwargs, dict) else {},
    )
    chunk_size_raw = (ingest_cfg or {}).get("chunk_size", 500)
    try:
        chunk_size = int(chunk_size_raw)
    except Exception:
        chunk_size = 500
    branching_factor = _infer_hierarchy_branching_factor(config)
    cache_dir = resolve_llm_cache_namespace_dir(
        str(output_dir),
        dataset_label=_infer_dataset_label(config),
        method_label=method_label,
        hierarchy_branching_factor=branching_factor,
        chunk_size=chunk_size,
        active_entity_config_label=_infer_active_entity_config_cache_label(config),
        llm_label=_infer_llm_label(config),
        active_entity_selector_label=_infer_active_entity_selector_cache_label(config),
    )
    if not (cache_dir / "index.json").exists():
        loaded = load_llm_history_cache_index(
            output_dir=str(output_dir),
            dataset_label=_infer_dataset_label(config),
            method_label=method_label,
            hierarchy_branching_factor=branching_factor,
            chunk_size=chunk_size,
            active_entity_config_label=_infer_active_entity_config_cache_label(config),
            llm_label=_infer_llm_label(config),
            active_entity_selector_label=_infer_active_entity_selector_cache_label(config),
        )
        if isinstance(loaded, dict) and loaded.get("index_path"):
            return str(loaded["index_path"])
    return str(cache_dir / "index.json")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cfg: dict[str, Any] = {}
    config_base_dir: Path | None = None
    if args.config:
        config_path = Path(args.config).resolve()
        config_base_dir = config_path.parent
        cfg = load_config(config_path)
        validate_experiment(cfg)

    retrieval_cfg = cfg.get("retrieval", {}) or {}
    if not isinstance(retrieval_cfg, dict):
        retrieval_cfg = {}
    cached_cfg = cfg.get("cached_retrieval", {}) or {}
    if not isinstance(cached_cfg, dict):
        cached_cfg = {}

    method = str(_pick_from_sections(args, key="method", default="mbr", sections=[cached_cfg, cfg]) or "mbr")
    active_entity_selector = _resolve_active_entity_selector(cfg)
    active_entity_limits = _configured_active_entity_limits(cfg)
    active_entity_config = _infer_active_entity_config_cache_label(cfg)
    llm_cache_index_json = _resolve_path(
        _pick_from_sections(args, key="llm_cache_index_json", default=None, sections=[cached_cfg, cfg]),
        base_dir=config_base_dir,
    )
    if not llm_cache_index_json:
        llm_cache_index_json = _resolve_cache_index_from_config(cfg)
    if not llm_cache_index_json:
        raise ValueError(
            "Could not resolve llm_cache_index_json. Provide it explicitly or include output_dir/dataset/ingest/model in the config."
        )

    qa_json = _resolve_path(
        _pick_from_sections(args, key="qa_json", default=None, sections=[cached_cfg, cfg]),
        base_dir=config_base_dir,
    )
    qa_entries: list[dict[str, Any]] | None = None
    if qa_json:
        pass
    elif isinstance(cfg.get("dataset"), dict):
        dataset_cfg = cfg.get("dataset", {})
        allowed_doc_ids: set[str] | None = None
        dataset_name = str((dataset_cfg or {}).get("name", "") or "").strip().lower()
        if dataset_name in {"novelhopqa", "abhaygupta1266/novelhopqa"}:
            cached_doc_ids = load_doc_ids_from_cache_index(llm_cache_index_json)
            if cached_doc_ids:
                allowed_doc_ids = set(cached_doc_ids)
        qa_entries, _ = load_qa_entries_from_config(
            dataset_cfg,
            base_dir=config_base_dir,
            allowed_doc_ids=allowed_doc_ids,
        )
    else:
        raise ValueError(
            "Could not resolve QA inputs. Provide qa_json explicitly or include a dataset config."
        )

    output_dir = _pick_from_sections(args, key="output_dir", default=None, sections=[cached_cfg])
    scope_value = str(_pick_from_sections(args, key="scope", default="per_document", sections=[cached_cfg, retrieval_cfg, cfg]) or "per_document")
    scopes = ["global", "per_document"] if scope_value == "both" else [scope_value]
    mbr_sections = [cached_cfg, retrieval_cfg, cfg]
    lambda_t = _float_from_sections(args, key="lambda_t", default=DEFAULT_MBR_LAMBDA_T, sections=mbr_sections)
    lambda_n = _float_from_sections(args, key="lambda_n", default=DEFAULT_MBR_LAMBDA_M, sections=mbr_sections)
    lambda_h = _float_from_sections(args, key="lambda_h", default=DEFAULT_MBR_LAMBDA_H, sections=mbr_sections)
    memory_name_weight = _float_from_sections(args, key="memory_name_weight", default=1.0, sections=mbr_sections)
    memory_alias_weight = _float_from_sections(args, key="memory_alias_weight", default=0.0, sections=mbr_sections)
    memory_description_weight = _float_from_sections(args, key="memory_description_weight", default=0.0, sections=mbr_sections)
    eval_k = _pick_from_sections(args, key="eval_k", default=5, sections=[cached_cfg, retrieval_cfg, cfg])
    eval_k = 5 if eval_k is None else int(eval_k)

    summaries: list[dict[str, Any]] = []
    for scope in scopes:
        out = run_cached_retrieval_eval(
            llm_cache_index_json=str(llm_cache_index_json),
            qa_json=str(qa_json) if qa_json else None,
            qa_entries=qa_entries,
            max_qa_entries=cfg.get("sample", {}).get("max_qa_entries"),
            method=method,
            output_dir=output_dir,
            retriever_name=str(_pick_from_sections(args, key="retriever", default="bm25", sections=[cached_cfg, retrieval_cfg, cfg]) or "bm25"),
            retrieve_k=int(_pick_from_sections(args, key="retrieve_k", default=10, sections=[cached_cfg, retrieval_cfg, cfg])),
            eval_k=eval_k,
            retrieval_scope=scope,
            augmentation_j=int(_pick_from_sections(args, key="augmentation_j", default=3, sections=[cached_cfg, cfg])),
            max_aliases=int(_pick_from_sections(args, key="max_aliases", default=2, sections=[cached_cfg, cfg])),
            max_desc=int(_pick_from_sections(args, key="max_desc", default=1, sections=[cached_cfg, cfg])),
            active_entity_selector=active_entity_selector,
            lambda_t=lambda_t,
            lambda_n=lambda_n,
            lambda_h=lambda_h,
            memory_name_weight=memory_name_weight,
            memory_alias_weight=memory_alias_weight,
            memory_description_weight=memory_description_weight,
            memory_aggregation=str(_pick_from_sections(args, key="memory_aggregation", default=DEFAULT_MBR_MEMORY_AGGREGATION, sections=[cached_cfg, retrieval_cfg, cfg]) or DEFAULT_MBR_MEMORY_AGGREGATION),
            normalization_method=str(_pick_from_sections(args, key="normalization_method", default="none", sections=[cached_cfg, retrieval_cfg, cfg]) or "none"),
        )
        summary = {
            "metric": "HR",
            "eval_k": out["report"].eval_k,
            "hit_rate": out["report"].hit_rate,
            "hit_rate_at_k": out["report"].hit_rate_at_k,
            "n_eligible": out["report"].n_union_eligible,
            "method": out["method_label"],
            "scope": scope,
            "active_entity_config": active_entity_config,
            "active_entity_selector": out.get("active_entity_selector", active_entity_selector),
            "active_entity_limits": {
                "max_entities": int(active_entity_limits[0]),
                "max_aliases": int(active_entity_limits[1]),
                "max_desc": int(active_entity_limits[2]),
            },
            "report_dir": str(out["artifacts"]["report_dir"]),
            "retrieval_results_json": str(out["artifacts"]["retrieval_results_json"]),
            "eval_report_json": str(out["artifacts"]["eval_report_json"]),
            "report_index_json": str(out["artifacts"]["report_index_json"]),
            "cached_documents": int((out.get("qa_alignment") or {}).get("cached_documents", 0)),
            "qa_entries_loaded": int((out.get("qa_alignment") or {}).get("qa_entries_loaded", 0)),
            "qa_entries_matched_cached_docs": int((out.get("qa_alignment") or {}).get("qa_entries_matched_cached_docs", 0)),
            "qa_entries_unmatched_cached_docs": int((out.get("qa_alignment") or {}).get("qa_entries_unmatched_cached_docs", 0)),
            "retrieval_examples_built": int((out.get("qa_alignment") or {}).get("retrieval_examples_built", 0)),
        }
        if method in {"mbr", "mbr_linear", "mbr_pool"}:
            summary["lambda_t"] = lambda_t
            summary["lambda_n"] = lambda_n
            summary["lambda_h"] = lambda_h
            if method == "mbr":
                summary["memory_name_weight"] = memory_name_weight
                summary["memory_alias_weight"] = memory_alias_weight
                summary["memory_description_weight"] = memory_description_weight
            summary["memory_aggregation"] = str(_pick_from_sections(args, key="memory_aggregation", default=DEFAULT_MBR_MEMORY_AGGREGATION, sections=[cached_cfg, retrieval_cfg, cfg]) or DEFAULT_MBR_MEMORY_AGGREGATION)
            summary["normalization_method"] = str(_pick_from_sections(args, key="normalization_method", default="none", sections=[cached_cfg, retrieval_cfg, cfg]) or "none")
            summary["mbr_diagnostics_jsonl"] = (
                str(out["artifacts"]["mbr_diagnostics_jsonl"])
                if out["artifacts"].get("mbr_diagnostics_jsonl") is not None
                else None
            )
        summaries.append(summary)

    print(json.dumps(summaries[0] if len(summaries) == 1 else {"runs": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
