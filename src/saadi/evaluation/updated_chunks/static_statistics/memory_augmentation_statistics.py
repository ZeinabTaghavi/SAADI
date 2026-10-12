"""Static statistics for retrieval-time memory augmentation (e.g. L0AJ)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from ....entities.active_selection import normalize_active_entity_selector
from ....indexing.retrieval_methods.l0aj import (
    ENTITY_CONTEXT_HEADER,
    build_l0aj_rows,
    format_entity_context_block,
    format_l0aj_method_label,
)
from ....io.hierarchy_cache import load_canonicalized_chunk_records_from_cache_index
from ....io.jsonl import write_jsonl
from ....io.manifests import build_manifest, write_manifest
from ..common import (
    configured_active_entity_selector,
    infer_dataset_label,
    resolve_updated_chunks_output_dir,
)
from .common import (
    build_token_counter,
    load_cached_report_index,
    parse_l0aj_method_label,
    safe_mean,
    value_count_distribution,
)

MAIN_PHASE_NAME = "Static Statistics / Cost / Structure"
SUB_PHASE_ID = "2_3_Memory_Augmentation_Statistics"
SUB_PHASE_NAME = "Memory Augmentation Statistics"

DEFAULT_AUGMENTATION_J = 3
DEFAULT_MAX_ALIASES = 2
DEFAULT_MAX_DESC = 1
DEFAULT_SELECTOR = "word"


def _normalize_positive_int(value: Any, *, default: int, field_name: str) -> int:
    raw = value if value is not None else default
    try:
        out = int(raw)
    except Exception as exc:
        raise ValueError(f"{field_name} must be a positive integer (got: {value!r})") from exc
    if out <= 0:
        raise ValueError(f"{field_name} must be > 0 (got: {out})")
    return out


def _normalize_nonnegative_int(value: Any, *, default: int, field_name: str) -> int:
    raw = value if value is not None else default
    try:
        out = int(raw)
    except Exception as exc:
        raise ValueError(f"{field_name} must be a non-negative integer (got: {value!r})") from exc
    if out < 0:
        raise ValueError(f"{field_name} must be >= 0 (got: {out})")
    return out


def _resolve_selector(value: Any, *, experiment_config: dict[str, Any]) -> tuple[str, str]:
    text = str(value or "auto").strip().lower()
    if text in {"", "auto", "default"}:
        configured = configured_active_entity_selector(experiment_config)
        if configured is not None:
            return configured, "experiment_config"
        return DEFAULT_SELECTOR, "default"
    return normalize_active_entity_selector(text), "config"


def _infer_augmentation_j_from_report_index(payload: dict[str, Any] | None) -> tuple[int | None, str | None]:
    if not isinstance(payload, dict):
        return None, None

    direct = parse_l0aj_method_label(payload.get("retrieval_method"))
    if direct is not None:
        return direct, "report_index.retrieval_method"

    methods = payload.get("methods_executed", []) or []
    if isinstance(methods, list):
        for method in methods:
            parsed = parse_l0aj_method_label(method)
            if parsed is not None:
                return parsed, "report_index.methods_executed"

    runs = payload.get("runs", []) or []
    if isinstance(runs, list):
        for row in runs:
            if not isinstance(row, dict):
                continue
            parsed = parse_l0aj_method_label(row.get("method"))
            if parsed is not None:
                return parsed, "report_index.runs"
    return None, None


def resolve_memory_augmentation_settings(
    *,
    experiment_config: dict[str, Any],
    llm_cache_index_json: str | Path,
    memory_augmentation_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = memory_augmentation_config if isinstance(memory_augmentation_config, dict) else {}
    report_index_payload, report_index_path = load_cached_report_index(llm_cache_index_json)

    augmentation_source = "default"
    augmentation_value = cfg.get("augmentation_j", "auto")
    if isinstance(augmentation_value, str) and augmentation_value.strip().lower() in {"", "auto", "default"}:
        inferred, inferred_source = _infer_augmentation_j_from_report_index(report_index_payload)
        if inferred is not None:
            augmentation_j = int(inferred)
            augmentation_source = inferred_source or "report_index"
        else:
            augmentation_j = DEFAULT_AUGMENTATION_J
    elif augmentation_value is None:
        augmentation_j = DEFAULT_AUGMENTATION_J
    else:
        augmentation_j = _normalize_positive_int(
            augmentation_value,
            default=DEFAULT_AUGMENTATION_J,
            field_name="memory_augmentation.augmentation_j",
        )
        augmentation_source = "config"

    max_aliases = _normalize_nonnegative_int(
        cfg.get("max_aliases", DEFAULT_MAX_ALIASES),
        default=DEFAULT_MAX_ALIASES,
        field_name="memory_augmentation.max_aliases",
    )
    max_desc = _normalize_nonnegative_int(
        cfg.get("max_desc", DEFAULT_MAX_DESC),
        default=DEFAULT_MAX_DESC,
        field_name="memory_augmentation.max_desc",
    )
    active_entity_selector, selector_source = _resolve_selector(
        cfg.get("active_entity_selector"),
        experiment_config=experiment_config,
    )

    return {
        "augmentation_j": int(augmentation_j),
        "max_aliases": int(max_aliases),
        "max_desc": int(max_desc),
        "active_entity_selector": str(active_entity_selector),
        "method_label": format_l0aj_method_label(int(augmentation_j)),
        "augmentation_j_source": augmentation_source,
        "active_entity_selector_source": selector_source,
        "report_index_json": str(report_index_path),
        "report_index_found": bool(report_index_payload is not None),
    }


def _appended_memory_text(
    *,
    base_text: str,
    selected_entities: list[dict[str, Any]],
    header: str = ENTITY_CONTEXT_HEADER,
) -> str:
    block = format_entity_context_block(selected_entities, header=header)
    if not block:
        return ""
    if str(base_text or "").strip():
        return f"\n\n{block}"
    return block


def build_memory_augmentation_records(
    rows: list[dict[str, Any]],
    *,
    dataset_name: str,
    count_tokens: Callable[[str], int],
    augmentation_settings: dict[str, Any],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    configured_budget = int(augmentation_settings["augmentation_j"])
    for row in rows:
        base_chunk_text = str(row.get("canonicalized_chunk_text", "") or "")
        selected_entities = [
            entity for entity in (row.get("selected_entities", []) or [])
            if isinstance(entity, dict)
        ]
        appended_memory_text = _appended_memory_text(
            base_text=base_chunk_text,
            selected_entities=selected_entities,
        )
        appended_entity_count = int(len(selected_entities))
        appended_memory_token_count = int(count_tokens(appended_memory_text)) if appended_memory_text else 0
        base_chunk_token_count = int(count_tokens(base_chunk_text))
        retrieval_text_token_count = int(count_tokens(str(row.get("retrieval_text", "") or "")))
        appended_memory_percent_of_base_chunk = float(
            (100.0 * appended_memory_token_count / base_chunk_token_count)
            if base_chunk_token_count
            else 0.0
        )
        appended_alias_count = int(
            sum(len(entity.get("aliases", []) or []) for entity in selected_entities)
        )
        appended_description_count = int(
            sum(len(entity.get("description", []) or []) for entity in selected_entities)
        )
        usage_ratio = float(appended_entity_count / configured_budget) if configured_budget else 0.0

        out.append(
            {
                "item_id": f"{dataset_name}:{row.get('doc_id')}:{row.get('chunk_id')}:{row.get('chunk_index')}",
                "main_phase": MAIN_PHASE_NAME,
                "sub_phase_id": SUB_PHASE_ID,
                "sub_phase_name": SUB_PHASE_NAME,
                "dataset_name": dataset_name,
                "document_id": str(row.get("doc_id", "") or ""),
                "chunk_id": str(row.get("chunk_id", "") or ""),
                "chunk_index": int(row.get("chunk_index", 0) or 0),
                "base_chunk_token_count": base_chunk_token_count,
                "retrieval_text_token_count": retrieval_text_token_count,
                "document_memory_size": int(row.get("document_memory_size", 0) or 0),
                "has_appended_memory": bool(appended_entity_count > 0),
                "appended_active_entity_count": appended_entity_count,
                "appended_memory_token_count": appended_memory_token_count,
                "appended_memory_percent_of_base_chunk": appended_memory_percent_of_base_chunk,
                "configured_memory_budget_active_entities": configured_budget,
                "actual_memory_usage_relative_to_configured_budget": usage_ratio,
                "hits_configured_memory_budget": bool(appended_entity_count == configured_budget),
                "appended_alias_count": appended_alias_count,
                "appended_description_count": appended_description_count,
                "method_label": str(augmentation_settings["method_label"]),
                "active_entity_selector": str(augmentation_settings["active_entity_selector"]),
                "configured_max_aliases": int(augmentation_settings["max_aliases"]),
                "configured_max_desc": int(augmentation_settings["max_desc"]),
            }
        )
    return out


def aggregate_memory_augmentation_statistics(
    rows: list[dict[str, Any]],
    *,
    dataset_name: str,
    augmentation_settings: dict[str, Any],
    token_counting: dict[str, Any],
    document_id: str | None = None,
) -> dict[str, Any]:
    processed_items = int(len(rows))
    augmented_rows = [row for row in rows if bool(row.get("has_appended_memory"))]
    augmented_count = int(len(augmented_rows))

    entity_counts_all = [int(row.get("appended_active_entity_count", 0) or 0) for row in rows]
    entity_counts_aug = [int(row.get("appended_active_entity_count", 0) or 0) for row in augmented_rows]
    token_counts_all = [float(row.get("appended_memory_token_count", 0) or 0) for row in rows]
    token_counts_aug = [float(row.get("appended_memory_token_count", 0) or 0) for row in augmented_rows]
    percent_all = [float(row.get("appended_memory_percent_of_base_chunk", 0.0) or 0.0) for row in rows]
    percent_aug = [float(row.get("appended_memory_percent_of_base_chunk", 0.0) or 0.0) for row in augmented_rows]
    usage_all = [
        float(row.get("actual_memory_usage_relative_to_configured_budget", 0.0) or 0.0)
        for row in rows
    ]
    usage_aug = [
        float(row.get("actual_memory_usage_relative_to_configured_budget", 0.0) or 0.0)
        for row in augmented_rows
    ]
    alias_counts_all = [float(row.get("appended_alias_count", 0) or 0) for row in rows]
    alias_counts_aug = [float(row.get("appended_alias_count", 0) or 0) for row in augmented_rows]
    desc_counts_all = [float(row.get("appended_description_count", 0) or 0) for row in rows]
    desc_counts_aug = [float(row.get("appended_description_count", 0) or 0) for row in augmented_rows]
    budget_hits_all = int(sum(1 for row in rows if bool(row.get("hits_configured_memory_budget"))))
    budget_hits_aug = int(sum(1 for row in augmented_rows if bool(row.get("hits_configured_memory_budget"))))

    documents = {
        str(row.get("document_id", "")).strip()
        for row in rows
        if str(row.get("document_id", "")).strip()
    }
    documents_covered = int(len(documents))

    summary = {
        "main_phase": MAIN_PHASE_NAME,
        "sub_phase_id": SUB_PHASE_ID,
        "sub_phase_name": SUB_PHASE_NAME,
        "dataset_name": dataset_name,
        "method_label": str(augmentation_settings["method_label"]),
        "processed_items": processed_items,
        "items_with_appended_memory": augmented_count,
        "fraction_items_with_appended_memory": float((augmented_count / processed_items) if processed_items else 0.0),
        "average_appended_active_entities": safe_mean([float(value) for value in entity_counts_aug]),
        "average_appended_active_entities_per_item": safe_mean([float(value) for value in entity_counts_all]),
        "average_appended_memory_tokens": safe_mean(token_counts_aug),
        "average_appended_memory_tokens_per_item": safe_mean(token_counts_all),
        "average_appended_memory_percent_of_base_chunk": safe_mean(percent_aug),
        "average_appended_memory_percent_of_base_chunk_per_item": safe_mean(percent_all),
        "configured_memory_budget_active_entities": int(augmentation_settings["augmentation_j"]),
        "configured_max_aliases": int(augmentation_settings["max_aliases"]),
        "configured_max_desc": int(augmentation_settings["max_desc"]),
        "active_entity_selector": str(augmentation_settings["active_entity_selector"]),
        "average_actual_memory_usage_relative_to_configured_budget": safe_mean(usage_aug),
        "average_actual_memory_usage_relative_to_configured_budget_per_item": safe_mean(usage_all),
        "fraction_items_hitting_configured_budget": float((budget_hits_all / processed_items) if processed_items else 0.0),
        "fraction_augmented_items_hitting_configured_budget": float((budget_hits_aug / augmented_count) if augmented_count else 0.0),
        "average_aliases_in_appended_memory": safe_mean(alias_counts_aug),
        "average_aliases_in_appended_memory_per_item": safe_mean(alias_counts_all),
        "average_descriptions_in_appended_memory": safe_mean(desc_counts_aug),
        "average_descriptions_in_appended_memory_per_item": safe_mean(desc_counts_all),
        "documents_covered": documents_covered,
        "appended_active_entity_count_distribution": value_count_distribution(entity_counts_all),
        "appended_memory_token_count_distribution": value_count_distribution([int(value) for value in token_counts_all]),
        "token_counting": dict(token_counting),
        "augmentation_settings": dict(augmentation_settings),
    }
    if document_id is not None:
        summary["document_id"] = str(document_id)
    return summary


def build_per_document_memory_augmentation_statistics(
    rows: list[dict[str, Any]],
    *,
    dataset_name: str,
    augmentation_settings: dict[str, Any],
    token_counting: dict[str, Any],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("document_id", "") or ""), []).append(row)

    out: list[dict[str, Any]] = []
    for document_id in sorted(doc_id for doc_id in grouped.keys() if doc_id):
        out.append(
            aggregate_memory_augmentation_statistics(
                grouped[document_id],
                dataset_name=dataset_name,
                augmentation_settings=augmentation_settings,
                token_counting=token_counting,
                document_id=document_id,
            )
        )
    return out


def _summary_markdown(summary: dict[str, Any]) -> str:
    return (
        f"# {SUB_PHASE_ID}\n\n"
        f"| Field | Value |\n"
        f"|---|---:|\n"
        f"| Processed items | {summary.get('processed_items', 0)} |\n"
        f"| Items with appended memory | {summary.get('items_with_appended_memory', 0)} |\n"
        f"| Average appended active entities | {summary.get('average_appended_active_entities', 0.0):.3f} |\n"
        f"| Average appended memory tokens | {summary.get('average_appended_memory_tokens', 0.0):.3f} |\n"
        f"| Average appended memory % of base chunk | {summary.get('average_appended_memory_percent_of_base_chunk', 0.0):.3f} |\n"
        f"| Configured memory budget (entities) | {summary.get('configured_memory_budget_active_entities', 0)} |\n"
        f"| Average budget usage | {summary.get('average_actual_memory_usage_relative_to_configured_budget', 0.0):.3f} |\n"
        f"| Fraction augmented items hitting budget | {summary.get('fraction_augmented_items_hitting_configured_budget', 0.0):.4f} |\n"
        f"| Average aliases in appended memory | {summary.get('average_aliases_in_appended_memory', 0.0):.3f} |\n"
        f"| Average descriptions in appended memory | {summary.get('average_descriptions_in_appended_memory', 0.0):.3f} |\n"
    )


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return str(path)


def run_memory_augmentation_statistics_eval(
    *,
    experiment_config: dict[str, Any],
    llm_cache_index_json: str,
    output_dir: str | Path | None = None,
    memory_augmentation_config: dict[str, Any] | None = None,
    token_counting_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dataset_name = infer_dataset_label(experiment_config)
    augmentation_settings = resolve_memory_augmentation_settings(
        experiment_config=experiment_config,
        llm_cache_index_json=llm_cache_index_json,
        memory_augmentation_config=memory_augmentation_config,
    )
    token_counter, token_counting_metadata = build_token_counter(
        token_counting_config=token_counting_config,
        experiment_config=experiment_config,
    )
    records = load_canonicalized_chunk_records_from_cache_index(llm_cache_index_json)
    l0aj_rows = build_l0aj_rows(
        records,
        augmentation_j=int(augmentation_settings["augmentation_j"]),
        max_aliases=int(augmentation_settings["max_aliases"]),
        max_desc=int(augmentation_settings["max_desc"]),
        active_entity_selector=str(augmentation_settings["active_entity_selector"]),
    )
    per_item_rows = build_memory_augmentation_records(
        l0aj_rows,
        dataset_name=dataset_name,
        count_tokens=token_counter,
        augmentation_settings=augmentation_settings,
    )
    per_document_rows = build_per_document_memory_augmentation_statistics(
        per_item_rows,
        dataset_name=dataset_name,
        augmentation_settings=augmentation_settings,
        token_counting=token_counting_metadata,
    )
    summary = aggregate_memory_augmentation_statistics(
        per_item_rows,
        dataset_name=dataset_name,
        augmentation_settings=augmentation_settings,
        token_counting=token_counting_metadata,
    )

    artifact_root = (
        Path(output_dir).resolve()
        if output_dir is not None
        else resolve_updated_chunks_output_dir(
            experiment_config,
            subphase_id=SUB_PHASE_ID,
            run_name="default",
        )
    )
    artifact_root.mkdir(parents=True, exist_ok=True)

    per_item_path = artifact_root / "per_item_stats.jsonl"
    per_document_path = artifact_root / "per_document_stats.jsonl"
    aggregate_path = artifact_root / "aggregate_stats.json"
    summary_md_path = artifact_root / "aggregate_stats.md"
    manifest_path = artifact_root / "run_manifest.json"

    summary["output_dir"] = str(artifact_root)
    summary["llm_cache_index_json"] = str(Path(llm_cache_index_json).resolve())

    write_jsonl(per_item_path, per_item_rows)
    write_jsonl(per_document_path, per_document_rows)
    _write_json(aggregate_path, summary)
    summary_md_path.write_text(_summary_markdown(summary), encoding="utf-8")

    manifest = build_manifest(
        config={
            "main_phase": MAIN_PHASE_NAME,
            "sub_phase_id": SUB_PHASE_ID,
            "sub_phase_name": SUB_PHASE_NAME,
            "experiment_config": experiment_config,
            "memory_augmentation": dict(memory_augmentation_config or {}),
            "token_counting": dict(token_counting_config or {}),
        },
        artifacts={
            "per_item_stats_jsonl": str(per_item_path),
            "per_document_stats_jsonl": str(per_document_path),
            "aggregate_stats_json": str(aggregate_path),
            "aggregate_stats_markdown": str(summary_md_path),
        },
        metadata={
            "dataset_name": dataset_name,
            "processed_items": len(per_item_rows),
            "augmentation_settings": augmentation_settings,
            "token_counting": token_counting_metadata,
        },
        cwd=str(Path.cwd()),
    )
    write_manifest(manifest_path, manifest)

    return {
        "main_phase": MAIN_PHASE_NAME,
        "sub_phase_id": SUB_PHASE_ID,
        "sub_phase_name": SUB_PHASE_NAME,
        "dataset_name": dataset_name,
        "augmentation_settings": augmentation_settings,
        "token_counting": token_counting_metadata,
        "summary": summary,
        "artifacts": {
            "artifact_root": artifact_root,
            "per_item_stats_jsonl": per_item_path,
            "per_document_stats_jsonl": per_document_path,
            "aggregate_stats_json": aggregate_path,
            "aggregate_stats_markdown": summary_md_path,
            "run_manifest_json": manifest_path,
        },
    }
