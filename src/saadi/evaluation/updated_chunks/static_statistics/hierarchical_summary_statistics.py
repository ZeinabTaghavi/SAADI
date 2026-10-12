"""Static statistics for hierarchical summaries produced over updated chunks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from ....io.jsonl import write_jsonl
from ....io.manifests import build_manifest, write_manifest
from ..common import (
    infer_dataset_label,
    infer_hierarchy_branching_factor,
    resolve_updated_chunks_output_dir,
)
from ..faithfulness.hierarchical_summary import build_hierarchical_summary_items_from_cache_index
from .common import build_token_counter, safe_mean, safe_median, value_count_distribution

MAIN_PHASE_NAME = "Static Statistics / Cost / Structure"
SUB_PHASE_ID = "2_4_Hierarchical_Summary_Statistics"
SUB_PHASE_NAME = "Hierarchical Summary Statistics"


def _summary_level_sort_key(level_label: str) -> tuple[int, str]:
    try:
        if str(level_label).startswith("H"):
            return (int(str(level_label)[1:]), str(level_label))
    except Exception:
        pass
    return (10_000, str(level_label))


def build_hierarchical_summary_records(
    items: list[dict[str, Any]],
    *,
    dataset_name: str,
    count_tokens: Callable[[str], int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        summary_text = str(item.get("generated_summary_text", "") or "")
        source_inputs = [row for row in (item.get("source_inputs", []) or []) if isinstance(row, dict)]
        summary_token_count = int(count_tokens(summary_text))
        source_input_token_count = int(
            sum(count_tokens(str(source.get("text", "") or "")) for source in source_inputs)
        )
        compression_ratio_summary_to_source = float(
            (summary_token_count / source_input_token_count)
            if source_input_token_count
            else 0.0
        )
        rows.append(
            {
                "item_id": str(item.get("item_id", "") or ""),
                "main_phase": MAIN_PHASE_NAME,
                "sub_phase_id": SUB_PHASE_ID,
                "sub_phase_name": SUB_PHASE_NAME,
                "dataset_name": dataset_name,
                "document_id": str(item.get("document_id", "") or ""),
                "summary_id": str(item.get("summary_id", "") or ""),
                "summary_level": str(item.get("summary_level", "") or ""),
                "summary_level_index": int(item.get("summary_level_index", 0) or 0),
                "summary_cycle": int(item.get("summary_cycle", 0) or 0),
                "summary_step": int(item.get("summary_step", 0) or 0),
                "source_level": str(item.get("source_level", "") or ""),
                "source_input_kind": str(item.get("source_input_kind", "") or ""),
                "source_input_count": int(item.get("source_input_count", 0) or 0),
                "summary_token_count": summary_token_count,
                "source_input_token_count": source_input_token_count,
                "compression_ratio_summary_to_source": compression_ratio_summary_to_source,
            }
        )
    rows.sort(
        key=lambda row: (
            str(row.get("document_id", "")),
            int(row.get("summary_level_index", -1)),
            int(row.get("summary_step", -1)),
            str(row.get("summary_id", "")),
        )
    )
    return rows


def aggregate_hierarchical_summary_overall_statistics(
    rows: list[dict[str, Any]],
    *,
    dataset_name: str,
    token_counting: dict[str, Any],
) -> dict[str, Any]:
    processed_summaries = int(len(rows))
    documents = {
        str(row.get("document_id", "")).strip()
        for row in rows
        if str(row.get("document_id", "")).strip()
    }
    documents_covered = int(len(documents))
    summary_token_counts = [int(row.get("summary_token_count", 0) or 0) for row in rows]
    source_token_counts = [int(row.get("source_input_token_count", 0) or 0) for row in rows]
    compression_ratios = [float(row.get("compression_ratio_summary_to_source", 0.0) or 0.0) for row in rows]
    source_input_counts = [int(row.get("source_input_count", 0) or 0) for row in rows]

    summary_counts_by_level: dict[str, int] = {}
    for row in rows:
        level = str(row.get("summary_level", "") or "").strip()
        if not level:
            continue
        summary_counts_by_level[level] = int(summary_counts_by_level.get(level, 0)) + 1

    return {
        "main_phase": MAIN_PHASE_NAME,
        "sub_phase_id": SUB_PHASE_ID,
        "sub_phase_name": SUB_PHASE_NAME,
        "dataset_name": dataset_name,
        "total_summaries": processed_summaries,
        "documents_covered": documents_covered,
        "summary_levels_observed": sorted(summary_counts_by_level.keys(), key=_summary_level_sort_key),
        "summary_counts_by_level": dict(sorted(summary_counts_by_level.items(), key=lambda item: _summary_level_sort_key(item[0]))),
        "total_summary_tokens": int(sum(summary_token_counts)),
        "average_summary_tokens": safe_mean([float(value) for value in summary_token_counts]),
        "median_summary_tokens": safe_median([float(value) for value in summary_token_counts]),
        "average_source_input_tokens": safe_mean([float(value) for value in source_token_counts]),
        "average_source_inputs_per_summary": safe_mean([float(value) for value in source_input_counts]),
        "average_compression_ratio_summary_to_source": safe_mean(compression_ratios),
        "average_summaries_per_document": float((processed_summaries / documents_covered) if documents_covered else 0.0),
        "summary_token_count_distribution": value_count_distribution(summary_token_counts),
        "token_counting": dict(token_counting),
    }


def build_hierarchical_summary_level_statistics(
    rows: list[dict[str, Any]],
    *,
    dataset_name: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("summary_level", "") or ""), []).append(row)

    summaries: list[dict[str, Any]] = []
    for summary_level in sorted((level for level in grouped.keys() if level), key=_summary_level_sort_key):
        level_rows = grouped[summary_level]
        summary_token_counts = [int(row.get("summary_token_count", 0) or 0) for row in level_rows]
        source_token_counts = [int(row.get("source_input_token_count", 0) or 0) for row in level_rows]
        compression_ratios = [float(row.get("compression_ratio_summary_to_source", 0.0) or 0.0) for row in level_rows]
        source_input_counts = [int(row.get("source_input_count", 0) or 0) for row in level_rows]
        documents = {
            str(row.get("document_id", "")).strip()
            for row in level_rows
            if str(row.get("document_id", "")).strip()
        }
        source_level_counts: dict[str, int] = {}
        for row in level_rows:
            source_level = str(row.get("source_level", "") or "").strip()
            if source_level:
                source_level_counts[source_level] = int(source_level_counts.get(source_level, 0)) + 1
        summaries.append(
            {
                "main_phase": MAIN_PHASE_NAME,
                "sub_phase_id": SUB_PHASE_ID,
                "sub_phase_name": SUB_PHASE_NAME,
                "dataset_name": dataset_name,
                "summary_level": summary_level,
                "summary_level_index": int(level_rows[0].get("summary_level_index", 0) or 0),
                "total_summaries": int(len(level_rows)),
                "documents_covered": int(len(documents)),
                "average_summary_tokens": safe_mean([float(value) for value in summary_token_counts]),
                "median_summary_tokens": safe_median([float(value) for value in summary_token_counts]),
                "average_source_input_tokens": safe_mean([float(value) for value in source_token_counts]),
                "median_source_input_tokens": safe_median([float(value) for value in source_token_counts]),
                "average_source_inputs_per_summary": safe_mean([float(value) for value in source_input_counts]),
                "average_compression_ratio_summary_to_source": safe_mean(compression_ratios),
                "total_summary_tokens": int(sum(summary_token_counts)),
                "summary_token_count_distribution": value_count_distribution(summary_token_counts),
                "source_level_counts": dict(sorted(source_level_counts.items(), key=lambda item: _summary_level_sort_key(item[0]))),
            }
        )
    return summaries


def build_hierarchical_summary_document_statistics(
    rows: list[dict[str, Any]],
    *,
    dataset_name: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("document_id", "") or ""), []).append(row)

    summaries: list[dict[str, Any]] = []
    for document_id in sorted(doc_id for doc_id in grouped.keys() if doc_id):
        doc_rows = grouped[document_id]
        summary_token_counts = [int(row.get("summary_token_count", 0) or 0) for row in doc_rows]
        summary_counts_by_level: dict[str, int] = {}
        total_summary_tokens_by_level: dict[str, int] = {}
        for row in doc_rows:
            level = str(row.get("summary_level", "") or "").strip()
            if not level:
                continue
            summary_counts_by_level[level] = int(summary_counts_by_level.get(level, 0)) + 1
            total_summary_tokens_by_level[level] = int(
                total_summary_tokens_by_level.get(level, 0) + int(row.get("summary_token_count", 0) or 0)
            )
        summaries.append(
            {
                "main_phase": MAIN_PHASE_NAME,
                "sub_phase_id": SUB_PHASE_ID,
                "sub_phase_name": SUB_PHASE_NAME,
                "dataset_name": dataset_name,
                "document_id": document_id,
                "total_summaries": int(len(doc_rows)),
                "total_summary_tokens": int(sum(summary_token_counts)),
                "average_summary_tokens": safe_mean([float(value) for value in summary_token_counts]),
                "median_summary_tokens": safe_median([float(value) for value in summary_token_counts]),
                "summary_counts_by_level": dict(sorted(summary_counts_by_level.items(), key=lambda item: _summary_level_sort_key(item[0]))),
                "total_summary_tokens_by_level": dict(sorted(total_summary_tokens_by_level.items(), key=lambda item: _summary_level_sort_key(item[0]))),
            }
        )
    return summaries


def _summary_markdown(summary: dict[str, Any], per_level_rows: list[dict[str, Any]]) -> str:
    lines = [
        f"# {SUB_PHASE_ID}",
        "",
        "| Field | Value |",
        "|---|---:|",
        f"| Total summaries | {summary.get('total_summaries', 0)} |",
        f"| Documents covered | {summary.get('documents_covered', 0)} |",
        f"| Total summary tokens | {summary.get('total_summary_tokens', 0)} |",
        f"| Average summary tokens | {summary.get('average_summary_tokens', 0.0):.3f} |",
        f"| Median summary tokens | {summary.get('median_summary_tokens', 0.0):.3f} |",
        f"| Average summaries per document | {summary.get('average_summaries_per_document', 0.0):.3f} |",
        f"| Average compression ratio | {summary.get('average_compression_ratio_summary_to_source', 0.0):.3f} |",
    ]
    if per_level_rows:
        lines.extend(
            [
                "",
                "## Per Summary Level",
                "",
                "| Level | Count | Avg tokens | Median tokens | Total tokens | Avg compression ratio |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in sorted(per_level_rows, key=lambda item: (int(item.get("summary_level_index", 0)), str(item.get("summary_level", "")))):
            lines.append(
                f"| {row.get('summary_level', '')} | {row.get('total_summaries', 0)} | "
                f"{row.get('average_summary_tokens', 0.0):.3f} | {row.get('median_summary_tokens', 0.0):.3f} | "
                f"{row.get('total_summary_tokens', 0)} | {row.get('average_compression_ratio_summary_to_source', 0.0):.3f} |"
            )
    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return str(path)


def run_hierarchical_summary_statistics_eval(
    *,
    experiment_config: dict[str, Any],
    llm_cache_index_json: str,
    output_dir: str | Path | None = None,
    token_counting_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dataset_name = infer_dataset_label(experiment_config)
    branching_factor = infer_hierarchy_branching_factor(experiment_config)
    token_counter, token_counting_metadata = build_token_counter(
        token_counting_config=token_counting_config,
        experiment_config=experiment_config,
    )
    items = build_hierarchical_summary_items_from_cache_index(
        llm_cache_index_json,
        dataset_name=dataset_name,
        branching_factor=branching_factor,
    )
    per_summary_rows = build_hierarchical_summary_records(
        items,
        dataset_name=dataset_name,
        count_tokens=token_counter,
    )
    per_level_rows = build_hierarchical_summary_level_statistics(
        per_summary_rows,
        dataset_name=dataset_name,
    )
    per_document_rows = build_hierarchical_summary_document_statistics(
        per_summary_rows,
        dataset_name=dataset_name,
    )
    summary = aggregate_hierarchical_summary_overall_statistics(
        per_summary_rows,
        dataset_name=dataset_name,
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

    per_summary_path = artifact_root / "per_summary_stats.jsonl"
    per_level_path = artifact_root / "per_level_stats.jsonl"
    per_document_path = artifact_root / "per_document_stats.jsonl"
    aggregate_path = artifact_root / "aggregate_stats.json"
    summary_md_path = artifact_root / "aggregate_stats.md"
    manifest_path = artifact_root / "run_manifest.json"

    summary["output_dir"] = str(artifact_root)
    summary["llm_cache_index_json"] = str(Path(llm_cache_index_json).resolve())

    write_jsonl(per_summary_path, per_summary_rows)
    write_jsonl(per_level_path, per_level_rows)
    write_jsonl(per_document_path, per_document_rows)
    _write_json(aggregate_path, summary)
    summary_md_path.write_text(_summary_markdown(summary, per_level_rows), encoding="utf-8")

    manifest = build_manifest(
        config={
            "main_phase": MAIN_PHASE_NAME,
            "sub_phase_id": SUB_PHASE_ID,
            "sub_phase_name": SUB_PHASE_NAME,
            "experiment_config": experiment_config,
            "token_counting": dict(token_counting_config or {}),
        },
        artifacts={
            "per_summary_stats_jsonl": str(per_summary_path),
            "per_level_stats_jsonl": str(per_level_path),
            "per_document_stats_jsonl": str(per_document_path),
            "aggregate_stats_json": str(aggregate_path),
            "aggregate_stats_markdown": str(summary_md_path),
        },
        metadata={
            "dataset_name": dataset_name,
            "total_summaries": len(per_summary_rows),
            "summary_levels_observed": summary.get("summary_levels_observed", []),
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
        "summary": summary,
        "token_counting": token_counting_metadata,
        "artifacts": {
            "artifact_root": artifact_root,
            "per_summary_stats_jsonl": per_summary_path,
            "per_level_stats_jsonl": per_level_path,
            "per_document_stats_jsonl": per_document_path,
            "aggregate_stats_json": aggregate_path,
            "aggregate_stats_markdown": summary_md_path,
            "run_manifest_json": manifest_path,
        },
    }
