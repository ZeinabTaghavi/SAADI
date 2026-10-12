"""Static statistics for chunk-level changes in updated L0 chunks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from ....core.types import CanonicalizedChunkRecord
from ....io.hierarchy_cache import load_canonicalized_chunk_records_from_cache_index
from ....io.jsonl import write_jsonl
from ....io.manifests import build_manifest, write_manifest
from ..common import (
    infer_dataset_label,
    infer_linking_sentence,
    materialized_parenthetical_insertions,
    resolve_updated_chunks_output_dir,
)
from .common import build_token_counter, compute_percentiles, safe_mean, safe_median

MAIN_PHASE_NAME = "Static Statistics / Cost / Structure"
SUB_PHASE_ID = "2_1_Chunk_Level_Changes"
SUB_PHASE_NAME = "Chunk-Level Changes"


def _build_item_id(dataset_name: str, record: CanonicalizedChunkRecord) -> str:
    return f"{dataset_name}:{record.doc_id}:{record.chunk_id}:{record.chunk_index}"


def build_chunk_level_change_records(
    records: list[CanonicalizedChunkRecord],
    *,
    dataset_name: str,
    count_tokens: Callable[[str], int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda row: (str(row.doc_id), int(row.chunk_index), str(row.chunk_id))):
        raw_chunk_text = str(record.raw_chunk_text or "")
        updated_chunk_text = str(record.canonicalized_chunk_text or "")
        raw_token_count = int(count_tokens(raw_chunk_text))
        updated_token_count = int(count_tokens(updated_chunk_text))
        token_increase = int(updated_token_count - raw_token_count)
        absolute_token_change = int(abs(token_increase))
        percent_token_increase = float((100.0 * token_increase / raw_token_count) if raw_token_count else 0.0)

        linking_sentence, linking_source = infer_linking_sentence(record)
        visible_insertions = materialized_parenthetical_insertions(record)
        linking_sentence_token_count = int(count_tokens(linking_sentence)) if linking_sentence else 0
        insertion_count = int(len(visible_insertions))

        rows.append(
            {
                "item_id": _build_item_id(dataset_name, record),
                "main_phase": MAIN_PHASE_NAME,
                "sub_phase_id": SUB_PHASE_ID,
                "sub_phase_name": SUB_PHASE_NAME,
                "dataset_name": dataset_name,
                "document_id": str(record.doc_id),
                "chunk_id": str(record.chunk_id),
                "chunk_index": int(record.chunk_index),
                "strategy_name": str(record.strategy_name or ""),
                "parser_status": str(record.parser_status or "ok"),
                "raw_token_count": raw_token_count,
                "updated_token_count": updated_token_count,
                "token_increase_tokens": token_increase,
                "absolute_token_change_tokens": absolute_token_change,
                "percent_token_increase": percent_token_increase,
                "has_linking_sentence": bool(linking_sentence),
                "linking_sentence_source": str(linking_source),
                "linking_sentence_token_count": linking_sentence_token_count,
                "parenthetical_insertions_count": insertion_count,
                "has_parenthetical_insertion": bool(insertion_count > 0),
            }
        )
    return rows


def aggregate_chunk_level_change_statistics(
    rows: list[dict[str, Any]],
    *,
    dataset_name: str,
    token_counting: dict[str, Any],
    document_id: str | None = None,
) -> dict[str, Any]:
    processed_chunks = int(len(rows))
    raw_lengths = [float(row.get("raw_token_count", 0) or 0) for row in rows]
    updated_lengths = [float(row.get("updated_token_count", 0) or 0) for row in rows]
    token_increases = [float(row.get("token_increase_tokens", 0) or 0) for row in rows]
    absolute_token_changes = [float(row.get("absolute_token_change_tokens", 0) or 0) for row in rows]
    percent_token_increases = [float(row.get("percent_token_increase", 0.0) or 0.0) for row in rows]
    linking_lengths = [
        float(row.get("linking_sentence_token_count", 0) or 0)
        for row in rows
        if bool(row.get("has_linking_sentence"))
    ]
    insertion_counts = [float(row.get("parenthetical_insertions_count", 0) or 0) for row in rows]
    parser_status_counts: dict[str, int] = {}
    linking_source_counts: dict[str, int] = {}
    for row in rows:
        parser_status = str(row.get("parser_status", "") or "").strip()
        if parser_status:
            parser_status_counts[parser_status] = int(parser_status_counts.get(parser_status, 0)) + 1
        linking_source = str(row.get("linking_sentence_source", "") or "").strip()
        if linking_source:
            linking_source_counts[linking_source] = int(linking_source_counts.get(linking_source, 0)) + 1

    chunks_with_linking_sentence = int(sum(1 for row in rows if bool(row.get("has_linking_sentence"))))
    chunks_with_parenthetical_insertion = int(sum(1 for row in rows if bool(row.get("has_parenthetical_insertion"))))

    summary = {
        "main_phase": MAIN_PHASE_NAME,
        "sub_phase_id": SUB_PHASE_ID,
        "sub_phase_name": SUB_PHASE_NAME,
        "dataset_name": dataset_name,
        "processed_chunks": processed_chunks,
        "documents_covered": int(len({str(row.get("document_id", "")).strip() for row in rows if str(row.get("document_id", "")).strip()})),
        "average_raw_chunk_tokens": safe_mean(raw_lengths),
        "average_updated_chunk_tokens": safe_mean(updated_lengths),
        "average_token_increase_tokens": safe_mean(token_increases),
        "average_absolute_token_change_tokens": safe_mean(absolute_token_changes),
        "average_percent_token_increase": safe_mean(percent_token_increases),
        "median_token_increase_tokens": safe_median(token_increases),
        "token_increase_percentiles_tokens": compute_percentiles(token_increases),
        "average_linking_sentence_tokens_when_present": safe_mean(linking_lengths),
        "chunks_with_linking_sentence": chunks_with_linking_sentence,
        "fraction_with_linking_sentence": float((chunks_with_linking_sentence / processed_chunks) if processed_chunks else 0.0),
        "average_parenthetical_insertions_per_chunk": safe_mean(insertion_counts),
        "chunks_with_parenthetical_insertion": chunks_with_parenthetical_insertion,
        "fraction_with_parenthetical_insertion": float((chunks_with_parenthetical_insertion / processed_chunks) if processed_chunks else 0.0),
        "parser_status_counts": dict(sorted(parser_status_counts.items())),
        "linking_sentence_source_counts": dict(sorted(linking_source_counts.items())),
        "token_counting": dict(token_counting),
    }
    if document_id is not None:
        summary["document_id"] = str(document_id)
    return summary


def build_per_document_chunk_level_statistics(
    rows: list[dict[str, Any]],
    *,
    dataset_name: str,
    token_counting: dict[str, Any],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("document_id", "")), []).append(row)
    summaries: list[dict[str, Any]] = []
    for document_id in sorted(doc_id for doc_id in grouped.keys() if doc_id):
        summaries.append(
            aggregate_chunk_level_change_statistics(
                grouped[document_id],
                dataset_name=dataset_name,
                token_counting=token_counting,
                document_id=document_id,
            )
        )
    return summaries


def _summary_markdown(summary: dict[str, Any]) -> str:
    percentiles = summary.get("token_increase_percentiles_tokens", {}) or {}
    return (
        f"# {SUB_PHASE_ID}\n\n"
        f"| Field | Value |\n"
        f"|---|---:|\n"
        f"| Processed chunks | {summary.get('processed_chunks', 0)} |\n"
        f"| Average raw chunk tokens | {summary.get('average_raw_chunk_tokens', 0.0):.3f} |\n"
        f"| Average updated chunk tokens | {summary.get('average_updated_chunk_tokens', 0.0):.3f} |\n"
        f"| Average token increase | {summary.get('average_token_increase_tokens', 0.0):.3f} |\n"
        f"| Average percent token increase | {summary.get('average_percent_token_increase', 0.0):.3f} |\n"
        f"| Median token increase | {summary.get('median_token_increase_tokens', 0.0):.3f} |\n"
        f"| P95 token increase | {float(percentiles.get('p95', 0.0)):.3f} |\n"
        f"| Fraction with linking sentence | {summary.get('fraction_with_linking_sentence', 0.0):.4f} |\n"
        f"| Avg linking-sentence tokens (when present) | {summary.get('average_linking_sentence_tokens_when_present', 0.0):.3f} |\n"
        f"| Fraction with parenthetical insertion | {summary.get('fraction_with_parenthetical_insertion', 0.0):.4f} |\n"
        f"| Avg parenthetical insertions / chunk | {summary.get('average_parenthetical_insertions_per_chunk', 0.0):.3f} |\n"
    )


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return str(path)


def run_chunk_level_changes_eval(
    *,
    experiment_config: dict[str, Any],
    llm_cache_index_json: str,
    output_dir: str | Path | None = None,
    token_counting_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dataset_name = infer_dataset_label(experiment_config)
    token_counter, token_counting_metadata = build_token_counter(
        token_counting_config=token_counting_config,
        experiment_config=experiment_config,
    )
    records = load_canonicalized_chunk_records_from_cache_index(llm_cache_index_json)
    per_chunk_rows = build_chunk_level_change_records(
        records,
        dataset_name=dataset_name,
        count_tokens=token_counter,
    )
    per_document_rows = build_per_document_chunk_level_statistics(
        per_chunk_rows,
        dataset_name=dataset_name,
        token_counting=token_counting_metadata,
    )
    summary = aggregate_chunk_level_change_statistics(
        per_chunk_rows,
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

    per_chunk_path = artifact_root / "per_chunk_stats.jsonl"
    per_document_path = artifact_root / "per_document_stats.jsonl"
    aggregate_path = artifact_root / "aggregate_stats.json"
    summary_md_path = artifact_root / "aggregate_stats.md"
    manifest_path = artifact_root / "run_manifest.json"

    summary["output_dir"] = str(artifact_root)
    summary["llm_cache_index_json"] = str(Path(llm_cache_index_json).resolve())

    write_jsonl(per_chunk_path, per_chunk_rows)
    write_jsonl(per_document_path, per_document_rows)
    _write_json(aggregate_path, summary)
    summary_md_path.write_text(_summary_markdown(summary), encoding="utf-8")

    manifest = build_manifest(
        config={
            "main_phase": MAIN_PHASE_NAME,
            "sub_phase_id": SUB_PHASE_ID,
            "sub_phase_name": SUB_PHASE_NAME,
            "experiment_config": experiment_config,
            "token_counting": dict(token_counting_config or {}),
        },
        artifacts={
            "per_chunk_stats_jsonl": str(per_chunk_path),
            "per_document_stats_jsonl": str(per_document_path),
            "aggregate_stats_json": str(aggregate_path),
            "aggregate_stats_markdown": str(summary_md_path),
        },
        metadata={
            "dataset_name": dataset_name,
            "processed_chunks": len(per_chunk_rows),
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
            "per_chunk_stats_jsonl": per_chunk_path,
            "per_document_stats_jsonl": per_document_path,
            "aggregate_stats_json": aggregate_path,
            "aggregate_stats_markdown": summary_md_path,
            "run_manifest_json": manifest_path,
        },
    }
