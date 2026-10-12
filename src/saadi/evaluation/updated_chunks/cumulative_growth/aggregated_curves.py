"""Dataset-level aggregated cumulative curves built from per-document growth logs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from ....io.jsonl import read_jsonl, write_jsonl
from ....io.manifests import build_manifest, write_manifest
from ..common import infer_dataset_label, resolve_updated_chunks_output_dir
from .common import resolve_per_document_growth_output_dir

MAIN_PHASE_NAME = "Cumulative Growth Analysis"
SUB_PHASE_ID = "3_2_Dataset_Level_Aggregated_Cumulative_Curves"
SUB_PHASE_NAME = "Dataset-Level Aggregated Cumulative Curves"
SOURCE_SUB_PHASE_ID = "3_1_Per_Document_Cumulative_Growth_Logging"
DEFAULT_NORMALIZED_GRID_SIZE = 101

_COUNT_FIELDS = (
    "cumulative_entity_names",
    "cumulative_aliases",
    "cumulative_descriptions",
    "cumulative_total_modifications",
)


def _normalize_positive_int(value: Any, *, default: int, field_name: str) -> int:
    raw = default if value is None else value
    try:
        out = int(raw)
    except Exception as exc:
        raise ValueError(f"{field_name} must be a positive integer (got: {value!r})") from exc
    if out <= 0:
        raise ValueError(f"{field_name} must be > 0 (got: {out})")
    return out


def _safe_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(stdev(values))


def _safe_sem(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(_safe_std(values) / math.sqrt(len(values)))


def _grid_positions(grid_size: int) -> list[float]:
    if grid_size <= 1:
        return [0.0]
    return [float(index / (grid_size - 1)) for index in range(grid_size)]


def _to_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _to_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def group_per_document_growth_rows(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        document_id = str(row.get("document_id", "") or "").strip()
        if not document_id:
            continue
        grouped.setdefault(document_id, []).append(row)
    for document_id, doc_rows in grouped.items():
        doc_rows.sort(
            key=lambda row: (
                _to_int(row.get("chunk_order_in_document"), default=_to_int(row.get("chunk_index"), default=0)),
                _to_int(row.get("chunk_index"), default=0),
                str(row.get("chunk_id", "") or ""),
            )
        )
    return grouped


def _carry_forward_row_at_position(doc_rows: list[dict[str, Any]], normalized_position: float) -> dict[str, Any]:
    if not doc_rows:
        return {}
    selected = doc_rows[0]
    target = float(normalized_position)
    for row in doc_rows:
        row_position = _to_float(row.get("normalized_chunk_position"), default=0.0)
        if row_position <= target:
            selected = row
        else:
            break
    return selected


def build_normalized_position_curve(
    per_chunk_rows: list[dict[str, Any]],
    *,
    dataset_name: str,
    grid_size: int = DEFAULT_NORMALIZED_GRID_SIZE,
) -> list[dict[str, Any]]:
    grid_size = _normalize_positive_int(grid_size, default=DEFAULT_NORMALIZED_GRID_SIZE, field_name="normalized_grid_size")
    grouped = group_per_document_growth_rows(per_chunk_rows)
    positions = _grid_positions(grid_size)
    document_ids = sorted(grouped)
    out: list[dict[str, Any]] = []

    for grid_index, normalized_position in enumerate(positions):
        selected_rows = [
            _carry_forward_row_at_position(grouped[document_id], normalized_position)
            for document_id in document_ids
            if grouped.get(document_id)
        ]
        contributing_documents = int(len(selected_rows))
        row: dict[str, Any] = {
            "item_id": f"{dataset_name}:normalized:{grid_index}",
            "main_phase": MAIN_PHASE_NAME,
            "sub_phase_id": SUB_PHASE_ID,
            "sub_phase_name": SUB_PHASE_NAME,
            "dataset_name": dataset_name,
            "curve_space": "normalized_position",
            "normalized_position": float(normalized_position),
            "grid_index": int(grid_index),
            "contributing_documents": contributing_documents,
            "interpolation_method": "stepwise_carry_forward",
        }
        for source_key, target_key in (
            ("cumulative_entity_names", "entity_names"),
            ("cumulative_aliases", "aliases"),
            ("cumulative_descriptions", "descriptions"),
            ("cumulative_total_modifications", "total_modifications"),
        ):
            values = [_to_float(selected.get(source_key), default=0.0) for selected in selected_rows]
            row[f"average_cumulative_{target_key}"] = float(mean(values) if values else 0.0)
            row[f"std_cumulative_{target_key}"] = float(_safe_std(values))
            row[f"sem_cumulative_{target_key}"] = float(_safe_sem(values))
        out.append(row)
    return out


def build_raw_chunk_index_curve(
    per_chunk_rows: list[dict[str, Any]],
    *,
    dataset_name: str,
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in per_chunk_rows:
        chunk_index = _to_int(row.get("chunk_index"), default=0)
        grouped.setdefault(chunk_index, []).append(row)

    out: list[dict[str, Any]] = []
    for chunk_index in sorted(grouped):
        rows = grouped[chunk_index]
        row: dict[str, Any] = {
            "item_id": f"{dataset_name}:raw_chunk_index:{chunk_index}",
            "main_phase": MAIN_PHASE_NAME,
            "sub_phase_id": SUB_PHASE_ID,
            "sub_phase_name": SUB_PHASE_NAME,
            "dataset_name": dataset_name,
            "curve_space": "raw_chunk_index",
            "raw_chunk_index": int(chunk_index),
            "contributing_documents": int(len(rows)),
        }
        for source_key, target_key in (
            ("cumulative_entity_names", "entity_names"),
            ("cumulative_aliases", "aliases"),
            ("cumulative_descriptions", "descriptions"),
            ("cumulative_total_modifications", "total_modifications"),
        ):
            values = [_to_float(item.get(source_key), default=0.0) for item in rows]
            row[f"average_cumulative_{target_key}"] = float(mean(values) if values else 0.0)
            row[f"std_cumulative_{target_key}"] = float(_safe_std(values))
            row[f"sem_cumulative_{target_key}"] = float(_safe_sem(values))
        out.append(row)
    return out


def summarize_aggregated_cumulative_curves(
    per_chunk_rows: list[dict[str, Any]],
    normalized_curve_rows: list[dict[str, Any]],
    raw_curve_rows: list[dict[str, Any]],
    *,
    dataset_name: str,
    normalized_grid_size: int,
    source_per_chunk_path: str | Path,
) -> dict[str, Any]:
    grouped = group_per_document_growth_rows(per_chunk_rows)
    final_rows = [doc_rows[-1] for doc_rows in grouped.values() if doc_rows]
    summary: dict[str, Any] = {
        "main_phase": MAIN_PHASE_NAME,
        "sub_phase_id": SUB_PHASE_ID,
        "sub_phase_name": SUB_PHASE_NAME,
        "dataset_name": dataset_name,
        "source_sub_phase_id": SOURCE_SUB_PHASE_ID,
        "source_per_chunk_cumulative_growth_jsonl": str(Path(source_per_chunk_path).resolve()),
        "processed_documents": int(len(grouped)),
        "processed_chunks": int(len(per_chunk_rows)),
        "normalized_grid_size": int(normalized_grid_size),
        "normalized_curve_points": int(len(normalized_curve_rows)),
        "raw_chunk_curve_points": int(len(raw_curve_rows)),
        "normalized_curve_interpolation_method": "stepwise_carry_forward",
    }
    for source_key, target_key in (
        ("cumulative_entity_names", "entity_names"),
        ("cumulative_aliases", "aliases"),
        ("cumulative_descriptions", "descriptions"),
        ("cumulative_total_modifications", "total_modifications"),
    ):
        values = [_to_float(row.get(source_key), default=0.0) for row in final_rows]
        summary[f"average_final_cumulative_{target_key}"] = float(mean(values) if values else 0.0)
        summary[f"std_final_cumulative_{target_key}"] = float(_safe_std(values))
        summary[f"sem_final_cumulative_{target_key}"] = float(_safe_sem(values))
    return summary


def _summary_markdown(summary: dict[str, Any]) -> str:
    return (
        f"# {SUB_PHASE_ID}\n\n"
        "| Field | Value |\n"
        "|---|---:|\n"
        f"| Processed documents | {summary.get('processed_documents', 0)} |\n"
        f"| Processed chunks | {summary.get('processed_chunks', 0)} |\n"
        f"| Normalized grid size | {summary.get('normalized_grid_size', 0)} |\n"
        f"| Normalized curve points | {summary.get('normalized_curve_points', 0)} |\n"
        f"| Raw chunk curve points | {summary.get('raw_chunk_curve_points', 0)} |\n"
        f"| Avg final cumulative entity names | {summary.get('average_final_cumulative_entity_names', 0.0):.3f} |\n"
        f"| Avg final cumulative aliases | {summary.get('average_final_cumulative_aliases', 0.0):.3f} |\n"
        f"| Avg final cumulative descriptions | {summary.get('average_final_cumulative_descriptions', 0.0):.3f} |\n"
        f"| Avg final cumulative total modifications | {summary.get('average_final_cumulative_total_modifications', 0.0):.3f} |\n"
    )


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return str(path)


def run_dataset_level_aggregated_cumulative_curves_eval(
    *,
    experiment_config: dict[str, Any],
    per_chunk_cumulative_growth_jsonl: str,
    normalized_grid_size: int = DEFAULT_NORMALIZED_GRID_SIZE,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    dataset_name = infer_dataset_label(experiment_config)
    per_chunk_rows = read_jsonl(per_chunk_cumulative_growth_jsonl)
    normalized_grid_size = _normalize_positive_int(
        normalized_grid_size,
        default=DEFAULT_NORMALIZED_GRID_SIZE,
        field_name="normalized_grid_size",
    )
    normalized_curve_rows = build_normalized_position_curve(
        per_chunk_rows,
        dataset_name=dataset_name,
        grid_size=normalized_grid_size,
    )
    raw_curve_rows = build_raw_chunk_index_curve(
        per_chunk_rows,
        dataset_name=dataset_name,
    )
    summary = summarize_aggregated_cumulative_curves(
        per_chunk_rows,
        normalized_curve_rows,
        raw_curve_rows,
        dataset_name=dataset_name,
        normalized_grid_size=normalized_grid_size,
        source_per_chunk_path=per_chunk_cumulative_growth_jsonl,
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

    normalized_curve_path = artifact_root / "normalized_position_curve.jsonl"
    raw_curve_path = artifact_root / "raw_chunk_index_curve.jsonl"
    summary_path = artifact_root / "aggregate_summary.json"
    summary_md_path = artifact_root / "aggregate_summary.md"
    manifest_path = artifact_root / "run_manifest.json"

    summary["output_dir"] = str(artifact_root)

    write_jsonl(normalized_curve_path, normalized_curve_rows)
    write_jsonl(raw_curve_path, raw_curve_rows)
    _write_json(summary_path, summary)
    summary_md_path.write_text(_summary_markdown(summary), encoding="utf-8")

    manifest = build_manifest(
        config={
            "main_phase": MAIN_PHASE_NAME,
            "sub_phase_id": SUB_PHASE_ID,
            "sub_phase_name": SUB_PHASE_NAME,
            "experiment_config": experiment_config,
            "source_per_chunk_cumulative_growth_jsonl": str(per_chunk_cumulative_growth_jsonl),
            "normalized_grid_size": int(normalized_grid_size),
        },
        artifacts={
            "normalized_position_curve_jsonl": str(normalized_curve_path),
            "raw_chunk_index_curve_jsonl": str(raw_curve_path),
            "aggregate_summary_json": str(summary_path),
            "aggregate_summary_markdown": str(summary_md_path),
        },
        metadata={
            "dataset_name": dataset_name,
            "processed_documents": summary.get("processed_documents", 0),
            "processed_chunks": summary.get("processed_chunks", 0),
            "source_sub_phase_id": SOURCE_SUB_PHASE_ID,
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
        "artifacts": {
            "artifact_root": artifact_root,
            "normalized_position_curve_jsonl": normalized_curve_path,
            "raw_chunk_index_curve_jsonl": raw_curve_path,
            "aggregate_summary_json": summary_path,
            "aggregate_summary_markdown": summary_md_path,
            "run_manifest_json": manifest_path,
        },
    }


def resolve_source_per_chunk_growth_path(
    experiment_config: dict[str, Any],
    *,
    source_run_name: str = "default",
) -> Path:
    return resolve_per_document_growth_output_dir(
        experiment_config,
        run_name=source_run_name,
    ) / "per_chunk_cumulative_growth.jsonl"
