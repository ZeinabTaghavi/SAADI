"""Reporting-friendly summaries for cumulative growth analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ....io.jsonl import write_jsonl
from ....io.manifests import build_manifest, write_manifest
from .comparative_plotting import (
    build_comparative_curve_export_rows,
    load_dataset_curve_entries,
    plot_comparative_cumulative_figures,
)

MAIN_PHASE_NAME = "Cumulative Growth Analysis"
SUB_PHASE_ID = "3_4_Methodology_Summary_and_Reporting_Outputs"
SUB_PHASE_NAME = "Methodology Summary and Reporting Outputs"
SOURCE_SUB_PHASE_ID = "3_2_Dataset_Level_Aggregated_Cumulative_Curves"
DEFAULT_MILESTONES = (0.25, 0.50, 0.75, 1.00)
DEFAULT_PLOT_FORMATS = ("png",)


def _normalize_milestones(value: Any) -> list[float]:
    if value is None:
        return [float(x) for x in DEFAULT_MILESTONES]
    raw_values = value if isinstance(value, list) else [value]
    out: list[float] = []
    for raw in raw_values:
        try:
            milestone = float(raw)
        except Exception as exc:
            raise ValueError(f"Invalid milestone value: {raw!r}") from exc
        if not (0.0 <= milestone <= 1.0):
            raise ValueError(f"Milestone must be within [0.0, 1.0] (got: {milestone})")
        if milestone not in out:
            out.append(float(milestone))
    return sorted(out)


def _normalize_plot_formats(value: Any) -> list[str]:
    allowed = {"png", "pdf", "svg"}
    if value is None:
        return [str(fmt) for fmt in DEFAULT_PLOT_FORMATS]
    raw_values = value if isinstance(value, list) else [value]
    out: list[str] = []
    for raw in raw_values:
        item = str(raw or "").strip().lower()
        if not item:
            continue
        if item not in allowed:
            raise ValueError(f"Unsupported plot format: {raw!r}. Expected one of: {sorted(allowed)}")
        if item not in out:
            out.append(item)
    return out or [str(fmt) for fmt in DEFAULT_PLOT_FORMATS]


def _normalize_positive_int(value: Any, *, default: int, field_name: str) -> int:
    raw = default if value is None else value
    try:
        out = int(raw)
    except Exception as exc:
        raise ValueError(f"{field_name} must be a positive integer (got: {value!r})") from exc
    if out <= 0:
        raise ValueError(f"{field_name} must be > 0 (got: {out})")
    return out


def _milestone_label(value: float) -> str:
    return f"{int(round(value * 100.0))}_percent"


def _nearest_curve_row(curve_rows: list[dict[str, Any]], target_position: float) -> dict[str, Any]:
    if not curve_rows:
        return {}
    best = curve_rows[0]
    best_distance = abs(float(best.get("normalized_position", 0.0) or 0.0) - target_position)
    for row in curve_rows[1:]:
        position = float(row.get("normalized_position", 0.0) or 0.0)
        distance = abs(position - target_position)
        if distance < best_distance:
            best = row
            best_distance = distance
    return best


def build_final_summary_rows(dataset_curves: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in dataset_curves:
        curve_rows = entry.get("curve_rows", []) or []
        if not curve_rows:
            continue
        final_row = max(
            curve_rows,
            key=lambda row: float(row.get("normalized_position", 0.0) or 0.0),
        )
        rows.append(
            {
                "dataset_name": str(entry.get("dataset_name", "") or ""),
                "dataset_label": str(entry.get("dataset_label", "") or ""),
                "source_sub_phase_id": SOURCE_SUB_PHASE_ID,
                "normalized_position": float(final_row.get("normalized_position", 0.0) or 0.0),
                "contributing_documents": int(final_row.get("contributing_documents", 0) or 0),
                "average_final_cumulative_entity_names": float(final_row.get("average_cumulative_entity_names", 0.0) or 0.0),
                "sem_final_cumulative_entity_names": float(final_row.get("sem_cumulative_entity_names", 0.0) or 0.0),
                "average_final_cumulative_aliases": float(final_row.get("average_cumulative_aliases", 0.0) or 0.0),
                "sem_final_cumulative_aliases": float(final_row.get("sem_cumulative_aliases", 0.0) or 0.0),
                "average_final_cumulative_descriptions": float(final_row.get("average_cumulative_descriptions", 0.0) or 0.0),
                "sem_final_cumulative_descriptions": float(final_row.get("sem_cumulative_descriptions", 0.0) or 0.0),
                "average_final_cumulative_total_modifications": float(final_row.get("average_cumulative_total_modifications", 0.0) or 0.0),
                "sem_final_cumulative_total_modifications": float(final_row.get("sem_cumulative_total_modifications", 0.0) or 0.0),
                "source_normalized_curve_jsonl": str(entry.get("normalized_curve_jsonl", "") or ""),
            }
        )
    rows.sort(key=lambda row: str(row.get("dataset_name", "")))
    return rows


def build_milestone_summary_rows(
    dataset_curves: list[dict[str, Any]],
    *,
    milestones: list[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in dataset_curves:
        curve_rows = entry.get("curve_rows", []) or []
        if not curve_rows:
            continue
        dataset_name = str(entry.get("dataset_name", "") or "")
        dataset_label = str(entry.get("dataset_label", "") or dataset_name)
        for milestone in milestones:
            selected = _nearest_curve_row(curve_rows, float(milestone))
            rows.append(
                {
                    "dataset_name": dataset_name,
                    "dataset_label": dataset_label,
                    "milestone_label": _milestone_label(float(milestone)),
                    "milestone_normalized_position_target": float(milestone),
                    "selected_normalized_position": float(selected.get("normalized_position", 0.0) or 0.0),
                    "contributing_documents": int(selected.get("contributing_documents", 0) or 0),
                    "average_cumulative_entity_names": float(selected.get("average_cumulative_entity_names", 0.0) or 0.0),
                    "sem_cumulative_entity_names": float(selected.get("sem_cumulative_entity_names", 0.0) or 0.0),
                    "average_cumulative_aliases": float(selected.get("average_cumulative_aliases", 0.0) or 0.0),
                    "sem_cumulative_aliases": float(selected.get("sem_cumulative_aliases", 0.0) or 0.0),
                    "average_cumulative_descriptions": float(selected.get("average_cumulative_descriptions", 0.0) or 0.0),
                    "sem_cumulative_descriptions": float(selected.get("sem_cumulative_descriptions", 0.0) or 0.0),
                    "average_cumulative_total_modifications": float(selected.get("average_cumulative_total_modifications", 0.0) or 0.0),
                    "sem_cumulative_total_modifications": float(selected.get("sem_cumulative_total_modifications", 0.0) or 0.0),
                    "source_normalized_curve_jsonl": str(entry.get("normalized_curve_jsonl", "") or ""),
                }
            )
    rows.sort(
        key=lambda row: (
            str(row.get("dataset_name", "")),
            float(row.get("milestone_normalized_position_target", 0.0)),
        )
    )
    return rows


def _final_summary_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        f"# {SUB_PHASE_ID} Final Values",
        "",
        "| Dataset | Final entity names | Final aliases | Final descriptions | Final total modifications |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('dataset_label', row.get('dataset_name', ''))} | "
            f"{float(row.get('average_final_cumulative_entity_names', 0.0) or 0.0):.3f} | "
            f"{float(row.get('average_final_cumulative_aliases', 0.0) or 0.0):.3f} | "
            f"{float(row.get('average_final_cumulative_descriptions', 0.0) or 0.0):.3f} | "
            f"{float(row.get('average_final_cumulative_total_modifications', 0.0) or 0.0):.3f} |"
        )
    return "\n".join(lines) + "\n"


def _milestone_summary_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        f"# {SUB_PHASE_ID} Milestones",
        "",
        "| Dataset | Milestone | Entity names | Aliases | Descriptions | Total modifications |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('dataset_label', row.get('dataset_name', ''))} | "
            f"{row.get('milestone_label', '')} | "
            f"{float(row.get('average_cumulative_entity_names', 0.0) or 0.0):.3f} | "
            f"{float(row.get('average_cumulative_aliases', 0.0) or 0.0):.3f} | "
            f"{float(row.get('average_cumulative_descriptions', 0.0) or 0.0):.3f} | "
            f"{float(row.get('average_cumulative_total_modifications', 0.0) or 0.0):.3f} |"
        )
    return "\n".join(lines) + "\n"


def _report_overview_markdown(
    final_rows: list[dict[str, Any]],
    milestone_rows: list[dict[str, Any]],
    *,
    include_plots: bool,
    figure_records: list[dict[str, Any]] | None = None,
) -> str:
    lines = [
        f"# {SUB_PHASE_ID} Reporting Bundle",
        "",
        f"- Datasets: {len(final_rows)}",
        f"- Final summary rows: {len(final_rows)}",
        f"- Milestone summary rows: {len(milestone_rows)}",
    ]
    if include_plots:
        lines.append(f"- Comparative figures: {len(figure_records or [])}")
    lines.extend(
        [
            "",
            "## Included Tables",
            "",
            "- `final_summary_table.jsonl`",
            "- `milestone_summary_table.jsonl`",
            "- `final_summary_table.md`",
            "- `milestone_summary_table.md`",
        ]
    )
    if include_plots:
        lines.extend(
            [
                "",
                "## Included Figures",
                "",
            ]
        )
        for record in figure_records or []:
            first_path = ""
            paths = record.get("paths") or {}
            if isinstance(paths, dict) and paths:
                first_path = str(next(iter(paths.values())))
            lines.append(
                f"- Figure {record.get('figure_index')}: {record.get('title')} "
                f"(`{first_path}`)"
            )
    return "\n".join(lines) + "\n"


def summarize_reporting_outputs(
    dataset_curves: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    milestone_rows: list[dict[str, Any]],
    *,
    milestones: list[float],
    include_plots: bool,
    figure_records: list[dict[str, Any]] | None = None,
    figure_index_json: str | None = None,
) -> dict[str, Any]:
    return {
        "main_phase": MAIN_PHASE_NAME,
        "sub_phase_id": SUB_PHASE_ID,
        "sub_phase_name": SUB_PHASE_NAME,
        "source_sub_phase_id": SOURCE_SUB_PHASE_ID,
        "number_of_datasets": int(len(dataset_curves)),
        "datasets_included": [
            {
                "dataset_name": str(entry.get("dataset_name", "") or ""),
                "dataset_label": str(entry.get("dataset_label", "") or ""),
                "normalized_curve_jsonl": str(entry.get("normalized_curve_jsonl", "") or ""),
            }
            for entry in dataset_curves
        ],
        "milestones": [float(value) for value in milestones],
        "final_summary_rows": int(len(final_rows)),
        "milestone_summary_rows": int(len(milestone_rows)),
        "plots_included": bool(include_plots),
        "number_of_figures": int(len(figure_records or [])),
        "figure_index_json": str(figure_index_json or ""),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return str(path)


def run_methodology_summary_and_reporting_outputs_eval(
    *,
    dataset_entries: list[dict[str, Any]],
    config_base_dir: Path | None = None,
    output_dir: str | Path,
    run_name: str = "default",
    default_source_run_name: str = "default",
    selected_dataset_names: list[str] | None = None,
    milestones: list[float] | None = None,
    include_plots: bool = True,
    plot_formats: list[str] | None = None,
    plot_dpi: int = 200,
    plot_use_sem_bands: bool = True,
) -> dict[str, Any]:
    normalized_milestones = _normalize_milestones(milestones)
    normalized_plot_formats = _normalize_plot_formats(plot_formats)
    normalized_plot_dpi = _normalize_positive_int(plot_dpi, default=200, field_name="plotting.dpi")
    artifact_root = Path(output_dir).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)

    dataset_curves = load_dataset_curve_entries(
        dataset_entries=dataset_entries,
        config_base_dir=config_base_dir,
        default_source_run_name=default_source_run_name,
        selected_dataset_names=selected_dataset_names,
    )
    if not dataset_curves:
        raise ValueError("No dataset curves were selected for reporting summaries.")

    final_rows = build_final_summary_rows(dataset_curves)
    milestone_rows = build_milestone_summary_rows(
        dataset_curves,
        milestones=normalized_milestones,
    )

    final_jsonl_path = artifact_root / "final_summary_table.jsonl"
    milestone_jsonl_path = artifact_root / "milestone_summary_table.jsonl"
    final_md_path = artifact_root / "final_summary_table.md"
    milestone_md_path = artifact_root / "milestone_summary_table.md"
    report_overview_path = artifact_root / "report_overview.md"
    summary_path = artifact_root / "run_summary.json"
    manifest_path = artifact_root / "run_manifest.json"
    figures_dir = artifact_root / "figures"
    figure_records: list[dict[str, Any]] = []
    figure_index_path: Path | None = None
    curve_values_path: Path | None = None

    write_jsonl(final_jsonl_path, final_rows)
    write_jsonl(milestone_jsonl_path, milestone_rows)
    final_md_path.write_text(_final_summary_markdown(final_rows), encoding="utf-8")
    milestone_md_path.write_text(_milestone_summary_markdown(milestone_rows), encoding="utf-8")

    if include_plots:
        curve_values_rows = build_comparative_curve_export_rows(dataset_curves)
        figures_dir.mkdir(parents=True, exist_ok=True)
        curve_values_path = figures_dir / "normalized_position_curve_values.jsonl"
        write_jsonl(curve_values_path, curve_values_rows)
        figure_records = plot_comparative_cumulative_figures(
            dataset_curves=dataset_curves,
            output_dir=figures_dir,
            formats=normalized_plot_formats,
            dpi=normalized_plot_dpi,
            use_sem_bands=bool(plot_use_sem_bands),
        )
        figure_index_path = figures_dir / "figure_index.json"
        _write_json(figure_index_path, {"figures": figure_records})

    report_overview_path.write_text(
        _report_overview_markdown(
            final_rows,
            milestone_rows,
            include_plots=bool(include_plots),
            figure_records=figure_records,
        ),
        encoding="utf-8",
    )

    summary = summarize_reporting_outputs(
        dataset_curves,
        final_rows,
        milestone_rows,
        milestones=normalized_milestones,
        include_plots=bool(include_plots),
        figure_records=figure_records,
        figure_index_json=str(figure_index_path) if figure_index_path else None,
    )
    summary["run_name"] = str(run_name or "default")
    summary["output_dir"] = str(artifact_root)
    _write_json(summary_path, summary)

    manifest = build_manifest(
        config={
            "main_phase": MAIN_PHASE_NAME,
            "sub_phase_id": SUB_PHASE_ID,
            "sub_phase_name": SUB_PHASE_NAME,
            "run_name": str(run_name or "default"),
            "source_sub_phase_id": SOURCE_SUB_PHASE_ID,
            "source_run_name": str(default_source_run_name or "default"),
            "datasets": [
                {
                    "dataset_name": entry["dataset_name"],
                    "dataset_label": entry["dataset_label"],
                    "experiment_config_path": entry["experiment_config_path"],
                    "normalized_curve_jsonl": entry["normalized_curve_jsonl"],
                }
                for entry in dataset_curves
            ],
            "milestones": normalized_milestones,
            "plotting": {
                "enabled": bool(include_plots),
                "formats": list(normalized_plot_formats) if include_plots else [],
                "dpi": int(normalized_plot_dpi),
                "use_sem_bands": bool(plot_use_sem_bands),
            },
        },
        artifacts={
            "final_summary_table_jsonl": str(final_jsonl_path),
            "milestone_summary_table_jsonl": str(milestone_jsonl_path),
            "final_summary_table_markdown": str(final_md_path),
            "milestone_summary_table_markdown": str(milestone_md_path),
            "report_overview_markdown": str(report_overview_path),
            "run_summary_json": str(summary_path),
            **(
                {
                    "normalized_position_curve_values_jsonl": str(curve_values_path),
                    "figure_index_json": str(figure_index_path),
                    **{
                        f"figure_{record['figure_index']}_{fmt}": str(path)
                        for record in figure_records
                        for fmt, path in (record.get("paths") or {}).items()
                    },
                }
                if include_plots and curve_values_path and figure_index_path
                else {}
            ),
        },
        metadata={
            "number_of_datasets": len(dataset_curves),
            "final_summary_rows": len(final_rows),
            "milestone_summary_rows": len(milestone_rows),
            "number_of_figures": len(figure_records),
        },
        cwd=str(Path.cwd()),
    )
    write_manifest(manifest_path, manifest)

    return {
        "main_phase": MAIN_PHASE_NAME,
        "sub_phase_id": SUB_PHASE_ID,
        "sub_phase_name": SUB_PHASE_NAME,
        "summary": summary,
        "artifacts": {
            "artifact_root": artifact_root,
            "final_summary_table_jsonl": final_jsonl_path,
            "milestone_summary_table_jsonl": milestone_jsonl_path,
            "final_summary_table_markdown": final_md_path,
            "milestone_summary_table_markdown": milestone_md_path,
            "report_overview_markdown": report_overview_path,
            "run_summary_json": summary_path,
            "run_manifest_json": manifest_path,
            **(
                {
                    "normalized_position_curve_values_jsonl": curve_values_path,
                    "figure_index_json": figure_index_path,
                    **{
                        f"figure_{record['figure_index']}_{fmt}": Path(path)
                        for record in figure_records
                        for fmt, path in (record.get("paths") or {}).items()
                    },
                }
                if include_plots and curve_values_path and figure_index_path
                else {}
            ),
        },
    }
