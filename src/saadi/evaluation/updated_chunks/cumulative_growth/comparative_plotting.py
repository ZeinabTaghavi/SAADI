"""Comparative plotting across datasets for cumulative growth analysis."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from ....config.loader import load_config
from ....io.jsonl import read_jsonl, write_jsonl
from ....io.manifests import build_manifest, write_manifest
from ..common import infer_dataset_label, resolve_path_from_base
from .common import resolve_aggregated_growth_output_dir

MAIN_PHASE_NAME = "Cumulative Growth Analysis"
SUB_PHASE_ID = "3_3_Comparative_Plotting_Across_Datasets"
SUB_PHASE_NAME = "Comparative Plotting Across Datasets"
SOURCE_SUB_PHASE_ID = "3_2_Dataset_Level_Aggregated_Cumulative_Curves"

_METRIC_SPECS: list[dict[str, str | int]] = [
    {
        "figure_index": 1,
        "metric_key": "average_cumulative_entity_names",
        "sem_key": "sem_cumulative_entity_names",
        "title": "Cumulative New Entity Names Across Datasets",
        "y_label": "Average cumulative new entity names",
        "slug": "cumulative_entity_names",
    },
    {
        "figure_index": 2,
        "metric_key": "average_cumulative_aliases",
        "sem_key": "sem_cumulative_aliases",
        "title": "Cumulative New Aliases Across Datasets",
        "y_label": "Average cumulative new aliases",
        "slug": "cumulative_aliases",
    },
    {
        "figure_index": 3,
        "metric_key": "average_cumulative_descriptions",
        "sem_key": "sem_cumulative_descriptions",
        "title": "Cumulative New Descriptions Across Datasets",
        "y_label": "Average cumulative new descriptions",
        "slug": "cumulative_descriptions",
    },
    {
        "figure_index": 4,
        "metric_key": "average_cumulative_total_modifications",
        "sem_key": "sem_cumulative_total_modifications",
        "title": "Cumulative Total Modifications Across Datasets",
        "y_label": "Average cumulative total modifications",
        "slug": "cumulative_total_modifications",
    },
]


def _safe_path_slug(value: Any) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "")).strip("._")
    return clean or "value"


def _normalize_output_formats(value: Any) -> list[str]:
    allowed = {"png", "pdf", "svg"}
    if value is None:
        return ["png"]
    if isinstance(value, str):
        raw_values = [value]
    elif isinstance(value, list):
        raw_values = value
    else:
        raise ValueError(f"plotting.formats must be a string or list[str] (got: {value!r})")
    out: list[str] = []
    for raw in raw_values:
        item = str(raw or "").strip().lower()
        if not item:
            continue
        if item not in allowed:
            raise ValueError(f"Unsupported plot format: {raw!r}. Expected one of: {sorted(allowed)}")
        if item not in out:
            out.append(item)
    return out or ["png"]


def _normalize_positive_int(value: Any, *, default: int, field_name: str) -> int:
    raw = default if value is None else value
    try:
        out = int(raw)
    except Exception as exc:
        raise ValueError(f"{field_name} must be a positive integer (got: {value!r})") from exc
    if out <= 0:
        raise ValueError(f"{field_name} must be > 0 (got: {out})")
    return out


def _default_dataset_label(name: str) -> str:
    aliases = {
        "qasper": "QASPER",
        "nqa": "NarrativeQA",
        "narrativeqa": "NarrativeQA",
        "loogle": "LooGLE",
        "novelqa": "NovelQA",
        "novelhopqa": "NovelHopQA",
        "quality": "QuALITY",
    }
    key = str(name or "").strip().lower()
    return aliases.get(key, str(name or "").strip() or key or "Dataset")


def _load_dataset_entry(
    raw_entry: dict[str, Any],
    *,
    config_base_dir: Path | None,
    default_source_run_name: str,
) -> dict[str, Any]:
    experiment_config_path = resolve_path_from_base(
        raw_entry.get("experiment_config"),
        base_dir=config_base_dir,
    )
    if not experiment_config_path:
        raise ValueError("Each dataset entry must provide experiment_config.")
    experiment_cfg = load_config(experiment_config_path)
    experiment_cfg = dict(experiment_cfg)
    experiment_base_dir = Path(experiment_config_path).resolve().parent
    resolved_output_dir = resolve_path_from_base(experiment_cfg.get("output_dir"), base_dir=experiment_base_dir)
    if resolved_output_dir:
        experiment_cfg["output_dir"] = resolved_output_dir

    dataset_name = str(raw_entry.get("name") or infer_dataset_label(experiment_cfg)).strip()
    if not dataset_name:
        dataset_name = infer_dataset_label(experiment_cfg)
    dataset_label = str(raw_entry.get("label") or _default_dataset_label(dataset_name)).strip() or dataset_name
    source_run_name = str(raw_entry.get("source_run_name") or default_source_run_name or "default").strip() or "default"
    normalized_curve_path = resolve_path_from_base(
        raw_entry.get("normalized_curve_jsonl"),
        base_dir=config_base_dir,
    )
    if not normalized_curve_path:
        normalized_curve_path = str(
            resolve_aggregated_growth_output_dir(
                experiment_cfg,
                run_name=source_run_name,
            )
            / "normalized_position_curve.jsonl"
        )
    return {
        "dataset_name": dataset_name,
        "dataset_label": dataset_label,
        "source_run_name": source_run_name,
        "experiment_config": experiment_cfg,
        "experiment_config_path": str(Path(experiment_config_path).resolve()),
        "normalized_curve_jsonl": str(Path(normalized_curve_path).resolve()),
    }


def load_dataset_curve_entries(
    *,
    dataset_entries: list[dict[str, Any]],
    config_base_dir: Path | None = None,
    default_source_run_name: str = "default",
    selected_dataset_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    selected = {str(name).strip().lower() for name in (selected_dataset_names or []) if str(name).strip()}
    out: list[dict[str, Any]] = []
    for raw_entry in dataset_entries:
        if not isinstance(raw_entry, dict):
            continue
        entry = _load_dataset_entry(
            raw_entry,
            config_base_dir=config_base_dir,
            default_source_run_name=default_source_run_name,
        )
        if selected and entry["dataset_name"].strip().lower() not in selected:
            continue
        curve_path = Path(entry["normalized_curve_jsonl"])
        if not curve_path.exists():
            raise FileNotFoundError(
                f"Missing normalized-position curve for dataset '{entry['dataset_name']}': {curve_path}"
            )
        curve_rows = read_jsonl(curve_path)
        curve_rows = sorted(
            [
                row for row in curve_rows
                if str(row.get("curve_space", "normalized_position")) == "normalized_position"
            ],
            key=lambda row: float(row.get("normalized_position", 0.0) or 0.0),
        )
        if not curve_rows:
            raise ValueError(f"No normalized-position curve rows found for dataset '{entry['dataset_name']}'.")
        entry["curve_rows"] = curve_rows
        out.append(entry)
    return out


def build_comparative_curve_export_rows(dataset_curves: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset_entry in dataset_curves:
        dataset_name = str(dataset_entry.get("dataset_name", "") or "")
        dataset_label = str(dataset_entry.get("dataset_label", "") or dataset_name)
        for row in dataset_entry.get("curve_rows", []) or []:
            rows.append(
                {
                    "dataset_name": dataset_name,
                    "dataset_label": dataset_label,
                    "normalized_position": float(row.get("normalized_position", 0.0) or 0.0),
                    "contributing_documents": int(row.get("contributing_documents", 0) or 0),
                    "average_cumulative_entity_names": float(row.get("average_cumulative_entity_names", 0.0) or 0.0),
                    "sem_cumulative_entity_names": float(row.get("sem_cumulative_entity_names", 0.0) or 0.0),
                    "average_cumulative_aliases": float(row.get("average_cumulative_aliases", 0.0) or 0.0),
                    "sem_cumulative_aliases": float(row.get("sem_cumulative_aliases", 0.0) or 0.0),
                    "average_cumulative_descriptions": float(row.get("average_cumulative_descriptions", 0.0) or 0.0),
                    "sem_cumulative_descriptions": float(row.get("sem_cumulative_descriptions", 0.0) or 0.0),
                    "average_cumulative_total_modifications": float(row.get("average_cumulative_total_modifications", 0.0) or 0.0),
                    "sem_cumulative_total_modifications": float(row.get("sem_cumulative_total_modifications", 0.0) or 0.0),
                }
            )
    rows.sort(key=lambda row: (str(row["dataset_name"]), float(row["normalized_position"])))
    return rows


def _prepare_matplotlib():
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "saadi_matplotlib"),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return matplotlib, plt


def plot_comparative_cumulative_figures(
    *,
    dataset_curves: list[dict[str, Any]],
    output_dir: str | Path,
    formats: list[str],
    dpi: int,
    use_sem_bands: bool,
) -> list[dict[str, Any]]:
    _, plt = _prepare_matplotlib()
    artifact_root = Path(output_dir).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    colors = plt.get_cmap("tab10").colors
    figure_records: list[dict[str, Any]] = []

    for spec_index, spec in enumerate(_METRIC_SPECS):
        metric_key = str(spec["metric_key"])
        sem_key = str(spec["sem_key"])
        slug = str(spec["slug"])
        figure_index = int(spec["figure_index"])
        title = str(spec["title"])
        y_label = str(spec["y_label"])

        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        for dataset_index, dataset_entry in enumerate(dataset_curves):
            curve_rows = dataset_entry.get("curve_rows", []) or []
            if not curve_rows:
                continue
            xs = [float(row.get("normalized_position", 0.0) or 0.0) for row in curve_rows]
            ys = [float(row.get(metric_key, 0.0) or 0.0) for row in curve_rows]
            sems = [float(row.get(sem_key, 0.0) or 0.0) for row in curve_rows]
            color = colors[dataset_index % len(colors)]
            label = str(dataset_entry.get("dataset_label", dataset_entry.get("dataset_name", "")) or "")

            ax.plot(xs, ys, label=label, color=color, linewidth=2.2)
            if use_sem_bands:
                lower = [max(0.0, y - s) for y, s in zip(ys, sems)]
                upper = [y + s for y, s in zip(ys, sems)]
                ax.fill_between(xs, lower, upper, color=color, alpha=0.14, linewidth=0.0)

        ax.set_title(title)
        ax.set_xlabel("Normalized document position")
        ax.set_ylabel(y_label)
        ax.set_xlim(0.0, 1.0)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, loc="best")
        fig.tight_layout()

        output_paths: dict[str, str] = {}
        for fmt in formats:
            output_path = artifact_root / f"figure_{figure_index}_{slug}.{fmt}"
            fig.savefig(output_path, dpi=dpi if fmt == "png" else None, bbox_inches="tight")
            output_paths[fmt] = str(output_path)
        plt.close(fig)

        figure_records.append(
            {
                "figure_index": figure_index,
                "metric_key": metric_key,
                "title": title,
                "y_label": y_label,
                "slug": slug,
                "formats": list(formats),
                "paths": output_paths,
            }
        )

    return figure_records


def summarize_comparative_plotting_run(
    dataset_curves: list[dict[str, Any]],
    figure_records: list[dict[str, Any]],
    *,
    formats: list[str],
    use_sem_bands: bool,
    curve_values_jsonl: str | Path,
) -> dict[str, Any]:
    return {
        "main_phase": MAIN_PHASE_NAME,
        "sub_phase_id": SUB_PHASE_ID,
        "sub_phase_name": SUB_PHASE_NAME,
        "source_sub_phase_id": SOURCE_SUB_PHASE_ID,
        "datasets_included": [
            {
                "dataset_name": str(entry.get("dataset_name", "")),
                "dataset_label": str(entry.get("dataset_label", "")),
                "normalized_curve_jsonl": str(entry.get("normalized_curve_jsonl", "")),
            }
            for entry in dataset_curves
        ],
        "number_of_datasets": int(len(dataset_curves)),
        "number_of_figures": int(len(figure_records)),
        "output_formats": list(formats),
        "use_sem_bands": bool(use_sem_bands),
        "curve_values_jsonl": str(Path(curve_values_jsonl).resolve()),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return str(path)


def run_comparative_plotting_across_datasets_eval(
    *,
    dataset_entries: list[dict[str, Any]],
    config_base_dir: Path | None = None,
    output_dir: str | Path,
    run_name: str = "default",
    default_source_run_name: str = "default",
    selected_dataset_names: list[str] | None = None,
    formats: list[str] | None = None,
    dpi: int = 200,
    use_sem_bands: bool = True,
) -> dict[str, Any]:
    artifact_root = Path(output_dir).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)

    normalized_formats = _normalize_output_formats(formats)
    normalized_dpi = _normalize_positive_int(dpi, default=200, field_name="plotting.dpi")

    dataset_curves = load_dataset_curve_entries(
        dataset_entries=dataset_entries,
        config_base_dir=config_base_dir,
        default_source_run_name=default_source_run_name,
        selected_dataset_names=selected_dataset_names,
    )
    if not dataset_curves:
        raise ValueError("No dataset curves were selected for comparative plotting.")

    curve_values_rows = build_comparative_curve_export_rows(dataset_curves)
    curve_values_path = artifact_root / "normalized_position_curve_values.jsonl"
    write_jsonl(curve_values_path, curve_values_rows)

    figure_records = plot_comparative_cumulative_figures(
        dataset_curves=dataset_curves,
        output_dir=artifact_root,
        formats=normalized_formats,
        dpi=normalized_dpi,
        use_sem_bands=bool(use_sem_bands),
    )
    figure_index_path = artifact_root / "figure_index.json"
    _write_json(figure_index_path, {"figures": figure_records})

    summary = summarize_comparative_plotting_run(
        dataset_curves,
        figure_records,
        formats=normalized_formats,
        use_sem_bands=bool(use_sem_bands),
        curve_values_jsonl=curve_values_path,
    )
    summary["run_name"] = str(run_name or "default")
    summary["output_dir"] = str(artifact_root)
    summary_path = artifact_root / "run_summary.json"
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
            "plotting": {
                "formats": list(normalized_formats),
                "dpi": int(normalized_dpi),
                "use_sem_bands": bool(use_sem_bands),
            },
        },
        artifacts={
            "normalized_position_curve_values_jsonl": str(curve_values_path),
            "figure_index_json": str(figure_index_path),
            "run_summary_json": str(summary_path),
            **{
                f"figure_{record['figure_index']}_{fmt}": str(path)
                for record in figure_records
                for fmt, path in (record.get("paths") or {}).items()
            },
        },
        metadata={
            "number_of_datasets": len(dataset_curves),
            "number_of_figures": len(figure_records),
        },
        cwd=str(Path.cwd()),
    )
    manifest_path = artifact_root / "run_manifest.json"
    write_manifest(manifest_path, manifest)

    return {
        "main_phase": MAIN_PHASE_NAME,
        "sub_phase_id": SUB_PHASE_ID,
        "sub_phase_name": SUB_PHASE_NAME,
        "summary": summary,
        "artifacts": {
            "artifact_root": artifact_root,
            "normalized_position_curve_values_jsonl": curve_values_path,
            "figure_index_json": figure_index_path,
            "run_summary_json": summary_path,
            "run_manifest_json": manifest_path,
            **{
                f"figure_{record['figure_index']}_{fmt}": Path(path)
                for record in figure_records
                for fmt, path in (record.get("paths") or {}).items()
            },
        },
    }
