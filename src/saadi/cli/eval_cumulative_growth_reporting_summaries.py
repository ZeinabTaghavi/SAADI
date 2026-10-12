"""CLI command for cumulative-growth reporting summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..config.loader import load_config
from ..evaluation.updated_chunks.common import resolve_path_from_base
from ..evaluation.updated_chunks.cumulative_growth.reporting_summaries import (
    SUB_PHASE_ID,
    run_methodology_summary_and_reporting_outputs_eval,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build reporting-friendly summary tables for cumulative growth analysis."
    )
    parser.add_argument("--config", required=False, help="Evaluation config YAML/JSON")
    parser.add_argument("--output_dir", required=False, help="Optional explicit artifact output directory")
    parser.add_argument("--datasets", required=False, help="Optional comma-separated dataset names to include")
    parser.add_argument("--include_plots", default=None, help="Optional bool-like override for including comparative plots")
    parser.add_argument("--formats", required=False, help="Optional comma-separated output formats, e.g. png,pdf")
    parser.add_argument("--dpi", type=int, default=None, help="Optional figure DPI")
    parser.add_argument("--use_sem_bands", default=None, help="Optional bool-like override for SEM confidence bands")
    return parser


def _pick(args: argparse.Namespace, config: dict[str, Any], key: str, default: Any = None) -> Any:
    value = getattr(args, key, None)
    if value is not None:
        return value
    if key in config and config.get(key) is not None:
        return config.get(key)
    return default


def _parse_dataset_filter(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def _pick_nested(args: argparse.Namespace, config: dict[str, Any], *, section: str, key: str, default: Any = None) -> Any:
    value = getattr(args, key, None)
    if value is not None:
        return value
    raw = config.get(section, {}) or {}
    if isinstance(raw, dict) and raw.get(key) is not None:
        return raw.get(key)
    return default


def _parse_formats(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    if not text:
        return None
    return [item.strip() for item in text.split(",") if item.strip()]


def _parse_boolish(value: Any, *, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected a bool-like value (got: {value!r}).")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    eval_cfg: dict[str, Any] = {}
    eval_base_dir: Path | None = None
    if args.config:
        eval_config_path = Path(args.config).resolve()
        eval_base_dir = eval_config_path.parent
        eval_cfg = load_config(eval_config_path)

    dataset_entries = eval_cfg.get("datasets", []) or []
    if not isinstance(dataset_entries, list) or not dataset_entries:
        raise ValueError("Reporting-summary config must provide a non-empty 'datasets' list.")

    run_name = str(eval_cfg.get("run_name", "default") or "default")
    output_root = resolve_path_from_base(
        _pick(args, eval_cfg, "output_root", "outputs"),
        base_dir=eval_base_dir,
    )
    if not output_root:
        output_root = str((Path.cwd() / "outputs").resolve())
    output_dir = resolve_path_from_base(_pick(args, eval_cfg, "output_dir"), base_dir=eval_base_dir)
    if not output_dir:
        output_dir = str(
            Path(output_root).resolve()
            / "evaluation"
            / "updated_chunks"
            / str(eval_cfg.get("sub_phase_id", SUB_PHASE_ID) or SUB_PHASE_ID)
            / run_name
        )

    out = run_methodology_summary_and_reporting_outputs_eval(
        dataset_entries=dataset_entries,
        config_base_dir=eval_base_dir,
        output_dir=output_dir,
        run_name=run_name,
        default_source_run_name=str(eval_cfg.get("source_run_name", "default") or "default"),
        selected_dataset_names=_parse_dataset_filter(args.datasets),
        milestones=eval_cfg.get("milestones"),
        include_plots=_parse_boolish(
            _pick_nested(args, eval_cfg, section="plotting", key="include_plots", default=True),
            default=True,
        ),
        plot_formats=_parse_formats(_pick_nested(args, eval_cfg, section="plotting", key="formats", default=None)),
        plot_dpi=int(_pick_nested(args, eval_cfg, section="plotting", key="dpi", default=200) or 200),
        plot_use_sem_bands=_parse_boolish(
            _pick_nested(args, eval_cfg, section="plotting", key="use_sem_bands", default=True),
            default=True,
        ),
    )

    payload = {
        "main_phase": out["main_phase"],
        "sub_phase_id": out["sub_phase_id"],
        "sub_phase_name": out["sub_phase_name"],
        "summary": out["summary"],
        "artifacts": {key: str(value) for key, value in (out.get("artifacts") or {}).items()},
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
