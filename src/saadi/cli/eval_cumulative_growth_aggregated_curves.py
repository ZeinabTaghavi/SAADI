"""CLI command for dataset-level aggregated cumulative growth curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..config.loader import load_config
from ..evaluation.updated_chunks.common import (
    resolve_path_from_base,
    resolve_updated_chunks_output_dir,
)
from ..evaluation.updated_chunks.cumulative_growth.aggregated_curves import (
    DEFAULT_NORMALIZED_GRID_SIZE,
    SOURCE_SUB_PHASE_ID,
    SUB_PHASE_ID,
    resolve_source_per_chunk_growth_path,
    run_dataset_level_aggregated_cumulative_curves_eval,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate per-document cumulative growth logs into dataset-level curves."
    )
    parser.add_argument("--config", required=False, help="Evaluation config YAML/JSON")
    parser.add_argument("--experiment_config", required=False, help="Experiment config YAML/JSON for the cached dataset run")
    parser.add_argument("--per_chunk_cumulative_growth_jsonl", required=False, help="Optional explicit path to the 3_1 per-chunk cumulative growth JSONL artifact")
    parser.add_argument("--output_dir", required=False, help="Optional explicit artifact output directory")
    parser.add_argument("--normalized_grid_size", type=int, default=None, help="Number of grid points for normalized-position aggregation")
    return parser


def _pick(args: argparse.Namespace, config: dict[str, Any], key: str, default: Any = None) -> Any:
    value = getattr(args, key, None)
    if value is not None:
        return value
    if key in config and config.get(key) is not None:
        return config.get(key)
    return default


def _pick_nested(args: argparse.Namespace, config: dict[str, Any], *, section: str, key: str, default: Any = None) -> Any:
    value = getattr(args, key, None)
    if value is not None:
        return value
    raw = config.get(section, {}) or {}
    if isinstance(raw, dict) and raw.get(key) is not None:
        return raw.get(key)
    return default


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    eval_cfg: dict[str, Any] = {}
    eval_base_dir: Path | None = None
    if args.config:
        eval_config_path = Path(args.config).resolve()
        eval_base_dir = eval_config_path.parent
        eval_cfg = load_config(eval_config_path)

    experiment_config_path = resolve_path_from_base(
        _pick(args, eval_cfg, "experiment_config"),
        base_dir=eval_base_dir,
    )
    if not experiment_config_path:
        raise ValueError("Provide --experiment_config directly or through the evaluation config.")
    experiment_config_path = str(Path(experiment_config_path).resolve())
    experiment_base_dir = Path(experiment_config_path).resolve().parent
    experiment_cfg = load_config(experiment_config_path)

    experiment_cfg = dict(experiment_cfg)
    resolved_output_dir = resolve_path_from_base(experiment_cfg.get("output_dir"), base_dir=experiment_base_dir)
    if resolved_output_dir:
        experiment_cfg["output_dir"] = resolved_output_dir

    per_chunk_cumulative_growth_jsonl = resolve_path_from_base(
        _pick(args, eval_cfg, "per_chunk_cumulative_growth_jsonl"),
        base_dir=eval_base_dir,
    )
    if not per_chunk_cumulative_growth_jsonl:
        source_run_name = str(_pick(args, eval_cfg, "source_run_name", "default") or "default")
        auto_path = resolve_source_per_chunk_growth_path(
            experiment_cfg,
            source_run_name=source_run_name,
        )
        if auto_path.exists():
            per_chunk_cumulative_growth_jsonl = str(auto_path)
    if not per_chunk_cumulative_growth_jsonl:
        raise ValueError(
            f"Could not resolve {SOURCE_SUB_PHASE_ID} per-chunk logs. Provide --per_chunk_cumulative_growth_jsonl or run {SOURCE_SUB_PHASE_ID} first."
        )

    run_name = str(eval_cfg.get("run_name", "default") or "default")
    output_dir = resolve_path_from_base(_pick(args, eval_cfg, "output_dir"), base_dir=eval_base_dir)
    if not output_dir:
        output_dir = str(
            resolve_updated_chunks_output_dir(
                experiment_cfg,
                subphase_id=str(eval_cfg.get("sub_phase_id", SUB_PHASE_ID) or SUB_PHASE_ID),
                run_name=run_name,
            )
        )

    out = run_dataset_level_aggregated_cumulative_curves_eval(
        experiment_config=experiment_cfg,
        per_chunk_cumulative_growth_jsonl=str(Path(per_chunk_cumulative_growth_jsonl).resolve()),
        normalized_grid_size=int(
            _pick_nested(
                args,
                eval_cfg,
                section="aggregation",
                key="normalized_grid_size",
                default=DEFAULT_NORMALIZED_GRID_SIZE,
            )
            or DEFAULT_NORMALIZED_GRID_SIZE
        ),
        output_dir=output_dir,
    )

    payload = {
        "main_phase": out["main_phase"],
        "sub_phase_id": out["sub_phase_id"],
        "sub_phase_name": out["sub_phase_name"],
        "dataset_name": out["dataset_name"],
        "summary": out["summary"],
        "artifacts": {key: str(value) for key, value in (out.get("artifacts") or {}).items()},
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
