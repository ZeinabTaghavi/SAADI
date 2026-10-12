"""CLI command for faithfulness evaluation of linking sentences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..config.loader import load_config
from ..evaluation.updated_chunks.common import (
    resolve_cache_index_from_experiment_config,
    resolve_path_from_base,
    resolve_updated_chunks_output_dir,
)
from ..evaluation.updated_chunks.faithfulness.linking_sentence import (
    SUB_PHASE_ID,
    run_linking_sentence_correctness_eval,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate faithfulness of linking sentences in updated chunks."
    )
    parser.add_argument("--config", required=False, help="Evaluation config YAML/JSON")
    parser.add_argument("--experiment_config", required=False, help="Experiment config YAML/JSON for the cached dataset run")
    parser.add_argument("--judge_model_config", required=False, help="Judge-model config YAML/JSON")
    parser.add_argument("--llm_cache_index_json", required=False, help="Optional explicit llm_cache index.json path")
    parser.add_argument("--output_dir", required=False, help="Optional explicit artifact output directory")
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
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

    judge_model_config_path = resolve_path_from_base(
        _pick(args, eval_cfg, "judge_model_config", "configs/models/qwen_vllm.yaml"),
        base_dir=eval_base_dir,
    )
    if not judge_model_config_path:
        raise ValueError("Could not resolve judge model config.")
    judge_model_cfg = load_config(judge_model_config_path)

    llm_cache_index_json = resolve_path_from_base(
        _pick(args, eval_cfg, "llm_cache_index_json"),
        base_dir=eval_base_dir,
    )
    if not llm_cache_index_json:
        llm_cache_index_json = resolve_cache_index_from_experiment_config(experiment_cfg)
    if not llm_cache_index_json:
        raise ValueError(
            "Could not resolve llm_cache index. Provide --llm_cache_index_json or ensure the experiment config contains output_dir/dataset/ingest/model."
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

    out = run_linking_sentence_correctness_eval(
        experiment_config=experiment_cfg,
        judge_model_config=judge_model_cfg,
        llm_cache_index_json=str(llm_cache_index_json),
        output_dir=output_dir,
        sample_size=int(_pick_nested(args, eval_cfg, section="sampling", key="sample_size", default=100) or 100),
        seed=int(_pick_nested(args, eval_cfg, section="sampling", key="seed", default=13) or 13),
        batch_size=int(_pick_nested(args, eval_cfg, section="judging", key="batch_size", default=16) or 16),
        max_tokens=int(_pick_nested(args, eval_cfg, section="judging", key="max_tokens", default=1024) or 1024),
        temperature=float(_pick_nested(args, eval_cfg, section="judging", key="temperature", default=0.0) or 0.0),
        stop=list((_pick_nested(args, eval_cfg, section="judging", key="stop", default=None) or [])),
    )

    payload = {
        "main_phase": out["main_phase"],
        "sub_phase_id": out["sub_phase_id"],
        "sub_phase_name": out["sub_phase_name"],
        "dataset_name": out["dataset_name"],
        "judge_model": out["judge_model"],
        "eligible_items": out["eligible_items"],
        "sampled_items": out["sampled_items"],
        "summary": out["summary"],
        "artifacts": {key: str(value) for key, value in (out.get("artifacts") or {}).items()},
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
