"""Faithfulness evaluation for hierarchical summaries built over updated chunks."""

from __future__ import annotations

import json
import random
from pathlib import Path
from statistics import mean
from typing import Any

from ....core.prompts import json_schema_string, prompt_hash
from ....io.hierarchy_cache import _resolve_cache_file_path
from ....io.jsonl import read_jsonl, write_jsonl
from ....io.manifests import build_manifest, write_manifest
from ....llm.factory import build_backend_from_config
from ..common import (
    infer_dataset_label,
    infer_hierarchy_branching_factor,
    infer_model_label,
    resolve_updated_chunks_output_dir,
)
from ..judging import (
    DEFAULT_JUDGE_STOP,
    build_chat_prompt,
    extract_json_object_from_text,
    generate_outputs,
    normalize_boolish,
    normalize_score_1_to_5,
)

MAIN_PHASE_NAME = "Faithfulness"
SUB_PHASE_ID = "1_4_Hierarchical_Summary_Faithfulness"
SUB_PHASE_NAME = "Hierarchical Summary Faithfulness"
PROMPT_VERSION = "v1"

JUDGE_RESPONSE_SCHEMA = {
    "is_faithful": True,
    "has_hallucination": False,
    "has_distortion": False,
    "score": 5,
    "reason": "The summary is fully supported by the provided source inputs and does not introduce unsupported claims.",
    "summary_quality_label": "faithful",
}

JUDGE_SYSTEM_PROMPT = """You are a strict research evaluator for hierarchical-summary faithfulness.

Task: judge whether one generated hierarchical summary is faithful to its source inputs.

Evaluate:
- factual consistency with the source inputs
- whether the summary introduces unsupported claims or hallucinations
- whether important content is distorted, conflated, or misstated
- whether severe omission should reduce the score, when it makes the summary misleading

Important rules:
- Judge the generated summary only against the provided source inputs.
- Do not reward style, fluency, or compression by itself.
- A summary can be concise and still faithful.
- If the summary adds claims not grounded in the inputs, mark `has_hallucination=true`.
- If the summary misstates relationships, scope, chronology, or attribution, mark `has_distortion=true`.
- If either hallucination or distortion is serious, `is_faithful` should usually be false.
- Score rubric:
  1 = clearly unfaithful, many unsupported or distorted claims
  2 = mostly unfaithful
  3 = mixed, partially faithful but with important issues
  4 = mostly faithful with minor issues
  5 = fully faithful and well-supported

Return JSON only, matching this schema:
"""


def _build_item_id(dataset_name: str, doc_id: str, level_index: int, step: int) -> str:
    return f"{dataset_name}:{doc_id}:H{level_index}:{step}"


def _read_index_json(index_path: str | Path) -> dict[str, Any]:
    path = Path(index_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _read_level_rows(path: Path) -> dict[int, str]:
    rows = read_jsonl(path)
    out: dict[int, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            step = int(row.get("step"))
        except Exception:
            continue
        chunk = row.get("chunk")
        if chunk is None:
            continue
        out[int(step)] = str(chunk)
    return out


def _read_step_metadata(path: Path) -> tuple[dict[int, str], str]:
    rows = read_jsonl(path)
    chunk_ids: dict[int, str] = {}
    strategy_name = ""
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            step = int(row.get("step", row.get("chunk_index")))
        except Exception:
            continue
        doc_id = str(row.get("doc_id", "") or "")
        chunk_ids[int(step)] = str(row.get("chunk_id", f"{doc_id}:{step}") or f"{doc_id}:{step}")
        if not strategy_name:
            strategy_name = str(row.get("strategy_name", "") or "")
    return chunk_ids, strategy_name


def _summary_level_sort_key(level_label: str) -> tuple[int, str]:
    try:
        if str(level_label).startswith("H"):
            return (int(str(level_label)[1:]), str(level_label))
    except Exception:
        pass
    return (10_000, str(level_label))


def _normalize_quality_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def build_hierarchical_summary_items_from_cache_index(
    index_path: str | Path,
    *,
    dataset_name: str,
    branching_factor: int | None = None,
    hierarchy_k: int | None = None,
) -> list[dict[str, Any]]:
    index = _read_index_json(index_path)
    docs = index.get("docs", []) or []
    if not isinstance(docs, list):
        return []

    raw_b = branching_factor if branching_factor is not None else hierarchy_k
    if raw_b is None:
        raw_b = index.get(
            "branching_factor",
            index.get("hierarchy_branching_factor", index.get("hierarchy_k", index.get("k"))),
        )
    try:
        b = int(raw_b)
    except Exception:
        b = None
    if b is None or b <= 1:
        return []

    index_base_dir = Path(index_path).resolve().parent
    strategy_fallback = str(index.get("strategy_name", "") or "")
    items: list[dict[str, Any]] = []

    for doc in docs:
        if not isinstance(doc, dict):
            continue
        doc_id = str(doc.get("doc_id", "") or "")
        file_map = doc.get("files", {}) or {}
        if not doc_id or not isinstance(file_map, dict):
            continue

        steps_index_ref = file_map.get("steps_index")
        l0_ref = file_map.get("L_0")
        if not steps_index_ref or not l0_ref:
            continue

        steps_index_path = _resolve_cache_file_path(
            steps_index_ref, base_dir=index_base_dir
        )
        l0_path = _resolve_cache_file_path(l0_ref, base_dir=index_base_dir)
        if not steps_index_path.exists() or not l0_path.exists():
            continue

        chunk_ids_by_step, strategy_name = _read_step_metadata(steps_index_path)
        texts_by_level: dict[int, dict[int, str]] = {0: _read_level_rows(l0_path)}
        if not texts_by_level[0]:
            continue

        for key, path_value in file_map.items():
            if not isinstance(key, str) or not key.startswith("L_") or key == "L_0":
                continue
            try:
                level_index = int(key.split("_", 1)[1])
            except Exception:
                continue
            if level_index <= 0:
                continue
            level_path = _resolve_cache_file_path(
                path_value, base_dir=index_base_dir
            )
            if not level_path.exists():
                continue
            by_step = _read_level_rows(level_path)
            if by_step:
                texts_by_level[level_index] = by_step

        for level_index in sorted(level for level in texts_by_level if level >= 1):
            summary_by_step = texts_by_level.get(level_index, {})
            lower_by_step = texts_by_level.get(level_index - 1, {})
            if not summary_by_step or not lower_by_step:
                continue
            for step, summary_text in sorted(summary_by_step.items()):
                lower_steps = [candidate for candidate in sorted(lower_by_step) if candidate <= step]
                if len(lower_steps) < b:
                    continue
                source_steps = lower_steps[-b:]
                source_inputs: list[dict[str, Any]] = []
                for source_step in source_steps:
                    source_text = str(lower_by_step[source_step] or "")
                    if level_index == 1:
                        chunk_id = chunk_ids_by_step.get(source_step, f"{doc_id}:{source_step}")
                        source_inputs.append(
                            {
                                "source_id": chunk_id,
                                "source_type": "chunk",
                                "source_level": "L0",
                                "step": int(source_step),
                                "text": source_text,
                            }
                        )
                    else:
                        source_inputs.append(
                            {
                                "source_id": f"{doc_id}:H{level_index - 1}:{source_step}",
                                "source_type": "summary",
                                "source_level": f"H{level_index - 1}",
                                "step": int(source_step),
                                "text": source_text,
                            }
                        )
                items.append(
                    {
                        "item_id": _build_item_id(dataset_name, doc_id, level_index, int(step)),
                        "main_phase": MAIN_PHASE_NAME,
                        "sub_phase_id": SUB_PHASE_ID,
                        "sub_phase_name": SUB_PHASE_NAME,
                        "dataset_name": dataset_name,
                        "document_id": doc_id,
                        "strategy_name": strategy_name or strategy_fallback,
                        "summary_level": f"H{level_index}",
                        "summary_level_index": int(level_index),
                        "summary_cycle": int(b**level_index),
                        "summary_step": int(step),
                        "summary_id": f"{doc_id}:H{level_index}:{step}",
                        "source_input_kind": "chunks" if level_index == 1 else "lower_level_summaries",
                        "source_level": "L0" if level_index == 1 else f"H{level_index - 1}",
                        "source_input_count": len(source_inputs),
                        "source_inputs": source_inputs,
                        "generated_summary_text": str(summary_text or ""),
                    }
                )

    items.sort(
        key=lambda row: (
            str(row.get("document_id", "")),
            int(row.get("summary_level_index", -1)),
            int(row.get("summary_step", -1)),
            str(row.get("summary_id", "")),
        )
    )
    return items


def sample_hierarchical_summary_items(
    items: list[dict[str, Any]],
    *,
    sample_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    if sample_size <= 0 or not items:
        return []
    rng = random.Random(int(seed))
    pool = list(items)
    if len(pool) <= sample_size:
        selected = pool
    else:
        selected = rng.sample(pool, int(sample_size))
    for rank, item in enumerate(selected, start=1):
        item["sample_rank"] = int(rank)
        item["sampling_seed"] = int(seed)
    return selected


def build_hierarchical_summary_judge_prompt(item: dict[str, Any]) -> str:
    user_payload = {
        "dataset_name": item.get("dataset_name"),
        "document_id": item.get("document_id"),
        "summary_level": item.get("summary_level"),
        "summary_id": item.get("summary_id"),
        "source_level": item.get("source_level"),
        "source_input_kind": item.get("source_input_kind"),
        "source_inputs": item.get("source_inputs", []),
        "generated_summary_text": item.get("generated_summary_text"),
    }
    return (
        "Evaluate whether the generated hierarchical summary is faithful to its source inputs.\n\n"
        "Return JSON only.\n\n"
        f"SCHEMA:\n{json_schema_string(JUDGE_RESPONSE_SCHEMA)}\n\n"
        f"EVALUATION_ITEM_JSON:\n{json.dumps(user_payload, ensure_ascii=False, indent=2)}\n\n"
        "Focus on factual consistency, hallucination, and distortion. Mention omission only if it makes the summary misleading."
    )


def parse_hierarchical_summary_judgment(raw_text: str) -> dict[str, Any]:
    parsed, parser_status = extract_json_object_from_text(raw_text)
    if parsed is None:
        return {
            "parser_status": "failed",
            "is_faithful": None,
            "has_hallucination": None,
            "has_distortion": None,
            "score": None,
            "reason": "",
            "summary_quality_label": None,
        }

    is_faithful = normalize_boolish(parsed.get("is_faithful"))
    has_hallucination = normalize_boolish(parsed.get("has_hallucination"))
    has_distortion = normalize_boolish(parsed.get("has_distortion"))
    score = normalize_score_1_to_5(parsed.get("score"))
    reason = str(parsed.get("reason", "") or "").strip()
    summary_quality_label = _normalize_quality_label(parsed.get("summary_quality_label"))
    if summary_quality_label is None and is_faithful is True and has_hallucination is False and has_distortion is False:
        summary_quality_label = "faithful"

    return {
        "parser_status": parser_status,
        "is_faithful": is_faithful,
        "has_hallucination": has_hallucination,
        "has_distortion": has_distortion,
        "score": score,
        "reason": reason,
        "summary_quality_label": summary_quality_label,
    }


def _percent_true(rows: list[dict[str, Any]], key: str) -> float:
    eligible = [row for row in rows if row.get(key) is not None]
    if not eligible:
        return 0.0
    hits = sum(1 for row in eligible if bool(row.get(key)))
    return 100.0 * hits / len(eligible)


def _level_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        level = str(item.get("summary_level", "") or "")
        if not level:
            continue
        counts[level] = int(counts.get(level, 0)) + 1
    return counts


def aggregate_hierarchical_summary_statistics(
    *,
    dataset_name: str,
    eligible_items: list[dict[str, Any]],
    sampled_items: list[dict[str, Any]],
    parsed_rows: list[dict[str, Any]],
    seed: int,
    judge_model_label: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    valid_scores = [int(row["score"]) for row in parsed_rows if row.get("score") is not None]
    quality_label_counts: dict[str, int] = {}
    for row in parsed_rows:
        label = str(row.get("summary_quality_label", "") or "").strip()
        if not label:
            continue
        quality_label_counts[label] = int(quality_label_counts.get(label, 0)) + 1

    eligible_by_level = _level_counts(eligible_items)
    sampled_by_level = _level_counts(sampled_items)
    levels = sorted(set(eligible_by_level) | set(sampled_by_level), key=_summary_level_sort_key)
    per_level: dict[str, dict[str, Any]] = {}
    for level in levels:
        rows = [row for row in parsed_rows if str(row.get("summary_level", "") or "") == level]
        level_scores = [int(row["score"]) for row in rows if row.get("score") is not None]
        per_level[level] = {
            "summary_level": level,
            "eligible_summaries": int(eligible_by_level.get(level, 0)),
            "number_sampled": int(sampled_by_level.get(level, 0)),
            "parsed_judgments": int(len(level_scores)),
            "parse_failures": int(sum(1 for row in rows if row.get("parser_status") == "failed")),
            "percent_faithful": float(_percent_true(rows, "is_faithful")),
            "percent_with_hallucination": float(_percent_true(rows, "has_hallucination")),
            "percent_with_distortion": float(_percent_true(rows, "has_distortion")),
            "average_score": float(mean(level_scores) if level_scores else 0.0),
        }

    overall = {
        "main_phase": MAIN_PHASE_NAME,
        "sub_phase_id": SUB_PHASE_ID,
        "sub_phase_name": SUB_PHASE_NAME,
        "dataset_name": dataset_name,
        "prompt_version": PROMPT_VERSION,
        "sampling_seed": int(seed),
        "judge_model": judge_model_label,
        "eligible_summaries": int(len(eligible_items)),
        "number_sampled": int(len(sampled_items)),
        "parsed_judgments": int(len(valid_scores)),
        "parse_failures": int(sum(1 for row in parsed_rows if row.get("parser_status") == "failed")),
        "percent_faithful": float(_percent_true(parsed_rows, "is_faithful")),
        "percent_with_hallucination": float(_percent_true(parsed_rows, "has_hallucination")),
        "percent_with_distortion": float(_percent_true(parsed_rows, "has_distortion")),
        "average_score": float(mean(valid_scores) if valid_scores else 0.0),
        "summary_quality_label_counts": dict(sorted(quality_label_counts.items())),
        "summary_levels_observed": levels,
    }
    return overall, per_level


def _summary_markdown(summary: dict[str, Any], per_level: dict[str, dict[str, Any]]) -> str:
    lines = [
        f"# {SUB_PHASE_ID}",
        "",
        "| Field | Value |",
        "|---|---:|",
        f"| Eligible summaries | {summary.get('eligible_summaries', 0)} |",
        f"| Number sampled | {summary.get('number_sampled', 0)} |",
        f"| Parsed judgments | {summary.get('parsed_judgments', 0)} |",
        f"| Parse failures | {summary.get('parse_failures', 0)} |",
        f"| Percent faithful | {summary.get('percent_faithful', 0.0):.2f} |",
        f"| Percent with hallucination | {summary.get('percent_with_hallucination', 0.0):.2f} |",
        f"| Percent with distortion | {summary.get('percent_with_distortion', 0.0):.2f} |",
        f"| Average score | {summary.get('average_score', 0.0):.3f} |",
    ]
    if per_level:
        lines.extend(
            [
                "",
                "## Per Summary Level",
                "",
                "| Level | Eligible | Sampled | Faithful % | Hallucination % | Distortion % | Avg score |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for level in sorted(per_level, key=_summary_level_sort_key):
            row = per_level[level]
            lines.append(
                f"| {level} | {row.get('eligible_summaries', 0)} | {row.get('number_sampled', 0)} | "
                f"{row.get('percent_faithful', 0.0):.2f} | {row.get('percent_with_hallucination', 0.0):.2f} | "
                f"{row.get('percent_with_distortion', 0.0):.2f} | {row.get('average_score', 0.0):.3f} |"
            )
    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return str(path)


def run_hierarchical_summary_faithfulness_eval(
    *,
    experiment_config: dict[str, Any],
    judge_model_config: dict[str, Any],
    llm_cache_index_json: str,
    sample_size: int = 100,
    seed: int = 13,
    output_dir: str | Path | None = None,
    batch_size: int = 16,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    stop: list[str] | None = None,
) -> dict[str, Any]:
    dataset_name = infer_dataset_label(experiment_config)
    judge_model_label = infer_model_label(judge_model_config)
    branching_factor = infer_hierarchy_branching_factor(experiment_config)
    eligible_items = build_hierarchical_summary_items_from_cache_index(
        llm_cache_index_json,
        dataset_name=dataset_name,
        branching_factor=branching_factor,
    )
    sampled_items = sample_hierarchical_summary_items(
        eligible_items,
        sample_size=int(sample_size),
        seed=int(seed),
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

    sampled_path = artifact_root / "sampled_items.jsonl"
    raw_outputs_path = artifact_root / "raw_judge_outputs.jsonl"
    parsed_path = artifact_root / "parsed_judgments.jsonl"
    aggregate_path = artifact_root / "aggregate_stats.json"
    aggregate_by_level_path = artifact_root / "aggregate_stats_by_level.json"
    summary_md_path = artifact_root / "aggregate_stats.md"
    manifest_path = artifact_root / "run_manifest.json"

    raw_rows: list[dict[str, Any]] = []
    parsed_rows: list[dict[str, Any]] = []

    if sampled_items:
        backend = build_backend_from_config(judge_model_config)
        prompts = [
            build_chat_prompt(
                backend=backend,
                system_prompt=f"{JUDGE_SYSTEM_PROMPT}\n{json_schema_string(JUDGE_RESPONSE_SCHEMA)}",
                user_prompt=build_hierarchical_summary_judge_prompt(item),
            )
            for item in sampled_items
        ]
        outputs = generate_outputs(
            backend,
            prompts,
            batch_size=int(batch_size),
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            stop=list(stop or DEFAULT_JUDGE_STOP),
        )
        for item, prompt_text, raw_output in zip(sampled_items, prompts, outputs):
            prompt_sha = prompt_hash(prompt_text)
            raw_rows.append(
                {
                    "item_id": item["item_id"],
                    "dataset_name": item["dataset_name"],
                    "document_id": item["document_id"],
                    "summary_level": item["summary_level"],
                    "summary_id": item["summary_id"],
                    "summary_step": item["summary_step"],
                    "judge_model": judge_model_label,
                    "judge_prompt_hash": prompt_sha,
                    "raw_judge_output": str(raw_output or ""),
                }
            )
            parsed = parse_hierarchical_summary_judgment(str(raw_output or ""))
            parsed_rows.append(
                {
                    "item_id": item["item_id"],
                    "dataset_name": item["dataset_name"],
                    "document_id": item["document_id"],
                    "summary_level": item["summary_level"],
                    "summary_id": item["summary_id"],
                    "summary_step": item["summary_step"],
                    "judge_model": judge_model_label,
                    "judge_prompt_hash": prompt_sha,
                    **parsed,
                }
            )

    summary, per_level = aggregate_hierarchical_summary_statistics(
        dataset_name=dataset_name,
        eligible_items=eligible_items,
        sampled_items=sampled_items,
        parsed_rows=parsed_rows,
        seed=int(seed),
        judge_model_label=judge_model_label,
    )
    summary["output_dir"] = str(artifact_root)
    summary["llm_cache_index_json"] = str(Path(llm_cache_index_json).resolve())
    summary["per_summary_level"] = per_level

    write_jsonl(sampled_path, sampled_items)
    write_jsonl(raw_outputs_path, raw_rows)
    write_jsonl(parsed_path, parsed_rows)
    _write_json(aggregate_path, summary)
    _write_json(aggregate_by_level_path, per_level)
    summary_md_path.write_text(_summary_markdown(summary, per_level), encoding="utf-8")

    manifest = build_manifest(
        config={
            "main_phase": MAIN_PHASE_NAME,
            "sub_phase_id": SUB_PHASE_ID,
            "sub_phase_name": SUB_PHASE_NAME,
            "experiment_config": experiment_config,
            "judge_model_config": judge_model_config,
            "sampling": {"sample_size": int(sample_size), "seed": int(seed)},
            "judging": {
                "batch_size": int(batch_size),
                "max_tokens": int(max_tokens),
                "temperature": float(temperature),
                "stop": list(stop or DEFAULT_JUDGE_STOP),
            },
        },
        artifacts={
            "sampled_items_jsonl": str(sampled_path),
            "raw_judge_outputs_jsonl": str(raw_outputs_path),
            "parsed_judgments_jsonl": str(parsed_path),
            "aggregate_stats_json": str(aggregate_path),
            "aggregate_stats_by_level_json": str(aggregate_by_level_path),
            "aggregate_stats_markdown": str(summary_md_path),
        },
        metadata={
            "dataset_name": dataset_name,
            "judge_model": judge_model_label,
            "eligible_summaries": len(eligible_items),
            "number_sampled": len(sampled_items),
            "prompt_version": PROMPT_VERSION,
        },
        cwd=str(Path.cwd()),
    )
    write_manifest(manifest_path, manifest)

    return {
        "main_phase": MAIN_PHASE_NAME,
        "sub_phase_id": SUB_PHASE_ID,
        "sub_phase_name": SUB_PHASE_NAME,
        "dataset_name": dataset_name,
        "judge_model": judge_model_label,
        "eligible_items": len(eligible_items),
        "sampled_items": len(sampled_items),
        "summary": summary,
        "artifacts": {
            "artifact_root": artifact_root,
            "sampled_items_jsonl": sampled_path,
            "raw_judge_outputs_jsonl": raw_outputs_path,
            "parsed_judgments_jsonl": parsed_path,
            "aggregate_stats_json": aggregate_path,
            "aggregate_stats_by_level_json": aggregate_by_level_path,
            "aggregate_stats_markdown": summary_md_path,
            "run_manifest_json": manifest_path,
        },
    }
