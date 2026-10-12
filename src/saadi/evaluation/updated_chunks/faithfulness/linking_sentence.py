"""Faithfulness evaluation for added linking sentences in updated chunks."""

from __future__ import annotations

import json
import random
from pathlib import Path
from statistics import mean
from typing import Any

from ....core.prompts import json_schema_string, prompt_hash
from ....core.types import CanonicalizedChunkRecord
from ....io.hierarchy_cache import load_canonicalized_chunk_records_from_cache_index
from ....io.jsonl import write_jsonl
from ....io.manifests import build_manifest, write_manifest
from ....llm.factory import build_backend_from_config
from ..common import (
    infer_dataset_label,
    infer_linking_sentence,
    infer_model_label,
    resolve_updated_chunks_output_dir,
)
from ..judging import (
    DEFAULT_JUDGE_STOP,
    build_chat_prompt,
    extract_json_object_from_text,
    generate_outputs,
    normalize_boolish,
    normalize_problem_type,
    normalize_score_1_to_5,
)

MAIN_PHASE_NAME = "Faithfulness"
SUB_PHASE_ID = "1_2_Linking_Sentence_Correctness"
SUB_PHASE_NAME = "Linking Sentence Correctness"
PROMPT_VERSION = "v1"

JUDGE_RESPONSE_SCHEMA = {
    "is_supported": True,
    "is_faithful": True,
    "is_helpful": True,
    "is_minimal": True,
    "score": 5,
    "reason": "The linking sentence is supported by the previous and current chunks, helpful as a bridge, and minimal.",
    "problem_type": "none",
}

JUDGE_SYSTEM_PROMPT = """You are a strict research evaluator for updated-chunk faithfulness.

Task: judge only the short linking sentence added at the start of a chunk.

A linking sentence is good only if it is:
- supported by the previous chunk and current raw chunk
- faithful and non-hallucinated
- helpful as a bridge into the current chunk
- minimal rather than overly long, interpretive, or redundant

Important rules:
- Judge only the linking sentence, not the whole updated chunk.
- The current raw chunk and the immediately previous chunk are the primary evidence.
- Allowed context may help, but do not infer unsupported facts.
- If the sentence adds claims, entities, motivations, or summaries not grounded in the evidence, mark `is_faithful=false`.
- If it is accurate but unnecessary or too interpretive, `is_minimal` or `is_helpful` may be false.
- Score rubric:
  1 = clearly hallucinated or harmful
  2 = mostly unsupported or strongly over-interpretive
  3 = mixed, weakly useful, or partly grounded
  4 = supported and useful with minor excess
  5 = fully supported, helpful, and minimal

Return JSON only, matching this schema:
"""


def _hierarchy_context_before_chunk(previous_record: CanonicalizedChunkRecord | None) -> list[dict[str, Any]]:
    if previous_record is None:
        return []
    raw = (previous_record.state_snapshot or {}).get("hierarchy_summaries", {}) or {}
    if not isinstance(raw, dict):
        return []
    rows: list[dict[str, Any]] = []
    for cycle_key, by_step in raw.items():
        if not isinstance(by_step, dict) or not by_step:
            continue
        try:
            cycle_size = int(cycle_key)
        except Exception:
            try:
                cycle_size = int(str(cycle_key).strip())
            except Exception:
                continue
        latest_step: int | None = None
        latest_summary = ""
        for step_key, summary in by_step.items():
            try:
                step = int(step_key)
            except Exception:
                continue
            if latest_step is None or step > latest_step:
                latest_step = step
                latest_summary = str(summary or "")
        if latest_step is None or not latest_summary.strip():
            continue
        rows.append(
            {
                "cycle_size": int(cycle_size),
                "latest_step": int(latest_step),
                "summary": latest_summary,
            }
        )
    rows.sort(key=lambda row: (int(row["cycle_size"]), int(row["latest_step"])))
    return rows


def _previous_active_entities(previous_record: CanonicalizedChunkRecord | None) -> list[dict[str, Any]]:
    if previous_record is None:
        return []
    raw = (previous_record.state_snapshot or {}).get("active_entities", []) or []
    return [row for row in raw if isinstance(row, dict)]


def _build_item_id(dataset_name: str, record: CanonicalizedChunkRecord) -> str:
    return f"{dataset_name}:{record.doc_id}:{record.chunk_id}:{record.chunk_index}"


def build_linking_sentence_items(
    records: list[CanonicalizedChunkRecord],
    *,
    dataset_name: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[CanonicalizedChunkRecord]] = {}
    for record in sorted(records, key=lambda row: (str(row.doc_id), int(row.chunk_index), str(row.chunk_id))):
        grouped.setdefault(str(record.doc_id), []).append(record)

    items: list[dict[str, Any]] = []
    for doc_id, doc_records in grouped.items():
        previous_record: CanonicalizedChunkRecord | None = None
        for record in doc_records:
            linking_sentence, source = infer_linking_sentence(record)
            if not linking_sentence:
                previous_record = record
                continue
            items.append(
                {
                    "item_id": _build_item_id(dataset_name, record),
                    "main_phase": MAIN_PHASE_NAME,
                    "sub_phase_id": SUB_PHASE_ID,
                    "sub_phase_name": SUB_PHASE_NAME,
                    "dataset_name": dataset_name,
                    "document_id": doc_id,
                    "chunk_id": str(record.chunk_id),
                    "chunk_index": int(record.chunk_index),
                    "strategy_name": str(record.strategy_name or ""),
                    "parser_status": str(record.parser_status or "ok"),
                    "previous_chunk": (
                        str(previous_record.canonicalized_chunk_text or "")
                        if previous_record is not None
                        else ""
                    ),
                    "previous_raw_chunk": (
                        str(previous_record.raw_chunk_text or "")
                        if previous_record is not None
                        else ""
                    ),
                    "original_current_chunk": str(record.raw_chunk_text or ""),
                    "updated_chunk": str(record.canonicalized_chunk_text or ""),
                    "linking_sentence": linking_sentence,
                    "linking_sentence_source": source,
                    "linking_sentence_word_count": len([tok for tok in linking_sentence.split() if tok]),
                    "allowed_context": {
                        "previous_active_entities": _previous_active_entities(previous_record),
                        "hierarchy_context": _hierarchy_context_before_chunk(previous_record),
                    },
                }
            )
            previous_record = record
    return items


def sample_linking_sentence_items(
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


def build_linking_sentence_judge_prompt(item: dict[str, Any]) -> str:
    user_payload = {
        "dataset_name": item.get("dataset_name"),
        "document_id": item.get("document_id"),
        "chunk_id": item.get("chunk_id"),
        "chunk_index": item.get("chunk_index"),
        "previous_chunk": item.get("previous_chunk"),
        "previous_raw_chunk": item.get("previous_raw_chunk"),
        "original_current_chunk": item.get("original_current_chunk"),
        "updated_chunk": item.get("updated_chunk"),
        "linking_sentence": item.get("linking_sentence"),
        "linking_sentence_source": item.get("linking_sentence_source"),
        "allowed_context": item.get("allowed_context", {}),
    }
    return (
        "Evaluate whether the linking sentence is correct and useful.\n\n"
        "Return JSON only.\n\n"
        f"SCHEMA:\n{json_schema_string(JUDGE_RESPONSE_SCHEMA)}\n\n"
        f"EVALUATION_ITEM_JSON:\n{json.dumps(user_payload, ensure_ascii=False, indent=2)}\n\n"
        "Focus only on whether the linking sentence is grounded, helpful as a bridge, and minimal."
    )


def parse_linking_sentence_judgment(raw_text: str) -> dict[str, Any]:
    parsed, parser_status = extract_json_object_from_text(raw_text)
    if parsed is None:
        return {
            "parser_status": "failed",
            "is_supported": None,
            "is_faithful": None,
            "is_helpful": None,
            "is_minimal": None,
            "score": None,
            "reason": "",
            "problem_type": "parse_failure",
        }

    is_supported = normalize_boolish(parsed.get("is_supported"))
    is_faithful = normalize_boolish(parsed.get("is_faithful"))
    is_helpful = normalize_boolish(parsed.get("is_helpful"))
    is_minimal = normalize_boolish(parsed.get("is_minimal"))
    score = normalize_score_1_to_5(parsed.get("score"))
    reason = str(parsed.get("reason", "") or "").strip()
    problem_type = normalize_problem_type(parsed.get("problem_type"))
    if (
        problem_type is None
        and is_supported is True
        and is_faithful is True
        and is_helpful is True
        and is_minimal is True
    ):
        problem_type = "none"

    return {
        "parser_status": parser_status,
        "is_supported": is_supported,
        "is_faithful": is_faithful,
        "is_helpful": is_helpful,
        "is_minimal": is_minimal,
        "score": score,
        "reason": reason,
        "problem_type": problem_type,
    }


def _percent_true(rows: list[dict[str, Any]], key: str) -> float:
    eligible = [row for row in rows if row.get(key) is not None]
    if not eligible:
        return 0.0
    hits = sum(1 for row in eligible if bool(row.get(key)))
    return 100.0 * hits / len(eligible)


def aggregate_linking_sentence_statistics(
    *,
    dataset_name: str,
    eligible_count: int,
    sampled_count: int,
    parsed_rows: list[dict[str, Any]],
    seed: int,
    judge_model_label: str,
) -> dict[str, Any]:
    valid_scores = [int(row["score"]) for row in parsed_rows if row.get("score") is not None]
    problem_type_counts: dict[str, int] = {}
    for row in parsed_rows:
        problem_type = str(row.get("problem_type") or "").strip()
        if not problem_type:
            continue
        problem_type_counts[problem_type] = int(problem_type_counts.get(problem_type, 0)) + 1

    return {
        "main_phase": MAIN_PHASE_NAME,
        "sub_phase_id": SUB_PHASE_ID,
        "sub_phase_name": SUB_PHASE_NAME,
        "dataset_name": dataset_name,
        "prompt_version": PROMPT_VERSION,
        "sampling_seed": int(seed),
        "judge_model": judge_model_label,
        "eligible_chunks": int(eligible_count),
        "number_sampled": int(sampled_count),
        "parsed_judgments": int(len(valid_scores)),
        "parse_failures": int(sum(1 for row in parsed_rows if row.get("parser_status") == "failed")),
        "percent_supported": float(_percent_true(parsed_rows, "is_supported")),
        "percent_faithful": float(_percent_true(parsed_rows, "is_faithful")),
        "percent_helpful": float(_percent_true(parsed_rows, "is_helpful")),
        "percent_minimal": float(_percent_true(parsed_rows, "is_minimal")),
        "average_score": float(mean(valid_scores) if valid_scores else 0.0),
        "problem_type_counts": dict(sorted(problem_type_counts.items())),
    }


def _summary_markdown(summary: dict[str, Any]) -> str:
    return (
        f"# {SUB_PHASE_ID}\n\n"
        "| Field | Value |\n"
        "|---|---:|\n"
        f"| Eligible chunks | {summary.get('eligible_chunks', 0)} |\n"
        f"| Number sampled | {summary.get('number_sampled', 0)} |\n"
        f"| Parsed judgments | {summary.get('parsed_judgments', 0)} |\n"
        f"| Parse failures | {summary.get('parse_failures', 0)} |\n"
        f"| Percent supported | {summary.get('percent_supported', 0.0):.2f} |\n"
        f"| Percent faithful | {summary.get('percent_faithful', 0.0):.2f} |\n"
        f"| Percent helpful | {summary.get('percent_helpful', 0.0):.2f} |\n"
        f"| Percent minimal | {summary.get('percent_minimal', 0.0):.2f} |\n"
        f"| Average score | {summary.get('average_score', 0.0):.3f} |\n"
    )


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return str(path)


def run_linking_sentence_correctness_eval(
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
    records = load_canonicalized_chunk_records_from_cache_index(llm_cache_index_json)
    eligible_items = build_linking_sentence_items(records, dataset_name=dataset_name)
    sampled_items = sample_linking_sentence_items(
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
                user_prompt=build_linking_sentence_judge_prompt(item),
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
                    "chunk_id": item["chunk_id"],
                    "chunk_index": item["chunk_index"],
                    "judge_model": judge_model_label,
                    "judge_prompt_hash": prompt_sha,
                    "raw_judge_output": str(raw_output or ""),
                }
            )
            parsed = parse_linking_sentence_judgment(str(raw_output or ""))
            parsed_rows.append(
                {
                    "item_id": item["item_id"],
                    "dataset_name": item["dataset_name"],
                    "document_id": item["document_id"],
                    "chunk_id": item["chunk_id"],
                    "chunk_index": item["chunk_index"],
                    "judge_model": judge_model_label,
                    "judge_prompt_hash": prompt_sha,
                    **parsed,
                }
            )

    summary = aggregate_linking_sentence_statistics(
        dataset_name=dataset_name,
        eligible_count=len(eligible_items),
        sampled_count=len(sampled_items),
        parsed_rows=parsed_rows,
        seed=int(seed),
        judge_model_label=judge_model_label,
    )
    summary["output_dir"] = str(artifact_root)
    summary["llm_cache_index_json"] = str(Path(llm_cache_index_json).resolve())

    write_jsonl(sampled_path, sampled_items)
    write_jsonl(raw_outputs_path, raw_rows)
    write_jsonl(parsed_path, parsed_rows)
    _write_json(aggregate_path, summary)
    summary_md_path.write_text(_summary_markdown(summary), encoding="utf-8")

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
            "aggregate_stats_markdown": str(summary_md_path),
        },
        metadata={
            "dataset_name": dataset_name,
            "judge_model": judge_model_label,
            "eligible_chunks": len(eligible_items),
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
            "aggregate_stats_markdown": summary_md_path,
            "run_manifest_json": manifest_path,
        },
    }
