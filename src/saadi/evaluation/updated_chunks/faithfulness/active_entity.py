"""Faithfulness evaluation for extracted active entities in updated chunks."""

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
    current_active_entities,
    hierarchy_context_before_chunk,
    infer_dataset_label,
    infer_model_label,
    previous_active_entities,
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
SUB_PHASE_ID = "1_3_Active_Entity_Faithfulness"
SUB_PHASE_NAME = "Active Entity Faithfulness"
PROMPT_VERSION = "v2-active-entity"

JUDGE_RESPONSE_SCHEMA = {
    "entity_names_grounded": True,
    "descriptions_supported": True,
    "aliases_supported": True,
    "is_faithful_overall": True,
    "score": 5,
    "reason": "All active entities are grounded and supported by the chunk or allowed context.",
    "num_supported_entities": 3,
    "num_problematic_entities": 0,
    "problem_type": "none",
}

JUDGE_SYSTEM_PROMPT = """You are a strict research evaluator for updated-chunk faithfulness.

Task: judge the active entities extracted for one chunk.

Each active entity may contain:
- an entity name
- short descriptions
- aliases

Evaluate whether:
- entity names are grounded in the raw chunk or allowed context
- descriptions are supported by the evidence
- aliases are justified by the evidence
- the overall entity set is faithful and not over-interpreted

Important rules:
- Judge the extracted active entities, not the whole updated chunk.
- The raw chunk is the primary evidence.
- Allowed context can help resolve references, but do not infer unsupported claims.
- If any entity contains an unsupported name, description, or alias, that should count against faithfulness.
- If a field is absent (for example, no aliases), do not penalize it just for being absent.
- Score rubric:
  1 = many hallucinated or badly unsupported entities
  2 = mostly unsupported or over-interpreted
  3 = mixed quality
  4 = mostly grounded with minor issues
  5 = fully grounded and faithful

Return JSON only, matching this schema:
"""


def _build_item_id(dataset_name: str, record: CanonicalizedChunkRecord) -> str:
    return f"{dataset_name}:{record.doc_id}:{record.chunk_id}:{record.chunk_index}"


def _count_nonempty_aliases(entities: list[dict[str, Any]]) -> int:
    total = 0
    for entity in entities:
        aliases = entity.get("aliases", []) or []
        if not isinstance(aliases, list):
            continue
        total += sum(1 for alias in aliases if str(alias).strip())
    return total


def _count_nonempty_descriptions(entities: list[dict[str, Any]]) -> int:
    total = 0
    for entity in entities:
        descriptions = entity.get("description", []) or []
        if not isinstance(descriptions, list):
            continue
        total += sum(1 for desc in descriptions if str(desc).strip())
    return total


def build_active_entity_items(
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
            active_entities = current_active_entities(record)
            if not active_entities:
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
                    "raw_chunk": str(record.raw_chunk_text or ""),
                    "updated_chunk": str(record.canonicalized_chunk_text or ""),
                    "active_entities": active_entities,
                    "number_of_active_entities": len(active_entities),
                    "number_of_aliases": _count_nonempty_aliases(active_entities),
                    "number_of_descriptions": _count_nonempty_descriptions(active_entities),
                    "allowed_context": {
                        "previous_chunk": (
                            str(previous_record.canonicalized_chunk_text or "")
                            if previous_record is not None
                            else ""
                        ),
                        "previous_active_entities": previous_active_entities(previous_record),
                        "hierarchy_context": hierarchy_context_before_chunk(previous_record),
                    },
                }
            )
            previous_record = record
    return items


def sample_active_entity_items(
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


def build_active_entity_judge_prompt(item: dict[str, Any]) -> str:
    user_payload = {
        "dataset_name": item.get("dataset_name"),
        "document_id": item.get("document_id"),
        "chunk_id": item.get("chunk_id"),
        "chunk_index": item.get("chunk_index"),
        "raw_chunk": item.get("raw_chunk"),
        "active_entities": item.get("active_entities", []),
        "allowed_context": item.get("allowed_context", {}),
    }
    return (
        "Evaluate whether the extracted active entities are grounded and faithful.\n\n"
        "Return JSON only.\n\n"
        f"SCHEMA:\n{json_schema_string(JUDGE_RESPONSE_SCHEMA)}\n\n"
        f"EVALUATION_ITEM_JSON:\n{json.dumps(user_payload, ensure_ascii=False, indent=2)}\n\n"
        "Focus on grounding of entity names, support for descriptions, support for aliases, and overall faithfulness."
    )


def parse_active_entity_judgment(raw_text: str) -> dict[str, Any]:
    parsed, parser_status = extract_json_object_from_text(raw_text)
    if parsed is None:
        return {
            "parser_status": "failed",
            "entity_names_grounded": None,
            "descriptions_supported": None,
            "aliases_supported": None,
            "is_faithful_overall": None,
            "score": None,
            "reason": "",
            "num_supported_entities": None,
            "num_problematic_entities": None,
            "problem_type": "parse_failure",
        }

    def _normalize_optional_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            out = int(float(value))
        except Exception:
            return None
        if out < 0:
            return None
        return out

    entity_names_grounded = normalize_boolish(parsed.get("entity_names_grounded"))
    descriptions_supported = normalize_boolish(parsed.get("descriptions_supported"))
    aliases_supported = normalize_boolish(parsed.get("aliases_supported"))
    is_faithful_overall = normalize_boolish(parsed.get("is_faithful_overall"))
    score = normalize_score_1_to_5(parsed.get("score"))
    reason = str(parsed.get("reason", "") or "").strip()
    num_supported_entities = _normalize_optional_int(
        parsed.get("num_supported_entities", parsed.get("num_supported_objects"))
    )
    num_problematic_entities = _normalize_optional_int(
        parsed.get("num_problematic_entities", parsed.get("num_problematic_objects"))
    )
    problem_type = normalize_problem_type(parsed.get("problem_type"))
    if (
        problem_type is None
        and entity_names_grounded is True
        and descriptions_supported is True
        and aliases_supported is True
        and is_faithful_overall is True
    ):
        problem_type = "none"

    return {
        "parser_status": parser_status,
        "entity_names_grounded": entity_names_grounded,
        "descriptions_supported": descriptions_supported,
        "aliases_supported": aliases_supported,
        "is_faithful_overall": is_faithful_overall,
        "score": score,
        "reason": reason,
        "num_supported_entities": num_supported_entities,
        "num_problematic_entities": num_problematic_entities,
        "problem_type": problem_type,
    }


def _percent_true(rows: list[dict[str, Any]], key: str) -> float:
    eligible = [row for row in rows if row.get(key) is not None]
    if not eligible:
        return 0.0
    hits = sum(1 for row in eligible if bool(row.get(key)))
    return 100.0 * hits / len(eligible)


def aggregate_active_entity_statistics(
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
    supported_entity_counts = [int(row["num_supported_entities"]) for row in parsed_rows if row.get("num_supported_entities") is not None]
    problematic_entity_counts = [int(row["num_problematic_entities"]) for row in parsed_rows if row.get("num_problematic_entities") is not None]
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
        "percent_faithful_overall": float(_percent_true(parsed_rows, "is_faithful_overall")),
        "percent_grounded_entity_names": float(_percent_true(parsed_rows, "entity_names_grounded")),
        "percent_supported_descriptions": float(_percent_true(parsed_rows, "descriptions_supported")),
        "percent_supported_aliases": float(_percent_true(parsed_rows, "aliases_supported")),
        "average_score": float(mean(valid_scores) if valid_scores else 0.0),
        "average_num_supported_entities": float(mean(supported_entity_counts) if supported_entity_counts else 0.0),
        "average_num_problematic_entities": float(mean(problematic_entity_counts) if problematic_entity_counts else 0.0),
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
        f"| Percent faithful overall | {summary.get('percent_faithful_overall', 0.0):.2f} |\n"
        f"| Percent grounded entity names | {summary.get('percent_grounded_entity_names', 0.0):.2f} |\n"
        f"| Percent supported descriptions | {summary.get('percent_supported_descriptions', 0.0):.2f} |\n"
        f"| Percent supported aliases | {summary.get('percent_supported_aliases', 0.0):.2f} |\n"
        f"| Average score | {summary.get('average_score', 0.0):.3f} |\n"
    )


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return str(path)


def run_active_entity_faithfulness_eval(
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
    eligible_items = build_active_entity_items(records, dataset_name=dataset_name)
    sampled_items = sample_active_entity_items(
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
                user_prompt=build_active_entity_judge_prompt(item),
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
            parsed = parse_active_entity_judgment(str(raw_output or ""))
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

    summary = aggregate_active_entity_statistics(
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


# Deprecated callable aliases for existing analysis notebooks.
build_active_object_items = build_active_entity_items
sample_active_object_items = sample_active_entity_items
build_active_object_judge_prompt = build_active_entity_judge_prompt
parse_active_object_judgment = parse_active_entity_judgment
aggregate_active_object_statistics = aggregate_active_entity_statistics
run_active_object_faithfulness_eval = run_active_entity_faithfulness_eval
