"""Static statistics for extracted active entities in updated chunks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ....core.types import CanonicalizedChunkRecord
from ....io.hierarchy_cache import load_canonicalized_chunk_records_from_cache_index
from ....io.jsonl import write_jsonl
from ....io.manifests import build_manifest, write_manifest
from ..common import infer_dataset_label, resolve_updated_chunks_output_dir
from .common import active_entities_from_record, safe_mean, value_count_distribution

MAIN_PHASE_NAME = "Static Statistics / Cost / Structure"
SUB_PHASE_ID = "2_2_Active_Entity_Statistics"
SUB_PHASE_NAME = "Active Entity Statistics"


def _build_item_id(dataset_name: str, record: CanonicalizedChunkRecord) -> str:
    return f"{dataset_name}:{record.doc_id}:{record.chunk_id}:{record.chunk_index}"


def build_active_entity_chunk_records(
    records: list[CanonicalizedChunkRecord],
    *,
    dataset_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    chunk_rows: list[dict[str, Any]] = []
    entity_rows: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda row: (str(row.doc_id), int(row.chunk_index), str(row.chunk_id))):
        active_entities = active_entities_from_record(record)
        entity_count = int(len(active_entities))
        chunk_item_id = _build_item_id(dataset_name, record)
        chunk_rows.append(
            {
                "item_id": chunk_item_id,
                "main_phase": MAIN_PHASE_NAME,
                "sub_phase_id": SUB_PHASE_ID,
                "sub_phase_name": SUB_PHASE_NAME,
                "dataset_name": dataset_name,
                "document_id": str(record.doc_id),
                "chunk_id": str(record.chunk_id),
                "chunk_index": int(record.chunk_index),
                "strategy_name": str(record.strategy_name or ""),
                "parser_status": str(record.parser_status or "ok"),
                "active_entity_count": entity_count,
                "has_active_entity": bool(entity_count > 0),
            }
        )
        for entity_index, active_entity in enumerate(active_entities):
            aliases = list(active_entity.get("aliases", []) or [])
            descriptions = list(active_entity.get("description", []) or [])
            entity_rows.append(
                {
                    "item_id": f"{chunk_item_id}:active_entity:{entity_index}",
                    "parent_chunk_item_id": chunk_item_id,
                    "main_phase": MAIN_PHASE_NAME,
                    "sub_phase_id": SUB_PHASE_ID,
                    "sub_phase_name": SUB_PHASE_NAME,
                    "dataset_name": dataset_name,
                    "document_id": str(record.doc_id),
                    "chunk_id": str(record.chunk_id),
                    "chunk_index": int(record.chunk_index),
                    "active_entity_index": int(entity_index),
                    "active_entity_name": str(active_entity.get("name", "") or ""),
                    "alias_count": int(len(aliases)),
                    "description_count": int(len(descriptions)),
                    "has_alias": bool(aliases),
                    "has_description": bool(descriptions),
                }
            )
    return chunk_rows, entity_rows


def aggregate_active_entity_statistics(
    chunk_rows: list[dict[str, Any]],
    entity_rows: list[dict[str, Any]],
    *,
    dataset_name: str,
    document_id: str | None = None,
) -> dict[str, Any]:
    processed_chunks = int(len(chunk_rows))
    chunks_with_active_entity = int(sum(1 for row in chunk_rows if bool(row.get("has_active_entity"))))
    entity_count_per_chunk = [int(row.get("active_entity_count", 0) or 0) for row in chunk_rows]
    eligible_chunk_counts = [count for count in entity_count_per_chunk if count > 0]

    documents = {
        str(row.get("document_id", "")).strip()
        for row in chunk_rows
        if str(row.get("document_id", "")).strip()
    }
    documents_covered = int(len(documents))

    alias_counts = [int(row.get("alias_count", 0) or 0) for row in entity_rows]
    description_counts = [int(row.get("description_count", 0) or 0) for row in entity_rows]
    active_entities_with_alias = int(sum(1 for row in entity_rows if bool(row.get("has_alias"))))
    active_entities_with_description = int(sum(1 for row in entity_rows if bool(row.get("has_description"))))
    total_active_entities = int(len(entity_rows))

    parser_status_counts: dict[str, int] = {}
    for row in chunk_rows:
        parser_status = str(row.get("parser_status", "") or "").strip()
        if parser_status:
            parser_status_counts[parser_status] = int(parser_status_counts.get(parser_status, 0)) + 1

    summary = {
        "main_phase": MAIN_PHASE_NAME,
        "sub_phase_id": SUB_PHASE_ID,
        "sub_phase_name": SUB_PHASE_NAME,
        "dataset_name": dataset_name,
        "processed_chunks": processed_chunks,
        "chunks_with_active_entity": chunks_with_active_entity,
        "fraction_chunks_with_active_entity": float((chunks_with_active_entity / processed_chunks) if processed_chunks else 0.0),
        "average_active_entities_per_chunk": safe_mean([float(value) for value in entity_count_per_chunk]),
        "average_active_entities_per_eligible_chunk": safe_mean([float(value) for value in eligible_chunk_counts]),
        "documents_covered": documents_covered,
        "total_active_entities": total_active_entities,
        "average_active_entities_per_document": float((total_active_entities / documents_covered) if documents_covered else 0.0),
        "average_descriptions_per_active_entity": safe_mean([float(value) for value in description_counts]),
        "average_aliases_per_active_entity": safe_mean([float(value) for value in alias_counts]),
        "fraction_active_entities_with_alias": float((active_entities_with_alias / total_active_entities) if total_active_entities else 0.0),
        "fraction_active_entities_with_description": float((active_entities_with_description / total_active_entities) if total_active_entities else 0.0),
        "active_entity_count_distribution_per_chunk": value_count_distribution(entity_count_per_chunk),
        "descriptions_per_entity_distribution": value_count_distribution(description_counts),
        "aliases_per_entity_distribution": value_count_distribution(alias_counts),
        "parser_status_counts": dict(sorted(parser_status_counts.items())),
    }
    if document_id is not None:
        summary["document_id"] = str(document_id)
    return summary


def build_per_document_active_entity_statistics(
    chunk_rows: list[dict[str, Any]],
    entity_rows: list[dict[str, Any]],
    *,
    dataset_name: str,
) -> list[dict[str, Any]]:
    chunks_by_doc: dict[str, list[dict[str, Any]]] = {}
    entities_by_doc: dict[str, list[dict[str, Any]]] = {}
    for row in chunk_rows:
        doc_id = str(row.get("document_id", "") or "")
        chunks_by_doc.setdefault(doc_id, []).append(row)
    for row in entity_rows:
        doc_id = str(row.get("document_id", "") or "")
        entities_by_doc.setdefault(doc_id, []).append(row)

    summaries: list[dict[str, Any]] = []
    for document_id in sorted(doc_id for doc_id in chunks_by_doc.keys() if doc_id):
        summary = aggregate_active_entity_statistics(
            chunks_by_doc[document_id],
            entities_by_doc.get(document_id, []),
            dataset_name=dataset_name,
            document_id=document_id,
        )
        summary["document_id"] = document_id
        summaries.append(summary)
    return summaries


def _summary_markdown(summary: dict[str, Any]) -> str:
    return (
        f"# {SUB_PHASE_ID}\n\n"
        f"| Field | Value |\n"
        f"|---|---:|\n"
        f"| Processed chunks | {summary.get('processed_chunks', 0)} |\n"
        f"| Chunks with active entities | {summary.get('chunks_with_active_entity', 0)} |\n"
        f"| Average active entities / chunk | {summary.get('average_active_entities_per_chunk', 0.0):.3f} |\n"
        f"| Average active entities / eligible chunk | {summary.get('average_active_entities_per_eligible_chunk', 0.0):.3f} |\n"
        f"| Average active entities / document | {summary.get('average_active_entities_per_document', 0.0):.3f} |\n"
        f"| Average descriptions / active entity | {summary.get('average_descriptions_per_active_entity', 0.0):.3f} |\n"
        f"| Average aliases / active entity | {summary.get('average_aliases_per_active_entity', 0.0):.3f} |\n"
        f"| Fraction active entities with alias | {summary.get('fraction_active_entities_with_alias', 0.0):.4f} |\n"
        f"| Fraction active entities with description | {summary.get('fraction_active_entities_with_description', 0.0):.4f} |\n"
    )


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return str(path)


def run_active_entity_statistics_eval(
    *,
    experiment_config: dict[str, Any],
    llm_cache_index_json: str,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    dataset_name = infer_dataset_label(experiment_config)
    records = load_canonicalized_chunk_records_from_cache_index(llm_cache_index_json)
    per_chunk_rows, per_active_entity_rows = build_active_entity_chunk_records(
        records,
        dataset_name=dataset_name,
    )
    per_document_rows = build_per_document_active_entity_statistics(
        per_chunk_rows,
        per_active_entity_rows,
        dataset_name=dataset_name,
    )
    summary = aggregate_active_entity_statistics(
        per_chunk_rows,
        per_active_entity_rows,
        dataset_name=dataset_name,
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
    per_active_entity_path = artifact_root / "per_active_entity_stats.jsonl"
    per_document_path = artifact_root / "per_document_stats.jsonl"
    aggregate_path = artifact_root / "aggregate_stats.json"
    summary_md_path = artifact_root / "aggregate_stats.md"
    manifest_path = artifact_root / "run_manifest.json"

    summary["output_dir"] = str(artifact_root)
    summary["llm_cache_index_json"] = str(Path(llm_cache_index_json).resolve())

    write_jsonl(per_chunk_path, per_chunk_rows)
    write_jsonl(per_active_entity_path, per_active_entity_rows)
    write_jsonl(per_document_path, per_document_rows)
    _write_json(aggregate_path, summary)
    summary_md_path.write_text(_summary_markdown(summary), encoding="utf-8")

    manifest = build_manifest(
        config={
            "main_phase": MAIN_PHASE_NAME,
            "sub_phase_id": SUB_PHASE_ID,
            "sub_phase_name": SUB_PHASE_NAME,
            "experiment_config": experiment_config,
        },
        artifacts={
            "per_chunk_stats_jsonl": str(per_chunk_path),
            "per_active_entity_stats_jsonl": str(per_active_entity_path),
            "per_document_stats_jsonl": str(per_document_path),
            "aggregate_stats_json": str(aggregate_path),
            "aggregate_stats_markdown": str(summary_md_path),
        },
        metadata={
            "dataset_name": dataset_name,
            "processed_chunks": len(per_chunk_rows),
            "total_active_entities": len(per_active_entity_rows),
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
            "per_chunk_stats_jsonl": per_chunk_path,
            "per_active_entity_stats_jsonl": per_active_entity_path,
            "per_document_stats_jsonl": per_document_path,
            "aggregate_stats_json": aggregate_path,
            "aggregate_stats_markdown": summary_md_path,
            "run_manifest_json": manifest_path,
        },
    }
