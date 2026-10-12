"""Per-document cumulative growth logging for updated-chunk augmentation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ....core.types import CanonicalizedChunkRecord
from ....io.hierarchy_cache import load_canonicalized_chunk_records_from_cache_index
from ....io.jsonl import write_jsonl
from ....io.manifests import build_manifest, write_manifest
from ..common import infer_dataset_label, resolve_updated_chunks_output_dir
from .common import (
    active_entities_from_record,
    normalize_alias_key,
    normalize_description_key,
    normalize_entity_name,
)

MAIN_PHASE_NAME = "Cumulative Growth Analysis"
SUB_PHASE_ID = "3_1_Per_Document_Cumulative_Growth_Logging"
SUB_PHASE_NAME = "Per-Document Cumulative Growth Logging"


def _build_item_id(dataset_name: str, record: CanonicalizedChunkRecord) -> str:
    return f"{dataset_name}:{record.doc_id}:{record.chunk_id}:{record.chunk_index}"


def _normalized_chunk_position(order_index: int, total_chunks: int) -> float:
    if total_chunks <= 1:
        return 0.0
    return float(order_index / (total_chunks - 1))


def build_per_document_cumulative_growth_records(
    records: list[CanonicalizedChunkRecord],
    *,
    dataset_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[CanonicalizedChunkRecord]] = {}
    for record in sorted(records, key=lambda row: (str(row.doc_id), int(row.chunk_index), str(row.chunk_id))):
        grouped.setdefault(str(record.doc_id), []).append(record)

    per_chunk_rows: list[dict[str, Any]] = []
    per_document_rows: list[dict[str, Any]] = []

    for document_id in sorted(doc_id for doc_id in grouped if doc_id):
        doc_records = grouped[document_id]
        total_chunks = int(len(doc_records))
        seen_entity_names: set[str] = set()
        seen_aliases: set[tuple[str, str]] = set()
        seen_descriptions: set[tuple[str, str]] = set()

        cumulative_entity_names = 0
        cumulative_aliases = 0
        cumulative_descriptions = 0

        for order_index, record in enumerate(doc_records):
            active_entities = active_entities_from_record(record)
            new_entity_names: list[str] = []
            new_aliases: list[dict[str, str]] = []
            new_descriptions: list[dict[str, str]] = []

            for active_entity in active_entities:
                entity_name_raw = active_entity.get("name", "")
                entity_name_key = normalize_entity_name(entity_name_raw)
                if not entity_name_key:
                    continue

                if entity_name_key not in seen_entity_names:
                    seen_entity_names.add(entity_name_key)
                    new_entity_names.append(str(entity_name_raw or "").strip())

                for alias in active_entity.get("aliases", []) or []:
                    alias_key = normalize_alias_key(entity_name=entity_name_raw, alias=alias)
                    if alias_key is None or alias_key in seen_aliases:
                        continue
                    seen_aliases.add(alias_key)
                    new_aliases.append(
                        {
                            "entity_name": str(entity_name_raw or "").strip(),
                            "alias": str(alias or "").strip(),
                        }
                    )

                for description in active_entity.get("description", []) or []:
                    description_key = normalize_description_key(entity_name=entity_name_raw, description=description)
                    if description_key is None or description_key in seen_descriptions:
                        continue
                    seen_descriptions.add(description_key)
                    new_descriptions.append(
                        {
                            "entity_name": str(entity_name_raw or "").strip(),
                            "description": str(description or "").strip(),
                        }
                    )

            new_entity_name_count = int(len(new_entity_names))
            new_alias_count = int(len(new_aliases))
            new_description_count = int(len(new_descriptions))
            new_total_modifications = int(new_entity_name_count + new_alias_count + new_description_count)

            cumulative_entity_names += new_entity_name_count
            cumulative_aliases += new_alias_count
            cumulative_descriptions += new_description_count
            cumulative_total_modifications = int(
                cumulative_entity_names + cumulative_aliases + cumulative_descriptions
            )

            per_chunk_rows.append(
                {
                    "item_id": _build_item_id(dataset_name, record),
                    "main_phase": MAIN_PHASE_NAME,
                    "sub_phase_id": SUB_PHASE_ID,
                    "sub_phase_name": SUB_PHASE_NAME,
                    "dataset_name": dataset_name,
                    "document_id": document_id,
                    "chunk_id": str(record.chunk_id),
                    "chunk_index": int(record.chunk_index),
                    "chunk_order_in_document": int(order_index),
                    "total_chunks_in_document": total_chunks,
                    "normalized_chunk_position": _normalized_chunk_position(order_index, total_chunks),
                    "parser_status": str(record.parser_status or "ok"),
                    "new_entity_names_introduced_at_chunk": new_entity_name_count,
                    "new_aliases_introduced_at_chunk": new_alias_count,
                    "new_descriptions_introduced_at_chunk": new_description_count,
                    "new_total_modifications_at_chunk": new_total_modifications,
                    "cumulative_entity_names": int(cumulative_entity_names),
                    "cumulative_aliases": int(cumulative_aliases),
                    "cumulative_descriptions": int(cumulative_descriptions),
                    "cumulative_total_modifications": int(cumulative_total_modifications),
                    "new_entity_names": new_entity_names,
                    "new_aliases": new_aliases,
                    "new_descriptions": new_descriptions,
                }
            )

        final_row = per_chunk_rows[-1] if per_chunk_rows and per_chunk_rows[-1]["document_id"] == document_id else None
        per_document_rows.append(
            {
                "item_id": f"{dataset_name}:{document_id}:summary",
                "main_phase": MAIN_PHASE_NAME,
                "sub_phase_id": SUB_PHASE_ID,
                "sub_phase_name": SUB_PHASE_NAME,
                "dataset_name": dataset_name,
                "document_id": document_id,
                "total_chunks_in_document": total_chunks,
                "final_chunk_index": int(doc_records[-1].chunk_index) if doc_records else -1,
                "final_cumulative_entity_names": int(cumulative_entity_names),
                "final_cumulative_aliases": int(cumulative_aliases),
                "final_cumulative_descriptions": int(cumulative_descriptions),
                "final_cumulative_total_modifications": int(
                    cumulative_entity_names + cumulative_aliases + cumulative_descriptions
                ),
                "final_normalized_chunk_position": float(final_row["normalized_chunk_position"]) if final_row else 0.0,
            }
        )

    return per_chunk_rows, per_document_rows


def summarize_per_document_cumulative_growth(
    per_chunk_rows: list[dict[str, Any]],
    per_document_rows: list[dict[str, Any]],
    *,
    dataset_name: str,
) -> dict[str, Any]:
    return {
        "main_phase": MAIN_PHASE_NAME,
        "sub_phase_id": SUB_PHASE_ID,
        "sub_phase_name": SUB_PHASE_NAME,
        "dataset_name": dataset_name,
        "processed_documents": int(len(per_document_rows)),
        "processed_chunks": int(len(per_chunk_rows)),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return str(path)


def run_per_document_cumulative_growth_logging_eval(
    *,
    experiment_config: dict[str, Any],
    llm_cache_index_json: str,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    dataset_name = infer_dataset_label(experiment_config)
    records = load_canonicalized_chunk_records_from_cache_index(llm_cache_index_json)
    per_chunk_rows, per_document_rows = build_per_document_cumulative_growth_records(
        records,
        dataset_name=dataset_name,
    )
    summary = summarize_per_document_cumulative_growth(
        per_chunk_rows,
        per_document_rows,
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

    per_chunk_path = artifact_root / "per_chunk_cumulative_growth.jsonl"
    per_document_path = artifact_root / "per_document_summary.jsonl"
    summary_path = artifact_root / "run_summary.json"
    manifest_path = artifact_root / "run_manifest.json"

    summary["output_dir"] = str(artifact_root)
    summary["llm_cache_index_json"] = str(Path(llm_cache_index_json).resolve())

    write_jsonl(per_chunk_path, per_chunk_rows)
    write_jsonl(per_document_path, per_document_rows)
    _write_json(summary_path, summary)

    manifest = build_manifest(
        config={
            "main_phase": MAIN_PHASE_NAME,
            "sub_phase_id": SUB_PHASE_ID,
            "sub_phase_name": SUB_PHASE_NAME,
            "experiment_config": experiment_config,
        },
        artifacts={
            "per_chunk_cumulative_growth_jsonl": str(per_chunk_path),
            "per_document_summary_jsonl": str(per_document_path),
            "run_summary_json": str(summary_path),
        },
        metadata={
            "dataset_name": dataset_name,
            "processed_documents": len(per_document_rows),
            "processed_chunks": len(per_chunk_rows),
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
            "per_chunk_cumulative_growth_jsonl": per_chunk_path,
            "per_document_summary_jsonl": per_document_path,
            "run_summary_json": summary_path,
            "run_manifest_json": manifest_path,
        },
    }
