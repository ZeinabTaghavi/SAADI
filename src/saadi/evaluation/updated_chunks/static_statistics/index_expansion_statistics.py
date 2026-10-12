"""Static statistics for searchable index expansion across raw, L0, and L0AJ representations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from ....core.text_edits import materialize_chunk_text
from ....io.hierarchy_cache import load_canonicalized_chunk_records_from_cache_index
from ....io.jsonl import write_jsonl
from ....io.manifests import build_manifest, write_manifest
from ....indexing.retrieval_methods.l0aj import build_l0aj_rows
from ..common import (
    infer_linking_sentence,
    infer_dataset_label,
    materialized_parenthetical_insertions,
    resolve_updated_chunks_output_dir,
)
from .common import build_token_counter, safe_mean
from .memory_augmentation_statistics import resolve_memory_augmentation_settings

MAIN_PHASE_NAME = "Static Statistics / Cost / Structure"
SUB_PHASE_ID = "2_5_Index_Expansion_Statistics"
SUB_PHASE_NAME = "Index Expansion Statistics"


def _to_bool(value: Any, *, default: bool = True) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def resolve_index_expansion_settings(
    *,
    experiment_config: dict[str, Any],
    llm_cache_index_json: str | Path,
    index_expansion_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = index_expansion_config if isinstance(index_expansion_config, dict) else {}
    include_l0aj = _to_bool(cfg.get("include_l0aj"), default=True)
    out = {
        "include_l0aj": bool(include_l0aj),
    }
    if include_l0aj:
        out["memory_augmentation"] = resolve_memory_augmentation_settings(
            experiment_config=experiment_config,
            llm_cache_index_json=llm_cache_index_json,
            memory_augmentation_config=cfg.get("memory_augmentation", {}) or {},
        )
    return out


def _build_item_id(dataset_name: str, *, doc_id: str, chunk_id: str, chunk_index: int) -> str:
    return f"{dataset_name}:{doc_id}:{chunk_id}:{chunk_index}"


def _l0_growth_breakdown(
    *,
    raw_text: str,
    linking_sentence: str,
    insertions: list[Any],
    count_tokens: Callable[[str], int],
) -> tuple[int, int]:
    raw_token_count = int(count_tokens(raw_text))
    linking_contribution = 0
    insertion_contribution = 0
    if linking_sentence:
        with_linking = materialize_chunk_text(
            raw_text,
            linking_sentence=linking_sentence,
            insertions=[],
        ).text
        linking_contribution = int(count_tokens(with_linking) - raw_token_count)
    if insertions:
        with_insertions = materialize_chunk_text(
            raw_text,
            linking_sentence="",
            insertions=list(insertions),
        ).text
        insertion_contribution = int(count_tokens(with_insertions) - raw_token_count)
    return linking_contribution, insertion_contribution


def build_index_expansion_records(
    records: list[Any],
    *,
    dataset_name: str,
    count_tokens: Callable[[str], int],
    index_expansion_settings: dict[str, Any],
) -> list[dict[str, Any]]:
    l0aj_by_chunk_id: dict[str, dict[str, Any]] = {}
    augmentation_settings = index_expansion_settings.get("memory_augmentation")
    if index_expansion_settings.get("include_l0aj") and isinstance(augmentation_settings, dict):
        l0aj_rows = build_l0aj_rows(
            records,
            augmentation_j=int(augmentation_settings["augmentation_j"]),
            max_aliases=int(augmentation_settings["max_aliases"]),
            max_desc=int(augmentation_settings["max_desc"]),
            active_entity_selector=str(augmentation_settings["active_entity_selector"]),
        )
        l0aj_by_chunk_id = {str(row.get("chunk_id", "")): row for row in l0aj_rows}

    out: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda row: (str(row.doc_id), int(row.chunk_index), str(row.chunk_id))):
        raw_text = str(record.raw_chunk_text or "")
        l0_text = str(record.canonicalized_chunk_text or "")
        raw_token_count = int(count_tokens(raw_text))
        l0_token_count = int(count_tokens(l0_text))
        raw_to_l0_growth_tokens = int(l0_token_count - raw_token_count)
        raw_to_l0_growth_percent = float((100.0 * raw_to_l0_growth_tokens / raw_token_count) if raw_token_count else 0.0)

        linking_sentence, _ = infer_linking_sentence(record)
        insertions = materialized_parenthetical_insertions(record)
        linking_contribution_tokens, parenthetical_contribution_tokens = _l0_growth_breakdown(
            raw_text=raw_text,
            linking_sentence=linking_sentence,
            insertions=insertions,
            count_tokens=count_tokens,
        )
        unattributed_l0_growth_tokens = int(
            raw_to_l0_growth_tokens - linking_contribution_tokens - parenthetical_contribution_tokens
        )

        l0aj_row = l0aj_by_chunk_id.get(str(record.chunk_id))
        l0aj_token_count: int | None = None
        raw_to_l0aj_growth_tokens: int | None = None
        raw_to_l0aj_growth_percent: float | None = None
        retrieval_time_memory_contribution_tokens: int | None = None
        if l0aj_row is not None:
            l0aj_token_count = int(count_tokens(str(l0aj_row.get("retrieval_text", "") or "")))
            raw_to_l0aj_growth_tokens = int(l0aj_token_count - raw_token_count)
            raw_to_l0aj_growth_percent = float((100.0 * raw_to_l0aj_growth_tokens / raw_token_count) if raw_token_count else 0.0)
            retrieval_time_memory_contribution_tokens = int(l0aj_token_count - l0_token_count)

        out.append(
            {
                "item_id": _build_item_id(
                    dataset_name,
                    doc_id=str(record.doc_id),
                    chunk_id=str(record.chunk_id),
                    chunk_index=int(record.chunk_index),
                ),
                "main_phase": MAIN_PHASE_NAME,
                "sub_phase_id": SUB_PHASE_ID,
                "sub_phase_name": SUB_PHASE_NAME,
                "dataset_name": dataset_name,
                "document_id": str(record.doc_id),
                "chunk_id": str(record.chunk_id),
                "chunk_index": int(record.chunk_index),
                "raw_token_count": raw_token_count,
                "l0_token_count": l0_token_count,
                "l0aj_token_count": l0aj_token_count,
                "raw_to_l0_growth_tokens": raw_to_l0_growth_tokens,
                "raw_to_l0_growth_percent": raw_to_l0_growth_percent,
                "raw_to_l0aj_growth_tokens": raw_to_l0aj_growth_tokens,
                "raw_to_l0aj_growth_percent": raw_to_l0aj_growth_percent,
                "linking_sentence_contribution_tokens": int(linking_contribution_tokens),
                "parenthetical_insertion_contribution_tokens": int(parenthetical_contribution_tokens),
                "unattributed_l0_growth_tokens": int(unattributed_l0_growth_tokens),
                "retrieval_time_memory_contribution_tokens": retrieval_time_memory_contribution_tokens,
            }
        )
    return out


def aggregate_index_expansion_statistics(
    rows: list[dict[str, Any]],
    *,
    dataset_name: str,
    index_expansion_settings: dict[str, Any],
    token_counting: dict[str, Any],
    document_id: str | None = None,
) -> dict[str, Any]:
    processed_chunks = int(len(rows))
    total_raw_token_count = int(sum(int(row.get("raw_token_count", 0) or 0) for row in rows))
    total_l0_token_count = int(sum(int(row.get("l0_token_count", 0) or 0) for row in rows))
    include_l0aj = bool(index_expansion_settings.get("include_l0aj"))
    total_l0aj_token_count = None
    if include_l0aj:
        total_l0aj_token_count = int(
            sum(int(row.get("l0aj_token_count", 0) or 0) for row in rows if row.get("l0aj_token_count") is not None)
        )

    percent_token_growth_raw_to_l0 = float(
        (100.0 * (total_l0_token_count - total_raw_token_count) / total_raw_token_count)
        if total_raw_token_count
        else 0.0
    )
    percent_token_growth_raw_to_l0aj = None
    if include_l0aj and total_l0aj_token_count is not None:
        percent_token_growth_raw_to_l0aj = float(
            (100.0 * (total_l0aj_token_count - total_raw_token_count) / total_raw_token_count)
            if total_raw_token_count
            else 0.0
        )

    documents_covered = int(
        len({str(row.get("document_id", "")).strip() for row in rows if str(row.get("document_id", "")).strip()})
    )

    summary = {
        "main_phase": MAIN_PHASE_NAME,
        "sub_phase_id": SUB_PHASE_ID,
        "sub_phase_name": SUB_PHASE_NAME,
        "dataset_name": dataset_name,
        "processed_chunks": processed_chunks,
        "documents_covered": documents_covered,
        "total_raw_token_count": total_raw_token_count,
        "total_l0_token_count": total_l0_token_count,
        "total_l0aj_token_count": total_l0aj_token_count,
        "percent_token_growth_raw_to_l0": percent_token_growth_raw_to_l0,
        "percent_token_growth_raw_to_l0aj": percent_token_growth_raw_to_l0aj,
        "average_raw_searchable_tokens_per_chunk": safe_mean([float(row.get("raw_token_count", 0) or 0) for row in rows]),
        "average_l0_searchable_tokens_per_chunk": safe_mean([float(row.get("l0_token_count", 0) or 0) for row in rows]),
        "average_l0aj_searchable_tokens_per_chunk": (
            safe_mean([float(row.get("l0aj_token_count", 0) or 0) for row in rows])
            if include_l0aj
            else None
        ),
        "total_linking_sentence_contribution_tokens": int(
            sum(int(row.get("linking_sentence_contribution_tokens", 0) or 0) for row in rows)
        ),
        "total_parenthetical_insertion_contribution_tokens": int(
            sum(int(row.get("parenthetical_insertion_contribution_tokens", 0) or 0) for row in rows)
        ),
        "total_unattributed_l0_growth_tokens": int(
            sum(int(row.get("unattributed_l0_growth_tokens", 0) or 0) for row in rows)
        ),
        "total_retrieval_time_memory_contribution_tokens": (
            int(sum(int(row.get("retrieval_time_memory_contribution_tokens", 0) or 0) for row in rows))
            if include_l0aj
            else None
        ),
        "token_counting": dict(token_counting),
        "index_expansion_settings": dict(index_expansion_settings),
    }
    if document_id is not None:
        summary["document_id"] = str(document_id)
    return summary


def build_per_document_index_expansion_statistics(
    rows: list[dict[str, Any]],
    *,
    dataset_name: str,
    index_expansion_settings: dict[str, Any],
    token_counting: dict[str, Any],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("document_id", "") or ""), []).append(row)
    out: list[dict[str, Any]] = []
    for document_id in sorted(doc_id for doc_id in grouped.keys() if doc_id):
        out.append(
            aggregate_index_expansion_statistics(
                grouped[document_id],
                dataset_name=dataset_name,
                index_expansion_settings=index_expansion_settings,
                token_counting=token_counting,
                document_id=document_id,
            )
        )
    return out


def _summary_markdown(summary: dict[str, Any]) -> str:
    total_l0aj_tokens = summary.get("total_l0aj_token_count")
    percent_raw_to_l0aj = summary.get("percent_token_growth_raw_to_l0aj")
    avg_l0aj_tokens = summary.get("average_l0aj_searchable_tokens_per_chunk")
    retrieval_memory_tokens = summary.get("total_retrieval_time_memory_contribution_tokens")
    total_l0aj_tokens_text = str(total_l0aj_tokens) if total_l0aj_tokens is not None else "n/a"
    percent_raw_to_l0aj_text = f"{float(percent_raw_to_l0aj):.3f}" if percent_raw_to_l0aj is not None else "n/a"
    avg_l0aj_tokens_text = f"{float(avg_l0aj_tokens):.3f}" if avg_l0aj_tokens is not None else "n/a"
    retrieval_memory_tokens_text = (
        str(retrieval_memory_tokens) if retrieval_memory_tokens is not None else "n/a"
    )
    return (
        f"# {SUB_PHASE_ID}\n\n"
        f"| Field | Value |\n"
        f"|---|---:|\n"
        f"| Processed chunks | {summary.get('processed_chunks', 0)} |\n"
        f"| Total raw tokens | {summary.get('total_raw_token_count', 0)} |\n"
        f"| Total L0 tokens | {summary.get('total_l0_token_count', 0)} |\n"
        f"| Total L0AJ tokens | {total_l0aj_tokens_text} |\n"
        f"| Percent growth raw -> L0 | {summary.get('percent_token_growth_raw_to_l0', 0.0):.3f} |\n"
        f"| Percent growth raw -> L0AJ | {percent_raw_to_l0aj_text} |\n"
        f"| Avg raw tokens / chunk | {summary.get('average_raw_searchable_tokens_per_chunk', 0.0):.3f} |\n"
        f"| Avg L0 tokens / chunk | {summary.get('average_l0_searchable_tokens_per_chunk', 0.0):.3f} |\n"
        f"| Avg L0AJ tokens / chunk | {avg_l0aj_tokens_text} |\n"
        f"| Linking contribution tokens | {summary.get('total_linking_sentence_contribution_tokens', 0)} |\n"
        f"| Parenthetical contribution tokens | {summary.get('total_parenthetical_insertion_contribution_tokens', 0)} |\n"
        f"| Retrieval-time memory contribution tokens | {retrieval_memory_tokens_text} |\n"
    )


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return str(path)


def run_index_expansion_statistics_eval(
    *,
    experiment_config: dict[str, Any],
    llm_cache_index_json: str,
    output_dir: str | Path | None = None,
    index_expansion_config: dict[str, Any] | None = None,
    token_counting_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dataset_name = infer_dataset_label(experiment_config)
    index_expansion_settings = resolve_index_expansion_settings(
        experiment_config=experiment_config,
        llm_cache_index_json=llm_cache_index_json,
        index_expansion_config=index_expansion_config,
    )
    token_counter, token_counting_metadata = build_token_counter(
        token_counting_config=token_counting_config,
        experiment_config=experiment_config,
    )
    records = load_canonicalized_chunk_records_from_cache_index(llm_cache_index_json)
    per_chunk_rows = build_index_expansion_records(
        records,
        dataset_name=dataset_name,
        count_tokens=token_counter,
        index_expansion_settings=index_expansion_settings,
    )
    per_document_rows = build_per_document_index_expansion_statistics(
        per_chunk_rows,
        dataset_name=dataset_name,
        index_expansion_settings=index_expansion_settings,
        token_counting=token_counting_metadata,
    )
    summary = aggregate_index_expansion_statistics(
        per_chunk_rows,
        dataset_name=dataset_name,
        index_expansion_settings=index_expansion_settings,
        token_counting=token_counting_metadata,
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
    per_document_path = artifact_root / "per_document_stats.jsonl"
    aggregate_path = artifact_root / "aggregate_stats.json"
    summary_md_path = artifact_root / "aggregate_stats.md"
    manifest_path = artifact_root / "run_manifest.json"

    summary["output_dir"] = str(artifact_root)
    summary["llm_cache_index_json"] = str(Path(llm_cache_index_json).resolve())

    write_jsonl(per_chunk_path, per_chunk_rows)
    write_jsonl(per_document_path, per_document_rows)
    _write_json(aggregate_path, summary)
    summary_md_path.write_text(_summary_markdown(summary), encoding="utf-8")

    manifest = build_manifest(
        config={
            "main_phase": MAIN_PHASE_NAME,
            "sub_phase_id": SUB_PHASE_ID,
            "sub_phase_name": SUB_PHASE_NAME,
            "experiment_config": experiment_config,
            "index_expansion": dict(index_expansion_config or {}),
            "token_counting": dict(token_counting_config or {}),
        },
        artifacts={
            "per_chunk_stats_jsonl": str(per_chunk_path),
            "per_document_stats_jsonl": str(per_document_path),
            "aggregate_stats_json": str(aggregate_path),
            "aggregate_stats_markdown": str(summary_md_path),
        },
        metadata={
            "dataset_name": dataset_name,
            "processed_chunks": len(per_chunk_rows),
            "index_expansion_settings": index_expansion_settings,
            "token_counting": token_counting_metadata,
        },
        cwd=str(Path.cwd()),
    )
    write_manifest(manifest_path, manifest)

    return {
        "main_phase": MAIN_PHASE_NAME,
        "sub_phase_id": SUB_PHASE_ID,
        "sub_phase_name": SUB_PHASE_NAME,
        "dataset_name": dataset_name,
        "index_expansion_settings": index_expansion_settings,
        "token_counting": token_counting_metadata,
        "summary": summary,
        "artifacts": {
            "artifact_root": artifact_root,
            "per_chunk_stats_jsonl": per_chunk_path,
            "per_document_stats_jsonl": per_document_path,
            "aggregate_stats_json": aggregate_path,
            "aggregate_stats_markdown": summary_md_path,
            "run_manifest_json": manifest_path,
        },
    }
