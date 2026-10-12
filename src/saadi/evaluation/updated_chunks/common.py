"""Shared helpers for updated-chunk evaluation packages."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ...core.text_edits import materialize_chunk_text
from ...core.types import CanonicalizedChunkRecord, TextInsertion
from ...entities import active_entity_config_label, resolve_active_entity_limits
from ...entities.active_selection import active_entity_selector_label, normalize_active_entity_selector
from ...io.hierarchy_cache import load_llm_history_cache_index, resolve_llm_cache_namespace_dir


def resolve_path_from_base(value: Any, *, base_dir: Path | None = None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return str(path)

    candidate_bases: list[Path] = []
    if base_dir is not None:
        base_dir = base_dir.resolve()
        candidate_bases.append(base_dir)
        candidate_bases.extend(base_dir.parents)
    cwd = Path.cwd().resolve()
    candidate_bases.append(cwd)
    candidate_bases.extend(cwd.parents)

    seen: set[str] = set()
    for candidate_base in candidate_bases:
        candidate = (candidate_base / path).resolve()
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return str(candidate)

    if candidate_bases:
        return str((candidate_bases[0] / path).resolve())
    return str(path.resolve())


def infer_dataset_label(config: dict[str, Any]) -> str:
    ds = config.get("dataset", {}) or {}
    if isinstance(ds, dict):
        for key in ("name", "dataset", "id"):
            value = ds.get(key)
            if value:
                return str(value)
        hf_dataset = ds.get("hf_dataset")
        if hf_dataset:
            parts = [part for part in str(hf_dataset).split("/") if part]
            if parts:
                return parts[-1]
    value = config.get("dataset_name")
    return str(value) if value else "unknown_dataset"


def infer_llm_label(config: dict[str, Any]) -> str:
    model_cfg = config.get("model", {}) or {}
    if isinstance(model_cfg, dict):
        for key in ("alias", "name", "llm"):
            value = model_cfg.get(key)
            if value:
                return str(value)
        model_id = model_cfg.get("model_id")
        if model_id:
            parts = [part for part in str(model_id).split("/") if part]
            if len(parts) >= 2:
                return parts[0]
            if parts:
                return parts[-1]
        backend = model_cfg.get("backend")
        if backend:
            return str(backend)
    return "unknown_llm"


def infer_model_label(model_cfg: dict[str, Any] | None) -> str:
    cfg = model_cfg or {}
    for key in ("alias", "name", "llm"):
        value = cfg.get(key)
        if value:
            return str(value)
    model_id = cfg.get("model_id")
    if model_id:
        parts = [part for part in str(model_id).split("/") if part]
        if len(parts) >= 2:
            return parts[-1]
        if parts:
            return parts[0]
    backend = cfg.get("backend")
    return str(backend or "unknown_llm")


def infer_hierarchy_branching_factor(config: dict[str, Any]) -> int | None:
    ingest_cfg = config.get("ingest", {}) or {}
    if not isinstance(ingest_cfg, dict):
        return None
    raw_kwargs = ingest_cfg.get("strategy_kwargs", {}) or {}
    if isinstance(raw_kwargs, dict):
        for key in ("branching_factor", "hierarchy_branching_factor", "k"):
            if raw_kwargs.get(key) is not None:
                try:
                    return int(raw_kwargs.get(key))
                except Exception:
                    return None
    strategy_name = str(ingest_cfg.get("strategy", "hierarchical") or "hierarchical").strip().lower()
    if strategy_name in {
        "hierarchical",
        "saadi_hierarchical",
        "accumulative",
        "limited_hierarchical",
        "limited_hierarchical_summary",
        "saadi_limited_hierarchical",
    }:
        return 5
    return None


def configured_active_entity_selector(config: dict[str, Any]) -> str | None:
    ingest_cfg = config.get("ingest", {}) or {}
    if not isinstance(ingest_cfg, dict):
        return None
    raw_kwargs = ingest_cfg.get("strategy_kwargs", {}) or {}
    if not isinstance(raw_kwargs, dict):
        return None
    value = raw_kwargs.get("active_entity_selector")
    if value is None:
        value = raw_kwargs.get("active_object_selector")
    if value is None:
        return None
    return normalize_active_entity_selector(str(value))


def configured_active_entity_limits(config: dict[str, Any]) -> tuple[int, int, int]:
    ingest_cfg = config.get("ingest", {}) or {}
    if not isinstance(ingest_cfg, dict):
        return resolve_active_entity_limits(None)
    raw_kwargs = ingest_cfg.get("strategy_kwargs", {}) or {}
    return resolve_active_entity_limits(raw_kwargs if isinstance(raw_kwargs, dict) else None)


def infer_active_entity_selector_cache_label(config: dict[str, Any]) -> str | None:
    configured = configured_active_entity_selector(config)
    if configured is None:
        return None
    return active_entity_selector_label(configured)


def infer_active_entity_config_cache_label(config: dict[str, Any]) -> str:
    max_entities, max_aliases, max_desc = configured_active_entity_limits(config)
    return active_entity_config_label(max_entities, max_aliases, max_desc)


def resolve_cache_index_from_experiment_config(config: dict[str, Any]) -> str | None:
    output_dir = config.get("output_dir")
    if not output_dir:
        return None
    ingest_cfg = config.get("ingest", {}) or {}
    strategy_name = str((ingest_cfg or {}).get("strategy", "hierarchical") or "hierarchical")
    chunk_size_raw = (ingest_cfg or {}).get("chunk_size", 500)
    try:
        chunk_size = int(chunk_size_raw)
    except Exception:
        chunk_size = 500
    branching_factor = infer_hierarchy_branching_factor(config)
    cache_dir = resolve_llm_cache_namespace_dir(
        str(output_dir),
        dataset_label=infer_dataset_label(config),
        method_label=strategy_name,
        hierarchy_branching_factor=branching_factor,
        chunk_size=chunk_size,
        active_entity_config_label=infer_active_entity_config_cache_label(config),
        llm_label=infer_llm_label(config),
        active_entity_selector_label=infer_active_entity_selector_cache_label(config),
    )
    index_path = cache_dir / "index.json"
    if index_path.exists():
        return str(index_path)
    loaded = load_llm_history_cache_index(
        output_dir=str(output_dir),
        dataset_label=infer_dataset_label(config),
        method_label=strategy_name,
        hierarchy_branching_factor=branching_factor,
        chunk_size=chunk_size,
        active_entity_config_label=infer_active_entity_config_cache_label(config),
        llm_label=infer_llm_label(config),
        active_entity_selector_label=infer_active_entity_selector_cache_label(config),
    )
    if isinstance(loaded, dict) and loaded.get("index_path"):
        return str(loaded["index_path"])
    return str(index_path)


def infer_hierarchy_k(config: dict[str, Any]) -> int | None:
    """Backward-compatible alias for code that still asks for hierarchy_k."""
    return infer_hierarchy_branching_factor(config)


def resolve_updated_chunks_output_dir(
    experiment_config: dict[str, Any],
    *,
    subphase_id: str,
    run_name: str = "default",
) -> Path:
    root = Path(str(experiment_config.get("output_dir", "outputs"))).resolve()
    return root / "evaluation" / "updated_chunks" / str(subphase_id) / str(run_name or "default")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def materialized_parenthetical_insertions(record: CanonicalizedChunkRecord) -> list[TextInsertion]:
    if not record.insertions:
        return []
    materialized = materialize_chunk_text(
        record.raw_chunk_text,
        linking_sentence=record.linking_sentence,
        insertions=list(record.insertions),
    )
    if not materialized.insertions:
        return []

    canonicalized_text = normalize_whitespace(record.canonicalized_chunk_text)
    materialized_text = normalize_whitespace(materialized.text)
    if canonicalized_text == materialized_text:
        return list(materialized.insertions)

    verified: list[TextInsertion] = []
    for insertion in materialized.insertions:
        marker = f"({str(insertion.name).strip()})"
        if marker and marker in record.canonicalized_chunk_text:
            verified.append(insertion)
    return verified


def _body_without_linking(record: CanonicalizedChunkRecord) -> str:
    materialized = materialize_chunk_text(
        record.raw_chunk_text,
        linking_sentence="",
        insertions=list(record.insertions),
    )
    return str(materialized.text or "")


def infer_linking_sentence(record: CanonicalizedChunkRecord) -> tuple[str, str]:
    explicit = str(record.linking_sentence or "").strip()
    updated = str(record.canonicalized_chunk_text or "")
    body = _body_without_linking(record)

    if explicit:
        expected = materialize_chunk_text(
            record.raw_chunk_text,
            linking_sentence=explicit,
            insertions=list(record.insertions),
        ).text
        if normalize_whitespace(expected) == normalize_whitespace(updated):
            return explicit, "explicit"
        if normalize_whitespace(updated).startswith(normalize_whitespace(explicit)):
            return explicit, "explicit"

    if not updated or not body:
        return "", "none"

    if normalize_whitespace(updated) == normalize_whitespace(body):
        return "", "none"

    if updated.endswith(body):
        prefix = updated[: len(updated) - len(body)].strip()
        if prefix:
            return prefix, "inferred"

    pos = updated.find(body)
    if pos > 0:
        prefix = updated[:pos].strip()
        if prefix:
            return prefix, "inferred"

    normalized_updated = normalize_whitespace(updated)
    normalized_body = normalize_whitespace(body)
    if normalized_body and normalized_updated.endswith(normalized_body):
        normalized_prefix = normalized_updated[: len(normalized_updated) - len(normalized_body)].strip()
        if normalized_prefix:
            return normalized_prefix, "inferred_normalized"

    return "", "none"


def hierarchy_context_before_chunk(previous_record: CanonicalizedChunkRecord | None) -> list[dict[str, Any]]:
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


def previous_active_entities(previous_record: CanonicalizedChunkRecord | None) -> list[dict[str, Any]]:
    if previous_record is None:
        return []
    raw = (previous_record.state_snapshot or {}).get("active_entities", []) or []
    return [row for row in raw if isinstance(row, dict)]


def current_active_entities(record: CanonicalizedChunkRecord) -> list[dict[str, Any]]:
    raw = (record.state_snapshot or {}).get("active_entities", []) or []
    return [row for row in raw if isinstance(row, dict)]
