"""Saves/loads canonicalized chunks and exports hierarchy cache files by dataset/method/LLM."""

from __future__ import annotations

import json
import re
from functools import wraps
from pathlib import Path
from typing import Any

from ..core.types import CanonicalizedChunkRecord, ChunkRecord, EntityRecord, TextInsertion
from ..entities.merge import merge_active_entities
from .jsonl import read_jsonl, write_jsonl


def _accept_legacy_entity_cache_keywords(func):
    """Translate deprecated cache keyword names before signature binding."""
    aliases = {
        "active_object_config_label": "active_entity_config_label",
        "active_object_selector_label": "active_entity_selector_label",
        "active_object_min_relevance": "active_entity_min_relevance",
    }

    @wraps(func)
    def wrapped(*args, **kwargs):
        kwargs = dict(kwargs)
        for legacy_key, canonical_key in aliases.items():
            if legacy_key not in kwargs:
                continue
            if canonical_key in kwargs and kwargs[canonical_key] != kwargs[legacy_key]:
                raise ValueError(
                    f"Conflicting cache settings: {canonical_key!r} and deprecated {legacy_key!r}."
                )
            kwargs.setdefault(canonical_key, kwargs[legacy_key])
            kwargs.pop(legacy_key, None)
        return func(*args, **kwargs)

    return wrapped


def _safe_path_component(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return clean or "doc"


def _canonicalize_legacy_entity_path(path: Path) -> Path:
    """Map historical entity-cache path segments without mutating artifacts."""
    replacements = {"AOWord": "AEWord", "AObm25": "AEBM25"}
    parts = [
        replacements.get(part, "AE_" + part[3:] if part.startswith("AO_") else part)
        for part in path.parts
    ]
    return Path(*parts)


def legacy_active_entity_cache_labels(
    config_label: str | None,
    selector_label: str | None,
) -> tuple[str | None, str | None]:
    """Map canonical AE labels to their historical AO cache equivalents."""
    legacy_config = config_label
    if isinstance(legacy_config, str) and legacy_config.startswith("AE_"):
        legacy_config = "AO_" + legacy_config[3:]

    selector_map = {
        "AEWord": "AOWord",
        "AEBM25": "AObm25",
        "AEbm25": "AObm25",
    }
    legacy_selector = selector_map.get(str(selector_label), selector_label)
    return legacy_config, legacy_selector


@_accept_legacy_entity_cache_keywords
def resolve_llm_cache_namespace_dir(
    output_dir: str | Path,
    *,
    dataset_label: str | None,
    method_label: str | None,
    hierarchy_branching_factor: int | None = None,
    hierarchy_k: int | None = None,
    chunk_size: int | None = None,
    active_entity_config_label: str | None = None,
    llm_label: str | None = None,
    active_entity_selector_label: str | None = None,
) -> Path:
    dataset_part = _safe_path_component((dataset_label or "unknown_dataset").lower())
    method_part = _safe_path_component((method_label or "unknown_method").lower())
    b = _coerce_branching_factor(hierarchy_branching_factor, hierarchy_k)
    branching_part = f"branching_factor_{b}" if isinstance(b, int) and b > 0 else "branching_factor_unknown"
    chunk_part = (
        f"chunk_{chunk_size}" if isinstance(chunk_size, int) and chunk_size > 0 else "chunk_unknown"
    )
    # Keep user-defined LLM labels (e.g. "Qwen30BIns") as-is for readable folder naming.
    llm_part = _safe_path_component(llm_label or "unknown_llm")
    out = Path(output_dir) / "llm_cache" / dataset_part / method_part / branching_part / chunk_part
    if active_entity_config_label:
        out = out / _safe_path_component(active_entity_config_label)
    out = out / llm_part
    if active_entity_selector_label:
        out = out / _safe_path_component(active_entity_selector_label)
    return out


def _coerce_branching_factor(*values: int | None) -> int | None:
    for value in values:
        try:
            b = int(value) if value is not None else None
        except Exception:
            b = None
        if b is not None and b > 0:
            return b
    return None


@_accept_legacy_entity_cache_keywords
def _legacy_llm_cache_namespace_dir(
    output_dir: str | Path,
    *,
    dataset_label: str | None,
    method_label: str | None,
    branching_factor: int | None,
    chunk_size: int | None = None,
    active_entity_config_label: str | None = None,
    llm_label: str | None = None,
    active_entity_selector_label: str | None = None,
) -> Path:
    dataset_part = _safe_path_component((dataset_label or "unknown_dataset").lower())
    method_part = _safe_path_component((method_label or "unknown_method").lower())
    legacy_part = f"k_{branching_factor}" if isinstance(branching_factor, int) and branching_factor > 0 else "k_unknown"
    chunk_part = (
        f"chunk_{chunk_size}" if isinstance(chunk_size, int) and chunk_size > 0 else "chunk_unknown"
    )
    llm_part = _safe_path_component(llm_label or "unknown_llm")
    out = Path(output_dir) / "llm_cache" / dataset_part / method_part / legacy_part / chunk_part
    if active_entity_config_label:
        out = out / _safe_path_component(active_entity_config_label)
    out = out / llm_part
    if active_entity_selector_label:
        out = out / _safe_path_component(active_entity_selector_label)
    return out


@_accept_legacy_entity_cache_keywords
def resolve_canonicalized_chunk_cache_path(
    output_dir: str | Path,
    *,
    dataset_label: str | None,
    method_label: str | None,
    hierarchy_branching_factor: int | None = None,
    hierarchy_k: int | None = None,
    chunk_size: int | None = None,
    active_entity_config_label: str | None = None,
    llm_label: str | None = None,
    active_entity_selector_label: str | None = None,
) -> Path:
    return (
        resolve_llm_cache_namespace_dir(
            output_dir,
            dataset_label=dataset_label,
            method_label=method_label,
            hierarchy_branching_factor=hierarchy_branching_factor,
            hierarchy_k=hierarchy_k,
            chunk_size=chunk_size,
            active_entity_config_label=active_entity_config_label,
            llm_label=llm_label,
            active_entity_selector_label=active_entity_selector_label,
        )
        / "canonicalized_chunks_cache.jsonl"
    )


def _cycle_to_level(cycle_size: int, branching_factor: int | None) -> int | None:
    if cycle_size <= 1:
        return None
    b = _coerce_branching_factor(branching_factor)
    if b is None or b <= 1:
        return None
    level = 0
    cur = 1
    while cur < cycle_size:
        cur *= b
        level += 1
    return level if cur == cycle_size else None


def _normalize_hierarchy_snapshot(snapshot: dict[str, Any]) -> dict[int, dict[int, str]]:
    out: dict[int, dict[int, str]] = {}
    raw = snapshot.get("hierarchy_summaries", {}) or {}
    if not isinstance(raw, dict):
        return out
    for key, vals in raw.items():
        try:
            cycle = int(key)
        except Exception:
            continue
        # New format: {cycle: {step: summary}}
        if isinstance(vals, dict):
            by_step: dict[int, str] = {}
            for sk, sv in vals.items():
                try:
                    step = int(sk)
                except Exception:
                    continue
                if sv is None:
                    continue
                by_step[step] = str(sv)
            out[cycle] = by_step
            continue
        # Backward compatibility for old snapshots {cycle: [summary,...]}
        if isinstance(vals, list):
            by_step = {}
            for idx, sv in enumerate(vals):
                if sv is None:
                    continue
                step = (idx + 1) * cycle - 1
                by_step[step] = str(sv)
            out[cycle] = by_step
    return out


def _entity_from_any(obj: Any) -> EntityRecord | None:
    if isinstance(obj, EntityRecord):
        return obj
    if not isinstance(obj, dict):
        return None
    name = str(obj.get("name", "") or "").strip()
    if not name:
        return None
    desc = obj.get("description", []) or []
    aliases = obj.get("aliases", []) or []
    if not isinstance(desc, list):
        desc = [desc]
    if not isinstance(aliases, list):
        aliases = [aliases]
    return EntityRecord(
        name=name,
        description=[str(x) for x in desc if str(x).strip()],
        aliases=[str(x) for x in aliases if str(x).strip()],
    )


def _insertion_from_any(obj: Any) -> TextInsertion | None:
    if isinstance(obj, TextInsertion):
        return obj
    if not isinstance(obj, dict):
        return None
    name = str(obj.get("name", obj.get("object_name", obj.get("entity_name", ""))) or "").strip()
    anchor_text = str(obj.get("anchor_text", "") or "").strip()
    if not anchor_text or not name:
        return None
    try:
        occurrence = int(obj.get("occurrence", 1) or 1)
    except Exception:
        occurrence = 1
    return TextInsertion(
        anchor_text=anchor_text,
        left_context=str(obj.get("left_context", "") or ""),
        right_context=str(obj.get("right_context", "") or ""),
        occurrence=max(1, occurrence),
        name=name,
    )


def _resolve_cache_file_path(path_value: Any, *, base_dir: Path | None = None) -> Path:
    """
    Resolve cache file references stored in index.json.

    Older or hand-moved caches may store paths as:
    - absolute paths
    - paths relative to index.json's directory
    - repo-root-relative paths such as `outputs/...`

    To support all three, try the base directory first and then walk its
    ancestors until a matching file exists.
    """
    path = Path(str(path_value))
    canonical_path = _canonicalize_legacy_entity_path(path)
    path_variants = [path]
    if canonical_path != path:
        path_variants.append(canonical_path)
    if path.is_absolute():
        for path_variant in path_variants:
            if path_variant.exists():
                return path_variant
        if base_dir is None:
            return canonical_path
        # Evaluation directories are commonly copied between machines while
        # index.json retains the original server absolute paths.  Resolve the
        # exact repository-relative suffix; do not search by basename.
        base = base_dir.resolve()
        for path_variant in path_variants:
            parts = path_variant.parts
            for ancestor in (base, *base.parents):
                for anchor in ("outputs", "data", "configs"):
                    if anchor not in parts:
                        continue
                    candidate = ancestor.joinpath(*parts[parts.index(anchor) :]).resolve()
                    if candidate.exists():
                        return candidate
        return canonical_path

    candidate_bases: list[Path] = []
    if base_dir is not None:
        base_dir = base_dir.resolve()
        candidate_bases.append(base_dir)
        candidate_bases.extend(base_dir.parents)
    candidate_bases.append(Path.cwd().resolve())

    seen: set[str] = set()
    for path_variant in path_variants:
        for candidate_base in candidate_bases:
            candidate = (candidate_base / path_variant).resolve()
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            if candidate.exists():
                return candidate

    if candidate_bases:
        return (candidate_bases[0] / canonical_path).resolve()
    return canonical_path.resolve()


def _load_v2_doc_records(
    *,
    doc_id: str,
    file_map: dict[str, Any],
    hierarchy_branching_factor: int | None = None,
    hierarchy_k: int | None = None,
    strategy_name_fallback: str | None,
    stores_hierarchy: bool = True,
    base_dir: Path | None = None,
) -> list[CanonicalizedChunkRecord]:
    def _resolve_path(path_value: Any) -> Path:
        return _resolve_cache_file_path(path_value, base_dir=base_dir)

    steps_path = file_map.get("steps_index")
    l0_path = file_map.get("L_0")
    if not steps_path or not l0_path:
        return []

    steps_file = _resolve_path(steps_path)
    l0_file = _resolve_path(l0_path)
    if not steps_file.exists() or not l0_file.exists():
        return []

    l0_rows = read_jsonl(l0_file)
    l0_by_step: dict[int, str] = {}
    for row in l0_rows:
        if not isinstance(row, dict):
            continue
        try:
            step = int(row.get("step"))
        except Exception:
            continue
        chunk = row.get("chunk")
        if chunk is None:
            continue
        l0_by_step[step] = str(chunk)
    if not l0_by_step:
        return []

    # Load all hierarchy files once; each step snapshot slices <= current step.
    hierarchy_by_cycle: dict[int, dict[int, str]] = (
        {1: dict(l0_by_step)} if stores_hierarchy else {}
    )
    branching_factor = _coerce_branching_factor(hierarchy_branching_factor, hierarchy_k)
    for key, path in file_map.items():
        if not isinstance(key, str) or not key.startswith("L_"):
            continue
        try:
            level = int(key.split("_", 1)[1])
        except Exception:
            continue
        if level <= 0:
            continue
        if not isinstance(branching_factor, int) or branching_factor <= 1:
            continue
        cycle = branching_factor**level
        p = _resolve_path(path)
        if not p.exists():
            continue
        rows = read_jsonl(p)
        by_step: dict[int, str] = {}
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
            by_step[step] = str(chunk)
        if by_step:
            hierarchy_by_cycle[cycle] = by_step

    steps_rows = read_jsonl(steps_file)
    steps_rows = [r for r in steps_rows if isinstance(r, dict)]

    def _step_sort_key(row: dict[str, Any]) -> int:
        try:
            return int(row.get("step", row.get("chunk_index", -1)))
        except Exception:
            return -1

    steps_rows.sort(key=_step_sort_key)
    if not steps_rows:
        return []

    entity_memory: list[EntityRecord] = []
    out: list[CanonicalizedChunkRecord] = []
    # Original chunk text per step, so a restored state can show the model raw
    # text as L0 history while the anchored copies in L_0 remain the fold inputs.
    raw_text_by_step: dict[int, str] = {}

    for row in steps_rows:
        try:
            step = int(row.get("step", row.get("chunk_index")))
        except Exception:
            continue
        if step not in l0_by_step:
            continue
        raw_text_by_step[step] = str(row.get("raw_chunk_text", "") or "")

        active_raw = row.get("active_entities", []) or []
        if not isinstance(active_raw, list):
            active_raw = []
        active_entities = [e for e in (_entity_from_any(x) for x in active_raw) if e is not None]

        entity_memory = merge_active_entities(entity_memory, active_entities)

        hierarchy_snapshot: dict[int, dict[int, str]] = {}
        for cycle, by_step in hierarchy_by_cycle.items():
            included = {int(s): str(t) for s, t in by_step.items() if int(s) <= step}
            if included:
                hierarchy_snapshot[int(cycle)] = included

        snapshot = {
            "prev_chunk_text": l0_by_step.get(step, ""),
            "entity_memory": [e.to_dict() for e in entity_memory],
            "active_entities": [e.to_dict() for e in active_entities],
            "hierarchy_summaries": hierarchy_snapshot,
            "raw_chunk_texts": dict(raw_text_by_step) if stores_hierarchy else {},
        }
        out.append(
            CanonicalizedChunkRecord(
                doc_id=str(row.get("doc_id", doc_id) or doc_id),
                chunk_id=str(row.get("chunk_id", f"{doc_id}:{step}") or f"{doc_id}:{step}"),
                chunk_index=int(step),
                raw_chunk_text=str(row.get("raw_chunk_text", "") or ""),
                canonicalized_chunk_text=l0_by_step.get(step, ""),
                state_snapshot=snapshot,
                strategy_name=str(row.get("strategy_name", strategy_name_fallback or "") or ""),
                parser_status=str(row.get("parser_status", "ok") or "ok"),  # type: ignore[arg-type]
                raw_model_text=str(row.get("raw_model_text", "") or ""),
                linking_sentence=str(row.get("linking_sentence", "") or ""),
                insertions=[x for x in (_insertion_from_any(item) for item in (row.get("insertions", []) or [])) if x is not None],
                rejected_entities=[e for e in (_entity_from_any(item) for item in (row.get("rejected_entities", []) or [])) if e is not None],
            )
        )
    return out

@_accept_legacy_entity_cache_keywords
def load_llm_history_cache_index(
    *,
    output_dir: str | Path,
    dataset_label: str | None,
    method_label: str | None,
    hierarchy_branching_factor: int | None = None,
    hierarchy_k: int | None = None,
    chunk_size: int | None = None,
    active_entity_config_label: str | None = None,
    llm_label: str | None = None,
    active_entity_selector_label: str | None = None,
) -> dict[str, Any] | None:
    branching_factor = _coerce_branching_factor(hierarchy_branching_factor, hierarchy_k)
    cache_dir = resolve_llm_cache_namespace_dir(
        output_dir,
        dataset_label=dataset_label,
        method_label=method_label,
        hierarchy_branching_factor=branching_factor,
        chunk_size=chunk_size,
        active_entity_config_label=active_entity_config_label,
        llm_label=llm_label,
        active_entity_selector_label=active_entity_selector_label,
    )
    index_path = cache_dir / "index.json"
    if not index_path.exists():
        legacy_config_label, legacy_selector_label = legacy_active_entity_cache_labels(
            active_entity_config_label,
            active_entity_selector_label,
        )
        candidates = [
            resolve_llm_cache_namespace_dir(
                output_dir,
                dataset_label=dataset_label,
                method_label=method_label,
                hierarchy_branching_factor=branching_factor,
                chunk_size=chunk_size,
                active_entity_config_label=legacy_config_label,
                llm_label=llm_label,
                active_entity_selector_label=legacy_selector_label,
            ),
            _legacy_llm_cache_namespace_dir(
                output_dir,
                dataset_label=dataset_label,
                method_label=method_label,
                branching_factor=branching_factor,
                chunk_size=chunk_size,
                active_entity_config_label=active_entity_config_label,
                llm_label=llm_label,
                active_entity_selector_label=active_entity_selector_label,
            ),
            _legacy_llm_cache_namespace_dir(
                output_dir,
                dataset_label=dataset_label,
                method_label=method_label,
                branching_factor=branching_factor,
                chunk_size=chunk_size,
                active_entity_config_label=legacy_config_label,
                llm_label=llm_label,
                active_entity_selector_label=legacy_selector_label,
            ),
        ]
        for candidate_dir in candidates:
            candidate_index = candidate_dir / "index.json"
            if candidate_index.exists():
                index_path = candidate_index
                break
    if not index_path.exists():
        return None
    with index_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return None
    # The on-disk index may retain its original AO-era location for provenance.
    # Runtime resolution must use the directory in which the index was found.
    data["index_path"] = str(index_path)
    data["cache_namespace_dir"] = str(index_path.parent)
    return data


def load_raw_chunk_records_from_cache_index(index_path: str | Path) -> list[ChunkRecord]:
    """
    Load raw chunk records from v2 llm cache `index.json` using per-doc `steps_index.jsonl`.
    This allows retrieval-only runs without `raw_chunks.jsonl`.
    """
    p = Path(index_path).resolve()
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        index = json.load(f)
    if not isinstance(index, dict):
        return []
    docs = index.get("docs", []) or []
    if not isinstance(docs, list):
        return []

    out: list[ChunkRecord] = []
    base_dir = p.parent
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        default_doc_id = str(doc.get("doc_id", "") or "")
        files = doc.get("files", {}) or {}
        if not isinstance(files, dict):
            continue
        steps_path = files.get("steps_index")
        if not steps_path:
            continue
        steps_file = _resolve_cache_file_path(steps_path, base_dir=base_dir)
        if not steps_file.exists():
            continue
        rows = read_jsonl(steps_file)
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                step = int(row.get("step", row.get("chunk_index")))
            except Exception:
                continue
            raw_chunk_text = row.get("raw_chunk_text")
            if raw_chunk_text is None:
                continue
            doc_id = str(row.get("doc_id", default_doc_id) or default_doc_id)
            chunk_id = str(row.get("chunk_id", f"{doc_id}:{step}") or f"{doc_id}:{step}")
            out.append(
                ChunkRecord(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    chunk_index=int(step),
                    raw_text=str(raw_chunk_text),
                )
            )

    out.sort(key=lambda c: (str(c.doc_id), int(c.chunk_index), str(c.chunk_id)))
    return out


def load_doc_ids_from_cache_index(index_path: str | Path) -> list[str]:
    p = Path(index_path).resolve()
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        index = json.load(f)
    if not isinstance(index, dict):
        return []
    docs = index.get("docs", []) or []
    if not isinstance(docs, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        doc_id = str(doc.get("doc_id", "") or "").strip()
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        out.append(doc_id)
    return out


def _canonicalized_chunk_from_dict(row: dict[str, Any]) -> CanonicalizedChunkRecord | None:
    try:
        return CanonicalizedChunkRecord(
            doc_id=str(row["doc_id"]),
            chunk_id=str(row["chunk_id"]),
            chunk_index=int(row["chunk_index"]),
            raw_chunk_text=str(row.get("raw_chunk_text", "") or ""),
            canonicalized_chunk_text=str(row.get("canonicalized_chunk_text", "") or ""),
            state_snapshot=row.get("state_snapshot", {}) or {},
            strategy_name=str(row.get("strategy_name", "") or ""),
            parser_status=str(row.get("parser_status", "ok") or "ok"),  # type: ignore[arg-type]
            raw_model_text=str(row.get("raw_model_text", "") or ""),
            linking_sentence=str(row.get("linking_sentence", "") or ""),
            insertions=[x for x in (_insertion_from_any(item) for item in (row.get("insertions", []) or [])) if x is not None],
            rejected_entities=[e for e in (_entity_from_any(item) for item in (row.get("rejected_entities", []) or [])) if e is not None],
        )
    except Exception:
        return None


@_accept_legacy_entity_cache_keywords
def load_canonicalized_chunk_records_cache(
    *,
    output_dir: str | Path,
    dataset_label: str | None,
    method_label: str | None,
    hierarchy_branching_factor: int | None = None,
    hierarchy_k: int | None = None,
    chunk_size: int | None = None,
    active_entity_config_label: str | None = None,
    llm_label: str | None = None,
    active_entity_selector_label: str | None = None,
) -> list[CanonicalizedChunkRecord]:
    # v2 cache layout: per-doc files (L_0/L_n/memory/steps_index) described by index.json.
    index = load_llm_history_cache_index(
        output_dir=output_dir,
        dataset_label=dataset_label,
        method_label=method_label,
        hierarchy_branching_factor=hierarchy_branching_factor,
        hierarchy_k=hierarchy_k,
        chunk_size=chunk_size,
        active_entity_config_label=active_entity_config_label,
        llm_label=llm_label,
        active_entity_selector_label=active_entity_selector_label,
    )
    if isinstance(index, dict):
        docs = index.get("docs", []) or []
        if isinstance(docs, list):
            out_v2: list[CanonicalizedChunkRecord] = []
            b_from_index = index.get(
                "branching_factor",
                index.get("hierarchy_branching_factor", index.get("hierarchy_k", index.get("k"))),
            )
            b_for_restore = _coerce_branching_factor(b_from_index, hierarchy_branching_factor, hierarchy_k)
            stores_hierarchy = bool(index.get("stores_hierarchy", True))
            base_dir = Path(str(index.get("cache_namespace_dir") or "")).resolve() if index.get("cache_namespace_dir") else resolve_llm_cache_namespace_dir(
                output_dir,
                dataset_label=dataset_label,
                method_label=method_label,
                hierarchy_branching_factor=b_for_restore,
                chunk_size=chunk_size,
                active_entity_config_label=active_entity_config_label,
                llm_label=llm_label,
                active_entity_selector_label=active_entity_selector_label,
            )
            for doc in docs:
                if not isinstance(doc, dict):
                    continue
                doc_id = str(doc.get("doc_id", "") or "")
                files = doc.get("files", {}) or {}
                if not doc_id or not isinstance(files, dict):
                    continue
                out_v2.extend(
                    _load_v2_doc_records(
                        doc_id=doc_id,
                        file_map=files,
                        hierarchy_branching_factor=b_for_restore,
                        strategy_name_fallback=str(index.get("strategy_name", "") or ""),
                        stores_hierarchy=stores_hierarchy,
                        base_dir=base_dir,
                    )
                )
            if out_v2:
                out_v2.sort(key=lambda r: (str(r.doc_id), int(r.chunk_index), str(r.chunk_id)))
                return out_v2

    # Legacy fallback: monolithic canonicalized_chunks_cache.jsonl.
    path = resolve_canonicalized_chunk_cache_path(
        output_dir,
        dataset_label=dataset_label,
        method_label=method_label,
        hierarchy_branching_factor=hierarchy_branching_factor,
        hierarchy_k=hierarchy_k,
        chunk_size=chunk_size,
        active_entity_config_label=active_entity_config_label,
        llm_label=llm_label,
        active_entity_selector_label=active_entity_selector_label,
    )
    if not path.exists():
        legacy_config_label, legacy_selector_label = legacy_active_entity_cache_labels(
            active_entity_config_label,
            active_entity_selector_label,
        )
        candidate_dirs = [
            resolve_llm_cache_namespace_dir(
                output_dir,
                dataset_label=dataset_label,
                method_label=method_label,
                hierarchy_branching_factor=hierarchy_branching_factor,
                hierarchy_k=hierarchy_k,
                chunk_size=chunk_size,
                active_entity_config_label=legacy_config_label,
                llm_label=llm_label,
                active_entity_selector_label=legacy_selector_label,
            ),
            _legacy_llm_cache_namespace_dir(
                output_dir,
                dataset_label=dataset_label,
                method_label=method_label,
                branching_factor=_coerce_branching_factor(hierarchy_branching_factor, hierarchy_k),
                chunk_size=chunk_size,
                active_entity_config_label=legacy_config_label,
                llm_label=llm_label,
                active_entity_selector_label=legacy_selector_label,
            ),
        ]
        for candidate_dir in candidate_dirs:
            candidate_path = candidate_dir / "canonicalized_chunks_cache.jsonl"
            if candidate_path.exists():
                path = candidate_path
                break
        if not path.exists():
            return []
    rows = read_jsonl(path)
    out: list[CanonicalizedChunkRecord] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        rec = _canonicalized_chunk_from_dict(row)
        if rec is not None:
            out.append(rec)
    return out


@_accept_legacy_entity_cache_keywords
def write_canonicalized_chunk_records_cache(
    records: list[CanonicalizedChunkRecord],
    *,
    output_dir: str | Path,
    dataset_label: str | None,
    method_label: str | None,
    hierarchy_branching_factor: int | None = None,
    hierarchy_k: int | None = None,
    chunk_size: int | None = None,
    active_entity_config_label: str | None = None,
    llm_label: str | None = None,
    active_entity_selector_label: str | None = None,
) -> str:
    path = resolve_canonicalized_chunk_cache_path(
        output_dir,
        dataset_label=dataset_label,
        method_label=method_label,
        hierarchy_branching_factor=hierarchy_branching_factor,
        hierarchy_k=hierarchy_k,
        chunk_size=chunk_size,
        active_entity_config_label=active_entity_config_label,
        llm_label=llm_label,
        active_entity_selector_label=active_entity_selector_label,
    )
    rows = [
        r.to_dict()
        for r in sorted(records, key=lambda x: (str(x.doc_id), int(x.chunk_index), str(x.chunk_id)))
    ]
    return write_jsonl(path, rows)


@_accept_legacy_entity_cache_keywords
def export_llm_history_cache(
    canonicalized_chunks: list[CanonicalizedChunkRecord],
    *,
    output_dir: str | Path,
    dataset_label: str | None = None,
    method_label: str | None = None,
    strategy_name: str | None = None,
    hierarchy_branching_factor: int | None = None,
    hierarchy_k: int | None = None,
    chunk_size: int | None = None,
    active_entity_config_label: str | None = None,
    llm_label: str | None = None,
    active_entity_selector_label: str | None = None,
    active_entity_min_relevance: float | None = None,
    memory_update_mode: str | None = None,
    stores_hierarchy: bool = True,
    history_l0_text: str | None = None,
    anchor_matching: str | None = None,
) -> dict[str, Any]:
    root = Path(output_dir) / "llm_cache"
    branching_factor = _coerce_branching_factor(hierarchy_branching_factor, hierarchy_k)
    cache_dir = resolve_llm_cache_namespace_dir(
        output_dir,
        dataset_label=dataset_label,
        method_label=method_label or strategy_name,
        hierarchy_branching_factor=branching_factor,
        chunk_size=chunk_size,
        active_entity_config_label=active_entity_config_label,
        llm_label=llm_label,
        active_entity_selector_label=active_entity_selector_label,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[CanonicalizedChunkRecord]] = {}
    for rec in canonicalized_chunks:
        grouped.setdefault(str(rec.doc_id), []).append(rec)

    used_doc_dirs: set[str] = set()
    docs_index: list[dict[str, Any]] = []

    for doc_id, recs in grouped.items():
        recs_sorted = sorted(recs, key=lambda r: (int(r.chunk_index), str(r.chunk_id)))

        base = _safe_path_component(doc_id)
        doc_dir_name = base
        suffix = 1
        while doc_dir_name in used_doc_dirs:
            suffix += 1
            doc_dir_name = f"{base}_{suffix}"
        used_doc_dirs.add(doc_dir_name)
        doc_dir = cache_dir / doc_dir_name

        level_rows: dict[int, list[dict[str, Any]]] = {0: []}
        prev_hier_max_step: dict[int, int] = {}
        steps_index_rows: list[dict[str, Any]] = []

        for rec in recs_sorted:
            step = int(rec.chunk_index)
            level_rows[0].append({"doc_id": doc_id, "chunk": rec.canonicalized_chunk_text, "step": step})
            active_entities = (rec.state_snapshot or {}).get("active_entities", []) or []
            if not isinstance(active_entities, list):
                active_entities = []
            steps_index_rows.append(
                {
                    "doc_id": doc_id,
                    "step": step,
                    "chunk_id": str(rec.chunk_id),
                    "raw_chunk_text": str(rec.raw_chunk_text),
                    "strategy_name": str(rec.strategy_name),
                    "parser_status": str(rec.parser_status),
                    "raw_model_text": str(rec.raw_model_text or ""),
                    "insertions": [x.to_dict() for x in rec.insertions],
                    "active_entities": active_entities,
                    "rejected_entities": [e.to_dict() for e in rec.rejected_entities],
                    "rejected_entity_count": len(rec.rejected_entities),
                }
            )

            cur_hier = _normalize_hierarchy_snapshot(rec.state_snapshot or {})
            for cycle_size, by_step in sorted(cur_hier.items()):
                level = _cycle_to_level(cycle_size, branching_factor)
                if level is None:
                    continue
                prev_step = prev_hier_max_step.get(cycle_size, -1)
                for h_step in sorted(by_step.keys()):
                    if h_step <= prev_step:
                        continue
                    if h_step > step:
                        continue
                    level_rows.setdefault(level, []).append({"doc_id": doc_id, "chunk": by_step[h_step], "step": h_step})
                    prev_step = h_step
                prev_hier_max_step[cycle_size] = prev_step

        file_map: dict[str, str] = {}
        for level in sorted(level_rows):
            rows = level_rows[level]
            if level > 0 and not rows:
                continue
            path = doc_dir / f"L_{level}.jsonl"
            write_jsonl(path, rows)
            file_map[f"L_{level}"] = str(path)

        steps_index_path = doc_dir / "steps_index.jsonl"
        write_jsonl(steps_index_path, steps_index_rows)
        file_map["steps_index"] = str(steps_index_path)

        final_snapshot = recs_sorted[-1].state_snapshot if recs_sorted else {}
        final_memory_raw = (final_snapshot or {}).get("entity_memory", []) or []
        if not isinstance(final_memory_raw, list):
            final_memory_raw = []
        final_memory_entities = [
            entity.to_dict()
            for entity in (_entity_from_any(item) for item in final_memory_raw)
            if entity is not None
        ]
        rejected_entity_count = sum(len(rec.rejected_entities) for rec in recs_sorted)
        memory_path = doc_dir / "memory.json"
        with memory_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "doc_id": doc_id,
                    "total_chunks": len(recs_sorted),
                    "final_chunk_id": str(recs_sorted[-1].chunk_id) if recs_sorted else None,
                    "final_chunk_index": int(recs_sorted[-1].chunk_index) if recs_sorted else None,
                    "entities": final_memory_entities,
                    "rejected_entity_count": rejected_entity_count,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        file_map["memory"] = str(memory_path)

        docs_index.append(
            {
                "doc_id": doc_id,
                "doc_cache_dir": str(doc_dir),
                "files": file_map,
                "rejected_entity_count": rejected_entity_count,
            }
        )

    index = {
        "format_version": 2,
        "cache_root_dir": str(root),
        "cache_namespace_dir": str(cache_dir),
        "cache_layout": "hierarchy_files_with_steps_index",
        "dataset": dataset_label or "unknown_dataset",
        "method": (method_label or strategy_name or "unknown_method"),
        "branching_factor": branching_factor,
        "chunk_size": chunk_size,
        "active_entity_config": active_entity_config_label,
        # Compatibility metadata for older report readers. Remove only after
        # all archived-analysis consumers have migrated.
        "active_object_config": active_entity_config_label,
        "llm": llm_label or "unknown_llm",
        "active_entity_selector": active_entity_selector_label,
        "active_object_min_relevance": active_entity_min_relevance,
        "active_object_selector": active_entity_selector_label,
        "active_entity_min_relevance": active_entity_min_relevance,
        "strategy_name": strategy_name,
        "memory_update_mode": memory_update_mode,
        # Which text the model saw as L0 history: "raw" (original chunks) or
        # "anchored" (copies with the canonical name placed after each anchor).
        # Caches written before this field existed were built with "anchored".
        "history_l0_text": history_l0_text,
        # How proposed anchors were located: "strict" or "tolerant". Caches
        # written before this field existed used strict matching.
        "anchor_matching": anchor_matching,
        "stores_hierarchy": bool(stores_hierarchy),
        "hierarchy_branching_factor": branching_factor,
        "rejected_entity_count": sum(
            int(doc.get("rejected_entity_count", 0) or 0) for doc in docs_index
        ),
        "docs": docs_index,
    }
    index_path = cache_dir / "index.json"
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    index["index_path"] = str(index_path)
    return index


def load_canonicalized_chunk_records_from_cache_index(index_path: str | Path) -> list[CanonicalizedChunkRecord]:
    """
    Load canonicalized chunk records from v2 llm cache `index.json`.

    L0 text is read from each doc's `L_0.jsonl`; `steps_index.jsonl` provides
    chunk_id/raw text and step metadata.
    """
    p = Path(index_path).resolve()
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        index = json.load(f)
    if not isinstance(index, dict):
        return []
    docs = index.get("docs", []) or []
    if not isinstance(docs, list):
        return []

    branching_factor_from_index = index.get(
        "branching_factor",
        index.get("hierarchy_branching_factor", index.get("hierarchy_k", index.get("k"))),
    )
    branching_factor_for_restore = _coerce_branching_factor(branching_factor_from_index)
    strategy_name_fallback = str(index.get("strategy_name", "") or "")
    stores_hierarchy = bool(index.get("stores_hierarchy", True))

    out: list[CanonicalizedChunkRecord] = []
    base_dir = p.parent
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        doc_id = str(doc.get("doc_id", "") or "")
        file_map = doc.get("files", {}) or {}
        if not doc_id or not isinstance(file_map, dict):
            continue
        out.extend(
            _load_v2_doc_records(
                doc_id=doc_id,
                file_map=file_map,
                hierarchy_branching_factor=branching_factor_for_restore,
                strategy_name_fallback=strategy_name_fallback,
                stores_hierarchy=stores_hierarchy,
                base_dir=base_dir,
            )
        )

    out.sort(key=lambda r: (str(r.doc_id), int(r.chunk_index), str(r.chunk_id)))
    return out
