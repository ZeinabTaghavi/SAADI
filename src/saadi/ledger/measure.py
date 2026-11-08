"""Measurement helpers for the probe steps: stubs as records, creation and promotion steps.

The probe's step 6-8 code (experiments/probe_bridge, src/saadi/rebuttal/frozen_records.py)
is reused by import; this module adds what the ledger profile needs:

- :func:`load_ledger`: the exact ledger of a document, replayed from its event log.
- :func:`ledger_frozen_chunk_fields`: the E2 frozen-records fields plus, for every
  chunk, which of its linked names were still stubs at that step, with their
  creation and promotion steps.
- :func:`verify_ledger_against_replay`: the ledger's steps agree with the
  construction records (name first seen at ``created_step``; a promoted stub's
  sentence first appears at ``promoted_step``).
- :func:`recalled_at_step`: what was in the prompt at a step (both groups).
- :func:`same_as_decisions`, :func:`provenance_counts`, :func:`title_conflicts`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..rebuttal.frozen_records import FrozenChunk, ReplayedDocument, frozen_chunk_fields
from .ledger import DocumentLedger, LedgerStore, name_key, stripped_tokens, title_tokens


@dataclass
class StubAnnotation:
    name: str
    entry_id: str
    created_step: int
    promoted_step: int | None
    stub_at_chunk_step: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_store(store_dir: str | Path) -> LedgerStore:
    return LedgerStore(Path(store_dir))


def load_ledger(store_dir: str | Path, doc_id: str) -> DocumentLedger:
    return load_store(store_dir).doc(doc_id)


def ledger_frozen_chunk_fields(
    document_index: Any,
    replay: ReplayedDocument,
    ledger: DocumentLedger,
) -> tuple[dict[str, FrozenChunk], dict[str, list[StubAnnotation]]]:
    """E2 frozen fields (unchanged) plus stub annotations per chunk.

    A linked name that was a stub at the chunk's step contributes only its name
    to the frozen record (no alias, no description), exactly what the replay
    yields; the annotation makes that explicit and dates the promotion.
    """

    frozen = frozen_chunk_fields(document_index, replay)
    annotations: dict[str, list[StubAnnotation]] = {}
    for chunk_id, row in frozen.items():
        rows: list[StubAnnotation] = []
        for name in row.linked_names:
            entry = ledger.by_name(name)
            if entry is None:
                continue
            rows.append(
                StubAnnotation(
                    name=name,
                    entry_id=entry.id,
                    created_step=entry.created_step,
                    promoted_step=entry.promoted_step,
                    stub_at_chunk_step=entry.is_stub_at(row.step),
                )
            )
        annotations[chunk_id] = rows
    return frozen, annotations


def apply_description_cap(document_index: Any, ledger: DocumentLedger) -> dict[str, int]:
    """Remove from a loaded index the descriptions the ledger logged as replaced.

    The construction cache rebuilds a document's memory by merging every step's
    active entities (uncapped), so a pure-cache rerun re-exports memory.json with
    the replaced descriptions. The ledger is the source of truth for the cap:
    this filter restores the capped memory before any scoring. Returns counts.
    """

    dropped_by_name: dict[str, set[str]] = {}
    for entry in ledger.entries.values():
        if entry.dropped_descriptions:
            dropped_by_name[entry.name] = set(entry.dropped_descriptions)
    removed = 0
    for name, dropped in dropped_by_name.items():
        if name in document_index.entity_to_descriptions:
            before = list(document_index.entity_to_descriptions[name])
            after = [d for d in before if d not in dropped]
            removed += len(before) - len(after)
            document_index.entity_to_descriptions[name] = after
        for entity in document_index.memory_entities:
            if entity.name == name:
                entity.descriptions = [d for d in entity.descriptions if d not in dropped]
    if removed:
        live = {d for descs in document_index.entity_to_descriptions.values() for d in descs}
        document_index.description_pool = [d for d in document_index.description_pool if d in live]
        for chunk_id, names in document_index.chunk_to_names.items():
            wanted: list[str] = []
            for n in names:
                for d in document_index.entity_to_descriptions.get(n, []):
                    if d not in wanted:
                        wanted.append(d)
            document_index.chunk_to_descriptions[chunk_id] = wanted
        rebuilt: dict[str, list[str]] = {}
        for chunk_id, descs in document_index.chunk_to_descriptions.items():
            for d in descs:
                rebuilt.setdefault(d, []).append(chunk_id)
        document_index.description_to_chunks = rebuilt
    return {"records_with_replacements": len(dropped_by_name), "descriptions_removed_from_index": removed}


def memory_json_drift(memory_entities: list[dict[str, Any]], ledger: DocumentLedger) -> list[dict[str, Any]]:
    """Descriptions present in memory.json that the ledger logged as replaced (should be empty after a live finish)."""

    out: list[dict[str, Any]] = []
    for row in memory_entities:
        entry = ledger.by_name(str(row.get("name", "")))
        if entry is None or not entry.dropped_descriptions:
            continue
        present = [d for d in (row.get("description") or []) if d in set(entry.dropped_descriptions)]
        if present:
            out.append({"id": entry.id, "name": entry.name, "replaced_but_present": present})
    return out


def verify_ledger_against_replay(ledger: DocumentLedger, replay: ReplayedDocument) -> list[str]:
    """Problems found when the ledger and the construction records disagree (empty when consistent)."""

    problems: list[str] = []
    for entry in ledger.entries.values():
        if "recovered" in entry.provenance:
            continue
        first = replay.name_first.get(name_key(entry.name))
        if first is None:
            problems.append(f"{entry.id} {entry.name!r}: in the ledger but never in a step record")
            continue
        if first != entry.created_step:
            problems.append(f"{entry.id} {entry.name!r}: ledger created_step={entry.created_step}, records first see it at {first}")
        if entry.promoted_step is not None and entry.sentence:
            desc_first = replay.desc_first.get((name_key(entry.name), entry.sentence))
            if desc_first is None:
                problems.append(
                    f"{entry.id} {entry.name!r}: promoted at {entry.promoted_step} but its sentence is not a description in the records"
                )
            elif desc_first != entry.promoted_step:
                problems.append(f"{entry.id} {entry.name!r}: promoted_step={entry.promoted_step}, sentence first appears at {desc_first}")
    for key in replay.name_first:
        if ledger.by_name(key) is None:
            problems.append(f"{replay.final_names.get(key, key)!r}: in the step records but not in the ledger")
    return problems


def recalled_at_step(ledger: DocumentLedger, step: int) -> dict[str, list[str]]:
    """Names in the prompt at ``step``: the overlap group, the ledger group, and what the cap left out."""

    for event in ledger.events:
        if int(event["step"]) == int(step):

            def names(ids: list[str]) -> list[str]:
                out = []
                for entry_id in ids:
                    entry = ledger.get(entry_id)
                    out.append(entry.name if entry else entry_id)
                return out

            return {
                "overlap": list(event.get("recalled_overlap_names", [])),
                "ledger": names(event.get("recalled_ledger_ids", [])),
                "left_out": names(event.get("ledger_left_out_ids", [])),
                "cap_hit": bool(event.get("ledger_cap_hit", False)),
                "candidates": [c["text"] for c in event.get("candidates", [])],
                "decisions": list(event.get("decisions", [])),
                "decision_status": event.get("decision_status"),
            }
    return {"overlap": [], "ledger": [], "left_out": [], "cap_hit": False, "candidates": [], "decisions": [], "decision_status": None}


def same_as_decisions(ledger: DocumentLedger) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in ledger.events:
        for d in event.get("decisions", []):
            if d.get("decision") == "SAME_AS" and d.get("resolved_id"):
                rows.append({"doc_id": ledger.doc_id, "step": event["step"], "chunk_id": event["chunk_id"], **d})
    return rows


def provenance_counts(ledger: DocumentLedger) -> dict[str, Any]:
    """Records and stubs by provenance (candidate net, the LLM's own additions, both)."""

    out: Counter = Counter()
    for entry in ledger.entries.values():
        prov = set(entry.provenance)
        has_net, has_llm = "candidate_net" in prov, "llm" in prov
        source = (
            "both"
            if has_net and has_llm
            else "candidate_net"
            if has_net
            else "llm"
            if has_llm
            else "recovered"
            if "recovered" in prov
            else "unknown"
        )
        out[f"{entry.kind}:{source}"] += 1
        out[entry.kind] += 1
        out[source] += 1
    # ledger.counters holds every per-step count plus the op-derived ones (promotions,
    # aliases_added, description_replacements, created_record, created_stub).
    totals = Counter(ledger.counters)
    return {"entries": dict(out), "event_totals": dict(totals), "n_events": len(ledger.events)}


def title_conflicts(ledger: DocumentLedger, *, exclude_tokens: set[str] | None = None) -> list[dict[str, Any]]:
    """Records whose names/aliases share a stripped name but carry different titles (Mr./Mrs. type joins).

    ``exclude_tokens`` removes the invented persona names so the count is about
    the books' real characters.
    """

    exclude = exclude_tokens or set()
    rows: list[dict[str, Any]] = []
    for entry in ledger.entries.values():
        names = entry.all_names()
        if len(names) < 2:
            continue
        groups: dict[tuple[str, ...], set[tuple[str, ...]]] = {}
        for n in names:
            s = stripped_tokens(n)
            if not s or set(s) & exclude:
                continue
            groups.setdefault(s, set()).add(title_tokens(n))
        conflicts = {
            s: t
            for s, t in groups.items()
            if len({x for x in t if x}) >= 2
            or (len(t) >= 2 and any(t) and () in t and len({x for x in t if x}) >= 1 and len([x for x in t if x]) >= 2)
        }
        for s, titles in conflicts.items():
            rows.append(
                {
                    "doc_id": ledger.doc_id,
                    "id": entry.id,
                    "name": entry.name,
                    "aliases": list(entry.aliases),
                    "stripped": " ".join(s),
                    "titles": sorted(" ".join(t) for t in titles),
                }
            )
    return rows


__all__ = [
    "StubAnnotation",
    "apply_description_cap",
    "memory_json_drift",
    "ledger_frozen_chunk_fields",
    "load_ledger",
    "load_store",
    "provenance_counts",
    "recalled_at_step",
    "same_as_decisions",
    "title_conflicts",
    "verify_ledger_against_replay",
]
