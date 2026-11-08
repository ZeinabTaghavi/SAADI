"""The per-document ledger: records, stubs, name lookup, and the event log.

A ledger entry is one name in memory. ``kind`` is ``record`` (described by the
LLM at least once) or ``stub`` (a candidate marked NEW but never described:
name, chunk id, one sentence; no description, no budget). A stub matched by a
later SAME AS decision is promoted to a record; its sentence becomes its first
description.

Every change is an *op* applied through :meth:`DocumentLedger.apply_op`, both
live and when the event log is replayed. One event per processed chunk is
appended to ``<store_dir>/<doc_id>/events.jsonl``; the build can therefore
resume from the construction cache and the measurement can rebuild the ledger
exactly, step by step. The entity memory itself stays in SAADI's own state
(stubs are entity records with no description), so the published loaders,
``chunk_to_names`` and the query-time scoring work unchanged.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..entities.active_selection import tokenize_active_entity_text

# Titles stripped before names are compared (the probe's list plus common forms).
TITLES = frozenset(
    {
        "mr",
        "mrs",
        "miss",
        "ms",
        "mx",
        "dr",
        "doctor",
        "captain",
        "capt",
        "professor",
        "prof",
        "father",
        "mother",
        "judge",
        "madame",
        "madam",
        "mme",
        "mlle",
        "mademoiselle",
        "monsieur",
        "sergeant",
        "colonel",
        "major",
        "general",
        "sir",
        "lady",
        "lord",
        "master",
        "mistress",
        "reverend",
        "rev",
        "saint",
        "st",
        "count",
        "countess",
        "duke",
        "duchess",
        "prince",
        "princess",
        "king",
        "queen",
        "baron",
        "baroness",
        "earl",
        "senor",
        "senora",
        "senorita",
        "signor",
        "signora",
        "herr",
        "frau",
        "don",
        "dona",
        "uncle",
        "aunt",
        "cousin",
        "widow",
        "squire",
        "lieutenant",
        "admiral",
        "commander",
        "sheriff",
        "deacon",
        "bishop",
        "dame",
        "brother",
        "sister",
    }
)

_WS = re.compile(r"\s+")


def name_key(name: str) -> str:
    """Case-folded, whitespace-normalized key (the memory's own identity rule)."""

    return _WS.sub(" ", str(name or "").strip()).casefold()


def name_tokens(text: str) -> tuple[str, ...]:
    """Unicode-aware, case-folded, stopword-free tokens (the active-memory tokenizer)."""

    return tuple(tokenize_active_entity_text(str(text or "")))


def stripped_tokens(text: str) -> tuple[str, ...]:
    """Name tokens with titles removed; the basis of candidate matching."""

    return tuple(t for t in name_tokens(text) if t not in TITLES)


def title_tokens(text: str) -> tuple[str, ...]:
    return tuple(t for t in name_tokens(text) if t in TITLES)


def is_alias_shaped(text: str) -> bool:
    """An alias is a name or a title plus a name, never a descriptive phrase.

    Rule: at most 5 words, contains a capital letter, and is not only titles.
    """

    words = str(text or "").split()
    if not words or len(words) > 5:
        return False
    if not any(ch.isalpha() and ch.isupper() for ch in text):
        return False
    return bool(stripped_tokens(text))


def match_kind(candidate: str, name: str) -> str | None:
    """How a candidate matches a stored name or alias after title stripping.

    ``exact``: same token set. ``partial``: one token set contains the other
    (e.g. "Linton" against "Mr. Edgar Linton"). ``None``: no match.
    """

    c = stripped_tokens(candidate)
    n = stripped_tokens(name)
    if not c or not n:
        return None
    cs, ns = set(c), set(n)
    if cs == ns:
        return "exact"
    if cs <= ns or ns <= cs:
        return "partial"
    return None


def description_strength(text: str) -> int:
    """The "weakest" description has the fewest content tokens (ties: the oldest)."""

    return len(name_tokens(text))


@dataclass
class LedgerEntry:
    id: str
    name: str
    kind: str  # "record" | "stub"
    created_step: int
    created_chunk_id: str
    sentence: str = ""
    promoted_step: int | None = None
    promoted_chunk_id: str | None = None
    provenance: list[str] = field(default_factory=list)  # candidate_net | llm | parse_failure | recovered
    aliases: list[str] = field(default_factory=list)
    alias_steps: dict[str, int] = field(default_factory=dict)
    linked_chunks: list[str] = field(default_factory=list)
    described_steps: list[int] = field(default_factory=list)
    same_as_count: int = 0
    dropped_descriptions: list[str] = field(default_factory=list)
    last_step: int = -1

    def all_names(self) -> list[str]:
        return [self.name, *self.aliases]

    def is_stub_at(self, step: int) -> bool:
        if self.created_step > step:
            return False
        return self.promoted_step is None or self.promoted_step > step

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> LedgerEntry:
        return cls(**{k: row[k] for k in cls.__dataclass_fields__ if k in row})  # type: ignore[arg-type]


class DocumentLedger:
    """Records and stubs of one document, rebuilt exactly from its event log."""

    def __init__(self, doc_id: str) -> None:
        self.doc_id = str(doc_id)
        self.entries: dict[str, LedgerEntry] = {}
        self.by_key: dict[str, str] = {}  # name key -> id
        self.alias_index: dict[str, set[str]] = {}  # alias key -> ids
        self.events: list[dict[str, Any]] = []
        self.counters: Counter = Counter()
        self._next = 1

    # -- lookup --------------------------------------------------------------
    @property
    def last_step(self) -> int:
        return int(self.events[-1]["step"]) if self.events else -1

    def get(self, entry_id: str) -> LedgerEntry | None:
        return self.entries.get(str(entry_id))

    def by_name(self, name: str) -> LedgerEntry | None:
        """Entry whose name is exactly this name (casefold)."""

        entry_id = self.by_key.get(name_key(name))
        return self.entries.get(entry_id) if entry_id else None

    def by_name_or_alias(self, text: str) -> LedgerEntry | None:
        entry = self.by_name(text)
        if entry is not None:
            return entry
        ids = sorted(self.alias_index.get(name_key(text), ()))
        return self.entries[ids[0]] if ids else None

    def matches(self, candidate: str) -> list[tuple[LedgerEntry, str]]:
        """Entries whose name or alias matches the candidate after title stripping."""

        out: list[tuple[LedgerEntry, str]] = []
        for entry in self.entries.values():
            best: str | None = None
            for stored in entry.all_names():
                kind = match_kind(candidate, stored)
                if kind == "exact":
                    best = "exact"
                    break
                if kind == "partial":
                    best = "partial"
            if best:
                out.append((entry, best))
        return out

    def recall(
        self,
        candidates: Iterable[str],
        *,
        exclude_keys: set[str] | None = None,
        cap: int = 5,
    ) -> tuple[list[LedgerEntry], list[LedgerEntry], bool, dict[str, list[str]]]:
        """Ledger recall: entries matching any candidate, outside the overlap slots.

        Ranking: exact matches before partial ones, then the most recently
        touched entry first, then creation order. Returns
        ``(recalled, left_out, cap_hit, matched_candidates_by_id)``.
        """

        exclude = {k for k in (exclude_keys or set())}
        scored: dict[str, tuple[int, LedgerEntry]] = {}
        matched: dict[str, list[str]] = {}
        for cand in candidates:
            for entry, kind in self.matches(cand):
                if name_key(entry.name) in exclude:
                    continue
                rank = 0 if kind == "exact" else 1
                prev = scored.get(entry.id)
                if prev is None or rank < prev[0]:
                    scored[entry.id] = (rank, entry)
                matched.setdefault(entry.id, []).append(cand)
        ordered = sorted(scored.values(), key=lambda item: (item[0], -item[1].last_step, int(item[1].id[1:])))
        entries = [e for _, e in ordered]
        recalled, left_out = entries[: max(0, int(cap))], entries[max(0, int(cap)) :]
        return recalled, left_out, bool(left_out), matched

    # -- ops -----------------------------------------------------------------
    def new_id(self) -> str:
        entry_id = f"E{self._next:04d}"
        self._next += 1
        return entry_id

    def apply_op(self, op: dict[str, Any]) -> None:
        """Apply one op (live or replay). Ops are plain dicts so the log is the source of truth."""

        kind = op["op"]
        if kind == "create":
            entry = LedgerEntry(
                id=str(op["id"]),
                name=str(op["name"]),
                kind=str(op["kind"]),
                created_step=int(op["step"]),
                created_chunk_id=str(op.get("chunk_id", "")),
                sentence=str(op.get("sentence", "") or ""),
                provenance=sorted(set(op.get("provenance", []) or [])),
                last_step=int(op["step"]),
            )
            self.entries[entry.id] = entry
            self.by_key[name_key(entry.name)] = entry.id
            try:
                self._next = max(self._next, int(entry.id[1:]) + 1)
            except ValueError:
                pass
            self.counters[f"created_{entry.kind}"] += 1
        elif kind == "alias":
            entry = self.entries[str(op["id"])]
            alias = str(op["alias"])
            if alias not in entry.aliases and name_key(alias) != name_key(entry.name):
                entry.aliases.append(alias)
                entry.alias_steps[alias] = int(op["step"])
                self.alias_index.setdefault(name_key(alias), set()).add(entry.id)
                self.counters["aliases_added"] += 1
            entry.last_step = max(entry.last_step, int(op["step"]))
        elif kind == "promote":
            entry = self.entries[str(op["id"])]
            if entry.kind == "stub":
                entry.kind = "record"
                entry.promoted_step = int(op["step"])
                entry.promoted_chunk_id = str(op.get("chunk_id", "") or "")
                self.counters["promotions"] += 1
            entry.last_step = max(entry.last_step, int(op["step"]))
        elif kind == "link":
            entry = self.entries[str(op["id"])]
            chunk_id = str(op["chunk_id"])
            if chunk_id not in entry.linked_chunks:
                entry.linked_chunks.append(chunk_id)
            entry.last_step = max(entry.last_step, int(op["step"]))
        elif kind == "describe":
            entry = self.entries[str(op["id"])]
            step = int(op["step"])
            if step not in entry.described_steps:
                entry.described_steps.append(step)
            entry.last_step = max(entry.last_step, step)
        elif kind == "same_as":
            entry = self.entries[str(op["id"])]
            entry.same_as_count += 1
        elif kind == "provenance":
            entry = self.entries[str(op["id"])]
            entry.provenance = sorted(set(entry.provenance) | set(op.get("add", [])))
        elif kind == "drop_description":
            entry = self.entries[str(op["id"])]
            text = str(op["text"])
            if text not in entry.dropped_descriptions:
                entry.dropped_descriptions.append(text)
            self.counters["description_replacements"] += 1
        else:
            raise ValueError(f"Unknown ledger op {kind!r}")

    def apply_event(self, event: dict[str, Any]) -> None:
        for op in event.get("ops", []) or []:
            self.apply_op(op)
        for key, value in (event.get("counts") or {}).items():
            self.counters[key] += int(value)
        self.events.append(event)

    def replay(self, events: Iterable[dict[str, Any]]) -> None:
        for event in sorted(events, key=lambda e: int(e["step"])):
            self.apply_event(event)

    def truncate(self, step: int) -> int:
        """Forget every event at or after ``step`` (that chunk is being regenerated)."""

        kept = [e for e in self.events if int(e["step"]) < int(step)]
        removed = len(self.events) - len(kept)
        if removed:
            self.entries.clear()
            self.by_key.clear()
            self.alias_index.clear()
            self.counters = Counter()
            self.events = []
            self._next = 1
            self.replay(kept)
        return removed

    # -- views -----------------------------------------------------------------
    def kinds_at(self, step: int) -> dict[str, str]:
        """Kind of every entry that existed at ``step`` (after that step's update), keyed by name key."""

        out: dict[str, str] = {}
        for entry in self.entries.values():
            if entry.created_step <= step:
                out[name_key(entry.name)] = "stub" if entry.is_stub_at(step) else "record"
        return out

    def snapshot(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "n_events": len(self.events),
            "last_step": self.last_step,
            "counters": dict(self.counters),
            "entries": [e.to_dict() for e in sorted(self.entries.values(), key=lambda e: int(e.id[1:]))],
        }


class LedgerStore:
    """Per-document ledgers under one directory: ``<root>/<doc_id>/events.jsonl`` and ``ledger.json``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._docs: dict[str, DocumentLedger] = {}

    def _safe(self, doc_id: str) -> str:
        return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(doc_id)) or "doc"

    def doc_dir(self, doc_id: str) -> Path:
        return self.root / self._safe(doc_id)

    def events_path(self, doc_id: str) -> Path:
        return self.doc_dir(doc_id) / "events.jsonl"

    def ledger_path(self, doc_id: str) -> Path:
        return self.doc_dir(doc_id) / "ledger.json"

    def read_events(self, doc_id: str) -> list[dict[str, Any]]:
        path = self.events_path(doc_id)
        if not path.exists():
            return []
        out: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def doc(self, doc_id: str) -> DocumentLedger:
        ledger = self._docs.get(str(doc_id))
        if ledger is None:
            ledger = DocumentLedger(doc_id)
            ledger.replay(self.read_events(doc_id))
            self._docs[str(doc_id)] = ledger
        return ledger

    def truncate(self, doc_id: str, step: int) -> int:
        ledger = self.doc(doc_id)
        removed = ledger.truncate(step)
        if removed:
            self._rewrite(doc_id, ledger)
        return removed

    def _rewrite(self, doc_id: str, ledger: DocumentLedger) -> None:
        path = self.events_path(doc_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            for event in ledger.events:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        tmp.replace(path)
        self._write_snapshot(doc_id, ledger)

    def _write_snapshot(self, doc_id: str, ledger: DocumentLedger) -> None:
        path = self.ledger_path(doc_id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(ledger.snapshot(), ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)

    def append_event(self, doc_id: str, event: dict[str, Any]) -> None:
        """Apply the event to the live ledger and persist it (one line, then the snapshot)."""

        ledger = self.doc(doc_id)
        ledger.apply_event(event)
        path = self.events_path(doc_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._write_snapshot(doc_id, ledger)

    def doc_ids(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir() if (p / "events.jsonl").exists())


__all__ = [
    "TITLES",
    "DocumentLedger",
    "LedgerEntry",
    "LedgerStore",
    "description_strength",
    "is_alias_shaped",
    "match_kind",
    "name_key",
    "name_tokens",
    "stripped_tokens",
    "title_tokens",
]
