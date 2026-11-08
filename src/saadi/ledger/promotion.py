"""Decisions -> ledger ops, stubs, promotions, aliases, chunk links, description cap.

All mutations go through :meth:`DocumentLedger.apply_op`, and every op is
returned so the step event carries it for an exact replay.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from ..core.types import DiscourseState, EntityRecord
from .accounting import NEW, NOT_AN_ENTITY, SAME_AS, DecisionBlock
from .candidate_net import Candidate
from .ledger import DocumentLedger, LedgerEntry, description_strength, is_alias_shaped, match_kind, name_key
from .p1b import P1bStep, entry_title_conflict, filter_aliases, redirect_described, resolve_final_decisions, title_conflict, type_conflict


@dataclass
class StepPlan:
    """What build_prompt decided for one chunk; consumed by apply_output."""

    doc_id: str
    chunk_id: str
    step: int
    candidates: list[Candidate]
    dropped_candidates: list[dict[str, Any]]
    candidate_raw_count: int
    candidate_seconds: float
    candidate_cap_hit: bool
    candidate_device: str
    overlap_ids: list[str]
    overlap_names: list[str]
    ledger_ids: list[str]
    left_out_ids: list[str]
    cap_hit: bool
    matched_candidates: dict[str, list[str]]
    reconcile_ops: list[dict[str, Any]]
    recovered_names: list[str]
    dropped_descriptions_removed: int
    prompt_system: str
    prompt_user: str
    # Profile p1b_ledger only (None on the p1 path): the per-chunk context of its rules.
    p1b: P1bStep | None = None
    recall_matches: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class StepOutcome:
    active_entities: list[EntityRecord]
    ops: list[dict[str, Any]]
    counts: Counter
    decisions: list[dict[str, Any]]
    new_entries: list[dict[str, Any]] = field(default_factory=list)
    promotions: list[str] = field(default_factory=list)
    aliases_added: list[list[str]] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    llm_additions: list[str] = field(default_factory=list)
    described: list[str] = field(default_factory=list)
    redirects: list[dict[str, Any]] = field(default_factory=list)  # p1b
    alias_rejections: list[dict[str, Any]] = field(default_factory=list)  # p1b


def _match_described(candidate: Candidate, described: list[EntityRecord]) -> EntityRecord | None:
    """The described record a NEW candidate refers to: exact name, exact alias, then partial name."""

    best: tuple[int, int, EntityRecord] | None = None
    for idx, rec in enumerate(described):
        rank: int | None = None
        kind = match_kind(candidate.text, rec.name)
        if kind == "exact" or name_key(candidate.text) == name_key(rec.name):
            rank = 0
        elif any(name_key(a) == name_key(candidate.text) or match_kind(candidate.text, a) == "exact" for a in rec.aliases):
            rank = 1
        elif kind == "partial":
            rank = 2
        if rank is not None and (best is None or (rank, idx) < (best[0], best[1])):
            best = (rank, idx, rec)
    return best[2] if best else None


def apply_decisions(
    *,
    ledger: DocumentLedger,
    plan: StepPlan,
    block: DecisionBlock,
    described: list[EntityRecord],
    cut_names: list[str],
) -> StepOutcome:
    """Turn the decision block and the described records into ledger ops and the step's active entities."""

    step, chunk_id = plan.step, plan.chunk_id
    ops: list[dict[str, Any]] = []
    counts: Counter = Counter()
    decisions_log: list[dict[str, Any]] = []
    extra: dict[str, EntityRecord] = {}  # name key -> ledger-derived additions (stubs, aliases, promotions)
    promoted_now: dict[str, str] = {}  # id -> sentence
    new_entries: list[dict[str, Any]] = []
    promotions: list[str] = []
    aliases_added: list[list[str]] = []
    links: list[str] = []
    llm_additions: list[str] = []
    claimed: set[int] = set()  # indices of described records matched by a candidate

    def do(op: dict[str, Any]) -> None:
        ledger.apply_op(op)
        ops.append(op)

    ctx = plan.p1b

    def set_label(entry: LedgerEntry, label: str | None) -> None:
        if ctx is not None and label and entry.id not in ledger.labels:
            do({"op": "label", "id": entry.id, "label": str(label)})

    def create(name: str, kind: str, *, sentence: str = "", provenance: list[str]) -> LedgerEntry:
        entry_id = ledger.new_id()
        do(
            {
                "op": "create",
                "id": entry_id,
                "name": name,
                "kind": kind,
                "step": step,
                "chunk_id": chunk_id,
                "sentence": sentence,
                "provenance": sorted(set(provenance)),
            }
        )
        entry = ledger.get(entry_id)
        assert entry is not None
        new_entries.append({"id": entry_id, "name": name, "kind": kind})
        return entry

    def link(entry: LedgerEntry) -> None:
        if chunk_id not in entry.linked_chunks:
            do({"op": "link", "id": entry.id, "chunk_id": chunk_id, "step": step})
        if entry.id not in links:
            links.append(entry.id)

    def add_alias(entry: LedgerEntry, alias: str) -> bool:
        alias = " ".join(str(alias or "").split())
        if not alias or name_key(alias) == name_key(entry.name) or any(name_key(alias) == name_key(a) for a in entry.aliases):
            return False
        if not is_alias_shaped(alias):
            counts["alias_rejected_descriptive"] += 1
            return False
        do({"op": "alias", "id": entry.id, "alias": alias, "step": step})
        aliases_added.append([entry.id, alias])
        return True

    def promote(entry: LedgerEntry) -> None:
        if entry.kind == "stub":
            do({"op": "promote", "id": entry.id, "step": step, "chunk_id": chunk_id})
            promoted_now[entry.id] = entry.sentence
            promotions.append(entry.id)

    def extra_record(entry: LedgerEntry) -> EntityRecord:
        key = name_key(entry.name)
        rec = extra.get(key)
        if rec is None:
            rec = EntityRecord(name=entry.name)
            extra[key] = rec
        return rec

    def ensure_entry_for_record(rec: EntityRecord, provenance: list[str]) -> LedgerEntry:
        entry = ledger.by_name(rec.name)
        if entry is None:
            owner = ledger.by_name_or_alias(rec.name)
            if owner is not None:
                # The model named the record by a stored alias: keep one record under the stored name.
                counts["described_name_was_alias"] += 1
                old = rec.name
                rec.name = owner.name
                if name_key(old) != name_key(owner.name) and all(name_key(a) != name_key(old) for a in rec.aliases):
                    rec.aliases.append(old)
                entry = owner
            else:
                entry = create(rec.name, "record", provenance=provenance)
                if "candidate_net" not in provenance:
                    llm_additions.append(entry.id)
        if entry.kind == "stub":
            promote(entry)
        do({"op": "describe", "id": entry.id, "step": step})
        do({"op": "provenance", "id": entry.id, "add": sorted(set(provenance))})
        link(entry)
        return entry

    def resolve_target(target: str | None) -> LedgerEntry | None:
        if not target:
            return None
        entry = ledger.get(target.strip().upper())
        if entry is not None:
            return entry
        return ledger.by_name_or_alias(target)

    parse_failure = block.status == "failed" and bool(plan.candidates)
    if parse_failure:
        counts["decision_block_parse_failures"] += 1
    by_index = block.by_index()

    finals = None
    redirects: list[dict[str, Any]] = []
    if ctx is not None:
        # p1b: rule 3 on the model's SAME AS, auto-join, then redirect the JSON records of SAME AS candidates.
        finals = resolve_final_decisions(
            ledger=ledger,
            step=ctx,
            candidates=plan.candidates,
            by_index=by_index,
            parse_failure=parse_failure,
            described=described,
            resolve_target=resolve_target,
            counts=counts,
            kinds=(NEW, SAME_AS, NOT_AN_ENTITY),
        )
        if ctx.config.join_direction:
            described, redirects = redirect_described(described, finals, plan.candidates, same_as_kind=SAME_AS, counts=counts)

    # 1. One outcome per candidate.
    for i, cand in enumerate(plan.candidates):
        final = finals[i] if finals is not None else None
        if final is not None:
            kind, target, reason, matched_by = final.kind, final.target, final.reason, final.matched_by
        else:
            decision = by_index.get(i)
            if parse_failure or decision is None:
                kind, target, reason = NEW, None, ""
                matched_by = "parse_failure" if parse_failure else "omitted"
                counts["parse_failure_candidates" if parse_failure else "omitted_candidates"] += 1
            else:
                kind, target, reason, matched_by = decision.kind, decision.target, decision.reason, decision.matched_by
        row: dict[str, Any] = {
            "n": i + 1,
            "candidate": cand.text,
            "label": cand.label,
            "decision": kind,
            "target": target,
            "reason": reason,
            "matched_by": matched_by,
        }
        if final is not None:
            row.update(final.log)
        if kind == NOT_AN_ENTITY:
            counts["not_an_entity"] += 1
            row["outcome"] = "not_an_entity"
            decisions_log.append(row)
            continue
        entry: LedgerEntry | None = None
        if kind == SAME_AS:
            entry = final.entry if final is not None else resolve_target(target)
            if entry is None:
                counts["same_as_unresolved"] += 1
                row["note"] = "target not found; treated as NEW"
                kind = NEW
            else:
                counts["same_as"] += 1
                set_label(entry, cand.label)
                do({"op": "same_as", "id": entry.id})
                was_stub = entry.kind == "stub"
                added = add_alias(entry, cand.text)
                if was_stub:
                    promote(entry)
                do({"op": "provenance", "id": entry.id, "add": ["candidate_net"]})
                link(entry)
                rec = extra_record(entry)
                if added and cand.text not in rec.aliases:
                    rec.aliases.append(cand.text)
                row["resolved_id"] = entry.id
                row["resolved_name"] = entry.name
                row["outcome"] = "promoted" if was_stub else "linked"
                if added:
                    row["alias_added"] = cand.text
        if kind == NEW:
            rec_match = _match_described(cand, described)
            if rec_match is not None and ctx is not None and ctx.config.same_as_rule:
                # p1b: a NEW candidate folded into a described record is a join too; no title or type conflict.
                owner = ledger.by_name_or_alias(rec_match.name)
                clash = title_conflict(cand.text, rec_match.name) or (
                    owner is not None and entry_title_conflict(cand.text, owner) is not None
                )
                if not clash and owner is not None:
                    clash = type_conflict(cand.label, ledger.labels.get(owner.id, ""))
                if clash:
                    counts["described_match_rejected_conflict"] += 1
                    row["described_match_rejected"] = rec_match.name
                    rec_match = None
            if rec_match is not None:
                claimed.add(described.index(rec_match))
                entry = ensure_entry_for_record(rec_match, ["candidate_net", "llm"])
                set_label(entry, cand.label)
                counts["new_described"] += 1
                if name_key(cand.text) != name_key(entry.name) and all(name_key(a) != name_key(cand.text) for a in rec_match.aliases):
                    if add_alias(entry, cand.text):
                        rec_match.aliases.append(cand.text)
                row["resolved_id"] = entry.id
                row["resolved_name"] = entry.name
                row["outcome"] = "record"
            else:
                existing = ledger.by_name_or_alias(cand.text)
                if existing is not None:
                    counts["new_but_exists"] += 1
                    entry = existing
                    do({"op": "provenance", "id": entry.id, "add": ["candidate_net"]})
                    link(entry)
                    extra_record(entry)
                    row["resolved_id"] = entry.id
                    row["resolved_name"] = entry.name
                    row["outcome"] = "linked_existing"
                else:
                    provenance = ["candidate_net"] + (["parse_failure"] if parse_failure else [])
                    entry = create(cand.text, "stub", sentence=cand.sentence, provenance=provenance)
                    set_label(entry, cand.label)
                    counts["new_stub"] += 1
                    if any(match_kind(cand.text, n) == "exact" for n in cut_names):
                        counts["stub_described_but_cut"] += 1
                        row["described_but_cut"] = True
                    link(entry)
                    extra_record(entry)
                    row["resolved_id"] = entry.id
                    row["resolved_name"] = entry.name
                    row["outcome"] = "stub"
        decisions_log.append(row)

    # 2. Described records the candidate net did not propose: the model's own additions.
    for idx, rec in enumerate(described):
        if idx in claimed:
            continue
        before = len(ledger.entries)
        entry = ensure_entry_for_record(rec, ["llm"])
        if len(ledger.entries) > before:
            counts["llm_additions"] += 1
        else:
            counts["described_existing"] += 1

    # 3. Final active entities of the step: described records plus ledger-derived additions.
    described_by_key: dict[str, EntityRecord] = {}
    for rec in described:
        described_by_key.setdefault(name_key(rec.name), rec)
    merged: list[EntityRecord] = []
    seen: set[str] = set()
    for rec in described:
        key = name_key(rec.name)
        if key in seen:
            continue
        seen.add(key)
        add = extra.pop(key, None)
        if add is not None:
            for a in add.aliases:
                if all(name_key(a) != name_key(x) for x in rec.aliases) and name_key(a) != key:
                    rec.aliases.append(a)
        merged.append(rec)
    for key, rec in extra.items():
        if key not in seen:
            seen.add(key)
            merged.append(rec)
    # A stub promoted this step: its stored sentence becomes its first description.
    for entry_id, sentence in promoted_now.items():
        entry = ledger.get(entry_id)
        if entry is None or not sentence:
            continue
        key = name_key(entry.name)
        rec = next((r for r in merged if name_key(r.name) == key), None)
        if rec is None:
            rec = EntityRecord(name=entry.name)
            merged.append(rec)
        if sentence not in rec.description:
            rec.description.insert(0, sentence)
    alias_rejections: list[dict[str, Any]] = []
    if ctx is not None and ctx.config.aliases_from_spans:
        # p1b alias hygiene: every alias written to memory this step is a span of the chunk (or its
        # title-stripped form) with no title or type conflict against its record.
        alias_rejections = filter_aliases(merged, plan.candidates, ledger, counts=counts)
    counts["described_records"] += len(described)
    counts["cut_by_budget"] += len(cut_names)
    return StepOutcome(
        active_entities=merged,
        ops=ops,
        counts=counts,
        decisions=decisions_log,
        new_entries=new_entries,
        promotions=promotions,
        aliases_added=aliases_added,
        links=links,
        llm_additions=llm_additions,
        described=[r.name for r in described],
        redirects=redirects,
        alias_rejections=alias_rejections,
    )


def cap_descriptions(
    ledger: DocumentLedger, memory: list[EntityRecord], *, cap: int, step: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Enforce the description cap on the updated memory: drop the weakest, log every replacement.

    The weakest description has the fewest content tokens; ties go to the
    oldest. Returns (ops, replacements).
    """

    ops: list[dict[str, Any]] = []
    replacements: list[dict[str, Any]] = []
    if cap is None or int(cap) < 0:
        return ops, replacements
    for rec in memory:
        while len(rec.description) > int(cap):
            weakest = min(range(len(rec.description)), key=lambda i: (description_strength(rec.description[i]), i))
            text = rec.description.pop(weakest)
            entry = ledger.by_name(rec.name)
            if entry is not None:
                op = {"op": "drop_description", "id": entry.id, "text": text, "step": step}
                ledger.apply_op(op)
                ops.append(op)
            replacements.append({"id": entry.id if entry else None, "name": rec.name, "dropped": text, "kept": list(rec.description)})
    return ops, replacements


def reconcile(ledger: DocumentLedger, state: DiscourseState, *, step: int, chunk_id: str) -> tuple[list[dict[str, Any]], list[str], int]:
    """Make a (possibly restored) state agree with the ledger before a step.

    Removes descriptions the ledger logged as replaced (the cache replay re-adds
    them) and creates a ``recovered`` entry for any memory name the ledger does
    not know (should not happen; counted). Returns (ops, recovered names, removed).
    """

    ops: list[dict[str, Any]] = []
    recovered: list[str] = []
    removed = 0
    for rec in state.entity_memory:
        entry = ledger.by_name(rec.name)
        if entry is None:
            entry_id = ledger.new_id()
            op = {
                "op": "create",
                "id": entry_id,
                "name": rec.name,
                "kind": "record",
                "step": step,
                "chunk_id": chunk_id,
                "sentence": "",
                "provenance": ["recovered"],
            }
            ledger.apply_op(op)
            ops.append(op)
            recovered.append(rec.name)
            entry = ledger.get(entry_id)
        if entry is not None and entry.dropped_descriptions:
            dropped = set(entry.dropped_descriptions)
            kept = [d for d in rec.description if d not in dropped]
            removed += len(rec.description) - len(kept)
            rec.description = kept
    return ops, recovered, removed


__all__ = ["StepOutcome", "StepPlan", "apply_decisions", "cap_descriptions", "reconcile"]
