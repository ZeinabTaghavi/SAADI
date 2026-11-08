"""Profile ``p1b_ledger``: three changes on top of ``p1_ledger`` (method update tracker, run 1b).

Everything here is switched on by keys of the ``ledger`` mapping; with the keys
absent (profile ``p1_ledger``) none of it runs and the run-1 code path is
unchanged (``tests/ledger/test_p1b_rules.py`` and the p1 tests check both).

1. Join direction, enforced in code (``join_direction: enforced``).
   - An accepted SAME AS attaches the candidate as an alias of the target; no
     record is created for it (the p1 code already did this; kept).
   - A described JSON record whose title-stripped name equals a candidate that
     ended as SAME AS is *redirected*: renamed to the target, so its
     descriptions merge into the target and no record is created.
   - A NEW candidate whose title-stripped name equals the name or an alias of a
     ledger entry shown in the prompt, with no title or type conflict, becomes
     SAME AS that entry (*auto-join*).
   - Optional, off unless ``json_alias_join: true``: a NEW candidate whose own
     JSON record lists, as an alias, the name or an alias of a shown entry
     (title-stripped equality) becomes SAME AS that entry if rule 3 accepts it.
2. Recall at the reveal (``recall: chunk_tokens``): ledger recall matches every
   name and alias of every entry (title stripped, tokens of 3+ characters, no
   stopwords) against the chunk text, independent of the candidate net.
   Priority under the cap: full name match, then full alias match, then a
   single-token match; ties by the most recent mention.
3. SAME AS acceptance (``same_as_rule: p1b``): accepted only with a shared name
   token or the two names in one sentence of the chunk; rejected on a title
   conflict (same stripped name, different titles) or a type conflict (person
   against place or organization, from the candidate-net labels stored per
   entry). Alias hygiene (``aliases_from_spans: true``): every alias written to
   memory must be a candidate-net span of the chunk or its title-stripped form,
   and carry no title or type conflict with its record.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from ..core.types import EntityRecord
from .candidate_net import Candidate, split_sentences
from .ledger import TITLES, DocumentLedger, LedgerEntry, name_key, name_tokens, stripped_tokens

# Titles that name the same form of address are compared as one.
_TITLE_EQUIV = {
    "dr": "doctor",
    "capt": "captain",
    "prof": "professor",
    "rev": "reverend",
    "st": "saint",
    "madam": "madame",
    "mme": "madame",
    "mlle": "mademoiselle",
}
PERSON_LABELS = frozenset({"person"})
PLACE_ORG_LABELS = frozenset({"place", "organization"})
MIN_RECALL_TOKEN_LEN = 3

# Recall match classes, in priority order.
RECALL_FULL_NAME, RECALL_FULL_ALIAS, RECALL_SINGLE_TOKEN = 0, 1, 2
RECALL_CLASS_NAMES = {RECALL_FULL_NAME: "full_name", RECALL_FULL_ALIAS: "full_alias", RECALL_SINGLE_TOKEN: "single_token"}


@dataclass
class P1bConfig:
    """Which of the p1b rules are on (all off = the p1 code path)."""

    join_direction: bool = False
    chunk_token_recall: bool = False
    same_as_rule: bool = False
    aliases_from_spans: bool = False
    json_alias_join: bool = False

    @property
    def any(self) -> bool:
        return self.join_direction or self.chunk_token_recall or self.same_as_rule or self.aliases_from_spans or self.json_alias_join

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any]) -> P1bConfig:
        join = str(cfg.get("join_direction", "model") or "model").strip().lower()
        recall = str(cfg.get("recall", "candidates") or "candidates").strip().lower()
        rule = str(cfg.get("same_as_rule", "none") or "none").strip().lower()
        if join not in {"model", "enforced"}:
            raise ValueError(f"ledger.join_direction must be model or enforced, got {join!r}")
        if recall not in {"candidates", "chunk_tokens"}:
            raise ValueError(f"ledger.recall must be candidates or chunk_tokens, got {recall!r}")
        if rule not in {"none", "p1b"}:
            raise ValueError(f"ledger.same_as_rule must be none or p1b, got {rule!r}")
        out = cls(
            join_direction=join == "enforced",
            chunk_token_recall=recall == "chunk_tokens",
            same_as_rule=rule == "p1b",
            aliases_from_spans=bool(cfg.get("aliases_from_spans", False)),
            json_alias_join=bool(cfg.get("json_alias_join", False)),
        )
        if out.json_alias_join and not out.join_direction:
            raise ValueError("ledger.json_alias_join needs join_direction: enforced")
        return out


@dataclass
class P1bStep:
    """Per-chunk context the p1b rules need in apply_decisions (carried on the StepPlan)."""

    config: P1bConfig
    chunk_text: str
    shown_ids: list[str] = field(default_factory=list)  # overlap group + ledger group


# ---------------------------------------------------------------------------
# Names, titles, types
# ---------------------------------------------------------------------------


def canonical_titles(text: str) -> frozenset[str]:
    return frozenset(_TITLE_EQUIV.get(t, t) for t in name_tokens(text) if t in TITLES)


def title_conflict(a: str, b: str) -> bool:
    """Same stripped name (one contains the other), both titled, and the titles differ (Mr. against Mrs.)."""

    sa, sb = set(stripped_tokens(a)), set(stripped_tokens(b))
    if not sa or not sb or not (sa <= sb or sb <= sa):
        return False
    ta, tb = canonical_titles(a), canonical_titles(b)
    return bool(ta) and bool(tb) and ta != tb


def entry_title_conflict(text: str, entry: LedgerEntry) -> str | None:
    """The stored name or alias of ``entry`` that conflicts with ``text`` by title, if any."""

    for stored in entry.all_names():
        if title_conflict(text, stored):
            return stored
    return None


def _label(label: str | None) -> str:
    return str(label or "").strip().lower()


def type_conflict(a: str | None, b: str | None) -> bool:
    """Person against place or organization (candidate-net labels). Unknown labels never conflict."""

    la, lb = _label(a), _label(b)
    return (la in PERSON_LABELS and lb in PLACE_ORG_LABELS) or (lb in PERSON_LABELS and la in PLACE_ORG_LABELS)


def shares_name_token(text: str, entry: LedgerEntry) -> bool:
    toks = set(stripped_tokens(text))
    return bool(toks) and any(toks & set(stripped_tokens(n)) for n in entry.all_names())


def same_sentence(chunk_text: str, text: str, entry: LedgerEntry) -> str | None:
    """A sentence of the chunk holding ``text`` and a stored name or alias of ``entry`` (all their tokens)."""

    toks = set(stripped_tokens(text))
    if not toks:
        return None
    names = [set(stripped_tokens(n)) for n in entry.all_names()]
    names = [n for n in names if n]
    for start, end in split_sentences(chunk_text):
        sentence = chunk_text[start:end]
        st = set(name_tokens(sentence))
        if toks <= st and any(n <= st for n in names):
            return sentence.strip()
    return None


@dataclass
class Verdict:
    accepted: bool
    reason: str  # shared_token | same_sentence | title_conflict | type_conflict | no_shared_token_or_sentence
    detail: str = ""


def accept_same_as(candidate: Candidate, entry: LedgerEntry, ledger: DocumentLedger, chunk_text: str) -> Verdict:
    """Rule 3: conflicts reject first; then a shared name token or one sentence holding both names accepts."""

    stored = entry_title_conflict(candidate.text, entry)
    if stored is not None:
        return Verdict(False, "title_conflict", f"{candidate.text!r} against {stored!r}")
    target_label = ledger.labels.get(entry.id, "")
    if type_conflict(candidate.label, target_label):
        return Verdict(False, "type_conflict", f"{candidate.label} against {target_label}")
    if shares_name_token(candidate.text, entry):
        return Verdict(True, "shared_token")
    sentence = same_sentence(chunk_text, candidate.text, entry)
    if sentence is not None:
        return Verdict(True, "same_sentence", sentence[:200])
    return Verdict(False, "no_shared_token_or_sentence")


def no_conflict(candidate: Candidate, entry: LedgerEntry, ledger: DocumentLedger) -> tuple[bool, str]:
    stored = entry_title_conflict(candidate.text, entry)
    if stored is not None:
        return False, f"title_conflict ({candidate.text!r} against {stored!r})"
    if type_conflict(candidate.label, ledger.labels.get(entry.id, "")):
        return False, f"type_conflict ({candidate.label} against {ledger.labels.get(entry.id, '')})"
    return True, ""


# ---------------------------------------------------------------------------
# Change 2: recall by chunk tokens
# ---------------------------------------------------------------------------


def recall_tokens(text: str) -> list[str]:
    return [t for t in stripped_tokens(text) if len(t) >= MIN_RECALL_TOKEN_LEN]


def recall_by_chunk_tokens(
    ledger: DocumentLedger,
    chunk_text: str,
    *,
    exclude_keys: set[str] | None = None,
    cap: int = 8,
) -> tuple[list[LedgerEntry], list[LedgerEntry], bool, dict[str, dict[str, Any]]]:
    """Every entry whose name or alias matches the chunk text by tokens; ranked, capped.

    A stored name matches when at least one of its recall tokens (title
    stripped, 3+ characters, no stopwords) is a token of the chunk. Class:
    the entry's name fully present (0), an alias fully present (1), otherwise
    a single-token match (2). Ties: the most recently mentioned entry first,
    then creation order. Returns ``(recalled, left_out, cap_hit, matches)``
    where ``matches[id] = {"class", "via", "tokens"}``.
    """

    exclude = set(exclude_keys or set())
    chunk_t = {t for t in name_tokens(chunk_text) if len(t) >= MIN_RECALL_TOKEN_LEN}
    scored: list[tuple[int, int, int, LedgerEntry]] = []
    matches: dict[str, dict[str, Any]] = {}
    for entry in ledger.entries.values():
        if name_key(entry.name) in exclude:
            continue
        best: tuple[int, str, list[str]] | None = None
        for i, stored in enumerate(entry.all_names()):
            toks = recall_tokens(stored)
            if not toks:
                continue
            hit = sorted({t for t in toks if t in chunk_t})
            if not hit:
                continue
            if len(hit) == len(set(toks)):
                cls = RECALL_FULL_NAME if i == 0 else RECALL_FULL_ALIAS
            else:
                cls = RECALL_SINGLE_TOKEN
            if best is None or cls < best[0]:
                best = (cls, stored, hit)
        if best is None:
            continue
        matches[entry.id] = {"class": RECALL_CLASS_NAMES[best[0]], "via": best[1], "tokens": best[2]}
        scored.append((best[0], -int(entry.last_step), int(entry.id[1:]), entry))
    scored.sort(key=lambda row: row[:3])
    entries = [row[3] for row in scored]
    n = max(0, int(cap))
    recalled, left_out = entries[:n], entries[n:]
    return recalled, left_out, bool(left_out), matches


# ---------------------------------------------------------------------------
# Change 1 and 3: the final decision per candidate
# ---------------------------------------------------------------------------


@dataclass
class FinalDecision:
    kind: str  # NEW | SAME_AS | NOT_AN_ENTITY (accounting constants)
    entry: LedgerEntry | None
    target: str | None
    reason: str
    matched_by: str
    log: dict[str, Any] = field(default_factory=dict)


def _matching_shown(candidate: Candidate, ledger: DocumentLedger, shown_ids: list[str]) -> list[LedgerEntry]:
    toks = frozenset(stripped_tokens(candidate.text))
    if not toks:
        return []
    out: list[LedgerEntry] = []
    for entry_id in dict.fromkeys(shown_ids):
        entry = ledger.get(entry_id)
        if entry is None:
            continue
        if any(frozenset(stripped_tokens(n)) == toks for n in entry.all_names()):
            out.append(entry)
    return out


def _own_record(candidate: Candidate, described: list[EntityRecord]) -> EntityRecord | None:
    """The JSON record that is the candidate itself: same title-stripped name, or the candidate as an exact alias."""

    toks = frozenset(stripped_tokens(candidate.text))
    for rec in described:
        if toks and frozenset(stripped_tokens(rec.name)) == toks:
            return rec
    for rec in described:
        if any(name_key(a) == name_key(candidate.text) for a in rec.aliases):
            return rec
    return None


def resolve_final_decisions(
    *,
    ledger: DocumentLedger,
    step: P1bStep,
    candidates: list[Candidate],
    by_index: dict[int, Any],
    parse_failure: bool,
    described: list[EntityRecord],
    resolve_target,
    counts: Counter,
    kinds: tuple[str, str, str],
) -> list[FinalDecision]:
    """Apply rule 3 to the model's SAME AS decisions, then auto-join (and the optional JSON-alias join)."""

    NEW, SAME_AS, NOT_AN_ENTITY = kinds
    cfg = step.config
    out: list[FinalDecision] = []
    for i, cand in enumerate(candidates):
        decision = by_index.get(i)
        if parse_failure or decision is None:
            matched_by = "parse_failure" if parse_failure else "omitted"
            counts["parse_failure_candidates" if parse_failure else "omitted_candidates"] += 1
            out.append(FinalDecision(NEW, None, None, "", matched_by, {"model_decision": None}))
            continue
        kind, target, reason, matched_by = decision.kind, decision.target, decision.reason, decision.matched_by
        log: dict[str, Any] = {"model_decision": kind, "model_target": target}
        entry: LedgerEntry | None = None
        explicit_new = kind == NEW
        if kind == SAME_AS:
            entry = resolve_target(target)
            if entry is None:
                counts["same_as_unresolved"] += 1
                log["note"] = "target not found; treated as NEW"
                kind = NEW
            elif cfg.same_as_rule:
                verdict = accept_same_as(cand, entry, ledger, step.chunk_text)
                log["acceptance"] = verdict.reason
                if verdict.detail:
                    log["acceptance_detail"] = verdict.detail
                if verdict.accepted:
                    counts[f"same_as_accepted_{verdict.reason}"] += 1
                else:
                    counts[f"same_as_rejected_{verdict.reason}"] += 1
                    counts["same_as_rejected"] += 1
                    log["rejected_target_id"] = entry.id
                    log["rejected_target_name"] = entry.name
                    kind, entry = NEW, None
        if kind == NEW and cfg.join_direction and (explicit_new or "rejected_target_id" in log or "note" in log):
            shown = _matching_shown(cand, ledger, step.shown_ids)
            ok = []
            for e in shown:
                fine, why = no_conflict(cand, e, ledger)
                if fine:
                    ok.append(e)
                else:
                    counts["auto_join_blocked_by_conflict"] += 1
                    log.setdefault("auto_join_blocked", []).append({"id": e.id, "name": e.name, "why": why})
            if "rejected_target_id" in log:
                ok = [e for e in ok if e.id != log["rejected_target_id"]]
            if len(ok) == 1:
                entry, kind = ok[0], SAME_AS
                counts["auto_join"] += 1
                log["auto_join"] = {"id": entry.id, "name": entry.name}
                if reason:
                    log["model_reason"] = reason
                reason = "auto-join: the title-stripped name equals a shown record's name or alias"
            elif len(ok) > 1:
                counts["auto_join_ambiguous"] += 1
                log["auto_join_ambiguous"] = [e.id for e in ok]
        if kind == NEW and cfg.json_alias_join and explicit_new:
            rec = _own_record(cand, described)
            if rec is not None:
                hits: dict[str, LedgerEntry] = {}
                for alias in rec.aliases:
                    probe = Candidate(text=alias, label=cand.label, score=1.0, start=-1, end=-1, sentence="")
                    for e in _matching_shown(probe, ledger, step.shown_ids):
                        hits[e.id] = e
                accepted = []
                for e in hits.values():
                    verdict = accept_same_as(cand, e, ledger, step.chunk_text)
                    if verdict.accepted:
                        accepted.append((e, verdict))
                    else:
                        counts[f"json_alias_join_rejected_{verdict.reason}"] += 1
                if len(accepted) == 1:
                    entry, verdict = accepted[0]
                    kind = SAME_AS
                    counts["json_alias_join"] += 1
                    log["json_alias_join"] = {"id": entry.id, "name": entry.name, "acceptance": verdict.reason, "record": rec.name}
                    reason = f"JSON-alias join: the JSON record {rec.name!r} lists the shown record's name as an alias"
                elif len(accepted) > 1:
                    counts["json_alias_join_ambiguous"] += 1
        if kind == NOT_AN_ENTITY:
            entry = None
        out.append(FinalDecision(kind, entry, entry.id if entry is not None else target, reason, matched_by, log))
    return out


def redirect_described(
    described: list[EntityRecord],
    finals: list[FinalDecision],
    candidates: list[Candidate],
    *,
    same_as_kind: str,
    counts: Counter,
) -> tuple[list[EntityRecord], list[dict[str, Any]]]:
    """Rename JSON records that are a SAME AS candidate to the target; merge duplicates. Returns (described, log)."""

    targets: dict[frozenset[str], LedgerEntry] = {}
    for cand, final in zip(candidates, finals, strict=True):
        if final.kind == same_as_kind and final.entry is not None:
            toks = frozenset(stripped_tokens(cand.text))
            if toks:
                targets.setdefault(toks, final.entry)
    log: list[dict[str, Any]] = []
    if targets:
        for rec in described:
            entry = targets.get(frozenset(stripped_tokens(rec.name)))
            if entry is None or name_key(rec.name) == name_key(entry.name):
                continue
            old = rec.name
            rec.name = entry.name
            if all(name_key(a) != name_key(old) for a in rec.aliases):
                rec.aliases.append(old)
            counts["redirected"] += 1
            log.append({"from": old, "to": entry.name, "id": entry.id})
    merged: list[EntityRecord] = []
    by_key: dict[str, EntityRecord] = {}
    for rec in described:
        key = name_key(rec.name)
        first = by_key.get(key)
        if first is None:
            by_key[key] = rec
            merged.append(rec)
            continue
        for d in rec.description:
            if d not in first.description:
                first.description.append(d)
        for a in rec.aliases:
            if all(name_key(a) != name_key(x) for x in first.aliases):
                first.aliases.append(a)
        counts["redirected_duplicates_merged"] += 1
    return merged, log


# ---------------------------------------------------------------------------
# Alias hygiene
# ---------------------------------------------------------------------------


def title_stripped_form(text: str) -> str:
    words = [w for w in str(text or "").split() if not (name_tokens(w) and all(t in TITLES for t in name_tokens(w)))]
    return " ".join(words)


def allowed_alias_keys(candidates: list[Candidate]) -> dict[str, str]:
    """name key -> candidate label, for every span of the chunk and its title-stripped form."""

    out: dict[str, str] = {}
    for c in candidates:
        for form in (c.text, title_stripped_form(c.text)):
            key = name_key(form)
            if key and stripped_tokens(form):
                out.setdefault(key, c.label)
    return out


def filter_aliases(
    records: list[EntityRecord],
    candidates: list[Candidate],
    ledger: DocumentLedger,
    *,
    counts: Counter,
) -> list[dict[str, Any]]:
    """Drop aliases that are not a span (or its title-stripped form) or conflict with their record. Returns the log."""

    allowed = allowed_alias_keys(candidates)
    log: list[dict[str, Any]] = []
    for rec in records:
        entry = ledger.by_name(rec.name)
        rec_label = ledger.labels.get(entry.id, "") if entry is not None else allowed.get(name_key(rec.name), "")
        names = entry.all_names() if entry is not None else [rec.name]
        kept: list[str] = []
        for alias in rec.aliases:
            key = name_key(alias)
            why = None
            if key not in allowed:
                why = "not_a_span"
            elif any(title_conflict(alias, n) for n in names):
                why = "title_conflict"
            elif type_conflict(allowed.get(key), rec_label):
                why = "type_conflict"
            if why is None:
                kept.append(alias)
            else:
                counts[f"alias_rejected_{why}"] += 1
                if len(log) < 20:
                    log.append({"record": rec.name, "alias": alias, "why": why})
        rec.aliases = kept
    return log


__all__ = [
    "FinalDecision",
    "P1bConfig",
    "P1bStep",
    "RECALL_CLASS_NAMES",
    "Verdict",
    "accept_same_as",
    "allowed_alias_keys",
    "canonical_titles",
    "entry_title_conflict",
    "filter_aliases",
    "recall_by_chunk_tokens",
    "redirect_described",
    "resolve_final_decisions",
    "same_sentence",
    "shares_name_token",
    "title_conflict",
    "title_stripped_form",
    "type_conflict",
]
