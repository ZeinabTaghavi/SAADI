"""Forced accounting: parse one decision per candidate from the model's reply.

Format (spec, section E), one line per candidate between ``[DECISIONS]`` and
``[END_DECISIONS]``::

    <n>. <candidate> -> NEW | SAME AS <record id> (<reason>) | NOT AN ENTITY

The parser is lenient about arrows, numbering punctuation, case and spacing,
but strict about meaning: a candidate without a decision is *omitted* (treated
as NEW and counted), an unparseable block is a *parse failure* (every candidate
becomes a stub; counted). Nothing is silently dropped.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

NEW = "NEW"
SAME_AS = "SAME_AS"
NOT_AN_ENTITY = "NOT_AN_ENTITY"

DECISIONS_OPEN = "[DECISIONS]"
DECISIONS_CLOSE = "[END_DECISIONS]"

_ARROW = r"(?:->|→|=>|-->|—>|–>|::|:|\s[-—–]\s)"
_LINE = re.compile(r"^\s*(?:(\d{1,3})\s*[.):\]-]+\s*)?(.*?)\s*" + _ARROW + r"\s*(.+?)\s*$")
_DECISION = re.compile(
    r"^(?:\*\*)?(?P<kind>NEW|NOT\s+AN?\s+ENTITY|NOT\s+ENTITY|NONE|SAME\s+AS)(?:\*\*)?\b\s*(?P<rest>.*)$",
    re.IGNORECASE,
)
_ENTRY_ID = re.compile(r"\b([ES]\d{3,6})\b")
_PAREN_REASON = re.compile(r"\((?P<reason>.*)\)\s*\.?\s*$", re.DOTALL)
_JSON_START = re.compile(r"\{\s*\"", re.DOTALL)


@dataclass
class Decision:
    candidate_index: int  # 0-based position in the candidate list
    candidate_text: str
    kind: str  # NEW | SAME_AS | NOT_AN_ENTITY
    target: str | None = None  # entry id or name text as written
    reason: str = ""
    raw_line: str = ""
    matched_by: str = ""  # number | text | omitted | parse_failure

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionBlock:
    decisions: list[Decision]
    status: str  # ok | partial | failed | empty
    omitted: list[int] = field(default_factory=list)
    unmatched_lines: list[str] = field(default_factory=list)
    block_text: str = ""

    def by_index(self) -> dict[int, Decision]:
        return {d.candidate_index: d for d in self.decisions}

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "omitted": list(self.omitted),
            "unmatched_lines": list(self.unmatched_lines),
            "decisions": [d.to_dict() for d in self.decisions],
        }


def _norm(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip().strip("\"'“”‘’*`").strip()
    return text.casefold()


def split_reply(raw_text: str) -> tuple[str, str]:
    """Split the reply into the decision text and the JSON text.

    The decision text is what lies between the markers when both are present;
    otherwise everything before the first JSON object. The JSON text starts at
    the first ``{"`` after the decision text.
    """

    text = str(raw_text or "")
    open_at = text.find(DECISIONS_OPEN)
    close_at = text.find(DECISIONS_CLOSE)
    if open_at >= 0 and close_at > open_at:
        decisions = text[open_at + len(DECISIONS_OPEN) : close_at]
        rest = text[close_at + len(DECISIONS_CLOSE) :]
    else:
        m = _JSON_START.search(text)
        if m:
            decisions = text[: m.start()]
            rest = text[m.start() :]
        else:
            decisions = text
            rest = ""
        decisions = decisions.replace(DECISIONS_OPEN, "").replace(DECISIONS_CLOSE, "")
    m = _JSON_START.search(rest)
    json_text = rest[m.start() :] if m else ""
    return decisions, json_text


def _parse_decision_text(text: str) -> tuple[str, str | None, str] | None:
    """Return (kind, target, reason) for the right-hand side of a decision line."""

    m = _DECISION.match(str(text or "").strip())
    if not m:
        return None
    kind_raw = re.sub(r"\s+", " ", m.group("kind").upper())
    rest = (m.group("rest") or "").strip()
    if kind_raw == "NEW":
        return NEW, None, ""
    if kind_raw.startswith("NOT") or kind_raw == "NONE":
        return NOT_AN_ENTITY, None, ""
    # SAME AS <id> (<reason>)
    reason = ""
    pm = _PAREN_REASON.search(rest)
    head = rest
    if pm:
        reason = pm.group("reason").strip()
        head = rest[: pm.start()].strip()
    idm = _ENTRY_ID.search(head) or _ENTRY_ID.search(rest)
    if idm:
        target = idm.group(1).upper()
        if not reason:
            tail = rest[idm.end() :].strip(" -–—:;,.")
            reason = tail
    else:
        head = re.sub(r"^(?:record|stub|entry|the record|the stub)\s+", "", head, flags=re.IGNORECASE)
        target = head.strip(" \"'“”‘’`*") or None
        if not reason:
            reason = ""
    return SAME_AS, target, reason


def parse_decisions(raw_text: str, candidates: list[str]) -> DecisionBlock:
    """Parse the decision block for the given numbered candidates (1-based in the prompt)."""

    decisions_text, _json_text = split_reply(raw_text)
    norm_candidates = [_norm(c) for c in candidates]
    found: dict[int, Decision] = {}
    unmatched: list[str] = []
    for line in decisions_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("[") or stripped.startswith("#"):
            continue
        lm = _LINE.match(stripped)
        if not lm:
            if re.search(r"\b(NEW|SAME AS|NOT AN ENTITY)\b", stripped, re.IGNORECASE):
                unmatched.append(stripped)
            continue
        number, cand_text, rhs = lm.group(1), lm.group(2), lm.group(3)
        parsed = _parse_decision_text(rhs)
        if parsed is None:
            unmatched.append(stripped)
            continue
        kind, target, reason = parsed
        index: int | None = None
        matched_by = ""
        if number is not None:
            n = int(number) - 1
            if 0 <= n < len(candidates):
                index, matched_by = n, "number"
        if index is None:
            key = _norm(cand_text)
            if key:
                for i, nc in enumerate(norm_candidates):
                    if i not in found and (nc == key or nc.rstrip("s") == key.rstrip("s")):
                        index, matched_by = i, "text"
                        break
        if index is None:
            unmatched.append(stripped)
            continue
        if index in found:
            # The model wrote two lines for one candidate; keep the first one.
            unmatched.append(stripped)
            continue
        found[index] = Decision(
            candidate_index=index,
            candidate_text=str(candidates[index]),
            kind=kind,
            target=target,
            reason=reason,
            raw_line=stripped,
            matched_by=matched_by,
        )
    omitted = [i for i in range(len(candidates)) if i not in found]
    if not candidates:
        status = "empty"
    elif not found:
        status = "failed"
    elif omitted:
        status = "partial"
    else:
        status = "ok"
    decisions = [found[i] for i in sorted(found)]
    return DecisionBlock(decisions=decisions, status=status, omitted=omitted, unmatched_lines=unmatched, block_text=decisions_text.strip())


__all__ = [
    "DECISIONS_CLOSE",
    "DECISIONS_OPEN",
    "Decision",
    "DecisionBlock",
    "NEW",
    "NOT_AN_ENTITY",
    "SAME_AS",
    "parse_decisions",
    "split_reply",
]
