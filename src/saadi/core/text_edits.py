"""Applies structured chunk edit operations deterministically."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .types import TextInsertion

DEFAULT_LINKING_SENTENCE_WORD_LIMIT = 12
DEFAULT_INSERTION_WORD_LIMIT = 10

# How a proposed anchor is located in the chunk.
#   strict:   the left and right contexts must match the text character for character
#             (the behavior that produced the paper's saved constructions).
#   tolerant: strict first; if that fails, contexts are compared after trimming
#             whitespace and removing a repeated anchor, and an anchor that occurs
#             exactly once in the chunk is accepted on its own. The anchor itself
#             must always appear verbatim, so grounding is unchanged.
ANCHOR_MATCHING_STRICT = "strict"
ANCHOR_MATCHING_TOLERANT = "tolerant"
ANCHOR_MATCHING_OPTIONS = (ANCHOR_MATCHING_STRICT, ANCHOR_MATCHING_TOLERANT)


def normalize_anchor_matching(value: str | None) -> str:
    raw = str(value or ANCHOR_MATCHING_STRICT).strip().lower()
    if raw not in ANCHOR_MATCHING_OPTIONS:
        raise ValueError(f"Invalid anchor_matching: {value!r}. Expected one of: {', '.join(ANCHOR_MATCHING_OPTIONS)}.")
    return raw


@dataclass(slots=True)
class MaterializedChunk:
    text: str
    linking_sentence: str
    insertions: list[TextInsertion] = field(default_factory=list)
    dropped_insertions: list[TextInsertion] = field(default_factory=list)
    linking_sentence_dropped: bool = False


def _word_count(text: str) -> int:
    return len([tok for tok in str(text or "").strip().split() if tok])


def _match_positions(text: str, token: str) -> list[tuple[int, int]]:
    if not token:
        return []
    return [(m.start(), m.end()) for m in re.finditer(re.escape(token), text)]


_WHITESPACE_RE = re.compile(r"\s+")


def _squash(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", str(text or ""))


def _clean_left_context(context: str, anchor: str) -> str:
    # Models often repeat the anchor inside its own context ("by Mira Chen" for
    # the anchor "Mira Chen"); keep only what precedes the anchor.
    text = _squash(context)
    position = text.find(anchor)
    if position >= 0:
        text = text[:position]
    return text.strip()


def _clean_right_context(context: str, anchor: str) -> str:
    text = _squash(context).strip()
    if text.startswith(anchor):
        text = text[len(anchor) :]
    return text.strip()


def _resolve_insertion_match_tolerant(raw_text: str, insertion: TextInsertion) -> tuple[int, int] | None:
    positions = _match_positions(raw_text, insertion.anchor_text)
    if not positions:
        return None
    anchor = _squash(insertion.anchor_text)
    left = _clean_left_context(insertion.left_context, anchor)
    right = _clean_right_context(insertion.right_context, anchor)
    candidates = [
        (start, end)
        for start, end in positions
        if _squash(raw_text[:start]).rstrip().endswith(left) and _squash(raw_text[end:]).lstrip().startswith(right)
    ]
    occurrence = max(1, int(insertion.occurrence))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1 and occurrence > 1 and len(candidates) >= occurrence:
        return candidates[occurrence - 1]
    if len(positions) == 1:
        # Contexts only disambiguate; a unique verbatim anchor locates itself.
        return positions[0]
    return None


def _resolve_insertion_match(
    raw_text: str, insertion: TextInsertion, *, anchor_matching: str = ANCHOR_MATCHING_STRICT
) -> tuple[int, int] | None:
    match = _resolve_insertion_match_strict(raw_text, insertion)
    if match is None and anchor_matching == ANCHOR_MATCHING_TOLERANT:
        match = _resolve_insertion_match_tolerant(raw_text, insertion)
    return match


def _resolve_insertion_match_strict(raw_text: str, insertion: TextInsertion) -> tuple[int, int] | None:
    candidates: list[tuple[int, int]] = []
    for start, end in _match_positions(raw_text, insertion.anchor_text):
        if insertion.left_context and not raw_text[:start].endswith(insertion.left_context):
            continue
        if insertion.right_context and not raw_text[end:].startswith(insertion.right_context):
            continue
        candidates.append((start, end))

    if not candidates:
        return None

    occurrence = max(1, int(insertion.occurrence))
    if len(candidates) >= occurrence:
        selected = candidates[occurrence - 1]
    else:
        return None

    if occurrence == 1 and len(candidates) > 1:
        return None
    return selected


def materialize_chunk_text(
    raw_text: str,
    *,
    linking_sentence: str = "",
    insertions: list[TextInsertion] | None = None,
    linking_sentence_word_limit: int = DEFAULT_LINKING_SENTENCE_WORD_LIMIT,
    insertion_word_limit: int = DEFAULT_INSERTION_WORD_LIMIT,
    anchor_matching: str = ANCHOR_MATCHING_STRICT,
) -> MaterializedChunk:
    anchor_matching = normalize_anchor_matching(anchor_matching)
    base = str(raw_text or "")
    valid_linking = str(linking_sentence or "").strip()
    linking_sentence_dropped = False
    if valid_linking and _word_count(valid_linking) > int(linking_sentence_word_limit):
        valid_linking = ""
        linking_sentence_dropped = True

    accepted: list[tuple[int, int, TextInsertion]] = []
    dropped: list[TextInsertion] = []

    for insertion in insertions or []:
        cleaned = TextInsertion(
            anchor_text=str(insertion.anchor_text or ""),
            left_context=str(insertion.left_context or ""),
            right_context=str(insertion.right_context or ""),
            occurrence=max(1, int(insertion.occurrence)),
            name=str(insertion.name or "").strip(),
        )
        if not cleaned.anchor_text or not cleaned.name:
            dropped.append(cleaned)
            continue
        if _word_count(cleaned.name) > int(insertion_word_limit):
            dropped.append(cleaned)
            continue
        match = _resolve_insertion_match(base, cleaned, anchor_matching=anchor_matching)
        if match is None:
            dropped.append(cleaned)
            continue
        start, end = match
        conflict_found = False
        for existing_start, existing_end, existing in accepted:
            same_span = (start, end) == (existing_start, existing_end)
            overlaps = not (end <= existing_start or start >= existing_end)
            if same_span and existing.name == cleaned.name:
                conflict_found = True
                break
            if same_span or overlaps:
                dropped.append(cleaned)
                conflict_found = True
                break
        if conflict_found:
            continue
        accepted.append((start, end, cleaned))

    out = base
    for _, end, insertion in sorted(accepted, key=lambda item: (item[1], item[0]), reverse=True):
        out = f"{out[:end]} ({insertion.name}){out[end:]}"

    if valid_linking:
        out = f"{valid_linking} {out}" if out else valid_linking

    return MaterializedChunk(
        text=out or base,
        linking_sentence=valid_linking,
        insertions=[item[2] for item in sorted(accepted, key=lambda item: (item[0], item[1]))],
        dropped_insertions=dropped,
        linking_sentence_dropped=linking_sentence_dropped,
    )
