"""Applies structured chunk edit operations deterministically."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .types import TextInsertion

DEFAULT_LINKING_SENTENCE_WORD_LIMIT = 12
DEFAULT_INSERTION_WORD_LIMIT = 10


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


def _resolve_insertion_match(raw_text: str, insertion: TextInsertion) -> tuple[int, int] | None:
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
) -> MaterializedChunk:
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
        match = _resolve_insertion_match(base, cleaned)
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
