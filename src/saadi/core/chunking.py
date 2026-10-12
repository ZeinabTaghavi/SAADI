"""Normalizes text and splits documents into sentence-aware chunk records."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from .types import ChunkRecord


def html_to_text(input_text: str) -> str:
    if "<html>" in input_text.lower():
        try:
            from bs4 import BeautifulSoup  # type: ignore
        except Exception:
            return re.sub(r"\n+", "\n", input_text)
        soup = BeautifulSoup(input_text, "html.parser")
        text = soup.get_text()
        return re.sub(r"\n+", "\n", text)
    return re.sub(r"\n+", "\n", input_text)


def coerce_to_text(obj: Any) -> Optional[str]:
    if obj is None:
        return None
    if isinstance(obj, str):
        s = obj.strip()
        return html_to_text(s) if s else None
    if isinstance(obj, dict):
        for key in ("text", "document", "story", "summary", "content", "plot", "passage", "article", "context", "paragraphs"):
            if key in obj and obj[key]:
                return coerce_to_text(obj[key])
        parts = [coerce_to_text(v) for v in obj.values()]
        joined = " ".join([p for p in parts if p])
        return joined or None
    if isinstance(obj, (list, tuple)):
        parts = [coerce_to_text(v) for v in obj]
        joined = " ".join([p for p in parts if p])
        return joined or None
    s = str(obj).strip()
    return html_to_text(s) if s else None


def _encode(text: str, tokenizer=None):
    if tokenizer is None:
        return text.split()
    return tokenizer.encode(text, add_special_tokens=False)


def _decode(tokens, tokenizer=None):
    if tokenizer is None:
        return " ".join(tokens)
    return tokenizer.decode(tokens, skip_special_tokens=True)


_CLAUSE_BREAK_RE = re.compile(r'(?<=[;:])\s+')
_SENTENCE_EXTRACT_RE = re.compile(r'.+?(?:[.!?;:](?:["\')\]]+)?|$)', flags=re.DOTALL)


@dataclass(slots=True)
class _TextUnit:
    text: str
    tokens: list


@dataclass(slots=True)
class _TokenizedUnit:
    text: str
    token_start: int
    token_end: int


def _split_sentences(text: str) -> list[str]:
    src = str(text or "").strip()
    if not src:
        return []
    pieces = [m.group(0).strip() for m in _SENTENCE_EXTRACT_RE.finditer(src)]
    return [p for p in pieces if p]


def _split_clauses(text: str) -> list[str]:
    pieces = [p.strip() for p in _CLAUSE_BREAK_RE.split(str(text or "").strip())]
    return [p for p in pieces if p]


def _split_tokens_hard(text: str, tokens: list, *, tokenizer, chunk_size: int) -> list[_TextUnit]:
    out: list[_TextUnit] = []
    start = 0
    n = len(tokens)
    while start < n:
        end = min(start + int(chunk_size), n)
        piece_tokens = tokens[start:end]
        piece_text = _decode(piece_tokens, tokenizer=tokenizer).strip()
        if piece_tokens and piece_text:
            out.append(_TextUnit(text=piece_text, tokens=piece_tokens))
        start = end
    return out


def _build_text_units(text: str, *, tokenizer, chunk_size: int) -> list[_TextUnit]:
    sentence_texts = _split_sentences(text)
    if not sentence_texts:
        sentence_texts = [str(text or "").strip()]

    units: list[_TextUnit] = []
    for sentence in sentence_texts:
        sentence_tokens = _encode(sentence, tokenizer=tokenizer)
        if not sentence_tokens:
            continue
        if len(sentence_tokens) <= chunk_size:
            units.append(_TextUnit(text=sentence, tokens=sentence_tokens))
            continue

        # Oversized sentence: split by clauses first, then hard token split.
        clause_texts = _split_clauses(sentence)
        if len(clause_texts) <= 1:
            units.extend(_split_tokens_hard(sentence, sentence_tokens, tokenizer=tokenizer, chunk_size=chunk_size))
            continue

        for clause in clause_texts:
            clause_tokens = _encode(clause, tokenizer=tokenizer)
            if not clause_tokens:
                continue
            if len(clause_tokens) <= chunk_size:
                units.append(_TextUnit(text=clause, tokens=clause_tokens))
            else:
                units.extend(_split_tokens_hard(clause, clause_tokens, tokenizer=tokenizer, chunk_size=chunk_size))
    return units


def _tokenize_units_with_spans(units: list[_TextUnit]) -> list[_TokenizedUnit]:
    out: list[_TokenizedUnit] = []
    offset = 0
    for unit in units:
        toks = unit.tokens
        if not toks:
            continue
        start = offset
        end = offset + len(toks)
        out.append(_TokenizedUnit(text=unit.text, token_start=start, token_end=end))
        offset = end
    return out


def _chunk_text_token_windows(
    text: str,
    *,
    tokenizer=None,
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[str, int, int]]:
    tokens = _encode(text, tokenizer=tokenizer)
    if not tokens:
        return []
    chunks: list[tuple[str, int, int]] = []
    start = 0
    n = len(tokens)
    while start < n:
        end = min(start + chunk_size, n)
        ct = _decode(tokens[start:end], tokenizer=tokenizer)
        chunks.append((ct, start, end))
        if end >= n:
            break
        start = end - chunk_overlap
    return chunks


def chunk_text(
    text: str,
    *,
    tokenizer=None,
    chunk_size: int = 500,
    chunk_overlap: int = 0,
) -> list[tuple[str, int, int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    # If there are no sentence boundaries at all, keep legacy token-window behavior.
    if len(_split_sentences(text)) <= 1:
        return _chunk_text_token_windows(
            text,
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    units = _tokenize_units_with_spans(_build_text_units(text, tokenizer=tokenizer, chunk_size=chunk_size))
    if not units:
        return []

    chunks: list[tuple[str, int, int]] = []
    i = 0
    n = len(units)
    while i < n:
        cur_tokens = 0
        j = i
        while j < n:
            unit_len = units[j].token_end - units[j].token_start
            if cur_tokens > 0 and cur_tokens + unit_len > chunk_size:
                break
            if cur_tokens == 0 and unit_len > chunk_size:
                # Should not happen because oversized units are split defensively.
                break
            cur_tokens += unit_len
            j += 1

        # Defensive fallback if packing did not advance.
        if j == i:
            j = i + 1

        chunk_units = units[i:j]
        chunk_text_str = " ".join(u.text.strip() for u in chunk_units if u.text.strip()).strip()
        token_start = chunk_units[0].token_start
        token_end = chunk_units[-1].token_end
        chunks.append((chunk_text_str, token_start, token_end))

        if j >= n:
            break

        if chunk_overlap == 0:
            i = j
            continue

        # Sentence-level overlap: keep as many suffix units as fit into overlap token budget.
        overlap_tokens = 0
        overlap_count = 0
        for k in range(j - 1, i - 1, -1):
            unit_len = units[k].token_end - units[k].token_start
            if overlap_tokens + unit_len > chunk_overlap:
                break
            overlap_tokens += unit_len
            overlap_count += 1

        next_i = j - overlap_count
        if next_i <= i:
            next_i = i + 1
        i = next_i
    return chunks


def chunk_documents(documents: list[Any], tokenizer=None, chunk_size: int = 500, chunk_overlap: int = 0) -> list[str]:
    all_chunks: list[str] = []
    for doc in documents:
        text = coerce_to_text(doc)
        if not text:
            continue
        all_chunks.extend([c for c, _, _ in chunk_text(text, tokenizer=tokenizer, chunk_size=chunk_size, chunk_overlap=chunk_overlap)])
    return all_chunks


def chunk_documents_grouped(documents: list[Any], tokenizer=None, chunk_size: int = 500, chunk_overlap: int = 0) -> list[list[str]]:
    grouped: list[list[str]] = []
    for doc in documents:
        text = coerce_to_text(doc)
        if not text:
            grouped.append([])
            continue
        grouped.append([c for c, _, _ in chunk_text(text, tokenizer=tokenizer, chunk_size=chunk_size, chunk_overlap=chunk_overlap)])
    return grouped


def chunk_documents_grouped_records(
    documents: list[Any],
    *,
    doc_ids: list[str] | None = None,
    tokenizer=None,
    chunk_size: int = 500,
    chunk_overlap: int = 0,
) -> list[list[ChunkRecord]]:
    grouped: list[list[ChunkRecord]] = []
    if doc_ids is None:
        doc_ids = [str(i) for i in range(len(documents))]
    if len(doc_ids) != len(documents):
        raise ValueError("doc_ids length must match documents length")

    for doc_id, doc in zip(doc_ids, documents):
        text = coerce_to_text(doc)
        if not text:
            grouped.append([])
            continue
        recs: list[ChunkRecord] = []
        for idx, (chunk_str, t_start, t_end) in enumerate(chunk_text(text, tokenizer=tokenizer, chunk_size=chunk_size, chunk_overlap=chunk_overlap)):
            recs.append(
                ChunkRecord(
                    doc_id=str(doc_id),
                    chunk_id=f"{doc_id}:{idx}",
                    chunk_index=idx,
                    raw_text=chunk_str,
                    token_start=t_start,
                    token_end=t_end,
                )
            )
        grouped.append(recs)
    return grouped
