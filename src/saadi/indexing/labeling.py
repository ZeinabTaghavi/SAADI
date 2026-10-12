"""Maps QA answers/evidence spans to chunk-level gold and silver labels."""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import defaultdict
from typing import Any, Tuple

from ..core.types import ChunkRecord, RetrievalExample
from ..runtime.progress import new_progress_bar

logger = logging.getLogger(__name__)


def _tokenize_match_text(text: str) -> list[str]:
    normalized = (
        str(text or "")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\xa0", " ")
        .replace("\ufeff", " ")
    )
    normalized = unicodedata.normalize("NFKD", normalized)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = re.sub(r"[^A-Za-z0-9]+", " ", normalized.casefold())
    return [tok for tok in normalized.split() if tok]


def _kmp_contains(text_tokens: list[str], pat_tokens: list[str]) -> bool:
    if not pat_tokens:
        return True
    pi = [0] * len(pat_tokens)
    j = 0
    for i in range(1, len(pat_tokens)):
        while j and pat_tokens[i] != pat_tokens[j]:
            j = pi[j - 1]
        if pat_tokens[i] == pat_tokens[j]:
            j += 1
            pi[i] = j
    j = 0
    for tok in text_tokens:
        while j and tok != pat_tokens[j]:
            j = pi[j - 1]
        if tok == pat_tokens[j]:
            j += 1
            if j == len(pat_tokens):
                return True
    return False


def _ordered_subsequence_contains(text_tokens: list[str], pat_tokens: list[str]) -> bool:
    if not pat_tokens:
        return True
    if len(pat_tokens) > len(text_tokens):
        return False
    j = 0
    for tok in text_tokens:
        if tok != pat_tokens[j]:
            continue
        j += 1
        if j == len(pat_tokens):
            return True
    return False


def classify_overlap_tokens(a_t: list[str], b_t: list[str]) -> Tuple[str, int, str]:
    if not b_t:
        return ("full", 0, "")
    if _kmp_contains(a_t, b_t):
        return ("full", len(b_t), " ".join(b_t))
    if _ordered_subsequence_contains(a_t, b_t):
        return ("full", len(b_t), " ".join(b_t))
    max_end = min(len(a_t), len(b_t) - 1)
    k_end = max_end
    while k_end > 0 and a_t[-k_end:] != b_t[:k_end]:
        k_end -= 1
    max_start = min(len(a_t), len(b_t) - 1)
    k_start = max_start
    while k_start > 0 and a_t[:k_start] != b_t[-k_start:]:
        k_start -= 1
    if k_end > 0 or k_start > 0:
        if k_end >= k_start:
            return ("partial", k_end, " ".join(b_t[:k_end]))
        return ("partial", k_start, " ".join(b_t[-k_start:]))
    return ("none", 0, "")


def classify_overlap(a: str, b: str) -> Tuple[str, int, str]:
    a_t = _tokenize_match_text(a)
    b_t = _tokenize_match_text(b)
    return classify_overlap_tokens(a_t, b_t)


def _normalize_retrieval_span_mode(value: str | None) -> str:
    raw = str(value or "text").strip().lower()
    alias = {
        "default": "text",
        "legacy": "text",
        "evidence": "text",
        "window_text": "window",
        "gold_window": "window",
    }
    out = alias.get(raw, raw)
    if out not in {"text", "window"}:
        raise ValueError(f"Unsupported retrieval_span_mode: {value!r}")
    return out


def _effective_retrieval_span_mode(qa_entry: dict[str, Any]) -> str:
    override = qa_entry.get("retrieval_span_mode")
    if isinstance(override, str) and override.strip():
        return _normalize_retrieval_span_mode(override)
    return "text"


def _extract_answer_texts(qa_entry: dict[str, Any]) -> list[str]:
    out: list[str] = []
    raw = qa_entry.get("answers")
    if isinstance(raw, str):
        txt = raw.strip()
        return [txt] if txt else []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if isinstance(item, str):
            txt = item.strip()
            if txt:
                out.append(txt)
            continue
        if isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                out.append(text.strip())
    return out


def _extract_retrieval_spans(qa_entry: dict[str, Any]) -> list[str]:
    out: list[str] = []
    raw = qa_entry.get("retrieval_spans")
    if isinstance(raw, str):
        txt = raw.strip()
        return [txt] if txt else []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
        return out

    # Backward compatibility for older QA fixtures/loaders that stored spans under answers.
    answers = qa_entry.get("answers")
    if not isinstance(answers, list):
        return out
    for item in answers:
        if not isinstance(item, dict):
            continue
        raw_span = item.get("span")
        if isinstance(raw_span, str):
            txt = raw_span.strip()
            if txt:
                out.append(txt)
        elif isinstance(raw_span, list):
            out.extend([s.strip() for s in raw_span if isinstance(s, str) and s.strip()])
    return out

def _boundary_overlap_size(a_t: list[str], b_t: list[str]) -> int:
    max_end = min(len(a_t), len(b_t))
    k_end = max_end
    while k_end > 0 and a_t[-k_end:] != b_t[:k_end]:
        k_end -= 1
    max_start = min(len(a_t), len(b_t))
    k_start = max_start
    while k_start > 0 and a_t[:k_start] != b_t[-k_start:]:
        k_start -= 1
    return max(k_end, k_start)


def _window_overlaps_chunk(chunk_text: str, span_text: str) -> bool:
    chunk_t = _tokenize_match_text(chunk_text)
    span_t = _tokenize_match_text(span_text)
    return _window_overlaps_tokens(chunk_t, span_t)


def _window_overlaps_tokens(chunk_t: list[str], span_t: list[str]) -> bool:
    if not chunk_t or not span_t:
        return False
    if _kmp_contains(chunk_t, span_t) or _kmp_contains(span_t, chunk_t):
        return True
    return _boundary_overlap_size(chunk_t, span_t) > 0


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _match_span_chunks(
    chunks: list[ChunkRecord],
    span: str,
    *,
    retrieval_span_mode: str,
    tokenized_chunks: list[tuple[ChunkRecord, list[str]]] | None = None,
) -> tuple[list[str], list[str]]:
    span_t = _tokenize_match_text(span)
    if tokenized_chunks is None:
        tokenized_chunks = [(chunk, _tokenize_match_text(chunk.raw_text)) for chunk in chunks]
    if retrieval_span_mode == "window":
        matched_ids = _dedupe_preserve_order(
            [str(chunk.chunk_id) for chunk, chunk_t in tokenized_chunks if _window_overlaps_tokens(chunk_t, span_t)]
        )
        if len(matched_ids) == 1:
            return matched_ids, []
        return [], matched_ids

    full_ids: list[str] = []
    partial_ids: list[str] = []
    for chunk, chunk_t in tokenized_chunks:
        kind, _, _ = classify_overlap_tokens(chunk_t, span_t)
        if kind == "full":
            full_ids.append(str(chunk.chunk_id))
        elif kind == "partial":
            partial_ids.append(str(chunk.chunk_id))
    full_ids = _dedupe_preserve_order(full_ids)
    if full_ids:
        return full_ids, []
    partial_ids = _dedupe_preserve_order(partial_ids)
    if len(partial_ids) > 1:
        return [], partial_ids
    return [], []


def _build_support_targets(
    qa_entry: dict[str, Any],
    chunks: list[ChunkRecord],
    *,
    tokenized_chunks: list[tuple[ChunkRecord, list[str]]] | None = None,
) -> tuple[list[str], list[str], list[list[str]], list[int] | None]:
    retrieval_span_mode = _effective_retrieval_span_mode(qa_entry)
    spans = _extract_retrieval_spans(qa_entry)
    if not spans:
        spans = _extract_answer_texts(qa_entry)
        retrieval_span_mode = "text"
    if not spans:
        return [], [], [], None

    chunk_pos_by_id = {str(chunk.chunk_id): idx for idx, chunk in enumerate(chunks)}
    labels = [0] * len(chunks)
    gold_ids: list[str] = []
    silver_ids: list[str] = []
    silver_groups: list[list[str]] = []
    seen_groups: set[tuple[str, ...]] = set()

    for span in spans:
        gold_matches, silver_group = _match_span_chunks(
            chunks,
            span,
            retrieval_span_mode=retrieval_span_mode,
            tokenized_chunks=tokenized_chunks,
        )
        for chunk_id in gold_matches:
            idx = chunk_pos_by_id.get(str(chunk_id))
            if idx is None:
                continue
            labels[idx] = 2
            gold_ids.append(str(chunk_id))

        group = _dedupe_preserve_order([str(chunk_id) for chunk_id in silver_group if str(chunk_id) in chunk_pos_by_id])
        if not group:
            continue
        group_key = tuple(group)
        if group_key not in seen_groups:
            seen_groups.add(group_key)
            silver_groups.append(group)
        for chunk_id in group:
            idx = chunk_pos_by_id[str(chunk_id)]
            if labels[idx] < 1:
                labels[idx] = 1
            silver_ids.append(chunk_id)

    labels_out = labels if any(labels) else None
    return (
        _dedupe_preserve_order(gold_ids),
        _dedupe_preserve_order(silver_ids),
        silver_groups,
        labels_out,
    )


def find_gold_chunk_labels(
    qa_entry: dict[str, Any],
    chunks: list[ChunkRecord],
) -> list[int] | None:
    _, _, _, labels = _build_support_targets(qa_entry, chunks)
    return labels


def build_retrieval_examples(
    qa_entries: list[dict[str, Any]],
    chunks_by_doc: dict[str, list[ChunkRecord]],
) -> list[RetrievalExample]:
    examples: list[RetrievalExample] = []
    total_entries = len(qa_entries)
    logger.info("Building retrieval examples qa_entries=%d cached_docs=%d", total_entries, len(chunks_by_doc))
    progress = new_progress_bar(total=total_entries, desc="Labeling retrieval examples", unit="qa", leave=False)
    processed = 0
    entries_by_doc: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for i, entry in enumerate(qa_entries):
        doc_id = str(entry.get("document_id", "")).strip()
        if not doc_id:
            processed += 1
            if progress is not None:
                progress.update(1)
            continue
        entries_by_doc[doc_id].append((i, entry))

    logged_interval = max(100, total_entries // 20) if total_entries else 100
    for doc_id, doc_entries in entries_by_doc.items():
        doc_chunks = chunks_by_doc.get(doc_id)
        if not doc_chunks:
            processed += len(doc_entries)
            if progress is not None:
                progress.update(len(doc_entries))
            continue
        tokenized_chunks = [(chunk, _tokenize_match_text(chunk.raw_text)) for chunk in doc_chunks]
        for i, entry in doc_entries:
            question = str(entry.get("question", "")).strip()
            if question:
                gold, silver, silver_groups, labels = _build_support_targets(
                    entry,
                    doc_chunks,
                    tokenized_chunks=tokenized_chunks,
                )
                if labels:
                    examples.append(
                        RetrievalExample(
                            query_id=str(entry.get("id", f"q{i}")),
                            doc_id=doc_id,
                            question=question,
                            gold_chunk_ids=[str(chunk_id) for chunk_id in gold],
                            silver_chunk_ids=[str(chunk_id) for chunk_id in silver],
                            silver_chunk_groups=[[str(chunk_id) for chunk_id in group] for group in silver_groups],
                        )
                    )
            processed += 1
            if progress is not None:
                progress.update(1)
            elif processed % logged_interval == 0 or processed == total_entries:
                logger.info(
                    "Labeling retrieval examples progress=%d/%d built=%d",
                    processed,
                    total_entries,
                    len(examples),
                )
    if progress is not None:
        progress.close()
    logger.info("Built retrieval examples count=%d qa_entries=%d", len(examples), total_entries)
    return examples
