"""Ranks and selects relevant entities from memory for the current chunk."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Callable

from ..core.types import EntityRecord
from .active_entity_config import (
    DEFAULT_MAX_ALIASES,
    DEFAULT_MAX_DESC,
    DEFAULT_MAX_ENTITIES,
    DEFAULT_MIN_RELEVANCE_SCORE,
    normalize_active_entity_min_relevance,
)

_SELECTOR_ALIASES = {
    "word": "word",
    "words": "word",
    "overlap": "word",
    "word_overlap": "word",
    "token_overlap": "word",
    "aoword": "word",
    "aeword": "word",
    "bm25": "bm25",
    "aobm25": "bm25",
    "aebm25": "bm25",
}

# Deliberately local to the active-memory selector.  The downstream document
# retriever is an experimental control and must keep its existing tokenizer.
_ACTIVE_MEMORY_STOPWORDS = frozenset(
    {
        # English function words (including the possessive token produced by
        # apostrophe splitting).
        "a", "an", "and", "are", "as", "at", "be", "been", "being", "by",
        "for", "from", "had", "has", "have", "he", "her", "hers", "him", "his",
        "i", "in", "is", "it", "its", "of", "on", "or", "s", "she", "that",
        "the", "their", "them", "they", "this", "to", "was", "were", "will",
        "with", "you", "your",
        # Common Spanish function words.  These are particularly important for
        # names such as "Cofradía de Pescadores", which must not match a chunk
        # merely because both contain "de" or an article.
        "al", "como", "con", "de", "del", "el", "ella", "ellos", "en", "es",
        "esta", "este", "la", "las", "lo", "los", "o", "para", "por", "que",
        "se", "sin", "su", "sus", "un", "una", "unos", "unas", "y",
    }
)


def tokenize_active_entity_text(text: str) -> list[str]:
    """Return Unicode-aware, case-folded, stopword-free selector tokens."""
    if not isinstance(text, str):
        return []
    normalized = unicodedata.normalize("NFKC", text).casefold()
    # ``[^\W_]`` means Unicode alphanumeric without underscore under Python's
    # Unicode regex rules.  Accented letters therefore remain intact.
    tokens = re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)
    return [token for token in tokens if token and token not in _ACTIVE_MEMORY_STOPWORDS]


def _tokenize_basic(text: str) -> set[str]:
    """Backward-compatible set form used by the overlap selector."""
    return set(tokenize_active_entity_text(text))


def normalize_active_entity_selector(value: str | None) -> str:
    raw = str(value or "bm25").strip().lower()
    out = _SELECTOR_ALIASES.get(raw, raw)
    if out not in {"word", "bm25"}:
        raise ValueError(
            f"Invalid active_entity_selector: {value!r}. Expected one of: word, bm25."
        )
    return out


def active_entity_selector_label(value: str | None) -> str:
    selector = normalize_active_entity_selector(value)
    if selector == "bm25":
        return "AEBM25"
    return "AEWord"


def _entity_parts(obj: EntityRecord | dict) -> tuple[str, list[str], list[str]]:
    if isinstance(obj, EntityRecord):
        return obj.name, list(obj.aliases), list(obj.description)
    return (
        str(obj.get("name", "") or ""),
        [x for x in (obj.get("aliases") or []) if isinstance(x, str)],
        [x for x in (obj.get("description") or []) if isinstance(x, str)],
    )


def flatten_active_entity_text(obj: EntityRecord | dict, *, separator: str = ", ") -> str:
    name, aliases, descriptions = _entity_parts(obj)
    parts = [name.strip()] if isinstance(name, str) and name.strip() else []
    parts.extend(str(x).strip() for x in aliases if str(x).strip())
    parts.extend(str(x).strip() for x in descriptions if str(x).strip())
    return separator.join(parts)


def string_token_overlap_similarity(obj: EntityRecord | dict, chunk_text: str) -> float:
    chunk_tokens = _tokenize_basic(chunk_text)
    name = getattr(obj, "name", None) if not isinstance(obj, dict) else obj.get("name", "")
    aliases = getattr(obj, "aliases", None) if not isinstance(obj, dict) else obj.get("aliases", [])
    candidates = []
    if isinstance(name, str):
        candidates.append(name)
    if isinstance(aliases, list):
        candidates.extend([a for a in aliases if isinstance(a, str)])
    best = 0.0
    for cand in candidates:
        toks = _tokenize_basic(cand)
        if toks:
            best = max(best, len(toks & chunk_tokens) / max(1, len(toks)))
    return best


def _rank_entries_word(entries: list[str], chunk_text: str) -> list[str]:
    chunk_toks = _tokenize_basic(chunk_text)
    scored = []
    for idx, s in enumerate(entries or []):
        if not isinstance(s, str):
            continue
        toks = _tokenize_basic(s)
        score = len(toks & chunk_toks) / max(1, len(toks)) if toks else 0.0
        scored.append((score, idx, s))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [s for _, _, s in scored]


def _rank_entries_bm25(entries: list[str], query_text: str) -> list[str]:
    valid_entries = [str(s) for s in (entries or []) if isinstance(s, str)]
    if not valid_entries:
        return []
    scores = _bm25_scores(
        [tokenize_active_entity_text(entry) for entry in valid_entries],
        tokenize_active_entity_text(query_text),
    )
    ranked = sorted(enumerate(scores), key=lambda item: (-item[1], item[0]))
    return [valid_entries[idx] for idx, _ in ranked]


def _bm25_scores(corpus_tokens: list[list[str]], query_tokens: list[str]) -> list[float]:
    """Small native BM25 scorer over the selector's cleaned Unicode tokens."""
    if not corpus_tokens:
        return []
    if not query_tokens:
        return [0.0] * len(corpus_tokens)
    n_docs = len(corpus_tokens)
    lengths = [len(tokens) for tokens in corpus_tokens]
    avg_len = sum(lengths) / max(1, n_docs)
    term_freqs = [Counter(tokens) for tokens in corpus_tokens]
    document_frequency: Counter[str] = Counter()
    for tf in term_freqs:
        document_frequency.update(tf.keys())

    k1 = 1.5
    b = 0.75
    # Repeating a query word in a long chunk must not multiply its evidence.
    unique_query_tokens = list(dict.fromkeys(query_tokens))
    scores: list[float] = []
    for length, tf in zip(lengths, term_freqs):
        score = 0.0
        for token in unique_query_tokens:
            frequency = tf.get(token, 0)
            if frequency <= 0:
                continue
            df = document_frequency[token]
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            denominator = frequency + k1 * (
                1.0 - b + b * length / max(1e-9, avg_len)
            )
            score += idf * (frequency * (k1 + 1.0)) / denominator
        scores.append(float(score))
    return scores


def _rank_entities_bm25(
    previous_entities: list[EntityRecord] | list[dict],
    chunk_text: str,
) -> list[tuple[EntityRecord | dict, float]]:
    if not previous_entities:
        return []
    corpus_tokens = [
        tokenize_active_entity_text(flatten_active_entity_text(entity))
        for entity in previous_entities
    ]
    scores = _bm25_scores(corpus_tokens, tokenize_active_entity_text(chunk_text))
    ranked = sorted(enumerate(scores), key=lambda item: (-item[1], item[0]))
    return [(previous_entities[idx], float(score)) for idx, score in ranked]


def select_active_entities_with_scores(
    previous_entities: list[EntityRecord] | list[dict],
    chunk_text: str,
    *,
    max_entities: int = DEFAULT_MAX_ENTITIES,
    max_aliases: int = DEFAULT_MAX_ALIASES,
    max_desc: int = DEFAULT_MAX_DESC,
    selector: str = "bm25",
    min_relevance_score: float = DEFAULT_MIN_RELEVANCE_SCORE,
    similarity_fn: Callable | None = None,
) -> list[dict]:
    """Select prompt entries and expose their scores for diagnostics.

    Each item is ``{"score": float, "entry": <normal prompt entry>}``.  The
    ordinary :func:`select_active_entities` API continues to return only entries,
    so scores never leak into the LLM prompt.
    """
    if not previous_entities or max_entities <= 0:
        return []
    threshold = normalize_active_entity_min_relevance(min_relevance_score)
    normalized_selector = normalize_active_entity_selector(selector)
    rank_entries = _rank_entries_bm25 if normalized_selector == "bm25" else _rank_entries_word

    if normalized_selector == "bm25":
        scored = _rank_entities_bm25(previous_entities, chunk_text)
    else:
        score_fn = similarity_fn or string_token_overlap_similarity
        scored = []
        for index, obj in enumerate(previous_entities):
            try:
                score = float(score_fn(obj, chunk_text))
            except Exception:
                score = 0.0
            if not math.isfinite(score):
                score = 0.0
            scored.append((obj, score, index))
        scored.sort(key=lambda item: (-item[1], item[2]))
        scored = [(obj, score) for obj, score, _ in scored]

    selected: list[dict] = []
    for obj, score in scored:
        # Strict comparison is intentional: a zero threshold excludes zero-score
        # records and permits a genuinely empty result.
        if not math.isfinite(score) or score <= threshold:
            continue
        name, aliases, desc = _entity_parts(obj)
        entry = {
            "name": name,
            "description": rank_entries(desc, chunk_text)[:max_desc] if max_desc >= 0 else rank_entries(desc, chunk_text),
            "aliases": rank_entries(aliases, chunk_text)[:max_aliases] if max_aliases >= 0 else rank_entries(aliases, chunk_text),
        }
        selected.append({"score": float(score), "entry": entry})
        if len(selected) >= max_entities:
            break
    return selected


def select_active_entities(
    previous_entities: list[EntityRecord] | list[dict],
    chunk_text: str,
    *,
    max_entities: int = DEFAULT_MAX_ENTITIES,
    max_aliases: int = DEFAULT_MAX_ALIASES,
    max_desc: int = DEFAULT_MAX_DESC,
    selector: str = "bm25",
    min_relevance_score: float = DEFAULT_MIN_RELEVANCE_SCORE,
    similarity_fn: Callable | None = None,
) -> list[dict]:
    diagnostics = select_active_entities_with_scores(
        previous_entities,
        chunk_text,
        max_entities=max_entities,
        max_aliases=max_aliases,
        max_desc=max_desc,
        selector=selector,
        min_relevance_score=min_relevance_score,
        similarity_fn=similarity_fn,
    )
    return [item["entry"] for item in diagnostics]


# Compatibility surface for older scripts/configs. These wrappers deliberately
# retain the historical keyword names and cache labels.
tokenize_active_object_text = tokenize_active_entity_text
normalize_active_object_selector = normalize_active_entity_selector
flatten_active_object_text = flatten_active_entity_text


def active_object_selector_label(value: str | None) -> str:
    selector = normalize_active_entity_selector(value)
    return "AObm25" if selector == "bm25" else "AOWord"


def select_active_objects_with_scores(
    prev_objects: list[EntityRecord] | list[dict],
    chunk_text: str,
    *,
    max_objects: int = DEFAULT_MAX_ENTITIES,
    max_aliases: int = DEFAULT_MAX_ALIASES,
    max_desc: int = DEFAULT_MAX_DESC,
    selector: str = "bm25",
    min_relevance_score: float = DEFAULT_MIN_RELEVANCE_SCORE,
    similarity_fn: Callable | None = None,
) -> list[dict]:
    return select_active_entities_with_scores(
        prev_objects,
        chunk_text,
        max_entities=max_objects,
        max_aliases=max_aliases,
        max_desc=max_desc,
        selector=selector,
        min_relevance_score=min_relevance_score,
        similarity_fn=similarity_fn,
    )


def select_active_objects(
    prev_objects: list[EntityRecord] | list[dict],
    chunk_text: str,
    *,
    max_objects: int = DEFAULT_MAX_ENTITIES,
    max_aliases: int = DEFAULT_MAX_ALIASES,
    max_desc: int = DEFAULT_MAX_DESC,
    selector: str = "bm25",
    min_relevance_score: float = DEFAULT_MIN_RELEVANCE_SCORE,
    similarity_fn: Callable | None = None,
) -> list[dict]:
    return select_active_entities(
        prev_objects,
        chunk_text,
        max_entities=max_objects,
        max_aliases=max_aliases,
        max_desc=max_desc,
        selector=selector,
        min_relevance_score=min_relevance_score,
        similarity_fn=similarity_fn,
    )
