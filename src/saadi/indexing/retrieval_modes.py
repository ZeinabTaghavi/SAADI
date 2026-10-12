"""Retrieval corpus mode registry and helpers.

This module centralizes how retrieval text is selected so new retrieval modes can
be added without rewriting pipeline logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..core.types import CanonicalizedChunkRecord, ChunkRecord, CorpusArtifact
from .corpus_builder import build_corpus_from_canonicalized, build_corpus_from_raw_chunks


@dataclass(frozen=True)
class RetrievalModeSpec:
    """Defines a retrieval mode and which chunk representation it indexes."""

    name: str
    source: str
    description: str


RETRIEVAL_MODE_REGISTRY: dict[str, RetrievalModeSpec] = {
    "l0": RetrievalModeSpec(
        name="l0",
        source="canonicalized",
        description="Retrieval over L0 canonicalized chunk text.",
    ),
}

RETRIEVAL_MODE_ALIASES: dict[str, str] = {
    "canonicalized": "l0",
    "stateful": "l0",
    "contextualized": "l0",
    "canonical": "l0",
}


def normalize_retrieval_mode(value: str | None, *, allow_auto: bool = False) -> str:
    raw = str(value or "").strip().lower()
    if not raw and allow_auto:
        return "auto"
    if raw == "auto":
        if allow_auto:
            return "auto"
        raise ValueError("'auto' is not allowed in this context")
    normalized = RETRIEVAL_MODE_ALIASES.get(raw, raw)
    if normalized not in RETRIEVAL_MODE_REGISTRY:
        options = ", ".join(sorted(RETRIEVAL_MODE_REGISTRY.keys()))
        raise ValueError(f"Invalid retrieval mode: {value!r}. Expected one of: {options}")
    return normalized


def resolve_retrieval_mode(
    *,
    requested_mode: str | None,
    raw_chunks: list[ChunkRecord] | None,
    canonicalized_chunks: list[CanonicalizedChunkRecord] | None,
) -> str:
    mode = normalize_retrieval_mode(requested_mode, allow_auto=True)
    if mode != "auto":
        return mode
    if canonicalized_chunks is not None:
        return "l0"
    raise ValueError("Cannot resolve retrieval mode: no chunk inputs were provided")


def _build_corpus_from_l0(chunks: Iterable[CanonicalizedChunkRecord] | None) -> CorpusArtifact:
    if chunks is None:
        raise ValueError("l0 retrieval mode requires canonicalized_chunks")
    return build_corpus_from_canonicalized(chunks, metadata={"source": "canonicalized", "retrieval_mode": "l0"})


def build_corpus_for_mode(
    *,
    mode: str,
    raw_chunks: list[ChunkRecord] | None,
    canonicalized_chunks: list[CanonicalizedChunkRecord] | None,
) -> CorpusArtifact:
    resolved = normalize_retrieval_mode(mode)
    if resolved == "l0":
        return _build_corpus_from_l0(canonicalized_chunks)
    raise ValueError(f"Unsupported retrieval mode: {mode!r}")


def build_corpus_by_doc_for_mode(
    *,
    mode: str,
    raw_chunks: list[ChunkRecord] | None,
    canonicalized_chunks: list[CanonicalizedChunkRecord] | None,
) -> dict[str, CorpusArtifact]:
    resolved = normalize_retrieval_mode(mode)
    grouped_canon: dict[str, list[CanonicalizedChunkRecord]] = {}
    for chunk in canonicalized_chunks or []:
        grouped_canon.setdefault(str(chunk.doc_id), []).append(chunk)
    return {
        doc_id: build_corpus_from_canonicalized(chunks, metadata={"source": "canonicalized", "retrieval_mode": "l0"})
        for doc_id, chunks in grouped_canon.items()
    }
