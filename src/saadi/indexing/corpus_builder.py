"""Builds retrieval corpora from raw or canonicalized chunks."""

from __future__ import annotations

from typing import Iterable

from ..core.types import CanonicalizedChunkRecord, ChunkRecord, CorpusArtifact


def build_corpus_from_raw_chunks(chunks: Iterable[ChunkRecord], *, metadata: dict | None = None) -> CorpusArtifact:
    chunk_ids = []
    chunk_texts = []
    for c in chunks:
        chunk_ids.append(c.chunk_id)
        chunk_texts.append(c.raw_text)
    return CorpusArtifact(chunk_ids=chunk_ids, chunk_texts=chunk_texts, metadata=metadata or {"source": "raw"})


def build_corpus_from_canonicalized(chunks: Iterable[CanonicalizedChunkRecord], *, metadata: dict | None = None) -> CorpusArtifact:
    chunk_ids = []
    chunk_texts = []
    for c in chunks:
        chunk_ids.append(c.chunk_id)
        chunk_texts.append(c.canonicalized_chunk_text)
    return CorpusArtifact(chunk_ids=chunk_ids, chunk_texts=chunk_texts, metadata=metadata or {"source": "canonicalized"})
