"""Exports retriever adapters and the retriever factory."""

from __future__ import annotations

from .bge_m3 import BGEM3Retriever
from .bm25 import BM25Retriever
from .contriever import ContrieverRetriever
from .e5 import E5Retriever
from .jina import JinaRetriever
from .qwen import QwenRetriever
from .scoring import (
    EncodedTextPool,
    build_encoded_text_pool,
    compute_similarity,
    encode_documents,
    encode_query,
)

BUILTIN_RETRIEVERS = ("bm25", "contriever", "e5", "bge_m3", "qwen", "jina")


def get_retriever(name: str):
    normalized = str(name or "").strip().lower()
    if normalized in {"bm25", "bm25s"}:
        return BM25Retriever
    if normalized == "contriever":
        return ContrieverRetriever
    if normalized in {"e5", "intfloat-e5", "intfloat_e5"}:
        return E5Retriever
    if normalized in {"bge", "bge_m3", "bge-m3", "bge_m3_embedding", "bge-m3-embedding"}:
        return BGEM3Retriever
    if normalized in {"qwen", "qwen_embedding", "qwen-embedding"}:
        return QwenRetriever
    if normalized in {"jina", "jina_embedding", "jina-embedding"}:
        return JinaRetriever
    # Retrievers registered at runtime through saadi.retrievers; imported lazily
    # so this package stays import-light and free of cycles.
    from ...retrievers.registry import lookup_retriever, registered_retrievers

    registered = lookup_retriever(normalized)
    if registered is not None:
        return registered
    known = ", ".join([*BUILTIN_RETRIEVERS, *registered_retrievers()])
    raise ValueError(f"Unknown retriever: {name}. Known retrievers: {known}")


__all__ = [
    "BUILTIN_RETRIEVERS",
    "BM25Retriever",
    "BGEM3Retriever",
    "ContrieverRetriever",
    "E5Retriever",
    "JinaRetriever",
    "QwenRetriever",
    "EncodedTextPool",
    "build_encoded_text_pool",
    "compute_similarity",
    "encode_documents",
    "encode_query",
    "get_retriever",
]
