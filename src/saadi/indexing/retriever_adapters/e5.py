"""Dense retriever adapter backed by an E5 embedding model."""

from __future__ import annotations

from dataclasses import dataclass

from .hf_embedding import HFEmbeddingRetriever


@dataclass
class E5Retriever(HFEmbeddingRetriever):
    model_name: str = "intfloat/e5-base-v2"
    query_prefix: str = "query: "
    document_prefix: str = "passage: "
    pooling: str = "mean"

    ENV_PREFIX = "SAADI_E5"


__all__ = ["E5Retriever"]
