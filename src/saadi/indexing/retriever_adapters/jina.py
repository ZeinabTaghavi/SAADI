"""Dense retriever adapter backed by a Jina embedding model."""

from __future__ import annotations

from dataclasses import dataclass

from .hf_embedding import HFEmbeddingRetriever


@dataclass
class JinaRetriever(HFEmbeddingRetriever):
    model_name: str = "jinaai/jina-embeddings-v3"
    trust_remote_code: bool = True
    pooling: str = "mean"

    ENV_PREFIX = "SAADI_JINA"


__all__ = ["JinaRetriever"]
