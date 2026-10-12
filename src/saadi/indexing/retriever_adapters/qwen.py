"""Dense retriever adapter backed by a Qwen embedding model."""

from __future__ import annotations

from dataclasses import dataclass

from .hf_embedding import HFEmbeddingRetriever


@dataclass
class QwenRetriever(HFEmbeddingRetriever):
    model_name: str = "Qwen/Qwen3-Embedding-8B"
    trust_remote_code: bool = True
    pooling: str = "last_token"

    ENV_PREFIX = "SAADI_QWEN"


__all__ = ["QwenRetriever"]
