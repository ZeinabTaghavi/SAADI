"""Dense retriever adapter backed by BGE M3."""

from __future__ import annotations

from dataclasses import dataclass

from .hf_embedding import HFEmbeddingRetriever


@dataclass
class BGEM3Retriever(HFEmbeddingRetriever):
    model_name: str = "BAAI/bge-m3"
    batch_size: int = 16
    max_length: int = 8192
    pooling: str = "cls"

    ENV_PREFIX = "SAADI_BGE_M3"


__all__ = ["BGEM3Retriever"]
