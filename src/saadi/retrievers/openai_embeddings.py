"""Dense retrieval through an OpenAI-compatible embeddings endpoint.

Works with OpenAI itself and with servers that implement ``/v1/embeddings``
such as Ollama, LM Studio, and ``vllm serve``. Nothing here needs a GPU or
PyTorch, which makes it the natural partner of the hosted LLM backends.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np

from ..llm.openai_backend import OLLAMA_DEFAULT_BASE_URL, OPENAI_DEFAULT_BASE_URL, resolve_api_key
from .base import EmbeddingRetriever


class OpenAIEmbeddingRetriever(EmbeddingRetriever):
    """Embeddings from ``client.embeddings.create`` over the OpenAI protocol.

    Options can be given as constructor keywords or, because the fusion step
    constructs retrievers with no options, through environment variables named
    ``<ENV_PREFIX>_MODEL``, ``<ENV_PREFIX>_BASE_URL`` and ``<ENV_PREFIX>_BATCH_SIZE``.
    """

    DEFAULT_MODEL: ClassVar[str] = "text-embedding-3-small"
    DEFAULT_BASE_URL: ClassVar[str | None] = None
    DEFAULT_API_KEY_ENV: ClassVar[str] = "OPENAI_API_KEY"
    ENV_PREFIX: ClassVar[str] = "SAADI_OPENAI_EMBEDDING"
    PROVIDER_LABEL: ClassVar[str] = "openai"
    _CLIENT_CACHE: ClassVar[dict[tuple[str, str], Any]] = {}

    def __init__(
        self,
        corpus: Sequence[str],
        chunk_ids: Sequence[str] | None = None,
        k: int = 10,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
        batch_size: int | None = None,
        dimensions: int | None = None,
        timeout: float = 120.0,
        max_retries: int = 4,
        client: Any | None = None,
    ) -> None:
        env = os.environ
        self.model = str(model or env.get(f"{self.ENV_PREFIX}_MODEL") or self.DEFAULT_MODEL)
        resolved_base = base_url or env.get(f"{self.ENV_PREFIX}_BASE_URL") or self.DEFAULT_BASE_URL
        if resolved_base is None and self.PROVIDER_LABEL == "openai":
            resolved_base = env.get("OPENAI_BASE_URL") or OPENAI_DEFAULT_BASE_URL
        self.base_url = str(resolved_base or OPENAI_DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key
        self.api_key_env = api_key_env or self.DEFAULT_API_KEY_ENV
        self.batch_size = max(1, int(batch_size or env.get(f"{self.ENV_PREFIX}_BATCH_SIZE") or 100))
        self.dimensions = int(dimensions) if dimensions else None
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self._client = client
        super().__init__(corpus, chunk_ids, k)

    def client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import openai  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise RuntimeError(
                "The OpenAI-compatible embedding retriever needs the `openai` package. Install it with `pip install 'saadi[openai]'`."
            ) from exc
        key = resolve_api_key(
            self.api_key, self.api_key_env, base_url=self.base_url, provider_label=f"{self.PROVIDER_LABEL} embedding retriever"
        )
        cache_key = (self.base_url, key)
        cached = self._CLIENT_CACHE.get(cache_key)
        if cached is None:
            cached = openai.OpenAI(api_key=key, base_url=self.base_url, timeout=self.timeout, max_retries=self.max_retries)
            self._CLIENT_CACHE[cache_key] = cached
        self._client = cached
        return cached

    def embed(self, texts: list[str], *, text_kind: str = "document") -> np.ndarray:
        client = self.client()
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [str(text) if str(text).strip() else " " for text in texts[start : start + self.batch_size]]
            request: dict[str, Any] = {"model": self.model, "input": batch}
            if self.dimensions:
                request["dimensions"] = self.dimensions
            response = client.embeddings.create(**request)
            items = sorted(response.data, key=lambda item: int(getattr(item, "index", 0)))
            if len(items) != len(batch):
                raise RuntimeError(f"Embedding server returned {len(items)} vectors for {len(batch)} texts")
            vectors.extend([float(value) for value in item.embedding] for item in items)
        return np.asarray(vectors, dtype=np.float32)


class OllamaEmbeddingRetriever(OpenAIEmbeddingRetriever):
    """Embeddings from a local Ollama server, for example ``nomic-embed-text``."""

    DEFAULT_MODEL = "nomic-embed-text"
    DEFAULT_BASE_URL = OLLAMA_DEFAULT_BASE_URL
    DEFAULT_API_KEY_ENV = "OLLAMA_API_KEY"
    ENV_PREFIX = "SAADI_OLLAMA_EMBEDDING"
    PROVIDER_LABEL = "ollama"

    def __init__(self, corpus: Sequence[str], chunk_ids: Sequence[str] | None = None, k: int = 10, **options: Any) -> None:
        options.setdefault(
            "base_url", os.environ.get("OLLAMA_BASE_URL") or os.environ.get(f"{self.ENV_PREFIX}_BASE_URL") or self.DEFAULT_BASE_URL
        )
        super().__init__(corpus, chunk_ids, k, **options)


__all__ = ["OllamaEmbeddingRetriever", "OpenAIEmbeddingRetriever"]
