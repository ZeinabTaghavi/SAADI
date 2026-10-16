"""Pluggable query-time retrievers.

Select a retriever by name wherever ``retriever`` is configured. Built in:
``bm25`` (default, no extra dependencies), the Hugging Face dense adapters
``contriever``, ``bge_m3``, ``e5``, ``jina`` and ``qwen`` (``saadi[dense]``),
and the hosted ``openai_embedding`` and ``ollama_embedding`` retrievers
(``saadi[openai]``). Add your own with :func:`register_retriever`::

    from saadi.retrievers import BaseRetriever, register_retriever

    @register_retriever("keyword_overlap")
    class KeywordOverlap(BaseRetriever):
        def score(self, query):
            words = set(query.lower().split())
            return [len(words & set(text.lower().split())) for text in self.corpus]

or pass an object to :func:`resolve_retriever`: a reranker with
``score(query, texts)``, or an embedding model with ``encode(texts)``.
"""

from .base import BaseRetriever, EmbeddingRetriever, Retriever, make_embedding_retriever, make_scorer_retriever
from .openai_embeddings import OllamaEmbeddingRetriever, OpenAIEmbeddingRetriever
from .registry import (
    lookup_retriever,
    normalize_retriever_name,
    register_retriever,
    registered_retrievers,
    resolve_retriever,
    unregister_retriever,
)

register_retriever("openai_embedding", OpenAIEmbeddingRetriever)
register_retriever("openai-embedding", OpenAIEmbeddingRetriever)
register_retriever("ollama_embedding", OllamaEmbeddingRetriever)
register_retriever("ollama-embedding", OllamaEmbeddingRetriever)

__all__ = [
    "BaseRetriever",
    "EmbeddingRetriever",
    "OllamaEmbeddingRetriever",
    "OpenAIEmbeddingRetriever",
    "Retriever",
    "lookup_retriever",
    "make_embedding_retriever",
    "make_scorer_retriever",
    "normalize_retriever_name",
    "register_retriever",
    "registered_retrievers",
    "resolve_retriever",
    "unregister_retriever",
]
