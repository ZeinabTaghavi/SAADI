"""Reusable encoding and scoring helpers over project retriever adapters.

These helpers prepare independent text pools for later retrieval methods
without changing the existing end-to-end retrieval pipeline.
"""

from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from .bm25 import BM25Retriever, _tokenize


@dataclass(slots=True)
class EncodedTextPool:
    """Retriever-compatible representation of a text pool."""

    texts: list[str]
    chunk_ids: list[str]
    retriever: Any
    document_representations: Any
    representation_kind: str


# Per-pool document encodings are query-independent, so memoize them. Without
# this, MBR (and any per-document retrieval) re-encodes every document's chunk,
# name, alias, description, and summary pools once per query, which dominates
# runtime for large embedding models. Keyed on (retriever, kwargs, texts, ids)
# so distinct pools never collide, and bounded with an LRU to cap host memory.
_ENCODED_POOL_CACHE: "OrderedDict[tuple[str, str], EncodedTextPool]" = OrderedDict()


def _encoded_pool_cache_maxsize() -> int:
    raw = os.getenv("SAADI_ENCODED_POOL_CACHE_SIZE", "8192")
    try:
        value = int(raw)
    except Exception:
        return 8192
    return max(0, value)


def _encoded_pool_cache_key(
    retriever_name: str,
    texts: list[str],
    chunk_ids: list[str] | None,
    retriever_kwargs: dict[str, Any] | None,
) -> tuple[str, str]:
    digest = hashlib.sha1()
    digest.update(repr(sorted((retriever_kwargs or {}).items())).encode("utf-8"))
    digest.update(b"\x00texts\x00")
    for text in texts:
        digest.update(str(text).encode("utf-8"))
        digest.update(b"\x01")
    digest.update(b"\x00ids\x00")
    for chunk_id in chunk_ids or []:
        digest.update(str(chunk_id).encode("utf-8"))
        digest.update(b"\x01")
    return (str(retriever_name), digest.hexdigest())


def _is_embedding_retriever(retriever: Any) -> bool:
    return (
        not isinstance(retriever, BM25Retriever)
        and hasattr(retriever, "_encode_texts")
        and hasattr(retriever, "_corpus_embeddings")
    )


def _encode_dense_texts(retriever: Any, texts: list[str], *, text_kind: str) -> Any:
    try:
        return retriever._encode_texts(list(texts), text_kind=text_kind)
    except TypeError:
        return retriever._encode_texts(list(texts))


def build_encoded_text_pool(
    texts: list[str],
    *,
    retriever_name: str,
    chunk_ids: list[str] | None = None,
    retriever_kwargs: dict[str, Any] | None = None,
) -> EncodedTextPool:
    """Instantiate a retriever on one pool and expose its reusable document state.

    Results are memoized per (retriever, kwargs, texts, ids): document encodings
    do not depend on the query, so repeated calls (e.g. one per query over the
    same document pools) reuse the cached encoding instead of recomputing it.
    """

    from . import get_retriever

    text_list = list(texts)
    id_list = list(chunk_ids) if chunk_ids is not None else None

    cache_maxsize = _encoded_pool_cache_maxsize()
    cache_key: tuple[str, str] | None = None
    if cache_maxsize > 0:
        cache_key = _encoded_pool_cache_key(retriever_name, text_list, id_list, retriever_kwargs)
        cached = _ENCODED_POOL_CACHE.get(cache_key)
        if cached is not None:
            _ENCODED_POOL_CACHE.move_to_end(cache_key)
            return cached

    kwargs = dict(retriever_kwargs or {})
    Retriever = get_retriever(retriever_name)
    retriever = Retriever(corpus=text_list, chunk_ids=id_list, **kwargs)
    if _is_embedding_retriever(retriever):
        doc_representations = retriever._corpus_embeddings
        representation_kind = "embedding"
    else:
        doc_representations = encode_documents(text_list, retriever)
        representation_kind = "scoring_index"
    pool = EncodedTextPool(
        texts=text_list,
        chunk_ids=list(retriever.chunk_ids or []),
        retriever=retriever,
        document_representations=doc_representations,
        representation_kind=representation_kind,
    )

    if cache_key is not None:
        _ENCODED_POOL_CACHE[cache_key] = pool
        _ENCODED_POOL_CACHE.move_to_end(cache_key)
        while len(_ENCODED_POOL_CACHE) > cache_maxsize:
            _ENCODED_POOL_CACHE.popitem(last=False)

    return pool


def encode_documents(texts: list[str], retriever: Any) -> Any:
    """Encode a document pool into retriever-compatible representations."""

    if _is_embedding_retriever(retriever):
        return _encode_dense_texts(retriever, list(texts), text_kind="document")
    if isinstance(retriever, BM25Retriever):
        return [_tokenize(text) for text in texts]
    raise TypeError(f"Unsupported retriever type for document encoding: {type(retriever)!r}")


def encode_query(query: str, retriever: Any) -> Any:
    """Encode a query into retriever-compatible representations."""

    if _is_embedding_retriever(retriever):
        return _encode_dense_texts(retriever, [query], text_kind="query").squeeze(0)
    if isinstance(retriever, BM25Retriever):
        return str(query or "")
    raise TypeError(f"Unsupported retriever type for query encoding: {type(retriever)!r}")


def compute_similarity(query_representation: Any, document_representations: Any, retriever: Any) -> list[float]:
    """Compute one score per document in the pool."""

    if _is_embedding_retriever(retriever):
        scores = document_representations @ query_representation
        return [float(value) for value in scores.tolist()]
    if isinstance(retriever, BM25Retriever):
        return [float(value) for value in retriever._native_scores(str(query_representation or ""))]
    raise TypeError(f"Unsupported retriever type for similarity computation: {type(retriever)!r}")


__all__ = [
    "EncodedTextPool",
    "build_encoded_text_pool",
    "compute_similarity",
    "encode_documents",
    "encode_query",
]
