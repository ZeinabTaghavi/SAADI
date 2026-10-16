"""Retriever contract for the query-time scoring channels.

SAADI's fusion step scores a query against several text pools of one document
(the original chunks, the entity names, aliases and descriptions in memory, and
the hierarchical summaries) and combines the scores. Each pool gets its own
retriever instance, built from the retriever *class* that a name resolves to.

A retriever therefore only needs to do one thing: given a corpus, return one
relevance score per text for a query, higher meaning more relevant. Scores are
z-normalised per channel before fusion, so their scale does not matter.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Retriever(Protocol):
    """What the fusion step and the retrieval pipeline require."""

    corpus: list[str]
    chunk_ids: list[str]

    def score(self, query: str) -> list[float]:
        """Return one relevance score per corpus text."""

    def retrieve(self, query: str, *, k: int | None = None) -> tuple[list[str], list[float], list[int]]:
        """Return the top-k ``(chunk_ids, scores, corpus_indices)`` in descending order."""


class BaseRetriever:
    """Implements ``retrieve`` on top of ``score``; subclasses implement ``score``.

    Constructor keyword names match the built-in retrievers, so any subclass
    can be registered and selected by name wherever ``retriever`` is configured.
    """

    def __init__(self, corpus: Sequence[str], chunk_ids: Sequence[str] | None = None, k: int = 10, **options: Any) -> None:
        self.corpus = [str(text) for text in corpus]
        self.chunk_ids = [str(item) for item in chunk_ids] if chunk_ids is not None else [str(i) for i in range(len(self.corpus))]
        if len(self.chunk_ids) != len(self.corpus):
            raise ValueError("chunk_ids length must match corpus length")
        if len(set(self.chunk_ids)) != len(self.chunk_ids):
            raise ValueError("chunk_ids must be unique")
        self.k = int(k)
        if options:
            unknown = ", ".join(sorted(options))
            raise TypeError(f"{type(self).__name__} got unexpected options: {unknown}")

    def score(self, query: str) -> list[float]:
        raise NotImplementedError

    def retrieve(self, query: str, *, k: int | None = None) -> tuple[list[str], list[float], list[int]]:
        if not self.corpus:
            return [], [], []
        top_k = min(int(k or self.k), len(self.corpus))
        if top_k <= 0:
            return [], [], []
        scores = [float(value) for value in self.score(query)]
        if len(scores) != len(self.corpus):
            raise ValueError(f"{type(self).__name__}.score returned {len(scores)} scores for {len(self.corpus)} texts")
        ranked = sorted(range(len(scores)), key=lambda index: (-scores[index], index))[:top_k]
        return [self.chunk_ids[i] for i in ranked], [scores[i] for i in ranked], ranked

    def retrieve_data(self, query: str) -> tuple[list[str], list[float]]:
        _ids, scores, indices = self.retrieve(query)
        return [self.corpus[i] for i in indices], scores


EmbedFunction = Callable[[list[str]], Any]


def _as_matrix(vectors: Any, *, count: int) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim == 1 and count == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[0] != count:
        raise ValueError(f"embed returned an array of shape {matrix.shape} for {count} texts")
    return matrix


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class EmbeddingRetriever(BaseRetriever):
    """Dense retriever over any embedding function.

    Subclasses set ``embed_fn`` (a function from a list of texts to a 2-D array
    of vectors) or override :meth:`embed`. Corpus vectors are computed once at
    construction; a query is scored by dot product, cosine when ``normalize``
    is on. The class also exposes the two attributes the fusion code uses to
    recognise dense retrievers, so it runs on the same path as the built-in
    Hugging Face adapters.
    """

    embed_fn: EmbedFunction | None = None
    normalize: bool = True

    def __init__(self, corpus: Sequence[str], chunk_ids: Sequence[str] | None = None, k: int = 10, **options: Any) -> None:
        super().__init__(corpus, chunk_ids, k, **options)
        self._corpus_embeddings = self._encode_texts(self.corpus, text_kind="document")

    def embed(self, texts: list[str], *, text_kind: str = "document") -> Any:
        fn = type(self).embed_fn
        if fn is None:
            raise NotImplementedError(f"{type(self).__name__} needs embed_fn or an embed() override")
        return fn(list(texts))

    def _encode_texts(self, texts: list[str], *, text_kind: str = "document") -> np.ndarray:
        texts = list(texts)
        if not texts:
            width = self._corpus_embeddings.shape[1] if getattr(self, "_corpus_embeddings", None) is not None else 1
            return np.zeros((0, width), dtype=np.float32)
        matrix = _as_matrix(self.embed(texts, text_kind=text_kind), count=len(texts))
        return _normalize_rows(matrix) if self.normalize else matrix

    def score(self, query: str) -> list[float]:
        if not self.corpus:
            return []
        query_vector = self._encode_texts([str(query or "")], text_kind="query")[0]
        return [float(value) for value in (self._corpus_embeddings @ query_vector).tolist()]


def _class_name(text: str, default: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(text or "")).strip("_")
    return safe or default


def make_embedding_retriever(embed: EmbedFunction, *, name: str | None = None, normalize: bool = True) -> type[EmbeddingRetriever]:
    """Turn ``embed(texts) -> vectors`` into a retriever class ready for registration."""

    if not callable(embed):
        raise TypeError("make_embedding_retriever needs a callable that maps texts to vectors")
    class_name = _class_name(name or f"{getattr(embed, '__name__', 'custom')}_embedding_retriever", "custom_embedding_retriever")
    return type(class_name, (EmbeddingRetriever,), {"embed_fn": staticmethod(embed), "normalize": bool(normalize)})


ScoreFunction = Callable[[str, list[str]], Any]


def make_scorer_retriever(scorer: Any, *, name: str | None = None) -> type[BaseRetriever]:
    """Turn a stateless scorer into a retriever class ready for registration.

    ``scorer`` is either a function ``(query, texts) -> scores`` or an object
    with such a ``score`` method, for example a cross-encoder reranker. Higher
    scores must mean more relevant; the scale does not matter.
    """

    score_fn = getattr(scorer, "score", None)
    if not callable(score_fn):
        score_fn = scorer
    if not callable(score_fn):
        raise TypeError("make_scorer_retriever needs a function (query, texts) -> scores or an object with score()")
    label = name or getattr(scorer, "name", None) or getattr(scorer, "__name__", None) or type(scorer).__name__
    class_name = _class_name(f"{label}_scorer_retriever", "custom_scorer_retriever")

    def score(self: BaseRetriever, query: str) -> list[float]:
        return [float(value) for value in score_fn(str(query or ""), list(self.corpus))]

    return type(class_name, (BaseRetriever,), {"score": score})
