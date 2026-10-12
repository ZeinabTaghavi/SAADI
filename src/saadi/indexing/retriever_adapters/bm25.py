"""Implements the BM25 baseline retriever with a native fallback."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable


def _normalize_token(tok: str) -> str:
    # Lightweight normalization so the pure-Python fallback is less brittle
    # for simple tense/plural variants (sign/signed, approve/approved, plans/plan).
    if len(tok) > 5 and tok.endswith("ing"):
        return tok[:-3]
    if len(tok) > 4 and tok.endswith("ed"):
        return tok[:-2]
    if len(tok) > 4 and tok.endswith("es"):
        return tok[:-2]
    if len(tok) > 3 and tok.endswith("s"):
        return tok[:-1]
    return tok


def _tokenize(text: str) -> list[str]:
    return [_normalize_token(t) for t in re.findall(r"[A-Za-z0-9]+", (text or "").lower())]


@dataclass
class BM25Retriever:
    corpus: list[str]
    chunk_ids: list[str] | None = None
    k: int = 10
    use_bm25s_if_available: bool = True
    _backend: str = "native"

    def __post_init__(self):
        self.chunk_ids = self.chunk_ids or [str(i) for i in range(len(self.corpus))]
        if len(self.chunk_ids) != len(self.corpus):
            raise ValueError("chunk_ids length must match corpus length")
        if len(set(self.chunk_ids)) != len(self.chunk_ids):
            raise ValueError("chunk_ids must be unique")
        if len(set(self.corpus)) != len(self.corpus):
            # duplicates are allowed in some corpora, but mapping by string becomes ambiguous.
            pass
        self._bm25s = None
        self._build_native()
        if self.use_bm25s_if_available:
            try:
                import bm25s  # type: ignore
                self._bm25s = bm25s
                tokenized = bm25s.tokenize(self.corpus)
                self._retriever = bm25s.BM25(corpus=self.corpus)
                self._retriever.index(tokenized)
                self._backend = "bm25s"
                return
            except Exception:
                self._bm25s = None

    def _build_native(self):
        self._backend = "native"
        self.docs_tokens = [_tokenize(t) for t in self.corpus]
        self.doc_len = [len(t) for t in self.docs_tokens]
        self.avgdl = sum(self.doc_len) / max(1, len(self.doc_len))
        self.term_freqs = [Counter(toks) for toks in self.docs_tokens]
        self.df = defaultdict(int)
        for tf in self.term_freqs:
            for term in tf.keys():
                self.df[term] += 1
        self.N = len(self.corpus)

    def _native_scores(self, query: str) -> list[float]:
        q_terms = _tokenize(query)
        if not q_terms:
            return [0.0] * len(self.corpus)
        k1 = 1.5
        b = 0.75
        scores = [0.0] * len(self.corpus)
        for i, tf in enumerate(self.term_freqs):
            dl = self.doc_len[i] or 1
            s = 0.0
            for t in q_terms:
                f = tf.get(t, 0)
                if f == 0:
                    continue
                df = self.df.get(t, 0)
                idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
                denom = f + k1 * (1 - b + b * dl / max(1e-9, self.avgdl))
                s += idf * (f * (k1 + 1)) / denom
            scores[i] = s
        return scores

    def retrieve(self, query: str, *, k: int | None = None) -> tuple[list[str], list[float], list[int]]:
        topk = k or self.k
        if self._backend == "bm25s":
            q_tokens = self._bm25s.tokenize(query)
            docs, scores = self._retriever.retrieve(q_tokens, k=topk)
            top_docs = list(docs[0])
            top_scores = [float(s) for s in scores[0]]
            # map strings back to first index occurrence, deterministic but ambiguous if duplicates.
            idxs = [self.corpus.index(d) for d in top_docs]
            return [self.chunk_ids[i] for i in idxs], top_scores, idxs
        scores = self._native_scores(query)
        ranked = sorted(enumerate(scores), key=lambda x: (-x[1], x[0]))[:topk]
        idxs = [i for i, _ in ranked]
        return [self.chunk_ids[i] for i in idxs], [float(s) for _, s in ranked], idxs

    def retrieve_data(self, query: str):
        ids, scores, idxs = self.retrieve(query)
        texts = [self.corpus[i] for i in idxs]
        return texts, scores


def get_retriever(name: str):
    from . import get_retriever as package_get_retriever

    return package_get_retriever(name)
