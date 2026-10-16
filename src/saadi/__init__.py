"""Saadi: index-time document state for single-pass long-document retrieval."""

from ._version import __version__
from .document import Answer, HistoryItem, IndexReport, MemoryEntity, RetrievedChunk, Saadi
from .llm import resolve_backend
from .prompts import load_prompt_profile
from .retrievers import register_retriever, resolve_retriever

__all__ = [
    "Answer",
    "HistoryItem",
    "IndexReport",
    "MemoryEntity",
    "RetrievedChunk",
    "Saadi",
    "__version__",
    "load_prompt_profile",
    "register_retriever",
    "resolve_backend",
    "resolve_retriever",
]
