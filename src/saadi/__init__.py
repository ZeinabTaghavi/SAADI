"""Saadi: index-time document state for single-pass long-document retrieval."""

from .llm import resolve_backend
from .retrievers import register_retriever, resolve_retriever

__all__ = ["register_retriever", "resolve_backend", "resolve_retriever"]
