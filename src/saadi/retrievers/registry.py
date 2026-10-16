"""Name registry for retrievers added at runtime.

Built-in retrievers (``bm25`` and the Hugging Face dense adapters) are resolved
first by ``saadi.indexing.retriever_adapters.get_retriever``; names registered
here are consulted afterwards, so a user class becomes selectable wherever a
retriever name is configured.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

from .base import EmbeddingRetriever, make_embedding_retriever

_REGISTRY: dict[str, type] = {}


def normalize_retriever_name(name: str) -> str:
    return str(name or "").strip().lower()


def register_retriever(name: str, retriever_class: type | None = None) -> Any:
    """Register a retriever class under ``name``. Usable as a decorator.

    The class is constructed as ``cls(corpus=..., chunk_ids=..., k=...)``, so
    subclassing :class:`saadi.retrievers.BaseRetriever` is the easy route.
    Registering the same class twice under one name is allowed; a different
    class under an existing name is an error.
    """

    key = normalize_retriever_name(name)
    if not key:
        raise ValueError("Retriever name must not be empty")

    def _register(cls: type) -> type:
        if not inspect.isclass(cls):
            raise TypeError("register_retriever expects a class; use make_embedding_retriever for a function")
        existing = _REGISTRY.get(key)
        if existing is not None and existing is not cls:
            raise ValueError(f"A different retriever is already registered as {key!r}")
        _REGISTRY[key] = cls
        return cls

    if retriever_class is None:
        return _register
    return _register(retriever_class)


def lookup_retriever(name: str) -> type | None:
    return _REGISTRY.get(normalize_retriever_name(name))


def registered_retrievers() -> list[str]:
    return sorted(_REGISTRY)


def unregister_retriever(name: str) -> None:
    _REGISTRY.pop(normalize_retriever_name(name), None)


def resolve_retriever(spec: Any) -> str:
    """Return a retriever *name* for a name, a retriever class, or an embedding function.

    The pipelines pass retrievers around by name because the fusion step builds
    one instance per text pool. A class is registered under its own name if it
    is not registered yet; a function ``embed(texts) -> vectors`` is wrapped in
    an :class:`EmbeddingRetriever` and registered.
    """

    if isinstance(spec, str):
        from ..indexing.retriever_adapters import get_retriever

        get_retriever(spec)  # raises for unknown names
        return normalize_retriever_name(spec)
    if inspect.isclass(spec):
        for key, cls in _REGISTRY.items():
            if cls is spec:
                return key
        key = normalize_retriever_name(getattr(spec, "name", "") or spec.__name__)
        register_retriever(key, spec)
        return key
    if callable(spec):
        cls = make_embedding_retriever(spec)
        key = normalize_retriever_name(cls.__name__)
        register_retriever(key, cls)
        return key
    raise TypeError(f"Cannot use an object of type {type(spec).__name__} as a retriever")


__all__ = [
    "EmbeddingRetriever",
    "lookup_retriever",
    "normalize_retriever_name",
    "register_retriever",
    "registered_retrievers",
    "resolve_retriever",
    "unregister_retriever",
]
