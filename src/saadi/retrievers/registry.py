"""Name registry for retrievers added at runtime.

Built-in retrievers (``bm25`` and the Hugging Face dense adapters) are resolved
first by ``saadi.indexing.retriever_adapters.get_retriever``; names registered
here are consulted afterwards, so a user class becomes selectable wherever a
retriever name is configured.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

from .base import EmbeddingRetriever, make_embedding_retriever, make_scorer_retriever

_REGISTRY: dict[str, type] = {}
# Objects wrapped by resolve_retriever, keyed by identity, so resolving the same
# object twice yields the same name and the object stays alive while registered.
_WRAPPED: dict[int, tuple[Any, str]] = {}


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
    key = normalize_retriever_name(name)
    _REGISTRY.pop(key, None)
    for object_id, (_obj, wrapped_key) in list(_WRAPPED.items()):
        if wrapped_key == key:
            del _WRAPPED[object_id]


def _required_positional_count(fn: Any) -> int | None:
    try:
        parameters = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return None
    return sum(
        1
        for p in parameters
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and p.default is inspect.Parameter.empty
    )


def _unique_key(base: str, cls: type) -> str:
    key = normalize_retriever_name(base) or "retriever"
    existing = _REGISTRY.get(key)
    if existing is None or existing is cls:
        return key
    return f"{key}_{id(cls) & 0xFFFFFF:06x}"


def _register_wrapped(obj: Any, cls: type) -> str:
    known = _WRAPPED.get(id(obj))
    if known is not None and known[0] is obj and _REGISTRY.get(known[1]) is not None:
        return known[1]
    key = _unique_key(cls.__name__, cls)
    register_retriever(key, cls)
    _WRAPPED[id(obj)] = (obj, key)
    return key


def resolve_retriever(spec: Any) -> str:
    """Return a retriever *name* for a name, a class, an instance, or a function.

    The pipelines pass retrievers around by name because the fusion step builds
    one retriever per text pool. Accepted inputs:

    - a registered or built-in name;
    - a retriever class, registered under its own name if needed;
    - a reranker object with ``score(query, texts)`` or a function of that shape;
    - an embedding model object with ``encode(texts)`` or ``embed(texts)``, such
      as a sentence-transformers model, or a function ``embed(texts) -> vectors``.

    A per-pool retriever *instance* (one that already holds a corpus) cannot be
    reused across pools; pass its class instead.
    """

    if isinstance(spec, str):
        from ..indexing.retriever_adapters import get_retriever

        get_retriever(spec)  # raises for unknown names
        return normalize_retriever_name(spec)
    if inspect.isclass(spec):
        for key, cls in _REGISTRY.items():
            if cls is spec:
                return key
        key = _unique_key(getattr(spec, "name", "") or spec.__name__, spec)
        register_retriever(key, spec)
        return key
    score = getattr(spec, "score", None)
    if callable(score) and not inspect.isfunction(spec) and not inspect.ismethod(spec):
        if (_required_positional_count(score) or 0) >= 2:
            return _register_wrapped(spec, make_scorer_retriever(spec))
        if hasattr(spec, "corpus"):
            raise TypeError(
                f"{type(spec).__name__} is a retriever instance bound to one corpus; pass the class "
                "(or its registered name) so the fusion step can build one retriever per text pool"
            )
    for attribute in ("encode", "embed", "embed_documents"):
        encoder = getattr(spec, attribute, None)
        if callable(encoder) and not inspect.isfunction(spec) and not inspect.ismethod(spec):
            name = f"{getattr(spec, 'name', None) or type(spec).__name__}_embedding_retriever"
            return _register_wrapped(spec, make_embedding_retriever(encoder, name=name))
    if callable(spec):
        if (_required_positional_count(spec) or 1) >= 2:
            return _register_wrapped(spec, make_scorer_retriever(spec))
        return _register_wrapped(spec, make_embedding_retriever(spec))
    raise TypeError(f"Cannot use an object of type {type(spec).__name__} as a retriever")


__all__ = [
    "EmbeddingRetriever",
    "make_scorer_retriever",
    "lookup_retriever",
    "normalize_retriever_name",
    "register_retriever",
    "registered_retrievers",
    "resolve_retriever",
    "unregister_retriever",
]
