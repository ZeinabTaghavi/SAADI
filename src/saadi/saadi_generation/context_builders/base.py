"""Shared context-builder contracts for SAADI generation experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ...core.types import RetrievalResult


@dataclass(frozen=True, slots=True)
class ContextBuilderConfig:
    experiment_name: str
    top_k: int
    summary_k: int
    include_bridge: bool = True
    use_entity_memory: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RetrievedContextItem:
    chunk_id: str
    rank: int
    score: float | None = None
    raw_text: str = ""
    folded_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class QueryContextRequest:
    retrieval_result: RetrievalResult
    retrieved_items: list[RetrievedContextItem]
    question: str
    qa_entry: dict[str, Any]


@dataclass(frozen=True, slots=True)
class BuiltContext:
    context_text: str
    debug_payload: dict[str, Any]


class ContextBuilder(Protocol):
    name: str

    def build(self, request: QueryContextRequest) -> BuiltContext:
        ...

