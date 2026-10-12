"""Defines the context strategy interface and shared strategy config fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..core.types import CanonicalizationOutput, ChunkRecord, DiscourseState
from ..entities.active_entity_config import (
    DEFAULT_MAX_ALIASES,
    DEFAULT_MAX_DESC,
    DEFAULT_MAX_ENTITIES,
    DEFAULT_MIN_RELEVANCE_SCORE,
)
from ..llm.base import GenerateParams, LLMBackend


class ContextStrategy(Protocol):
    name: str

    def build_prompt(self, chunk: ChunkRecord, state: DiscourseState) -> str:
        ...

    def apply_output(
        self,
        chunk: ChunkRecord,
        state: DiscourseState,
        raw_model_text: str,
        *,
        backend: LLMBackend | None = None,
        generate: GenerateParams | None = None,
    ) -> tuple[CanonicalizationOutput, DiscourseState]:
        ...


@dataclass
class StrategyCommonConfig:
    max_entities: int = DEFAULT_MAX_ENTITIES
    max_aliases: int = DEFAULT_MAX_ALIASES
    max_desc: int = DEFAULT_MAX_DESC
    active_entity_selector: str = "bm25"
    active_entity_min_relevance: float = DEFAULT_MIN_RELEVANCE_SCORE
    desc_word_limit: int = 10
    max_tokens: int = 1024
    temperature: float = 0.0
    stop: list[str] | None = None

    def stop_tokens(self) -> list[str]:
        return self.stop or ["[END_JSON]"]
