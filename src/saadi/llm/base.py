"""Defines the backend protocol and generation parameter object for LLM calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


# Deterministic indexing defaults. vLLM uses ``top_k=-1`` to disable top-k
# sampling.  Keep these values in one place so both the backend and run
# manifests describe the same effective decoding setup.
DEFAULT_GENERATION_SEED = 42
DEFAULT_GENERATION_TOP_P = 1.0
DEFAULT_GENERATION_TOP_K = -1
DEFAULT_REPETITION_PENALTY = 1.0


@dataclass(slots=True)
class GenerateParams:
    max_tokens: int = 1024
    temperature: float = 0.0
    stop: list[str] = field(default_factory=lambda: ["[END_JSON]"])


class LLMBackend(Protocol):
    def generate_batch(
        self,
        prompts: list[str],
        *,
        max_tokens: int,
        temperature: float,
        stop: list[str],
    ) -> list[str]:
        ...
