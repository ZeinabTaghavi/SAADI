"""Provides a mock LLM backend for tests and local debugging."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field

from ..core.parsers import extract_tagged_section
from .base import RenderedStringBackend

MockGenerator = Callable[[str], str]


def _default_mock_generator(prompt: str) -> str:
    rollup_items = extract_tagged_section(prompt, "ROLLUP_ITEMS_JSON")
    if rollup_items:
        try:
            items = json.loads(rollup_items)
        except Exception:
            items = []
        if not isinstance(items, list):
            items = []
        joined = " ".join(str(x).strip() for x in items if str(x).strip())
        words = joined.split()
        summary = " ".join(words[:60]) if words else ""
        return summary + ("\n[END_SUMMARY]" if summary else "[END_SUMMARY]")

    _ = extract_tagged_section(prompt, "HIERARCHY_CONTEXT_JSON")
    obj = {
        "active_entities": [],
        "insertions": [],
    }
    return json.dumps(obj, ensure_ascii=False) + "\n[END_JSON]"


@dataclass
class MockBackend(RenderedStringBackend):
    """CPU-only stand-in that answers from the prompt text alone.

    ``generator`` receives the rendered prompt (see :meth:`ChatPrompt.as_text`)
    and returns the model text. Every rendered prompt is appended to ``calls``.
    """

    generator: MockGenerator = _default_mock_generator
    calls: list[str] = field(default_factory=list)
    runtime_metadata: dict[str, object] = field(default_factory=lambda: {"backend": "mock", "model_name": "mock"})

    def generate_batch(
        self,
        prompts: list[str],
        *,
        max_tokens: int,
        temperature: float,
        stop: list[str],
    ) -> list[str]:
        outs: list[str] = []
        for p in prompts:
            self.calls.append(p)
            outs.append(self.generator(p))
        return outs
