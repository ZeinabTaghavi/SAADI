"""Adapts a plain Python function to the backend contract."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from typing import Any

from ._concurrency import map_concurrently
from .base import BaseLLMBackend, ChatPrompt


def _accepts_generation_keywords(fn: Callable[..., Any]) -> bool:
    try:
        parameters = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return True
    names = {p.name for p in parameters}
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters):
        return True
    return {"max_tokens", "temperature", "stop"} <= names


class CallableBackend(BaseLLMBackend):
    """Wrap ``fn(system, user, *, max_tokens, temperature, stop) -> str`` as a backend.

    A function that takes only ``(system, user)`` works as well. Pass an
    object's bound ``generate`` method to adapt an existing client.
    """

    def __init__(self, fn: Callable[..., Any], *, max_workers: int = 1, name: str = "callable") -> None:
        if not callable(fn):
            raise TypeError("CallableBackend needs a callable")
        self.fn = fn
        self.max_workers = int(max_workers)
        self.name = str(name)
        self._with_keywords = _accepts_generation_keywords(fn)
        self.runtime_metadata: dict[str, Any] = {"backend": "callable", "model_name": self.name}

    def generate_chat_batch(
        self,
        prompts: Sequence[ChatPrompt],
        *,
        max_tokens: int,
        temperature: float,
        stop: list[str],
    ) -> list[str]:
        def one(prompt: ChatPrompt) -> str:
            if self._with_keywords:
                out = self.fn(prompt.system, prompt.user, max_tokens=int(max_tokens), temperature=float(temperature), stop=list(stop))
            else:
                out = self.fn(prompt.system, prompt.user)
            return str(out or "")

        return map_concurrently(one, prompts, max_workers=self.max_workers)
