"""Backend contract shared by every language-model integration.

Pipelines describe each request as a :class:`ChatPrompt` (a system instruction
plus a user message). Each backend turns that into its provider's own format:
the vLLM backend applies the model's chat template, hosted APIs send a messages
array, and the mock backend renders a fixed text form for tests.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..core.parsers import _strip_reasoning_markup as _strip_closed_reasoning_blocks
from ..core.prompts import chatml_wrap, format_chat_prompt

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


@dataclass(frozen=True, slots=True)
class ChatPrompt:
    """One chat request: a system instruction and a user message."""

    system: str = ""
    user: str = ""

    def messages(self) -> list[dict[str, str]]:
        """Return the request as a provider-neutral messages array."""

        messages: list[dict[str, str]] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": self.user})
        return messages

    def as_text(self) -> str:
        """Render the request as one string for logs, hashes, and the mock backend."""

        return chatml_wrap(self.system, self.user)


@runtime_checkable
class LLMBackend(Protocol):
    """What every pipeline requires from a backend."""

    def generate_chat_batch(
        self,
        prompts: Sequence[ChatPrompt],
        *,
        max_tokens: int,
        temperature: float,
        stop: list[str],
    ) -> list[str]: ...


class BaseLLMBackend:
    """Shared behaviour for the built-in backends.

    Subclasses implement :meth:`generate_chat_batch`. The other methods have
    sensible defaults: :meth:`render_prompt` gives the text form a backend
    would send, and :meth:`generate_batch` keeps the older raw-string path
    working by treating each string as a user message.
    """

    chat_tokenizer: Any = None

    def generate_chat_batch(
        self,
        prompts: Sequence[ChatPrompt],
        *,
        max_tokens: int,
        temperature: float,
        stop: list[str],
    ) -> list[str]:
        raise NotImplementedError

    def render_prompt(self, prompt: ChatPrompt) -> str:
        return prompt.as_text()

    def generate_batch(
        self,
        prompts: list[str],
        *,
        max_tokens: int,
        temperature: float,
        stop: list[str],
    ) -> list[str]:
        return self.generate_chat_batch(
            [parse_chatml(str(prompt)) for prompt in prompts],
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )

    def close(self) -> None:
        """Release network or GPU resources. Safe to call more than once."""


_CHATML_PATTERN = re.compile(
    r"\A<\|system\|>\n(?P<system>.*?)\n<\|user\|>\n(?P<user>.*)\n<\|assistant\|>\n\Z",
    re.DOTALL,
)


def parse_chatml(text: str) -> ChatPrompt:
    """Invert :func:`chatml_wrap`; any other string becomes a plain user message.

    Older callers hand backends pre-rendered strings. Recovering the two roles
    lets hosted APIs receive a real system message instead of markup.
    """

    match = _CHATML_PATTERN.match(str(text or ""))
    if match is None:
        return ChatPrompt(user=str(text or ""))
    return ChatPrompt(system=match.group("system"), user=match.group("user"))


def render_prompt(backend: Any, prompt: ChatPrompt) -> str:
    """Return the text a backend sends for ``prompt``, for artifacts and hashes."""

    renderer = getattr(backend, "render_prompt", None)
    if callable(renderer):
        return str(renderer(prompt))
    return prompt.as_text()


_UNCLOSED_REASONING = re.compile(r"<think>|<\|begin_of_thought\|>", re.IGNORECASE)


def strip_reasoning_markup(text: str) -> str:
    """Remove the reasoning markup some chat models emit before the answer.

    Closed ``<think>...</think>`` and ``<|begin_of_thought|>...<|end_of_thought|>``
    blocks are removed with the same helper the output parsers use. An opening
    tag without a close means the output was cut off while still reasoning, so
    everything from that tag on is dropped; callers then see an empty answer
    and fall back instead of treating reasoning text as the result.
    """

    cleaned = _strip_closed_reasoning_blocks(str(text or ""))
    unclosed = _UNCLOSED_REASONING.search(cleaned)
    if unclosed is not None:
        cleaned = cleaned[: unclosed.start()].rstrip()
    return cleaned


def package_version(name: str) -> str | None:
    """Installed version of ``name``, or None when it is not installed."""

    try:
        from importlib import metadata

        return metadata.version(name)
    except Exception:
        return None


def truncate_at_stop(text: str, stop: Sequence[str]) -> str:
    """Cut ``text`` at the first stop sequence, for providers that echo it."""

    out = str(text or "")
    cut = len(out)
    for marker in stop or []:
        marker = str(marker or "")
        if not marker:
            continue
        position = out.find(marker)
        if position != -1 and position < cut:
            cut = position
    return out[:cut]


class RenderedStringBackend(BaseLLMBackend):
    """Base for backends whose engine takes rendered prompt strings.

    Subclasses implement the raw :meth:`generate_batch`; chat prompts are
    rendered with :meth:`render_prompt` first. The vLLM and mock backends and
    the legacy adapter share this path.
    """

    def generate_chat_batch(
        self,
        prompts: Sequence[ChatPrompt],
        *,
        max_tokens: int,
        temperature: float,
        stop: list[str],
    ) -> list[str]:
        return list(
            self.generate_batch(
                [self.render_prompt(prompt) for prompt in prompts],
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
            )
        )

    def generate_batch(
        self,
        prompts: list[str],
        *,
        max_tokens: int,
        temperature: float,
        stop: list[str],
    ) -> list[str]:
        raise NotImplementedError


class LegacyStringBackend(RenderedStringBackend):
    """Adapt an object that only implements the old ``generate_batch(list[str])`` API."""

    def __init__(self, inner: Any) -> None:
        if not callable(getattr(inner, "generate_batch", None)):
            raise TypeError("LegacyStringBackend needs an object with a generate_batch method")
        self.inner = inner
        self.chat_tokenizer = getattr(inner, "chat_tokenizer", None)
        self.runtime_metadata = dict(getattr(inner, "runtime_metadata", {}) or {})

    def render_prompt(self, prompt: ChatPrompt) -> str:
        return format_chat_prompt(
            tokenizer=self.chat_tokenizer,
            system_prompt=prompt.system,
            user_prompt=prompt.user,
        )

    def generate_batch(
        self,
        prompts: list[str],
        *,
        max_tokens: int,
        temperature: float,
        stop: list[str],
    ) -> list[str]:
        return list(self.inner.generate_batch(prompts, max_tokens=max_tokens, temperature=temperature, stop=stop))


def describe_llm_provenance(backend: Any, reproducibility: dict[str, Any]) -> dict[str, Any]:
    """Make a run's reproducibility block truthful for non-vLLM backends.

    ``build_reproducibility_metadata`` fills the sampling fields with the vLLM
    engine defaults. Hosted and mock backends never send those, so for them the
    sampling block keeps only the fields that were actually applied, and the
    backend's own runtime description (provider, model, base URL, SDK version)
    is attached as ``llm_backend``.
    """

    runtime = dict(getattr(backend, "runtime_metadata", {}) or {})
    out = dict(reproducibility)
    if runtime:
        out["llm_backend"] = runtime
    if runtime.get("backend") not in (None, "vllm"):
        sampling = dict(out.get("sampling") or {})
        applied: dict[str, Any] = {}
        if "temperature" in sampling:
            applied["temperature"] = sampling["temperature"]
        applied.update(runtime.get("sampling") or {})
        out["sampling"] = applied
    return out
