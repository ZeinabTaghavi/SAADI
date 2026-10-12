"""Wraps a vLLM model behind the SAADI LLM backend interface."""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import (
    DEFAULT_GENERATION_SEED,
    DEFAULT_GENERATION_TOP_K,
    DEFAULT_GENERATION_TOP_P,
    DEFAULT_REPETITION_PENALTY,
    LLMBackend,
)


@dataclass
class VLLMBackend(LLMBackend):
    model: object
    sampling_defaults: dict[str, object] = field(default_factory=dict)
    chat_tokenizer: object | None = None
    runtime_metadata: dict[str, object] = field(default_factory=dict)

    def generate_batch(self, prompts: list[str], *, max_tokens: int, temperature: float, stop: list[str]) -> list[str]:
        try:
            from vllm import SamplingParams  # type: ignore
        except Exception as exc:
            raise RuntimeError("vLLM is not installed. Install with `pip install .[vllm]`.") from exc
        # These defaults make every indexing call deterministic even when an
        # older experiment config omits its sampling section. Explicit config
        # values remain supported and are recorded in the run manifest.
        params_kwargs: dict[str, object] = {
            "seed": DEFAULT_GENERATION_SEED,
            "top_p": DEFAULT_GENERATION_TOP_P,
            "top_k": DEFAULT_GENERATION_TOP_K,
            "repetition_penalty": DEFAULT_REPETITION_PENALTY,
        }
        params_kwargs.update(self.sampling_defaults or {})
        params_kwargs.update(
            {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": stop,
            }
        )
        params = SamplingParams(**params_kwargs)
        outs = self.model.generate(prompts, params)
        out_texts: list[str] = []
        for out in outs:
            outputs = getattr(out, "outputs", None) or []
            if not outputs:
                out_texts.append("")
                continue
            out_texts.append(str(getattr(outputs[0], "text", "") or ""))
        return out_texts
