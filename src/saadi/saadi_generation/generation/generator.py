"""Answer-generation boundary for SAADI context experiments."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ...core.prompts import prompt_hash
from ...llm.base import ChatPrompt, GenerateParams, render_prompt
from ...llm.factory import build_backend_from_config, resolve_backend
from .prompts import (
    DEFAULT_SAADI_GENERATION_SYSTEM_PROMPT,
    DEFAULT_SAADI_GENERATION_USER_TEMPLATE,
    format_choices_block,
    format_generated_bridge_prompt,
)


@dataclass(slots=True)
class GenerationConfig:
    system_prompt: str = DEFAULT_SAADI_GENERATION_SYSTEM_PROMPT
    user_template: str = DEFAULT_SAADI_GENERATION_USER_TEMPLATE
    prompt_version: str = "saadi_generation_v1"
    batch_size: int = 1
    max_context_tokens: int | None = None
    overflow_policy: str = "fail"
    max_output_tokens: int = 1024
    temperature: float = 0.0
    stop: list[str] | None = None


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    example_id: str
    question: str
    qa_entry: dict[str, Any]
    context: str
    metadata: dict[str, Any]


class SaadiAnswerGenerator:

    def __init__(self, config: GenerationConfig, *, backend: Any | None = None) -> None:
        self.config = config
        self.backend = resolve_backend(backend) if backend is not None else build_backend_from_config({"backend": "vllm"})
        self.tokenizer = getattr(self.backend, "chat_tokenizer", None)

    @classmethod
    def from_model_config(
        cls,
        model_cfg: dict[str, Any] | None,
        *,
        generation_cfg: dict[str, Any] | None = None,
        backend: Any | None = None,
    ) -> "SaadiAnswerGenerator":
        cfg = model_cfg or {"backend": "vllm"}
        gen_cfg = generation_cfg or {}
        params = _build_generate_params(cfg)
        if gen_cfg.get("max_output_tokens") is not None:
            params.max_tokens = int(gen_cfg["max_output_tokens"])
        if gen_cfg.get("temperature") is not None:
            params.temperature = float(gen_cfg["temperature"])
        if "stop" in gen_cfg:
            stop_raw = gen_cfg.get("stop")
            if stop_raw is None:
                params.stop = []
            elif isinstance(stop_raw, list):
                params.stop = [str(item) for item in stop_raw if str(item).strip()]
            else:
                params.stop = [str(stop_raw)]
        generation_config = GenerationConfig(
            system_prompt=str(gen_cfg.get("system_prompt", DEFAULT_SAADI_GENERATION_SYSTEM_PROMPT) or DEFAULT_SAADI_GENERATION_SYSTEM_PROMPT),
            user_template=str(gen_cfg.get("user_template", DEFAULT_SAADI_GENERATION_USER_TEMPLATE) or DEFAULT_SAADI_GENERATION_USER_TEMPLATE),
            prompt_version=str(gen_cfg.get("prompt_version", "sadi_folded_context_v1") or "sadi_folded_context_v1"),
            batch_size=int(gen_cfg.get("batch_size", 1) or 1),
            max_context_tokens=(None if gen_cfg.get("max_context_tokens") in (None, "", "null") else int(gen_cfg.get("max_context_tokens"))),
            overflow_policy=str(gen_cfg.get("overflow_policy", "fail") or "fail"),
            max_output_tokens=int(params.max_tokens),
            temperature=float(params.temperature),
            stop=list(params.stop),
        )
        if generation_config.batch_size <= 0:
            raise ValueError("SAADI generation batch_size must be > 0")
        return cls(generation_config, backend=backend if backend is not None else build_backend_from_config(cfg))

    def count_tokens(self, text: str) -> int:
        if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
            try:
                return len(self.tokenizer.encode(str(text or ""), add_special_tokens=False))
            except TypeError:
                return len(self.tokenizer.encode(str(text or "")))
            except Exception:
                pass
        return len(str(text or "").split())

    def build_prompt(self, request: GenerationRequest) -> tuple[ChatPrompt, str, dict[str, Any]]:
        context = str(request.context or "")
        context_tokens = self.count_tokens(context)
        if self.config.max_context_tokens is not None and context_tokens > self.config.max_context_tokens:
            policy = str(self.config.overflow_policy or "fail")
            raise ValueError(
                f"Context for example_id={request.example_id} has {context_tokens} tokens, "
                f"exceeding max_context_tokens={self.config.max_context_tokens}; "
                f"overflow_policy={policy!r} is not implemented in SAADI generation yet."
            )
        user_prompt = self.config.user_template.format(
            context=context,
            question=str(request.question or "").strip(),
            choices_block=format_choices_block(request.qa_entry),
        )
        prompt = ChatPrompt(system=self.config.system_prompt, user=user_prompt)
        return prompt, user_prompt, {
            "context_token_length": context_tokens,
            "context_tokens": context_tokens,
            "prompt_tokens": self.count_tokens(render_prompt(self.backend, prompt)),
            "prompt_hash": prompt_hash(self.config.system_prompt, user_prompt),
        }

    def generate_batch(self, requests: list[GenerationRequest]) -> list[dict[str, Any]]:
        prompts: list[ChatPrompt] = []
        prompt_meta: list[dict[str, Any]] = []
        for request in requests:
            prompt, _user_prompt, meta = self.build_prompt(request)
            prompts.append(prompt)
            prompt_meta.append(meta)
        started = time.perf_counter()
        outputs = self.backend.generate_chat_batch(
            prompts,
            max_tokens=int(self.config.max_output_tokens),
            temperature=float(self.config.temperature),
            stop=list(self.config.stop or []),
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if len(outputs) != len(requests):
            raise RuntimeError(f"Backend returned {len(outputs)} outputs for {len(requests)} prompts.")
        per_item_ms = elapsed_ms / max(1, len(outputs))
        rows: list[dict[str, Any]] = []
        for request, meta, raw_output in zip(requests, prompt_meta, outputs):
            raw_text = str(raw_output or "").strip()
            rows.append(
                {
                    **request.metadata,
                    **meta,
                    "example_id": request.example_id,
                    "query_id": request.example_id,
                    "question": request.question,
                    "generated_answer": raw_text,
                    "raw_output": raw_text,
                    "status": "ok",
                    "prompt_version": self.config.prompt_version,
                    "max_output_tokens": int(self.config.max_output_tokens),
                    "temperature": float(self.config.temperature),
                    "stop": list(self.config.stop or []),
                    "generation_latency_ms": per_item_ms,
                    "answer_tokens": self.count_tokens(raw_text),
                }
            )
        return rows

    def generate_bridge(
        self,
        *,
        query: str,
        chunk_a: str,
        summaries: str,
        chunk_b: str,
    ) -> dict[str, Any]:
        bridge_prompt = format_generated_bridge_prompt(
            query=query,
            chunk_a=chunk_a,
            summaries=summaries,
            chunk_b=chunk_b,
        )
        started = time.perf_counter()
        outputs = self.backend.generate_batch(
            [bridge_prompt],
            max_tokens=30,
            temperature=float(self.config.temperature),
            stop=list(self.config.stop or []),
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        generated_bridge = str(outputs[0] if outputs else "").strip()
        return {
            "bridge_prompt": bridge_prompt,
            "generated_bridge": generated_bridge,
            "generated_bridge_token_length": self.count_tokens(generated_bridge),
            "bridge_generation_latency_ms": elapsed_ms,
        }


def _build_generate_params(model_cfg: dict[str, Any]) -> GenerateParams:
    raw = model_cfg.get("generate", {}) or {}
    if not isinstance(raw, dict):
        raw = {}
    stop_raw = raw.get("stop")
    if "stop" not in raw:
        stop = ["<|im_end|>", "<|endoftext|>"]
    elif stop_raw is None:
        stop = []
    elif isinstance(stop_raw, list):
        stop = [str(item) for item in stop_raw if str(item).strip()]
    else:
        text = str(stop_raw).strip()
        stop = [text] if text else []
    return GenerateParams(
        max_tokens=int(raw.get("max_tokens", 1024)),
        temperature=float(raw.get("temperature", 0.0)),
        stop=stop,
    )
