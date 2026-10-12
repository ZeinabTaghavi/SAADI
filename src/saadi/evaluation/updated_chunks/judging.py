"""Shared judge-model helpers for updated-chunk evaluations."""

from __future__ import annotations

import json
import math
import re
from typing import Any, Iterable

from ...core.parsers import extract_balanced_json, extract_balanced_json_candidates
from ...core.prompts import format_chat_prompt
from ...llm.base import LLMBackend

DEFAULT_JUDGE_STOP = ["[END_JSON]", "<|im_end|>", "<|endoftext|>"]


def _strip_reasoning_markup(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(
        r"<\|begin_of_thought\|>.*?<\|end_of_thought\|>",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return cleaned.strip()


def extract_json_object_from_text(text: str) -> tuple[dict[str, Any] | None, str]:
    cleaned = _strip_reasoning_markup(text)
    candidates = extract_balanced_json_candidates(cleaned)
    for candidate in reversed(candidates):
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed, "ok"

    candidate = extract_balanced_json(cleaned)
    if candidate is not None:
        try:
            parsed = json.loads(candidate)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            return parsed, "fallback"

    return None, "failed"


def normalize_boolish(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    text = str(value or "").strip().lower()
    if text in {"true", "yes", "y", "supported", "faithful", "1"}:
        return True
    if text in {"false", "no", "n", "unsupported", "unfaithful", "0"}:
        return False
    return None


def normalize_score_1_to_5(value: Any) -> int | None:
    if value is None:
        return None
    try:
        score = int(float(value))
    except Exception:
        return None
    if not (1 <= score <= 5):
        return None
    return score


def normalize_problem_type(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        out = ", ".join(str(item).strip() for item in value if str(item).strip())
        return out or None
    text = str(value).strip()
    return text or None


def build_chat_prompt(*, backend: LLMBackend, system_prompt: str, user_prompt: str) -> str:
    tokenizer = getattr(backend, "chat_tokenizer", None)
    return format_chat_prompt(
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


def iter_batches(items: list[Any], batch_size: int | None) -> Iterable[list[Any]]:
    if batch_size is None or int(batch_size) <= 0:
        yield list(items)
        return
    step = int(batch_size)
    for idx in range(0, len(items), step):
        yield items[idx : idx + step]


def generate_outputs(
    backend: LLMBackend,
    prompts: list[str],
    *,
    batch_size: int | None,
    max_tokens: int,
    temperature: float,
    stop: list[str] | None = None,
) -> list[str]:
    outputs: list[str] = []
    stop_tokens = list(stop or DEFAULT_JUDGE_STOP)
    for batch in iter_batches(prompts, batch_size):
        if not batch:
            continue
        outputs.extend(
            backend.generate_batch(
                batch,
                max_tokens=int(max_tokens),
                temperature=float(temperature),
                stop=stop_tokens,
            )
        )
    return outputs

