"""Builds chat-formatted prompts and prompt hashes for reproducibility."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def chatml_wrap(system_prompt: str, user_prompt: str) -> str:
    return f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n"


def format_chat_prompt(*, tokenizer: Any | None, system_prompt: str, user_prompt: str) -> str:
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        # Qwen3.5 (and Qwen3) think by default; force non-thinking (instruct) output.
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            # Tokenizer/template does not accept enable_thinking; retry without it.
            try:
                return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        except Exception:
            pass
    return chatml_wrap(system_prompt, user_prompt)


def prompt_hash(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\n---\n")
    return h.hexdigest()


def json_schema_string(schema: dict[str, Any]) -> str:
    return json.dumps(schema, ensure_ascii=False)
