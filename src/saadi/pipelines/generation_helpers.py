"""Prompt and context helpers shared by Saadi answer generation."""
from __future__ import annotations
from typing import Any
from ..core.prompts import format_chat_prompt

DEFAULT_SYSTEM_PROMPT = (
    "Answer the question using only the provided context. "
    "If the context does not contain the answer, say that you do not know. "
    "Keep the answer concise."
)

DEFAULT_USER_TEMPLATE = """Context:
{context}

Question:
{question}{choices_block}

Answer:"""

def _token_ids(text: str, tokenizer: Any | None) -> list[int] | None:
    if tokenizer is None or not hasattr(tokenizer, "encode"):
        return None
    try:
        return list(tokenizer.encode(str(text or ""), add_special_tokens=False))
    except TypeError:
        try:
            return list(tokenizer.encode(str(text or "")))
        except Exception:
            return None
    except Exception:
        return None

def _decode_token_ids(token_ids: list[int], tokenizer: Any | None) -> str:
    if tokenizer is None or not hasattr(tokenizer, "decode"):
        return " ".join(str(token_id) for token_id in token_ids)
    try:
        return str(tokenizer.decode(token_ids, skip_special_tokens=False))
    except TypeError:
        return str(tokenizer.decode(token_ids))

def _count_tokens(text: str, tokenizer: Any | None) -> int:
    ids = _token_ids(text, tokenizer)
    if ids is not None:
        return len(ids)
    return len(str(text or "").split())

def _trim_context(
    context: str,
    *,
    tokenizer: Any | None,
    max_context_tokens: int | None,
    overflow_policy: str,
) -> tuple[str, dict[str, Any]]:
    ids = _token_ids(context, tokenizer)
    if ids is None:
        token_count = len(str(context or "").split())
        return str(context or ""), {
            "context_tokens": token_count,
            "context_truncated": False,
            "original_context_tokens": token_count,
            "overflow_policy": overflow_policy,
        }
    token_count = len(ids)
    if max_context_tokens is None or token_count <= max_context_tokens:
        return str(context or ""), {
            "context_tokens": token_count,
            "context_truncated": False,
            "original_context_tokens": token_count,
            "overflow_policy": overflow_policy,
        }

    policy = str(overflow_policy or "fail").strip().lower()
    if policy == "skip":
        return "", {
            "context_tokens": token_count,
            "context_truncated": False,
            "original_context_tokens": token_count,
            "overflow_policy": policy,
            "skip_reason": f"context token count {token_count} exceeds max_context_tokens={max_context_tokens}",
        }
    if policy == "fail":
        raise ValueError(
            f"Context token count {token_count} exceeds max_context_tokens={max_context_tokens}. "
            "Set generation.long_context.overflow_policy to skip, truncate_head, truncate_tail, or truncate_middle."
        )
    if policy == "truncate_head":
        trimmed_ids = ids[-max_context_tokens:]
    elif policy == "truncate_tail":
        trimmed_ids = ids[:max_context_tokens]
    elif policy == "truncate_middle":
        left = max_context_tokens // 2
        right = max_context_tokens - left
        trimmed_ids = ids[:left] + ids[-right:]
    else:
        raise ValueError(
            "Invalid generation.long_context.overflow_policy. Expected one of: "
            "fail, skip, truncate_head, truncate_tail, truncate_middle."
        )
    trimmed = _decode_token_ids(trimmed_ids, tokenizer)
    return trimmed, {
        "context_tokens": len(trimmed_ids),
        "context_truncated": True,
        "original_context_tokens": token_count,
        "overflow_policy": policy,
    }

def _choices_block(entry: dict[str, Any]) -> str:
    choices = entry.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    lines = ["", "", "Answer choices:"]
    for index, choice in enumerate(choices, start=1):
        text = str(choice or "").strip()
        if text:
            lines.append(f"{index}. {text}")
    if len(lines) <= 3:
        return ""
    lines.append("")
    lines.append("If answer choices are provided, select the best answer and include the answer text.")
    return "\n".join(lines)

def build_long_context_prompt(
    *,
    context: str,
    question: str,
    qa_entry: dict[str, Any],
    tokenizer: Any | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    user_template: str = DEFAULT_USER_TEMPLATE,
) -> tuple[str, str]:
    user_prompt = user_template.format(
        context=str(context or ""),
        question=str(question or "").strip(),
        choices_block=_choices_block(qa_entry),
    )
    prompt = format_chat_prompt(
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    return prompt, user_prompt
