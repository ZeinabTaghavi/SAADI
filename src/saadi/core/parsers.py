"""Extracts and parses JSON-like LLM outputs with fallback heuristics."""

from __future__ import annotations

import json
import os
import re
import tempfile
from typing import Any, Dict, List, Tuple

FAILED_JSONL_PATH = os.environ.get(
    "SAADI_FAILED_JSONL_PATH",
    os.path.join(tempfile.gettempdir(), "saadi_failed_json_parse.jsonl"),
)


def _log_json_parse_fail(stage: str, text: str) -> None:
    try:
        from ..io.jsonl import append_jsonl

        append_jsonl(FAILED_JSONL_PATH, [{"stage": stage, "raw_len": len(text), "raw_model_text": text}])
    except Exception:
        pass


def extract_balanced_json(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract_balanced_json_candidates(text: str) -> list[str]:
    out: list[str] = []
    start = None
    depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
            continue
        if ch == "}":
            if depth <= 0:
                continue
            depth -= 1
            if depth == 0 and start is not None:
                out.append(text[start : i + 1])
                start = None
    return out


def _strip_reasoning_markup(text: str) -> str:
    # Qwen/Gemma and other reasoning models may emit explicit thought sections.
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<\|begin_of_thought\|>.*?<\|end_of_thought\|>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def _extract_json_string_value(raw_text: str, key: str, next_keys: List[str] | None = None) -> str | None:
    if next_keys is None:
        next_keys = ['"active_entities"', '"active_objects"', '"updated_current_chunk"', '"linking_sentence"', '"insertions"']
    kpos = raw_text.find(key)
    if kpos == -1:
        return None
    colon = raw_text.find(":", kpos)
    if colon == -1:
        return None
    start_quote = raw_text.find('"', colon)
    if start_quote == -1:
        return None
    i = start_quote + 1
    while i < len(raw_text):
        if raw_text[i] == '"':
            bs = 0
            j = i - 1
            while j >= 0 and raw_text[j] == "\\":
                bs += 1
                j -= 1
            if bs % 2 == 0:
                k = i + 1
                while k < len(raw_text) and raw_text[k].isspace():
                    k += 1
                if k < len(raw_text) and raw_text[k] == ",":
                    k2 = k + 1
                    while k2 < len(raw_text) and raw_text[k2].isspace():
                        k2 += 1
                    if any(raw_text.startswith(nk, k2) for nk in next_keys):
                        return raw_text[start_quote + 1 : i]
                if k < len(raw_text) and raw_text[k] == "}":
                    return raw_text[start_quote + 1 : i]
                if raw_text.startswith("[END_JSON]", k):
                    return raw_text[start_quote + 1 : i]
        i += 1
    return None


def _extract_array_for_key(raw_text: str, key: str) -> list[Any]:
    pos = raw_text.find(key)
    if pos == -1:
        return []
    start = raw_text.find("[", pos)
    if start == -1:
        return []
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(raw_text)):
        ch = raw_text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                try:
                    val = json.loads(raw_text[start : i + 1])
                except Exception:
                    return []
                return val if isinstance(val, list) else []
    return []


def fallback_extract_fields(raw_text: str) -> Dict[str, Any] | None:
    try:
        has_active = any(
            key in raw_text
            for key in ('"active_entities"', '"active_objects"', '"objects_active"')
        )
        has_updated = ('"updated_current_chunk"' in raw_text) or ('"chunk_resolved"' in raw_text)
        has_linking = '"linking_sentence"' in raw_text
        has_insertions = '"insertions"' in raw_text
        if not (has_active or has_updated or has_linking or has_insertions):
            return None
        updated = _extract_json_string_value(raw_text, '"updated_current_chunk"') or _extract_json_string_value(
            raw_text, '"chunk_resolved"'
        )
        linking_sentence = _extract_json_string_value(
            raw_text,
            '"linking_sentence"',
            next_keys=['"active_entities"', '"active_objects"', '"insertions"', '"updated_current_chunk"'],
        ) or ""
        active_entities = (
            _extract_array_for_key(raw_text, '"active_entities"')
            or _extract_array_for_key(raw_text, '"active_objects"')
            or _extract_array_for_key(raw_text, '"objects_active"')
        )
        insertions = _extract_array_for_key(raw_text, '"insertions"')
        out = {
            "active_entities": active_entities,
        }
        if has_updated:
            out["updated_current_chunk"] = updated or ""
        if has_linking:
            out["linking_sentence"] = linking_sentence
        if has_insertions:
            out["insertions"] = insertions
        return out
    except Exception:
        return None


def _is_canonicalization_payload(parsed: dict[str, Any]) -> bool:
    return any(
        key in parsed
        for key in (
            "active_entities",
            "active_objects",
            "objects_active",
            "updated_current_chunk",
            "chunk_resolved",
            "linking_sentence",
            "insertions",
        )
    )


def safe_json_from_text(text: str) -> Tuple[dict[str, Any] | None, str]:
    normalized = _strip_reasoning_markup(str(text or ""))
    json_region = normalized.split("[END_JSON]", 1)[0] if "[END_JSON]" in normalized else normalized

    # Prefer the last balanced object near the end marker; models may emit preambles.
    candidates = extract_balanced_json_candidates(json_region)
    for candidate in reversed(candidates):
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if not isinstance(parsed, dict):
            continue
        if _is_canonicalization_payload(parsed):
            return parsed, "ok"
    fallback_candidate: dict[str, Any] | None = None
    for candidate in reversed(candidates):
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            fallback_candidate = parsed
            break

    # Backward-compatible single-object fallback.
    candidate = extract_balanced_json(json_region)
    if candidate is not None:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and _is_canonicalization_payload(parsed):
                return parsed, "ok"
        except Exception:
            pass

    fb = fallback_extract_fields(json_region)
    if fb is not None:
        return fb, "fallback"
    if fallback_candidate is not None:
        return fallback_candidate, "fallback"
    _log_json_parse_fail("safe_json_from_text", normalized)
    return None, "failed"


def extract_tagged_section(prompt: str, tag: str) -> str:
    pattern = rf"\[{re.escape(tag)}\]\n(.*?)(?=\n\[[A-Z0-9_]+\]\n|\Z)"
    m = re.search(pattern, prompt, flags=re.DOTALL)
    return m.group(1).strip() if m else ""
