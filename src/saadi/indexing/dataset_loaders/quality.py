"""Loads QuALITY documents and QA entries into the SAADI schema."""

from __future__ import annotations

import logging
from typing import Any

from ...core.chunking import coerce_to_text

logger = logging.getLogger(__name__)

_DATASET_ID = "tasksource/QuALITY"
_VALID_SPLITS = {"train", "validation"}
_SPLIT_ALIASES = {
    "default": "validation",
    "dev": "validation",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "test": "validation",
    "train": "train",
}


def _datasets():
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "datasets package is required for QuALITY loading. Install with `pip install .[datasets]`."
        ) from exc
    return load_dataset


def _normalize_split(split: str | None) -> str:
    raw = str(split or "validation").strip().lower()
    out = _SPLIT_ALIASES.get(raw, raw)
    if out not in _VALID_SPLITS:
        raise ValueError(
            "Unsupported QuALITY split. Use one of: train, validation "
            f"(got: {split!r})."
        )
    return out


def _load_quality_split(split_name: str):
    load_dataset = _datasets()
    try:
        return load_dataset(_DATASET_ID, split=split_name)
    except TypeError:
        return load_dataset(_DATASET_ID)[split_name]


def _row_doc_id(row: dict[str, Any], *, fallback_index: int) -> str:
    for key in ("article_id", "document_id", "doc_id"):
        value = row.get(key)
        if isinstance(value, (int, str)) and str(value).strip():
            return f"article:{str(value).strip()}"
    value = row.get("title")
    if isinstance(value, str) and value.strip():
        safe = "_".join(value.strip().split())
        return f"article:{safe}"
    return f"article:{fallback_index}"


def _options_list(row: dict[str, Any]) -> list[str]:
    raw = row.get("options")
    if isinstance(raw, list):
        return [txt for txt in (coerce_to_text(item) for item in raw) if txt]
    return []


def _gold_option_index(row: dict[str, Any], options: list[str]) -> int | None:
    for key in ("gold_label", "writer_label"):
        value = row.get(key)
        try:
            idx = int(value)
        except Exception:
            continue
        if 1 <= idx <= len(options):
            return idx - 1
        if 0 <= idx < len(options):
            return idx
    return None


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("title", "source", "author", "topic", "year"):
        value = coerce_to_text(row.get(key))
        if value:
            out[key] = value
    difficult = row.get("difficult")
    try:
        if difficult is not None:
            out["difficult"] = int(difficult)
    except Exception:
        pass
    return out


def load_quality_documents(
    *,
    split: str = "validation",
    config_name: str | None = "default",
) -> dict[str, str]:
    _ = config_name
    split_name = _normalize_split(split)
    logger.info("Loading QuALITY documents split=%s", split_name)
    rows = _load_quality_split(split_name)
    out: dict[str, str] = {}
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        doc_id = _row_doc_id(row, fallback_index=i)
        text = coerce_to_text(row.get("article")) or ""
        if text and doc_id not in out:
            out[doc_id] = text
    logger.info("Loaded QuALITY documents count=%d", len(out))
    return out


def load_quality_qa_entries(
    *,
    split: str = "validation",
    n: int | str = "all",
    selection_method: str = "random",
    config_name: str | None = "default",
) -> list[dict[str, Any]]:
    import random

    _ = config_name
    split_name = _normalize_split(split)
    logger.info("Loading QuALITY QA entries split=%s", split_name)
    rows = _load_quality_split(split_name)
    docs = load_quality_documents(split=split_name, config_name=config_name)
    qa_entries: list[dict[str, Any]] = []

    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        doc_id = _row_doc_id(row, fallback_index=i)
        if doc_id not in docs:
            continue
        question = coerce_to_text(row.get("question"))
        if not question:
            continue
        options = _options_list(row)
        gold_idx = _gold_option_index(row, options)
        if gold_idx is None or gold_idx >= len(options):
            continue
        answer_text = options[gold_idx]
        if not answer_text:
            continue
        entry: dict[str, Any] = {
            "id": str(row.get("question_unique_id") or f"{doc_id}:q{i}"),
            "question": question,
            "document_id": doc_id,
            "answers": [answer_text],
            "retrieval_spans": [],
            "choices": options,
            "gold_option_index": gold_idx,
            "gold_option_label": gold_idx + 1,
        }
        meta = _metadata(row)
        if meta:
            entry["metadata"] = meta
        qa_entries.append(entry)

    total = len(qa_entries)
    if isinstance(n, str):
        n = total if n.lower() == "all" else int(n)
    n = min(int(n), total)
    idxs = list(range(total))
    if n < total:
        idxs = random.sample(idxs, n) if selection_method == "random" else idxs[:n]
    out = [{"id": qa_entries[i].get("id", i), **{k: v for k, v in qa_entries[i].items() if k != "id"}} for i in idxs]
    logger.info("Loaded QuALITY QA entries selected=%d total=%d", len(out), total)
    return out
