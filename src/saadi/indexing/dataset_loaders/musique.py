"""Loads MuSiQue documents and QA entries into the SAADI schema."""

from __future__ import annotations

import logging
from typing import Any, Iterable

from ...core.chunking import coerce_to_text

logger = logging.getLogger(__name__)

_DATASET_ID = "dgslibisey/MuSiQue"


def _datasets():
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "datasets package is required for MuSiQue loading. Install with `pip install .[datasets]`."
        ) from exc
    return load_dataset


def _load_musique_dataset(*, config_name: str | None):
    load_dataset = _datasets()
    kwargs: dict[str, Any] = {"revision": "refs/convert/parquet"}
    if config_name and str(config_name).strip().lower() != "default":
        kwargs["name"] = config_name
    try:
        return load_dataset(_DATASET_ID, **kwargs)
    except TypeError:
        kwargs.pop("revision", None)
        return load_dataset(_DATASET_ID, **kwargs)


def _iter_rows_for_split(ds: Any, *, split: str) -> Iterable[dict[str, Any]]:
    rows = ds[split]
    for row in rows:
        if isinstance(row, dict):
            yield row


def _extract_doc_id(row: dict[str, Any], *, fallback_index: int) -> str:
    value = row.get("id")
    if isinstance(value, (str, int)) and str(value).strip():
        return str(value).strip()
    return f"doc_{fallback_index}"


def _paragraphs(row: dict[str, Any]) -> list[dict[str, Any]]:
    raw = row.get("paragraphs")
    if not isinstance(raw, list):
        return []
    return [paragraph for paragraph in raw if isinstance(paragraph, dict)]


def _paragraph_text(paragraph: dict[str, Any]) -> str:
    return coerce_to_text(paragraph.get("paragraph_text")) or ""


def load_musique_documents(*, split: str = "validation", config_name: str | None = "default") -> dict[str, str]:
    logger.info("Loading MuSiQue documents split=%s config=%s", split, config_name)
    ds = _load_musique_dataset(config_name=config_name)
    out: dict[str, str] = {}
    for i, row in enumerate(_iter_rows_for_split(ds, split=split)):
        doc_id = _extract_doc_id(row, fallback_index=i)
        text = "\n\n".join(text for text in (_paragraph_text(p) for p in _paragraphs(row)) if text)
        if text and doc_id not in out:
            out[doc_id] = text
    logger.info("Loaded MuSiQue documents count=%d", len(out))
    return out


def load_musique_qa_entries(
    *,
    split: str = "validation",
    n: int | str = "all",
    selection_method: str = "random",
    config_name: str | None = "default",
) -> list[dict[str, Any]]:
    import random

    logger.info("Loading MuSiQue QA entries split=%s config=%s", split, config_name)
    ds = _load_musique_dataset(config_name=config_name)
    qa_entries: list[dict[str, Any]] = []
    for i, row in enumerate(_iter_rows_for_split(ds, split=split)):
        question = coerce_to_text(row.get("question"))
        answer = coerce_to_text(row.get("answer"))
        if not question or not answer:
            continue
        doc_id = _extract_doc_id(row, fallback_index=i)
        spans = [
            text
            for paragraph in _paragraphs(row)
            if paragraph.get("is_supporting") is True
            for text in [_paragraph_text(paragraph)]
            if text
        ]
        qa_entries.append(
            {
                "question": question,
                "document_id": doc_id,
                "answers": [answer],
                "retrieval_spans": spans,
            }
        )

    total = len(qa_entries)
    if isinstance(n, str):
        n = total if n.lower() == "all" else int(n)
    n = min(int(n), total)
    idxs = list(range(total))
    if n < total:
        idxs = random.sample(idxs, n) if selection_method == "random" else idxs[:n]
    out = [{"id": i, **qa_entries[i]} for i in idxs]
    logger.info("Loaded MuSiQue QA entries selected=%d total=%d", len(out), total)
    return out
