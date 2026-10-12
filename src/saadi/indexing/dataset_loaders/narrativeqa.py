"""Loads NarrativeQA documents and QA entries into the SAADI schema."""

from __future__ import annotations

import logging
from typing import Any

from ...core.chunking import coerce_to_text

logger = logging.getLogger(__name__)


def _datasets():
    try:
        from datasets import load_dataset, get_dataset_config_names  # type: ignore
    except Exception as exc:
        raise RuntimeError("datasets package is required for NarrativeQA loading. Install with `pip install .[datasets]`.") from exc
    return load_dataset, get_dataset_config_names


def _resolve_config(requested: str | None, dataset_name: str = "deepmind/narrativeqa") -> str | None:
    _, get_dataset_config_names = _datasets()
    try:
        configs = get_dataset_config_names(dataset_name) or []
    except Exception:
        configs = []
    if not configs:
        return None
    if requested in configs:
        return requested
    return configs[0]


def _load_narrativeqa_dataset(*, cfg: str | None):
    load_dataset, _ = _datasets()
    try:
        return load_dataset("deepmind/narrativeqa", name=cfg) if cfg else load_dataset("deepmind/narrativeqa")
    except ValueError as exc:
        msg = str(exc)
        if "Feature type 'List' not found" in msg:
            raise RuntimeError(
                "Loading 'deepmind/narrativeqa' failed because your installed `datasets` package is too old "
                "for this dataset metadata. Upgrade datasets (e.g. "
                "`python -m pip install -U \"datasets>=4\"`) and retry."
            ) from exc
        raise


def load_narrativeqa_qa_entries(*, split: str = "test", n: int | str = "all", selection_method: str = "random", config_name: str | None = "default") -> list[dict[str, Any]]:
    import random
    cfg = _resolve_config(config_name)
    logger.info("Loading NarrativeQA QA entries split=%s config=%s", split, cfg)
    ds = _load_narrativeqa_dataset(cfg=cfg)
    qa_entries: list[dict[str, Any]] = []
    for row in ds[split]:
        q = row.get("question")
        if isinstance(q, dict):
            q = q.get("text")
        if not isinstance(q, str) or not q.strip():
            continue
        doc = row.get("document") or {}
        doc_id = doc.get("id") or row.get("document_id") or row.get("doc_id")
        if not doc_id:
            continue
        answer_text = coerce_to_text(row.get("answers"))
        if not answer_text:
            continue
        qa_entries.append(
            {
                "question": q.strip(),
                "document_id": str(doc_id),
                "answers": [answer_text.strip()],
                "retrieval_spans": [],
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
    logger.info("Loaded NarrativeQA QA entries selected=%d total=%d", len(out), total)
    return out


def load_narrativeqa_documents(*, split: str = "test", config_name: str | None = "default") -> dict[str, str]:
    cfg = _resolve_config(config_name)
    logger.info("Loading NarrativeQA documents split=%s config=%s", split, cfg)
    ds = _load_narrativeqa_dataset(cfg=cfg)
    out: dict[str, str] = {}
    for row in ds[split]:
        doc = row.get("document") or {}
        doc_id = doc.get("id") or row.get("document_id") or row.get("doc_id")
        if not doc_id:
            continue
        text = coerce_to_text(doc.get("text") if isinstance(doc, dict) else doc) or ""
        if text and str(doc_id) not in out:
            out[str(doc_id)] = text
    logger.info("Loaded NarrativeQA documents count=%d", len(out))
    return out
