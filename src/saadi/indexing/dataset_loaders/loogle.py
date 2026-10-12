"""Loads LooGLE documents and QA entries into the SAADI schema."""

from __future__ import annotations

import ast
import json
import logging
import os
from importlib.metadata import version as pkg_version
from typing import Any, Iterable

from ...core.chunking import coerce_to_text

logger = logging.getLogger(__name__)

_DATASET_IDS = ("bigai-nlco/LooGLE", "bigainlco/LooGLE")
_LEGACY_CONFIG_ALIASES = {
    "longdep_summarization": "summarization",
}


def _datasets():
    try:
        from datasets import load_dataset, get_dataset_config_names  # type: ignore
    except Exception as exc:
        raise RuntimeError("datasets package is required for LooGLE loading. Install with `pip install .[datasets]`.") from exc
    return load_dataset, get_dataset_config_names


def _datasets_version_major() -> int | None:
    try:
        raw = str(pkg_version("datasets")).strip()
        if not raw:
            return None
        return int(raw.split(".", 1)[0])
    except Exception:
        return None


def _normalize_requested_config(requested: str | None) -> str:
    raw = str(requested or "").strip()
    if not raw:
        return "shortdep_qa"
    return _LEGACY_CONFIG_ALIASES.get(raw, raw)


def _config_candidates(requested: str | None) -> list[str]:
    normalized = _normalize_requested_config(requested)
    out = [normalized]
    for old_name, new_name in _LEGACY_CONFIG_ALIASES.items():
        if normalized == new_name:
            out.append(old_name)
        elif normalized == old_name:
            out.append(new_name)
    return out


def _get_config_names(dataset_name: str) -> list[str]:
    _, get_dataset_config_names = _datasets()
    major = _datasets_version_major()
    kwargs: dict[str, Any] = {}
    if major is not None and major >= 4:
        kwargs["revision"] = "refs/convert/parquet"
    try:
        try:
            return list(get_dataset_config_names(dataset_name, **kwargs) or [])
        except TypeError:
            return list(get_dataset_config_names(dataset_name) or [])
    except Exception:
        return []


def _resolve_dataset_and_config(requested_config_name: str | None) -> tuple[str, str | None]:
    desired_candidates = _config_candidates(requested_config_name)
    for dataset_name in _DATASET_IDS:
        configs = _get_config_names(dataset_name)
        if not configs:
            continue
        for cfg in desired_candidates:
            if cfg in configs:
                return dataset_name, cfg
        # Prefer shortdep_qa when available because it is the standard QA benchmark config.
        if "shortdep_qa" in configs:
            return dataset_name, "shortdep_qa"
        return dataset_name, configs[0]
    return _DATASET_IDS[0], desired_candidates[0] if desired_candidates else None


def _load_loogle_dataset(*, cfg: str | None):
    load_dataset, _ = _datasets()
    major = _datasets_version_major()
    preferred_name, preferred_cfg = _resolve_dataset_and_config(cfg)
    dataset_order = [preferred_name] + [name for name in _DATASET_IDS if name != preferred_name]
    cfg_candidates: list[str | None] = [preferred_cfg]
    for alt in _config_candidates(cfg):
        if alt not in cfg_candidates:
            cfg_candidates.append(alt)
    kwargs_candidates: list[dict[str, Any]] = []
    if major is not None and major >= 4:
        kwargs_candidates.append({"revision": "refs/convert/parquet"})
    kwargs_candidates.append({})
    if major is None or major < 4:
        kwargs_candidates.append({"trust_remote_code": True})

    def _attempt_load(*, force_offline: bool) -> Any:
        prev_hub_offline = os.environ.get("HF_HUB_OFFLINE")
        prev_datasets_offline = os.environ.get("HF_DATASETS_OFFLINE")
        offline_download_config = None
        if force_offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            try:
                from datasets import DownloadConfig  # type: ignore

                offline_download_config = DownloadConfig(local_files_only=True)
            except Exception:
                offline_download_config = None
        last_exc: Exception | None = None
        try:
            for dataset_name in dataset_order:
                for cfg_name in cfg_candidates:
                    for kwargs in kwargs_candidates:
                        call_kwargs = dict(kwargs)
                        if force_offline and offline_download_config is not None and "download_config" not in call_kwargs:
                            call_kwargs["download_config"] = offline_download_config
                        try:
                            return (
                                load_dataset(dataset_name, name=cfg_name, **call_kwargs)
                                if cfg_name
                                else load_dataset(dataset_name, **call_kwargs)
                            )
                        except TypeError:
                            # Backward compatibility for older `datasets` signatures.
                            try:
                                return (
                                    load_dataset(dataset_name, name=cfg_name)
                                    if cfg_name
                                    else load_dataset(dataset_name)
                                )
                            except Exception as exc:  # pragma: no cover - exercised via normal exception path
                                last_exc = exc
                        except Exception as exc:
                            last_exc = exc
                            continue
            return last_exc
        finally:
            if force_offline:
                if prev_hub_offline is None:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                else:
                    os.environ["HF_HUB_OFFLINE"] = prev_hub_offline
                if prev_datasets_offline is None:
                    os.environ.pop("HF_DATASETS_OFFLINE", None)
                else:
                    os.environ["HF_DATASETS_OFFLINE"] = prev_datasets_offline

    first_try = _attempt_load(force_offline=False)
    if not isinstance(first_try, Exception):
        return first_try
    # Offline retry helps when cache exists but network is unavailable/flaky.
    second_try = _attempt_load(force_offline=True)
    if not isinstance(second_try, Exception):
        return second_try

    requested = cfg if cfg is not None else "default"
    raise RuntimeError(
        f"Failed to load LooGLE dataset (requested config={requested!r}) from any known dataset id: {_DATASET_IDS}."
    ) from second_try


def _extract_document_text(row: dict[str, Any]) -> str | None:
    for key in ("context", "input", "document", "text", "article", "passage"):
        text = coerce_to_text(row.get(key))
        if text:
            return text
    return None


def _extract_doc_id(row: dict[str, Any], *, fallback_index: int) -> str:
    for key in ("doc_id", "document_id", "docid", "title"):
        value = row.get(key)
        if isinstance(value, (str, int)) and str(value).strip():
            return str(value).strip()
    return f"doc_{fallback_index}"


def _to_text_list(value: Any) -> list[str]:
    out: list[str] = []
    if value is None:
        return out
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    if isinstance(value, (list, tuple)):
        for item in value:
            out.extend(_to_text_list(item))
        return out
    text = coerce_to_text(value)
    if text:
        out.append(text)
    return out


def _parse_qa_pairs(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, str):
        txt = raw.strip()
        if not txt:
            return []
        try:
            parsed = json.loads(txt)
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
        except Exception:
            pass
        try:
            parsed = ast.literal_eval(txt)
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
        except Exception:
            pass
    return []


def _iter_rows_for_split(ds: Any, *, split: str) -> Iterable[dict[str, Any]]:
    rows = ds[split]
    for row in rows:
        if isinstance(row, dict):
            yield row


def load_loogle_documents(*, split: str = "test", config_name: str | None = "shortdep_qa") -> dict[str, str]:
    logger.info("Loading LooGLE documents split=%s config=%s", split, config_name)
    ds = _load_loogle_dataset(cfg=config_name)
    out: dict[str, str] = {}
    for i, row in enumerate(_iter_rows_for_split(ds, split=split)):
        doc_id = _extract_doc_id(row, fallback_index=i)
        text = _extract_document_text(row) or ""
        if text and doc_id not in out:
            out[doc_id] = text
    logger.info("Loaded LooGLE documents count=%d", len(out))
    return out


def load_loogle_qa_entries(*, split: str = "test", n: int | str = "all", selection_method: str = "random", config_name: str | None = "shortdep_qa") -> list[dict[str, Any]]:
    import random

    logger.info("Loading LooGLE QA entries split=%s config=%s", split, config_name)
    ds = _load_loogle_dataset(cfg=config_name)
    qa_entries: list[dict[str, Any]] = []
    for i, row in enumerate(_iter_rows_for_split(ds, split=split)):
        doc_id = _extract_doc_id(row, fallback_index=i)
        qa_pairs = _parse_qa_pairs(row.get("qa_pairs"))
        if qa_pairs:
            for pair in qa_pairs:
                question = coerce_to_text(pair.get("Q") or pair.get("question"))
                if not question:
                    continue
                answer_texts = _to_text_list(pair.get("A") or pair.get("answer"))
                spans = _to_text_list(pair.get("S") or pair.get("evidence"))
                if answer_texts or spans:
                    qa_entries.append(
                        {
                            "question": question,
                            "document_id": doc_id,
                            "answers": answer_texts,
                            "retrieval_spans": spans,
                        }
                    )
            continue

        question = coerce_to_text(row.get("question") or row.get("Q") or row.get("query"))
        if not question:
            continue
        answer_texts = _to_text_list(row.get("answer") or row.get("A") or row.get("answers"))
        spans = _to_text_list(row.get("evidence") or row.get("S") or row.get("span"))
        if answer_texts or spans:
            qa_entries.append(
                {
                    "question": question,
                    "document_id": doc_id,
                    "answers": answer_texts,
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
    logger.info("Loaded LooGLE QA entries selected=%d total=%d", len(out), total)
    return out
