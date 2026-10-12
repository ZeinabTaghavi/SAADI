"""Loads QASPER documents and QA entries into the SAADI schema."""

from __future__ import annotations

import logging
from importlib.metadata import version as pkg_version
from typing import Any

from ...core.chunking import coerce_to_text

logger = logging.getLogger(__name__)


def _datasets():
    try:
        from datasets import load_dataset, get_dataset_config_names  # type: ignore
    except Exception as exc:
        raise RuntimeError("datasets package is required for QASPER loading. Install with `pip install .[datasets]`.") from exc
    return load_dataset, get_dataset_config_names


def _datasets_version_major() -> int | None:
    try:
        raw = str(pkg_version("datasets")).strip()
        if not raw:
            return None
        return int(raw.split(".", 1)[0])
    except Exception:
        return None


def _resolve_config(requested: str | None, dataset_name: str = "allenai/qasper") -> str | None:
    _, get_dataset_config_names = _datasets()
    major = _datasets_version_major()
    kwargs = {}
    # Legacy script loading path for datasets<4.
    if major is None or major < 4:
        kwargs["trust_remote_code"] = True
    # datasets>=4 removed script execution; load config list from parquet conversion revision.
    if major is not None and major >= 4:
        kwargs["revision"] = "refs/convert/parquet"
    try:
        try:
            configs = get_dataset_config_names(dataset_name, **kwargs) or []
        except TypeError:
            configs = get_dataset_config_names(dataset_name) or []
    except Exception:
        configs = []
    if not configs:
        return None
    if requested in configs:
        return requested
    return configs[0]


def _load_qasper_dataset(*, cfg: str | None):
    load_dataset, _ = _datasets()
    major = _datasets_version_major()
    if major is not None and major >= 4:
        # datasets>=4 path: load parquet-converted branch to avoid script execution.
        modern_kwargs = {"revision": "refs/convert/parquet"}
        try:
            return (
                load_dataset("allenai/qasper", name=cfg, **modern_kwargs)
                if cfg
                else load_dataset("allenai/qasper", **modern_kwargs)
            )
        except TypeError:
            # Backward compatibility for odd signatures while still attempting a non-script load.
            return load_dataset("allenai/qasper", name=cfg) if cfg else load_dataset("allenai/qasper")
        except Exception as exc:
            msg = str(exc)
            if "Dataset scripts are no longer supported" in msg and "qasper.py" in msg:
                raise RuntimeError(
                    "Loading 'allenai/qasper' failed with datasets>=4 because script execution is disabled "
                    "and parquet fallback could not be resolved. Use dataset.source=files (local docs/qa JSON), "
                    "or run QASPER in a separate environment with `datasets<4`."
                ) from exc
            raise

    # datasets<4 legacy path.
    legacy_kwargs = {"trust_remote_code": True}
    try:
        try:
            return load_dataset("allenai/qasper", name=cfg, **legacy_kwargs) if cfg else load_dataset("allenai/qasper", **legacy_kwargs)
        except TypeError:
            # Backward compatibility for datasets versions without trust_remote_code parameter.
            return load_dataset("allenai/qasper", name=cfg) if cfg else load_dataset("allenai/qasper")
    except RuntimeError as exc:
        msg = str(exc)
        if "trust_remote_code" in msg:
            raise RuntimeError(
                "Loading 'allenai/qasper' requires script loading. Use `datasets<4` and pass "
                "`trust_remote_code=True` (handled by this loader)."
            ) from exc
        raise


def load_qasper_qa_entries(*, split: str = "test", n: int | str = "all", selection_method: str = "random", config_name: str | None = "default") -> list[dict[str, Any]]:
    import random
    cfg = _resolve_config(config_name)
    logger.info("Loading QASPER QA entries split=%s config=%s", split, cfg)
    ds = _load_qasper_dataset(cfg=cfg)
    qa_entries: list[dict[str, Any]] = []
    for row in ds[split]:
        doc_id = str(row.get("id"))
        qas = row.get("qas") or {}
        qs = qas.get("question") or []
        answers_all = qas.get("answers") or []
        for q, ans_dict in zip(qs, answers_all):
            answer_texts: list[str] = []
            retrieval_spans: list[str] = []
            for ans in (ans_dict or {}).get("answer", []) or []:
                if not isinstance(ans, dict) or ans.get("unanswerable"):
                    continue
                if ans.get("extractive_spans"):
                    txt = " ".join([s for s in ans.get("extractive_spans", []) if isinstance(s, str)])
                elif isinstance(ans.get("free_form_answer"), str) and ans.get("free_form_answer", "").strip():
                    txt = ans["free_form_answer"].strip()
                elif ans.get("yes_no") is not None:
                    txt = "Yes" if bool(ans["yes_no"]) else "No"
                else:
                    txt = ""
                ev = ans.get("evidence") or ans.get("highlighted_evidence") or []
                if txt:
                    answer_texts.append(txt)
                if isinstance(ev, str) and ev.strip():
                    retrieval_spans.append(ev.strip())
                elif isinstance(ev, list):
                    retrieval_spans.extend([s.strip() for s in ev if isinstance(s, str) and s.strip()])
            if answer_texts or retrieval_spans:
                qa_entries.append(
                    {
                        "question": str(q).strip(),
                        "document_id": doc_id,
                        "answers": answer_texts,
                        "retrieval_spans": retrieval_spans,
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
    logger.info("Loaded QASPER QA entries selected=%d total=%d", len(out), total)
    return out


def load_qasper_documents(*, split: str = "test", config_name: str | None = "default") -> dict[str, str]:
    cfg = _resolve_config(config_name)
    logger.info("Loading QASPER documents split=%s config=%s", split, cfg)
    ds = _load_qasper_dataset(cfg=cfg)
    out: dict[str, str] = {}
    for row in ds[split]:
        doc_id = str(row.get("id"))
        full_text = (row.get("full_text") or {}).get("paragraphs", "")
        if isinstance(full_text, list):
            if full_text and isinstance(full_text[0], list):
                text = "\n".join([p for section in full_text for p in section if isinstance(p, str)])
            else:
                text = "\n".join([p for p in full_text if isinstance(p, str)])
        else:
            text = coerce_to_text(full_text) or ""
        out[doc_id] = text
    logger.info("Loaded QASPER documents count=%d", len(out))
    return out
