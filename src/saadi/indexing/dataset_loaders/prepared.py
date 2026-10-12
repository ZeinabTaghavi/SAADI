"""Loads experiment datasets from config (files or HF dataset loaders)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .loogle import load_loogle_documents, load_loogle_qa_entries
from .musique import load_musique_documents, load_musique_qa_entries
from .narrativeqa import load_narrativeqa_documents, load_narrativeqa_qa_entries
from .novelhopqa import load_novelhopqa_documents, load_novelhopqa_qa_entries
from .novelqa import load_novelqa_documents, load_novelqa_qa_entries
from .quality import load_quality_documents, load_quality_qa_entries
from .qasper import load_qasper_documents, load_qasper_qa_entries

logger = logging.getLogger(__name__)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_path(raw: str | None, *, base_dir: Path | None) -> Path | None:
    if raw is None:
        return None
    p = Path(str(raw))
    if p.is_absolute():
        return p
    if base_dir is not None:
        return (base_dir / p).resolve()
    return p.resolve()


def _load_from_files(dataset_cfg: dict[str, Any], *, base_dir: Path | None) -> tuple[dict[str, str], list[dict[str, Any]], dict[str, Any]]:
    docs_path = _resolve_path(dataset_cfg.get("documents_json"), base_dir=base_dir)
    qa_path = _resolve_path(dataset_cfg.get("qa_json"), base_dir=base_dir)
    if docs_path is None or qa_path is None:
        raise ValueError(
            "File dataset source requires dataset.documents_json and dataset.qa_json."
        )
    if not docs_path.exists():
        raise FileNotFoundError(f"documents_json not found: {docs_path}")
    if not qa_path.exists():
        raise FileNotFoundError(f"qa_json not found: {qa_path}")

    docs = _read_json(docs_path)
    qa = _read_json(qa_path)
    if not isinstance(docs, dict):
        raise ValueError(f"documents_json must contain a dict doc_id->text: {docs_path}")
    if not isinstance(qa, list):
        raise ValueError(f"qa_json must contain a list of QA entries: {qa_path}")

    docs_out = {str(k): str(v) for k, v in docs.items()}
    logger.info("Loaded local dataset files docs=%d qa_entries=%d", len(docs_out), len(qa))
    return (
        docs_out,
        list(qa),
        {
            "dataset_source": "files",
            "documents_json": str(docs_path),
            "qa_json": str(qa_path),
            "cached": False,
        },
    )


def _load_qa_only_from_files(dataset_cfg: dict[str, Any], *, base_dir: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    qa_path = _resolve_path(dataset_cfg.get("qa_json"), base_dir=base_dir)
    if qa_path is None:
        raise ValueError("File dataset source requires dataset.qa_json for QA-only loading.")
    if not qa_path.exists():
        raise FileNotFoundError(f"qa_json not found: {qa_path}")
    qa = _read_json(qa_path)
    if not isinstance(qa, list):
        raise ValueError(f"qa_json must contain a list of QA entries: {qa_path}")
    logger.info("Loaded local QA file qa_entries=%d", len(qa))
    return (
        list(qa),
        {
            "dataset_source": "files",
            "qa_json": str(qa_path),
            "cached": False,
        },
    )


def _load_from_hf(dataset_cfg: dict[str, Any]) -> tuple[dict[str, str], list[dict[str, Any]], str]:
    raw_name = str(dataset_cfg.get("name") or "").strip().lower()
    default_split = "validation" if raw_name in {"musique", "dgslibisey/musique"} else "test"
    split = str(dataset_cfg.get("split", default_split))
    config_name = dataset_cfg.get("config_name", "default")
    docs_config_name = dataset_cfg.get("docs_config_name", config_name)
    qa_config_name = dataset_cfg.get("qa_config_name", config_name)
    qa_n = dataset_cfg.get("qa_n", "all")
    qa_selection_method = str(dataset_cfg.get("qa_selection_method", "first"))
    books_root = dataset_cfg.get("books_root")
    raw_allowed_doc_ids = dataset_cfg.get("allowed_doc_ids")
    allowed_doc_ids: set[str] | None = None
    if raw_allowed_doc_ids is not None:
        if isinstance(raw_allowed_doc_ids, (str, bytes)):
            raise ValueError("dataset.allowed_doc_ids must be a sequence of document IDs, not a string.")
        try:
            allowed_doc_ids = {
                str(doc_id).strip()
                for doc_id in raw_allowed_doc_ids
                if str(doc_id).strip()
            }
        except TypeError as exc:
            raise ValueError("dataset.allowed_doc_ids must be a sequence of document IDs.") from exc
    logger.info(
        "Loading Hugging Face dataset name=%s split=%s docs_config=%s qa_config=%s qa_n=%s qa_selection=%s",
        raw_name,
        split,
        docs_config_name,
        qa_config_name,
        qa_n,
        qa_selection_method,
    )

    if raw_name in {"qasper", "allenai/qasper"}:
        docs = load_qasper_documents(split=split, config_name=docs_config_name)
        qa = load_qasper_qa_entries(
            split=split,
            n=qa_n,
            selection_method=qa_selection_method,
            config_name=qa_config_name,
        )
        return docs, qa, "qasper"

    if raw_name in {"nqa", "narrativeqa", "deepmind/narrativeqa"}:
        docs = load_narrativeqa_documents(split=split, config_name=docs_config_name)
        qa = load_narrativeqa_qa_entries(
            split=split,
            n=qa_n,
            selection_method=qa_selection_method,
            config_name=qa_config_name,
        )
        return docs, qa, "narrativeqa"

    if raw_name in {"loogle", "bigai-nlco/loogle", "bigainlco/loogle"}:
        docs = load_loogle_documents(split=split, config_name=docs_config_name)
        qa = load_loogle_qa_entries(
            split=split,
            n=qa_n,
            selection_method=qa_selection_method,
            config_name=qa_config_name,
        )
        return docs, qa, "loogle"

    if raw_name in {"musique", "dgslibisey/musique"}:
        docs = load_musique_documents(split=split, config_name=docs_config_name)
        qa = load_musique_qa_entries(
            split=split,
            n=qa_n,
            selection_method=qa_selection_method,
            config_name=qa_config_name,
        )
        return docs, qa, "musique"

    if raw_name in {"novelqa", "novelqa/novelqa"}:
        docs = load_novelqa_documents(split=split, config_name=docs_config_name)
        qa = load_novelqa_qa_entries(
            split=split,
            n=qa_n,
            selection_method=qa_selection_method,
            config_name=qa_config_name,
        )
        return docs, qa, "novelqa"

    if raw_name in {"novelhopqa", "abhaygupta1266/novelhopqa"}:
        loader_kwargs: dict[str, Any] = {}
        if books_root is not None:
            loader_kwargs["books_root"] = books_root
        if allowed_doc_ids is not None:
            loader_kwargs["allowed_doc_ids"] = allowed_doc_ids
        docs = load_novelhopqa_documents(split=split, config_name=docs_config_name, **loader_kwargs)
        qa = load_novelhopqa_qa_entries(
            split=split,
            n=qa_n,
            selection_method=qa_selection_method,
            config_name=qa_config_name,
            **loader_kwargs,
        )
        return docs, qa, "novelhopqa"

    if raw_name in {"quality", "tasksource/quality"}:
        docs = load_quality_documents(split=split, config_name=docs_config_name)
        qa = load_quality_qa_entries(
            split=split,
            n=qa_n,
            selection_method=qa_selection_method,
            config_name=qa_config_name,
        )
        return docs, qa, "quality"

    raise ValueError(
        "Unsupported dataset.name for HF loading. Use one of: "
        "'qasper', 'narrativeqa'/'nqa', 'loogle', 'musique', 'novelqa', 'novelhopqa', 'quality', or set dataset.source=files."
    )


def load_dataset_from_config(
    dataset_cfg: dict[str, Any] | None,
    *,
    base_dir: str | Path | None = None,
) -> tuple[dict[str, str], list[dict[str, Any]], dict[str, Any]]:
    """
    Resolve documents/QA from dataset config.

    Supported modes:
    - dataset.source=files with dataset.documents_json + dataset.qa_json
    - dataset.name in {qasper, narrativeqa, nqa, loogle, musique, novelqa, novelhopqa, quality} via HF loaders
    """
    cfg = dict(dataset_cfg or {})
    source = str(cfg.get("source", "")).strip().lower()
    base = Path(base_dir).resolve() if base_dir is not None else None

    if source == "files" or (cfg.get("documents_json") and cfg.get("qa_json")):
        return _load_from_files(cfg, base_dir=base)

    if not cfg.get("name"):
        raise ValueError(
            "Missing dataset config. Provide dataset.name "
            "(e.g., qasper/narrativeqa/loogle/musique/novelqa/novelhopqa/quality) "
            "or dataset.source=files with dataset.documents_json + dataset.qa_json."
        )

    # Backward compatibility: old dataset-cache knobs are intentionally ignored.
    ignored_keys = [k for k in ("cache_prepared_json", "refresh_cache", "cache_dir") if k in cfg]
    if ignored_keys:
        logger.info(
            "Ignoring deprecated dataset cache config keys (prepared cache removed): %s",
            ", ".join(sorted(ignored_keys)),
        )

    name = str(cfg.get("name"))
    split = str(cfg.get("split", "test"))
    config_name = cfg.get("config_name", "default")
    docs_config_name = cfg.get("docs_config_name", config_name)
    qa_config_name = cfg.get("qa_config_name", config_name)
    qa_n = cfg.get("qa_n", "all")
    qa_selection_method = str(cfg.get("qa_selection_method", "first"))
    books_root = cfg.get("books_root")
    allowed_doc_ids = cfg.get("allowed_doc_ids")

    docs, qa, canonical_name = _load_from_hf(
        {
            "name": name,
            "split": split,
            "config_name": config_name,
            "docs_config_name": docs_config_name,
            "qa_config_name": qa_config_name,
            "qa_n": qa_n,
            "qa_selection_method": qa_selection_method,
            "books_root": books_root,
            "allowed_doc_ids": allowed_doc_ids,
        }
    )
    docs_out = {str(k): str(v) for k, v in docs.items()}
    qa_out = list(qa)
    logger.info("Prepared dataset loaded name=%s docs=%d qa_entries=%d", canonical_name, len(docs_out), len(qa_out))

    return (
        docs_out,
        qa_out,
        {
            "dataset_source": "hf",
            "dataset_name": canonical_name,
            "split": split,
            "docs_config_name": docs_config_name,
            "qa_config_name": qa_config_name,
            "cached": False,
        },
    )


def load_qa_entries_from_config(
    dataset_cfg: dict[str, Any] | None,
    *,
    base_dir: str | Path | None = None,
    allowed_doc_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Resolve QA entries from dataset config without loading documents.

    Supported modes:
    - dataset.source=files with dataset.qa_json
    - dataset.name in {qasper, narrativeqa, nqa, loogle, novelqa, novelhopqa, quality} via HF loaders
    """
    cfg = dict(dataset_cfg or {})
    source = str(cfg.get("source", "")).strip().lower()
    base = Path(base_dir).resolve() if base_dir is not None else None

    if source == "files" or cfg.get("qa_json"):
        return _load_qa_only_from_files(cfg, base_dir=base)

    if not cfg.get("name"):
        raise ValueError(
            "Missing dataset config. Provide dataset.name (e.g., qasper/narrativeqa/loogle/novelqa/novelhopqa/quality) "
            "or dataset.source=files with dataset.qa_json."
        )

    raw_name = str(cfg.get("name") or "").strip().lower()
    split = str(cfg.get("split", "test"))
    config_name = cfg.get("config_name", "default")
    qa_config_name = cfg.get("qa_config_name", config_name)
    qa_n = cfg.get("qa_n", "all")
    qa_selection_method = str(cfg.get("qa_selection_method", "first"))
    books_root = cfg.get("books_root")

    if raw_name in {"qasper", "allenai/qasper"}:
        qa = load_qasper_qa_entries(
            split=split,
            n=qa_n,
            selection_method=qa_selection_method,
            config_name=qa_config_name,
        )
        canonical_name = "qasper"
    elif raw_name in {"nqa", "narrativeqa", "deepmind/narrativeqa"}:
        qa = load_narrativeqa_qa_entries(
            split=split,
            n=qa_n,
            selection_method=qa_selection_method,
            config_name=qa_config_name,
        )
        canonical_name = "narrativeqa"
    elif raw_name in {"loogle", "bigai-nlco/loogle", "bigainlco/loogle"}:
        qa = load_loogle_qa_entries(
            split=split,
            n=qa_n,
            selection_method=qa_selection_method,
            config_name=qa_config_name,
        )
        canonical_name = "loogle"
    elif raw_name in {"novelqa", "novelqa/novelqa"}:
        qa = load_novelqa_qa_entries(
            split=split,
            n=qa_n,
            selection_method=qa_selection_method,
            config_name=qa_config_name,
        )
        canonical_name = "novelqa"
    elif raw_name in {"novelhopqa", "abhaygupta1266/novelhopqa"}:
        loader_kwargs: dict[str, Any] = {}
        if books_root is not None:
            loader_kwargs["books_root"] = books_root
        if allowed_doc_ids is not None:
            loader_kwargs["allowed_doc_ids"] = set(allowed_doc_ids)
        qa = load_novelhopqa_qa_entries(
            split=split,
            n=qa_n,
            selection_method=qa_selection_method,
            config_name=qa_config_name,
            **loader_kwargs,
        )
        canonical_name = "novelhopqa"
    elif raw_name in {"quality", "tasksource/quality"}:
        qa = load_quality_qa_entries(
            split=split,
            n=qa_n,
            selection_method=qa_selection_method,
            config_name=qa_config_name,
        )
        canonical_name = "quality"
    else:
        raise ValueError(
            "Unsupported dataset.name for HF loading. Use one of: "
            "'qasper', 'narrativeqa'/'nqa', 'loogle', 'novelqa', 'novelhopqa', 'quality', or set dataset.source=files."
        )

    qa_out = list(qa)
    logger.info("Prepared QA entries loaded name=%s qa_entries=%d", canonical_name, len(qa_out))
    return (
        qa_out,
        {
            "dataset_source": "hf",
            "dataset_name": canonical_name,
            "split": split,
            "qa_config_name": qa_config_name,
            "cached": False,
        },
    )
