"""Loads NovelQA documents and QA entries into the SAADI schema."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from ...core.chunking import coerce_to_text

logger = logging.getLogger(__name__)

_DATASET_REPO_ID = "NovelQA/NovelQA"
_CONFIG_ALIASES = {
    "default": "demonstration",
    "demo": "demonstration",
    "public": "publicdomain",
    "public_domain": "publicdomain",
    "copyright": "copyrightprotected",
    "copyright_protected": "copyrightprotected",
}


def _find_child_dir_ci(root: Path, name: str) -> Path | None:
    try:
        for child in root.iterdir():
            if child.is_dir() and child.name.lower() == name.lower():
                return child
    except Exception:
        return None
    return None


def _find_top_file_ci(root: Path, name: str) -> Path | None:
    try:
        for child in root.iterdir():
            if child.is_file() and child.name.lower() == name.lower():
                return child
    except Exception:
        return None
    return None


def _looks_like_dataset_root(root: Path) -> bool:
    if not root.exists() or not root.is_dir():
        return False
    if _find_top_file_ci(root, "bookmeta.json") is not None:
        return True
    if _find_child_dir_ci(root, "Demonstration") is not None:
        return True
    data_dir = _find_child_dir_ci(root, "Data")
    if data_dir and (
        _find_child_dir_ci(data_dir, "PublicDomain") is not None
        or _find_child_dir_ci(data_dir, "CopyrightProtected") is not None
    ):
        return True
    return False


def _coerce_dataset_root(raw_root: Path) -> Path:
    root = raw_root.expanduser().resolve()
    if _looks_like_dataset_root(root):
        return root
    try:
        for child in root.iterdir():
            if child.is_dir() and _looks_like_dataset_root(child):
                return child.resolve()
    except Exception:
        pass
    return root


def _snapshot_download(*, local_files_only: bool = False) -> str:
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "NovelQA loading requires `huggingface_hub` (installed with `datasets`). "
            "Install with `pip install .[datasets]`."
        ) from exc

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    kwargs: dict[str, Any] = {
        "repo_id": _DATASET_REPO_ID,
        "repo_type": "dataset",
        "local_files_only": bool(local_files_only),
    }
    if token:
        kwargs["token"] = token
    return str(snapshot_download(**kwargs))


def _resolve_root() -> Path:
    local_override = os.environ.get("NOVELQA_DATASET_DIR")
    if local_override:
        root = _coerce_dataset_root(Path(local_override))
        if _looks_like_dataset_root(root):
            return root
        logger.warning(
            "NOVELQA_DATASET_DIR is set but expected NovelQA layout was not found: %s",
            root,
        )

    online_exc: Exception | None = None
    local_cache_exc: Exception | None = None
    last_root: Path | None = None
    for local_only in (False, True):
        try:
            root = _coerce_dataset_root(Path(_snapshot_download(local_files_only=local_only)))
            if _looks_like_dataset_root(root):
                return root
            last_root = root
        except Exception as exc:
            if local_only:
                local_cache_exc = exc
            else:
                online_exc = exc

    if last_root is not None:
        raise RuntimeError(
            "NovelQA snapshot was resolved but expected files were not found under "
            f"{last_root}. Expected one of: bookmeta.json, Demonstration/, Data/PublicDomain/."
        )
    if online_exc is not None:
        details = f"Online download failed with: {online_exc!r}."
        if local_cache_exc is not None:
            details += f" Local-cache fallback failed with: {local_cache_exc!r}."
        raise RuntimeError(
            "Failed to access NovelQA dataset. Ensure you accepted access at "
            "https://huggingface.co/datasets/NovelQA/NovelQA and that HF_TOKEN (or "
            "HUGGINGFACE_TOKEN) has permission. "
            + details
        ) from online_exc
    raise RuntimeError(
        "Failed to access NovelQA dataset. Ensure you accepted access at "
        "https://huggingface.co/datasets/NovelQA/NovelQA and set HF_TOKEN if needed."
    ) from local_cache_exc


def _normalize_mode(config_name: str | None) -> str:
    raw = str(config_name or "demonstration").strip().lower()
    return _CONFIG_ALIASES.get(raw, raw)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_bookmeta_entries(payload: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                out.append(item)
        return out
    if isinstance(payload, dict):
        books = payload.get("books")
        if isinstance(books, list):
            for item in books:
                if isinstance(item, dict):
                    out.append(item)
            return out
        for key, value in payload.items():
            if not isinstance(value, dict):
                continue
            row = dict(value)
            row.setdefault("BID", key)
            out.append(row)
    return out


def _resolve_doc_id(record: dict[str, Any], *, fallback: str) -> str:
    for key in ("BID", "bid", "book_id", "bookid", "doc_id", "document_id", "id"):
        value = record.get(key)
        if isinstance(value, (int, str)) and str(value).strip():
            return str(value).strip()
    return fallback


def _resolve_qa_doc_id(record: dict[str, Any], *, fallback: str) -> str:
    # QA rows often carry an `id` field for the question itself. Restrict the
    # document-id lookup to explicit book/document keys so we preserve the
    # book-level mapping inferred from the QA file/bookmeta.
    for key in ("BID", "bid", "book_id", "bookid", "doc_id", "document_id"):
        value = record.get(key)
        if isinstance(value, (int, str)) and str(value).strip():
            return str(value).strip()
    return fallback


def _candidate_text_paths(root: Path, file_name: str) -> list[Path]:
    p = Path(str(file_name))
    if p.is_absolute():
        return [p]
    if len(p.parts) > 1:
        return [(root / p).resolve()]
    return [
        (root / "Demonstration" / p).resolve(),
        (root / "demonstration" / p).resolve(),
        (root / "Books" / "PublicDomain" / p).resolve(),
        (root / "Books" / "publicdomain" / p).resolve(),
        (root / "Books" / "CopyrightProtected" / p).resolve(),
        (root / "Books" / "copyrightprotected" / p).resolve(),
        (root / "Books" / p).resolve(),
        (root / p).resolve(),
    ]


def _first_str_by_suffix(record: dict[str, Any], suffix: str) -> str | None:
    suffix_l = suffix.lower()
    for value in record.values():
        if isinstance(value, str) and value.strip() and value.strip().lower().endswith(suffix_l):
            return value.strip()
    return None


def _load_docs_from_bookmeta(root: Path) -> tuple[dict[str, str], dict[str, str]]:
    docs: dict[str, str] = {}
    qa_file_to_doc: dict[str, str] = {}
    bookmeta_path = _find_top_file_ci(root, "bookmeta.json")
    if bookmeta_path is None or not bookmeta_path.exists():
        return docs, qa_file_to_doc
    payload = _read_json(bookmeta_path)
    for i, row in enumerate(_iter_bookmeta_entries(payload)):
        doc_id = _resolve_doc_id(row, fallback=f"book_{i}")
        qa_name = (
            row.get("jsonfile")
            or row.get("json_file")
            or row.get("qa_file")
            or row.get("qa_path")
            or _first_str_by_suffix(row, ".json")
        )
        if isinstance(qa_name, str) and qa_name.strip():
            qa_basename = Path(qa_name.strip()).name
            qa_file_to_doc[qa_basename] = doc_id
            qa_file_to_doc[Path(qa_basename).stem] = doc_id
            qa_file_to_doc[qa_basename.lower()] = doc_id
            qa_file_to_doc[Path(qa_basename).stem.lower()] = doc_id
        txt_name = (
            row.get("txtfile")
            or row.get("txt_file")
            or row.get("book_file")
            or row.get("txt_path")
            or row.get("text_file")
            or _first_str_by_suffix(row, ".txt")
        )
        if not isinstance(txt_name, str) or not txt_name.strip():
            continue
        for candidate in _candidate_text_paths(root, txt_name.strip()):
            if candidate.exists():
                text = candidate.read_text(encoding="utf-8", errors="ignore").strip()
                if text and doc_id not in docs:
                    docs[doc_id] = text
                break
    return docs, qa_file_to_doc


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def _attach_doc_from_qa_stem_if_available(root: Path, qa_path: Path, *, doc_id: str, docs: dict[str, str]) -> None:
    if doc_id in docs:
        return
    stem = qa_path.stem
    candidates = [
        qa_path.with_suffix(".txt"),
        root / "Demonstration" / f"{stem}.txt",
        root / "demonstration" / f"{stem}.txt",
        root / "Books" / "PublicDomain" / f"{stem}.txt",
        root / "Books" / "publicdomain" / f"{stem}.txt",
        root / "Books" / "CopyrightProtected" / f"{stem}.txt",
        root / "Books" / "copyrightprotected" / f"{stem}.txt",
        root / "Books" / f"{stem}.txt",
    ]
    try:
        for candidate in root.rglob(f"{stem}.txt"):
            candidates.append(candidate)
        for candidate in root.rglob(f"{stem}.TXT"):
            candidates.append(candidate)
    except Exception:
        pass
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if not candidate.exists():
            continue
        text = _load_text(candidate)
        if text:
            docs[doc_id] = text
            return


def _discover_qa_files(root: Path, *, mode: str) -> list[Path]:
    try:
        all_json = [p for p in root.rglob("*.json") if p.is_file()]
    except Exception:
        all_json = []
    files: list[Path] = []
    for p in all_json:
        parts_l = [part.lower() for part in p.parts]
        has_demo = "demonstration" in parts_l
        has_pd = "data" in parts_l and "publicdomain" in parts_l
        has_cp = "data" in parts_l and "copyrightprotected" in parts_l
        if mode == "demonstration":
            if has_demo:
                files.append(p)
        elif mode == "publicdomain":
            if has_pd:
                files.append(p)
        elif mode == "copyrightprotected":
            if has_cp:
                files.append(p)
        elif mode == "all":
            # "all" should include every available split, including the
            # Demonstration sample set. NovelQA mixes book coverage across
            # these folders, and retrieval experiments may intentionally pair
            # demonstration QA with an "all docs" corpus.
            if has_pd or has_cp or has_demo:
                files.append(p)
        else:
            if has_demo:
                files.append(p)
    return sorted(files)


def _to_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        txt = value.strip()
        return [txt] if txt else []
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            out.extend(_to_text_list(item))
        return out
    text = coerce_to_text(value)
    return [text] if text else []


def _options_map(record: dict[str, Any]) -> dict[str, str]:
    raw = record.get("Options") or record.get("options") or record.get("choices")
    out: dict[str, str] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            key = str(k).strip()
            val = coerce_to_text(v)
            if key and val:
                out[key] = val
        return out
    if isinstance(raw, list):
        for i, v in enumerate(raw):
            val = coerce_to_text(v)
            if val:
                out[str(i)] = val
        return out
    return out


def _extract_answers(record: dict[str, Any], options: dict[str, str]) -> tuple[list[str], list[str]]:
    raw_answer = (
        record.get("Answer")
        or record.get("answer")
        or record.get("Golden")
        or record.get("golden")
        or record.get("gold_answer")
        or record.get("golden_answer")
        or record.get("GoldenAnswer")
        or record.get("correct_answer")
    )
    answer_texts = _to_text_list(raw_answer)
    if len(answer_texts) == 1:
        maybe_key = answer_texts[0]
        if maybe_key in options:
            answer_texts = [options[maybe_key]]

    raw_evidence = (
        record.get("Evidence")
        or record.get("Evidences")
        or record.get("evidence")
        or record.get("supporting_evidence")
        or record.get("gold_evidence")
    )
    spans = _to_text_list(raw_evidence)
    return answer_texts, spans


def _iter_qa_rows(payload: Any) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    if isinstance(payload, list):
        for i, item in enumerate(payload):
            if isinstance(item, dict):
                if len(item) == 1:
                    key = next(iter(item.keys()))
                    val = item[key]
                    if isinstance(val, dict):
                        rows.append((str(key), val))
                        continue
                qid = item.get("QID") or item.get("qid") or item.get("id") or f"q_{i}"
                rows.append((str(qid), item))
        return rows
    if isinstance(payload, dict):
        # Some files store doc-level fields plus QID-keyed objects.
        for key, value in payload.items():
            if isinstance(value, dict) and any(k in value for k in ("Question", "question", "Options", "options", "Answer", "answer")):
                rows.append((str(key), value))
        if rows:
            return rows
        nested = payload.get("questions") or payload.get("qas") or payload.get("data")
        if isinstance(nested, list):
            for i, item in enumerate(nested):
                if not isinstance(item, dict):
                    continue
                qid = item.get("QID") or item.get("qid") or item.get("id") or f"q_{i}"
                rows.append((str(qid), item))
    return rows


def _load_novelqa_all(
    *,
    mode: str,
    split: str,
    require_labeled_qa: bool = True,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    # `split` is kept for API compatibility; NovelQA is organized by folders/configs.
    _ = split
    root = _resolve_root()
    docs, qa_file_to_doc = _load_docs_from_bookmeta(root)
    qa_files = _discover_qa_files(root, mode=mode)
    qa_entries: list[dict[str, Any]] = []

    for qa_path in qa_files:
        payload = _read_json(qa_path)
        mapped_doc_id = (
            qa_file_to_doc.get(qa_path.name)
            or qa_file_to_doc.get(qa_path.stem)
            or qa_file_to_doc.get(qa_path.name.lower())
            or qa_file_to_doc.get(qa_path.stem.lower())
            or qa_path.stem
        )
        _attach_doc_from_qa_stem_if_available(root, qa_path, doc_id=mapped_doc_id, docs=docs)
        doc_level_text = None
        if isinstance(payload, dict):
            doc_level_text = (
                coerce_to_text(payload.get("context"))
                or coerce_to_text(payload.get("book_text"))
                or coerce_to_text(payload.get("content"))
            )
        if doc_level_text and mapped_doc_id not in docs:
            docs[mapped_doc_id] = doc_level_text

        for qid, row in _iter_qa_rows(payload):
            rec_doc_id = _resolve_qa_doc_id(row, fallback=mapped_doc_id)
            _attach_doc_from_qa_stem_if_available(root, qa_path, doc_id=rec_doc_id, docs=docs)
            rec_text = (
                coerce_to_text(row.get("context"))
                or coerce_to_text(row.get("book_text"))
                or coerce_to_text(row.get("content"))
            )
            if rec_text and rec_doc_id not in docs:
                docs[rec_doc_id] = rec_text

            question = coerce_to_text(row.get("Question") or row.get("question") or row.get("query"))
            if not question:
                continue
            if rec_doc_id not in docs:
                continue
            options = _options_map(row)
            answer_texts, spans = _extract_answers(row, options)
            if not answer_texts and not spans:
                # NovelQA public subsets often hide gold labels; skip unlabeled rows.
                continue
            qa_entries.append(
                {
                    "id": str(qid),
                    "question": question,
                    "document_id": rec_doc_id,
                    "answers": answer_texts,
                    "retrieval_spans": spans,
                }
            )

    if not docs:
        raise RuntimeError(
            "NovelQA loader resolved a dataset root but found zero documents. "
            f"root={root} mode={mode} split={split}."
        )
    if require_labeled_qa and mode in {"demonstration", "all"} and not qa_entries:
        raise RuntimeError(
            "NovelQA loader found zero labeled QA entries for a labeled mode. "
            f"root={root} mode={mode} split={split}. "
            "Confirm dataset access approval and files inside Demonstration/."
        )

    return docs, qa_entries


def load_novelqa_documents(*, split: str = "test", config_name: str | None = "demonstration") -> dict[str, str]:
    mode = _normalize_mode(config_name)
    logger.info("Loading NovelQA documents split=%s config=%s", split, mode)
    docs, _ = _load_novelqa_all(
        mode=mode,
        split=split,
        require_labeled_qa=False,
    )
    logger.info("Loaded NovelQA documents count=%d", len(docs))
    return docs


def load_novelqa_qa_entries(*, split: str = "test", n: int | str = "all", selection_method: str = "random", config_name: str | None = "demonstration") -> list[dict[str, Any]]:
    import random

    mode = _normalize_mode(config_name)
    logger.info("Loading NovelQA QA entries split=%s config=%s", split, mode)
    docs, rows = _load_novelqa_all(
        mode=mode,
        split=split,
        require_labeled_qa=True,
    )
    valid_doc_ids = set(docs.keys())
    qa_entries = [row for row in rows if str(row.get("document_id", "")) in valid_doc_ids]

    total = len(qa_entries)
    if isinstance(n, str):
        n = total if n.lower() == "all" else int(n)
    n = min(int(n), total)
    idxs = list(range(total))
    if n < total:
        idxs = random.sample(idxs, n) if selection_method == "random" else idxs[:n]
    out = [{"id": qa_entries[i].get("id", i), **{k: v for k, v in qa_entries[i].items() if k != "id"}} for i in idxs]
    logger.info("Loaded NovelQA QA entries selected=%d total=%d", len(out), total)
    return out
