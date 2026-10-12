"""Loads NovelHopQA documents and QA entries into the SAADI schema."""

from __future__ import annotations

import json
import logging
import os
import re
import time
import unicodedata
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any

from ...core.chunking import coerce_to_text

logger = logging.getLogger(__name__)

_DATASET_ID = "abhaygupta1266/novelhopqa"
_VALID_SPLITS = ("hop_1", "hop_2", "hop_3", "hop_4")
_CONFIG_ALIASES = {
    "default": "all",
    "full": "all",
    "all_hops": "all",
}


def _datasets():
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:
        raise RuntimeError("datasets package is required for NovelHopQA loading. Install with `pip install .[datasets]`.") from exc
    return load_dataset


def _datasets_version_major() -> int | None:
    try:
        raw = str(pkg_version("datasets")).strip()
        if not raw:
            return None
        return int(raw.split(".", 1)[0])
    except Exception:
        return None


def _normalize_config(config_name: str | None) -> str:
    raw = str(config_name or "all").strip().lower()
    norm = _CONFIG_ALIASES.get(raw, raw)
    if norm == "all" or norm in _VALID_SPLITS:
        return norm
    raise ValueError(
        "Unsupported NovelHopQA config_name. Use one of: all/default, hop_1, hop_2, hop_3, hop_4."
    )


def _selected_splits(mode: str) -> list[str]:
    if mode == "all":
        return list(_VALID_SPLITS)
    return [mode]


def _load_novelhopqa_split(split_name: str):
    load_dataset = _datasets()
    major = _datasets_version_major()
    kwargs: dict[str, Any] = {}
    if major is not None and major >= 4:
        kwargs["revision"] = "refs/convert/parquet"
    try:
        return load_dataset(_DATASET_ID, "default", split=split_name, **kwargs)
    except TypeError:
        try:
            return load_dataset(_DATASET_ID, split=split_name, **kwargs)
        except TypeError:
            return load_dataset(_DATASET_ID, split=split_name)
    except Exception:
        # Fallback path for environments that only support main branch/script loading.
        return load_dataset(_DATASET_ID, "default", split=split_name)


def _safe_component(value: str, *, default: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "")).strip("._")
    return out or default


def _normalize_book_key(value: str | None) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = (
        text.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\xa0", " ")
        .replace("\ufeff", " ")
    )
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[\(\[\{].*?[\)\]\}]", " ", text)
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().casefold()


def _query_id(row: dict[str, Any], *, split_name: str, index: int) -> str:
    base = row.get("qid") or row.get("question_id") or row.get("id") or index
    return f"{split_name}:{_safe_component(str(base), default=str(index))}"


def _find_top_file_ci(root: Path, name: str) -> Path | None:
    try:
        for child in root.iterdir():
            if child.is_file() and child.name.lower() == name.lower():
                return child
    except Exception:
        return None
    return None


def _find_child_dir_ci(root: Path, name: str) -> Path | None:
    try:
        for child in root.iterdir():
            if child.is_dir() and child.name.lower() == name.lower():
                return child
    except Exception:
        return None
    return None


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def _looks_like_books_root(root: Path) -> bool:
    if not root.exists():
        return False
    if root.is_file():
        return root.suffix.lower() == ".txt"
    if _find_top_file_ci(root, "bookmeta.json") is not None:
        return True
    books_dir = _find_child_dir_ci(root, "Books")
    if books_dir is not None:
        return True
    try:
        return any(child.is_file() and child.suffix.lower() == ".txt" for child in root.iterdir())
    except Exception:
        return False


def _coerce_books_root(raw_root: Path) -> Path:
    root = raw_root.expanduser().resolve()
    if _looks_like_books_root(root):
        return root
    try:
        for child in root.iterdir():
            if child.is_dir() and _looks_like_books_root(child):
                return child.resolve()
    except Exception:
        pass
    return root


def _resolve_books_root(raw_root: str | os.PathLike[str] | None) -> Path:
    configured_raw = str(raw_root).strip() if raw_root is not None else ""
    env_books_root = str(os.environ.get("NOVELHOPQA_BOOKS_ROOT") or "").strip()
    candidates = [
        ("NOVELHOPQA_BOOKS_ROOT", env_books_root),
        ("dataset.books_root", configured_raw),
    ]
    usable_attempts: list[str] = []
    saw_candidate = False
    for label, configured in candidates:
        if not configured:
            continue
        saw_candidate = True
        root = _coerce_books_root(Path(configured))
        if _looks_like_books_root(root):
            if usable_attempts:
                logger.warning(
                    "Ignoring unusable NovelHopQA corpus root candidate(s): %s. Using %s=%s",
                    ", ".join(usable_attempts),
                    label,
                    root,
                )
            return root
        usable_attempts.append(f"{label}={root}")

    if not saw_candidate:
        raise RuntimeError(
            "NovelHopQA whole-book loading requires a book corpus root. "
            "Set dataset.books_root or NOVELHOPQA_BOOKS_ROOT to a directory containing "
            "bookmeta.json + book text files (or a title-mapped text corpus)."
        )
    raise RuntimeError(
        "NovelHopQA whole-book loading could not find a usable corpus root at "
        f"{'; '.join(usable_attempts)}. Expected bookmeta.json, a Books/ directory, or .txt files."
    )


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


def _candidate_text_paths(root: Path, file_name: str) -> list[Path]:
    p = Path(str(file_name))
    if p.is_absolute():
        return [p]
    if len(p.parts) > 1:
        return [(root / p).resolve()]
    return [
        (root / "Books" / "PublicDomain" / p).resolve(),
        (root / "Books" / "publicdomain" / p).resolve(),
        (root / "Books" / "CopyrightProtected" / p).resolve(),
        (root / "Books" / "copyrightprotected" / p).resolve(),
        (root / "Books" / p).resolve(),
        (root / p).resolve(),
    ]


def _book_doc_id(title: str, *, fallback: str) -> str:
    return f"book:{_safe_component(title, default=fallback)}"


def _book_title(row: dict[str, Any]) -> str | None:
    return coerce_to_text(row.get("book") or row.get("title") or row.get("book_title"))


def _env_flag(name: str) -> bool:
    raw = str(os.environ.get(name) or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _iter_title_variants(value: str | None) -> list[str]:
    raw = str(value or "").strip()
    if not raw:
        return []
    variants: list[str] = [raw]
    cleanup_suffixes = (
        " complete",
        " unabridged",
        " illustrated",
        " with illustrations",
    )
    for sep in (":", ";", ",", " - ", " — ", " – "):
        if sep in raw:
            head = raw.split(sep, 1)[0].strip()
            if head:
                variants.append(head)
    for candidate in list(variants):
        lowered = candidate.casefold()
        for prefix in ("the ", "a ", "an "):
            if lowered.startswith(prefix):
                trimmed = candidate[len(prefix):].strip()
                if trimmed:
                    variants.append(trimmed)
        for suffix in cleanup_suffixes:
            if lowered.endswith(suffix):
                trimmed = candidate[: -len(suffix)].strip(" ,;-:")
                if trimmed:
                    variants.append(trimmed)
    out: list[str] = []
    seen: set[str] = set()
    for candidate in variants:
        key = _normalize_book_key(candidate)
        if key and key not in seen:
            seen.add(key)
            out.append(candidate)
    return out


def _title_like_lines(text: str, *, limit: int = 5) -> list[str]:
    raw_candidates: list[str] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.replace("\ufeff", "").replace("\xa0", " ").strip()
        if not line:
            continue
        lowered = line.casefold()
        if "project gutenberg ebook of" in lowered:
            match = re.search(
                r"project gutenberg ebook of\s+(.+?)(?:,\s+by\b|$)",
                line,
                flags=re.IGNORECASE,
            )
            if match:
                title = match.group(1).strip(" .,:;!-")
                if title:
                    raw_candidates.append(title)
                    if len(raw_candidates) >= limit:
                        break
            continue
        if (
            "project gutenberg" in lowered
            or "www.gutenberg.org" in lowered
            or lowered.startswith("***")
            or lowered.startswith("by ")
            or lowered.startswith("translated by ")
            or lowered.startswith("produced by ")
            or lowered.startswith("release date")
            or lowered.startswith("language:")
            or lowered.startswith("contents")
            or lowered.startswith("chapter ")
            or lowered.startswith("book ")
        ):
            continue
        if len(line) > 160:
            continue
        if sum(ch.isalpha() for ch in line) < 3:
            continue
        raw_candidates.append(line)
        if len(raw_candidates) >= limit:
            break

    out: list[str] = []
    seen: set[str] = set()
    for candidate in raw_candidates:
        normalized = candidate.strip(" .,:;!-")
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            out.append(normalized)

    # Some title pages split the title across multiple short lines, e.g.
    # "The Adventures of" + "Huckleberry Finn". Stitch together the early
    # title-like lines so HF book titles can resolve back to the .txt file.
    stitched_parts: list[str] = []
    total_words = 0
    for candidate in raw_candidates[:3]:
        lowered = candidate.casefold()
        if stitched_parts and (
            lowered.startswith("by ")
            or lowered.startswith("translated by ")
            or lowered.startswith("produced by ")
        ):
            break
        if len(candidate) > 80:
            break
        stitched_parts.append(candidate.strip(" .,:;!-"))
        total_words += len(candidate.split())
        if len(stitched_parts) < 2 or total_words > 12:
            continue
        stitched = " ".join(part for part in stitched_parts if part).strip(" .,:;!-")
        key = stitched.casefold()
        if stitched and key not in seen:
            seen.add(key)
            out.append(stitched)
    return out


def _register_book_aliases(
    books: dict[str, tuple[str, str]],
    *,
    doc_id: str,
    text: str,
    candidates: list[str],
) -> None:
    for raw in candidates:
        for variant in _iter_title_variants(raw):
            key = _normalize_book_key(variant)
            if key:
                books.setdefault(key, (doc_id, text))


def _report_dir() -> Path:
    configured = str(os.environ.get("NOVELHOPQA_REPORT_DIR") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[4] / "novelhopqa"


def _write_title_report(path: Path, titles: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted({str(title).strip() for title in titles if str(title).strip()}, key=str.casefold)
    content = "\n".join(ordered)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def _load_books_from_root(root: Path) -> tuple[dict[str, tuple[str, str]], dict[str, str]]:
    books: dict[str, tuple[str, str]] = {}
    titles_by_doc_id: dict[str, str] = {}
    bookmeta_path = _find_top_file_ci(root, "bookmeta.json")
    if bookmeta_path is not None and bookmeta_path.exists():
        started = time.monotonic()
        processed = 0
        payload = _read_json(bookmeta_path)
        for i, row in enumerate(_iter_bookmeta_entries(payload)):
            title = coerce_to_text(row.get("title") or row.get("book") or row.get("name"))
            doc_id = str(row.get("BID") or row.get("bid") or _book_doc_id(title, fallback=f"book_{i}")).strip()
            txt_name = row.get("txtfile") or row.get("txt_file") or row.get("book_file") or row.get("text_file")
            candidate_names: list[str] = []
            if isinstance(txt_name, str) and txt_name.strip():
                candidate_names.append(txt_name.strip())
            if doc_id:
                candidate_names.extend(
                    [
                        f"{doc_id}.txt",
                        f"{doc_id.upper()}.txt",
                        f"{doc_id.lower()}.txt",
                    ]
                )
            if not title or not candidate_names:
                continue
            seen_candidates: set[Path] = set()
            for raw_name in candidate_names:
                for candidate in _candidate_text_paths(root, raw_name):
                    if candidate in seen_candidates:
                        continue
                    seen_candidates.add(candidate)
                    if not candidate.exists():
                        continue
                    text = _load_text(candidate)
                    if not text:
                        continue
                    aliases = [title, doc_id, Path(raw_name).stem, candidate.stem, *_title_like_lines(text)]
                    canonical_title = str(title or candidate.stem or doc_id).strip()
                    if canonical_title:
                        titles_by_doc_id.setdefault(doc_id, canonical_title)
                    _register_book_aliases(books, doc_id=doc_id, text=text, candidates=aliases)
                    processed += 1
                    if processed % 50 == 0:
                        elapsed = time.monotonic() - started
                        logger.info(
                            "NovelHopQA corpus scan progress via bookmeta.json: loaded=%d elapsed=%.1fs root=%s",
                            processed,
                            elapsed,
                            root,
                        )
                    break
                else:
                    continue
                break
        if books:
            elapsed = time.monotonic() - started
            logger.info(
                "NovelHopQA corpus scan done via bookmeta.json: loaded=%d elapsed=%.1fs root=%s",
                processed,
                elapsed,
                root,
            )
            return books, titles_by_doc_id

    if root.is_file() and root.suffix.lower() == ".txt":
        title = root.stem
        text = _load_text(root)
        if not text:
            return {}, {}
        doc_id = _book_doc_id(title, fallback="book")
        titles_by_doc_id[doc_id] = title
        _register_book_aliases(
            books,
            doc_id=doc_id,
            text=text,
            candidates=[title, * _title_like_lines(text)],
        )
        return books, titles_by_doc_id

    try:
        started = time.monotonic()
        processed = 0
        txt_files = sorted(p for p in root.rglob("*.txt") if p.is_file())
    except Exception:
        txt_files = []
    for path in txt_files:
        text = _load_text(path)
        if not text:
            continue
        stem = path.stem
        doc_id = _book_doc_id(stem, fallback=stem)
        title_candidates = _title_like_lines(text)
        canonical_title = str(title_candidates[0] if title_candidates else stem).strip()
        if canonical_title:
            titles_by_doc_id.setdefault(doc_id, canonical_title)
        _register_book_aliases(
            books,
            doc_id=doc_id,
            text=text,
            candidates=[stem, *title_candidates],
        )
        processed += 1
        if processed % 200 == 0:
            elapsed = time.monotonic() - started
            logger.info(
                "NovelHopQA corpus scan progress via rglob: loaded=%d/%d elapsed=%.1fs root=%s",
                processed,
                len(txt_files),
                elapsed,
                root,
            )
    if txt_files:
        elapsed = time.monotonic() - started
        logger.info(
            "NovelHopQA corpus scan done via rglob: loaded=%d/%d elapsed=%.1fs root=%s",
            processed,
            len(txt_files),
            elapsed,
            root,
        )
    return books, titles_by_doc_id

def _load_novelhopqa_all(
    *,
    mode: str,
    split: str,
    books_root: str | os.PathLike[str] | None = None,
    allowed_doc_ids: set[str] | None = None,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    # `split` is kept for API compatibility; NovelHopQA provides hop-based splits.
    _ = split
    root = _resolve_books_root(books_root)
    books_by_key, titles_by_doc_id = _load_books_from_root(root)
    if not books_by_key:
        raise RuntimeError(
            "NovelHopQA whole-book loading found zero books under "
            f"{root}. Expected title-mapped .txt files or bookmeta.json entries."
        )

    allowed_doc_id_set = {str(doc_id).strip() for doc_id in (allowed_doc_ids or set()) if str(doc_id).strip()}
    allowed_title_keys: set[str] | None = None
    if allowed_doc_id_set:
        allowed_title_keys = {
            _normalize_book_key(title)
            for doc_id, title in titles_by_doc_id.items()
            if str(doc_id).strip() in allowed_doc_id_set and str(title).strip()
        }

    docs: dict[str, str] = {}
    qa_entries: list[dict[str, Any]] = []
    missing_books: set[str] = set()
    used_doc_ids: set[str] = set()
    for split_name in _selected_splits(mode):
        rows = _load_novelhopqa_split(split_name)
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            book_title = _book_title(row)
            if not book_title:
                continue
            book_key = _normalize_book_key(book_title)
            if allowed_title_keys is not None and book_key not in allowed_title_keys:
                continue
            book_record = books_by_key.get(book_key)
            if book_record is None:
                missing_books.add(book_title)
                continue
            doc_id, document_text = book_record
            if allowed_doc_id_set and str(doc_id).strip() not in allowed_doc_id_set:
                continue
            used_doc_ids.add(doc_id)
            if document_text and doc_id not in docs:
                docs[doc_id] = document_text

            context = coerce_to_text(row.get("context") or row.get("passage") or row.get("document"))
            question = coerce_to_text(row.get("question") or row.get("query"))
            answer = coerce_to_text(row.get("answer") or row.get("gold_answer"))
            if not context:
                continue
            if not question or not answer:
                continue
            qa_entries.append(
                {
                    "id": _query_id(row, split_name=split_name, index=i),
                    "question": question,
                    "document_id": doc_id,
                    "book_title": book_title,
                    "gold_context_window": context,
                    "retrieval_span_mode": "window",
                    "answers": [answer],
                    "retrieval_spans": [context],
                }
            )

    report_dir = _report_dir()
    _write_title_report(report_dir / "novelhop_query_remained.txt", missing_books)
    unused_books = {
        title
        for doc_id, title in titles_by_doc_id.items()
        if doc_id not in used_doc_ids
    }
    _write_title_report(report_dir / "novelhop_book_remained.txt", unused_books)

    if missing_books:
        sample = ", ".join(sorted(missing_books)[:5])
        if not _env_flag("NOVELHOPQA_SUBSET_MODE"):
            raise RuntimeError(
                "NovelHopQA whole-book loading could not resolve book texts for "
                f"{len(missing_books)} title(s) under {root}. Example title(s): {sample}. "
                "Provide dataset.books_root or NOVELHOPQA_BOOKS_ROOT pointing to a corpus with matching titles."
            )
        logger.warning(
            "NovelHopQA subset mode enabled; skipped %d unresolved title(s) under %s. "
            "Example title(s): %s. Reports written to %s",
            len(missing_books),
            root,
            sample,
            report_dir,
        )
    return docs, qa_entries


def load_novelhopqa_documents(
    *,
    split: str = "test",
    config_name: str | None = "all",
    books_root: str | os.PathLike[str] | None = None,
    allowed_doc_ids: set[str] | None = None,
) -> dict[str, str]:
    mode = _normalize_config(config_name)
    logger.info("Loading NovelHopQA documents split=%s config=%s", split, mode)
    docs, _ = _load_novelhopqa_all(mode=mode, split=split, books_root=books_root, allowed_doc_ids=allowed_doc_ids)
    logger.info("Loaded NovelHopQA documents count=%d", len(docs))
    return docs


def load_novelhopqa_qa_entries(
    *,
    split: str = "test",
    n: int | str = "all",
    selection_method: str = "random",
    config_name: str | None = "all",
    books_root: str | os.PathLike[str] | None = None,
    allowed_doc_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    import random

    mode = _normalize_config(config_name)
    logger.info("Loading NovelHopQA QA entries split=%s config=%s", split, mode)
    docs, qa_entries = _load_novelhopqa_all(
        mode=mode,
        split=split,
        books_root=books_root,
        allowed_doc_ids=allowed_doc_ids,
    )
    valid_doc_ids = set(docs.keys())
    qa_entries = [row for row in qa_entries if str(row.get("document_id", "")) in valid_doc_ids]
    total = len(qa_entries)
    if isinstance(n, str):
        n = total if n.lower() == "all" else int(n)
    n = min(int(n), total)
    idxs = list(range(total))
    if n < total:
        idxs = random.sample(idxs, n) if selection_method == "random" else idxs[:n]
    out = [{"id": qa_entries[i].get("id", i), **{k: v for k, v in qa_entries[i].items() if k != "id"}} for i in idxs]
    logger.info("Loaded NovelHopQA QA entries selected=%d total=%d", len(out), total)
    return out
