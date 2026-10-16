"""Readable exports of a document state: entity memory, history levels, summaries."""

from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .core.prompts import prompt_hash
from .llm.base import ChatPrompt, render_prompt
from .prompts import PromptProfile

DEFAULT_FORBIDDEN_ALIASES = frozenset(
    {
        "i", "me", "my", "mine", "we", "us", "our", "ours", "you", "your", "yours",
        "he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", "their", "theirs",
        "patient", "therapist",
    }
)
MEMORY_FILE = "memory.json"
HISTORY_DIR = "history"
SUMMARY_FILE = "summary.txt"


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def sanitize_memory_summary(path: Path | None, *, forbidden_aliases: Iterable[str] = DEFAULT_FORBIDDEN_ALIASES) -> int:
    """Remove role labels and standalone pronouns from the exported aliases.

    Prompt guidance is the primary control; this deterministic guard keeps an
    occasional model slip out of downstream use. Returns how many were removed.
    """

    if path is None or not path.is_file():
        return 0
    forbidden = {str(value).strip().casefold() for value in forbidden_aliases}
    payload = json.loads(path.read_text(encoding="utf-8"))
    removed = 0
    for entity in payload.get("entities", []):
        kept = []
        for alias in entity.get("aliases", []) or []:
            text = alias.get("text", "") if isinstance(alias, dict) else str(alias)
            if str(text).strip().casefold() in forbidden:
                removed += 1
            else:
                kept.append(alias)
        entity["aliases"] = kept
    payload["export_sanitization"] = {"removed_forbidden_aliases": removed, "rule": "role labels and standalone pronouns"}
    write_json(path, payload)
    return removed


@dataclass(frozen=True, slots=True)
class ReadableState:
    memory_path: Path | None
    history_dir: Path
    levels: tuple[int, ...]
    summary_path: Path | None
    aliases_removed: int


def find_cache_index(artifacts: Any) -> Path | None:
    """Locate the construction cache's ``index.json`` from the ingest artifacts."""

    if artifacts is None:
        return None
    candidate = Path(str(getattr(artifacts, "llm_cache_index_json", "") or ""))
    if candidate.is_file():
        return candidate
    cache_dir = Path(str(getattr(artifacts, "llm_cache_dir", "") or ""))
    matches = sorted(cache_dir.rglob("index.json")) if cache_dir.is_dir() else []
    return matches[0] if matches else None


def _resolve_indexed_cache_file(index_path: Path, raw_path: str, doc_cache_dir: str) -> Path | None:
    recorded = Path(str(raw_path or ""))
    if recorded.is_file():
        return recorded
    for candidate in (index_path.parent / Path(str(doc_cache_dir or "")).name / recorded.name, index_path.parent / recorded.name):
        if candidate.is_file():
            return candidate
    matches = sorted(index_path.parent.rglob(recorded.name)) if recorded.name else []
    return matches[0] if matches else None


def export_readable_state(
    state_dir: Path,
    *,
    artifacts: Any,
    memory_summary_path: Path | None,
    document_summary_path: Path | None = None,
) -> ReadableState:
    """Write ``memory.json``, ``history/L_n.{jsonl,txt}``, and ``summary.txt`` into ``state_dir``."""

    state_dir = Path(state_dir)
    history_dir = state_dir / HISTORY_DIR
    history_dir.mkdir(parents=True, exist_ok=True)
    for stale in history_dir.glob("L_*"):
        stale.unlink()

    index_path = find_cache_index(artifacts)
    index: dict[str, Any] = {}
    docs: list[Any] = []
    if index_path is not None:
        loaded = json.loads(index_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            index = loaded
            docs = loaded.get("docs", []) if isinstance(loaded.get("docs"), list) else []
    branching_factor = int(index.get("hierarchy_branching_factor") or index.get("branching_factor") or 5)

    levels: list[int] = []
    for doc in docs:
        if not isinstance(doc, dict) or index_path is None:
            continue
        level_items: list[tuple[int, str]] = []
        for key, raw_path in (doc.get("files", {}) or {}).items():
            match = re.fullmatch(r"L_(\d+)", str(key))
            if match:
                level_items.append((int(match.group(1)), str(raw_path)))
        for level, raw_path in sorted(level_items):
            source_path = _resolve_indexed_cache_file(index_path, raw_path, str(doc.get("doc_cache_dir", "")))
            if source_path is None:
                continue
            rows: list[dict[str, Any]] = []
            span_size = branching_factor**level
            for line in source_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                source_row = json.loads(line)
                step = int(source_row.get("step", len(rows)))
                rows.append(
                    {
                        "level": level,
                        "index": len(rows),
                        "start_step": max(0, step - span_size + 1),
                        "end_step": step,
                        "span_size": span_size,
                        "kind": "chunk" if level == 0 else "summary",
                        "text": str(source_row.get("chunk", "") or ""),
                    }
                )
            (history_dir / f"L_{level}.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
            )
            (history_dir / f"L_{level}.txt").write_text(
                "\n\n".join(f"[L{level} item {row['index']} | steps {row['start_step']}-{row['end_step']}]\n{row['text']}" for row in rows)
                + ("\n" if rows else ""),
                encoding="utf-8",
            )
            levels.append(level)

    memory_path: Path | None = None
    aliases_removed = 0
    if memory_summary_path is not None and Path(memory_summary_path).is_file():
        memory_path = state_dir / MEMORY_FILE
        shutil.copyfile(memory_summary_path, memory_path)
        aliases_removed = sanitize_memory_summary(memory_path)
    summary_path: Path | None = None
    if document_summary_path is not None and Path(document_summary_path).is_file():
        summary_path = state_dir / SUMMARY_FILE
        shutil.copyfile(document_summary_path, summary_path)
    return ReadableState(
        memory_path=memory_path,
        history_dir=history_dir,
        levels=tuple(sorted(set(levels))),
        summary_path=summary_path,
        aliases_removed=aliases_removed,
    )


def generate_document_summary(
    *,
    profile: PromptProfile,
    canonicalized_chunks: list[Any],
    backend: Any,
    document_id: str,
    out_dir: Path,
    max_tokens: int = 384,
    temperature: float = 0.2,
) -> Path:
    """Summarize every L0 chunk in one call with the profile's document-summary prompt."""

    if not profile.document_summary_prompt:
        raise ValueError(
            f"Prompt profile {profile.name!r} has no document_summary.txt; add one to the profile folder "
            "to enable whole-document summaries."
        )
    ordered = sorted(canonicalized_chunks, key=lambda chunk: (int(chunk.chunk_index), str(chunk.chunk_id)))
    items = [str(getattr(chunk, "canonicalized_chunk_text", "") or "").strip() for chunk in ordered]
    items = [item for item in items if item]
    if not items:
        raise ValueError("There are no chunks to summarize; index a document first")

    user_prompt = (
        "[ROLLUP_META]\n"
        + json.dumps({"scope": "full_document", "document_id": document_id, "input_level": "L0", "items": len(items)}, ensure_ascii=False)
        + "\n\n[ROLLUP_ITEMS_JSON]\n"
        + json.dumps(items, ensure_ascii=False)
        + "\n\n[INSTRUCTIONS]\n"
        + "Synthesize all items into one complete summary. Include important information from the final item. "
        + "Output one plain-text paragraph, then [END_SUMMARY]."
    )
    chat_prompt = ChatPrompt(system=profile.document_summary_prompt, user=user_prompt)
    started = time.perf_counter()
    raw_output = backend.generate_chat_batch(
        [chat_prompt], max_tokens=int(max_tokens), temperature=float(temperature), stop=["[END_SUMMARY]", "<|im_end|>", "<|endoftext|>"]
    )[0]
    latency_ms = (time.perf_counter() - started) * 1000.0
    summary = str(raw_output or "").strip()
    if "[END_SUMMARY]" in summary:
        summary = summary.split("[END_SUMMARY]", 1)[0].strip()
    if not summary:
        raise RuntimeError("The document summary came back empty")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.txt"
    (out_dir / "prompt.txt").write_text(render_prompt(backend, chat_prompt), encoding="utf-8")
    summary_path.write_text(summary + "\n", encoding="utf-8")
    write_json(
        out_dir / "generation_record.json",
        {
            "status": "completed",
            "document_id": document_id,
            "prompt_profile": profile.name,
            "prompt_content_hash": profile.content_hash,
            "input_items": len(items),
            "input_chars": sum(len(item) for item in items),
            "summary_words": len(summary.split()),
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "generation_latency_ms": latency_ms,
            "prompt_hash": prompt_hash(profile.document_summary_prompt, user_prompt),
        },
    )
    return summary_path


__all__ = [
    "DEFAULT_FORBIDDEN_ALIASES",
    "HISTORY_DIR",
    "MEMORY_FILE",
    "ReadableState",
    "SUMMARY_FILE",
    "export_readable_state",
    "find_cache_index",
    "generate_document_summary",
    "sanitize_memory_summary",
    "write_json",
]
