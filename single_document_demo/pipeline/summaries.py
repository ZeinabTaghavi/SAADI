from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from saadi.core.prompts import prompt_hash
from saadi.llm.base import ChatPrompt, render_prompt

from .artifacts import sanitize_memory_summary, write_json
from .config import PromptedPipelineConfig
from .prompts import MemoryPromptProfile


@dataclass(frozen=True, slots=True)
class CleanedOutputs:
    directory: Path
    manifest_path: Path
    memory_summary_path: Path | None
    document_summary_path: Path | None
    hierarchy_levels: tuple[int, ...]
    aliases_removed: int


def _resolve_indexed_cache_file(
    index_path: Path, raw_path: str, doc_cache_dir: str
) -> Path | None:
    recorded = Path(str(raw_path or ""))
    if recorded.is_file():
        return recorded
    candidates = [
        index_path.parent / Path(str(doc_cache_dir or "")).name / recorded.name,
        index_path.parent / recorded.name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = sorted(index_path.parent.rglob(recorded.name)) if recorded.name else []
    return matches[0] if matches else None


def export_cleaned_outputs(
    ingest: Any,
    run_dir: Path,
    *,
    memory_summary_path: Path | None,
    document_summary_path: Path | None,
) -> CleanedOutputs | None:
    """Publish human-facing memory, summary, and hierarchy files together."""

    output_dir = run_dir / "cleaned_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    exported_levels: list[dict[str, Any]] = []
    index_path: Path | None = None
    index: dict[str, Any] = {}
    docs: list[Any] = []
    if ingest.artifacts and ingest.artifacts.llm_cache_index_json:
        candidate = Path(ingest.artifacts.llm_cache_index_json)
        if candidate.is_file():
            loaded_index = json.loads(candidate.read_text(encoding="utf-8"))
            if isinstance(loaded_index, dict):
                index_path = candidate
                index = loaded_index
                loaded_docs = index.get("docs", [])
                if isinstance(loaded_docs, list):
                    docs = loaded_docs

    branching_factor = int(
        index.get("hierarchy_branching_factor") or index.get("branching_factor") or 5
    )

    for doc in docs:
        if not isinstance(doc, dict):
            continue
        file_map = doc.get("files", {}) or {}
        level_items: list[tuple[int, str]] = []
        for key, raw_path in file_map.items():
            match = re.fullmatch(r"L_(\d+)", str(key))
            if match:
                level_items.append((int(match.group(1)), str(raw_path)))
        for level, raw_path in sorted(level_items):
            if index_path is None:
                continue
            source_path = _resolve_indexed_cache_file(
                index_path, raw_path, str(doc.get("doc_cache_dir", ""))
            )
            if source_path is None:
                continue
            rows: list[dict[str, Any]] = []
            for line in source_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                source_row = json.loads(line)
                step = int(source_row.get("step", len(rows)))
                span_size = branching_factor**level
                rows.append(
                    {
                        "doc_id": str(source_row.get("doc_id", doc.get("doc_id", ""))),
                        "level": level,
                        "item_index": len(rows),
                        "start_step": max(0, step - span_size + 1),
                        "end_step": step,
                        "span_size": span_size,
                        "text": str(source_row.get("chunk", "") or ""),
                        "kind": "canonicalized_chunk" if level == 0 else "hierarchy_summary",
                    }
                )
            jsonl_path = output_dir / f"L_{level}.jsonl"
            jsonl_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            text_path = output_dir / f"L_{level}.txt"
            text_path.write_text(
                "\n\n".join(
                    f"[L{level} item {row['item_index']} | steps "
                    f"{row['start_step']}-{row['end_step']}]\n{row['text']}"
                    for row in rows
                )
                + ("\n" if rows else ""),
                encoding="utf-8",
            )
            exported_levels.append(
                {
                    "level": level,
                    "kind": "canonicalized_chunk" if level == 0 else "hierarchy_summary",
                    "items": len(rows),
                    "span_size": branching_factor**level,
                    "jsonl_path": str(jsonl_path),
                    "text_path": str(text_path),
                    "source_cache_path": str(source_path),
                }
            )

    cleaned_memory_path: Path | None = None
    aliases_removed = 0
    if memory_summary_path is not None and memory_summary_path.is_file():
        cleaned_memory_path = output_dir / "memory_summary.json"
        shutil.copyfile(memory_summary_path, cleaned_memory_path)
        aliases_removed = sanitize_memory_summary(cleaned_memory_path)

    cleaned_document_summary_path: Path | None = None
    if document_summary_path is not None and document_summary_path.is_file():
        cleaned_document_summary_path = output_dir / "document_summary.txt"
        shutil.copyfile(document_summary_path, cleaned_document_summary_path)

    if not exported_levels and cleaned_memory_path is None and cleaned_document_summary_path is None:
        return None
    manifest_path = write_json(
        output_dir / "manifest.json",
        {
            "purpose": "Human-facing cleaned outputs for downstream use.",
            "cleaned_outputs_directory": str(output_dir),
            "memory_summary_json": str(cleaned_memory_path) if cleaned_memory_path else None,
            "document_summary_txt": (
                str(cleaned_document_summary_path)
                if cleaned_document_summary_path
                else None
            ),
            "source_cache_index": str(index_path) if index_path else None,
            "branching_factor": branching_factor,
            "levels_available": [item["level"] for item in exported_levels],
            "levels": exported_levels,
            "level_semantics": {
                "L0": "Canonicalized chunks; these are not generated roll-up summaries.",
                "L1_and_above": (
                    "Generated summaries for complete branching-factor groups. "
                    "A level may not cover trailing L0 chunks."
                ),
                "complete_document": (
                    "Use document_summary.txt when present; it always reads every L0 item."
                ),
            },
        },
    )
    return CleanedOutputs(
        directory=output_dir,
        manifest_path=manifest_path,
        memory_summary_path=cleaned_memory_path,
        document_summary_path=cleaned_document_summary_path,
        hierarchy_levels=tuple(item["level"] for item in exported_levels),
        aliases_removed=aliases_removed,
    )


def generate_document_summary(
    config: PromptedPipelineConfig,
    profile: MemoryPromptProfile,
    canonicalized_chunks: list[Any],
    *,
    backend: Any,
    run_dir: Path,
) -> Path | None:
    """Generate the therapy workflow's compact summary over every L0 chunk."""

    if not config.document_summary_enabled or not profile.document_summary_prompt:
        return None
    ordered_chunks = sorted(
        canonicalized_chunks,
        key=lambda chunk: (int(chunk.chunk_index), str(chunk.chunk_id)),
    )
    items = [str(chunk.canonicalized_chunk_text or "").strip() for chunk in ordered_chunks]
    items = [item for item in items if item]
    if not items:
        return None

    user_prompt = (
        "[ROLLUP_META]\n"
        + json.dumps(
            {
                "scope": "full_document",
                "document_id": config.document_id,
                "input_level": "L0",
                "items": len(items),
            },
            ensure_ascii=False,
        )
        + "\n\n[ROLLUP_ITEMS_JSON]\n"
        + json.dumps(items, ensure_ascii=False)
        + "\n\n[INSTRUCTIONS]\n"
        + "Synthesize all items into one complete session summary. Include important "
        + "information from the final item. Output one plain-text paragraph of at most "
        + "120 words, then [END_SUMMARY]."
    )
    chat_prompt = ChatPrompt(system=profile.document_summary_prompt, user=user_prompt)
    prompt = render_prompt(backend, chat_prompt)
    started = time.perf_counter()
    raw_output = backend.generate_chat_batch(
        [chat_prompt],
        max_tokens=int(config.document_summary_max_tokens),
        temperature=float(config.document_summary_temperature),
        stop=["[END_SUMMARY]", "<|im_end|>", "<|endoftext|>"],
    )[0]
    latency_ms = (time.perf_counter() - started) * 1000.0
    summary = str(raw_output or "").strip()
    if "[END_SUMMARY]" in summary:
        summary = summary.split("[END_SUMMARY]", 1)[0].strip()
    if not summary:
        raise RuntimeError("Full-session summary generation returned empty output")

    output_dir = run_dir / "document_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = output_dir / "prompt.txt"
    user_prompt_path = output_dir / "user_prompt.txt"
    summary_path = output_dir / "summary.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    user_prompt_path.write_text(user_prompt, encoding="utf-8")
    summary_path.write_text(summary + "\n", encoding="utf-8")
    write_json(
        output_dir / "generation_record.json",
        {
            "status": "completed",
            "document_id": config.document_id,
            "prompt_profile": profile.name,
            "prompt_content_hash": profile.content_hash,
            "input_level": "L0",
            "input_items": len(items),
            "input_chars": sum(len(item) for item in items),
            "summary_chars": len(summary),
            "summary_words": len(summary.split()),
            "max_tokens": int(config.document_summary_max_tokens),
            "temperature": float(config.document_summary_temperature),
            "generation_latency_ms": latency_ms,
            "prompt_hash": prompt_hash(profile.document_summary_prompt, user_prompt),
            "artifacts": {
                "prompt_txt": str(prompt_path),
                "user_prompt_txt": str(user_prompt_path),
                "summary_txt": str(summary_path),
            },
        },
    )
    return summary_path
