from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_FORBIDDEN_ALIASES = frozenset(
    {
        "i",
        "me",
        "my",
        "mine",
        "we",
        "us",
        "our",
        "ours",
        "you",
        "your",
        "yours",
        "he",
        "him",
        "his",
        "she",
        "her",
        "hers",
        "it",
        "its",
        "they",
        "them",
        "their",
        "theirs",
        "patient",
        "therapist",
    }
)


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def sanitize_memory_summary(
    path: Path | None,
    *,
    forbidden_aliases: Iterable[str] = DEFAULT_FORBIDDEN_ALIASES,
) -> int:
    """Remove role labels and pronouns from the public memory export.

    Prompt instructions remain the primary control. This deterministic export
    guard prevents occasional LLM violations from leaking into downstream use.
    """

    if path is None or not path.is_file():
        return 0
    forbidden = {str(value).strip().casefold() for value in forbidden_aliases}
    payload = json.loads(path.read_text(encoding="utf-8"))
    removed = 0
    for entity in payload.get("entities", []):
        aliases = entity.get("aliases", []) or []
        kept = []
        for alias in aliases:
            text = alias.get("text", "") if isinstance(alias, dict) else str(alias)
            if str(text).strip().casefold() in forbidden:
                removed += 1
            else:
                kept.append(alias)
        entity["aliases"] = kept
    payload["export_sanitization"] = {
        "removed_forbidden_aliases": removed,
        "rule": "role labels and standalone pronouns",
    }
    write_json(path, payload)
    return removed


def attach_cleaned_outputs(run_manifest: Path, cleaned_manifest: Path | None) -> None:
    if cleaned_manifest is None or not run_manifest.is_file():
        return
    payload = json.loads(run_manifest.read_text(encoding="utf-8"))
    cleaned_payload = json.loads(cleaned_manifest.read_text(encoding="utf-8"))
    artifacts = payload.setdefault("artifacts", {})
    artifacts["cleaned_outputs_manifest_json"] = str(cleaned_manifest)
    artifacts["cleaned_memory_summary_json"] = cleaned_payload.get(
        "memory_summary_json"
    )
    artifacts["cleaned_document_summary_txt"] = cleaned_payload.get(
        "document_summary_txt"
    )
    payload.setdefault("summary", {})["hierarchy_levels"] = cleaned_payload.get(
        "levels_available", []
    )
    write_json(run_manifest, payload)


def attach_document_summary(run_manifest: Path, summary_path: Path | None) -> None:
    if summary_path is None or not run_manifest.is_file():
        return
    payload = json.loads(run_manifest.read_text(encoding="utf-8"))
    payload.setdefault("artifacts", {})["document_summary_txt"] = str(summary_path)
    payload["artifacts"]["document_summary_generation_record_json"] = str(
        summary_path.parent / "generation_record.json"
    )
    payload.setdefault("summary", {})["document_summary_words"] = len(
        summary_path.read_text(encoding="utf-8").split()
    )
    write_json(run_manifest, payload)
