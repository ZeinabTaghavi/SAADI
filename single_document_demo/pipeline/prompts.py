from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MemoryPromptProfile:
    name: str
    directory: Path
    entity_extraction_prompt: str
    hierarchy_summary_prompt: str
    document_summary_prompt: str | None
    content_hash: str


def load_prompt_profile(prompt_root: str | Path, profile: str | Path) -> MemoryPromptProfile:
    root = Path(prompt_root).expanduser()
    requested = Path(profile).expanduser()
    directory = requested if requested.is_dir() else root / requested
    if not directory.is_dir():
        raise FileNotFoundError(f"Memory prompt profile does not exist: {directory}")

    entity_path = directory / "entity_extraction.txt"
    hierarchy_path = directory / "hierarchy_summary.txt"
    missing = [str(path) for path in (entity_path, hierarchy_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Prompt profile is missing required files: " + ", ".join(missing))

    entity_prompt = entity_path.read_text(encoding="utf-8").strip()
    hierarchy_prompt = hierarchy_path.read_text(encoding="utf-8").strip()
    document_summary_path = directory / "document_summary.txt"
    document_summary_prompt = (
        document_summary_path.read_text(encoding="utf-8").strip()
        if document_summary_path.is_file()
        else None
    )
    if not entity_prompt or not hierarchy_prompt:
        raise ValueError(f"Prompt profile contains an empty prompt file: {directory}")
    if document_summary_path.is_file() and not document_summary_prompt:
        raise ValueError(f"Prompt profile contains an empty prompt file: {document_summary_path}")

    digest = hashlib.sha256(
        (
            "entity\0"
            + entity_prompt
            + "\0summary\0"
            + hierarchy_prompt
            + "\0document_summary\0"
            + str(document_summary_prompt or "")
        ).encode("utf-8")
    ).hexdigest()
    return MemoryPromptProfile(
        name=directory.name,
        directory=directory.resolve(),
        entity_extraction_prompt=entity_prompt,
        hierarchy_summary_prompt=hierarchy_prompt,
        document_summary_prompt=document_summary_prompt,
        content_hash=digest,
    )
