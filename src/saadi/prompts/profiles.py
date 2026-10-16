"""Prompt profiles: domain guidance for entity extraction and summaries.

A profile is a folder with ``entity_extraction.txt`` and ``hierarchy_summary.txt``
and, optionally, ``document_summary.txt``. Each file holds guidance only; the
library adds the fixed input and output contract (see :mod:`saadi.prompts.contract`).
Two profiles ship with the package: ``default`` (the paper's prompts) and
``conversation`` (role-labelled transcripts, with the therapy prompts as the
example; ``therapy`` is accepted as an alias).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from .contract import (
    CONTRACT_VERSION,
    compose_document_summary_prompt,
    compose_entity_system_prompt,
    compose_summary_system_prompt,
    validate_entity_guidance,
    validate_summary_guidance,
)

ENTITY_GUIDANCE_FILE = "entity_extraction.txt"
SUMMARY_GUIDANCE_FILE = "hierarchy_summary.txt"
DOCUMENT_SUMMARY_GUIDANCE_FILE = "document_summary.txt"
BUILTIN_PROFILES = ("default", "conversation")
PROFILE_ALIASES = {"therapy": "conversation"}


def builtin_profiles_root() -> Path:
    """Directory holding the packaged profiles."""

    return Path(str(resources.files("saadi.prompts").joinpath("profiles")))


def builtin_profile_names() -> list[str]:
    return sorted(BUILTIN_PROFILES)


@dataclass(frozen=True)
class PromptProfile:
    """Guidance for one domain plus the composed prompts the pipeline sends."""

    name: str
    directory: Path
    entity_guidance: str
    summary_guidance: str
    document_summary_guidance: str | None = None
    contract_version: str = CONTRACT_VERSION

    @property
    def entity_system_prompt(self) -> str:
        return compose_entity_system_prompt(self.entity_guidance)

    @property
    def summary_system_prompt(self) -> str:
        return compose_summary_system_prompt(self.summary_guidance)

    @property
    def document_summary_prompt(self) -> str | None:
        if not self.document_summary_guidance:
            return None
        return compose_document_summary_prompt(self.document_summary_guidance)

    @property
    def content_hash(self) -> str:
        """Hash of the prompts the model actually sees, so provenance follows the composed text."""

        digest = hashlib.sha256()
        for part in (
            "contract",
            self.contract_version,
            "entity",
            self.entity_system_prompt,
            "summary",
            self.summary_system_prompt,
            "document_summary",
            self.document_summary_prompt or "",
        ):
            digest.update(part.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "directory": str(self.directory),
            "content_hash": self.content_hash,
            "contract_version": self.contract_version,
            "has_document_summary": self.document_summary_guidance is not None,
        }


def resolve_profile_directory(spec: str | Path) -> Path:
    """Map a built-in name, alias, or directory path to the profile folder."""

    text = str(spec).strip()
    key = PROFILE_ALIASES.get(text.lower(), text.lower())
    if key in BUILTIN_PROFILES:
        return builtin_profiles_root() / key
    path = Path(text).expanduser()
    if path.is_dir():
        return path.resolve()
    raise ValueError(
        f"Unknown prompt profile {spec!r}. Built-in profiles: {', '.join(builtin_profile_names())} "
        f"(alias: {', '.join(f'{a} -> {b}' for a, b in PROFILE_ALIASES.items())}); otherwise pass a directory "
        f"containing {ENTITY_GUIDANCE_FILE} and {SUMMARY_GUIDANCE_FILE}."
    )


def _read_guidance(path: Path, *, required: bool) -> str | None:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Prompt profile is missing {path.name}: {path}")
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Prompt profile file is empty: {path}")
    return text


def _reject_contract_phrases(markers: list[str], *, path: Path) -> None:
    if markers:
        raise ValueError(
            f"{path} contains {', '.join(repr(m) for m in markers)}. The output contract is fixed by the "
            "library and added to every prompt; keep profile files to domain guidance."
        )


def load_prompt_profile(spec: str | Path | PromptProfile) -> PromptProfile:
    """Load a profile by built-in name, alias, or directory. A profile object passes through."""

    if isinstance(spec, PromptProfile):
        return spec
    directory = resolve_profile_directory(spec)
    entity_path = directory / ENTITY_GUIDANCE_FILE
    summary_path = directory / SUMMARY_GUIDANCE_FILE
    document_path = directory / DOCUMENT_SUMMARY_GUIDANCE_FILE
    entity_guidance = _read_guidance(entity_path, required=True) or ""
    summary_guidance = _read_guidance(summary_path, required=True) or ""
    document_guidance = _read_guidance(document_path, required=False)
    _reject_contract_phrases(validate_entity_guidance(entity_guidance), path=entity_path)
    _reject_contract_phrases(validate_summary_guidance(summary_guidance), path=summary_path)
    if document_guidance is not None:
        _reject_contract_phrases(validate_summary_guidance(document_guidance), path=document_path)
    return PromptProfile(
        name=directory.name,
        directory=directory,
        entity_guidance=entity_guidance,
        summary_guidance=summary_guidance,
        document_summary_guidance=document_guidance,
    )


def apply_profile_to_strategy_kwargs(profile: PromptProfile, strategy_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    """Return strategy keyword arguments with the profile's prompts filled in.

    Explicit ``system_prompt`` or ``rollup_system_prompt`` values in the mapping
    win, so an experiment can still pin a literal prompt.
    """

    merged = dict(strategy_kwargs or {})
    merged.setdefault("system_prompt", profile.entity_system_prompt)
    merged.setdefault("rollup_system_prompt", profile.summary_system_prompt)
    return merged


def profile_cache_suffix(profile: PromptProfile) -> str:
    """Cache-namespace component so constructions with different prompts never share a cache."""

    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in profile.name) or "profile"
    return f"prompt_{safe}_{profile.content_hash[:10]}"


__all__ = [
    "BUILTIN_PROFILES",
    "DOCUMENT_SUMMARY_GUIDANCE_FILE",
    "ENTITY_GUIDANCE_FILE",
    "PROFILE_ALIASES",
    "PromptProfile",
    "SUMMARY_GUIDANCE_FILE",
    "apply_profile_to_strategy_kwargs",
    "builtin_profile_names",
    "builtin_profiles_root",
    "load_prompt_profile",
    "profile_cache_suffix",
    "resolve_profile_directory",
]
