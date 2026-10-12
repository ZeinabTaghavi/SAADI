"""Builds and writes run manifests, including git metadata when available."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from ..core.types import ExperimentManifest
from ..llm.base import (
    DEFAULT_GENERATION_SEED,
    DEFAULT_GENERATION_TOP_K,
    DEFAULT_GENERATION_TOP_P,
    DEFAULT_REPETITION_PENALTY,
)


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except Exception:
        return None


def git_commit(cwd: str | None = None) -> str | None:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=cwd)
        return res.stdout.strip()
    except Exception:
        return None


GIT_PROVENANCE_FIELDS = (
    "git_commit",
    "git_dirty",
    "git_worktree_diff_sha256",
)
_GIT_PROVENANCE_ENV = "SAADI_GIT_PROVENANCE_JSON"


def _environment_git_provenance() -> dict[str, Any] | None:
    """Read a validated provenance snapshot pinned by a parent runner."""
    raw = os.environ.get(_GIT_PROVENANCE_ENV)
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except Exception:
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("git_dirty"), bool):
        return None
    fingerprint = str(payload.get("git_worktree_diff_sha256", "") or "").lower()
    if len(fingerprint) != 64 or any(ch not in "0123456789abcdef" for ch in fingerprint):
        return None
    commit_raw = payload.get("git_commit")
    commit = str(commit_raw).strip() if commit_raw is not None else None
    return {
        "git_commit": commit or None,
        "git_dirty": bool(payload["git_dirty"]),
        "git_worktree_diff_sha256": fingerprint,
    }


def _hash_untracked_path(
    digest: "hashlib._Hash",
    *,
    repository_root: Path,
    relative_path_bytes: bytes,
) -> None:
    """Add one untracked path, its executable/type bits, and contents to a hash."""
    relative_path = os.fsdecode(relative_path_bytes)
    path = repository_root / relative_path
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode):
        kind = b"symlink"
        mode = b"120000"
        content = os.fsencode(os.readlink(path))
    elif stat.S_ISREG(info.st_mode):
        kind = b"file"
        mode = b"100755" if info.st_mode & 0o111 else b"100644"
        content = None
    else:
        # Git normally reports files rather than directories here. Preserve a
        # deterministic identity for any unusual filesystem entry without
        # attempting to read a device/socket/FIFO.
        kind = b"special"
        mode = f"{stat.S_IFMT(info.st_mode):o}".encode("ascii")
        content = b""

    digest.update(b"untracked\0")
    digest.update(len(relative_path_bytes).to_bytes(8, "big"))
    digest.update(relative_path_bytes)
    digest.update(len(kind).to_bytes(8, "big"))
    digest.update(kind)
    digest.update(len(mode).to_bytes(8, "big"))
    digest.update(mode)
    if content is not None:
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
        return

    digest.update(int(info.st_size).to_bytes(8, "big"))
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)


def git_provenance(
    cwd: str | None = None,
    *,
    allow_environment: bool = True,
) -> dict[str, Any]:
    """Return commit plus a reproducible fingerprint of runtime worktree changes.

    The fingerprint hashes the binary/full-index diff from ``HEAD`` to the
    working files (therefore covering staged and unstaged tracked-file content)
    followed by every non-ignored untracked path, type/mode, and byte content in
    Git's deterministic path order. Ignored files are deliberately absent, just
    as they are from the normal Git worktree status.

    A parent experiment runner may pin a snapshot in the environment so child
    ingestion/retrieval processes all report the same pre-run code identity even
    after they have created output artifacts.
    """
    if allow_environment:
        inherited = _environment_git_provenance()
        if inherited is not None:
            return inherited

    commit = git_commit(cwd=cwd)
    try:
        root_result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
            cwd=cwd,
        )
        repository_root = Path(root_result.stdout.strip())
        tracked_result = subprocess.run(
            [
                "git",
                "diff",
                "--binary",
                "--full-index",
                "--no-ext-diff",
                "--no-textconv",
                "HEAD",
                "--",
                ".",
            ],
            capture_output=True,
            check=True,
            cwd=repository_root,
        )
        untracked_result = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z", "--", "."],
            capture_output=True,
            check=True,
            cwd=repository_root,
        )
        tracked_diff = bytes(tracked_result.stdout)
        untracked_paths = sorted(
            path for path in bytes(untracked_result.stdout).split(b"\0") if path
        )

        digest = hashlib.sha256()
        digest.update(b"saadi-git-worktree-diff-v1\0")
        digest.update(b"tracked\0")
        digest.update(len(tracked_diff).to_bytes(8, "big"))
        digest.update(tracked_diff)
        for relative_path_bytes in untracked_paths:
            _hash_untracked_path(
                digest,
                repository_root=repository_root,
                relative_path_bytes=relative_path_bytes,
            )
        return {
            "git_commit": commit,
            "git_dirty": bool(tracked_diff or untracked_paths),
            "git_worktree_diff_sha256": digest.hexdigest(),
        }
    except Exception:
        # Preserve the historical best-effort behavior outside a Git checkout.
        return {
            "git_commit": commit,
            "git_dirty": None,
            "git_worktree_diff_sha256": None,
        }


def pin_git_provenance_environment(
    provenance: dict[str, Any] | None = None,
    *,
    cwd: str | None = None,
) -> dict[str, Any]:
    """Pin one provenance snapshot for this process and all child processes."""
    resolved = dict(provenance or git_provenance(cwd=cwd, allow_environment=False))
    if resolved.get("git_dirty") is not None and resolved.get("git_worktree_diff_sha256"):
        os.environ[_GIT_PROVENANCE_ENV] = json.dumps(
            {field: resolved.get(field) for field in GIT_PROVENANCE_FIELDS},
            sort_keys=True,
            separators=(",", ":"),
        )
    return resolved


def write_manifest(path: str | Path, manifest: ExperimentManifest) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, ensure_ascii=False, indent=2)
    return str(p)


def build_reproducibility_metadata(
    *,
    config: dict[str, Any],
    backend: object | None = None,
    generate: object | None = None,
    cwd: str | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the resolved model, decoding, and code identity for a run."""

    model_cfg = config.get("model", {}) or {}
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    generate_cfg = model_cfg.get("generate", {}) or {}
    if not isinstance(generate_cfg, dict):
        generate_cfg = {}
    sampling_cfg = model_cfg.get("sampling", {}) or {}
    if not isinstance(sampling_cfg, dict):
        sampling_cfg = {}

    runtime = getattr(backend, "runtime_metadata", {}) if backend is not None else {}
    if not isinstance(runtime, dict):
        runtime = {}
    runtime_sampling = runtime.get("sampling", {}) or {}
    if not isinstance(runtime_sampling, dict):
        runtime_sampling = {}

    sampling: dict[str, Any] = {
        "temperature": float(
            getattr(generate, "temperature", generate_cfg.get("temperature", 0.0))
        ),
        "seed": DEFAULT_GENERATION_SEED,
        "top_p": DEFAULT_GENERATION_TOP_P,
        "top_k": DEFAULT_GENERATION_TOP_K,
        "repetition_penalty": DEFAULT_REPETITION_PENALTY,
    }
    sampling.update(sampling_cfg)
    sampling.update(runtime_sampling)
    # Temperature and the output limit are supplied per call, so those values
    # take precedence over vLLM SamplingParams defaults.
    sampling["temperature"] = float(
        getattr(generate, "temperature", generate_cfg.get("temperature", 0.0))
    )
    sampling["seed"] = int(sampling.get("seed", DEFAULT_GENERATION_SEED))
    sampling["top_p"] = float(sampling.get("top_p", DEFAULT_GENERATION_TOP_P))
    sampling["top_k"] = int(sampling.get("top_k", DEFAULT_GENERATION_TOP_K))
    sampling["top_k_enabled"] = sampling["top_k"] >= 0
    sampling["repetition_penalty"] = float(
        sampling.get("repetition_penalty", DEFAULT_REPETITION_PENALTY)
    )

    model_name = runtime.get("model_name") or model_cfg.get("model_id") or model_cfg.get("name")
    configured_revision = (
        model_cfg.get("revision")
        or model_cfg.get("model_revision")
        or model_cfg.get("hf_revision")
    )
    max_tokens_raw = (
        getattr(generate, "max_tokens")
        if generate is not None and hasattr(generate, "max_tokens")
        else generate_cfg.get("max_tokens")
    )
    max_tokens = int(max_tokens_raw) if max_tokens_raw is not None else None
    code_provenance = dict(provenance or git_provenance(cwd=cwd))
    return {
        **{field: code_provenance.get(field) for field in GIT_PROVENANCE_FIELDS},
        "model_name": (str(model_name) if model_name is not None else None),
        "hf_revision": str(runtime.get("hf_revision") or configured_revision or "main"),
        "hf_commit_hash": (
            str(runtime["hf_commit_hash"])
            if runtime.get("hf_commit_hash") is not None
            else None
        ),
        "dtype": (
            str(runtime.get("dtype") or model_cfg.get("dtype"))
            if runtime.get("dtype") is not None or model_cfg.get("dtype") is not None
            else None
        ),
        "tensor_parallel_size": (
            int(runtime.get("tensor_parallel_size") or model_cfg.get("tensor_parallel_size"))
            if runtime.get("tensor_parallel_size") is not None or model_cfg.get("tensor_parallel_size") is not None
            else None
        ),
        "vllm_version": (
            str(runtime["vllm_version"])
            if runtime.get("vllm_version") is not None
            else _package_version("vllm")
        ),
        "max_tokens": max_tokens,
        "sampling": sampling,
    }


def build_manifest(
    *,
    config: dict[str, Any],
    artifacts: dict[str, str],
    metadata: dict[str, Any] | None = None,
    cwd: str | None = None,
    backend: object | None = None,
    generate: object | None = None,
    provenance: dict[str, Any] | None = None,
) -> ExperimentManifest:
    md = dict(metadata or {})
    code_provenance = dict(provenance or git_provenance(cwd=cwd))
    for field in GIT_PROVENANCE_FIELDS:
        md.setdefault(field, code_provenance.get(field))
        code_provenance[field] = md.get(field)
    md.setdefault(
        "reproducibility",
        build_reproducibility_metadata(
            config=config,
            backend=backend,
            generate=generate,
            cwd=cwd,
            provenance=code_provenance,
        ),
    )
    return ExperimentManifest.now(config=config, artifacts=artifacts, metadata=md)
