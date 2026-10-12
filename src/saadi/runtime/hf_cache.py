"""Centralized Hugging Face cache environment setup."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


DEFAULT_CACHE_ROOT = ".cache"


def resolve_cache_root(raw_root: str | os.PathLike[str] | None = None) -> Path:
    """Resolve cache root from explicit arg or SAADI_CACHE_ROOT env."""
    raw = str(raw_root).strip() if raw_root is not None else ""
    if not raw:
        raw = str(os.getenv("SADI_CACHE_DIR", os.getenv("SAADI_CACHE_ROOT", DEFAULT_CACHE_ROOT))).strip() or DEFAULT_CACHE_ROOT
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def _set_env_default(name: str, value: str, *, force: bool) -> str:
    current = os.getenv(name)
    if force or current is None or not str(current).strip():
        os.environ[name] = value
        return value
    return str(current)


def _ensure_dir(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as exc:

        logger.warning("Could not create cache directory %s: %s", path, exc)


def configure_hf_cache_env(
    cache_root: str | os.PathLike[str] | None = None,
    *,
    force: bool = False,
) -> dict[str, str]:
    """
    Configure Hugging Face cache env vars to one root directory.

    If vars are already set, they are preserved unless force=True.
    Returns the effective env mapping.
    """
    root = resolve_cache_root(cache_root)
    saadi_root = _set_env_default("SAADI_CACHE_ROOT", str(root), force=force)
    hf_home_default = str(Path(saadi_root) / "huggingface")
    hf_home = _set_env_default("HF_HOME", hf_home_default, force=force)
    hf_hub = _set_env_default("HF_HUB_CACHE", str(Path(hf_home) / "hub"), force=force)
    hf_datasets = _set_env_default("HF_DATASETS_CACHE", str(Path(hf_home) / "datasets"), force=force)
    hf_assets = _set_env_default("HF_ASSETS_CACHE", str(Path(hf_home) / "assets"), force=force)
    transformers_cache = _set_env_default("TRANSFORMERS_CACHE", str(Path(hf_home) / "transformers"), force=force)

    _set_env_default("HUGGINGFACE_HUB_CACHE", hf_hub, force=force)

    for p in (
        Path(saadi_root),
        Path(hf_home),
        Path(hf_hub),
        Path(hf_datasets),
        Path(hf_assets),
        Path(transformers_cache),
    ):
        _ensure_dir(p)

    return {
        "SAADI_CACHE_ROOT": saadi_root,
        "HF_HOME": hf_home,
        "HF_HUB_CACHE": hf_hub,
        "HF_DATASETS_CACHE": hf_datasets,
        "HF_ASSETS_CACHE": hf_assets,
        "TRANSFORMERS_CACHE": transformers_cache,
    }


def configure_hf_cache_from_config(config: dict[str, Any] | None) -> dict[str, str]:
    """
    Configure HF cache using runtime config if present.

    Supports:
    runtime:
      cache_root: /path/to/cache
    """
    cfg = config or {}
    runtime_cfg = cfg.get("runtime", {}) if isinstance(cfg, dict) else {}
    if not isinstance(runtime_cfg, dict):
        runtime_cfg = {}
    cache_root = runtime_cfg.get("cache_root")
    if cache_root is None or not str(cache_root).strip():
        return configure_hf_cache_env()
    return configure_hf_cache_env(str(cache_root), force=True)
