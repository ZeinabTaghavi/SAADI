"""Shared progress-bar helpers for long-running local pipelines."""

from __future__ import annotations

import os
import sys

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None


def progress_enabled() -> bool:
    raw = str(os.getenv("SAADI_PROGRESS", "1")).strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    return sys.stderr.isatty()


def new_progress_bar(*, total: int, desc: str, unit: str, leave: bool = False) -> object | None:
    if _tqdm is None or not progress_enabled():
        return None
    return _tqdm(total=total, desc=desc, unit=unit, leave=leave, dynamic_ncols=True)

