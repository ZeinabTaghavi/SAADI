"""Package version, read from the installed metadata."""

from __future__ import annotations

try:
    from importlib.metadata import version as _installed_version

    __version__ = _installed_version("saadi")
except Exception:  # pragma: no cover - source checkout without metadata
    __version__ = "0.1.0"
