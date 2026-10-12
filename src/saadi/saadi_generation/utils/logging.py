"""Logging setup for standalone SAADI generation scripts."""

from __future__ import annotations

import logging
import os


def configure_logging() -> None:
    level_name = str(os.getenv("SAADI_LOG_LEVEL", "INFO")).strip().upper()
    level = getattr(logging, level_name, logging.INFO)
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

