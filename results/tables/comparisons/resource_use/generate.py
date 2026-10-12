#!/usr/bin/env python3
"""Regenerate this final paper table from saved outputs."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "pyproject.toml").is_file() and (p / "src/saadi").is_dir())
sys.path.insert(0, str(ROOT / "src"))
from saadi.reporting.cli import comparison_main
if __name__ == "__main__":
    raise SystemExit(comparison_main('resource_use', table_dir=HERE, project_root=ROOT))
