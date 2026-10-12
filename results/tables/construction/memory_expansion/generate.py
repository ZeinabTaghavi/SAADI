#!/usr/bin/env python3
"""Generate the Saadi memory expansion statistics table from verified study outputs."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = next((p for p in HERE.parents if (p / "pyproject.toml").is_file() and (p / "src/saadi").is_dir()), None)
if ROOT is not None:
    sys.path.insert(0, str(ROOT / "src"))
from saadi.reporting.cli import table_main

if __name__ == "__main__":
    raise SystemExit(table_main('active_entities_expansion_statistics', table_dir=HERE, project_root=ROOT))
