#!/usr/bin/env python3
"""Check generated Saadi tables against independently extracted paper values."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from saadi.reporting.paper_consistency import audit_paper


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-root", type=Path, default=ROOT / "results/tables")
    parser.add_argument("--specification-root", type=Path, default=ROOT / "results/paper_consistency")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/paper_consistency")
    parser.add_argument("--paper", type=Path, help="Optionally verify the supplied PDF's checksum")
    args = parser.parse_args()
    report = audit_paper(args.tables_root, args.specification_root, args.output_dir, paper=args.paper)
    print(json.dumps({key: report[key] for key in ("all_checked_cells_match", "checked_cells", "counts", "by_table")}, indent=2))
    raise SystemExit(0 if report["all_checked_cells_match"] else 1)
