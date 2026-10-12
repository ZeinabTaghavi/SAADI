#!/usr/bin/env python3
"""Validate saved comparison CSVs; optionally register intentional value edits."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from saadi.reporting.comparison_inputs import load_other_methods

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--other-methods-root", type=Path, default=ROOT / "outputs/other_methods")
    parser.add_argument("--update-checksums", action="store_true", help="Explicitly accept edited, schema-valid CSV values; update source provenance first")
    args = parser.parse_args()
    data = load_other_methods(args.other_methods_root, update_checksums=args.update_checksums)
    print(json.dumps({"retrieval_rows": len(data["retrieval"]), "resource_rows": len(data["resources"]),
                      "files": len(data["evidence"]), "checksums_updated": args.update_checksums}, indent=2))
