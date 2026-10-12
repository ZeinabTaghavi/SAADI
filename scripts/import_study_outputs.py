#!/usr/bin/env python3
"""Copy archived Saadi outputs and optionally full construction caches into this release."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from saadi.reporting.output_sources import import_outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="Original outputs directory containing the four dataset ablation folders")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--reference", type=Path, default=ROOT / "outputs/sadi/reference")
    parser.add_argument("--include-caches", action="store_true", help="Also copy Saadi entity memory and hierarchy construction artifacts")
    args = parser.parse_args()
    catalog = import_outputs(args.source_root, args.output_root, args.reference, include_caches=args.include_caches)
    print(f"Imported {catalog['total_files']} Saadi files ({catalog['total_bytes'] / 1e9:.2f} GB).")
