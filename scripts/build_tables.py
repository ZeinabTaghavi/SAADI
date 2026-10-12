#!/usr/bin/env python3
"""Build final paper tables from Saadi outputs and saved comparison-method CSVs."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from saadi.reporting.release_tables import build_release_tables, ALL_TABLES

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT / "outputs/sadi/reference")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/tables")
    parser.add_argument("--outputs-root", type=Path, default=ROOT / "outputs", help="Restored full experiment outputs (optional separate archive)")
    parser.add_argument("--input-source", choices=["auto", "outputs", "reference"], default="auto")
    parser.add_argument("--study", action="append", choices=list(ALL_TABLES), help="Generate one table (repeatable); default: all")
    parser.add_argument("--other-methods-root", type=Path, help="Default: outputs-root/other_methods")
    parser.add_argument("--comparison-inputs", type=Path, help="Default: outputs-root/sadi/comparison_inputs")
    parser.add_argument("--figures", action="store_true", help="Also produce the HR-versus-depth figure (requires matplotlib)")
    args = parser.parse_args()
    print(json.dumps(build_release_tables(args.reference, args.output_dir, outputs_root=args.outputs_root,
        input_mode=args.input_source, studies=args.study, figures=args.figures,
        other_methods_root=args.other_methods_root, comparison_root=args.comparison_inputs), indent=2))
