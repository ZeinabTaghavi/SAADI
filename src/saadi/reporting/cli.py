"""Shared command-line interface for the generator beside each Saadi table."""
import argparse
import json
from pathlib import Path

from .tables import build_tables
from .layout import TABLES


def table_main(study, *, table_dir, project_root):
    parser = argparse.ArgumentParser(description=TABLES[study][1])
    parser.add_argument("--reference", type=Path, default=project_root / "outputs/sadi/reference" if project_root else None)
    parser.add_argument("--outputs-root", type=Path, default=project_root / "outputs" if project_root else None)
    parser.add_argument("--input-source", choices=["auto", "outputs", "reference"], default="auto")
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args()
    if args.reference is None:
        parser.error("Pass --reference or run this generator inside the Saadi repository")
    # The public layout is category/study, two levels below the table root.
    report = build_tables(args.reference, Path(table_dir).parents[1], outputs_root=args.outputs_root,
                          input_mode=args.input_source, studies=[study], figures=args.figures)
    print(json.dumps(report, indent=2))
    return 0


def comparison_main(study, *, table_dir, project_root):
    from .comparison_tables import COMPARISON_TABLES, build_comparison_tables
    parser = argparse.ArgumentParser(description=COMPARISON_TABLES[study][1])
    parser.add_argument("--reference", type=Path, default=project_root / "outputs/sadi/reference")
    parser.add_argument("--outputs-root", type=Path, default=project_root / "outputs")
    parser.add_argument("--other-methods-root", type=Path)
    parser.add_argument("--comparison-inputs", type=Path)
    parser.add_argument("--input-source", choices=["auto", "outputs", "reference"], default="auto")
    args = parser.parse_args()
    report = build_comparison_tables(args.reference, Path(table_dir).parents[1], outputs_root=args.outputs_root,
        studies=[study], input_mode=args.input_source, other_methods_root=args.other_methods_root,
        comparison_root=args.comparison_inputs)
    print(json.dumps(report, indent=2))
    return 0
