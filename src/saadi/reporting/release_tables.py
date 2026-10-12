"""Build the public table collection from Saadi and comparison-method outputs."""
import json
from pathlib import Path

from .layout import TABLES
from .tables import build_tables
from .comparison_tables import COMPARISON_TABLES, build_comparison_tables

ALL_TABLES = {**TABLES, **COMPARISON_TABLES}


def build_release_tables(reference, output, *, outputs_root, input_mode="auto", studies=None,
                         figures=False, other_methods_root=None, comparison_root=None):
    selected = set(ALL_TABLES if studies is None else studies)
    if not selected or selected - set(ALL_TABLES):
        raise ValueError("Unknown or empty table selection")
    report = {}
    sadi = selected & set(TABLES)
    if sadi:
        report["sadi_studies"] = build_tables(reference, output, outputs_root=outputs_root, input_mode=input_mode,
                                               studies=None if sadi == set(TABLES) else sadi, figures=figures)
    comparison = selected & set(COMPARISON_TABLES)
    if comparison:
        report["final_tables"] = build_comparison_tables(reference, output, outputs_root=outputs_root, input_mode=input_mode,
            studies=comparison, other_methods_root=other_methods_root, comparison_root=comparison_root)
    if studies is None:
        output = Path(output)
        links = [f"- [{title}]({path}/README.md)" for path, title in COMPARISON_TABLES.values()]
        study_links = [f"- [{title}]({path}/README.md)" for path, title in TABLES.values()]
        (output / "README.md").write_text("# Tables\n\n"
            "Regenerate all tables with `python scripts/build_tables.py` from the repository root. "
            "Each folder also contains a standalone `generate.py`. Inputs live under `outputs/`; "
            "this directory contains generated tables, code, and provenance.\n\n"
            "## Final comparisons\n\nSaadi measurements are joined to saved CSVs in `outputs/other_methods/`. "
            "No other method is executed.\n\n" + "\n".join(links) + "\n\n## Saadi studies\n\n" + "\n".join(study_links) + "\n\n"
            "See [the paper audit](../../results/paper_consistency/README.md) for the cell-by-cell comparison with the paper. "
            "The Saadi-only [combined metrics](combined/all_metrics.csv) retain all saved study configurations.\n")
        (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
