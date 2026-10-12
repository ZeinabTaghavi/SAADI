"""Compare generated numerical cells with independently recorded paper values."""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

from .output_sources import safe_path, sha256_file
from .tables import write_csv


def compare_cell(cell, rows):
    selector = json.loads(cell["row_selector"])
    matches = [row for row in rows if all(row.get(k) == str(v) for k, v in selector.items())]
    if not matches:
        return "missing_row", ""
    if len(matches) != 1:
        raise ValueError(f"Ambiguous table row: {cell['result_file']}: {selector}")
    row = matches[0]
    if cell["column"] not in row:
        return "missing_column", ""
    value = row[cell["column"]]
    if cell["decimals"] == "text":
        actual, expected = value, cell["paper_value"]
    else:
        digits = int(cell["decimals"])
        actual = f"{float(value) * float(cell['scale']):.{digits}f}"
        expected = f"{float(cell['paper_value']):.{digits}f}"
    return ("match" if actual == expected else "mismatch"), actual


def audit_paper(tables_root, specification_root, output_dir, *, paper=None):
    tables_root, specification_root, output_dir = map(Path, (tables_root, specification_root, output_dir))
    metadata = json.loads((specification_root / "paper.json").read_text())
    expected_path = specification_root / "expected_cells.csv"
    if sha256_file(expected_path) != metadata["expected_cells_sha256"]:
        raise ValueError("The paper-cell specification changed; review its extraction and provenance")
    if paper is not None and sha256_file(paper) != metadata["paper_sha256"]:
        raise ValueError("This is a different paper PDF; its tables need a new comparison specification")
    with expected_path.open(newline="") as stream:
        cells = list(csv.DictReader(stream))
    if Counter(cell["table"] for cell in cells) != metadata["counts_by_table"]:
        raise ValueError("Paper-cell coverage differs from the reviewed specification")
    tables, hashes, comparisons = {}, {}, []
    for cell in cells:
        name = cell["result_file"]
        if name not in tables:
            path = safe_path(tables_root, name)
            if path.is_file():
                with path.open(newline="") as stream:
                    tables[name] = list(csv.DictReader(stream))
                hashes[name] = sha256_file(path)
            else:
                tables[name] = []
        status, actual = compare_cell(cell, tables[name])
        comparisons.append({**cell, "generated_at_paper_precision": actual, "status": status})
    by_table = defaultdict(Counter)
    for row in comparisons:
        by_table[row["table"]][row["status"]] += 1
    counts = Counter(row["status"] for row in comparisons)
    report = {"paper_sha256": metadata["paper_sha256"], "scope": metadata["scope"],
              "all_checked_cells_match": counts.get("match", 0) == len(cells),
              "checked_cells": len(cells), "counts": dict(counts),
              "by_table": {table: dict(values) for table, values in by_table.items()},
              "generated_table_sha256": hashes,
              "limitations": metadata["limitations"]}
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "comparison.csv", comparisons)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Paper consistency check", "",
             "Numerical comparison with the supplied 53-page `SAADI_ICLR.pdf`, identified by its SHA-256 in `paper.json`. "
             "Expected values were extracted independently from the paper. "
             "Generated values are rounded to the paper's stated precision before comparison.", "",
             f"Checked **{len(cells)} cells**: **{counts.get('match', 0)} match**, "
             f"**{counts.get('mismatch', 0)} differ**, and "
             f"**{counts.get('missing_row', 0) + counts.get('missing_column', 0)} are missing from the generated tables**.", "",
             "| Paper table | Matching cells | Different cells | Missing cells |",
             "| --- | ---: | ---: | ---: |"]
    lines.extend(f"| {table} | {values['match']} | {values['mismatch']} | {values['missing_row'] + values['missing_column']} |"
                 for table, values in by_table.items())
    lines.extend(["", "The [cell comparison](comparison.csv) lists each paper value, its generated value, row selector, and status. "
                  "[The machine-readable report](report.json) records the table hashes and coverage.", "",
                  "## Scope and notes", ""])
    lines.extend(f"- {note}" for note in metadata["limitations"])
    lines.extend(["", "## Reproduce this check", "", "Run from the repository root:", "", "```bash",
                  "python scripts/build_tables.py --output-dir results/tables",
                  "python scripts/audit_paper.py", "```", "",
                  "The audit exits with code 1 when any checked cell differs or is missing. "
                  "Agreement with the saved inputs (`verification.json`) and agreement with the paper (this check) are separate checks.", ""])
    (output_dir / "README.md").write_text("\n".join(lines))
    return report
