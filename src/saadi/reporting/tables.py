"""Rebuild Saadi tables from checksummed, portable result records."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from saadi.evaluation.hr import evaluate_rows
from .layout import TABLES, STATISTICS, table_directory, generator_script
from .output_sources import StudyInputs

IDENTITY = ("study", "dataset", "retriever", "condition", "source_id")
REPORTED_PROVENANCE = {"reported_summary", "paper_reported_summary"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checked_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Result path escapes its bundle: {relative}")
    return path


def verify_bundle(root: Path) -> dict:
    root = Path(root)
    checksums = read_json(root / "checksums.json")
    actual = {str(p.relative_to(root)) for p in root.rglob("*")
              if p.is_file() and p.name != "checksums.json"}
    if actual != set(checksums):
        raise ValueError("Reference bundle contains missing or unregistered files")
    for name, expected in checksums.items():
        path = checked_path(root, name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Reference checksum mismatch: {name}")
    manifest = read_json(root / "manifest.json")
    if manifest.get("schema_version") != 1 or manifest.get("metric") != "HR":
        raise ValueError("Unsupported result schema")
    for source in manifest["sources"]:
        if checksums.get(source["path"]) != source["sha256"]:
            raise ValueError(f"Source checksum mismatch: {source['source_id']}")
        if source["provenance"] == "paper_reported_summary":
            document = source.get("source_document", {})
            if (not all(document.get(k) for k in ("filename", "sha256", "page", "table", "panel"))
                    or document["sha256"] != source["original_sha256"]):
                raise ValueError("Paper-reported summaries require a checksummed document and table locator")
    return manifest


def rankings(root, source):
    with gzip.open(checked_path(root, source["path"]), "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def key(row):
    return tuple(row[name] for name in IDENTITY) + (row["cutoff"],)


def reported_metrics(root, source, *, inputs=None):
    for row in inputs.read(source) if inputs is not None else read_json(checked_path(root, source["path"])):
        condition = "; ".join(f"{name}={row[name]}" for name in (
            "branching_factor", "model_key", "score_formulation", "normalization",
            "conditioning", "budget", "selector", "weights") if row.get(name)) or "default"
        for column, value in row.items():
            if column.startswith("hr_percent_at_"):
                yield dict(study=source["study"], dataset=row["dataset"], retriever=row["retriever"],
                           condition=condition, source_id=source["source_id"], cutoff=int(column.rsplit("_", 1)[1]),
                           queries=None, eligible=None, hits=None, hr=float(value) / 100,
                           provenance=source["provenance"])


def compute_metrics(root, manifest, *, inputs=None, studies=None):
    values = []
    populations = {}
    for source in manifest["sources"]:
        if studies is not None and source["study"] not in studies:
            continue
        if source["provenance"] in REPORTED_PROVENANCE:
            values.extend(reported_metrics(root, source, inputs=inputs))
        elif source["provenance"] == "per_query":
            rows = inputs.read(source) if inputs is not None else rankings(root, source)
            population = hashlib.sha256(json.dumps(sorted(r["query_id"] for r in rows)).encode()).hexdigest()
            if population != source["query_ids_sha256"]:
                raise ValueError(f"Query population changed: {source['source_id']}")
            if source["study"] == "main":
                previous = populations.setdefault(source["dataset"], population)
                if previous != population:
                    raise ValueError(f"Main retrievers use different queries: {source['dataset']}")
            for metric in evaluate_rows(rows, retrieve_k=source["saved_depth"]):
                values.append({**{name: source[name] for name in IDENTITY}, **metric,
                               "provenance": "per_query"})
    expected = read_json(root / "expected_metrics.json")
    if studies is not None:
        expected = [row for row in expected if row["study"] in studies]
    actual_by_key = {key(row): row for row in values}
    expected_by_key = {key(row): row for row in expected}
    if len(actual_by_key) != len(values) or len(expected_by_key) != len(expected):
        raise ValueError("Duplicate result identity")
    if actual_by_key.keys() != expected_by_key.keys():
        raise ValueError("Result coverage changed")
    for identity, actual in actual_by_key.items():
        reference = expected_by_key[identity]
        if "provenance" in reference and actual["provenance"] != reference["provenance"]:
            raise ValueError(f"Changed evidence provenance: {identity}")
        for name in ("queries", "eligible", "hits", "hr"):
            left, right = actual[name], reference[name]
            equal = (left == right) if name != "hr" or left is None or right is None else math.isclose(left, right, abs_tol=1e-12, rel_tol=0)
            if not equal:
                raise ValueError(f"Changed {name}: {identity}: {left} != {right}")
    return values


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(name for row in rows for name in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def latex(value):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#", "$": r"\$", "{": r"\{", "}": r"\}"}
    return "".join(replacements.get(c, c) for c in str(value))


def write_table(path, rows):
    write_csv(path.with_suffix(".csv"), rows)
    if not rows:
        return
    fields = list(rows[0])
    def display(value):
        return "—" if value is None else f"{value:.2f}" if isinstance(value, float) else str(value)
    cells = [[display(row.get(field)) for field in fields] for row in rows]
    markdown = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    markdown.extend("| " + " | ".join(c.replace("|", r"\|") for c in row) + " |" for row in cells)
    path.with_suffix(".md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    lines = [r"\begin{tabular}{" + "l" * len(fields) + "}", r"\hline",
             " & ".join(latex(f) for f in fields) + r" \\", r"\hline"]
    lines.extend(" & ".join(latex(c) for c in row) + r" \\" for row in cells)
    lines.extend([r"\hline", r"\end{tabular}"])
    path.with_suffix(".tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_tables(reference: Path, output: Path, *, outputs_root=None, input_mode="auto", studies=None, figures=False):
    reference, output = Path(reference).resolve(), Path(output).resolve()
    if output == reference or output.is_relative_to(reference):
        raise ValueError("Tables must be written outside the reference bundle")
    manifest = verify_bundle(reference)
    selected = set(TABLES if studies is None else studies)
    if not selected or selected - set(TABLES):
        raise ValueError(f"Unknown or empty study selection: {sorted(selected - set(TABLES))}")
    inputs = StudyInputs(reference, outputs_root=outputs_root, mode=input_mode)
    values = compute_metrics(reference, manifest, inputs=inputs, studies=selected)
    output.mkdir(parents=True, exist_ok=True)
    table_rows = defaultdict(dict)
    for row in values:
        identity = tuple(row[name] for name in ("dataset", "retriever", "condition", "source_id"))
        table_row = table_rows[row["study"]].setdefault(identity, {
            "Dataset": row["dataset"], "Retriever": row["retriever"], "Condition": row["condition"],
            "Eligible queries": row["eligible"], "Evidence": row["provenance"]})
        table_row[f"HR@{row['cutoff']} (%)"] = None if row["hr"] is None else row["hr"] * 100
    for study in selected:
        table_directory(output, study).mkdir(parents=True, exist_ok=True)
    for study, rows in table_rows.items():
        directory = table_directory(output, study)
        write_table(directory / "table", list(rows.values()))
        write_csv(directory / "metrics.csv", [row for row in values if row["study"] == study])
    sources = {s["source_id"]: s for s in manifest["sources"]}
    for statistic in manifest["statistics"]:
        source = sources[statistic["source_id"]]
        if source["study"] not in selected:
            continue
        rows = inputs.read(source)
        target = table_directory(output, source["study"]) / STATISTICS[statistic["study"]]
        if len(rows) < 100:
            write_table(target, rows)
        else:
            write_csv(target.with_suffix(".csv"), rows)
    curves = []
    for source in manifest["sources"]:
        if source["study"] == "retrieval_depth" and "retrieval_depth" in selected:
            for metric in evaluate_rows(inputs.read(source), cutoffs=range(1, 21), retrieve_k=20):
                curves.append({"dataset": source["dataset"], **metric})
    if curves:
        depth_dir = table_directory(output, "retrieval_depth")
        write_csv(depth_dir / "curve.csv", curves)
        if figures:
            from .figures import plot_depth_curve
            plot_depth_curve(depth_dir)
    report = {"metric": "HR", "verified_cells": len(values),
              "recomputed_cells": sum(v["provenance"] == "per_query" for v in values),
              "reported_cells": sum(v["provenance"] in REPORTED_PROVENANCE for v in values),
              "paper_reported_cells": sum(v["provenance"] == "paper_reported_summary" for v in values),
              "input_source": inputs.mode, "studies": sorted(selected), "coverage": manifest["coverage"]}
    for study in selected:
        directory = table_directory(output, study)
        study_values = [row for row in values if row["study"] == study]
        source_rows = {json.dumps(row, sort_keys=True): row for row in inputs.records if row["study"] == study}
        (directory / "sources.json").write_text(json.dumps(list(source_rows.values()), indent=2) + "\n")
        (directory / "verification.json").write_text(json.dumps({"metric": "HR", "study": study,
            "input_source": inputs.mode, "verified_cells": len(study_values),
            "recomputed_cells": sum(row["provenance"] == "per_query" for row in study_values),
            "reported_cells": sum(row["provenance"] in REPORTED_PROVENANCE for row in study_values),
            "paper_reported_cells": sum(row["provenance"] == "paper_reported_summary" for row in study_values)}, indent=2) + "\n")
        (directory / "generate.py").write_text(generator_script(study))
        note = ("HR is recomputed from per-query rankings and checked against the saved study values."
                if any(row["provenance"] == "per_query" for row in study_values)
                else "These are saved Saadi summary values, preserved with their source hashes.")
        if any(row["provenance"] == "paper_reported_summary" for row in study_values):
            note = ("These values are transcribed from Table 6 of the paper and labeled `paper_reported_summary`; "
                    "the PDF hash, page, and panel are recorded in `sources.json`. "
                    "They are a transcription, not a recomputation: the query-level outputs of that run are not included.")
        (directory / "README.md").write_text(f"# {TABLES[study][1]}\n\n"
            "This folder contains the Saadi table, its generator, and its input provenance.\n\n"
            "```bash\npython generate.py\n# Require the restored full output archive:\npython generate.py --input-source outputs\n```\n\n"
            "Run from this folder, or invoke the script by its path from the repository root. "
            "The default reads the restored output archive when it is present and the bundled reference inputs otherwise.\n\n"
            f"{note} Presentation HR values are percentages; `metrics.csv`, when present, stores fractions. "
            "See [table.md](table.md), [sources.json](sources.json), and [verification.json](verification.json).\n")
    if studies is None:
        combined = output / "combined"
        combined.mkdir(exist_ok=True)
        write_csv(combined / "all_metrics.csv", values)
        (combined / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
        links = [f"- [{title}]({path}/README.md)" for path, title in TABLES.values()]
        (output / "README.md").write_text("# Saadi study tables\n\n"
            "Each study folder contains its table, `generate.py`, input paths/hashes, and verification results. "
            "All tables report Saadi only.\n\n" + "\n".join(links) + "\n\n"
            "The [combined HR data](combined/all_metrics.csv) contains every available configuration. "
            "[Verification](combined/verification.json) records source coverage and recomputed versus reported values.\n")
    return report
