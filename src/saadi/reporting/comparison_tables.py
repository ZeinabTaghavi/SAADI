"""Final paper tables: measured Saadi outputs joined to saved comparison CSVs.

No generated table is an input, and no comparison method is executed here.
"""
import json
import math
from pathlib import Path

from saadi.evaluation.hr import evaluate_rows
from .comparison_inputs import DATASET_LABELS, RETRIEVER_LABELS, METHOD_LABELS, AGENTS, load_other_methods, read_csv
from .output_sources import StudyInputs, read_rankings, safe_path, sha256_file
from .tables import verify_bundle, compute_metrics, write_csv, write_table

COMPARISON_TABLES = {
    "retrieval_comparison": ("main/retrieval_comparison", "Retrieval comparison (Table 1)"),
    "matched_exposure": ("main/matched_exposure", "Retrieval at matched evidence exposure (Table 2)"),
    "fixed_and_agentic": ("comparisons/fixed_and_agentic", "Fixed-depth and agentic retrieval (Table 22)"),
    "multihop_comparison": ("comparisons/multihop", "Retrieval by hop count (Table 21)"),
    "context_comparison": ("comparisons/context_expansion", "Context expansion comparison (Table 26)"),
    "resource_use": ("comparisons/resource_use", "Model calls and tokens (Table 24)"),
    "calls_per_document": ("construction/calls_per_document", "Saadi construction calls per document (Table 25)"),
}
CONDITIONS = ("no_expansion", "32K_expansion", "64K_expansion", "128K_expansion", "256K_expansion")
DATASET_KEYS = {label: key for key, label in DATASET_LABELS.items()}


def only(rows, **selector):
    matches = [r for r in rows if all(str(r[k]) == str(v) for k, v in selector.items())]
    if len(matches) != 1:
        raise ValueError(f"Expected one input for {selector}, found {len(matches)}")
    return matches[0]


def verify_auxiliary(root):
    root = Path(root)
    checksums = json.loads((root / "checksums.json").read_text())
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p.name != "checksums.json"}
    if actual != set(checksums):
        raise ValueError("Saadi comparison inputs are missing or unregistered")
    for name, digest in checksums.items():
        if sha256_file(safe_path(root, name)) != digest:
            raise ValueError(f"Saadi comparison checksum changed: {name}")
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported Saadi comparison schema")
    for source in manifest["rankings"]:
        if checksums.get(source["path"]) != source["sha256"]:
            raise ValueError("Saadi expansion source checksum disagrees with manifest")
    return manifest


class ComparisonInputs:
    def __init__(self, reference, outputs_root, other_methods_root, comparison_root, mode):
        self.reference = Path(reference)
        self.outputs_root = Path(outputs_root)
        self.other_root = Path(other_methods_root)
        self.aux_root = Path(comparison_root)
        self.manifest = verify_bundle(self.reference)
        self.inputs = StudyInputs(reference, outputs_root=outputs_root, mode=mode)
        self.other = load_other_methods(self.other_root)
        self.aux = verify_auxiliary(self.aux_root)
        self.evidence = []
        self._cache = {}
        self.other_kinds = set()

    def read(self, source):
        if source["source_id"] not in self._cache:
            self._cache[source["source_id"]] = self.inputs.read(source)
        return self._cache[source["source_id"]]

    def aux_csv(self, name):
        path = self.aux_root / name
        rows = read_csv(path)
        self.evidence.append({"path": "outputs/sadi/comparison_inputs/" + name, "sha256": sha256_file(path),
                              "provenance": "reported_summary", "source_ids": sorted({r["source_id"] for r in rows})})
        return rows

    def external(self, study):
        self.other_kinds.add("retrieval")
        rows = [dict(row) for row in self.other["retrieval"] if row["study"] == study]
        for row in rows:
            for field in ("hr_percent", "mean_chunks"):
                row[field] = float(row[field]) if row[field] else None
        return rows

    def main(self):
        return [{**r, "method": "sadi", "variant": "default", "hr_percent": 100 * r["hr"]}
                for r in compute_metrics(self.reference, self.manifest, inputs=self, studies={"main"})]

    def evaluated(self, source, cutoffs):
        rows = self.read(source)
        for metric in evaluate_rows(rows, cutoffs=cutoffs, retrieve_k=source["saved_depth"]):
            yield {"method": "sadi", "dataset": source["dataset"], "retriever": source.get("retriever", "bm25"),
                   "condition": source.get("condition", "standard"), "source_id": source["source_id"],
                   **metric, "hr_percent": metric["hr"] * 100, "provenance": "per_query",
                   "mean_chunks": sum(min(metric["cutoff"], len(r["retrieved_chunk_ids"])) for r in rows) / len(rows)}

    def depth(self):
        compute_metrics(self.reference, self.manifest, inputs=self, studies={"retrieval_depth"})
        return [r for s in self.manifest["sources"] if s["study"] == "retrieval_depth"
                for r in self.evaluated(s, range(1, 21))]

    def context(self):
        compute_metrics(self.reference, self.manifest, inputs=self, studies={"context_expansion"})
        result = [r for s in self.manifest["sources"] if s["study"] == "context_expansion"
                  for r in self.evaluated(s, (5, 10, 15, 20))]
        for source in self.aux["rankings"]:
            portable = read_rankings(self.aux_root / source["path"])
            rows = portable
            relative = "outputs/sadi/comparison_inputs/" + source["path"]
            digest = source["sha256"]
            if self.inputs.mode == "outputs":
                relative = source["original_source"]
                path = safe_path(self.outputs_root, relative.removeprefix("outputs/"))
                digest = source["original_sha256"]
                if sha256_file(path) != digest:
                    raise ValueError(f"Changed Saadi expansion output: {relative}")
                retained = {row["query_id"] for row in portable}
                rows = [r for r in read_rankings(path) if r["query_id"] in retained]
                if sorted(rows, key=lambda r: r["query_id"]) != sorted(portable, key=lambda r: r["query_id"]):
                    raise ValueError("Copied Saadi expansion rankings differ from portable inputs")
            if len(rows) != source["selected_queries"] or len({r["query_id"] for r in rows}) != len(rows):
                raise ValueError("Saadi expansion query population changed")
            metrics = evaluate_rows(rows, retrieve_k=10)
            if metrics != source["expected_metrics"]:
                raise ValueError("Saadi expansion metrics changed")
            self.evidence.append({"path": relative, "sha256": digest, "source_id": source["source_id"],
                                  "provenance": "per_query", "queries": len(rows), "selection": source["selection"]})
            for metric in metrics:
                result.append({"method": "sadi", "dataset": source["dataset"], "condition": source["condition"],
                               "retriever": "contriever", "source_id": source["source_id"], **metric,
                               "hr_percent": 100 * metric["hr"], "provenance": "per_query",
                               "mean_chunks": sum(min(metric["cutoff"], len(r["retrieved_chunk_ids"])) for r in rows) / len(rows)})
        return result

    def hops(self):
        source = only(self.manifest["sources"], study="multihop")
        exposure = self.aux_csv("multihop_exposure.csv")
        result = []
        for row in self.read(source):
            dataset, hop = DATASET_KEYS[row["Dataset"]], row["Hops"].lower()
            count = only(exposure, dataset=dataset, hop=hop, cutoff=row["Cutoff"])
            if int(count["queries"]) != row["Queries"]:
                raise ValueError("Saadi multihop exposure and HR use different populations")
            if not math.isclose(row["HR (%)"], 100 * row["Hits"] / row["Eligible queries"], abs_tol=1e-10):
                raise ValueError("Saadi multihop HR disagrees with saved hit counts")
            result.append({"method": "sadi", "dataset": dataset, "hop": hop, "cutoff": row["Cutoff"],
                           "retriever": row["Retriever"], "hr_percent": row["HR (%)"], "mean_chunks": float(count["mean_chunks"]),
                           "queries": row["Queries"], "eligible": row["Eligible queries"], "hits": row["Hits"],
                           "source_id": source["source_id"], "provenance": "reported_summary"})
        return result

    def resources(self):
        self.other_kinds.add("resources")
        costs = self.aux_csv("costs.csv")
        result = []
        for dataset in DATASET_LABELS:
            for method in AGENTS:
                row = only(self.other["resources"], method=method, dataset=dataset, condition="standard")
                sadi = only(costs, comparison_method=method, dataset=dataset)
                if any(row[k] != sadi[k] for k in ("queries", "documents")):
                    raise ValueError("Cost comparison uses inconsistent populations")
                out = dict(row)
                for unit in ("calls", "input_tokens", "generated_tokens"):
                    online = float(row[f"{unit}_per_query"])
                    difference = float(sadi[f"{unit}_per_document"]) - float(row[f"offline_{unit}_per_document"])
                    if online <= 0 or difference < 0:
                        raise ValueError("Crossover requires positive online cost and nonnegative construction difference")
                    out[f"{unit}_crossover"] = difference / online
                    out[f"sadi_{unit}_per_document"] = float(sadi[f"{unit}_per_document"])
                result.append(out)
        return result

    def provenance(self, rows):
        methods = {r.get("method") for r in rows} - {"sadi", None}
        external = [r for r in self.other["evidence"] if Path(r["path"]).parent.name in methods
                    and Path(r["path"]).stem in self.other_kinds]
        return {"sadi_inputs": list({json.dumps(r, sort_keys=True): r for r in self.inputs.records + self.evidence}.values()),
                "other_method_inputs": external, "archived_other_method_sources": self.other["sources"] if methods else {},
                "archived_sadi_comparison_sources": self.aux["sources"] if self.evidence else {}}


def main_table(inputs):
    metrics = inputs.main() + inputs.external("main")
    result = []
    for retriever, label in RETRIEVER_LABELS.items():
        methods = [("vanilla", "default")]
        if retriever == "jina":
            methods.extend(("late_chunking", v) for v in ("no_overlap", "low_overlap", "more_overlap"))
        methods.extend((m, "default") for m in ("llamaindex", "lqca", "sadi"))
        for method, variant in methods:
            row = {"Retriever": label, "Method": METHOD_LABELS[method] + (f" ({variant.replace('_', ' ')})" if variant != "default" else "")}
            for dataset, dataset_label in DATASET_LABELS.items():
                for k in (5, 10):
                    row[f"{dataset_label} HR@{k} (%)"] = only(metrics, method=method, variant=variant, retriever=retriever, dataset=dataset, cutoff=k)["hr_percent"]
            result.append(row)
    return result, metrics


def fixed_table(inputs):
    metrics = inputs.external("agentic") + [r for r in inputs.depth() if r["cutoff"] in (3, 4, 5, 10, 15, 20)]
    result = []
    methods = [(m, k) for m in ("hipporag_v2", "raptor") for k in (5, 10)] + [(m, "") for m in AGENTS] + [("sadi", k) for k in (3, 4, 5, 10, 15, 20)]
    for method, cutoff in methods:
        row = {"Method": METHOD_LABELS[method] + (" (BM25)" if method == "sadi" else ""), "Cutoff": cutoff or "Trajectory"}
        for dataset, label in DATASET_LABELS.items():
            metric = only(metrics, method=method, dataset=dataset, cutoff=cutoff)
            row[f"{label} Avg. chunks"] = metric["mean_chunks"]
            row[f"{label} HR (%)"] = metric["hr_percent"]
        result.append(row)
    return result, metrics


def matched_table(inputs):
    agents, depth, resources = inputs.external("agentic"), inputs.depth(), inputs.resources()
    rows, metrics = [], []
    for dataset, label in DATASET_LABELS.items():
        for method in AGENTS:
            agent = only(agents, method=method, dataset=dataset, cutoff="")
            resource = only(resources, method=method, dataset=dataset)
            exposure = agent["mean_chunks"]
            if not math.isclose(exposure, float(resource["mean_chunks"]), abs_tol=1e-12):
                raise ValueError("Retrieval and resource summaries disagree on evidence exposure")
            lower, upper = math.floor(exposure), math.ceil(exposure)
            low = only(depth, dataset=dataset, cutoff=lower)
            high = only(depth, dataset=dataset, cutoff=upper)
            rows.append({"Dataset": label, "Method": METHOD_LABELS[method], "Avg. chunks": exposure,
                         "Method HR (%)": agent["hr_percent"], "Lower k": lower, "Saadi lower HR (%)": low["hr_percent"],
                         "Upper k": upper, "Saadi upper HR (%)": high["hr_percent"],
                         "Calls/query": float(resource["calls_per_query"]), "Call crossover (queries/document)": resource["calls_crossover"]})
            metrics.extend([agent, low, high])
    return rows, metrics


def hop_table(inputs):
    metrics = inputs.hops() + [r for r in inputs.external("multihop") if r["retriever"] == "native" or (r["retriever"] == "bm25" and r["cutoff"] == "10")]
    rows = []
    for dataset in ("musique", "novelhopqa"):
        methods = [(m, "") for m in AGENTS] + [("vanilla", 10)] + [("sadi", k) for k in ((3, 4, 5, 10, 20) if dataset == "musique" else (3, 4, 5, 10))]
        for method, cutoff in methods:
            row = {"Dataset": DATASET_LABELS[dataset], "Method": METHOD_LABELS[method] + (" (BM25)" if method in {"sadi", "vanilla"} else ""), "Cutoff": cutoff or "Trajectory"}
            for hop in ("overall", "1-hop", "2-hop", "3-hop", "4-hop"):
                metric = None if dataset == "musique" and hop == "1-hop" else only(metrics, method=method, dataset=dataset, hop=hop, cutoff=cutoff)
                row[f"{hop} Avg. chunks"] = metric["mean_chunks"] if metric else None
                row[f"{hop} HR (%)"] = metric["hr_percent"] if metric else None
            rows.append(row)
    return rows, metrics


def context_table(inputs):
    metrics = inputs.context() + inputs.external("context_expansion")
    rows = []
    methods = [(m, "native", "") for m in AGENTS] + [(m, r, k) for r in ("bm25", "contriever") for m in ("vanilla", "sadi") for k in ((5, 10, 15, 20) if r == "bm25" else (5, 10))]
    for dataset in ("musique", "qasper"):
        for method, retriever, cutoff in methods:
            row = {"Dataset": "MuSiQue" if dataset == "musique" else "QASPER", "Method": METHOD_LABELS[method],
                   "Retriever": RETRIEVER_LABELS.get(retriever, "—"), "Cutoff": cutoff or "Trajectory"}
            for condition in CONDITIONS:
                metric = only(metrics, dataset=dataset, method=method, retriever=retriever, cutoff=cutoff, condition=condition)
                label = "No expansion" if condition == "no_expansion" else condition.split("_")[0]
                row[f"{label} Avg. chunks"] = metric["mean_chunks"]
                row[f"{label} HR (%)"] = metric["hr_percent"]
            rows.append(row)
    return rows, metrics


def resource_table(inputs):
    metrics = inputs.resources()
    rows = []
    for metric in metrics:
        rows.append({"Dataset": DATASET_LABELS[metric["dataset"]], "Method": METHOD_LABELS[metric["method"]],
                     "Avg. chunks": float(metric["mean_chunks"]),
                     "Calls/query": float(metric["calls_per_query"]), "Input tokens/query": float(metric["input_tokens_per_query"]),
                     "Generated tokens/query": float(metric["generated_tokens_per_query"]),
                     **{f"{unit.replace('_', ' ').title()} crossover (queries/document)": metric[f"{unit}_crossover"] for unit in ("calls", "input_tokens", "generated_tokens")}})
    return rows, metrics


def construction_table(inputs):
    chunk_source = only(inputs.manifest["sources"], study="active_entities_statistics")
    summary_source = next(s for s in inputs.manifest["sources"] if s["original_source"].endswith("appendix_construction_hierarchical_summary_statistics_values.csv"))
    chunks, summaries = inputs.read(chunk_source), inputs.read(summary_source)
    rows = []
    for dataset, label in DATASET_LABELS.items():
        chunk, summary = only(chunks, dataset_key=dataset), only(summaries, dataset_key=dataset)
        if chunk["documents"] != summary["documents"]:
            raise ValueError("Construction summaries use different document populations")
        documents = int(chunk["documents"])
        memory, hierarchy = int(chunk["chunks"]) / documents, int(summary["total_summaries"]) / documents
        rows.append({"Dataset": label, "Documents": documents, "Chunks/document": memory, "Memory calls/document": memory,
                     "Hierarchy calls/document": hierarchy, "Total calls/document": memory + hierarchy})
    return rows, [{"method": "sadi", "provenance": "reported_statistics", **row} for row in rows]


RENDERERS = {"retrieval_comparison": main_table, "matched_exposure": matched_table, "fixed_and_agentic": fixed_table,
             "multihop_comparison": hop_table, "context_comparison": context_table,
             "resource_use": resource_table, "calls_per_document": construction_table}


def generator_script(study):
    return f'''#!/usr/bin/env python3
"""Regenerate this final paper table from saved outputs."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "pyproject.toml").is_file() and (p / "src/saadi").is_dir())
sys.path.insert(0, str(ROOT / "src"))
from saadi.reporting.cli import comparison_main
if __name__ == "__main__":
    raise SystemExit(comparison_main({study!r}, table_dir=HERE, project_root=ROOT))
'''


def build_comparison_tables(reference, output, *, outputs_root, input_mode="auto", studies=None, other_methods_root=None, comparison_root=None):
    output, outputs_root = Path(output).resolve(), Path(outputs_root).resolve()
    other_methods_root = Path(other_methods_root or outputs_root / "other_methods")
    comparison_root = Path(comparison_root or outputs_root / "sadi/comparison_inputs")
    for source_root in (reference, other_methods_root, comparison_root):
        if output.is_relative_to(Path(source_root).resolve()):
            raise ValueError("Generated tables must be outside their source inputs")
    selected = set(COMPARISON_TABLES if studies is None else studies)
    if not selected or selected - set(COMPARISON_TABLES):
        raise ValueError("Unknown or empty comparison table selection")
    report = {}
    for study in COMPARISON_TABLES:
        if study not in selected:
            continue
        inputs = ComparisonInputs(reference, outputs_root, other_methods_root, comparison_root, input_mode)
        rows, metrics = RENDERERS[study](inputs)
        directory = output / COMPARISON_TABLES[study][0]
        directory.mkdir(parents=True, exist_ok=True)
        write_table(directory / "table", rows)
        if study in {"retrieval_comparison", "matched_exposure", "resource_use", "multihop_comparison", "context_comparison"}:
            def display_value(column, value):
                if not isinstance(value, float):
                    return value
                digits = 2
                if (study in {"retrieval_comparison", "multihop_comparison", "context_comparison"}) and "HR" in column:
                    digits = 1
                if "crossover" in column or "tokens/query" in column:
                    digits = 0
                return f"{value:.{digits}f}"
            display = [{k: display_value(k, v) for k, v in row.items()} for row in rows]
            write_table(directory / "table", display)
            write_csv(directory / "table.csv", rows)  # retain full precision in machine-readable tables
        unique = list({json.dumps(r, sort_keys=True): r for r in metrics}.values())
        write_csv(directory / "metrics.csv", unique)
        (directory / "sources.json").write_text(json.dumps(inputs.provenance(unique), indent=2) + "\n")
        validation = {"table": study, "input_source": inputs.inputs.mode, "rows": len(rows),
                      "sadi_input_rows": sum(r.get("method") == "sadi" for r in unique),
                      "other_method_input_rows": sum(r.get("method") != "sadi" for r in unique),
                      "comparison_methods_executed": False, "input_hashes_verified": True}
        (directory / "verification.json").write_text(json.dumps(validation, indent=2) + "\n")
        (directory / "generate.py").write_text(generator_script(study))
        (directory / "README.md").write_text(f"# {COMPARISON_TABLES[study][1]}\n\n"
            "Run `python generate.py` here, or invoke this script by its path. "
            "Saadi inputs come from `outputs/sadi/` (or the restored full experiment archive); "
            "comparison methods come from `outputs/other_methods/<method>/*.csv`. No comparison method is rerun.\n\n"
            "[table.md](table.md), [table.csv](table.csv), and [table.tex](table.tex) are generated together. "
            "CSV values retain full precision; HR columns are percentages. "
            "[metrics.csv](metrics.csv) records the inputs used, including per-query versus reported-summary evidence. "
            "[sources.json](sources.json) records input hashes and [verification.json](verification.json) records coverage.\n\n"
            "Use `--input-source reference` to require the compact Saadi inputs, or `--input-source outputs` to require the restored archive. "
            "Use `--other-methods-root` for a separately validated set of comparison CSVs. "
            "Agreement with the paper is checked separately by the paper-consistency report.\n")
        report[study] = validation
    return report
