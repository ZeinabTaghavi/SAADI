#!/usr/bin/env python3
"""Import saved comparison summaries and Saadi comparison inputs from a research archive.

No baseline implementation or language model is executed. Regeneration uses the
resulting portable files and does not require the original research directory.
"""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from saadi.evaluation.hr import evaluate_rows
from saadi.reporting.comparison_inputs import RETRIEVAL_FIELDS, RESOURCE_FIELDS, METHOD_LABELS, read_csv, load_other_methods
from saadi.reporting.output_sources import read_rankings, sha256_file

METHODS = {label: key for key, label in METHOD_LABELS.items()}
DATASETS = {"LooGLE": "loogle", "MuSiQue": "musique", "MuSiQue-32K": "musique",
            "QASPER": "qasper", "QASPER-64K": "qasper", "NovelHopQA": "novelhopqa",
            "loogle": "loogle", "musique_32k": "musique", "qasper_64k": "qasper", "novelhopqa": "novelhopqa"}
CONDITIONS = {"No expansion": "no_expansion", **{f"{k}K": f"{k}K_expansion" for k in (32, 64, 128, 256)}}


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def import_inputs(source_root, outputs_root):
    source_root, outputs_root = Path(source_root).resolve(), Path(outputs_root).resolve()
    tables = source_root / "outputs/Tables"
    sources = {}

    def source(identifier, relative):
        path = tables / relative
        sources[identifier] = {"original_source": str(path.relative_to(source_root)), "sha256": sha256_file(path)}
        return path

    main = source("main_fixed_depth", "Final_table.txt")
    agents = read_csv(source("fixed_and_agentic", "appendix_results_fixed_vs_agentic_retrieval/appendix_results_fixed_vs_agentic_retrieval.csv"))
    resources = read_csv(source("resource_use", "main_results_computational_profile/main_computational_profile_values.csv"))
    hops = read_csv(source("multihop", "appendix_results_multihop_retrieval/appendix_results_multihop_retrieval.csv"))
    expansion = read_csv(source("context_expansion", "appendix_results_context_expansion/appendix_results_context_expansion.csv"))
    retrieval_rows, resource_rows = [], []

    def add(**kwargs):
        row = {key: "" for key in RETRIEVAL_FIELDS}
        row.update(condition="standard", retriever="native", variant="default", hop="overall", provenance="reported_summary")
        row.update(kwargs)
        retrieval_rows.append(row)

    retriever = None
    parsed_rows = 0
    for line in main.read_text().splitlines():
        if "&" not in line:
            continue
        parts = [part.strip() for part in line.split("&")]
        match = re.search(r"\\multirow\{\d+\}\{\*\}\{([^}]+)\}", parts[0])
        if match:
            retriever = {"BGE-M3": "bge_m3"}.get(match[1], match[1].lower())
        if len(parts) != 10 or retriever is None:
            continue
        values = re.findall(r"\d+\.\d+", " & ".join(parts[2:]))
        if len(values) != 8:
            raise ValueError(f"Unrecognized main-table row: {line}")
        parsed_rows += 1
        method = parts[1]
        if method in {"SAADI", "SADI", "Saadi"}:
            continue  # Saadi always comes from its ranking outputs, never this summary.
        variant = "default"
        if method.startswith("LateC("):
            variant = {"LateC(No overlap)": "no_overlap", "LateC(Low overlap)": "low_overlap", "LateC(More overlap)": "more_overlap"}[method]
            method = "late_chunking"
        else:
            method = METHODS[method]
        for dataset, pair in zip(("loogle", "musique", "qasper", "novelhopqa"), (values[i:i+2] for i in range(0, 8, 2))):
            for cutoff, value in zip((5, 10), pair):
                add(study="main", method=method, dataset=dataset, retriever=retriever, variant=variant,
                    cutoff=cutoff, hr_percent=value, source_id="main_fixed_depth")
    if parsed_rows != 23:
        raise ValueError(f"Expected 23 archived main-table rows, found {parsed_rows}")
    for row in agents:
        if row["method"].startswith(("SAADI", "SADI", "Saadi")):
            continue
        add(study="agentic", method=METHODS[row["method"]], dataset=DATASETS[row["dataset_key"]],
            cutoff="" if row["cutoff"] == "Trajectory" else row["cutoff"],
            hr_percent=row["union_strict_hit_rate"], mean_chunks=row["mean_evaluated_chunks"], source_id="fixed_and_agentic")
    for row in hops:
        if any(name in row["approach"] for name in ("SAADI", "SADI", "Saadi")):
            continue
        method = "vanilla" if row["approach"].endswith(" Vanilla") else METHODS[row["approach"]]
        retriever = {"BM25 Vanilla": "bm25", "BGE-M3 Vanilla": "bge_m3"}.get(row["approach"], "native")
        add(study="multihop", method=method, dataset=DATASETS[row["dataset"]],
            retriever=retriever, hop=row["hop"].lower(),
            cutoff="" if row["cutoff"] == "Trajectory" else row["cutoff"], hr_percent=row["union_strict"],
            mean_chunks=row["avg_chunks"], queries=row["n"], eligible_queries=row["union_strict_denominator"],
            hits=row["union_strict_hits"], source_id="multihop")
    for row in expansion:
        if row["method"].startswith("mbr_"):
            continue
        base = dict(study="context_expansion", method=row["method"], dataset=DATASETS[row["dataset"]],
                    condition=CONDITIONS[row["extension"]], queries=row["result_count"], source_id="context_expansion")
        if row["method"] == "vanilla":
            for k in (5, 10, 15, 20):
                if row[f"union_s_hr_at_{k}"]:
                    add(**base, retriever=row["retriever"].lower(), cutoff=k,
                        mean_chunks=row[f"mean_evaluated_chunks_at_{k}"], hr_percent=100 * float(row[f"union_s_hr_at_{k}"]))
        else:
            add(**base, hr_percent=100 * float(row["union_s_retrieved"]), mean_chunks=row["mean_evaluated_chunks"])
    sadi_costs = []
    for row in resources:
        out = {key: "" for key in RESOURCE_FIELDS}
        out.update(method=METHODS[row["method"]], dataset=DATASETS[row["dataset_key"]], condition="standard",
                   queries=row["queries"], documents=row["documents"], mean_chunks=row["avg_retrieved_contexts"],
                   input_tokens_basis=row["llm_input_tokens_status"], generated_tokens_basis=row["llm_generated_tokens_status"],
                   provenance="reported_summary", source_id="resource_use")
        for field in ("calls", "input_tokens", "generated_tokens"):
            out[field + "_per_query"] = row["llm_" + field + "_per_query"]
            out["offline_" + field + "_per_document"] = row["method_offline_" + field + "_per_document"]
        resource_rows.append(out)
        sadi_costs.append({"dataset": out["dataset"], "comparison_method": out["method"], "queries": row["queries"],
                           "documents": row["documents"], **{field + "_per_document": row["saadi_" + field + "_per_document_reference"]
                           for field in ("calls", "input_tokens", "generated_tokens")},
                           "provenance": "reported_summary", "source_id": "resource_use"})

    other_root = outputs_root / "other_methods"
    files = {}
    for kind, rows, fields in (("retrieval", retrieval_rows, RETRIEVAL_FIELDS), ("resources", resource_rows, RESOURCE_FIELDS)):
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["method"]].append(row)
        for method, method_rows in sorted(grouped.items()):
            name = f"{method}/{kind}.csv"
            write_csv(other_root / name, method_rows, fields)
            files[name] = sha256_file(other_root / name)
    (other_root / "manifest.json").write_text(json.dumps({"schema_version": 1, "files": files, "sources": sources}, indent=2) + "\n")
    load_other_methods(other_root)

    # Supplement the existing Saadi bundle with the Contriever expansion rankings.
    sadi_root = outputs_root / "sadi/comparison_inputs"
    sadi_root.mkdir(parents=True, exist_ok=True)
    write_csv(sadi_root / "costs.csv", sadi_costs, list(sadi_costs[0]))
    exposures = [{"dataset": DATASETS[row["dataset"]], "hop": row["hop"].lower(),
                  "cutoff": row["cutoff"], "mean_chunks": row["avg_chunks"],
                  "queries": row["n"], "source_id": "multihop"}
                 for row in hops if row["approach"] == "BM25 SAADI"]
    write_csv(sadi_root / "multihop_exposure.csv", exposures, list(exposures[0]))
    additional = []
    method = "mbr_lt1_ln0.5_lh0.3_norm_zscore_mnw1_maw1_mdw1"
    run = "mbr_B_lt1.0_lm0.5_lh0.3_mn1.0_ma1.0_md1.0_norm_zscore"
    for dataset in ("musique", "qasper"):
        conditions = {}
        for condition in CONDITIONS.values():
            relative = f"outputs/{dataset}_ablation/{condition}/100_tokens/llm_cache/{dataset}/hierarchical/branching_factor_5/chunk_100/AE_5_2_3/Qwen3-30B-A3B-Instruct-2507/AEWord/mbrB_ablation/{run}/contriever/retrieval_reports/retrieval_results__contriever__{method}__per_document.json"
            path = source_root / relative
            conditions[condition] = (path, read_rankings(path))
        common_ids = set.intersection(*({row["query_id"] for row in rows} for _, rows in conditions.values()))
        for condition, (path, rows) in conditions.items():
            standard = condition == {"musique": "32K_expansion", "qasper": "64K_expansion"}[dataset]
            selected = rows if standard else [row for row in rows if row["query_id"] in common_ids]
            archived = next(r for r in expansion if DATASETS[r["dataset"]] == dataset and CONDITIONS[r["extension"]] == condition
                            and r["retriever"] == "Contriever" and r["method"] == method)
            metrics = evaluate_rows(selected, retrieve_k=10)
            for metric in metrics:
                if not math.isclose(metric["hr"], float(archived[f"union_s_hr_at_{metric['cutoff']}"]), abs_tol=1e-12):
                    raise ValueError(f"Contriever expansion source disagrees with the final saved summary: {dataset}/{condition}")
            identifier = hashlib.sha256(f"{dataset}/{condition}/contriever".encode()).hexdigest()[:20]
            relative = f"rankings/{identifier}.jsonl.gz"
            destination = sadi_root / relative
            destination.parent.mkdir(exist_ok=True)
            destination.write_bytes(gzip.compress(''.join(json.dumps(row, separators=(',', ':')) + '\n' for row in selected).encode(), mtime=0))
            additional.append({"source_id": identifier, "dataset": dataset, "condition": condition, "retriever": "contriever",
                               "path": relative, "sha256": sha256_file(destination), "saved_depth": 10,
                               "original_source": str(path.relative_to(source_root)), "original_sha256": sha256_file(path),
                               "raw_queries": len(rows), "selected_queries": len(selected),
                               "selection": "standard_full_population" if standard else "five_condition_query_intersection",
                               "expected_metrics": metrics})
    (sadi_root / "manifest.json").write_text(json.dumps({"schema_version": 1, "rankings": additional,
        "sources": {key: sources[key] for key in ("resource_use", "context_expansion", "multihop")}}, indent=2) + "\n")
    checksums = {str(p.relative_to(sadi_root)): sha256_file(p) for p in sadi_root.rglob("*") if p.is_file() and p.name != "checksums.json"}
    (sadi_root / "checksums.json").write_text(json.dumps(checksums, indent=2) + "\n")
    return {"other_method_retrieval_rows": len(retrieval_rows), "resource_rows": len(resource_rows), "sadi_expansion_sources": len(additional)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="Experiment workspace containing outputs/Tables and the dataset ablation folders")
    parser.add_argument("--outputs-root", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    print(json.dumps(import_inputs(args.source_root, args.outputs_root), indent=2))
