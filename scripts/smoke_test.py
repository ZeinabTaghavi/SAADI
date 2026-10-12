#!/usr/bin/env python3
"""Exercise Saadi construction, BM25 retrieval, and HR evaluation offline on synthetic data."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def run_smoke(output):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty smoke-test output directory")
    output.mkdir(parents=True, exist_ok=True)
    document = "Mira leads the Aurora team. The Aurora team studies Lake Alder. Leon records the measurements."
    (output / "documents.json").write_text(json.dumps({"sample": document}))
    (output / "qa.json").write_text(json.dumps([
        {"id": "q1", "document_id": "sample", "question": "Who leads the Aurora team?",
         "answers": ["Mira"], "retrieval_spans": ["Mira leads the Aurora team"]},
        {"id": "q2", "document_id": "sample", "question": "Which lake does the Aurora team study?",
         "answers": ["Lake Alder"], "retrieval_spans": ["The Aurora team studies Lake Alder"]}]))
    config = {"output_dir": str(output / "index"),
              "dataset": {"name": "synthetic", "source": "files", "documents_json": str(output / "documents.json"), "qa_json": str(output / "qa.json")},
              "ingest": {"enabled": True, "use_cache": True, "strategy": "hierarchical", "chunk_size": 8, "chunk_overlap": 0,
                         "strategy_kwargs": {"branching_factor": 2, "active_entity_selector": "bm25", "max_entities": 5, "max_aliases": 2, "max_desc": 3}},
              "model": {"backend": "mock", "alias": "mock"},
              "retrieval": {"mode": "canonicalized", "scope": "per_document", "retrieve_k": 10, "eval_k": 5, "retriever": "bm25"},
              "cached_retrieval": {"method": "mbr", "lambda_t": 1.0, "lambda_n": 0.5, "lambda_h": 0.3, "normalization_method": "zscore"}}
    config_path = output / "config.json"
    config_path.write_text(json.dumps(config))
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1", "SAADI_CACHE_ROOT": str(output / ".cache")}
    def command(*args):
        subprocess.run([sys.executable, "-m", "saadi.cli.main", *map(str, args)], cwd=ROOT, env=env, check=True)
    command("run-experiment", "--config", config_path, "--ingest_only")
    command("retrieve-cached", "--config", config_path, "--retriever", "bm25", "--output_dir", output / "retrieval")
    result_paths = list((output / "retrieval").rglob("retrieval_results__*.json"))
    if len(result_paths) != 1:
        raise AssertionError(f"Expected one Saadi result file, found {len(result_paths)}")
    command("eval", "--results_json", result_paths[0], "--out", output / "evaluation.json")
    report = json.loads((output / "evaluation.json").read_text())
    if report["metric"] != "HR" or report["n_examples"] != 2 or report["n_eligible"] != 2 or report["hit_rate"] != 1.0:
        raise AssertionError(f"Unexpected smoke-test report: {report}")
    if any("vanilla" in str(p.relative_to(output)).lower() for p in output.rglob("*")):
        raise AssertionError("Unexpected baseline output")
    print(f"Saadi offline smoke test passed: {output}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.output_dir:
        run_smoke(args.output_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="sadi-smoke-") as temporary:
            run_smoke(Path(temporary))
