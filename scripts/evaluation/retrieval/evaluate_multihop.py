#!/usr/bin/env python3
"""Evaluate Saadi retrieval separately for each question hop count."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

from saadi.config.loader import load_config
from saadi.evaluation.retrieval_metrics import (
    evaluate_retrieval_results,
    load_retrieval_results_json_with_metadata,
    saved_retrieval_depth,
)
from saadi.indexing.dataset_loaders import load_qa_entries_from_config


def hop_count(entry):
    for key in ("id", "source_document_id", "document_id"):
        value = str(entry.get(key, ""))
        match = re.search(r"(?:^|[:_])hop[_-]?([1-4])(?:[:_]|$)", value)
        if match:
            return int(match.group(1))
        match = re.match(r"([2-4])hop", value)
        if match:
            return int(match.group(1))
    raise ValueError(f"Cannot infer hop count for QA entry {entry.get('id')!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval_k", type=int, default=5, help="Primary HR cutoff (default: 5)")
    parser.add_argument("--retrieve_k", type=int, help="Saved ranking depth when no report index is present")
    args = parser.parse_args()
    cfg = load_config(args.config)
    qa, _ = load_qa_entries_from_config(cfg["dataset"], base_dir=args.config.parent)
    hops = {str(entry["id"]): hop_count(entry) for entry in qa}
    files = sorted(args.results_dir.glob("retrieval_results__*.json"))
    if not files:
        raise FileNotFoundError(f"No retrieval results in {args.results_dir}")
    for path in files:
        rows, metadata = load_retrieval_results_json_with_metadata(path)
        depth = saved_retrieval_depth(path)
        if depth is not None and args.retrieve_k is not None and depth != args.retrieve_k:
            raise ValueError(f"--retrieve_k={args.retrieve_k} conflicts with saved retrieve_k={depth} for {path.name}")
        groups = defaultdict(list)
        for row in rows:
            groups[hops[row.query_id]].append(row)
        report = {
            str(hop): evaluate_retrieval_results(
                items, eval_k=args.eval_k,
                retrieve_k=depth if depth is not None else args.retrieve_k,
                metadata=metadata,
            ).to_dict()
            for hop, items in sorted(groups.items())
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / path.name.replace("retrieval_results__", "hop_evaluation__")).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
