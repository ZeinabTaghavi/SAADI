"""CLI command that evaluates retrieval results JSON and writes a metric summary."""

from __future__ import annotations

import argparse
import json

from ..evaluation.retrieval_metrics import (
    evaluate_retrieval_results,
    load_retrieval_results_json_with_metadata,
    saved_retrieval_depth,
    write_eval_report_json,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate HR@5 and HR@10")
    p.add_argument("--results_json", required=True)
    p.add_argument("--split", choices=["auto", "stateful"], default="auto")
    p.add_argument("--out", default="eval_report.json")
    p.add_argument("--eval_k", type=int, default=5, help="Primary HR cutoff (default: 5)")
    p.add_argument("--retrieve_k", type=int, help="Saved ranking depth; inferred from report_index.json when present")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows, metadata = load_retrieval_results_json_with_metadata(args.results_json, split=args.split)
    depth = saved_retrieval_depth(args.results_json)
    if depth is not None and args.retrieve_k is not None and depth != args.retrieve_k:
        raise ValueError(f"--retrieve_k={args.retrieve_k} conflicts with saved retrieve_k={depth}")
    report = evaluate_retrieval_results(rows, eval_k=args.eval_k,
                                       retrieve_k=depth if depth is not None else args.retrieve_k,
                                       metadata=metadata)
    write_eval_report_json(args.out, report)
    print(json.dumps(report.to_dict(), indent=2))
    return 0
