#!/usr/bin/env python3
"""C3 steps 02 (BM25) and 03 (dense): score every variant on the pairs that passed check 1.

For every query of a ready pair (work/ready_pairs.json, written by step 01), the six variants
are ranked over the document's full chunk pool by the published Equation 2
(``combine_mbr_variant_chunk_scores``) with the published weights, one channel weight set to 0.
Per query and variant: the 1-based rank at which the saved hit definition is first met
(a gold chunk, or the last chunk of a complete evidence group), from which HR@5, HR@10 and the
median rank follow. HR is also computed with the repository's ``evaluate_retrieval_results`` on
the top-10, and both must agree.

Inputs are only read; the stream files must have the sha256 step 01 recorded.
Writes results/rebuttal/c3_channel_ablation/per_query/<dataset>__<retriever>.csv.gz (small,
committed, so step 04 can combine laptop and server runs) and reports/step_NN.{md,json}.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    ALL_DATASETS,
    ALL_RETRIEVERS,
    BIG_RANK,
    DATASET_DISPLAY,
    N_BOOT,
    REPO_ROOT,
    RESULTS_DIR,
    RETRIEVE_K,
    RETRIEVER_DISPLAY,
    SEED,
    VARIANT_LABEL,
    VARIANTS,
    WORK_DIR,
    StepReport,
    StreamFiles,
    _hr,
    _rebuild_for_repo,
    combine,
    hit_rank,
    job_name,
    load_pair,
    load_published_weights,
    mcnemar_exact_p,
    rel,
    run_step,
    selected,
    sha256_file,
    table,
)
from saadi.rebuttal.frozen_records import paired_bootstrap_ci

READY_PATH = WORK_DIR / "ready_pairs.json"
PER_QUERY_DIR = RESULTS_DIR / "per_query"
NEVER = -1  # stored rank when no gold chunk or complete evidence group is in the pool

# Shown in this step's report as a first look; step 04 writes the full comparison set.
QUICK_COMPARISONS = (
    ("all_three", "text_summary"),
    ("all_three", "text_memory"),
    ("text_summary", "text_only"),
    ("text_memory", "text_only"),
    ("all_three", "text_only"),
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pair_label(d: str, r: str) -> str:
    return f"{DATASET_DISPLAY[d]} / {RETRIEVER_DISPLAY[r]}"


def per_query_path(d: str, r: str) -> Path:
    return PER_QUERY_DIR / f"{job_name(d, r)}.csv.gz"


def score_pair(entry: dict[str, Any], w) -> dict[str, Any]:
    d, r = entry["dataset"], entry["retriever"]
    files = StreamFiles(REPO_ROOT / entry["stream_files"][0], REPO_ROOT / entry["stream_files"][1], Path("/nonexistent"))
    for p, key in ((files.parquet, "parquet_sha256"), (files.queries, "queries_sha256")):
        if sha256_file(p) != entry[key]:
            raise RuntimeError(f"{rel(p)} changed since step 01 (sha256 differs); rerun step 01")
    t0 = time.perf_counter()
    queries, meta, rows = load_pair(d, r, files)
    n = len(queries)
    ranks = np.full((n, len(VARIANTS)), NEVER, dtype=np.int64)
    tops: dict[str, list[list[str]]] = {v: [] for v in VARIANTS}
    for qi, q in enumerate(queries):
        index, streams = _rebuild_for_repo(q, r)
        for vi, v in enumerate(VARIANTS):
            ranking = [x.chunk_id for x in combine(index, streams, w, v).ranked_chunks]
            tops[v].append(ranking[:RETRIEVE_K])
            hr0 = hit_rank(q, ranking)
            ranks[qi, vi] = NEVER if hr0 >= BIG_RANK else hr0 + 1
    eligible = np.asarray([bool(q.eligible) for q in queries])

    # Guards: all three still equals the saved rankings; rank-based HR equals the repo's HR.
    saved_top = [[str(x) for x in row.get("retrieved_chunk_ids", [])][:RETRIEVE_K] for row in rows]
    n_same_saved = sum(1 for a, b in zip(tops["all_three"], saved_top, strict=True) if a == b)
    agree = {}
    hr: dict[str, dict[str, float]] = {}
    for vi, v in enumerate(VARIANTS):
        repo = _hr(rows, tops[v])
        hr[v] = {}
        for k in (5, 10):
            ours_hits = int(((ranks[:, vi] >= 1) & (ranks[:, vi] <= k) & eligible).sum())
            agree[f"{v}@{k}"] = ours_hits == int(repo["hits"][str(k)]) and int(eligible.sum()) == int(repo["eligible"])
            hr[v][str(k)] = 100.0 * ours_hits / max(1, int(eligible.sum()))

    PER_QUERY_DIR.mkdir(parents=True, exist_ok=True)
    out = per_query_path(d, r)
    tmp = out.with_name(out.name + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", newline="") as handle:
        wr = csv.writer(handle)
        wr.writerow(["dataset", "retriever", "query_id", "doc_id", "eligible", *[f"rank_{v}" for v in VARIANTS]])
        for qi, (q, m) in enumerate(zip(queries, meta, strict=True)):
            wr.writerow([d, r, m["query_id"], m["doc_id"], int(eligible[qi]), *ranks[qi].tolist()])
    os.replace(tmp, out)

    e = eligible
    med = {}
    never = {}
    for vi, v in enumerate(VARIANTS):
        x = ranks[e, vi].astype(float)
        x[x == NEVER] = np.inf
        med[v] = float(np.median(x)) if x.size else float("nan")
        never[v] = int((ranks[e, vi] == NEVER).sum())
    quick = {}
    for a, b in QUICK_COMPARISONS:
        ia, ib = VARIANTS.index(a), VARIANTS.index(b)
        for k in (5, 10):
            ha = (ranks[e, ia] >= 1) & (ranks[e, ia] <= k)
            hb = (ranks[e, ib] >= 1) & (ranks[e, ib] <= k)
            mean, lo, hi = paired_bootstrap_ci(100.0 * (ha.astype(float) - hb.astype(float)), n_resamples=N_BOOT, seed=SEED)
            only_a, only_b = int((ha & ~hb).sum()), int((hb & ~ha).sum())
            quick[f"{a}-{b}@{k}"] = {"mean": mean, "lo": lo, "hi": hi, "a_only": only_a, "b_only": only_b, "p": mcnemar_exact_p(only_a, only_b)}
    return {
        "dataset": d,
        "retriever": r,
        "n_queries": n,
        "n_eligible": int(e.sum()),
        "hr": hr,
        "median_rank": med,
        "never_hit": never,
        "all_three_equals_saved": n_same_saved,
        "hr_agrees_with_repo": agree,
        "quick": quick,
        "per_query_file": rel(out),
        "seconds": time.perf_counter() - t0,
    }


def body(rep: StepReport, default_retrievers: tuple[str, ...]) -> int:
    datasets = selected("C3_DATASETS", ALL_DATASETS)
    retrievers = tuple(r for r in selected("C3_RETRIEVERS", ALL_RETRIEVERS) if r in default_retrievers)
    w, problems = load_published_weights()
    if problems:
        rep.add("**Config inconsistent with the saved rankings:**", "", *[f"- {p}" for p in problems])
        return 1
    if not READY_PATH.exists():
        rep.add(f"`{rel(READY_PATH)}` is missing: run step 01 first.")
        return 1
    ready_doc = json.loads(READY_PATH.read_text(encoding="utf-8"))
    if ready_doc.get("weights") != w.as_dict():
        rep.add("Step 01 ran with other weights than the config now has; rerun step 01.")
        return 1
    todo = [x for x in ready_doc["ready"] if x["dataset"] in datasets and x["retriever"] in retrievers]
    todo.sort(key=lambda x: (ALL_RETRIEVERS.index(x["retriever"]), ALL_DATASETS.index(x["dataset"])))
    skipped = [x for x in ready_doc.get("not_ready", []) if x["dataset"] in datasets and x["retriever"] in retrievers]
    rep.add(
        "## Settings",
        "",
        f"- Weights from `configs/methods/sadi.yaml`: (lambda_T, lambda_M, lambda_S) = ({w.lambda_t:g}, {w.lambda_n:g}, {w.lambda_h:g}), "
        f"field weights ({w.mu_n:g}, {w.mu_a:g}, {w.mu_d:g}), {w.normalization_method}. A variant sets the dropped channel's weight to 0 "
        "in the published Equation 2; nothing else changes.",
        "- Rank = 1-based position over the full chunk pool at which the saved hit definition is first met (gold chunk, or the last "
        "chunk of a complete evidence group). Median over eligible queries; 'never' = no gold chunk or complete group in the pool.",
        f"- Paired differences in HR points: 95% bootstrap over queries ({N_BOOT} resamples, seed {SEED}); McNemar exact two-sided p.",
        f"- Pairs from `{rel(READY_PATH)}` (step 01, check 1 passed): {', '.join(job_name(x['dataset'], x['retriever']) for x in todo) or 'none'}.",
        "",
    )
    if skipped:
        rep.add("Not scored (did not pass step 01): " + "; ".join(f"{job_name(x['dataset'], x['retriever'])} ({x['reason']})" for x in skipped), "")
    if not todo:
        rep.add("No ready pair for this step. Run step 01 for these retrievers first.")
        return 1

    results, failed = [], False
    for entry in todo:
        try:
            log(f"variants for {job_name(entry['dataset'], entry['retriever'])}")
            res = score_pair(entry, w)
            results.append(res)
            rep.wrote(res["per_query_file"])
            log(f"  done in {res['seconds']:.0f}s")
        except Exception:
            import traceback

            failed = True
            tb = traceback.format_exc()
            rep.add(f"### {job_name(entry['dataset'], entry['retriever'])}: exception", "", "```", tb.rstrip()[-3000:], "```", "")
    rep.data["pairs"] = results

    rep.add("## Guards", "")
    rows_g = []
    for res in results:
        ok_saved = res["all_three_equals_saved"] == res["n_queries"]
        ok_hr = all(res["hr_agrees_with_repo"].values())
        failed |= not (ok_saved and ok_hr)
        rows_g.append(
            [pair_label(res["dataset"], res["retriever"]), f"{res['all_three_equals_saved']}/{res['n_queries']}", "yes" if ok_hr else "NO", res["n_eligible"], f"{res['seconds']:.0f} s"]
        )
    rep.add(*table(["Pair", "All three = saved top-10", "Rank HR = repo HR (all variants, k=5,10)", "Eligible queries", "Time"], rows_g), "")

    rep.add("## HR@5 / HR@10 (%) and median rank", "")
    rows_h = []
    for res in results:
        for v in VARIANTS:
            m = res["median_rank"][v]
            rows_h.append(
                [
                    pair_label(res["dataset"], res["retriever"]),
                    VARIANT_LABEL[v],
                    f"{res['hr'][v]['5']:.2f}",
                    f"{res['hr'][v]['10']:.2f}",
                    "never" if m == float("inf") else f"{m:g}",
                    res["never_hit"][v],
                ]
            )
    rep.add(*table(["Pair", "Variant", "HR@5", "HR@10", "Median rank", "Never hit"], rows_h), "")
    rep.add("Summary alone and memory alone are for information only: a chunk without that channel falls back to its text score (counts in step 01).", "")

    rep.add("## Paired differences at HR@10 (first look; step 04 has the full set)", "")
    rows_q = []
    for res in results:
        for a, b in QUICK_COMPARISONS:
            x = res["quick"][f"{a}-{b}@10"]
            rows_q.append(
                [
                    pair_label(res["dataset"], res["retriever"]),
                    f"{VARIANT_LABEL[a]} - {VARIANT_LABEL[b]}",
                    f"{x['mean']:+.2f}",
                    f"[{x['lo']:+.2f}, {x['hi']:+.2f}]",
                    f"{x['a_only']} / {x['b_only']}",
                    f"{x['p']:.3g}",
                ]
            )
    rep.add(*table(["Pair", "Comparison", "Diff (pp)", "95% CI", "Hit only in first / only in second", "McNemar p"], rows_q), "")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, required=True, choices=(2, 3))
    args, _ = parser.parse_known_args()
    if args.step == 2:
        title, sh, default = "variants for BM25", "02_bm25.sh", ("bm25",)
    else:
        title, sh, default = "variants for the dense retrievers", "03_dense.sh", tuple(r for r in ALL_RETRIEVERS if r != "bm25")
    cmd = f"bash experiments/rebuttal/c3_channel_ablation/sh/{sh}"
    if os.environ.get("C3_RETRIEVERS_USER"):
        cmd = f"C3_RETRIEVERS={os.environ['C3_RETRIEVERS_USER']} " + cmd
    rep = StepReport(args.step, title, cmd)
    return run_step(rep, lambda r: body(r, default))


if __name__ == "__main__":
    sys.exit(main())
