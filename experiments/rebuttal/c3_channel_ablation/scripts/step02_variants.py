#!/usr/bin/env python3
"""C3 steps 02 (BM25) and 03 (dense): the new variants, next to Vanilla and SAADI as saved.

Not re-scored:
- all three (SAADI) = the paper's saved rankings (the files Table 1 was computed from);
- text only (Vanilla) = the text-only top-10 saved with the channel scores (C2's stream step,
  lambda_M = lambda_S = 0), checked against Table 1's Vanilla in step 01.
Scored here, on the pairs that passed step 01's check 1, by the published Equation 2
(``combine_mbr_variant_chunk_scores``) with the published weights and one channel weight set
to 0: text + summary, text + memory, and (for information) summary alone, memory alone.

Per query and variant: the 1-based rank within the top-10 at which the saved hit definition is
met (11 = not within the top-10), so HR@5 and HR@10 follow exactly; HR is also computed with the
repository's ``evaluate_retrieval_results`` and both must agree. Median rank over eligible queries
(">10" when fewer than half are hit within the top-10).

Inputs are only read; the stream files must have the sha256 step 01 recorded. Writes
results/rebuttal/c3_channel_ablation/per_query/<dataset>__<retriever>.csv.gz (small, committed, so
step 04 can combine laptop and server runs) and reports/step_NN.{md,json}.
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
    DATASET_DISPLAY,
    N_BOOT,
    NEW_VARIANTS,
    NOT_IN_TOP,
    REPO_ROOT,
    RESULTS_DIR,
    RETRIEVE_K,
    RETRIEVER_DISPLAY,
    SEED,
    VANILLA_DATASET_NAME,
    VARIANT_LABEL,
    VARIANT_SOURCE,
    VARIANTS,
    WORK_DIR,
    StepReport,
    StreamFiles,
    _hr,
    _rebuild_for_repo,
    combine,
    expected_main_metrics,
    job_name,
    load_pair,
    load_published_weights,
    mcnemar_exact_p,
    published_vanilla,
    rel,
    run_step,
    selected,
    sha256_file,
    table,
    top10_rank,
)
from saadi.rebuttal.frozen_records import paired_bootstrap_ci
from saadi.rebuttal.heldout_fusion import evidence_of

READY_PATH = WORK_DIR / "ready_pairs.json"
PER_QUERY_DIR = RESULTS_DIR / "per_query"

# Shown in this step's report as a first look; step 04 writes the full set.
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


def median_text(ranks: np.ndarray) -> str:
    if ranks.size == 0:
        return "-"
    m = float(np.median(ranks))
    return ">10" if m > RETRIEVE_K else f"{m:g}"


def score_pair(entry: dict[str, Any], w) -> dict[str, Any]:
    d, r = entry["dataset"], entry["retriever"]
    files = StreamFiles(REPO_ROOT / entry["stream_files"][0], REPO_ROOT / entry["stream_files"][1], Path("/nonexistent"))
    for p, key in ((files.parquet, "parquet_sha256"), (files.queries, "queries_sha256")):
        if sha256_file(p) != entry[key]:
            raise RuntimeError(f"{rel(p)} changed since step 01 (sha256 differs); rerun step 01")
    t0 = time.perf_counter()
    queries, meta, rows = load_pair(d, r, files)
    n = len(queries)

    tops: dict[str, list[list[str]]] = {v: [] for v in VARIANTS}
    tops["all_three"] = [[str(x) for x in row.get("retrieved_chunk_ids", [])][:RETRIEVE_K] for row in rows]
    tops["text_only"] = [list(m["vanilla_top10"])[:RETRIEVE_K] for m in meta]
    for q in queries:
        index, streams = _rebuild_for_repo(q, r)
        for v in NEW_VARIANTS:
            tops[v].append([x.chunk_id for x in combine(index, streams, w, v).ranked_chunks[:RETRIEVE_K]])

    ranks = np.full((n, len(VARIANTS)), NOT_IN_TOP, dtype=np.int64)
    eligible = np.zeros(n, dtype=bool)
    for qi, row in enumerate(rows):
        gold, _silver, groups = evidence_of(row)
        groups = [g for g in groups if g]
        eligible[qi] = bool(gold) or bool(groups)
        for vi, v in enumerate(VARIANTS):
            ranks[qi, vi] = top10_rank(gold, groups, tops[v][qi])

    # Guard: rank-based HR equals the repository's HR for every variant.
    hr: dict[str, dict[str, float]] = {}
    agree = {}
    for vi, v in enumerate(VARIANTS):
        repo = _hr(rows, tops[v])
        hr[v] = {}
        for k in (5, 10):
            hits = int(((ranks[:, vi] <= k) & eligible).sum())
            agree[f"{v}@{k}"] = hits == int(repo["hits"][str(k)]) and int(eligible.sum()) == int(repo["eligible"])
            hr[v][str(k)] = 100.0 * hits / max(1, int(eligible.sum()))

    PER_QUERY_DIR.mkdir(parents=True, exist_ok=True)
    out = per_query_path(d, r)
    tmp = out.with_name(out.name + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", newline="") as handle:
        wr = csv.writer(handle)
        wr.writerow(["dataset", "retriever", "query_id", "doc_id", "eligible", *[f"rank_{v}" for v in VARIANTS]])
        for qi, m in enumerate(meta):
            wr.writerow([d, r, m["query_id"], m["doc_id"], int(eligible[qi]), *ranks[qi].tolist()])
    os.replace(tmp, out)

    e = eligible
    med = {v: median_text(ranks[e, vi]) for vi, v in enumerate(VARIANTS)}
    quick = {}
    for a, b in QUICK_COMPARISONS:
        ia, ib = VARIANTS.index(a), VARIANTS.index(b)
        for k in (5, 10):
            ha, hb = ranks[e, ia] <= k, ranks[e, ib] <= k
            mean, lo, hi = paired_bootstrap_ci(100.0 * (ha.astype(float) - hb.astype(float)), n_resamples=N_BOOT, seed=SEED)
            only_a, only_b = int((ha & ~hb).sum()), int((hb & ~ha).sum())
            quick[f"{a}-{b}@{k}"] = {"mean": mean, "lo": lo, "hi": hi, "a_only": only_a, "b_only": only_b, "p": mcnemar_exact_p(only_a, only_b)}

    # The paper's numbers: Table 1 Vanilla (one decimal) and SAADI (expected_metrics.json, main/default).
    van = published_vanilla(REPO_ROOT)
    exp = expected_main_metrics(d, r, REPO_ROOT)
    paper = {
        "vanilla": {str(k): van.get((VANILLA_DATASET_NAME[d], r, k)) for k in (5, 10)},
        "saadi": {str(k): (100.0 * float(exp[k]["hr"]) if k in exp else None) for k in (5, 10)},
    }
    return {
        "dataset": d,
        "retriever": r,
        "n_queries": n,
        "n_eligible": int(e.sum()),
        "hr": hr,
        "median_rank": med,
        "hr_agrees_with_repo": agree,
        "quick": quick,
        "paper": paper,
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
        f"field weights ({w.mu_n:g}, {w.mu_a:g}, {w.mu_d:g}), {w.normalization_method}. A new variant sets the dropped channel's weight "
        "to 0 in the published Equation 2; nothing else changes.",
        "- Where each variant comes from: " + "; ".join(f"{VARIANT_LABEL[v]}: {VARIANT_SOURCE[v]}" for v in VARIANTS) + ".",
        "- Rank = 1-based position within the top-10 at which the saved hit definition is met (gold chunk, or every chunk of one "
        "evidence group); median over eligible queries, '>10' when fewer than half are hit within the top-10.",
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
            log(f"new variants for {job_name(entry['dataset'], entry['retriever'])}")
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

    rep.add("## Vanilla and SAADI against the paper", "")
    rows_p = []
    for res in results:
        p = res["paper"]
        rows_p.append(
            [
                pair_label(res["dataset"], res["retriever"]),
                f"{res['hr']['text_only']['5']:.2f} / {res['hr']['text_only']['10']:.2f}",
                f"{p['vanilla']['5']} / {p['vanilla']['10']}",
                f"{res['hr']['all_three']['5']:.2f} / {res['hr']['all_three']['10']:.2f}",
                f"{p['saadi']['5']:.2f} / {p['saadi']['10']:.2f}" if p["saadi"]["5"] is not None else "-",
            ]
        )
    rep.add(
        *table(["Pair", "Text only HR@5 / HR@10 (here)", "Table 1 Vanilla", "All three HR@5 / HR@10 (here)", "Paper SAADI"], rows_p),
        "",
        "Text only differs from Table 1's Vanilla on LooGLE and NovelHopQA (a separate run in the paper; step 01 explains it). "
        "All three is the paper's saved run itself, so it equals the paper's SAADI numbers.",
        "",
    )

    rep.add("## Guards", "")
    rows_g = []
    for res in results:
        ok_hr = all(res["hr_agrees_with_repo"].values())
        failed |= not ok_hr
        rows_g.append([pair_label(res["dataset"], res["retriever"]), "yes" if ok_hr else "NO", res["n_eligible"], f"{res['seconds']:.0f} s"])
    rep.add(*table(["Pair", "Rank HR = repo HR (all variants, k=5,10)", "Eligible queries", "Time"], rows_g), "")

    rep.add("## HR@5 / HR@10 (%) and median rank, all variants", "")
    rows_h = []
    for res in results:
        for v in VARIANTS:
            rows_h.append([pair_label(res["dataset"], res["retriever"]), VARIANT_LABEL[v], f"{res['hr'][v]['5']:.2f}", f"{res['hr'][v]['10']:.2f}", res["median_rank"][v]])
    rep.add(*table(["Pair", "Variant", "HR@5", "HR@10", "Median rank"], rows_h), "")
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
        title, sh, default = "new variants for BM25", "02_bm25.sh", ("bm25",)
    else:
        title, sh, default = "new variants for the dense retrievers", "03_dense.sh", tuple(r for r in ALL_RETRIEVERS if r != "bm25")
    cmd = f"bash experiments/rebuttal/c3_channel_ablation/sh/{sh}"
    if os.environ.get("C3_RETRIEVERS_USER"):
        cmd = f"C3_RETRIEVERS={os.environ['C3_RETRIEVERS_USER']} " + cmd
    rep = StepReport(args.step, title, cmd)
    return run_step(rep, lambda r: body(r, default))


if __name__ == "__main__":
    sys.exit(main())
