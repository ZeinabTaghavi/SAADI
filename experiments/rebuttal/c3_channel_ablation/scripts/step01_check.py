#!/usr/bin/env python3
"""C3 step 01: find the inputs and run the two checks that come before any result.

1. Inputs. For every dataset x retriever pair: the paper's saved rankings, and the
   per-chunk channel scores. BM25 scores are recomputed here on CPU from the published
   construction caches (C2's ``compute_job_streams``; written to work/streams/). Dense
   scores are only read, from C3_STREAMS_DIR (default: C2's streams); a pair without them
   is reported as "waiting for channel scores" with the exact missing files.
2. Check 1. "All three" (published Equation 2, published weights from the config) must give
   exactly the paper's saved top-10 for every query of the pair; HR hits must equal
   expected_metrics.json. A pair with any mismatch fails and is not used later; the report
   shows its first 5 mismatches with the fused scores.
3. Check 2. "Text only" against Table 1's Vanilla (reported to one decimal), against the
   saved per-query text-only rankings where they exist, and against C2's text-only top-10.
   Reported, with the reason when it differs; it does not stop the run.

Writes work/ready_pairs.json (the pairs steps 02/03 may use) and reports/step_01.{md,json}.
Exit code 0: every pair that has channel scores passed check 1. 1: a pair failed, or the
config is inconsistent, or an exception (the report still gets written).
"""

from __future__ import annotations

import argparse
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
    BM25_STREAMS_DIR,
    CONFIG_PATH,
    DATASET_DISPLAY,
    DATASETS,
    EXPECTED_TOTAL_QUERIES,
    REPO_ROOT,
    RETRIEVE_K,
    RETRIEVER_DISPLAY,
    STREAMS_DIR,
    STREAMS_MARKERS_DIR,
    VANILLA_DATASET_NAME,
    VARIANT_LABEL,
    VARIANTS,
    WORK_DIR,
    StepReport,
    _hr,
    _rebuild_for_repo,
    combine,
    compute_bm25_streams,
    expected_main_metrics,
    input_streams,
    job_name,
    load_pair,
    load_published_weights,
    n_saved_rows,
    per_query_vanilla_file,
    published_vanilla,
    rel,
    run_step,
    selected,
    sha256_file,
    stream_files,
    table,
    variant_lambdas,
    write_json,
)

READY_PATH = WORK_DIR / "ready_pairs.json"
N_EXAMPLES = 5


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pair_label(d: str, r: str) -> str:
    return f"{DATASET_DISPLAY[d]} / {RETRIEVER_DISPLAY[r]}"


def short(cid: str) -> str:
    return str(cid).rsplit(":", 1)[-1]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def preserved_sources_check() -> dict[str, Any]:
    path = REPO_ROOT / "docs" / "preserved_sources.json"
    entries = json.loads(path.read_text(encoding="utf-8"))
    changed, missing = [], []
    for e in entries:
        p = REPO_ROOT / e["path"]
        if not p.exists():
            missing.append(e["path"])
        elif sha256_file(p) != e["sha256"]:
            changed.append(e["path"])
    return {"n_files": len(entries), "changed": changed, "missing": missing}


def inventory(datasets: tuple[str, ...], retrievers: tuple[str, ...]) -> list[dict[str, Any]]:
    out = []
    for d in datasets:
        spec = DATASETS[d]
        for r in retrievers:
            saved = REPO_ROOT / spec.saved_results_json(r)
            item: dict[str, Any] = {
                "dataset": d,
                "retriever": r,
                "saved_rankings": rel(saved),
                "saved_exists": saved.exists(),
                "n_saved": n_saved_rows(d, r) if saved.exists() else None,
                "missing": [],
            }
            if not saved.exists():
                item["missing"].append(rel(saved))
            if r == "bm25":
                idx = REPO_ROOT / spec.index_json
                item["source"] = "recomputed here (CPU) from the construction cache"
                item["cache_index"] = rel(idx)
                if not idx.exists():
                    item["missing"].append(rel(idx))
                item["status"] = "to compute" if not item["missing"] else "missing inputs"
            else:
                files = stream_files(STREAMS_DIR, d, r)
                job = job_name(d, r)
                item["source"] = f"read from {rel(STREAMS_DIR)}"
                item["stream_files"] = [rel(files.parquet), rel(files.queries)]
                if files.present and not item["missing"]:
                    item["status"] = "channel scores found"
                    if files.check.exists():
                        c2 = json.loads(files.check.read_text(encoding="utf-8"))
                        item["c2_check"] = {
                            "identical_top10": c2.get("identical_top10"),
                            "n_queries": c2.get("n_queries"),
                            "drift_accepted": c2.get("drift_accepted"),
                        }
                else:
                    item["missing"].extend(files.missing())
                    if (STREAMS_MARKERS_DIR / f"{job}.FAILED").exists():
                        why = "C2 job FAILED its reproduction gate (its quarantined copy in streams_failed/ is not used)"
                    elif (STREAMS_MARKERS_DIR / f"{job}.DONE").exists():
                        why = "C2 marker says DONE but the stream files are not in C3_STREAMS_DIR"
                    else:
                        why = "no C2 marker for this job"
                    item["status"] = "waiting for channel scores"
                    item["why"] = why
            out.append(item)
    return out


def compare_streams(ours: Path, theirs: Path) -> dict[str, Any]:
    """BM25 recomputed here vs C2's BM25 streams (same code, so they should be identical)."""

    import pandas as pd

    a, b =pd.read_parquet(ours), pd.read_parquet(theirs)
    out: dict[str, Any] = {"rows_here": len(a), "rows_c2": len(b)}
    if len(a) != len(b):
        out["identical"] = False
        return out
    same_ids = bool((a["chunk_id"].astype(str).to_numpy() == b["chunk_id"].astype(str).to_numpy()).all())
    same_q = bool((a["query_id"].astype(str).to_numpy() == b["query_id"].astype(str).to_numpy()).all())
    chans = {}
    for c in ("text", "name", "alias", "description", "summary"):
        chans[c] = bool(np.array_equal(a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float), equal_nan=True))
    out.update({"same_chunk_ids": same_ids, "same_query_ids": same_q, "channels_identical": chans})
    out["identical"] = same_ids and same_q and all(chans.values())
    return out


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_pair(d: str, r: str, w) -> dict[str, Any]:
    files = input_streams(d, r)
    t0 = time.perf_counter()
    queries, meta, rows = load_pair(d, r, files)
    n_saved = n_saved_rows(d, r)
    all_top, all_scores, text_top = [], [], []
    for q in queries:
        index, streams = _rebuild_for_repo(q, r)
        a = combine(index, streams, w, "all_three")
        t = combine(index, streams, w, "text_only")
        all_top.append([x.chunk_id for x in a.ranked_chunks[:RETRIEVE_K]])
        all_scores.append([float(x.final_score) for x in a.ranked_chunks[:RETRIEVE_K]])
        text_top.append([x.chunk_id for x in t.ranked_chunks[:RETRIEVE_K]])
    saved_top = [[str(x) for x in row.get("retrieved_chunk_ids", [])][:RETRIEVE_K] for row in rows]
    identical = [x == y for x, y in zip(all_top, saved_top, strict=True)]

    examples, score_diffs = [], []
    for i, same in enumerate(identical):
        row = rows[i]
        if same:
            if row.get("scores"):
                score_diffs.extend(abs(a - float(b)) for a, b in zip(all_scores[i], row["scores"], strict=False))
            continue
        if len(examples) < N_EXAMPLES:
            first = next(j for j in range(max(len(all_top[i]), len(saved_top[i]))) if j >= len(all_top[i]) or j >= len(saved_top[i]) or all_top[i][j] != saved_top[i][j])
            examples.append(
                {
                    "query_id": str(row.get("query_id")),
                    "doc_id": str(row.get("doc_id")),
                    "first_differing_rank": first + 1,
                    "ours_top10": all_top[i],
                    "ours_scores": all_scores[i],
                    "saved_top10": saved_top[i],
                    "saved_scores": [float(s) for s in (row.get("scores") or [])[:RETRIEVE_K]],
                }
            )

    hr_ours, hr_saved = _hr(rows, all_top), _hr(rows, saved_top)
    expected = expected_main_metrics(d, r, REPO_ROOT)
    exp_rows = {}
    exp_ok = bool(expected)
    for k in (5, 10):
        e = expected.get(k)
        if e is None:
            exp_ok = False
            continue
        exp_rows[str(k)] = {"eligible": int(e["eligible"]), "hits": int(e["hits"])}
        if int(e["eligible"]) != hr_ours["eligible"] or int(e["hits"]) != int(hr_ours["hits"][str(k)]):
            exp_ok = False
    coverage = len(queries) == n_saved
    check1 = {
        "n_queries_streams": len(queries),
        "n_queries_saved": n_saved,
        "complete": coverage,
        "identical_top10": int(sum(identical)),
        "hr_all_three": hr_ours,
        "hr_saved": hr_saved,
        "expected_metrics": exp_rows,
        "expected_hits_match": exp_ok,
        "max_abs_score_diff_identical": max(score_diffs) if score_diffs else None,
        "mismatch_examples": examples,
        "passed": bool(coverage and all(identical) and exp_ok),
    }

    # Check 2: text only.
    hr_text = _hr(rows, text_top)
    van = published_vanilla(REPO_ROOT)
    vt: dict[str, Any] = {"hr_text_only": hr_text, "table1_vanilla": {}, "diff_points": {}, "equal_to_table1": True}
    for k in (5, 10):
        ref = van.get((VANILLA_DATASET_NAME[d], r, k))
        ours = hr_text["hr_percent"][str(k)]
        vt["table1_vanilla"][str(k)] = ref
        if ref is None or ours is None:
            vt["equal_to_table1"] = None
            continue
        vt["diff_points"][str(k)] = ours - ref
        if abs(ours - ref) > 0.05 + 1e-9:  # Table 1 has one decimal
            vt["equal_to_table1"] = False
    pq = per_query_vanilla_file(d, r, REPO_ROOT)
    if pq is not None:
        saved_text = {
            (str(x.get("doc_id")), str(x.get("query_id"))): [str(c) for c in x.get("retrieved_chunk_ids", [])][:RETRIEVE_K]
            for x in json.loads(pq.read_text(encoding="utf-8"))
        }
        vt["per_query_file"] = rel(pq)
        vt["per_query_identical"] = sum(
            1 for row, top in zip(rows, text_top, strict=True) if saved_text.get((str(row.get("doc_id")), str(row.get("query_id")))) == top
        )
    vt["c2_text_only_identical"] = sum(1 for m, top in zip(meta, text_top, strict=True) if list(m.get("vanilla_top10", [])) == top)
    vt["n"] = len(rows)

    n_chunks = [len(q.chunk_ids) for q in queries]
    missing_channel = {
        "summary": int(sum(int(np.isnan(q.raw[:, 4]).sum()) for q in queries)),
        "memory_all_fields": int(sum(int(np.isnan(q.raw[:, 1:4]).all(axis=1).sum()) for q in queries)),
        "chunk_query_pairs": int(sum(n_chunks)),
    }
    return {
        "dataset": d,
        "retriever": r,
        "stream_files": [rel(files.parquet), rel(files.queries)],
        "parquet_sha256": sha256_file(files.parquet),
        "queries_sha256": sha256_file(files.queries),
        "n_docs": len({q.doc_id for q in queries}),
        "chunks_per_query_median": float(np.median(n_chunks)) if n_chunks else None,
        "missing_channel_counts": missing_channel,
        "check1": check1,
        "check2": vt,
        "seconds": time.perf_counter() - t0,
    }


# ---------------------------------------------------------------------------
# Step body
# ---------------------------------------------------------------------------


def body(rep: StepReport) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="recompute the BM25 channel scores even if work/streams has them")
    args = parser.parse_args()
    datasets = selected("C3_DATASETS", ALL_DATASETS)
    retrievers = selected("C3_RETRIEVERS", ALL_RETRIEVERS)
    failed = False

    # --- Settings -----------------------------------------------------------
    w, problems = load_published_weights()
    rep.data["weights"] = w.as_dict()
    rep.data["config"] = rel(CONFIG_PATH)
    rep.add(
        "## Settings",
        "",
        f"- Config: `{rel(CONFIG_PATH)}` (block `cached_retrieval`). Fusion weights (lambda_T, lambda_M, lambda_S) = "
        f"({w.lambda_t:g}, {w.lambda_n:g}, {w.lambda_h:g}); field weights (mu_N, mu_A, mu_D) = ({w.mu_n:g}, {w.mu_a:g}, {w.mu_d:g}); "
        f"normalization {w.normalization_method}; memory aggregation {w.memory_aggregation}.",
        "- A variant drops a channel by setting its weight to 0 in the published Equation 2 "
        "(`combine_mbr_variant_chunk_scores`, MBR2). The remaining weights are not re-tuned or rescaled; Equation 2's own "
        "per-chunk division by the sum of the weights present is unchanged.",
        f"- Dense channel scores read from `C3_STREAMS_DIR` = `{rel(STREAMS_DIR)}`; C2 markers from `{rel(STREAMS_MARKERS_DIR)}`.",
        f"- BM25 channel scores written to `{rel(BM25_STREAMS_DIR)}`.",
        f"- Datasets: {', '.join(datasets)}. Retrievers: {', '.join(retrievers)}.",
        "",
    )
    rep.add(*table(["Variant", "lambda_T", "lambda_M", "lambda_S"], [[VARIANT_LABEL[v], *variant_lambdas(w, v)] for v in VARIANTS]))
    rep.add("", "Summary alone and memory alone are for information only (steps 02-04).", "")
    if problems:
        rep.add("**Config inconsistent with the saved rankings; stopping before any check:**", "", *[f"- {p}" for p in problems], "")
        rep.data["config_problems"] = problems
        return 1

    pres = preserved_sources_check()
    rep.data["preserved_sources"] = pres
    rep.add(
        f"- Preserved sources (`docs/preserved_sources.json`): {pres['n_files']} files, {len(pres['changed'])} changed, "
        f"{len(pres['missing'])} missing." + ("" if not (pres["changed"] or pres["missing"]) else " **Not by C3; check before going on.**"),
        "",
    )
    for p in pres["changed"][:10]:
        rep.add(f"  - changed: `{p}`")

    # --- Inputs ------------------------------------------------------------
    inv = inventory(datasets, retrievers)
    rep.data["inventory"] = inv
    rep.add("## 1. Inputs", "")
    rep.add(
        *table(
            ["Pair", "Saved rankings (queries)", "Channel scores", "Status"],
            [[pair_label(i["dataset"], i["retriever"]), i["n_saved"] if i["saved_exists"] else "MISSING", i["source"], i["status"]] for i in inv],
        )
    )
    waiting = [i for i in inv if i["status"] != "channel scores found" and i["retriever"] != "bm25"]
    if waiting:
        rep.add("", "Waiting for channel scores (exact missing files):", "")
        for i in waiting:
            rep.add(f"- {pair_label(i['dataset'], i['retriever'])}: {i.get('why', '')}")
            for m in i["missing"]:
                rep.add(f"  - `{m}`")
    per_ds = {}
    for d in datasets:
        first = next((i for i in inv if i["dataset"] == d and i["saved_exists"]), None)
        per_ds[d] = first["n_saved"] if first else None
    total = sum(v for v in per_ds.values() if v)
    rep.data["queries_per_dataset"] = per_ds
    rep.add(
        "",
        "Queries per dataset (saved rankings): " + ", ".join(f"{DATASET_DISPLAY[d]} {n}" for d, n in per_ds.items()) + f"; total {total}"
        + (f" (E2: {EXPECTED_TOTAL_QUERIES})." if set(datasets) == set(ALL_DATASETS) else "."),
    )
    unequal = [
        pair_label(i["dataset"], i["retriever"]) for i in inv if i["saved_exists"] and per_ds.get(i["dataset"]) not in (None, i["n_saved"])
    ]
    if unequal:
        rep.add(f"- Different query counts across retrievers: {', '.join(unequal)}")
    rep.add("")

    # --- BM25 channel scores ------------------------------------------------
    bm25_runs: dict[str, Any] = {}
    if "bm25" in retrievers:
        rep.add("## 2. BM25 channel scores (CPU)", "")
        rows_t = []
        for d in datasets:
            item = next(i for i in inv if i["dataset"] == d and i["retriever"] == "bm25")
            if item["missing"]:
                bm25_runs[d] = {"status": "missing inputs", "missing": item["missing"]}
                rows_t.append([DATASET_DISPLAY[d], "missing inputs", "-", "-", "-"])
                failed = True
                continue
            try:
                log(f"BM25 channel scores for {d}")
                res = compute_bm25_streams(d, force=args.force, log=log)
                for f in res["files"]:
                    rep.wrote(f)
                theirs = stream_files(STREAMS_DIR, d, "bm25")
                if theirs.present:
                    res["vs_c2_streams"] = compare_streams(BM25_STREAMS_DIR / f"{job_name(d, 'bm25')}.parquet", theirs.parquet)
                bm25_runs[d] = res
                vs = res.get("vs_c2_streams")
                rows_t.append(
                    [
                        DATASET_DISPLAY[d],
                        res["status"],
                        res.get("n_queries", "-"),
                        f"{res['seconds']:.0f} s" if "seconds" in res else "-",
                        "-" if vs is None else ("identical" if vs["identical"] else f"DIFFERENT: {vs}"),
                    ]
                )
            except Exception as exc:  # reported per dataset; the other datasets still run
                import traceback

                failed = True
                tb = traceback.format_exc()
                bm25_runs[d] = {"status": "exception", "traceback": tb}
                rows_t.append([DATASET_DISPLAY[d], f"exception: {type(exc).__name__}", "-", "-", "-"])
                rep.add("```", tb.rstrip(), "```", "")
        rep.add(*table(["Dataset", "Status", "Queries", "Time", "Same as C2's BM25 streams"], rows_t), "")
    rep.data["bm25_streams"] = bm25_runs

    # --- Checks -------------------------------------------------------------
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for i in inv:
        d, r = i["dataset"], i["retriever"]
        if not input_streams(d, r).present or not i["saved_exists"]:
            continue
        try:
            log(f"checks for {job_name(d, r)}")
            res = check_pair(d, r, w)
            results.append(res)
            log(f"  {job_name(d, r)}: identical top-10 {res['check1']['identical_top10']}/{res['check1']['n_queries_saved']}, {res['seconds']:.0f}s")
        except Exception:
            import traceback

            failed = True
            errors.append({"pair": job_name(d, r), "traceback": traceback.format_exc()})
    rep.data["checks"] = results
    rep.data["check_errors"] = errors

    rep.add("## 3. Check 1: all three reproduces the paper's saved rankings", "")
    rows_c = []
    for res in results:
        c = res["check1"]
        h, s = c["hr_all_three"]["hr_percent"], c["hr_saved"]["hr_percent"]
        rows_c.append(
            [
                pair_label(res["dataset"], res["retriever"]),
                f"{c['n_queries_streams']}/{c['n_queries_saved']}",
                f"{c['identical_top10']}/{c['n_queries_saved']}",
                f"{h['5']:.2f} / {s['5']:.2f}",
                f"{h['10']:.2f} / {s['10']:.2f}",
                "yes" if c["expected_hits_match"] else "NO",
                "-" if c["max_abs_score_diff_identical"] is None else f"{c['max_abs_score_diff_identical']:.1e}",
                "PASS" if c["passed"] else "FAIL",
            ]
        )
    rep.add(
        *table(
            ["Pair", "Queries scored", "Identical top-10", "HR@5 ours / saved", "HR@10 ours / saved", "Hits = expected_metrics", "Max abs score diff", "Check 1"],
            rows_c,
        )
    )
    n_pass = sum(1 for res in results if res["check1"]["passed"])
    rep.add("", f"Matching queries in total: {sum(res['check1']['identical_top10'] for res in results)} of "
            f"{sum(res['check1']['n_queries_saved'] for res in results)} ({n_pass} of {len(results)} pairs pass).", "")
    for res in results:
        c = res["check1"]
        if c["passed"]:
            continue
        failed = True
        rep.add(f"### {pair_label(res['dataset'], res['retriever'])}: first {len(c['mismatch_examples'])} mismatches", "")
        if not c["complete"]:
            rep.add(f"- The channel scores cover {c['n_queries_streams']} of {c['n_queries_saved']} saved queries.")
        for e in c["mismatch_examples"]:
            rep.add(
                f"- query {e['query_id']} (doc `{e['doc_id']}`), first difference at rank {e['first_differing_rank']}:",
                f"  - ours : {' '.join(short(x) for x in e['ours_top10'])} | " + " ".join(f"{s:.4f}" for s in e["ours_scores"]),
                f"  - saved: {' '.join(short(x) for x in e['saved_top10'])} | " + " ".join(f"{s:.4f}" for s in e["saved_scores"]),
            )
        rep.add("")
    for e in errors:
        rep.add(f"### {e['pair']}: exception", "", "```", e["traceback"].rstrip()[-3000:], "```", "")

    rep.add("## 4. Check 2: text only against the paper's Vanilla", "")
    rows_v = []
    differs = []
    for res in results:
        v = res["check2"]
        h = v["hr_text_only"]["hr_percent"]
        t1 = v["table1_vanilla"]
        eq = v["equal_to_table1"]
        if eq is False:
            differs.append(res["dataset"])
        rows_v.append(
            [
                pair_label(res["dataset"], res["retriever"]),
                f"{h['5']:.2f} / {h['10']:.2f}",
                f"{t1['5']} / {t1['10']}",
                "yes" if eq else ("NO" if eq is False else "-"),
                f"{v['per_query_identical']}/{v['n']}" if "per_query_file" in v else "no file",
                f"{v['c2_text_only_identical']}/{v['n']}",
            ]
        )
    rep.add(
        *table(
            ["Pair", "Text only HR@5 / HR@10", "Table 1 Vanilla HR@5 / HR@10", "Equal to 1 decimal", "Saved per-query text-only file identical", "Same as C2 text-only"],
            rows_v,
        ),
        "",
    )
    diff_ds = sorted(set(differs), key=ALL_DATASETS.index)
    if diff_ds:
        expected_diff = {"loogle", "novelhopqa"}
        rep.add(
            f"Text only differs from Table 1's Vanilla on: {', '.join(DATASET_DISPLAY[d] for d in diff_ds)}. Why: Table 1's Vanilla is a "
            "reported summary (`outputs/other_methods/vanilla/retrieval.csv`, source `main_fixed_depth` = `outputs/Tables/Final_table.txt`) "
            "with no per-query rankings. On LooGLE and NovelHopQA (500-token chunks) it comes from a separate run, not from the text channel "
            "of SAADI's run on the same chunks; where the saved per-query text-only rankings on SAADI's chunks exist (BM25, LooGLE and "
            "NovelHopQA: `retrieval_results__bm25__mbr_lt1_ln0_lh0_norm_none__per_document.json`), they equal C3's text only query by query "
            "(column 5). C3 uses the text channel on SAADI's own chunks as 'text only', so every comparison is paired on the same chunks; "
            "Table 1's Vanilla is shown next to it in the final report.",
        )
        if set(diff_ds) - expected_diff:
            rep.add(
                "",
                f"**Unexpected:** a difference on {', '.join(DATASET_DISPLAY[d] for d in sorted(set(diff_ds) - expected_diff, key=ALL_DATASETS.index))}, "
                "where Table 1's Vanilla was the text channel in C2. Check this before step 02.",
            )
        rep.add(
            "",
            "Table 1 reports LooGLE/Contriever Vanilla as 72.4 / 73.3 and NovelHopQA/Contriever as 11.8 / 11.8 (HR@5 about equal "
            "to HR@10), which suggests a transcription problem in that table row; it does not affect C3.",
        )
    else:
        rep.add("Text only equals Table 1's Vanilla (to one decimal) on every pair checked.")
    rep.add("")

    # --- Missing channels (for the 'alone' variants) ------------------------
    rep.add("## 5. Chunks without a summary or memory score", "")
    rep.add(
        "Equation 2 renormalizes over the channels a chunk has. In 'summary alone' and 'memory alone' a chunk without that channel "
        "has no weight left, and the published code then falls back to its text score. Counts over (query, chunk) pairs:",
        "",
    )
    rep.add(
        *table(
            ["Pair", "Median chunks per query", "No summary score", "No memory field", "(query, chunk) pairs"],
            [
                [
                    pair_label(res["dataset"], res["retriever"]),
                    res["chunks_per_query_median"],
                    res["missing_channel_counts"]["summary"],
                    res["missing_channel_counts"]["memory_all_fields"],
                    res["missing_channel_counts"]["chunk_query_pairs"],
                ]
                for res in results
            ],
        ),
        "",
    )

    # --- Ready list ----------------------------------------------------------
    ready = [
        {
            "dataset": res["dataset"],
            "retriever": res["retriever"],
            "stream_files": res["stream_files"],
            "parquet_sha256": res["parquet_sha256"],
            "queries_sha256": res["queries_sha256"],
            "n_queries": res["check1"]["n_queries_saved"],
            "check2": {k: res["check2"][k] for k in ("hr_text_only", "table1_vanilla", "equal_to_table1")},
        }
        for res in results
        if res["check1"]["passed"]
    ]
    not_ready = [
        {"dataset": i["dataset"], "retriever": i["retriever"], "reason": i["status"] if i["status"] not in ("channel scores found", "to compute") else "check 1 failed or exception"}
        for i in inv
        if not any(x["dataset"] == i["dataset"] and x["retriever"] == i["retriever"] for x in ready)
    ]
    # Merge with an earlier run (e.g. BM25 first, dense later): pairs checked now replace their old entries.
    checked = {(i["dataset"], i["retriever"]) for i in inv}
    if READY_PATH.exists():
        prev = json.loads(READY_PATH.read_text(encoding="utf-8"))
        if prev.get("weights") == w.as_dict():
            ready = [x for x in prev.get("ready", []) if (x["dataset"], x["retriever"]) not in checked] + ready
            not_ready = [x for x in prev.get("not_ready", []) if (x["dataset"], x["retriever"]) not in checked] + not_ready
    write_json(READY_PATH, {"written_at": time.strftime("%Y-%m-%d %H:%M:%S"), "streams_dir": rel(STREAMS_DIR), "weights": w.as_dict(), "ready": ready, "not_ready": not_ready})
    rep.wrote(READY_PATH)
    rep.data["ready"] = [job_name(x["dataset"], x["retriever"]) for x in ready]
    rep.data["not_ready"] = not_ready
    rep.add("## 6. Next steps", "")
    bm = [pair_label(x["dataset"], x["retriever"]) for x in ready if x["retriever"] == "bm25"]
    de = [pair_label(x["dataset"], x["retriever"]) for x in ready if x["retriever"] != "bm25"]
    rep.add(
        f"- Step 02 (BM25) can run on: {', '.join(bm) if bm else 'none'}.",
        f"- Step 03 (dense) can run on: {', '.join(de) if de else 'none'}.",
        f"- Not used: " + ("; ".join(f"{pair_label(x['dataset'], x['retriever'])} ({x['reason']})" for x in not_ready) if not_ready else "none") + ".",
        "",
    )
    return 1 if failed else 0


def main() -> int:
    cmd = "bash experiments/rebuttal/c3_channel_ablation/sh/01_check.sh" + ("" if len(sys.argv) < 2 else " " + " ".join(sys.argv[1:]))
    if os.environ.get("C3_STREAMS_DIR"):
        cmd = f"C3_STREAMS_DIR={os.environ['C3_STREAMS_DIR']} " + cmd
    rep = StepReport(1, "inputs and checks", cmd)
    return run_step(rep, body)


if __name__ == "__main__":
    sys.exit(main())
