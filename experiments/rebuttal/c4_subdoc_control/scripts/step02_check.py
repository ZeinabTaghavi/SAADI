#!/usr/bin/env python3
"""C4 step 02: are the inputs complete, and does the full ranking reproduce the paper?

For every (setting, retriever) of the paper on the concatenated settings:
  1. Inventory: what the paper saved (top-k lists only, or full per-chunk scores).
  2. Full per-chunk scores ("streams", C2 format): BM25 is re-scored here on CPU from the
     saved construction caches (resumable; nothing rebuilt, no LLM). Dense retrievers are
     read from $C4_DENSE_STREAMS_DIR; when absent the pair is "waiting for scores" and the
     exact missing files are listed.
  3. Reproduction gate on the full (unrestricted) ranking, with the published weights read
     from configs/methods/sadi.yaml:
       SAADI   top-k == every saved paper ranking of the pair, per query; HR@5/HR@10 hits
               and eligible counts == the paper's expected counts;
       Vanilla (text channel alone, lambda_M = lambda_S = 0) top-10 == the repository's own
               combination at stream time; its HR next to the paper's Vanilla (aggregate only:
               no per-query Vanilla ranking is saved for these settings), reported, not gated;
       the vectorised fusion == combine_mbr_variant_chunk_scores (full rankings) on 20
               seeded queries per pair, for both systems.
     BM25 must match exactly. Dense pairs follow C2's accepted rule (identical top-10 for at
     least 99% of queries and HR within 0.1 points; Jina's embedding drift accepted when HR
     stays within 0.5 points); every non-identical query is counted and the first 5 shown.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from c4_common import (
    DENSE_STREAMS_DIR,
    MAIN_CONDITION,
    MAP_DIR,
    ROOT,
    SEED,
    Setting,
    StepReport,
    fnum,
    load_ranking_rows,
    paper_sources,
    published_settings,
    qid_sort_key,
    read_json,
    rel,
    require_step_ok,
    run_step,
    selected_retrievers,
    selected_settings,
)
from c4_streams import BM25_STREAMS_DIR, compute_bm25_streams, compute_streams, job_name, labels_equal, saved_rows, stream_paths

from saadi.rebuttal.heldout_fusion import (  # noqa: E402
    BIG_RANK,
    DENSE_HR_TOLERANCE_POINTS,
    DENSE_MIN_IDENTICAL_TOP10,
    DRIFT_ACCEPTED_RETRIEVERS,
    DRIFT_MAX_HR_POINTS,
    final_scores,
    hit_ranks,
    load_query_arrays,
    ranking_order,
    repo_ranking,
)

N_REPO_CHECK = 20


def paper_vanilla() -> dict[tuple[str, str, str, int], float]:
    """The paper's Vanilla HR (percent): Table 26 (context_expansion) and Table 1 (main)."""

    path = ROOT / "outputs" / "other_methods" / "vanilla" / "retrieval.csv"
    out: dict[tuple[str, str, str, int], float] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("method") != "vanilla" or row.get("hop") != "overall":
                continue
            if row.get("study") == "context_expansion":
                out[("table26", row["dataset"], row["condition"], row["retriever"], int(row["cutoff"]))] = float(row["hr_percent"])
            elif row.get("study") == "main" and row.get("source_id") == "main_fixed_depth":
                out[("table1", row["dataset"], "", row["retriever"], int(row["cutoff"]))] = float(row["hr_percent"])
    return out


def _bm25_job(key: str, paper_qids: list[str], force: bool) -> dict[str, Any]:
    from c4_common import ALL_SETTINGS

    setting = ALL_SETTINGS[key]
    rows = saved_rows(setting, "bm25", paper_qids)
    return compute_bm25_streams(setting, rows, log=lambda m: print(m, flush=True), force=force)


def check_pair(setting: Setting, retriever: str, srcs: list[dict[str, Any]], cfg: dict[str, Any], vanilla_ref: dict) -> dict[str, Any]:
    out: dict[str, Any] = {"setting": setting.key, "display": setting.display, "retriever": retriever, "problems": [], "notes": []}
    P = out["problems"]
    paper = []
    for src in srcs:
        rows = {str(r["query_id"]): r for r in load_ranking_rows(src)}
        paper.append((src, rows))
    paper_qids = sorted(paper[0][1], key=qid_sort_key)
    out["n_queries"] = len(paper_qids)
    out["paper_saved_depth"] = sorted({int(s.get("saved_depth") or 0) for s in srcs})
    saved_json = setting.saved_results_json(retriever)
    out["saved_per_document_file"] = rel(saved_json) if saved_json.exists() else None
    sp, qp = stream_paths(setting, retriever)
    out["stream_files"] = [rel(sp), rel(qp)]
    if not (sp.exists() and qp.exists()):
        out["status"] = "waiting_for_scores"
        out["missing_files"] = [rel(p) for p in (sp, qp) if not p.exists()]
        markers = DENSE_STREAMS_DIR.parent / "markers"
        for m in (markers / f"{job_name(setting, retriever)}.DONE", markers / f"{job_name(setting, retriever)}.FAILED"):
            if m.exists():
                out["notes"].append(f"C2 marker present: {rel(m)}")
        return out

    queries = load_query_arrays(sp, qp, retriever)
    by_q = {q.query_id: q for q in queries}
    missing = [q for q in paper_qids if q not in by_q]
    if missing:
        P.append(f"{len(missing)} paper queries missing from the stream (first: {missing[:5]})")
        out["status"] = "failed"
        return out
    out["stream_queries"] = len(queries)
    out["stream_extra_queries"] = len(set(by_q) - set(paper_qids))
    meta = {str(m["query_id"]): m for m in read_json(qp)["queries"]}

    lam_n, lam_h = float(cfg["lambda_n"]), float(cfg["lambda_h"])
    mu_a, mu_d = float(cfg["memory_alias_weight"]), float(cfg["memory_description_weight"])
    if float(cfg["memory_name_weight"]) != 1.0 or float(cfg["lambda_t"]) != 1.0:
        raise RuntimeError("the vectorised fusion assumes lambda_T = mu_N = 1 (as in the published run)")
    pub_pt = (np.array([lam_n]), np.array([lam_h]), np.array([mu_a]), np.array([mu_d]))
    van_pt = (np.array([0.0]), np.array([0.0]), np.array([mu_a]), np.array([mu_d]))

    depth = max(out["paper_saved_depth"] + [10])
    n_ident = {s["source_id"]: 0 for s, _ in paper}
    label_mismatch = 0
    stream_pub_mismatch = 0
    van_stream_mismatch = 0
    mismatches: list[dict[str, Any]] = []
    elig = 0
    hits = {"pub": {5: 0, 10: 0}, "van": {5: 0, 10: 0}}
    for qid in paper_qids:
        q = by_q[qid]
        m = meta[qid]
        if not labels_equal({"gold_chunk_ids": m["gold_chunk_ids"], "silver_chunk_groups": m["silver_chunk_groups"]}, paper[0][1][qid]):
            label_mismatch += 1
        f_pub = final_scores(q, *pub_pt)
        f_van = final_scores(q, *van_pt)
        top_pub = [q.chunk_ids[i] for i in ranking_order(q, f_pub)[0][:depth]]
        top_van = [q.chunk_ids[i] for i in ranking_order(q, f_van)[0][:10]]
        if top_pub[:10] != list(q.published_top10):
            stream_pub_mismatch += 1
        if top_van != list(q.vanilla_top10):
            van_stream_mismatch += 1
        for src, rows in paper:
            saved = [str(x) for x in rows[qid].get("retrieved_chunk_ids", [])]
            if top_pub[: len(saved)] == saved:
                n_ident[src["source_id"]] += 1
            elif len(mismatches) < 5:
                mismatches.append({"query_id": qid, "source_id": src["source_id"], "paper_top10": saved[:10], "ours_top10": top_pub[:10]})
        if q.eligible:
            elig += 1
            r_pub = int(hit_ranks(q, f_pub)[0])
            r_van = int(hit_ranks(q, f_van)[0])
            for k in (5, 10):
                hits["pub"][k] += int(r_pub < k)
                hits["van"][k] += int(r_van < k)
    n = len(paper_qids)
    out["eligible"] = elig
    out["hits"] = hits
    out["hr"] = {s: {k: 100.0 * hits[s][k] / elig if elig else None for k in (5, 10)} for s in hits}
    out["identical_topk_per_source"] = n_ident
    out["label_mismatch"] = label_mismatch
    out["stream_time_published_top10_mismatch"] = stream_pub_mismatch
    out["vanilla_vs_stream_time_top10_mismatch"] = van_stream_mismatch
    out["mismatch_examples"] = mismatches

    # Vectorised fusion vs the repository function, full rankings, seeded sample.
    rng = np.random.default_rng(SEED)
    sample = [paper_qids[i] for i in sorted(rng.choice(n, size=min(N_REPO_CHECK, n), replace=False))]
    repo_bad = 0
    for qid in sample:
        q = by_q[qid]
        for pt, arr in (((lam_n, lam_h, mu_a, mu_d), pub_pt), ((0.0, 0.0, mu_a, mu_d), van_pt)):
            ours = [q.chunk_ids[i] for i in ranking_order(q, final_scores(q, *arr))[0]]
            if ours != repo_ranking(q, retriever, pt):
                repo_bad += 1
    out["repo_full_ranking_check"] = {"queries": len(sample), "mismatching_rankings": repo_bad}

    # ---- gate
    exact = retriever == "bm25"
    if label_mismatch:
        P.append(f"labels in the stream differ from the paper's for {label_mismatch} queries")
    if repo_bad:
        P.append(f"vectorised fusion != combine_mbr_variant_chunk_scores on {repo_bad} sampled full rankings")
    if van_stream_mismatch:
        P.append(f"text-only top-10 != stream-time repository ranking for {van_stream_mismatch} queries")
    drift: list[str] = []
    gaps = []
    for src, _rows in paper:
        ident = n_ident[src["source_id"]]
        if exact and ident != n:
            P.append(f"SAADI top-k differs from saved ranking {src['source_id'][:8]} for {n - ident} of {n} queries")
        elif not exact and ident / n < DENSE_MIN_IDENTICAL_TOP10:
            drift.append(f"identical top-k {ident}/{n} vs {src['source_id'][:8]} (< {DENSE_MIN_IDENTICAL_TOP10:.0%})")
        for k in (5, 10):
            e = src["expected"].get(k)
            if e is None:
                P.append(f"no expected hit count at k={k} for source {src['source_id'][:8]}")
                continue
            if int(e["eligible"]) != elig:
                P.append(f"eligible {elig} vs paper {e['eligible']} ({src['source_id'][:8]})")
            gap = abs(100.0 * hits["pub"][k] / elig - 100.0 * e["hits"] / e["eligible"]) if elig else float("inf")
            gaps.append(gap)
            if exact and int(e["hits"]) != hits["pub"][k]:
                P.append(f"HR@{k} hits {hits['pub'][k]} vs paper {e['hits']} ({src['source_id'][:8]})")
            elif not exact and gap > DENSE_HR_TOLERANCE_POINTS + 1e-9:
                drift.append(f"HR@{k} differs from the paper by {gap:.2f} points ({src['source_id'][:8]})")
    out["max_hr_gap_points"] = max(gaps) if gaps else None
    out["drift_accepted"] = False
    if drift:
        if retriever in DRIFT_ACCEPTED_RETRIEVERS and out["max_hr_gap_points"] is not None and out["max_hr_gap_points"] <= DRIFT_MAX_HR_POINTS + 1e-9:
            out["drift_accepted"] = True
            out["notes"].append("embedding drift accepted as in C2: " + "; ".join(drift))
        else:
            P.extend(drift)
    elif not exact and any(n_ident[s["source_id"]] != n for s, _ in paper):
        out["notes"].append(f"dense pair within C2's rule; identical top-k {min(n_ident.values())}/{n}")

    # Vanilla vs the paper's Vanilla (aggregate, reported only).
    van = {}
    for table, cond in (("table26", setting.condition), ("table1", "")):
        if table == "table1" and setting.condition != MAIN_CONDITION[setting.dataset]:
            continue  # Table 1 holds MuSiQue-32K and QASPER-64K only
        for k in (5, 10):
            ref = vanilla_ref.get((table, setting.dataset, cond, retriever, k))
            if ref is not None and out["hr"]["van"][k] is not None:
                van[f"{table}@{k}"] = {"paper": ref, "ours": out["hr"]["van"][k], "diff": out["hr"]["van"][k] - ref}
    out["vanilla_vs_paper"] = van
    out["status"] = "failed" if P else "ready"
    return out


def body(report: StepReport, args: argparse.Namespace) -> None:
    require_step_ok(1)
    cfg = published_settings()
    report.data["published_settings"] = cfg
    settings = selected_settings(args.settings)
    retrievers = selected_retrievers(args.retrievers)
    sources = paper_sources()
    pairs = [(s, r) for s in settings for r in retrievers if sources.get((s.key, r))]
    for s in settings:
        mp = read_json(MAP_DIR / f"{s.key}.json")
        if not mp.get("ok"):
            raise RuntimeError(f"{s.key}: the sub-document map of step 01 is not ok")

    # ---- BM25 streams (CPU, parallel over settings, resumable)
    jobs = []
    for s, r in pairs:
        if r != "bm25":
            continue
        qids = sorted({str(row["query_id"]) for row in load_ranking_rows(sources[(s.key, r)][0])}, key=qid_sort_key)
        jobs.append((s.key, qids))
    stream_info: dict[str, Any] = {}
    if jobs:
        print(f"[c4] step 02: BM25 scores for {len(jobs)} settings, {args.workers} workers", flush=True)
        with ProcessPoolExecutor(max_workers=max(1, min(args.workers, len(jobs)))) as pool:
            futs = {pool.submit(_bm25_job, key, qids, args.force): key for key, qids in jobs}
            for fut in as_completed(futs):
                key = futs[fut]
                try:
                    stream_info[key] = fut.result()
                    print(f"[c4] step 02: BM25 scores for {key} ready", flush=True)
                except Exception as exc:
                    import traceback

                    stream_info[key] = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
                    report.fail(f"{key}/bm25: scoring failed: {type(exc).__name__}: {exc}")
    report.data["bm25_streams"] = stream_info

    # ---- dense streams computed here when asked (--score-dense contriever); one process, the
    # retriever's own encoder on cuda / mps / cpu. Resumable like BM25.
    dense_info: dict[str, Any] = {}
    score_dense = [r.strip() for r in (args.score_dense or "").split(",") if r.strip()]
    bad = [r for r in score_dense if r == "bm25" or r not in retrievers]
    if bad:
        raise RuntimeError(f"--score-dense {bad}: must be dense retrievers among the selected ones ({retrievers})")
    for s, r in pairs:
        if r not in score_dense:
            continue
        name = job_name(s, r)
        print(f"[c4] step 02: {r} scores for {s.key}", flush=True)
        try:
            qids = sorted({str(row["query_id"]) for row in load_ranking_rows(sources[(s.key, r)][0])}, key=qid_sort_key)
            dense_info[name] = compute_streams(s, r, saved_rows(s, r, qids), log=lambda m: print(m, flush=True), force=args.force)
            print(f"[c4] step 02: {r} scores for {s.key} ready", flush=True)
        except Exception as exc:
            import traceback

            dense_info[name] = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
            report.fail(f"{name}: scoring failed: {type(exc).__name__}: {exc}")
    report.data["dense_streams"] = dense_info

    # ---- per-pair checks
    vanilla_ref = paper_vanilla()
    results = []
    for s, r in pairs:
        if (r == "bm25" and "error" in stream_info.get(s.key, {})) or "error" in dense_info.get(job_name(s, r), {}):
            results.append({"setting": s.key, "display": s.display, "retriever": r, "status": "failed", "problems": ["BM25 scoring failed"], "notes": []})
            continue
        print(f"[c4] step 02: checking {s.key}/{r}", flush=True)
        try:
            res = check_pair(s, r, sources[(s.key, r)], cfg, vanilla_ref)
        except Exception as exc:
            import traceback

            res = {"setting": s.key, "display": s.display, "retriever": r, "status": "failed", "problems": [f"{type(exc).__name__}: {exc}"], "notes": [], "traceback": traceback.format_exc()}
        results.append(res)
        for p in res["problems"]:
            report.fail(f"{s.key}/{r}: {p}")
    report.data["pairs"] = results

    report.p("Published weights read from `configs/methods/sadi.yaml` (cached_retrieval): "
             + ", ".join(f"{k}={v}" for k, v in cfg.items()) + ".")
    report.h("1. What the paper saved")
    report.p("No full ranking and no per-chunk score is saved for any pair: the paper keeps SAADI's top-k list per query "
             "(depth 10 or 20) and Vanilla only as aggregate HR. Full per-chunk channel scores therefore come from "
             "BM25 re-scoring here (CPU) or, for the dense retrievers, from C2's recomputed streams.")
    rows = []
    for x in results:
        rows.append([x["display"], x["retriever"], "/".join(str(d) for d in x.get("paper_saved_depth", [])) or "-",
                     "yes" if x.get("saved_per_document_file") else "no",
                     {"ready": "available", "failed": "available" if x.get("stream_queries") else "-", "waiting_for_scores": "waiting for scores"}.get(x["status"], x["status"]),
                     "re-scored here" if Path((x.get("stream_files") or [""])[0]).parent == Path(rel(BM25_STREAMS_DIR))
                     else rel(DENSE_STREAMS_DIR)])
    report.table(["Setting", "Retriever", "Paper top-k depth", "Saved per-document top-10 file", "Full per-chunk scores", "Source of full scores"], rows)
    waiting = [x for x in results if x["status"] == "waiting_for_scores"]
    if waiting:
        report.p("Missing full-score files (pairs marked \"waiting for scores\"; C4 reads them later from "
                 "`$C4_DENSE_STREAMS_DIR` with no code change):")
        report.p("")
        for x in waiting:
            report.bullet(f"{x['display']} / {x['retriever']}: " + ", ".join(f"`{f}`" for f in x["missing_files"]))
            report.warn(f"{x['setting']}/{x['retriever']}: waiting for scores")
    if stream_info:
        parts = []
        for k, v in sorted(stream_info.items()):
            if "error" in v:
                parts.append(f"{k}: FAILED")
            elif not v.get("computed"):
                parts.append(f"{k}: reused from {v.get('path')}")
            else:
                parts.append(f"{k}: {v.get('docs', 0)} documents ({v.get('docs_from_checkpoint', 0)} from checkpoints), {fnum(v.get('seconds'), 0)} s")
        report.p("")
        report.p("BM25 scoring: " + "; ".join(parts) + ".")
    if dense_info:
        parts = []
        for k, v in sorted(dense_info.items()):
            if "error" in v:
                parts.append(f"{k}: FAILED")
            elif not v.get("computed"):
                parts.append(f"{k}: reused from {v.get('path')}")
            else:
                parts.append(f"{k}: {v.get('docs', 0)} documents ({v.get('docs_from_checkpoint', 0)} from checkpoints) on {v.get('device')}, {fnum(v.get('seconds'), 0)} s")
        report.p("")
        report.p("Dense scoring here: " + "; ".join(parts) + ".")

    report.h("2. Reproduction on the full ranking")
    rows = []
    for x in results:
        if "hits" not in x:
            continue
        idt = x["identical_topk_per_source"]
        rows.append([x["display"], x["retriever"], x["n_queries"], x["eligible"],
                     " / ".join(f"{v}/{x['n_queries']}" for v in idt.values()),
                     f"{fnum(x['hr']['pub'][5])} / {fnum(x['hr']['pub'][10])}",
                     fnum(x["max_hr_gap_points"], 2), x["label_mismatch"],
                     f"{x['repo_full_ranking_check']['mismatching_rankings']}/{2 * x['repo_full_ranking_check']['queries']}",
                     x["vanilla_vs_stream_time_top10_mismatch"], "PASS" if x["status"] == "ready" else "FAIL"])
    report.table(["Setting", "Retriever", "Queries", "Eligible", "SAADI top-k identical to each paper ranking", "SAADI HR@5 / HR@10",
                  "Max HR gap vs paper (points)", "Label mismatches", "Repo fusion check (bad/rankings)", "Vanilla != stream-time top-10", "Gate"], rows)
    notes = [f"{x['display']}/{x['retriever']}: {n}" for x in results for n in x.get("notes", [])]
    if notes:
        report.p("Notes: " + " ".join(notes))
    ex = [(x["display"], x["retriever"], m) for x in results for m in x.get("mismatch_examples", [])]
    if ex:
        report.p("")
        report.p("First 5 mismatching queries (paper top-10 vs ours):")
        report.code("\n".join(f"{d}/{r} q{m['query_id']} ({m['source_id'][:8]}):\n  paper {m['paper_top10']}\n  ours  {m['ours_top10']}" for d, r, m in ex[:5]))

    report.h("3. Vanilla (text channel) against the paper's Vanilla (aggregate, not gated)")
    rows = []
    for x in results:
        v = x.get("vanilla_vs_paper") or {}
        if "hr" not in x:
            continue
        cell = lambda key: (f"{fnum(v[key]['paper'])} ({v[key]['diff']:+.2f})" if key in v else "-")  # noqa: E731
        rows.append([x["display"], x["retriever"], f"{fnum(x['hr']['van'][5])} / {fnum(x['hr']['van'][10])}",
                     f"{cell('table26@5')} / {cell('table26@10')}", f"{cell('table1@5')} / {cell('table1@10')}"])
    report.table(["Setting", "Retriever", "Text channel HR@5 / HR@10 (ours)", "Table 26 Vanilla (ours - paper)", "Table 1 Vanilla (ours - paper; MuSiQue-32K and QASPER-64K only)"], rows)
    report.p("Rebuttal Vanilla = SAADI's text channel on the same chunks (decided in C1). The paper's Vanilla rows are "
             "aggregate numbers from a separate run, so they are shown here and not used as a gate.")
    for x in results:
        if x.get("traceback"):
            report.h(f"Traceback ({x['display']}/{x['retriever']})")
            report.code(x["traceback"])
    for k, v in list(stream_info.items()) + list(dense_info.items()):
        if v.get("traceback"):
            report.h(f"Traceback (scoring, {k})")
            report.code(v["traceback"])
    report.data["ready_pairs"] = [f"{x['setting']}__{x['retriever']}" for x in results if x["status"] == "ready"]
    report.data["waiting_pairs"] = [f"{x['setting']}__{x['retriever']}" for x in waiting]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--settings", default=None)
    ap.add_argument("--retrievers", default=None)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("C4_WORKERS", "4")), help="parallel BM25 settings (default 4)")
    ap.add_argument("--force", action="store_true", help="re-score even when the scores exist")
    ap.add_argument("--score-dense", default=os.environ.get("C4_SCORE_DENSE", ""),
                    help="comma list of dense retrievers to score here (e.g. contriever); needs torch + transformers")
    args = ap.parse_args()
    cmd = "bash experiments/rebuttal/c4_subdoc_control/sh/02_check.sh" + ("" if len(sys.argv) == 1 else " " + " ".join(sys.argv[1:]))
    report = StepReport(2, "input completeness and reproduction of the paper", command=cmd)
    return run_step(report, lambda r: body(r, args))


if __name__ == "__main__":
    raise SystemExit(main())
