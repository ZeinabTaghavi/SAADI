#!/usr/bin/env python3
"""C3 step 04: tables, paired comparisons and the final report, from the per-query files only.

Reads results/rebuttal/c3_channel_ablation/per_query/<dataset>__<retriever>.csv.gz (written by
steps 02 and 03) and the paper's numbers (Table 1 Vanilla, expected_metrics SAADI). Nothing is
re-scored and nothing is tuned; CPU only, a few seconds. Runs on the laptop or the server.

Guards (the step fails, but still writes everything it can, when one does not hold):
- every pair that step 01 marked ready has a per-query file;
- the same (query_id, doc_id) set for every retriever of a dataset (the comparison is paired);
- HR@5/HR@10 and the five paired differences recomputed here equal steps 02/03's JSON;
- all three = the paper's SAADI (expected_metrics.json) for every pair.

Writes to results/rebuttal/c3_channel_ablation/: table.csv, paired.csv, REPORT.md,
table_paper.tex; and reports/step_04.{md,json}.
"""

from __future__ import annotations

import csv
import gzip
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    ALL_DATASETS,
    ALL_RETRIEVERS,
    DATASET_DISPLAY,
    INFO_VARIANTS,
    MAIN_VARIANTS,
    N_BOOT,
    REPO_ROOT,
    REPORT_DIR,
    RESULTS_DIR,
    RETRIEVER_DISPLAY,
    SEED,
    VANILLA_DATASET_NAME,
    VARIANT_LABEL,
    VARIANT_SOURCE,
    VARIANTS,
    StepReport,
    expected_main_metrics,
    job_name,
    load_published_weights,
    mcnemar_exact_p,
    published_vanilla,
    rel,
    run_step,
    selected,
    sha256_file,
    table,
)
from saadi.rebuttal.frozen_records import paired_bootstrap_ci
from step02_variants import QUICK_COMPARISONS, median_text, pair_label, per_query_path

# The five comparisons of the task, plus text + summary - text + memory (for the count of
# pairs where text + summary beats text + memory, with its interval).
COMPARISONS = (*QUICK_COMPARISONS, ("text_summary", "text_memory"))
KS = (5, 10)
# Table 1's Vanilla is the text channel on SAADI's chunks on these datasets only (step 01, check 2).
VANILLA_SAME_RUN = ("musique_32k", "qasper_64k")
TOL = 1e-9

OUT_TABLE = RESULTS_DIR / "table.csv"
OUT_PAIRED = RESULTS_DIR / "paired.csv"
OUT_REPORT = RESULTS_DIR / "REPORT.md"
OUT_TEX = RESULTS_DIR / "table_paper.tex"


def cmp_label(a: str, b: str) -> str:
    return f"{VARIANT_LABEL[a]} - {VARIANT_LABEL[b]}"


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def load_per_query(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = [(r["query_id"], r["doc_id"]) for r in rows]
    eligible = np.array([r["eligible"] == "1" for r in rows], dtype=bool)
    ranks = np.array([[int(r[f"rank_{v}"]) for v in VARIANTS] for r in rows], dtype=np.int64).reshape(len(rows), len(VARIANTS))
    return {"keys": keys, "eligible": eligible, "ranks": ranks}


def analyse_pair(d: str, r: str, pq: dict[str, Any], van: dict, exp: dict) -> dict[str, Any]:
    e = pq["eligible"]
    ranks = pq["ranks"][e]
    n = int(e.sum())
    hits = {v: {k: int((ranks[:, vi] <= k).sum()) for k in KS} for vi, v in enumerate(VARIANTS)}
    hr = {v: {k: 100.0 * hits[v][k] / max(1, n) for k in KS} for v in VARIANTS}
    med = {v: median_text(ranks[:, vi]) for vi, v in enumerate(VARIANTS)}
    paired = {}
    for a, b in COMPARISONS:
        ia, ib = VARIANTS.index(a), VARIANTS.index(b)
        for k in KS:
            ha, hb = ranks[:, ia] <= k, ranks[:, ib] <= k
            mean, lo, hi = paired_bootstrap_ci(100.0 * (ha.astype(float) - hb.astype(float)), n_resamples=N_BOOT, seed=SEED)
            only_a, only_b = int((ha & ~hb).sum()), int((hb & ~ha).sum())
            paired[(a, b, k)] = {"mean": mean, "lo": lo, "hi": hi, "a_only": only_a, "b_only": only_b, "p": mcnemar_exact_p(only_a, only_b)}
    paper = {
        "vanilla": {k: van.get((VANILLA_DATASET_NAME[d], r, k)) for k in KS},
        "saadi": {k: (100.0 * float(exp[k]["hr"]) if k in exp else None) for k in KS},
    }
    return {"dataset": d, "retriever": r, "n_rows": len(pq["keys"]), "n_eligible": n, "hits": hits, "hr": hr, "median": med, "paired": paired, "paper": paper}


def guard_against_step_json(res: dict[str, Any], step_pairs: dict[str, dict]) -> tuple[bool | None, str]:
    """HR and the five quick differences must equal what step 02/03 wrote for this pair."""

    src = step_pairs.get(job_name(res["dataset"], res["retriever"]))
    if src is None:
        return None, "no step 02/03 JSON entry"
    bad = []
    if src["n_eligible"] != res["n_eligible"]:
        bad.append("eligible count")
    for v in VARIANTS:
        for k in KS:
            if abs(src["hr"][v][str(k)] - res["hr"][v][k]) > TOL:
                bad.append(f"HR@{k} {v}")
    for a, b in QUICK_COMPARISONS:
        for k in KS:
            x, y = src["quick"][f"{a}-{b}@{k}"], res["paired"][(a, b, k)]
            if any(abs(x[f] - y[f]) > TOL for f in ("mean", "lo", "hi")) or x["a_only"] != y["a_only"] or x["b_only"] != y["b_only"]:
                bad.append(f"{a}-{b}@{k}")
    return (not bad), ("identical" if not bad else "differs: " + ", ".join(bad[:6]))


def share(num: float, den: float) -> str:
    return f"{100.0 * num / den:.0f}%" if den > 0 else "n/a (no gain)"


def diff10(res: dict[str, Any], a: str, b: str) -> float:
    return res["hr"][a][10] - res["hr"][b][10]


def tex_escape(s: str) -> str:
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def write_tex(results: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "% C3 query-time channel ablation (generated by experiments/rebuttal/c3_channel_ablation/scripts/step04_report.py).",
        "% Needs \\usepackage{booktabs}. HR@10 in %. Text only = text channel on SAADI's chunks (equals Table 1 Vanilla on",
        "% MuSiQue-32K and QASPER-64K); All three = SAADI as published (the paper's saved run).",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Query-time channel ablation, HR@10 (\%). One channel weight of the published Equation~2 is set to 0 "
        r"($\lambda_T{=}1$, $\lambda_M{=}0.5$, $\lambda_S{=}0.3$ otherwise); no re-tuning, same index. "
        r"$^{+}$/$^{-}$: the 95\% paired bootstrap interval of the difference to text only (10{,}000 resamples over queries) "
        r"lies above/below zero. Best per row in bold.}",
        r"\label{tab:c3-channel-ablation}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Dataset & Retriever & Text only & Text + summary & Text + memory & All three \\",
        r"\midrule",
    ]
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for res in results:
        by_dataset.setdefault(res["dataset"], []).append(res)
    first = True
    for d in ALL_DATASETS:
        if d not in by_dataset:
            continue
        if not first:
            lines.append(r"\midrule")
        first = False
        for i, res in enumerate(by_dataset[d]):
            vals = [round(res["hr"][v][10], 1) for v in MAIN_VARIANTS]
            best = max(vals)
            cells = []
            for v, x in zip(MAIN_VARIANTS, vals, strict=True):
                cell = f"{x:.1f}"
                if x == best:
                    cell = rf"\textbf{{{cell}}}"
                if v != "text_only":
                    c = res["paired"][(v, "text_only", 10)]
                    if c["lo"] > 0:
                        cell += r"$^{+}$"
                    elif c["hi"] < 0:
                        cell += r"$^{-}$"
                cells.append(cell)
            name = tex_escape(DATASET_DISPLAY[d]) if i == 0 else ""
            lines.append(f"{name} & {tex_escape(RETRIEVER_DISPLAY[res['retriever']])} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def summary_numbers(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)

    def count(a: str, b: str, k: int = 10) -> dict[str, Any]:
        xs = [res["paired"][(a, b, k)] for res in results]
        names = [pair_label(res["dataset"], res["retriever"]) for res in results]
        return {
            "above": sum(x["mean"] > 0 for x in xs),
            "below": sum(x["mean"] < 0 for x in xs),
            "tie": sum(x["mean"] == 0 for x in xs),
            "ci_above": [nm for nm, x in zip(names, xs, strict=True) if x["lo"] > 0],
            "ci_below": [nm for nm, x in zip(names, xs, strict=True) if x["hi"] < 0],
        }

    counts = {f"{a}-{b}": count(a, b) for a, b in COMPARISONS}
    counts5 = {f"{a}-{b}": count(a, b, 5) for a, b in COMPARISONS}
    full = [diff10(r, "all_three", "text_only") for r in results]
    ts = [diff10(r, "text_summary", "text_only") for r in results]
    tm = [diff10(r, "text_memory", "text_only") for r in results]
    pos = [i for i, x in enumerate(full) if x > 0]

    def by(key_fn, keys):
        out = {}
        for key in keys:
            idx = [i for i, r in enumerate(results) if key_fn(r) == key]
            if idx:
                out[key] = {
                    "n": len(idx),
                    "full": float(np.mean([full[i] for i in idx])),
                    "text_summary": float(np.mean([ts[i] for i in idx])),
                    "text_memory": float(np.mean([tm[i] for i in idx])),
                }
        return out

    return {
        "n_pairs": n,
        "counts_hr10": counts,
        "counts_hr5": counts5,
        "mean_full_gain": float(np.mean(full)) if n else float("nan"),
        "mean_text_summary_gain": float(np.mean(ts)) if n else float("nan"),
        "mean_text_memory_gain": float(np.mean(tm)) if n else float("nan"),
        "share_text_summary_all": share(sum(ts), sum(full)),
        "share_text_memory_all": share(sum(tm), sum(full)),
        "pairs_with_full_gain": len(pos),
        "share_text_summary_gain_pairs": share(sum(ts[i] for i in pos), sum(full[i] for i in pos)),
        "share_text_memory_gain_pairs": share(sum(tm[i] for i in pos), sum(full[i] for i in pos)),
        "by_retriever": by(lambda r: r["retriever"], ALL_RETRIEVERS),
        "by_dataset": by(lambda r: r["dataset"], ALL_DATASETS),
    }


def opening_summary(results, s, not_scored, n_queries_per_retriever, missing_share) -> list[str]:
    """The 5-8 sentence paragraph for the rebuttal document; every number comes from the files."""

    n = s["n_pairs"]
    retr = [r for r in ALL_RETRIEVERS if r in s["by_retriever"]]
    retr_txt = ", ".join(RETRIEVER_DISPLAY[r] for r in retr)
    ds_txt = ", ".join(DATASET_DISPLAY[d] for d in ALL_DATASETS if d in s["by_dataset"])
    c_ts = s["counts_hr10"]["all_three-text_summary"]
    c_tm = s["counts_hr10"]["all_three-text_memory"]
    c_sm = s["counts_hr10"]["text_summary-text_memory"]
    c_full = s["counts_hr10"]["all_three-text_only"]

    def names(xs: list[str]) -> str:
        return f" ({'; '.join(xs)})" if xs else ""

    nq = sorted(set(n_queries_per_retriever.values()))
    nq_txt = f"{nq[0]:,} queries per retriever" if len(nq) == 1 else "queries per retriever: " + ", ".join(f"{RETRIEVER_DISPLAY[r]} {v:,}" for r, v in n_queries_per_retriever.items())
    sents = [
        f"We removed one query-time channel at a time from SAADI's published score fusion (Equation 2) by setting its weight to 0, "
        f"keeping the published weights (lambda_T, lambda_M, lambda_S) = (1, 0.5, 0.3), the index and the queries fixed, on {n} "
        f"dataset x retriever pairs ({retr_txt} on {ds_txt}; {nq_txt}).",
        f"Averaged over the {n} pairs (unweighted), all three channels add {s['mean_full_gain']:+.2f} pp HR@10 over text only; "
        f"text + summary adds {s['mean_text_summary_gain']:+.2f} pp ({s['share_text_summary_all']} of the full gain) and text + memory "
        f"{s['mean_text_memory_gain']:+.2f} pp ({s['share_text_memory_all']}).",
        f"All three is above text only at HR@10 in {c_full['above']}/{n} pairs, with the 95% paired bootstrap interval above zero in "
        f"{len(c_full['ci_above'])}{names(c_full['ci_above'])}" + (f" and below zero in {len(c_full['ci_below'])}{names(c_full['ci_below'])}" if c_full["ci_below"] else "") + ".",
        f"All three is above text + summary in {c_ts['above']}/{n} pairs (interval above zero in {len(c_ts['ci_above'])}{names(c_ts['ci_above'])}) "
        f"and above text + memory in {c_tm['above']}/{n} (interval above zero in {len(c_tm['ci_above'])}{names(c_tm['ci_above'])}).",
        f"Text + summary is above text + memory in {c_sm['above']}/{n} pairs, below in {c_sm['below']}"
        + (f", tied in {c_sm['tie']}" if c_sm["tie"] else "") + ".",
    ]
    br = s["by_retriever"]
    parts = []
    for r in retr:
        x = br[r]
        larger = "summary" if x["text_summary"] > x["text_memory"] else ("memory" if x["text_memory"] > x["text_summary"] else "neither")
        parts.append(
            f"{RETRIEVER_DISPLAY[r]} {x['full']:+.2f} (text + summary {x['text_summary']:+.2f}, text + memory {x['text_memory']:+.2f}; larger single channel: {larger})"
        )
    sents.append("Mean HR@10 gain over text only by retriever, in pp over the four datasets: " + "; ".join(parts) + ".")
    bd = s["by_dataset"]
    sents.append(
        "By dataset (mean over retrievers), the full gain is "
        + ", ".join(f"{DATASET_DISPLAY[d]} {bd[d]['full']:+.2f}" for d in ALL_DATASETS if d in bd)
        + " pp; text only and all three are not re-scored (all three is the paper's saved run; text only is the text channel on "
        "SAADI's chunks, equal to Table 1's Vanilla on MuSiQue-32K and QASPER-64K, where Table 1's Vanilla is that same ranking)."
    )
    if not_scored:
        sents.append(
            "Not included: "
            + "; ".join(sorted({RETRIEVER_DISPLAY[x["retriever"]] + ": " + x["reason"] for x in not_scored}))
            + "."
        )
    return sents


def body(rep: StepReport) -> int:
    datasets = selected("C3_DATASETS", ALL_DATASETS)
    retrievers = selected("C3_RETRIEVERS", ALL_RETRIEVERS)
    w, problems = load_published_weights()
    failed = False
    if problems:
        rep.add("**Config inconsistent with the saved rankings:**", "", *[f"- {p}" for p in problems], "")
        failed = True

    step1 = read_json(REPORT_DIR / "step_01.json") or {}
    ready = set(step1.get("ready", []))
    inv = {job_name(i["dataset"], i["retriever"]): i for i in step1.get("inventory", [])}
    missing_counts = {job_name(c["dataset"], c["retriever"]): c.get("missing_channel_counts") for c in step1.get("checks", [])}
    step_pairs: dict[str, dict] = {}
    step_files = {}
    for n_ in (2, 3):
        doc = read_json(REPORT_DIR / f"step_{n_:02d}.json")
        if doc:
            step_files[f"step_{n_:02d}.json"] = {"status": doc.get("status"), "commit": doc.get("git", {}).get("commit", "")[:12]}
            for p in doc.get("pairs", []):
                step_pairs[job_name(p["dataset"], p["retriever"])] = p

    # Which pairs have per-query files; the rest are listed with the reason.
    found, not_scored = [], []
    for r in retrievers:
        for d in datasets:
            j = job_name(d, r)
            path = per_query_path(d, r)
            if path.exists():
                found.append((d, r, path))
                continue
            if j in ready:
                reason = "passed step 01 but has no per-query file (run step 02/03)"
                failed = True
            elif j in inv:
                reason = f"step 01: {inv[j]['status']}" + (f"; missing {', '.join(inv[j]['missing'][:3])}" if inv[j].get("missing") else "")
            else:
                reason = "not checked by step 01 in this run (no channel scores were looked for; C3 recomputes no embeddings)"
            not_scored.append({"dataset": d, "retriever": r, "reason": reason})

    rep.add(
        "## Settings",
        "",
        f"- Weights from `configs/methods/sadi.yaml`: (lambda_T, lambda_M, lambda_S) = ({w.lambda_t:g}, {w.lambda_n:g}, {w.lambda_h:g}), "
        f"field weights ({w.mu_n:g}, {w.mu_a:g}, {w.mu_d:g}), {w.normalization_method}.",
        "- Where each variant comes from: " + "; ".join(f"{VARIANT_LABEL[v]}: {VARIANT_SOURCE[v]}" for v in VARIANTS) + ".",
        f"- Paired differences in HR points: 95% bootstrap over queries ({N_BOOT} resamples, seed {SEED}); McNemar exact two-sided p.",
        "- Input: only the per-query files below and the paper's numbers. Nothing is re-scored here.",
        "- Steps 02/03 JSON: " + ("; ".join(f"`{k}` {v['status']} at {v['commit']}" for k, v in step_files.items()) or "none found"),
        "",
    )
    if not found:
        rep.add("No per-query file found. Run steps 02 and 03 first.")
        return 1

    van = published_vanilla(REPO_ROOT)
    results = []
    keysets: dict[str, set] = {}
    key_ok = True
    rows_in = []
    for d, r, path in found:
        pq = load_per_query(path)
        ks = set(pq["keys"])
        if d in keysets and keysets[d] != ks:
            key_ok = False
        keysets.setdefault(d, ks)
        res = analyse_pair(d, r, pq, van, expected_main_metrics(d, r, REPO_ROOT))
        res["file"] = rel(path)
        res["sha256"] = sha256_file(path)
        results.append(res)
        rows_in.append([f"`{rel(path)}`", res["n_rows"], res["n_eligible"], res["sha256"][:16]])
    rep.add("## Inputs", "", *table(["Per-query file", "Queries", "Eligible", "sha256 (first 16)"], rows_in), "")
    if not_scored:
        rep.add("Not scored:", "", *table(["Pair", "Reason"], [[job_name(x["dataset"], x["retriever"]), x["reason"]] for x in not_scored]), "")

    # Guards.
    rows_g = []
    for res in results:
        same, why = guard_against_step_json(res, step_pairs)
        saadi_ok = all(
            res["paper"]["saadi"][k] is not None and abs(res["hr"]["all_three"][k] - res["paper"]["saadi"][k]) < 1e-6 for k in KS
        )
        van_note = "-"
        if res["dataset"] in VANILLA_SAME_RUN:
            vv = res["paper"]["vanilla"]
            van_ok = all(vv[k] is not None and abs(round(res["hr"]["text_only"][k], 1) - float(vv[k])) <= 0.05 + 1e-9 for k in KS)
            van_note = "yes" if van_ok else "NO"
            failed |= not van_ok
        failed |= (same is False) or not saadi_ok
        res["guards"] = {"step_json": why, "saadi_equals_paper": saadi_ok, "vanilla_equals_table1": van_note}
        rows_g.append([pair_label(res["dataset"], res["retriever"]), why, "yes" if saadi_ok else "NO", van_note])
    failed |= not key_ok
    rep.add(
        "## Guards",
        "",
        f"- Same (query_id, doc_id) set for every retriever of a dataset: {'yes' if key_ok else '**NO**'}.",
        "",
        *table(["Pair", "HR and paired differences vs step 02/03 JSON", "All three = paper SAADI (HR@5, HR@10)", "Text only = Table 1 Vanilla (MuSiQue, QASPER only)"], rows_g),
        "",
    )

    s = summary_numbers(results)
    n_q = {}
    for res in results:
        n_q[res["retriever"]] = n_q.get(res["retriever"], 0) + res["n_eligible"]
    miss_share = {}
    for res in results:
        mc = missing_counts.get(job_name(res["dataset"], res["retriever"]))
        if mc and mc.get("chunk_query_pairs"):
            miss_share[res["dataset"]] = (100.0 * mc["summary"] / mc["chunk_query_pairs"], 100.0 * mc["memory_all_fields"] / mc["chunk_query_pairs"])

    # ---- table.csv
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_TABLE.open("w", encoding="utf-8", newline="") as handle:
        wr = csv.writer(handle)
        wr.writerow(["dataset", "retriever", "variant", "variant_label", "source", "n_eligible", "hits_at_5", "hits_at_10", "hr_at_5", "hr_at_10", "median_rank", "paper_hr_at_5", "paper_hr_at_10", "paper_source"])
        for res in results:
            for v in VARIANTS:
                if v == "text_only":
                    pp, psrc = res["paper"]["vanilla"], "Table 1 Vanilla (one decimal)" + ("" if res["dataset"] in VANILLA_SAME_RUN else "; separate run, not this ranking")
                elif v == "all_three":
                    pp, psrc = res["paper"]["saadi"], "expected_metrics.json (paper SAADI)"
                else:
                    pp, psrc = {5: None, 10: None}, ""
                wr.writerow(
                    [res["dataset"], res["retriever"], v, VARIANT_LABEL[v], VARIANT_SOURCE[v], res["n_eligible"], res["hits"][v][5], res["hits"][v][10],
                     f"{res['hr'][v][5]:.4f}", f"{res['hr'][v][10]:.4f}", res["median"][v],
                     "" if pp[5] is None else pp[5], "" if pp[10] is None else pp[10], psrc]
                )
    rep.wrote(OUT_TABLE)

    # ---- paired.csv
    with OUT_PAIRED.open("w", encoding="utf-8", newline="") as handle:
        wr = csv.writer(handle)
        wr.writerow(["dataset", "retriever", "comparison", "variant_a", "variant_b", "k", "n_eligible", "diff_pp", "ci_low", "ci_high", "ci_excludes_zero", "hit_only_a", "hit_only_b", "mcnemar_p", "in_task_list"])
        for res in results:
            for a, b in COMPARISONS:
                for k in KS:
                    x = res["paired"][(a, b, k)]
                    wr.writerow(
                        [res["dataset"], res["retriever"], cmp_label(a, b), a, b, k, res["n_eligible"], f"{x['mean']:.4f}", f"{x['lo']:.4f}", f"{x['hi']:.4f}",
                         int(x["lo"] > 0 or x["hi"] < 0), x["a_only"], x["b_only"], f"{x['p']:.6g}", int((a, b) in QUICK_COMPARISONS)]
                    )
    rep.wrote(OUT_PAIRED)

    # ---- table_paper.tex
    write_tex(results, OUT_TEX)
    rep.wrote(OUT_TEX)

    # ---- REPORT.md
    summary = opening_summary(results, s, not_scored, n_q, miss_share)
    main_rows = []
    for res in results:
        p = res["paper"]
        main_rows.append(
            [pair_label(res["dataset"], res["retriever"]), *[f"{res['hr'][v][10]:.2f}" for v in MAIN_VARIANTS],
             f"{diff10(res, 'all_three', 'text_only'):+.2f}",
             share(diff10(res, "text_summary", "text_only"), diff10(res, "all_three", "text_only")),
             share(diff10(res, "text_memory", "text_only"), diff10(res, "all_three", "text_only")),
             "-" if p["vanilla"][10] is None else f"{p['vanilla'][10]}" + ("" if res["dataset"] in VANILLA_SAME_RUN else " *")]
        )
    hr5_rows = [[pair_label(res["dataset"], res["retriever"]), *[f"{res['hr'][v][5]:.2f}" for v in MAIN_VARIANTS], "-" if res["paper"]["vanilla"][5] is None else f"{res['paper']['vanilla'][5]}" + ("" if res["dataset"] in VANILLA_SAME_RUN else " *")] for res in results]
    info_rows = [[pair_label(res["dataset"], res["retriever"]), *[f"{res['hr'][v][5]:.2f} / {res['hr'][v][10]:.2f}" for v in INFO_VARIANTS], *[res["median"][v] for v in VARIANTS]] for res in results]
    paired_rows = []
    for res in results:
        for a, b in COMPARISONS:
            x = res["paired"][(a, b, 10)]
            paired_rows.append([pair_label(res["dataset"], res["retriever"]), cmp_label(a, b), f"{x['mean']:+.2f}", f"[{x['lo']:+.2f}, {x['hi']:+.2f}]", f"{x['a_only']} / {x['b_only']}", f"{x['p']:.3g}"])
    count_rows = []
    for a, b in COMPARISONS:
        for k, cs in ((10, s["counts_hr10"]), (5, s["counts_hr5"])):
            c = cs[f"{a}-{b}"]
            count_rows.append([cmp_label(a, b), f"HR@{k}", f"{c['above']} / {c['below']} / {c['tie']}", len(c["ci_above"]), len(c["ci_below"])])
    n = s["n_pairs"]
    md = [
        "# C3: query-time channel ablation by re-scoring",
        "",
        "For reviewers zXpi and iVU8. Generated by `experiments/rebuttal/c3_channel_ablation/sh/04_report.sh`; every number below comes from "
        "`per_query/*.csv.gz` (steps 02 and 03) and the paper's saved numbers.",
        "",
        "## Summary",
        "",
        " ".join(summary),
        "",
        "## What was done",
        "",
        "- Published Equation 2 (`combine_mbr_variant_chunk_scores`, MBR2), published weights read from `configs/methods/sadi.yaml`; a dropped "
        "channel gets weight 0. The remaining weights are not re-tuned or rescaled; Equation 2's own per-chunk division by the sum of the "
        "weights present is unchanged. Same index, same chunks, same queries.",
        "- Not re-scored: all three (SAADI) is the paper's saved run (top-10 identical for every query, step 01 check 1); text only is the "
        "text-only top-10 saved with the channel scores (C2's stream step), checked against Table 1 in step 01 check 2.",
        "- Scored by C3: text + summary, text + memory, and for information summary alone and memory alone.",
        f"- HR over eligible queries; paired differences with a 95% bootstrap over queries ({N_BOOT:,} resamples, seed {SEED}) and an exact McNemar test.",
        "- This is not the index-time ablation of the paper's Table 12 (history-conditioned extraction), and it differs from the score-fusion "
        "table, which varies the weights but never sets one to 0.",
        "",
        "## HR@10 (%), main variants",
        "",
        *table(["Pair", "Text only", "Text + summary", "Text + memory", "All three (SAADI)", "Full gain (pp)", "Share: text + summary", "Share: text + memory", "Table 1 Vanilla"], main_rows),
        "",
        "Share = (variant - text only) / (all three - text only), at HR@10; 'n/a' when all three is not above text only. A share above 100% "
        "means that variant is above all three; a negative share means it is below text only.",
        "\\* Table 1's Vanilla on LooGLE and NovelHopQA is a separate run, not the text channel on SAADI's chunks (step 01 check 2); shown for reference only.",
        "",
        "## HR@5 (%), main variants",
        "",
        *table(["Pair", "Text only", "Text + summary", "Text + memory", "All three (SAADI)", "Table 1 Vanilla"], hr5_rows),
        "",
        "## Summary counts",
        "",
        f"Over {n} pairs. Above / below / tied = sign of the HR difference; 'CI > 0' and 'CI < 0' count pairs whose 95% interval lies entirely above or below zero.",
        "",
        *table(["Comparison", "k", "Above / below / tied", "CI > 0", "CI < 0"], count_rows),
        "",
        f"- Mean HR@10 gain over text only (unweighted over the {n} pairs): all three {s['mean_full_gain']:+.2f} pp, text + summary "
        f"{s['mean_text_summary_gain']:+.2f} pp, text + memory {s['mean_text_memory_gain']:+.2f} pp.",
        f"- Share of the full gain reached by text + summary: {s['share_text_summary_all']} over all pairs; {s['share_text_summary_gain_pairs']} "
        f"over the {s['pairs_with_full_gain']} pairs where all three is above text only. Text + memory: {s['share_text_memory_all']} and "
        f"{s['share_text_memory_gain_pairs']}.",
        "",
        "Mean HR@10 gain over text only, by retriever and by dataset (pp, unweighted):",
        "",
        *table(
            ["Group", "Pairs", "All three", "Text + summary", "Text + memory"],
            [[RETRIEVER_DISPLAY[k], v["n"], f"{v['full']:+.2f}", f"{v['text_summary']:+.2f}", f"{v['text_memory']:+.2f}"] for k, v in s["by_retriever"].items()]
            + [[DATASET_DISPLAY[k], v["n"], f"{v['full']:+.2f}", f"{v['text_summary']:+.2f}", f"{v['text_memory']:+.2f}"] for k, v in s["by_dataset"].items()],
        ),
        "",
        "## Paired differences at HR@10",
        "",
        "The five comparisons of the task, plus text + summary - text + memory. HR@5 is in `paired.csv`.",
        "",
        *table(["Pair", "Comparison", "Diff (pp)", "95% CI", "Hit only in first / only in second", "McNemar p"], paired_rows),
        "",
        "## For information: one channel alone, and median ranks",
        "",
        "HR@5 / HR@10 of summary alone and memory alone; median rank of the first hit within the top-10 over eligible queries ('>10' when fewer than half are hit).",
        "",
        *table(["Pair", "Summary alone", "Memory alone", *[f"Median: {VARIANT_LABEL[v]}" for v in VARIANTS]], info_rows),
        "",
        "In the alone variants a chunk without that channel falls back to its text score (Equation 2's published rule when the weights present "
        "sum to 0)"
        + (
            "; chunk-query pairs without a summary / without any memory field: "
            + ", ".join(f"{DATASET_DISPLAY[d]} {a:.1f}% / {b:.1f}%" for d, (a, b) in miss_share.items())
            if miss_share
            else ""
        )
        + ".",
        "",
        "## Files",
        "",
        "- `table.csv`: one row per dataset x retriever x variant (HR@5, HR@10, hits, median rank, the paper's number where one exists).",
        "- `paired.csv`: one row per pair x comparison x k (difference, 95% CI, discordant counts, McNemar p).",
        "- `table_paper.tex`: HR@10, four main variants, rows = dataset x retriever (booktabs).",
        "- `per_query/*.csv.gz`: rank of the first hit per query and variant (11 = not within the top-10).",
        "",
    ]
    if not_scored:
        md += ["## Not included", "", *table(["Pair", "Reason"], [[job_name(x["dataset"], x["retriever"]), x["reason"]] for x in not_scored]), ""]
    OUT_REPORT.write_text("\n".join(md), encoding="utf-8")
    rep.wrote(OUT_REPORT)

    # ---- step report
    rep.add("## Opening summary (as written to REPORT.md)", "", *[f"> {x}" for x in summary], "")
    rep.add("## HR@10 (%), main variants", "", *table(["Pair", "Text only", "Text + summary", "Text + memory", "All three (SAADI)", "Full gain (pp)", "Share: text + summary", "Share: text + memory", "Table 1 Vanilla"], main_rows), "")
    rep.add("## Summary counts", "", *table(["Comparison", "k", "Above / below / tied", "CI > 0", "CI < 0"], count_rows), "")
    rep.add(
        f"- Share of the full HR@10 gain: text + summary {s['share_text_summary_all']} (all pairs), {s['share_text_summary_gain_pairs']} "
        f"({s['pairs_with_full_gain']} pairs with a gain); text + memory {s['share_text_memory_all']}, {s['share_text_memory_gain_pairs']}.",
        f"- Full tables and all paired differences: `{rel(OUT_REPORT)}`.",
        "",
    )
    rep.data.update(
        {
            "weights": w.as_dict(),
            "inputs": [{"file": r_["file"], "sha256": r_["sha256"], "n_eligible": r_["n_eligible"]} for r_ in results],
            "not_scored": not_scored,
            "guards": {job_name(r_["dataset"], r_["retriever"]): r_["guards"] for r_ in results},
            "same_query_sets": key_ok,
            "summary": s,
            "opening_summary": summary,
            "hr": {job_name(r_["dataset"], r_["retriever"]): {v: r_["hr"][v] for v in VARIANTS} for r_ in results},
        }
    )
    return 1 if failed else 0


def main() -> int:
    rep = StepReport(4, "tables, paired comparisons, final report", "bash experiments/rebuttal/c3_channel_ablation/sh/04_report.sh")
    return run_step(rep, body)


if __name__ == "__main__":
    sys.exit(main())
