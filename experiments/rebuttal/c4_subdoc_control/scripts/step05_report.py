#!/usr/bin/env python3
"""C4 step 05: paired bootstrap, cross-pair summary, and the final tables and report.

Reads the per-query ranks of steps 03 (BM25) and 04 (dense; optional) and writes
  results/rebuttal/c4_subdoc_control/table.csv      every measure, per pair x set x method
  results/rebuttal/c4_subdoc_control/paired.csv     SAADI - Vanilla with 95% CI (10,000 resamples over queries, seed 42)
  results/rebuttal/c4_subdoc_control/REPORT.md      summary for the rebuttal + tables
  results/rebuttal/c4_subdoc_control/table_paper.tex
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from typing import Any

import numpy as np

from c4_common import (
    ALL_SETTINGS,
    N_BOOT,
    REPORT_DIR,
    RESULTS_DIR,
    RETRIEVER_DISPLAY,
    RETRIEVERS,
    SEED,
    SIZES,
    DATASET_DISPLAY,
    StepReport,
    fnum,
    read_json,
    rel,
    require_step_ok,
    run_step,
)
from c4_measures import PAIRED_MEASURES, SETS, load_perquery, paired_diffs, perquery_path, summarize

from saadi.rebuttal.frozen_records import paired_bootstrap_ci  # noqa: E402

MAIN_SET = os.environ.get("C4_MAIN_SET", "R")
OTHER_SET = "S" if MAIN_SET == "R" else "R"
SET_DISPLAY = {"all": "all eligible queries", "S": "set S (all paper evidence inside the gold sub-document)",
               "R": "set R (labels re-derived inside the gold sub-document)"}


def _ci(d: dict[str, Any] | None, nd: int = 1) -> str:
    if not d or d.get("diff") is None or (isinstance(d["diff"], float) and math.isnan(d["diff"])):
        return "-"
    return f"{d['diff']:+.{nd}f} [{d['lo']:+.{nd}f}, {d['hi']:+.{nd}f}]"


def _order(pair: str) -> tuple[int, int, int]:
    key, r = pair.split("__")
    s = ALL_SETTINGS[key]
    return (RETRIEVERS.index(r), ("musique", "qasper").index(s.dataset), SIZES.index(s.size))


def body(report: StepReport, args: argparse.Namespace) -> None:
    require_step_ok(1)
    require_step_ok(2)
    s03 = require_step_ok(3)
    pairs = list(s03.get("measured_pairs", []))
    s04_path = REPORT_DIR / "step_04.json"
    if s04_path.exists():
        s04 = require_step_ok(4)
        pairs += list(s04.get("measured_pairs", []))
        waiting = [x["pair"] for x in s04.get("skipped", [])]
    else:
        waiting = ["(step 04 not run: no dense retriever included)"]
        report.warn("step 04 has not run; the report covers BM25 only")
    pairs = sorted(set(pairs), key=_order)
    if not pairs:
        raise RuntimeError("no measured pair")
    report.data["pairs"] = pairs
    report.data["waiting"] = waiting
    report.data["main_set"] = MAIN_SET

    frames = {}
    for p in pairs:
        key, r = p.split("__")
        path = perquery_path(ALL_SETTINGS[key], r)
        if not path.exists():
            raise RuntimeError(f"{p}: per-query file missing ({rel(path)}); rerun step 03/04")
        frames[p] = load_perquery(path)

    # ---------------- table.csv
    table_rows: list[dict[str, Any]] = []
    for p, fr in frames.items():
        key, r = p.split("__")
        s = ALL_SETTINGS[key]
        for set_name in ("all",) + SETS:
            for meth in ("van", "pub"):
                table_rows.append({"setting": key, "dataset": s.dataset, "size": s.size, "retriever": r, "set": set_name,
                                   "method": {"van": "vanilla", "pub": "saadi"}[meth], **summarize(fr, set_name, meth)})
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tpath = RESULTS_DIR / "table.csv"
    cols = list(dict.fromkeys(k for row in table_rows for k in row))
    with tpath.open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=cols)
        w.writeheader()
        for row in table_rows:
            w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in cols})
    report.wrote(tpath)

    # ---------------- paired.csv
    paired: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    prow: list[dict[str, Any]] = []
    for p, fr in frames.items():
        key, r = p.split("__")
        for set_name in ("all",) + SETS:
            for measure, restr, label in PAIRED_MEASURES:
                if set_name == "all" and restr:
                    continue
                a, b, d = paired_diffs(fr, set_name, measure, restr)
                mean, lo, hi = paired_bootstrap_ci(d, n_resamples=N_BOOT, seed=SEED)
                rec = {"setting": key, "retriever": r, "set": set_name, "restriction": restr or "full_document", "measure": measure,
                       "label": label, "n": int(d.size), "saadi": float(a.mean()) if a.size else None,
                       "vanilla": float(b.mean()) if b.size else None, "diff": mean, "lo": lo, "hi": hi,
                       "ci_above_zero": bool(lo > 0) if d.size else False, "ci_below_zero": bool(hi < 0) if d.size else False}
                paired[(p, set_name, measure, restr)] = rec
                prow.append(rec)
    ppath = RESULTS_DIR / "paired.csv"
    with ppath.open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(prow[0]))
        w.writeheader()
        w.writerows(prow)
    report.wrote(ppath)

    def P(p, set_name, measure, restr=""):
        return paired.get((p, set_name, measure, restr))

    def T(p, set_name, meth):
        key, r = p.split("__")
        for row in table_rows:
            if (row["setting"], row["retriever"], row["set"], row["method"]) == (key, r, set_name, meth):
                return row
        return None

    # ---------------- cross-pair summary
    def counts(set_name: str, restr: str, measure: str = "in_rr") -> dict[str, Any]:
        recs = [P(p, set_name, measure, restr) for p in pairs]
        recs = [x for x in recs if x and x["n"]]
        return {"pairs": len(recs), "gain": sum(x["diff"] > 0 for x in recs), "ci_above": sum(x["ci_above_zero"] for x in recs),
                "ci_below": sum(x["ci_below_zero"] for x in recs),
                "diffs": [x["diff"] for x in recs]}

    summ = {f"{s}_{r}": counts(s, r) for s in SETS for r in ("keep", "renorm")}
    summ_hr5 = {s: counts(s, "keep", "in_hr5") for s in SETS}
    summ_sub = counts(MAIN_SET, "", "subdoc_hit10")
    summ_full = counts(MAIN_SET, "", "full_hr10")
    report.data["summary_counts"] = {"mrr": summ, "hr5": summ_hr5, "subdoc_hit10": summ_sub, "full_hr10": summ_full}

    def verdict(p: str) -> str:
        full = P(p, MAIN_SET, "full_hr10")
        sub = P(p, MAIN_SET, "subdoc_hit10")
        ins = P(p, MAIN_SET, "in_rr", "keep")
        s_ok, i_ok = bool(sub and sub["ci_above_zero"]), bool(ins and ins["ci_above_zero"])
        lead = "full gain" if full and full["ci_above_zero"] else "no clear full gain"
        if s_ok and i_ok:
            return f"{lead}; both"
        if s_ok:
            return f"{lead}; sub-document selection"
        if i_ok:
            return f"{lead}; inside ranking"
        return f"{lead}; neither"

    # ---------------- REPORT.md
    L: list[str] = []
    n_pairs = len(pairs)
    retr_used = sorted({p.split("__")[1] for p in pairs}, key=RETRIEVERS.index)
    mk = summ[f"{MAIN_SET}_keep"]
    mr = summ[f"{MAIN_SET}_renorm"]
    ok_ = summ[f"{OTHER_SET}_keep"]
    md = mk["diffs"]
    bm = [p for p in pairs if p.endswith("__bm25")]

    def trend_text(ds: str) -> str | None:
        vals = []
        for size in SIZES:
            x = P(f"{ds}_{size.lower()}__bm25", MAIN_SET, "in_rr", "keep")
            if x:
                vals.append(f"{x['diff']:+.1f} at {size}")
        return f"{DATASET_DISPLAY[ds]} " + ", ".join(vals) if vals else None

    trends = [t for t in (trend_text("musique"), trend_text("qasper")) if t]

    def sub_sentence() -> str:
        n, up, ci_up = summ_sub["pairs"], summ_sub["gain"], summ_sub["ci_above"]
        base = (f"SAADI's full-document top 10 contains a gold-sub-document chunk more often than Vanilla's in {up} of {n} pairs "
                f"(CI above zero in {ci_up}, below zero in {summ_sub['ci_below']})")
        if ci_up and mk["ci_above"]:
            return base + ", so the full-document gain comes partly from choosing the right sub-document and partly from ranking inside it."
        if mk["ci_above"]:
            return base + ", so choosing the sub-document does not explain the full-document gain; ranking inside it does."
        if ci_up:
            return base + ", so the full-document gain comes mainly from choosing the right sub-document."
        return base + "."
    sents = [
        f"To separate SAADI's within-document ranking from sub-document selection, we re-ranked only the chunks of each query's "
        f"gold sub-document in every concatenated MuSiQue and QASPER setting (32K to 256K tokens), for {n_pairs} setting-retriever "
        f"pairs ({', '.join(RETRIEVER_DISPLAY[r] for r in retr_used)}), comparing SAADI with Vanilla (the text channel alone) on identical chunk scores.",
        "On the whole document both methods first reproduce the paper's saved rankings and hit rates "
        + ("exactly." if all(p.endswith("__bm25") for p in pairs) else "(exactly for BM25, within the C2 tolerance for the dense retrievers)."),
        ("On all queries, with the evidence labels re-derived inside the gold sub-document by the paper's own labeler, "
         if MAIN_SET == "R" else "On the queries whose original evidence lies entirely inside the gold sub-document, ")
        + f"SAADI's MRR inside the gold sub-document is higher than Vanilla's in {mk['gain']} of {mk['pairs']} pairs, and the 95% "
        f"bootstrap CI of the gain lies fully above zero in {mk['ci_above']}"
        + (f" (fully below zero in {mk['ci_below']})" if mk["ci_below"] else "") + ".",
        (f"The in-sub-document MRR gain ranges from {min(md):+.1f} to {max(md):+.1f} points (median {float(np.median(md)):+.1f}); "
         f"inside HR@5 rises in {summ_hr5[MAIN_SET]['gain']} of {summ_hr5[MAIN_SET]['pairs']} pairs.") if md else "",
        sub_sentence(),
        (f"For BM25 the in-sub-document MRR gain is {'; '.join(trends)}.") if trends else "",
        ("Restricting to the queries whose original evidence lies entirely inside the gold sub-document (no relabeling) "
         if MAIN_SET == "R" else "Using all queries, with the labels re-derived inside the gold sub-document by the same labeler, ")
        + f"{'gives the same picture' if ((ok_['ci_above'] > 0) == (mk['ci_above'] > 0) and (2 * ok_['gain'] > ok_['pairs']) == (2 * mk['gain'] > mk['pairs'])) else 'gives'}: an MRR gain in "
        f"{ok_['gain']} of {ok_['pairs']} pairs (CI above zero in {ok_['ci_above']}).",
        f"Renormalising the scores over the gold sub-document only, instead of keeping the full-document scores, gives a gain in "
        f"{mr['gain']} of {mr['pairs']} pairs (CI above zero in {mr['ci_above']}).",
    ]
    L += ["# C4: SAADI vs Vanilla inside the gold sub-document (reviewer iVU8, Q2)", "", "## Summary for the rebuttal", ""]
    L.append(" ".join(s for s in sents if s))
    if waiting and any(not w.startswith("(") for w in waiting):
        L += ["", f"Not included (scores not available on CPU, waiting for a recomputation): {', '.join(waiting)}."]
    elif waiting:
        L += ["", f"Not included: {waiting[0]}."]

    L += ["", "## Setup", "",
          f"- Main query set: {SET_DISPLAY[MAIN_SET]}; the other set ({OTHER_SET}) is reported alongside.",
          "- Restriction (main): full-document scores kept, chunks outside the gold sub-document dropped. Second version: every "
          "channel z-scored again over the gold sub-document's chunks (SAADI normalises per query).",
          "- Hit = a sufficient chunk, or every chunk of one evidence group, in the top k (the paper's definition). "
          "Full-ranking measures use the paper's labels on the whole document.",
          "- Sub-document hit@10: at least one of the full-document top-10 chunks belongs to the gold sub-document "
          "(a chunk crossing a boundary belongs to both sub-documents).",
          f"- Paired differences SAADI - Vanilla, percent points, 95% CI from {N_BOOT:,} bootstrap resamples over queries, seed {SEED}. Nothing tuned."]

    L += ["", f"## 1. Is the full-document gain explained by sub-document selection, by the ranking inside it, or both? (set {MAIN_SET})", ""]
    L.append("| Pair | n | Full HR@10 gain | Sub-doc hit@10 gain | Inside HR@5 gain | Inside MRR gain | Reading |")
    L.append("| --- | --- | --- | --- | --- | --- | --- |")
    for p in pairs:
        key, r = p.split("__")
        x = P(p, MAIN_SET, "full_hr10")
        L.append(f"| {ALL_SETTINGS[key].display} {RETRIEVER_DISPLAY[r]} | {x['n'] if x else '-'} | {_ci(x)} | {_ci(P(p, MAIN_SET, 'subdoc_hit10'))} | "
                 f"{_ci(P(p, MAIN_SET, 'in_hr5', 'keep'))} | {_ci(P(p, MAIN_SET, 'in_rr', 'keep'))} | {verdict(p)} |")
    L += ["", "Reading: \"sub-document selection\" when the sub-document hit@10 CI is above zero, \"inside ranking\" when the "
          "inside-MRR CI is above zero, \"both\" when both are. A fixed rule for reading the table, not a test of a decomposition."]

    L += ["", "## 2. Measures per pair", ""]
    for set_name in (MAIN_SET, OTHER_SET):
        L += [f"### {SET_DISPLAY[set_name][:1].upper() + SET_DISPLAY[set_name][1:]}", ""]
        L.append("| Pair | Method | n | Full HR@5 | Full HR@10 | Full med. rank | Sub-doc hit@10 | Top-10 share | In HR@1 | In HR@3 | In HR@5 | In HR@10 | In HR@10 (>10 ch.) | In MRR | In med. rank |")
        L.append("| " + " | ".join(["---"] * 15) + " |")
        for p in pairs:
            key, r = p.split("__")
            for meth in ("van", "pub"):
                t = T(p, set_name, {"van": "vanilla", "pub": "saadi"}[meth])
                if not t:
                    continue
                L.append(f"| {ALL_SETTINGS[key].display} {RETRIEVER_DISPLAY[r]} | {'Vanilla' if meth == 'van' else 'SAADI'} | {t['n']} | "
                         f"{fnum(t['full_hr5'])} | {fnum(t['full_hr10'])} | {fnum(t['full_median_rank'])} | {fnum(t['subdoc_hit10'])} | "
                         f"{fnum(t['subdoc_share10'])} | {fnum(t['keep_in_hr1'])} | {fnum(t['keep_in_hr3'])} | {fnum(t['keep_in_hr5'])} | "
                         f"{fnum(t['keep_in_hr10'])} | {fnum(t['keep_in_hr10_gt'])} (n={t['n_gt10']}) | {fnum(t['keep_in_mrr'])} | {fnum(t['keep_in_median_rank'])} |")
        L.append("")

    L += ["## 3. Paired differences (SAADI - Vanilla, 95% CI)", ""]
    for set_name in (MAIN_SET, OTHER_SET):
        L += [f"### {SET_DISPLAY[set_name][:1].upper() + SET_DISPLAY[set_name][1:]}", ""]
        L.append("| Pair | Full HR@10 | Sub-doc hit@10 | In HR@3 keep / renorm | In HR@5 keep / renorm | In MRR keep / renorm | In HR@10 (>10 ch.) keep / renorm |")
        L.append("| --- | --- | --- | --- | --- | --- | --- |")
        for p in pairs:
            key, r = p.split("__")
            cell = lambda m: f"{_ci(P(p, set_name, m, 'keep'))} / {_ci(P(p, set_name, m, 'renorm'))}"  # noqa: E731
            L.append(f"| {ALL_SETTINGS[key].display} {RETRIEVER_DISPLAY[r]} | {_ci(P(p, set_name, 'full_hr10'))} | "
                     f"{_ci(P(p, set_name, 'subdoc_hit10'))} | {cell('in_hr3')} | {cell('in_hr5')} | {cell('in_rr')} | {cell('in_hr10_gt')} |")
        L.append("")
    L += ["### All eligible queries (full document only)", "", "| Pair | n | Full HR@10 | Sub-doc hit@10 |", "| --- | --- | --- | --- |"]
    for p in pairs:
        key, r = p.split("__")
        x = P(p, "all", "full_hr10")
        L.append(f"| {ALL_SETTINGS[key].display} {RETRIEVER_DISPLAY[r]} | {x['n']} | {_ci(x)} | {_ci(P(p, 'all', 'subdoc_hit10'))} |")

    L += ["", "## 4. Across pairs", ""]
    L.append("| Query set | Restriction | Pairs | Inside MRR gain > 0 | CI fully above 0 | CI fully below 0 |")
    L.append("| --- | --- | --- | --- | --- | --- |")
    for s in (MAIN_SET, OTHER_SET):
        for r in ("keep", "renorm"):
            c = summ[f"{s}_{r}"]
            L.append(f"| {s} | {r} | {c['pairs']} | {c['gain']} | {c['ci_above']} | {c['ci_below']} |")
    L += ["", f"Trend from 32K to 256K (set {MAIN_SET}, keep; SAADI - Vanilla):", ""]
    trend_retr = [r for r in retr_used if sum(f"{ds}_{sz.lower()}__{r}" in frames for ds in ("musique", "qasper") for sz in SIZES) > 2]
    L.append("| Retriever | Dataset | Measure | " + " | ".join(SIZES) + " |")
    L.append("| --- | --- | --- | " + " | ".join("---" for _ in SIZES) + " |")
    for r in trend_retr:
        for ds in ("musique", "qasper"):
            for measure, restr, label in (("full_hr10", "", "Full HR@10"), ("subdoc_hit10", "", "Sub-doc hit@10"), ("in_rr", "keep", "Inside MRR")):
                cells = [_ci(P(f"{ds}_{sz.lower()}__{r}", MAIN_SET, measure, restr)) for sz in SIZES]
                L.append(f"| {RETRIEVER_DISPLAY[r]} | {DATASET_DISPLAY[ds]} | {label} | " + " | ".join(cells) + " |")
    if not trend_retr:
        L.append("| - | - | - | " + " | ".join("-" for _ in SIZES) + " |")
    L += ["", "Only retrievers with scores at the longer settings appear in the trend table; the dense retrievers' "
          "full scores exist for MuSiQue-32K and QASPER-64K (C2), plus whatever was recomputed into `$C4_DENSE_STREAMS_DIR`."]

    L += ["", "## 5. Notes", "",
          "- Query sets. The paper's labels come from the repository labeler run over the whole concatenated document. A span "
          "repeated in another source document gives a second sufficient chunk outside the gold sub-document, and a span that fits "
          "in no single chunk becomes an evidence group of every chunk that shares a boundary token run with it, anywhere in the "
          "document. Set S keeps only queries with all such evidence inside; set R re-derives the labels with the same labeler "
          "on the gold sub-document's chunks only (see step 01).",
          "- The share of queries with evidence outside the gold sub-document grows with the setting length (step 01, section 3), "
          "so the paper's full-document hit rates at the longer settings also count partial-match groups that cross sub-documents.",
          "- Vanilla's ranking is the same under both restrictions: z-scoring a single channel keeps its order.",
          "- Files: `table.csv` (every measure), `paired.csv` (every CI), `table_paper.tex`, per-query ranks under "
          "`experiments/rebuttal/c4_subdoc_control/work/perquery/`."]
    rpath = RESULTS_DIR / "REPORT.md"
    rpath.write_text("\n".join(L) + "\n", encoding="utf-8")
    report.wrote(rpath)

    # ---------------- table_paper.tex
    f1 = lambda x: "-" if x is None else f"{float(x):.1f}"  # noqa: E731
    tex = [r"% C4 sub-document control (generated by step05_report.py). Requires booktabs.",
           r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
           r"\begin{tabular}{llrrrrrrl}", r"\toprule",
           r" & & \multicolumn{2}{c}{Sub-doc hit@10} & \multicolumn{2}{c}{Inside HR@5} & \multicolumn{2}{c}{Inside MRR} & \\",
           r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
           r"Setting & Retriever & Van. & SAADI & Van. & SAADI & Van. & SAADI & $\Delta$MRR [95\% CI] \\", r"\midrule"]
    prev_r = None
    for p in pairs:
        key, r = p.split("__")
        if prev_r is not None and r != prev_r:
            tex.append(r"\midrule")
        prev_r = r
        v, s_ = T(p, MAIN_SET, "vanilla"), T(p, MAIN_SET, "saadi")
        d = P(p, MAIN_SET, "in_rr", "keep")
        best = lambda a, b: (f"\\textbf{{{f1(b)}}}" if b is not None and a is not None and b > a else f1(b))  # noqa: E731
        ci = f"{d['diff']:+.1f} [{d['lo']:+.1f}, {d['hi']:+.1f}]" if d else "-"
        tex.append(f"{ALL_SETTINGS[key].display} & {RETRIEVER_DISPLAY[r]} & {f1(v['subdoc_hit10'])} & {best(v['subdoc_hit10'], s_['subdoc_hit10'])} & "
                   f"{f1(v['keep_in_hr5'])} & {best(v['keep_in_hr5'], s_['keep_in_hr5'])} & {f1(v['keep_in_mrr'])} & "
                   f"{best(v['keep_in_mrr'], s_['keep_in_mrr'])} & {ci} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\caption{Ranking inside the gold sub-document of each query on the concatenated settings "
            + ("(queries with labels re-derived inside the gold sub-document). " if MAIN_SET == "R" else "(queries whose evidence lies in the gold sub-document). ")
            + r"Vanilla is the text channel alone; SAADI uses the published weights; both use the same full-document scores, with "
              r"chunks outside the gold sub-document removed. Sub-doc hit@10: a gold-sub-document chunk in the full-document top 10. "
              r"$\Delta$: SAADI $-$ Vanilla, 95\% CI from 10{,}000 bootstrap resamples over queries.}",
            r"\label{tab:subdoc-control}", r"\end{table}"]
    xpath = RESULTS_DIR / "table_paper.tex"
    xpath.write_text("\n".join(tex) + "\n", encoding="utf-8")
    report.wrote(xpath)

    # ---------------- step report (short; REPORT.md holds the full tables)
    report.p(f"Pairs: {n_pairs} ({', '.join(pairs)}). Main set {MAIN_SET}. Full tables: `{rel(rpath)}`.")
    report.h("Summary (as written at the top of REPORT.md)")
    report.p(" ".join(s for s in sents if s))
    report.h(f"Decomposition (set {MAIN_SET})")
    report.table(["Pair", "Full HR@10 gain", "Sub-doc hit@10 gain", "Inside MRR gain", "Reading"],
                 [[p, _ci(P(p, MAIN_SET, "full_hr10")), _ci(P(p, MAIN_SET, "subdoc_hit10")), _ci(P(p, MAIN_SET, "in_rr", "keep")), verdict(p)] for p in pairs])
    report.h("Across pairs")
    report.table(["Set", "Restriction", "Pairs", "MRR gain > 0", "CI above 0", "CI below 0"],
                 [[s, r, summ[f"{s}_{r}"]["pairs"], summ[f"{s}_{r}"]["gain"], summ[f"{s}_{r}"]["ci_above"], summ[f"{s}_{r}"]["ci_below"]]
                  for s in (MAIN_SET, OTHER_SET) for r in ("keep", "renorm")])
    if waiting:
        report.p("Not included: " + ", ".join(waiting) + ".")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.parse_args()
    cmd = "bash experiments/rebuttal/c4_subdoc_control/sh/05_report.sh" + ("" if len(sys.argv) == 1 else " " + " ".join(sys.argv[1:]))
    report = StepReport(5, "paired bootstrap and final report", command=cmd)
    return run_step(report, lambda r: body(r, ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
