#!/usr/bin/env python3
"""C4 step 06: which SAADI channel carries the gain inside the gold sub-document?

iVU8 (Q2) suspects the summary channel mostly identifies the right sub-document. On the same
scores and the same queries as steps 03-05 (set R, full-document scores kept), four weightings
of the published fusion are compared:
  text      text channel alone (= Vanilla)
  +summary  text + summary channel (lambda_M = 0, lambda_S published)
  +memory   text + memory channel  (lambda_M published, lambda_S = 0)
  full      the published SAADI
For each: MRR and HR@5 inside the gold sub-document, and on the full ranking how often a
top-10 chunk comes from the gold sub-document and what share of the top 10 does. Each variant
is compared with text alone (paired bootstrap over queries, 10,000 resamples, seed 42).
Nothing is tuned; the channel weights are the published ones or zero.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np

from c4_common import (
    ALL_SETTINGS,
    DATASET_DISPLAY,
    N_BOOT,
    REPORT_DIR,
    RESULTS_DIR,
    RETRIEVER_DISPLAY,
    RETRIEVERS,
    SEED,
    SIZES,
    WORK_DIR,
    StepReport,
    fnum,
    published_settings,
    rel,
    require_step_ok,
    run_step,
)
from c4_measures import compute_pair, load_perquery

from saadi.rebuttal.frozen_records import paired_bootstrap_ci  # noqa: E402

VARIANTS = ("text", "summ", "mem", "full")
VARIANT_DISPLAY = {"text": "Text only (Vanilla)", "summ": "Text + summary", "mem": "Text + memory", "full": "Full SAADI"}
CHANNEL_DIR = WORK_DIR / "channels"
MEASURES = (
    # key, label, compared with text alone
    ("in_rr", "MRR inside the gold document", True),
    ("in_hr5", "HR@5 inside the gold document", True),
    ("share10", "Share of the full-document top 10 from the gold document", True),
    ("hit10", "At least one full-document top-10 chunk from the gold document", True),
)


def variant_points(cfg: dict[str, Any]) -> dict[str, tuple[np.ndarray, ...]]:
    if float(cfg["memory_name_weight"]) != 1.0 or float(cfg["lambda_t"]) != 1.0:
        raise RuntimeError("the vectorised fusion assumes lambda_T = mu_N = 1 (as in the published run)")
    lm, ls = float(cfg["lambda_n"]), float(cfg["lambda_h"])
    mu = (np.array([float(cfg["memory_alias_weight"])]), np.array([float(cfg["memory_description_weight"])]))
    a = lambda x: np.array([x])  # noqa: E731
    return {"text": (a(0.0), a(0.0), *mu), "summ": (a(0.0), a(ls), *mu), "mem": (a(lm), a(0.0), *mu), "full": (a(lm), a(ls), *mu)}


def _job(key: str, retriever: str, cfg: dict[str, Any]) -> dict[str, Any]:
    try:
        out = CHANNEL_DIR / f"{key}__{retriever}.csv.gz"
        return compute_pair(ALL_SETTINGS[key], retriever, cfg, points=variant_points(cfg), out_path=out,
                            log=lambda m: print(m, flush=True))
    except Exception as exc:
        return {"setting": key, "retriever": retriever, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def values(frame, v: str) -> dict[str, np.ndarray]:
    """Per-query values on set R (labels re-derived inside the gold document), full-document scores kept."""

    f = frame[frame["in_R"] == 1]
    ins = f[f"inR_keep_rank_{v}"].to_numpy(dtype=float)
    return {
        "in_rr": np.where(np.isfinite(ins), 1.0 / ins, 0.0),
        "in_hr5": (ins <= 5).astype(float),
        "share10": f[f"subdoc_share10_{v}"].to_numpy(dtype=float),
        "hit10": f[f"subdoc_hit10_{v}"].to_numpy(dtype=float),
        "full_hr10": (f[f"full_rank_{v}"].to_numpy(dtype=float) <= 10).astype(float),
    }


def _order(pair: str) -> tuple[int, int, int]:
    key, r = pair.split("__")
    s = ALL_SETTINGS[key]
    return (RETRIEVERS.index(r), ("musique", "qasper").index(s.dataset), SIZES.index(s.size))


def body(report: StepReport, args: argparse.Namespace) -> None:
    require_step_ok(1)
    s02 = require_step_ok(2)
    s03 = require_step_ok(3)
    cfg = published_settings()
    if s02.get("published_settings") != cfg:
        raise RuntimeError("the config's published weights changed since step 02; rerun step 02")
    pairs = list(s03.get("measured_pairs", []))
    if (REPORT_DIR / "step_04.json").exists():
        pairs += list(require_step_ok(4).get("measured_pairs", []))
    pairs = sorted(set(pairs), key=_order)
    if not pairs:
        raise RuntimeError("no measured pair in steps 03/04")

    results: dict[str, dict[str, Any]] = {}
    print(f"[c4] step 06: {len(pairs)} pairs, {args.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=max(1, min(args.workers, len(pairs)))) as pool:
        futs = {pool.submit(_job, *p.split("__"), cfg): p for p in pairs}
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result()

    rows: list[dict[str, Any]] = []
    for p in pairs:
        res = results[p]
        if "error" in res:
            report.fail(f"{p}: {res['error']}")
            continue
        for msg in res["problems"][:5]:
            report.fail(f"{p}: {msg}")
        frame = load_perquery(CHANNEL_DIR / f"{p}.csv.gz")
        report.wrote(CHANNEL_DIR / f"{p}.csv.gz")
        # The text-only and full rankings must equal the ones of steps 03-04 (same scores, same code).
        base = load_perquery(WORK_DIR / "perquery" / f"{p}.csv.gz")
        for v, old in (("text", "van"), ("full", "pub")):
            a = frame[f"inR_keep_rank_{v}"].to_numpy(dtype=float)
            b = base[f"inR_keep_rank_{old}"].to_numpy(dtype=float)
            if len(a) != len(b) or not np.array_equal(np.nan_to_num(a, nan=-1), np.nan_to_num(b, nan=-1)):
                report.fail(f"{p}: {v} ranks differ from step 03/04 ({old}); rerun steps 03/04")
        key, r = p.split("__")
        s = ALL_SETTINGS[key]
        vt = values(frame, "text")
        for v in VARIANTS:
            vv = values(frame, v)
            rec: dict[str, Any] = {"setting": key, "dataset": s.dataset, "size": s.size, "retriever": r, "variant": v,
                                   "n": int(len(vv["in_rr"]))}
            for m in ("in_rr", "in_hr5", "share10", "hit10", "full_hr10"):
                rec[m] = 100.0 * float(vv[m].mean())
                if v != "text":
                    mean, lo, hi = paired_bootstrap_ci(100.0 * (vv[m] - vt[m]), n_resamples=N_BOOT, seed=SEED)
                    rec[f"{m}_diff"], rec[f"{m}_lo"], rec[f"{m}_hi"] = mean, lo, hi
                    rec[f"{m}_sig_up"], rec[f"{m}_sig_down"] = bool(lo > 0), bool(hi < 0)
            rows.append(rec)
    report.data["rows"] = rows
    if not rows:
        return

    out = RESULTS_DIR / "channels.csv"
    cols = list(dict.fromkeys(k for row in rows for k in row))
    with out.open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in cols})
    report.wrote(out)

    def get(key, r, v):
        return next((x for x in rows if (x["setting"], x["retriever"], x["variant"]) == (key, r, v)), None)

    groups = [(r, ds) for r in RETRIEVERS for ds in ("musique", "qasper")
              if any(get(f"{ds}_{sz.lower()}", r, "text") for sz in SIZES)]

    def cell(r, ds, v, m):
        xs = [get(f"{ds}_{sz.lower()}", r, v) for sz in SIZES]
        xs = [x for x in xs if x]
        if not xs:
            return "-"
        mean = sum(x[f"{m}_diff"] for x in xs) / len(xs)
        up = sum(x[f"{m}_sig_up"] for x in xs)
        dn = sum(x[f"{m}_sig_down"] for x in xs)
        return f"{mean:+.1f} ({up}/{len(xs)}{', ' + str(dn) + ' lower' if dn else ''})"

    counts = {}
    for v in ("summ", "mem", "full"):
        for m in ("in_rr", "share10", "hit10"):
            xs = [x for x in rows if x["variant"] == v]
            counts[(v, m)] = (sum(x[f"{m}_diff"] > 0 for x in xs), sum(x[f"{m}_sig_up"] for x in xs), sum(x[f"{m}_sig_down"] for x in xs), len(xs))
    report.data["counts"] = {f"{v}:{m}": c for (v, m), c in counts.items()}

    # ---------------- CHANNELS.md (short; for the rebuttal)
    L = ["# C4 addendum: which channel carries the gain inside the gold sub-document", "",
         "Same scores, queries and restriction as the main C4 analysis (labels re-derived inside the gold document, "
         "full-document scores kept). Each SAADI channel is added to the text channel alone with its published weight; "
         "nothing is tuned. Values: gain over text alone in points, mean over the four lengths; in brackets, the "
         "number of lengths where the 95% CI (10,000 paired bootstrap resamples over queries) is above zero.", ""]
    for m, label in (("in_rr", "MRR inside the gold document"), ("share10", "Share of the full-document top 10 from the gold document"),
                     ("hit10", "At least one full-document top-10 chunk from the gold document")):
        L += [f"**{label}**", "", "| | Text + summary | Text + memory | Full SAADI |", "| --- | --- | --- | --- |"]
        for r, ds in groups:
            L.append(f"| {DATASET_DISPLAY[ds]}, {RETRIEVER_DISPLAY[r]} | {cell(r, ds, 'summ', m)} | {cell(r, ds, 'mem', m)} | {cell(r, ds, 'full', m)} |")
        L.append("")
    def abs_cell(r, ds, v, m):
        xs = [x for x in (get(f"{ds}_{sz.lower()}", r, v) for sz in SIZES) if x]
        if not xs:
            return "-"
        mean = sum(x[m] for x in xs) / len(xs)
        if v == "text":
            return f"{mean:.1f}"
        return f"{mean:.1f} ({sum(x[f'{m}_sig_up'] for x in xs)}/{len(xs)})"

    L += ["**Absolute values (%, mean over the four lengths; in brackets, lengths with the 95% CI of the gain over text only above zero)**", ""]
    for m, label in (("in_rr", "MRR inside the gold document"), ("share10", "Share of the full-document top 10 from the gold document"),
                     ("hit10", "At least one full-document top-10 chunk from the gold document")):
        L += [f"*{label}*", "", "| | Text only (Vanilla) | Text + summary | Text + memory | Full SAADI |", "| --- | --- | --- | --- | --- |"]
        for r, ds in groups:
            L.append(f"| {DATASET_DISPLAY[ds]}, {RETRIEVER_DISPLAY[r]} | " + " | ".join(abs_cell(r, ds, v, m) for v in VARIANTS) + " |")
        L.append("")
    L += ["Across all pairs (gain > 0 / CI above zero / CI below zero, out of N):", "",
          "| | Text + summary | Text + memory | Full SAADI |", "| --- | --- | --- | --- |"]
    for m, label in (("in_rr", "MRR inside the gold document"), ("share10", "Top-10 share from the gold document"),
                     ("hit10", "At least one top-10 chunk from the gold document")):
        L.append(f"| {label} | " + " | ".join(f"{counts[(v, m)][0]} / {counts[(v, m)][1]} / {counts[(v, m)][2]} of {counts[(v, m)][3]}"
                                             for v in ("summ", "mem", "full")) + " |")
    L += ["", "Per-setting values: `channels.csv`."]
    md = RESULTS_DIR / "CHANNELS.md"
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    report.wrote(md)

    # ---------------- step report
    report.p(f"Pairs: {len(pairs)}. Set R, full-document scores kept (the main C4 analysis). Channel weights: "
             f"lambda_M={cfg['lambda_n']}, lambda_S={cfg['lambda_h']} (published) or 0.")
    report.p("Check: the text-only and full rankings equal those of steps 03-04 for every pair (else a problem is listed).")
    report.h("Gain over text alone, mean over the four lengths (lengths with CI above zero)")
    for m, label in (("in_rr", "MRR inside the gold document"), ("in_hr5", "HR@5 inside the gold document"),
                     ("share10", "Share of the full-document top 10 from the gold document"),
                     ("hit10", "At least one full-document top-10 chunk from the gold document")):
        report.p(f"**{label}**")
        report.p("")
        report.table(["", "Text + summary", "Text + memory", "Full SAADI"],
                     [[f"{DATASET_DISPLAY[ds]}, {RETRIEVER_DISPLAY[r]}", cell(r, ds, "summ", m), cell(r, ds, "mem", m), cell(r, ds, "full", m)]
                      for r, ds in groups])
    report.h("Per setting: MRR inside the gold document (text alone, and gain of each variant)")
    tab = []
    for p in pairs:
        key, r = p.split("__")
        t = get(key, r, "text")
        tab.append([ALL_SETTINGS[key].display, RETRIEVER_DISPLAY[r], fnum(t["in_rr"])]
                   + [f"{get(key, r, v)['in_rr_diff']:+.1f}{'*' if get(key, r, v)['in_rr_sig_up'] else ''}" for v in ("summ", "mem", "full")])
    report.table(["Setting", "Retriever", "Text only", "+ summary", "+ memory", "Full"], tab)
    report.p("* = 95% CI above zero.")
    for name, res in results.items():
        if res.get("traceback"):
            report.h(f"Traceback ({name})")
            report.code(res["traceback"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=int(os.environ.get("C4_WORKERS", "4")))
    args = ap.parse_args()
    cmd = "bash experiments/rebuttal/c4_subdoc_control/sh/06_channels.sh" + ("" if len(sys.argv) == 1 else " " + " ".join(sys.argv[1:]))
    report = StepReport(6, "channel breakdown inside the gold sub-document", command=cmd)
    return run_step(report, lambda r: body(r, args))


if __name__ == "__main__":
    raise SystemExit(main())
