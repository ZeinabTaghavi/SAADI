"""C8 step 03 (CPU): score one dataset from its saved answers.

All comparisons use the COMMON queries: those answered in all four C8 conditions (the table gives
their number; queries skipped because their document does not fit, or not started at the deadline,
drop out of every condition alike). Per condition: EM, token F1 and answer containment (percent) with
a 95% interval (10,000 bootstrap resamples over queries, seed 42). Paired differences on the same
queries: full document - SAADI top 10, SAADI - Vanilla (7B reader), plus a few extra pairs. SAADI top
10 read by C5's reader (Qwen3-30B-A3B) is added from C5's saved answers on the same queries. Cost per
query: input and output tokens, seconds per request, amortised wall seconds; for the full document
also the uncached seconds (the document read without the prefix cache). Accuracy split by where the
gold evidence sits in the document (first / middle / last third).
Writes results/rebuttal/c8_long_context/per_dataset/<dataset>/{table,paired,cost,position}.csv and
summary.json. Scores are recomputed from the saved outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from statistics import mean, median
from typing import Any

import c8_common as c8
import c8_reader as rd

cc = c8.cc
C5_READER = "saadi_c5_reader"
DISPLAY = {**c8.CONDITION_DISPLAY, C5_READER: "SAADI top 10, C5 reader (Qwen3-30B-A3B)"}


def log(msg: str) -> None:
    print(f"[c8-03] {cc.now()} {msg}", flush=True)


def ci(vals: list[float], n_boot: int) -> dict[str, Any]:
    if not vals:
        return {"n": 0, "mean": None, "lo": None, "hi": None}
    m, lo, hi = c8.paired_bootstrap_ci([100.0 * v for v in vals], n_resamples=n_boot, seed=c8.SEED)
    return {"n": len(vals), "mean": m, "lo": lo, "hi": hi}


def write_csv(path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(k for r in rows for k in r)) or ["empty"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def load_answers(paths: c8.C8Paths, inp: rd.Inputs, mode: str) -> dict[str, dict[str, dict[str, Any]]]:
    out = {}
    for cond in c8.CONDITIONS:
        p = paths.answers_file(inp.dataset, cond)
        if not p.exists():
            out[cond] = {}
            continue
        key = inp.key(cond, mode)
        rows = {}
        for r in cc.read_jsonl(p):
            if r.get("key") != key:
                continue
            r.update(cc.score_answer(r["output"], r["gold"]))
            rows[r["query_id"]] = r
        out[cond] = rows
    return out


def load_c5_reader(paths: c8.C8Paths, ds: str) -> tuple[dict[str, dict[str, Any]], str]:
    p = paths.c5_answers_file(ds, "saadi")
    if not p.exists():
        return {}, f"not found ({paths.rel(p)}): C5's reader has not answered yet"
    rows = {}
    for r in cc.read_jsonl(p):
        r.update(cc.score_answer(r["output"], r["gold"]))
        rows[str(r["query_id"])] = r
    return rows, paths.rel(p)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=c8.DATASETS)
    args = ap.parse_args(argv)
    paths = c8.C8Paths.from_env()
    ds = args.dataset
    n_boot = int(os.environ.get("C8_BOOTSTRAP_RESAMPLES", 10000))
    info: dict[str, Any] = {"step": "03", "dataset": ds, "commit": cc.git_commit(paths.root), "host": os.uname().nodename, "files": []}
    status, code, t0 = "OK", 0, time.time()
    # Old scores of this dataset go first: a failed scoring must not leave them for step 99.
    for name in ("table.csv", "paired.csv", "cost.csv", "position.csv", "summary.json"):
        (paths.results / "per_dataset" / ds / name).unlink(missing_ok=True)
    try:
        inp = rd.Inputs(paths, ds)
        rep02 = paths.reports / f"step_02_{ds}_reader.json"
        r02 = cc.read_json(rep02) if rep02.exists() else {}
        mode = os.environ.get("C8_SERVED_MODE") or (r02.get("server") or {}).get("mode") or "dca"
        info["mode"] = mode
        ans = load_answers(paths, inp, mode)
        if not any(ans.values()):
            raise RuntimeError(f"no answers under {paths.rel(paths.results / 'answers' / ds)} for server mode {mode}")
        c5r, c5src = load_c5_reader(paths, ds)
        info["c5_reader_answers"] = c5src
        sel = {q["query_id"] for q in inp.selected()}
        common = sorted(set.intersection(*[set(ans[c]) for c in c8.CONDITIONS]) & sel, key=lambda q: inp.by_id[q]["subset_rank"])
        info.update({"selected": len(sel), "answered": {c: len(v) for c, v in ans.items()}, "common": len(common),
                     "c5_reader_on_common": len(set(common) & set(c5r))})
        if not common:
            raise RuntimeError("no query is answered in all four conditions")
        conds = list(c8.CONDITIONS) + ([C5_READER] if c5r else [])
        get = lambda c, q: (c5r if c == C5_READER else ans[c]).get(q)  # noqa: E731
        # Accuracy.
        table = []
        for c in conds:
            qs = [q for q in common if get(c, q) is not None]
            row = {"dataset": ds, "condition": c, "condition_display": DISPLAY[c], "reader": "Qwen3-30B-A3B (C5)" if c == C5_READER else "Qwen2.5-7B-Instruct-1M", "n": len(qs)}
            for m in c8.MEASURES:
                s = ci([get(c, q)[m] for q in qs], n_boot)
                row.update({m: s["mean"], f"{m}_lo": s["lo"], f"{m}_hi": s["hi"]})
            table.append(row)
        # Paired differences.
        paired = []
        pairs = list(c8.PAIRS) + list(c8.PAIRS_EXTRA) + ([("full_doc", C5_READER), ("saadi", C5_READER)] if c5r else [])
        for a, b in pairs:
            qs = [q for q in common if get(a, q) is not None and get(b, q) is not None]
            row = {"dataset": ds, "a": a, "b": b, "pair": f"{DISPLAY[a]} - {DISPLAY[b]}", "n": len(qs), "main": (a, b) in c8.PAIRS}
            for m in c8.MEASURES:
                d, lo, hi = c8.paired_bootstrap_ci([100.0 * (get(a, q)[m] - get(b, q)[m]) for q in qs], n_resamples=n_boot, seed=c8.SEED) if qs else (None, None, None)
                row.update({m: d, f"{m}_lo": lo, f"{m}_hi": hi})
            paired.append(row)
        # Cost.
        timing = [r for r in (cc.read_jsonl(paths.results / "timing" / "timing.jsonl") if (paths.results / "timing" / "timing.jsonl").exists() else []) if r["dataset"] == ds]
        docs_p = paths.answers_file(ds, "full_doc").with_name("full_doc_documents.jsonl")
        doc_rows = {}
        if docs_p.exists():
            for r in cc.read_jsonl(docs_p):
                if r.get("key") == inp.key("full_doc", mode):
                    doc_rows[r["doc_id"]] = r  # the latest run of each document
        cost = []
        for c in conds:
            rows = [get(c, q) for q in common if get(c, q) is not None]
            if not rows:
                continue
            ptok = [r.get("server_prompt_tokens") or r.get("prompt_tokens") or 0 for r in rows]
            otok = [r.get("completion_tokens") for r in rows if r.get("completion_tokens") is not None]
            lat = [r["latency_s"] for r in rows if r.get("latency_s") is not None]
            row = {"dataset": ds, "condition": c, "condition_display": DISPLAY[c], "n": len(rows), "input_tokens_mean": mean(ptok), "input_tokens_median": median(ptok),
                   "output_tokens_mean": mean(otok) if otok else None, "request_latency_s_mean": mean(lat) if lat else None}
            tim = [t for t in timing if t["condition"] == c]
            if c == "full_doc":
                unc = [t["seconds"] for t in tim if t["kind"] == "uncached"]
                cac = [t["seconds"] for t in tim if t["kind"] in ("cached", "cached_repeat")]
                row["single_request_s_uncached_timing"] = mean(unc) if unc else None
                row["single_request_s_cached_timing"] = mean(cac) if cac else None
                # Without the prefix cache every question reads its whole document: the document's
                # measured uncached seconds, averaged over the questions.
                per_q = [doc_rows[r["doc_id"]]["uncached_seconds"] for r in rows if r["doc_id"] in doc_rows]
                row["seconds_per_query_without_cache"] = mean(per_q) if per_q else None
                used = [d for d in doc_rows.values() if any(r["doc_id"] == d["doc_id"] for r in rows)]
                n_used = sum(1 + d["other_answered"] for d in used)
                row["seconds_per_query_with_cache_amortised"] = (sum(d["document_wall_s"] for d in used) / n_used) if n_used else None
            else:
                row["single_request_s_timing"] = mean(t["seconds"] for t in tim) if tim else None
                cinfo = (r02.get("conditions") or {}).get(c) or {}
                row["seconds_per_query_amortised"] = (cinfo["wall_seconds"] / cinfo["asked"]) if cinfo.get("asked") and cinfo.get("wall_seconds") else None
            if c == C5_READER:
                row["note"] = "C5's reader on C5's server (Qwen3 tokenizer counts); seconds not comparable"
            cost.append(row)
        # Evidence position.
        position = []
        for c in ("full_doc", "saadi", "vanilla"):
            for subset, keep in (("all", lambda q: True), ("evidence_in_one_third", lambda q: inp.by_id[q].get("evidence_spread") is False)):
                for third in c8.THIRDS:
                    qs = [q for q in common if inp.by_id[q].get("evidence_third") == third and keep(q)]
                    row = {"dataset": ds, "condition": c, "subset": subset, "third": third, "n": len(qs)}
                    for m in c8.MEASURES:
                        s = ci([ans[c][q][m] for q in qs], n_boot)
                        row.update({m: s["mean"], f"{m}_lo": s["lo"], f"{m}_hi": s["hi"]})
                    position.append(row)
            qs = [q for q in common if inp.by_id[q].get("evidence_third") is None]
            position.append({"dataset": ds, "condition": c, "subset": "all", "third": "no located evidence", "n": len(qs),
                             **{m: (mean(100 * ans[c][q][m] for q in qs) if qs else None) for m in c8.MEASURES}})
        # Skipped and not started.
        info["skipped"] = {c: (r02.get("conditions") or {}).get(c, {}).get("skipped") for c in c8.CONDITIONS}
        info["not_started_deadline"] = ((r02.get("conditions") or {}).get("full_doc") or {}).get("not_started_deadline")
        out = paths.results / "per_dataset" / ds
        for name, rows in (("table", table), ("paired", paired), ("cost", cost), ("position", position)):
            write_csv(out / f"{name}.csv", rows)
            info["files"].append(paths.rel(out / f"{name}.csv"))
        summary = {"dataset": ds, "mode": mode, "selected": len(sel), "common": len(common), "answered": info["answered"], "table": table, "paired": paired,
                   "cost": cost, "position": position, "c5_reader_answers": c5src, "skipped": info["skipped"], "not_started_deadline": info["not_started_deadline"],
                   "subset": c8.subset_size(ds), "evidence_source": inp.queries[0].get("evidence_source")}
        cc.write_json(out / "summary.json", summary)
        info["files"].append(paths.rel(out / "summary.json"))
        info.update({"table": table, "paired": paired, "cost": cost, "position": position})
        fd = next(r for r in table if r["condition"] == "full_doc")
        sa = next(r for r in table if r["condition"] == "saadi")
        pd = next(r for r in paired if (r["a"], r["b"]) == ("full_doc", "saadi"))
        info["one_line"] = f"F1 full doc {fd['f1']:.1f}, SAADI {sa['f1']:.1f} (7B, n={len(common)}); full-SAADI {pd['f1']:+.1f} [{pd['f1_lo']:+.1f}, {pd['f1_hi']:+.1f}]"
    except Exception as exc:
        status, code = "FAILED", 1
        info["traceback"] = traceback.format_exc()
        info["one_line"] = f"FAILED: {type(exc).__name__}: {str(exc)[:110]}"
    info["seconds"] = round(time.time() - t0)
    info["status"] = status
    write_report(paths, info, status)
    log(f"done: {status}")
    return code


def write_report(paths: c8.C8Paths, info: dict[str, Any], status: str) -> None:
    f = c8.fmt
    ds = info["dataset"]
    rep = paths.reports / f"step_03_{ds}_score.md"
    L = [f"# C8 long-context reader - step 03 scores, {c8.DATASET_DISPLAY[ds]}: {status}", "",
         f"Written {cc.now()}; commit {info['commit']}; host {info['host']}; {info['seconds']} s. Server mode {info.get('mode')}.", ""]
    if info.get("table"):
        L += [f"Common queries (answered in all four conditions): {info['common']} of {info['selected']} selected. Answered per condition: {info['answered']}. "
              f"C5 reader answers: {info['c5_reader_answers']} ({info['c5_reader_on_common']} on the common queries).", "",
              "Accuracy (percent, 95% bootstrap interval over queries, 10,000 resamples, seed 42):", ""]
        L += cc.md_table(["Condition", "Reader", "n", "EM", "F1", "Containment"],
                         [[r["condition_display"], r["reader"], r["n"]] + [f"{f(r[m])} [{f(r[m + '_lo'])}, {f(r[m + '_hi'])}]" for m in c8.MEASURES] for r in info["table"]]) + [""]
        L += ["Paired differences (points, same queries):", ""]
        L += cc.md_table(["Pair", "n", "EM", "F1", "Containment"],
                         [[("**" + r["pair"] + "**") if r["main"] else r["pair"], r["n"]] + [f"{r[m]:+.1f} [{r[m + '_lo']:+.1f}, {r[m + '_hi']:+.1f}]" if r[m] is not None else "—" for m in c8.MEASURES]
                          for r in info["paired"]]) + [""]
        L += ["Cost per query:", ""]
        L += cc.md_table(["Condition", "n", "Input tokens (mean)", "Output tokens (mean)", "Request latency in the run (s)", "Single request alone, timing (s)", "Amortised wall s / query", "Full doc without cache (s / query)"],
                         [[r["condition_display"], r["n"], f(r["input_tokens_mean"], 0), f(r.get("output_tokens_mean"), 1), f(r.get("request_latency_s_mean"), 2),
                           (f"{f(r.get('single_request_s_uncached_timing'), 1)} uncached / {f(r.get('single_request_s_cached_timing'), 2)} cached" if r["condition"] == "full_doc" else f(r.get("single_request_s_timing"), 2)),
                           f(r.get("seconds_per_query_with_cache_amortised") if r["condition"] == "full_doc" else r.get("seconds_per_query_amortised"), 2),
                           f(r.get("seconds_per_query_without_cache"), 1)] for r in info["cost"]]) + [""]
        L += ["F1 by the position of the gold evidence in the document (all common queries, and those whose evidence lies in one third):", ""]
        L += cc.md_table(["Condition", "Subset", "Third", "n", "EM", "F1", "Containment"],
                         [[c8.CONDITION_DISPLAY[r["condition"]], r["subset"], r["third"], r["n"]] + [f(r.get(m)) for m in c8.MEASURES] for r in info["position"]]) + [""]
        if any(info.get("skipped", {}).values()) or info.get("not_started_deadline"):
            L += [f"Left out: skipped (prompt longer than the served window) {info['skipped']}; full-document queries not started at the deadline: {info.get('not_started_deadline')}.", ""]
    if info.get("traceback"):
        L += ["## Error", "", "```text", *info["traceback"].splitlines()[-60:], "```", ""]
    L += ["## Files written", ""] + [f"- `{x}`" for x in info["files"]] + [f"- `{paths.rel(rep)}`"]
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text("\n".join(L) + "\n", encoding="utf-8")
    cc.write_json(rep.with_suffix(".json"), info)
    rep.with_suffix(".summary.txt").write_text(info.get("one_line", status) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
