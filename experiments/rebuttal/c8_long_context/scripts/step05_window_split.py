"""C8 step 05 (CPU): NovelHopQA split by the long-context reader's native window.

The full-document condition reads only the books inside the native 262,144-token window (10 books,
526 questions). The retrieval conditions answered every question, so their accuracy can also be
given on the 6 longer books (265K-760K tokens, Qwen2.5-7B-Instruct-1M tokenizer, 382 questions) and
on all 16 books. Same scores and bootstrap as step 03: questions in step 00's seed-42 order of C5's
query list, 10,000 resamples, seed 42. Self-check: the in-window rows must equal step 03's
per_dataset/novelhopqa/table.csv exactly, otherwise the step fails.

Reads only pushed files: results/rebuttal/c8_long_context/answers/novelhopqa/*.jsonl,
full_doc.skipped.json, prepare/doc_lengths.csv and C5's prepare/labels_para_novelhopqa.jsonl.gz.
Writes per_dataset/novelhopqa/window_split.csv and a section of REPORT.md (between markers, so a
rerun replaces it; sh/99_report.sh runs this step after step 99).
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
import traceback
from typing import Any

import numpy as np

import c8_common as c8

cc = c8.cc
DS = "novelhopqa"
CONDS = ("saadi", "vanilla", "closed_book")
BEGIN, END = "<!-- c8-window-split:begin -->", "<!-- c8-window-split:end -->"


def log(msg: str) -> None:
    print(f"[c8-05] {cc.now()} {msg}", flush=True)


def load(path) -> dict[str, dict[str, Any]]:
    rows, keys = {}, set()
    for r in cc.read_jsonl(path):
        r.update(cc.score_answer(r["output"], r["gold"]))
        rows[str(r["query_id"])] = r
        keys.add(r.get("key"))
    if len(keys) != 1:
        raise RuntimeError(f"{path} holds answers from {len(keys)} different inputs/server modes; rerun step 02/03 first")
    return rows


def main() -> int:
    paths = c8.C8Paths.from_env()
    n_boot = int(os.environ.get("C8_BOOTSTRAP_RESAMPLES", 10000))
    rep = paths.reports / "step_05_window_split.md"
    out_csv = paths.results / "per_dataset" / DS / "window_split.csv"
    info: dict[str, Any] = {"step": "05", "dataset": DS, "commit": cc.git_commit(paths.root), "files": []}
    status, code, t0 = "OK", 0, time.time()
    try:
        adir = paths.results / "answers" / DS
        beyond = {str(r["query_id"]) for r in json.load(open(adir / "full_doc.skipped.json", encoding="utf-8"))}
        inside = set(load(adir / "full_doc.jsonl"))
        ans = {c: load(adir / f"{c}.jsonl") for c in CONDS}
        # Step 00's order: seed-42 permutation of C5's query list (C5's paragraph labels keep that list).
        order = list(dict.fromkeys(str(r["query_id"]) for r in cc.read_jsonl(paths.c5_para_labels(DS))))
        perm = np.random.default_rng(c8.SEED).permutation(len(order))
        rank = {order[int(j)]: i for i, j in enumerate(perm)}
        lengths = {r["doc_id"]: int(r["tokens"]) for r in csv.DictReader(open(paths.results / "prepare" / "doc_lengths.csv", encoding="utf-8")) if r["dataset"] == DS}
        groups = {"inside_window": inside, "beyond_window": beyond, "all_books": inside | beyond}
        rows = []
        for g, qs in groups.items():
            for c in CONDS:
                sel = sorted((q for q in ans[c] if q in qs), key=lambda q: rank[q])
                docs = {ans[c][q]["doc_id"] for q in sel}
                row = {"dataset": DS, "group": g, "condition": c, "condition_display": c8.CONDITION_DISPLAY[c], "reader": "Qwen2.5-7B-Instruct-1M",
                       "n": len(sel), "books": len(docs), "tokens_min": min(lengths[d] for d in docs), "tokens_max": max(lengths[d] for d in docs)}
                for m in c8.MEASURES:
                    mu, lo, hi = c8.paired_bootstrap_ci([100.0 * ans[c][q][m] for q in sel], n_resamples=n_boot, seed=c8.SEED)
                    row.update({m: mu, f"{m}_lo": lo, f"{m}_hi": hi})
                rows.append(row)
        # Self-check against step 03.
        t03 = {r["condition"]: r for r in csv.DictReader(open(paths.results / "per_dataset" / DS / "table.csv", encoding="utf-8"))}
        for r in rows:
            if r["group"] != "inside_window":
                continue
            for k in ("n", "f1", "f1_lo", "f1_hi", "em_lo", "em_hi"):
                if abs(float(r[k]) - float(t03[r["condition"]][k])) > 1e-9:
                    raise RuntimeError(f"self-check failed: {r['condition']} {k} {r[k]} != step 03's {t03[r['condition']][k]}")
        info["self_check"] = "inside-window rows equal per_dataset/novelhopqa/table.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        info["files"].append(paths.rel(out_csv))
        R = {(r["group"], r["condition"]): r for r in rows}
        b, i = R[("beyond_window", "saadi")], R[("inside_window", "saadi")]
        ci = lambda r: f"{r['f1']:.1f} [{r['f1_lo']:.1f}, {r['f1_hi']:.1f}]"  # noqa: E731
        sec = [BEGIN, "", "## NovelHopQA beyond the native window", "",
               f"The full-document condition reads only the {i['books']} books inside the native 262,144-token window. The retrieval conditions answered every question, so their "
               f"accuracy is also given on the {b['books']} longer books ({b['tokens_min']:,}-{b['tokens_max']:,} tokens, Qwen2.5-7B-Instruct-1M tokenizer, {b['n']} questions): "
               f"with the same 7B reader, SAADI's top 10 gives F1 {ci(b)} there, against {ci(i)} inside the window.", "",
               "F1 in percent, 95% bootstrap interval over questions (10,000 resamples, seed 42; same order as step 03). File: `" + paths.rel(out_csv) + "`.", ""]
        sec += cc.md_table(["Books", "Condition", "n", "Books (tokens)", "EM", "F1", "Containment"],
                           [[r["group"], r["condition_display"], r["n"], f"{r['books']} ({r['tokens_min']:,}-{r['tokens_max']:,})"]
                            + [f"{r[m]:.1f} [{r[m + '_lo']:.1f}, {r[m + '_hi']:.1f}]" for m in c8.MEASURES] for r in rows]) + ["", END]
        report_md = paths.results / "REPORT.md"
        text = report_md.read_text(encoding="utf-8") if report_md.exists() else ""
        if BEGIN in text and END in text:
            text = text[: text.index(BEGIN)].rstrip() + "\n\n" + text[text.index(END) + len(END):].lstrip()
        report_md.write_text(text.rstrip() + "\n\n" + "\n".join(sec) + "\n", encoding="utf-8")
        info["files"].append(paths.rel(report_md))
        info["rows"] = rows
        info["one_line"] = f"SAADI (7B) F1 beyond the window {ci(b)} (n={b['n']}), inside {ci(i)} (n={i['n']})"
        info["section"] = sec
    except Exception as exc:
        status, code = "FAILED", 1
        info["traceback"] = traceback.format_exc()
        info["one_line"] = f"FAILED: {type(exc).__name__}: {str(exc)[:110]}"
    info["seconds"] = round(time.time() - t0)
    info["status"] = status
    L = [f"# C8 long-context reader - step 05 NovelHopQA split by the native window: {status}", "", f"Written {cc.now()}; commit {info['commit']}.", ""]
    if info.get("section"):
        L += [x for x in info["section"] if x not in (BEGIN, END)] + ["", f"Self-check: {info['self_check']}.", ""]
    if info.get("traceback"):
        L += ["## Error", "", "```text", *info["traceback"].splitlines()[-40:], "```", ""]
    L += ["## Files written", ""] + [f"- `{x}`" for x in info["files"]] + [f"- `{paths.rel(rep)}`"]
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text("\n".join(L) + "\n", encoding="utf-8")
    cc.write_json(rep.with_suffix(".json"), {k: v for k, v in info.items() if k != "section"})
    rep.with_suffix(".summary.txt").write_text(info["one_line"] + "\n", encoding="utf-8")
    log(f"done: {status}: {info['one_line']}")
    return code


if __name__ == "__main__":
    sys.exit(main())
