"""C5 step 03 (CPU): score one dataset from its saved answers.

Per condition: EM, token F1 and answer containment (percent) on all queries, and split by
"gold evidence in the context" (yes / no / no evidence labels); hit rate at 10 and the evidence
rate on the same queries; context length in tokens (mean, max). Paired differences over the
queries both conditions answered, 95% interval from 10,000 bootstrap resamples over queries
(seed 42, E2's paired_bootstrap_ci). Scores are recomputed from the saved outputs here, so a
change of the scorer never needs the GPU again.
Writes results/rebuttal/c5_end_to_end/per_dataset/<dataset>/{table.csv,paired.csv,summary.json}.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from statistics import mean
from typing import Any

import c5_common as cc

SUBSETS = (("all", None), ("evidence_yes", True), ("evidence_no", False), ("no_labels", "none"))


def log(msg: str) -> None:
    print(f"[c5-03] {cc.now()} {msg}", flush=True)


def load_answers(paths: cc.C5Paths, ds: str) -> dict[str, dict[str, dict[str, Any]]]:
    out = {}
    for cond in cc.ALL_CONDITIONS:
        p = paths.answers_file(ds, cond)
        if not p.exists():
            continue
        rows = {}
        for r in cc.read_jsonl(p):
            r.update(cc.score_answer(r["output"], r["gold"]))  # rescore from the saved output
            rows[r["query_id"]] = r
        out[cond] = rows
    return out


def summarise(ds: str, answers: dict[str, dict[str, dict[str, Any]]], manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    table, paired, summary = [], [], {"dataset": ds, "conditions": {}, "paired": {}}
    for cond, rows in answers.items():
        vals = list(rows.values())
        expected = (manifest["conditions"].get(cond) or {}).get("n")
        s: dict[str, Any] = {"n": len(vals), "expected": expected, "complete": expected == len(vals)}
        for name, flag in SUBSETS:
            if flag is None:
                sub = vals
            elif flag == "none":
                sub = [r for r in vals if r.get("evidence_in_context") is None]
            else:
                sub = [r for r in vals if r.get("evidence_in_context") is flag]
            s[name] = {"n": len(sub)}
            for m in cc.MEASURES:
                v = 100.0 * mean(r[m] for r in sub) if sub else None
                s[name][m] = v
                table.append({"dataset": ds, "condition": cond, "subset": name, "measure": m, "n": len(sub), "value": v})
        hr = cc.hit_rate(r.get("retrieval_hit10") for r in vals)
        ev = cc.hit_rate(r.get("evidence_in_context") for r in vals)
        s["hr10"], s["evidence_in_context"] = hr, ev
        ctx = [r.get("context_tokens") or 0 for r in vals]
        s["context_tokens_mean"], s["context_tokens_max"] = (mean(ctx), max(ctx)) if ctx else (None, None)
        s["passages_mean"] = mean(r.get("n_passages") or 0 for r in vals) if vals else None
        for m, v, n in (("hr10", hr["hr_percent"], hr["eligible"]), ("evidence_in_context", ev["hr_percent"], ev["eligible"]),
                        ("context_tokens_mean", s["context_tokens_mean"], len(vals)), ("context_tokens_max", s["context_tokens_max"], len(vals)),
                        ("passages_mean", s["passages_mean"], len(vals))):
            table.append({"dataset": ds, "condition": cond, "subset": "all", "measure": m, "n": n, "value": v})
        summary["conditions"][cond] = s
    seed, n_boot = cc.SEED, int(os.environ.get("C5_BOOTSTRAP_RESAMPLES", 10000))
    for a, b in cc.PAIRS + cc.PAIRS_EXTRA:
        if a not in answers or b not in answers:
            continue
        common = [q for q in answers[a] if q in answers[b]]
        res = {"n": len(common)}
        for m in cc.MEASURES + ("hr10", "evidence"):
            if m in ("hr10", "evidence"):
                key = "retrieval_hit10" if m == "hr10" else "evidence_in_context"
                qs = [q for q in common if answers[a][q].get(key) is not None and answers[b][q].get(key) is not None]
                diffs = [100.0 * (float(answers[a][q][key]) - float(answers[b][q][key])) for q in qs]
            else:
                qs = common
                diffs = [100.0 * (answers[a][q][m] - answers[b][q][m]) for q in qs]
            if not diffs:
                continue
            d, lo, hi = cc.paired_bootstrap_ci(diffs, n_resamples=n_boot, seed=seed)
            res[m] = {"n": len(qs), "diff": d, "ci": [lo, hi]}
            paired.append({"dataset": ds, "a": a, "b": b, "measure": m, "n": len(qs), "diff": d, "ci_low": lo, "ci_high": hi})
        summary["paired"][f"{a}-{b}"] = res
    return table, paired, summary


def write_csv(path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(k for r in rows for k in r)) or ["empty"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def f(v: Any, nd: int = 1) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=cc.DATASETS)
    args = ap.parse_args(argv)
    paths = cc.C5Paths.from_env()
    ds = args.dataset
    info: dict[str, Any] = {"step": "03", "dataset": ds, "commit": cc.git_commit(paths.root), "host": os.uname().nodename, "files": []}
    status, code, t0 = "OK", 0, time.time()
    summary: dict[str, Any] = {}
    try:
        manifest = cc.read_json(paths.context_manifest(ds))
        answers = load_answers(paths, ds)
        if not answers:
            raise RuntimeError(f"no answers under {paths.results / 'answers' / ds}")
        table, paired, summary = summarise(ds, answers, manifest)
        summary["token_budget"] = manifest.get("token_budget")
        out = paths.results / "per_dataset" / ds
        write_csv(out / "table.csv", table)
        write_csv(out / "paired.csv", paired)
        cc.write_json(out / "summary.json", summary)
        info["files"] += [paths.rel(out / n) for n in ("table.csv", "paired.csv", "summary.json")]
        incomplete = [c for c, s in summary["conditions"].items() if not s["complete"]]
        if incomplete:
            info["warnings"] = [f"incomplete conditions: {incomplete}"]
        sa = summary["conditions"].get("saadi", {})
        pv = summary["paired"].get("saadi-vanilla", {}).get("f1")
        info["one_line"] = (f"F1 SAADI {f(sa.get('all', {}).get('f1'))}" + (f", SAADI-Vanilla {pv['diff']:+.1f} [{pv['ci'][0]:+.1f}, {pv['ci'][1]:+.1f}]" if pv else "")
                            + (f"; incomplete {incomplete}" if incomplete else ""))
    except Exception as exc:
        status, code = "FAILED", 1
        info["traceback"] = traceback.format_exc()
        info["one_line"] = f"FAILED: {type(exc).__name__}: {str(exc)[:110]}"
    info["seconds"] = round(time.time() - t0, 1)
    rep = paths.reports / f"step_03_{ds}_score.md"
    L = [f"# C5 end-to-end QA - step 03 scores, {cc.DATASET_DISPLAY[ds]}: {status}", "",
         f"Written {cc.now()}; commit {info['commit']}; host {info['host']}.", "",
         "Scores in percent. EM and token F1 after SQuAD normalisation (best over the gold answers); Contain = a gold answer inside the output. "
         "Evidence yes/no = the paper's hit rule (a gold chunk or a whole silver group) applied to the chunks in the context. "
         "HR@10 = the condition's retrieval hit (RAPTOR: its upper-bound HR@10; A-RAG: hit over the chunks it read; none for the cut SAADI rows, whose retrieval measure is the evidence in the cut context).", ""]
    if summary.get("conditions"):
        rows = []
        for c in cc.ALL_CONDITIONS:
            s = summary["conditions"].get(c)
            if not s:
                continue
            rows.append([cc.CONDITION_DISPLAY[c], f"{s['n']}" + ("" if s["complete"] else f"/{s['expected']} INCOMPLETE"), f(s["all"]["em"]), f(s["all"]["f1"]), f(s["all"]["contain"]),
                         f"{f(s['evidence_yes']['f1'])} ({s['evidence_yes']['n']})", f"{f(s['evidence_no']['f1'])} ({s['evidence_no']['n']})",
                         f"{f(s['no_labels']['f1'])} ({s['no_labels']['n']})", f(s["hr10"]["hr_percent"]), f(s["evidence_in_context"]["hr_percent"]),
                         f"{f(s['context_tokens_mean'], 0)} / {s['context_tokens_max']}"])
        L += cc.md_table(["Condition", "n", "EM", "F1", "Contain", "F1 evidence yes (n)", "F1 evidence no (n)", "F1 no labels (n)", "HR@10", "Evidence in context", "Context tokens mean / max"], rows)
        L += ["", f"Token-matched rows are cut to {summary.get('token_budget')} tokens (mean SAADI top-10 context).", "", "Paired differences (points; 95% bootstrap interval, 10,000 resamples over queries, seed 42):", ""]
        rows = []
        for key, res in summary["paired"].items():
            a, b = key.split("-", 1)
            cells = [f"{cc.CONDITION_DISPLAY[a]} − {cc.CONDITION_DISPLAY[b]}", res["n"]]
            for m in ("em", "f1", "contain", "hr10", "evidence"):
                r = res.get(m)
                cells.append("—" if not r else f"{r['diff']:+.1f} [{r['ci'][0]:+.1f}, {r['ci'][1]:+.1f}]")
            rows.append(cells)
        L += cc.md_table(["Pair", "n", "EM", "F1", "Contain", "HR@10", "Evidence in context"], rows)
    for w in info.get("warnings", []):
        L += ["", f"Warning: {w}"]
    if info.get("traceback"):
        L += ["", "## Error", "", "```text", *info["traceback"].splitlines()[-60:], "```"]
    L += ["", "## Files written", ""] + [f"- `{x}`" for x in info["files"]] + [f"- `{paths.rel(rep)}`"]
    rep.write_text("\n".join(L) + "\n", encoding="utf-8")
    cc.write_json(rep.with_suffix(".json"), {**info, "summary": summary})
    rep.with_suffix(".summary.txt").write_text(info["one_line"] + "\n", encoding="utf-8")
    log(f"done: {status}")
    return code


if __name__ == "__main__":
    sys.exit(main())
