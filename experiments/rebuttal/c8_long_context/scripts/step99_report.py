"""C8 step 99 (CPU): merge the datasets into results/rebuttal/c8_long_context/{table.csv, paired.csv,
cost.csv, REPORT.md, table_paper.tex} and write reports/step_99_report.{md,json}.

REPORT.md opens with plain sentences written from the numbers (accuracy, tokens and seconds per
query, evidence position, caveats), whichever way they come out.
"""

from __future__ import annotations

import csv
import os
import sys
import time
import traceback
from typing import Any

import c8_common as c8

cc = c8.cc


def log(msg: str) -> None:
    print(f"[c8-99] {cc.now()} {msg}", flush=True)


def write_csv(path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(k for r in rows for k in r)) or ["empty"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def pick(rows: list[dict[str, Any]], **kw: Any) -> dict[str, Any] | None:
    for r in rows:
        if all(r.get(k) == v for k, v in kw.items()):
            return r
    return None


def ci_txt(r: dict[str, Any] | None, m: str = "f1", signed: bool = False) -> str:
    if not r or r.get(m) is None:
        return "—"
    s = "+.1f" if signed else ".1f"
    return f"{r[m]:{s}} [{r[m + '_lo']:{s}}, {r[m + '_hi']:{s}}]"


def num(v: Any, nd: int = 0) -> str:
    return c8.fmt(float(v), nd) if v is not None else "—"


def sentences(key: str, s: dict[str, Any], doc_stats: dict[str, Any]) -> list[str]:
    name = c8.DATASET_DISPLAY[key]
    t, p, c = s["table"], s["paired"], s["cost"]
    fd, sa, va, c5 = pick(t, condition="full_doc"), pick(t, condition="saadi"), pick(t, condition="vanilla"), pick(t, condition="saadi_c5_reader")
    d = pick(p, a="full_doc", b="saadi")
    sv = pick(p, a="saadi", b="vanilla")
    cf, cs = pick(c, condition="full_doc"), pick(c, condition="saadi")
    ds = doc_stats.get(key, {})  # the documents actually read with the full document (see used_documents)
    out = []
    if fd and sa and d:
        rel = "higher than" if d["f1_lo"] > 0 else ("lower than" if d["f1_hi"] < 0 else "not distinguishable from")
        txt = (f"On {name} ({s['common']} queries over {ds.get('documents', '?')} documents of {num(ds.get('min'))} to {num(ds.get('max'))} tokens), "
               f"Qwen2.5-7B-Instruct-1M reading the whole document reaches F1 {ci_txt(fd)}, {rel} the same model reading SAADI's top 10 ({ci_txt(sa)}; "
               f"difference {ci_txt(d, signed=True)} points)")
        if c5:
            txt += f"; SAADI's top 10 read by C5's reader (Qwen3-30B-A3B) gives {ci_txt(c5)}"
        out.append(txt + ".")
        if sv and va:
            out.append(f"With the 7B reader, SAADI minus Vanilla BM25 top 10 is {ci_txt(sv, signed=True)} F1 points (EM {ci_txt(sv, 'em', True)}).")
    if cf and cs:
        ratio = (cf["input_tokens_mean"] / cs["input_tokens_mean"]) if cs["input_tokens_mean"] else None
        txt = (f"The full document costs {num(cf['input_tokens_mean'])} input tokens per query against {num(cs['input_tokens_mean'])} for SAADI's top 10"
               + (f" ({ratio:.0f}x)" if ratio else "") + "; ")
        txt += (f"in wall-clock time it took {num(cf.get('seconds_per_query_with_cache_amortised'), 2)} s per query when the questions on one document share its prefix cache "
                f"and {num(cf.get('seconds_per_query_without_cache'), 1)} s per query without the cache, against {num(cs.get('seconds_per_query_amortised'), 2)} s for SAADI's top 10 "
                f"(single request alone: {num(cf.get('single_request_s_uncached_timing'), 1)} s uncached full document, {num(cs.get('single_request_s_timing'), 2)} s SAADI; 4 RTX A6000).")
        out.append(txt)
    pos = [r for r in s["position"] if r["subset"] == "all" and r["third"] in c8.THIRDS]
    pf = {r["third"]: r for r in pos if r["condition"] == "full_doc"}
    ps = {r["third"]: r for r in pos if r["condition"] == "saadi"}
    if len(pf) == 3 and all(pf[x]["f1"] is not None for x in c8.THIRDS):
        spread = max(pf[x]["f1"] for x in c8.THIRDS) - min(pf[x]["f1"] for x in c8.THIRDS)
        out.append(f"By the position of the gold evidence, full-document F1 is {pf['first']['f1']:.1f} / {pf['middle']['f1']:.1f} / {pf['last']['f1']:.1f} in the first / middle / last third "
                   f"(n = {pf['first']['n']} / {pf['middle']['n']} / {pf['last']['n']}; range {spread:.1f} points), and SAADI's {ps['first']['f1']:.1f} / {ps['middle']['f1']:.1f} / {ps['last']['f1']:.1f}.")
    return out


def used_documents(paths: c8.C8Paths, s00: dict[str, Any], key: str) -> dict[str, Any]:
    """Length range of the documents read in the full-document condition, and of those skipped
    (from step 00's per-document lengths and step 02's skipped documents)."""

    rows = [r for r in (s00.get("doc_rows") or []) if r["dataset"] == key and r.get("queries")]
    p2 = paths.reports / f"step_02_{key}_reader.json"
    r2 = cc.read_json(p2) if p2.exists() else {}
    skipped = set(((r2.get("conditions") or {}).get("full_doc") or {}).get("skipped_docs") or [])
    used = [r for r in rows if r["doc_id"] not in skipped]
    sk = [r for r in rows if r["doc_id"] in skipped]
    out: dict[str, Any] = {"documents": len(used), "min": min((r["tokens"] for r in used), default=None), "max": max((r["tokens"] for r in used), default=None),
                           "skipped_docs": sorted(skipped), "skipped_queries": sum(r["queries"] for r in sk)}
    if sk:
        out.update({"skipped_min": min(r["tokens"] for r in sk), "skipped_max": max(r["tokens"] for r in sk)})
    return out


def caveats(sums: dict[str, dict[str, Any]], s01: dict[str, Any], used: dict[str, dict[str, Any]]) -> list[str]:
    out = []
    for key, s in sums.items():
        name = c8.DATASET_DISPLAY[key]
        sk = (s.get("skipped") or {}).get("full_doc")
        u = used.get(key, {})
        if sk:
            why = ("the server's vLLM has no dual chunk attention, so the window is the model's native 262,144 tokens"
                   if s.get("mode") == "native" else "they do not fit the served window")
            out.append(f"{name}: {why}; the {len(u.get('skipped_docs', []))} longer books ({num(u.get('skipped_min'))} to {num(u.get('skipped_max'))} tokens, {sk} queries) "
                       f"were not read in full and their queries are left out of every condition, so the full-document comparison covers the {u.get('documents')} "
                       f"shorter books ({s['common']} queries).")
        if s.get("not_started_deadline"):
            out.append(f"{name}: {s['not_started_deadline']} full-document queries were not started before the deadline; the comparison uses the {s['common']} queries answered in all conditions.")
        if s.get("subset"):
            out.append(f"{name}: a subset of {s['subset']} queries (first of a seed-42 permutation), the same for all conditions.")
        if not pick(s["table"], condition="saadi_c5_reader"):
            out.append(f"{name}: C5's reader answers were not available, so the C5-reader row is missing.")
    if "novelhopqa" in sums:
        out.append("NovelHopQA's evidence positions use C5's paragraph labels (the paper's window labels accept one-word matches); answer scores do not depend on labels.")
    for k, v in (s01.get("per_dataset") or {}).items():
        if v.get("prefix_cache_effective") is False:
            out.append(f"{c8.DATASET_DISPLAY[k]}: the timing run found the prefix cache ineffective (cached/uncached {v.get('cached_over_uncached_median')}).")
    return out


def main() -> int:
    paths = c8.C8Paths.from_env()
    info: dict[str, Any] = {"step": "99", "commit": cc.git_commit(paths.root), "host": os.uname().nodename, "files": []}
    status, code, t0 = "OK", 0, time.time()
    try:
        sums = {}
        for key in c8.DATASETS:
            p = paths.results / "per_dataset" / key / "summary.json"
            if p.exists():
                sums[key] = cc.read_json(p)
        if not sums:
            raise RuntimeError("no per-dataset scores (step 03) found")
        s00 = cc.read_json(paths.reports / "step_00.json") if (paths.reports / "step_00.json").exists() else {}
        s01 = cc.read_json(paths.reports / "step_01_timing.json") if (paths.reports / "step_01_timing.json").exists() else {}
        used = {k: used_documents(paths, s00, k) for k in sums}
        doc_stats = used
        table = [r for s in sums.values() for r in s["table"]]
        paired = [r for s in sums.values() for r in s["paired"]]
        cost = [r for s in sums.values() for r in s["cost"]]
        position = [r for s in sums.values() for r in s["position"]]
        res = paths.results
        for name, rows in (("table", table), ("paired", paired), ("cost", cost), ("position", position)):
            write_csv(res / f"{name}.csv", rows)
            info["files"].append(paths.rel(res / f"{name}.csv"))
        opening = [x for k, s in sums.items() for x in sentences(k, s, doc_stats)]
        cav = caveats(sums, s01, used)
        if cav:
            opening.append("Caveats: " + " ".join(cav))
        # Side-by-side table.
        side = []
        for k, s in sums.items():
            for cond, label in (("full_doc", "Full document, 7B"), ("saadi", "SAADI top 10, 7B"), ("saadi_c5_reader", "SAADI top 10, C5 reader (30B-A3B)"),
                                ("vanilla", "Vanilla top 10, 7B"), ("closed_book", "Closed-book, 7B")):
                r, c = pick(s["table"], condition=cond), pick(s["cost"], condition=cond)
                if not r:
                    continue
                sec = (c or {}).get("seconds_per_query_with_cache_amortised") if cond == "full_doc" else (c or {}).get("seconds_per_query_amortised")
                side.append([c8.DATASET_DISPLAY[k], label, r["n"], ci_txt(r, "em"), ci_txt(r, "f1"), ci_txt(r, "contain"), num((c or {}).get("input_tokens_mean")),
                             num(sec, 2) if cond != "saadi_c5_reader" else "—", num((c or {}).get("seconds_per_query_without_cache"), 1) if cond == "full_doc" else "—"])
        L = ["# C8: long-context reader vs SAADI top 10 (reviewer iVU8, Q1)", "", *opening, "",
             f"Written {cc.now()}; commit {info['commit']}. Reader Qwen/Qwen2.5-7B-Instruct-1M (vLLM, 4 RTX A6000, temperature 0, seed 42, C5's prompt and scoring); "
             "the full document is SAADI's chunks in order, given first, the question last.", "",
             "## Accuracy and cost per query", "",
             *cc.md_table(["Dataset", "Condition", "n", "EM", "F1", "Containment", "Input tokens / query", "Wall s / query", "Full doc without cache, s / query"], side), "",
             "Intervals: 95% bootstrap over queries (10,000 resamples, seed 42). Wall s / query: amortised over the run (full document: questions on one document share "
             "its prefix cache); 'without cache' = the document's uncached read, paid by every question.", "",
             "## Paired differences (points, same queries)", ""]
        L += cc.md_table(["Dataset", "Pair", "n", "EM", "F1", "Containment"],
                         [[c8.DATASET_DISPLAY[r["dataset"]], r["pair"], r["n"]] + [ci_txt(r, m, True) for m in c8.MEASURES] for r in paired]) + [""]
        L += ["## F1 by the position of the gold evidence", ""]
        L += cc.md_table(["Dataset", "Condition", "Subset", "First third", "Middle third", "Last third"],
                         [[c8.DATASET_DISPLAY[k], c8.CONDITION_DISPLAY[c], sub] + [
                             (lambda r: f"{r['f1']:.1f} (n={r['n']})" if r and r.get("f1") is not None else "—")(pick(position, dataset=k, condition=c, subset=sub, third=th)) for th in c8.THIRDS]
                          for k in sums for c in ("full_doc", "saadi", "vanilla") for sub in ("all", "evidence_in_one_third")]) + [""]
        L += ["## Cost detail", ""]
        L += cc.md_table(["Dataset", "Condition", "Input tokens", "Output tokens", "Request latency in the run (s)", "Single request alone (s)", "Amortised wall s", "Without cache s"],
                         [[c8.DATASET_DISPLAY[r["dataset"]], r["condition_display"], num(r["input_tokens_mean"]), num(r.get("output_tokens_mean"), 1), num(r.get("request_latency_s_mean"), 2),
                           (f"{num(r.get('single_request_s_uncached_timing'), 1)} / {num(r.get('single_request_s_cached_timing'), 2)}" if r["condition"] == "full_doc" else num(r.get("single_request_s_timing"), 2)),
                           num(r.get("seconds_per_query_with_cache_amortised") if r["condition"] == "full_doc" else r.get("seconds_per_query_amortised"), 2),
                           num(r.get("seconds_per_query_without_cache"), 1)] for r in cost]) + [""]
        all_docs = s00.get("doc_stats") or {}
        if all_docs:
            L += ["## Documents", ""]
            L += cc.md_table(["Dataset", "Distinct documents", "Tokens min", "median", "max", "Read in full (documents, tokens)", "Not read in full"],
                             [[c8.DATASET_DISPLAY[k], v["documents"], num(v["min"]), num(v["median"]), num(v["max"]),
                               f"{used.get(k, {}).get('documents')} ({num(used.get(k, {}).get('min'))} to {num(used.get(k, {}).get('max'))})",
                               (", ".join(used[k]["skipped_docs"]) + f" ({used[k]['skipped_queries']} queries)") if used.get(k, {}).get("skipped_docs") else "—"]
                              for k, v in all_docs.items()]) + [""]
        L += ["## Files", "", "- `results/rebuttal/c8_long_context/{table,paired,cost,position}.csv`, `answers/<dataset>/<condition>.jsonl`, `table_paper.tex`",
              "- step reports: `experiments/rebuttal/c8_long_context/reports/`", ""]
        (res / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
        info["files"].append(paths.rel(res / "REPORT.md"))
        # LaTeX.
        tex = ["% C8 long-context reader baseline (generated by step99_report.py)", "\\begin{tabular}{llrrrr}", "\\toprule",
               "Dataset & Reader input & EM & F1 & Input tokens & s / query \\\\", "\\midrule"]
        for row in side:
            tex.append(" & ".join(str(x).replace("_", "\\_").replace("%", "\\%") for x in (row[0], row[1], row[3].split(" ")[0], row[4].split(" ")[0], row[6], row[7])) + " \\\\")
        tex += ["\\bottomrule", "\\end{tabular}"]
        (res / "table_paper.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
        info["files"].append(paths.rel(res / "table_paper.tex"))
        info["opening"] = opening
        info["one_line"] = " | ".join(f"{c8.DATASET_DISPLAY[k]} F1 full {pick(s['table'], condition='full_doc')['f1']:.1f} vs SAADI {pick(s['table'], condition='saadi')['f1']:.1f}" for k, s in sums.items())
    except Exception as exc:
        status, code = "FAILED", 1
        info["traceback"] = traceback.format_exc()
        info["one_line"] = f"FAILED: {type(exc).__name__}: {str(exc)[:110]}"
    info["seconds"] = round(time.time() - t0)
    rep = paths.reports / "step_99_report.md"
    L = [f"# C8 long-context reader - step 99 final report: {status}", "", f"Written {cc.now()}; commit {info['commit']}; {info['seconds']} s.", "",
         "Command:", "", "```bash", "bash experiments/rebuttal/c8_long_context/sh/99_report.sh", "```", ""]
    if info.get("opening"):
        L += ["## Opening of REPORT.md", ""] + info["opening"] + [""]
    if info.get("traceback"):
        L += ["## Error", "", "```text", *info["traceback"].splitlines()[-60:], "```", ""]
    L += ["## Files written", ""] + [f"- `{x}`" for x in info["files"]] + [f"- `{paths.rel(rep)}`"]
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text("\n".join(L) + "\n", encoding="utf-8")
    cc.write_json(rep.with_suffix(".json"), info)
    rep.with_suffix(".summary.txt").write_text(info.get("one_line", status) + "\n", encoding="utf-8")
    log(f"done: {status}")
    return code


if __name__ == "__main__":
    sys.exit(main())
