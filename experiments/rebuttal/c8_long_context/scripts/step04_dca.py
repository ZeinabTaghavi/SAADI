"""C8 step 04: NovelHopQA with every book read whole, through Qwen's dual chunk attention.

The main C8 run served the 7B reader without dual chunk attention (vLLM 0.23 has no such backend),
so its window was the native 262,144 tokens and 6 of the 16 books (265K-760K tokens) were skipped.
This step reads all 16 books with the model card's setup: vLLM 0.10.1.1, V0 engine,
VLLM_ATTENTION_BACKEND=DUAL_CHUNK_FLASH_ATTN, chunked prefill of 131,072 tokens, eager mode, same
prompt, decoding and scoring as before. Only the full-document condition is asked again: SAADI,
Vanilla and closed-book prompts are at most ~7.5K tokens, where dual chunk attention is the same as
standard attention, so their answers from the main run are reused unchanged.

  plan  (CPU)  splits the books into C8_DCA_SHARDS work units (default 2, one per 4-GPU server),
               balanced by an estimated cost; inside a unit the books beyond 262,144 tokens come first
               (shortest first), then the others. The plan is written once and reused on a restart.
  score (CPU)  merges the units' answers and scores them against the main run's SAADI / Vanilla /
               closed-book answers and C5's 30B reader, on all books, on the 6 long books and on the
               10 books within the native window (there also against the native full-document answers),
               and on every book read whole in the mode its length needs ("all_mixed": the main run's
               native read inside 262,144 tokens, dual chunk attention beyond).

Answers: results/rebuttal/c8_long_context/dca/shard_<k>/answers/novelhopqa/full_doc.jsonl (step 02
writes them; the main run's answer files are never touched). Results: results/rebuttal/
c8_long_context/dca/{REPORT.md,table.csv,paired.csv,per_book.csv,cost.csv}.
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
from pathlib import Path
from statistics import mean
from typing import Any

import c8_common as c8
import c8_reader as rd

cc = c8.cc
DS = "novelhopqa"
NATIVE = 262144
COND_DISPLAY = {
    "full_dca": "Full book, dual chunk attention (7B)",
    "full_native": "Full book, native window (7B, main run)",
    "full_mixed": "Full book: native read up to 262K, dual chunk attention beyond (7B)",
    "saadi": "SAADI top 10 (7B)",
    "vanilla": "Vanilla BM25 top 10 (7B)",
    "closed_book": "Closed-book (7B)",
    "saadi_c5": "SAADI top 10, C5 reader (Qwen3-30B-A3B)",
}
PAIRS = [("full_dca", "saadi"), ("full_dca", "saadi_c5"), ("saadi", "vanilla"), ("full_dca", "vanilla"), ("full_dca", "closed_book")]
PAIRS_WITHIN = [("full_dca", "full_native"), ("full_native", "saadi")]
PAIRS_MIXED = [("full_mixed", "saadi"), ("full_mixed", "saadi_c5"), ("saadi", "vanilla"), ("full_mixed", "closed_book")]


def log(msg: str) -> None:
    print(f"[c8-04] {cc.now()} {msg}", flush=True)


def dca_dir(paths: c8.C8Paths) -> Path:
    return paths.results / "dca"


def plan_file(paths: c8.C8Paths) -> Path:
    return dca_dir(paths) / "plan.json"


# ---------------------------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------------------------


def est_seconds(tokens: int, n_q: int) -> float:
    """Rough relative cost on one 4-GPU server: one uncached read (fit to the main run's native
    reads, 133K -> 38 s, 253K -> 88 s) plus the cached questions (8 in flight)."""

    k = tokens / 1000.0
    return 0.21 * k + 5.5e-4 * k * k + n_q * 0.0037 * k


def do_plan(paths: c8.C8Paths, n_shards: int) -> int:
    pf = plan_file(paths)
    if pf.exists() and os.environ.get("C8_DCA_REPLAN", "0") != "1":
        plan = cc.read_json(pf)
        if len(plan["shards"]) == n_shards:
            log(f"plan kept: {pf}")
            for s in plan["shards"]:
                print(f"{s['shard']}\t{','.join(s['docs'])}")
            return 0
    inp = rd.Inputs(paths, DS)
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q in inp.selected():
        by_doc[q["doc_id"]].append(q)
    docs = []
    for d, qs in by_doc.items():
        tok = qs[0]["doc_tokens"]
        docs.append({"doc_id": d, "tokens": tok, "queries": len(qs), "over_native": any(q.get("over_native") for q in qs),
                     "prompt_tokens_max": max(q["prompt_tokens"]["full_doc"] for q in qs), "est_s": est_seconds(tok, len(qs))})
    shards = [{"shard": k, "docs": [], "est_s": 0.0, "queries": 0} for k in range(n_shards)]
    for d in sorted(docs, key=lambda x: -x["est_s"]):
        s = min(shards, key=lambda x: x["est_s"])
        s["docs"].append(d["doc_id"])
        s["est_s"] += d["est_s"]
        s["queries"] += d["queries"]
    info = {d["doc_id"]: d for d in docs}
    for s in shards:
        s["docs"].sort(key=lambda x: (not info[x]["over_native"], info[x]["tokens"]))
        s["est_s"] = round(s["est_s"])
    plan = {"created": cc.now(), "dataset": DS, "shards": shards, "docs": sorted(docs, key=lambda x: -x["tokens"])}
    pf.parent.mkdir(parents=True, exist_ok=True)
    cc.write_json(pf, plan)
    log(f"plan written: {pf}")
    for s in shards:
        print(f"{s['shard']}\t{','.join(s['docs'])}")
    return 0


# ---------------------------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------------------------


def ci(vals: list[float], n_boot: int) -> dict[str, Any]:
    if not vals:
        return {"n": 0, "mean": None, "lo": None, "hi": None}
    m, lo, hi = c8.paired_bootstrap_ci([100.0 * v for v in vals], n_resamples=n_boot, seed=c8.SEED)
    return {"n": len(vals), "mean": m, "lo": lo, "hi": hi}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(k for r in rows for k in r)) or ["empty"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def scored(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    out = {}
    for r in rows:
        if r.get("key") != key:
            continue
        r.update(cc.score_answer(r["output"], r["gold"]))
        out[str(r["query_id"])] = r
    return out


def do_score(paths: c8.C8Paths) -> int:
    n_boot = int(os.environ.get("C8_BOOTSTRAP_RESAMPLES", 10000))
    rep = paths.reports / "step_04_dca_score.md"
    info: dict[str, Any] = {"step": "04", "dataset": DS, "commit": cc.git_commit(paths.root), "host": os.uname().nodename, "files": []}
    status, code, t0 = "OK", 0, time.time()
    out = dca_dir(paths)
    for name in ("table.csv", "paired.csv", "per_book.csv", "cost.csv", "summary.json", "REPORT.md"):
        (out / name).unlink(missing_ok=True)
    try:
        inp = rd.Inputs(paths, DS)
        sel = {q["query_id"]: q for q in inp.selected()}
        # Main run (native mode): SAADI, Vanilla, closed-book, and full document on the 10 books.
        ans: dict[str, dict[str, dict[str, Any]]] = {}
        for c, src in (("saadi", "saadi"), ("vanilla", "vanilla"), ("closed_book", "closed_book"), ("full_native", "full_doc")):
            p = paths.answers_file(DS, src)
            ans[c] = scored(cc.read_jsonl(p), inp.key(src, "native")) if p.exists() else {}
        # This step: full document with dual chunk attention, from every work unit.
        dca_rows, doc_rows, servers = [], {}, []
        for sd in sorted(out.glob("shard_*")):
            p = sd / "answers" / DS / "full_doc.jsonl"
            if p.exists():
                dca_rows += cc.read_jsonl(p)
            dp = p.with_name("full_doc_documents.jsonl")
            if dp.exists():
                for r in cc.read_jsonl(dp):
                    if r.get("key") == inp.key("full_doc", "dca"):
                        doc_rows[r["doc_id"]] = r
            sp = paths.reports / "dca" / sd.name / "server_check.txt"
            if sp.exists():
                servers.append({"shard": sd.name, "check": sp.read_text(encoding="utf-8").strip()})
        ans["full_dca"] = scored(dca_rows, inp.key("full_doc", "dca"))
        # Every book read whole, each in the mode its length needs: the main run's native read for the
        # books inside 262,144 tokens, dual chunk attention for the books beyond.
        ans["full_mixed"] = {}
        for q in sel:
            src = ans["full_dca"] if sel[q].get("over_native") else ans["full_native"]
            if q in src:
                ans["full_mixed"][q] = src[q]
        c5p = paths.c5_answers_file(DS, "saadi")
        ans["saadi_c5"] = {str(r["query_id"]): {**r, **cc.score_answer(r["output"], r["gold"])} for r in cc.read_jsonl(c5p)} if c5p.exists() else {}
        info["answered"] = {c: len(set(v) & set(sel)) for c, v in ans.items()}
        info["servers"] = servers
        if not ans["full_dca"]:
            raise RuntimeError("no full-document answers with dual chunk attention yet (dca/shard_*/answers/novelhopqa/full_doc.jsonl)")
        if not ans["saadi"]:
            raise RuntimeError(f"the main run's SAADI answers are missing ({paths.rel(paths.answers_file(DS, 'saadi'))})")
        base = sorted(set(ans["full_dca"]) & set(ans["saadi"]) & set(ans["vanilla"]) & set(ans["closed_book"]) & set(sel), key=lambda q: sel[q]["subset_rank"])
        missing = sorted(set(sel) - set(ans["full_dca"]))
        info["selected"], info["common"] = len(sel), len(base)
        info["missing_by_doc"] = {d: sum(1 for q in missing if sel[q]["doc_id"] == d) for d in sorted({sel[q]["doc_id"] for q in missing})}
        base_mixed = sorted(set(ans["full_mixed"]) & set(ans["saadi"]) & set(ans["vanilla"]) & set(ans["closed_book"]) & set(sel), key=lambda q: sel[q]["subset_rank"])
        info["common_mixed"] = len(base_mixed)
        groups = {
            "all_mixed": base_mixed,
            "all": base,
            "beyond_native": [q for q in base if sel[q].get("over_native")],
            "within_native": [q for q in base if not sel[q].get("over_native")],
        }
        info["groups"] = {g: {"n": len(qs), "books": len({sel[q]["doc_id"] for q in qs}),
                              "tokens_min": min((sel[q]["doc_tokens"] for q in qs), default=None),
                              "tokens_max": max((sel[q]["doc_tokens"] for q in qs), default=None)} for g, qs in groups.items()}
        table, paired = [], []
        for g, qs in groups.items():
            if not qs:
                continue
            full = ["full_mixed"] if g == "all_mixed" else ["full_dca"] + (["full_native"] if g == "within_native" else [])
            conds = full + ["saadi", "vanilla", "closed_book"] + (["saadi_c5"] if ans["saadi_c5"] else [])
            for c in conds:
                qq = [q for q in qs if q in ans[c]]
                row = {"group": g, "condition": c, "condition_display": COND_DISPLAY[c], "n": len(qq)}
                for m in c8.MEASURES:
                    s = ci([ans[c][q][m] for q in qq], n_boot)
                    row.update({m: s["mean"], f"{m}_lo": s["lo"], f"{m}_hi": s["hi"]})
                table.append(row)
            for a, b in (PAIRS_MIXED if g == "all_mixed" else PAIRS + (PAIRS_WITHIN if g == "within_native" else [])):
                qq = [q for q in qs if q in ans[a] and q in ans[b]]
                row = {"group": g, "a": a, "b": b, "pair": f"{COND_DISPLAY[a]} - {COND_DISPLAY[b]}", "n": len(qq)}
                for m in c8.MEASURES:
                    d, lo, hi = c8.paired_bootstrap_ci([100.0 * (ans[a][q][m] - ans[b][q][m]) for q in qq], n_resamples=n_boot, seed=c8.SEED) if qq else (None, None, None)
                    row.update({m: d, f"{m}_lo": lo, f"{m}_hi": hi})
                paired.append(row)
        # Per book.
        per_book = []
        for d in sorted({sel[q]["doc_id"] for q in sel}, key=lambda x: next(sel[q]["doc_tokens"] for q in sel if sel[q]["doc_id"] == x)):
            qs_all = [q for q in sel if sel[q]["doc_id"] == d]
            qs = [q for q in base if sel[q]["doc_id"] == d]
            dr = doc_rows.get(d, {})
            cached = [ans["full_dca"][q] for q in qs if ans["full_dca"][q].get("cache_state") == "cached_document"]
            hit = [r for r in cached if (r.get("cached_tokens") or 0) >= 0.9 * (r.get("server_prompt_tokens") or r.get("prompt_tokens") or 1)]
            row = {"doc_id": d, "tokens": sel[qs_all[0]]["doc_tokens"], "beyond_native": bool(sel[qs_all[0]].get("over_native")), "queries": len(qs_all), "answered": len(qs),
                   "f1_full_dca": mean(100 * ans["full_dca"][q]["f1"] for q in qs) if qs else None,
                   "f1_saadi": mean(100 * ans["saadi"][q]["f1"] for q in qs) if qs else None,
                   "f1_full_native": mean(100 * ans["full_native"][q]["f1"] for q in qs if q in ans["full_native"]) if any(q in ans["full_native"] for q in qs) else None,
                   "uncached_s": dr.get("uncached_seconds"), "cached_mean_latency_s": dr.get("other_mean_latency_s"), "document_wall_s": dr.get("document_wall_s"),
                   "cached_questions": len(cached), "cached_questions_with_cache_hit": len(hit)}
            row["f1_full_dca_minus_saadi"] = (row["f1_full_dca"] - row["f1_saadi"]) if qs else None
            per_book.append(row)
        info["cache_hits"] = {"cached_questions": sum(r["cached_questions"] for r in per_book), "with_hit": sum(r["cached_questions_with_cache_hit"] for r in per_book)}
        # Cost per group.
        r02p = paths.reports / f"step_02_{DS}_reader.json"
        r02 = cc.read_json(r02p) if r02p.exists() else {}
        sa_c = (r02.get("conditions") or {}).get("saadi") or {}
        saadi_amort = (sa_c["wall_seconds"] / sa_c["asked"]) if sa_c.get("asked") and sa_c.get("wall_seconds") else None
        cost = []
        for g, qs in groups.items():
            if not qs or g == "all_mixed":
                continue
            fd = [ans["full_dca"][q] for q in qs]
            used = [doc_rows[d] for d in {sel[q]["doc_id"] for q in qs} if d in doc_rows]
            n_used = sum(1 + u["other_answered"] for u in used)
            cost.append({"group": g, "n": len(qs),
                         "full_input_tokens_mean": mean(r.get("server_prompt_tokens") or r["prompt_tokens"] for r in fd),
                         "saadi_input_tokens_mean": mean(ans["saadi"][q].get("server_prompt_tokens") or ans["saadi"][q]["prompt_tokens"] for q in qs),
                         "full_seconds_per_query_without_cache": mean(doc_rows[r["doc_id"]]["uncached_seconds"] for r in fd if r["doc_id"] in doc_rows) if doc_rows else None,
                         "full_seconds_per_query_with_cache_amortised": (sum(u["document_wall_s"] for u in used) / n_used) if n_used else None,
                         "saadi_seconds_per_query_amortised_main_run": saadi_amort})
        for name, rows in (("table", table), ("paired", paired), ("per_book", per_book), ("cost", cost)):
            write_csv(out / f"{name}.csv", rows)
            info["files"].append(paths.rel(out / f"{name}.csv"))
        info.update({"table": table, "paired": paired, "per_book": per_book, "cost": cost})
        cc.write_json(out / "summary.json", {k: info[k] for k in ("answered", "selected", "common", "common_mixed", "missing_by_doc", "groups", "table", "paired", "per_book", "cost", "cache_hits", "servers")})
        info["files"].append(paths.rel(out / "summary.json"))
        g_all = {r["condition"]: r for r in table if r["group"] == "all"}
        p_all = next(r for r in paired if r["group"] == "all" and (r["a"], r["b"]) == ("full_dca", "saadi"))
        info["one_line"] = (f"all books read whole: F1 full book {g_all['full_dca']['f1']:.1f}, SAADI {g_all['saadi']['f1']:.1f} (7B, n={len(base)} of {len(sel)}); "
                            f"full-SAADI {p_all['f1']:+.1f} [{p_all['f1_lo']:+.1f}, {p_all['f1_hi']:+.1f}]")
        p_mix = next((r for r in paired if r["group"] == "all_mixed" and (r["a"], r["b"]) == ("full_mixed", "saadi")), None)
        if p_mix and p_mix.get("f1") is not None:
            info["one_line"] += f"; mixed full read - SAADI {p_mix['f1']:+.1f} [{p_mix['f1_lo']:+.1f}, {p_mix['f1_hi']:+.1f}] (n={p_mix['n']})"
        if missing:
            status = "PARTIAL"
            info["one_line"] += f"; {len(missing)} questions not answered yet"
    except Exception as exc:
        status, code = "FAILED", 1
        info["traceback"] = traceback.format_exc()
        info["one_line"] = f"FAILED: {type(exc).__name__}: {str(exc)[:110]}"
    info["seconds"] = round(time.time() - t0)
    info["status"] = status
    text = render(paths, info, status)
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(text, encoding="utf-8")
    cc.write_json(rep.with_suffix(".json"), info)
    rep.with_suffix(".summary.txt").write_text(info.get("one_line", status) + "\n", encoding="utf-8")
    if status != "FAILED":
        (out / "REPORT.md").write_text(text, encoding="utf-8")
    log(f"done: {status}")
    return code


def pm(r: dict[str, Any], m: str = "f1") -> str:
    return f"{r[m]:+.1f} [{r[m + '_lo']:+.1f}, {r[m + '_hi']:+.1f}]" if r.get(m) is not None else "—"


def val(r: dict[str, Any], m: str = "f1") -> str:
    return f"{r[m]:.1f} [{r[m + '_lo']:.1f}, {r[m + '_hi']:.1f}]" if r.get(m) is not None else "—"


def render(paths: c8.C8Paths, info: dict[str, Any], status: str) -> str:
    f = c8.fmt
    L = [f"# C8 step 04 - NovelHopQA, every book read whole with dual chunk attention: {status}", "",
         f"Written {cc.now()}; commit {info['commit']}; host {info['host']}; {info['seconds']} s.", ""]
    if info.get("table"):
        T = {(r["group"], r["condition"]): r for r in info["table"]}
        P = {(r["group"], r["a"], r["b"]): r for r in info["paired"]}
        G = info["groups"]
        cost = {r["group"]: r for r in info["cost"]}
        s = []
        ga = G["all"]
        if ("all_mixed", "full_mixed") in T and ("all_mixed", "full_mixed", "saadi") in P:
            gm = G["all_mixed"]
            s.append(f"Every NovelHopQA book read whole, each in the mode its length needs (the main run's native read for the books inside 262,144 tokens, dual chunk attention beyond; "
                     f"{gm['books']} books, {gm['n']} questions): F1 {val(T[('all_mixed', 'full_mixed')])} against {val(T[('all_mixed', 'saadi')])} for SAADI's top 10 with the same 7B reader; "
                     f"paired difference {pm(P[('all_mixed', 'full_mixed', 'saadi')])} (EM {pm(P[('all_mixed', 'full_mixed', 'saadi')], 'em')}, containment {pm(P[('all_mixed', 'full_mixed', 'saadi')], 'contain')}).")
        s.append(f"The 7B 1M-context reader (Qwen2.5-7B-Instruct-1M) read every NovelHopQA book whole with Qwen's dual chunk attention (vLLM 0.10.1.1, V0 engine, as on the model card): "
                 f"{ga['books']} books of {f(ga['tokens_min'])}-{f(ga['tokens_max'])} tokens, {ga['n']} of {info['selected']} questions answered.")
        s.append(f"Reading the whole book gives F1 {val(T[('all', 'full_dca')])}; the same reader with SAADI's top 10 gives {val(T[('all', 'saadi')])}; "
                 f"paired difference full book - SAADI {pm(P[('all', 'full_dca', 'saadi')])} (EM {pm(P[('all', 'full_dca', 'saadi')], 'em')}, containment {pm(P[('all', 'full_dca', 'saadi')], 'contain')}).")
        if ("beyond_native", "full_dca") in T:
            gb = G["beyond_native"]
            s.append(f"On the {gb['books']} books beyond the native 262,144-token window ({f(gb['tokens_min'])}-{f(gb['tokens_max'])} tokens, {gb['n']} questions), the ones the main run had to skip: "
                     f"full book {val(T[('beyond_native', 'full_dca')])}, SAADI {val(T[('beyond_native', 'saadi')])}, difference {pm(P[('beyond_native', 'full_dca', 'saadi')])}.")
        if ("within_native", "full_dca") in T and ("within_native", "full_dca", "full_native") in P:
            s.append(f"On the {G['within_native']['books']} books inside the native window, dual chunk attention gives F1 {T[('within_native', 'full_dca')]['f1']:.1f} against {T[('within_native', 'full_native')]['f1']:.1f} "
                     f"for the main run's native read of the same books (difference {pm(P[('within_native', 'full_dca', 'full_native')])}).")
        c = cost.get("all") or {}
        if c:
            s.append(f"Input per question: {f(c['full_input_tokens_mean'], 0)} tokens for the whole book against {f(c['saadi_input_tokens_mean'], 0)} for SAADI "
                     f"({c['full_input_tokens_mean'] / c['saadi_input_tokens_mean']:.0f}x); a fresh read of a book takes {f(c['full_seconds_per_query_without_cache'], 1)} s per question on average, "
                     f"{f(c['full_seconds_per_query_with_cache_amortised'], 2)} s with the book kept in the prefix cache, against {f(c['saadi_seconds_per_query_amortised_main_run'], 2)} s for SAADI.")
        if info.get("missing_by_doc"):
            s.append(f"Not answered (deadline or errors): {sum(info['missing_by_doc'].values())} questions on {', '.join(f'{d} ({n})' for d, n in info['missing_by_doc'].items())}.")
        L += [" ".join(s), ""]
        L += ["Accuracy (percent, 95% bootstrap interval over questions, 10,000 resamples, seed 42). SAADI, Vanilla and closed-book are the main run's answers (their prompts are below 8K tokens, where dual chunk attention equals standard attention).", ""]
        L += cc.md_table(["Books", "Condition", "n", "EM", "F1", "Containment"],
                         [[r["group"], r["condition_display"], r["n"], val(r, "em"), val(r, "f1"), val(r, "contain")] for r in info["table"]]) + [""]
        L += ["Paired differences (points, same questions):", ""]
        L += cc.md_table(["Books", "Pair", "n", "EM", "F1", "Containment"],
                         [[r["group"], r["pair"], r["n"], pm(r, "em"), pm(r, "f1"), pm(r, "contain")] for r in info["paired"]]) + [""]
        L += ["Cost per question:", ""]
        L += cc.md_table(["Books", "n", "Input tokens, full book", "Input tokens, SAADI", "Full book without cache (s)", "Full book with cache, amortised (s)", "SAADI, amortised (s, main run)"],
                         [[r["group"], r["n"], f(r["full_input_tokens_mean"], 0), f(r["saadi_input_tokens_mean"], 0), f(r["full_seconds_per_query_without_cache"], 1),
                           f(r["full_seconds_per_query_with_cache_amortised"], 2), f(r["saadi_seconds_per_query_amortised_main_run"], 2)] for r in info["cost"]]) + [""]
        L += ["Per book (shortest first):", ""]
        L += cc.md_table(["Book", "Tokens", "Beyond 262K", "Questions", "Answered", "F1 full (DCA)", "F1 SAADI", "Full - SAADI", "F1 full (native, main run)", "Uncached read (s)", "Cached question latency (s)", "Cache hits"],
                         [[r["doc_id"], f(r["tokens"]), r["beyond_native"], r["queries"], r["answered"], f(r["f1_full_dca"]), f(r["f1_saadi"]),
                           (f"{r['f1_full_dca_minus_saadi']:+.1f}" if r["f1_full_dca_minus_saadi"] is not None else "—"), f(r["f1_full_native"]),
                           f(r["uncached_s"], 1), f(r["cached_mean_latency_s"], 2), f"{r['cached_questions_with_cache_hit']}/{r['cached_questions']}"] for r in info["per_book"]]) + [""]
        ch = info.get("cache_hits") or {}
        L += [f"Prefix cache: {ch.get('with_hit')} of {ch.get('cached_questions')} questions asked after their book's first question found at least 90% of their prompt cached.", ""]
        if info.get("servers"):
            L += ["Servers (attention backend lines from the vLLM logs):", ""] + [f"- {x['shard']}: {x['check']}" for x in info["servers"]] + [""]
    if info.get("traceback"):
        L += ["## Error", "", "```text", *info["traceback"].splitlines()[-60:], "```", ""]
    L += ["## Files written", ""] + [f"- `{x}`" for x in info["files"]] + [f"- `{paths.rel(paths.reports / 'step_04_dca_score.md')}`"]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("plan", "score"))
    ap.add_argument("--shards", type=int, default=int(os.environ.get("C8_DCA_SHARDS", 2)))
    args = ap.parse_args(argv)
    paths = c8.C8Paths.from_env()
    if args.action == "plan":
        return do_plan(paths, args.shards)
    return do_score(paths)


if __name__ == "__main__":
    sys.exit(main())
