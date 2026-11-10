"""C8 step 01 (GPU, server already up): time C8_TIMING_QUERIES queries per dataset, estimate the run.

The sample is the first C8_TIMING_QUERIES queries of step 00's seed-42 permutation. Every request is
sent alone (one at a time), so the seconds are single-request latencies:
  - full document: per sampled document, the prefix cache is emptied, its first sampled question is
    asked (uncached: the whole document is read), the same question again (cached document), then
    the document's other sampled questions (cached);
  - SAADI top 10, Vanilla top 10, closed-book: each sampled question once.
From these: per-document uncached seconds as a function of length (a*L + b*L^2, least squares),
cached seconds per question (c + d*L), and the estimated GPU hours of the full run per dataset, with
and without the prefix cache. The estimate is an upper bound (the full run sends a document's cached
questions and the short conditions several at a time). Policy over C8_MAX_GPU_HOURS:
C8_ON_OVER_BUDGET=run (default: run anyway, the estimate is reported) or stop (skip that dataset).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import c8_common as c8
import c8_reader as rd

cc = c8.cc


def log(msg: str) -> None:
    print(f"[c8-01] {cc.now()} {msg}", flush=True)


def kv_from_log(path: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return out
    m = re.findall(r"# GPU blocks: (\d+)", text)
    if m:
        out["gpu_blocks"] = int(m[-1])
        out["kv_capacity_tokens"] = int(m[-1]) * 16
    m = re.findall(r"GPU KV cache size: ([\d,]+) tokens", text)
    if m:
        out["kv_capacity_tokens"] = int(m[-1].replace(",", ""))
    m = re.findall(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x", text)
    if m:
        out["max_concurrency"] = {"tokens_per_request": int(m[-1][0].replace(",", "")), "x": float(m[-1][1])}
    m = re.findall(r"(?i)(dual.?chunk[^\n]{0,120})", text)
    if m:
        out["dca_log_lines"] = list(dict.fromkeys(x.strip() for x in m))[:5]
    m = re.findall(r"(?i)(prefix caching[^\n]{0,120})", text)
    if m:
        out["prefix_cache_log_lines"] = list(dict.fromkeys(x.strip() for x in m))[:3]
    m = re.findall(r"vLLM API server version ([^\s]+)", text) or re.findall(r"vLLM version ([^\s]+)", text)
    if m:
        out["vllm_version"] = m[-1]
    return out


def fit_cold(points: list[tuple[int, float]]) -> dict[str, Any]:
    """seconds = a*L + b*L^2 (no intercept: an empty document costs nothing to read)."""

    import numpy as np

    if not points:
        return {"a": None, "b": None, "n": 0}
    L = np.array([p[0] for p in points], dtype=float)
    y = np.array([p[1] for p in points], dtype=float)
    if len(points) >= 2:
        X = np.stack([L, L * L], axis=1)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        a, b = float(coef[0]), float(coef[1])
        if a < 0 or b < 0:  # keep the term that explains the data
            a1 = float((L @ y) / (L @ L))
            b1 = float(((L * L) @ y) / ((L * L) @ (L * L)))
            r1 = float(((a1 * L - y) ** 2).sum())
            r2 = float(((b1 * L * L - y) ** 2).sum())
            a, b = (a1, 0.0) if r1 <= r2 else (0.0, b1)
    else:
        a, b = float(y[0] / L[0]), 0.0
    return {"a": a, "b": b, "n": len(points)}


def fit_warm(points: list[tuple[int, float]]) -> dict[str, Any]:
    import numpy as np

    if not points:
        return {"c": None, "d": None, "n": 0}
    L = np.array([p[0] for p in points], dtype=float)
    y = np.array([p[1] for p in points], dtype=float)
    if len(set(L.tolist())) >= 2:
        X = np.stack([np.ones_like(L), L], axis=1)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        c, d = float(coef[0]), float(coef[1])
        if d < 0 or c < 0:
            c, d = float(y.mean()), 0.0
    else:
        c, d = float(y.mean()), 0.0
    return {"c": c, "d": d, "n": len(points)}


def cold_s(f: dict[str, Any], L: int) -> float:
    return (f["a"] or 0.0) * L + (f["b"] or 0.0) * L * L


def warm_s(f: dict[str, Any], L: int) -> float:
    return (f["c"] or 0.0) + (f["d"] or 0.0) * L


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--datasets", default=os.environ.get("C8_DATASETS", ",".join(c8.DATASETS)))
    args = ap.parse_args(argv)
    paths = c8.C8Paths.from_env()
    sv = rd.server_env()
    datasets = [d for d in args.datasets.split(",") if d]
    client = rd.Client(args.port, sv["model"], sv["max_out"], sv["timeout"])
    max_gpu_h = float(os.environ.get("C8_MAX_GPU_HOURS", 6))
    policy = os.environ.get("C8_ON_OVER_BUDGET", "run")
    info: dict[str, Any] = {"step": "01", "commit": cc.git_commit(paths.root), "host": os.uname().nodename, "server": sv,
                            "gpus": os.environ.get("CUDA_VISIBLE_DEVICES"), "policy": policy, "max_gpu_hours": max_gpu_h,
                            "per_dataset": {}, "files": [], "stop_datasets": [], "warnings": []}
    status, code, t0 = "OK", 0, time.time()
    rows_all: list[dict[str, Any]] = []
    try:
        info["server_log"] = kv_from_log(sv["server_log"])
        for ds in datasets:
            inp = rd.Inputs(paths, ds)
            sample = [q for q in inp.queries if q["timing"]]
            d: dict[str, Any] = {"sample": len(sample), "skipped": []}
            info["per_dataset"][ds] = d
            # Short conditions: each sampled question once, alone.
            def timed(msgs: list[dict[str, str]], tries: int, what: str) -> tuple[dict[str, Any] | None, float | None]:
                try:
                    return client.chat(msgs, tries=tries)
                except Exception as exc:  # recorded; the timing goes on with the other requests
                    info.setdefault("errors", []).append(f"{ds} {what}: {str(exc)[:200]}")
                    log(f"{ds} {what}: request failed: {str(exc)[:200]}")
                    if not client.alive():
                        raise RuntimeError(f"the vLLM server stopped answering ({ds} {what}): {exc}") from exc
                    return None, None

            for cond in ("closed_book", "vanilla", "saadi"):
                for q in sample:
                    resp, sec = timed(inp.messages(cond, q["query_id"]), 3, f"{cond} {q['query_id']}")
                    if resp is None:
                        continue
                    rows_all.append({"dataset": ds, "condition": cond, "query_id": q["query_id"], "doc_id": q["doc_id"], "doc_tokens": q["doc_tokens"],
                                     "kind": "single", "seconds": sec, **rd.usage_of(resp), "prompt_tokens": q["prompt_tokens"][cond]})
                log(f"{ds}/{cond}: {len(sample)} timed")
            # Full document, grouped by document.
            by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for q in sample:
                why = inp.runnable(q, "full_doc", sv["max_model_len"], sv["max_out"])
                if why:
                    d["skipped"].append({"query_id": q["query_id"], "doc_id": q["doc_id"], "reason": why})
                    continue
                by_doc[q["doc_id"]].append(q)
            for doc_id, qs in sorted(by_doc.items(), key=lambda x: inp.docs[x[0]]["n_tokens"]):
                reset = client.reset_prefix_cache()
                first = qs[0]
                msgs = inp.messages("full_doc", first["query_id"])
                for kind in ("uncached", "cached_repeat"):
                    resp, sec = timed(msgs, 1, f"full_doc {doc_id} {kind}")
                    if resp is None:
                        break
                    rows_all.append({"dataset": ds, "condition": "full_doc", "query_id": first["query_id"], "doc_id": doc_id, "doc_tokens": first["doc_tokens"],
                                     "kind": kind, "cache_reset": reset if kind == "uncached" else None, "seconds": sec, **rd.usage_of(resp),
                                     "prompt_tokens": first["prompt_tokens"]["full_doc"], "answer": cc.clean_output(resp["choices"][0]["message"].get("content"))})
                    log(f"{ds}/full_doc {doc_id} ({first['doc_tokens']:,} tokens) {kind}: {sec:.1f} s")
                for q in qs[1:]:
                    resp, sec = timed(inp.messages("full_doc", q["query_id"]), 1, f"full_doc {doc_id} {q['query_id']}")
                    if resp is None:
                        continue
                    rows_all.append({"dataset": ds, "condition": "full_doc", "query_id": q["query_id"], "doc_id": doc_id, "doc_tokens": q["doc_tokens"],
                                     "kind": "cached", "seconds": sec, **rd.usage_of(resp), "prompt_tokens": q["prompt_tokens"]["full_doc"]})
            # Estimates for the whole dataset (the queries the full run will ask).
            sel = inp.selected()
            run_full = [q for q in sel if not inp.runnable(q, "full_doc", sv["max_model_len"], sv["max_out"])]
            dr = [r for r in rows_all if r["dataset"] == ds]
            fc = fit_cold([(r["prompt_tokens"], r["seconds"]) for r in dr if r["kind"] == "uncached"])
            fw = fit_warm([(r["prompt_tokens"], r["seconds"]) for r in dr if r["kind"] in ("cached", "cached_repeat")])
            per_doc_q: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for q in run_full:
                per_doc_q[q["doc_id"]].append(q)
            cold_total = sum(cold_s(fc, max(q["prompt_tokens"]["full_doc"] for q in qs)) for qs in per_doc_q.values())
            warm_total = sum(warm_s(fw, q["prompt_tokens"]["full_doc"]) for qs in per_doc_q.values() for q in qs[1:])
            nocache_total = sum(cold_s(fc, q["prompt_tokens"]["full_doc"]) + warm_s(fw, 0) for q in run_full)
            short_total = 0.0
            short_mean = {}
            for cond in ("closed_book", "vanilla", "saadi"):
                secs = [r["seconds"] for r in dr if r["condition"] == cond]
                short_mean[cond] = mean(secs) if secs else None
                short_total += (short_mean[cond] or 0.0) * len(sel)
            wall_h = (cold_total + warm_total + short_total) / 3600
            gpu_h = 4 * wall_h
            unc = [r for r in dr if r["kind"] == "uncached"]
            rep = {r["doc_id"]: r for r in dr if r["kind"] == "cached_repeat"}
            ratios = [(rep[r["doc_id"]]["seconds"] / r["seconds"]) for r in unc if r["doc_id"] in rep and r["seconds"] > 0 and r["doc_tokens"] >= 20000]
            cached_tok = [rep[r["doc_id"]].get("cached_tokens") for r in unc if r["doc_id"] in rep]
            cache_ok = bool(ratios) and median(ratios) < 0.5
            d.update({"selected": len(sel), "full_doc_runnable": len(run_full), "documents": len(per_doc_q), "fit_uncached": fc, "fit_cached": fw,
                      "short_mean_seconds": short_mean, "est_cold_h": cold_total / 3600, "est_warm_h": warm_total / 3600, "est_short_h": short_total / 3600,
                      "est_wall_h": wall_h, "est_gpu_h": gpu_h, "est_full_doc_nocache_gpu_h": 4 * nocache_total / 3600,
                      "cached_over_uncached_median": median(ratios) if ratios else None, "cached_tokens_on_repeat": cached_tok, "prefix_cache_effective": cache_ok,
                      "over_budget": gpu_h > max_gpu_h})
            if not cache_ok:
                info["warnings"].append(f"{ds}: the repeated question on a cached document took {d['cached_over_uncached_median']} of the uncached time: the prefix cache does not seem to work; "
                                        "the full run's per-document cost is then paid by every question")
            if gpu_h > max_gpu_h:
                if policy == "stop":
                    info["stop_datasets"].append(ds)
                    info["warnings"].append(f"{ds}: estimated {gpu_h:.1f} GPU h > {max_gpu_h} GPU h: NOT run (C8_ON_OVER_BUDGET=stop). Set C8_SUBSET_{ds.upper()} or C8_ON_OVER_BUDGET=run and start again.")
                else:
                    info["warnings"].append(f"{ds}: estimated {gpu_h:.1f} GPU h > {max_gpu_h} GPU h; run anyway (C8_ON_OVER_BUDGET=run). The run stops starting new documents at the deadline (C8_DEADLINE_HOURS).")
        out = paths.results / "timing" / "timing.jsonl"
        cc.write_jsonl(out, rows_all)
        info["files"].append(paths.rel(out))
        info["one_line"] = "; ".join(f"{c8.DATASET_DISPLAY[k]} est. {v['est_gpu_h']:.1f} GPU h (upper bound)" for k, v in info["per_dataset"].items())
        if info["stop_datasets"]:
            info["one_line"] += f"; over budget, not run: {','.join(info['stop_datasets'])}"
    except Exception as exc:
        status, code = "FAILED", 1
        info["traceback"] = traceback.format_exc()
        info["one_line"] = f"FAILED: {type(exc).__name__}: {str(exc)[:110]}"
        if rows_all:
            out = paths.results / "timing" / "timing.jsonl"
            cc.write_jsonl(out, rows_all)
            info["files"].append(paths.rel(out))
    info["seconds"] = round(time.time() - t0)
    info["status"] = status
    write_report(paths, info, rows_all, status)
    log(f"done: {status}")
    return code


def write_report(paths: c8.C8Paths, info: dict[str, Any], rows: list[dict[str, Any]], status: str) -> None:
    f = c8.fmt
    rep = paths.reports / "step_01_timing.md"
    sv = info["server"]
    L = [f"# C8 long-context reader - step 01 timing: {status}", "",
         f"Written {cc.now()}; commit {info['commit']}; host {info['host']}; GPUs {info['gpus']}; {info['seconds']} s.", "",
         f"Server: `{sv['model']}`, mode **{sv['mode']}** (dca = dual chunk attention backend, V0 engine; native = default backend, window capped at the native 262,144 tokens), "
         f"--max-model-len {f(sv['max_model_len'])}; log `{sv['server_log']}`.", ""]
    kv = info.get("server_log") or {}
    if kv:
        L += [f"From the server log: vLLM {kv.get('vllm_version')}; KV cache capacity {f(kv.get('kv_capacity_tokens'))} tokens"
              + (f"; max concurrency {kv['max_concurrency']['x']}x at {f(kv['max_concurrency']['tokens_per_request'])} tokens" if kv.get("max_concurrency") else "") + ".", ""]
        for x in kv.get("dca_log_lines", []):
            L.append(f"- log: `{x[:160]}`")
        L.append("")
    if info.get("warnings"):
        L += ["## Warnings", ""] + [f"- {w}" for w in info["warnings"]] + [""]
    if info.get("errors"):
        L += [f"## Failed requests ({len(info['errors'])}; not timed)", ""] + [f"- {e}" for e in info["errors"][:20]] + [""]
    if info["per_dataset"]:
        L += ["## Estimate of the full run (upper bound: every request alone)", ""]
        L += cc.md_table(["Dataset", "Queries", "Full doc runnable", "Documents", "Uncached reads (h)", "Cached questions (h)", "SAADI + Vanilla + closed-book (h)", "Wall (h)", "GPU h", f"> {info['max_gpu_hours']} GPU h", "Full doc without cache (GPU h)", "Prefix cache works"],
                         [[c8.DATASET_DISPLAY[k], v.get("selected"), v.get("full_doc_runnable"), v.get("documents"), f(v.get("est_cold_h"), 2), f(v.get("est_warm_h"), 2), f(v.get("est_short_h"), 2),
                           f(v.get("est_wall_h"), 2), f(v.get("est_gpu_h"), 1), v.get("over_budget"), f(v.get("est_full_doc_nocache_gpu_h"), 0), v.get("prefix_cache_effective")]
                          for k, v in info["per_dataset"].items()]) + [""]
        L += [f"Policy over budget: C8_ON_OVER_BUDGET={info['policy']}." + (f" Not run: {info['stop_datasets']}." if info["stop_datasets"] else ""), ""]
        L += ["Fits: uncached seconds = a*L + b*L^2, cached seconds = c + d*L (L = prompt tokens):", ""]
        L += cc.md_table(["Dataset", "a (s/token)", "b (s/token^2)", "points", "c (s)", "d (s/token)", "points", "cached / uncached (median, docs >= 20K tokens)"],
                         [[c8.DATASET_DISPLAY[k], f"{v['fit_uncached']['a']:.3g}" if v.get("fit_uncached", {}).get("a") is not None else "—",
                           f"{v['fit_uncached']['b']:.3g}" if v.get("fit_uncached", {}).get("b") is not None else "—", v.get("fit_uncached", {}).get("n"),
                           f"{v['fit_cached']['c']:.3g}" if v.get("fit_cached", {}).get("c") is not None else "—",
                           f"{v['fit_cached']['d']:.3g}" if v.get("fit_cached", {}).get("d") is not None else "—", v.get("fit_cached", {}).get("n"),
                           f(v.get("cached_over_uncached_median"), 3)] for k, v in info["per_dataset"].items() if "fit_uncached" in v]) + [""]
    if rows:
        L += ["## Single-request latency of the sample (seconds)", ""]
        agg = defaultdict(list)
        for r in rows:
            agg[(r["dataset"], r["condition"], r["kind"])].append(r)
        L += cc.md_table(["Dataset", "Condition", "Kind", "Requests", "Mean s", "Median s", "Max s", "Mean prompt tokens", "Mean output tokens"],
                         [[c8.DATASET_DISPLAY[k[0]], c8.CONDITION_DISPLAY[k[1]], k[2], len(v), f(mean(x["seconds"] for x in v), 2), f(median(x["seconds"] for x in v), 2),
                           f(max(x["seconds"] for x in v), 2), f(mean(x["prompt_tokens"] for x in v), 0), f(mean((x.get("completion_tokens") or 0) for x in v), 1)]
                          for k, v in sorted(agg.items())]) + [""]
        unc = [r for r in rows if r["kind"] == "uncached"]
        if unc:
            L += ["Full document, per sampled document:", ""]
            rep_ = {(r["dataset"], r["doc_id"]): r for r in rows if r["kind"] == "cached_repeat"}
            L += cc.md_table(["Dataset", "Document", "Prompt tokens", "Uncached s", "Same question, cached s", "Cached tokens reported", "Cache reset ok"],
                             [[c8.DATASET_DISPLAY[r["dataset"]], r["doc_id"], f(r["prompt_tokens"]), f(r["seconds"], 1), f(rep_.get((r["dataset"], r["doc_id"]), {}).get("seconds"), 2),
                               f(rep_.get((r["dataset"], r["doc_id"]), {}).get("cached_tokens")), r.get("cache_reset")] for r in unc]) + [""]
    for k, v in info["per_dataset"].items():
        if v.get("skipped"):
            L += [f"{c8.DATASET_DISPLAY[k]}: {len(v['skipped'])} sampled queries skipped for the full document: {v['skipped'][0]['reason']}.", ""]
    if info.get("traceback"):
        L += ["## Error", "", "```text", *info["traceback"].splitlines()[-60:], "```", ""]
    L += ["## Files written", ""] + [f"- `{x}`" for x in info["files"]] + [f"- `{paths.rel(rep)}`"]
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text("\n".join(L) + "\n", encoding="utf-8")
    cc.write_json(rep.with_suffix(".json"), info)
    rep.with_suffix(".summary.txt").write_text(info.get("one_line", status) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
