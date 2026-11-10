"""C8 step 02 (GPU, server already up): the 7B reader answers every selected query of one dataset in
the four conditions. Order: closed-book, Vanilla, SAADI (cheap), then the full documents.

Full document: documents in a seed-42 random order; per document the prefix cache is emptied, the
first question is sent ALONE (it reads the whole document: its seconds are the uncached cost), then
the document's other questions together (they reuse the cached document). No new document is
started after the deadline (C8_DEADLINE_EPOCH, set by the GPU script); the report says how many
queries that left out. A prompt longer than the served window is never cut: the query is skipped
and listed. Every answer is appended to results/rebuttal/c8_long_context/answers/<dataset>/
<condition>.jsonl as it arrives; a restart skips what is done (same inputs and server mode).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from statistics import mean
from typing import Any

import c8_common as c8
import c8_reader as rd

cc = c8.cc
ORDER = ("closed_book", "vanilla", "saadi", "full_doc")


def log(msg: str) -> None:
    print(f"[c8-02] {cc.now()} {msg}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=c8.DATASETS)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--conditions", default=",".join(ORDER))
    args = ap.parse_args(argv)
    paths = c8.C8Paths.from_env()
    sv = rd.server_env()
    ds = args.dataset
    client = rd.Client(args.port, sv["model"], sv["max_out"], sv["timeout"])
    conc = max(1, sv["concurrency"])
    info: dict[str, Any] = {"step": "02", "dataset": ds, "server": sv, "commit": cc.git_commit(paths.root), "host": os.uname().nodename,
                            "gpus": os.environ.get("CUDA_VISIBLE_DEVICES"), "subset": c8.subset_size(ds), "conditions": {}, "files": [], "docs": []}
    status, code, t0 = "OK", 0, time.time()
    try:
        inp = rd.Inputs(paths, ds)
        sel = inp.selected()
        info["selected"] = len(sel)
        for cond in [c for c in args.conditions.split(",") if c]:
            key = inp.key(cond, sv["mode"])
            out = paths.answers_file(ds, cond)
            out.parent.mkdir(parents=True, exist_ok=True)
            have, dropped = rd.done_ids(out, key)
            skipped = []
            todo = []
            for q in sel:
                if q["query_id"] in have:
                    continue
                why = inp.runnable(q, cond, sv["max_model_len"], sv["max_out"])
                if why:
                    skipped.append({"query_id": q["query_id"], "doc_id": q["doc_id"], "reason": why})
                else:
                    todo.append(q)
            ci = {"selected": len(sel), "done_before": len(have), "dropped_stale_rows": dropped, "skipped": len(skipped), "skipped_examples": skipped[:5],
                  "skipped_docs": sorted({s["doc_id"] for s in skipped}), "asked": 0, "errors": 0, "finish_length": 0, "prompt_token_mismatch": 0,
                  "not_started_deadline": 0, "key": key}
            info["conditions"][cond] = ci
            log(f"{ds}/{cond}: {len(sel)} selected, {len(have)} done, {len(todo)} to ask, {len(skipped)} skipped")
            t1 = time.time()

            def ask(q: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], float]:
                resp, sec = client.chat(inp.messages(cond, q["query_id"]), tries=1 if cond == "full_doc" else 3)
                return q, resp, sec

            def save(fh: Any, q: dict[str, Any], resp: dict[str, Any], sec: float, **extra: Any) -> None:
                ch = resp["choices"][0]
                u = rd.usage_of(resp)
                if ch.get("finish_reason") == "length":
                    ci["finish_length"] += 1
                if u["server_prompt_tokens"] is not None and u["server_prompt_tokens"] != q["prompt_tokens"][cond]:
                    ci["prompt_token_mismatch"] += 1
                fh.write(json.dumps(rd.answer_row(inp, cond, q["query_id"], ch["message"].get("content") or "", key, finish_reason=ch.get("finish_reason"),
                                                  latency_s=round(sec, 4), served_mode=sv["mode"], **u, **extra), ensure_ascii=False) + "\n")
                fh.flush()

            def error(exc: Exception) -> None:
                ci["errors"] += 1
                if ci["errors"] <= 5:
                    ci.setdefault("error_examples", []).append(str(exc)[:300])

            with out.open("a", encoding="utf-8") as fh:
                if cond != "full_doc":
                    with cf.ThreadPoolExecutor(max_workers=conc) as pool:
                        futs = [pool.submit(ask, q) for q in todo]
                        for n, fut in enumerate(cf.as_completed(futs), start=1):
                            try:
                                q, resp, sec = fut.result()
                            except Exception as exc:
                                error(exc)
                                continue
                            ci["asked"] += 1
                            save(fh, q, resp, sec, batch="concurrent")
                            if n % 500 == 0:
                                log(f"{ds}/{cond}: {n}/{len(todo)}")
                else:
                    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
                    for q in todo:
                        by_doc[q["doc_id"]].append(q)
                    order = [d for d in rd.doc_order(list(by_doc)) if d in by_doc]
                    docs_path = paths.answers_file(ds, "full_doc").with_name("full_doc_documents.jsonl")
                    with docs_path.open("a", encoding="utf-8") as dfh:
                        for i, doc_id in enumerate(order, start=1):
                            qs = by_doc[doc_id]
                            if sv["deadline"] and time.time() > sv["deadline"]:
                                ci["not_started_deadline"] += sum(len(by_doc[d]) for d in order[i - 1:])
                                info["deadline_hit"] = True
                                log(f"{ds}/full_doc: deadline reached; {ci['not_started_deadline']} queries on {len(order) - i + 1} documents not started")
                                break
                            reset = client.reset_prefix_cache()
                            td = time.time()
                            try:
                                q, resp, sec = ask(qs[0])
                            except Exception as exc:
                                error(exc)
                                log(f"{ds}/full_doc {doc_id}: first request failed: {str(exc)[:200]}")
                                if not client.alive():
                                    raise RuntimeError(f"the vLLM server stopped answering during {doc_id}: {exc}") from exc
                                continue
                            ci["asked"] += 1
                            first_u = rd.usage_of(resp)
                            save(fh, q, resp, sec, batch="alone", cache_state="first_after_reset" if reset else "first_on_document")
                            rest_secs = []
                            with cf.ThreadPoolExecutor(max_workers=conc) as pool:
                                futs = [pool.submit(ask, x) for x in qs[1:]]
                                for fut in cf.as_completed(futs):
                                    try:
                                        x, r2, s2 = fut.result()
                                    except Exception as exc:
                                        error(exc)
                                        continue
                                    ci["asked"] += 1
                                    rest_secs.append(s2)
                                    save(fh, x, r2, s2, batch="concurrent", cache_state="cached_document")
                            drow = {"doc_id": doc_id, "doc_tokens": qs[0]["doc_tokens"], "prompt_tokens_first": qs[0]["prompt_tokens"]["full_doc"],
                                    "first_query": qs[0]["query_id"], "uncached_seconds": round(sec, 3), "cache_reset": reset,
                                    "first_cached_tokens": first_u["cached_tokens"], "other_queries": len(qs) - 1, "other_answered": len(rest_secs),
                                    "other_mean_latency_s": round(mean(rest_secs), 3) if rest_secs else None,
                                    "document_wall_s": round(time.time() - td, 3), "key": key, "served_mode": sv["mode"]}
                            dfh.write(json.dumps(drow) + "\n")
                            dfh.flush()
                            info["docs"].append(drow)
                            log(f"{ds}/full_doc {i}/{len(order)} {doc_id} ({qs[0]['doc_tokens']:,} tokens): uncached {sec:.1f} s, "
                                f"{len(rest_secs)} more in {time.time() - td - sec:.1f} s")
                    info["files"].append(paths.rel(docs_path))
            ci["wall_seconds"] = round(time.time() - t1, 1)
            have, _ = rd.done_ids(out, key)
            ci["answered"] = len(have & {q["query_id"] for q in sel})
            ci["complete"] = ci["answered"] + len(skipped) == len(sel)
            info["files"].append(paths.rel(out))
            sk_path = out.with_name(f"{cond}.skipped.json")
            if skipped:
                cc.write_json(sk_path, skipped)
                info["files"].append(paths.rel(sk_path))
            elif sk_path.exists():
                sk_path.unlink()
        bad = [c for c, v in info["conditions"].items() if not v["complete"] and not v["not_started_deadline"]]
        if bad:
            raise RuntimeError(f"unanswered queries remain in {bad} (errors in the report); start the GPU script again to resume")
        parts = [f"{c} {v['answered']}" + (f" (+{v['skipped']} skipped)" if v["skipped"] else "") for c, v in info["conditions"].items()]
        info["one_line"] = "answered " + ", ".join(parts)
        if info.get("deadline_hit"):
            status = "PARTIAL (deadline)"
            info["one_line"] += f"; deadline: {info['conditions'].get('full_doc', {}).get('not_started_deadline')} full-document queries not started"
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
    sv = info["server"]
    rep = paths.reports / f"step_02_{ds}_reader.md"
    L = [f"# C8 long-context reader - step 02 reader, {c8.DATASET_DISPLAY[ds]}: {status}", "",
         f"Written {cc.now()}; commit {info['commit']}; host {info['host']}; GPUs {info['gpus']}; {info['seconds']} s.", "",
         f"Reader `{sv['model']}` (vLLM, tensor parallel 4, mode {sv['mode']}, --max-model-len {f(sv['max_model_len'])}), temperature 0, top_p 1, seed 42, "
         f"max output tokens {sv['max_out']}, {sv['concurrency']} requests in flight (full document: the first question of each document alone). "
         f"Queries: {info.get('selected')}" + (f" (subset: first {info['subset']} of the seed-42 permutation)" if info.get("subset") else " (all)") + ".", ""]
    rows = [[c8.CONDITION_DISPLAY[c], v["selected"], v["done_before"], v["asked"], v.get("answered"), v["skipped"], v["errors"], v["not_started_deadline"],
             v["finish_length"], v["prompt_token_mismatch"], v.get("wall_seconds"), "complete" if v.get("complete") else "INCOMPLETE"] for c, v in info["conditions"].items()]
    L += cc.md_table(["Condition", "Selected", "Done before", "Asked", "Answered", "Skipped (too long)", "Errors", "Not started (deadline)", "Cut at max tokens",
                      "Prompt tokens != step 00", "Wall s", "Status"], rows) + [""]
    for c, v in info["conditions"].items():
        if v["skipped"]:
            L += [f"{c8.CONDITION_DISPLAY[c]}: {v['skipped']} queries skipped on documents {', '.join(v['skipped_docs'])}: {v['skipped_examples'][0]['reason']}.", ""]
    if info.get("docs"):
        L += ["Full document, per document (this run):", ""]
        L += cc.md_table(["Document", "Tokens", "Uncached first question (s)", "Cache reset", "Other questions", "Their mean latency (s)", "Document wall (s)"],
                         [[d["doc_id"], f(d["doc_tokens"]), f(d["uncached_seconds"], 1), d["cache_reset"], d["other_answered"], f(d["other_mean_latency_s"], 2), f(d["document_wall_s"], 1)]
                          for d in info["docs"][:60]]) + [""]
    ex = {c: v["error_examples"] for c, v in info["conditions"].items() if v.get("error_examples")}
    if ex:
        L += ["Error examples:", "", "```json", json.dumps(ex, indent=1)[:3000], "```", ""]
    if info.get("traceback"):
        L += ["## Error", "", "```text", *info["traceback"].splitlines()[-60:], "```", ""]
    L += ["## Files written", ""] + [f"- `{x}`" for x in dict.fromkeys(info["files"])] + [f"- `{paths.rel(rep)}`"]
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text("\n".join(L) + "\n", encoding="utf-8")
    cc.write_json(rep.with_suffix(".json"), info)
    rep.with_suffix(".summary.txt").write_text(info.get("one_line", status) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
