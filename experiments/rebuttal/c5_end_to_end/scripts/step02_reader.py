"""C5 step 02 (GPU): the reader answers every query of every available condition of one dataset.

One vLLM server per dataset (started by sh/run_group.sh), one model, one prompt template, greedy
decoding (temperature 0, seed 42, fixed max output tokens). Every answer is appended to
results/rebuttal/c5_end_to_end/answers/<dataset>/<condition>.jsonl as soon as it arrives, with its
scores and the evidence-in-context flag; a restart skips the queries already answered. Answers
whose context file changed since (other sha256) are not mixed: the condition is redone.
"A-RAG own answer" needs no call: its saved final answers are written and scored the same way.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import c5_common as cc


def log(msg: str) -> None:
    print(f"[c5-02] {cc.now()} {msg}", flush=True)


def post(url: str, payload: dict[str, Any], timeout: float = 600.0, tries: int = 5) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = exc
            body = ""
            if isinstance(exc, urllib.error.HTTPError):
                try:
                    body = exc.read().decode("utf-8")[:500]
                except Exception:
                    pass
                if 400 <= exc.code < 500:  # a request error will not get better by retrying
                    raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"request failed after {tries} tries: {last}")


def answer_row(rec: dict[str, Any], output: str, ctx_sha: str, **extra: Any) -> dict[str, Any]:
    scores = cc.score_answer(output, rec["gold_answers"])
    return {
        "query_id": rec["query_id"],
        "doc_id": rec["doc_id"],
        "condition": rec["condition"],
        "question": rec["question"],
        "gold": rec["gold_answers"],
        "output": output,
        "answer": cc.clean_output(output),
        **scores,
        "evidence_in_context": rec.get("evidence_in_context"),
        "retrieval_hit10": rec.get("retrieval_hit10"),
        "n_passages": rec.get("n_passages"),
        "context_tokens": rec.get("context_tokens"),
        "prompt_tokens": rec.get("prompt_tokens"),
        "context_sha256": ctx_sha,
        **extra,
    }


def done_ids(path: Path, ctx_sha: str) -> tuple[set[str], int]:
    """Query ids already answered from the same contexts file; rows of an older file are dropped."""

    if not path.exists():
        return set(), 0
    keep, dropped = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except Exception:
            dropped += 1  # a line cut by a crash
            continue
        if r.get("context_sha256") == ctx_sha:
            keep.append(r)
        else:
            dropped += 1
    if dropped:
        cc.write_jsonl(path, keep)
    return {r["query_id"] for r in keep}, dropped


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=cc.DATASETS)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--conditions", default=",".join(cc.ALL_CONDITIONS))
    args = ap.parse_args(argv)
    paths = cc.C5Paths.from_env()
    env = dict(os.environ)
    ds = args.dataset
    model = env["C5_READER_MODEL"]
    max_out = int(env.get("C5_MAX_OUTPUT_TOKENS", 64))
    conc = int(env.get("C5_READER_CONCURRENCY", 48))
    url = f"http://127.0.0.1:{args.port}/v1/chat/completions"
    info: dict[str, Any] = {"step": "02", "dataset": ds, "port": args.port, "model": model, "max_output_tokens": max_out, "concurrency": conc,
                            "sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0, "seed": cc.SEED, "max_tokens": max_out},
                            "commit": cc.git_commit(paths.root), "host": os.uname().nodename, "conditions": {}, "files": []}
    status, code, t0 = "OK", 0, time.time()
    try:
        manifest = cc.read_json(paths.context_manifest(ds))
        for cond in [c for c in args.conditions.split(",") if c]:
            meta = manifest["conditions"].get(cond) or {}
            if not meta.get("available"):
                info["conditions"][cond] = {"status": "no contexts (missing in step 00)"}
                continue
            ctx_path = paths.context_file(ds, cond)
            ctx_sha = cc.sha256_file(ctx_path)
            if ctx_sha != meta.get("sha256"):
                raise RuntimeError(f"{ctx_path} changed after step 00 wrote its manifest; rerun step 00")
            recs = cc.read_jsonl(ctx_path)
            out = paths.answers_file(ds, cond)
            out.parent.mkdir(parents=True, exist_ok=True)
            have, dropped = done_ids(out, ctx_sha)
            todo = [r for r in recs if r["query_id"] not in have]
            c_info = {"queries": len(recs), "already_done": len(have), "dropped_stale_rows": dropped, "asked": len(todo), "errors": 0,
                      "finish_length": 0, "prompt_token_mismatch": 0}
            info["conditions"][cond] = c_info
            log(f"{ds}/{cond}: {len(recs)} queries, {len(have)} done, {len(todo)} to ask")
            t1 = time.time()
            with out.open("a", encoding="utf-8") as fh:
                if cond in cc.SCORED_ONLY:
                    for r in todo:
                        fh.write(json.dumps(answer_row(r, str(r.get("prediction") or ""), ctx_sha, source="agent's saved final answer",
                                                       termination=r.get("termination"), agent_error=r.get("agent_error")), ensure_ascii=False) + "\n")
                else:
                    def ask(r: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
                        payload = {"model": model, "messages": cc.build_messages(r["question"], r["passages"]),
                                   "temperature": 0.0, "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0,
                                   "seed": cc.SEED, "max_tokens": max_out, "n": 1}
                        return r, post(url, payload)

                    with cf.ThreadPoolExecutor(max_workers=conc) as pool:
                        futs = [pool.submit(ask, r) for r in todo]
                        for n, fut in enumerate(cf.as_completed(futs), start=1):
                            try:
                                r, resp = fut.result()
                            except Exception as exc:  # recorded, the query stays undone for the next start
                                c_info["errors"] += 1
                                if c_info["errors"] <= 5:
                                    c_info.setdefault("error_examples", []).append(str(exc)[:300])
                                continue
                            choice = resp["choices"][0]
                            usage = resp.get("usage") or {}
                            if choice.get("finish_reason") == "length":
                                c_info["finish_length"] += 1
                            if usage.get("prompt_tokens") is not None and usage["prompt_tokens"] != r["prompt_tokens"]:
                                c_info["prompt_token_mismatch"] += 1
                            fh.write(json.dumps(answer_row(r, choice["message"].get("content") or "", ctx_sha,
                                                           finish_reason=choice.get("finish_reason"), server_prompt_tokens=usage.get("prompt_tokens"),
                                                           completion_tokens=usage.get("completion_tokens")), ensure_ascii=False) + "\n")
                            fh.flush()
                            if n % 500 == 0:
                                log(f"{ds}/{cond}: {n}/{len(todo)}")
            c_info["seconds"] = round(time.time() - t1, 1)
            have, _ = done_ids(out, ctx_sha)
            c_info["answered"] = len(have)
            c_info["complete"] = len(have) == len(recs)
            info["files"].append(paths.rel(out))
        incomplete = [c for c, v in info["conditions"].items() if "complete" in v and not v["complete"]]
        if incomplete:
            raise RuntimeError(f"unanswered queries remain in {incomplete} (errors are in the report); start the script again to resume")
        n_ans = sum(v.get("answered", 0) for v in info["conditions"].values())
        info["one_line"] = f"reader answered {n_ans} queries over {sum(1 for v in info['conditions'].values() if v.get('complete'))} conditions"
    except Exception as exc:
        status, code = "FAILED", 1
        info["traceback"] = traceback.format_exc()
        info["one_line"] = f"FAILED: {type(exc).__name__}: {str(exc)[:110]}"
    info["seconds"] = round(time.time() - t0)
    rep = paths.reports / f"step_02_{ds}_reader.md"
    L = [f"# C5 end-to-end QA - step 02 reader, {cc.DATASET_DISPLAY[ds]}: {status}", "",
         f"Written {cc.now()}; commit {info['commit']}; host {info['host']}; GPUs {env.get('CUDA_VISIBLE_DEVICES')}; port {args.port}; {info['seconds']} s.", "",
         f"Reader `{model}` (vLLM, tensor parallel {env.get('C5_TENSOR_PARALLEL')}), temperature 0, top_p 1, seed 42, max output tokens {max_out}, "
         f"{conc} requests in flight. Prompt template: the one in step 00's report (c5_common.READER_TEMPLATE).", ""]
    rows = [[c, v.get("queries"), v.get("already_done"), v.get("asked"), v.get("answered"), v.get("errors"), v.get("finish_length"),
             v.get("prompt_token_mismatch"), v.get("seconds"), v.get("status", "complete" if v.get("complete") else "INCOMPLETE")] for c, v in info["conditions"].items()]
    L += cc.md_table(["Condition", "Queries", "Done before", "Asked", "Answered", "Errors", "Cut at max tokens", "Prompt tokens != step 00", "Seconds", "Status"], rows)
    ex = {c: v["error_examples"] for c, v in info["conditions"].items() if v.get("error_examples")}
    if ex:
        L += ["", "Error examples:", "", "```json", json.dumps(ex, indent=1)[:3000], "```"]
    if info.get("traceback"):
        L += ["", "## Error", "", "```text", *info["traceback"].splitlines()[-60:], "```"]
    L += ["", "## Files written", ""] + [f"- `{f}`" for f in info["files"]] + [f"- `{paths.rel(rep)}`"]
    rep.write_text("\n".join(L) + "\n", encoding="utf-8")
    cc.write_json(rep.with_suffix(".json"), info)
    rep.with_suffix(".summary.txt").write_text(info["one_line"] + "\n", encoding="utf-8")
    log(f"done: {status}")
    return code


if __name__ == "__main__":
    sys.exit(main())
