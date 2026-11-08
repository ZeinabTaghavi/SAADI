#!/usr/bin/env python3
"""B2 Contextual Retrieval, whole-document variant: one LLM call per chunk writes the chunk's context.

Prompt: Anthropic's Contextual Retrieval post ("Introducing Contextual Retrieval", 2024),
verbatim, as one user message: the document, then the chunk, then the instruction. The model
(Qwen3-30B-A3B-Instruct-2507 on the local vLLM server, temperature 0) sees as much of the book
as the served window allows: the whole book when it fits; otherwise the longest prefix of
whole chunks that fits next to the longest chunk of that book and the output budget. The
document part is identical for every chunk of a book, so the server's prefix cache computes
it once per book (one warm-up call, then the rest in parallel).

For every chunk it records which chunks of the book were visible; retrieve.py derives from it
whether each persona's reveal was inside the visible context. Resumable per chunk
(outputs/probe_baselines/b2_contextual/contexts/<book_id>.jsonl).

Environment: PBL_VLLM_URL (http://127.0.0.1:<port>/v1), PBL_MODEL (served model name),
PBL_MODEL_ID (Hugging Face id, for the tokenizer), PBL_B2_CONCURRENCY (16).
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common"))
import chunk_map as CM  # noqa: E402
import report_utils as RPT  # noqa: E402

SYSTEM = "b2_contextual"
DOC_PROMPT = "<document>\n{doc}\n</document>\n"
CHUNK_PROMPT = (
    "Here is the chunk we want to situate within the whole document\n"
    "<chunk>\n{chunk}\n</chunk>\n"
    "Please give a short succinct context to situate this chunk within the overall document "
    "for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."
)
DOC_SEPARATOR = "\n\n"
MAX_TOKENS = 200  # the post's contexts are 50-100 tokens; this is a ceiling, not a target
MARGIN = 1024
URL = os.environ.get("PBL_VLLM_URL", "http://127.0.0.1:8047/v1").rstrip("/")
MODEL = os.environ.get("PBL_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
MODEL_ID = os.environ.get("PBL_MODEL_ID", "Qwen/Qwen3-30B-A3B-Instruct-2507")
CONCURRENCY = int(os.environ.get("PBL_B2_CONCURRENCY", "16"))
SEED = int(os.environ.get("PBL_SEED", "42"))


def http_json(url: str, payload: dict | None = None, timeout: float = 3600) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def served_window() -> int:
    models = http_json(URL + "/models", timeout=60)["data"]
    model = next((m for m in models if m.get("id") == MODEL), models[0])
    return int(model["max_model_len"])


def prompt_for(doc: str, chunk: str) -> str:
    return DOC_PROMPT.format(doc=doc) + CHUNK_PROMPT.format(chunk=chunk)


def call(prompt: str) -> dict:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
        "seed": SEED,
    }
    last = None
    for attempt in range(4):
        try:
            t0 = time.time()
            out = http_json(URL + "/chat/completions", payload)
            choice = out["choices"][0]
            usage = out.get("usage") or {}
            return {
                "context": (choice["message"].get("content") or "").strip(),
                "finish_reason": choice.get("finish_reason"),
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
                "cached_tokens": int(((usage.get("prompt_tokens_details") or {}) or {}).get("cached_tokens") or 0),
                "seconds": round(time.time() - t0, 2),
            }
        except Exception as exc:  # transient server errors: retry with backoff
            last = exc
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"chat completion failed 4 times: {last}")


def plan_book(tok, window: int, chunks: list[str]) -> dict:
    """How much of the book is visible: whole book or the longest prefix of whole chunks that fits."""

    overhead = len(tok.apply_chat_template([{"role": "user", "content": prompt_for("", "")}], tokenize=True, add_generation_prompt=True))
    lens = [len(tok.encode(c, add_special_tokens=False)) for c in chunks]
    budget = window - overhead - max(lens) - MAX_TOKENS - MARGIN
    total, last = 0, -1
    for i, n in enumerate(lens):
        add = n + (1 if i else 0)
        if total + add > budget:
            break
        total += add
        last = i
    # Verify with the exact tokenization of the longest prompt of this book.
    while last >= 0:
        doc = DOC_SEPARATOR.join(chunks[: last + 1])
        longest = max(range(len(chunks)), key=lambda i: lens[i])
        n_prompt = len(
            tok.apply_chat_template(
                [{"role": "user", "content": prompt_for(doc, chunks[longest])}], tokenize=True, add_generation_prompt=True
            )
        )
        if n_prompt + MAX_TOKENS <= window:
            break
        last -= 1
    if last < 0:
        raise RuntimeError("not even the first chunk fits the served window")
    return {
        "window": window,
        "overhead_tokens": overhead,
        "chunk_tokens": lens,
        "visible_last_chunk": last,
        "whole_book": last == len(chunks) - 1,
        "doc_tokens": sum(lens[: last + 1]) + last,
        "book_tokens": sum(lens) + len(lens) - 1,
        "longest_prompt_tokens": n_prompt,
    }


def main() -> int:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    window = served_window()
    print(f"B2: served window {window} tokens; model {MODEL}; concurrency {CONCURRENCY}", flush=True)
    ctx_dir = CM.system_dir(SYSTEM) / "contexts"
    ctx_dir.mkdir(parents=True, exist_ok=True)
    per_book, unexpected = {}, []
    t_all = time.time()
    for book_id in CM.book_ids():
        inp = CM.load_inputs(book_id)
        chunks, ids = inp["chunks"], inp["chunk_ids"]
        plan = plan_book(tok, window, chunks)
        doc = DOC_SEPARATOR.join(chunks[: plan["visible_last_chunk"] + 1])
        path = ctx_dir / f"{book_id}.jsonl"
        done = {r["chunk_id"]: r for r in CM.read_jsonl(path)} if path.exists() else {}
        todo = [i for i, cid in enumerate(ids) if cid not in done]
        print(
            f"B2 {book_id}: {len(chunks)} chunks, book {plan['book_tokens']} tokens, visible through chunk {plan['visible_last_chunk']} ({'whole book' if plan['whole_book'] else 'prefix'}); {len(todo)} to do",
            flush=True,
        )
        before = CM.vllm_counters(URL)
        tb = time.time()
        lock = threading.Lock()

        def run(i: int, doc=doc, chunks=chunks, ids=ids, plan=plan, lock=lock, path=path) -> None:
            res = call(prompt_for(doc, chunks[i]))
            row = {
                "chunk_id": ids[i],
                "chunk_index": i,
                "visible_last_chunk": plan["visible_last_chunk"],
                "whole_book": plan["whole_book"],
                **res,
            }
            with lock, path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        if todo:
            run(todo[0])  # warm-up: computes the book's prefix once for the cache
            with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
                futures = [pool.submit(run, i) for i in todo[1:]]
                for n, fut in enumerate(as_completed(futures), 1):
                    fut.result()
                    if n % 50 == 0:
                        print(f"  {book_id}: {n + 1}/{len(todo)} contexts, {time.time() - tb:.0f} s", flush=True)
        rows = CM.read_jsonl(path)
        delta = CM.counter_delta(before, CM.vllm_counters(URL))
        lengths = [len(tok.encode(r["context"], add_special_tokens=False)) for r in rows]
        per_book[book_id] = {
            "chunks": len(chunks),
            "calls": len(rows),
            "visible_last_chunk": plan["visible_last_chunk"],
            "whole_book": plan["whole_book"],
            "book_tokens": plan["book_tokens"],
            "doc_tokens": plan["doc_tokens"],
            "input_tokens": sum(r["prompt_tokens"] for r in rows),
            "generated_tokens": sum(r["completion_tokens"] for r in rows),
            "cached_tokens_reported": sum(r.get("cached_tokens", 0) for r in rows),
            "server_counters_this_invocation": delta,
            "context_tokens_median": statistics.median(lengths) if lengths else None,
            "context_tokens_max": max(lengths) if lengths else None,
            "stopped_at_max_tokens": sum(1 for r in rows if r.get("finish_reason") == "length"),
            "seconds_this_invocation": round(time.time() - tb, 1),
            "index_chars": sum(len(r["context"]) for r in rows) + sum(len(c) for c in chunks),
        }
        if len(rows) != len(chunks):
            unexpected.append(f"{book_id}: {len(rows)} contexts for {len(chunks)} chunks")
        CM.write_numbers(
            SYSTEM,
            "build",
            {
                "window": window,
                "model": MODEL,
                "max_tokens": MAX_TOKENS,
                "temperature": 0.0,
                "seed": SEED,
                "per_book": per_book,
                "unexpected": unexpected,
                "seconds_total": round(time.time() - t_all, 1),
            },
        )
    rows = [
        [
            b,
            v["chunks"],
            v["book_tokens"],
            "whole" if v["whole_book"] else f"prefix to chunk {v['visible_last_chunk']}",
            v["calls"],
            v["input_tokens"],
            v["generated_tokens"],
            v["context_tokens_median"],
            v["stopped_at_max_tokens"],
            v["seconds_this_invocation"],
        ]
        for b, v in per_book.items()
    ]
    frag = f"### B2 contexts per book (served window {window} tokens)\n\n" + RPT.md_table(
        [
            "book",
            "chunks",
            "book tokens",
            "visible",
            "calls",
            "input tokens",
            "generated tokens",
            "median context tokens",
            "cut at max tokens",
            "seconds",
        ],
        rows,
    )
    frag += "\n\nInput tokens are the logical prompt tokens of every call (the whole visible book each time); the server's prefix cache recomputes only the chunk part after the first call of a book (server counters per book are in the JSON)."
    (CM.system_dir(SYSTEM) / "report_build.md").write_text(frag + "\n", encoding="utf-8")
    (CM.system_dir(SYSTEM) / "files_build.txt").write_text(str(ctx_dir) + "/<book>.jsonl\n", encoding="utf-8")
    return 1 if unexpected else 0


if __name__ == "__main__":
    raise SystemExit(main())
