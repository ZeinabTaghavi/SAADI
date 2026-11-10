"""C8 reader client and the per-query inputs (used by steps 01 and 02).

One vLLM server (sh/run_gpus_4567.sh), C5's prompt and decoding: temperature 0, top_p 1, seed 42,
C5's max output tokens. The full-document prompt is built here from step 00's documents; the other
three conditions use C5's context files as they are (sha256 checked against step 00's manifest).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import c8_common as c8

cc = c8.cc


class Client:
    def __init__(self, port: int, model: str, max_out: int, timeout: float):
        self.base = f"http://127.0.0.1:{port}"
        self.model = model
        self.max_out = max_out
        self.timeout = timeout

    def _post(self, path: str, payload: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
        data = json.dumps(payload or {}).encode("utf-8")
        req = urllib.request.Request(self.base + path, data=data, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
        return json.loads(body) if body.strip() else {}

    def chat(self, messages: list[dict[str, str]], tries: int = 2) -> tuple[dict[str, Any], float]:
        """One greedy answer; returns (response, seconds of the successful request). A 4xx error is
        not retried (it will not get better)."""

        payload = {"model": self.model, "messages": messages, "temperature": 0.0, "top_p": 1.0, "top_k": -1,
                   "repetition_penalty": 1.0, "seed": c8.SEED, "max_tokens": self.max_out, "n": 1}
        last: Exception | None = None
        for attempt in range(tries):
            t0 = time.perf_counter()
            try:
                resp = self._post("/v1/chat/completions", payload, self.timeout)
                return resp, time.perf_counter() - t0
            except urllib.error.HTTPError as exc:
                body = ""
                try:
                    body = exc.read().decode("utf-8")[:400]
                except Exception:
                    pass
                if 400 <= exc.code < 500:
                    raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
                last = RuntimeError(f"HTTP {exc.code}: {body}")
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                last = exc
            time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"request failed after {tries} tries: {last}")

    def reset_prefix_cache(self) -> bool:
        """vLLM's dev endpoint (server started with VLLM_SERVER_DEV_MODE=1). Best effort."""

        try:
            self._post("/reset_prefix_cache", None, 120)
            return True
        except Exception:
            return False

    def alive(self) -> bool:
        try:
            with urllib.request.urlopen(self.base + "/v1/models", timeout=10) as r:
                return r.status == 200
        except Exception:
            return False


def usage_of(resp: dict[str, Any]) -> dict[str, Any]:
    u = resp.get("usage") or {}
    det = u.get("prompt_tokens_details") or {}
    return {"server_prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
            "cached_tokens": det.get("cached_tokens") if isinstance(det, dict) else None}


# ---------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------


class Inputs:
    """Step 00's documents and query index of one dataset, and C5's contexts of conditions 2-4."""

    def __init__(self, paths: c8.C8Paths, dataset: str):
        self.paths = paths
        self.dataset = dataset
        man_p = paths.manifest(dataset)
        if not man_p.exists():
            raise RuntimeError(f"{man_p} missing: run sh/00_prepare.sh first")
        self.manifest = cc.read_json(man_p)
        for part in ("documents", "queries"):
            p = paths.root / self.manifest[part]["file"] if not Path(self.manifest[part]["file"]).is_absolute() else Path(self.manifest[part]["file"])
            if cc.sha256_file(p) != self.manifest[part]["sha256"]:
                raise RuntimeError(f"{p} changed after step 00 wrote its manifest; rerun step 00")
        self.docs = {d["doc_id"]: d for d in cc.read_jsonl(paths.documents_file(dataset))}
        self.queries = cc.read_jsonl(paths.queries_file(dataset))
        self.by_id = {q["query_id"]: q for q in self.queries}
        self.full_key = "c8:" + self.manifest["documents"]["sha256"][:16] + ":" + self.manifest["queries"]["sha256"][:16]
        self._c5: dict[str, dict[str, dict[str, Any]]] = {}
        self.c5_sha = {f["condition"]: f["sha256"] for f in self.manifest["c5_inputs"]["files"]}

    def c5_rows(self, condition: str) -> dict[str, dict[str, Any]]:
        if condition not in self._c5:
            p = self.paths.c5_context_file(self.dataset, condition)
            if cc.sha256_file(p) != self.c5_sha[condition]:
                raise RuntimeError(f"C5's {p} changed after C8's step 00; rerun C8's step 00")
            self._c5[condition] = {r["query_id"]: r for r in cc.read_jsonl(p)}
        return self._c5[condition]

    def key(self, condition: str, mode: str) -> str:
        base = self.full_key if condition == "full_doc" else "c5:" + self.c5_sha[condition][:32]
        return f"{base}:{mode}"

    def messages(self, condition: str, qid: str) -> list[dict[str, str]]:
        q = self.by_id[qid]
        if condition == "full_doc":
            return c8.full_doc_messages(q["question"], self.docs[q["doc_id"]]["text"])
        r = self.c5_rows(condition)[qid]
        return cc.build_messages(r["question"], r["passages"])

    def selected(self) -> list[dict[str, Any]]:
        """All queries, or the first N of the seed-42 permutation when C8_SUBSET_<DATASET> is set."""

        n = c8.subset_size(self.dataset)
        qs = self.queries if n is None else [q for q in self.queries if q["subset_rank"] < n]
        return sorted(qs, key=lambda q: q["subset_rank"])

    def runnable(self, q: dict[str, Any], condition: str, max_model_len: int, max_out: int) -> str | None:
        """None when the prompt fits the served window, else the reason it is skipped (never cut)."""

        if q["prompt_tokens"][condition] + max_out <= max_model_len:
            return None
        if condition == "full_doc" and q.get("over_native") and max_model_len <= 262144 + 8192:
            return "document beyond the native 262,144-token window and the server runs without dual chunk attention"
        return f"prompt {q['prompt_tokens'][condition]:,} + {max_out} output tokens > --max-model-len {max_model_len:,}"


def answer_row(inp: Inputs, condition: str, qid: str, output: str, key: str, **extra: Any) -> dict[str, Any]:
    q = inp.by_id[qid]
    scores = cc.score_answer(output, q["gold_answers"])
    return {"query_id": qid, "doc_id": q["doc_id"], "condition": condition, "question": q["question"], "gold": q["gold_answers"],
            "output": output, "answer": cc.clean_output(output), **scores,
            "prompt_tokens": q["prompt_tokens"][condition], "context_tokens": q["context_tokens"][condition],
            "evidence_third": q.get("evidence_third"), "evidence_spread": q.get("evidence_spread"), "key": key, **extra}


def done_ids(path: Path, key: str) -> tuple[set[str], int]:
    """Query ids already answered with the same inputs and server mode; other rows are dropped."""

    if not path.exists():
        return set(), 0
    keep, dropped = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except Exception:
            dropped += 1
            continue
        if r.get("key") == key:
            keep.append(r)
        else:
            dropped += 1
    if dropped:
        cc.write_jsonl(path, keep)
    return {r["query_id"] for r in keep}, dropped


def doc_order(doc_ids: list[str]) -> list[str]:
    """Documents in a seed-42 random order (a run stopped at the deadline is not biased by length)."""

    import numpy as np

    ids = sorted(set(doc_ids))
    perm = np.random.default_rng(c8.SEED).permutation(len(ids))
    return [ids[int(i)] for i in perm]


def server_env() -> dict[str, Any]:
    e = os.environ
    return {"mode": e.get("C8_SERVED_MODE", "dca"), "max_model_len": int(e.get("C8_SERVED_MAX_MODEL_LEN", "0") or 0),
            "server_log": e.get("C8_SERVER_LOG", ""), "model": e.get("C8_MODEL", "Qwen/Qwen2.5-7B-Instruct-1M"),
            "max_out": int(e.get("C8_MAX_OUTPUT_TOKENS", 64)), "timeout": float(e.get("C8_REQUEST_TIMEOUT", 5400)),
            "concurrency": int(e.get("C8_READER_CONCURRENCY", 8)), "deadline": float(e.get("C8_DEADLINE_EPOCH", "0") or 0)}


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
