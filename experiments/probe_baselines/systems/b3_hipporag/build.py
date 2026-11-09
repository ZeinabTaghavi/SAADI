#!/usr/bin/env python3
"""B3 HippoRAG v2: index every book with the official package (hipporag 2.0.0a4), one index per book.

Runs in third_party/venv_hipporag, installed the way the user's SAADI_Hipporag repository
does (its tested set: torch 2.5.1, transformers 4.45.2, vllm 0.6.6.post1, openai 1.58.1,
litellm 1.55.8; hipporag with --no-deps, since its own metadata pins openai==1.91.1, which is
not on PyPI). SAADI_Hipporag's two parser fixes are applied as its runner applies them (see
make_hipporag). Otherwise package defaults (decided with the user; the paper's Appendix E
was not available): OpenIE (NER, then triples) with the
construction LLM through the local vLLM server's OpenAI-compatible endpoint, NV-Embed-v2
embeddings, its graph (fact, passage and synonymy edges). The units given to HippoRAG are
the probe's chunks, unchanged, so passages map back to chunk ids by their text.

Resumable per book: HippoRAG caches every LLM reply (llm_cache/*.sqlite) and its OpenIE
results in the book's save_dir, and a finished book leaves index_done.json.

Environment: PBL_VLLM_URL, PBL_MODEL (served model name; also HippoRAG's llm_name).
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common"))
import chunk_map as CM  # noqa: E402
import report_utils as RPT  # noqa: E402

SYSTEM = "b3_hipporag"
URL = os.environ.get("PBL_VLLM_URL", "http://127.0.0.1:8047/v1").rstrip("/")
MODEL = os.environ.get("PBL_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
EMBEDDING = os.environ.get("PBL_HIPPORAG_EMBEDDING", "nvidia/NV-Embed-v2")  # the package default
os.environ.setdefault("OPENAI_API_KEY", "EMPTY")  # the local server needs no key; the client wants one


def save_dir(book_id: str) -> Path:
    return CM.system_dir(SYSTEM) / book_id


def dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


SHR_SRC = Path(os.environ.get("PBL_SHR_DIR", HERE.parents[1] / "third_party" / "SAADI_Hipporag")) / "src"
# SAADI_Hipporag's own settings for the adapter (its configs/novelhopqa.json).
OPENIE_PARSE_RETRIES, OPENIE_RETRY_EMPTY, OPENIE_WORKERS = 1, True, 16


class StructuredOutputsLLM:
    """The package's LLM client, with one request field renamed for this server.

    SAADI_Hipporag's adapter asks for schema-constrained rescue replies with vLLM 0.6.6's
    `guided_json` (and `guided_decoding_backend`). The construction server here is vLLM 0.23,
    which accepts only `structured_outputs` and ignores the old fields without an error, so
    the rescue would not be constrained. Only that field is rewritten; everything else is
    passed through to the package's client unchanged.
    """

    def __init__(self, llm):
        self._llm = llm
        self.structured_requests = 0

    def infer(self, messages, **kwargs):
        extra = dict(kwargs.get("extra_body") or {})
        if "guided_json" in extra:
            schema = extra.pop("guided_json")
            extra.pop("guided_decoding_backend", None)
            extra["structured_outputs"] = {"json": schema}
            kwargs["extra_body"] = extra
            self.structured_requests += 1
        return self._llm.infer(messages=messages, **kwargs)

    def __getattr__(self, name):
        return getattr(self._llm, name)


def make_hipporag(book_id: str):
    """The package's HippoRAG with SAADI_Hipporag's two parser fixes, as its runner applies them.

    1. OpenIE: RobustOpenIE (saadi_hypporag/openie_compat.py) replaces the package's online
       OpenIE object. Same prompts and flow; it parses Qwen's list/typed-object variants
       safely (no eval), retries one malformed or empty reply, then up to two
       schema-constrained rescue requests, and keeps a per-chunk audit.
    2. Fact filter: the recognition-memory filter's reply goes through the same parser.
    """

    from hipporag import HippoRAG

    if str(SHR_SRC) not in sys.path:
        sys.path.insert(0, str(SHR_SRC))
    from saadi_hypporag.openie_compat import RobustOpenIE, parse_triples

    h = HippoRAG(save_dir=str(save_dir(book_id)), llm_model_name=MODEL, llm_base_url=URL, embedding_model_name=EMBEDDING)
    h.openie = RobustOpenIE(
        StructuredOutputsLLM(h.llm_model),
        max_parse_retries=OPENIE_PARSE_RETRIES,
        retry_empty=OPENIE_RETRY_EMPTY,
        max_workers=OPENIE_WORKERS,
    )
    h.rerank_filter.parse_filter = lambda response: list(parse_triples(str(response or "")).values)
    return h


def openie_audit(h, book_id: str) -> dict:
    """Write RobustOpenIE's per-chunk audit and count what SAADI_Hipporag refuses to report."""

    rows = h.openie.audit_rows()
    with (save_dir(book_id) / "openie_audit.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {
        "audited_calls": len(rows),
        "parse_failed": sum(r["final_status"] == "failed" for r in rows),
        "empty": sum(r["final_count"] == 0 for r in rows),
        "retried": sum(len(r["attempts"]) > 1 for r in rows),
        "structured_rescue": sum(bool(r["guided_rescue_used"]) for r in rows),
    }


def main() -> int:
    from importlib.metadata import version

    import hipporag
    import torch

    versions = {"hipporag": getattr(hipporag, "__version__", "2.0.0a4"), "torch": torch.__version__}
    for name in ("transformers", "openai", "litellm", "vllm"):
        try:
            versions[name] = version(name)
        except Exception:
            pass
    versions["SAADI_Hipporag"] = os.environ.get("PBL_SHR_COMMIT", "?")
    per_book, unexpected = {}, []
    for book_id in CM.book_ids():
        done_file = save_dir(book_id) / "index_done.json"
        if done_file.exists():
            per_book[book_id] = json.loads(done_file.read_text(encoding="utf-8"))
            print(f"B3 {book_id}: indexed earlier; skipped", flush=True)
            continue
        inp = CM.load_inputs(book_id)
        save_dir(book_id).mkdir(parents=True, exist_ok=True)
        before = CM.vllm_counters(URL)
        t0 = time.time()
        h = make_hipporag(book_id)
        h.index(docs=list(inp["chunks"]))
        info = h.get_graph_info()
        audit = openie_audit(h, book_id)
        if audit["parse_failed"]:
            unexpected.append(f"{book_id}: {audit['parse_failed']} OpenIE replies still unparsed after the retries (openie_audit.jsonl)")
        delta = CM.counter_delta(before, CM.vllm_counters(URL))
        row = {
            "chunks": len(inp["chunks"]),
            "seconds": round(time.time() - t0, 1),
            "llm_calls": int(delta.get("request_success_total", 0)),
            "input_tokens": int(delta.get("prompt_tokens_total", 0)),
            "generated_tokens": int(delta.get("generation_tokens_total", 0)),
            "graph": {k: (int(v) if isinstance(v, (int, float)) else v) for k, v in info.items()},
            "index_bytes": dir_bytes(save_dir(book_id)),
            "openie_file": str(h.openie_results_path),
            "openie_audit": audit,
            "structured_requests": h.openie.llm_model.structured_requests,
        }
        done_file.write_text(json.dumps(row, indent=1), encoding="utf-8")
        per_book[book_id] = row
        print(f"B3 {book_id}: indexed in {row['seconds']:.0f} s, {row['llm_calls']} LLM calls, graph {row['graph']}", flush=True)
        del h
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        CM.write_numbers(
            SYSTEM, "build", {"model": MODEL, "embedding": EMBEDDING, "versions": versions, "per_book": per_book, "unexpected": unexpected}
        )
    CM.write_numbers(
        SYSTEM, "build", {"model": MODEL, "embedding": EMBEDDING, "versions": versions, "per_book": per_book, "unexpected": unexpected}
    )
    rows = [
        [
            b,
            v["chunks"],
            v.get("graph", {}).get("num_extracted_triples", "?"),
            v.get("graph", {}).get("num_phrase_nodes", "?"),
            v["llm_calls"],
            v["input_tokens"],
            v["generated_tokens"],
            v.get("openie_audit", {}).get("retried", "?"),
            v.get("openie_audit", {}).get("structured_rescue", "?"),
            v.get("openie_audit", {}).get("parse_failed", "?"),
            f"{v['index_bytes'] / 1e6:.0f} MB",
            v["seconds"],
        ]
        for b, v in per_book.items()
    ]
    frag = f"### B3 HippoRAG v2 indexing per book (hipporag {versions.get('hipporag')}, {EMBEDDING}, LLM {MODEL})\n\n" + RPT.md_table(
        [
            "book",
            "chunks",
            "triples",
            "phrase nodes",
            "LLM calls",
            "input tokens",
            "generated tokens",
            "OpenIE replies retried",
            "structured rescue",
            "still unparsed",
            "index size",
            "seconds",
        ],
        rows,
    )
    (CM.system_dir(SYSTEM) / "report_build.md").write_text(frag + "\n", encoding="utf-8")
    (CM.system_dir(SYSTEM) / "files_build.txt").write_text(
        str(CM.system_dir(SYSTEM)) + "/<book>/ (HippoRAG save_dir: OpenIE results, embeddings, graph, LLM cache)\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
