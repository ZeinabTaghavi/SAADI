#!/usr/bin/env python3
"""B3 HippoRAG v2: index every book with the official package (hipporag 2.0.0a4), one index per book.

Runs in third_party/venv_hipporag (the package pins torch 2.5.1, transformers 4.45.2 and
vllm 0.6.6, which clash with the study venv). Package defaults throughout (decided with the
user; the paper's Appendix E was not available): OpenIE (NER, then triples) with the
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


def make_hipporag(book_id: str):
    from hipporag import HippoRAG

    return HippoRAG(save_dir=str(save_dir(book_id)), llm_model_name=MODEL, llm_base_url=URL, embedding_model_name=EMBEDDING)


def main() -> int:
    import hipporag
    import torch

    versions = {"hipporag": getattr(hipporag, "__version__", "2.0.0a4"), "torch": torch.__version__}
    try:
        import transformers

        versions["transformers"] = transformers.__version__
    except Exception:
        pass
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
            f"{v['index_bytes'] / 1e6:.0f} MB",
            v["seconds"],
        ]
        for b, v in per_book.items()
    ]
    frag = f"### B3 HippoRAG v2 indexing per book (hipporag {versions.get('hipporag')}, {EMBEDDING}, LLM {MODEL})\n\n" + RPT.md_table(
        ["book", "chunks", "triples", "phrase nodes", "LLM calls", "input tokens", "generated tokens", "index size", "seconds"], rows
    )
    (CM.system_dir(SYSTEM) / "report_build.md").write_text(frag + "\n", encoding="utf-8")
    (CM.system_dir(SYSTEM) / "files_build.txt").write_text(
        str(CM.system_dir(SYSTEM)) + "/<book>/ (HippoRAG save_dir: OpenIE results, embeddings, graph, LLM cache)\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
