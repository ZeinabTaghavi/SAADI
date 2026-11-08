#!/usr/bin/env python3
"""B3 HippoRAG v2: retrieve for every persona with the official package, map passages to chunk ids.

hipporag.retrieve(queries, num_to_retrieve=<all chunks of the book>): fact retrieval,
recognition-memory filtering with the LLM, PPR over the graph (or the package's own dense
fallback when no fact survives). Asking for every chunk only sets the output depth (the
package default 200 would truncate the larger books); the ranking itself is the package's.
Returned passages are mapped back to the paper's chunk ids by their text.

Link formation: whether the extracted triples of the book hold a triple with form 1 in one
element and form 2 in another (case-insensitive string match), i.e. an edge that connects the
two names in HippoRAG's graph.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common"))
sys.path.insert(0, str(HERE))
import chunk_map as CM  # noqa: E402
import report_utils as RPT  # noqa: E402

import build as B  # noqa: E402

SYSTEM = "b3_hipporag"


def triples_of(openie_file: Path) -> list[list[str]]:
    data = json.loads(Path(openie_file).read_text(encoding="utf-8"))
    out = []
    for doc in data.get("docs", []):
        for t in doc.get("extracted_triples", []) or []:
            if isinstance(t, (list, tuple)):
                out.append([str(x) for x in t])
    return out


def links(triples: list[list[str]], form1: str, form2: str) -> list[list[str]]:
    f1, f2 = form1.casefold(), form2.casefold()
    hits = []
    for t in triples:
        low = [x.casefold() for x in t]
        if any(f1 in a and f2 in b for i, a in enumerate(low) for j, b in enumerate(low) if i != j):
            hits.append(t)
    return hits


def main() -> int:
    import torch

    rows_all, per_book, unexpected = [], {}, []
    for book_id in CM.book_ids():
        inp = CM.load_inputs(book_id)
        before = CM.vllm_counters(B.URL)
        t0 = time.time()
        h = B.make_hipporag(book_id)
        queries = [p["query"] for p in inp["personas"]]
        solutions = h.retrieve(queries=queries, num_to_retrieve=len(inp["chunks"]))
        delta = CM.counter_delta(before, CM.vllm_counters(B.URL))
        cmap = CM.ChunkMap(inp["chunk_ids"], inp["chunks"])
        triples = triples_of(Path(h.openie_results_path))
        n = len(inp["personas"])
        calls, tin, tout = (
            int(delta.get("request_success_total", 0)),
            int(delta.get("prompt_tokens_total", 0)),
            int(delta.get("generation_tokens_total", 0)),
        )
        rows = []
        for p, sol in zip(inp["personas"], solutions, strict=True):
            ranked = cmap.map_ranked_texts(list(sol.docs))
            hit = links(triples, p["form1"], p["form2"])
            rows.append(
                CM.ranking_row(
                    p | {"book_id": book_id},
                    ranked,
                    n_chunks=len(inp["chunks"]),
                    model_calls=round(calls / n),
                    input_tokens=round(tin / n),
                    generated_tokens=round(tout / n),
                    link_formed=bool(hit),
                    linking_triples=hit[:3],
                    query_cost_basis="book total over the book's queries, divided evenly",
                )
            )
        if cmap.unmapped:
            unexpected.append(f"{book_id}: {len(cmap.unmapped)} returned passages did not map to a chunk id")
        CM.write_rankings(SYSTEM, book_id, rows)
        rows_all += rows
        per_book[book_id] = {
            "queries": n,
            "seconds": round(time.time() - t0, 1),
            "llm_calls": calls,
            "input_tokens": tin,
            "generated_tokens": tout,
            "triples": len(triples),
            "unmapped": len(cmap.unmapped),
        }
        print(f"B3 {book_id}: {n} queries in {per_book[book_id]['seconds']:.0f} s", flush=True)
        del h
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    cells: dict[int, list[dict]] = {}
    for r in rows_all:
        cells.setdefault(r["cell"], []).append(r)
    frag = "### B3: a triple connecting form 1 and form 2 in the book's graph (string match)\n\n" + RPT.md_table(
        ["cell", "n", "linking triple found"], [[c, len(g), sum(r["link_formed"] for r in g)] for c, g in sorted(cells.items())]
    )
    frag += "\n\n### B3 query-time cost per book\n\n" + RPT.md_table(
        ["book", "queries", "LLM calls", "input tokens", "generated tokens", "seconds", "unmapped passages"],
        [
            [b, v["queries"], v["llm_calls"], v["input_tokens"], v["generated_tokens"], v["seconds"], v["unmapped"]]
            for b, v in per_book.items()
        ],
    )
    (CM.system_dir(SYSTEM) / "report_retrieve.md").write_text(frag + "\n", encoding="utf-8")
    CM.write_numbers(SYSTEM, "retrieve", {"per_book": per_book, "unexpected": unexpected})
    (CM.system_dir(SYSTEM) / "files_retrieve.txt").write_text(str(CM.RANKINGS_DIR / SYSTEM) + "/<book>.jsonl\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
