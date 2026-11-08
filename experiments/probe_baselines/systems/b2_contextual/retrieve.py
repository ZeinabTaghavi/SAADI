#!/usr/bin/env python3
"""B2 Contextual Retrieval: BM25 over "context + chunk", original chunk ids returned.

Per persona it records whether the reveal was inside the context the model saw when it wrote
the gold chunk's context (reveal_visible; None for cell 5, which has no reveal), and whether
that written context contains form 2 (link_formed, case-insensitive string match) or form 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common"))
import chunk_map as CM  # noqa: E402
import load_probe as LP  # noqa: E402
import report_utils as RPT  # noqa: E402

SYSTEM = "b2_contextual"


def main() -> int:
    ctx_dir = CM.system_dir(SYSTEM) / "contexts"
    rows_all, unexpected = [], []
    for book_id in CM.book_ids():
        inp = CM.load_inputs(book_id)
        ctx = {r["chunk_id"]: r for r in CM.read_jsonl(ctx_dir / f"{book_id}.jsonl")}
        missing = [cid for cid in inp["chunk_ids"] if cid not in ctx]
        if missing:
            raise RuntimeError(f"{book_id}: {len(missing)} chunks have no context yet (build.py did not finish)")
        texts = [ctx[cid]["context"] + "\n\n" + chunk for cid, chunk in zip(inp["chunk_ids"], inp["chunks"], strict=True)]
        rows = []
        for p in inp["personas"]:
            gold = ctx[p["gold_chunk_id"]]
            reveal_idx = p.get("reveal_chunk_index")
            visible = None if reveal_idx is None or int(p["cell"]) == 5 else bool(int(reveal_idx) <= int(gold["visible_last_chunk"]))
            text = gold["context"].casefold()
            ranked = LP.bm25_ranking(texts, inp["chunk_ids"], p["query"])
            rows.append(
                CM.ranking_row(
                    p | {"book_id": book_id},
                    ranked,
                    n_chunks=len(inp["chunks"]),
                    reveal_visible=visible,
                    link_formed=str(p["form2"]).casefold() in text,
                    context_has_form1=str(p["form1"]).casefold() in text,
                    gold_context=gold["context"],
                )
            )
        CM.write_rankings(SYSTEM, book_id, rows)
        rows_all += rows
    import scorer as SC  # noqa: E402

    lines = []
    for cell, g in sorted(SC.by_cell(rows_all).items()):
        vis = [r for r in g if r["reveal_visible"] is True]
        lines.append(
            [
                cell,
                len(g),
                f"{len(vis)}/{sum(1 for r in g if r['reveal_visible'] is not None)}",
                sum(r["link_formed"] for r in g),
                sum(r["context_has_form1"] for r in g),
            ]
        )
    frag = "### B2: reveal visible when the gold chunk's context was written, and the written context\n\n" + RPT.md_table(
        ["cell", "n", "reveal visible", "context holds form 2", "context holds form 1"], lines
    )
    (CM.system_dir(SYSTEM) / "report_retrieve.md").write_text(frag + "\n", encoding="utf-8")
    CM.write_numbers(SYSTEM, "retrieve", {"personas": len(rows_all), "unexpected": unexpected})
    (CM.system_dir(SYSTEM) / "files_retrieve.txt").write_text(str(CM.RANKINGS_DIR / SYSTEM) + "/<book>.jsonl\n", encoding="utf-8")
    print(f"B2: {len(rows_all)} personas ranked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
