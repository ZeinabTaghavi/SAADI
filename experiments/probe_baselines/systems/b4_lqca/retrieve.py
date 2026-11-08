#!/usr/bin/env python3
"""B4 LQCA: BM25 over the coreference-rewritten chunks, results returned as the original chunk ids.

Link formation: the rewritten gold chunk contains form 2 (case-insensitive string match).
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common"))
import chunk_map as CM  # noqa: E402
import load_probe as LP  # noqa: E402
import report_utils as RPT  # noqa: E402

SYSTEM = "b4_lqca"


def main() -> int:
    rows_all = []
    for book_id in CM.book_ids():
        inp = CM.load_inputs(book_id)
        rewritten = {}
        for r in CM.read_jsonl(CM.system_dir(SYSTEM) / book_id / "segments.jsonl"):
            rewritten.update(dict(zip(r["chunk_ids"], r["rewritten"], strict=True)))
        missing = [cid for cid in inp["chunk_ids"] if cid not in rewritten]
        if missing:
            raise RuntimeError(f"{book_id}: {len(missing)} chunks were not rewritten (build.py did not finish)")
        texts = [rewritten[cid] for cid in inp["chunk_ids"]]
        original = dict(zip(inp["chunk_ids"], inp["chunks"], strict=True))
        rows = []
        for p in inp["personas"]:
            gold = rewritten[p["gold_chunk_id"]]
            rows.append(
                CM.ranking_row(
                    p | {"book_id": book_id},
                    LP.bm25_ranking(texts, inp["chunk_ids"], p["query"]),
                    n_chunks=len(texts),
                    link_formed=str(p["form2"]).casefold() in gold.casefold(),
                    gold_chunk_changed=gold != original[p["gold_chunk_id"]],
                )
            )
        CM.write_rankings(SYSTEM, book_id, rows)
        rows_all += rows
    cells: dict[int, list[dict]] = {}
    for r in rows_all:
        cells.setdefault(r["cell"], []).append(r)
    frag = "### B4: the rewritten gold chunk\n\n" + RPT.md_table(
        ["cell", "n", "gold chunk rewritten at all", "rewritten gold chunk holds form 2"],
        [[c, len(g), sum(r["gold_chunk_changed"] for r in g), sum(r["link_formed"] for r in g)] for c, g in sorted(cells.items())],
    )
    (CM.system_dir(SYSTEM) / "report_retrieve.md").write_text(frag + "\n", encoding="utf-8")
    CM.write_numbers(SYSTEM, "retrieve", {"personas": len(rows_all)})
    (CM.system_dir(SYSTEM) / "files_retrieve.txt").write_text(str(CM.RANKINGS_DIR / SYSTEM) + "/<book>.jsonl\n", encoding="utf-8")
    print(f"B4: {len(rows_all)} personas ranked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
