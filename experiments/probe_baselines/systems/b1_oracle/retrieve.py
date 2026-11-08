#!/usr/bin/env python3
"""B1 Oracle: Vanilla BM25 over the oracle copy of each book (form 2 in every revealed persona's gold chunk)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common"))
import chunk_map as CM  # noqa: E402
import load_probe as LP  # noqa: E402

SYSTEM = "b1_oracle"


def main() -> int:
    rows_all = []
    for book_id in CM.book_ids():
        inp = CM.load_inputs(book_id)
        oracle = json.loads((CM.system_dir(SYSTEM) / "books" / f"{book_id}.json").read_text(encoding="utf-8"))
        assert oracle["chunk_ids"] == inp["chunk_ids"]
        rows = []
        for p in inp["personas"]:
            ranked = LP.bm25_ranking(oracle["chunks"], oracle["chunk_ids"], p["query"])
            rows.append(CM.ranking_row(p | {"book_id": book_id}, ranked, n_chunks=len(inp["chunks"]), link_formed=int(p["cell"]) != 5))
        CM.write_rankings(SYSTEM, book_id, rows)
        rows_all += rows
    CM.write_numbers(SYSTEM, "retrieve", {"personas": len(rows_all)})
    (CM.system_dir(SYSTEM) / "files_retrieve.txt").write_text(str(CM.RANKINGS_DIR / SYSTEM) + "/<book>.jsonl\n", encoding="utf-8")
    print(f"B1: {len(rows_all)} personas ranked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
