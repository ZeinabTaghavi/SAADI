#!/usr/bin/env python3
"""B1 Oracle: a copy of each book where every persona's gold chunk also carries form 2 (the query name).

This is the perfect link: what any bridging method could reach if form 2 were attached to
the gold chunk. Form 2 is appended, after a space, to the gold chunk's text for every persona
whose form 2 is revealed in the book (cells 1-4 and 6-8). Cell 5 has no reveal: there is no
link to make, so its gold chunks stay unchanged and the control keeps its meaning (a perfect
link adds nothing where the book never connects the two names). No LLM.
Output: outputs/probe_baselines/b1_oracle/books/<book_id>.json (the oracle chunk texts).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common"))
import chunk_map as CM  # noqa: E402

SYSTEM = "b1_oracle"
NO_REVEAL_CELLS = (5,)


def main() -> int:
    out_dir = CM.system_dir(SYSTEM) / "books"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_book, appended = {}, 0
    for book_id in CM.book_ids():
        inp = CM.load_inputs(book_id)
        chunks = list(inp["chunks"])
        index = {cid: i for i, cid in enumerate(inp["chunk_ids"])}
        n = 0
        for p in inp["personas"]:
            if int(p["cell"]) in NO_REVEAL_CELLS:
                continue
            i = index[p["gold_chunk_id"]]
            chunks[i] = chunks[i] + " " + p["form2"]
            n += 1
        appended += n
        (out_dir / f"{book_id}.json").write_text(
            json.dumps({"book_id": book_id, "chunk_ids": inp["chunk_ids"], "chunks": chunks}, ensure_ascii=False), encoding="utf-8"
        )
        per_book[book_id] = {"gold_chunks_with_form2": n, "index_chars": sum(len(c) for c in chunks)}
    CM.write_numbers(
        SYSTEM,
        "build",
        {
            "per_book": per_book,
            "appended": appended,
            "rule": "form 2 appended to the gold chunk of every persona with a reveal (cells 1-4, 6-8); cell 5 unchanged",
        },
    )
    (CM.system_dir(SYSTEM) / "files_build.txt").write_text(str(out_dir) + "/<book>.json\n", encoding="utf-8")
    print(f"B1 build: form 2 appended to {appended} gold chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
