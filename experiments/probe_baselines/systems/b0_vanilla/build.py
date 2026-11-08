#!/usr/bin/env python3
"""B0 Vanilla BM25: nothing to build ahead of time.

The repository's BM25 indexes the chunk texts of a book in memory at query time (as the
probe's step 7 does), so the index is the chunk text itself. This script records its size.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common"))
import chunk_map as CM  # noqa: E402

if __name__ == "__main__":
    sizes = {b: sum(len(c) for c in CM.load_inputs(b)["chunks"]) for b in CM.book_ids()}
    CM.write_numbers(
        "b0_vanilla", "build", {"index": "in-memory BM25 over the raw chunk texts (built per query, as in the probe)", "index_chars": sizes}
    )
    print(f"B0 build: nothing to build; {sum(sizes.values())} characters over {len(sizes)} books")
