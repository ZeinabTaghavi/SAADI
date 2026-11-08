#!/usr/bin/env python3
"""B0 Vanilla BM25: rank every chunk of each book by the persona's query with the probe's own BM25 path.

The very function the probe's step 7 uses for its Vanilla column (MBR's text pool over
the raw chunk texts, retriever "bm25", ties broken by chunk position). As a check it
compares every persona's gold rank with the probe's own Vanilla rank (run 0's
per_persona.jsonl) when that file exists, and per cell with results/probe_bridge/table.csv.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common"))
import chunk_map as CM  # noqa: E402
import load_probe as LP  # noqa: E402
import report_utils as RPT  # noqa: E402

SYSTEM = "b0_vanilla"


def main() -> int:
    t0 = time.time()
    rows_all, per_book = [], {}
    for book_id in CM.book_ids():
        inp = CM.load_inputs(book_id)
        tb = time.time()
        rows = [
            CM.ranking_row(
                p | {"book_id": book_id}, LP.bm25_ranking(inp["chunks"], inp["chunk_ids"], p["query"]), n_chunks=len(inp["chunks"])
            )
            for p in inp["personas"]
        ]
        CM.write_rankings(SYSTEM, book_id, rows)
        rows_all += rows
        per_book[book_id] = {
            "chunks": len(inp["chunks"]),
            "index_chars": sum(len(c) for c in inp["chunks"]),
            "seconds": round(time.time() - tb, 2),
        }

    # Check against the probe's own Vanilla.
    unexpected, lines = [], []
    run0 = LP.ROOT / "outputs" / "probe_bridge" / "per_persona.jsonl"
    if run0.exists():
        ref = {p["persona_id"]: int(p["rank_vanilla"]) for p in CM.read_jsonl(run0)}
        diff = [r["persona_id"] for r in rows_all if ref.get(r["persona_id"]) != r["gold_rank"]]
        lines.append(
            f"Per persona against run 0's Vanilla rank ({run0.relative_to(LP.ROOT)}): {len(rows_all) - len(diff)} of {len(rows_all)} identical."
        )
        if diff:
            unexpected.append(f"B0 differs from the probe's Vanilla for {len(diff)} personas (first: {diff[:5]}).")
    table = LP.ROOT / "results" / "probe_bridge" / "table.csv"
    if table.exists():
        exp = {}
        with table.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["method"] == "vanilla":
                    exp[int(r["cell"])] = r
        import scorer as SC  # noqa: E402

        trows = []
        for cell, g in sorted(SC.by_cell(rows_all).items()):
            m = SC.cell_metrics(g)
            e = exp.get(cell, {})
            same = (
                e
                and abs(m["hr10"]["hr"] - float(e["hr10"])) < 0.06
                and abs(m["hr5"]["hr"] - float(e["hr5"])) < 0.06
                and abs(m["median_rank"] - float(e["median_rank"])) < 1e-6
            )
            trows.append(
                [
                    cell,
                    f"{m['hr5']['hr']:.1f} / {e.get('hr5', '?')}",
                    f"{m['hr10']['hr']:.1f} / {e.get('hr10', '?')}",
                    f"{m['median_rank']:.1f} / {e.get('median_rank', '?')}",
                    "identical" if same else "DIFFERS",
                ]
            )
            if not same:
                unexpected.append(f"B0 cell {cell} differs from results/probe_bridge/table.csv's Vanilla.")
        lines.append(
            "Per cell against results/probe_bridge/table.csv (B0 / probe Vanilla):\n\n"
            + RPT.md_table(["cell", "HR@5", "HR@10", "median rank", "check"], trows)
        )
    numbers = {"per_book": per_book, "personas": len(rows_all), "seconds": round(time.time() - t0, 1), "unexpected": unexpected}
    CM.write_numbers(SYSTEM, "retrieve", numbers)
    frag = "### B0 Vanilla BM25 against the probe's own Vanilla\n\n" + "\n\n".join(lines or ["No reference file found."])
    frag += "\n\n### Cost per book\n\n" + RPT.md_table(
        ["book", "chunks", "index size (chars)", "seconds"], [[b, v["chunks"], v["index_chars"], v["seconds"]] for b, v in per_book.items()]
    )
    (CM.system_dir(SYSTEM) / "report_retrieve.md").write_text(frag + "\n", encoding="utf-8")
    (CM.system_dir(SYSTEM) / "files_retrieve.txt").write_text(str(CM.RANKINGS_DIR / SYSTEM) + "/<book>.jsonl\n", encoding="utf-8")
    print(f"B0: {len(rows_all)} personas ranked in {time.time() - t0:.0f} s")
    return 1 if unexpected else 0


if __name__ == "__main__":
    raise SystemExit(main())
