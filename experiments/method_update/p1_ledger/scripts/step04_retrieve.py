#!/usr/bin/env python3
"""Step 4 (CPU): rank all chunks of each book for every persona's query, four ways, next to run 0.

Vanilla BM25 (text channel alone), SAADI live, SAADI frozen (E2 construction,
with the ledger's stub annotations) and SAADI without memory (lambda_n = 0).
HR@5, HR@10 and the median gold rank per cell with 95% bootstrap intervals
(10,000 resamples, seed 42), the paired live minus frozen difference with its
interval and an exact McNemar test. Checks: cell 5 stays at zero; cell 4 is not
below cells 1-3. Published weights (1.0, 0.5, 0.3), nothing tuned.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import report_utils as RU  # noqa: E402
from common import CELL_LABELS, LEDGER_DIR, OUT_DIR, PUBLISHED_SETTINGS, StepReport, read_jsonl, run0_file, run_step, write_jsonl  # noqa: E402

MEMORY_JSONL = OUT_DIR / "personas_memory.jsonl"
PER_PERSONA = OUT_DIR / "per_persona.jsonl"
METHODS = RU.METHODS


def main(report: StepReport, argv: list[str]) -> None:
    from saadi.indexing.retrieval_methods.mbr import (
        MBR_VARIANT_MBR2,
        _find_doc_entry,
        _load_doc_records,
        _read_index_payload,
        build_mbr_document_memory_index,
        score_mbr_variant_document_pools,
    )
    from saadi.ledger.ledger import LedgerStore
    from saadi.ledger.measure import apply_description_cap, ledger_frozen_chunk_fields
    from saadi.rebuttal.frozen_records import _frozen_streams, replay_first_added, score_query_variants

    personas = read_jsonl(MEMORY_JSONL)
    index_path = common.find_run1_cache_index()
    payload = _read_index_payload(index_path)
    store = LedgerStore(LEDGER_DIR)
    report.commands.append("score_mbr_variant_document_pools + combine (published settings) per persona; frozen via saadi.rebuttal.frozen_records with saadi.ledger.measure annotations; vanilla = the text channel alone")
    run0 = {r["persona_id"]: r for r in read_jsonl(run0_file("per_persona.jsonl"))} if run0_file("per_persona.jsonl") else {}
    if not run0:
        report.unexpected.append("Run 0's per_persona.jsonl was not found under outputs/probe_bridge; run 0 columns show n/a.")

    by_book: dict[str, list[dict]] = defaultdict(list)
    for p in personas:
        by_book[p["book_id"]].append(p)
    stub_rows = []
    for book, group in sorted(by_book.items()):
        doc_entry = _find_doc_entry(payload, doc_id=book)
        records = _load_doc_records(index_path=index_path, index_payload=payload, doc_entry=doc_entry, doc_id=book)
        replay = replay_first_added(book, records)
        document_index = build_mbr_document_memory_index(index_path, doc_id=book)
        ledger = store.doc(book)
        apply_description_cap(document_index, ledger)
        frozen, notes = ledger_frozen_chunk_fields(document_index, replay, ledger)
        n_chunks = len(document_index.chunk_pool)
        chunk_order = [c.chunk_id for c in document_index.chunk_pool]
        position = {cid: i for i, cid in enumerate(chunk_order)}
        chunks_with_stub = sum(1 for cid in chunk_order if any(n.stub_at_chunk_step for n in notes.get(cid, [])))
        stub_rows.append([book, n_chunks, chunks_with_stub, sum(len(frozen[c].linked_names) for c in chunk_order) / max(1, n_chunks)])
        for p in group:
            gold = p["gold_chunk_id"]
            rankings = score_query_variants(document_index, frozen, query=p["query"], retriever_name="bm25", retrieve_k=n_chunks, settings=PUBLISHED_SETTINGS)
            streams = score_mbr_variant_document_pools(
                document_index, query=p["query"], retriever_name="bm25", variant=MBR_VARIANT_MBR2,
                memory_name_weight=1.0, memory_alias_weight=1.0, memory_description_weight=1.0, memory_aggregation="max",
            )
            vanilla = sorted(chunk_order, key=lambda cid: (-float(streams.text_scores.get(cid, 0.0)), position[cid]))
            ranked = {"vanilla": vanilla, "live": rankings["live"], "frozen": rankings["frozen"], "no_memory": rankings["no_memory"]}
            p["n_chunks"] = n_chunks
            for method, ids in ranked.items():
                rank = ids.index(gold) + 1 if gold in ids else n_chunks + 1
                p[f"rank_{method}"] = rank
                p[f"hit5_{method}"] = rank <= 5
                p[f"hit10_{method}"] = rank <= 10
                p[f"top3_{method}"] = ids[:3]
            mem_live = common.probe.memory_channel_ranks(streams, document_index)
            mem_frozen = common.probe.memory_channel_ranks(_frozen_streams(streams, frozen, memory_aggregation="max"), document_index)
            p["memory_rank_live"] = mem_live.get(gold)
            p["memory_rank_frozen"] = mem_frozen.get(gold)
            p["gold_linked_stubs_at_step"] = [n.name for n in notes.get(gold, []) if n.stub_at_chunk_step]
            r0 = run0.get(p["persona_id"], {})
            p.setdefault("run0", {}).update({f"rank_{m}": r0.get(f"rank_{m}") for m in METHODS} | {f"hit10_{m}": r0.get(f"hit10_{m}") for m in METHODS} | {f"hit5_{m}": r0.get(f"hit5_{m}") for m in METHODS})
    write_jsonl(PER_PERSONA, personas)
    report.files_written.append(str(PER_PERSONA))
    report.table("Stubs in the frozen view", ["book", "chunks", "chunks linked to at least one stub at their own step", "linked names per chunk (mean)"], [[b, n, s, f"{m:.2f}"] for b, n, s, m in stub_rows])

    cells = sorted({p["cell"] for p in personas})

    def r0_group(cell):
        return [p["run0"] for p in personas if p["cell"] == cell and p.get("run0", {}).get("rank_live") is not None]

    rows = []
    for cell in cells:
        g = [p for p in personas if p["cell"] == cell]
        g0 = r0_group(cell)
        cols = []
        for m in METHODS:
            r1 = f"{RU.hr(g, m, '10'):.0f} / {RU.median_rank(g, f'rank_{m}'):.0f}"
            r0t = f"{RU.hr(g0, m, '10'):.0f} / {RU.median_rank(g0, f'rank_{m}'):.0f}" if g0 else None
            cols.append(RU.side_by_side(r1, r0t))
        rows.append([f"{cell} ({CELL_LABELS.get(cell, '')})", len(g), *cols, RU.paired_text(g, "live", "frozen"), RU.paired_text(g, "live", "vanilla")])
    report.table("HR@10 (%) / median gold rank per cell, run 1 (run 0 in parentheses)", ["cell", "n", *[RU.METHOD_LABELS[m] for m in METHODS], "Live - Frozen @10", "Live - Vanilla @10"], rows)
    rows5 = []
    for cell in cells:
        g = [p for p in personas if p["cell"] == cell]
        g0 = r0_group(cell)
        rows5.append([cell, len(g), *[RU.side_by_side(f"{RU.hr(g, m, '5'):.0f}", f"{RU.hr(g0, m, '5'):.0f}" if g0 else None) for m in METHODS], f"{RU.median_rank(g, 'memory_rank_live'):.0f}", f"{RU.median_rank(g, 'memory_rank_frozen'):.0f}"])
    report.table("HR@5 (%) and the memory-channel-only median rank of the gold chunk", ["cell", "n", *[RU.METHOD_LABELS[m] for m in METHODS], "mem-rank live", "mem-rank frozen"], rows5)
    for k in ("5", "10"):
        report.table(f"HR@{k} (%) with 95% bootstrap interval over personas (10,000 resamples, seed 42)", ["cell", "n", *METHODS], [[c, len(g), *[RU.hr_with_ci(g, m, k) for m in METHODS]] for c in cells for g in [[p for p in personas if p["cell"] == c]]])
    report.table("Share of personas with the gold chunk at rank 1 (%)", ["cell", "n", *METHODS], [[c, len(g), *[f"{100.0 * sum(p[f'rank_{m}'] == 1 for p in g) / len(g):.0f}" for m in METHODS]] for c in cells for g in [[p for p in personas if p["cell"] == c]]])

    bridged_rows = []
    for cell in cells:
        g = [p for p in personas if p["cell"] == cell and p.get("form2_in_that_record_final") and p.get("gold_chunk_linked_to_that_record")]
        if not g:
            bridged_rows.append([cell, 0, "-", "-", "-", "-", "-"])
            continue
        bridged_rows.append([cell, len(g), *[f"{RU.hr(g, m, '10'):.0f}" for m in METHODS], RU.paired_text(g, "live", "frozen")])
    report.table("Bridge-reached personas only: HR@10 (%)", ["cell", "n", *[RU.METHOD_LABELS[m] for m in METHODS], "Live - Frozen @10"], bridged_rows)

    # Checks.
    c5 = [p for p in personas if p["cell"] == 5]
    c4 = [p for p in personas if p["cell"] == 4]
    c123 = [p for p in personas if p["cell"] in (1, 2, 3)]
    cell5_zero = all(not p["hit10_live"] for p in c5)
    cell4_ok = RU.hr(c4, "live", "10") >= min(RU.hr([p for p in personas if p["cell"] == c], "live", "10") for c in (1, 2, 3))
    live_frozen_123 = RU.paired_numbers(c123, "live", "frozen")
    report.table(
        "Checks",
        ["check", "result"],
        [
            ["cell 5 (no reveal) stays at zero: live HR@10 = 0 and HR@5 = 0", f"{'holds' if cell5_zero and all(not p['hit5_live'] for p in c5) else 'DOES NOT HOLD'} (live HR@10 {RU.hr(c5, 'live', '10'):.0f}%, HR@5 {RU.hr(c5, 'live', '5'):.0f}%)"],
            ["cell 4 (reveal before gold) not below cells 1-3 at live HR@10", f"{'holds' if cell4_ok else 'DOES NOT HOLD'} (cell 4 {RU.hr(c4, 'live', '10'):.0f}%, cells 1-3 {[round(RU.hr([p for p in personas if p['cell'] == c], 'live', '10')) for c in (1, 2, 3)]})"],
            ["cells 1-3 pooled: live minus frozen @10", f"{live_frozen_123['mean_pp']:+.1f} pp [{live_frozen_123['ci_lo']:+.1f}, {live_frozen_123['ci_hi']:+.1f}], McNemar p={live_frozen_123['mcnemar_p']:.3f}"],
            ["cells 1-3 pooled: live HR@10 (run 0)", f"{RU.hr(c123, 'live', '10'):.0f}% ({RU.hr([p['run0'] for p in c123 if p['run0'].get('rank_live') is not None], 'live', '10'):.0f}%)" if run0 else f"{RU.hr(c123, 'live', '10'):.0f}%"],
        ],
    )
    report.numbers.update(
        {
            "cells": cells,
            "cell5_zero": cell5_zero,
            "cell4_not_below_cells_1_3": cell4_ok,
            "live_minus_frozen_cells_1_3": live_frozen_123,
            "per_cell": {
                str(c): {
                    "n": len(g),
                    **{f"hr10_{m}": RU.hr(g, m, "10") for m in METHODS},
                    **{f"hr5_{m}": RU.hr(g, m, "5") for m in METHODS},
                    **{f"median_{m}": RU.median_rank(g, f"rank_{m}") for m in METHODS},
                    "live_minus_frozen": RU.paired_numbers(g, "live", "frozen"),
                    "run0": {**{f"hr10_{m}": RU.hr(g0, m, "10") for m in METHODS}, **{f"median_{m}": RU.median_rank(g0, f"rank_{m}") for m in METHODS}} if (g0 := r0_group(c)) else None,
                }
                for c in cells
                for g in [[p for p in personas if p["cell"] == c]]
            },
        }
    )
    report.add("Vanilla leak check", "A cell where Vanilla succeeds means a query word reached the gold chunk through the text; expected to fail everywhere (HR near 0), as in run 0.")
    report.add("Hand-off", "Paste this report back. Step 5 writes the final tables and the Google Doc text.")


if __name__ == "__main__":
    raise SystemExit(run_step(4, "Retrieve and score", main, requires=[2, 3]))
