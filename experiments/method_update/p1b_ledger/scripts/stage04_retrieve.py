#!/usr/bin/env python3
"""Stage 04 (CPU): retrieval per cell, run 1b next to run 1 and run 0, plus the fusion landscape.

First runs run 1's step 4 (p1_ledger/scripts/step04_retrieve.py, imported
unchanged) on run 1b: Vanilla, No memory, Frozen and Live at the published
weights (1.0, 0.5, 0.3), written to outputs/method_update/p1b_ledger/per_persona.jsonl;
its own tables go to reports/step_04_run1_code.md. Then: HR@5, HR@10 and the
median gold rank per cell (1b / 1 / 0); live minus frozen with a 95% bootstrap
interval (10,000 resamples, seed 42) and the exact McNemar test; the controls
(cell 5 at zero, cell 4 not below cells 1-3).

Fusion landscape, for information only (used nowhere else, nothing is tuned):
Live HR@10 per cell at memory weight lambda_n = 0.5 (published), 1.0, 2.0 and
the memory channel alone, for runs 1b, 1 and 0; and for every bridged persona
the smallest lambda_n on a fixed grid at which the gold chunk enters the top 10.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_p1b as C  # noqa: E402

METHODS = C.RU.METHODS  # vanilla, no_memory, frozen, live
LANDSCAPE_WEIGHTS = (0.5, 1.0, 2.0)
GRID = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0)


def landscape(index_path, ledger_dir, personas: list[dict]) -> dict[str, dict]:
    """Per persona: live rank at every grid weight and the memory-channel-only rank."""

    from saadi.indexing.retrieval_methods.mbr import MBR_VARIANT_MBR2, build_mbr_document_memory_index, combine_mbr_variant_chunk_scores, score_mbr_variant_document_pools
    from saadi.ledger.ledger import LedgerStore
    from saadi.ledger.measure import apply_description_cap

    S = C.p1.PUBLISHED_SETTINGS
    store = LedgerStore(ledger_dir) if ledger_dir is not None else None
    by_book: dict[str, list[dict]] = defaultdict(list)
    for p in personas:
        by_book[p["book_id"]].append(p)
    out: dict[str, dict] = {}
    for book, group in sorted(by_book.items()):
        idx = build_mbr_document_memory_index(index_path, doc_id=book)
        if store is not None:
            apply_description_cap(idx, store.doc(book))
        n = len(idx.chunk_pool)
        for p in group:
            gold = p["gold_chunk_id"]
            streams = score_mbr_variant_document_pools(
                idx, query=p["query"], retriever_name="bm25", variant=MBR_VARIANT_MBR2,
                memory_name_weight=S["memory_name_weight"], memory_alias_weight=S["memory_alias_weight"],
                memory_description_weight=S["memory_description_weight"], memory_aggregation=S["memory_aggregation"],
            )
            ranks = {}
            for w in GRID:
                res = combine_mbr_variant_chunk_scores(
                    idx, streams, variant=MBR_VARIANT_MBR2, lambda_t=S["lambda_t"], lambda_n=w, lambda_h=S["lambda_h"],
                    normalization_method=S["normalization_method"], memory_name_weight=S["memory_name_weight"],
                    memory_alias_weight=S["memory_alias_weight"], memory_description_weight=S["memory_description_weight"],
                )
                ids = [r.chunk_id for r in res.ranked_chunks]
                ranks[w] = ids.index(gold) + 1 if gold in ids else n + 1
            mem = C.probe.memory_channel_ranks(streams, idx).get(gold, n + 1)
            out[p["persona_id"]] = {"ranks": ranks, "memory_only": mem, "n": n}
    return out


def smallest_weight(ranks: dict[float, int]) -> str:
    for w in GRID:
        if ranks[w] <= 10:
            return f"{w:g}"
    return f">{GRID[-1]:g}"


def main(report: C.StepReport, argv: list[str]) -> None:
    s4 = C.load_p1_step("step04_retrieve")
    detail = C.DetailReport(step=4, title="Retrieve and score (run 1's step 4 code on run 1b)")
    s4.main(detail, argv)
    detail_path = C.write_detail(detail)
    report.files_written += [str(detail_path), *detail.files_written]

    rows1b = C.read_jsonl(C.OUT_DIR / "per_persona.jsonl")
    for p in rows1b:
        p["bridge"] = bool(p.get("form2_in_that_record_final") and p.get("gold_chunk_linked_to_that_record"))
    run1_path = C.run1_file("per_persona.jsonl")
    rows1 = {p["persona_id"]: p for p in C.read_jsonl(run1_path)} if run1_path else {}
    for p in rows1.values():
        p["bridge"] = bool(p.get("form2_in_that_record_final") and p.get("gold_chunk_linked_to_that_record"))
    if not rows1:
        report.unexpected.append(f"Run 1's per_persona.jsonl was not found under {C.RUN1_OUT_DIR}; run 1 columns show n/a.")
    cells = sorted({p["cell"] for p in rows1b})
    RU = C.RU

    def grp(cs):
        return [p for p in rows1b if p["cell"] in cs]

    def g1(g):
        return [rows1[p["persona_id"]] for p in g if p["persona_id"] in rows1]

    def g0(g):
        return [p["run0"] for p in g if p.get("run0", {}).get("rank_live") is not None]

    def cell_text(g, m, k):
        a = f"{RU.hr(g, m, k):.0f}"
        b = f"{RU.hr(g1(g), m, k):.0f}" if g1(g) else None
        c = f"{RU.hr(g0(g), m, k):.0f}" if g0(g) else None
        return C.three(a, b, c)

    def med_text(g, m):
        a = f"{RU.median_rank(g, f'rank_{m}'):.0f}"
        b = f"{RU.median_rank(g1(g), f'rank_{m}'):.0f}" if g1(g) else None
        c = f"{RU.median_rank(g0(g), f'rank_{m}'):.0f}" if g0(g) else None
        return C.three(a, b, c)

    for k in ("10", "5"):
        report.table(
            f"HR@{k} (%) per cell: run 1b / run 1 / run 0",
            ["cell", "n", *[RU.METHOD_LABELS[m] for m in METHODS]],
            [[f"{c} ({C.CELL_LABELS.get(c, '')})", len(grp((c,))), *[cell_text(grp((c,)), m, k) for m in METHODS]] for c in cells],
        )
    report.table("Median gold rank per cell: run 1b / run 1 / run 0", ["cell", "n", *[RU.METHOD_LABELS[m] for m in METHODS]], [[c, len(grp((c,))), *[med_text(grp((c,)), m) for m in METHODS]] for c in cells])
    paired_rows = []
    for label, cs in (("cells 1-3 pooled", (1, 2, 3)), *[(f"cell {c}", (c,)) for c in cells]):
        g = grp(cs)
        paired_rows.append([label, len(g), RU.paired_text(g, "live", "frozen"), RU.paired_text(g1(g), "live", "frozen") if g1(g) else "n/a"])
    report.table("Live minus Frozen @10: mean difference in pp, 95% bootstrap interval (10,000 resamples, seed 42), exact McNemar (discordant pairs)", ["group", "n", "run 1b", "run 1"], paired_rows)

    c5, c4, c123 = grp((5,)), grp((4,)), grp((1, 2, 3))
    cell5_zero = all(not p["hit10_live"] and not p["hit5_live"] for p in c5)
    min123 = min(RU.hr(grp((c,)), "live", "10") for c in (1, 2, 3))
    cell4_ok = RU.hr(c4, "live", "10") >= min123
    lf = RU.paired_numbers(c123, "live", "frozen")
    lf1 = RU.paired_numbers(g1(c123), "live", "frozen") if g1(c123) else None
    report.table(
        "Controls",
        ["check", "result"],
        [
            ["cell 5 (no reveal) at zero: live HR@10 = HR@5 = 0", f"{'holds' if cell5_zero else 'DOES NOT HOLD'} (live HR@10 {RU.hr(c5, 'live', '10'):.0f}%)"],
            ["cell 4 (reveal before gold) not below cells 1-3 at live HR@10", f"{'holds' if cell4_ok else 'DOES NOT HOLD'} (cell 4 {RU.hr(c4, 'live', '10'):.0f}%, lowest of cells 1-3 {min123:.0f}%)"],
        ],
    )
    bridged = [p for p in rows1b if p["bridge"]]
    report.table(
        "Bridge-reached personas only: live HR@10",
        ["group", "n (1b / 1)", "live HR@10 (1b / 1)"],
        [[label, f"{len([p for p in bridged if p['cell'] in cs])} / {len([p for p in g1(grp(cs)) if p['bridge']])}", f"{RU.hr([p for p in bridged if p['cell'] in cs], 'live', '10') if any(p['cell'] in cs for p in bridged) else 0:.0f} / {RU.hr([p for p in g1(grp(cs)) if p['bridge']], 'live', '10') if any(p['bridge'] for p in g1(grp(cs))) else 0:.0f}"] for label, cs in (("cells 1-3", (1, 2, 3)), ("all cells", tuple(cells)))],
    )

    # ---- fusion landscape (information only) ----
    print("Fusion landscape: run 1b", flush=True)
    land = {"run1b": landscape(C.p1.find_run1_cache_index(), C.LEDGER_DIR, rows1b)}
    idx1 = C.run1_cache_index()
    if idx1 is not None and rows1:
        print("Fusion landscape: run 1", flush=True)
        land["run1"] = landscape(idx1, C.RUN1_LEDGER_DIR if C.RUN1_LEDGER_DIR.exists() else None, list(rows1.values()))
    idx0 = C.run0_cache_index()
    if idx0 is not None:
        print("Fusion landscape: run 0", flush=True)
        land["run0"] = landscape(idx0, None, rows1b)
    mismatch = [p["persona_id"] for p in rows1b if land["run1b"][p["persona_id"]]["ranks"][0.5] != p["rank_live"]]
    if mismatch:
        report.unexpected.append(f"The landscape's rank at lambda_n=0.5 differs from the Live rank for {len(mismatch)} personas (first {mismatch[:3]}); the landscape is not comparable.")

    def hr_land(run: str, ids: list[str], w) -> str:
        L = land.get(run)
        if not L:
            return "n/a"
        vals = [L[i]["memory_only"] <= 10 if w == "mem" else L[i]["ranks"][w] <= 10 for i in ids if i in L]
        return f"{100.0 * sum(vals) / len(vals):.0f}" if vals else "-"

    rows = []
    for c in cells:
        ids = [p["persona_id"] for p in grp((c,))]
        rows.append([f"{c} ({C.CELL_LABELS.get(c, '')})", len(ids), *[C.three(hr_land("run1b", ids, w), hr_land("run1", ids, w), hr_land("run0", ids, w)) for w in (*LANDSCAPE_WEIGHTS, "mem")]])
    report.table(
        "Fusion landscape, information only: Live HR@10 (%) by memory weight lambda_n (run 1b / run 1 / run 0); text 1.0 and history 0.3 fixed",
        ["cell", "n", "lambda_n 0.5 (published)", "lambda_n 1.0", "lambda_n 2.0", "memory channel only"],
        rows,
    )
    brows = []
    for p in sorted(bridged, key=lambda x: (x["cell"], x["persona_id"])):
        L = land["run1b"][p["persona_id"]]
        brows.append([p["persona_id"], p["cell"], L["ranks"][0.5], smallest_weight(L["ranks"]), L["memory_only"]])
    report.table(
        f"Bridged personas of run 1b ({len(bridged)}): rank at the published weight and the smallest lambda_n on the grid {list(GRID)} that puts the gold chunk in the top 10",
        ["persona", "cell", "rank at 0.5", "smallest lambda_n for top 10", "memory-channel-only rank"],
        brows or [["-", "-", "-", "-", "-"]],
    )
    report.numbers.update(
        {
            "cells": cells,
            "cell5_zero": cell5_zero,
            "cell4_not_below_cells_1_3": cell4_ok,
            "live_minus_frozen_cells_1_3": lf,
            "live_minus_frozen_cells_1_3_run1": lf1,
            "hr10_live_cells_1_3": {"run1b": RU.hr(c123, "live", "10"), "run1": RU.hr(g1(c123), "live", "10") if g1(c123) else None, "run0": RU.hr(g0(c123), "live", "10") if g0(c123) else None},
            "hr10_frozen_cells_1_3": {"run1b": RU.hr(c123, "frozen", "10"), "run1": RU.hr(g1(c123), "frozen", "10") if g1(c123) else None},
            "per_cell": {str(c): {"n": len(grp((c,))), **{f"hr10_{m}": RU.hr(grp((c,)), m, "10") for m in METHODS}, **{f"hr5_{m}": RU.hr(grp((c,)), m, "5") for m in METHODS}, **{f"median_{m}": RU.median_rank(grp((c,)), f"rank_{m}") for m in METHODS}} for c in cells},
            "landscape_rows": rows,
            "bridged_smallest_weight": {r[0]: r[3] for r in brows if r[0] != "-"},
            "bridged_cells_1_3_live_hr10": RU.hr([p for p in bridged if p["cell"] in (1, 2, 3)], "live", "10") if any(p["cell"] in (1, 2, 3) for p in bridged) else 0.0,
        }
    )


if __name__ == "__main__":
    raise SystemExit(C.run_stage(4, "Retrieve and score, fusion landscape (run 1b / run 1 / run 0)", main, requires=[2, 3]))
