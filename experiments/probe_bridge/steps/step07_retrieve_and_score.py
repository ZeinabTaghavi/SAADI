#!/usr/bin/env python3
"""Step 7: rank all chunks of each book for every persona's query, four ways.

Methods: Vanilla BM25 (text channel alone), SAADI live, SAADI frozen (E2's
construction) and SAADI without memory (lambda_n = 0), plus the
memory-channel-only rank (live and frozen). Per-persona results go to
outputs/probe_bridge/per_persona.jsonl; the report carries the metric tables.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402
from common import (  # noqa: E402
    OUT_DIR,
    PUBLISHED_SETTINGS,
    StepReport,
    mcnemar_exact_p,
    memory_channel_ranks,
    paired_bootstrap_ci,
    read_jsonl,
    run_step,
    write_jsonl,
)

MEMORY_JSONL = OUT_DIR / "personas_memory.jsonl"
PER_PERSONA = OUT_DIR / "per_persona.jsonl"
METHODS = ("vanilla", "no_memory", "frozen", "live")


def main(report: StepReport, argv: list[str]) -> None:
    from saadi.indexing.retrieval_methods.mbr import (
        MBR_VARIANT_MBR2,
        _find_doc_entry,
        _load_doc_records,
        _read_index_payload,
        build_mbr_document_memory_index,
        score_mbr_variant_document_pools,
    )
    from saadi.rebuttal.frozen_records import (
        _frozen_streams,
        frozen_chunk_fields,
        replay_first_added,
    )

    personas = read_jsonl(MEMORY_JSONL)
    index_path = common.find_cache_index()
    payload = _read_index_payload(index_path)
    report.commands.append(
        "score_mbr_variant_document_pools + combine (published settings) per persona; "
        "frozen via saadi.rebuttal.frozen_records; vanilla = the text channel alone"
    )

    by_book: dict[str, list[dict]] = {}
    for p in personas:
        by_book.setdefault(p["book_id"], []).append(p)

    from saadi.rebuttal.frozen_records import score_query_variants

    for book, group in sorted(by_book.items()):
        doc_entry = _find_doc_entry(payload, doc_id=book)
        records = _load_doc_records(index_path=index_path, index_payload=payload, doc_entry=doc_entry, doc_id=book)
        replay = replay_first_added(book, records)
        document_index = build_mbr_document_memory_index(index_path, doc_id=book)
        frozen = frozen_chunk_fields(document_index, replay)
        n_chunks = len(document_index.chunk_pool)
        chunk_order = [c.chunk_id for c in document_index.chunk_pool]
        position = {cid: i for i, cid in enumerate(chunk_order)}

        for p in group:
            gold = p["gold_chunk_id"]
            rankings = score_query_variants(
                document_index,
                frozen,
                query=p["query"],
                retriever_name="bm25",
                retrieve_k=n_chunks,
                settings=PUBLISHED_SETTINGS,
            )
            streams = score_mbr_variant_document_pools(
                document_index,
                query=p["query"],
                retriever_name="bm25",
                variant=MBR_VARIANT_MBR2,
                memory_name_weight=1.0,
                memory_alias_weight=1.0,
                memory_description_weight=1.0,
                memory_aggregation="max",
            )
            vanilla = sorted(
                chunk_order,
                key=lambda cid: (-float(streams.text_scores.get(cid, 0.0)), position[cid]),
            )
            ranked = {
                "vanilla": vanilla,
                "live": rankings["live"],
                "frozen": rankings["frozen"],
                "no_memory": rankings["no_memory"],
            }
            p["n_chunks"] = n_chunks
            for method, ids in ranked.items():
                rank = ids.index(gold) + 1 if gold in ids else n_chunks + 1
                p[f"rank_{method}"] = rank
                p[f"hit5_{method}"] = rank <= 5
                p[f"hit10_{method}"] = rank <= 10
                p[f"top3_{method}"] = ids[:3]
            mem_live = memory_channel_ranks(streams, document_index)
            mem_frozen = memory_channel_ranks(_frozen_streams(streams, frozen, memory_aggregation="max"), document_index)
            p["memory_rank_live"] = mem_live.get(gold)
            p["memory_rank_frozen"] = mem_frozen.get(gold)

    write_jsonl(PER_PERSONA, personas)
    report.files_written.append(str(PER_PERSONA))

    # ---- metric tables -----------------------------------------------------
    def hr(group, method, k) -> float:
        return 100.0 * sum(p[f"hit{k}_{method}"] for p in group) / len(group) if group else float("nan")

    def median_rank(group, key) -> float:
        vals = [p[key] for p in group if p.get(key) is not None]
        return float(statistics.median(vals)) if vals else float("nan")

    def ci_and_mcnemar(group, a, b, k="10") -> str:
        diffs = [(float(p[f"hit{k}_{a}"]) - float(p[f"hit{k}_{b}"])) * 100.0 for p in group]
        mean, lo, hi = paired_bootstrap_ci(diffs)
        x = sum(1 for p in group if p[f"hit{k}_{a}"] and not p[f"hit{k}_{b}"])
        y = sum(1 for p in group if p[f"hit{k}_{b}"] and not p[f"hit{k}_{a}"])
        return f"{mean:+.1f} pp [{lo:+.1f}, {hi:+.1f}], McNemar p={mcnemar_exact_p(x, y):.3f} ({x} vs {y})"

    cells = sorted({p["cell"] for p in personas})
    rows = []
    for cell in cells:
        group = [p for p in personas if p["cell"] == cell]
        rows.append(
            [
                cell,
                len(group),
                *[f"{hr(group, m, '10'):.0f} / {median_rank(group, f'rank_{m}'):.0f}" for m in METHODS],
                ci_and_mcnemar(group, "live", "frozen"),
                ci_and_mcnemar(group, "live", "vanilla"),
            ]
        )
    report.table(
        "HR@10 (%) / median gold rank, per cell",
        ["cell", "n", "Vanilla", "No memory", "Frozen", "Live", "Live-Frozen @10", "Live-Vanilla @10"],
        rows,
    )
    rows5 = []
    for cell in cells:
        group = [p for p in personas if p["cell"] == cell]
        rows5.append(
            [
                cell,
                len(group),
                *[f"{hr(group, m, '5'):.0f}" for m in METHODS],
                f"{median_rank(group, 'memory_rank_live'):.0f}",
                f"{median_rank(group, 'memory_rank_frozen'):.0f}",
            ]
        )
    report.table(
        "HR@5 (%) and the memory-channel-only median rank of the gold chunk",
        ["cell", "n", "Vanilla", "No memory", "Frozen", "Live", "mem-rank live", "mem-rank frozen"],
        rows5,
    )

    # Bridge-reached subset (from step 6): form 2 in the gold chunk's live record.
    bridged_rows = []
    for cell in cells:
        group = [
            p for p in personas if p["cell"] == cell and p.get("form2_in_that_record_final") and p.get("gold_chunk_linked_to_that_record")
        ]
        if not group:
            bridged_rows.append([cell, 0, "-", "-", "-", "-", "-"])
            continue
        bridged_rows.append([cell, len(group), *[f"{hr(group, m, '10'):.0f}" for m in METHODS], ci_and_mcnemar(group, "live", "frozen")])
    report.table(
        "Bridge-reached personas only: HR@10 (%)",
        ["cell", "n", "Vanilla", "No memory", "Frozen", "Live", "Live-Frozen @10"],
        bridged_rows,
    )
    small = [c for c in cells if len([p for p in personas if p["cell"] == c]) < 30]
    if small:
        report.unexpected.append(f"Cells with fewer than 30 personas: {small} - too small to conclude.")
    report.numbers["cells"] = cells
    report.add(
        "Vanilla leak check",
        "A cell where Vanilla succeeds means a query word reached the gold chunk through the text; "
        "see the Vanilla column (expected: fails everywhere, HR near 0).",
    )


if __name__ == "__main__":
    raise SystemExit(run_step(7, "Retrieve and score", main, requires=[5, 6]))
