#!/usr/bin/env python3
"""Step 1: load the 18 NovelHopQA books, chunk them, select the 10 shortest."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402
from common import BOOKS_JSON, ROOT, SPEC_DIR, StepReport, chunk_book, run_step, write_json  # noqa: E402

QUESTION_SET_TXT = SPEC_DIR / "question_set.txt"

# Position feasibility under geometry v2 (gold window [0.05N, 0.45N], decided
# at the step-1 hand-off): the window holds 27 anchored positions 3 chunks
# apart plus the cell-4 clearance, which needs about N >= 240 (verified by
# simulation at the real book sizes; 213 is provably too small). Selection
# takes the 10 shortest books at or above this floor and reports any skipped.
MIN_FEASIBLE_CHUNKS = 240


def main(report: StepReport, argv: list[str]) -> None:
    from saadi.indexing.dataset_loaders.novelhopqa import (
        load_novelhopqa_documents,
        load_novelhopqa_qa_entries,
    )

    books_root = os.environ.get("NOVELHOPQA_BOOKS_ROOT", "")
    report.commands.append(f"load_novelhopqa_documents(config_name='all', books_root={books_root!r}); chunk_text(tokenizer=None, 500, 0)")
    docs = load_novelhopqa_documents(config_name="all", books_root=books_root or None)
    if not docs:
        raise RuntimeError(
            f"No NovelHopQA books loaded (NOVELHOPQA_BOOKS_ROOT={books_root!r}). "
            "Supply the book corpus as described in docs/data_and_artifacts.md."
        )
    qa = load_novelhopqa_qa_entries(config_name="all", n="all", selection_method="first", books_root=books_root or None)

    rows = []
    for doc_id, text in sorted(docs.items()):
        chunks = chunk_book(text)
        rows.append(
            {
                "doc_id": str(doc_id),
                "n_chunks": len(chunks),
                "n_words": len(text.split()),
                "n_chars": len(text),
            }
        )
    rows.sort(key=lambda r: (r["n_chunks"], r["doc_id"]))
    for row in rows:
        row["feasible_positions"] = row["n_chunks"] >= MIN_FEASIBLE_CHUNKS
    skipped = [r["doc_id"] for r in rows if not r["feasible_positions"]]
    selected = [r for r in rows if r["feasible_positions"]][: common.N_BOOKS]
    if len(selected) < common.N_BOOKS:
        raise RuntimeError(
            f"Only {len(selected)} books have at least {MIN_FEASIBLE_CHUNKS} chunks; the probe needs 10. Stop and decide at the hand-off."
        )

    # Cross-check against the paper's construction cache when it is present
    # (reported only; a different book edition would show up here).
    cache_counts = {}
    cache_glob = sorted(
        (ROOT / "outputs" / "novelhopqa_ablation").glob(
            "500_tokens/llm_cache/novelhopqa/hierarchical/branching_factor_5/chunk_500/*/Qwen3-30B-A3B-Instruct-2507/*/index.json"
        )
    )
    if cache_glob:
        import json

        payload = json.loads(cache_glob[0].read_text(encoding="utf-8"))
        for doc in payload.get("docs", []):
            steps_path = (doc.get("files", {}) or {}).get("steps_index")
            if steps_path:
                from saadi.indexing.retrieval_methods.mbr import _resolve_cache_file_path

                p = _resolve_cache_file_path(steps_path, base_dir=cache_glob[0].parent)
                if Path(p).exists():
                    cache_counts[str(doc["doc_id"])] = sum(1 for _ in open(p, encoding="utf-8"))
    for row in selected:
        row["paper_cache_chunks"] = cache_counts.get(row["doc_id"])

    write_json(BOOKS_JSON, {"selected": selected, "all_books": rows, "chunk_size": common.CHUNK_SIZE})
    report.files_written.append(str(BOOKS_JSON))

    # The question-set text used by the absence check (check 1).
    parts = []
    for entry in qa:
        parts.append(str(entry.get("question", "")))
        answers = entry.get("answers", [])
        if isinstance(answers, list):
            parts.extend(str(a) for a in answers)
        spans = entry.get("retrieval_spans", [])
        if isinstance(spans, list):
            parts.extend(str(s) for s in spans)
    QUESTION_SET_TXT.parent.mkdir(parents=True, exist_ok=True)
    QUESTION_SET_TXT.write_text("\n".join(parts), encoding="utf-8")
    report.files_written.append(str(QUESTION_SET_TXT))

    report.table(
        "The 10 selected books (shortest by chunk count)",
        ["doc_id", "chunks", "words", "paper cache chunks", "position windows feasible?"],
        [[r["doc_id"], r["n_chunks"], r["n_words"], r["paper_cache_chunks"], r["feasible_positions"]] for r in selected],
    )
    total = sum(r["n_chunks"] for r in selected)
    report.add(
        "Totals",
        f"Books loaded: {len(rows)}. Selected: {len(selected)}. "
        f"Total chunks to index: **{total}**. QA entries in the question set: {len(qa)}.",
    )
    if skipped:
        report.add(
            "Skipped books",
            f"Too short for geometry v2 (under {MIN_FEASIBLE_CHUNKS} chunks): {skipped}. "
            "The next-shortest feasible books take their places.",
        )
    report.numbers.update(
        {
            "selected": selected,
            "total_chunks": total,
            "qa_entries": len(qa),
            "skipped_books": skipped,
        }
    )


if __name__ == "__main__":
    raise SystemExit(run_step(1, "Select books", main, requires=[]))
