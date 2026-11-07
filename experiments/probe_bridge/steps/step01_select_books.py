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

# Position feasibility: in a tight book all 24 sentences anchored to the gold
# window (21 golds, 3 cell-1 reveal slots, 3 pinned cell-4 slots) must fit
# [0.10N, 0.30N] at least 3 chunks apart: 0.20N >= 78 plus layout headroom, so
# books under about 450 chunks fail at step 3. This is a design limit of the
# note's windows with 24 personas per book - decide at this hand-off whether to
# widen the windows, reduce personas per short book, or pick longer books.
MIN_FEASIBLE_CHUNKS = 450


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
    selected = rows[: common.N_BOOKS]
    for row in selected:
        row["feasible_positions"] = row["n_chunks"] >= MIN_FEASIBLE_CHUNKS

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
    infeasible = [r["doc_id"] for r in selected if not r["feasible_positions"]]
    if infeasible:
        report.unexpected.append(
            f"These books have fewer than {MIN_FEASIBLE_CHUNKS} chunks: {infeasible}. "
            "The note's position windows (21 golds in [0.10N, 0.30N], 3 chunks apart) cannot all fit there. "
            "STOP: decide at this hand-off whether to widen the windows or relax the spacing before step 2."
        )
    report.numbers.update(
        {
            "selected": selected,
            "total_chunks": total,
            "qa_entries": len(qa),
            "infeasible_books": infeasible,
        }
    )


if __name__ == "__main__":
    raise SystemExit(run_step(1, "Select books", main, requires=[]))
