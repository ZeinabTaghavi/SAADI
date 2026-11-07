#!/usr/bin/env python3
"""Step 4: insert the sentences, re-chunk, run checks 7-9, freeze the spec.

Writes data/books/<id>/{original,modified}.txt, insertions.json, documents.json
and qa.json, fills the chunk indices into personas.jsonl, and freezes the spec
(spec_frozen.json with the SHA-256 of personas.jsonl). After this step the spec
is never edited.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402
from common import (  # noqa: E402
    BOOKS_JSON,
    DATA_DIR,
    DOCUMENTS_JSON,
    INSERTIONS_JSON,
    PERSONAS_JSONL,
    QA_JSON,
    SPEC_FROZEN_JSON,
    StepReport,
    chunk_book,
    read_json,
    read_jsonl,
    run_step,
    sha256_file,
    whitespace_token_offsets,
    write_json,
    write_jsonl,
)

MIN_SEPARATION = 3
_SENT_BOUNDARY = re.compile(r'[.!?]["\')\]]*\s+')
_PARA_BOUNDARY = re.compile(r"\n[ \t]*\n")


def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def insertion_point(text: str, char_pos: int) -> tuple[int, str]:
    """The note's rule: a paragraph boundary within two sentences of the target,
    otherwise the nearest sentence boundary. Returns (position, boundary kind)."""
    sentence_ends = [m.end() for m in _SENT_BOUNDARY.finditer(text)]
    if not sentence_ends:
        return char_pos, "raw"
    import bisect

    i = bisect.bisect_left(sentence_ends, char_pos)
    window = sentence_ends[max(0, i - 2) : i + 3]
    nearest = min(window or sentence_ends, key=lambda p: abs(p - char_pos))
    lo, hi = (min(window), max(window)) if window else (nearest, nearest)
    paragraphs = [m.end() for m in _PARA_BOUNDARY.finditer(text, max(0, lo - 1), hi + 1)]
    if paragraphs:
        return min(paragraphs, key=lambda p: abs(p - char_pos)), "paragraph"
    return nearest, "sentence"


def main(report: StepReport, argv: list[str]) -> None:
    personas = read_jsonl(PERSONAS_JSONL)
    book_ids = [b["doc_id"] for b in read_json(BOOKS_JSON)["selected"]]
    from saadi.indexing.dataset_loaders.novelhopqa import load_novelhopqa_documents

    books_root = os.environ.get("NOVELHOPQA_BOOKS_ROOT", "") or None
    docs = load_novelhopqa_documents(config_name="all", books_root=books_root)

    insertions: list[dict] = []
    documents: dict[str, str] = {}
    chunk_counts: list[list] = []
    sample_chunks: list[str] = []

    by_book: dict[str, list[dict]] = {}
    for p in personas:
        by_book.setdefault(p["book_id"], []).append(p)

    from step03_check_and_place import draw_positions

    redraw_counts: dict[str, int] = {}
    for book_ordinal, book in enumerate(book_ids):
        text = docs[book]
        chunks = chunk_book(text)
        offsets = whitespace_token_offsets(text)
        group = by_book.get(book, [])

        # Insertion shifts chunk boundaries, so two targets 3 apart can land 2
        # apart after re-chunking. The note says positions are redrawn until
        # check 7 holds, so a failing book gets fresh (deterministic) draws.
        plan: list[dict] = []
        modified = text
        new_chunks: list = []
        chunk_texts: list[str] = []
        for attempt in range(50):
            plan = []
            for p in group:
                for kind, pos in (("gold", p["target_gold_pos"]), ("reveal", p["target_reveal_pos"])):
                    if pos is None:
                        continue
                    sentence = p["gold_sentence"] if kind == "gold" else p["reveal_sentence"]
                    char_pos = offsets[chunks[int(pos)][1]]
                    where, boundary = insertion_point(text, char_pos)
                    plan.append(
                        {
                            "persona_id": p["persona_id"],
                            "kind": kind,
                            "target_chunk": int(pos),
                            "char_pos": where,
                            "boundary": boundary,
                            "sentence": sentence,
                        }
                    )
            # Apply from the end so earlier offsets stay valid; stable for ties.
            modified = text
            for item in sorted(plan, key=lambda x: (-x["char_pos"], x["persona_id"], x["kind"])):
                sep = "\n\n" if item["boundary"] == "paragraph" else " "
                modified = modified[: item["char_pos"]] + item["sentence"] + sep + modified[item["char_pos"] :]

            new_chunks = chunk_book(modified)
            chunk_texts = [_norm_ws(c) for c, _, _ in new_chunks]
            for item in plan:
                needle = _norm_ws(item["sentence"])
                hits = [i for i, c in enumerate(chunk_texts) if needle in c]
                if len(hits) != 1:
                    raise RuntimeError(
                        f"Check 8 failed for {item['persona_id']}/{item['kind']} in {book}: "
                        f"the sentence lies in {len(hits)} chunks ({hits})."
                    )
                item["chunk_index"] = hits[0]
                item["chunk_id"] = f"{book}:{hits[0]}"

            # Check 7 after re-chunking; on failure, redraw this book's targets.
            placed = sorted(item["chunk_index"] for item in plan)
            if all(b - a >= MIN_SEPARATION for a, b in zip(placed, placed[1:], strict=False)):
                redraw_counts[book] = attempt
                break
            import numpy as np

            redraw_rng = np.random.default_rng([common.SEED, book_ordinal, attempt + 1])
            draw_positions(group, {book: len(chunks)}, redraw_rng)
        else:
            raise RuntimeError(f"Check 7 still failing in {book} after 50 position redraws; stop and decide at the hand-off.")

        for p in by_book.get(book, []):
            mine = {i["kind"]: i for i in plan if i["persona_id"] == p["persona_id"]}
            p["gold_chunk_index"] = mine["gold"]["chunk_index"]
            p["gold_chunk_id"] = mine["gold"]["chunk_id"]
            if "reveal" in mine:
                p["reveal_chunk_index"] = mine["reveal"]["chunk_index"]
                p["reveal_chunk_id"] = mine["reveal"]["chunk_id"]
                if p["reveal_chunk_index"] == p["gold_chunk_index"]:
                    raise RuntimeError(f"Check 8 failed: gold and reveal share a chunk for {p['persona_id']}")

        book_dir = DATA_DIR / "books" / book
        book_dir.mkdir(parents=True, exist_ok=True)
        (book_dir / "original.txt").write_text(text, encoding="utf-8")
        (book_dir / "modified.txt").write_text(modified, encoding="utf-8")
        documents[book] = modified
        insertions += plan
        chunk_counts.append([book, len(chunks), len(new_chunks), len(new_chunks) - len(chunks), len(plan), redraw_counts[book]])

        if not sample_chunks:  # one gold, one reveal, one untouched chunk, from the first book
            gold0 = next(i for i in plan if i["kind"] == "gold")
            reveal0 = next((i for i in plan if i["kind"] == "reveal"), None)
            untouched = next(i for i in range(len(new_chunks)) if i not in {x["chunk_index"] for x in plan})
            sample_chunks.append(f"**Gold chunk {gold0['chunk_id']}:**\n\n> {chunk_texts[gold0['chunk_index']][:900]}")
            if reveal0:
                sample_chunks.append(f"**Reveal chunk {reveal0['chunk_id']}:**\n\n> {chunk_texts[reveal0['chunk_index']][:900]}")
            sample_chunks.append(f"**Untouched chunk {book}:{untouched}:**\n\n> {chunk_texts[untouched][:900]}")

    # Check 9: the study runner's own evidence mapping must give the same chunks.
    from saadi.core.types import ChunkRecord
    from saadi.indexing.labeling import build_retrieval_examples

    qa_entries = [
        {
            "id": p["persona_id"],
            "question": p["query"],
            "document_id": p["book_id"],
            "answers": [p["answer"]],
            "retrieval_spans": [p["gold_sentence"]],
        }
        for p in personas
    ]
    chunks_by_doc = {}
    for book in book_ids:
        recs = []
        for idx, (c, t0, t1) in enumerate(chunk_book(documents[book])):
            recs.append(ChunkRecord(doc_id=book, chunk_id=f"{book}:{idx}", chunk_index=idx, raw_text=c, token_start=t0, token_end=t1))
        chunks_by_doc[book] = recs
    examples = build_retrieval_examples(qa_entries, chunks_by_doc)
    by_id = {e.query_id: e for e in examples}
    label_mismatches = []
    for p in personas:
        example = by_id.get(p["persona_id"])
        if example is None or example.gold_chunk_ids != [p["gold_chunk_id"]]:
            label_mismatches.append((p["persona_id"], example.gold_chunk_ids if example else None, p["gold_chunk_id"]))
    if label_mismatches:
        raise RuntimeError(
            f"Check 9 failed: the runner's labels differ from insertions.json for "
            f"{len(label_mismatches)} personas; first: {label_mismatches[:3]}"
        )

    write_json(INSERTIONS_JSON, insertions)
    write_json(DOCUMENTS_JSON, documents)
    write_json(QA_JSON, qa_entries)
    write_jsonl(PERSONAS_JSONL, personas)
    spec_hash = sha256_file(PERSONAS_JSONL)
    write_json(
        SPEC_FROZEN_JSON,
        {
            "personas_jsonl_sha256": spec_hash,
            "books_json_sha256": sha256_file(BOOKS_JSON),
            "templates_json_sha256": sha256_file(common.TEMPLATES_JSON),
            "documents_json_sha256": sha256_file(DOCUMENTS_JSON),
            "qa_json_sha256": sha256_file(QA_JSON),
            "n_personas": len(personas),
        },
    )
    report.files_written += [
        str(INSERTIONS_JSON),
        str(DOCUMENTS_JSON),
        str(QA_JSON),
        str(PERSONAS_JSONL),
        str(SPEC_FROZEN_JSON),
        str(DATA_DIR / "books" / "<id>" / "{original,modified}.txt"),
    ]
    report.commands.append("insert at chunk starts (paragraph/sentence boundary), re-chunk, checks 7-9, freeze spec")
    report.table(
        "Chunks before and after",
        ["book", "before", "after", "growth", "sentences inserted", "position redraws"],
        chunk_counts,
    )
    report.add(
        "Spec frozen",
        f"`personas.jsonl` SHA-256: **{spec_hash}**. After this step the spec is read-only; step 5 refuses to run if the hash changes.",
    )
    report.add("Three modified chunks", "\n\n".join(sample_chunks))
    report.numbers.update({"spec_sha256": spec_hash, "chunk_counts": chunk_counts, "labels_checked": len(personas)})


if __name__ == "__main__":
    raise SystemExit(run_step(4, "Insert and re-chunk", main, requires=[1, 2, 3]))
