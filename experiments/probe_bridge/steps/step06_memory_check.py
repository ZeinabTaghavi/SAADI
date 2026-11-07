#!/usr/bin/env python3
"""Step 6: inspect the memory before any retrieval - did the bridge form?

For every persona, this step fills the memory-check fields into a copy of
personas.jsonl under outputs/probe_bridge/ and reports per-cell rates. This is
the gate: if the bridge word reached the gold chunk's record in fewer than half
of the personas of cells 1 to 3, the weak point is extraction, and the
retrieval numbers of step 7 must be read with that in mind.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402
from common import (  # noqa: E402
    OUT_DIR,
    PERSONAS_JSONL,
    SEED,
    StepReport,
    content_token_set,
    read_jsonl,
    run_step,
    tokens,
    write_jsonl,
)

MEMORY_JSONL = OUT_DIR / "personas_memory.jsonl"
GATE_CELLS = (1, 2, 3)
GATE_RATE = 0.5


def _field_tokens(values) -> list[set[str]]:
    return [set(tokens(v)) for v in values]


def classify_form2(entity, form2_t: set[str]) -> str:
    if form2_t <= set(tokens(entity.name)):
        return "name"
    if any(form2_t <= s for s in _field_tokens(entity.aliases)):
        return "alias"
    if any(form2_t <= s for s in _field_tokens(entity.descriptions)):
        return "description"
    return "none"


def find_record(document_index, form_t: set[str]):
    """The record whose name or aliases carry every token of the form."""
    best = None
    for entity in document_index.memory_entities:
        name_alias = set(tokens(entity.name))
        for alias in entity.aliases:
            name_alias |= set(tokens(alias))
        if form_t <= name_alias:
            if best is None or form_t == set(tokens(entity.name)):
                best = entity
            if form_t == set(tokens(entity.name)):
                break
    return best


def main(report: StepReport, argv: list[str]) -> None:
    import numpy as np

    from saadi.indexing.retrieval_methods.mbr import (
        _find_doc_entry,
        _load_doc_records,
        _read_index_payload,
        build_mbr_document_memory_index,
    )
    from saadi.rebuttal.frozen_records import frozen_chunk_fields, replay_first_added

    personas = read_jsonl(PERSONAS_JSONL)
    index_path = common.find_cache_index()
    payload = _read_index_payload(index_path)
    report.commands.append(f"inspect {index_path} with the E2 replay (saadi.rebuttal.frozen_records)")

    by_book: dict[str, list[dict]] = {}
    for p in personas:
        by_book.setdefault(p["book_id"], []).append(p)

    growth_rows = []
    for book, group in by_book.items():
        doc_entry = _find_doc_entry(payload, doc_id=book)
        if doc_entry is None:
            raise RuntimeError(f"Book {book} is not in the cache index {index_path}")
        records = _load_doc_records(index_path=index_path, index_payload=payload, doc_entry=doc_entry, doc_id=book)
        replay = replay_first_added(book, records)
        document_index = build_mbr_document_memory_index(index_path, doc_id=book)
        frozen = frozen_chunk_fields(document_index, replay)
        rec_by_index = {int(r.chunk_index): r for r in records}
        growth_rows.append([book, len(records)])

        for p in group:
            gold_id, reveal_id = p["gold_chunk_id"], p.get("reveal_chunk_id")
            form1_t = content_token_set(p["form1"])
            form2_t = content_token_set(p["form2"])
            f1 = find_record(document_index, form1_t)
            f2 = find_record(document_index, form2_t)

            p["form1_record_found"] = f1 is not None
            p["form1_record_name"] = f1.name if f1 else None
            p["form2_field"] = classify_form2(f1, form2_t) if f1 else "none"
            p["form2_in_that_record_final"] = p["form2_field"] != "none"
            linked = document_index.chunk_to_names.get(gold_id, [])
            p["gold_chunk_linked_to_that_record"] = bool(f1 and f1.name in linked)
            p["reveal_chunk_linked_to_that_record"] = bool(f1 and reveal_id and f1.name in document_index.chunk_to_names.get(reveal_id, []))
            if f1 is None and f2 is None:
                p["merge_direction"] = "no_record"
            elif f1 is not None and p["form2_in_that_record_final"]:
                p["merge_direction"] = "form2_into_form1"
            elif f2 is not None and (
                form1_t <= set(tokens(f2.name))
                or any(form1_t <= s for s in _field_tokens(f2.aliases))
                or any(form1_t <= s for s in _field_tokens(f2.descriptions))
            ):
                p["merge_direction"] = "form1_into_form2"
            else:
                p["merge_direction"] = "never_joined"

            fc = frozen.get(gold_id)
            frozen_tokens: set[str] = set()
            if fc:
                for v in fc.linked_names + fc.frozen_aliases + fc.frozen_descriptions:
                    frozen_tokens |= set(tokens(v))
            p["form2_in_gold_frozen_record"] = bool(form2_t and form2_t <= frozen_tokens)

            arrival = None
            if f1 is not None:
                key = str(f1.name or "").strip().lower()
                candidates = []
                if form2_t <= set(tokens(f1.name)):
                    candidates.append(replay.name_first.get(key))
                for alias in f1.aliases:
                    if form2_t <= set(tokens(alias)):
                        candidates.append(replay.alias_first.get((key, alias)))
                for desc in f1.descriptions:
                    if form2_t <= set(tokens(desc)):
                        candidates.append(replay.desc_first.get((key, desc)))
                candidates = [c for c in candidates if c is not None]
                arrival = min(candidates) if candidates else None
            p["form2_arrival_step"] = arrival

            # Parser status of the chunks that hold inserted sentences.
            for kind, idx in (("gold", p["gold_chunk_index"]), ("reveal", p.get("reveal_chunk_index"))):
                if idx is None:
                    continue
                rec = rec_by_index.get(int(idx))
                p[f"{kind}_parser_status"] = rec.parser_status if rec else "missing"

    write_jsonl(MEMORY_JSONL, personas)
    report.files_written.append(str(MEMORY_JSONL))

    # Per-cell outcome rates.
    def rate(group, key) -> str:
        vals = [bool(p.get(key)) for p in group]
        return f"{sum(vals)}/{len(vals)} ({100.0 * sum(vals) / len(vals):.0f}%)" if vals else "-"

    rows = []
    for cell in sorted({p["cell"] for p in personas}):
        group = [p for p in personas if p["cell"] == cell]
        with_reveal = cell != 5
        rows.append(
            [
                cell,
                len(group),
                rate(group, "form1_record_found"),
                rate(group, "form2_in_that_record_final") if with_reveal else "-",
                rate(group, "gold_chunk_linked_to_that_record"),
                rate(group, "form2_in_gold_frozen_record"),
                {d: sum(1 for p in group if p["merge_direction"] == d) for d in sorted({p["merge_direction"] for p in group})},
            ]
        )
    report.table(
        "Memory outcomes per cell",
        ["cell", "n", "form1 record", "form2 in final record", "gold chunk linked", "form2 in gold FROZEN record", "merge directions"],
        rows,
    )
    field_rows = []
    for cell in sorted({p["cell"] for p in personas}):
        group = [p for p in personas if p["cell"] == cell and p["cell"] != 5]
        counts = {f: sum(1 for p in group if p.get("form2_field") == f) for f in ("name", "alias", "description", "none")}
        field_rows.append([cell, counts["name"], counts["alias"], counts["description"], counts["none"]])
    report.table("Where form 2 sits in the form-1 record", ["cell", "name", "alias", "description", "none"], field_rows)

    # The gate.
    gate_group = [p for p in personas if p["cell"] in GATE_CELLS]
    bridged = [p for p in gate_group if p["form2_in_that_record_final"] and p["gold_chunk_linked_to_that_record"]]
    gate_rate = len(bridged) / len(gate_group) if gate_group else 0.0
    verdict = (
        f"The bridge word reached the gold chunk's record in {len(bridged)} of {len(gate_group)} personas "
        f"of cells 1-3 ({100 * gate_rate:.0f}%)."
    )
    if gate_rate < GATE_RATE:
        verdict += (
            " **That is below half: the weak point is extraction, and the retrieval numbers of step 7 must be read with that in mind.**"
        )
    report.add("Gate", verdict)
    report.numbers.update({"gate_rate": gate_rate, "gate_n": len(gate_group), "gate_bridged": len(bridged)})

    parser_bad = [
        (p["persona_id"], k) for p in personas for k in ("gold_parser_status", "reveal_parser_status") if p.get(k) not in (None, "ok")
    ]
    if parser_bad:
        report.unexpected.append(f"Parser status not ok for inserted chunks: {parser_bad[:10]} ({len(parser_bad)} total).")
    report.table("Chunk records per book (growth check is in step 4's table)", ["book", "chunk records"], growth_rows)

    rng = np.random.default_rng(SEED)
    picks = sorted(int(x) for x in rng.choice(len(personas), size=5, replace=False))
    lines = []
    for i in picks:
        p = personas[i]
        lines.append(
            f"- **{p['persona_id']}** (cell {p['cell']}): form1 record={p['form1_record_name']!r}, "
            f"form2 field={p['form2_field']}, merge={p['merge_direction']}, arrival step={p['form2_arrival_step']}, "
            f"gold step={p['gold_chunk_index']}, gold linked={p['gold_chunk_linked_to_that_record']}, "
            f"frozen has form2={p['form2_in_gold_frozen_record']}"
        )
    report.add("Five personas in full (seeded draw)", "\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(run_step(6, "Memory check", main, requires=[4, 5]))
