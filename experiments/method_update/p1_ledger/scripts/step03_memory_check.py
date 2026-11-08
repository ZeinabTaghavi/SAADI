#!/usr/bin/env python3
"""Step 3 (CPU): the memory check with stubs, next to run 0.

For every persona: form 1 present after the gold chunk; form 2 present after
the reveal; both present; earlier record in the prompt at the reveal; joined
when both present; bridge formed (form 2 in the gold chunk's live record);
merge direction. Provenance (candidate net / LLM / both), NOT AN ENTITY counts,
parse failures, promotions, cap hits. Merge audit: all SAME AS decisions with
reasons (file), a random sample of 30 (seed 42), and the Mr./Mrs. check on the
books' real characters. Five full ledger entries (seed 42). This is the gate
for step 4 (hand-off).

The probe's step 6 code is imported, not copied: find_record and classify_form2
run on an index that now contains the stubs (they are entity records with no
description), so stubs count as records.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import report_utils as RU  # noqa: E402
from common import (  # noqa: E402
    CELL_LABELS,
    GATE_CELLS,
    LEDGER_DIR,
    OUT_DIR,
    REVEAL_CELLS,
    RUN0_OUT_DIR,
    SEED,
    StepReport,
    content_token_set,
    load_personas,
    load_probe_step,
    pct,
    run0_file,
    run_step,
    tokens,
    write_jsonl,
)

MEMORY_JSONL = OUT_DIR / "personas_memory.jsonl"
SAME_AS_JSONL = OUT_DIR / "same_as_decisions.jsonl"
TITLE_CONFLICTS_JSONL = OUT_DIR / "title_conflicts.jsonl"
GATE_RATE = 0.5


def _holds(text: str, form_t: set[str]) -> bool:
    return bool(form_t) and form_t <= set(tokens(text))


def _name_alias_holds(entity, form_t: set[str], *, alias_steps: dict | None = None, before_step: int | None = None) -> bool:
    name = entity.get("name") if isinstance(entity, dict) else getattr(entity, "name", "")
    if _holds(name, form_t):
        return True
    aliases = entity.get("aliases", []) if isinstance(entity, dict) else getattr(entity, "aliases", [])
    for a in aliases or []:
        if before_step is not None and alias_steps is not None and alias_steps.get(a, -1) >= before_step:
            continue
        if _holds(a, form_t):
            return True
    return False


def main(report: StepReport, argv: list[str]) -> None:
    from saadi.indexing.retrieval_methods.mbr import _find_doc_entry, _load_doc_records, _read_index_payload, build_mbr_document_memory_index
    from saadi.ledger.ledger import LedgerStore
    from saadi.ledger.measure import (
        apply_description_cap,
        ledger_frozen_chunk_fields,
        provenance_counts,
        recalled_at_step,
        same_as_decisions,
        title_conflicts,
        verify_ledger_against_replay,
    )
    from saadi.rebuttal.frozen_records import replay_first_added

    step06 = load_probe_step("step06_memory_check.py")
    personas = load_personas()
    index_path = common.find_run1_cache_index()
    payload = _read_index_payload(index_path)
    store = LedgerStore(LEDGER_DIR)
    report.commands.append(f"inspect {index_path} + ledger {LEDGER_DIR} with the probe's step 6 (imported) and saadi.ledger.measure")

    run0_mem = {r["persona_id"]: r for r in common.read_jsonl(run0_file("personas_memory.jsonl"))} if run0_file("personas_memory.jsonl") else {}
    run0_q2 = {r["persona_id"]: r for r in common.read_jsonl(run0_file("diagnose/q2_recall.jsonl"))} if run0_file("diagnose/q2_recall.jsonl") else {}
    if not run0_mem:
        report.unexpected.append(f"Run 0's personas_memory.jsonl was not found under {RUN0_OUT_DIR}; run 0 columns show n/a.")
    if not run0_q2:
        report.unexpected.append(f"Run 0's diagnose/q2_recall.jsonl was not found under {RUN0_OUT_DIR}; the run 0 'in prompt at reveal' column shows n/a.")

    by_book: dict[str, list[dict]] = defaultdict(list)
    for p in personas:
        by_book[p["book_id"]].append(p)

    persona_tokens: set[str] = set()
    for p in personas:
        persona_tokens |= content_token_set(p["form1"]) | content_token_set(p["form2"])

    growth_rows, prov_total, all_same_as, all_conflicts, replay_problems, drift_rows = [], Counter(), [], [], [], []
    for book in sorted(by_book):
        doc_entry = _find_doc_entry(payload, doc_id=book)
        if doc_entry is None:
            raise RuntimeError(f"Book {book} is not in the cache index {index_path}")
        records = _load_doc_records(index_path=index_path, index_payload=payload, doc_entry=doc_entry, doc_id=book)
        replay = replay_first_added(book, records)
        document_index = build_mbr_document_memory_index(index_path, doc_id=book)
        ledger = store.doc(book)
        drift = apply_description_cap(document_index, ledger)
        drift_rows.append([book, drift["records_with_replacements"], drift["descriptions_removed_from_index"]])
        problems = verify_ledger_against_replay(ledger, replay)
        if problems:
            replay_problems.append(f"{book}: {len(problems)} (first: {problems[0]})")
        frozen, notes = ledger_frozen_chunk_fields(document_index, replay, ledger)
        kinds = Counter(e.kind for e in ledger.entries.values())
        growth_rows.append([book, len(records), kinds.get("record", 0), kinds.get("stub", 0), ledger.counters.get("promotions", 0), len(ledger.events)])
        pc = provenance_counts(ledger)
        prov_total.update(pc["entries"])
        prov_total.update({f"ev:{k}": v for k, v in pc["event_totals"].items()})
        all_same_as += same_as_decisions(ledger)
        all_conflicts += title_conflicts(ledger, exclude_tokens=persona_tokens)
        rec_by_index = {int(r.chunk_index): r for r in records}

        def memory_after(step: int) -> list[dict]:
            rec = rec_by_index.get(int(step))
            return list((rec.state_snapshot or {}).get("entity_memory", []) or []) if rec else []

        for p in by_book[book]:
            gold_id, reveal_id = p["gold_chunk_id"], p.get("reveal_chunk_id")
            gold_step, reveal_step = int(p["gold_chunk_index"]), (int(p["reveal_chunk_index"]) if p.get("reveal_chunk_index") is not None else None)
            form1_t, form2_t = content_token_set(p["form1"]), content_token_set(p["form2"])
            # The probe's own step-6 fields, on an index that includes the stubs.
            f1 = step06.find_record(document_index, form1_t)
            f2 = step06.find_record(document_index, form2_t)
            p["form1_record_found"] = f1 is not None
            p["form1_record_name"] = f1.name if f1 else None
            p["form1_record_kind"] = (ledger.by_name(f1.name).kind if f1 and ledger.by_name(f1.name) else None)
            p["form2_record_found"] = f2 is not None
            p["form2_record_name"] = f2.name if f2 else None
            p["form2_field"] = step06.classify_form2(f1, form2_t) if f1 else "none"
            p["form2_in_that_record_final"] = p["form2_field"] != "none"
            linked = document_index.chunk_to_names.get(gold_id, [])
            p["gold_chunk_linked_to_that_record"] = bool(f1 and f1.name in linked)
            p["reveal_chunk_linked_to_that_record"] = bool(f1 and reveal_id and f1.name in document_index.chunk_to_names.get(reveal_id, []))
            p["gold_chunk_linked_names"] = list(linked)
            if f1 is None and f2 is None:
                p["merge_direction"] = "no_record"
            elif f1 is not None and p["form2_in_that_record_final"]:
                p["merge_direction"] = "form2_into_form1"
            elif f2 is not None and (
                form1_t <= set(tokens(f2.name))
                or any(form1_t <= s for s in step06._field_tokens(f2.aliases))
                or any(form1_t <= s for s in step06._field_tokens(f2.descriptions))
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
            p["gold_chunk_stub_names_at_its_step"] = [n.name for n in notes.get(gold_id, []) if n.stub_at_chunk_step]
            # New with the ledger: presence over time and the prompt at the link step.
            p["form1_present_after_gold"] = any(_name_alias_holds(e, form1_t) for e in memory_after(gold_step))
            p["form2_present_after_reveal"] = any(_name_alias_holds(e, form2_t) for e in memory_after(reveal_step)) if reveal_step is not None else None
            p["both_present_final"] = bool(f1 is not None and f2 is not None)
            p["joined_when_both_present"] = bool(p["both_present_final"] and (f1 is f2 or p["form2_in_that_record_final"] or p["merge_direction"] == "form1_into_form2"))
            # Earlier record in the prompt at the step where the link could form (reveal; the gold chunk in cell 4).
            if p["cell"] == 4:
                link_step, earlier_t = gold_step, form2_t
            else:
                link_step, earlier_t = reveal_step, form1_t
            if link_step is not None:
                shown = recalled_at_step(ledger, link_step)
                p["prompt_at_link_step"] = {k: shown[k] for k in ("overlap", "ledger", "left_out", "cap_hit", "decision_status")}
                hit = False
                for name in shown["overlap"] + shown["ledger"]:
                    entry = ledger.by_name(name)
                    if entry is None:
                        continue
                    if _name_alias_holds({"name": entry.name, "aliases": entry.aliases}, earlier_t, alias_steps=entry.alias_steps, before_step=link_step):
                        hit = True
                        break
                p["earlier_record_in_prompt_at_link_step"] = hit
                earlier_existed = any(_name_alias_holds(e, earlier_t) for e in memory_after(link_step - 1)) if link_step > 0 else False
                p["earlier_record_existed_before_link_step"] = earlier_existed
                p["earlier_left_out_by_cap"] = (not hit) and earlier_existed and any(
                    _name_alias_holds({"name": ledger.by_name(n).name, "aliases": ledger.by_name(n).aliases}, earlier_t, alias_steps=ledger.by_name(n).alias_steps, before_step=link_step)
                    for n in shown["left_out"] if ledger.by_name(n)
                )
            else:
                p["earlier_record_in_prompt_at_link_step"] = None
                p["earlier_record_existed_before_link_step"] = None
                p["earlier_left_out_by_cap"] = None
            # Arrival step of form 2 in the form-1 record (as in step 6).
            arrival = None
            if f1 is not None:
                key = str(f1.name or "").strip().lower()
                cands = []
                if form2_t <= set(tokens(f1.name)):
                    cands.append(replay.name_first.get(key))
                for alias in f1.aliases:
                    if form2_t <= set(tokens(alias)):
                        cands.append(replay.alias_first.get((key, alias)))
                for desc in f1.descriptions:
                    if form2_t <= set(tokens(desc)):
                        cands.append(replay.desc_first.get((key, desc)))
                cands = [c for c in cands if c is not None]
                arrival = min(cands) if cands else None
            p["form2_arrival_step"] = arrival
            for kind, idx in (("gold", gold_step), ("reveal", reveal_step)):
                if idx is None:
                    continue
                rec = rec_by_index.get(int(idx))
                p[f"{kind}_parser_status"] = rec.parser_status if rec else "missing"
                ev = next((e for e in ledger.events if int(e["step"]) == int(idx)), None)
                p[f"{kind}_decision_status"] = ev.get("decision_status") if ev else "missing"
                p[f"{kind}_candidates"] = [c["text"] for c in ev.get("candidates", [])] if ev else []
            # Run 0 values next to it.
            r0 = run0_mem.get(p["persona_id"], {})
            p["run0"] = {
                "form1_record_found": r0.get("form1_record_found"),
                "form2_in_that_record_final": r0.get("form2_in_that_record_final"),
                "gold_chunk_linked_to_that_record": r0.get("gold_chunk_linked_to_that_record"),
                "merge_direction": r0.get("merge_direction"),
                "bridge": bool(r0.get("form2_in_that_record_final") and r0.get("gold_chunk_linked_to_that_record")) if r0 else None,
                "joined": (r0.get("merge_direction") in ("form2_into_form1", "form1_into_form2")) if r0 else None,
                "both_present": (r0.get("merge_direction") in ("form2_into_form1", "form1_into_form2", "never_joined")) if r0 else None,
            }
            q2 = run0_q2.get(p["persona_id"], {})
            at = q2.get("at_gold") if p["cell"] == 4 else q2.get("at_reveal")
            p["run0"]["earlier_record_in_prompt_at_link_step"] = (at.get("class") == "recalled") if at else None
            p["run0"]["earlier_record_existed"] = (at.get("class") != "no earlier record") if at else None

    write_jsonl(MEMORY_JSONL, personas)
    report.files_written.append(str(MEMORY_JSONL))
    write_jsonl(SAME_AS_JSONL, all_same_as)
    write_jsonl(TITLE_CONFLICTS_JSONL, all_conflicts)
    report.files_written += [str(SAME_AS_JSONL), str(TITLE_CONFLICTS_JSONL)]
    if replay_problems:
        report.unexpected.append("Ledger and construction records disagree: " + "; ".join(replay_problems))
    report.table("Books: chunk records, records, stubs, promotions, events", ["book", "chunk records", "records", "stubs", "promotions", "ledger events"], growth_rows)
    report.table("Description cap applied to the loaded index (ledger replacements removed before scoring)", ["book", "records with replacements", "descriptions removed"], drift_rows)

    # ---- per-cell outcomes next to run 0 ----
    def both(group, key, r0key=None):
        r1 = RU.rate_text(group, key)
        if not run0_mem:
            return RU.side_by_side(r1, None)
        vals = [p["run0"].get(r0key or key) for p in group]
        known = [v for v in vals if v is not None]
        return RU.side_by_side(r1, f"{sum(bool(v) for v in known)}/{len(known)} ({100.0 * sum(bool(v) for v in known) / len(known):.0f}%)" if known else "n/a")

    cells = sorted({p["cell"] for p in personas})
    rows = []
    for cell in cells:
        g = [p for p in personas if p["cell"] == cell]
        with_reveal = cell != 5
        rows.append(
            [
                f"{cell} ({CELL_LABELS.get(cell, '')})",
                len(g),
                both(g, "form1_present_after_gold", "form1_record_found"),
                both(g, "form2_present_after_reveal", "form2_in_that_record_final") if with_reveal else "-",
                both(g, "both_present_final", "both_present") if with_reveal else "-",
                both(g, "earlier_record_in_prompt_at_link_step", "earlier_record_in_prompt_at_link_step") if with_reveal else "-",
                both(g, "joined_when_both_present", "joined") if with_reveal else "-",
                both([{**p, "bridge": p["form2_in_that_record_final"] and p["gold_chunk_linked_to_that_record"]} for p in g], "bridge", "bridge") if with_reveal else "-",
            ]
        )
    report.table(
        "Memory outcomes per cell, run 1 (run 0 in parentheses)",
        ["cell", "n", "form 1 present after the gold chunk", "form 2 present after the reveal", "both present (final)", "earlier record in the prompt at the reveal (gold chunk for cell 4)", "joined when both present", "bridge formed (form 2 in the gold chunk's live record)"],
        rows,
    )
    report.add(
        "Reading the columns",
        "Run 0's 'form 1 present' is its step-6 'form1 record found' (final memory); run 0's 'both present' counts personas whose merge direction was not 'no_record' "
        "(both names in some record); run 0's 'in the prompt' is step 9's recall class at the reveal (the gold chunk for cell 4). Stubs count as records throughout.",
    )
    mrows = []
    for cell in cells:
        g = [p for p in personas if p["cell"] == cell]
        mrows.append([cell, {d: sum(1 for p in g if p["merge_direction"] == d) for d in sorted({p["merge_direction"] for p in g})}, {d: sum(1 for p in g if p["run0"].get("merge_direction") == d) for d in sorted({str(p["run0"].get("merge_direction")) for p in g})} if run0_mem else "n/a"])
    report.table("Merge directions per cell", ["cell", "run 1", "run 0"], mrows)
    frows = []
    for cell in cells:
        g = [p for p in personas if p["cell"] == cell and p["cell"] != 5]
        counts = Counter(p.get("form2_field") for p in g)
        frows.append([cell, counts.get("name", 0), counts.get("alias", 0), counts.get("description", 0), counts.get("none", 0), RU.rate_text(g, "form2_in_gold_frozen_record")])
    report.table("Where form 2 sits in the form-1 record (run 1)", ["cell", "name", "alias", "description", "none", "form 2 in gold FROZEN record"], frows)

    # Recall at the link step, in detail.
    with_reveal = [p for p in personas if p["cell"] in REVEAL_CELLS]
    existed = [p for p in with_reveal if p.get("earlier_record_existed_before_link_step")]
    in_prompt = [p for p in existed if p.get("earlier_record_in_prompt_at_link_step")]
    left_out = [p for p in existed if p.get("earlier_left_out_by_cap")]
    report.add(
        "Earlier record in the prompt where the link could form",
        f"Earlier record existed before the link step: {pct(len(existed), len(with_reveal))}; of those, in the prompt: {pct(len(in_prompt), len(existed))} "
        f"(run 0: 6 of 72); existed but left out by the ledger-recall cap: {len(left_out)}. Decision block status at the link step: "
        f"{dict(Counter(p.get('reveal_decision_status') if p['cell'] != 4 else p.get('gold_decision_status') for p in with_reveal))}.",
    )

    # The gate.
    gate_group = [p for p in personas if p["cell"] in GATE_CELLS]
    bridged = [p for p in gate_group if p["form2_in_that_record_final"] and p["gold_chunk_linked_to_that_record"]]
    gate_rate = len(bridged) / len(gate_group) if gate_group else 0.0
    verdict = f"The bridge word reached the gold chunk's record in {len(bridged)} of {len(gate_group)} personas of cells 1-3 ({100 * gate_rate:.0f}%); run 0: 1 of 90."
    if gate_rate < GATE_RATE:
        verdict += " **Below half: the retrieval numbers of step 4 must be read with that in mind.**"
    report.add("Gate", verdict)
    report.numbers.update({"gate_rate": gate_rate, "gate_n": len(gate_group), "gate_bridged": len(bridged)})

    # Provenance.
    ev = {k[3:]: v for k, v in prov_total.items() if k.startswith("ev:")}
    report.table(
        "Provenance of records and stubs (all books)",
        ["measure", "value"],
        [
            ["records: from the candidate net only / from the LLM only / from both", f"{prov_total.get('record:candidate_net', 0)} / {prov_total.get('record:llm', 0)} / {prov_total.get('record:both', 0)}"],
            ["stubs: from the candidate net (incl. parse-failure stubs)", prov_total.get("stub:candidate_net", 0)],
            ["entries recovered (memory name unknown to the ledger; should be 0)", prov_total.get("recovered", 0)],
            ["NOT AN ENTITY decisions", ev.get("not_an_entity", 0)],
            ["SAME AS decisions / unresolved", f"{ev.get('same_as', 0)} / {ev.get('same_as_unresolved', 0)}"],
            ["decision-block parse failures / candidates stubbed by that policy / omitted candidates", f"{ev.get('decision_block_parse_failures', 0)} / {ev.get('parse_failure_candidates', 0)} / {ev.get('omitted_candidates', 0)}"],
            ["promotions", ev.get("promotions", 0)],
            ["aliases added / rejected as descriptive", f"{ev.get('aliases_added', 0)} / {ev.get('alias_rejected_descriptive', 0)}"],
            ["described records named by a stored alias (kept under the stored name)", ev.get("described_name_was_alias", 0)],
            ["description replacements", ev.get("description_replacements", 0)],
        ],
    )
    report.numbers["provenance"] = dict(prov_total)

    # Merge audit.
    sample = RU.seeded_sample(all_same_as, 30)
    lines = [f"- {d['doc_id']} step {d['step']}: '{d['candidate']}' -> {d.get('resolved_name')!r} [{d.get('resolved_id')}], {d.get('outcome')}: {d.get('reason') or '(no reason given)'}" for d in sample]
    report.add(f"Merge audit: 30 of {len(all_same_as)} SAME AS decisions (seed {SEED}; all in same_as_decisions.jsonl)", "\n".join(lines) or "No SAME AS decision.")
    no_reason = sum(1 for d in all_same_as if not d.get("reason"))
    report.add(
        "Mr./Mrs. check on the books' real characters",
        f"Records whose names/aliases share a stripped name but carry different titles (persona names excluded): **{len(all_conflicts)}** (list in title_conflicts.jsonl). "
        f"SAME AS decisions without a reason: {no_reason} of {len(all_same_as)}. Run 0 had no SAME AS mechanism; step 9 counted 10 title-variant duplicate pairs among persona records.",
    )
    if all_conflicts:
        report.table("Up to ten title conflicts", ["book", "record", "aliases", "stripped name", "titles"], [[c["doc_id"], c["name"], c["aliases"][:6], c["stripped"], c["titles"]] for c in all_conflicts[:10]])
    report.numbers.update({"same_as_total": len(all_same_as), "same_as_no_reason": no_reason, "title_conflicts": len(all_conflicts)})

    # Five full ledger entries.
    entries = []
    for book in sorted(by_book):
        for e in store.doc(book).entries.values():
            entries.append((book, e))
    picks = RU.seeded_sample(entries, 5)
    lines = []
    for book, e in picks:
        lines.append(f"- **{book} {e.id} {e.name!r}** kind={e.kind} created step {e.created_step} ({e.created_chunk_id}) promoted={e.promoted_step} provenance={e.provenance} aliases={e.aliases} linked chunks={len(e.linked_chunks)} described at={e.described_steps} same_as={e.same_as_count} sentence={e.sentence[:160]!r}")
    report.add(f"Five full ledger entries (seed {SEED})", "\n".join(lines))

    status_counts = Counter()
    for p in personas:
        for k in ("gold_parser_status", "reveal_parser_status"):
            if p.get(k) is not None:
                status_counts[p[k]] += 1
    report.add("JSON parser status of chunks holding inserted sentences", str(dict(sorted(status_counts.items()))))
    report.add("Hand-off", "Paste this report back. Step 4 (retrieval) waits for it.")


if __name__ == "__main__":
    raise SystemExit(run_step(3, "Memory check with stubs", main, requires=[2]))
