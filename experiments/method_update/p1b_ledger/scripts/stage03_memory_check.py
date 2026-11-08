#!/usr/bin/env python3
"""Stage 03 (CPU): memory check per cell, run 1b next to run 1 and run 0.

First runs run 1's step 3 (p1_ledger/scripts/step03_memory_check.py, imported
unchanged) on run 1b's build: it writes outputs/method_update/p1b_ledger/
personas_memory.jsonl, same_as_decisions.jsonl and title_conflicts.jsonl, and its
own tables go to reports/step_03_run1_code.md. Then, from those rows and run 1's
(outputs/method_update/p1_ledger/personas_memory.jsonl, read only):

- per cell: form 1 after gold; form 2 after reveal; both present; earlier record
  in the prompt at the reveal; joined when both present; bridge formed;
- why the earlier record was not in the prompt (cut by the cap, not matched, no
  earlier record), join direction, the decision on form 2 at the reveal, and
  where form 1 sits when it went into a separate form-2 record;
- the counts of the three changes; the merge audit (30 final SAME AS, seed 42),
  title conflicts, alias examples.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_p1b as C  # noqa: E402

DIRECTIONS = ("form2_into_form1", "form1_into_form2", "never_joined", "no_record")
DIRECTION_LABELS = {"form2_into_form1": "into the gold record", "form1_into_form2": "into a new form-2 record", "never_joined": "never joined", "no_record": "no record"}
MEASURES = [
    # (label, key in run 1b / run 1 rows, key in the run-0 sub-row (as in run 1's step 3))
    ("form 1 after gold", "form1_present_after_gold", "form1_record_found"),
    ("form 2 after reveal", "form2_present_after_reveal", "form2_in_that_record_final"),
    ("both present (final)", "both_present_final", "both_present"),
    ("earlier record in prompt at the reveal", "earlier_record_in_prompt_at_link_step", "earlier_record_in_prompt_at_link_step"),
    ("joined when both present", "joined_when_both_present", "joined"),
    ("bridge formed", "bridge", "bridge"),
]


def bridge(p: dict) -> bool:
    return bool(p.get("form2_in_that_record_final") and p.get("gold_chunk_linked_to_that_record"))


def with_bridge(rows: list[dict]) -> list[dict]:
    return [{**p, "bridge": bridge(p)} for p in rows]


def why_not_in_prompt(p: dict) -> str:
    if not p.get("earlier_record_existed_before_link_step"):
        return "no earlier record"
    if p.get("earlier_record_in_prompt_at_link_step"):
        return "in the prompt"
    if p.get("earlier_left_out_by_cap"):
        return "cut by the cap"
    return "not matched by recall"


def why_not_run0(r0: dict) -> str | None:
    if r0.get("earlier_record_existed") is None:
        return None
    if not r0.get("earlier_record_existed"):
        return "no earlier record"
    return "in the prompt" if r0.get("earlier_record_in_prompt_at_link_step") else "not in the prompt"


def decision_on_form2(store, p: dict) -> str:
    """What happened to form 2 at the reveal: the model's decision and the final outcome."""

    from saadi.ledger.ledger import match_kind, stripped_tokens

    if p.get("reveal_chunk_index") is None or p["cell"] == 4:
        return "n/a"
    ledger = store.doc(p["book_id"])
    step = int(p["reveal_chunk_index"])
    ev = next((e for e in ledger.events if int(e["step"]) == step), None)
    if ev is None:
        return "no event"
    rows = [r for r in ev.get("decisions", []) if match_kind(r.get("candidate", ""), p["form2"]) in ("exact", "partial")]
    if not rows:
        return "form 2 not a candidate"
    row = rows[0]
    model = row.get("model_decision", row.get("decision")) or "omitted"
    final = row.get("decision")
    f1 = set(stripped_tokens(p["form1"]))
    target = ledger.get(row.get("resolved_id", "")) if row.get("resolved_id") else None
    into_form1 = bool(target and f1 and any(f1 <= set(stripped_tokens(n)) for n in target.all_names()))
    if final == "SAME_AS":
        how = "auto-join" if row.get("auto_join") else "JSON-alias join" if row.get("json_alias_join") else "model"
        return f"model {model} -> joined ({how}) {'into the form-1 record' if into_form1 else 'into another record'}"
    if row.get("rejected_target_id"):
        return f"model SAME_AS rejected ({row.get('acceptance')}) -> {row.get('outcome')}"
    return f"model {model} -> {row.get('outcome', final)}"


def form1_location(index, p: dict, step06) -> str | None:
    """Where form 1 sits in the separate form-2 record (only for the wrong direction)."""

    if p.get("merge_direction") != "form1_into_form2":
        return None
    rec = next((e for e in index.memory_entities if e.name == p.get("form2_record_name")), None)
    return step06.classify_form2(rec, C.content_token_set(p["form1"])) if rec is not None else "record not found"


def indexes_for(index_path, ledger_dir, books):
    from saadi.indexing.retrieval_methods.mbr import build_mbr_document_memory_index
    from saadi.ledger.ledger import LedgerStore
    from saadi.ledger.measure import apply_description_cap

    out = {}
    if index_path is None:
        return out
    store = LedgerStore(ledger_dir) if ledger_dir is not None else None
    for book in books:
        idx = build_mbr_document_memory_index(index_path, doc_id=book)
        if store is not None:
            apply_description_cap(idx, store.doc(book))
        out[book] = idx
    return out


def main(report: C.StepReport, argv: list[str]) -> None:
    from saadi.ledger.ledger import LedgerStore
    from saadi.ledger.measure import same_as_decisions

    s3 = C.load_p1_step("step03_memory_check")
    step06 = C.load_probe_step("step06_memory_check.py")
    detail = C.DetailReport(step=3, title="Memory check with stubs (run 1's step 3 code on run 1b)")
    s3.main(detail, argv)
    detail_path = C.write_detail(detail)
    report.files_written += [str(detail_path), *detail.files_written]
    report.unexpected += [f"(from run 1's step 3 code) {u}" for u in detail.unexpected if "not found under" not in u]

    rows1b = with_bridge(C.read_jsonl(C.OUT_DIR / "personas_memory.jsonl"))
    run1_path = C.run1_file("personas_memory.jsonl")
    rows1 = {p["persona_id"]: p for p in with_bridge(C.read_jsonl(run1_path))} if run1_path else {}
    if not rows1:
        report.unexpected.append(f"Run 1's personas_memory.jsonl was not found under {C.RUN1_OUT_DIR}; run 1 columns show n/a.")
    cells = sorted({p["cell"] for p in rows1b})

    def g1b(cell):
        return [p for p in rows1b if p["cell"] in cell]

    def g1(group):
        return [rows1[p["persona_id"]] for p in group if p["persona_id"] in rows1]

    def g0(group):
        return [p.get("run0", {}) for p in group]

    # ---- per-cell table ----
    table = []
    for cell in cells:
        g = g1b((cell,))
        row = [f"{cell} ({C.CELL_LABELS.get(cell, '')})", len(g)]
        for label, key, key0 in MEASURES:
            if cell == 5 and key != "form1_present_after_gold":
                row.append("-")
                continue
            row.append(C.three(C.rate_text(C.rate(g, key)), C.rate_text(C.rate(g1(g), key)) if rows1 else None, C.rate_text(C.rate(g0(g), key0))))
        table.append(row)
    report.table("Memory check per cell: run 1b / run 1 / run 0", ["cell", "n", *[m[0] for m in MEASURES]], table)
    report.add(
        "Reading the table",
        "Stubs count as records. Run 0 columns use the proxies of run 1's step 3 (form 2 after reveal = form 2 in the form-1 record; "
        "both present = some record holds both names; in prompt = step 9's recall class). Cell 7's form 2 is an occupation phrase: "
        "it lands in descriptions, so 'both present' is 0 there and the bridge column is the measure that counts.",
    )

    c123 = g1b((1, 2, 3))
    reveal = [p for p in rows1b if p["cell"] != 5]
    numbers: dict = {}
    for name, grp in (("cells_1_3", c123), ("reveal", reveal)):
        numbers[name] = {key: {"run1b": C.rate(grp, key), "run1": C.rate(g1(grp), key) if rows1 else None, "run0": C.rate(g0(grp), key0)} for _l, key, key0 in MEASURES}
    report.numbers["pooled"] = numbers

    # ---- join direction ----
    rows = []
    for label, grp in (("cells 1-3", c123), *[(f"cell {c}", g1b((c,))) for c in cells if c != 5]):
        d1b, d1, d0 = Counter(p.get("merge_direction") for p in grp), Counter(p.get("merge_direction") for p in g1(grp)), Counter(p.get("merge_direction") for p in g0(grp))
        has0 = any(p.get("merge_direction") for p in g0(grp))
        rows.append([label, *[C.three(d1b.get(d, 0), d1.get(d, 0) if rows1 else None, d0.get(d, 0) if has0 else None) for d in DIRECTIONS]])
    report.table("Join direction (run 1b / run 1 / run 0)", ["group", *[DIRECTION_LABELS[d] for d in DIRECTIONS]], rows)
    report.numbers["join_direction_cells_1_3"] = {"run1b": dict(Counter(p.get("merge_direction") for p in c123)), "run1": dict(Counter(p.get("merge_direction") for p in g1(c123))), "run0": dict(Counter(p.get("merge_direction") for p in g0(c123)))}

    # ---- why the earlier record was not in the prompt ----
    why1b, why1, why0 = Counter(why_not_in_prompt(p) for p in reveal), Counter(why_not_in_prompt(p) for p in g1(reveal)), Counter(x for x in (why_not_run0(r) for r in g0(reveal)) if x)
    keys = ["in the prompt", "cut by the cap", "not matched by recall", "no earlier record"]
    report.table(
        "Earlier record at the link step (the reveal; the gold chunk in cell 4), all cells with a reveal",
        ["outcome", "run 1b", "run 1", "run 0"],
        [[k, why1b.get(k, 0), why1.get(k, 0) if rows1 else "n/a", (why0.get(k, why0.get("not in the prompt", 0) if k == "not matched by recall" else 0) if why0 else "n/a")] for k in keys],
    )
    report.numbers["earlier_record"] = {"run1b": dict(why1b), "run1": dict(why1), "run0": dict(why0)}

    # ---- the decision on form 2 at the reveal ----
    store1b, store1 = LedgerStore(C.LEDGER_DIR), LedgerStore(C.RUN1_LEDGER_DIR)
    run1_ledgers = C.RUN1_LEDGER_DIR.exists()
    grp = [p for p in rows1b if p["cell"] in (1, 2, 3, 6, 8)]
    dec1b = Counter(decision_on_form2(store1b, p) for p in grp)
    dec1 = Counter(decision_on_form2(store1, p) for p in grp) if run1_ledgers else Counter()
    report.table(
        "Form 2 at the reveal, cells 1-3, 6 and 8 (n = %d): the model's decision and the final outcome" % len(grp),
        ["outcome", "run 1b", "run 1"],
        [[k, dec1b.get(k, 0), dec1.get(k, 0) if run1_ledgers else "n/a"] for k in sorted(set(dec1b) | set(dec1), key=lambda k: -(dec1b.get(k, 0) + dec1.get(k, 0)))],
    )
    report.numbers["form2_at_reveal"] = {"run1b": dict(dec1b), "run1": dict(dec1)}

    # ---- where form 1 sits in a separate form-2 record ----
    books = sorted({p["book_id"] for p in rows1b})
    idx1b = indexes_for(C.p1.find_run1_cache_index(), C.LEDGER_DIR, books)
    idx1 = indexes_for(C.run1_cache_index(), C.RUN1_LEDGER_DIR if run1_ledgers else None, books)
    loc1b = Counter(x for x in (form1_location(idx1b[p["book_id"]], p, step06) for p in rows1b if p["cell"] != 5) if x)
    loc1 = Counter(x for x in (form1_location(idx1[p["book_id"]], rows1[p["persona_id"]], step06) for p in rows1b if p["cell"] != 5 and p["persona_id"] in rows1 and idx1) if x)
    report.table(
        "Wrong direction: where form 1 sits in the separate form-2 record (all cells with a reveal)",
        ["field", "run 1b", "run 1"],
        [[k, loc1b.get(k, 0), loc1.get(k, 0) if idx1 else "n/a"] for k in ("name", "alias", "description", "none", "record not found")],
    )
    report.numbers["form1_location"] = {"run1b": dict(loc1b), "run1": dict(loc1)}

    # ---- the three changes ----
    tot = Counter()
    same_as_rows, rejected_rows = [], []
    for book in books:
        ledger = store1b.doc(book)
        tot.update({k: v for k, v in ledger.counters.items() if isinstance(v, int)})
        same_as_rows += same_as_decisions(ledger)
        for ev in ledger.events:
            for d in ev.get("decisions", []):
                if d.get("rejected_target_id"):
                    rejected_rows.append({"doc_id": book, "step": ev["step"], **d})
    report.table(
        "The three changes at work (counts over all books)",
        ["measure", "value"],
        [
            ["JSON records redirected into a SAME AS target", tot["redirected"]],
            ["auto-join (NEW equal to a shown entry) / blocked by a conflict / ambiguous", f"{tot['auto_join']} / {tot['auto_join_blocked_by_conflict']} / {tot['auto_join_ambiguous']}"],
            ["JSON-alias join (optional switch)", tot["json_alias_join"]],
            ["model SAME AS rejected, by reason", {k[17:]: v for k, v in tot.items() if k.startswith("same_as_rejected_")} or 0],
            ["model SAME AS accepted, by evidence", {k[17:]: v for k, v in tot.items() if k.startswith("same_as_accepted_")}],
            ["NEW folded into a described record, rejected for a conflict", tot["described_match_rejected_conflict"]],
            ["aliases rejected, by reason", {k[15:]: v for k, v in tot.items() if k.startswith("alias_rejected_")} or 0],
        ],
    )
    report.numbers["p1b_totals"] = dict(tot)

    # ---- merge audit ----
    def source(d: dict) -> str:
        return "auto-join" if d.get("auto_join") else "JSON-alias join" if d.get("json_alias_join") else f"model, {d.get('acceptance', '?')}"

    sample = C.RU.seeded_sample(same_as_rows, 30)
    lines = [f"- {d['doc_id']} step {d['step']}: '{d['candidate']}' -> '{d.get('resolved_name')}' [{d.get('resolved_id')}] ({source(d)}): {d.get('reason') or '(no reason)'}" for d in sample]
    report.add(f"Merge audit: 30 of {len(same_as_rows)} final SAME AS decisions (seed {C.SEED})", "\n".join(lines) or "None.")
    rej = C.RU.seeded_sample(rejected_rows, 10)
    report.add(
        f"Ten of {len(rejected_rows)} rejected SAME AS (seed {C.SEED})",
        "\n".join(f"- {d['doc_id']} step {d['step']}: '{d['candidate']}' -> '{d.get('rejected_target_name')}' rejected: {d.get('acceptance')} {d.get('acceptance_detail', '')}; model's reason: {d.get('reason') or '-'}" for d in rej) or "None.",
    )
    run1_s3 = C.run1_report_numbers(3)
    report.add(
        "Mr./Mrs. check and wrong merges",
        f"Records joining one stripped name under different titles (books' real characters): **{detail.numbers.get('title_conflicts')}** (run 1: {run1_s3.get('title_conflicts', 'n/a')}). "
        "Whether a sampled SAME AS is wrong is a reading, not a count the code can make: run 1's sample of 30 was read as 7 to 8 wrong; the sample above needs the same reading.",
    )
    report.numbers.update({"same_as_total": len(same_as_rows), "rejected_total": len(rejected_rows), "title_conflicts": detail.numbers.get("title_conflicts"), "title_conflicts_run1": run1_s3.get("title_conflicts"), "audit_sample": sample})

    # ---- alias statistics and examples ----
    def alias_stats(idxs):
        counts = [len(v) for idx in idxs.values() for v in idx.entity_to_aliases.values()]
        return (sum(counts), max(counts) if counts else 0, sum(counts) / max(1, len(counts)))

    a1b, a1 = alias_stats(idx1b), alias_stats(idx1) if idx1 else None
    top = sorted(((book, name, aliases) for book, idx in idx1b.items() for name, aliases in idx.entity_to_aliases.items()), key=lambda r: -len(r[2]))[:10]
    report.add(
        "Aliases in memory",
        f"Aliases in total / most on one record / mean per record: run 1b {a1b[0]} / {a1b[1]} / {a1b[2]:.2f}; run 1 "
        + (f"{a1[0]} / {a1[1]} / {a1[2]:.2f}" if a1 else "n/a")
        + ". The ten records with the most aliases in run 1b (for a reader to judge pollution):\n\n"
        + "\n".join(f"- {b} '{n}': {a}" for b, n, a in top),
    )
    report.numbers["aliases"] = {"run1b": a1b, "run1": a1}


if __name__ == "__main__":
    raise SystemExit(C.run_stage(3, "Memory check (run 1b / run 1 / run 0)", main, requires=[2]))
