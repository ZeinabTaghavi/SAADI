#!/usr/bin/env python3
"""Step 5 (CPU): the final tables and the report text for the Google Doc.

Writes results/method_update/p1_ledger/{README.md, table.md, table.csv,
REPORT_FOR_GDOC.md}. REPORT_FOR_GDOC.md has sections R1 to R5 exactly as the
sub-tab "3.1 Issue 1: Report" lists them, plus the one-line Result and
Conclusion for run 1 of the tracker table, and the expected-against-observed
table. Everything is assembled from the step 1-4 reports (JSON twins) and the
per-persona files; nothing is recomputed with different settings.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_utils as RU  # noqa: E402
from common import CELL_LABELS, GATE_DIR, OUT_DIR, REPORT_DIR, RESULTS_DIR, SPEC_FROZEN_JSON, StepReport, md_table, read_json, read_jsonl, run_step, software_versions  # noqa: E402

PER_PERSONA = OUT_DIR / "per_persona.jsonl"
METHODS = RU.METHODS


def step_numbers(step: int) -> dict:
    path = REPORT_DIR / f"step_{step:02d}.json"
    return read_json(path).get("numbers", {}) if path.exists() else {}


def main(report: StepReport, argv: list[str]) -> None:
    personas = read_jsonl(PER_PERSONA)
    s1, s2, s3, s4 = (step_numbers(i) for i in (1, 2, 3, 4))
    frozen_spec = read_json(SPEC_FROZEN_JSON)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cells = sorted({p["cell"] for p in personas})
    has_run0 = any(p.get("run0", {}).get("rank_live") is not None for p in personas)

    def group(cell):
        return [p for p in personas if p["cell"] == cell]

    def r0group(cell):
        return [p["run0"] for p in group(cell) if p.get("run0", {}).get("rank_live") is not None]

    # ---- R4 retrieval table ----
    main_rows, csv_rows = [], []
    for cell in cells:
        g, g0 = group(cell), r0group(cell)
        cols = []
        for m in METHODS:
            r1 = f"{RU.hr(g, m):.0f} / {RU.median_rank(g, f'rank_{m}'):.0f}"
            r0 = f"{RU.hr(g0, m):.0f} / {RU.median_rank(g0, f'rank_{m}'):.0f}" if g0 else None
            cols.append(RU.side_by_side(r1, r0))
        main_rows.append([f"{cell} ({CELL_LABELS[cell]})", len(g), *cols, RU.paired_text(g, "live", "frozen")])
        for m in METHODS:
            csv_rows.append({"run": "run1_p1_ledger", "cell": cell, "n": len(g), "method": m, "hr5": round(RU.hr(g, m, "5"), 1), "hr10": round(RU.hr(g, m), 1), "median_rank": RU.median_rank(g, f"rank_{m}")})
            if g0:
                csv_rows.append({"run": "run0_published", "cell": cell, "n": len(g0), "method": m, "hr5": round(RU.hr(g0, m, "5"), 1), "hr10": round(RU.hr(g0, m), 1), "median_rank": RU.median_rank(g0, f"rank_{m}")})
        d = RU.paired_numbers(g, "live", "frozen")
        csv_rows.append({"run": "run1_p1_ledger", "cell": cell, "n": len(g), "method": "live_minus_frozen_hr10_pp", "hr10": round(d["mean_pp"], 2), "hr5": None, "median_rank": f"CI [{d['ci_lo']:+.2f},{d['ci_hi']:+.2f}] McNemar p={d['mcnemar_p']:.4f}"})
    retrieval_table = md_table(["Cell", "n", *[RU.METHOD_LABELS[m] + " HR@10 / median rank" for m in METHODS], "Live - Frozen @10"], main_rows)

    # ---- R3 memory table ----
    def rate_pair(g, key, r0key):
        r1 = RU.rate_text(g, key)
        vals = [p.get("run0", {}).get(r0key) for p in g]
        known = [v for v in vals if v is not None]
        r0 = f"{sum(bool(v) for v in known)}/{len(known)} ({100.0 * sum(bool(v) for v in known) / len(known):.0f}%)" if known else None
        return RU.side_by_side(r1, r0)

    mem_rows = []
    for cell in cells:
        g = group(cell)
        bridged = [{**p, "bridge": bool(p.get("form2_in_that_record_final") and p.get("gold_chunk_linked_to_that_record"))} for p in g]
        if cell == 5:
            mem_rows.append([f"{cell} ({CELL_LABELS[cell]})", len(g), rate_pair(g, "form1_present_after_gold", "form1_record_found"), "-", "-", "-", "-"])
            continue
        mem_rows.append([f"{cell} ({CELL_LABELS[cell]})", len(g), rate_pair(g, "both_present_final", "both_present"), rate_pair(g, "earlier_record_in_prompt_at_link_step", "earlier_record_in_prompt_at_link_step"), rate_pair(g, "joined_when_both_present", "joined"), rate_pair(bridged, "bridge", "bridge"), rate_pair(g, "form1_present_after_gold", "form1_record_found")])
    memory_table = md_table(["Cell", "n", "both names present", "earlier record in prompt at the reveal", "joined when both present", "bridge formed", "form 1 present after gold"], mem_rows)

    # ---- expected against observed ----
    reveal = [p for p in personas if p["cell"] != 5]
    c123 = [p for p in personas if p["cell"] in (1, 2, 3)]
    both_n = sum(bool(p.get("both_present_final")) for p in reveal)
    both0 = [p.get("run0", {}).get("both_present") for p in reveal]
    both0_n, both0_d = sum(bool(v) for v in both0 if v is not None), sum(1 for v in both0 if v is not None)
    prompt_n = sum(bool(p.get("earlier_record_in_prompt_at_link_step")) for p in reveal if p.get("earlier_record_existed_before_link_step"))
    prompt_d = sum(1 for p in reveal if p.get("earlier_record_existed_before_link_step"))
    bridge_n = sum(bool(p.get("form2_in_that_record_final") and p.get("gold_chunk_linked_to_that_record")) for p in c123)
    c5_zero = s4.get("cell5_zero")
    calls_ok = not any("Calls per book differ" in u for u in read_json(REPORT_DIR / "step_02.json").get("unexpected", [])) if (REPORT_DIR / "step_02.json").exists() else None
    eo_rows = [
        ["both names present (cells with a reveal)", "from ~30% to >80%", f"{both_n}/{len(reveal)} ({100.0 * both_n / max(1, len(reveal)):.0f}%); run 0 {both0_n}/{both0_d} ({100.0 * both0_n / max(1, both0_d):.0f}%)" if both0_d else f"{both_n}/{len(reveal)} ({100.0 * both_n / max(1, len(reveal)):.0f}%)", "held" if both_n / max(1, len(reveal)) > 0.8 else "not held"],
        ["earlier record in the prompt at the reveal (when it existed)", "from 6 of 72 to most", f"{prompt_n}/{prompt_d} ({100.0 * prompt_n / max(1, prompt_d):.0f}%)", "held" if prompt_d and prompt_n / prompt_d > 0.5 else "not held"],
        ["bridge formed, cells 1-3", "from 1 of 90 to a clear majority, if the join works", f"{bridge_n}/{len(c123)} ({100.0 * bridge_n / max(1, len(c123)):.0f}%)", "held" if bridge_n / max(1, len(c123)) > 0.5 else "not held"],
        ["cell 5 stays at zero (live HR@10)", "0", f"{RU.hr(group(5), 'live'):.0f}%", "held" if c5_zero else "not held"],
        ["calls per book unchanged", "equal to run 0", "equal" if calls_ok else ("differ" if calls_ok is False else "n/a"), "held" if calls_ok else "not held"],
    ]
    eo_table = md_table(["Quantity", "Expected", "Observed", "Verdict"], eo_rows)

    # ---- R5 sentences (templated from the numbers) ----
    lf = s4.get("live_minus_frozen_cells_1_3", {})
    link_sentence = f"The link formed in {bridge_n} of {len(c123)} characters of cells 1-3 (run 0: 1 of 90); both names were present for {both_n} of {len(reveal)} characters with a reveal, and the earlier record was in the prompt at the reveal in {prompt_n} of {prompt_d} cases where it existed."
    if lf:
        follow = "followed" if lf.get("ci_lo", 0) > 0 else ("did not follow" if lf.get("ci_hi", 0) <= 0 else "did not clearly follow")
        retrieval_sentence = (
            f"Retrieval {follow}: in cells 1-3 live HR@10 is {RU.hr(c123, 'live'):.0f}% against frozen {RU.hr(c123, 'frozen'):.0f}% "
            f"(live minus frozen {lf.get('mean_pp', float('nan')):+.1f} pp, 95% CI [{lf.get('ci_lo', float('nan')):+.1f}, {lf.get('ci_hi', float('nan')):+.1f}], "
            f"McNemar p={lf.get('mcnemar_p', float('nan')):.3f})"
        )
        if has_run0:
            retrieval_sentence += f"; run 0 live was {RU.hr([p['run0'] for p in c123 if p['run0'].get('rank_live') is not None], 'live'):.0f}%."
        else:
            retrieval_sentence += "."
    else:
        retrieval_sentence = "Retrieval numbers are missing (step 4 did not finish)."
    tok = s2.get("tokens", {})
    ls = s2.get("ledger_stats", {})
    n_records = sum(v.get("records", 0) for v in ls.values())
    n_stubs = sum(v.get("stubs", 0) for v in ls.values())
    cost_sentence = (
        f"It cost {tok.get('run1_input_mean', float('nan')):.0f} input and {tok.get('run1_output_mean', float('nan')):.0f} output tokens per call "
        f"(run 0: {tok.get('run0_input_mean', float('nan')):.0f} and {tok.get('run0_output_mean', float('nan')):.0f}), {s2.get('wall_clock_s_total', 0) / 3600:.2f} h of wall clock on 4 GPUs "
        f"(run 0: 3.36 h), the same number of calls, and {n_records} records plus {n_stubs} stubs over the 10 books (run 0: records only)."
    )
    problems = []
    pf = sum(v.get("decision_parse_failures", 0) for v in ls.values())
    if pf:
        problems.append(f"{pf} decision blocks could not be parsed (every candidate of those chunks became a stub).")
    cap_hits = sum(v.get("ledger_cap_hits", 0) for v in ls.values())
    if cap_hits:
        problems.append(f"The ledger-recall cap of 5 was hit at {cap_hits} chunks; what was left out is logged per chunk.")
    if s3.get("title_conflicts"):
        problems.append(f"{s3['title_conflicts']} records join names that differ only by a title (Mr./Mrs. type); see title_conflicts.jsonl.")
    if s3.get("same_as_no_reason"):
        problems.append(f"{s3['same_as_no_reason']} of {s3.get('same_as_total', 0)} SAME AS decisions carry no reason.")
    if not s4.get("cell4_not_below_cells_1_3", True):
        problems.append("Cell 4 (reveal before gold) is below cells 1-3 at live HR@10.")
    if not c5_zero:
        problems.append("Cell 5 (no reveal) is not at zero: a query word reached a gold chunk without a reveal.")
    unresolved = s3.get("provenance", {}).get("ev:same_as_unresolved", 0)
    if unresolved:
        problems.append(f"{unresolved} SAME AS targets could not be resolved to a shown record (treated as NEW).")
    if bridge_n and RU.hr(c123, "live") < 50:
        problems.append("Links form but the text channel still wins in most cases: the fusion weights (fixed here) are the next limit.")
    if not problems:
        problems.append("None exposed by the counts above.")

    gate_passed = s1.get("gate_passed")
    tracker_result = f"Bridge {bridge_n} of {len(c123)}; live HR@10 cells 1-3 {RU.hr(c123, 'live'):.0f}% vs frozen {RU.hr(c123, 'frozen'):.0f}%; both names present {both_n}/{len(reveal)}; in prompt at reveal {prompt_n}/{prompt_d}"
    tracker_conclusion = (link_sentence.split(";")[0] + ". " + (retrieval_sentence.split(":")[0] + "." if retrieval_sentence else "")).strip()

    versions = software_versions()
    gdoc = f"""# Issue 1 report: profile p1_ledger

Filled by the coding agent from the step reports (experiments/method_update/p1_ledger/reports/step_01..05). Numbers next to run 0 values; headings as in the tab.

## R1. Candidate net gate (step 1)

Model chosen: `{s1.get('candidate_net', {}).get('model', 'n/a')}` ({s1.get('candidate_net', {}).get('device', '')}; labels {s1.get('candidate_net', {}).get('labels', [])}; threshold {s1.get('candidate_net', {}).get('threshold', '')}). Alternatives considered and the reason for the choice are in step 1's report (table "Candidate-net models considered").
Names caught on the probe books: form 1 {s1.get('form1_caught', 'n/a')}/{s1.get('form1_n', 'n/a')} ({100.0 * s1.get('form1_rate', 0):.0f}%), form 2 {s1.get('form2_caught', 'n/a')}/{s1.get('form2_n', 'n/a')} ({100.0 * s1.get('form2_rate', 0):.0f}%); gate {'passed' if gate_passed else 'NOT passed'} (floor 80% on form 1).
Time per chunk: {s1.get('seconds_per_chunk_mean', float('nan')):.2f} s on CPU ({s1.get('candidates_per_chunk_mean', float('nan')):.1f} candidates per chunk). In the build: {sum(v.get('candidate_seconds_mean', 0) for v in ls.values()) / max(1, len(ls)):.3f} s per chunk on {next(iter(ls.values())).get('candidate_device', '') if ls else ''}.
The exact prompt text of the LLM call: `{GATE_DIR / 'system_prompt.txt'}` and one filled example in `{GATE_DIR / 'example_user_prompt.txt'}` (both printed verbatim in step 1's report).

## R2. Build (step 2)

Books built: {s2.get('books_finished', 'n/a')} of 10. Calls per book: {'equal to run 0' if calls_ok else ('DIFFER from run 0' if calls_ok is False else 'run 0 unavailable for comparison')} (per-book table in step 2's report).
Tokens per call: input {tok.get('run1_input_mean', float('nan')):.0f} (run 0: {tok.get('run0_input_mean', float('nan')):.0f}), output {tok.get('run1_output_mean', float('nan')):.0f} (run 0: {tok.get('run0_output_mean', float('nan')):.0f}).
Wall clock: {s2.get('wall_clock_s_total', 0) / 3600:.2f} h over {len(s2.get('invocations', []))} invocation(s) (run 0: 3.36 h).
Decision-block parse failures: {pf}; ledger-recall cap hits: {cap_hits} chunks; records {n_records}, stubs {n_stubs}, promotions {sum(v.get('promotions', 0) for v in ls.values())}.

## R3. Memory check (step 3)

{memory_table}

Provenance: {s3.get('provenance', {}).get('record:candidate_net', 0)} records from the candidate net only, {s3.get('provenance', {}).get('record:llm', 0)} from the LLM only, {s3.get('provenance', {}).get('record:both', 0)} from both; {s3.get('provenance', {}).get('stub:candidate_net', 0)} stubs; NOT AN ENTITY decisions {s3.get('provenance', {}).get('ev:not_an_entity', 0)}; SAME AS decisions {s3.get('same_as_total', 0)}.
Merge audit: a 30-decision sample (seed 42) is in step 3's report; all decisions with reasons in `{OUT_DIR / 'same_as_decisions.jsonl'}`. Mr./Mrs. type joins on the books' real characters: {s3.get('title_conflicts', 'n/a')}.

## R4. Retrieval (step 4)

{retrieval_table}

Cell 5 at zero: {'yes' if c5_zero else 'NO'}. Cell 4 not below cells 1-3: {'yes' if s4.get('cell4_not_below_cells_1_3') else 'NO'}. Cells 1-3 pooled live minus frozen @10: {lf.get('mean_pp', float('nan')):+.1f} pp [{lf.get('ci_lo', float('nan')):+.1f}, {lf.get('ci_hi', float('nan')):+.1f}], McNemar p={lf.get('mcnemar_p', float('nan')):.3f}.

## R5. Conclusion (step 5)

{link_sentence} {retrieval_sentence} {cost_sentence}

Open problems this run exposed:
{chr(10).join('- ' + x for x in problems)}

## Expected against observed

{eo_table}

## Tracker table, run 1 (paste into the Result and Conclusion cells)

Result: {tracker_result}
Conclusion: {tracker_conclusion}
"""
    (RESULTS_DIR / "REPORT_FOR_GDOC.md").write_text(gdoc, encoding="utf-8")

    readme = f"""# p1_ledger: candidate-first ledger on the persona probe (run 1 of the method update tracker)

{link_sentence}

{retrieval_sentence}

{cost_sentence}

## Retrieval (HR@10 % / median gold rank; run 0 in parentheses; 95% bootstrap CIs, 10,000 resamples, seed 42; exact McNemar)

{retrieval_table}

## Memory check (stubs count as records; run 0 in parentheses)

{memory_table}

## Expected against observed

{eo_table}

## Open problems

{chr(10).join('- ' + x for x in problems)}

## Reproduction

- Spec hash (personas.jsonl, frozen before indexing): `{frozen_spec['personas_jsonl_sha256']}`
- Profile: `configs/profiles/p1_ledger.yaml` (published base + ledger); code in `src/saadi/ledger/`
- Commands: `bash experiments/method_update/p1_ledger/sh/01_gate.sh` ... `05_report.sh`, in order; step 2 on GPUs 0-3
- Versions: {", ".join(f"{k}={v}" for k, v in versions.items())}
- Date: {time.strftime("%Y-%m-%d %H:%M:%S")}
"""
    (RESULTS_DIR / "README.md").write_text(readme, encoding="utf-8")
    (RESULTS_DIR / "table.md").write_text("# p1_ledger tables\n\n" + retrieval_table + "\n\n" + memory_table + "\n\n" + eo_table + "\n", encoding="utf-8")
    with (RESULTS_DIR / "table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "cell", "n", "method", "hr5", "hr10", "median_rank"])
        writer.writeheader()
        writer.writerows(csv_rows)
    report.files_written += [str(RESULTS_DIR / n) for n in ("README.md", "table.md", "table.csv", "REPORT_FOR_GDOC.md")]
    report.commands.append("assemble results/method_update/p1_ledger from the step 1-4 JSON twins and per_persona.jsonl")
    report.add("REPORT_FOR_GDOC.md (verbatim; paste into the tab '3.1 Issue 1: Report' and the run 1 cells of the tracker)", gdoc)
    report.numbers.update({"tracker_result": tracker_result, "tracker_conclusion": tracker_conclusion, "problems": problems, "bridge_cells_1_3": [bridge_n, len(c123)], "both_present": [both_n, len(reveal)], "in_prompt": [prompt_n, prompt_d]})


if __name__ == "__main__":
    raise SystemExit(run_step(5, "Final report", main, requires=[3, 4]))
