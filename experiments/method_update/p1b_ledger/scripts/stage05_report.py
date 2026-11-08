#!/usr/bin/env python3
"""Stage 05 (CPU): reports/FINAL.md, results/.../REPORT_FOR_GDOC.md, the tracker row "1b", table.md, table.csv.

Everything is read from the stage reports (reports/step_0N.json) and the
per-persona rows; numbers are written as they came out, next to run 1 and run 0.
The expected movement is fixed text written before the numbers.
"""

from __future__ import annotations

import csv
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_p1b as C  # noqa: E402

FINAL_MD = C.REPORT_DIR / "FINAL.md"
GDOC_MD = C.RESULTS_DIR / "REPORT_FOR_GDOC.md"
TRACKER_MD = C.RESULTS_DIR / "TRACKER_ROW_1b.md"
TABLE_MD = C.RESULTS_DIR / "table.md"
TABLE_CSV = C.RESULTS_DIR / "table.csv"

EXPECTED = [
    ("Bridge formed, cells 1-3", "from 14 of 90 to above 40 of 90"),
    ("Join into the gold record (cells 1-3)", "from 24 of 49 joins to most joins"),
    ("Earlier record in the prompt at the reveal", "from 82 of 159 to above 130 of 159"),
    ("Wrong SAME AS in the audit sample of 30", "from 7-8 of 30 to under 2 of 30"),
    ("Retrieval (live HR@10, cells 1-3)", "probably still flat at fixed weights; the fusion landscape shows whether the bridges would be retrievable at a higher memory weight"),
    ("Cell 5 (no reveal)", "stays at zero"),
    ("Calls per book", "unchanged (equal to run 1)"),
]


def numbers(step: int) -> dict:
    path = C.REPORT_DIR / f"step_{step:02d}.json"
    return C.read_json(path).get("numbers", {}) if path.exists() else {}


def status(step: int) -> str:
    path = C.REPORT_DIR / f"step_{step:02d}.json"
    return C.read_json(path).get("status", "missing") if path.exists() else "missing"


def section(md: str, heading: str) -> str:
    """The body of '### heading' in a stage report (first heading that starts with the text)."""

    m = re.search(r"^### " + re.escape(heading) + r"[^\n]*\n\n(.*?)(?=\n### |\n## |\Z)", md, flags=re.DOTALL | re.MULTILINE)
    return m.group(1).strip() if m else f"(section '{heading}' not found)"


def frac(r) -> str:
    if not r:
        return "n/a"
    n, d = r
    return f"{n}/{d}"


def main(report: C.StepReport, argv: list[str]) -> None:
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    C.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    s1, s2, s3, s4 = numbers(1), numbers(2), numbers(3), numbers(4)
    md = {k: (C.REPORT_DIR / f"step_{k:02d}.md").read_text(encoding="utf-8") if (C.REPORT_DIR / f"step_{k:02d}.md").exists() else "" for k in (1, 2, 3, 4)}
    personas = C.read_jsonl(C.OUT_DIR / "per_persona.jsonl")
    for p in personas:
        p["bridge"] = bool(p.get("form2_in_that_record_final") and p.get("gold_chunk_linked_to_that_record"))
    c123 = [p for p in personas if p["cell"] in (1, 2, 3)]
    pooled = s3.get("pooled", {})
    br = pooled.get("cells_1_3", {}).get("bridge", {})
    prompt = s3.get("earlier_record", {})
    jd = s3.get("join_direction_cells_1_3", {})
    joins1b = jd.get("run1b", {}).get("form2_into_form1", 0) + jd.get("run1b", {}).get("form1_into_form2", 0)
    into1b = jd.get("run1b", {}).get("form2_into_form1", 0)
    joins1 = jd.get("run1", {}).get("form2_into_form1", 0) + jd.get("run1", {}).get("form1_into_form2", 0)
    into1 = jd.get("run1", {}).get("form2_into_form1", 0)

    def in_prompt(run):
        d = prompt.get(run, {})
        existed = sum(v for k, v in d.items() if k not in ("no earlier record",))
        return d.get("in the prompt", 0), existed

    ip1b, ex1b = in_prompt("run1b")
    ip1, ex1 = in_prompt("run1")
    bridge_n, bridge_d = br.get("run1b") or (0, 90)
    hr = s4.get("hr10_live_cells_1_3", {})
    hrf = s4.get("hr10_frozen_cells_1_3", {})
    lf = s4.get("live_minus_frozen_cells_1_3", {})
    calls_equal = not any("Calls per book differ" in u for u in C.read_json(C.REPORT_DIR / "step_02.json").get("unexpected", [])) if (C.REPORT_DIR / "step_02.json").exists() else None
    observed = [
        f"{bridge_n}/{bridge_d} (run 1: {frac(br.get('run1'))}; run 0: {frac(br.get('run0'))})",
        f"{into1b} of {joins1b} joins into the gold record, {jd.get('run1b', {}).get('form1_into_form2', 0)} into a new form-2 record (run 1: {into1} of {joins1})",
        f"{ip1b}/{ex1b} (run 1: {ip1}/{ex1})",
        "to be read: the 30 sampled decisions are listed below; the code cannot judge a merge",
        f"live {hr.get('run1b', float('nan')):.0f}% against frozen {hrf.get('run1b', float('nan')):.0f}% (run 1: {hr.get('run1') or float('nan'):.0f}% / {hrf.get('run1') or float('nan'):.0f}%); live minus frozen {lf.get('mean_pp', float('nan')):+.1f} pp [{lf.get('ci_lo', float('nan')):+.1f}, {lf.get('ci_hi', float('nan')):+.1f}], McNemar p={lf.get('mcnemar_p', float('nan')):.3f}",
        "zero" if s4.get("cell5_zero") else "NOT zero",
        "equal" if calls_equal else ("differ" if calls_equal is False else "n/a"),
    ]
    verdicts = [
        "held" if bridge_n > 40 else "not held",
        "held" if joins1b and into1b > joins1b / 2 else "not held",
        "held" if ip1b > 130 else "not held",
        "to be read",
        "flat" if lf and lf.get("ci_lo", 0) <= 0 else "moved",
        "held" if s4.get("cell5_zero") else "not held",
        "held" if calls_equal else "not held",
    ]
    eo_table = C.md_table(["Quantity", "Expected (written before the run)", "Observed", "Verdict"], [[q, e, o, v] for (q, e), o, v in zip(EXPECTED, observed, verdicts, strict=True)])

    tok = s2.get("tokens", {})
    tt = s2.get("p1b_totals", {})
    st = s2.get("ledger_stats", {})
    rec1b = sum(v.get("records", 0) for v in st.values())
    stub1b = sum(v.get("stubs", 0) for v in st.values())
    caps = sum(v.get("ledger_cap_hits", 0) for v in st.values())
    events = sum(v.get("events", 0) for v in st.values()) or 1
    secs = s2.get("wall_clock_s_total", 0) / 3600
    run1_s2 = C.run1_report_numbers(2)
    st1 = run1_s2.get("ledger_stats", {})
    cost = C.md_table(
        ["Measure", "run 1b", "run 1", "run 0"],
        [
            ["calls per book", "equal to run 1" if calls_equal else "DIFFER (see stage 02)", "equal to run 0", "-"],
            ["input tokens per call (mean of book means)", f"{tok.get('run1b_input_mean', float('nan')):.0f}", f"{tok.get('run1_input_mean') or float('nan'):.0f}", f"{tok.get('run0_input_mean') or float('nan'):.0f}"],
            ["output tokens per call", f"{tok.get('run1b_output_mean', float('nan')):.0f}", f"{tok.get('run1_output_mean') or float('nan'):.0f}", f"{tok.get('run0_output_mean') or float('nan'):.0f}"],
            ["replies at the output cap with no JSON", tok.get("likely_truncated", "n/a"), run1_s2.get("likely_truncated_replies", "n/a"), "-"],
            ["records / stubs (10 books)", f"{rec1b} / {stub1b}", f"{sum(v.get('records', 0) for v in st1.values())} / {sum(v.get('stubs', 0) for v in st1.values())}", "records only"],
            ["recall cap hits (chunks)", f"{caps} ({100.0 * caps / events:.0f}%, cap 8)", f"{sum(v.get('ledger_cap_hits', 0) for v in st1.values())} (cap 5)", "-"],
            ["wall clock on 4 GPUs", f"{secs:.2f} h", f"{run1_s2.get('wall_clock_s_total', 0) / 3600:.2f} h", "3.36 h"],
            ["candidate-net seconds per chunk (GPU, in the build)", f"{sum(v.get('candidate_seconds_mean', 0) for v in st.values()) / max(1, len(st)):.3f}", "0.082", "-"],
            ["candidate net, threshold / candidates per chunk", f"0.3 / {sum(v.get('candidates_per_chunk', 0) for v in st.values()) / max(1, len(st)):.1f}", f"0.5 / {sum(v.get('candidates_per_chunk', 0) for v in st1.values()) / max(1, len(st1)):.1f}", "-"],
        ],
    )

    link_sentence = (
        f"The link formed in {bridge_n} of {bridge_d} characters of cells 1-3 (run 1: {frac(br.get('run1'))}, run 0: {frac(br.get('run0'))}); "
        f"{into1b} of {joins1b} joins went into the gold record, and the earlier record was in the prompt at the reveal in {ip1b} of {ex1b} cases where it existed (run 1: {ip1} of {ex1})."
    )
    follow = "did not clearly follow" if lf.get("ci_lo", 0) <= 0 else "followed"
    retrieval_sentence = f"Retrieval {follow}: live HR@10 in cells 1-3 is {hr.get('run1b', float('nan')):.0f}% against frozen {hrf.get('run1b', float('nan')):.0f}% (live minus frozen {lf.get('mean_pp', float('nan')):+.1f} pp, 95% CI [{lf.get('ci_lo', float('nan')):+.1f}, {lf.get('ci_hi', float('nan')):+.1f}], McNemar p={lf.get('mcnemar_p', float('nan')):.3f}); run 1 live was {hr.get('run1') or float('nan'):.0f}%."
    cost_sentence = f"It cost {tok.get('run1b_input_mean', float('nan')):.0f} input and {tok.get('run1b_output_mean', float('nan')):.0f} output tokens per call (run 1: {tok.get('run1_input_mean') or float('nan'):.0f} and {tok.get('run1_output_mean') or float('nan'):.0f}), {secs:.2f} h on 4 GPUs (run 1: {run1_s2.get('wall_clock_s_total', 0) / 3600:.2f} h), and the same number of calls."

    problems = []
    loc = s3.get("form1_location", {}).get("run1b", {})
    if jd.get("run1b", {}).get("form1_into_form2", 0):
        problems.append(f"{jd['run1b']['form1_into_form2']} joins in cells 1-3 still put form 1 into a separate form-2 record; form 1 sits there as {loc}. " + ("The optional JSON-alias join was off." if not s2.get("json_alias_join") else "The JSON-alias join was on."))
    why = prompt.get("run1b", {})
    if why.get("cut by the cap") or why.get("not matched by recall"):
        problems.append(f"Earlier record not in the prompt at the link step: {why.get('cut by the cap', 0)} cut by the cap of 8, {why.get('not matched by recall', 0)} not matched by recall.")
    rej = {k[17:]: v for k, v in tt.items() if k.startswith("same_as_rejected_")}
    if rej:
        problems.append(f"SAME AS rejected by rule 3: {rej}; the ten-row sample in stage 03 shows whether the rule cut true joins.")
    if s3.get("title_conflicts"):
        problems.append(f"{s3['title_conflicts']} records still join one stripped name under different titles (run 1: {s3.get('title_conflicts_run1', 'n/a')}).")
    if tok.get("likely_truncated"):
        problems.append(f"{tok['likely_truncated']} replies hit the output cap with no JSON (more candidates at 0.3 lengthen the decision block).")
    if not s4.get("cell4_not_below_cells_1_3", True):
        problems.append("Cell 4 is below cells 1-3 at live HR@10.")
    if not s4.get("cell5_zero", True):
        problems.append("Cell 5 is not at zero.")
    problems.append("Fusion: the memory channel's ranking does not reach the top 10 at the published weight; the landscape above is information only (Issue 6).")

    sample = s3.get("audit_sample", [])
    audit_lines = [f"{i + 1}. {d['doc_id']} step {d['step']}: '{d['candidate']}' -> '{d.get('resolved_name')}' ({'auto-join' if d.get('auto_join') else 'JSON-alias join' if d.get('json_alias_join') else 'model, ' + str(d.get('acceptance', '?'))}): {d.get('reason') or '-'}" for i, d in enumerate(sample)]

    commit = C.git_commit()
    statuses = {C.STAGES[k]: status(k) for k in (1, 2, 3, 4)}
    final = f"""# Run 1b (profile p1b_ledger): final report

Written {time.strftime('%Y-%m-%d %H:%M')} by stage 05 from the stage reports in this folder. Commit `{commit}`. Stage status: {statuses}.
Every number is run 1b first, then run 1, then run 0 where it exists. Nothing was tuned; fusion weights (1.0, 0.5, 0.3) fixed.
Optional JSON-alias join: **{'on' if s2.get('json_alias_join') else 'off'}**.

## Expected against observed (expectations written before the run)

{eo_table}

## Memory check per cell (run 1b / run 1 / run 0)

{section(md[3], 'Memory check per cell')}

### Join direction

{section(md[3], 'Join direction')}

### Earlier record at the link step

{section(md[3], 'Earlier record at the link step')}

### Form 2 at the reveal: decision and outcome

{section(md[3], 'Form 2 at the reveal')}

### Wrong direction: where form 1 sits

{section(md[3], 'Wrong direction')}

### The three changes at work

{section(md[3], 'The three changes at work')}

## Retrieval per cell

### HR@10 (%) (run 1b / run 1 / run 0)

{section(md[4], 'HR@10 (%) per cell')}

### HR@5 (%)

{section(md[4], 'HR@5 (%) per cell')}

### Median gold rank

{section(md[4], 'Median gold rank per cell')}

### Live minus Frozen @10

{section(md[4], 'Live minus Frozen @10')}

### Controls

{section(md[4], 'Controls')}

## Fusion landscape (information only, not used anywhere else)

{section(md[4], 'Fusion landscape')}

{section(md[4], 'Bridged personas of run 1b')}

## Merge audit

30 of {s3.get('same_as_total', 'n/a')} final SAME AS decisions, seed 42 (to be read; run 1's sample was read as 7-8 wrong of 30):

{chr(10).join(audit_lines) or 'None.'}

{section(md[3], 'Ten of')}

Mr./Mrs. type joins on the books' real characters: {s3.get('title_conflicts', 'n/a')} (run 1: {s3.get('title_conflicts_run1', 'n/a')}). 

{section(md[3], 'Aliases in memory')}

## Cost

{cost}

## Conclusion

{link_sentence} {retrieval_sentence} {cost_sentence}

## Open problems

{chr(10).join('- ' + x for x in problems)}

## Commands and commit

- `nohup bash experiments/method_update/p1b_ledger/run_all.sh > experiments/method_update/p1b_ledger/logs/run_all_nohup.out 2>&1 &`
- Build: {next((c for c in C.read_json(C.REPORT_DIR / 'step_02.json').get('commands', []) if 'run_study.py' in c), 'n/a') if (C.REPORT_DIR / 'step_02.json').exists() else 'n/a'}
- Commit `{commit}`; stage reports `experiments/method_update/p1b_ledger/reports/step_01..05.md`; run 1's own step code on run 1b in `step_03_run1_code.md`, `step_04_run1_code.md`.
"""
    FINAL_MD.write_text(final, encoding="utf-8")
    report.files_written.append(str(FINAL_MD))

    verdict = f"Run 1b: bridge {bridge_n} of {bridge_d} in cells 1-3 (run 1: {frac(br.get('run1'))}); retrieval {'flat' if lf.get('ci_lo', 0) <= 0 else 'up'} at fixed weights."
    f1 = s1.get("form1_rate")
    gdoc = f"""# Issue 1 report, run 1b (profile p1b_ledger)

**{verdict}**

## R1. Candidate net (stage 01)

- Same model, urchade/gliner_large-v2.1; threshold 0.3 instead of 0.5.
- Names caught: form 1 {100 * (f1 or float('nan')):.0f}%, form 2 {100 * (s1.get('form2_rate') or float('nan')):.0f}% (run 1 at 0.5: 89%, 80%); candidates per chunk {s1.get('candidates_per_chunk_mean', float('nan')):.1f} (run 1: 9.0).
- Prompt text unchanged from run 1; recall no longer depends on the candidate list.

## R2. Build (stage 02)

- Calls per book: {'equal to run 1' if calls_equal else 'DIFFER from run 1'}. Tokens per call: input {tok.get('run1b_input_mean', float('nan')):.0f} (run 1: {tok.get('run1_input_mean') or float('nan'):.0f}), output {tok.get('run1b_output_mean', float('nan')):.0f} (run 1: {tok.get('run1_output_mean') or float('nan'):.0f}).
- Wall clock {secs:.2f} h (run 1: {run1_s2.get('wall_clock_s_total', 0) / 3600:.2f} h). Records {rec1b}, stubs {stub1b}. Recall cap 8 hit on {100.0 * caps / events:.0f}% of chunks.
- Redirected JSON records {tt.get('redirected', 0)}; auto-joins {tt.get('auto_join', 0)}; SAME AS rejected {tt.get('same_as_rejected', 0)} ({rej or 'none'}); aliases rejected {sum(v for k, v in tt.items() if k.startswith('alias_rejected_'))}.

## R3. Memory check (stage 03)

- Bridge formed, cells 1-3: {bridge_n}/{bridge_d} (run 1: {frac(br.get('run1'))}, run 0: {frac(br.get('run0'))}).
- Joins into the gold record: {into1b} of {joins1b} (run 1: {into1} of {joins1}).
- Earlier record in the prompt at the reveal: {ip1b}/{ex1b} (run 1: {ip1}/{ex1}).
- Title conflicts on the real characters: {s3.get('title_conflicts', 'n/a')} (run 1: {s3.get('title_conflicts_run1', 'n/a')}). Merge audit sample: in FINAL.md, to be read.

## R4. Retrieval (stage 04)

- Live HR@10 cells 1-3: {hr.get('run1b', float('nan')):.0f}% against frozen {hrf.get('run1b', float('nan')):.0f}%; live minus frozen {lf.get('mean_pp', float('nan')):+.1f} pp [{lf.get('ci_lo', float('nan')):+.1f}, {lf.get('ci_hi', float('nan')):+.1f}], McNemar p={lf.get('mcnemar_p', float('nan')):.3f}.
- Cell 5 at zero: {'yes' if s4.get('cell5_zero') else 'NO'}. Cell 4 not below cells 1-3: {'yes' if s4.get('cell4_not_below_cells_1_3') else 'NO'}.
- Fusion landscape (information only): see FINAL.md.

## R5. Conclusion (stage 05)

- {link_sentence}
- {retrieval_sentence}
- {cost_sentence}
- Open problems: {' '.join(problems)}
"""
    GDOC_MD.write_text(gdoc, encoding="utf-8")
    tracker = (
        "Tracker row 1b\n\n"
        f"Result: Bridge {bridge_n} of {bridge_d} (run 1: {frac(br.get('run1'))}); joins into gold record {into1b}/{joins1b}; earlier record in prompt {ip1b}/{ex1b}; live HR@10 cells 1-3 {hr.get('run1b', float('nan')):.0f}% vs frozen {hrf.get('run1b', float('nan')):.0f}%; cell 5 {'0' if s4.get('cell5_zero') else 'not 0'}; {secs:.2f} h.\n\n"
        f"Conclusion: {link_sentence.split(';')[0]}. {retrieval_sentence.split(':')[0]}.\n"
    )
    TRACKER_MD.write_text(tracker, encoding="utf-8")
    table = "# p1b_ledger tables\n\n" + section(md[3], "Memory check per cell") + "\n\n" + section(md[4], "HR@10 (%) per cell") + "\n\n" + eo_table + "\n"
    TABLE_MD.write_text(table, encoding="utf-8")
    with TABLE_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "cell", "n", "hr10_vanilla", "hr10_no_memory", "hr10_frozen", "hr10_live", "hr5_live", "median_live", "bridge_formed"])
        for c, v in sorted(s4.get("per_cell", {}).items(), key=lambda kv: int(kv[0])):
            g = [p for p in personas if str(p["cell"]) == c]
            w.writerow(["run1b_p1b_ledger", c, v["n"], *[round(v[f"hr10_{m}"], 2) for m in ("vanilla", "no_memory", "frozen", "live")], round(v["hr5_live"], 2), v["median_live"], sum(p["bridge"] for p in g)])
    report.files_written += [str(GDOC_MD), str(TRACKER_MD), str(TABLE_MD), str(TABLE_CSV)]
    report.add("Verdict", verdict)
    report.add("Expected against observed", eo_table)
    report.numbers.update({"verdict": verdict, "bridge_cells_1_3": [bridge_n, bridge_d], "joins": [into1b, joins1b], "in_prompt": [ip1b, ex1b], "tracker": tracker})
    lines = final.count("\n")
    if lines > 330:
        report.unexpected.append(f"FINAL.md has {lines} lines (target about 300).")


if __name__ == "__main__":
    raise SystemExit(C.run_stage(5, "Final report", main, requires=[2, 3, 4]))
