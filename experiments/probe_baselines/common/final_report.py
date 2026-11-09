#!/usr/bin/env python3
"""07_report: reports/FINAL.md and results/probe_baselines/REPORT_FOR_GDOC.md from whatever finished.

Reads results/probe_baselines/scores.json (06_score), the stage reports, and the numbers
the systems left in outputs/probe_baselines/<system>/. Every number is reported as it came
out; a system that did not finish is named as missing.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chunk_map as CM  # noqa: E402
import report_utils as RPT  # noqa: E402
import scorer as SC  # noqa: E402
from load_probe import ROOT  # noqa: E402

FINAL = RPT.REPORT_DIR / "FINAL.md"
GDOC = SC.RESULTS_DIR / "REPORT_FOR_GDOC.md"
REAL = ("b0_vanilla", "b2_contextual", "b3_hipporag", "b4_lqca", "saadi_run0_live", "saadi_run1_live", "saadi_run1b_live")
ORDER = ("b1_oracle", "b0_vanilla", "b2_contextual", "b3_hipporag", "b4_lqca", "saadi_run0_live", "saadi_run1_live", "saadi_run1b_live")
EXPECTED = [
    ("B1 oracle", "well above everything; cells 1-3 HR@10 above 50% (what the scoring costs even with a perfect link)"),
    ("B2 Contextual Retrieval", "the strongest real competitor on cells 1-3 where the reveal was visible; near Vanilla where it was not"),
    ("B3 HippoRAG v2", "between Vanilla and B2"),
    ("B4 LQCA", "near Vanilla (a surname-to-given-name reveal is not a pronoun)"),
    ("Cell 5", "zero for every system"),
]


def f(x, nd=0) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def numbers(system: str) -> dict:
    out = {}
    for p in sorted((CM.OUT_DIR / system).glob("numbers_*.json")) if (CM.OUT_DIR / system).exists() else []:
        try:
            out[p.stem[8:]] = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return out


def stage_json(stage: str) -> dict:
    p = RPT.REPORT_DIR / f"{stage}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def hr10(entry: dict | None, group: str = "cells_1_3") -> float | None:
    return None if entry is None else entry[group]["hr10"]["hr"]


def paired_text(p: dict | None) -> str:
    if not p or not p.get("n"):
        return "n/a"
    return f"{p['mean_pp']:+.1f} pp [{p['ci_lo']:+.1f}, {p['ci_hi']:+.1f}], McNemar p={p['mcnemar_p']:.3f} ({p['a_only']} vs {p['b_only']})"


def cost_rows() -> list[list]:
    rows = []
    for system, stage in (
        ("b0_vanilla", "01_b0_vanilla"),
        ("b1_oracle", "02_b1_oracle"),
        ("b2_contextual", "03_b2_contextual"),
        ("b3_hipporag", "04_b3_hipporag"),
        ("b4_lqca", "05_b4_lqca"),
    ):
        n = numbers(system)
        b, r = n.get("build", {}), n.get("retrieve", {})
        per_book = b.get("per_book", {}) or {}
        calls = sum(int(v.get("calls", v.get("llm_calls", 0)) or 0) for v in per_book.values() if isinstance(v, dict))
        tin = sum(int(v.get("input_tokens", 0) or 0) for v in per_book.values() if isinstance(v, dict))
        tout = sum(int(v.get("generated_tokens", 0) or 0) for v in per_book.values() if isinstance(v, dict))
        qb = r.get("per_book", {}) or {}
        qcalls = sum(int(v.get("llm_calls", 0) or 0) for v in qb.values() if isinstance(v, dict))
        size = (
            sum(int(v.get("index_bytes", v.get("index_chars", 0)) or 0) for v in per_book.values() if isinstance(v, dict))
            if per_book
            else sum((b.get("index_chars") or {}).values())
            if isinstance(b.get("index_chars"), dict)
            else 0
        )
        secs = stage_json(stage).get("seconds")
        nb = max(1, len(per_book) or 10)
        rows.append(
            [
                SC.SYSTEM_LABELS[system],
                RPT.stage_status(stage),
                f"{calls / nb:.0f}" if calls else "0",
                f"{tin / nb:,.0f}" if tin else "0",
                f"{tout / nb:,.0f}" if tout else "0",
                f"{qcalls}" if qcalls else "0",
                f"{(secs or 0) / 3600:.2f} h" if secs else "n/a",
                f"{size / nb / 1e6:.1f} MB" if size else "n/a",
            ]
        )
    s2 = ROOT / "experiments" / "method_update" / "p1_ledger" / "reports" / "step_02.json"
    if s2.exists():
        n = json.loads(s2.read_text(encoding="utf-8"))["numbers"]
        tok = n.get("tokens", {})
        calls = sum(v.get("chunk_records", 0) for v in n.get("books", {}).values())
        rows.append(
            [
                "SAADI run 1 (p1_ledger), for reference",
                "done",
                f"{calls / 10:.0f}",
                f"{tok.get('run1_input_mean', 0) * calls / 10:,.0f}",
                f"{tok.get('run1_output_mean', 0) * calls / 10:,.0f}",
                "0",
                f"{n.get('wall_clock_s_total', 0) / 3600:.2f} h",
                "n/a",
            ]
        )
    return rows


def main() -> int:
    scores_path = SC.RESULTS_DIR / "scores.json"
    if not scores_path.exists():
        raise FileNotFoundError(f"{scores_path} is missing; 06_score did not run.")
    S = json.loads(scores_path.read_text(encoding="utf-8"))
    sys_ = S["systems"]
    present = [s for s in ORDER if s in sys_]
    cells = sorted({int(c) for e in sys_.values() for c in e["per_cell"]})
    c123 = {s: hr10(sys_.get(s)) for s in ORDER}

    # ---- expected against observed ----
    b2 = sys_.get("b2_contextual")
    vis = b2.get("split_reveal_visible", {}).get("cells_1_3", {}) if b2 else {}
    b2_vis, b2_invis = (
        (vis.get("true", {}).get("hr10", {}).get("hr") if vis else None),
        (vis.get("false", {}).get("hr10", {}).get("hr") if vis else None),
    )
    real_others = {s: c123[s] for s in REAL if s != "b2_contextual" and c123.get(s) is not None}
    leaks = [SC.SYSTEM_LABELS.get(s, s) for s, e in sys_.items() if e.get("cell5_leak")]
    obs = [
        f"cells 1-3 HR@10 {f(c123.get('b1_oracle'))}%" if c123.get("b1_oracle") is not None else "missing",
        (
            f"cells 1-3 HR@10 {f(c123.get('b2_contextual'))}%; reveal visible {f(b2_vis)}% (n={vis.get('true', {}).get('n', 0)}), not visible {f(b2_invis)}% (n={vis.get('false', {}).get('n', 0)}); best other real system {max(real_others, key=real_others.get) if real_others else 'n/a'} at {f(max(real_others.values())) if real_others else 'n/a'}%"
        )
        if b2
        else "missing",
        f"cells 1-3 HR@10 {f(c123.get('b3_hipporag'))}% (Vanilla {f(c123.get('b0_vanilla'))}%, B2 {f(c123.get('b2_contextual'))}%)"
        if sys_.get("b3_hipporag")
        else "missing",
        (f"cells 1-3 HR@10 {f(c123.get('b4_lqca'))}% (Vanilla {f(c123.get('b0_vanilla'))}%)" if sys_.get("b4_lqca") else "missing"),
        ("missing" if not sys_ else "zero for every system" if not leaks else "ABOVE ZERO for: " + ", ".join(leaks)),
    ]

    def verdict(i: int) -> str:
        if obs[i] == "missing":
            return "not measured"
        if i == 0:
            return "held" if (c123.get("b1_oracle") or 0) > 50 else "not held"
        if i == 1:
            strongest = bool(real_others) and (b2_vis or 0) > max(real_others.values())
            n_invisible = vis.get("false", {}).get("n", 0) if vis else 0
            near = None
            if n_invisible:
                near = abs((b2_invis or 0) - (c123.get("b0_vanilla") or 0)) <= 5
            text = "held" if strongest and near is not False else "not held"
            return text + (
                ""
                if n_invisible
                else " for the visible part; every reveal in cells 1-3 was visible, so 'near Vanilla where not visible' is not testable"
            )
        if i == 2:
            lo, hi = sorted([c123.get("b0_vanilla") or 0, c123.get("b2_contextual") or 0])
            return "held" if lo <= (c123.get("b3_hipporag") or 0) <= hi else "not held"
        if i == 3:
            d = abs((c123.get("b4_lqca") or 0) - (c123.get("b0_vanilla") or 0))
            return "held" if d <= 5 else "not held"
        return "held" if not leaks else "not held"

    eo = RPT.md_table(
        ["System", "Expected (written before the run)", "Observed", "Verdict"],
        [[e[0], e[1], obs[i], verdict(i)] for i, e in enumerate(EXPECTED)],
    )

    # ---- main table ----
    def cell_text(s, c):
        m = sys_[s]["per_cell"].get(str(c))
        return f"{f(m['hr10']['hr'])} / {f(m['median_rank'])}" if m else "-"

    rows = [
        [SC.SYSTEM_LABELS.get(s, s), *[cell_text(s, c) for c in cells], f(c123.get(s)), "LEAK" if sys_[s].get("cell5_leak") else "0"]
        for s in present
    ]
    for s, agg in S.get("saadi_tables", {}).items():
        rows.append(
            [
                SC.SYSTEM_LABELS.get(s, s) + " (table.csv)",
                *[
                    f"{f(agg[str(c)]['hr10'] if str(c) in agg else agg.get(c, {}).get('hr10'))} / {f(agg[str(c)]['median_rank'] if str(c) in agg else agg.get(c, {}).get('median_rank'))}"
                    for c in cells
                ],
                "n/a",
                "-",
            ]
        )
    main_table = RPT.md_table(["System", *[f"cell {c}" for c in cells], "cells 1-3 HR@10", "cell 5"], rows)
    ci_rows = [
        [
            SC.SYSTEM_LABELS.get(s, s),
            *[
                f"{f(sys_[s]['per_cell'][str(c)]['hr10']['hr'])} [{f(sys_[s]['per_cell'][str(c)]['hr10']['lo'])}, {f(sys_[s]['per_cell'][str(c)]['hr10']['hi'])}]"
                if str(c) in sys_[s]["per_cell"]
                else "-"
                for c in cells
            ],
        ]
        for s in present
    ]
    ci_table = RPT.md_table(["System", *[f"cell {c}" for c in cells]], ci_rows)
    hr5_rows = [
        [
            SC.SYSTEM_LABELS.get(s, s),
            *[f(sys_[s]["per_cell"][str(c)]["hr5"]["hr"]) if str(c) in sys_[s]["per_cell"] else "-" for c in cells],
        ]
        for s in present
    ]
    hr5_table = RPT.md_table(["System", *[f"cell {c}" for c in cells]], hr5_rows)
    gap_rows = [
        [SC.SYSTEM_LABELS.get(s, s), f(sys_[s].get("gap_to_oracle_hr10_pp", {}).get("cells_1_3"), 1)]
        for s in present
        if s != "b1_oracle" and "gap_to_oracle_hr10_pp" in sys_[s]
    ]
    gap_table = RPT.md_table(["System", "HR@10 minus the oracle, cells 1-3 (pp)"], gap_rows) if gap_rows else "Oracle missing."

    # ---- B2 visibility split ----
    if b2 and b2.get("split_reveal_visible"):
        sp = b2["split_reveal_visible"]
        vrows = []
        for grp in ("cells_1_3", "reveal_cells"):
            g = sp[grp]
            vrows.append(
                [
                    {"cells_1_3": "cells 1-3", "reveal_cells": "all cells with a reveal"}[grp],
                    f"{g['true']['n']} / {g['false']['n']}",
                    f"{f(g['true']['hr10']['hr'])} [{f(g['true']['hr10']['lo'])}, {f(g['true']['hr10']['hi'])}]",
                    f"{f(g['false']['hr10']['hr'])} [{f(g['false']['hr10']['lo'])}, {f(g['false']['hr10']['hi'])}]",
                    f"{f(g['true']['median_rank'])} / {f(g['false']['median_rank'])}",
                ]
            )
        per_cell_vis = sp.get("per_cell_true", {})
        vis_table = (
            RPT.md_table(["group", "n visible / not visible", "HR@10 visible", "HR@10 not visible", "median rank visible / not"], vrows)
            + "\n\nReveal visible, per cell: "
            + ", ".join(f"cell {c} {v}/30" for c, v in per_cell_vis.items())
        )
    else:
        vis_table = "B2 missing."

    # ---- what B2 wrote: a seeded sample of gold-chunk contexts, and every cell-5 hit ----
    b2_samples = ""
    b2_rows = CM.read_rankings("b2_contextual") if b2 else []
    if b2_rows:
        persona_text = {}
        for book_id in {r["book_id"] for r in b2_rows}:
            for p in CM.load_inputs(book_id)["personas"]:
                persona_text[p["persona_id"]] = p

        def show(r: dict) -> str:
            p = persona_text.get(r["persona_id"], {})
            ctx = " ".join(str(r.get("gold_context", "")).split())
            return f"- **{r['persona_id']}** (cell {r['cell']}, gold rank {r['gold_rank']}): query *{p.get('query')}*; gold sentence *{p.get('gold_sentence')}*; reveal *{p.get('reveal_sentence') or '-'}*\n  context: {ctx[:600]}"

        import random

        pool = sorted((r for r in b2_rows if r["cell"] in (1, 2, 3)), key=lambda r: r["persona_id"])
        sample = sorted(random.Random(42).sample(pool, min(6, len(pool))), key=lambda r: r["persona_id"])
        hits5 = [r for r in b2_rows if r["cell"] == 5 and r["gold_rank"] <= 10]
        b2_samples = "Six gold-chunk contexts of cells 1-3 (seed 42):\n\n" + "\n".join(show(r) for r in sample)
        if hits5:
            b2_samples += (
                "\n\nEvery cell-5 persona B2 put in the top 10 (cell 5 has no reveal, so no context can hold form 2; a hit comes from other query words matching the context):\n\n"
                + "\n".join(show(r) for r in hits5)
            )

    # ---- link formation ----
    lrows = []
    for s, how in (
        ("b1_oracle", "form 2 appended (by construction)"),
        ("b2_contextual", "gold chunk's context holds form 2"),
        ("b3_hipporag", "a triple connects form 1 and form 2"),
        ("b4_lqca", "rewritten gold chunk holds form 2"),
    ):
        e = sys_.get(s)
        if e and e.get("split_link_formed"):
            pc = e["split_link_formed"]["per_cell_true"]
            sl = e["split_link_formed"]["cells_1_3"]
            lrows.append(
                [
                    SC.SYSTEM_LABELS[s],
                    how,
                    *[pc.get(str(c), 0) for c in cells],
                    f"{f(sl['true']['hr10']['hr'])} (n={sl['true']['n']}) / {f(sl['false']['hr10']['hr'])} (n={sl['false']['n']})",
                ]
            )
    run1 = SC.SAADI_RUNS["run1"]["per_persona"]
    if run1.exists():
        rr = SC.saadi_rows(run1, "live")
        lrows.append(
            [
                "SAADI run 1, for reference",
                "form 2 in the gold chunk's live record",
                *[sum(1 for r in rr if r["cell"] == c and r["bridge"]) for c in cells],
                "-",
            ]
        )
    link_table = (
        RPT.md_table(["System", "link counted when", *[f"cell {c}" for c in cells], "cells 1-3 HR@10 linked / not linked"], lrows)
        if lrows
        else "No link counts (systems missing)."
    )

    # ---- paired against SAADI run 1 live ----
    prows = [
        [
            SC.SYSTEM_LABELS.get(s, s),
            paired_text(sys_[s]["vs_saadi_run1_live"]["cells_1_3"]),
            paired_text(sys_[s]["vs_saadi_run1_live"]["reveal_cells"]),
        ]
        for s in present
        if "vs_saadi_run1_live" in sys_[s]
    ]
    paired_table = (
        RPT.md_table(["System minus SAADI run 1 live, HR@10", "cells 1-3 (n=90)", "all cells with a reveal (n=210)"], prows)
        if prows
        else S.get("paired_note", "SAADI run 1 missing.")
    )

    # ---- conclusion ----
    best = max(((s, c123[s]) for s in REAL if c123.get(s) is not None), key=lambda x: x[1], default=(None, None))
    oracle_v = c123.get("b1_oracle")
    solvable = (
        f"The probe is solvable by retrieval once the link exists: the oracle reaches {f(oracle_v)}% HR@10 in cells 1-3."
        if oracle_v is not None
        else "The oracle did not run, so the ceiling is unknown."
    )
    who = (
        f"The best real system is {SC.SYSTEM_LABELS.get(best[0], best[0])} at {f(best[1])}% (Vanilla {f(c123.get('b0_vanilla'))}%, SAADI run 1 {f(c123.get('saadi_run1_live'))}%)."
        if best[0]
        else "No real system finished."
    )
    costs = cost_rows()
    cost_line = next((r for r in costs if best[0] and r[0] == SC.SYSTEM_LABELS.get(best[0])), None)
    at_cost = (
        f"Its index-time cost per book: {cost_line[2]} LLM calls, {cost_line[3]} input and {cost_line[4]} generated tokens."
        if cost_line
        else "SAADI's cost per book is in the cost table."
    )
    conclusion = f"{solvable} {who} {at_cost}"

    problems = [f"{SC.SYSTEM_LABELS.get(s, s)}: {why}" for s, why in S.get("missing", {}).items()]
    problems += [f"Cell-5 leak: {x}" for x in leaks]
    for stage in RPT.STAGES[:6]:
        u = stage_json(stage).get("unexpected") or []
        problems += [f"{stage}: {x.splitlines()[0][:300]}" for x in u if x and not x.startswith("Last 50 log lines")]
    problems += S.get("saadi_notes", [])
    problems.append(
        "Settings of B3 and B4 follow the packages' defaults (decided with the user); they were not checked against the paper's Appendix E, which was not available."
    )

    chk = stage_json("00_check").get("numbers", {}).get("check", {})
    versions = chk.get("vllm", {})
    status = {s: RPT.stage_status(s) for s in RPT.STAGES}
    final = f"""# Probe baselines: final report

Written {time.strftime("%Y-%m-%d %H:%M")} by 07_report. Commit `{RPT.git_commit()}`. Stage status: {status}.
Persona probe: 10 modified NovelHopQA books, frozen spec `{chk.get("spec", {}).get("spec_sha256", "n/a")}`, 240 personas, 8 cells of 30. HR in % over personas; a hit when the gold chunk's rank is at most k; median of the gold ranks; 95% bootstrap intervals (10,000 resamples, seed 42); exact McNemar. Nothing was tuned on the probe.

## Expected against observed (expectations written before the run)

{eo}

## HR@10 (%) / median gold rank per cell, all systems and SAADI runs

{main_table}

HR@10 with 95% bootstrap interval:

{ci_table}

HR@5 (%):

{hr5_table}

Distance to the oracle:

{gap_table}

## B2: was the reveal visible when the gold chunk's context was written?

{vis_table}

{b2_samples}

## Link formation (does the system hold form 2 where the gold chunk can use it?)

{link_table}

## Cost per book (index time; query-time LLM calls in total)

{RPT.md_table(["System", "stage", "LLM calls per book", "input tokens per book", "generated tokens per book", "query-time LLM calls", "wall clock (last run of the stage)", "index size per book"], costs)}

B2 reads the visible book for every chunk: its input tokens are the logical prompt tokens; the server's prefix cache computes the book once per book (stage 03's report has the server counters).

## Paired comparison with SAADI run 1 (live) on the same personas

{paired_table}

## Conclusion

{conclusion}

## Open problems

{chr(10).join("- " + p for p in problems) or "- none recorded"}

## Commands, versions, commit

- `nohup bash experiments/probe_baselines/run_all.sh > experiments/probe_baselines/logs/run_all_nohup.out 2>&1 &`
- vLLM {versions.get("version", "n/a")} serving {versions.get("model", "n/a")} at a window of {versions.get("window", "n/a")} tokens; GPUs 4-7, tensor parallel 4, temperature 0.
- Third party: {json.dumps(chk.get("third_party", {}))}; system configs in experiments/probe_baselines/systems/*/config.yaml.
- Stage reports: experiments/probe_baselines/reports/00_check.md ... 07_report.md; scores: results/probe_baselines/scores.json, table.csv.
"""
    FINAL.parent.mkdir(parents=True, exist_ok=True)
    FINAL.write_text(final, encoding="utf-8")
    verdict_line = f"Verdict: oracle {f(oracle_v)}% HR@10 in cells 1-3; best real system {SC.SYSTEM_LABELS.get(best[0], 'n/a')} at {f(best[1])}%; SAADI run 1 {f(c123.get('saadi_run1_live'))}%; Vanilla {f(c123.get('b0_vanilla'))}%."
    gdoc = (
        f"""# 2.1 Probe baselines

**{verdict_line}**

## Setup
- Same probe as SAADI's runs: 10 books, 240 personas, 8 cells, the same chunks and the probe's own HR; one scorer for every system.
- Systems: B0 Vanilla BM25; B1 oracle (form 2 appended to every revealed persona's gold chunk); B2 Contextual Retrieval (whole book, Qwen3-30B at a 262K window); B3 HippoRAG v2 (package defaults); B4 LQCA (official code on 8-chunk segments).
- LLM for B2 and B3: Qwen3-30B-A3B-Instruct-2507, temperature 0, GPUs 4-7. LQCA uses no LLM.

## Results (cells 1-3, HR@10)
"""
        + "\n".join(f"- {SC.SYSTEM_LABELS.get(s, s)}: {f(c123.get(s))}%" for s in present)
        + f"""
- Cell 5 (no reveal): {"zero for every system" if not leaks else "above zero for " + ", ".join(leaks)}.
- B2 reveal visible / not visible: {f(b2_vis)}% / {f(b2_invis)}%.

## Conclusion
- {conclusion}

## Open problems
"""
        + "\n".join(f"- {p}" for p in problems[:8])
        + "\n"
    )
    SC.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    GDOC.write_text(gdoc, encoding="utf-8")
    CM.write_numbers("report", "report", {"final": str(FINAL), "gdoc": str(GDOC), "lines": final.count("\n")})
    (CM.system_dir("report") / "report_report.md").write_text(
        f"### Written\n\n- `{FINAL}` ({final.count(chr(10))} lines)\n- `{GDOC}`\n", encoding="utf-8"
    )
    print(FINAL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
