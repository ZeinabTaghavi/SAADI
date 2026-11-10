"""C5 step 99 (CPU): merge both datasets into the final tables and REPORT.md.

Reads results/rebuttal/c5_end_to_end/per_dataset/<dataset>/{table.csv,paired.csv,summary.json}
(step 03) and the step 00 reports (settings, reproduction gate). Writes
results/rebuttal/c5_end_to_end/{table.csv,paired.csv,table_paper.tex,REPORT.md}.
The opening sentences of REPORT.md are generated from the numbers, with the same wording rule
for every comparison: a difference is called better/worse only when its 95% interval excludes 0.
Token F1 is the lead measure; EM and containment are in the tables.
"""

from __future__ import annotations

import csv
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import c5_common as cc

LEAD = "f1"


def log(msg: str) -> None:
    print(f"[c5-99] {cc.now()} {msg}", flush=True)


def read_csv(path: Path) -> list[dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def step00_info(paths: cc.C5Paths, ds: str) -> dict[str, Any]:
    for name in (f"step_00_{ds}.json", "step_00.json"):
        p = paths.reports / name
        if p.exists():
            info = cc.read_json(p)
            if ds in (info.get("per_dataset") or {}):
                return info["per_dataset"][ds]
    return {}


def v(s: dict[str, Any], cond: str, subset: str = "all", m: str = LEAD) -> float | None:
    return (((s.get("conditions") or {}).get(cond) or {}).get(subset) or {}).get(m)


def pr(s: dict[str, Any], a: str, b: str, m: str = LEAD) -> dict[str, Any] | None:
    return ((s.get("paired") or {}).get(f"{a}-{b}") or {}).get(m)


def d(res: dict[str, Any] | None) -> str:
    if not res:
        return "not available"
    return f"{res['diff']:+.1f} points [{res['ci'][0]:+.1f}, {res['ci'][1]:+.1f}]"


def verdict(res: dict[str, Any] | None) -> str:
    if not res:
        return "n/a"
    lo, hi = res["ci"]
    return "better" if lo > 0 else ("worse" if hi < 0 else "not different")


def n1(x: Any) -> str:
    return "—" if x is None else f"{x:.1f}"


def sentences(S: dict[str, dict[str, Any]]) -> list[str]:
    out = []
    names = [k for k in cc.DATASETS if k in S]
    # 1-2. Does the retrieval gain turn into better answers?
    for ds in names:
        s = S[ds]
        res = pr(s, "saadi", "vanilla")
        hv, hs = (s["conditions"].get("vanilla") or {}).get("hr10", {}).get("hr_percent"), (s["conditions"].get("saadi") or {}).get("hr10", {}).get("hr_percent")
        if res is None:
            continue
        tail = {"better": f"so SAADI's retrieval gain (HR@10 {n1(hv)} to {n1(hs)}) carries over to the answers",
                "worse": f"so the answers are worse although HR@10 goes from {n1(hv)} to {n1(hs)}",
                "not different": f"so the retrieval gain (HR@10 {n1(hv)} to {n1(hs)}) does not show up as a clear answer gain"}[verdict(res)]
        out.append(f"On {cc.DATASET_DISPLAY[ds]}, the reader reaches {n1(v(s, 'saadi'))} token F1 with SAADI's top 10 and {n1(v(s, 'vanilla'))} with Vanilla BM25's top 10 "
                   f"(SAADI − Vanilla {d(res)}, 95% interval), {tail}.")
    # 3. Contextual Retrieval.
    parts = [f"{d(pr(S[ds], 'saadi', 'cr'))} on {cc.DATASET_DISPLAY[ds]} ({verdict(pr(S[ds], 'saadi', 'cr'))})" for ds in names if pr(S[ds], "saadi", "cr")]
    if parts:
        out.append("Against Contextual Retrieval (same chunks, same BM25), SAADI − CR in F1 is " + " and ".join(parts) + ".")
    # 4. RAPTOR, as retrieved and token-matched.
    parts = []
    for ds in names:
        s = S[ds]
        if pr(s, "saadi", "raptor"):
            tr, ts = (s["conditions"].get("raptor") or {}).get("context_tokens_mean"), (s["conditions"].get("saadi") or {}).get("context_tokens_mean")
            parts.append(f"{d(pr(s, 'saadi', 'raptor'))} on {cc.DATASET_DISPLAY[ds]} (RAPTOR context {tr:.0f} vs SAADI {ts:.0f} tokens on average; "
                         f"with SAADI cut to RAPTOR's length per query {d(pr(s, 'saadi_at_raptor_len', 'raptor'))})")
    if parts:
        out.append("Against RAPTOR with its default settings, SAADI − RAPTOR in F1 is " + " and ".join(parts) + ".")
    # 5. The agent.
    parts = []
    for ds in names:
        s = S[ds]
        if pr(s, "saadi", "arag"):
            own = (s["conditions"].get("arag_own") or {}).get("all") or {}
            parts.append(f"{d(pr(s, 'saadi', 'arag'))} on {cc.DATASET_DISPLAY[ds]} (A-RAG read {n1((s['conditions'].get('arag') or {}).get('passages_mean'))} chunks per query; "
                         f"with SAADI cut to A-RAG's length per query {d(pr(s, 'saadi_at_arag_len', 'arag'))}; "
                         f"its own final answers: F1 {n1(own.get('f1'))}, containment {n1(own.get('contain'))})")
    if parts:
        out.append("Given the chunks the A-RAG agent read, SAADI − A-RAG in F1 is " + " and ".join(parts) + ".")
    # 6. Closed-book and oracle.
    cb = [f"{n1(v(S[ds], 'closed_book'))} on {cc.DATASET_DISPLAY[ds]}" for ds in names]
    orc = [f"{n1(v(S[ds], 'oracle'))}" for ds in names]
    def gap_text(ds: str) -> str:
        above = (v(S[ds], 'saadi') or 0) - (v(S[ds], 'closed_book') or 0)
        to_oracle = (v(S[ds], 'oracle') or 0) - (v(S[ds], 'saadi') or 0)
        return f"{n1(above)} above closed-book and " + (f"{n1(to_oracle)} below the oracle" if to_oracle >= 0 else f"{n1(-to_oracle)} above the oracle")
    gap = [gap_text(ds) for ds in names]
    if names:
        out.append(f"Closed-book F1 is {' and '.join(cb)}, the oracle with the gold evidence chunks reaches {' and '.join(orc)}; SAADI is " + " and ".join(f"{g} on {cc.DATASET_DISPLAY[ds]}" for g, ds in zip(gap, names)) + ".")
    # 6b. Paragraph labels (NovelHopQA): the paper's labels and the oracle built from them.
    for ds in names:
        s = S[ds]
        if not (s["conditions"].get("oracle_para") or {}).get("n"):
            continue
        sa = s["conditions"].get("saadi") or {}
        out.append(f"On {cc.DATASET_DISPLAY[ds]} the paper's evidence labels match the dataset's whole evidence text as one span and accept a one-word overlap at a chunk boundary, "
                   f"so for multi-hop questions they rarely hold the evidence paragraphs; with each paragraph located on its own, the oracle reaches "
                   f"{n1(v(s, 'oracle_para'))} F1 ({n1(v(s, 'oracle_para', 'hop_1'))} on 1-hop and {n1(v(s, 'oracle_para', 'hop_2plus'))} on 2-4-hop questions; "
                   f"oracle − SAADI {d(pr(s, 'oracle_para', 'saadi'))}), and SAADI's top 10 holds every evidence paragraph for {n1(sa.get('evidence_para_all', {}).get('hr_percent'))}% "
                   f"of the queries and at least one for {n1(sa.get('evidence_para_any', {}).get('hr_percent'))}% (paper labels: HR@10 {n1(sa.get('hr10', {}).get('hr_percent'))}).")
    # 7. Dependence on retrieval.
    def split(ds: str) -> str:
        sa = S[ds]["conditions"].get("saadi") or {}
        if (sa.get("para_yes") or {}).get("n") or (sa.get("para_no") or {}).get("n"):
            yes, no, lab = "para_yes", "para_no", "every evidence paragraph (paragraph labels)"
        else:
            yes, no, lab = "evidence_yes", "evidence_no", "the gold evidence"
        return (f"{n1(v(S[ds], 'saadi', yes))} when {lab} is in SAADI's context and {n1(v(S[ds], 'saadi', no))} when it is not "
                f"({(sa.get(yes) or {}).get('n')} vs {(sa.get(no) or {}).get('n')} queries) on {cc.DATASET_DISPLAY[ds]}")
    parts = [split(ds) for ds in names]
    if parts:
        out.append("Split by retrieval, SAADI's F1 is " + "; ".join(parts) + ".")
    # 8. Caveats.
    out.append(f"Caveats: one reader ({os.environ.get('C5_READER_MODEL', 'Qwen3-30B-A3B-Instruct-2507')}, greedy) and one prompt for every condition; Vanilla is SAADI's text channel on the same chunks; "
               "RAPTOR on MuSiQue-32K was re-run with its saved default settings on SAADI's documents; token F1 leads, EM and containment are in the tables.")
    return out


def latex(S: dict[str, dict[str, Any]]) -> str:
    names = [k for k in cc.DATASETS if k in S]
    cols = "l" + "rrrrr" * len(names)
    L = ["% Generated by experiments/rebuttal/c5_end_to_end/scripts/step99_report.py", "\\begin{table}[t]", "\\centering", "\\small",
         "\\caption{End-to-end QA with one reader (Qwen3-30B-A3B-Instruct-2507, greedy) over each method's retrieved context. "
         "HR@10 and answer scores in percent; Tok.\\ = mean context tokens; HR@10 uses the paper's evidence labels (NovelHopQA: see the paragraph-label rows of the report). Token-matched rows cut the context to SAADI's mean length; equal-length rows cut SAADI's top 10 per query to the other method's context length.}",
         "\\label{tab:c5_end_to_end_qa}", "\\resizebox{\\linewidth}{!}{%", f"\\begin{{tabular}}{{{cols}}}", "\\toprule",
         " & " + " & ".join(f"\\multicolumn{{5}}{{c}}{{{cc.DATASET_DISPLAY[ds]}}}" for ds in names) + " \\\\",
         "".join(f"\\cmidrule(lr){{{2 + 5 * i}-{6 + 5 * i}}}" for i in range(len(names))),
         "Condition & " + " & ".join("HR@10 & EM & F1 & Cont. & Tok." for _ in names) + " \\\\", "\\midrule"]
    for c in cc.ALL_CONDITIONS:
        if not any(c in S[ds]["conditions"] for ds in names):
            continue
        cells = []
        for ds in names:
            s = S[ds]["conditions"].get(c)
            if not s:
                cells += ["--"] * 5
                continue
            hr = s["hr10"]["hr_percent"]
            cells += [n1(hr) if hr is not None else "--", n1(s["all"]["em"]), n1(s["all"]["f1"]), n1(s["all"]["contain"]),
                      f"{s['context_tokens_mean']:.0f}" if s.get("context_tokens_mean") else "0"]
        L.append(cc.CONDITION_DISPLAY[c].replace("%", "\\%") + " & " + " & ".join(cells) + " \\\\")
    L += ["\\midrule", "\\multicolumn{" + str(1 + 5 * len(names)) + "}{l}{\\textit{Paired F1 difference, points [95\\% CI]}} \\\\"]
    for a, b in cc.PAIRS + (("saadi_at_raptor_len", "raptor"), ("saadi_at_arag_len", "arag"), ("saadi", "arag_own")):
        cells = []
        for ds in names:
            r = pr(S[ds], a, b)
            cells.append(f"\\multicolumn{{5}}{{c}}{{{r['diff']:+.1f} [{r['ci'][0]:+.1f}, {r['ci'][1]:+.1f}]}}" if r else "\\multicolumn{5}{c}{--}")
        label = "SAADI" if a == "saadi" else "SAADI (equal length)"
        L.append(f"{label} $-$ {cc.CONDITION_DISPLAY[b]} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    return "\n".join(L) + "\n"


def main() -> int:
    paths = cc.C5Paths.from_env()
    info: dict[str, Any] = {"step": "99", "commit": cc.git_commit(paths.root), "host": os.uname().nodename, "files": []}
    status, code = "OK", 0
    S: dict[str, dict[str, Any]] = {}
    try:
        table, paired = [], []
        for ds in cc.DATASETS:
            base = paths.results / "per_dataset" / ds
            if not (base / "summary.json").exists():
                info.setdefault("missing", []).append(ds)
                continue
            S[ds] = cc.read_json(base / "summary.json")
            table += read_csv(base / "table.csv")
            paired += read_csv(base / "paired.csv")
        if not S:
            raise RuntimeError("no per-dataset results; run the GPU scripts first")
        for name, rows in (("table.csv", table), ("paired.csv", paired)):
            p = paths.results / name
            with open(p, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
                w.writeheader()
                w.writerows(rows)
            info["files"].append(paths.rel(p))
        (paths.results / "table_paper.tex").write_text(latex(S), encoding="utf-8")
        info["files"].append(paths.rel(paths.results / "table_paper.tex"))
        sent = sentences(S)
        R = ["# C5: end-to-end QA (reviewers mokb Q2, iVU8 Q1)", "",
             f"Written {cc.now()}; commit {info['commit']}. Datasets: {', '.join(cc.DATASET_DISPLAY[k] for k in S)}" + (f"; MISSING: {info['missing']}" if info.get("missing") else "") + ".", "",
             "## Summary", "", " ".join(sent), ""]
        for ds, s in S.items():
            st = step00_info(paths, ds)
            g = st.get("gate") or {}
            R += [f"## {cc.DATASET_DISPLAY[ds]}", "",
                  f"Queries {st.get('queries')}; reproduction gate {'passed' if g.get('passed') else 'see step 00'} (SAADI top 10 identical to the paper for {g.get('saadi_identical')} queries); "
                  f"token-matched budget {s.get('token_budget')} tokens.", ""]
            rows = []
            for c in cc.ALL_CONDITIONS:
                x = s["conditions"].get(c)
                if not x:
                    continue
                rows.append([cc.CONDITION_DISPLAY[c], x["n"], n1(x["all"]["em"]), n1(x["all"]["f1"]), n1(x["all"]["contain"]), n1(x["hr10"]["hr_percent"]),
                             f"{n1(x['evidence_yes']['f1'])} ({x['evidence_yes']['n']})", f"{n1(x['evidence_no']['f1'])} ({x['evidence_no']['n']})",
                             f"{x['context_tokens_mean']:.0f} / {x['context_tokens_max']}"])
            R += cc.md_table(["Condition", "n", "EM", "F1", "Contain", "HR@10", "F1, evidence in context (n)", "F1, evidence not in context (n)", "Context tokens mean / max"], rows)
            R += ["", "Paired differences, points [95% bootstrap interval]:", ""]
            rows = []
            for key, res in s["paired"].items():
                a, b = key.split("-", 1)
                rows.append([f"{cc.CONDITION_DISPLAY[a]} − {cc.CONDITION_DISPLAY[b]}", res.get("n")] +
                            [("—" if not res.get(m) else f"{res[m]['diff']:+.1f} [{res[m]['ci'][0]:+.1f}, {res[m]['ci'][1]:+.1f}]") for m in ("em", "f1", "contain", "hr10", "evidence")])
            R += cc.md_table(["Pair", "n", "EM", "F1", "Contain", "HR@10", "Evidence in context"], rows)
            if any((x.get("evidence_para_all") or {}).get("eligible") for x in s["conditions"].values()):
                pl = st.get("para_labels") or {}
                R += ["", "Paragraph labels. The paper's labels for this dataset match the dataset's whole evidence text as one span and accept a one-word overlap at a chunk boundary; "
                      "here each evidence paragraph is located on its own (" + str(pl.get("rule", "see step 00")) + ")", ""]
                if pl.get("by_hops"):
                    R += cc.md_table(["Hops", "Queries", "Paragraphs located %", "Paper-label chunks inside a paragraph %", "Paper labels cover every paragraph"],
                                     [[t["hops"], t["queries"], n1(t["paragraphs_located_pct"]), n1(t["paper_label_chunks_in_a_paragraph_pct"]), t["paper_labels_cover_every_paragraph"]] for t in pl["by_hops"]])
                    R += [""]
                rows = []
                for c in cc.ALL_CONDITIONS:
                    x = s["conditions"].get(c)
                    if not x:
                        continue
                    rows.append([cc.CONDITION_DISPLAY[c], x["n"], n1(x["evidence_para_any"]["hr_percent"]), n1(x["evidence_para_all"]["hr_percent"]),
                                 f"{n1(x['para_yes']['f1'])} ({x['para_yes']['n']})", f"{n1(x['para_no']['f1'])} ({x['para_no']['n']})",
                                 f"{n1(x['hop_1']['f1'])} ({x['hop_1']['n']})", f"{n1(x['hop_2plus']['f1'])} ({x['hop_2plus']['n']})"])
                R += cc.md_table(["Condition", "n", "Evidence any %", "Evidence all %", "F1, all paragraphs in context (n)", "F1, not all (n)", "F1 1-hop (n)", "F1 2-4 hops (n)"], rows)
                rows = []
                for key, res in s["paired"].items():
                    a, b = key.split("-", 1)
                    r = res.get("evidence_para_all")
                    if r:
                        rows.append([f"{cc.CONDITION_DISPLAY[a]} − {cc.CONDITION_DISPLAY[b]}", r["n"], f"{r['diff']:+.1f} [{r['ci'][0]:+.1f}, {r['ci'][1]:+.1f}]",
                                     (lambda q: "—" if not q else f"{q['diff']:+.1f} [{q['ci'][0]:+.1f}, {q['ci'][1]:+.1f}]")(res.get("evidence_para_any"))])
                R += ["", "Paired evidence differences with the paragraph labels, points [95% bootstrap interval]:", ""]
                R += cc.md_table(["Pair", "n", "Every paragraph", "At least one paragraph"], rows)
            sets = [f"- {name}: " + "; ".join(f"{k} = {val}" for k, val in (st.get(name) or {}).get("settings", {}).items()) for name in ("cr", "raptor", "arag") if (st.get(name) or {}).get("settings")]
            if sets:
                R += ["", "Method settings (defaults, nothing tuned):", "", *sets]
            R += [""]
        R += ["## Method", "",
              "- Same reader, prompt and queries for every condition (all queries of both datasets). Reader: vLLM, temperature 0, seed 42, fixed max output tokens.",
              "- Prompt template (one user message):", "", "```text", cc.READER_TEMPLATE, "```", "",
              "- Conditions: closed-book (no passages); oracle (the gold evidence chunks, document order); oracle from the paragraph labels (NovelHopQA: the chunks of every located evidence paragraph, document order); Vanilla = BM25 text channel top 10; SAADI = published profile and weights, top 10; "
              "Contextual Retrieval = C1's BM25 top 10, original chunk text; RAPTOR = its retrieved nodes (leaves and summaries) with default settings; A-RAG = the chunks it read "
              "(Table 2 exposure); A-RAG own = its saved final answer. Token-matched rows cut RAPTOR and A-RAG contexts to SAADI's mean context length; "
              "the equal-length rows cut SAADI's top 10, per query, to as many tokens as RAPTOR's or A-RAG's context for that query "
              "(RAPTOR and A-RAG give the reader fewer tokens than SAADI's top 10).",
              "- Measures: EM, token F1 (SQuAD normalisation, best over gold answers), containment; split by the paper's hit rule on the context's chunks. "
              "Paired bootstrap over queries, 10,000 resamples, seed 42.", "",
              "## Files", "", *[f"- `{f}`" for f in info["files"]], "- `results/rebuttal/c5_end_to_end/answers/<dataset>/<condition>.jsonl`",
              "- Step reports: `experiments/rebuttal/c5_end_to_end/reports/`", ""]
        (paths.results / "REPORT.md").write_text("\n".join(R), encoding="utf-8")
        info["files"].append(paths.rel(paths.results / "REPORT.md"))
        info["summary_sentences"] = sent
        info["one_line"] = "final tables and REPORT.md" + (f" (missing {info['missing']})" if info.get("missing") else "")
    except Exception as exc:
        status, code = "FAILED", 1
        info["traceback"] = traceback.format_exc()
        info["one_line"] = f"FAILED: {type(exc).__name__}: {str(exc)[:110]}"
    rep = paths.reports / "step_99_report.md"
    L = [f"# C5 end-to-end QA - step 99 final report: {status}", "", f"Written {cc.now()}; commit {info['commit']}; host {info['host']}.", "",
         "Command: `bash experiments/rebuttal/c5_end_to_end/sh/99_report.sh`", ""]
    if info.get("summary_sentences"):
        L += ["## Opening sentences of REPORT.md", "", " ".join(info["summary_sentences"]), ""]
    if info.get("missing"):
        L += [f"Datasets without results: {info['missing']}", ""]
    if info.get("traceback"):
        L += ["## Error", "", "```text", *info["traceback"].splitlines()[-60:], "```", ""]
    L += ["## Files written", "", *[f"- `{f}`" for f in info["files"]], f"- `{paths.rel(rep)}`"]
    rep.write_text("\n".join(L) + "\n", encoding="utf-8")
    cc.write_json(rep.with_suffix(".json"), info)
    rep.with_suffix(".summary.txt").write_text(info["one_line"] + "\n", encoding="utf-8")
    log(f"done: {status}")
    return code


if __name__ == "__main__":
    sys.exit(main())
