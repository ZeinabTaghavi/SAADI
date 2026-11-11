#!/usr/bin/env python3
"""C6 step 2c: SAADI vs each agent at that query's own exposure (re-scoring only).

For agent a and query q, k_q = the number of distinct chunks a showed its LLM for q. SAADI
gets exactly k_q chunks for the same query: the top k_q of its full ranking (c6_saadi_rankings.py;
identical to Table 2's ranking for the first 20). Both are scored with the paper's hit rule:
a hit if a gold chunk is among the chunks, or one silver evidence group is complete
(saadi.evaluation.retrieval_metrics). Labels are Table 2's (outputs/saadi/reference/rankings),
HR = hits / all Table 2 queries; a query the agent has no record for counts as 0 chunks, miss
(this is how the paper's agent numbers were computed).

Which saved list is "the chunks the agent showed its LLM" is not guessed: for every source file
and every chunk-id list in it, the script computes mean |list| and HR over the Table 2 queries,
and keeps the list that reproduces the paper's own Table 2 numbers for that agent
(outputs/other_methods/<m>/retrieval.csv: mean_chunks and hr_percent). The report shows the
check. A-RAG gets a second row whose budget also counts the chunks it only saw as search
previews (reads + previews), which is the reviewer's "including preview tokens".

Standard library only. Writes REPORT.md, table.csv, per_query.csv.gz and match.json to --results.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
SETTINGS = ("loogle", "musique_32k", "qasper_64k", "novelhopqa")
DISPLAY = {"loogle": "LooGLE", "musique_32k": "MuSiQue-32K", "qasper_64k": "QASPER-64K", "novelhopqa": "NovelHopQA"}
PAPER_DATASET = {"loogle": "loogle", "musique_32k": "musique", "qasper_64k": "qasper", "novelhopqa": "novelhopqa"}
METHODS = ("rag_r1", "agentic_r", "a_rag", "grip", "uma")
METHOD_DISPLAY = {"rag_r1": "RAG-R1", "agentic_r": "Agentic-R", "a_rag": "A-RAG", "grip": "GRIP", "uma": "UMA"}
MEAN_TOL = 1e-6
HR_TOL = 1e-6
BOOTSTRAP = 10000
SEED = 42
SAVED_DEPTH = 20


def read_jsonl_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def reference(setting: str) -> dict[str, dict]:
    manifest = json.loads((ROOT / "outputs/saadi/reference/manifest.json").read_text(encoding="utf-8"))
    (src,) = [s for s in manifest["sources"] if s.get("study") == "retrieval_depth" and s.get("retriever") == "bm25"
              and s.get("dataset") == PAPER_DATASET[setting]]
    return {str(r["query_id"]): r for r in read_jsonl_gz(ROOT / "outputs/saadi/reference" / src["path"])}


def paper_agent(method: str, setting: str) -> dict | None:
    path = ROOT / "outputs/other_methods" / method / "retrieval.csv"
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["study"] == "agentic" and r["dataset"] == PAPER_DATASET[setting]
                and r["cutoff"] == "" and r["hop"] == "overall"]
    if len(rows) != 1:
        return None
    return {"hr_percent": float(rows[0]["hr_percent"]), "mean_chunks": float(rows[0]["mean_chunks"])}


def is_hit(chunks, gold, groups) -> bool:
    s = set(chunks)
    return bool(s.intersection(gold)) or any(g and set(g) <= s for g in groups)


def dedupe(seq) -> list[str]:
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def candidates(exposure_dir: Path, method: str, setting: str):
    """Yield (source slug, {qid: compact record})."""

    d = exposure_dir / method / setting
    for f in sorted(d.glob("*.jsonl.gz")) if d.is_dir() else []:
        recs = {}
        for r in read_jsonl_gz(f):
            recs[str(r["query_id"])] = r  # last record wins for a duplicated id
        yield f.name[: -len(".jsonl.gz")], recs


def evaluate_list(ref: dict, recs: dict, key: str):
    n = len(ref)
    total = hits = covered = 0
    for qid, r in ref.items():
        rec = recs.get(qid)
        lst = dedupe(rec["lists"].get(key, [])) if rec else []
        if rec and key in rec["lists"]:
            covered += 1
        total += len(lst)
        hits += is_hit(lst, r["relevant_chunk_ids"], r["evidence_groups"])
    return {"mean_chunks": total / n, "hr_percent": 100 * hits / n, "covered": covered}


def choose(ref: dict, method: str, setting: str, exposure_dir: Path, paper: dict | None):
    tried = []
    for slug, recs in candidates(exposure_dir, method, setting):
        keys = sorted({k for r in recs.values() for k in r["lists"]})
        for key in keys:
            if key in ("preview_union", "preview_only"):
                continue
            m = evaluate_list(ref, recs, key)
            ok_mean = paper is not None and abs(m["mean_chunks"] - paper["mean_chunks"]) <= MEAN_TOL
            ok_hr = paper is not None and abs(m["hr_percent"] - paper["hr_percent"]) <= HR_TOL
            tried.append({"source": slug, "key": key, **m, "mean_matches_paper": ok_mean, "hr_matches_paper": ok_hr})
    exact = [t for t in tried if t["mean_matches_paper"] and t["hr_matches_paper"]]
    if exact:
        exact.sort(key=lambda t: (-t["covered"], t["source"], t["key"]))
        return exact[0], tried, "exact"
    if paper is not None and tried:
        best = min(tried, key=lambda t: (abs(t["mean_chunks"] - paper["mean_chunks"]) / max(paper["mean_chunks"], 1e-9)
                                         + abs(t["hr_percent"] - paper["hr_percent"]) / 100))
        return best, tried, "closest (paper numbers NOT reproduced)"
    return None, tried, "none"


def saadi_rankings(rank_dir: Path, setting: str) -> dict[str, list[str]] | None:
    p = rank_dir / f"{setting}.jsonl.gz"
    if not p.exists():
        return None
    return {str(r["query_id"]): r["ranking"] for r in read_jsonl_gz(p)}


def bootstrap_ci(diffs: list[int], seed: int = SEED, b: int = BOOTSTRAP) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(diffs)
    stats = []
    for _ in range(b):
        s = 0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        stats.append(100 * s / n)
    stats.sort()
    return stats[int(0.025 * b)], stats[int(math.ceil(0.975 * b)) - 1]


def compare(ref, recs, key, ranks, *, agent_key=None):
    """Per-query rows and the summary for one (agent list -> budget) choice."""

    rows = []
    for qid, r in ref.items():
        rec = recs.get(qid)
        budget_list = dedupe(rec["lists"].get(key, [])) if rec else []
        agent_list = dedupe(rec["lists"].get(agent_key or key, [])) if rec else []
        k = len(budget_list)
        ranking = ranks[qid]
        saadi_top = ranking[:k]
        rows.append({
            "query_id": qid, "has_agent_record": rec is not None, "k": k, "pool": len(ranking),
            "agent_hit": int(is_hit(agent_list, r["relevant_chunk_ids"], r["evidence_groups"])),
            "saadi_hit": int(is_hit(saadi_top, r["relevant_chunk_ids"], r["evidence_groups"])),
            "saadi_hit_saved20": int(is_hit(r["retrieved_chunk_ids"][: min(k, SAVED_DEPTH)], r["relevant_chunk_ids"], r["evidence_groups"])),
            # Not part of the matched comparison: SAADI's top 1 / top 3, used only to describe queries with k_q = 0.
            "saadi_hit_top1": int(is_hit(ranking[:1], r["relevant_chunk_ids"], r["evidence_groups"])),
            "saadi_hit_top3": int(is_hit(ranking[:3], r["relevant_chunk_ids"], r["evidence_groups"])),
        })
    n = len(rows)
    a = sum(x["agent_hit"] for x in rows)
    s = sum(x["saadi_hit"] for x in rows)
    both = sum(x["agent_hit"] and x["saadi_hit"] for x in rows)
    lo, hi = bootstrap_ci([x["saadi_hit"] - x["agent_hit"] for x in rows])
    ks = sorted(x["k"] for x in rows)
    return rows, {
        "queries": n, "agent_records": sum(x["has_agent_record"] for x in rows),
        "mean_k": sum(ks) / n, "median_k": ks[n // 2], "max_k": ks[-1],
        "queries_k_over_20": sum(x["k"] > SAVED_DEPTH for x in rows),
        "queries_k_over_pool": sum(x["k"] > x["pool"] for x in rows),
        "queries_k_zero": sum(x["k"] == 0 for x in rows),
        "k_zero_no_agent_record": sum(x["k"] == 0 and not x["has_agent_record"] for x in rows),
        "k_zero_agent_hits": sum(x["k"] == 0 and x["agent_hit"] for x in rows),
        "k_zero_saadi_top1_hits": sum(x["k"] == 0 and x["saadi_hit_top1"] for x in rows),
        "k_zero_saadi_top3_hits": sum(x["k"] == 0 and x["saadi_hit_top3"] for x in rows),
        "saadi_hr_if_k_zero_given_1": 100 * (s + sum(x["k"] == 0 and x["saadi_hit_top1"] for x in rows)) / n,
        "agent_hr": 100 * a / n, "saadi_hr": 100 * s / n, "diff": 100 * (s - a) / n, "ci95": [lo, hi],
        "both": both, "saadi_only": s - both, "agent_only": a - both, "neither": n - a - s + both,
        "saadi_hr_if_cut_at_20": 100 * sum(x["saadi_hit_saved20"] for x in rows) / n,
    }


def saadi_at(ref: dict, k: int) -> float:
    return 100 * sum(is_hit(r["retrieved_chunk_ids"][:k], r["relevant_chunk_ids"], r["evidence_groups"]) for r in ref.values()) / len(ref)


def f1(x) -> str:
    return "" if x is None else f"{x:.1f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--work", type=Path, default=ROOT / "outputs/rebuttal/c6_agent_exposure/per_query",
                    help="folder with saadi_rankings/ and exposure/<method>/")
    ap.add_argument("--exposure", type=Path, action="append", default=[], help="extra exposure folder(s) (e.g. the UMA copy); default <work>/exposure")
    ap.add_argument("--results", type=Path, default=ROOT / "results/rebuttal/c6_agent_exposure/per_query")
    args = ap.parse_args(argv)
    work = args.work.resolve()
    exposure_dirs = args.exposure or [work / "exposure"]
    res = args.results.resolve()
    res.mkdir(parents=True, exist_ok=True)

    table, details, per_query_out, notes = [], [], [], []
    for setting in SETTINGS:
        ref = reference(setting)
        ranks = saadi_rankings(work / "saadi_rankings", setting)
        if ranks is None:
            notes.append(f"{DISPLAY[setting]}: no SAADI full rankings yet (run c6_saadi_rankings.py), skipped.")
            continue
        missing = [q for q in ref if q not in ranks]
        if missing:
            raise RuntimeError(f"{setting}: SAADI rankings miss {len(missing)} Table 2 queries")
        for method in METHODS:
            paper = paper_agent(method, setting)
            exp_dir = next((d for d in exposure_dirs if (d / method / setting).is_dir()), None)
            if exp_dir is None:
                notes.append(f"{METHOD_DISPLAY[method]} on {DISPLAY[setting]}: no exposure files yet, skipped.")
                continue
            chosen, tried, how = choose(ref, method, setting, exp_dir, paper)
            details.append({"method": method, "setting": setting, "paper": paper, "selection": how, "chosen": chosen, "tried": tried})
            if chosen is None:
                notes.append(f"{METHOD_DISPLAY[method]} on {DISPLAY[setting]}: no chunk-id list found, skipped.")
                continue
            recs = dict(next(r for s, r in candidates(exp_dir, method, setting) if s == chosen["source"]))
            variants = [("exposure", chosen["key"], None)]
            if method == "a_rag" and any("preview_union" in r["lists"] for r in recs.values()):
                variants.append(("exposure + previews", "preview_union", chosen["key"]))
            for label, key, agent_key in variants:
                rows, summ = compare(ref, recs, key, ranks, agent_key=agent_key)
                lower, upper = (math.floor(paper["mean_chunks"]), math.ceil(paper["mean_chunks"])) if paper else (None, None)
                row = {
                    "dataset": DISPLAY[setting], "method": METHOD_DISPLAY[method], "budget": label,
                    "source": chosen["source"], "list": key, "selection": how,
                    "paper_mean_chunks": paper["mean_chunks"] if paper else None, "paper_agent_hr": paper["hr_percent"] if paper else None,
                    "paper_saadi_lower_k": lower, "paper_saadi_lower_hr": saadi_at(ref, lower) if paper else None,
                    "paper_saadi_upper_k": upper, "paper_saadi_upper_hr": saadi_at(ref, upper) if paper else None,
                    **{k: v for k, v in summ.items() if k != "ci95"}, "ci95_low": summ["ci95"][0], "ci95_high": summ["ci95"][1],
                }
                table.append(row)
                for x in rows:
                    per_query_out.append({"dataset": setting, "method": method, "budget": label, **x})

    # ---- outputs
    with (res / "table.csv").open("w", newline="", encoding="utf-8") as fh:
        if table:
            w = csv.DictWriter(fh, fieldnames=list(table[0].keys()))
            w.writeheader()
            w.writerows(table)
    with gzip.open(res / "per_query.csv.gz", "wt", newline="", encoding="utf-8") as fh:
        if per_query_out:
            w = csv.DictWriter(fh, fieldnames=list(per_query_out[0].keys()))
            w.writeheader()
            w.writerows(per_query_out)
    (res / "match.json").write_text(json.dumps({"table": table, "selection": details, "notes": notes}, indent=1), encoding="utf-8")

    L = ["# C6: SAADI vs each agent at that query's own exposure", ""]
    L.append("For every query, SAADI gets exactly as many chunks as the agent showed its LLM for that query "
             "(k_q), taken from the top of SAADI's ranking (BM25, published fusion point; identical to Table 2's ranking "
             "for the first 20, extended past 20 with the same scorer). Same labels and hit rule as Table 2; HR over all "
             "Table 2 queries. Re-scoring only: no agent or SAADI model was re-run.")
    L.append("")
    L.append("| Dataset | Method | Budget | Agent HR | SAADI HR at k_q | Diff (95% CI) | SAADI only / agent only | mean k_q (max) | k_q > 20 | Table 2: SAADI at floor / ceil |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in table:
        L.append(f"| {r['dataset']} | {r['method']} | {r['budget']} | {f1(r['agent_hr'])} | {f1(r['saadi_hr'])} | "
                 f"{r['diff']:+.1f} ({r['ci95_low']:+.1f}, {r['ci95_high']:+.1f}) | {r['saadi_only']} / {r['agent_only']} | "
                 f"{r['mean_k']:.2f} ({r['max_k']}) | {r['queries_k_over_20']} | "
                 f"k={r['paper_saadi_lower_k']}: {f1(r['paper_saadi_lower_hr'])} / k={r['paper_saadi_upper_k']}: {f1(r['paper_saadi_upper_hr'])} |")
    L.append("")
    L.append("## Queries where the agent showed its LLM no chunk (k_q = 0)")
    L.append("")
    L.append("On these queries the matched comparison gives SAADI zero chunks too, so both miss. This table shows how often "
             "that happens and how SAADI would do there with one or three chunks. Those columns are a different budget "
             "from the agent's and are not part of the comparison above; they only show what the zero budget costs SAADI. "
             "Budget = exposure (chunks the agent fully read); for A-RAG, search previews are not counted here.")
    L.append("")
    L.append("| Dataset | Method | Queries | k_q = 0 (%) | of which no agent record | agent hits there | SAADI top 1 hits there | SAADI top 3 hits there | SAADI HR at k_q | SAADI HR if k_q = 0 got 1 chunk |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in table:
        if r["budget"] != "exposure":
            continue
        z = r["queries_k_zero"]
        L.append(f"| {r['dataset']} | {r['method']} | {r['queries']} | {z} ({100 * z / r['queries']:.1f}%) | "
                 f"{r['k_zero_no_agent_record']} | {r['k_zero_agent_hits']} | {r['k_zero_saadi_top1_hits']} | "
                 f"{r['k_zero_saadi_top3_hits']} | {f1(r['saadi_hr'])} | {f1(r['saadi_hr_if_k_zero_given_1'])} |")
    L.append("")
    L.append("## Checks")
    L.append("")
    L.append("Each agent's exposure list is the saved list that reproduces the paper's own Table 2 numbers "
             "(mean chunks and HR, over the same queries).")
    L.append("")
    L.append("| Dataset | Method | Source | List | mean |list| (paper) | agent HR (paper) | selection |")
    L.append("|---|---|---|---|---|---|---|")
    for d in details:
        c = d["chosen"]
        if not c:
            continue
        p = d["paper"] or {}
        L.append(f"| {DISPLAY[d['setting']]} | {METHOD_DISPLAY[d['method']]} | `{c['source']}` | `{c['key']}` | "
                 f"{c['mean_chunks']:.4f} ({p.get('mean_chunks', float('nan')):.4f}) | {c['hr_percent']:.3f} ({p.get('hr_percent', float('nan')):.3f}) | {d['selection']} |")
    if notes:
        L.append("")
        L.append("## Not covered yet")
        L.append("")
        L += [f"- {n}" for n in notes]
    L.append("")
    L.append("Columns: *Diff* = SAADI HR minus agent HR, points, paired bootstrap over queries (10,000 resamples, seed 42). "
             "*SAADI only / agent only* = queries only one of the two hits. *k_q > 20* = queries that needed SAADI's ranking "
             "past the saved depth 20. *exposure + previews* (A-RAG) = the budget also counts chunks A-RAG only saw as search "
             "previews; A-RAG's hit is still from the chunks it read, as in the paper.")
    (res / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
