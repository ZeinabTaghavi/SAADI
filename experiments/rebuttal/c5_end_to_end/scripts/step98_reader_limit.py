"""C5 step 98 (CPU): how much of the gap to the oracle is the reader's.

For one dataset (default MuSiQue-32K), compares SAADI's answers with the oracle's on the same
questions, split by how much gold evidence SAADI's top 10 already holds:
  - any gold:  the paper's hit rule on SAADI's top 10 (a gold chunk, or a whole silver group);
               this is the evidence_in_context flag of the answer files;
  - all gold:  questions with at least one gold chunk, every one of them in SAADI's top 10;
  - no gold:   the paper's hit rule fails.
Inputs: the saved answers (results/rebuttal/c5_end_to_end/answers/<ds>/{saadi,oracle}.jsonl) and
the saved per-query labels with SAADI's top 10 (<ds>__bm25.queries.json in C5_STREAM_DIRS, or
--queries). Self-check: the any-gold flags recomputed here must equal the answer files' flags for
every question, else the script stops. Scores are re-computed from the saved outputs.
Writes results/rebuttal/c5_end_to_end/analysis/reader_limit_<ds>.{json,md}.
Usage: python step98_reader_limit.py [--dataset musique_32k] [--queries PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean

import c5_common as cc


def find_queries(ds: str, given: str | None) -> Path:
    if given:
        return Path(given)
    for d in [Path(x) for x in os.environ.get("C5_STREAM_DIRS", "").split(":") if x]:
        p = d / f"{ds}__bm25.queries.json"
        if p.exists():
            return p
    raise SystemExit(f"no {ds}__bm25.queries.json in C5_STREAM_DIRS; pass --queries")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="musique_32k", choices=cc.DATASETS)
    ap.add_argument("--queries", default=None)
    args = ap.parse_args(argv)
    paths = cc.C5Paths.from_env()
    ds = args.dataset
    qpath = find_queries(ds, args.queries)
    rows = {str(q["query_id"]): q for q in cc.read_json(qpath)["queries"]}
    ans = {}
    for cond in ("saadi", "oracle"):
        recs = {}
        for r in cc.read_jsonl(paths.answers_file(ds, cond)):
            r.update(cc.score_answer(r["output"], r["gold"]))
            recs[r["query_id"]] = r
        ans[cond] = recs
    qids = [q for q in ans["saadi"] if q in ans["oracle"]]
    missing = [q for q in qids if q not in rows]
    if missing:
        raise SystemExit(f"{len(missing)} answered questions have no saved labels, e.g. {missing[:3]}")
    labs = [cc.labels_from_row(rows[q]) for q in qids]
    top10 = [list(rows[q]["published_top10"])[:10] for q in qids]
    anyg = cc.evidence_hits(labs, top10, k=10)
    mism = [q for q, f in zip(qids, anyg) if f != ans["saadi"][q].get("evidence_in_context")]
    if mism:
        raise SystemExit(f"self-check failed: any-gold flag differs from the answer files for {len(mism)} questions, e.g. {mism[:3]}")
    allg = [(bool(lab.gold) and set(lab.gold) <= set(t)) if lab.gold else None for lab, t in zip(labs, top10)]

    def subset(name: str, sel: list[str]) -> dict:
        return {"subset": name, "n": len(sel),
                "saadi_f1": 100.0 * mean(ans["saadi"][q]["f1"] for q in sel) if sel else None,
                "oracle_f1": 100.0 * mean(ans["oracle"][q]["f1"] for q in sel) if sel else None}

    out = {"dataset": ds, "queries_file": paths.rel(qpath), "queries_sha256": cc.sha256_file(qpath), "answers": [paths.rel(paths.answers_file(ds, c)) for c in ("saadi", "oracle")],
           "self_check": f"any-gold flags equal the answer files' evidence_in_context for all {len(qids)} questions",
           "subsets": [subset("all questions", qids),
                       subset("any gold in SAADI's top 10 (paper hit rule)", [q for q, f in zip(qids, anyg) if f]),
                       subset("all gold chunks in SAADI's top 10", [q for q, f in zip(qids, allg) if f]),
                       subset("gold chunks, but not all in SAADI's top 10", [q for q, f in zip(qids, allg) if f is False]),
                       subset("no gold in SAADI's top 10 (paper hit rule fails)", [q for q, f in zip(qids, anyg) if f is False])],
           "questions_without_gold_chunks": sum(1 for lab in labs if not lab.gold)}
    base = paths.results / "analysis"
    base.mkdir(parents=True, exist_ok=True)
    cc.write_json(base / f"reader_limit_{ds}.json", out)
    L = [f"# C5 reader limit, {cc.DATASET_DISPLAY[ds]}", "", f"Labels and SAADI top 10: `{out['queries_file']}`. Answers: " + ", ".join(f"`{a}`" for a in out["answers"]) + ".",
         f"Labels file sha256 {out['queries_sha256']}. Self-check: {out['self_check']}. Questions without gold chunks (silver groups only): {out['questions_without_gold_chunks']}.", "",
         "| Subset | n | SAADI F1 | Oracle F1, same questions |", "| --- | --- | --- | --- |"]
    L += [f"| {s['subset']} | {s['n']} | {s['saadi_f1']:.1f} | {s['oracle_f1']:.1f} |" for s in out["subsets"] if s["n"]]
    (base / f"reader_limit_{ds}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
