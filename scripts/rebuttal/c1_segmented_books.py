#!/usr/bin/env python3
"""C1 x C5 on NovelHopQA: books Contextual Retrieval had to cut into segments vs books it saw whole.

Contextual Retrieval (CR) needs the whole document in every prompt. In C1, 7 of the 16 NovelHopQA
books did not fit Qwen3-30B-A3B's 262,144-token window, so CR split them into 2-3 segments and each
chunk saw only its own segment. SAADI reads every book once with a bounded state, whatever its length.
This script splits the 908 questions by that rule (fixed by C1's census before any result; nothing
chosen on outcomes) and compares SAADI, CR and Vanilla in each group, on C5's end-to-end answers
(one fixed reader over each method's BM25 top 10) and on retrieval.

Inputs (all saved results, no GPU, no LLM call):
  - C5 per-question answers: results/rebuttal/c5_end_to_end/answers/novelhopqa/{saadi,cr,vanilla}.jsonl
    (f1, em, contain, retrieval_hit10 per question, with the book id)
  - C1's NovelHopQA report: experiments/rebuttal/C1_contextual_retrieval/reports/novelhopqa.json
    (census.segmented_docs: the cut books and their number of segments)
Statistics: paired bootstrap over questions (10,000 resamples, seed 42), and, because questions
cluster in books, a book-level bootstrap as a robustness check. The interaction (SAADI - CR in cut
books minus SAADI - CR in whole books) is bootstrapped within each group.
Note: retrieval_hit10 uses the paper's NovelHopQA evidence labels (label issue still open); F1, EM and
containment compare the answer with the gold answer and do not depend on those labels.

    python scripts/rebuttal/c1_segmented_books.py            # writes results/rebuttal/C1_contextual_retrieval/segmented_books/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from saadi.rebuttal.frozen_records import paired_bootstrap_ci  # noqa: E402
from saadi.rebuttal.heldout_fusion import cluster_bootstrap_ci  # noqa: E402

SEED = 42
N_BOOT = 10_000
SYSTEMS = ("saadi", "cr", "vanilla")
DISPLAY = {"saadi": "SAADI", "cr": "CR", "vanilla": "Vanilla BM25"}
MEASURES = ("f1", "em", "contain", "retrieval_hit10")
MEASURE_DISPLAY = {"f1": "F1", "em": "EM", "contain": "Containment", "retrieval_hit10": "HR@10 (paper labels)"}
PAIRS = (("saadi", "cr"), ("saadi", "vanilla"), ("cr", "vanilla"))
GROUPS = ("cut", "whole")
GROUP_DISPLAY = {"cut": "Books CR had to cut (over the 262K window)", "whole": "Books CR saw whole"}


def load_answers(path: Path) -> dict[str, dict]:
    out = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                r = json.loads(line)
                out[str(r["query_id"])] = r
    return out


def value(row: dict, measure: str) -> float:
    v = row[measure]
    if isinstance(v, str):
        v = {"true": 1.0, "false": 0.0}.get(v.strip().lower(), v)
    return 100.0 * float(v)


def interaction_ci(cut: np.ndarray, whole: np.ndarray, *, n: int = N_BOOT, seed: int = SEED) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    a = cut[rng.integers(0, cut.size, size=(n, cut.size))].mean(1)
    b = whole[rng.integers(0, whole.size, size=(n, whole.size))].mean(1)
    d = a - b
    return float(cut.mean() - whole.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def fmt_ci(c: tuple[float, float, float], nd: int = 1) -> str:
    m, lo, hi = c
    star = "**" if (lo > 0 or hi < 0) else ""
    return f"{star}{m:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]{star}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--answers-dir", type=Path, default=REPO_ROOT / "results/rebuttal/c5_end_to_end/answers/novelhopqa")
    ap.add_argument("--c1-report", type=Path, default=REPO_ROOT / "experiments/rebuttal/C1_contextual_retrieval/reports/novelhopqa.json")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results/rebuttal/C1_contextual_retrieval/segmented_books")
    ap.add_argument("--inputs-from", default="this checkout", help="where the input files came from (recorded in the report)")
    args = ap.parse_args()

    census = json.loads(args.c1_report.read_text(encoding="utf-8"))["census"]
    seg = {s["doc_id"]: s for s in census["segmented_docs"]}
    answers = {s: load_answers(args.answers_dir / f"{s}.jsonl") for s in SYSTEMS}
    qids = sorted(answers["saadi"])
    for s in SYSTEMS:
        if set(answers[s]) != set(qids):
            raise SystemExit(f"{s}: question ids differ from SAADI's ({len(answers[s])} vs {len(qids)})")
    doc_of = {q: str(answers["saadi"][q]["doc_id"]) for q in qids}
    for s in SYSTEMS:
        bad = [q for q in qids if str(answers[s][q]["doc_id"]) != doc_of[q]]
        if bad:
            raise SystemExit(f"{s}: book ids differ for {len(bad)} questions, e.g. {bad[0]}")
    group_of = {q: ("cut" if doc_of[q] in seg else "whole") for q in qids}
    books = sorted(set(doc_of.values()))

    res: dict = {"written_at": time.strftime("%Y-%m-%d %H:%M:%S"), "inputs": {"from": args.inputs_from, "answers": "results/rebuttal/c5_end_to_end/answers/novelhopqa/{saadi,cr,vanilla}.jsonl",
                                                               "c1_report": "experiments/rebuttal/C1_contextual_retrieval/reports/novelhopqa.json"},
                 "cut_books": seg, "groups": {}, "interaction": {}, "per_book": []}
    for g in GROUPS:
        qs = [q for q in qids if group_of[q] == g]
        gb = sorted({doc_of[q] for q in qs})
        clusters = np.asarray([doc_of[q] for q in qs])
        entry = {"questions": len(qs), "books": gb, "means": {}, "diffs": {}, "diffs_book_cluster": {}}
        for m in MEASURES:
            vals = {s: np.asarray([value(answers[s][q], m) for q in qs]) for s in SYSTEMS}
            entry["means"][m] = {s: float(vals[s].mean()) for s in SYSTEMS}
            for a, b in PAIRS:
                d = vals[a] - vals[b]
                entry["diffs"].setdefault(m, {})[f"{a}-{b}"] = paired_bootstrap_ci(d, n_resamples=N_BOOT, seed=SEED)
                entry["diffs_book_cluster"].setdefault(m, {})[f"{a}-{b}"] = cluster_bootstrap_ci(d, clusters, n_resamples=N_BOOT, seed=SEED)
        res["groups"][g] = entry
    for m in MEASURES:
        for a, b in PAIRS:
            dc = np.asarray([value(answers[a][q], m) - value(answers[b][q], m) for q in qids if group_of[q] == "cut"])
            dw = np.asarray([value(answers[a][q], m) - value(answers[b][q], m) for q in qids if group_of[q] == "whole"])
            res["interaction"].setdefault(m, {})[f"{a}-{b}"] = interaction_ci(dc, dw)
    for bk in books:
        qs = [q for q in qids if doc_of[q] == bk]
        row = {"book": bk, "group": "cut" if bk in seg else "whole", "segments": seg.get(bk, {}).get("k", 1),
               "doc_tokens": seg.get(bk, {}).get("doc_tokens"), "questions": len(qs)}
        for s in SYSTEMS:
            row[f"{s}_f1"] = float(np.mean([value(answers[s][q], "f1") for q in qs]))
        row["saadi_minus_cr_f1"] = row["saadi_f1"] - row["cr_f1"]
        res["per_book"].append(row)

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    with (out / "per_book.csv").open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(res["per_book"][0].keys()))
        w.writeheader()
        w.writerows(res["per_book"])

    G = res["groups"]
    L = ["# C1 x C5: NovelHopQA books Contextual Retrieval had to cut vs books it saw whole", ""]
    L.append(f"Written {res['written_at']} by `scripts/rebuttal/c1_segmented_books.py`, from C5's saved answers (BM25 top 10, one reader) and C1's census. "
             f"Inputs from {args.inputs_from}. The split is CR's own window rule (262,144 tokens), fixed before any result. Cut books: "
             + ", ".join(f"{b} ({v['doc_tokens'] // 1000}K tokens, {v['k']} segments)" for b, v in seg.items()) + ".")
    L.append("")
    L.append("## Means per group (%)")
    L.append("")
    L.append("| Group | Books | Questions | Measure | SAADI | CR | Vanilla BM25 |")
    L.append("| --- | --- | --- | --- | --- | --- | --- |")
    for g in GROUPS:
        for m in MEASURES:
            mm = G[g]["means"][m]
            L.append(f"| {GROUP_DISPLAY[g]} | {len(G[g]['books'])} | {G[g]['questions']} | {MEASURE_DISPLAY[m]} | {mm['saadi']:.1f} | {mm['cr']:.1f} | {mm['vanilla']:.1f} |")
    L.append("")
    L.append("## Paired differences (points, 95% bootstrap over questions; **bold** = excludes 0)")
    L.append("")
    L.append("| Measure | Pair | Cut books | Whole books | Cut minus whole (interaction) | Cut books, book-level | Whole books, book-level |")
    L.append("| --- | --- | --- | --- | --- | --- | --- |")
    for m in MEASURES:
        for a, b in PAIRS:
            k = f"{a}-{b}"
            L.append(f"| {MEASURE_DISPLAY[m]} | {DISPLAY[a]} − {DISPLAY[b]} | {fmt_ci(G['cut']['diffs'][m][k])} | {fmt_ci(G['whole']['diffs'][m][k])} | "
                     f"{fmt_ci(res['interaction'][m][k])} | {fmt_ci(G['cut']['diffs_book_cluster'][m][k])} | {fmt_ci(G['whole']['diffs_book_cluster'][m][k])} |")
    L.append("")
    L.append("## Per book (F1 %)")
    L.append("")
    L.append("| Book | Group | Segments | Questions | SAADI | CR | Vanilla BM25 | SAADI − CR |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in sorted(res["per_book"], key=lambda x: (x["group"], x["book"])):
        L.append(f"| {r['book']} | {r['group']} | {r['segments']} | {r['questions']} | {r['saadi_f1']:.1f} | {r['cr_f1']:.1f} | {r['vanilla_f1']:.1f} | {r['saadi_minus_cr_f1']:+.1f} |")
    wins = {g: sum(1 for r in res["per_book"] if r["group"] == g and r["saadi_minus_cr_f1"] > 0) for g in GROUPS}
    nb = {g: sum(1 for r in res["per_book"] if r["group"] == g) for g in GROUPS}
    L.append("")
    L.append(f"SAADI's F1 is above CR's in {wins['cut']} of {nb['cut']} cut books and {wins['whole']} of {nb['whole']} whole books.")
    L.append("")
    L.append("Notes: HR@10 uses the paper's NovelHopQA evidence labels (label issue still open); F1, EM and containment do not depend on them. "
             "Questions cluster in books (7 cut, 9 whole), so the book-level bootstrap is the conservative interval.")
    (out / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
