#!/usr/bin/env python3
"""C1: qualitative NovelHopQA examples, Contextual Retrieval's context vs SAADI's memory for the same chunk.

Selection rule, fixed before looking at any example (so the examples are not hand-picked):
  1. Evidence = the chunks holding the question's annotated evidence paragraphs (C5's per-paragraph
     labels, results/rebuttal/c5_end_to_end/prepare/labels_para_novelhopqa.jsonl.gz), not the paper's
     window-mode labels.
  2. Group A ("SAADI finds, CR misses"): an evidence chunk is in SAADI's BM25 top 10 but not in CR's,
     and SAADI's answer F1 (C5, same reader) is higher than CR's. Group B is the mirror image.
  3. Examples are taken from each group in a seeded random order (seed 42); the counts of both
     groups are reported, so the reader can judge how typical group A is.
For every example: the question, the gold answer, both answers, the evidence chunk, its rank under
Vanilla / CR / SAADI, the context CR wrote for it, and SAADI's memory for it (the entities linked to
the chunk with their aliases and descriptions). Question words that appear in SAADI's memory or in
CR's context but not in the chunk itself are listed: they are what each method added to the chunk.

Runs on the server, where C1's contexts and rankings are (CPU only, about a minute):
    python scripts/rebuttal/c1_examples.py
Local test on the CPU dry run (stub contexts): --tag dry_run_full
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from saadi.indexing.retrieval_methods.mbr import build_mbr_document_memory_index  # noqa: E402
from saadi.rebuttal import contextual_retrieval as cr  # noqa: E402
from saadi.rebuttal.frozen_records import DATASETS  # noqa: E402

SEED = 42
DATASET = "novelhopqa"
STOP = set(
    "the a an and or of to in on at for with by from as is was were be been are it its this that these those who whom whose "
    "what which when where why how does did do has have had he she they them his her their him not no but if then than so "
    "into about after before over under again more most such can could would should will shall may might must one two".split()
)


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z'\-]{2,}", str(text).lower()) if w not in STOP}


def load_jsonl(path: Path) -> dict[str, dict]:
    out = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                r = json.loads(line)
                out[str(r["query_id"])] = r
    return out


def rank_of(chunk: str, ranking: list[str]) -> str:
    return str(ranking.index(chunk) + 1) if chunk in ranking else f">{len(ranking)}"


def memory_of(idx, chunk_id: str, qwords: set[str]) -> list[dict]:
    """Entities SAADI linked to this chunk, the ones whose fields share words with the question first."""

    out = []
    for name in idx.chunk_to_names.get(chunk_id, []) or []:
        aliases = list(idx.entity_to_aliases.get(name, []) or [])
        descs = list(idx.entity_to_descriptions.get(name, []) or [])
        hit = words(" ".join([name, *aliases, *descs])) & qwords
        out.append({"name": name, "aliases": aliases, "descriptions": descs, "question_words": sorted(hit)})
    out.sort(key=lambda e: (-len(e["question_words"]), e["name"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="", help="'' for the real run, dry_run_full for the CPU test")
    ap.add_argument("--answers-dir", type=Path, default=REPO_ROOT / "results/rebuttal/c5_end_to_end/answers/novelhopqa")
    ap.add_argument("--labels", type=Path, default=REPO_ROOT / "results/rebuttal/c5_end_to_end/prepare/labels_para_novelhopqa.jsonl.gz")
    ap.add_argument("--n", type=int, default=5, help="examples shown from group A (group B: half of it)")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    paths = cr.C1Paths(REPO_ROOT, tag=args.tag)
    out_dir = args.out_dir or (paths.results_dir / "examples")
    job = cr.load_job(paths, DATASET, "bm25")
    contexts = cr.load_all_contexts(paths, DATASET)
    ans = {s: load_jsonl(args.answers_dir / f"{s}.jsonl") for s in ("saadi", "cr")}
    labels = {}
    with gzip.open(args.labels, "rt", encoding="utf-8") as handle:
        for line in handle:
            r = json.loads(line)
            labels[str(r["query_id"])] = r
    spec = DATASETS[DATASET]
    index_cache: dict[str, object] = {}

    def index_for(doc_id: str):
        if doc_id not in index_cache:
            index_cache[doc_id] = build_mbr_document_memory_index(REPO_ROOT / spec.index_json, doc_id=doc_id)
        return index_cache[doc_id]

    groups = {"A": [], "B": []}
    counts = {"questions": 0, "with_evidence": 0, "saadi_only_evidence": 0, "cr_only_evidence": 0}
    for (doc_id, qid), q in job.items():
        counts["questions"] += 1
        lab = labels.get(qid)
        if lab is None or qid not in ans["saadi"] or qid not in ans["cr"]:
            continue
        paras = lab["paragraph_chunk_ids"]
        paras = json.loads(paras) if isinstance(paras, str) else paras
        evidence = sorted({c for p in paras for c in p})
        if not evidence:
            continue
        counts["with_evidence"] += 1
        top = {s: [q["chunk_ids"][p] for p in q["rank_pos"][s]] for s in ("vanilla", "cr", "saadi")}
        s10, c10 = set(top["saadi"][:10]), set(top["cr"][:10])
        saadi_only = [c for c in evidence if c in s10 and c not in c10]
        cr_only = [c for c in evidence if c in c10 and c not in s10]
        counts["saadi_only_evidence"] += bool(saadi_only)
        counts["cr_only_evidence"] += bool(cr_only)
        f_s, f_c = float(ans["saadi"][qid]["f1"]), float(ans["cr"][qid]["f1"])
        if saadi_only and f_s > f_c:
            groups["A"].append((doc_id, qid, saadi_only[0], top, f_s, f_c))
        if cr_only and f_c > f_s:
            groups["B"].append((doc_id, qid, cr_only[0], top, f_s, f_c))

    def describe(item) -> dict:
        doc_id, qid, chunk, top, f_s, f_c = item
        idx = index_for(doc_id)
        text = next(c.text for c in idx.chunk_pool if str(c.chunk_id) == chunk)
        qrow = ans["saadi"][qid]
        qw = words(qrow["question"])
        mem = memory_of(idx, chunk, qw)
        ctx = str((contexts.get(chunk) or {}).get("context", ""))
        chunk_w = words(text)
        mem_words = set().union(*[set(e["question_words"]) for e in mem]) if mem else set()
        return {
            "book": doc_id, "query_id": qid, "question": qrow["question"], "gold": qrow["gold"],
            "saadi_answer": qrow["answer"], "saadi_f1": f_s, "cr_answer": ans["cr"][qid]["answer"], "cr_f1": f_c,
            "evidence_chunk": chunk, "chunk_text": text,
            "rank": {s: rank_of(chunk, top[s]) for s in ("vanilla", "cr", "saadi")},
            "cr_context": ctx,
            "saadi_memory": mem,
            "added_by_saadi_memory": sorted(mem_words - chunk_w),
            "added_by_cr_context": sorted((words(ctx) & qw) - chunk_w),
        }

    rng = random.Random(SEED)
    picked = {}
    for g, n in (("A", args.n), ("B", max(1, args.n // 2))):
        items = sorted(groups[g], key=lambda x: (x[0], x[1]))
        rng.shuffle(items)
        picked[g] = [describe(x) for x in items[:n]]
    # Over all of group A: how often SAADI's memory adds question words the chunk lacks, and CR's context does not.
    a_all = [describe(x) for x in groups["A"]]
    mech = {
        "group_A": len(a_all),
        "memory_adds_question_words": sum(1 for x in a_all if x["added_by_saadi_memory"]),
        "cr_context_adds_question_words": sum(1 for x in a_all if x["added_by_cr_context"]),
        "memory_adds_and_cr_does_not": sum(1 for x in a_all if x["added_by_saadi_memory"] and not x["added_by_cr_context"]),
    }
    res = {"rule": __doc__.split("Selection rule")[1].split("For every example")[0].strip(), "counts": counts,
           "group_sizes": {g: len(v) for g, v in groups.items()}, "mechanism_group_A": mech, "examples": picked, "tag": args.tag}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "examples.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")

    L = [f"# C1 qualitative examples: CR's context vs SAADI's memory (NovelHopQA){' — ' + args.tag if args.tag else ''}", ""]
    L.append(f"Questions with located evidence: {counts['with_evidence']} of {counts['questions']}. An evidence chunk is in SAADI's top 10 but not CR's in "
             f"{counts['saadi_only_evidence']} questions; the reverse in {counts['cr_only_evidence']}.")
    L.append(f"Group A (SAADI finds it, CR does not, and SAADI's answer is better): {len(groups['A'])} questions. Group B (the reverse): {len(groups['B'])}.")
    L.append(f"In group A, SAADI's memory adds question words the chunk itself lacks in {mech['memory_adds_question_words']} of {mech['group_A']}; "
             f"CR's context does so in {mech['cr_context_adds_question_words']}; memory adds them and CR's context does not in {mech['memory_adds_and_cr_does_not']}.")
    L.append("Examples are drawn in a seeded random order (seed 42) from each group; selection rule in examples.json.")
    for g in ("A", "B"):
        L += ["", f"## Group {g}: {'SAADI finds the evidence, CR does not' if g == 'A' else 'CR finds the evidence, SAADI does not'}", ""]
        for i, x in enumerate(picked[g], 1):
            L.append(f"### {g}{i}. {x['book']} {x['query_id']}")
            L.append(f"- Question: {x['question']}")
            L.append(f"- Gold: {x['gold']}")
            L.append(f"- SAADI's answer: {x['saadi_answer']} (F1 {100 * x['saadi_f1']:.0f}); CR's answer: {x['cr_answer']} (F1 {100 * x['cr_f1']:.0f})")
            L.append(f"- Evidence chunk {x['evidence_chunk']}: rank {x['rank']['saadi']} with SAADI, {x['rank']['cr']} with CR, {x['rank']['vanilla']} with Vanilla BM25")
            L.append(f"- Chunk text: {x['chunk_text'][:700].replace(chr(10), ' ')}{'…' if len(x['chunk_text']) > 700 else ''}")
            L.append(f"- CR's context for this chunk: {x['cr_context'] or '(none)'}")
            L.append("- SAADI's memory for this chunk:")
            for e in x["saadi_memory"][:4]:
                al = ", ".join(e["aliases"][:6])
                ds = " | ".join(d[:220] for d in e["descriptions"][:2])
                L.append(f"  - {e['name']}" + (f" (also: {al})" if al else "") + (f": {ds}" if ds else "") + (f" [shares with the question: {', '.join(e['question_words'])}]" if e["question_words"] else ""))
            L.append(f"- Question words added by SAADI's memory (not in the chunk): {', '.join(x['added_by_saadi_memory']) or 'none'}; by CR's context: {', '.join(x['added_by_cr_context']) or 'none'}")
            L.append("")
    (out_dir / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:6]))
    print(f"report: {out_dir / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
