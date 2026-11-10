"""C5 step 00 (CPU only): find and check every saved retrieval output, reproduce the paper's
Vanilla and SAADI top-10 lists, and build one contexts file per dataset and condition.

Order (it stops at the first failed gate and still writes its report):
  1. Fixed inputs: published fusion point from configs/methods/sadi.yaml, preserved sources
     unchanged, saved queries and evidence labels (the paper's per-query BM25 results).
  2. Reproduction gate, before anything else is built: SAADI top 10 (recomputed from the BM25
     channel streams at the published point) must equal the paper's saved top 10 for every
     query with the paper's HR@5/HR@10 hit counts; Vanilla (text channel) must equal the saved
     text-only rankings (NovelHopQA) or Table 1's HR@5/HR@10 (MuSiQue-32K, no per-query file).
     On a mismatch: the first 5 mismatches are reported and the step exits 1.
  3. Saved comparison outputs: Contextual Retrieval (C1 rankings), RAPTOR (its saved run),
     A-RAG (its saved run): coverage of the C5 queries, same documents, and their own paper
     numbers reproduced.
  4. Contexts: closed-book, oracle, Vanilla, SAADI, CR, RAPTOR (+ token-matched), A-RAG
     (+ token-matched), A-RAG's own answers; token counts with the reader's tokenizer.
Nothing here calls a model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import traceback
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import c5_common as cc

STEP = "00"
REPORT_NAME = "step_00"  # "step_00_<tag>" with --tag (the re-prepare after a re-run)
OFFICIAL_RAPTOR_DEFAULTS = {
    # RetrievalAugmentationConfig / retrieve() of the official RAPTOR code (parthsarthi03/raptor).
    "tree_builder.max_tokens (leaf size)": 100,
    "tree_builder.num_layers": 5,
    "tree_builder.threshold": 0.5,
    "tree_builder.top_k": 5,
    "tree_builder.selection_mode": "top_k",
    "tree_builder.summarization_length": 100,
    "retrieve.top_k": 10,
    "retrieve.max_tokens": 3500,
    "retrieve.collapse_tree": True,
    "embedding model": "OpenAI text-embedding-ada-002",
    "summarization model": "gpt-3.5-turbo",
}


class GateFailed(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------


def log(msg: str) -> None:
    print(f"[c5-00] {cc.now()} {msg}", flush=True)


def strip_prefix(qid: Any) -> str:
    """'musique_32k::0' -> '0' (RAPTOR and A-RAG prefix the dataset); others unchanged."""

    s = str(qid)
    return s.split("::", 1)[1] if "::" in s else s


def chunk_index_of(chunk_id: str) -> int:
    m = re.search(r"(\d+)$", str(chunk_id))
    return int(m.group(1)) if m else 10**9


def load_yaml(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except ImportError:
        return {"_raw": text}


def published_point_from_config(path: Path) -> tuple[tuple[float, float, float, float], dict[str, Any]]:
    data = load_yaml(path)
    if "_raw" in data:  # no PyYAML: read the flat cached_retrieval block
        block, inside = {}, False
        for line in data["_raw"].splitlines():
            if line.startswith("cached_retrieval:"):
                inside = True
                continue
            if inside and line and not line.startswith(" "):
                break
            m = re.match(r"\s+(\w+):\s*([^#\s]+)", line)
            if inside and m:
                block[m.group(1)] = m.group(2)
    else:
        block = data["cached_retrieval"]
    block = {k: (float(v) if re.fullmatch(r"-?\d+(\.\d+)?", str(v)) else str(v)) for k, v in block.items()}
    fixed = {"lambda_t": 1.0, "memory_name_weight": 1.0, "normalization_method": "zscore", "memory_aggregation": "max"}
    for k, v in fixed.items():
        if block.get(k) != v:
            raise GateFailed(f"{path}: cached_retrieval.{k} = {block.get(k)!r}, the published scorer needs {v!r}")
    point = (float(block["lambda_n"]), float(block["lambda_h"]), float(block["memory_alias_weight"]), float(block["memory_description_weight"]))
    return point, block


def check_preserved(root: Path) -> dict[str, Any]:
    path = root / "docs/preserved_sources.json"
    if not path.exists():
        return {"checked": 0, "changed": [], "missing": [], "note": "docs/preserved_sources.json not found"}
    entries = cc.read_json(path)
    changed, missing = [], []
    for e in entries:
        p = root / e["path"]
        if not p.exists():
            missing.append(e["path"])
        elif cc.sha256_file(p) != e["sha256"]:
            changed.append(e["path"])
    return {"checked": len(entries), "changed": changed, "missing": missing}


# ---------------------------------------------------------------------------------------------
# Inputs shared by all conditions
# ---------------------------------------------------------------------------------------------


class Dataset:
    def __init__(self, key: str, root: Path):
        from saadi.indexing.retrieval_methods.mbr import build_mbr_document_memory_index
        from saadi.rebuttal.frozen_records import DATASETS, load_saved_live_rows

        self.key = key
        self.root = root
        self.spec = DATASETS[key]
        rows = load_saved_live_rows(self.spec, "bm25", repo_root=root)
        ids = [str(r["query_id"]) for r in rows]
        dup = [q for q, n in Counter(ids).items() if n > 1]
        if dup:
            raise GateFailed(f"{key}: duplicate query ids in the saved rows: {dup[:5]}")
        if len(rows) != cc.EXPECTED_QUERIES[key]:
            raise GateFailed(f"{key}: {len(rows)} saved queries, expected {cc.EXPECTED_QUERIES[key]}")
        self.rows = rows
        self.qids = ids
        self.by_id = {str(r["query_id"]): r for r in rows}
        self.labels = {q: cc.labels_from_row(self.by_id[q]) for q in ids}
        self.doc_of = {q: str(self.by_id[q]["doc_id"]) for q in ids}
        self.question = {q: cc.normalize_question(self.by_id[q].get("question")) for q in ids}
        self.chunks: dict[str, dict[str, Any]] = {}
        index_json = root / self.spec.index_json
        for doc in dict.fromkeys(self.doc_of.values()):
            idx = build_mbr_document_memory_index(index_json, doc_id=doc)
            for c in idx.chunk_pool:
                self.chunks[str(c.chunk_id)] = {"doc": doc, "index": int(c.step if c.step is not None else chunk_index_of(c.chunk_id)), "text": str(c.text)}
        self.n_docs = len(set(self.doc_of.values()))
        self.gold: dict[str, list[str]] = {}
        self.gold_source = ""
        # NovelHopQA only: the dataset's own evidence text per query (for the paragraph labels).
        self.context_text: dict[str, str] = {}
        self.hops: dict[str, int] = {}
        self.para: dict[str, dict[str, Any]] = {}

    def lab_list(self) -> list[cc.Labels]:
        return [self.labels[q] for q in self.qids]

    def passages(self, chunk_ids: list[str], kind: str = "chunk") -> list[dict[str, Any]]:
        out = []
        for cid in chunk_ids:
            c = self.chunks.get(cid)
            if c is None:
                raise KeyError(cid)
            out.append({"id": cid, "kind": kind, "text": c["text"], "chunk_ids": [cid]})
        return out


def load_gold_answers(ds: Dataset, env: dict[str, str], info: dict[str, Any]) -> None:
    gold: dict[str, list[str]] = {}
    questions: dict[str, str] = {}
    if ds.key == "musique_32k":
        path = Path(env["C5_MUSIQUE_QA"])
        source_of: dict[str, str] = {}
        for e in cc.read_json(path):
            gold[str(e["id"])] = [str(a) for a in e.get("answers", []) if str(a).strip()]
            questions[str(e["id"])] = cc.normalize_question(e.get("question"))
            source_of[str(e["id"])] = str(e.get("source_document_id", ""))
        ds.gold_source = cc.C5Paths.from_env().rel(path)
        # MuSiQue's official scoring also accepts the answer aliases; the prepared qa.json kept only
        # the answer. Add the aliases from the Hugging Face dataset (repo loader, offline cache)
        # when it is there; the report says whether they were used.
        alias_info: dict[str, Any] = {"used": False}
        if env.get("C5_MUSIQUE_ALIASES", "1") == "1":
            try:
                from saadi.indexing.dataset_loaders.musique import _iter_rows_for_split, _load_musique_dataset

                hf = {str(r.get("id")): r for r in _iter_rows_for_split(_load_musique_dataset(config_name="default"), split="validation")}
                added = mismatched = 0
                for q, sid in source_of.items():
                    r = hf.get(sid)
                    if r is None:
                        continue
                    if cc.normalize_answer(r.get("answer")) != cc.normalize_answer(gold[q][0] if gold[q] else ""):
                        mismatched += 1
                        continue
                    al = [str(a) for a in (r.get("answer_aliases") or []) if str(a).strip()]
                    new = [a for a in al if cc.normalize_answer(a) not in {cc.normalize_answer(g) for g in gold[q]}]
                    if new:
                        gold[q] = gold[q] + new
                        added += 1
                found = sum(1 for sid in source_of.values() if sid in hf)
                alias_info = {"used": True, "source": "Hugging Face dgslibisey/MuSiQue validation (offline cache)", "matched_questions": found,
                              "answer_mismatch": mismatched, "questions_with_aliases_added": added}
                ds.gold_source += " + answer aliases from dgslibisey/MuSiQue"
                if found != len(source_of) or mismatched:
                    info.setdefault("warnings", []).append(f"MuSiQue aliases: matched {found}/{len(source_of)} questions, answer differs for {mismatched}")
            except Exception as exc:
                alias_info = {"used": False, "reason": f"{type(exc).__name__}: {exc}"[:300]}
        info["musique_aliases"] = alias_info
    else:
        source = env.get("C5_NOVELHOPQA_QA_SOURCE", "auto")
        errors = []
        if source in ("auto", "hf"):
            try:
                from saadi.core.chunking import coerce_to_text
                from saadi.indexing.dataset_loaders.novelhopqa import _VALID_SPLITS, _load_novelhopqa_split, _query_id

                for split in _VALID_SPLITS:
                    for i, row in enumerate(_load_novelhopqa_split(split)):
                        if not isinstance(row, dict):
                            continue
                        qid = _query_id(row, split_name=split, index=i)
                        ans = coerce_to_text(row.get("answer") or row.get("gold_answer"))
                        if ans:
                            gold[qid] = [ans]
                            questions[qid] = cc.normalize_question(coerce_to_text(row.get("question") or row.get("query")))
                            ds.context_text[qid] = coerce_to_text(row.get("context") or row.get("passage") or row.get("document"))
                            try:
                                ds.hops[qid] = int(row.get("number_of_hops") or split.split("_")[-1])
                            except (TypeError, ValueError):
                                pass
                ds.gold_source = "Hugging Face abhaygupta1266/novelhopqa (repo loader, offline cache)"
            except Exception as exc:  # fall back to the RAPTOR run's copy
                errors.append(f"hf: {type(exc).__name__}: {exc}"[:400])
                gold.clear()
                questions.clear()
                ds.context_text.clear()
                ds.hops.clear()
        if not gold and source in ("auto", "raptor"):
            path = raptor_paths(env, ds.key)["run"] / "selection/qa_entries.json"
            for e in cc.read_json(path):
                qid = strip_prefix(e["query_id"])
                gold[qid] = [str(a) for a in e.get("reference_answers", []) if str(a).strip()]
                questions[qid] = cc.normalize_question(e.get("question"))
            ds.gold_source = f"{path} (RAPTOR run's copy of the repo loader output)"
        info["gold_errors"] = errors
    missing = [q for q in ds.qids if not gold.get(q)]
    qdiff = [q for q in ds.qids if q in questions and questions[q] != ds.question[q]]
    info["gold"] = {"source": ds.gold_source, "found": len(ds.qids) - len(missing), "missing": missing[:10], "question_mismatch": qdiff[:10]}
    if missing or qdiff:
        raise GateFailed(f"{ds.key}: gold answers missing for {len(missing)} queries, question text differs for {len(qdiff)} (examples {missing[:3]} {qdiff[:3]})")
    ds.gold = {q: gold[q] for q in ds.qids}


# ---------------------------------------------------------------------------------------------
# 2. Vanilla and SAADI: reproduction gate
# ---------------------------------------------------------------------------------------------


def find_streams(env: dict[str, str], key: str) -> tuple[Path, Path] | None:
    for d in [Path(x) for x in env.get("C5_STREAM_DIRS", "").split(":") if x]:
        p, q = d / f"{key}__bm25.parquet", d / f"{key}__bm25.queries.json"
        if p.exists() and q.exists():
            return p, q
    return None


def compute_streams(paths: cc.C5Paths, key: str) -> tuple[Path, Path]:
    from saadi.rebuttal.heldout_fusion import C2Paths, run_stream_job

    base = paths.out / "c2_streams"
    c2 = C2Paths(paths.root, out_base=base, results_base=base / "results", exp_base=base / "exp")
    summary = run_stream_job(key, "bm25", paths=c2, log=lambda m: log(str(m)))
    if summary.get("status") not in ("DONE", None) and not summary.get("skipped"):
        raise GateFailed(f"{key}: C2's stream job failed its own reproduction gate: {json.dumps(summary.get('check'))[:1500]}")
    return c2.stream_path(f"{key}__bm25"), c2.queries_path(f"{key}__bm25")


def reference_rows(paths: cc.C5Paths, env: dict[str, str], key: str) -> tuple[dict[str, Any], dict[str, list[str]], dict[int, int]]:
    ref_dir = Path(env.get("C5_REFERENCE_DIR", paths.root / "outputs/saadi/reference"))
    manifest = cc.read_json(ref_dir / "manifest.json")
    src = [s for s in manifest["sources"] if s.get("study") == "main" and s.get("dataset") == cc.METRICS_NAME[key] and s.get("retriever") == "bm25"]
    if len(src) != 1:
        raise GateFailed(f"{key}: expected one main bm25 source in the reference manifest, found {len(src)}")
    src = src[0]
    ranks = {str(r["query_id"]): list(r["retrieved_chunk_ids"])[:10] for r in cc.read_jsonl(ref_dir / src["path"])}
    expected = {int(m["cutoff"]): int(m["hits"]) for m in cc.read_json(ref_dir / "expected_metrics.json")
                if m.get("study") == "main" and m.get("dataset") == cc.METRICS_NAME[key] and m.get("retriever") == "bm25"}
    return src, ranks, expected


def table1_vanilla(env: dict[str, str], key: str) -> dict[int, float]:
    out = {}
    with open(env["C5_VANILLA_TABLE"], newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["study"] == "main" and r["method"] == "vanilla" and r["dataset"] == cc.METRICS_NAME[key] and r["retriever"] == "bm25" and r["hop"] == "overall":
                out[int(r["cutoff"])] = float(r["hr_percent"])
    return out


def reproduce(ds: Dataset, paths: cc.C5Paths, env: dict[str, str], point, info: dict[str, Any]) -> dict[str, list[str]]:
    from saadi.rebuttal.contextual_retrieval import VANILLA_POINT
    from saadi.rebuttal.heldout_fusion import load_query_arrays, repo_ranking

    found = find_streams(env, ds.key)
    if found is None:
        if env.get("C5_COMPUTE_STREAMS", "1") != "1":
            raise GateFailed(f"{ds.key}: no BM25 streams in C5_STREAM_DIRS and C5_COMPUTE_STREAMS=0")
        log(f"{ds.key}: no saved BM25 streams; computing them on CPU (C2's run_stream_job)")
        found = compute_streams(paths, ds.key)
    stream, queries = found
    log(f"{ds.key}: streams {stream}")
    qs = {q.query_id: q for q in load_query_arrays(stream, queries, "bm25")}
    missing = [q for q in ds.qids if q not in qs]
    if missing:
        raise GateFailed(f"{ds.key}: {len(missing)} C5 queries are not in the streams (e.g. {missing[:5]})")
    saadi = {q: repo_ranking(qs[q], "bm25", point)[:10] for q in ds.qids}
    vanilla = {q: repo_ranking(qs[q], "bm25", VANILLA_POINT)[:10] for q in ds.qids}

    src, paper, expected = reference_rows(paths, env, ds.key)
    gate: dict[str, Any] = {"streams": paths.rel(stream), "reference_source": src["source_id"], "reference_path": src["path"]}
    mism = [q for q in ds.qids if saadi[q] != paper.get(q)]
    gate["saadi_identical"] = len(ds.qids) - len(mism)
    gate["saadi_mismatches"] = [{"query_id": q, "ours": saadi[q], "paper": paper.get(q)} for q in mism[:5]]
    labs = ds.lab_list()
    gate["saadi_hits"] = {k: cc.hit_rate(cc.evidence_hits(labs, [saadi[q] for q in ds.qids], k=k))["hits"] for k in (5, 10)}
    gate["paper_hits"] = expected
    gate["saadi_hr"] = {k: 100.0 * gate["saadi_hits"][k] / len(ds.qids) for k in (5, 10)}
    ok_saadi = not mism and all(gate["saadi_hits"][k] == expected.get(k) for k in (5, 10))

    t1 = table1_vanilla(env, ds.key)
    van_hits = {k: cc.hit_rate(cc.evidence_hits(labs, [vanilla[q] for q in ds.qids], k=k)) for k in (5, 10)}
    gate["vanilla_hr"] = {k: van_hits[k]["hr_percent"] for k in (5, 10)}
    gate["table1_vanilla_hr"] = t1
    text_only = paths.root / ds.spec.namespace / "retrieval_reports/retrieval_results__bm25__mbr_lt1_ln0_lh0_norm_none__per_document.json"
    if text_only.exists():
        saved = {str(r["query_id"]): list(r["retrieved_chunk_ids"])[:10] for r in cc.read_json(text_only)}
        vm = [q for q in ds.qids if vanilla[q] != saved.get(q)]
        gate["vanilla_reference"] = f"saved per-query text-only rankings ({paths.rel(text_only)})"
        gate["vanilla_identical"] = len(ds.qids) - len(vm)
        gate["vanilla_mismatches"] = [{"query_id": q, "ours": vanilla[q], "saved": saved.get(q)} for q in vm[:5]]
        ok_van = not vm
    else:
        # No per-query Vanilla file: Table 1's HR@5/HR@10 (one decimal) must be reproduced.
        gate["vanilla_reference"] = "Table 1 Vanilla HR@5/HR@10 (outputs/other_methods/vanilla/retrieval.csv; no per-query file exists)"
        diffs = {k: round(gate["vanilla_hr"][k], 1) - t1.get(k, float("nan")) for k in (5, 10)}
        gate["vanilla_hr_diff_vs_table1"] = diffs
        ok_van = all(abs(d) < 1e-9 for d in diffs.values())
        gate["vanilla_mismatches"] = [] if ok_van else [{"cutoff": k, "ours": gate["vanilla_hr"][k], "table1": t1.get(k)} for k in (5, 10)]
    gate["passed"] = bool(ok_saadi and ok_van)
    info["gate"] = gate
    ds.ranking_saadi, ds.ranking_vanilla = saadi, vanilla  # type: ignore[attr-defined]
    return saadi


# ---------------------------------------------------------------------------------------------
# 3. Comparison methods
# ---------------------------------------------------------------------------------------------


def load_cr(ds: Dataset, env: dict[str, str], info: dict[str, Any]) -> dict[str, list[str]] | None:
    from saadi.rebuttal import contextual_retrieval as c1

    base = Path(env["C5_C1_DIR"]) / ds.key
    path = base / "retrieval/bm25/rankings.pkl.gz"
    done = base / "markers/bm25.DONE"
    rec: dict[str, Any] = {"path": str(path), "done_marker": done.exists()}
    info["cr"] = rec
    if not path.exists():
        rec["status"] = "missing"
        return None
    cr, c1_van, c1_saadi = {}, {}, {}
    for d in c1.load_rankings(path):
        chunk_ids = [str(c) for c in d["chunk_ids"]]
        for q in d["queries"]:
            qid = str(q["query_id"])
            rank = {s: [chunk_ids[int(i)] for i in v] for s, v in q["rank"].items()}
            if str(d["doc_id"]) != ds.doc_of.get(qid, str(d["doc_id"])):
                rec.setdefault("doc_mismatch", []).append(qid)
            cr[qid], c1_van[qid], c1_saadi[qid] = rank["cr"][:10], rank["vanilla"][:10], rank["saadi"][:10]
    missing = [q for q in ds.qids if q not in cr]
    rec["records"] = len(cr)
    rec["covered"] = len(ds.qids) - len(missing)
    rec["vanilla_identical_to_ours"] = sum(c1_van.get(q) == ds.ranking_vanilla[q] for q in ds.qids)  # type: ignore[attr-defined]
    rec["saadi_identical_to_ours"] = sum(c1_saadi.get(q) == ds.ranking_saadi[q] for q in ds.qids)  # type: ignore[attr-defined]
    labs = ds.lab_list()
    rec["cr_hr10"] = cc.hit_rate(cc.evidence_hits(labs, [cr.get(q, []) for q in ds.qids], k=10))["hr_percent"]
    try:
        with open(env["C5_C1_TABLE"], newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["dataset"] == ds.key and r["retriever"] == "bm25":
                    rec["c1_report_cr_hr10"] = float(r["cr_hr10"])
    except Exception as exc:
        rec["c1_table_error"] = str(exc)
    rec["settings"] = {
        "construction LLM": c1.LLM_MODEL_ID,
        "context max tokens": c1.CONTEXT_MAX_TOKENS,
        "sampling": c1.SAMPLING,
        "prompt": "Anthropic's published contextualizer prompt (C1 ANTHROPIC_PROMPT)",
        "index": "SAADI's BM25 on context + '\\n\\n' + chunk; no reranker",
        "reader input": "the ORIGINAL chunk text of the CR top 10 (CR changes what is indexed, not the chunk)",
    }
    problems = []
    if missing:
        problems.append(f"{len(missing)} queries missing")
    if rec.get("doc_mismatch"):
        problems.append(f"{len(rec['doc_mismatch'])} queries on another document")
    if rec["vanilla_identical_to_ours"] != len(ds.qids) or rec["saadi_identical_to_ours"] != len(ds.qids):
        problems.append("C1's Vanilla/SAADI top 10 differ from the reproduced ones (chunks or scores differ)")
    if "c1_report_cr_hr10" in rec and abs(rec["cr_hr10"] - rec["c1_report_cr_hr10"]) > 1e-6:
        problems.append(f"CR HR@10 {rec['cr_hr10']:.2f} differs from C1's report {rec['c1_report_cr_hr10']:.2f}")
    rec["problems"] = problems
    rec["status"] = "ok" if not problems else "unusable"
    return cr if not problems else None


def raptor_paths(env: dict[str, str], key: str) -> dict[str, Path]:
    run_name = f"{key}_retrieval_ablation_raptor"
    roots = [Path(x) for x in [env.get("C5_RAPTOR_RERUN_DIR", ""), env.get("C5_RAPTOR_DIR", ""), *env.get("C5_RAPTOR_SEARCH", "").split(":")] if x]
    for r in roots:
        run = r / env.get("C5_RAPTOR_RUNS", "raptor_10_runs") / key / run_name
        if (run / "retrieval/retrieval_payloads.jsonl").exists():
            ev = r / env.get("C5_RAPTOR_EVALS", "raptor_10_evaluations") / key / run_name / "retrieval_upper_bound"
            return {"root": r, "run": run, "eval": ev}
    r = roots[-1]
    return {"root": r, "run": r / "raptor_10_runs" / key / run_name, "eval": r / "raptor_10_evaluations" / key / run_name / "retrieval_upper_bound"}


def other_methods_hr(env: dict[str, str], method: str, key: str, cutoff: str) -> dict[str, float | None]:
    path = Path(env["C5_OTHER_METHODS_DIR"]) / method / "retrieval.csv"
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["study"] == "agentic" and r["dataset"] == cc.METRICS_NAME[key] and r["cutoff"] == cutoff and r["hop"] == "overall":
                return {"hr_percent": float(r["hr_percent"]), "mean_chunks": float(r["mean_chunks"]) if r["mean_chunks"] else None}
    return {"hr_percent": None, "mean_chunks": None}


def load_raptor(ds: Dataset, env: dict[str, str], info: dict[str, Any]) -> dict[str, dict[str, Any]] | None:
    p = raptor_paths(env, ds.key)
    rec: dict[str, Any] = {"run": str(p["run"]), "evaluation": str(p["eval"]),
                           "source": ("C5 re-run on SAADI's documents (paper number is on RAPTOR's own corpus)"
                                      if env.get("C5_RAPTOR_RERUN_DIR") and p["root"] == Path(env["C5_RAPTOR_RERUN_DIR"]) else "saved run")}
    info["raptor"] = rec
    payload_path = p["run"] / "retrieval/retrieval_payloads.jsonl"
    if not payload_path.exists():
        rec["status"] = "missing"
        return None
    payloads = cc.read_jsonl(payload_path)
    metrics = {str(m["query_id"]): m for m in cc.read_jsonl(p["eval"] / "metrics_per_query.jsonl")} if (p["eval"] / "metrics_per_query.jsonl").exists() else {}
    labels = {str(m["query_id"]): m for m in cc.read_jsonl(p["eval"] / "labels.jsonl")} if (p["eval"] / "labels.jsonl").exists() else {}
    cfg_path = p["run"] / "config/raptor_run.yaml"
    cfg = load_yaml(cfg_path) if cfg_path.exists() else {}
    if isinstance(cfg, dict) and "_raw" not in cfg:
        tb, rs = cfg.get("tree_builder_settings", {}) or {}, cfg.get("retrieval_settings", {}) or {}
        rec["settings"] = {
            "tree_builder.max_tokens (leaf size)": tb.get("max_tokens"),
            "tree_builder.num_layers": tb.get("num_layers"),
            "tree_builder.threshold": tb.get("threshold"),
            "tree_builder.top_k": tb.get("top_k"),
            "tree_builder.selection_mode": tb.get("selection_mode"),
            "tree_builder.summarization_length": tb.get("summarization_length"),
            "retrieve.top_k": rs.get("top_k"),
            "retrieve.max_tokens": rs.get("max_tokens"),
            "retrieve.collapse_tree": rs.get("collapse_tree"),
            "embedding model": (cfg.get("embedding_model") or {}).get("model"),
            "summarization model": ((tb.get("summarization_model") or {}).get("model")),
        }
        rec["settings_differ_from_official"] = {k: [v, OFFICIAL_RAPTOR_DEFAULTS[k]] for k, v in rec["settings"].items() if k in OFFICIAL_RAPTOR_DEFAULTS and v != OFFICIAL_RAPTOR_DEFAULTS[k]}
    rec["records"] = len(payloads)
    rec["profiling_build_seconds"] = None
    bt = p["run"] / "profiling/build_times.jsonl"
    if bt.exists():
        t = [float(r.get("build_time_ms", 0)) / 1000.0 for r in cc.read_jsonl(bt)]
        rec["profiling_build_seconds"] = {"documents": len(t), "total": sum(t), "per_document": (sum(t) / len(t)) if t else None}
    # Paper check on RAPTOR's own query set: upper-bound union-strict HR@10 (fixed-vs-agentic table).
    h = [m["hits"]["union_strict"]["hr10"] for m in metrics.values() if (m.get("hits") or {}).get("union_strict")]
    rec["own_queries_hr10"] = 100.0 * mean(h) if h else None
    rec["own_queries_eligible"] = len(h)
    rec["paper_hr10"] = other_methods_hr(env, "raptor", ds.key, "10")["hr_percent"]
    by_id: dict[str, dict[str, Any]] = {}
    for x in payloads:
        by_id[strip_prefix(x["query_id"])] = x
    allow_other = env.get("C5_RAPTOR_ALLOW_OTHER_CORPUS", "0") == "1"
    # Same document = same id AND same text. Group ids alone are not enough on MuSiQue-32K: RAPTOR's own
    # 15-group corpus reuses the names group_001..group_015 for other texts.
    same_text: dict[str, bool] | None = None
    corpus = p["run"] / "corpus/documents.jsonl"
    saadi_docs = Path(env["C5_MUSIQUE_QA"]).parent / "documents.json"
    if ds.key == "musique_32k" and corpus.exists() and saadi_docs.exists():
        mine = {str(k): " ".join(str(v).split()) for k, v in cc.read_json(saadi_docs).items()}
        same_text = {str(r["doc_id"]): mine.get(str(r["doc_id"])) == " ".join(str(r.get("text", "")).split()) for r in cc.read_jsonl(corpus)}
        rec["document_text_check"] = {"raptor_documents": len(same_text), "identical_to_saadi": sum(same_text.values())}
    elif ds.key == "musique_32k":
        rec["document_text_check"] = "not possible (corpus/documents.jsonl or SAADI documents.json missing): ids only"
    else:
        rec["document_text_check"] = "ids only (whole books, loaded by the same NovelHopQA loader)"
    usable, other_doc, qdiff = {}, [], []
    for q in ds.qids:
        x = by_id.get(q)
        if x is None:
            continue
        if cc.normalize_question(x.get("question")) != ds.question[q]:
            qdiff.append(q)
            continue
        if str(x.get("doc_id")) != ds.doc_of[q] or (same_text is not None and not same_text.get(str(x.get("doc_id")), False)):
            other_doc.append(q)
            if not allow_other:
                continue
        usable[q] = x
    rec["same_question"] = len(ds.qids) - len([q for q in ds.qids if q not in by_id]) - len(qdiff)
    rec["other_document"] = len(other_doc)
    rec["other_document_examples"] = [{"query_id": q, "c5_doc": ds.doc_of[q], "raptor_doc": by_id[q].get("doc_id")} for q in other_doc[:3]]
    rec["question_mismatch"] = qdiff[:5]
    rec["covered"] = len(usable)
    rec["allow_other_corpus"] = allow_other
    if not usable or (len(usable) < len(ds.qids) and not allow_other):
        rec["status"] = "incomplete" if usable else "missing for the C5 corpus"
        if other_doc:
            rec["status"] = f"saved run is on another corpus ({len(other_doc)} of its questions sit in a different {ds.key} document)"
        if len(usable) < len(ds.qids) and not allow_other:
            return None
    out = {}
    for q, x in usable.items():
        layer = {int(li["node_index"]): int(li["layer_number"]) for li in x.get("layer_information", [])}
        via: dict[int, list[str]] = {}
        for e in x.get("expanded_retrieved_chunks", []):
            via.setdefault(int(e["via_node_index"]), []).append(str(e["chunk_id"]))
        passages = []
        for node in x.get("retrieved_nodes", []):
            ni = int(node["node_index"])
            lay = int(node.get("layer_number", layer.get(ni, 0)))
            passages.append({"id": f"raptor:{x['doc_id']}:{ni}", "kind": "leaf" if lay == 0 else f"summary_l{lay}",
                             "text": str(node.get("text", "")), "chunk_ids": via.get(ni, [])})
        lab = labels.get(str(x["query_id"]))
        m = metrics.get(str(x["query_id"]), {})
        u = ((m.get("hits") or {}).get("union_strict") or {})
        out[q] = {"passages": passages, "labels": cc.labels_from_row({**lab, "query_id": q}) if lab else None,
                  "retrieval_hit10": (None if u.get("hr10") is None else bool(u["hr10"])),
                  "context_token_count_raptor": x.get("context_token_count")}
    rec["status"] = "ok" if len(out) == len(ds.qids) else f"partial ({len(out)}/{len(ds.qids)}, other corpus allowed)"
    return out


def load_arag(ds: Dataset, env: dict[str, str], info: dict[str, Any]) -> dict[str, dict[str, Any]] | None:
    run = env["C5_ARAG_RUN_MUSIQUE_32K"] if ds.key == "musique_32k" else env["C5_ARAG_RUN_NOVELHOPQA"]
    cands = [Path(x) / "outputs" / ds.key / "arag" / run for x in [env.get("C5_ARAG_DIR", ""), *env.get("C5_ARAG_SEARCH", "").split(":")] if x]
    run_dir = next((c for c in cands if (c / "checkpoints").is_dir()), cands[0])
    rec: dict[str, Any] = {"run": str(run_dir), "completed_marker": (run_dir / "COMPLETED").exists()}
    info["arag"] = rec
    cps = sorted((run_dir / "checkpoints").glob("*.json")) if (run_dir / "checkpoints").is_dir() else []
    if not cps:
        rec["status"] = "missing"
        return None
    by_id: dict[str, dict[str, Any]] = {}
    bad = 0
    for f in cps:
        try:
            x = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            bad += 1
            continue
        by_id[strip_prefix(x.get("query_id"))] = x
    rec["records"] = len(by_id)
    rec["unreadable_checkpoints"] = bad
    man = run_dir / "run_manifest.json"
    if man.exists():
        m = cc.read_json(man)
        rc = (m.get("runtime_config") or {})
        rec["settings"] = {k: rc.get(k) for k in ("model", "max_loops", "max_context_tokens", "message_token_guard", "max_completion_tokens", "embedding_model", "workers")}
        rec["settings"]["upstream"] = m.get("upstream_repository")
        rec["settings"]["chunks"] = (m.get("dataset") or {}).get("chunks")
    missing, qdiff, ddiff, unknown, errors = [], [], [], 0, []
    out = {}
    for q in ds.qids:
        x = by_id.get(q)
        if x is None:
            missing.append(q)
            continue
        if cc.normalize_question(x.get("question")) != ds.question[q]:
            qdiff.append(q)
            continue
        if str(x.get("doc_id")) != ds.doc_of[q]:
            ddiff.append(q)
            continue
        read = [str(c) for c in (x.get("retrieved_chunk_ids") or [])]
        unk = [c for c in read if c not in ds.chunks]
        unknown += len(unk)
        if x.get("error"):
            errors.append(q)
        out[q] = {"read": read, "unique": list(dict.fromkeys(c for c in read if c in ds.chunks)), "prediction": x.get("prediction"),
                  "termination": x.get("termination_reason"), "error": x.get("error"),
                  "reference_answers": x.get("reference_answers"), "unique_read_tokens": x.get("unique_read_tokens"),
                  "search_snippet_tokens": x.get("search_snippet_tokens")}
    rec["covered"] = len(out)
    rec["missing"] = missing[:5]
    rec["n_missing"] = len(missing)
    rec["question_mismatch"] = qdiff[:5]
    rec["document_mismatch"] = ddiff[:5]
    rec["unknown_chunk_ids"] = unknown
    rec["errors"] = len(errors)
    rec["termination"] = dict(Counter(str(v["termination"]) for v in out.values()))
    labs = [ds.labels[q] for q in ds.qids if q in out]
    rec["hr"] = cc.hit_rate(cc.evidence_hits(labs, [out[q]["read"] for q in ds.qids if q in out]))["hr_percent"]
    rec["mean_read"] = mean(len(out[q]["read"]) for q in out) if out else None
    paper = other_methods_hr(env, "a_rag", ds.key, "")
    rec["paper_hr"], rec["paper_mean_chunks"] = paper["hr_percent"], paper["mean_chunks"]
    problems = []
    if missing or qdiff or ddiff:
        problems.append(f"{len(missing)} missing, {len(qdiff)} other question, {len(ddiff)} other document")
    if unknown:
        problems.append(f"{unknown} read chunk ids are not SAADI chunk ids")
    if rec["paper_hr"] is not None and rec["hr"] is not None and abs(round(rec["hr"], 1) - rec["paper_hr"]) > 0.05 + 1e-9:
        problems.append(f"HR {rec['hr']:.2f} does not reproduce Table 2 ({rec['paper_hr']})")
    if rec["paper_mean_chunks"] is not None and rec["mean_read"] is not None and abs(rec["mean_read"] - rec["paper_mean_chunks"]) > 0.005:
        problems.append(f"mean read chunks {rec['mean_read']:.4f} vs Table 2 {rec['paper_mean_chunks']}")
    rec["problems"] = problems
    rec["status"] = "ok" if not problems else "unusable"
    return out if not problems else None


# ---------------------------------------------------------------------------------------------
# 4. Contexts
# ---------------------------------------------------------------------------------------------


class Tok:
    def __init__(self, model: str):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model, local_files_only=os.environ.get("HF_HUB_OFFLINE", "1") == "1")
        self.model = model

    def count(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"]) if text else 0

    def prompt(self, messages: list[dict[str, str]]) -> int:
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        if hasattr(ids, "keys"):  # BatchEncoding in newer transformers
            ids = ids["input_ids"]
        return len(ids)

    def head(self, text: str, n: int) -> str:
        ids = self.tok(text, add_special_tokens=False)["input_ids"][: max(0, n)]
        return self.tok.decode(ids, skip_special_tokens=True)


def token_match(passages: list[dict[str, Any]], budget: int, tok: Tok) -> tuple[list[dict[str, Any]], bool]:
    """Keep passages in rank order while the context fits `budget` tokens; the next one is cut at
    the budget (marked partial; its chunks no longer count as evidence). Shorter contexts are kept."""

    if tok.count(cc.format_context(passages)) <= budget:
        return passages, False
    kept: list[dict[str, Any]] = []
    for p in passages:
        trial = kept + [p]
        if tok.count(cc.format_context(trial)) <= budget:
            kept = trial
            continue
        used = tok.count(cc.format_context(kept)) if kept else 0
        overhead = tok.count(cc.format_context(kept + [{**p, "text": "x"}])) - used
        room = budget - used - overhead
        if room > 0:
            kept.append({**p, "text": tok.head(" ".join(str(p["text"]).split()), room), "partial": True, "chunk_ids": []})
        break
    return kept, True


def record(ds: Dataset, q: str, condition: str, passages: list[dict[str, Any]], tok: Tok, **extra: Any) -> dict[str, Any]:
    msgs = cc.build_messages(ds.by_id[q]["question"], passages)
    return {
        "query_id": q,
        "doc_id": ds.doc_of[q],
        "question": ds.by_id[q]["question"],
        "gold_answers": ds.gold[q],
        "condition": condition,
        "passages": passages,
        "n_passages": len(passages),
        "context_tokens": tok.count(cc.format_context(passages)) if passages else 0,
        "prompt_tokens": tok.prompt(msgs),
        **extra,
    }


def build_para_labels(ds: Dataset, paths: cc.C5Paths, info: dict[str, Any]) -> None:
    """NovelHopQA: locate every paragraph of the dataset's evidence text in SAADI's chunks (see
    c5_common.paragraph_in_text). Sets ds.para[q] = {"paragraphs": [[chunk ids], ...], ...}; a
    paragraph found in no chunk is kept with an empty list and does not count."""

    rec: dict[str, Any] = {"available": False}
    info["para_labels"] = rec
    if ds.key != "novelhopqa":
        rec["reason"] = "only NovelHopQA (its labels are window-mode matches of the whole context)"
        return
    missing = [q for q in ds.qids if not ds.context_text.get(q)]
    if missing:
        rec["reason"] = f"the dataset's context text is missing for {len(missing)} queries (gold answers came from {ds.gold_source})"
        info.setdefault("warnings", []).append(f"{ds.key}: no paragraph labels: {rec['reason']}")
        return
    log(f"{ds.key}: paragraph labels (each evidence paragraph located in SAADI's chunks)")
    by_doc: dict[str, list[tuple[str, list[str]]]] = {}
    for cid, c in sorted(ds.chunks.items(), key=lambda x: (x[1]["doc"], x[1]["index"], x[0])):
        by_doc.setdefault(c["doc"], []).append((cid, cc.match_tokens(c["text"])))
    rows = []
    for q in ds.qids:
        paras = cc.split_paragraphs(ds.context_text[q])
        groups = []
        for p in paras:
            pt = cc.match_tokens(p)
            groups.append([cid for cid, ct in by_doc.get(ds.doc_of[q], []) if cc.paragraph_in_text(ct, pt)])
        lab = ds.labels[q]
        saved = lab.evidence_chunks()
        real = {c for g in groups for c in g}
        ds.para[q] = {"paragraphs": groups, "paragraph_tokens": [cc.match_tokens(p) for p in paras], "hops": ds.hops.get(q)}
        rows.append({"query_id": q, "doc_id": ds.doc_of[q], "hops": ds.hops.get(q), "n_paragraphs": len(paras),
                     "paragraph_chunk_ids": groups, "located": sum(1 for g in groups if g),
                     "paper_label_chunk_ids": saved, "paper_label_chunks_in_a_paragraph": [c for c in saved if c in real],
                     "paper_labels_cover_every_paragraph": any(groups) and all(set(g) & set(saved) for g in groups if g)})
    out = paths.para_labels_file(ds.key)
    cc.write_jsonl(out, rows)
    by_hops: dict[Any, list[dict[str, Any]]] = {}
    for r in rows:
        by_hops.setdefault(r["hops"], []).append(r)
    table = []
    for h in sorted(by_hops, key=lambda x: (x is None, x)):
        rs = by_hops[h]
        n_saved = sum(len(r["paper_label_chunk_ids"]) for r in rs)
        table.append({"hops": h, "queries": len(rs), "paragraphs": sum(r["n_paragraphs"] for r in rs),
                      "paragraphs_located_pct": 100.0 * sum(r["located"] for r in rs) / max(1, sum(r["n_paragraphs"] for r in rs)),
                      "queries_without_any_located": sum(1 for r in rs if not r["located"]),
                      "chunks_per_query_paragraph_labels": mean(len({c for g in r["paragraph_chunk_ids"] for c in g}) for r in rs),
                      "chunks_per_query_paper_labels": mean(len(r["paper_label_chunk_ids"]) for r in rs),
                      "paper_label_chunks_in_a_paragraph_pct": (100.0 * sum(len(r["paper_label_chunks_in_a_paragraph"]) for r in rs) / n_saved) if n_saved else None,
                      "paper_labels_cover_every_paragraph": sum(1 for r in rs if r["paper_labels_cover_every_paragraph"])})
    rec.update({"available": True, "file": paths.rel(out), "min_boundary_overlap_tokens": cc.PARA_MIN_OVERLAP, "by_hops": table,
                "rule": "each 'Paragraph i:' piece of the dataset's context is located on its own in SAADI's chunks of the same book "
                        f"(the chunk contains it, it contains the chunk, or they overlap at a boundary by >= min({cc.PARA_MIN_OVERLAP}, paragraph length) tokens; "
                        "same token normalisation as the paper's labeler). Evidence 'all' = every located paragraph is in the context; 'any' = at least one."})


def para_evidence_rows(ds: Dataset, ctx: dict[str, list[dict[str, Any]] | None]) -> list[dict[str, Any]]:
    """Per condition and query: which located paragraphs are in the context. A full SAADI chunk counts
    by its id; a RAPTOR leaf by its text (same rule as the labels); summaries and cut (partial)
    passages never count."""

    out: list[dict[str, Any]] = []
    cache: dict[tuple[str, int, int], bool] = {}
    tok_cache: dict[int, list[str]] = {}
    for cond in cc.ALL_CONDITIONS:
        for r in ctx.get(cond) or []:
            q = r["query_id"]
            info = ds.para.get(q)
            if info is None:
                continue
            idx = [i for i, g in enumerate(info["paragraphs"]) if g]
            found = []
            for i in idx:
                g = set(info["paragraphs"][i])
                hit = False
                for p in r["passages"]:
                    if p.get("partial") or str(p.get("kind", "")).startswith("summary"):
                        continue
                    if p["id"] in ds.chunks:
                        hit = p["id"] in g
                    else:  # a RAPTOR leaf: its own text
                        key = (q, i, hash(p["text"]))
                        if key not in cache:
                            h = hash(p["text"])
                            if h not in tok_cache:
                                tok_cache[h] = cc.match_tokens(p["text"])
                            cache[key] = cc.paragraph_in_text(tok_cache[h], info["paragraph_tokens"][i])
                        hit = cache[key]
                    if hit:
                        break
                found.append(hit)
            any_, all_ = cc.para_hit(found)
            out.append({"condition": cond, "query_id": q, "hops": info.get("hops"), "paragraphs_located": len(idx), "paragraphs_in_context": sum(found),
                        "evidence_para_any": any_, "evidence_para_all": all_})
    return out


def build_contexts(ds: Dataset, tok: Tok, cr, raptor, arag, info: dict[str, Any]) -> dict[str, list[dict[str, Any]] | None]:
    labs = ds.lab_list()
    out: dict[str, list[dict[str, Any]] | None] = {}

    def flags(lists: list[list[str]], k: int | None = None) -> list[bool | None]:
        return cc.evidence_hits(labs, lists, k=k)

    log(f"{ds.key}: closed-book, oracle, Vanilla, SAADI")
    # Closed-book: no passage, so the evidence is never in the context (None only without labels).
    out["closed_book"] = [record(ds, q, "closed_book", [], tok, evidence_in_context=(False if ds.labels[q].eligible else None), retrieval_hit10=None) for q in ds.qids]
    oracle_ids = [sorted(ds.labels[q].evidence_chunks(), key=lambda c: (ds.chunks.get(c, {}).get("index", 10**9), c)) for q in ds.qids]
    unknown = sum(c not in ds.chunks for ids in oracle_ids for c in ids)
    if unknown:
        info.setdefault("warnings", []).append(f"{ds.key}: {unknown} evidence chunk ids are not in the chunk pool (left out of the oracle)")
    oracle_ids = [[c for c in ids if c in ds.chunks] for ids in oracle_ids]
    of = flags(oracle_ids)
    out["oracle"] = [record(ds, q, "oracle", ds.passages(ids, "evidence"), tok, evidence_in_context=f, retrieval_hit10=None)
                     for q, ids, f in zip(ds.qids, oracle_ids, of, strict=True)]
    # Oracle from the paragraph labels: the chunks of every located paragraph, document order.
    if ds.para:
        pids = [sorted({c for g in ds.para[q]["paragraphs"] for c in g}, key=lambda c: (ds.chunks[c]["index"], c)) for q in ds.qids]
        pf = flags(pids)
        out["oracle_para"] = [record(ds, q, "oracle_para", ds.passages(ids, "evidence_para"), tok, evidence_in_context=f, retrieval_hit10=None)
                              for q, ids, f in zip(ds.qids, pids, pf, strict=True)]
    else:
        out["oracle_para"] = None
    for cond, ranking in (("vanilla", ds.ranking_vanilla), ("saadi", ds.ranking_saadi), ("cr", cr)):  # type: ignore[attr-defined]
        if ranking is None:
            out[cond] = None
            continue
        lists = [ranking[q] for q in ds.qids]
        f10 = flags(lists, k=10)
        out[cond] = [record(ds, q, cond, ds.passages(lst), tok, evidence_in_context=f, retrieval_hit10=f) for q, lst, f in zip(ds.qids, lists, f10, strict=True)]
    budget = int(round(mean(r["context_tokens"] for r in out["saadi"])))  # type: ignore[union-attr]
    info["token_budget"] = budget

    if raptor is None:
        out["raptor"] = out["raptor_tm"] = None
    else:
        log(f"{ds.key}: RAPTOR (+ token-matched to {budget} tokens)")
        rows, rows_tm = [], []
        for q in ds.qids:
            if q not in raptor:
                continue
            r = raptor[q]
            lab = r["labels"]
            def ev(ps: list[dict[str, Any]], leaf_only: bool = False) -> bool | None:
                if lab is None or not lab.eligible:
                    return None
                ids = [c for p in ps if not p.get("partial") and (not leaf_only or p["kind"] == "leaf") for c in p["chunk_ids"]]
                return cc.evidence_hits([lab], [ids])[0]
            rows.append(record(ds, q, "raptor", r["passages"], tok, evidence_in_context=ev(r["passages"]),
                               evidence_in_leaf_text=ev(r["passages"], True), retrieval_hit10=r["retrieval_hit10"]))
            tm, cut = token_match(r["passages"], budget, tok)
            rows_tm.append(record(ds, q, "raptor_tm", tm, tok, evidence_in_context=ev(tm), evidence_in_leaf_text=ev(tm, True),
                                  retrieval_hit10=r["retrieval_hit10"], cut=cut))
        out["raptor"], out["raptor_tm"] = rows, rows_tm

    if arag is None:
        out["arag"] = out["arag_tm"] = out["arag_own"] = None
    else:
        log(f"{ds.key}: A-RAG (+ token-matched to {budget} tokens) and its own answers")
        rows, rows_tm, own = [], [], []
        for q in ds.qids:
            if q not in arag:
                continue
            a = arag[q]
            ps = ds.passages(a["unique"])
            f = cc.evidence_hits([ds.labels[q]], [a["read"]])[0]
            rows.append(record(ds, q, "arag", ps, tok, evidence_in_context=f, retrieval_hit10=f, n_read=len(a["read"])))
            tm, cut = token_match(ps, budget, tok)
            ftm = cc.evidence_hits([ds.labels[q]], [[c for p in tm if not p.get("partial") for c in p["chunk_ids"]]])[0]
            rows_tm.append(record(ds, q, "arag_tm", tm, tok, evidence_in_context=ftm, retrieval_hit10=f, cut=cut))
            own.append({**record(ds, q, "arag_own", ps, tok, evidence_in_context=f, retrieval_hit10=f),
                        "prediction": a["prediction"], "termination": a["termination"], "agent_error": a["error"],
                        "agent_reference_answers": a["reference_answers"]})
        out["arag"], out["arag_tm"], out["arag_own"] = rows, rows_tm, own
    # SAADI at the competitor's length: per query, SAADI's top 10 cut to as many tokens as that method's
    # context for the same query (RAPTOR and A-RAG give the reader fewer tokens than SAADI's top 10).
    saadi_by_q = {r["query_id"]: r for r in out["saadi"]}  # type: ignore[union-attr]
    for name, rows_src in (("saadi_at_raptor_len", out.get("raptor")), ("saadi_at_arag_len", out.get("arag"))):
        if not rows_src:
            out[name] = None
            continue
        log(f"{ds.key}: {name} (SAADI cut per query to the other method's context length)")
        rows = []
        for r in rows_src:
            q = r["query_id"]
            ps, cut = token_match(saadi_by_q[q]["passages"], int(r["context_tokens"]), tok)
            f = cc.evidence_hits([ds.labels[q]], [[c for p in ps if not p.get("partial") for c in p["chunk_ids"]]])[0]
            rows.append(record(ds, q, name, ps, tok, evidence_in_context=f, retrieval_hit10=None,
                               cut=cut, target_tokens=int(r["context_tokens"])))
        out[name] = rows
        diff = [r["query_id"] for r in own if r["agent_reference_answers"] and
                not {cc.normalize_answer(x) for x in r["agent_reference_answers"]} <= {cc.normalize_answer(x) for x in r["gold_answers"]}]
        info["arag"]["gold_answer_disagreements"] = {"n": len(diff), "examples": diff[:5]}
    return out


def stats(rows: list[dict[str, Any]] | None) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    ctx = [r["context_tokens"] for r in rows]
    pr = [r["prompt_tokens"] for r in rows]
    ev = [r.get("evidence_in_context") for r in rows]
    h10 = [r.get("retrieval_hit10") for r in rows]
    return {
        "n": len(rows),
        "passages_mean": mean(r["n_passages"] for r in rows),
        "context_tokens_mean": mean(ctx), "context_tokens_max": max(ctx),
        "prompt_tokens_mean": mean(pr), "prompt_tokens_max": max(pr), "prompt_tokens_total": sum(pr),
        "evidence_in_context": cc.hit_rate(ev),
        "hr10": cc.hit_rate(h10),
        "cut": sum(1 for r in rows if r.get("cut")),
    }


# ---------------------------------------------------------------------------------------------
# Estimates (rough; for the decision of what to run)
# ---------------------------------------------------------------------------------------------
PREFILL_TOKENS_PER_S = 8000.0  # Qwen3-30B-A3B, TP 4 on RTX A6000, batched; conservative
DECODE_TOKENS_PER_S = 1500.0  # aggregate over the batch
VLLM_STARTUP_H = 0.15


def reader_hours(total_prompt_tokens: int, n_requests: int, max_out: int) -> float:
    return VLLM_STARTUP_H + total_prompt_tokens / PREFILL_TOKENS_PER_S / 3600 + n_requests * max_out / DECODE_TOKENS_PER_S / 3600


# ---------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------


def write_report(paths: cc.C5Paths, info: dict[str, Any], status: str, exit_code: int) -> None:
    rep = paths.reports / f"{REPORT_NAME}.md"
    L: list[str] = []
    L.append(f"# C5 end-to-end QA - step 00 (prepare): {status}")
    L.append("")
    L.append(f"Written {cc.now()}; commit {info.get('commit')}; host {info.get('host')}; exit code {exit_code}.")
    L.append("")
    L.append("Command: `bash experiments/rebuttal/c5_end_to_end/sh/00_prepare.sh " + " ".join(info.get("argv", [])) + "`")
    L.append("")
    L.append("## Summary")
    L.append("")
    for s in info.get("summary", []):
        L.append(f"- {s}")
    L.append("")
    L.append("## Fixed settings")
    L.append("")
    L.append(f"- Base retriever: BM25 (the retriever of the paper's agent comparison, Table 2; its SAADI inputs are the bm25 `retrieval_depth` rankings of outputs/saadi/reference).")
    L.append(f"- SAADI: published point (lambda_M, lambda_S, mu_A, mu_D) = {info.get('published_point')} from `{info.get('config')}` (lambda_T = 1, z-score, max aggregation).")
    L.append(f"- Reader: `{info.get('reader_model')}`, vLLM, temperature 0, seed 42, max output tokens {info.get('max_output_tokens')}, max model length {info.get('max_model_len')}. Tokenizer for all counts: the reader's.")
    L.append("- One prompt for every condition: a single user message, no system prompt, chat template of the model. The exact template:")
    L.append("")
    L.append("```text")
    L.append(cc.READER_TEMPLATE)
    L.append("```")
    L.append("")
    L.append(f"`{{context}}` is the passages in rank order, each `[i] <text>` (whitespace collapsed), separated by a blank line; closed-book gets `{cc.EMPTY_CONTEXT}`. Scoring: SQuAD normalisation (lower case, no punctuation, no articles), EM and token F1 as the max over the gold answers, containment = a normalised gold answer inside the normalised output. The output is first cut to its first non-empty line and a leading `Answer:` is dropped.")
    if info.get("example_prompt"):
        L.append("")
        L.append("Example (MuSiQue-32K, SAADI, first query; passages shortened here to 120 characters):")
        L.append("")
        L.append("```text")
        L.extend(info["example_prompt"].splitlines())
        L.append("```")
    L.append("")
    for key in info.get("datasets", []):
        d = info["per_dataset"].get(key, {})
        L.append(f"## {cc.DATASET_DISPLAY[key]}")
        L.append("")
        L.append(f"Queries {d.get('queries')}, documents {d.get('documents')}, chunks {d.get('chunks')}; gold answers: {((d.get('gold') or {}).get('source'))}.")
        if "musique_aliases" in d:
            L.append(f"MuSiQue answer aliases: {json.dumps(d['musique_aliases'])}.")
        g = d.get("gate")
        if g:
            L.append("")
            L.append(f"Reproduction gate: **{'PASSED' if g.get('passed') else 'FAILED'}** (streams `{g.get('streams')}`).")
            L.append("")
            rows = [["SAADI top 10 vs paper's saved ranking", f"{g.get('saadi_identical')}/{d.get('queries')} identical",
                     f"HR@5 {g['saadi_hr'][5]:.2f}, HR@10 {g['saadi_hr'][10]:.2f}", f"hits {g['saadi_hits']} vs paper {g['paper_hits']}"],
                    ["Vanilla top 10 (text channel)", (f"{g.get('vanilla_identical')}/{d.get('queries')} identical" if "vanilla_identical" in g else "no per-query file"),
                     f"HR@5 {g['vanilla_hr'][5]:.2f}, HR@10 {g['vanilla_hr'][10]:.2f}", f"Table 1: {g.get('table1_vanilla_hr')}; reference: {g.get('vanilla_reference')}"]]
            L.extend(cc.md_table(["List", "Identical", "Ours", "Paper"], rows))
            for name in ("saadi_mismatches", "vanilla_mismatches"):
                if g.get(name):
                    L.append("")
                    L.append(f"First {len(g[name])} {name.replace('_', ' ')}:")
                    L.append("")
                    L.append("```json")
                    L.extend(json.dumps(m) for m in g[name])
                    L.append("```")
        L.append("")
        L.append("Saved comparison outputs:")
        L.append("")
        rows = []
        cr = d.get("cr") or {}
        if cr:
            rows.append(["Contextual Retrieval (C1)", cr.get("status"), f"{cr.get('covered', 0)}/{d.get('queries')}",
                         f"CR HR@10 {cr.get('cr_hr10', float('nan')):.2f} (C1 report {cr.get('c1_report_cr_hr10')}); C1 Vanilla/SAADI = ours: {cr.get('vanilla_identical_to_ours')}/{cr.get('saadi_identical_to_ours')}; {'; '.join(cr.get('problems', []))}", cr.get("path")])
        ra = d.get("raptor") or {}
        if ra:
            rows.append(["RAPTOR", ra.get("status"), f"{ra.get('covered', 0)}/{d.get('queries')}",
                         f"{ra.get('source')}; document text check {ra.get('document_text_check')}; records {ra.get('records')}; own-queries HR@10 (upper bound) {fmt(ra.get('own_queries_hr10'))} vs paper {ra.get('paper_hr10')}; same question {ra.get('same_question')}, other document {ra.get('other_document')}", ra.get("run")])
        ar = d.get("arag") or {}
        if ar:
            rows.append(["A-RAG", ar.get("status"), f"{ar.get('covered', 0)}/{d.get('queries')}",
                         f"HR {fmt(ar.get('hr'))} vs Table 2 {ar.get('paper_hr')}; mean read {fmt(ar.get('mean_read'), 4)} vs {ar.get('paper_mean_chunks')}; unknown ids {ar.get('unknown_chunk_ids')}; terminations {ar.get('termination')}; {'; '.join(ar.get('problems', []))}", ar.get("run")])
        L.extend(cc.md_table(["Method", "Status", "C5 queries covered", "Checks", "Source"], rows))
        if ra.get("other_document_examples"):
            L.append("")
            L.append(f"RAPTOR other-document examples: {json.dumps(ra['other_document_examples'])}")
        st = d.get("stats")
        if st:
            L.append("")
            L.append(f"Contexts (token budget of the token-matched rows = mean SAADI top-10 context = {d.get('token_budget')} tokens):")
            L.append("")
            rows = []
            for c in cc.ALL_CONDITIONS:
                s = st.get(c) or {"n": 0}
                if not s.get("n"):
                    rows.append([c, 0, "—", "—", "—", "—", "—", "—", "missing"])
                    continue
                rows.append([c, s["n"], f"{s['passages_mean']:.2f}", f"{s['context_tokens_mean']:.0f}", s["context_tokens_max"], f"{s['prompt_tokens_mean']:.0f}", s["prompt_tokens_max"],
                             fmt(s["evidence_in_context"]["hr_percent"]) + f" ({s['evidence_in_context']['eligible']})", fmt(s["hr10"]["hr_percent"]) + (f"; cut {s['cut']}" if s.get("cut") else "")])
            L.extend(cc.md_table(["Condition", "n", "Passages", "Ctx tok mean", "Ctx tok max", "Prompt tok mean", "Prompt tok max", "Evidence in context % (eligible)", "HR@10 %"], rows))
        pl = d.get("para_labels") or {}
        if pl.get("available"):
            L.append("")
            L.append(f"Paragraph labels (`{pl['file']}`): {pl['rule']}")
            L.append("")
            L.extend(cc.md_table(["Hops", "Queries", "Paragraphs", "Located %", "Queries with none located", "Chunks per query: paragraph labels", "Chunks per query: paper labels",
                                  "Paper-label chunks inside a paragraph %", "Paper labels cover every paragraph"],
                                 [[t["hops"], t["queries"], t["paragraphs"], fmt(t["paragraphs_located_pct"], 1), t["queries_without_any_located"],
                                   fmt(t["chunks_per_query_paragraph_labels"], 2), fmt(t["chunks_per_query_paper_labels"], 2),
                                   fmt(t["paper_label_chunks_in_a_paragraph_pct"], 1), t["paper_labels_cover_every_paragraph"]] for t in pl["by_hops"]]))
            ev = pl.get("evidence") or {}
            if ev:
                L.append("")
                L.append("Evidence in context with the paragraph labels (any / every located paragraph; for the top-10 rows this is HR@10):")
                L.append("")
                L.extend(cc.md_table(["Condition", "Any %", "All %", "Eligible"],
                                     [[c, fmt(e["any"]["hr_percent"], 1), fmt(e["all"]["hr_percent"], 1), e["all"]["eligible"]] for c, e in ev.items()]))
        elif pl:
            L.append("")
            L.append(f"Paragraph labels: not built ({pl.get('reason')}).")
        L.append("")
    L.append("## Comparison-method settings (defaults; nothing tuned)")
    L.append("")
    for key in info.get("datasets", []):
        d = info["per_dataset"].get(key, {})
        for name in ("cr", "raptor", "arag"):
            s = (d.get(name) or {}).get("settings")
            if s:
                L.append(f"- {cc.DATASET_DISPLAY[key]} {name}: " + "; ".join(f"{k} = {v}" for k, v in s.items()))
            diff = (d.get(name) or {}).get("settings_differ_from_official")
            if diff:
                L.append(f"  - differs from official RAPTOR defaults [saved, official]: " + "; ".join(f"{k}: {v}" for k, v in diff.items()))
    L.append("")
    L.append("## What is missing and what running it would cost")
    L.append("")
    L.extend(cc.md_table(["Dataset", "Item", "Status", "Estimated wall hours (4 GPUs)", "GPU hours", "Basis"], info.get("missing_rows", [])) or ["Nothing missing."])
    L.append("")
    L.append("## Checks of unchanged inputs")
    L.append("")
    pv = info.get("preserved", {})
    L.append(f"- docs/preserved_sources.json: {pv.get('checked')} files checked, changed {len(pv.get('changed', []))}, missing {len(pv.get('missing', []))} {pv.get('changed', [])[:5]} {pv.get('note', '')}")
    L.append(f"- configs/methods/sadi.yaml sha256 {info.get('config_sha256')}")
    if info.get("warnings"):
        L.append("")
        L.append("## Warnings")
        L.append("")
        L.extend(f"- {w}" for w in info["warnings"])
    if info.get("traceback"):
        L.append("")
        L.append("## Error")
        L.append("")
        L.append("```text")
        L.extend(info["traceback"].splitlines()[-60:])
        L.append("```")
    L.append("")
    L.append("## Files written")
    L.append("")
    L.extend(f"- `{f}`" for f in info.get("files", []))
    rep.write_text("\n".join(L) + "\n", encoding="utf-8")
    cc.write_json(paths.reports / f"{REPORT_NAME}.json", info)
    (paths.reports / f"{REPORT_NAME}.summary.txt").write_text(info.get("one_line", status) + "\n", encoding="utf-8")


def fmt(v: Any, nd: int = 2) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v)


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", default=os.environ.get("C5_DATASETS", ",".join(cc.DATASETS)))
    ap.add_argument("--tag", default="", help="report name suffix, e.g. musique_32k for the re-prepare after a RAPTOR re-run")
    args = ap.parse_args(argv)
    global REPORT_NAME
    REPORT_NAME = f"step_00_{args.tag}" if args.tag else "step_00"
    env = dict(os.environ)
    paths = cc.C5Paths.from_env()
    datasets = [d for d in args.datasets.split(",") if d]
    info: dict[str, Any] = {"step": STEP, "argv": sys.argv[1:], "commit": cc.git_commit(paths.root), "host": os.uname().nodename,
                            "datasets": datasets, "per_dataset": {}, "files": [], "warnings": [], "summary": [], "missing_rows": [],
                            "reader_model": env.get("C5_READER_MODEL"), "max_output_tokens": int(env.get("C5_MAX_OUTPUT_TOKENS", 64)),
                            "max_model_len": int(env.get("C5_MAX_MODEL_LEN", 32768)), "config": env.get("C5_CONFIG")}
    exit_code, status = 0, "OK"
    try:
        from saadi.rebuttal.heldout_fusion import PUBLISHED_POINT

        cfg = Path(env["C5_CONFIG"])
        point, block = published_point_from_config(cfg)
        info["published_point"], info["config_sha256"] = point, cc.sha256_file(cfg)
        if tuple(point) != tuple(PUBLISHED_POINT):
            raise GateFailed(f"published point from {cfg} {point} differs from the repo's PUBLISHED_POINT {PUBLISHED_POINT}")
        info["preserved"] = check_preserved(paths.root)
        if info["preserved"]["changed"] or info["preserved"]["missing"]:
            raise GateFailed(f"preserved sources changed: {info['preserved']['changed'][:5]} missing: {info['preserved']['missing'][:5]}")

        # 1-2. Inputs and the reproduction gate for every dataset, before any context is built.
        dsets: dict[str, Dataset] = {}
        for key in datasets:
            log(f"{key}: loading saved queries, labels and chunk text")
            ds = Dataset(key, paths.root)
            d = info["per_dataset"].setdefault(key, {})
            d.update({"queries": len(ds.qids), "documents": ds.n_docs, "chunks": len(ds.chunks)})
            load_gold_answers(ds, env, d)
            reproduce(ds, paths, env, point, d)
            dsets[key] = ds
        failed = [k for k in datasets if not info["per_dataset"][k]["gate"]["passed"]]
        if failed:
            raise GateFailed(f"reproduction gate failed on {failed}; the first mismatches are in the report. Nothing else was built.")

        # 3-4. Saved comparison outputs and contexts.
        tok = Tok(env["C5_READER_MODEL"])
        max_prompt = info["max_model_len"] - info["max_output_tokens"]
        stats_rows = []
        for key, ds in dsets.items():
            d = info["per_dataset"][key]
            cr = load_cr(ds, env, d)
            raptor = load_raptor(ds, env, d)
            arag = load_arag(ds, env, d)
            build_para_labels(ds, paths, d)
            if (d.get("para_labels") or {}).get("available"):
                info["files"].append(d["para_labels"]["file"])
            ctx = build_contexts(ds, tok, cr, raptor, arag, d)
            d["token_budget"] = d.get("token_budget")
            manifest = {"dataset": key, "created": cc.now(), "commit": info["commit"], "reader_model": env["C5_READER_MODEL"],
                        "template_sha256": __import__("hashlib").sha256(cc.READER_TEMPLATE.encode()).hexdigest(),
                        "token_budget": d["token_budget"], "published_point": point, "conditions": {}}
            d["stats"] = {}
            for cond in cc.ALL_CONDITIONS:
                rows = ctx.get(cond)
                path = paths.context_file(key, cond)
                if not rows:
                    manifest["conditions"][cond] = {"available": False}
                    if path.exists():
                        path.unlink()
                    d["stats"][cond] = {"n": 0}
                    continue
                too_long = [r["query_id"] for r in rows if r["prompt_tokens"] > max_prompt]
                if too_long:
                    raise GateFailed(f"{key}/{cond}: {len(too_long)} prompts exceed C5_MAX_MODEL_LEN - C5_MAX_OUTPUT_TOKENS = {max_prompt} tokens (e.g. {too_long[:3]}); raise C5_MAX_MODEL_LEN")
                cc.write_jsonl(path, rows)
                manifest["conditions"][cond] = {"available": True, "file": paths.rel(path), "n": len(rows), "complete": len(rows) == len(ds.qids), "sha256": cc.sha256_file(path)}
                info["files"].append(paths.rel(path))
                s = stats(rows)
                d["stats"][cond] = s
                stats_rows.append({"dataset": key, "condition": cond, **{k: v for k, v in s.items() if not isinstance(v, dict)},
                                   "evidence_in_context_pct": s["evidence_in_context"]["hr_percent"], "evidence_eligible": s["evidence_in_context"]["eligible"],
                                   "hr10_pct": s["hr10"]["hr_percent"], "token_budget": d["token_budget"]})
            # Paragraph-label evidence, in its own file: the contexts files (and so the reader's saved
            # answers, keyed on their sha256) stay as they were.
            pe_path = paths.para_evidence_file(key)
            if ds.para:
                pe = para_evidence_rows(ds, ctx)
                cc.write_jsonl(pe_path, pe)
                manifest["para_evidence"] = {"available": True, "file": paths.rel(pe_path), "rows": len(pe), "sha256": cc.sha256_file(pe_path),
                                             "labels": d["para_labels"]["file"], "rule": d["para_labels"]["rule"]}
                info["files"].append(paths.rel(pe_path))
                pstat = {}
                for c in cc.ALL_CONDITIONS:
                    rs = [r for r in pe if r["condition"] == c]
                    if rs:
                        pstat[c] = {"any": cc.hit_rate(r["evidence_para_any"] for r in rs), "all": cc.hit_rate(r["evidence_para_all"] for r in rs)}
                d["para_labels"]["evidence"] = pstat
            else:
                manifest["para_evidence"] = {"available": False}
                if pe_path.exists():
                    pe_path.unlink()
            cc.write_json(paths.context_manifest(key), manifest)
            info["files"].append(paths.rel(paths.context_manifest(key)))
            # Missing items and costs.
            if not manifest["conditions"]["cr"]["available"]:
                info["missing_rows"].append([key, "Contextual Retrieval", (d.get("cr") or {}).get("status"), "—", "—", "C1 is a separate run (its own GPU scripts); not re-run inside C5"])
            if not manifest["conditions"]["raptor"]["available"]:
                prof = (d.get("raptor") or {}).get("profiling_build_seconds") or {}
                per_doc = prof.get("per_document")
                hrs = (per_doc * ds.n_docs / 3600 + 0.2) if per_doc else None
                info["missing_rows"].append([key, "RAPTOR (default settings, build trees for the C5 documents + retrieve)", (d.get("raptor") or {}).get("status"),
                                             fmt(hrs, 1), fmt(hrs * 4 if hrs else None, 1),
                                             f"saved run: {fmt(per_doc, 0)} s per document to build on 4 GPUs x {ds.n_docs} documents + vLLM start" if per_doc else "no saved profiling"])
            if not manifest["conditions"]["arag"]["available"]:
                info["missing_rows"].append([key, "A-RAG", (d.get("arag") or {}).get("status"), "—", "—", "see the A-RAG checks"])
            reader_req = sum(s.get("n", 0) for c, s in d["stats"].items() if c in cc.CONDITIONS)
            reader_tok = sum(s.get("prompt_tokens_total", 0) for c, s in d["stats"].items() if c in cc.CONDITIONS)
            h = reader_hours(reader_tok, reader_req, info["max_output_tokens"])
            info["missing_rows"].append([key, f"Reader, {reader_req} prompts ({reader_tok / 1e6:.1f}M prompt tokens) over the available conditions", "to run in the GPU script", fmt(h, 1), fmt(4 * h, 1),
                                         f"rough: {PREFILL_TOKENS_PER_S:.0f} prompt tok/s, {DECODE_TOKENS_PER_S:.0f} output tok/s, {VLLM_STARTUP_H} h start"])
        for key in dsets:
            rows_k = [r for r in stats_rows if r["dataset"] == key]
            if not rows_k:
                continue
            out_csv = paths.results / "prepare" / f"context_stats_{key}.csv"
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            cols = list(dict.fromkeys(k for r in rows_k for k in r))
            with open(out_csv, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=cols)
                w.writeheader()
                w.writerows(rows_k)
            info["files"].append(paths.rel(out_csv))
        # Example prompt.
        if "musique_32k" in dsets and (paths.context_file("musique_32k", "saadi")).exists():
            r = cc.read_jsonl(paths.context_file("musique_32k", "saadi"))[0]
            short = [{**p, "text": p["text"][:120] + ("..." if len(p["text"]) > 120 else "")} for p in r["passages"]]
            info["example_prompt"] = cc.build_prompt(r["question"], short)
        # Summary sentences.
        for key in datasets:
            d = info["per_dataset"][key]
            avail = [c for c in cc.ALL_CONDITIONS if (d.get("stats", {}).get(c) or {}).get("n")]
            miss = [c for c in cc.ALL_CONDITIONS if c not in avail and not (c == "oracle_para" and not (d.get("para_labels") or {}).get("available"))]
            g = d["gate"]
            info["summary"].append(f"{cc.DATASET_DISPLAY[key]}: gate passed (SAADI {g['saadi_identical']}/{d['queries']} identical to the paper, HR@10 {g['saadi_hr'][10]:.2f}; Vanilla HR@10 {g['vanilla_hr'][10]:.2f}). "
                                   f"Contexts built: {', '.join(avail)}." + (f" Missing: {', '.join(miss)}." if miss else ""))
        missing_all = [f"{k}:{c}" for k in datasets for c in ("cr", "raptor", "arag") if not ((info["per_dataset"][k].get("stats") or {}).get(c) or {}).get("n")]
        info["one_line"] = f"gate passed on {len(datasets)} datasets; contexts written" + (f"; missing {', '.join(missing_all)}" if missing_all else "; nothing missing")
        if missing_all:
            status = "OK (some comparison outputs missing)"
    except GateFailed as exc:
        exit_code, status = 1, "FAILED"
        info["summary"].append(f"Stopped: {exc}")
        info["one_line"] = f"FAILED: {str(exc)[:120]}"
        info["traceback"] = traceback.format_exc()
    except Exception as exc:  # unexpected: report with traceback
        exit_code, status = 1, "FAILED (unexpected error)"
        info["summary"].append(f"Unexpected error: {type(exc).__name__}: {exc}")
        info["one_line"] = f"FAILED: {type(exc).__name__}: {str(exc)[:100]}"
        info["traceback"] = traceback.format_exc()
    finally:
        for key in datasets:
            d = info["per_dataset"].get(key, {})
            for w in d.pop("warnings", []) if isinstance(d.get("warnings"), list) else []:
                info["warnings"].append(w)
        info["files"] += [paths.rel(paths.reports / f"{REPORT_NAME}.md"), paths.rel(paths.reports / f"{REPORT_NAME}.json")]
        write_report(paths, info, status, exit_code)
    log(f"done: {status}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
