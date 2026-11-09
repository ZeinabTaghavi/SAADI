"""C2: SAADI's fusion weights chosen on held-out queries (ICLR rebuttal).

Two steps, as in the published query-time scoring (indexing/retrieval_methods/mbr.py):

1. Streams (GPU for dense retrievers, CPU for BM25): for every saved query of a
   (dataset, retriever), ``score_mbr_variant_document_pools`` gives the raw per-chunk
   channel scores (text, name, alias, description, summary). The published
   combination (``combine_mbr_variant_chunk_scores``) must reproduce the paper's saved
   rankings before the streams are saved.
2. Analysis (CPU): z-score each channel with the repo's ``normalize_signal_scores``,
   then re-weight on a grid with a vectorized copy of the combination step that does
   the same floating-point operations in the same order. The copy is checked against
   the repo function (full rankings) at the published point and 20 random grid points.

Everything else (cross-validated selection, leave-one-dataset-out transfer, the Table 9
reconstruction test, bootstrap intervals, ranking in Table 1, sensitivity, a learned
fusion) runs on the saved per-query grid results. Seed 42 everywhere.

Nothing here changes the published scoring path: it is imported, never edited.
"""

from __future__ import annotations

import csv
import json
import math
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..core.types import RetrievalResult
from ..evaluation.retrieval_metrics import evaluate_retrieval_results
from ..indexing.retrieval_methods.mbr import (
    MBR_VARIANT_MBR2,
    MBRChunkPoolRecord,
    MBRChunkScoreRecord,
    MBRChunkScoreStreams,
    MBRDocumentMemoryIndex,
    build_mbr_document_memory_index,
    combine_mbr_variant_chunk_scores,
    normalize_signal_scores,
    score_mbr_variant_document_pools,
)
from .frozen_records import DATASETS, PUBLISHED_SETTINGS, RETRIEVE_K, load_saved_live_rows, paired_bootstrap_ci, reference_dir

EXPERIMENT_ID = "C2_heldout_fusion"
SEED = 42
N_FOLDS = 5
N_BOOT = 10_000

DATASET_KEYS = ("loogle", "musique_32k", "qasper_64k", "novelhopqa")
RETRIEVERS = ("bm25", "contriever", "bge_m3", "jina", "qwen")
DENSE_RETRIEVERS = tuple(r for r in RETRIEVERS if r != "bm25")
DATASET_DISPLAY = {"loogle": "LooGLE", "musique_32k": "MuSiQue-32K", "qasper_64k": "QASPER-64K", "novelhopqa": "NovelHopQA"}
RETRIEVER_DISPLAY = {"bm25": "BM25", "contriever": "Contriever", "bge_m3": "BGE-M3", "jina": "Jina", "qwen": "Qwen"}
# Names used by outputs/other_methods/vanilla/retrieval.csv
VANILLA_DATASET_NAME = {"loogle": "loogle", "musique_32k": "musique", "qasper_64k": "qasper", "novelhopqa": "novelhopqa"}

CHANNELS = ("text", "name", "alias", "description", "summary")

# Grid (lambda_T = mu_N = 1 fixed; only ratios matter).
LAMBDA_GRID = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0)
MU_GRID = (0.0, 0.3, 0.5, 1.0)
DRY_LAMBDA_GRID = (0.0, 0.3, 0.5, 1.0)
DRY_MU_GRID = (0.0, 1.0)
PUBLISHED_POINT = (
    float(PUBLISHED_SETTINGS["lambda_n"]),
    float(PUBLISHED_SETTINGS["lambda_h"]),
    float(PUBLISHED_SETTINGS["memory_alias_weight"]),
    float(PUBLISHED_SETTINGS["memory_description_weight"]),
)
N_VERIFY_POINTS = 20

# Dense reproduction gate (task C2, section 4).
DENSE_HR_TOLERANCE_POINTS = 0.1
DENSE_MIN_IDENTICAL_TOP10 = 0.99
# Decided with the user (2026-11-09) after the first server run: Jina's recomputed embeddings differ from the
# paper's run by about 1e-3 in score (68-83% identical top-10; HR within 0.33 points). For these retrievers the
# ranking-identity and 0.1-point gates become drift notes, provided HR@5/HR@10 stay within DRIFT_MAX_HR_POINTS of
# the saved values. Every system of such a pair (Vanilla, published, held-out) uses the same recomputed streams.
DRIFT_ACCEPTED_RETRIEVERS = ("jina",)
DRIFT_MAX_HR_POINTS = 0.5


def job_name(dataset: str, retriever: str) -> str:
    return f"{dataset}__{retriever}"


ALL_JOBS: tuple[tuple[str, str], ...] = tuple((d, r) for d in DATASET_KEYS for r in RETRIEVERS)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class C2Paths:
    repo_root: Path
    dry_run: bool = False
    # Optional overrides (tests); default layout below.
    out_base: Path | None = None
    results_base: Path | None = None
    exp_base: Path | None = None

    @property
    def out_dir(self) -> Path:
        base = self.out_base or (self.repo_root / "outputs" / "rebuttal" / EXPERIMENT_ID)
        return base / "dry_run" if self.dry_run else base

    @property
    def results_dir(self) -> Path:
        base = self.results_base or (self.repo_root / "results" / "rebuttal" / EXPERIMENT_ID)
        return base / "dry_run" if self.dry_run else base

    @property
    def exp_dir(self) -> Path:
        return self.exp_base or (self.repo_root / "experiments" / "rebuttal" / EXPERIMENT_ID)

    def as_kwargs(self) -> dict[str, Any]:
        return {
            "repo_root": str(self.repo_root),
            "dry_run": self.dry_run,
            "out_base": str(self.out_base) if self.out_base else None,
            "results_base": str(self.results_base) if self.results_base else None,
            "exp_base": str(self.exp_base) if self.exp_base else None,
        }

    @classmethod
    def from_kwargs(cls, kw: dict[str, Any]) -> C2Paths:
        return cls(
            Path(kw["repo_root"]),
            dry_run=bool(kw["dry_run"]),
            out_base=Path(kw["out_base"]) if kw.get("out_base") else None,
            results_base=Path(kw["results_base"]) if kw.get("results_base") else None,
            exp_base=Path(kw["exp_base"]) if kw.get("exp_base") else None,
        )

    @property
    def reports_dir(self) -> Path:
        return self.exp_dir / "reports" / ("dry_run" if self.dry_run else "")

    @property
    def logs_dir(self) -> Path:
        return self.exp_dir / "logs" / ("dry_run" if self.dry_run else "")

    @property
    def streams_dir(self) -> Path:
        return self.out_dir / "streams"

    @property
    def failed_dir(self) -> Path:
        return self.out_dir / "streams_failed"

    @property
    def markers_dir(self) -> Path:
        return self.out_dir / "markers"

    @property
    def work_dir(self) -> Path:
        return self.out_dir / "work"

    @property
    def grid_dir(self) -> Path:
        return self.out_dir / "grid"

    @property
    def plan_path(self) -> Path:
        return self.out_dir / "plan.json"

    def done_marker(self, job: str) -> Path:
        return self.markers_dir / f"{job}.DONE"

    def failed_marker(self, job: str) -> Path:
        return self.markers_dir / f"{job}.FAILED"

    def stream_path(self, job: str) -> Path:
        return self.streams_dir / f"{job}.parquet"

    def queries_path(self, job: str) -> Path:
        return self.streams_dir / f"{job}.queries.json"

    def check_path(self, job: str) -> Path:
        return self.streams_dir / f"{job}.check.json"

    def ensure(self) -> None:
        for p in (
            self.out_dir,
            self.results_dir,
            self.reports_dir,
            self.logs_dir,
            self.streams_dir,
            self.markers_dir,
            self.work_dir,
            self.grid_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def write_json(path: Path, payload: Any) -> None:
    _atomic_write_bytes(path, (json.dumps(payload, indent=2, sort_keys=False, default=_json_default) + "\n").encode("utf-8"))


def rel(path: Path, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(path)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serializable: {type(obj)!r}")


# ---------------------------------------------------------------------------
# Saved inputs
# ---------------------------------------------------------------------------


def rows_by_document(rows: list[dict[str, Any]]) -> dict[str, list[tuple[int, dict[str, Any]]]]:
    """Saved rows grouped by document, first-appearance order, saved order inside a document."""

    out: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for i, row in enumerate(rows):
        out.setdefault(str(row.get("doc_id", "")), []).append((i, row))
    return out


def evidence_of(row: dict[str, Any]) -> tuple[list[str], list[str], list[list[str]]]:
    gold = [str(x) for x in row.get("gold_chunk_ids", []) or []]
    silver = [str(x) for x in row.get("silver_chunk_ids", []) or []]
    groups = [[str(c) for c in g] for g in (row.get("silver_chunk_groups", []) or []) if isinstance(g, list)]
    return gold, silver, groups


def is_eligible(gold: list[str], groups: list[list[str]]) -> bool:
    # retrieval_metrics.evaluate_retrieval_results: eligible = bool(gold_set or silver_groups);
    # its _normalize_chunk_groups drops empty groups.
    return bool(gold) or any(len(g) > 0 for g in groups)


def published_vanilla(repo_root: Path) -> dict[tuple[str, str, int], float]:
    """Table 1 Vanilla HR (percent, one decimal) from outputs/other_methods/vanilla/retrieval.csv."""

    path = repo_root / "outputs" / "other_methods" / "vanilla" / "retrieval.csv"
    out: dict[tuple[str, str, int], float] = {}
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("study") != "main" or row.get("source_id") != "main_fixed_depth" or row.get("hop") != "overall":
                continue
            out[(row["dataset"], row["retriever"], int(row["cutoff"]))] = float(row["hr_percent"])
    return out


def per_query_vanilla_file(dataset: str, retriever: str, repo_root: Path) -> Path | None:
    """Text-only per-query rankings (lambda_n = lambda_h = 0), saved for BM25 on LooGLE and NovelHopQA."""

    spec = DATASETS[dataset]
    path = (
        repo_root / spec.namespace / "retrieval_reports" / f"retrieval_results__{retriever}__mbr_lt1_ln0_lh0_norm_none__per_document.json"
    )
    return path if path.exists() else None


def expected_main_metrics(dataset: str, retriever: str, repo_root: Path) -> dict[int, dict[str, Any]]:
    spec = DATASETS[dataset]
    with (reference_dir(repo_root) / "expected_metrics.json").open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    return {
        int(r["cutoff"]): r
        for r in rows
        if r.get("dataset") == spec.metrics_name
        and r.get("retriever") == retriever
        and r.get("study") == "main"
        and r.get("condition") == "default"
    }


# ---------------------------------------------------------------------------
# Step 1: streams (one job = one dataset x retriever)
# ---------------------------------------------------------------------------


def _published_combine(document_index: MBRDocumentMemoryIndex, streams: MBRChunkScoreStreams, *, lambda_n: float, lambda_h: float):
    cfg = PUBLISHED_SETTINGS
    return combine_mbr_variant_chunk_scores(
        document_index,
        streams,
        variant=MBR_VARIANT_MBR2,
        lambda_t=cfg["lambda_t"],
        lambda_n=lambda_n,
        lambda_h=lambda_h,
        normalization_method=cfg["normalization_method"],
        memory_name_weight=cfg["memory_name_weight"],
        memory_alias_weight=cfg["memory_alias_weight"],
        memory_description_weight=cfg["memory_description_weight"],
    )


def _score_document(document_index: MBRDocumentMemoryIndex, doc_rows: list[tuple[int, dict[str, Any]]], retriever: str) -> dict[str, Any]:
    cfg = PUBLISHED_SETTINGS
    chunk_ids = [str(c.chunk_id) for c in document_index.chunk_pool]
    doc: dict[str, Any] = {"doc_id": document_index.doc_id, "chunk_ids": chunk_ids, "chunk_index": None, "queries": [], "anomalies": []}
    for row_index, row in doc_rows:
        question = str(row.get("question", "") or "")
        streams = score_mbr_variant_document_pools(
            document_index,
            query=question,
            retriever_name=retriever,
            variant=MBR_VARIANT_MBR2,
            memory_name_weight=cfg["memory_name_weight"],
            memory_alias_weight=cfg["memory_alias_weight"],
            memory_description_weight=cfg["memory_description_weight"],
            memory_aggregation=cfg["memory_aggregation"],
        )
        if set(streams.text_scores) != set(chunk_ids):
            doc["anomalies"].append(f"query row {row_index}: text_scores keys differ from the chunk pool")
        published = _published_combine(document_index, streams, lambda_n=cfg["lambda_n"], lambda_h=cfg["lambda_h"])
        vanilla = _published_combine(document_index, streams, lambda_n=0.0, lambda_h=0.0)
        index_by_id = {r.chunk_id: int(r.chunk_index) for r in published.ranked_chunks}
        chunk_index = [index_by_id[c] for c in chunk_ids]
        if doc["chunk_index"] is None:
            doc["chunk_index"] = chunk_index
        elif doc["chunk_index"] != chunk_index:
            raise RuntimeError(f"{document_index.doc_id}: chunk indices differ between queries")
        raw = np.full((len(chunk_ids), len(CHANNELS)), np.nan, dtype=np.float64)
        for i, cid in enumerate(chunk_ids):
            rec = streams.chunk_scores[cid]
            raw[i, 0] = float(rec.text_score)
            for j, value in enumerate((rec.name_score, rec.alias_score, rec.description_score, rec.summary_score), start=1):
                if value is not None:
                    raw[i, j] = float(value)
        doc["queries"].append(
            {
                "row_index": int(row_index),
                "raw": raw,
                "published_top": [r.chunk_id for r in published.ranked_chunks[:RETRIEVE_K]],
                "published_scores": [float(r.final_score) for r in published.ranked_chunks[:RETRIEVE_K]],
                "vanilla_top": [r.chunk_id for r in vanilla.ranked_chunks[:RETRIEVE_K]],
            }
        )
    return doc


def compute_job_streams(
    dataset: str,
    retriever: str,
    *,
    paths: C2Paths,
    max_docs: int | None = None,
    log=print,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Score every saved query of one (dataset, retriever); per-document checkpoints make it resumable."""

    repo_root = paths.repo_root
    spec = DATASETS[dataset]
    index_path = repo_root / spec.index_json
    saved_rows = load_saved_live_rows(spec, retriever, repo_root=repo_root)
    by_doc = rows_by_document(saved_rows)
    doc_ids = list(by_doc)
    if max_docs is not None:
        doc_ids = doc_ids[: int(max_docs)]
    job = job_name(dataset, retriever)
    work = paths.work_dir / job
    work.mkdir(parents=True, exist_ok=True)
    timings = {"n_docs": len(doc_ids), "docs_from_checkpoint": 0, "seconds_scoring": 0.0}
    docs: list[dict[str, Any]] = []
    started = time.perf_counter()
    for di, doc_id in enumerate(doc_ids):
        ck = work / f"doc_{di:04d}.pkl"
        if ck.exists():
            with ck.open("rb") as handle:
                doc = pickle.load(handle)
            if doc.get("doc_id") == doc_id:
                docs.append(doc)
                timings["docs_from_checkpoint"] += 1
                continue
        t0 = time.perf_counter()
        document_index = build_mbr_document_memory_index(index_path, doc_id=doc_id)
        doc = _score_document(document_index, by_doc[doc_id], retriever)
        _atomic_write_bytes(ck, pickle.dumps(doc, protocol=pickle.HIGHEST_PROTOCOL))
        docs.append(doc)
        dt = time.perf_counter() - t0
        timings["seconds_scoring"] += dt
        log(f"[{job}] doc {di + 1}/{len(doc_ids)} {doc_id}: {len(doc['queries'])} queries, {len(doc['chunk_ids'])} chunks, {dt:.1f}s")
    timings["seconds_total"] = time.perf_counter() - started
    used_rows = [saved_rows[q["row_index"]] for d in docs for q in d["queries"]]
    return docs, used_rows, timings


def _results_from(rows: list[dict[str, Any]], rankings: list[list[str]]) -> list[RetrievalResult]:
    out = []
    for row, ranking in zip(rows, rankings, strict=True):
        gold, silver, groups = evidence_of(row)
        out.append(
            RetrievalResult(
                query_id=str(row.get("query_id", "")),
                doc_id=str(row.get("doc_id", "")),
                question=str(row.get("question", "")),
                retrieved_chunk_ids=list(ranking),
                retrieved_indices=[],
                scores=[],
                gold_chunk_ids=gold,
                silver_chunk_ids=silver,
                silver_chunk_groups=groups,
            )
        )
    return out


def _hr(rows: list[dict[str, Any]], rankings: list[list[str]]) -> dict[str, Any]:
    report = evaluate_retrieval_results(_results_from(rows, rankings), eval_k=5, retrieve_k=RETRIEVE_K, compute_per_query=False)
    return {
        "eligible": int(report.n_union_eligible),
        "hits": {k: report.hits_at_k.get(k) for k in ("5", "10")},
        "hr_percent": {k: (100.0 * report.hit_rate_at_k[k] if report.hit_rate_at_k.get(k) is not None else None) for k in ("5", "10")},
    }


def reproduction_check(
    dataset: str,
    retriever: str,
    docs: list[dict[str, Any]],
    used_rows: list[dict[str, Any]],
    *,
    repo_root: Path,
    full_dataset: bool,
) -> dict[str, Any]:
    """Published weights on the new streams vs the paper's saved rankings and expected_metrics.json."""

    queries = [q for d in docs for q in d["queries"]]
    saved_top = [[str(x) for x in row.get("retrieved_chunk_ids", [])][:RETRIEVE_K] for row in used_rows]
    ours_top = [q["published_top"] for q in queries]
    vanilla_top = [q["vanilla_top"] for q in queries]
    identical = [a == b for a, b in zip(ours_top, saved_top, strict=True)]
    n = len(queries)
    hr_ours = _hr(used_rows, ours_top)
    hr_saved = _hr(used_rows, saved_top)
    hr_vanilla = _hr(used_rows, vanilla_top)

    score_diffs = []
    for row, q, same in zip(used_rows, queries, identical, strict=True):
        if same and row.get("scores"):
            score_diffs.extend(abs(float(a) - float(b)) for a, b in zip(q["published_scores"], row["scores"], strict=False))
    examples = []
    for row, q, same in zip(used_rows, queries, identical, strict=True):
        if not same and len(examples) < 5:
            examples.append(
                {
                    "query_id": row.get("query_id"),
                    "doc_id": row.get("doc_id"),
                    "saved_top10": [str(x) for x in row.get("retrieved_chunk_ids", [])][:RETRIEVE_K],
                    "saved_scores": row.get("scores", [])[:RETRIEVE_K],
                    "ours_top10": q["published_top"],
                    "ours_scores": q["published_scores"],
                }
            )

    out: dict[str, Any] = {
        "dataset": dataset,
        "retriever": retriever,
        "n_queries": n,
        "full_dataset": bool(full_dataset),
        "identical_top10": int(sum(identical)),
        "identical_top10_share": (sum(identical) / n) if n else None,
        "hr_ours_published": hr_ours,
        "hr_saved": hr_saved,
        "max_abs_score_diff_identical_rankings": max(score_diffs) if score_diffs else None,
        "mismatch_examples": examples,
        "expected_metrics": None,
        "text_only": {"hr": hr_vanilla},
        "anomalies": [a for d in docs for a in d.get("anomalies", [])],
    }

    problems: list[str] = []
    drift: list[str] = []  # gate failures that only reflect embedding drift (dense retrievers)
    dense = retriever != "bm25"
    if not dense:
        if not all(identical):
            problems.append(f"BM25 top-10 rankings differ from the saved rankings for {n - sum(identical)} of {n} queries")
    else:
        if n and sum(identical) / n < DENSE_MIN_IDENTICAL_TOP10:
            drift.append(f"identical top-10 for {sum(identical)}/{n} queries (< {DENSE_MIN_IDENTICAL_TOP10:.0%})")
        for k in ("5", "10"):
            a, b = hr_ours["hr_percent"][k], hr_saved["hr_percent"][k]
            if a is None or b is None or abs(a - b) > DENSE_HR_TOLERANCE_POINTS + 1e-9:
                drift.append(f"HR@{k} {a:.2f} vs saved {b:.2f} (tolerance {DENSE_HR_TOLERANCE_POINTS} points)")

    if full_dataset:
        expected = expected_main_metrics(dataset, retriever, repo_root)
        exp_out: dict[str, Any] = {}
        for k in (5, 10):
            row = expected.get(k)
            if row is None:
                problems.append(f"no main/default expected_metrics row at cutoff {k}")
                continue
            ours_hits = hr_ours["hits"][str(k)]
            exp_out[str(k)] = {"eligible": row["eligible"], "hits": row["hits"], "hr_percent": 100.0 * float(row["hr"])}
            if int(row["eligible"]) != hr_ours["eligible"]:
                problems.append(f"eligible {hr_ours['eligible']} vs expected {row['eligible']}")
            if not dense and int(row["hits"]) != int(ours_hits):
                problems.append(f"HR@{k} hits {ours_hits} vs expected {row['hits']}")
            if dense and abs(100.0 * float(row["hr"]) - hr_ours["hr_percent"][str(k)]) > DENSE_HR_TOLERANCE_POINTS + 1e-9:
                drift.append(f"HR@{k} {hr_ours['hr_percent'][str(k)]:.2f} vs expected {100.0 * float(row['hr']):.2f}")
        out["expected_metrics"] = exp_out

    # Text channel alone vs the paper's Vanilla (reported to one decimal in Table 1).
    vanilla_ref = published_vanilla(repo_root)
    text_notes: list[str] = []
    vt: dict[str, Any] = {"reported_percent": {}, "diff_points": {}}
    for k in (5, 10):
        ref = vanilla_ref.get((VANILLA_DATASET_NAME[dataset], retriever, k))
        vt["reported_percent"][str(k)] = ref
        ours = hr_vanilla["hr_percent"][str(k)]
        if ref is not None and ours is not None:
            vt["diff_points"][str(k)] = ours - ref
            tol = 0.05 if not dense else 0.05 + DENSE_HR_TOLERANCE_POINTS
            if full_dataset and abs(ours - ref) > tol + 1e-9:
                text_notes.append(f"text-only HR@{k} {ours:.2f} vs Table 1 Vanilla {ref:.1f} (tolerance {tol:.2f})")
    pq = per_query_vanilla_file(dataset, retriever, repo_root)
    if pq is not None:
        with pq.open("r", encoding="utf-8") as handle:
            saved_text_rows = json.load(handle)
        by_key = {
            (str(r.get("doc_id")), str(r.get("query_id"))): [str(x) for x in r.get("retrieved_chunk_ids", [])][:RETRIEVE_K]
            for r in saved_text_rows
        }
        same = sum(
            1
            for row, top in zip(used_rows, vanilla_top, strict=True)
            if by_key.get((str(row.get("doc_id")), str(row.get("query_id")))) == top
        )
        vt["per_query_file"] = rel(pq, repo_root)
        vt["per_query_identical_top10"] = same
        if not dense and same != n:
            text_notes.append(f"text-only top-10 differs from {pq.name} for {n - same} of {n} queries")
    vt["passed"] = not text_notes
    vt["notes"] = text_notes
    out["text_only"].update(vt)

    max_hr_gap = max(
        (
            abs(hr_ours["hr_percent"][k] - hr_saved["hr_percent"][k])
            for k in ("5", "10")
            if hr_ours["hr_percent"][k] is not None and hr_saved["hr_percent"][k] is not None
        ),
        default=float("inf"),
    )
    out["max_hr_gap_points"] = max_hr_gap
    out["drift_accepted"] = False
    if drift:
        if retriever in DRIFT_ACCEPTED_RETRIEVERS and max_hr_gap <= DRIFT_MAX_HR_POINTS + 1e-9:
            out["drift_accepted"] = True
            out["drift_notes"] = drift + [
                f"accepted: {retriever} is in DRIFT_ACCEPTED_RETRIEVERS and the largest HR gap is {max_hr_gap:.2f} <= {DRIFT_MAX_HR_POINTS} points"
            ]
        else:
            problems.extend(drift)
            if retriever in DRIFT_ACCEPTED_RETRIEVERS:
                problems.append(f"largest HR gap {max_hr_gap:.2f} points exceeds the drift limit {DRIFT_MAX_HR_POINTS}")
    out["problems"] = problems
    out["passed"] = not problems
    return out


def save_streams(
    dataset: str, retriever: str, docs: list[dict[str, Any]], used_rows: list[dict[str, Any]], *, out_path: Path, queries_path: Path
) -> int:
    """One parquet row per (query, chunk); evidence labels as the saved results define them."""

    import pandas as pd

    cols: dict[str, list[Any]] = {
        k: []
        for k in ("qidx", "query_id", "doc_id", "chunk_id", "chunk_index", "chunk_pos", *CHANNELS, "is_gold", "is_evidence", "group_ids")
    }
    qmeta = []
    qidx = 0
    rows_iter = iter(used_rows)
    for doc in docs:
        chunk_ids = doc["chunk_ids"]
        for q in doc["queries"]:
            row = next(rows_iter)
            gold, silver, groups = evidence_of(row)
            gold_set = set(gold)
            member = {}
            for gi, g in enumerate(groups):
                for c in g:
                    member.setdefault(c, []).append(gi)
            evidence = gold_set | set(member)
            for pos, cid in enumerate(chunk_ids):
                cols["qidx"].append(qidx)
                cols["query_id"].append(str(row.get("query_id", "")))
                cols["doc_id"].append(doc["doc_id"])
                cols["chunk_id"].append(cid)
                cols["chunk_index"].append(int(doc["chunk_index"][pos]))
                cols["chunk_pos"].append(pos)
                for j, ch in enumerate(CHANNELS):
                    cols[ch].append(float(q["raw"][pos, j]))
                cols["is_gold"].append(cid in gold_set)
                cols["is_evidence"].append(cid in evidence)
                cols["group_ids"].append(",".join(str(g) for g in member.get(cid, [])))
            qmeta.append(
                {
                    "qidx": qidx,
                    "row_index": q["row_index"],
                    "query_id": str(row.get("query_id", "")),
                    "doc_id": doc["doc_id"],
                    "question": str(row.get("question", "")),
                    "gold_chunk_ids": gold,
                    "silver_chunk_ids": silver,
                    "silver_chunk_groups": groups,
                    "eligible": is_eligible(gold, groups),
                    "saved_top10": [str(x) for x in row.get("retrieved_chunk_ids", [])][:RETRIEVE_K],
                    "published_top10": q["published_top"],
                    "vanilla_top10": q["vanilla_top"],
                }
            )
            qidx += 1
    frame = pd.DataFrame(cols)
    for col in ("qidx", "chunk_index", "chunk_pos"):
        frame[col] = frame[col].astype("int32")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, out_path)
    write_json(queries_path, {"dataset": dataset, "retriever": retriever, "channels": list(CHANNELS), "queries": qmeta})
    return len(frame)


def run_stream_job(dataset: str, retriever: str, *, paths: C2Paths, max_docs: int | None = None, log=print) -> dict[str, Any]:
    """Streams -> reproduction gate -> save. Writes DONE or FAILED; returns the job summary."""

    job = job_name(dataset, retriever)
    if paths.done_marker(job).exists():
        return json.loads(paths.done_marker(job).read_text(encoding="utf-8")) | {"skipped": True}
    paths.ensure()
    t0 = time.perf_counter()
    docs, used_rows, timings = compute_job_streams(dataset, retriever, paths=paths, max_docs=max_docs, log=log)
    check = reproduction_check(dataset, retriever, docs, used_rows, repo_root=paths.repo_root, full_dataset=max_docs is None)
    summary: dict[str, Any] = {
        "job": job,
        "dataset": dataset,
        "retriever": retriever,
        "max_docs": max_docs,
        "timings": timings,
        "check": check,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if not check["passed"]:
        # Quarantined copy for diagnosis only; the analysis never reads streams_failed/.
        rows = save_streams(
            dataset,
            retriever,
            docs,
            used_rows,
            out_path=paths.failed_dir / f"{job}.parquet",
            queries_path=paths.failed_dir / f"{job}.queries.json",
        )
        summary["quarantined_rows"] = rows
        summary["status"] = "FAILED"
        summary["seconds"] = time.perf_counter() - t0
        write_json(paths.failed_marker(job), summary)
        return summary
    rows = save_streams(dataset, retriever, docs, used_rows, out_path=paths.stream_path(job), queries_path=paths.queries_path(job))
    summary["stream_rows"] = rows
    summary["files"] = [rel(p, paths.repo_root) for p in (paths.stream_path(job), paths.queries_path(job), paths.check_path(job))]
    summary["status"] = "DONE"
    summary["seconds"] = time.perf_counter() - t0
    write_json(paths.check_path(job), check)
    if paths.failed_marker(job).exists():
        paths.failed_marker(job).unlink()
    write_json(paths.done_marker(job), summary)
    return summary


# ---------------------------------------------------------------------------
# Step 2: grid, per-query arrays, vectorized combination
# ---------------------------------------------------------------------------


@dataclass
class Grid:
    points: list[tuple[float, float, float, float]]  # (lambda_M, lambda_S, mu_A, mu_D); lambda_T = mu_N = 1

    def __post_init__(self) -> None:
        arr = np.asarray(self.points, dtype=np.float64)
        self.lam_m, self.lam_s, self.mu_a, self.mu_d = arr[:, 0].copy(), arr[:, 1].copy(), arr[:, 2].copy(), arr[:, 3].copy()
        self.published_index = self.index_of(PUBLISHED_POINT)
        self.l1 = np.abs(arr - np.asarray(PUBLISHED_POINT)[None, :]).sum(axis=1)
        self.channel_only = np.flatnonzero((self.mu_a == 1.0) & (self.mu_d == 1.0))

    def __len__(self) -> int:
        return len(self.points)

    def index_of(self, point: tuple[float, float, float, float]) -> int:
        for i, p in enumerate(self.points):
            if all(abs(a - b) < 1e-12 for a, b in zip(p, point, strict=True)):
                return i
        raise KeyError(f"grid point {point} not in grid")

    def label(self, i: int) -> str:
        lm, ls, ma, md = self.points[int(i)]
        return f"lM={lm:g} lS={ls:g} mA={ma:g} mD={md:g}"


def make_grid(*, dry_run: bool = False) -> Grid:
    lams = DRY_LAMBDA_GRID if dry_run else LAMBDA_GRID
    mus = DRY_MU_GRID if dry_run else MU_GRID
    return Grid([(lm, ls, ma, md) for lm in lams for ls in lams for ma in mus for md in mus])


BIG_RANK = 1_000_000


@dataclass
class QueryArrays:
    qidx: int
    query_id: str
    doc_id: str
    chunk_ids: list[str]
    chunk_index: np.ndarray
    raw: np.ndarray  # (C, 5) raw channel scores, NaN = missing
    z: np.ndarray  # (C, 5) z-scored per channel over the chunks that have the signal
    r0: np.ndarray  # static tie-break rank: (-raw text, chunk_index, chunk_id)
    gold_pos: np.ndarray
    group_pos: list[np.ndarray]
    eligible: bool
    vanilla_top10: list[str] = field(default_factory=list)
    published_top10: list[str] = field(default_factory=list)


def _zscore_column(chunk_ids: list[str], values: np.ndarray, retriever: str) -> np.ndarray:
    # Same dict, same insertion order (chunk pool order) as combine_mbr_variant_chunk_scores.
    present = {cid: float(v) for cid, v in zip(chunk_ids, values, strict=True) if not math.isnan(v)}
    normed = normalize_signal_scores(present, method="zscore", retriever_name=retriever)
    out = np.full(len(chunk_ids), np.nan)
    for i, cid in enumerate(chunk_ids):
        if cid in normed:
            out[i] = normed[cid]
    return out


def load_query_arrays(stream_path: Path, queries_path: Path, retriever: str) -> list[QueryArrays]:
    import pandas as pd

    frame = pd.read_parquet(stream_path)
    meta = json.loads(queries_path.read_text(encoding="utf-8"))["queries"]
    qidx = frame["qidx"].to_numpy()
    bounds = np.flatnonzero(np.diff(qidx)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(frame)]])
    raw_all = frame[list(CHANNELS)].to_numpy(dtype=np.float64)
    chunk_id_all = frame["chunk_id"].astype(str).to_numpy()
    chunk_index_all = frame["chunk_index"].to_numpy(dtype=np.int64)
    out: list[QueryArrays] = []
    for s, e, m in zip(starts, ends, meta, strict=True):
        if int(qidx[s]) != int(m["qidx"]):
            raise RuntimeError(f"{stream_path.name}: parquet and queries.json out of order at qidx {m['qidx']}")
        chunk_ids = [str(c) for c in chunk_id_all[s:e]]
        raw = raw_all[s:e]
        z = np.column_stack([_zscore_column(chunk_ids, raw[:, j], retriever) for j in range(len(CHANNELS))])
        cidx = chunk_index_all[s:e]
        order = sorted(range(len(chunk_ids)), key=lambda i: (-float(raw[i, 0]), int(cidx[i]), chunk_ids[i]))
        r0 = np.empty(len(chunk_ids), dtype=np.int64)
        r0[np.asarray(order, dtype=np.int64)] = np.arange(len(chunk_ids))
        pos = {c: i for i, c in enumerate(chunk_ids)}
        gold_pos = np.asarray(sorted({pos[c] for c in m["gold_chunk_ids"] if c in pos}), dtype=np.int64)
        groups = []
        for g in m["silver_chunk_groups"]:
            if g and all(c in pos for c in g):  # a group with a chunk outside the pool can never be complete
                groups.append(np.asarray(sorted({pos[c] for c in g}), dtype=np.int64))
        out.append(
            QueryArrays(
                qidx=int(m["qidx"]),
                query_id=str(m["query_id"]),
                doc_id=str(m["doc_id"]),
                chunk_ids=chunk_ids,
                chunk_index=cidx,
                raw=raw,
                z=z,
                r0=r0,
                gold_pos=gold_pos,
                group_pos=groups,
                eligible=bool(m["eligible"]),
                vanilla_top10=list(m["vanilla_top10"]),
                published_top10=list(m["published_top10"]),
            )
        )
    return out


def final_scores(q: QueryArrays, lam_m: np.ndarray, lam_s: np.ndarray, mu_a: np.ndarray, mu_d: np.ndarray) -> np.ndarray:
    """(G, C) final scores; the operations and their order follow combine_mbr_variant_chunk_scores exactly.

    Memory: SM = sum_f (mu_f / sum_present mu) * z_f over present fields with mu_f > 0
    (name, alias, description in that order). Final: sum_c (lambda_c / sum_present lambda) * s_c over
    the present channels (text, memory, summary in that order). Adding 0.0 for an absent term is exact.
    """

    z = q.z
    h = ~np.isnan(z)
    z0 = np.where(h, z, 0.0)
    lam_m = lam_m[:, None]
    lam_s = lam_s[:, None]
    mu_a = mu_a[:, None]
    mu_d = mu_d[:, None]
    inc_n = np.broadcast_to(h[:, 1][None, :], (lam_m.shape[0], z.shape[0]))
    inc_a = h[:, 2][None, :] & (mu_a > 0)
    inc_d = h[:, 3][None, :] & (mu_d > 0)
    has_m = inc_n | inc_a | inc_d
    msum = 0.0 + np.where(inc_n, 1.0, 0.0)
    msum = msum + np.where(inc_a, mu_a, 0.0)
    msum = msum + np.where(inc_d, mu_d, 0.0)
    safe = np.where(has_m, msum, 1.0)
    mem = 0.0 + np.where(inc_n, (1.0 / safe) * z0[:, 1][None, :], 0.0)
    mem = mem + np.where(inc_a, (mu_a / safe) * z0[:, 2][None, :], 0.0)
    mem = mem + np.where(inc_d, (mu_d / safe) * z0[:, 3][None, :], 0.0)
    has_s = h[:, 4][None, :]
    ws = (0.0 + 1.0) + np.where(has_m, lam_m, 0.0)
    ws = ws + np.where(has_s, lam_s, 0.0)
    final = 0.0 + (1.0 / ws) * z0[:, 0][None, :]
    final = final + np.where(has_m, (lam_m / ws) * mem, 0.0)
    final = final + np.where(has_s, (lam_s / ws) * z0[:, 4][None, :], 0.0)
    return final


def ranking_order(q: QueryArrays, final: np.ndarray) -> np.ndarray:
    """(G, C) chunk positions in ranked order; key (-final, -raw text, chunk_index, chunk_id) as in the repo."""

    r0 = np.broadcast_to(q.r0[None, :], final.shape)
    return np.lexsort((r0, -final), axis=-1)


def evidence_positions(q: QueryArrays) -> np.ndarray:
    parts = [q.gold_pos, *q.group_pos]
    return np.unique(np.concatenate(parts)) if parts and sum(len(p) for p in parts) else np.zeros(0, dtype=np.int64)


def hit_ranks(q: QueryArrays, final: np.ndarray, *, block: int = 256) -> np.ndarray:
    """(G,) 0-based rank at which the saved hit definition is first met.

    Hit@k (retrieval_metrics): a gold chunk in the top k, or every chunk of one evidence group in
    the top k. So the hit rank is min(min over gold of rank, min over groups of max member rank).
    BIG_RANK when no gold chunk or complete group is in the pool.
    """

    ev = evidence_positions(q)
    G = final.shape[0]
    if ev.size == 0:
        return np.full(G, BIG_RANK, dtype=np.int64)
    col = {int(p): i for i, p in enumerate(ev)}
    r0e = q.r0[ev]
    out = np.empty(G, dtype=np.int64)
    for b in range(0, G, block):
        f = final[b : b + block]
        fe = f[:, ev]
        gt = (f[:, :, None] > fe[:, None, :]).sum(axis=1)
        eq = ((f[:, :, None] == fe[:, None, :]) & (q.r0[None, :, None] < r0e[None, None, :])).sum(axis=1)
        pos = gt + eq  # (g, E)
        best = np.full(pos.shape[0], BIG_RANK, dtype=np.int64)
        if q.gold_pos.size:
            best = np.minimum(best, pos[:, [col[int(p)] for p in q.gold_pos]].min(axis=1))
        for grp in q.group_pos:
            best = np.minimum(best, pos[:, [col[int(p)] for p in grp]].max(axis=1))
        out[b : b + block] = best
    return out


def _rebuild_for_repo(q: QueryArrays, retriever: str) -> tuple[MBRDocumentMemoryIndex, MBRChunkScoreStreams]:
    """Minimal objects for combine_mbr_variant_chunk_scores from the saved raw streams."""

    pool = [MBRChunkPoolRecord(chunk_id=cid, text="", step=int(ci)) for cid, ci in zip(q.chunk_ids, q.chunk_index, strict=True)]
    scores: dict[str, MBRChunkScoreRecord] = {}
    for i, cid in enumerate(q.chunk_ids):
        vals = [None if math.isnan(v) else float(v) for v in q.raw[i]]
        mem = [v for v in vals[1:4] if v is not None]
        scores[cid] = MBRChunkScoreRecord(
            text_score=float(vals[0]),
            name_score=vals[1],
            alias_score=vals[2],
            description_score=vals[3],
            # Stream-time mu = (1, 1, 1): the raw memory score exists iff any field exists; its value is not
            # used by the MBR2 combination (only whether it is None).
            memory_score=(sum(mem) / len(mem)) if mem else None,
            summary_score=vals[4],
        )
    index = MBRDocumentMemoryIndex(doc_id=q.doc_id, chunk_pool=pool)
    streams = MBRChunkScoreStreams(
        doc_id=q.doc_id,
        query="",
        retriever_name=retriever,
        text_scores={cid: float(q.raw[i, 0]) for i, cid in enumerate(q.chunk_ids)},
        chunk_scores=scores,
    )
    return index, streams


def repo_ranking(q: QueryArrays, retriever: str, point: tuple[float, float, float, float], *, normalization: str = "zscore") -> list[str]:
    index, streams = _rebuild_for_repo(q, retriever)
    lam_m, lam_s, mu_a, mu_d = point
    result = combine_mbr_variant_chunk_scores(
        index,
        streams,
        variant=MBR_VARIANT_MBR2,
        lambda_t=1.0,
        lambda_n=lam_m,
        lambda_h=lam_s,
        normalization_method=normalization,
        memory_name_weight=1.0,
        memory_alias_weight=mu_a,
        memory_description_weight=mu_d,
    )
    return [r.chunk_id for r in result.ranked_chunks]


def hit_rank_of_ranking(q: QueryArrays, ranking: list[str]) -> int:
    pos = {c: i for i, c in enumerate(ranking)}
    best = BIG_RANK
    for p in q.gold_pos:
        best = min(best, pos[q.chunk_ids[int(p)]])
    for grp in q.group_pos:
        best = min(best, max(pos[q.chunk_ids[int(p)]] for p in grp))
    return best


def verify_points(grid: Grid) -> list[int]:
    """The published point plus 20 random grid points (seed 42)."""

    rng = np.random.default_rng(SEED)
    others = [i for i in range(len(grid)) if i != grid.published_index]
    picks = rng.choice(len(others), size=min(N_VERIFY_POINTS, len(others)), replace=False)
    return [grid.published_index, *sorted(int(others[i]) for i in picks)]


# ---------------------------------------------------------------------------
# Folds and the learned fusion (part G)
# ---------------------------------------------------------------------------


def grouped_folds(doc_ids: list[str], *, n_folds: int = N_FOLDS, seed: int = SEED) -> tuple[np.ndarray, int]:
    """Folds over queries grouped by document: groups shuffled (seed 42), then each group goes to the
    fold with the fewest queries so far (ties: lowest fold). Returns (fold per query, number of folds)."""

    groups = sorted(set(doc_ids))
    k = min(int(n_folds), len(groups))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    sizes = {g: 0 for g in groups}
    for d in doc_ids:
        sizes[d] += 1
    load = np.zeros(k, dtype=np.int64)
    fold_of: dict[str, int] = {}
    for gi in order:
        g = groups[int(gi)]
        f = int(np.argmin(load))
        fold_of[g] = f
        load[f] += sizes[g]
    return np.asarray([fold_of[d] for d in doc_ids], dtype=np.int64), k


LEARNED_FEATURES = (
    "z_text",
    "z_name",
    "z_alias",
    "z_description",
    "z_summary",
    "miss_name",
    "miss_alias",
    "miss_description",
    "miss_summary",
)


def learned_features(q: QueryArrays) -> np.ndarray:
    h = ~np.isnan(q.z)
    z0 = np.where(h, q.z, 0.0)
    return np.column_stack([z0, (~h[:, 1:]).astype(np.float64)])


def _pairwise_diffs(queries: list[QueryArrays], feats: list[np.ndarray]) -> np.ndarray:
    blocks = []
    for q, x in zip(queries, feats, strict=True):
        ev = evidence_positions(q)
        if ev.size == 0:
            continue
        neg = np.setdiff1d(np.arange(len(q.chunk_ids)), ev)
        if neg.size == 0:
            continue
        blocks.append((x[ev][:, None, :] - x[neg][None, :, :]).reshape(-1, x.shape[1]))
    return np.concatenate(blocks) if blocks else np.zeros((0, len(LEARNED_FEATURES)))


def fit_pairwise_logistic(d: np.ndarray, *, c: float = 1.0, max_iter: int = 100, tol: float = 1e-10) -> tuple[np.ndarray, dict[str, Any]]:
    """L2 logistic regression without intercept on pair differences (gold minus non-gold).

    Objective (as sklearn's LogisticRegression with C, fit_intercept=False):
        C * sum_i log(1 + exp(-w . d_i)) + 0.5 * ||w||^2.
    Without an intercept the loss depends only on y_i * x_i, so randomly flipping pair orientation
    (the usual way to get two classes) does not change the fit; the seed has no effect here.
    Solved by Newton's method with backtracking; strictly convex, so the optimum is unique.
    """

    w = np.zeros(d.shape[1])

    def objective(v: np.ndarray) -> float:
        return float(c * np.logaddexp(0.0, -(d @ v)).sum() + 0.5 * v @ v)

    obj = objective(w)
    info = {"n_pairs": int(d.shape[0]), "iterations": 0, "converged": False}
    for it in range(1, max_iter + 1):
        m = d @ w
        p = 0.5 * (1.0 + np.tanh(-0.5 * m))  # sigmoid(-m), stable
        grad = -c * (d.T @ p) + w
        hess = c * (d.T * (p * (1.0 - p))) @ d + np.eye(d.shape[1])
        step = np.linalg.solve(hess, grad)
        t = 1.0
        while True:
            cand = w - t * step
            cand_obj = objective(cand)
            if cand_obj <= obj - 1e-4 * t * float(grad @ step) or t < 1e-8:
                break
            t *= 0.5
        w, obj = cand, cand_obj
        info["iterations"] = it
        if float(np.abs(t * step).max()) < tol:
            info["converged"] = True
            break
    info["objective"] = obj
    return w, info


# ---------------------------------------------------------------------------
# Per-job analysis worker (CPU)
# ---------------------------------------------------------------------------

TABLE9_SETTINGS = ((0.5, 0.3), (1.0, 1.0), (0.3, 1.0), (0.5, 1.0), (1.0, 0.5))  # (lambda_M, lambda_S), Table 9 order
TABLE9_JOBS = tuple((d, r) for d in ("loogle", "novelhopqa") for r in ("bm25", "contriever", "jina"))


def analyze_job(dataset: str, retriever: str, *, paths_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Verification, grid hit ranks, folds, learned fusion, Table 9 hit ranks for one job. Saves an npz."""

    t0 = time.perf_counter()
    paths = C2Paths.from_kwargs(paths_kwargs)
    grid = make_grid(dry_run=paths.dry_run)
    job = job_name(dataset, retriever)
    queries = load_query_arrays(paths.stream_path(job), paths.queries_path(job), retriever)
    vpoints = verify_points(grid)
    lam_m, lam_s, mu_a, mu_d = grid.lam_m, grid.lam_s, grid.mu_a, grid.mu_d
    vanilla_index = grid.index_of((0.0, 0.0, 1.0, 1.0))

    n_q = len(queries)
    hit = np.empty((n_q, len(grid)), dtype=np.int32)
    examples: list[dict[str, Any]] = []
    n_rank_diff = 0
    n_hit_diff = 0
    vanilla_fail = 0
    published_fail = 0
    t_grid = 0.0
    t_verify = 0.0
    for qi, q in enumerate(queries):
        ta = time.perf_counter()
        final = final_scores(q, lam_m, lam_s, mu_a, mu_d)
        hit[qi] = np.minimum(hit_ranks(q, final), np.iinfo(np.int32).max)
        tb = time.perf_counter()
        t_grid += tb - ta
        orders = ranking_order(q, final[vpoints])
        for j, gi in enumerate(vpoints):
            ours = [q.chunk_ids[int(p)] for p in orders[j]]
            repo = repo_ranking(q, retriever, grid.points[gi])
            if ours != repo:
                n_rank_diff += 1
                if len(examples) < 5:
                    examples.append(
                        {"query_id": q.query_id, "grid_point": grid.label(gi), "ours_top10": ours[:10], "repo_top10": repo[:10]}
                    )
            if min(hit_rank_of_ranking(q, repo), BIG_RANK) != int(hit[qi, gi]):
                n_hit_diff += 1
            if gi == grid.published_index and repo[:RETRIEVE_K] != q.published_top10:
                published_fail += 1
        van_order = ranking_order(q, final[[vanilla_index]])[0]
        if [q.chunk_ids[int(p)] for p in van_order[:RETRIEVE_K]] != q.vanilla_top10:
            vanilla_fail += 1
        t_verify += time.perf_counter() - tb

    doc_ids = [q.doc_id for q in queries]
    folds, k = grouped_folds(doc_ids)
    eligible = np.asarray([q.eligible for q in queries])

    # G: learned fusion in the same outer folds.
    feats = [learned_features(q) for q in queries]
    learned_rank = np.full(n_q, BIG_RANK, dtype=np.int64)
    weights = []
    for f in range(k):
        train = [i for i in range(n_q) if folds[i] != f and eligible[i]]
        d = _pairwise_diffs([queries[i] for i in train], [feats[i] for i in train])
        w, info = fit_pairwise_logistic(d)
        weights.append({"fold": f, "w": w.tolist(), **info})
        for i in range(n_q):
            if folds[i] == f:
                s = (feats[i] @ w)[None, :]
                learned_rank[i] = hit_ranks(queries[i], s)[0]

    # C: Table 9 settings without normalization (the z-score ones come from the grid).
    table9_none = None
    if (dataset, retriever) in TABLE9_JOBS:
        table9_none = np.empty((n_q, len(TABLE9_SETTINGS)), dtype=np.int64)
        for qi, q in enumerate(queries):
            for j, (lm, ls) in enumerate(TABLE9_SETTINGS):
                table9_none[qi, j] = hit_rank_of_ranking(q, repo_ranking(q, retriever, (lm, ls, 1.0, 1.0), normalization="none"))

    out_npz = paths.grid_dir / f"{job}.npz"
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        hit_rank=hit,
        eligible=eligible,
        doc_id=np.asarray(doc_ids),
        query_id=np.asarray([q.query_id for q in queries]),
        fold=folds,
        learned_rank=learned_rank,
        table9_none=table9_none if table9_none is not None else np.zeros((0, 0)),
        n_chunks=np.asarray([len(q.chunk_ids) for q in queries]),
    )
    return {
        "job": job,
        "dataset": dataset,
        "retriever": retriever,
        "n_queries": n_q,
        "n_folds": k,
        "verify_points": [grid.label(i) for i in vpoints],
        "verify_rankings_compared": n_q * len(vpoints),
        "verify_rankings_different": n_rank_diff,
        "verify_hit_rank_different": n_hit_diff,
        "verify_examples": examples,
        "published_top10_vs_streams_mismatch": published_fail,
        "vanilla_top10_mismatch": vanilla_fail,
        "learned_weights": weights,
        "seconds": time.perf_counter() - t0,
        "seconds_grid": t_grid,
        "seconds_verify": t_verify,
        "npz": str(out_npz),
    }


# ---------------------------------------------------------------------------
# Step 3: selection, statistics, report (CPU, after all jobs)
# ---------------------------------------------------------------------------


@dataclass
class JobGrid:
    dataset: str
    retriever: str
    hit_rank: np.ndarray  # (Q, G) eligible queries only
    doc_id: np.ndarray
    query_id: np.ndarray
    fold: np.ndarray
    learned_rank: np.ndarray
    table9_none: np.ndarray
    n_folds: int

    @property
    def h5(self) -> np.ndarray:
        return self.hit_rank < 5

    @property
    def h10(self) -> np.ndarray:
        return self.hit_rank < 10

    @property
    def rr(self) -> np.ndarray:
        return np.where(self.hit_rank < BIG_RANK, 1.0 / (self.hit_rank.astype(np.float64) + 1.0), 0.0)


def load_job_grid(path: Path, dataset: str, retriever: str, n_folds: int) -> JobGrid:
    data = np.load(path, allow_pickle=False)
    e = data["eligible"].astype(bool)
    t9 = data["table9_none"]
    return JobGrid(
        dataset=dataset,
        retriever=retriever,
        hit_rank=data["hit_rank"][e].astype(np.int64),
        doc_id=data["doc_id"][e],
        query_id=data["query_id"][e],
        fold=data["fold"][e],
        learned_rank=data["learned_rank"][e],
        table9_none=t9[e] if t9.size else t9,
        n_folds=n_folds,
    )


def choose_point(h10: np.ndarray, h5: np.ndarray, rr: np.ndarray, grid: Grid, candidates: np.ndarray) -> int:
    """Highest mean HR@10; ties: HR@5, then MRR, then smallest L1 distance to the published point,
    then grid order. h10/h5/rr are per-candidate scores (means or sums over the same queries)."""

    keys = (candidates, grid.l1[candidates], -np.round(rr, 12), -np.round(h5, 12), -np.round(h10, 12))
    return int(candidates[np.lexsort(keys)[0]])


def cross_validated(jg: JobGrid, grid: Grid, candidates: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Per-query held-out hit rank and the point chosen in each outer fold."""

    held = np.empty(len(jg.fold), dtype=np.int64)
    chosen: list[int] = []
    h10, h5, rr = jg.h10, jg.h5, jg.rr
    for f in range(jg.n_folds):
        tr = jg.fold != f
        te = ~tr
        gi = choose_point(h10[tr][:, candidates].mean(0), h5[tr][:, candidates].mean(0), rr[tr][:, candidates].mean(0), grid, candidates)
        chosen.append(gi)
        held[te] = jg.hit_rank[te, gi]
    return held, chosen


def cluster_bootstrap_ci(
    diffs: np.ndarray, clusters: np.ndarray, *, n_resamples: int = N_BOOT, seed: int = SEED
) -> tuple[float, float, float]:
    """Mean paired difference with a 95% CI from resampling documents (with replacement)."""

    keys, inv = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inv, weights=diffs, minlength=len(keys))
    counts = np.bincount(inv, minlength=len(keys)).astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(int(n_resamples), len(keys)))
    means = sums[idx].sum(1) / counts[idx].sum(1)
    return float(diffs.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _pct(x: np.ndarray) -> float:
    return 100.0 * float(np.mean(x)) if len(x) else float("nan")


def run_analysis(paths: C2Paths, *, workers: int = 4, log=print) -> dict[str, Any]:
    """Parts A-G over all finished jobs; writes tables, figures, REPORT.md, REPORT_FOR_GDOC.md."""

    from concurrent.futures import ProcessPoolExecutor

    t_start = time.perf_counter()
    grid = make_grid(dry_run=paths.dry_run)
    jobs = [(d, r) for d, r in ALL_JOBS if paths.done_marker(job_name(d, r)).exists()]
    if paths.dry_run:
        jobs = [(d, r) for d, r in jobs if r == "bm25"]
    log(f"analysis: {len(jobs)} jobs, grid {len(grid)} points, {workers} workers")
    worker_out: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futs = {pool.submit(analyze_job, d, r, paths_kwargs=paths.as_kwargs()): (d, r) for d, r in jobs}
        for fut in futs:
            d, r = futs[fut]
            res = fut.result()
            worker_out[job_name(d, r)] = res
            log(
                f"  {job_name(d, r)}: {res['n_queries']} queries, {res['seconds']:.0f}s, ranking differences vs repo {res['verify_rankings_different']}/{res['verify_rankings_compared']}"
            )

    grids: dict[tuple[str, str], JobGrid] = {}
    for d, r in jobs:
        w = worker_out[job_name(d, r)]
        grids[(d, r)] = load_job_grid(Path(w["npz"]), d, r, w["n_folds"])

    pub = grid.published_index
    van = grid.index_of((0.0, 0.0, 1.0, 1.0))
    all_points = np.arange(len(grid))
    chan_points = grid.channel_only

    per_job: dict[tuple[str, str], dict[str, Any]] = {}
    for key, jg in grids.items():
        a1, chosen1 = cross_validated(jg, grid, all_points)
        a2, chosen2 = cross_validated(jg, grid, chan_points)
        per_job[key] = {
            "a1": a1,
            "a2": a2,
            "chosen_a1": chosen1,
            "chosen_a2": chosen2,
            "vanilla": jg.hit_rank[:, van],
            "published": jg.hit_rank[:, pub],
            "learned": jg.learned_rank,
        }

    # B: leave-one-dataset-out transfer.
    datasets_present = sorted({d for d, _ in grids}, key=DATASET_KEYS.index)
    retrievers_present = sorted({r for _, r in grids}, key=RETRIEVERS.index)
    lodo: dict[tuple[str, str], int] = {}
    lodo_global: dict[str, int] = {}

    def _means(keys: list[tuple[str, str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h10 = np.mean([grids[k].h10.mean(0) for k in keys], axis=0)
        h5 = np.mean([grids[k].h5.mean(0) for k in keys], axis=0)
        rr = np.mean([grids[k].rr.mean(0) for k in keys], axis=0)
        return h10, h5, rr

    for d in datasets_present:
        others = [x for x in datasets_present if x != d]
        if not others:
            continue
        gkeys = [(o, r) for o in others for r in retrievers_present if (o, r) in grids]
        h10, h5, rr = _means(gkeys)
        lodo_global[d] = choose_point(h10, h5, rr, grid, all_points)
        for r in retrievers_present:
            keys = [(o, r) for o in others if (o, r) in grids]
            if (d, r) in grids and keys:
                h10, h5, rr = _means(keys)
                lodo[(d, r)] = choose_point(h10, h5, rr, grid, all_points)
    for key, jg in grids.items():
        d, r = key
        per_job[key]["b"] = jg.hit_rank[:, lodo[key]] if key in lodo else None
        per_job[key]["b_global"] = jg.hit_rank[:, lodo_global[d]] if d in lodo_global else None

    # D: statistics.
    rows_d: list[dict[str, Any]] = []
    for key in sorted(grids, key=lambda k: (DATASET_KEYS.index(k[0]), RETRIEVERS.index(k[1]))):
        jg, pj = grids[key], per_job[key]
        row: dict[str, Any] = {
            "dataset": key[0],
            "retriever": key[1],
            "n_queries": len(jg.fold),
            "n_docs": len(set(jg.doc_id.tolist())),
            "n_folds": jg.n_folds,
        }
        systems = {
            "vanilla": pj["vanilla"],
            "published": pj["published"],
            "a1": pj["a1"],
            "a2": pj["a2"],
            "b": pj["b"],
            "b_global": pj["b_global"],
            "learned": pj["learned"],
        }
        for name, ranks in systems.items():
            for k in (5, 10):
                row[f"{name}_hr{k}"] = _pct(ranks < k) if ranks is not None else None
        for name in ("a1", "a2", "b", "b_global", "learned"):
            if systems[name] is None:
                continue
            for base in ("vanilla", "published"):
                for k in (5, 10):
                    diff = 100.0 * ((systems[name] < k).astype(np.float64) - (systems[base] < k).astype(np.float64))
                    m, lo, hi = paired_bootstrap_ci(diff, n_resamples=N_BOOT, seed=SEED)
                    _, clo, chi = cluster_bootstrap_ci(diff, jg.doc_id)
                    row[f"{name}_minus_{base}_hr{k}"] = m
                    row[f"{name}_minus_{base}_hr{k}_ci"] = [lo, hi]
                    row[f"{name}_minus_{base}_hr{k}_cluster_ci"] = [clo, chi]
        row["chosen_a1"] = [grid.points[i] for i in pj["chosen_a1"]]
        row["chosen_a2"] = [grid.points[i] for i in pj["chosen_a2"]]
        row["chosen_b"] = grid.points[lodo[key]] if key in lodo else None
        row["chosen_b_global"] = grid.points[lodo_global[key[0]]] if key[0] in lodo_global else None
        # F: sensitivity over the channel grid with field weights (1, 1, 1).
        h10c = 100.0 * jg.h10[:, chan_points].mean(0)
        best = float(h10c.max())
        row["f_hr10_published"] = 100.0 * float(jg.h10[:, pub].mean())
        row["f_hr10_best"] = best
        row["f_hr10_best_point"] = grid.points[int(chan_points[int(np.argmax(h10c))])]
        row["f_hr10_worst"] = float(h10c.min())
        row["f_share_within_1pt"] = float(np.mean(h10c >= best - 1.0 - 1e-9))
        row["f_hr10_full_grid_best"] = 100.0 * float(jg.h10.mean(0).max())
        ch = pj["chosen_a1"]
        row["stab_a1_distinct"] = len(set(ch))
        row["stab_a1_published_folds"] = sum(1 for i in ch if i == pub)
        row["stab_a1_mean_l1"] = float(np.mean([grid.l1[i] for i in ch]))
        rows_d.append(row)

    summary_counts = {}
    for name in ("a1", "a2", "b", "b_global", "learned"):
        for k in (5, 10):
            rs = [r for r in rows_d if r.get(f"{name}_minus_vanilla_hr{k}") is not None]
            summary_counts[f"{name}_hr{k}"] = {
                "pairs": len(rs),
                "beats_vanilla": sum(1 for r in rs if r[f"{name}_minus_vanilla_hr{k}"] > 0),
                "ci_above_zero": sum(1 for r in rs if r[f"{name}_minus_vanilla_hr{k}_ci"][0] > 0),
                "ci_below_zero": sum(1 for r in rs if r[f"{name}_minus_vanilla_hr{k}_ci"][1] < 0),
                "cluster_ci_above_zero": sum(1 for r in rs if r[f"{name}_minus_vanilla_hr{k}_cluster_ci"][0] > 0),
                "beats_published": sum(1 for r in rs if r[f"{name}_minus_published_hr{k}"] > 0),
                "below_published_ci": sum(1 for r in rs if r[f"{name}_minus_published_hr{k}_ci"][1] < 0),
            }
    pub_counts = {}
    for k in (5, 10):
        diffs = []
        for key in grids:
            pj = per_job[key]
            diff = 100.0 * ((pj["published"] < k).astype(np.float64) - (pj["vanilla"] < k).astype(np.float64))
            diffs.append(paired_bootstrap_ci(diff, n_resamples=N_BOOT, seed=SEED))
        pub_counts[f"hr{k}"] = {
            "pairs": len(diffs),
            "beats_vanilla": sum(1 for m, _, _ in diffs if m > 0),
            "ci_above_zero": sum(1 for _, lo, _ in diffs if lo > 0),
        }

    # E: ranking in the main comparison table.
    ranking = ranking_stability(paths.repo_root, {(r["dataset"], r["retriever"]): r for r in rows_d})

    # C: Table 9 reconstruction test.
    table9 = table9_test(paths.repo_root, grids, grid)

    analysis = {
        "experiment": EXPERIMENT_ID,
        "dry_run": paths.dry_run,
        "grid_points": len(grid),
        "grid": {"lambda": list(DRY_LAMBDA_GRID if paths.dry_run else LAMBDA_GRID), "mu": list(DRY_MU_GRID if paths.dry_run else MU_GRID)},
        "jobs": [job_name(d, r) for d, r in jobs],
        "workers": worker_out,
        "per_job": rows_d,
        "counts": summary_counts,
        "published_counts": pub_counts,
        "lodo": {job_name(*k): grid.points[v] for k, v in lodo.items()},
        "lodo_global": {d: grid.points[v] for d, v in lodo_global.items()},
        "ranking": ranking,
        "table9": table9,
        "seconds": time.perf_counter() - t_start,
    }
    write_tables(paths, analysis, grids, grid)
    analysis["figures"] = write_heatmaps(paths, grids, grid)
    write_json(paths.results_dir / "analysis.json", analysis)
    write_report(paths, analysis, grid)
    write_json(paths.out_dir / "ANALYSIS_DONE.json", {"finished_at": time.strftime("%Y-%m-%d %H:%M:%S"), "jobs": analysis["jobs"]})
    return analysis


def ranking_stability(repo_root: Path, rows: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    """SAADI's rank in the 40 cells of Table 1, published vs A1 held-out (others unchanged)."""

    path = repo_root / "results" / "tables" / "main" / "retrieval_comparison" / "table.csv"
    with path.open("r", encoding="utf-8") as handle:
        table = list(csv.DictReader(handle))
    disp_to_ret = {v: k for k, v in RETRIEVER_DISPLAY.items()}
    cells = []
    for rdisp in [RETRIEVER_DISPLAY[r] for r in RETRIEVERS]:
        methods = [t for t in table if t["Retriever"] == rdisp]
        saadi = next(t for t in methods if t["Method"] == "Saadi")
        for d in DATASET_KEYS:
            for k in (5, 10):
                col = f"{DATASET_DISPLAY[d]} HR@{k} (%)"
                others = [(t["Method"], float(t[col])) for t in methods if t["Method"] != "Saadi"]
                v_pub = float(saadi[col])
                row = rows.get((d, disp_to_ret[rdisp]))
                v_new = row.get(f"a1_hr{k}") if row else None
                rank_pub = 1 + sum(1 for _, v in others if v > v_pub)
                # Held-out value compared at the table's printed precision (one decimal).
                rank_new = 1 + sum(1 for _, v in others if v > round(v_new, 1)) if v_new is not None else None
                cells.append(
                    {
                        "retriever": rdisp,
                        "dataset": DATASET_DISPLAY[d],
                        "cutoff": k,
                        "saadi_published": v_pub,
                        "saadi_heldout_a1": v_new,
                        "rank_published": rank_pub,
                        "rank_heldout": rank_new,
                        "best_other": max(others, key=lambda x: x[1]),
                        "changed": rank_new is not None and rank_new != rank_pub,
                        "available": v_new is not None,
                    }
                )
    avail = [c for c in cells if c["available"]]
    return {
        "source": rel(path, repo_root),
        "cells": cells,
        "top2_published": sum(1 for c in cells if c["rank_published"] <= 2),
        "top2_heldout": sum(1 for c in avail if c["rank_heldout"] <= 2)
        + sum(1 for c in cells if not c["available"] and c["rank_published"] <= 2),
        "n_cells": len(cells),
        "n_cells_recomputed": len(avail),
        "first_published": sum(1 for c in cells if c["rank_published"] == 1),
        "first_heldout": sum(1 for c in avail if c["rank_heldout"] == 1)
        + sum(1 for c in cells if not c["available"] and c["rank_published"] == 1),
    }


def _round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5 + 1e-9))


def table9_test(repo_root: Path, grids: dict[tuple[str, str], JobGrid], grid: Grid) -> dict[str, Any]:
    """Can the 100-query tuning sample be reconstructed? Compare candidate query sets with Table 9."""

    path = reference_dir(repo_root) / "summaries" / "43a8b999722a1192ad32.json"
    reported = json.loads(path.read_text(encoding="utf-8"))
    ref = {(x["dataset"], x["retriever"], x["weights"]): (int(x["hr_percent_at_5"]), int(x["hr_percent_at_10"])) for x in reported}
    candidates: dict[str, Any] = {}
    available = [k for k in TABLE9_JOBS if k in grids]

    def sets_for(jg: JobGrid) -> dict[str, np.ndarray]:
        n = len(jg.query_id)
        out = {"full test set": np.arange(n), "first 100 saved rows": np.arange(min(100, n))}
        try:
            ids = np.asarray([int(x) for x in jg.query_id])
            out["100 smallest query ids"] = np.argsort(ids, kind="stable")[:100]
        except ValueError:
            pass
        return out

    for norm in ("zscore", "none"):
        for d, r in available:
            jg = grids[(d, r)]
            for cname, sel in sets_for(jg).items():
                entry = candidates.setdefault(f"{cname}, normalization={norm}", {"values": [], "matches": 0, "compared": 0})
                for j, (lm, ls) in enumerate(TABLE9_SETTINGS):
                    if norm == "zscore":
                        ranks = jg.hit_rank[sel, grid.index_of((lm, ls, 1.0, 1.0))] if _in_grid(grid, (lm, ls, 1.0, 1.0)) else None
                    else:
                        ranks = jg.table9_none[sel, j] if jg.table9_none.size else None
                    if ranks is None:
                        continue
                    key = (DATASET_DISPLAY[d], RETRIEVER_DISPLAY[r], f"1.0/{lm:.1f}/{ls:.1f}")
                    rep = ref.get(key)
                    ours = (_pct(ranks < 5), _pct(ranks < 10))
                    ok = rep is not None and (_round_half_up(ours[0]), _round_half_up(ours[1])) == rep
                    entry["values"].append({"cell": list(key), "ours": ours, "reported": rep, "match": ok})
                    entry["compared"] += 1
                    entry["matches"] += int(ok)
    expected_cells = len(TABLE9_JOBS) * len(TABLE9_SETTINGS) * 2
    reconstructed = [
        n
        for n, c in candidates.items()
        if c["compared"] * 2 == expected_cells and c["matches"] == c["compared"] and not n.startswith("full")
    ]
    for c in candidates.values():
        c["cells_compared"] = 2 * c["compared"]
        c["cells_matching"] = sum(
            int(v["reported"] is not None and _round_half_up(v["ours"][0]) == v["reported"][0])
            + int(v["reported"] is not None and _round_half_up(v["ours"][1]) == v["reported"][1])
            for v in c["values"]
        )
    excluded = []
    if reconstructed:
        cname = reconstructed[0].split(", normalization=")[0]
        for (d, r), jg in grids.items():
            if d not in ("loogle", "novelhopqa"):
                continue
            sel = sets_for(jg)[cname]
            keep = np.setdiff1d(np.arange(len(jg.query_id)), sel)
            ranks = jg.hit_rank[keep, grid.published_index]
            excluded.append({"dataset": d, "retriever": r, "n_kept": int(keep.size), "hr5": _pct(ranks < 5), "hr10": _pct(ranks < 10)})
    return {
        "excluded_results": excluded,
        "source": rel(path, repo_root),
        "sampling_code": "load_*_qa_entries(..., selection_method='random') calls random.sample(idxs, n) on the global RNG; no seed is set anywhere in src/",
        "candidates": candidates,
        "expected_cells": expected_cells,
        "reconstructed": reconstructed,
    }


def _in_grid(grid: Grid, point: tuple[float, float, float, float]) -> bool:
    try:
        grid.index_of(point)
        return True
    except KeyError:
        return False


# ---------------------------------------------------------------------------
# Tables, figures, reports
# ---------------------------------------------------------------------------


def _f(x: Any, nd: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{nd}f}"


def _s(x: Any, nd: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:+.{nd}f}"


def _ci(ci: Any, nd: int = 1) -> str:
    if not ci:
        return "—"
    return f"[{ci[0]:+.{nd}f}, {ci[1]:+.{nd}f}]"


def _pt(p: Any) -> str:
    if p is None:
        return "—"
    lm, ls, ma, md = p
    return f"({lm:g}, {ls:g} | {ma:g}, {md:g})"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, list | tuple | dict) else v) for k, v in r.items()})


def write_tables(paths: C2Paths, analysis: dict[str, Any], grids: dict[tuple[str, str], JobGrid], grid: Grid) -> None:
    tdir = paths.results_dir / "tables"
    rows = analysis["per_job"]
    _write_csv(tdir / "d_per_pair.csv", rows)
    folds = []
    for r in rows:
        for f, (p1, p2) in enumerate(zip(r["chosen_a1"], r["chosen_a2"], strict=True)):
            folds.append({"dataset": r["dataset"], "retriever": r["retriever"], "fold": f, "a1_point": list(p1), "a2_point": list(p2)})
    _write_csv(tdir / "a_chosen_per_fold.csv", folds)
    _write_csv(
        tdir / "b_lodo.csv",
        [
            {
                "dataset": r["dataset"],
                "retriever": r["retriever"],
                "b_point": r["chosen_b"],
                "b_hr5": r["b_hr5"],
                "b_hr10": r["b_hr10"],
                "b_global_point": r["chosen_b_global"],
                "b_global_hr5": r["b_global_hr5"],
                "b_global_hr10": r["b_global_hr10"],
            }
            for r in rows
        ],
    )
    t9 = []
    for cname, c in analysis["table9"]["candidates"].items():
        for v in c["values"]:
            t9.append(
                {
                    "candidate": cname,
                    "dataset": v["cell"][0],
                    "retriever": v["cell"][1],
                    "weights": v["cell"][2],
                    "ours_hr5": v["ours"][0],
                    "ours_hr10": v["ours"][1],
                    "reported_hr5": (v["reported"] or [None, None])[0],
                    "reported_hr10": (v["reported"] or [None, None])[1],
                    "match": v["match"],
                }
            )
    _write_csv(tdir / "c_table9_reconstruction.csv", t9)
    _write_csv(tdir / "e_ranking_cells.csv", analysis["ranking"]["cells"])
    heat = []
    for (d, r), jg in grids.items():
        h10 = 100.0 * jg.h10.mean(0)
        h5 = 100.0 * jg.h5.mean(0)
        for gi in range(len(grid)):
            lm, ls, ma, md = grid.points[gi]
            heat.append(
                {"dataset": d, "retriever": r, "lambda_M": lm, "lambda_S": ls, "mu_A": ma, "mu_D": md, "hr5": h5[gi], "hr10": h10[gi]}
            )
    _write_csv(tdir / "f_grid_hr.csv", heat)
    _write_csv(
        tdir / "f_sensitivity.csv",
        [
            {
                k: r[k]
                for k in (
                    "dataset",
                    "retriever",
                    "f_hr10_published",
                    "f_hr10_best",
                    "f_hr10_best_point",
                    "f_hr10_worst",
                    "f_share_within_1pt",
                    "f_hr10_full_grid_best",
                    "stab_a1_distinct",
                    "stab_a1_published_folds",
                    "stab_a1_mean_l1",
                )
            }
            for r in rows
        ],
    )
    learned = []
    for job, w in analysis["workers"].items():
        for fw in w["learned_weights"]:
            learned.append(
                {
                    "job": job,
                    "fold": fw["fold"],
                    **{n: v for n, v in zip(LEARNED_FEATURES, fw["w"], strict=True)},
                    "n_pairs": fw["n_pairs"],
                    "iterations": fw["iterations"],
                    "converged": fw["converged"],
                }
            )
    _write_csv(tdir / "g_learned_weights.csv", learned)
    _write_csv(
        tdir / "verification.csv",
        [
            {
                k: w[k]
                for k in (
                    "job",
                    "n_queries",
                    "verify_rankings_compared",
                    "verify_rankings_different",
                    "verify_hit_rank_different",
                    "published_top10_vs_streams_mismatch",
                    "vanilla_top10_mismatch",
                    "seconds",
                )
            }
            for w in analysis["workers"].values()
        ],
    )


def write_heatmaps(paths: C2Paths, grids: dict[tuple[str, str], JobGrid], grid: Grid) -> dict[str, Any]:
    """HR@10 over (lambda_M, lambda_S) with field weights (1, 1, 1): BM25 per dataset, and the mean over retrievers."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on the environment
        return {"written": [], "skipped": f"matplotlib not importable ({exc}); the values are in tables/f_grid_hr.csv"}
    lams = sorted(set(grid.lam_m.tolist()))
    datasets = [d for d in DATASET_KEYS if any(k[0] == d for k in grids)]

    def matrix(keys: list[tuple[str, str]]) -> np.ndarray:
        m = np.zeros((len(lams), len(lams)))
        for i, lm in enumerate(lams):
            for j, ls in enumerate(lams):
                gi = grid.index_of((lm, ls, 1.0, 1.0))
                m[i, j] = np.mean([100.0 * grids[k].h10[:, gi].mean() for k in keys])
        return m

    written = []
    fig_dir = paths.results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for name, title, keys_for in (
        ("heatmap_hr10_bm25.png", "BM25", lambda d: [(d, "bm25")] if (d, "bm25") in grids else []),
        ("heatmap_hr10_mean_over_retrievers.png", "mean over retrievers", lambda d: [k for k in grids if k[0] == d]),
    ):
        ds = [d for d in datasets if keys_for(d)]
        if not ds:
            continue
        fig, axes = plt.subplots(1, len(ds), figsize=(4.2 * len(ds), 3.8), squeeze=False)
        for ax, d in zip(axes[0], ds, strict=True):
            m = matrix(keys_for(d))
            im = ax.imshow(m, origin="lower", cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(lams)), [f"{x:g}" for x in lams], fontsize=7)
            ax.set_yticks(range(len(lams)), [f"{x:g}" for x in lams], fontsize=7)
            ax.set_xlabel("lambda_S")
            ax.set_ylabel("lambda_M")
            pi, pj = lams.index(PUBLISHED_POINT[0]), lams.index(PUBLISHED_POINT[1])
            ax.scatter([pj], [pi], marker="o", facecolors="none", edgecolors="white", s=90, linewidths=1.5)
            bi, bj = np.unravel_index(int(np.argmax(m)), m.shape)
            ax.scatter([bj], [bi], marker="x", color="red", s=50)
            ax.set_title(f"{DATASET_DISPLAY[d]} ({title})", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="HR@10 (%)")
        fig.suptitle("HR@10 over channel weights, field weights (1,1,1); o = published, x = best", fontsize=9)
        fig.tight_layout()
        out = fig_dir / name
        fig.savefig(out, dpi=150)
        plt.close(fig)
        written.append(rel(out, paths.repo_root))
    return {"written": written}


SAVED_INDEX_TIME_ABLATIONS = (
    ("folding factor f (branching)", "results/tables/ablations/hierarchy_branching/"),
    ("recall budget E/A/D", "results/tables/ablations/memory_budget/"),
    ("memory selector", "results/tables/ablations/memory_selector/"),
    ("field weights (Table of mu, saved summaries)", "results/tables/ablations/memory_field_weights/"),
    ("channel weights (Table 9, saved summaries)", "results/tables/ablations/score_fusion/"),
)


def write_report(paths: C2Paths, a: dict[str, Any], grid: Grid) -> None:
    rows = a["per_job"]
    cnt = a["counts"]
    pc = a["published_counts"]
    rk = a["ranking"]
    t9 = a["table9"]
    n = len(rows)
    dry = a["dry_run"]

    def med(key: str) -> float:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(np.median(vals)) if vals else float("nan")

    def rng(key: str) -> tuple[float, float]:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return (min(vals), max(vals)) if vals else (float("nan"), float("nan"))

    def mean(key: str) -> float:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(vals)) if vals else float("nan")

    c1, c1_5 = cnt["a1_hr10"], cnt["a1_hr5"]
    kset = sorted({r["n_folds"] for r in rows})
    kfold = f"{kset[0]}-fold" if len(kset) == 1 else f"{min(kset)}- to {max(kset)}-fold"
    lo, hi = rng("a1_minus_published_hr10")
    L: list[str] = []
    L.append(
        "# C2: SAADI fusion weights chosen on held-out queries" + (" (DRY RUN: BM25, 2 documents per dataset, reduced grid)" if dry else "")
    )
    L.append("")
    L.append("## Summary")
    L.append("")
    L.append(
        f"1. mokb Q1. With all four weights chosen without the evaluated query ({kfold} CV grouped by document, {len(grid)}-point grid; A1), held-out SAADI is above the text-only Vanilla in {c1['beats_vanilla']}/{c1['pairs']} (dataset, retriever) pairs at HR@10 and {c1_5['beats_vanilla']}/{c1_5['pairs']} at HR@5; the 95% paired interval is above zero in {c1['ci_above_zero']}/{c1['pairs']} (HR@10) and {c1_5['ci_above_zero']}/{c1_5['pairs']} (HR@5). With the published weights the interval is above zero in {pc['hr10']['ci_above_zero']}/{pc['hr10']['pairs']} (HR@10) and {pc['hr5']['ci_above_zero']}/{pc['hr5']['pairs']} (HR@5)."
    )
    L.append(
        f"2. Held-out minus published weights, HR@10: mean {_s(mean('a1_minus_published_hr10'), 2)} points (range {_s(lo, 2)} to {_s(hi, 2)}); the interval is below zero in {c1['below_published_ci']}/{c1['pairs']} pairs. HR@5: mean {_s(mean('a1_minus_published_hr5'), 2)}."
    )
    L.append(
        f"3. Ranking (Table 1): SAADI is first or second in {rk['top2_heldout']}/{rk['n_cells']} cells with held-out (A1) weights vs {rk['top2_published']}/{rk['n_cells']} with published weights ({rk['n_cells_recomputed']} cells recomputed); {sum(1 for c in rk['cells'] if c['changed'])} cell(s) changed rank."
    )
    cb, cg = cnt["b_hr10"], cnt["b_global_hr10"]
    L.append(
        f"4. iVU8 Q3. Weights transferred from the other three datasets (B) beat Vanilla at HR@10 in {cb['beats_vanilla']}/{cb['pairs']} pairs (interval above zero: {cb['ci_above_zero']}); one global setting per held-out dataset: {cg['beats_vanilla']}/{cg['pairs']} ({cg['ci_above_zero']}). Median share of channel settings within 1 point of the best HR@10: {med('f_share_within_1pt'):.2f}."
    )
    cl = cnt["learned_hr10"]
    L.append(
        f"5. A learned pairwise logistic fusion (demonstration) gives mean HR@10 {mean('learned_hr10'):.1f} vs {mean('a1_hr10'):.1f} for A1 and beats Vanilla in {cl['beats_vanilla']}/{cl['pairs']} pairs. Part C: {'the tuning sample was reconstructed (' + t9['reconstructed'][0] + ')' if t9['reconstructed'] else 'the 100-query tuning sample could not be reconstructed, so C is skipped'}."
    )
    L.append("")
    L.append("## Definitions and commands")
    L.append("")
    L.append(
        "- Hit@k: a gold chunk in the top k, or every chunk of one evidence group in the top k (`evaluate_retrieval_results`). HR = hits / eligible queries. Hit rank = the 0-based rank at which this is first met; MRR = mean of 1/(hit rank + 1)."
    )
    L.append(
        f"- Score: Equation 2 as published (`combine_mbr_variant_chunk_scores`, variant MBR2, z-score per channel over the document's chunks, max aggregation, per-chunk missing-signal renormalization). lambda_T = mu_N = 1. Grid: lambda_M, lambda_S in {a['grid']['lambda']}; mu_A, mu_D in {a['grid']['mu']} ({len(grid)} points). Published point (lambda_M, lambda_S | mu_A, mu_D) = {_pt(PUBLISHED_POINT)}. Vanilla = the text channel alone (lambda_M = lambda_S = 0)."
    )
    L.append(
        "- A: 5 outer folds over queries grouped by document (groups shuffled with seed 42, each group to the fold with fewest queries); per training part the point with the highest mean HR@10 (ties: HR@5, MRR, smallest L1 distance to the published point). A1 = all four weights; A2 = channel weights only, mu = (1, 1, 1). B: per retriever, the point maximizing the mean of per-dataset HR@10 over the other three datasets; B-global: the same over all retrievers of the other three datasets."
    )
    L.append(
        "- D: 95% paired bootstrap over queries (10,000 resamples, seed 42; `paired_bootstrap_ci`) and a document-level cluster bootstrap (10,000, seed 42). Differences in HR points. Commands: `bash experiments/rebuttal/C2_heldout_fusion/run_gpus_0123.sh`, `bash experiments/rebuttal/C2_heldout_fusion/run_gpus_4567.sh`; analysis only: `... --analysis-only`."
    )
    L.append("")
    L.append("## Reproduction and checks")
    L.append("")
    L.append(
        "| Pair | Queries | Identical top-10 vs saved | HR@5 / HR@10 ours (published weights) | saved | text-only vs Table 1 Vanilla (HR@5 / HR@10) |"
    )
    L.append("| --- | --- | --- | --- | --- | --- |")
    for job in a["jobs"]:
        check = json.loads(paths.check_path(job).read_text(encoding="utf-8"))
        to = check["text_only"]
        drift_mark = " (drift accepted)" if check.get("drift_accepted") else ""
        L.append(
            f"| {job} | {check['n_queries']} | {check['identical_top10']}{drift_mark} | {_f(check['hr_ours_published']['hr_percent']['5'], 2)} / {_f(check['hr_ours_published']['hr_percent']['10'], 2)} | {_f(check['hr_saved']['hr_percent']['5'], 2)} / {_f(check['hr_saved']['hr_percent']['10'], 2)} | {_f(to['hr']['hr_percent']['5'], 2)} / {_f(to['hr']['hr_percent']['10'], 2)} vs {_f(to['reported_percent']['5'])} / {_f(to['reported_percent']['10'])}{'' if to['passed'] else ' (differs)'} |"
        )
    vw = a["workers"].values()
    drifted = [j for j in a["jobs"] if json.loads(paths.check_path(j).read_text(encoding="utf-8")).get("drift_accepted")]
    L.append("")
    if drifted:
        L.append(
            f"Embedding drift accepted (decided with the authors) for {', '.join(drifted)}: recomputed scores differ from the paper's run by about 1e-3, "
            f"so near-ties reorder (identical top-10 below 99%), with HR within {DRIFT_MAX_HR_POINTS} points of the saved values. For these pairs every system "
            "(Vanilla, published weights, held-out weights) is computed from the same recomputed streams, so the comparisons stay paired; the paper's saved values are in the column 'saved'."
        )
        L.append("")
    L.append(
        f"Vectorized grid vs the repo function (full rankings, published point + 20 random grid points, seed 42): {sum(w['verify_rankings_different'] for w in vw)} different rankings and {sum(w['verify_hit_rank_different'] for w in vw)} different hit ranks out of {sum(w['verify_rankings_compared'] for w in vw)} compared. Vanilla grid point vs saved text-only top-10: {sum(w['vanilla_top10_mismatch'] for w in vw)} mismatches."
    )
    L.append("")
    L.append("## A. Cross-validated weight selection (HR %, every query scored with weights chosen without it)")
    L.append("")
    L.append("| Dataset | Retriever | Vanilla @5 / @10 | Published @5 / @10 | A1 held-out @5 / @10 | A2 (channels only) @5 / @10 | folds |")
    L.append("| --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        L.append(
            f"| {DATASET_DISPLAY[r['dataset']]} | {RETRIEVER_DISPLAY[r['retriever']]} | {_f(r['vanilla_hr5'])} / {_f(r['vanilla_hr10'])} | {_f(r['published_hr5'])} / {_f(r['published_hr10'])} | {_f(r['a1_hr5'])} / {_f(r['a1_hr10'])} | {_f(r['a2_hr5'])} / {_f(r['a2_hr10'])} | {r['n_folds']} |"
        )
    L.append("")
    L.append("## B. Leave-one-dataset-out transfer")
    L.append("")
    L.append(
        "| Held-out dataset | Retriever | Point chosen on the other datasets | HR@5 / HR@10 | minus Vanilla @10 [95% CI] | Global point | Global HR@5 / HR@10 |"
    )
    L.append("| --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        L.append(
            f"| {DATASET_DISPLAY[r['dataset']]} | {RETRIEVER_DISPLAY[r['retriever']]} | {_pt(r['chosen_b'])} | {_f(r['b_hr5'])} / {_f(r['b_hr10'])} | {_s(r.get('b_minus_vanilla_hr10'))} {_ci(r.get('b_minus_vanilla_hr10_ci'))} | {_pt(r['chosen_b_global'])} | {_f(r['b_global_hr5'])} / {_f(r['b_global_hr10'])} |"
        )
    L.append("")
    L.append("## C. Excluding the original tuning sample")
    L.append("")
    L.append(f"- How the sample was drawn: {t9['sampling_code']}. The query ids of the sample are not saved, so they cannot be read back.")
    L.append(
        f"- Test: recompute the Table 9 cells ({t9['expected_cells']}: LooGLE and NovelHopQA x BM25, Contriever, Jina x 5 channel settings x HR@5/HR@10, mu = (1, 1, 1)) on candidate query sets and compare with the reported integer percentages ({t9['source']})."
    )
    L.append("")
    L.append("| Candidate query set | Cells compared | Cells matching |")
    L.append("| --- | --- | --- |")
    for cname, c in t9["candidates"].items():
        L.append(f"| {cname} | {c['cells_compared']} | {c['cells_matching']} |")
    L.append("")
    if t9["reconstructed"]:
        L.append(f"Reconstructed: {t9['reconstructed'][0]}. Published-weight results without those queries:")
        L.append("")
        L.append("| Dataset | Retriever | Queries kept | HR@5 | HR@10 |")
        L.append("| --- | --- | --- | --- | --- |")
        for e in t9["excluded_results"]:
            L.append(
                f"| {DATASET_DISPLAY[e['dataset']]} | {RETRIEVER_DISPLAY[e['retriever']]} | {e['n_kept']} | {_f(e['hr5'])} | {_f(e['hr10'])} |"
            )
    else:
        L.append(
            "No candidate 100-query set matches every cell, so the tuning sample is not reconstructed and C is skipped. A and B do not depend on it. Per-cell values: `tables/c_table9_reconstruction.csv`."
        )
    L.append("")
    L.append("## D. Paired differences (HR points; 95% paired bootstrap over queries; cluster = by document)")
    L.append("")
    for k in (10, 5):
        L.append(f"HR@{k}:")
        L.append("")
        L.append(
            "| Pair | A1 minus Vanilla [CI] | cluster CI | A1 minus published [CI] | cluster CI | A2 minus Vanilla [CI] | B minus Vanilla [CI] |"
        )
        L.append("| --- | --- | --- | --- | --- | --- | --- |")
        for r in rows:
            L.append(
                f"| {DATASET_DISPLAY[r['dataset']]} / {RETRIEVER_DISPLAY[r['retriever']]} | {_s(r[f'a1_minus_vanilla_hr{k}'])} {_ci(r[f'a1_minus_vanilla_hr{k}_ci'])} | {_ci(r[f'a1_minus_vanilla_hr{k}_cluster_ci'])} | {_s(r[f'a1_minus_published_hr{k}'])} {_ci(r[f'a1_minus_published_hr{k}_ci'])} | {_ci(r[f'a1_minus_published_hr{k}_cluster_ci'])} | {_s(r[f'a2_minus_vanilla_hr{k}'])} {_ci(r[f'a2_minus_vanilla_hr{k}_ci'])} | {_s(r.get(f'b_minus_vanilla_hr{k}'))} {_ci(r.get(f'b_minus_vanilla_hr{k}_ci'))} |"
            )
        L.append("")
    L.append("| System | HR@k | above Vanilla | CI above 0 | CI below 0 | cluster CI above 0 | above published | CI below published |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    names = {"a1": "A1 held-out", "a2": "A2 held-out", "b": "B transfer", "b_global": "B global", "learned": "G learned"}
    for name, disp in names.items():
        for k in (5, 10):
            c = cnt[f"{name}_hr{k}"]
            L.append(
                f"| {disp} | {k} | {c['beats_vanilla']}/{c['pairs']} | {c['ci_above_zero']} | {c['ci_below_zero']} | {c['cluster_ci_above_zero']} | {c['beats_published']} | {c['below_published_ci']} |"
            )
    L.append(f"| Published | 5 | {pc['hr5']['beats_vanilla']}/{pc['hr5']['pairs']} | {pc['hr5']['ci_above_zero']} | | | | |")
    L.append(f"| Published | 10 | {pc['hr10']['beats_vanilla']}/{pc['hr10']['pairs']} | {pc['hr10']['ci_above_zero']} | | | | |")
    L.append("")
    L.append("## E. Ranking stability in Table 1")
    L.append("")
    L.append(
        f"SAADI first or second: {rk['top2_published']}/{rk['n_cells']} with published weights, {rk['top2_heldout']}/{rk['n_cells']} with A1 held-out weights (other methods unchanged; held-out value rounded to one decimal as printed; {rk['n_cells_recomputed']} cells recomputed). First: {rk['first_published']} -> {rk['first_heldout']}. Source: `{rk['source']}`."
    )
    L.append("")
    changed = [c for c in rk["cells"] if c["changed"]]
    if changed:
        L.append("| Retriever | Dataset | HR@k | SAADI published | SAADI A1 | rank published -> held-out | best other |")
        L.append("| --- | --- | --- | --- | --- | --- | --- |")
        for c in changed:
            L.append(
                f"| {c['retriever']} | {c['dataset']} | {c['cutoff']} | {_f(c['saadi_published'])} | {_f(c['saadi_heldout_a1'])} | {c['rank_published']} -> {c['rank_heldout']} | {c['best_other'][0]} {_f(c['best_other'][1])} |"
            )
    else:
        L.append("No cell changed rank.")
    L.append("")
    L.append("## F. Sensitivity (channel grid with mu = (1, 1, 1); chosen points per outer fold)")
    L.append("")
    L.append(
        "| Pair | HR@10 published | best (point) | worst | share within 1 pt of best | A1 points chosen in the folds (lambda_M, lambda_S | mu_A, mu_D) |"
    )
    L.append("| --- | --- | --- | --- | --- | --- |")
    for r in rows:
        L.append(
            f"| {DATASET_DISPLAY[r['dataset']]} / {RETRIEVER_DISPLAY[r['retriever']]} | {_f(r['f_hr10_published'])} | {_f(r['f_hr10_best'])} {_pt(r['f_hr10_best_point'])} | {_f(r['f_hr10_worst'])} | {r['f_share_within_1pt']:.2f} | {'; '.join(_pt(p) for p in r['chosen_a1'])} |"
        )
    L.append("")
    figs = a.get("figures", {})
    L.append("Heatmaps: " + (", ".join(f"`{p}`" for p in figs.get("written", [])) or str(figs.get("skipped", "none"))) + ".")
    L.append("")
    L.append("## G. Learned fusion (demonstration, not the main result)")
    L.append("")
    L.append(
        "Pairwise logistic regression in the same outer folds: each evidence chunk against every non-evidence chunk of the same training query; features = z-scored text, name, alias, description, summary (0 when missing) and a missing indicator per memory/summary channel; L2, C = 1.0, no intercept (Newton's method; seed 42 has no effect without an intercept). Weights per fold: `tables/g_learned_weights.csv`."
    )
    L.append("")
    L.append("| Pair | A1 @5 / @10 | Learned @5 / @10 | Learned minus A1 @10 |")
    L.append("| --- | --- | --- | --- |")
    for r in rows:
        L.append(
            f"| {DATASET_DISPLAY[r['dataset']]} / {RETRIEVER_DISPLAY[r['retriever']]} | {_f(r['a1_hr5'])} / {_f(r['a1_hr10'])} | {_f(r['learned_hr5'])} / {_f(r['learned_hr10'])} | {_s(r['learned_hr10'] - r['a1_hr10'])} |"
        )
    L.append("")
    L.append("## Not part of C2 (index-time settings; need new LLM builds)")
    L.append("")
    for name, p in SAVED_INDEX_TIME_ABLATIONS:
        L.append(f"- {name}: `{p}`" + ("" if (paths.repo_root / p).exists() else " (not found)"))
    L.append(
        "- chunk size: suite `configs/suites/chunk_size.yaml`; no saved results found in the repository (only the 500-token LooGLE/NovelHopQA and 100-token MuSiQue/QASPER builds are present)."
    )
    L.append("")
    L.append("## Files")
    L.append("")
    L.append(
        f"- Tables: `{rel(paths.results_dir / 'tables', paths.repo_root)}/`; full numbers: `{rel(paths.results_dir / 'analysis.json', paths.repo_root)}`."
    )
    L.append(f"- Streams and grid arrays (not committed): `{rel(paths.out_dir, paths.repo_root)}/streams/`, `grid/`.")
    (paths.results_dir / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # Short version for the rebuttal document.
    G: list[str] = []
    G.append("# C2 held-out fusion weights: key numbers" + (" (DRY RUN)" if dry else ""))
    G.append("")
    G.append(
        f"- Setup: {n} (dataset, retriever) pairs; all weights re-chosen on a {len(grid)}-point grid without the evaluated queries ({kfold} CV grouped by document, seed 42)."
    )
    G.append(
        f"- Mean HR@10 over pairs: Vanilla {mean('vanilla_hr10'):.1f}, SAADI published {mean('published_hr10'):.1f}, SAADI held-out (A1) {mean('a1_hr10'):.1f}, channels-only held-out (A2) {mean('a2_hr10'):.1f}."
    )
    G.append(
        f"- Mean HR@5 over pairs: Vanilla {mean('vanilla_hr5'):.1f}, published {mean('published_hr5'):.1f}, A1 {mean('a1_hr5'):.1f}, A2 {mean('a2_hr5'):.1f}."
    )
    G.append(
        f"- A1 above Vanilla: {c1['beats_vanilla']}/{c1['pairs']} pairs at HR@10 ({c1['ci_above_zero']} with the 95% CI above zero; {c1['cluster_ci_above_zero']} with the document-cluster CI); {c1_5['beats_vanilla']}/{c1_5['pairs']} at HR@5 ({c1_5['ci_above_zero']} CI above zero)."
    )
    G.append(
        f"- Published weights for comparison: CI above zero in {pc['hr10']['ci_above_zero']}/{pc['hr10']['pairs']} (HR@10) and {pc['hr5']['ci_above_zero']}/{pc['hr5']['pairs']} (HR@5)."
    )
    G.append(
        f"- A1 minus published, HR@10: mean {_s(mean('a1_minus_published_hr10'), 2)} points (range {_s(lo, 2)} to {_s(hi, 2)}); significantly below published in {c1['below_published_ci']}/{c1['pairs']} pairs."
    )
    G.append(
        f"- Table 1 ranking: SAADI first or second in {rk['top2_heldout']}/{rk['n_cells']} cells with held-out weights (published: {rk['top2_published']}/{rk['n_cells']})."
    )
    G.append(
        f"- Transfer from the other three datasets: above Vanilla at HR@10 in {cb['beats_vanilla']}/{cb['pairs']} pairs per retriever, {cg['beats_vanilla']}/{cg['pairs']} with one global setting."
    )
    G.append(
        f"- Stability: mean number of distinct points chosen across the outer folds {mean('stab_a1_distinct'):.1f}; folds choosing exactly the published point: {sum(r['stab_a1_published_folds'] for r in rows)}/{sum(r['n_folds'] for r in rows)}; mean L1 distance to the published point {mean('stab_a1_mean_l1'):.2f}."
    )
    G.append(
        f"- Sensitivity: median share of the {len(grid.channel_only)} channel settings within 1 HR@10 point of the best: {med('f_share_within_1pt'):.2f}; median HR@10 gap best minus published {float(np.median([r['f_hr10_best'] - r['f_hr10_published'] for r in rows])):.2f} points."
    )
    G.append(f"- Learned pairwise logistic fusion (demonstration): mean HR@10 {mean('learned_hr10'):.1f} vs A1 {mean('a1_hr10'):.1f}.")
    if drifted:
        G.append(
            f"- Embedding drift: {', '.join(drifted)} recomputed with scores about 1e-3 off the paper's run (HR within {DRIFT_MAX_HR_POINTS} points); all their systems use the same recomputed scores."
        )
    G.append(
        "- Original 100-query tuning sample: "
        + (
            "reconstructed and excluded (see REPORT.md, part C)."
            if t9["reconstructed"]
            else "drawn with an unseeded random.sample; not reconstructable from the code or saved outputs, so it is not excluded separately (A and B do not depend on it)."
        )
    )
    G.append("- Full tables, intervals and definitions: results/rebuttal/C2_heldout_fusion/REPORT.md.")
    (paths.results_dir / "REPORT_FOR_GDOC.md").write_text("\n".join(G) + "\n", encoding="utf-8")
