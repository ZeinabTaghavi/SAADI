"""Full per-chunk channel scores ("streams") for C4, in C2's stream format.

One stream = one (setting, retriever): a parquet with one row per (query, chunk) holding the
raw text / name / alias / description / summary scores, plus a .queries.json with each
query's labels and the published / text-only top-10 computed at stream time by the
repository's own combination (combine_mbr_variant_chunk_scores).

BM25 streams are computed here on CPU with C2's code (score_mbr_variant_document_pools over
the saved construction cache; nothing is rebuilt). A dense retriever's streams are computed
here too when asked (step 02 --score-dense contriever: same code, the retriever's own encoder
on cuda, mps or cpu), into work/streams/; otherwise they are read from $C4_DENSE_STREAMS_DIR
(C2's output). work/streams/ wins when both exist.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any

from c4_common import DENSE_STREAMS_DIR, WORK_DIR, Setting, evidence_from_row, read_json, rel

from saadi.indexing.retriever_adapters.scoring import _ENCODED_POOL_CACHE  # noqa: E402
from saadi.indexing.retrieval_methods.mbr import build_mbr_document_memory_index  # noqa: E402
from saadi.rebuttal.heldout_fusion import (  # noqa: E402
    _atomic_write_bytes,
    _score_document,
    rows_by_document,
    save_streams,
)

BM25_STREAMS_DIR = WORK_DIR / "streams"


def job_name(setting: Setting, retriever: str) -> str:
    return f"{setting.key}__{retriever}"


def stream_paths(setting: Setting, retriever: str, *, for_writing: bool = False) -> tuple[Path, Path]:
    job = job_name(setting, retriever)
    local = (BM25_STREAMS_DIR / f"{job}.parquet", BM25_STREAMS_DIR / f"{job}.queries.json")
    if retriever == "bm25" or for_writing or (local[0].exists() and local[1].exists()):
        return local
    return DENSE_STREAMS_DIR / f"{job}.parquet", DENSE_STREAMS_DIR / f"{job}.queries.json"


def dense_device() -> str:
    """cuda if visible, else Apple mps, else cpu; $SAADI_RETRIEVER_DEVICE (C4_DENSE_DEVICE) wins."""

    import os

    want = os.environ.get("SAADI_RETRIEVER_DEVICE", "").strip()
    if want:
        return want
    import torch

    if torch.cuda.is_available():
        dev = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"
    os.environ["SAADI_RETRIEVER_DEVICE"] = dev  # read by the retriever adapters at construction
    return dev


def saved_rows(setting: Setting, retriever: str, paper_qids: list[str]) -> list[dict[str, Any]]:
    """The saved per-document result rows (question, labels, top-10) restricted to the paper's queries.

    The saved runs of QASPER 32K/128K/256K hold a few queries more than the paper's 1,321-query
    intersection; those are dropped. Every paper query must be present.
    """

    path = setting.saved_results_json(retriever)
    if not path.exists():
        raise FileNotFoundError(f"saved per-document results not found: {rel(path)}")
    rows = read_json(path)
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        qid = str(row.get("query_id", ""))
        if qid in by_id:
            raise RuntimeError(f"{rel(path)}: query {qid} appears twice")
        by_id[qid] = row
    missing = [q for q in paper_qids if q not in by_id]
    if missing:
        raise RuntimeError(f"{rel(path)}: {len(missing)} paper queries missing (first: {missing[:5]})")
    keep = set(paper_qids)
    return [row for row in rows if str(row.get("query_id", "")) in keep]  # saved order


def labels_equal(row_a: dict[str, Any], row_b: dict[str, Any]) -> bool:
    ga, gra = evidence_from_row(row_a)
    gb, grb = evidence_from_row(row_b)
    return (sorted(ga), sorted(sorted(g) for g in gra)) == (sorted(gb), sorted(sorted(g) for g in grb))


def compute_bm25_streams(setting: Setting, rows: list[dict[str, Any]], *, log=print, force: bool = False) -> dict[str, Any]:
    """Score every query of one setting with BM25 on CPU; per-document checkpoints make it resumable."""

    return compute_streams(setting, "bm25", rows, log=log, force=force)


def compute_streams(setting: Setting, retriever: str, rows: list[dict[str, Any]], *, log=print, force: bool = False) -> dict[str, Any]:
    """Score every query of one (setting, retriever); per-document checkpoints make it resumable."""

    out_path, queries_path = stream_paths(setting, retriever, for_writing=True)
    if out_path.exists() and queries_path.exists() and not force:
        return {"computed": False, "path": rel(out_path)}
    device = "cpu" if retriever == "bm25" else dense_device()
    job = job_name(setting, retriever)
    work = WORK_DIR / "streams_work" / job
    work.mkdir(parents=True, exist_ok=True)
    by_doc = rows_by_document(rows)
    docs: list[dict[str, Any]] = []
    t_start = time.time()
    from_ck = 0
    for di, doc_id in enumerate(by_doc):
        ck = work / f"doc_{di:04d}.pkl"
        if ck.exists() and not force:
            with ck.open("rb") as handle:
                doc = pickle.load(handle)
            if doc.get("doc_id") == doc_id and len(doc["queries"]) == len(by_doc[doc_id]):
                docs.append(doc)
                from_ck += 1
                continue
        t0 = time.time()
        index = build_mbr_document_memory_index(setting.index_json, doc_id=doc_id)
        doc = _score_document(index, by_doc[doc_id], retriever)
        # The encoded pools of a document are reused by its queries only; free them before the next one.
        _ENCODED_POOL_CACHE.clear()
        _atomic_write_bytes(ck, pickle.dumps(doc, protocol=pickle.HIGHEST_PROTOCOL))
        docs.append(doc)
        log(f"[{job}] doc {di + 1}/{len(by_doc)} {doc_id}: {len(doc['queries'])} queries, {len(doc['chunk_ids'])} chunks, {time.time() - t0:.0f}s")
    used = [rows[q["row_index"]] for d in docs for q in d["queries"]]
    n_rows = save_streams(setting.key, retriever, docs, used, out_path=out_path, queries_path=queries_path)
    anomalies = [a for d in docs for a in d.get("anomalies", [])]
    return {
        "computed": True,
        "path": rel(out_path),
        "parquet_rows": n_rows,
        "docs": len(docs),
        "docs_from_checkpoint": from_ck,
        "device": device,
        "seconds": time.time() - t_start,
        "anomalies": anomalies,
    }
