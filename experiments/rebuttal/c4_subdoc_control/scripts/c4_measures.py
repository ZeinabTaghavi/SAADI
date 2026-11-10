"""Per-query ranks and their summaries for C4 (shared by steps 03, 04 and 05).

For one (setting, retriever) pair and one method (Vanilla = text channel alone; SAADI =
published fusion, weights from the config), every query of the paper gets:

  A. full ranking (whole concatenated document, the paper's labels): 1-based hit rank;
  C. top-10 of the full ranking: does any chunk belong to the gold sub-document, and how
     many of the 10 do (a boundary chunk counts for each sub-document it overlaps);
  B. the 1-based hit rank inside the gold sub-document, for two restrictions:
       keep   - full-document z-scores and final scores, chunks outside dropped (main);
       renorm - each channel z-scored again over the gold sub-document's chunks only,
                then fused with the same weights (SAADI normalises per query);
     and two query sets:
       S - strict: every evidence chunk of the paper's labels lies in the gold sub-document;
           inside labels = the paper's labels;
       R - labels re-derived by the repository labeler on the gold sub-document's chunks
           (step 01); every query whose answer spans also match there.

The hit definition is the paper's (retrieval_metrics): a sufficient chunk in the top k, or a
whole evidence group in the top k. Ties break as in the repository (-raw text, chunk
index, chunk id). Nothing is tuned.
"""

from __future__ import annotations

import gzip
import math
from pathlib import Path
from typing import Any

import numpy as np

from c4_common import MAP_DIR, WORK_DIR, Setting, read_json, rel
from c4_streams import job_name, stream_paths

from saadi.rebuttal.heldout_fusion import (  # noqa: E402
    BIG_RANK,
    CHANNELS,
    QueryArrays,
    _zscore_column,
    final_scores,
    hit_ranks,
    load_query_arrays,
    ranking_order,
)

METHODS = ("van", "pub")  # Vanilla (text channel), SAADI (published)
METHOD_DISPLAY = {"van": "Vanilla", "pub": "SAADI"}
SETS = ("S", "R")
RESTRICTIONS = ("keep", "renorm")
INSIDE_K = (1, 3, 5, 10)
FULL_K = (5, 10)
PERQUERY_DIR = WORK_DIR / "perquery"

PERQUERY_COLUMNS = (
    ["query_id", "doc_id", "gold_subdoc", "n_gold_chunks", "in_S", "in_R"]
    + [f"full_rank_{m}" for m in METHODS]
    + [f"subdoc_hit10_{m}" for m in METHODS]
    + [f"subdoc_share10_{m}" for m in METHODS]
    + [f"in{s}_{r}_rank_{m}" for s in SETS for r in RESTRICTIONS for m in METHODS]
)


def perquery_path(setting: Setting, retriever: str) -> Path:
    return PERQUERY_DIR / f"{job_name(setting, retriever)}.csv.gz"


def method_points(cfg: dict[str, Any]) -> dict[str, tuple[np.ndarray, ...]]:
    if float(cfg["memory_name_weight"]) != 1.0 or float(cfg["lambda_t"]) != 1.0:
        raise RuntimeError("the vectorised fusion assumes lambda_T = mu_N = 1 (as in the published run)")
    if str(cfg["normalization_method"]) != "zscore" or str(cfg["memory_aggregation"]) != "max":
        raise RuntimeError("C4 reuses C2's z-score fusion; the config's normalisation/aggregation differ")
    mu_a, mu_d = float(cfg["memory_alias_weight"]), float(cfg["memory_description_weight"])
    return {
        "van": (np.array([0.0]), np.array([0.0]), np.array([mu_a]), np.array([mu_d])),
        "pub": (np.array([float(cfg["lambda_n"])]), np.array([float(cfg["lambda_h"])]), np.array([mu_a]), np.array([mu_d])),
    }


def _restrict(q: QueryArrays, idx: np.ndarray, gold_ids: list[str], group_ids: list[list[str]], *, renorm: bool, retriever: str) -> QueryArrays:
    """The query restricted to the chunks at positions idx (pool order kept)."""

    chunk_ids = [q.chunk_ids[int(i)] for i in idx]
    raw = q.raw[idx]
    if renorm:
        z = np.column_stack([_zscore_column(chunk_ids, raw[:, j], retriever) for j in range(len(CHANNELS))])
    else:
        z = q.z[idx]
    # Same tie-break key on the subset: the relative order of q.r0 is kept.
    r0 = np.empty(len(idx), dtype=np.int64)
    r0[np.argsort(q.r0[idx], kind="stable")] = np.arange(len(idx))
    pos = {c: i for i, c in enumerate(chunk_ids)}
    gold_pos = np.asarray(sorted({pos[c] for c in gold_ids if c in pos}), dtype=np.int64)
    groups = [np.asarray(sorted({pos[c] for c in g}), dtype=np.int64) for g in group_ids if g and all(c in pos for c in g)]
    return QueryArrays(
        qidx=q.qidx, query_id=q.query_id, doc_id=q.doc_id, chunk_ids=chunk_ids, chunk_index=q.chunk_index[idx],
        raw=raw, z=z, r0=r0, gold_pos=gold_pos, group_pos=groups, eligible=bool(gold_pos.size or groups),
    )


def _rank1(r: int) -> float:
    return math.inf if int(r) >= BIG_RANK else float(int(r) + 1)


def compute_pair(setting: Setting, retriever: str, cfg: dict[str, Any], *, log=print) -> dict[str, Any]:
    """Per-query ranks of one pair; writes work/perquery/<setting>__<retriever>.csv.gz."""

    import pandas as pd

    points = method_points(cfg)
    mp = read_json(MAP_DIR / f"{setting.key}.json")
    if not mp.get("ok"):
        raise RuntimeError(f"{setting.key}: step 01 map is not ok")
    chunk_sub: dict[str, list[str]] = mp["chunks"]
    mq: dict[str, dict[str, Any]] = mp["queries"]
    sp, qp = stream_paths(setting, retriever)
    queries = load_query_arrays(sp, qp, retriever)
    by_q = {q.query_id: q for q in queries}
    wanted = [qid for qid, m in mq.items() if retriever in (m.get("retrievers") or []) and m.get("eligible")]
    missing = [qid for qid in wanted if qid not in by_q]
    if missing:
        raise RuntimeError(f"{len(missing)} map queries missing from {rel(sp)} (first: {missing[:5]})")

    problems: list[str] = []
    diag = {"S_paper_label_outside": 0, "n_gold_chunks_mismatch": 0, "S_inside_not_eligible": 0, "R_inside_not_eligible": 0,
            "S_relabel_differs_from_paper": 0, "unmapped_chunks": 0}
    rows: list[dict[str, Any]] = []
    for qid in wanted:
        q = by_q[qid]
        m = mq[qid]
        gsd = m["gold_subdoc"]
        if q.doc_id != m["doc_id"]:
            raise RuntimeError(f"q{qid}: stream document {q.doc_id} != map document {m['doc_id']}")
        unm = [c for c in q.chunk_ids if c not in chunk_sub]
        if unm:
            diag["unmapped_chunks"] += len(unm)
            raise RuntimeError(f"q{qid}: {len(unm)} stream chunks not in the step 01 map (first {unm[:3]})")
        inside_mask = np.array([gsd in chunk_sub[c] for c in q.chunk_ids], dtype=bool)
        idx = np.flatnonzero(inside_mask)
        if idx.size != int(m["n_gold_chunks"]):
            diag["n_gold_chunks_mismatch"] += 1
        status = m["status"]
        in_S = status == "kept"
        in_R = bool(m.get("inside_eligible")) and status != "excluded_gold_subdoc_not_in_text"
        paper_gold = [q.chunk_ids[int(i)] for i in q.gold_pos]
        paper_groups = [[q.chunk_ids[int(i)] for i in g] for g in q.group_pos]
        if in_S:
            if not inside_mask[np.concatenate([q.gold_pos, *q.group_pos]).astype(np.int64)].all():
                diag["S_paper_label_outside"] += 1
                problems.append(f"q{qid}: in set S but a paper evidence chunk lies outside {gsd}")
            same = (sorted(paper_gold), sorted(sorted(g) for g in paper_groups)) == (
                sorted(m["inside_gold"]), sorted(sorted(g) for g in m["inside_groups"]))
            diag["S_relabel_differs_from_paper"] += int(not same)
        row: dict[str, Any] = {"query_id": qid, "doc_id": q.doc_id, "gold_subdoc": gsd, "n_gold_chunks": int(idx.size),
                               "in_S": int(in_S), "in_R": int(in_R)}
        for meth, pt in points.items():
            f = final_scores(q, *pt)
            row[f"full_rank_{meth}"] = _rank1(hit_ranks(q, f)[0])
            top10 = ranking_order(q, f)[0][:10]
            n_in = int(inside_mask[top10].sum())
            row[f"subdoc_hit10_{meth}"] = int(n_in > 0)
            row[f"subdoc_share10_{meth}"] = n_in / 10.0
        for set_name, member, gold_ids, group_ids in (
            ("S", in_S, paper_gold, paper_groups),
            ("R", in_R, list(m.get("inside_gold") or []), [list(g) for g in (m.get("inside_groups") or [])]),
        ):
            for restr in RESTRICTIONS:
                if not member:
                    for meth in METHODS:
                        row[f"in{set_name}_{restr}_rank_{meth}"] = math.nan
                    continue
                sub = _restrict(q, idx, gold_ids, group_ids, renorm=restr == "renorm", retriever=retriever)
                if not sub.eligible:
                    diag[f"{set_name}_inside_not_eligible"] += 1
                    problems.append(f"q{qid}: set {set_name} but no complete evidence inside {gsd}")
                for meth, pt in points.items():
                    row[f"in{set_name}_{restr}_rank_{meth}"] = _rank1(hit_ranks(sub, final_scores(sub, *pt))[0])
        rows.append(row)
    if diag["n_gold_chunks_mismatch"]:
        problems.append(f"{diag['n_gold_chunks_mismatch']} queries: gold sub-document chunk count in the stream != step 01 map")
    frame = pd.DataFrame(rows, columns=PERQUERY_COLUMNS)
    out = perquery_path(setting, retriever)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False)
    tmp.replace(out)
    log(f"[c4] {setting.key}/{retriever}: {len(frame)} queries -> {rel(out)}")
    return {"setting": setting.key, "retriever": retriever, "path": str(out), "n": len(frame), "diag": diag, "problems": problems}


def load_perquery(path: Path):
    import pandas as pd

    frame = pd.read_csv(path, dtype={"query_id": str, "doc_id": str, "gold_subdoc": str})
    for c in frame.columns:
        if "_rank_" in c:
            frame[c] = frame[c].astype(float)  # "inf" parses as inf
    return frame


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def set_frame(frame, set_name: str):
    if set_name == "all":
        return frame
    return frame[frame[f"in_{set_name}"] == 1]


def per_query_values(frame, set_name: str, meth: str, restr: str = "keep") -> dict[str, np.ndarray]:
    """Per-query 0/1 or real values of every measure (aligned rows), for one method."""

    sf = set_frame(frame, set_name)
    full = sf[f"full_rank_{meth}"].to_numpy(dtype=float)
    out: dict[str, np.ndarray] = {f"full_hr{k}": (full <= k).astype(float) for k in FULL_K}
    out["full_rank"] = full
    out["subdoc_hit10"] = sf[f"subdoc_hit10_{meth}"].to_numpy(dtype=float)
    out["subdoc_share10"] = sf[f"subdoc_share10_{meth}"].to_numpy(dtype=float)
    if set_name != "all":
        ins = sf[f"in{set_name}_{restr}_rank_{meth}"].to_numpy(dtype=float)
        nch = sf["n_gold_chunks"].to_numpy(dtype=int)
        for k in INSIDE_K:
            out[f"in_hr{k}"] = (ins <= k).astype(float)
            # HR@k on queries whose gold sub-document has more than k chunks (NaN elsewhere).
            out[f"in_hr{k}_gt"] = np.where(nch > k, (ins <= k).astype(float), np.nan)
        out["in_rr"] = np.where(np.isfinite(ins), 1.0 / ins, 0.0)
        out["in_rank"] = ins
    return out


def _median(x: np.ndarray) -> float | None:
    return float(np.median(x)) if x.size else None


def summarize(frame, set_name: str, meth: str) -> dict[str, Any]:
    """One row of table.csv: measures A and C, and B for both restrictions (percent / ranks)."""

    sf = set_frame(frame, set_name)
    v = per_query_values(frame, set_name, meth, "keep")
    n = len(sf)
    out: dict[str, Any] = {"n": n}
    for k in FULL_K:
        out[f"full_hr{k}"] = 100.0 * v[f"full_hr{k}"].mean() if n else None
    out["full_median_rank"] = _median(v["full_rank"])
    out["subdoc_hit10"] = 100.0 * v["subdoc_hit10"].mean() if n else None
    out["subdoc_share10"] = 100.0 * v["subdoc_share10"].mean() if n else None
    if set_name == "all":
        return out
    nch = sf["n_gold_chunks"].to_numpy(dtype=int)
    for restr in RESTRICTIONS:
        v = per_query_values(frame, set_name, meth, restr)
        p = f"{restr}_"
        for k in INSIDE_K:
            out[p + f"in_hr{k}"] = 100.0 * v[f"in_hr{k}"].mean() if n else None
            gt = v[f"in_hr{k}_gt"]
            ngt = int((nch > k).sum())
            out[p + f"in_hr{k}_gt"] = 100.0 * float(np.nanmean(gt)) if ngt else None
            out[f"n_gt{k}"] = ngt
        out[p + "in_mrr"] = 100.0 * v["in_rr"].mean() if n else None
        out[p + "in_median_rank"] = _median(v["in_rank"])
    return out


# The paired comparisons of the task (SAADI - Vanilla), per pair and query set.
PAIRED_MEASURES = (
    # (measure key, restriction or "", label)
    ("full_hr10", "", "Full ranking HR@10"),
    ("in_hr3", "keep", "Inside HR@3"),
    ("in_hr5", "keep", "Inside HR@5"),
    ("in_rr", "keep", "Inside MRR"),
    ("in_hr10_gt", "keep", "Inside HR@10 (> 10 chunks)"),
    ("subdoc_hit10", "", "Sub-document hit@10"),
    ("subdoc_share10", "", "Top-10 share from the gold sub-document"),
    ("in_hr3", "renorm", "Inside HR@3"),
    ("in_hr5", "renorm", "Inside HR@5"),
    ("in_rr", "renorm", "Inside MRR"),
    ("in_hr10_gt", "renorm", "Inside HR@10 (> 10 chunks)"),
)


def paired_diffs(frame, set_name: str, measure: str, restr: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(saadi, vanilla, saadi - vanilla) per query, NaN rows removed (in percent points)."""

    a = per_query_values(frame, set_name, "pub", restr or "keep")[measure]
    b = per_query_values(frame, set_name, "van", restr or "keep")[measure]
    keep = ~(np.isnan(a) | np.isnan(b))
    return 100.0 * a[keep], 100.0 * b[keep], 100.0 * (a[keep] - b[keep])
