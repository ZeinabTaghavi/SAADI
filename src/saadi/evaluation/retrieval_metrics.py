"""Computes the paper's HR at explicit retrieval cutoffs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Any

from ..core.types import RetrievalEvalReport, RetrievalResult


def saved_retrieval_depth(results_path: str | Path) -> int | None:
    """Read a ranking's saved depth from its adjacent report index."""
    path = Path(results_path).resolve()
    index_path = path.parent / "report_index.json"
    if not index_path.is_file():
        return None
    index = json.loads(index_path.read_text(encoding="utf-8"))
    depths = {
        int(row["retrieve_k"])
        for row in index.get("runs", [])
        if Path(str(row.get("results_json", ""))).name == path.name
        and row.get("retrieve_k") is not None
    }
    if len(depths) > 1:
        raise ValueError(f"Conflicting saved ranking depths in {index_path}")
    return next(iter(depths), None)


def _normalize_chunk_groups(raw: Any) -> list[list[str]]:
    out: list[list[str]] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, list):
            continue
        group = [str(x) for x in item if str(x).strip()]
        if group:
            out.append(group)
    return out


def evaluate_retrieval_results(
    results: Iterable[RetrievalResult],
    *,
    eval_k: int = 5,
    retrieve_k: int | None = None,
    compute_per_query: bool = True,
    metadata: dict | None = None,
) -> RetrievalEvalReport:
    """A hit retrieves a sufficient single chunk or a complete evidence group within top-k.

    Only queries with a sufficient chunk or nonempty evidence group are eligible.
    Report HR@5, HR@10, and the primary cutoff. ``retrieve_k`` records a known
    saved ranking limit; cutoffs beyond that limit are unavailable.
    """
    if isinstance(eval_k, bool) or not isinstance(eval_k, int) or eval_k <= 0:
        raise ValueError(f"eval_k must be a positive integer (got: {eval_k!r})")
    if retrieve_k is not None:
        if isinstance(retrieve_k, bool) or not isinstance(retrieve_k, int) or retrieve_k <= 0:
            raise ValueError(f"retrieve_k must be a positive integer (got: {retrieve_k!r})")
        if eval_k > retrieve_k:
            raise ValueError(f"HR@{eval_k} requires rankings saved to at least depth {eval_k}; retrieve_k={retrieve_k}")
    cutoffs = sorted({5, 10, eval_k})
    available_cutoffs = [k for k in cutoffs if retrieve_k is None or k <= retrieve_k]
    hits_at_k = {str(k): 0 if k in available_cutoffs else None for k in cutoffs}
    n_examples = 0
    n_gold_eligible = 0
    n_gold_hits = 0
    n_silver_eligible = 0
    n_silver_hits_strict = 0
    n_silver_hits_loose = 0
    n_union_eligible = 0
    n_union_hits = 0
    per_query: list[dict[str, Any]] = []

    for r in results:
        n_examples += 1
        ranking = list(r.retrieved_chunk_ids)
        retrieved_ids = ranking[:eval_k]
        retrieved_set = set(retrieved_ids)
        gold_set = set(r.gold_chunk_ids)
        silver_groups = _normalize_chunk_groups(getattr(r, "silver_chunk_groups", []))
        silver_set = set(r.silver_chunk_ids)
        gold_hit = False
        silver_hit_strict = False
        silver_hit_loose = False
        union_hit = False

        if gold_set:
            n_gold_eligible += 1
            gold_hit = bool(retrieved_set & gold_set)
            if gold_hit:
                n_gold_hits += 1
        if silver_groups:
            silver_hit_strict = any(set(group).issubset(retrieved_set) for group in silver_groups)
            silver_hit_loose = bool(retrieved_set & silver_set)
        if silver_groups:
            n_silver_eligible += 1
            if silver_hit_strict:
                n_silver_hits_strict += 1
            if silver_hit_loose:
                n_silver_hits_loose += 1
        eligible = bool(gold_set or silver_groups)
        query_hits = {str(k): None for k in cutoffs}
        if eligible:
            n_union_eligible += 1
            union_hit = gold_hit or silver_hit_strict
            if union_hit:
                n_union_hits += 1
            for k in available_cutoffs:
                topk = set(ranking[:k])
                hit = bool(gold_set & topk) or any(set(group).issubset(topk) for group in silver_groups)
                query_hits[str(k)] = hit
                hits_at_k[str(k)] += int(hit)

        if compute_per_query:
            per_query.append(
                {
                    "query_id": r.query_id,
                    "doc_id": r.doc_id,
                    "hit": union_hit,
                    "eligible": eligible,
                    "hit_at_k": query_hits,
                    "retrieved_chunk_ids": retrieved_ids,
                    "relevant_chunk_ids": list(r.gold_chunk_ids),
                    "evidence_groups": [list(group) for group in silver_groups],
                }
            )

    gold_hit_rate = (n_gold_hits / n_gold_eligible) if n_gold_eligible else 0.0
    silver_strict_hit_rate = (n_silver_hits_strict / n_silver_eligible) if n_silver_eligible else 0.0
    silver_loose_hit_rate = (n_silver_hits_loose / n_silver_eligible) if n_silver_eligible else 0.0
    union_hit_rate = (n_union_hits / n_union_eligible) if n_union_eligible else None
    hit_rate_at_k = {
        key: count / n_union_eligible if count is not None and n_union_eligible else None
        for key, count in hits_at_k.items()
    }
    return RetrievalEvalReport(
        n_examples=n_examples,
        n_gold_eligible=n_gold_eligible,
        n_gold_hits=n_gold_hits,
        n_silver_eligible=n_silver_eligible,
        n_silver_hits_strict=n_silver_hits_strict,
        n_silver_hits_loose=n_silver_hits_loose,
        n_union_eligible=n_union_eligible,
        n_union_hits=n_union_hits,
        gold_hit_rate=gold_hit_rate,
        silver_strict_hit_rate=silver_strict_hit_rate,
        silver_loose_hit_rate=silver_loose_hit_rate,
        union_hit_rate=union_hit_rate,
        per_query=per_query,
        metadata={**(metadata or {}), "metric": "HR", "eval_k": eval_k,
                  "retrieve_k": retrieve_k, "reported_cutoffs": cutoffs,
                  "unavailable_cutoffs": [k for k in cutoffs if k not in available_cutoffs]},
        eval_k=eval_k,
        hit_rate_at_k=hit_rate_at_k,
        hits_at_k=hits_at_k,
    )


def _normalize_split(split: str) -> str:
    value = str(split).strip().lower()
    if value not in {"auto", "stateful"}:
        raise ValueError("Invalid split. Expected one of: auto, stateful")
    return value


def _extract_rows(payload: Any, *, split: str = "auto") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split = _normalize_split(split)

    if isinstance(payload, list):
        rows = [row for row in payload if isinstance(row, dict)]
        if len(rows) != len(payload):
            raise ValueError("Invalid retrieval results JSON: all list rows must be objects")
        commits = {
            str(row.get("git_commit", "")).strip()
            for row in rows
            if row.get("git_commit") is not None
            and str(row.get("git_commit", "")).strip()
        }
        metadata: dict[str, Any] = {"selected_split": "flat"}
        if len(commits) == 1:
            metadata["git_commit"] = next(iter(commits))
        elif len(commits) > 1:
            metadata["git_commits"] = sorted(commits)
        dirty_values = {
            bool(row["git_dirty"])
            for row in rows
            if isinstance(row.get("git_dirty"), bool)
        }
        if len(dirty_values) == 1:
            metadata["git_dirty"] = next(iter(dirty_values))
        elif len(dirty_values) > 1:
            metadata["git_dirty_values"] = sorted(dirty_values)
        fingerprints = {
            str(row.get("git_worktree_diff_sha256", "")).strip()
            for row in rows
            if row.get("git_worktree_diff_sha256") is not None
            and str(row.get("git_worktree_diff_sha256", "")).strip()
        }
        if len(fingerprints) == 1:
            metadata["git_worktree_diff_sha256"] = next(iter(fingerprints))
        elif len(fingerprints) > 1:
            metadata["git_worktree_diff_sha256_values"] = sorted(fingerprints)
        return rows, metadata

    if not isinstance(payload, dict):
        raise ValueError("Invalid retrieval results JSON: expected a list or an object bundle")

    selected = split
    if selected == "auto":
        if isinstance(payload.get("stateful"), list):
            selected = "stateful"
        else:
            raise ValueError("Bundle JSON must contain stateful list")

    rows_raw = payload.get(selected)
    if not isinstance(rows_raw, list):
        raise ValueError(f"Bundle JSON has no '{selected}' result list")
    rows = [row for row in rows_raw if isinstance(row, dict)]
    if len(rows) != len(rows_raw):
        raise ValueError(f"Bundle JSON '{selected}' entries must be objects")

    metadata = {
        "selected_split": selected,
        "mode": str(payload.get("mode", "")),
        "scope": str(payload.get("scope", "")),
    }
    payload_commit = (
        str(payload.get("git_commit", "")).strip()
        if payload.get("git_commit") is not None
        else ""
    )
    if payload_commit:
        metadata["git_commit"] = payload_commit
    if isinstance(payload.get("git_dirty"), bool):
        metadata["git_dirty"] = bool(payload["git_dirty"])
    payload_fingerprint = str(
        payload.get("git_worktree_diff_sha256", "") or ""
    ).strip()
    if payload_fingerprint:
        metadata["git_worktree_diff_sha256"] = payload_fingerprint
    return rows, metadata


def _rows_to_results(rows: Iterable[dict[str, Any]]) -> list[RetrievalResult]:
    results: list[RetrievalResult] = []
    for row in rows:
        results.append(
            RetrievalResult(
                query_id=str(row.get("query_id", "")),
                doc_id=str(row.get("doc_id", row.get("document_id", ""))),
                question=str(row.get("question", "")),
                retrieved_chunk_ids=[str(x) for x in row.get("retrieved_chunk_ids", row.get("ordered_retrieved_chunk_ids", []))],
                retrieved_indices=[int(x) for x in row.get("retrieved_indices", [])],
                scores=[float(x) for x in row.get("scores", [])],
                gold_chunk_ids=[str(x) for x in row.get("relevant_chunk_ids", row.get("gold_chunk_ids", row.get("gold", [])))],
                silver_chunk_ids=[str(x) for x in row.get("silver_chunk_ids", row.get("silver", []))],
                silver_chunk_groups=_normalize_chunk_groups(row.get("evidence_groups", row.get("silver_chunk_groups", []))),
            )
        )
    return results


def load_retrieval_results_json_with_metadata(path: str | Path, *, split: str = "auto") -> tuple[list[RetrievalResult], dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    rows, metadata = _extract_rows(payload, split=split)
    return _rows_to_results(rows), metadata


def load_retrieval_results_json(path: str | Path, *, split: str = "auto") -> list[RetrievalResult]:
    rows, _ = load_retrieval_results_json_with_metadata(path, split=split)
    return rows


def write_eval_report_json(path: str | Path, report: RetrievalEvalReport) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
    return str(p)
