"""Evaluate portable ranking records with the same HR implementation as experiments."""
from __future__ import annotations

from .retrieval_metrics import _rows_to_results, evaluate_retrieval_results


def evaluate_rows(rows: list[dict], *, cutoffs=(5, 10), retrieve_k=None) -> list[dict]:
    ids = [row.get("query_id") for row in rows]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("Query IDs must be nonempty and unique")
    for row in rows:
        ranking = row.get("retrieved_chunk_ids", row.get("ordered_retrieved_chunk_ids", []))
        if len(ranking) != len(set(ranking)):
            raise ValueError(f"Duplicate retrieved chunk IDs for {row['query_id']}")
    results = _rows_to_results(rows)
    values = []
    for cutoff in cutoffs:
        report = evaluate_retrieval_results(results, eval_k=cutoff, retrieve_k=retrieve_k,
                                            compute_per_query=False)
        values.append({"cutoff": cutoff, "queries": report.n_examples,
                       "eligible": report.n_union_eligible, "hits": report.n_union_hits,
                       "hr": report.hit_rate})
    return values
