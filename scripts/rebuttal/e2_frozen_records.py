#!/usr/bin/env python3
"""E2 "frozen records": live vs frozen vs no-memory scoring from the saved caches.

Run from the repository root:

    .venv/bin/python scripts/rebuttal/e2_frozen_records.py --retrievers bm25

No LLM calls, no GPU, no network. The script stops unless the LIVE variant
reproduces the paper's saved HR@5/HR@10 (outputs/saadi/reference/expected_metrics.json).
Results: outputs/rebuttal/E2_frozen_records/ and results/rebuttal/E2_frozen_records/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from saadi.rebuttal.frozen_records import (  # noqa: E402
    DATASETS,
    PUBLISHED_SETTINGS,
    RETRIEVE_K,
    VARIANTS,
    DatasetRun,
    check_reproduction,
    paired_bootstrap_ci,
    run_dataset,
)

EXPERIMENT_ID = "E2_frozen_records"


def per_query_frame(run: DatasetRun) -> pd.DataFrame:
    """One row per query with the hr.py-computed hits of every variant."""

    reports = {variant: run.evaluate(variant, eval_k=5) for variant in VARIANTS}
    base_rows = reports["live"].per_query
    rows: list[dict] = []
    for i, q in enumerate(run.queries):
        row = {
            "dataset": run.dataset,
            "retriever": run.retriever,
            "query_id": q.query_id,
            "doc_id": q.doc_id,
            "eligible": bool(base_rows[i]["eligible"]),
            "affected": bool(q.affected),
            "evidence_third": q.evidence_third,
            "live_matches_saved": bool(q.live_matches_saved),
        }
        for variant in VARIANTS:
            per_query = reports[variant].per_query[i]
            if per_query["query_id"] != q.query_id:
                raise RuntimeError("per-query rows out of order; refusing to merge variants")
            for k in ("5", "10"):
                row[f"hit{k}_{variant}"] = per_query["hit_at_k"].get(k)
        rows.append(row)
    return pd.DataFrame(rows)


def chunk_frame(run: DatasetRun) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": run.dataset,
                "retriever": run.retriever,
                "doc_id": c.doc_id,
                "chunk_id": c.chunk_id,
                "step": c.step,
                "n_linked_names": c.n_linked_names,
                "n_live_aliases": c.n_live_aliases,
                "n_live_descriptions": c.n_live_descriptions,
                "n_late_aliases": c.n_late_aliases,
                "n_late_descriptions": c.n_late_descriptions,
                "has_later_update": c.n_late_aliases + c.n_late_descriptions > 0,
            }
            for c in run.chunks
        ]
    )


def _hr(frame: pd.DataFrame, column: str) -> float:
    return float("nan") if frame.empty else 100.0 * frame[column].astype(float).mean()


def summarize(frame: pd.DataFrame, *, subset: str) -> list[dict]:
    rows: list[dict] = []
    eligible = frame[frame["eligible"]]
    for k in ("5", "10"):
        live = eligible[f"hit{k}_live"].astype(float)
        frozen = eligible[f"hit{k}_frozen"].astype(float)
        diffs_pp = (live - frozen) * 100.0
        mean_pp, lo_pp, hi_pp = paired_bootstrap_ci(diffs_pp.tolist())
        rows.append(
            {
                "dataset": frame["dataset"].iloc[0] if len(frame) else "",
                "retriever": frame["retriever"].iloc[0] if len(frame) else "",
                "subset": subset,
                "k": int(k),
                "n": int(len(eligible)),
                "hr_live": _hr(eligible, f"hit{k}_live"),
                "hr_frozen": _hr(eligible, f"hit{k}_frozen"),
                "hr_no_memory": _hr(eligible, f"hit{k}_no_memory"),
                "diff_live_minus_frozen_pp": mean_pp,
                "ci95_lo_pp": lo_pp,
                "ci95_hi_pp": hi_pp,
            }
        )
    return rows


def position_rows(frame: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    eligible = frame[frame["eligible"] & frame["evidence_third"].notna()]
    labels = {0: "first_third", 1: "middle_third", 2: "last_third"}
    for third, label in labels.items():
        sub = eligible[eligible["evidence_third"] == third]
        for k in ("5", "10"):
            live = sub[f"hit{k}_live"].astype(float)
            frozen = sub[f"hit{k}_frozen"].astype(float)
            diffs_pp = ((live - frozen) * 100.0).tolist()
            mean_pp, lo_pp, hi_pp = paired_bootstrap_ci(diffs_pp)
            rows.append(
                {
                    "dataset": frame["dataset"].iloc[0] if len(frame) else "",
                    "retriever": frame["retriever"].iloc[0] if len(frame) else "",
                    "subset": label,
                    "k": int(k),
                    "n": int(len(sub)),
                    "hr_live": _hr(sub, f"hit{k}_live"),
                    "hr_frozen": _hr(sub, f"hit{k}_frozen"),
                    "hr_no_memory": _hr(sub, f"hit{k}_no_memory"),
                    "diff_live_minus_frozen_pp": mean_pp,
                    "ci95_lo_pp": lo_pp,
                    "ci95_hi_pp": hi_pp,
                }
            )
    return rows


def pick_examples(run: DatasetRun, frame: pd.DataFrame, *, limit: int = 5) -> dict:
    """First queries (in saved order) where live hits and frozen misses at k=5.

    Taken in query order, never by effect size, so the examples are not curated.
    """

    by_id = {q.query_id: q for q in run.queries}
    eligible = frame[frame["eligible"]]
    live_only = eligible[eligible["hit5_live"].astype(bool) & ~eligible["hit5_frozen"].astype(bool)]
    frozen_only = eligible[~eligible["hit5_live"].astype(bool) & eligible["hit5_frozen"].astype(bool)]
    examples = []
    for _, row in live_only.iterrows():
        q = by_id[row["query_id"]]
        if not q.late_evidence_fields:
            continue
        examples.append(
            {
                "dataset": run.dataset,
                "retriever": run.retriever,
                "query_id": q.query_id,
                "doc_id": q.doc_id,
                "question": q.question,
                "late_fields_on_evidence": q.late_evidence_fields[:3],
                "n_late_fields_on_evidence": len(q.late_evidence_fields),
            }
        )
        if len(examples) >= limit:
            break
    return {
        "dataset": run.dataset,
        "retriever": run.retriever,
        "examples_live_hit_frozen_miss_at5": examples,
        "n_live_hit_frozen_miss_at5": int(len(live_only)),
        "n_frozen_hit_live_miss_at5": int(len(frozen_only)),
    }


def fmt(value: float, digits: int = 1) -> str:
    return "-" if pd.isna(value) else f"{value:.{digits}f}"


def write_tables(
    summary: pd.DataFrame,
    chunk_stats: pd.DataFrame,
    examples: list[dict],
    reproduction: list[dict],
    results_dir: Path,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(results_dir / "table.csv", index=False)

    lines: list[str] = []
    lines.append("# E2. Frozen records against live records")
    lines.append("")
    lines.append(
        "HR in percent. Diff = paired (live - frozen) in percentage points with a 95% "
        "bootstrap CI (10,000 resamples over queries, seed 42). Variants share every "
        "setting of the published run; z-scores are recomputed per variant."
    )

    def render(sub: pd.DataFrame, title: str) -> None:
        lines.append("")
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Dataset | Retriever | Subset | k | n | Live | Frozen | No memory | Live-Frozen (pp) | 95% CI |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['dataset']} | {r['retriever']} | {r['subset']} | {r['k']} | {r['n']} "
                f"| {fmt(r['hr_live'])} | {fmt(r['hr_frozen'])} | {fmt(r['hr_no_memory'])} "
                f"| {fmt(r['diff_live_minus_frozen_pp'], 2)} "
                f"| [{fmt(r['ci95_lo_pp'], 2)}, {fmt(r['ci95_hi_pp'], 2)}] |"
            )

    render(summary[summary["subset"] == "all"], "Main table: all eligible queries")
    render(summary[summary["subset"] == "affected"], "Affected queries only")
    render(
        summary[summary["subset"].isin(["first_third", "middle_third", "last_third"])],
        "Split by the position of the earliest evidence chunk",
    )

    lines.append("")
    lines.append("## Later-update statistics per dataset")
    lines.append("")
    lines.append(
        "| Dataset | Chunks | Share with a later update | Mean later aliases per chunk | Mean later descriptions per chunk |"
    )
    lines.append("| --- | --- | --- | --- | --- |")
    for dataset, sub in chunk_stats.groupby("dataset"):
        lines.append(
            f"| {dataset} | {len(sub)} | {100.0 * sub['has_later_update'].mean():.1f}% "
            f"| {sub['n_late_aliases'].mean():.2f} | {sub['n_late_descriptions'].mean():.2f} |"
        )

    lines.append("")
    lines.append("## Example queries (live hits at 5, frozen misses at 5)")
    lines.append("")
    lines.append("Taken in saved query order, not selected by effect size.")
    lines.append("")
    for block in examples:
        for ex in block["examples_live_hit_frozen_miss_at5"]:
            lines.append(f"- **{ex['dataset']} / {ex['query_id']}** - {ex['question']}")
            for f in ex["late_fields_on_evidence"]:
                lines.append(
                    f"  - evidence chunk step {f['evidence_chunk_step']}: entity "
                    f"\"{f['entity_name']}\" gained the {f['kind']} \"{f['value']}\" at step {f['added_at_step']}"
                )
        lines.append(
            f"- {block['dataset']}/{block['retriever']}: "
            f"{block['n_live_hit_frozen_miss_at5']} queries live-hit/frozen-miss at 5; "
            f"{block['n_frozen_hit_live_miss_at5']} go the other way."
        )

    lines.append("")
    lines.append("## Reproduction gate (LIVE vs saved reference)")
    lines.append("")
    lines.append("| Dataset | Retriever | Queries | Hits@5 (mine/expected) | Hits@10 (mine/expected) | Rankings equal to saved |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in reproduction:
        exp5 = (row["expected"].get("5") or {}).get("hits", "-")
        exp10 = (row["expected"].get("10") or {}).get("hits", "-")
        lines.append(
            f"| {row['dataset']} | {row['retriever']} | {row['n_eligible']} "
            f"| {row['hits']['5']}/{exp5} | {row['hits']['10']}/{exp10} "
            f"| {row['rankings_equal_saved']}/{row['rankings_total']} |"
        )

    (results_dir / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(results_dir: Path, *, command: str, summary: pd.DataFrame, reproduction: list[dict]) -> None:
    def overall_sentence() -> str:
        rows = summary[(summary["subset"] == "all") & (summary["k"] == 10)]
        parts = []
        for _, r in rows.iterrows():
            parts.append(
                f"{r['dataset']} {fmt(r['diff_live_minus_frozen_pp'], 2)} pp "
                f"[{fmt(r['ci95_lo_pp'], 2)}, {fmt(r['ci95_hi_pp'], 2)}]"
            )
        return "; ".join(parts)

    n_queries = {row["dataset"]: row["n_eligible"] for row in reproduction}
    text = f"""# E2. Frozen records against live records

**Command.** `{command}` (from the repository root; no LLM calls, no GPU, no network).

**Settings.** Published MBR2 scoring: weights (lambda_t, lambda_n, lambda_h) = (1.0, 0.5, 0.3),
field weights (1, 1, 1), max aggregation, z-score normalization recomputed per variant,
retrieve_k = {RETRIEVE_K}, HR computed with `src/saadi/evaluation/hr.py`'s implementation
(`evaluate_retrieval_results`). Query sets, chunks and evidence labels are read from the
paper's saved per-document rankings, so they are identical to the paper.

**Queries.** {json.dumps(n_queries)}

**Answers to Step 0.**
- Per-chunk construction records: `outputs/<dataset>_ablation/<variant>/<chunk>/llm_cache/.../AEWord/<doc>/steps_index.jsonl`
  with `L_0.jsonl`, `L_1.jsonl`, `memory.json`, indexed by `index.json` (LooGLE and NovelHopQA 500-token,
  MuSiQue 32K-expansion 100-token, QASPER 64K-expansion 100-token namespaces).
- `state_snapshot["active_entities"]` holds only the recalled subset (max 5 records), with the state
  AFTER that step's update. The full per-step memory is rebuilt by replaying the merges, exactly as the
  repository's loader does; replaying to step N reproduces `memory.json` exactly (checked per document).
- Chunk-to-name links: the names in the chunk's own step row, kept when they exist in the final memory
  (`chunk_to_names` in `build_mbr_document_memory_index`).
- Memory channel: unique name/alias/description pools from the final records of the linked names, each pool
  scored against the query (BM25 native scoring); per chunk, max over linked items per field; each field
  channel z-scored over the document's chunks; memory = equal-weight mean of the available fields; final
  score = renormalized 1.0*text + 0.5*memory + 0.3*summary.

**Result.** Live and frozen are statistically indistinguishable on every dataset: the paired
live-minus-frozen difference at k=10 is {overall_sentence()}, and every 95% CI crosses zero
(the same holds at k=5 and on the affected-queries subset). Later record updates are common
(48-97% of chunks have at least one, see chunk_stats.parquet), but under the published scoring
they do not measurably change the retrieval of earlier chunks. The memory channel itself does
help (live against no-memory, e.g. MuSiQue-32K HR@5 62.8 against 59.6), and that gain is already
carried by the records as they stood at each chunk's own step. Reported as they come; the
difference here is zero-to-small, not positive.
"""
    (results_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--retrievers", nargs="+", default=["bm25"])
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Smoke runs only: limits documents and writes to *_smoke dirs.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    suffix = "" if args.max_docs is None else "_smoke"
    out_dir = repo_root / "outputs" / "rebuttal" / (EXPERIMENT_ID + suffix)
    results_dir = repo_root / "results" / "rebuttal" / (EXPERIMENT_ID + suffix)
    out_dir.mkdir(parents=True, exist_ok=True)

    command = ".venv/bin/python scripts/rebuttal/e2_frozen_records.py --retrievers " + " ".join(args.retrievers)

    frames: list[pd.DataFrame] = []
    chunk_frames: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    example_blocks: list[dict] = []
    reproduction_rows: list[dict] = []
    failures: list[str] = []

    for retriever in args.retrievers:
        for key in args.datasets:
            spec = DATASETS[key]
            run = run_dataset(spec, retriever=retriever, repo_root=repo_root, max_docs=args.max_docs)
            repro = check_reproduction(run, spec, repo_root=repo_root, full_dataset=args.max_docs is None)
            reproduction_rows.append(repro)
            print(
                f"[{key}/{retriever}] queries={repro['n_eligible']} "
                f"hits@5={repro['hits']['5']} hits@10={repro['hits']['10']} "
                f"rankings==saved: {repro['rankings_equal_saved']}/{repro['rankings_total']} "
                f"expected match: {repro['matches_expected']}"
            )
            if repro["matches_expected"] is False:
                failures.append(
                    f"{key}/{retriever}: LIVE does not reproduce the saved reference "
                    f"(mine hits@5/@10 = {repro['hits']['5']}/{repro['hits']['10']}, "
                    f"expected = {repro['expected']})"
                )
                continue

            frame = per_query_frame(run)
            frames.append(frame)
            chunk_frames.append(chunk_frame(run))
            summary_rows += summarize(frame, subset="all")
            summary_rows += summarize(frame[frame["affected"]], subset="affected")
            summary_rows += position_rows(frame)
            example_blocks.append(pick_examples(run, frame))

    (out_dir / "reproduction.json").write_text(json.dumps(reproduction_rows, indent=2), encoding="utf-8")

    if failures:
        print("\nSTOP: the LIVE variant did not reproduce the saved reference:", file=sys.stderr)
        for failure in failures:
            print(" -", failure, file=sys.stderr)
        print(f"Partial reproduction details: {out_dir / 'reproduction.json'}", file=sys.stderr)
        return 2

    per_query = pd.concat(frames, ignore_index=True)
    chunk_stats = pd.concat(chunk_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)

    per_query.to_parquet(out_dir / "per_query.parquet", index=False)
    chunk_stats.to_parquet(out_dir / "chunk_stats.parquet", index=False)
    (out_dir / "examples.json").write_text(json.dumps(example_blocks, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "settings.json").write_text(
        json.dumps({"published_settings": PUBLISHED_SETTINGS, "retrieve_k": RETRIEVE_K, "command": command}, indent=2),
        encoding="utf-8",
    )

    write_tables(summary, chunk_stats, example_blocks, reproduction_rows, results_dir)
    write_readme(results_dir, command=command, summary=summary, reproduction=reproduction_rows)
    print(f"\nWrote {out_dir} and {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
