#!/usr/bin/env python3
"""Table 1 with SAADI's held-out fusion weights (ICLR rebuttal, follow-up to C2).

Re-scoring only: no index is rebuilt, no LLM is called, no GPU is used. Everything comes from C2's saved
per-query grid arrays (outputs/rebuttal/C2_heldout_fusion/grid/<job>.npz, written by C2's analysis) and the
paper's Table 1 (results/tables/main/retrieval_comparison/table.csv). C2's code is imported, never edited:
  - hit rule and grid hit ranks: saadi.rebuttal.heldout_fusion (load_job_grid, make_grid);
  - held-out weights: heldout_fusion.cross_validated (5-fold CV grouped by document, seed 42, highest HR@10);
  - bootstrap: saadi.rebuttal.frozen_records.paired_bootstrap_ci (10,000 resamples over queries, seed 42).

Steps
  0. Pairs: every (dataset, retriever) with a C2 DONE marker and a grid array. A retriever is included when
     all four of its datasets are available; otherwise it is listed as left out, with the reason.
  1. Check: SAADI with the published weights must print the same value as Table 1 (HR@5 and HR@10, one
     decimal) in every cell. A difference in a pair that passed C2's strict reproduction gate stops the run
     (exit code 1, the report lists the differences). Pairs for which C2 accepted embedding drift (Jina, Qwen)
     are expected to differ by up to 0.5 points; their differences are listed and the run continues.
  2. Table 1 twice: SAADI with the published weights and SAADI with the held-out weights (both recomputed from
     the same C2 arrays); every other method keeps its Table 1 value.
  3. SAADI's rank among the methods of its retriever in every cell: 1 + number of methods whose Table 1 value is
     strictly higher than SAADI's unrounded value (the rule behind the paper's 'first or second in 34 of 40 cells');
     mean rank, first, first or second; overall, per dataset and per retriever.
  4. Held-out SAADI minus Vanilla per pair at HR@5 and HR@10 with a 95% paired bootstrap interval. Vanilla is
     the text channel alone (lambda_M = lambda_S = 0) on the same chunks and queries.

Usage (from the repository root; experiments/rebuttal/c2_table1/run.sh does this):
    python experiments/rebuttal/c2_table1/scripts/c2_table1.py
Writes results/rebuttal/c2_table1/{REPORT.md,results.json,table1_published.csv,table1_heldout.csv}.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

from saadi.rebuttal import heldout_fusion as hf  # noqa: E402
from saadi.rebuttal.frozen_records import paired_bootstrap_ci  # noqa: E402

EXPERIMENT_ID = "c2_table1"
TABLE1_PATH = Path("results") / "tables" / "main" / "retrieval_comparison" / "table.csv"
CUTOFFS = (5, 10)
SAADI_METHOD = "Saadi"
VANILLA_METHOD = "Vanilla"
VANILLA_POINT = (0.0, 0.0, 1.0, 1.0)  # text channel only; the field weights have no effect without memory channels
PRINT_TOL = 0.05 + 1e-9  # a value prints as the Table 1 entry when it is within half of the last printed digit


def round1(x: float) -> float:
    """One decimal, halves rounded up (the precision Table 1 prints)."""
    return math.floor(x * 10.0 + 0.5 + 1e-9) / 10.0


def git_commit(root: Path) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:  # pragma: no cover - depends on the environment
        return f"<git failed: {exc}>"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def load_table1(root: Path) -> list[dict[str, Any]]:
    with (root / TABLE1_PATH).open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def table1_col(dataset: str, k: int) -> str:
    return f"{hf.DATASET_DISPLAY[dataset]} HR@{k} (%)"


def load_c2_analysis(paths: hf.C2Paths) -> dict[str, Any] | None:
    p = paths.results_dir / "analysis.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def discover_pairs(paths: hf.C2Paths) -> tuple[list[str], dict[str, list[str]], dict[tuple[str, str], Path]]:
    """Retrievers with all four datasets available, reasons for the others, and the grid array of every used pair."""

    included: list[str] = []
    left_out: dict[str, list[str]] = {}
    arrays: dict[tuple[str, str], Path] = {}
    for r in hf.RETRIEVERS:
        reasons = []
        found: dict[tuple[str, str], Path] = {}
        for d in hf.DATASET_KEYS:
            job = hf.job_name(d, r)
            npz = paths.grid_dir / f"{job}.npz"
            if not paths.done_marker(job).exists():
                reasons.append(f"{job}: no C2 DONE marker")
            elif not npz.exists():
                reasons.append(f"{job}: no grid array (run C2's --analysis-only with this retriever)")
            elif paths.check_path(job).exists() and npz.stat().st_mtime < paths.check_path(job).stat().st_mtime:
                reasons.append(f"{job}: grid array older than the job's streams (rerun C2's --analysis-only)")
            else:
                found[(d, r)] = npz
        if reasons:
            left_out[r] = reasons
        else:
            included.append(r)
            arrays.update(found)
    return included, left_out, arrays


def drift_accepted(paths: hf.C2Paths, job: str) -> bool:
    p = paths.check_path(job)
    return bool(json.loads(p.read_text(encoding="utf-8")).get("drift_accepted")) if p.exists() else False


# ---------------------------------------------------------------------------
# Per pair: published, held-out and Vanilla hit ranks
# ---------------------------------------------------------------------------


def score_pair(npz: Path, dataset: str, retriever: str, grid: hf.Grid, n_folds: int | None) -> dict[str, Any]:
    raw = np.load(npz, allow_pickle=False)
    folds_all = raw["fold"][raw["eligible"].astype(bool)]
    k = int(n_folds) if n_folds else len(np.unique(folds_all))
    jg = hf.load_job_grid(npz, dataset, retriever, k)
    held, chosen = hf.cross_validated(jg, grid, np.arange(len(grid)))
    published = jg.hit_rank[:, grid.published_index]
    vanilla = jg.hit_rank[:, grid.index_of(VANILLA_POINT)]
    out: dict[str, Any] = {
        "dataset": dataset,
        "retriever": retriever,
        "n_queries": int(len(jg.fold)),
        "n_docs": int(len(set(jg.doc_id.tolist()))),
        "n_folds": k,
        "chosen_points": [list(grid.points[i]) for i in chosen],
    }
    for name, ranks in (("published", published), ("heldout", held), ("vanilla", vanilla)):
        for c in CUTOFFS:
            out[f"{name}_hr{c}"] = 100.0 * float(np.mean(ranks < c))
    for c in CUTOFFS:
        diff = 100.0 * ((held < c).astype(np.float64) - (vanilla < c).astype(np.float64))
        m, lo, hi = paired_bootstrap_ci(diff, n_resamples=hf.N_BOOT, seed=hf.SEED)
        out[f"heldout_minus_vanilla_hr{c}"] = m
        out[f"heldout_minus_vanilla_hr{c}_ci"] = [lo, hi]
    return out


# ---------------------------------------------------------------------------
# Check 1, Table 1 versions, ranks
# ---------------------------------------------------------------------------


def check_published(pairs: dict[tuple[str, str], dict[str, Any]], table1: list[dict[str, Any]], paths: hf.C2Paths) -> dict[str, Any]:
    cells = []
    for (d, r), p in pairs.items():
        saadi = next(t for t in table1 if t["Retriever"] == hf.RETRIEVER_DISPLAY[r] and t["Method"] == SAADI_METHOD)
        drift = drift_accepted(paths, hf.job_name(d, r))
        for c in CUTOFFS:
            paper = float(saadi[table1_col(d, c)])
            ours = p[f"published_hr{c}"]
            cells.append(
                {
                    "dataset": d,
                    "retriever": r,
                    "cutoff": c,
                    "table1": paper,
                    "recomputed": ours,
                    "recomputed_printed": round1(ours),
                    "difference": ours - paper,
                    "matches": abs(ours - paper) <= PRINT_TOL,
                    "drift_accepted_in_c2": drift,
                }
            )
    strict_fail = [c for c in cells if not c["matches"] and not c["drift_accepted_in_c2"]]
    drift_diff = [c for c in cells if not c["matches"] and c["drift_accepted_in_c2"]]
    return {
        "cells": cells,
        "n_cells": len(cells),
        "n_match": sum(1 for c in cells if c["matches"]),
        "strict_differences": strict_fail,
        "drift_differences": drift_diff,
        "passed": not strict_fail,
    }


def build_tables(
    pairs: dict[tuple[str, str], dict[str, Any]], table1: list[dict[str, Any]], retrievers: list[str]
) -> dict[str, list[dict[str, Any]]]:
    """Table 1 rows of the included retrievers, with SAADI replaced by the recomputed published / held-out values."""

    versions: dict[str, list[dict[str, Any]]] = {"published": [], "heldout": []}
    for version in versions:
        for r in retrievers:
            for t in table1:
                if t["Retriever"] != hf.RETRIEVER_DISPLAY[r]:
                    continue
                row: dict[str, Any] = {"Retriever": t["Retriever"], "Method": t["Method"]}
                for d in hf.DATASET_KEYS:
                    for c in CUTOFFS:
                        col = table1_col(d, c)
                        # SAADI unrounded, as the paper's count compares it; the report prints one decimal.
                        row[col] = pairs[(d, r)][f"{version}_hr{c}"] if t["Method"] == SAADI_METHOD else float(t[col])
                versions[version].append(row)
    return versions


def rank_cells(tables: dict[str, list[dict[str, Any]]], retrievers: list[str]) -> list[dict[str, Any]]:
    cells = []
    for r in retrievers:
        rdisp = hf.RETRIEVER_DISPLAY[r]
        for d in hf.DATASET_KEYS:
            for c in CUTOFFS:
                col = table1_col(d, c)
                cell: dict[str, Any] = {"retriever": r, "dataset": d, "cutoff": c}
                for version, rows in tables.items():
                    methods = [t for t in rows if t["Retriever"] == rdisp]
                    saadi = next(t[col] for t in methods if t["Method"] == SAADI_METHOD)
                    others = [(t["Method"], t[col]) for t in methods if t["Method"] != SAADI_METHOD]
                    best = max(others, key=lambda x: x[1])
                    cell[f"saadi_{version}"] = saadi
                    cell[f"rank_{version}"] = 1 + sum(1 for _, v in others if v > saadi + 1e-9)
                    cell[f"ties_{version}"] = sum(1 for _, v in others if abs(v - saadi) <= 1e-9)
                    cell["n_methods"] = len(methods)
                    cell[f"best_other_{version}"] = {"method": best[0], "value": best[1]}
                cells.append(cell)
    return cells


def rank_summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    def block(sel: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {"n_cells": len(sel)}
        for v in ("published", "heldout"):
            ranks = [c[f"rank_{v}"] for c in sel]
            out[v] = {
                "mean_rank": float(np.mean(ranks)) if ranks else None,
                "first": sum(1 for x in ranks if x == 1),
                "first_or_second": sum(1 for x in ranks if x <= 2),
            }
        out["cells_rank_changed"] = sum(1 for c in sel if c["rank_published"] != c["rank_heldout"])
        return out

    return {
        "all": block(cells),
        "by_dataset": {d: block([c for c in cells if c["dataset"] == d]) for d in hf.DATASET_KEYS},
        "by_retriever": {r: block([c for c in cells if c["retriever"] == r]) for r in dict.fromkeys(c["retriever"] for c in cells)},
        "by_cutoff": {str(k): block([c for c in cells if c["cutoff"] == k]) for k in CUTOFFS},
    }


def vanilla_summary(pairs: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for c in CUTOFFS:
        rows = list(pairs.values())
        out[f"hr{c}"] = {
            "pairs": len(rows),
            "positive": sum(1 for p in rows if p[f"heldout_minus_vanilla_hr{c}"] > 0),
            "ci_above_zero": sum(1 for p in rows if p[f"heldout_minus_vanilla_hr{c}_ci"][0] > 0),
            "ci_below_zero": sum(1 for p in rows if p[f"heldout_minus_vanilla_hr{c}_ci"][1] < 0),
            "mean_difference": float(np.mean([p[f"heldout_minus_vanilla_hr{c}"] for p in rows])) if rows else None,
        }
    return out


def c2_consistency(pairs: dict[tuple[str, str], dict[str, Any]], analysis: dict[str, Any] | None) -> dict[str, Any]:
    """The held-out and Vanilla values must equal C2's analysis.json (same arrays, same selection)."""

    if analysis is None:
        return {"compared": 0, "differences": [], "note": "C2 analysis.json not found"}
    rows = {(r["dataset"], r["retriever"]): r for r in analysis.get("per_job", [])}
    diffs = []
    compared = 0
    for key, p in pairs.items():
        row = rows.get(key)
        if row is None:
            continue
        for ours, theirs in (("heldout", "a1"), ("vanilla", "vanilla"), ("published", "published")):
            for c in CUTOFFS:
                compared += 1
                if abs(p[f"{ours}_hr{c}"] - row[f"{theirs}_hr{c}"]) > 1e-9:
                    diffs.append(
                        {"pair": hf.job_name(*key), "value": f"{ours}_hr{c}", "here": p[f"{ours}_hr{c}"], "c2": row[f"{theirs}_hr{c}"]}
                    )
        if [list(x) for x in row.get("chosen_a1", [])] != p["chosen_points"]:
            diffs.append({"pair": hf.job_name(*key), "value": "chosen points", "here": p["chosen_points"], "c2": row.get("chosen_a1")})
    return {"compared": compared, "differences": diffs, "pairs_in_c2": sum(1 for k in pairs if k in rows)}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _f(x: float, nd: int = 1) -> str:
    return f"{x:.{nd}f}"


def _s(x: float, nd: int = 1) -> str:
    return f"{x:+.{nd}f}"


def _pt(p: list[float]) -> str:
    return f"({p[0]:g}, {p[1]:g} | {p[2]:g}, {p[3]:g})"


def write_table_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def failure_report(res: dict[str, Any]) -> list[str]:
    ck = res["check_published"]
    L = ["# Table 1 with held-out fusion weights: STOPPED at the check", ""]
    L.append(
        f"SAADI with the published weights does not reproduce Table 1 in {len(ck['strict_differences'])} cell(s) of pairs that passed "
        "C2's strict reproduction gate, so the tables and ranks were not computed."
    )
    L.append("")
    L.append("| Pair | HR@k | Table 1 | Recomputed | Difference |")
    L.append("| --- | --- | --- | --- | --- |")
    for c in ck["strict_differences"]:
        L.append(
            f"| {hf.job_name(c['dataset'], c['retriever'])} | {c['cutoff']} | {_f(c['table1'])} | {_f(c['recomputed'], 2)} | {_s(c['difference'], 2)} |"
        )
    L.append("")
    L.extend(setup_lines(res))
    return L


def setup_lines(res: dict[str, Any]) -> list[str]:
    L = ["## Setup", ""]
    inc = ", ".join(hf.RETRIEVER_DISPLAY[r] for r in res["retrievers"]) or "none"
    L.append(f"- Retrievers included: {inc}.")
    for r, reasons in res["retrievers_left_out"].items():
        L.append(f"- Not included: {hf.RETRIEVER_DISPLAY[r]} ({'; '.join(reasons)}).")
    L.append(
        f"- Inputs: C2 grid arrays `{res['inputs']['grid_dir']}/<dataset>__<retriever>.npz` (per-query hit ranks for all {res['grid_points']} "
        f"weight settings) and Table 1 `{res['inputs']['table1']}`. Nothing is rebuilt; no LLM or GPU is used."
    )
    L.append(
        "- Held-out weights: C2's protocol (`heldout_fusion.cross_validated`): for each dataset and retriever separately, 5 folds grouped "
        "by document (seed 42); in each fold the weights with the highest HR@10 on the other four folds score the held-out fold."
    )
    L.append(
        "- Rank: 1 + number of Table 1 methods of the same retriever whose value is strictly higher than SAADI's unrounded "
        "recomputed value (the rule behind the paper's count of 34 of 40 cells first or second). Values are printed with one "
        "decimal, so a printed tie (e.g. 14.8 against 14.76) can still rank SAADI below."
    )
    L.append(
        "- Vanilla in part 4: the text channel alone (lambda_M = lambda_S = 0) on the same chunks and queries, so the comparison is "
        "paired. Its value can differ from the Vanilla row of Table 1, which is kept unchanged in the rank tables."
    )
    L.append("- Bootstrap: 95% paired interval over queries, 10,000 resamples, seed 42 (`frozen_records.paired_bootstrap_ci`).")
    L.append(f"- Commit {res['commit']}; host {res['host']}; {res['seconds']:.0f} s.")
    L.append("")
    return L


def report_lines(res: dict[str, Any]) -> list[str]:
    rs = res["ranks"]["all"]
    vs = res["vs_vanilla"]
    n = rs["n_cells"]
    m = vs["hr10"]["pairs"]
    inc = ", ".join(hf.RETRIEVER_DISPLAY[r] for r in res["retrievers"])
    L = ["# Table 1 with held-out fusion weights", ""]
    L.append("## Summary")
    L.append("")
    L.append(
        f"Mean rank: published {_f(rs['published']['mean_rank'], 3)}, held-out {_f(rs['heldout']['mean_rank'], 3)}. "
        f"First or second in {rs['published']['first_or_second']} of {n} cells (published) and {rs['heldout']['first_or_second']} of {n} cells (held-out). "
        f"With held-out weights SAADI improves over Vanilla in {vs['hr10']['positive']} of {m} pairs at HR@10, significantly in {vs['hr10']['ci_above_zero']}; "
        f"at HR@5 in {vs['hr5']['positive']} of {m}, significantly in {vs['hr5']['ci_above_zero']}."
    )
    L.append("")
    L.append(
        f"First in {rs['published']['first']} of {n} cells with the published weights and {rs['heldout']['first']} of {n} with the held-out weights; "
        f"{rs['cells_rank_changed']} cell(s) change rank. Retrievers: {inc}"
        + (
            f"; not included: {', '.join(hf.RETRIEVER_DISPLAY[r] for r in res['retrievers_left_out'])}"
            if res["retrievers_left_out"]
            else ""
        )
        + f" ({len(res['pairs'])} dataset-retriever pairs, {n} cells: HR@5 and HR@10 per pair)."
    )
    ck = res["check_published"]
    L.append("")
    L.append(
        f"Check: SAADI with the published weights prints the Table 1 value in {ck['n_match']} of {ck['n_cells']} cells."
        + (
            f" The {len(ck['drift_differences'])} other cell(s) belong to pairs where C2 accepted embedding drift (listed in part 1); "
            "for them both versions use the recomputed values."
            if ck["drift_differences"]
            else ""
        )
    )
    cc = res["c2_consistency"]
    if cc.get("compared"):
        L.append(
            f"The held-out, published and Vanilla values equal C2's `analysis.json` in {cc['compared'] - len([d for d in cc['differences'] if d['value'] != 'chosen points'])} "
            f"of {cc['compared']} comparisons, and the selected weights match in "
            f"{cc['pairs_in_c2'] - len([d for d in cc['differences'] if d['value'] == 'chosen points'])} of {cc['pairs_in_c2']} pairs."
        )
    L.append("")
    L.extend(setup_lines(res))

    # Part 1
    L.append("## 1. Check: SAADI with the published weights vs Table 1")
    L.append("")
    L.append("| Pair | HR@5 Table 1 / recomputed | HR@10 Table 1 / recomputed | Matches | C2 drift accepted |")
    L.append("| --- | --- | --- | --- | --- |")
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for c in ck["cells"]:
        by_pair.setdefault((c["dataset"], c["retriever"]), []).append(c)
    for (d, r), cs in by_pair.items():
        c5 = next(c for c in cs if c["cutoff"] == 5)
        c10 = next(c for c in cs if c["cutoff"] == 10)
        L.append(
            f"| {hf.DATASET_DISPLAY[d]} / {hf.RETRIEVER_DISPLAY[r]} | {_f(c5['table1'])} / {_f(c5['recomputed'], 2)} | {_f(c10['table1'])} / {_f(c10['recomputed'], 2)} "
            f"| {'yes' if c5['matches'] and c10['matches'] else 'no'} | {'yes' if c5['drift_accepted_in_c2'] else 'no'} |"
        )
    L.append("")

    # Part 2
    L.append("## 2. Table 1 with SAADI's published and held-out weights (HR %, other methods from Table 1)")
    L.append("")
    head = "| Retriever | Method | " + " | ".join(f"{hf.DATASET_DISPLAY[d]} @{c}" for d in hf.DATASET_KEYS for c in CUTOFFS) + " |"
    sep = "| --- | --- | " + " | ".join("---" for _ in hf.DATASET_KEYS for _ in CUTOFFS) + " |"
    L.append(head)
    L.append(sep)
    rank_of = {(c["retriever"], c["dataset"], c["cutoff"]): c for c in res["rank_cells"]}
    disp_to_ret = {v: k for k, v in hf.RETRIEVER_DISPLAY.items()}
    for row in res["tables"]["published"]:
        r = disp_to_ret[row["Retriever"]]
        if row["Method"] == SAADI_METHOD:
            for version, label in (("published", "SAADI, published weights"), ("heldout", "SAADI, held-out weights")):
                vals = []
                for d in hf.DATASET_KEYS:
                    for c in CUTOFFS:
                        cell = rank_of[(r, d, c)]
                        vals.append(f"**{_f(cell[f'saadi_{version}'])}** ({cell[f'rank_{version}']})")
                L.append(f"| {row['Retriever']} | {label} | " + " | ".join(vals) + " |")
        else:
            L.append(
                f"| {row['Retriever']} | {row['Method']} | "
                + " | ".join(_f(row[table1_col(d, c)]) for d in hf.DATASET_KEYS for c in CUTOFFS)
                + " |"
            )
    L.append("")
    L.append("SAADI's rank among the methods of its retriever is in parentheses.")
    L.append("")

    # Part 3
    L.append("## 3. SAADI's position")
    L.append("")
    L.append(
        "| Scope | Cells | Mean rank published / held-out | First published / held-out | First or second published / held-out | Cells that change rank |"
    )
    L.append("| --- | --- | --- | --- | --- | --- |")

    def rank_row(label: str, b: dict[str, Any]) -> str:
        p, h = b["published"], b["heldout"]
        if not b["n_cells"]:
            return f"| {label} | 0 | - | - | - | - |"
        return (
            f"| {label} | {b['n_cells']} | {_f(p['mean_rank'], 3)} / {_f(h['mean_rank'], 3)} | {p['first']} / {h['first']} | "
            f"{p['first_or_second']} / {h['first_or_second']} | {b['cells_rank_changed']} |"
        )

    L.append(rank_row("All cells", rs))
    for d in hf.DATASET_KEYS:
        L.append(rank_row(hf.DATASET_DISPLAY[d], res["ranks"]["by_dataset"][d]))
    for r, b in res["ranks"]["by_retriever"].items():
        L.append(rank_row(f"{hf.RETRIEVER_DISPLAY[r]} ({res['n_methods'][r]} methods)", b))
    for k, b in res["ranks"]["by_cutoff"].items():
        L.append(rank_row(f"HR@{k}", b))
    L.append("")
    L.append("Rank in every cell (published -> held-out; best other method of Table 1):")
    L.append("")
    L.append("| Retriever | Dataset | HR@k | SAADI published | SAADI held-out | Rank published -> held-out | Best other method |")
    L.append("| --- | --- | --- | --- | --- | --- | --- |")
    for c in res["rank_cells"]:
        b = c["best_other_published"]
        mark = " (changed)" if c["rank_published"] != c["rank_heldout"] else ""
        L.append(
            f"| {hf.RETRIEVER_DISPLAY[c['retriever']]} | {hf.DATASET_DISPLAY[c['dataset']]} | {c['cutoff']} | {_f(c['saadi_published'])} | {_f(c['saadi_heldout'])} | "
            f"{c['rank_published']} -> {c['rank_heldout']}{mark} | {b['method']} {_f(b['value'])} |"
        )
    L.append("")

    # Part 4
    L.append("## 4. Held-out SAADI minus Vanilla (text channel on the same chunks), HR points, 95% paired bootstrap")
    L.append("")
    L.append("| Pair | Queries | Vanilla @5 / @10 | Held-out @5 / @10 | Diff @5 [95% CI] | Diff @10 [95% CI] | Table 1 Vanilla @5 / @10 |")
    L.append("| --- | --- | --- | --- | --- | --- | --- |")
    t1 = {(t["Retriever"], t["Method"]): t for t in res["table1_rows"]}
    for (d, r), p in res["pairs"].items():
        tv = t1.get((hf.RETRIEVER_DISPLAY[r], VANILLA_METHOD))
        tvs = f"{_f(float(tv[table1_col(d, 5)]))} / {_f(float(tv[table1_col(d, 10)]))}" if tv else "-"
        ci5, ci10 = p["heldout_minus_vanilla_hr5_ci"], p["heldout_minus_vanilla_hr10_ci"]
        L.append(
            f"| {hf.DATASET_DISPLAY[d]} / {hf.RETRIEVER_DISPLAY[r]} | {p['n_queries']} | {_f(p['vanilla_hr5'])} / {_f(p['vanilla_hr10'])} | "
            f"{_f(p['heldout_hr5'])} / {_f(p['heldout_hr10'])} | {_s(p['heldout_minus_vanilla_hr5'])} [{_s(ci5[0])}, {_s(ci5[1])}] | "
            f"{_s(p['heldout_minus_vanilla_hr10'])} [{_s(ci10[0])}, {_s(ci10[1])}] | {tvs} |"
        )
    L.append("")
    for c in CUTOFFS:
        v = vs[f"hr{c}"]
        L.append(
            f"- HR@{c}: gain positive in {v['positive']} of {v['pairs']} pairs, interval above zero in {v['ci_above_zero']}, below zero in "
            f"{v['ci_below_zero']}; mean gain {_s(v['mean_difference'], 2)} points."
        )
    L.append("")
    L.append("## Selected weights per fold (lambda_M, lambda_S | mu_A, mu_D)")
    L.append("")
    L.append("| Pair | Folds |")
    L.append("| --- | --- |")
    for (d, r), p in res["pairs"].items():
        L.append(f"| {hf.DATASET_DISPLAY[d]} / {hf.RETRIEVER_DISPLAY[r]} | {'; '.join(_pt(x) for x in p['chosen_points'])} |")
    L.append("")
    L.append(
        f"Files: `{res['outputs']['results_json']}` (every number above), `{res['outputs']['table_published']}`, `{res['outputs']['table_heldout']}`."
    )
    return L


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(paths: hf.C2Paths, results_dir: Path, repo_root: Path) -> int:
    t0 = time.perf_counter()
    grid = hf.make_grid()
    table1 = load_table1(repo_root)
    analysis = load_c2_analysis(paths)
    retrievers, left_out, arrays = discover_pairs(paths)
    n_folds_c2 = {w: v.get("n_folds") for w, v in (analysis or {}).get("workers", {}).items()}
    print(f"[{EXPERIMENT_ID}] retrievers included: {retrievers or 'none'}; left out: {list(left_out) or 'none'}", flush=True)

    pairs: dict[tuple[str, str], dict[str, Any]] = {}
    for d in hf.DATASET_KEYS:
        for r in retrievers:
            job = hf.job_name(d, r)
            pairs[(d, r)] = score_pair(arrays[(d, r)], d, r, grid, n_folds_c2.get(job))
            p = pairs[(d, r)]
            print(
                f"[{EXPERIMENT_ID}] {job}: {p['n_queries']} queries; HR@10 published {p['published_hr10']:.2f}, held-out {p['heldout_hr10']:.2f}, "
                f"Vanilla {p['vanilla_hr10']:.2f}",
                flush=True,
            )
    order = sorted(pairs, key=lambda k: (hf.RETRIEVERS.index(k[1]), hf.DATASET_KEYS.index(k[0])))
    pairs = {k: pairs[k] for k in order}

    res: dict[str, Any] = {
        "experiment": EXPERIMENT_ID,
        "commit": git_commit(repo_root),
        "host": platform.node(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "grid_points": len(grid),
        "retrievers": retrievers,
        "retrievers_left_out": left_out,
        "inputs": {
            "grid_dir": hf.rel(paths.grid_dir, repo_root),
            "table1": str(TABLE1_PATH),
            "c2_analysis": hf.rel(paths.results_dir / "analysis.json", repo_root),
        },
        "outputs": {
            "report": hf.rel(results_dir / "REPORT.md", repo_root),
            "results_json": hf.rel(results_dir / "results.json", repo_root),
            "table_published": hf.rel(results_dir / "table1_published.csv", repo_root),
            "table_heldout": hf.rel(results_dir / "table1_heldout.csv", repo_root),
        },
        "table1_rows": [t for t in table1 if t["Retriever"] in {hf.RETRIEVER_DISPLAY[r] for r in retrievers}],
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    if not pairs:
        res["seconds"] = time.perf_counter() - t0
        res["status"] = "no pairs"
        hf.write_json(results_dir / "results.json", res)
        (results_dir / "REPORT.md").write_text(
            "# Table 1 with held-out fusion weights: no input\n\nNo retriever has C2 outputs for all four datasets.\n\n"
            + "\n".join(setup_lines(res))
            + "\n",
            encoding="utf-8",
        )
        print(f"[{EXPERIMENT_ID}] no retriever has C2 grid arrays for all four datasets", flush=True)
        return 1

    res["pairs_list"] = [hf.job_name(*k) for k in pairs]
    res["pairs"] = pairs
    res["check_published"] = check_published(pairs, table1, paths)
    res["c2_consistency"] = c2_consistency(pairs, analysis)
    if not res["check_published"]["passed"]:
        res["seconds"] = time.perf_counter() - t0
        res["status"] = "stopped: published weights do not reproduce Table 1"
        hf.write_json(results_dir / "results.json", _jsonable(res))
        (results_dir / "REPORT.md").write_text("\n".join(failure_report(res)) + "\n", encoding="utf-8")
        for c in res["check_published"]["strict_differences"]:
            print(
                f"[{EXPERIMENT_ID}] CHECK FAILED {hf.job_name(c['dataset'], c['retriever'])} HR@{c['cutoff']}: Table 1 {c['table1']}, recomputed {c['recomputed']:.2f}",
                flush=True,
            )
        return 1

    res["tables"] = build_tables(pairs, table1, retrievers)
    res["rank_cells"] = rank_cells(res["tables"], retrievers)
    res["ranks"] = rank_summary(res["rank_cells"])
    res["n_methods"] = {r: sum(1 for t in table1 if t["Retriever"] == hf.RETRIEVER_DISPLAY[r]) for r in retrievers}
    res["vs_vanilla"] = vanilla_summary(pairs)
    res["seconds"] = time.perf_counter() - t0
    res["status"] = "done"
    write_table_csv(results_dir / "table1_published.csv", res["tables"]["published"])
    write_table_csv(results_dir / "table1_heldout.csv", res["tables"]["heldout"])
    out = _jsonable(res)
    hf.write_json(results_dir / "results.json", out)
    lines = report_lines(out | {"pairs": res["pairs"]})
    (results_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[{EXPERIMENT_ID}] " + lines[4], flush=True)
    return 0


def _jsonable(res: dict[str, Any]) -> dict[str, Any]:
    out = dict(res)
    if isinstance(out.get("pairs"), dict):
        out["pairs"] = {hf.job_name(*k): v for k, v in out["pairs"].items()}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--c2-out", help="C2 outputs folder (default outputs/rebuttal/C2_heldout_fusion)")
    ap.add_argument("--c2-results", help="C2 results folder (default results/rebuttal/C2_heldout_fusion)")
    ap.add_argument("--results-dir", help="where to write (default results/rebuttal/c2_table1)")
    args = ap.parse_args()
    paths = hf.C2Paths(
        REPO_ROOT,
        out_base=Path(args.c2_out).resolve() if args.c2_out else None,
        results_base=Path(args.c2_results).resolve() if args.c2_results else None,
    )
    results_dir = Path(args.results_dir).resolve() if args.results_dir else REPO_ROOT / "results" / "rebuttal" / EXPERIMENT_ID
    return run(paths, results_dir, REPO_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
