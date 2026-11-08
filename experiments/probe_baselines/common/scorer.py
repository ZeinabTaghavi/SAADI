"""ONE scorer for every system on the persona probe: gold rank, HR@5, HR@10, median rank, 95% CIs, per cell.

Reads the shared ranking format (common/chunk_map.py) for the baselines, and SAADI's
runs from their per-persona files (run 0: outputs/probe_bridge/per_persona.jsonl;
run 1: outputs/method_update/p1_ledger/per_persona.jsonl; run 1b:
outputs/method_update/p1b_ledger/per_persona.jsonl), converted to the same format.
When a per-persona file is absent the run's results/.../table.csv is read instead
(aggregates only: no CI, no paired test) and the report says so.

The probe's HR (experiments/probe_bridge/steps/step07_retrieve_and_score.py): a hit
at k when the gold chunk's 1-based rank is <= k; HR in percent over personas; median
of the gold ranks; intervals by the probe's paired_bootstrap_ci (10,000 resamples,
seed 42) and its exact McNemar test. Those two functions are imported, not copied.

    python common/scorer.py score                     # 06_score
    python common/scorer.py self-test [--run run1]    # reproduce a SAADI run's table from its per-persona file
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chunk_map as CM  # noqa: E402
from load_probe import ROOT, probe  # noqa: E402

SYSTEMS = ("b0_vanilla", "b1_oracle", "b2_contextual", "b3_hipporag", "b4_lqca")
SYSTEM_LABELS = {
    "b0_vanilla": "B0 Vanilla BM25",
    "b1_oracle": "B1 Oracle (form 2 in the gold chunk)",
    "b2_contextual": "B2 Contextual Retrieval",
    "b3_hipporag": "B3 HippoRAG v2",
    "b4_lqca": "B4 LQCA",
    "saadi_run0_live": "SAADI run 0 (published), live",
    "saadi_run1_live": "SAADI run 1 (p1_ledger), live",
    "saadi_run1b_live": "SAADI run 1b (p1b_ledger), live",
}
SAADI_RUNS = {
    "run0": {
        "label": "run0_published",
        "per_persona": ROOT / "outputs" / "probe_bridge" / "per_persona.jsonl",
        "table": ROOT / "results" / "method_update" / "p1_ledger" / "table.csv",
    },
    "run1": {
        "label": "run1_p1_ledger",
        "per_persona": ROOT / "outputs" / "method_update" / "p1_ledger" / "per_persona.jsonl",
        "table": ROOT / "results" / "method_update" / "p1_ledger" / "table.csv",
    },
    "run1b": {
        "label": "run1b_p1b_ledger",
        "per_persona": ROOT / "outputs" / "method_update" / "p1b_ledger" / "per_persona.jsonl",
        "table": ROOT / "results" / "method_update" / "p1b_ledger" / "table.csv",
    },
}
# Local dry runs only: PBL_SAADI_<RUN>_PER_PERSONA points a run at another per-persona file.
import os as _os  # noqa: E402

for _run, _spec in SAADI_RUNS.items():
    _override = _os.environ.get(f"PBL_SAADI_{_run.upper()}_PER_PERSONA")
    if _override:
        _spec["per_persona"] = Path(_override)
SAADI_METHODS = ("vanilla", "no_memory", "frozen", "live")
RESULTS_DIR = Path(__import__("os").environ.get("PBL_RESULTS_DIR", ROOT / "results" / "probe_baselines"))
CELLS_1_3 = (1, 2, 3)
REVEAL_CELLS = (1, 2, 3, 4, 6, 7, 8)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def hit(row: dict, k: int) -> bool:
    return int(row["gold_rank"]) <= int(k)


def hr_ci(rows: list[dict], k: int) -> dict:
    if not rows:
        return {"hr": float("nan"), "lo": float("nan"), "hi": float("nan"), "hits": 0, "n": 0}
    mean, lo, hi = probe.paired_bootstrap_ci([100.0 * hit(r, k) for r in rows])
    return {"hr": mean, "lo": lo, "hi": hi, "hits": sum(hit(r, k) for r in rows), "n": len(rows)}


def median_rank(rows: list[dict]) -> float:
    return float(statistics.median(int(r["gold_rank"]) for r in rows)) if rows else float("nan")


def cell_metrics(rows: list[dict]) -> dict:
    return {"n": len(rows), "hr5": hr_ci(rows, 5), "hr10": hr_ci(rows, 10), "median_rank": median_rank(rows)}


def paired(a: list[dict], b: list[dict], k: int = 10) -> dict:
    """a minus b at HR@k over the personas both have: mean pp, 95% CI, exact McNemar."""

    bb = {r["persona_id"]: r for r in b}
    pairs = [(r, bb[r["persona_id"]]) for r in a if r["persona_id"] in bb]
    if not pairs:
        return {"n": 0}
    diffs = [100.0 * (hit(x, k) - hit(y, k)) for x, y in pairs]
    mean, lo, hi = probe.paired_bootstrap_ci(diffs)
    only_a = sum(1 for x, y in pairs if hit(x, k) and not hit(y, k))
    only_b = sum(1 for x, y in pairs if hit(y, k) and not hit(x, k))
    return {
        "n": len(pairs),
        "mean_pp": mean,
        "ci_lo": lo,
        "ci_hi": hi,
        "a_only": only_a,
        "b_only": only_b,
        "mcnemar_p": probe.mcnemar_exact_p(only_a, only_b),
    }


def by_cell(rows: list[dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        out[int(r["cell"])].append(r)
    return dict(out)


# ---------------------------------------------------------------------------
# SAADI runs in the shared format
# ---------------------------------------------------------------------------


def saadi_rows(per_persona: Path, method: str) -> list[dict]:
    """A SAADI per-persona file (probe step 7 / p1 step 4) as shared-format rows of one method."""

    out = []
    for p in CM.read_jsonl(per_persona):
        out.append(
            {
                "persona_id": p["persona_id"],
                "book_id": p["book_id"],
                "cell": int(p["cell"]),
                "ranked_chunk_ids": list(p.get(f"top3_{method}") or []),  # SAADI saved only its top 3
                "gold_rank": int(p[f"rank_{method}"]),
                "n_chunks": int(p.get("n_chunks") or 0),
                "model_calls": 0,
                "input_tokens": 0,
                "generated_tokens": 0,
                "bridge": bool(p.get("form2_in_that_record_final") and p.get("gold_chunk_linked_to_that_record")),
            }
        )
    return out


def saadi_table(table: Path, run_label: str) -> dict[int, dict]:
    """Aggregates from a results/method_update/*/table.csv (both schemas: p1's long one and p1b's wide one)."""

    out: dict[int, dict] = {}
    if not table.exists():
        return out
    with table.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        if r.get("run") != run_label:
            continue
        cell = int(r["cell"])
        if "method" in r and r.get("method") == "live":
            out[cell] = {"n": int(r["n"]), "hr5": float(r["hr5"]), "hr10": float(r["hr10"]), "median_rank": float(r["median_rank"])}
        elif "hr10_live" in r:
            out[cell] = {
                "n": int(r["n"]),
                "hr5": float(r.get("hr5_live") or "nan"),
                "hr10": float(r["hr10_live"]),
                "median_rank": float(r.get("median_live") or "nan"),
            }
    return out


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_saadi() -> tuple[dict[str, list[dict]], dict[str, dict], list[str]]:
    """{system: rows} for runs with per-persona files, {system: table aggregates} for the others, notes."""

    rows, tables, notes = {}, {}, []
    for run, spec in SAADI_RUNS.items():
        system = f"saadi_{run}_live"
        if spec["per_persona"].exists():
            rows[system] = saadi_rows(spec["per_persona"], "live")
            notes.append(f"{SYSTEM_LABELS[system]}: per persona from {_rel(spec['per_persona'])}")
        else:
            agg = saadi_table(spec["table"], spec["label"])
            if agg:
                tables[system] = agg
                notes.append(
                    f"{SYSTEM_LABELS[system]}: per-persona file absent; aggregates from {_rel(spec['table'])} (no CI, no paired test)"
                )
            else:
                notes.append(f"{SYSTEM_LABELS[system]}: no results found ({_rel(spec['per_persona'])} and {_rel(spec['table'])} absent)")
    return rows, tables, notes


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------


def split_metrics(rows: list[dict], key: str) -> dict:
    out = {}
    for value in (True, False):
        g = [r for r in rows if r.get(key) is value]
        out[str(value).lower()] = {"n": len(g), "hr10": hr_ci(g, 10), "median_rank": median_rank(g)}
    out["unknown"] = sum(1 for r in rows if r.get(key) is None)
    return out


def score(rankings_dir: Path | None = None) -> dict:
    systems: dict[str, list[dict]] = {}
    missing: dict[str, str] = {}
    for system in SYSTEMS:
        rows = CM.read_rankings(system, rankings_dir)
        if len(rows) == 240:
            systems[system] = rows
        elif rows:
            systems[system] = rows
            missing[system] = f"partial: {len(rows)} of 240 personas ranked"
        else:
            missing[system] = "no rankings (stage failed or did not run)"
    saadi, saadi_tables, notes = load_saadi()
    all_systems = {**systems, **saadi}
    result: dict = {"systems": {}, "missing": missing, "saadi_notes": notes, "saadi_tables": saadi_tables}
    for system, rows in all_systems.items():
        cells = by_cell(rows)
        entry = {
            "label": SYSTEM_LABELS.get(system, system),
            "n": len(rows),
            "per_cell": {str(c): cell_metrics(g) for c, g in sorted(cells.items())},
            "cells_1_3": cell_metrics([r for r in rows if r["cell"] in CELLS_1_3]),
            "reveal_cells": cell_metrics([r for r in rows if r["cell"] in REVEAL_CELLS]),
            "cell5_leak": any(hit(r, 10) or hit(r, 5) for r in cells.get(5, [])),
            "query_cost": {k: sum(int(r.get(k, 0) or 0) for r in rows) for k in ("model_calls", "input_tokens", "generated_tokens")},
        }
        for key in ("reveal_visible", "link_formed"):
            if any(key in r for r in rows):
                entry[f"split_{key}"] = {
                    "cells_1_3": split_metrics([r for r in rows if r["cell"] in CELLS_1_3], key),
                    "reveal_cells": split_metrics([r for r in rows if r["cell"] in REVEAL_CELLS], key),
                    "per_cell_true": {str(c): sum(1 for r in g if r.get(key) is True) for c, g in sorted(cells.items())},
                }
        result["systems"][system] = entry
    # Distance to the oracle, and every system against SAADI run 1 live.
    oracle = result["systems"].get("b1_oracle")
    run1 = saadi.get("saadi_run1_live")
    for system, rows in all_systems.items():
        entry = result["systems"][system]
        if oracle is not None:
            entry["gap_to_oracle_hr10_pp"] = {
                c: entry["per_cell"][c]["hr10"]["hr"] - oracle["per_cell"][c]["hr10"]["hr"]
                for c in entry["per_cell"]
                if c in oracle["per_cell"]
            }
            entry["gap_to_oracle_hr10_pp"]["cells_1_3"] = entry["cells_1_3"]["hr10"]["hr"] - oracle["cells_1_3"]["hr10"]["hr"]
        if run1 is not None and system != "saadi_run1_live":
            entry["vs_saadi_run1_live"] = {
                "cells_1_3": paired([r for r in rows if r["cell"] in CELLS_1_3], [r for r in run1 if r["cell"] in CELLS_1_3]),
                "reveal_cells": paired([r for r in rows if r["cell"] in REVEAL_CELLS], [r for r in run1 if r["cell"] in REVEAL_CELLS]),
                "per_cell": {
                    str(c): paired([r for r in rows if r["cell"] == c], [r for r in run1 if r["cell"] == c]) for c in sorted(by_cell(rows))
                },
            }
    if run1 is None:
        result["paired_note"] = "SAADI run 1's per-persona file is absent: no paired comparison."
    return result


def write_table(result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["system", "cell", "n", "hr5", "hr5_lo", "hr5_hi", "hr10", "hr10_lo", "hr10_hi", "median_rank"])
        for system, entry in result["systems"].items():
            for cell, m in entry["per_cell"].items():
                w.writerow(
                    [
                        system,
                        cell,
                        m["n"],
                        round(m["hr5"]["hr"], 2),
                        round(m["hr5"]["lo"], 2),
                        round(m["hr5"]["hi"], 2),
                        round(m["hr10"]["hr"], 2),
                        round(m["hr10"]["lo"], 2),
                        round(m["hr10"]["hi"], 2),
                        m["median_rank"],
                    ]
                )
        for system, agg in result.get("saadi_tables", {}).items():
            for cell, m in sorted(agg.items()):
                w.writerow([system, cell, m["n"], m["hr5"], "", "", m["hr10"], "", "", m["median_rank"]])


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------


def self_test(
    per_persona: Path, table: Path | None = None, run_label: str = "run1_p1_ledger", expected_json: Path | None = None
) -> tuple[bool, list[str]]:
    """Score a SAADI per-persona file through this scorer and compare with the run's own table.

    ``table`` is a results/method_update/*/table.csv (long schema: run, cell, n, method, hr5,
    hr10, median_rank); ``expected_json`` a p1-style step_04.json (numbers.per_cell).
    """

    expected: dict[tuple[int, str], dict] = {}
    if table is not None and table.exists():
        with table.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r.get("run") == run_label and r.get("method") in SAADI_METHODS:
                    expected[(int(r["cell"]), r["method"])] = {
                        "hr5": float(r["hr5"]),
                        "hr10": float(r["hr10"]),
                        "median_rank": float(r["median_rank"]),
                    }
    if expected_json is not None and expected_json.exists():
        per_cell = json.loads(expected_json.read_text(encoding="utf-8"))["numbers"]["per_cell"]
        for cell, v in per_cell.items():
            for m in SAADI_METHODS:
                expected[(int(cell), m)] = {
                    "hr5": float(v[f"hr5_{m}"]),
                    "hr10": float(v[f"hr10_{m}"]),
                    "median_rank": float(v[f"median_{m}"]),
                }
    if not expected:
        return False, [f"no expected values found ({table}, {expected_json})"]
    lines, ok = [], True
    for method in SAADI_METHODS:
        cells = by_cell(saadi_rows(per_persona, method))
        for cell, rows in sorted(cells.items()):
            exp = expected.get((cell, method))
            if exp is None:
                continue
            got = cell_metrics(rows)
            g = {"hr5": round(got["hr5"]["hr"], 1), "hr10": round(got["hr10"]["hr"], 1), "median_rank": got["median_rank"]}
            same = (
                abs(g["hr5"] - round(exp["hr5"], 1)) < 0.06
                and abs(g["hr10"] - round(exp["hr10"], 1)) < 0.06
                and abs(g["median_rank"] - exp["median_rank"]) < 1e-6
            )
            ok &= same
            lines.append(
                f"{'ok ' if same else 'DIFF'} cell {cell} {method:9s} scorer hr5={g['hr5']:.1f} hr10={g['hr10']:.1f} median={g['median_rank']:.1f} | table hr5={exp['hr5']:.1f} hr10={exp['hr10']:.1f} median={exp['median_rank']:.1f}"
            )
    return ok, lines


def main(argv: list[str]) -> int:
    if not argv or argv[0] == "score":
        result = score()
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "scores.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
        write_table(result, RESULTS_DIR / "table.csv")
        CM.write_numbers(
            "score",
            "score",
            {"missing": result["missing"], "saadi_notes": result["saadi_notes"], "scores_json": str(RESULTS_DIR / "scores.json")},
        )
        print(RESULTS_DIR / "scores.json")
        return 0
    if argv[0] == "self-test":
        run = argv[argv.index("--run") + 1] if "--run" in argv else "run1"
        per_persona = Path(argv[argv.index("--per-persona") + 1]) if "--per-persona" in argv else SAADI_RUNS[run]["per_persona"]
        table = Path(argv[argv.index("--table") + 1]) if "--table" in argv else SAADI_RUNS[run]["table"]
        expected_json = Path(argv[argv.index("--expected-json") + 1]) if "--expected-json" in argv else None
        if not per_persona.exists():
            print(f"self-test skipped: {per_persona} not found")
            return 2
        ok, lines = self_test(per_persona, table, SAADI_RUNS.get(run, {}).get("label", run), expected_json)
        print("\n".join(lines))
        print("SELF-TEST", "PASSED" if ok else "FAILED", f"({len(lines)} cell-method rows compared)")
        return 0 if ok else 1
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
