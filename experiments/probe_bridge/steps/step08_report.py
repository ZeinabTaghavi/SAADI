#!/usr/bin/env python3
"""Step 8: the final report - results/probe_bridge/{README.md,table.md,table.csv}
plus the distance figure. The step report is the README itself."""

from __future__ import annotations

import csv
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import banks  # noqa: E402
from common import (  # noqa: E402
    OUT_DIR,
    RESULTS_DIR,
    SEED,
    SPEC_FROZEN_JSON,
    StepReport,
    mcnemar_exact_p,
    paired_bootstrap_ci,
    read_json,
    read_jsonl,
    run_step,
    software_versions,
)

PER_PERSONA = OUT_DIR / "per_persona.jsonl"
METHODS = ("vanilla", "no_memory", "frozen", "live")
EXPECTED = {
    "1-3, 6-8 (reveal after gold)": {"vanilla": "fails", "frozen": "fails", "live": "succeeds, if the mechanism works"},
    "4 (reveal before gold)": {"vanilla": "fails", "frozen": "succeeds", "live": "succeeds"},
    "5 (no reveal)": {"vanilla": "fails", "frozen": "fails", "live": "fails"},
}


def hr(group, method, k="10") -> float:
    return 100.0 * sum(p[f"hit{k}_{method}"] for p in group) / len(group) if group else float("nan")


def median_rank(group, key) -> float:
    vals = [p[key] for p in group if p.get(key) is not None]
    return float(statistics.median(vals)) if vals else float("nan")


def verdict(value: float) -> str:
    if value >= 50:
        return "succeeds"
    if value < 20:
        return "fails"
    return "partial"


def main(report: StepReport, argv: list[str]) -> None:
    import numpy as np

    personas = read_jsonl(PER_PERSONA)
    frozen_spec = read_json(SPEC_FROZEN_JSON)
    cells = sorted({p["cell"] for p in personas})
    cell_labels = {c["cell"]: c["label"] for c in banks.CELLS}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def group_of(cell):
        return [p for p in personas if p["cell"] == cell]

    # ---- main table ----------------------------------------------------------
    main_rows, csv_rows = [], []
    for cell in cells:
        g = group_of(cell)
        diffs = [(float(p["hit10_live"]) - float(p["hit10_frozen"])) * 100.0 for p in g]
        mean, lo, hi = paired_bootstrap_ci(diffs)
        x = sum(1 for p in g if p["hit10_live"] and not p["hit10_frozen"])
        y = sum(1 for p in g if p["hit10_frozen"] and not p["hit10_live"])
        main_rows.append(
            [
                f"{cell} ({cell_labels[cell]})",
                len(g),
                *[f"{hr(g, m):.0f} / {median_rank(g, f'rank_{m}'):.0f}" for m in METHODS],
                f"{mean:+.1f} [{lo:+.1f}, {hi:+.1f}] pp, McNemar p={mcnemar_exact_p(x, y):.3f}",
            ]
        )
        for m in METHODS:
            csv_rows.append(
                {
                    "cell": cell,
                    "n": len(g),
                    "method": m,
                    "hr5": round(hr(g, m, "5"), 1),
                    "hr10": round(hr(g, m), 1),
                    "median_rank": median_rank(g, f"rank_{m}"),
                }
            )
        csv_rows.append(
            {
                "cell": cell,
                "n": len(g),
                "method": "live_minus_frozen_hr10_pp",
                "hr10": round(mean, 2),
                "hr5": None,
                "median_rank": f"CI [{lo:+.2f},{hi:+.2f}] McNemar p={mcnemar_exact_p(x, y):.4f}",
            }
        )

    def md_table(headers, rows):
        out = ["| " + " | ".join(map(str, headers)) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
        out += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
        return "\n".join(out)

    main_table = md_table(["Cell", "n", "Vanilla", "No memory", "Frozen", "Live", "Live - Frozen @10"], main_rows)

    # ---- one-sentence answer -------------------------------------------------
    g13 = [p for p in personas if p["cell"] in (1, 2, 3)]
    live_n = sum(p["hit10_live"] for p in g13)
    frozen_n = sum(p["hit10_frozen"] for p in g13)
    answer = (
        f"A later reveal makes the early chunk retrievable (top 10) in **{live_n} of {len(g13)}** personas "
        f"of cells 1-3, against **{frozen_n} of {len(g13)}** for frozen records "
        f"(Vanilla BM25: {sum(p['hit10_vanilla'] for p in g13)})."
    )

    # ---- expected vs observed ------------------------------------------------
    rows_eo = []
    obs = {}
    for label, cells_in in (
        ("1-3, 6-8 (reveal after gold)", (1, 2, 3, 6, 7, 8)),
        ("4 (reveal before gold)", (4,)),
        ("5 (no reveal)", (5,)),
    ):
        g = [p for p in personas if p["cell"] in cells_in]
        obs[label] = {m: f"{verdict(hr(g, m))} ({hr(g, m):.0f}%)" for m in ("vanilla", "frozen", "live")}
        held = all(obs[label][m].startswith(EXPECTED[label][m].split(",")[0]) for m in ("vanilla", "frozen", "live"))
        rows_eo.append(
            [
                label,
                f"{EXPECTED[label]['vanilla']} / obs {obs[label]['vanilla']}",
                f"{EXPECTED[label]['frozen']} / obs {obs[label]['frozen']}",
                f"{EXPECTED[label]['live']} / obs {obs[label]['live']}",
                "held" if held else "DID NOT HOLD",
            ]
        )
    eo_table = md_table(["Cells", "Vanilla (expected/observed)", "Frozen", "Live", "Pattern"], rows_eo)

    # ---- memory-check table ----------------------------------------------------
    mem_rows = []
    for cell in cells:
        g = group_of(cell)
        reached = [p for p in g if p.get("form2_in_that_record_final") and p.get("gold_chunk_linked_to_that_record")]
        no_rec = sum(1 for p in g if p.get("merge_direction") == "no_record")
        not_in = sum(1 for p in g if p.get("form1_record_found") and not p.get("form2_in_that_record_final"))
        reached_hit = sum(1 for p in reached if p["hit10_live"])
        mem_rows.append([cell, len(g), no_rec, not_in, len(reached), f"{reached_hit}/{len(reached)}" if reached else "-"])
    mem_table = md_table(["Cell", "n", "no record", "record without bridge word", "bridge reached", "reached and live hits@10"], mem_rows)

    # ---- distance and order ----------------------------------------------------
    dist_rows = []
    for cell in (1, 2, 3, 4):
        g = group_of(cell)
        dist = statistics.mean(abs(p["reveal_chunk_index"] - p["gold_chunk_index"]) for p in g if p.get("reveal_chunk_index") is not None)
        dist_rows.append([cell, cell_labels[cell], f"{dist:.0f}", *[f"{hr(g, m):.0f}" for m in METHODS]])
    dist_table = md_table(["Cell", "placement", "mean |reveal-gold| (chunks)", "Vanilla", "No memory", "Frozen", "Live"], dist_rows)
    fig_note = ""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = [float(r[2]) for r in dist_rows]
        fig, ax = plt.subplots(figsize=(6, 4))
        for m in METHODS:
            ax.plot(xs, [hr(group_of(c), m) for c in (1, 2, 3, 4)], marker="o", label=m)
        ax.set_xlabel("distance between reveal and gold (chunks; cell 4 = reveal before)")
        ax.set_ylabel("HR@10 (%)")
        ax.set_title("Bridge probe: distance and order (cells 1-4)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "fig_distance.png", dpi=150)
        fig_note = "![distance](fig_distance.png)"
    except Exception as exc:  # matplotlib may be absent on the server
        fig_note = f"(figure skipped: {exc}; the numbers are in the table above)"

    # ---- family comparison -----------------------------------------------------
    fam_rows = []
    for cell in (2, 6, 7, 8):
        g = group_of(cell)
        fam_rows.append(
            [
                cell,
                cell_labels[cell],
                *[f"{hr(g, m):.0f}" for m in METHODS],
                f"{median_rank(g, 'memory_rank_live'):.0f} / {median_rank(g, 'memory_rank_frozen'):.0f}",
            ]
        )
    fam_table = md_table(["Cell", "family", "Vanilla", "No memory", "Frozen", "Live", "mem-rank live/frozen"], fam_rows)

    # ---- ten seeded examples ----------------------------------------------------
    rng = np.random.default_rng(SEED)
    picks = sorted(int(x) for x in rng.choice(len(personas), size=min(10, len(personas)), replace=False))
    example_lines = []
    for i in picks:
        p = personas[i]
        example_lines.append(
            f'- **{p["persona_id"]}** (cell {p["cell"]}): query "{p["query"]}" -> {p["answer"]}\n'
            f"  - gold: {p['gold_sentence']}\n  - reveal: {p['reveal_sentence']}\n"
            f"  - gold record: {p.get('form1_record_name')!r}, form2 via {p.get('form2_field')}, "
            f"arrived step {p.get('form2_arrival_step')}\n"
            f"  - ranks: vanilla {p['rank_vanilla']}, no-memory {p['rank_no_memory']}, "
            f"frozen {p['rank_frozen']}, live {p['rank_live']} (of {p['n_chunks']})"
        )

    # ---- README -----------------------------------------------------------------
    versions = software_versions()
    gate = sum(1 for p in g13 if p.get("form2_in_that_record_final") and p.get("gold_chunk_linked_to_that_record"))
    readme = f"""# Bridge probe: do later reveals make early chunks retrievable?

{answer}

## Main table

HR@10 (%) / median rank of the gold chunk, per cell; 30 personas per cell give wide intervals and are
reported as they are (95% bootstrap CIs, 10,000 resamples, seed 42; exact McNemar on the hit indicators).

{main_table}

## Expected against observed

Success = HR@10 >= 50%, failure < 20%, otherwise partial (rule fixed with the metrics).

{eo_table}

## Where a failure happens (memory check, step 6)

{mem_table}

The bridge word reached the gold chunk's record in {gate} of {len(g13)} personas of cells 1-3.

## Distance and order (cells 1-4)

{dist_table}

{fig_note}

## Family comparison (cells 2, 6, 7, 8)

{fam_table}

## Ten examples (seeded draw, not chosen)

{chr(10).join(example_lines)}

## What changed since E2

E2 froze SAADI's own records on the four standard benchmarks and found live and frozen indistinguishable;
E2b showed those benchmarks almost never contain a query that needs a later alias (0 to 2.1%). The probe
creates that case on purpose inside real books, so the cells above measure the mechanism itself, with the
memory check separating extraction failures from scoring failures.

## Limits

Synthetic insertion of plain-style sentences, one construction model (Qwen3-30B-A3B-Instruct-2507,
published profile), 30 personas per cell, 10 books, BM25 only. Anything that went wrong in the run is in
the step reports (experiments/probe_bridge/reports/).

## Reproduction

- Spec hash (personas.jsonl, frozen before indexing): `{frozen_spec["personas_jsonl_sha256"]}`
- Commands: `bash experiments/probe_bridge/sh/01_select_books.sh` ... `08_report.sh`, in order
- Versions: {", ".join(f"{k}={v}" for k, v in versions.items())}
- Date: {time.strftime("%Y-%m-%d %H:%M:%S")}
"""
    (RESULTS_DIR / "README.md").write_text(readme, encoding="utf-8")
    (RESULTS_DIR / "table.md").write_text(
        "# Bridge probe tables\n\n" + main_table + "\n\n" + eo_table + "\n\n" + mem_table + "\n\n" + dist_table + "\n\n" + fam_table + "\n",
        encoding="utf-8",
    )
    with (RESULTS_DIR / "table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cell", "n", "method", "hr5", "hr10", "median_rank"])
        writer.writeheader()
        writer.writerows(csv_rows)
    report.files_written += [str(RESULTS_DIR / name) for name in ("README.md", "table.md", "table.csv")]
    report.commands.append("assemble results/probe_bridge from per_persona.jsonl")

    # The step report is the README itself.
    report.add("README (verbatim)", readme)
    report.numbers["one_sentence_answer"] = answer


if __name__ == "__main__":
    raise SystemExit(run_step(8, "Final report", main, requires=[6, 7]))
