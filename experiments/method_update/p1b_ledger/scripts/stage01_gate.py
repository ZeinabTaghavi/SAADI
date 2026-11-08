#!/usr/bin/env python3
"""Stage 01 (CPU): the candidate net at threshold 0.3 on the gold and reveal chunks, next to run 1's 0.5.

Form-1 and form-2 recall per family and per cell, candidates and the
false-candidate proxy per chunk, seconds per chunk, and the exact prompt of the
LLM call (unchanged from run 1) with one filled example built with run 1b's
recall. This stage reports only; it never blocks the build.
Reuses run 1's gate code (experiments/method_update/p1_ledger/scripts/step01_gate.py) by import.
"""

from __future__ import annotations

import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_p1b as C  # noqa: E402

FAMILY_NAMES = {"A": "A surname", "B": "B nickname", "C": "C occupation", "D": "D owned object"}


def main(report: C.StepReport, argv: list[str]) -> None:
    from saadi.ledger.candidate_net import build_candidate_net

    g = C.load_p1_step("step01_gate")
    C.p1.GATE_DIR.mkdir(parents=True, exist_ok=True)
    spec_hash = C.verify_frozen_spec()
    report.add("Frozen spec", f"personas.jsonl SHA-256 verified: `{spec_hash}` (same spec as runs 0 and 1).")
    personas = C.load_personas()
    cfg = C.p1.ledger_profile_config()
    if float(cfg["candidate_threshold"]) != 0.3 or cfg["candidate_model"] != "urchade/gliner_large-v2.1":
        raise RuntimeError(f"configs/profiles/p1b_ledger.yaml must keep urchade/gliner_large-v2.1 at threshold 0.3; got {cfg['candidate_model']} at {cfg['candidate_threshold']}")
    kind = C.arg_value(argv, "--candidate-net", cfg.get("candidate_net", "gliner"))
    net_cfg = dict(cfg, candidate_net=kind, candidate_device="cpu")
    report.commands.append(f"candidate net: {kind} model={cfg['candidate_model']} device=cpu threshold={cfg['candidate_threshold']} labels={cfg['candidate_labels']}")

    # Gold and reveal chunks, cut with the ingest pipeline's chunker (as in run 1's gate).
    needed: dict[str, tuple[str, int]] = {}
    for p in personas:
        needed[p["gold_chunk_id"]] = (p["book_id"], int(p["gold_chunk_index"]))
        if p["cell"] in C.REVEAL_CELLS and p.get("reveal_chunk_id"):
            needed[p["reveal_chunk_id"]] = (p["book_id"], int(p["reveal_chunk_index"]))
    chunks: dict[str, str] = {}
    for book in sorted({b for b, _ in needed.values()}):
        texts = C.p1.book_chunks(book)
        for chunk_id, (b, idx) in needed.items():
            if b == book:
                chunks[chunk_id] = texts[idx]
    missing = [p["persona_id"] for p in personas if p["gold_sentence"] not in chunks[p["gold_chunk_id"]]]
    if missing:
        raise RuntimeError(f"Gold sentences not found in their chunks (chunking differs from the probe): {missing[:5]}")
    limit = int(C.arg_value(argv, "--limit", "0") or 0)
    run_chunks = dict(list(chunks.items())[:limit]) if limit else chunks
    if limit:
        report.unexpected.append(f"--limit {limit}: only {limit} chunks were run (dry run); the recall shares are not the gate.")

    net = build_candidate_net(net_cfg)
    t0 = time.time()
    net.load()
    load_s = time.time() - t0
    describe = net.describe()
    report.numbers["candidate_net"] = describe
    report.add("Candidate net", f"`{describe['model']}` on `{describe.get('device')}`, loaded in {load_s:.1f} s; labels {describe['labels']}; threshold {describe.get('threshold')}; versions {describe.get('versions')}.")
    print(f"Running {describe['model']} at threshold {describe.get('threshold')} on {len(run_chunks)} chunks ...", flush=True)
    results, t_total = g.run_net_on_chunks(net, run_chunks)
    C.write_jsonl(C.p1.GATE_DIR / "candidates.jsonl", results.values())
    report.files_written.append(str(C.p1.GATE_DIR / "candidates.jsonl"))

    personas_run = [p for p in personas if p["gold_chunk_id"] in results] if limit else personas
    rec = g.recall_tables(personas_run, results)
    C.write_jsonl(C.p1.GATE_DIR / "recall_rows.jsonl", rec["rows"])
    report.files_written.append(str(C.p1.GATE_DIR / "recall_rows.jsonl"))
    run1 = C.run1_report_numbers(1)
    r1_fam = run1.get("per_family", {})
    r1_cell = {int(k): v for k, v in run1.get("per_cell", {}).items()}

    def frac(d: dict, key: str) -> str:
        n, tot = d.get(f"{key}_caught", 0), d.get(f"{key}_n", 0)
        return C.pct(n, tot) if tot else "-"

    report.table(
        "Names caught per family: threshold 0.3 (run 1b) against 0.5 (run 1)",
        ["family", "form 1 at 0.3", "form 1 at 0.5 (run 1)", "form 2 at 0.3", "form 2 at 0.5 (run 1)"],
        [[FAMILY_NAMES.get(f, f), frac(rec["per_family"][f], "form1"), frac(r1_fam.get(f, {}), "form1") if r1_fam else "n/a", frac(rec["per_family"][f], "form2"), frac(r1_fam.get(f, {}), "form2") if r1_fam else "n/a"] for f in sorted(rec["per_family"])],
    )
    report.table(
        "Names caught per cell: threshold 0.3 (run 1b) against 0.5 (run 1)",
        ["cell", "label", "form 1 at 0.3", "form 1 at 0.5", "form 2 at 0.3", "form 2 at 0.5"],
        [[c, C.CELL_LABELS.get(c, ""), frac(rec["per_cell"][c], "form1"), frac(r1_cell.get(c, {}), "form1") if r1_cell else "n/a", frac(rec["per_cell"][c], "form2"), frac(r1_cell.get(c, {}), "form2") if r1_cell else "n/a"] for c in sorted(rec["per_cell"])],
    )
    tot = rec["total"]
    f1, f2 = tot.get("form1_caught", 0) / max(1, tot.get("form1_n", 0)), tot.get("form2_caught", 0) / max(1, tot.get("form2_n", 0))
    n_c = [len(r["candidates"]) for r in results.values()]
    false_n = sum(1 for r in results.values() for c in r["candidates"] if g.false_proxy(c["text"], c["label"]))
    secs = [r["seconds"] for r in results.values()]
    labels = Counter(c["label"] for r in results.values() for c in r["candidates"])
    report.table(
        "Totals and cost: threshold 0.3 (run 1b) against 0.5 (run 1)",
        ["measure", "run 1b (0.3)", "run 1 (0.5)"],
        [
            ["form 1 caught", C.pct(tot.get("form1_caught", 0), tot.get("form1_n", 0)), C.pct(run1["form1_caught"], run1["form1_n"]) if run1 else "n/a"],
            ["form 2 caught", C.pct(tot.get("form2_caught", 0), tot.get("form2_n", 0)), C.pct(run1["form2_caught"], run1["form2_n"]) if run1 else "n/a"],
            ["candidates per chunk (mean)", f"{statistics.mean(n_c):.1f}", f"{run1.get('candidates_per_chunk_mean', float('nan')):.1f}" if run1 else "n/a"],
            ["false-candidate proxy per chunk (common word / title / no letters)", f"{false_n / max(1, len(results)):.2f}", f"{run1.get('false_proxy_per_chunk', float('nan')):.2f}" if run1 else "n/a"],
            ["chunks at the cap of 40 candidates", sum(1 for r in results.values() if r["cap_hit"]), "-"],
            ["seconds per chunk on CPU (mean)", f"{statistics.mean(secs):.2f}", f"{run1.get('seconds_per_chunk_mean', float('nan')):.2f}" if run1 else "n/a"],
            ["candidates per label", dict(labels.most_common()), dict(Counter(run1.get("labels", {})).most_common()) if run1 else "n/a"],
        ],
    )
    missed2 = [r for r in rec["rows"] if r["form2_caught"] is None and r["cell"] in C.REVEAL_CELLS]
    report.add(
        "Form 2 missed at 0.3",
        f"{len(missed2)} names; by family {dict(Counter(r['family'] for r in missed2))}. Family C form 2 is an occupation phrase, which a name tagger does not list (as in run 1); "
        "the recall of run 1b no longer depends on the candidate list, so a missed name can still bring its record back by its tokens.",
    )
    report.numbers.update({"form1_rate": f1, "form2_rate": f2, "per_family": rec["per_family"], "per_cell": rec["per_cell"], "candidates_per_chunk_mean": statistics.mean(n_c), "false_proxy_per_chunk": false_n / max(1, len(results)), "seconds_per_chunk_mean": statistics.mean(secs), "run1": {k: run1.get(k) for k in ("form1_rate", "form2_rate", "candidates_per_chunk_mean", "false_proxy_per_chunk")}})
    if f1 < 0.8 and not limit:
        report.unexpected.append(f"Form-1 recall is {100 * f1:.0f}% at threshold 0.3, below run 1's 80% floor. Reported only: this stage never blocks the build (run_all.sh continues).")

    # The prompt: the system prompt is run 1's, unchanged; the example shows run 1b's recall.
    system, user = g.example_prompt(report, net, personas, chunks)
    (C.p1.GATE_DIR / "system_prompt.txt").write_text(system + "\n", encoding="utf-8")
    (C.p1.GATE_DIR / "example_user_prompt.txt").write_text(user + "\n", encoding="utf-8")
    report.files_written += [str(C.p1.GATE_DIR / "system_prompt.txt"), str(C.p1.GATE_DIR / "example_user_prompt.txt")]
    from saadi.ledger.prompt import LEDGER_SYSTEM_PROMPT

    report.add("System prompt of the LLM call", "Identical to run 1's (`saadi.ledger.prompt.LEDGER_SYSTEM_PROMPT`, unchanged): " + ("yes" if system == LEDGER_SYSTEM_PROMPT else "**NO, it differs**") + f". Full text: `{C.p1.GATE_DIR / 'system_prompt.txt'}`.")
    marker = "[HIERARCHY_CONTEXT_JSON]\n"
    shown = user
    if marker in user:
        head, rest = user.split(marker, 1)
        hier, tail = rest.split("\n\n[", 1)
        if len(hier) > 500:
            hier = hier[:500] + f" ... [{len(hier) - 500} more characters; full text in example_user_prompt.txt]"
        shown = head + marker + hier + "\n\n[" + tail
    report.add(f"One filled user prompt with run 1b's recall (B31, gold chunk of persona c1_B31_000; {report.numbers.get('example_prompt_source')})", "```text\n" + shown + "\n```")
    report.add("Hand-off", "Reported only; run_all.sh starts the build next.")


if __name__ == "__main__":
    raise SystemExit(C.run_stage(1, "Candidate net at threshold 0.3 (CPU)", main))
