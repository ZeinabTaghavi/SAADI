#!/usr/bin/env python3
"""Stage 02 (GPUs 0-3): index the 10 probe books with profile p1b_ledger, then report the cost.

Runs the study runner exactly as runs 0 and 1 did (scripts/run_study.py --suite
probe_bridge, --profile p1b_ledger, tensor parallel 4, its own vLLM port and
output root). Resumable per book: rerunning continues from the construction
cache, and the ledger event log trims itself to the regenerated step.

Report, next to run 1 and run 0: calls per book (must equal run 1), input and
output tokens per call, records and stubs per book, recall cap hits, wall clock,
candidate-net time per chunk; and the counts of the three changes (redirected,
auto-join, SAME AS rejected by reason, aliases rejected, recall match classes).
Reuses run 1's build code (p1_ledger/scripts/step02_build.py) by import.
Flags: --force; --report-only (do not start the study runner; report the build that
is already on disk, e.g. after a manual resume or for a local dry run).
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_p1b as C  # noqa: E402

INVOCATIONS_JSON = C.OUT_DIR / "build_invocations.json"
EXPECTED = {"candidate_threshold": 0.3, "recall": "chunk_tokens", "recall_cap": 8, "join_direction": "enforced", "same_as_rule": "p1b", "aliases_from_spans": True}
RUN1_VLLM_PORT = "29517"


def chunk_records(index_path: Path | None) -> dict[str, int]:
    if index_path is None:
        return {}
    from saadi.indexing.retrieval_methods.mbr import _resolve_cache_file_path

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    out = {}
    for doc in payload.get("docs", []):
        steps = _resolve_cache_file_path((doc.get("files") or {}).get("steps_index", ""), base_dir=index_path.parent)
        if Path(steps).exists():
            out[str(doc.get("doc_id"))] = sum(1 for _ in open(steps, encoding="utf-8"))
    return out


def p1b_counts(books: list[str]) -> dict[str, Counter]:
    from saadi.ledger.ledger import LedgerStore

    store = LedgerStore(C.LEDGER_DIR)
    out: dict[str, Counter] = {}
    for book in books:
        ledger = store.doc(book)
        c = Counter({k: v for k, v in ledger.counters.items() if isinstance(v, int)})
        classes = Counter()
        for ev in ledger.events:
            matches = (ev.get("p1b") or {}).get("recall_matches", {})
            shown = set(ev.get("recalled_ledger_ids", []))
            for entry_id, m in matches.items():
                classes[("recalled_" if entry_id in shown else "left_out_") + m["class"]] += 1
        c.update(classes)
        c["labelled_entries"] = len(ledger.labels)
        out[book] = c
    return out


def main(report: C.StepReport, argv: list[str]) -> None:
    b = C.load_p1_step("step02_build")
    tp = C.arg_value(argv, "--tensor-parallel-size", os.environ.get("P1_TENSOR_PARALLEL", "4"))
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    spec_hash = C.verify_frozen_spec()
    report.add("Frozen spec", f"personas.jsonl SHA-256 verified: `{spec_hash}` (same spec as runs 0 and 1).")
    if not (C.ROOT / "outputs" / "saadi" / "reference" / "manifest.json").exists():
        raise RuntimeError("outputs/saadi/reference/manifest.json is missing; the published base of the profile needs it.")
    if os.environ.get("VLLM_PORT", "") in ("", RUN1_VLLM_PORT):
        raise RuntimeError(f"VLLM_PORT is {os.environ.get('VLLM_PORT')!r}; run 1b needs its own port (env.sh sets 29537; P1B_VLLM_PORT overrides).")

    scratch = C.StepReport(step=2, title="scratch")
    resolved = b.resolve_configuration(scratch)  # temperature 0, seed 42, model id, ungrounded keep, store_dir
    ledger_cfg = resolved["ingest"]["strategy_kwargs"]["ledger"]
    wrong = {k: (ledger_cfg.get(k), v) for k, v in EXPECTED.items() if ledger_cfg.get(k) != v}
    if wrong:
        raise RuntimeError(f"The resolved ledger settings are not p1b's: {wrong} (resolved, expected)")
    strategy = resolved["ingest"]["strategy_kwargs"]
    rows = [
        ["construction LLM", resolved["model"]["model_id"]],
        ["temperature / seed", f"{resolved['model']['generate']['temperature']} / {resolved['model']['sampling'].get('seed')}"],
        ["max output tokens", resolved["model"]["generate"].get("max_tokens")],
        ["budget shown (records, aliases, descriptions)", f"{strategy.get('max_entities')}, {strategy.get('max_aliases')}, {strategy.get('max_desc')}"],
        ["history text / anchor matching / ungrounded", f"{strategy.get('history_l0_text')} / {strategy.get('anchor_matching')} / {strategy.get('ungrounded_entities')}"],
        ["schedule / wavefront batch", f"{resolved['ingest'].get('schedule')} / {resolved['ingest'].get('wavefront_batch_size')}"],
    ] + [[f"ledger.{k}", v] for k, v in ledger_cfg.items()]
    report.table("Resolved construction settings (profile p1b_ledger)", ["setting", "value"], rows)
    report.numbers["resolved_ledger"] = ledger_cfg
    report.numbers["json_alias_join"] = bool(ledger_cfg.get("json_alias_join", False))
    b.environment_table(report, ledger_cfg, gpus)

    # Run 0's token counts were rebuilt by run 1 (its step 2); reuse them instead of rebuilding.
    run1_s2 = C.run1_report_numbers(2)
    tok_run1 = run1_s2.get("tokens", {})
    if tok_run1.get("run0") and not b.RUN0_TOKENS_JSON.exists():
        C.write_json(b.RUN0_TOKENS_JSON, tok_run1["run0"])

    cmd = [sys.executable, str(C.ROOT / "scripts" / "run_study.py"), "--suite", "probe_bridge", "--profile", C.PROFILE, "--datasets", "novelhopqa_probe", "--retrievers", "bm25", "--stage", "ingest", "--tensor-parallel-size", str(tp)]
    env = dict(os.environ, SADI_OUTPUT_DIR=str(C.OUT_DIR), CUDA_VISIBLE_DEVICES=gpus)
    report.commands.append(" ".join([f"SADI_OUTPUT_DIR={C.OUT_DIR}", f"CUDA_VISIBLE_DEVICES={gpus}", f"VLLM_PORT={env.get('VLLM_PORT', '')}", *cmd]))
    invocations = C.read_json(INVOCATIONS_JSON) if INVOCATIONS_JSON.exists() else []
    t0 = time.time()
    if "--report-only" in argv or os.environ.get("P1B_REPORT_ONLY_DRY_RUN") == "1":  # the env switch: local dry runs of run_all.sh only
        report.unexpected.append("--report-only: the study runner was not started; this reports the build already on disk.")
        returncode = 0
    else:
        returncode = subprocess.run(cmd, cwd=C.ROOT, env=env).returncode
        invocations.append({"started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)), "seconds": round(time.time() - t0), "returncode": returncode})
        C.write_json(INVOCATIONS_JSON, invocations)
        report.files_written.append(str(INVOCATIONS_JSON))
    elapsed = time.time() - t0
    total = sum(x["seconds"] for x in invocations)
    run1_hours = run1_s2.get("wall_clock_s_total", 0) / 3600 if run1_s2 else None
    report.numbers.update({"wall_clock_s_this": elapsed, "wall_clock_s_total": total, "invocations": invocations})

    finished, books = b.scan_cache(scratch)
    report.numbers["cache_index"] = scratch.numbers.get("cache_index")
    if returncode != 0:
        report.unexpected.append(f"run_study.py exited with code {returncode}. Rerun `run_all.sh --from 02_build` to resume; the construction cache and the ledger log are kept.")
    if not books:
        raise RuntimeError("No construction records were written; see the log tail.")

    stats = b.ledger_stats(books)
    counts = p1b_counts(sorted(books))
    calls1 = chunk_records(C.run1_cache_index())
    calls0 = b.run0_chunk_records()
    st1 = run1_s2.get("ledger_stats", {})
    rows, mismatch = [], []
    for book in sorted(books, key=lambda d: books[d]["chunk_records"]):
        bk, s, s1 = books[book], stats[book], st1.get(book, {})
        if bk["finished"] and calls1.get(book) is not None and calls1[book] != bk["chunk_records"]:
            mismatch.append(f"{book}: {bk['chunk_records']} vs run 1 {calls1[book]}")
        rows.append([book, "finished" if bk["finished"] else "in progress", C.three(bk["chunk_records"], calls1.get(book), calls0.get(book)), f"{s['records']} / {s1.get('records', 'n/a')}", f"{s['stubs']} / {s1.get('stubs', 'n/a')}", f"{s['ledger_cap_hits']} / {s1.get('ledger_cap_hits', 'n/a')}", s["decision_parse_failures"], f"{s['candidates_per_chunk']:.1f} / {s1.get('candidates_per_chunk', float('nan')):.1f}"])
    report.table("Books: calls (1b / 1 / 0), records, stubs, recall cap hits and candidates per chunk (1b / 1)", ["book", "state", "calls", "records", "stubs", "recall cap hits", "decision parse failures", "candidates per chunk"], rows)
    if mismatch:
        report.unexpected.append("Calls per book differ from run 1 (the chunking must be identical): " + "; ".join(mismatch))

    tot, tot1 = Counter(), Counter()
    for s in stats.values():
        tot.update({k: v for k, v in s.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
    for s in st1.values():
        tot1.update({k: v for k, v in s.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
    pc = Counter()
    for c in counts.values():
        pc.update(c)
    n_events = sum(s["events"] for s in stats.values()) or 1
    secs = statistics.mean(s["candidate_seconds_mean"] for s in stats.values() if s["events"])
    report.table(
        "Ledger totals (run 1b / run 1)",
        ["measure", "run 1b", "run 1"],
        [
            ["calls (events)", n_events, tot1.get("events", "n/a")],
            ["records / stubs", f"{tot['records']} / {tot['stubs']}", f"{tot1.get('records', 'n/a')} / {tot1.get('stubs', 'n/a')}"],
            ["promotions", tot["promotions"], tot1.get("promotions", "n/a")],
            ["final SAME AS (model accepted + auto-join + JSON-alias join) / unresolved", f"{tot['same_as']} / {tot['same_as_unresolved']}", f"{tot1.get('same_as', 'n/a')} / {tot1.get('same_as_unresolved', 'n/a')}"],
            ["NEW -> stub / NOT AN ENTITY", f"{tot['new_stub']} / {tot['not_an_entity']}", f"{tot1.get('new_stub', 'n/a')} / {tot1.get('not_an_entity', 'n/a')}"],
            ["decision-block parse failures / omitted candidates", f"{tot['decision_parse_failures']} / {tot['omitted_candidates']}", f"{tot1.get('decision_parse_failures', 'n/a')} / {tot1.get('omitted_candidates', 'n/a')}"],
            ["recall cap hits (chunks) / entries left out", f"{tot['ledger_cap_hits']} ({100.0 * tot['ledger_cap_hits'] / n_events:.0f}%) / {tot['ledger_left_out']}", f"{tot1.get('ledger_cap_hits', 'n/a')} / {tot1.get('ledger_left_out', 'n/a')}"],
            ["description replacements (cap 5)", tot["description_replacements"], tot1.get("description_replacements", "n/a")],
            ["candidate-net seconds per chunk (mean)", f"{secs:.3f}", "0.082"],
            ["JSON parser status not ok (fallback or failed)", tot["parser_json_not_ok"], tot1.get("parser_json_not_ok", "n/a")],
            ["memory.json entries holding a replaced description (should be 0)", tot["memory_json_drift"], tot1.get("memory_json_drift", "n/a")],
        ],
    )
    rejected = {k.replace("same_as_rejected_", ""): v for k, v in pc.items() if k.startswith("same_as_rejected_")}
    accepted = {k.replace("same_as_accepted_", ""): v for k, v in pc.items() if k.startswith("same_as_accepted_")}
    alias_rej = {k.replace("alias_rejected_", ""): v for k, v in pc.items() if k.startswith("alias_rejected_")}
    report.table(
        "The three changes at work (all books)",
        ["measure", "value"],
        [
            ["model SAME AS accepted, by evidence", dict(accepted)],
            ["model SAME AS rejected (became NEW), by reason", dict(rejected) or 0],
            ["auto-join (NEW equal to a shown entry) / blocked by a conflict / ambiguous", f"{pc['auto_join']} / {pc['auto_join_blocked_by_conflict']} / {pc['auto_join_ambiguous']}"],
            ["JSON-alias join (optional switch) / ambiguous", f"{pc['json_alias_join']} / {pc['json_alias_join_ambiguous']}" + ("" if report.numbers["json_alias_join"] else " (switch off)")],
            ["JSON records redirected into a SAME AS target / duplicates merged", f"{pc['redirected']} / {pc['redirected_duplicates_merged']}"],
            ["NEW folded into a described record, rejected for a conflict", pc["described_match_rejected_conflict"]],
            ["aliases rejected, by reason", dict(alias_rej) or 0],
            ["recall: entries shown, by match class", {k[9:]: v for k, v in pc.items() if k.startswith("recalled_")}],
            ["recall: entries left out by the cap of 8, by match class", {k[9:]: v for k, v in pc.items() if k.startswith("left_out_")}],
            ["entries with a candidate-net label", pc["labelled_entries"]],
        ],
    )
    report.numbers.update({"books": {d: {k: v for k, v in x.items() if k != "reply_chars"} for d, x in books.items()}, "ledger_stats": stats, "p1b_counts": {d: dict(c) for d, c in counts.items()}, "p1b_totals": dict(pc), "calls_run1": calls1, "calls_run0": calls0, "books_finished": finished})

    tok = b.load_tokenizer()
    if isinstance(tok, Exception):
        report.unexpected.append(f"The Qwen tokenizer could not be loaded offline ({tok}); token counts are characters/4.")
    r1b = b.run1_tokens(tok, books)
    r1, r0 = tok_run1.get("run1", {}), tok_run1.get("run0", {})
    rows = []
    for book in sorted(books, key=lambda d: books[d]["chunk_records"]):
        a = r1b.get(book, {})
        rows.append([book, C.three(f"{a.get('input_mean', float('nan')):.0f}", f"{r1[book]['input_mean']:.0f}" if book in r1 else None, f"{r0[book]['input_mean']:.0f}" if book in r0 else None), C.three(f"{a.get('output_mean', float('nan')):.0f}", f"{r1[book]['output_mean']:.0f}" if book in r1 else None, f"{r0[book]['output_mean']:.0f}" if book in r0 else None), a.get("likely_truncated", 0)])
    in1b = statistics.mean(v["input_mean"] for v in r1b.values() if v["n"])
    out1b = statistics.mean(v["output_mean"] for v in r1b.values() if v["n"])
    rows.append(["**all books (mean of book means)**", C.three(f"{in1b:.0f}", f"{tok_run1.get('run1_input_mean', float('nan')):.0f}", f"{tok_run1.get('run0_input_mean', float('nan')):.0f}"), C.three(f"{out1b:.0f}", f"{tok_run1.get('run1_output_mean', float('nan')):.0f}", f"{tok_run1.get('run0_output_mean', float('nan')):.0f}"), sum(v.get("likely_truncated", 0) for v in r1b.values())])
    report.table("Tokens per call, input and output (1b / 1 / 0; Qwen tokenizer), and replies at the output cap with no JSON", ["book", "input", "output", "likely truncated (1b)"], rows)
    report.numbers["tokens"] = {"run1b": r1b, "run1b_input_mean": in1b, "run1b_output_mean": out1b, "run1_input_mean": tok_run1.get("run1_input_mean"), "run1_output_mean": tok_run1.get("run1_output_mean"), "run0_input_mean": tok_run1.get("run0_input_mean"), "run0_output_mean": tok_run1.get("run0_output_mean"), "likely_truncated": sum(v.get("likely_truncated", 0) for v in r1b.values())}
    report.add("Wall clock", f"{elapsed / 3600:.2f} h this invocation; {total / 3600:.2f} h over {len(invocations)} invocation(s). Run 1: {f'{run1_hours:.2f} h' if run1_hours else 'n/a'}; run 0: 3.36 h.")

    if finished < C.N_BOOKS or returncode != 0:
        report.skip_marker = True
        raise RuntimeError(f"Only {finished}/{C.N_BOOKS} books have a final memory.json (run_study exit code {returncode}). Rerun `run_all.sh --from 02_build` to resume.")
    report.files_written += [str(report.numbers["cache_index"]), f"{C.LEDGER_DIR}/<book>/events.jsonl, ledger.json"]


if __name__ == "__main__":
    raise SystemExit(C.run_stage(2, "Index with profile p1b_ledger (GPUs 0-3)", main))
