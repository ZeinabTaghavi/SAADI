#!/usr/bin/env python3
"""Step 1 (CPU): the candidate-net gate, before any GPU time is spent.

Loads the candidate net of the profile, runs it on the gold and reveal chunks of
the 10 probe books, and reports: the model chosen and the alternatives looked
at; the share of the 240 form-1 and 210 form-2 names caught, per cell and per
family; a false-candidate proxy per chunk; seconds per chunk; and the exact
final prompt of the LLM call with one filled example. If form-1 recall is
below 80% the report says so and proposes a change (spec, section D).

Flags: --force; --model <hf id>; --device cpu|cuda|auto; --threshold 0.5;
--candidate-net gliner|regex (regex = no model, dry run only); --limit N
(first N chunks, dry runs); --compare-models a,b (same chunks, extra table);
--no-second-threshold (skip the informational pass at 0.3).
"""

from __future__ import annotations

import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CELL_LABELS,
    GATE_DIR,
    REVEAL_CELLS,
    StepReport,
    arg_value,
    book_chunks,
    find_run0_cache_index,
    ledger_profile_config,
    load_personas,
    pct,
    run_step,
    verify_frozen_spec,
    write_jsonl,
)

RECALL_FLOOR = 0.80
SECOND_THRESHOLD = 0.3

# Models looked at on Hugging Face on 8 Nov 2026 (model cards and the GLiNER
# README; numbers as the cards state them, none measured here). "Avg F1" is the
# knowledgator zero-shot table (CrossNER x5, MIT movie, MIT restaurant).
ALTERNATIVES = [
    ["urchade/gliner_large-v2.1 (default)", "DeBERTa-v3-large span model, 459M, Apache-2.0", "0.5754 (same table)", "Stable `gliner` API; the spec's default; the one measured below."],
    ["gliner-community/gliner_large-v2.5", "same architecture and training data (pile-mistral-v0.1), Apache-2.0", "card shows a chart image, no numbers in text", "No quotable gain over v2.1; same API, so a one-flag swap (--model) if the gate says so."],
    ["knowledgator/gliner-multitask-large-v0.5", "prompt-tunable multitask GLiNER, Apache-2.0 (Jun 2024)", "0.6276 (v0.5 family row)", "Best published average with the same API; the proposed swap if form-1 recall is below 80%."],
    ["knowledgator/gliner-multitask-v1.0", "multitask GLiNER, Apache-2.0", "0.6325", "Marginal over v0.5; same caveat (benchmarks use other label sets)."],
    ["numind/NuNER_Zero", "token classifier on the GLiNER architecture, 0.4B, MIT", "+3.1% token-level F1 over large-v2.1 (card)", "Needs lower-cased labels and the gliner 0.1.12 decoding path; different span semantics (merges adjacent tokens)."],
    ["EmergentMethods/gliner_large_news-v2.1", "v2.1 fine-tuned on synthetic news NER, Apache-2.0", "up to +7.5% over v2.1 on 18 sets (card)", "News-domain tuning; novels are not its domain."],
    ["fastino/gliner2.5-* (GLiNER2 / 2.5)", "schema-driven extraction, separate `gliner2` library (Aug-Sep 2026)", "vendor numbers only", "Different library and interface; not a drop-in candidate net."],
]
ALTERNATIVES_NOTE = (
    "Decision rule (spec D): keep the default unless the gate below shows it clearly insufficient. The published "
    "gains of the alternatives are on CrossNER/MIT label sets, not on the names this probe inserts; the gate "
    "measures the quantity that matters here directly. `--compare-models` runs any of them on the same chunks."
)


def caught(candidates: list[str], form: str) -> str | None:
    """How the inserted name is caught: a candidate equal to it after title stripping, or one containing it."""

    from saadi.ledger.ledger import match_kind

    best = None
    for cand in candidates:
        kind = match_kind(cand, form)
        if kind == "exact":
            return "exact"
        if kind == "partial":
            best = "partial"
    return best


_COMMON_FALSE = {
    "the", "a", "an", "and", "but", "or", "well", "oh", "yes", "no", "sir", "god", "lord", "heaven", "chapter", "i", "you", "he",
    "she", "it", "we", "they", "mr", "mrs", "miss", "ma'am", "madam", "monsieur", "madame", "doctor", "captain", "aunt", "uncle",
}


def false_proxy(cand_text: str, label: str) -> bool:
    """A conservative proxy for a false candidate (no gold labels exist for the books' own names)."""

    t = cand_text.strip().lower().strip("'\".,;:!?")
    return (not any(ch.isalpha() for ch in t)) or t in _COMMON_FALSE or len(t) <= 1


def run_net_on_chunks(net, chunks: dict[str, str]) -> tuple[dict[str, dict], float]:
    results: dict[str, dict] = {}
    t_total = 0.0
    for i, (chunk_id, text) in enumerate(chunks.items()):
        run = net.run(text)
        t_total += run.seconds
        results[chunk_id] = {
            "chunk_id": chunk_id,
            "seconds": run.seconds,
            "raw_count": run.raw_count,
            "cap_hit": run.cap_hit,
            "candidates": [c.to_dict() for c in run.candidates],
            "dropped": run.dropped,
        }
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(chunks)} chunks, {t_total / (i + 1):.2f} s per chunk", flush=True)
    return results, t_total


def recall_tables(personas: list[dict], results: dict[str, dict], score_floor: float | None = None) -> dict:
    def cands(chunk_id: str) -> list[str]:
        rows = results.get(chunk_id, {}).get("candidates", [])
        return [c["text"] for c in rows if score_floor is None or float(c["score"]) >= score_floor]

    per_cell: dict = defaultdict(lambda: Counter())
    per_family: dict = defaultdict(lambda: Counter())
    rows_out = []
    for p in personas:
        cell, fam = p["cell"], p["family"]
        f1 = caught(cands(p["gold_chunk_id"]), p["form1"])
        per_cell[cell]["form1_n"] += 1
        per_family[fam]["form1_n"] += 1
        if f1:
            per_cell[cell]["form1_caught"] += 1
            per_family[fam]["form1_caught"] += 1
            per_cell[cell][f"form1_{f1}"] += 1
        f2 = None
        if cell in REVEAL_CELLS and p.get("reveal_chunk_id"):
            f2 = caught(cands(p["reveal_chunk_id"]), p["form2"])
            per_cell[cell]["form2_n"] += 1
            per_family[fam]["form2_n"] += 1
            if f2:
                per_cell[cell]["form2_caught"] += 1
                per_family[fam]["form2_caught"] += 1
                per_cell[cell][f"form2_{f2}"] += 1
        rows_out.append({"persona_id": p["persona_id"], "cell": cell, "family": fam, "form1": p["form1"], "form1_caught": f1, "form2": p["form2"], "form2_caught": f2})
    tot = Counter()
    for c in per_cell.values():
        tot.update(c)
    return {"per_cell": {k: dict(v) for k, v in per_cell.items()}, "per_family": {k: dict(v) for k, v in per_family.items()}, "total": dict(tot), "rows": rows_out}


def example_prompt(report: StepReport, net, personas: list[dict], chunks: dict[str, str]) -> tuple[str, str]:
    """The exact system prompt and one filled user prompt (the first gold chunk of B31; run 0's memory before it when available)."""

    from saadi.context_strategies import get_strategy
    from saadi.core.state import init_discourse_state, state_from_snapshot
    from saadi.core.types import ChunkRecord

    p = next(x for x in personas if x["book_id"] == "B31" and x["cell"] == 1)
    book, step = p["book_id"], int(p["gold_chunk_index"])
    cfg = ledger_profile_config()
    tmp_store = GATE_DIR / "example_ledger"
    if tmp_store.exists():
        import shutil

        shutil.rmtree(tmp_store)
    cfg.update({"store_dir": str(tmp_store), "log_prompts": False})
    cfg["candidate_net"] = "gliner" if type(net).__name__ == "CandidateNet" else "regex"
    if cfg["candidate_net"] == "gliner":
        cfg["candidate_model"] = net.model_name
        cfg["candidate_device"] = net.device or "cpu"
    strategy = get_strategy(
        "hierarchical",
        branching_factor=5, active_entity_selector="word", max_entities=5, max_aliases=2, max_desc=3,
        history_l0_text="anchored", anchor_matching="strict", ungrounded_entities="keep", ledger=cfg,
    )
    strategy._ledger_runtime.net = net  # reuse the loaded model
    state = init_discourse_state(hierarchy_levels=[1])
    source = "an empty memory (cold start; run 0's construction records were not found under outputs/probe_bridge)"
    index_path = find_run0_cache_index()
    if index_path is not None:
        try:
            from saadi.indexing.retrieval_methods.mbr import _find_doc_entry, _load_doc_records, _read_index_payload

            payload = _read_index_payload(index_path)
            records = _load_doc_records(index_path=index_path, index_payload=payload, doc_entry=_find_doc_entry(payload, doc_id=book), doc_id=book)
            prev = next((r for r in records if int(r.chunk_index) == step - 1), None)
            if prev is not None:
                state = state_from_snapshot(prev.state_snapshot)
                source = f"run 0's memory and history after chunk {book}:{step - 1} (its records treated as ledger records)"
        except Exception as exc:  # the example must never block the gate
            report.unexpected.append(f"Could not restore run 0's state for the prompt example: {exc}")
    chunk = ChunkRecord(doc_id=book, chunk_id=f"{book}:{step}", chunk_index=step, raw_text=chunks[f"{book}:{step}"])
    prompt = strategy.build_prompt(chunk, state)
    report.numbers["example_prompt_source"] = source
    return prompt.system, prompt.user


def main(report: StepReport, argv: list[str]) -> None:
    from saadi.ledger.candidate_net import build_candidate_net

    GATE_DIR.mkdir(parents=True, exist_ok=True)
    spec_hash = verify_frozen_spec()
    report.add("Frozen spec", f"personas.jsonl SHA-256 verified: `{spec_hash}`")
    personas = load_personas()
    limit = int(arg_value(argv, "--limit", "0") or 0)

    cfg = ledger_profile_config()
    kind = arg_value(argv, "--candidate-net", cfg.get("candidate_net", "gliner"))
    model = arg_value(argv, "--model", cfg.get("candidate_model"))
    device = arg_value(argv, "--device", "cpu")  # the gate itself is a CPU step
    threshold = float(arg_value(argv, "--threshold", str(cfg.get("candidate_threshold", 0.5))))
    net_cfg = dict(cfg, candidate_net=kind, candidate_model=model, candidate_device=device, candidate_threshold=threshold)
    report.commands.append(f"candidate net: {kind} model={model} device={device} threshold={threshold} labels={cfg.get('candidate_labels')}")

    # Package check: the study venv must see `gliner` (installed by sh/01_gate.sh into P1_PYDEPS).
    versions = {}
    for mod in ("torch", "transformers", "gliner", "sentencepiece", "google.protobuf"):
        try:
            m = __import__(mod, fromlist=["__version__"])
            versions[mod] = str(getattr(m, "__version__", "installed"))
        except Exception as exc:
            versions[mod] = f"MISSING ({type(exc).__name__})"
    report.table("Python packages seen by the step", ["package", "version"], [[k, v] for k, v in versions.items()])
    if kind == "gliner" and any(v.startswith("MISSING") for k, v in versions.items() if k in ("torch", "transformers", "gliner")):
        raise RuntimeError(
            f"The candidate net needs torch, transformers and gliner; seen: {versions}. "
            "sh/01_gate.sh installs gliner into $P1_PYDEPS with --no-deps; torch/transformers must come from the study venv."
        )

    # Chunks: gold and reveal chunks of the 10 books, cut with the ingest pipeline's chunker.
    needed: dict[str, tuple[str, int]] = {}
    for p in personas:
        needed[p["gold_chunk_id"]] = (p["book_id"], int(p["gold_chunk_index"]))
        if p["cell"] in REVEAL_CELLS and p.get("reveal_chunk_id"):
            needed[p["reveal_chunk_id"]] = (p["book_id"], int(p["reveal_chunk_index"]))
    chunks: dict[str, str] = {}
    for book in sorted({b for b, _ in needed.values()}):
        texts = book_chunks(book)
        for chunk_id, (b, idx) in needed.items():
            if b == book:
                if idx >= len(texts):
                    raise RuntimeError(f"{chunk_id}: the modified book has only {len(texts)} chunks")
                chunks[chunk_id] = texts[idx]
    # Sanity: every inserted sentence is in its chunk.
    missing = [p["persona_id"] for p in personas if p["gold_sentence"] not in chunks[p["gold_chunk_id"]]]
    if missing:
        raise RuntimeError(f"Gold sentences not found in their chunks (chunking differs from the probe): {missing[:5]} ...")
    chunks_all = dict(chunks)
    if limit:
        chunks = dict(list(chunks.items())[:limit])
        report.unexpected.append(f"--limit {limit}: only the first {limit} chunks were run (dry run); the recall shares below are not the gate.")
    report.add("Chunks", f"{len(chunks)} distinct chunks ({len(personas)} gold, {sum(1 for p in personas if p['cell'] in REVEAL_CELLS)} reveal) from {len({b for b, _ in needed.values()})} books, chunker = `saadi.core.chunking.chunk_text` (tokenizer=None, 500, overlap 0).")

    # Load and run.
    net = build_candidate_net(net_cfg)
    t0 = time.time()
    net.load()
    load_s = time.time() - t0
    describe = net.describe()
    report.numbers["candidate_net"] = describe
    report.add("Candidate net loaded", f"`{describe['model']}` on `{describe.get('device')}` in {load_s:.1f} s; labels {describe['labels']}; threshold {describe.get('threshold')}; versions {describe.get('versions')}.")
    print(f"Running {describe['model']} on {len(chunks)} chunks ...", flush=True)
    results, t_total = run_net_on_chunks(net, chunks)
    write_jsonl(GATE_DIR / "candidates.jsonl", results.values())
    report.files_written.append(str(GATE_DIR / "candidates.jsonl"))

    # Recall of the inserted names.
    personas_run = [p for p in personas if p["gold_chunk_id"] in results] if limit else personas
    rec = recall_tables(personas_run, results)
    write_jsonl(GATE_DIR / "recall_rows.jsonl", rec["rows"])
    report.files_written.append(str(GATE_DIR / "recall_rows.jsonl"))
    cells = sorted(rec["per_cell"])
    report.table(
        f"Names caught by the candidate net, per cell (threshold {threshold})",
        ["cell", "label", "form 1 caught", "of which exact", "form 2 caught", "of which exact"],
        [
            [
                c,
                CELL_LABELS.get(c, ""),
                pct(rec["per_cell"][c].get("form1_caught", 0), rec["per_cell"][c].get("form1_n", 0)),
                rec["per_cell"][c].get("form1_exact", 0),
                pct(rec["per_cell"][c].get("form2_caught", 0), rec["per_cell"][c].get("form2_n", 0)) if rec["per_cell"][c].get("form2_n") else "-",
                rec["per_cell"][c].get("form2_exact", 0),
            ]
            for c in cells
        ],
    )
    fams = sorted(rec["per_family"])
    fam_names = {"A": "A surname", "B": "B nickname", "C": "C occupation", "D": "D owned object"}
    report.table(
        "Names caught per family",
        ["family", "form 1 caught", "form 2 caught"],
        [[fam_names.get(f, f), pct(rec["per_family"][f].get("form1_caught", 0), rec["per_family"][f].get("form1_n", 0)), pct(rec["per_family"][f].get("form2_caught", 0), rec["per_family"][f].get("form2_n", 0)) if rec["per_family"][f].get("form2_n") else "-"] for f in fams],
    )
    tot = rec["total"]
    f1_rate = tot.get("form1_caught", 0) / max(1, tot.get("form1_n", 0))
    f2_rate = tot.get("form2_caught", 0) / max(1, tot.get("form2_n", 0))
    report.numbers.update({"form1_caught": tot.get("form1_caught", 0), "form1_n": tot.get("form1_n", 0), "form2_caught": tot.get("form2_caught", 0), "form2_n": tot.get("form2_n", 0), "form1_rate": f1_rate, "form2_rate": f2_rate, "per_cell": rec["per_cell"], "per_family": rec["per_family"]})
    verdict = f"Form 1: {pct(tot.get('form1_caught', 0), tot.get('form1_n', 0))}; form 2: {pct(tot.get('form2_caught', 0), tot.get('form2_n', 0))}. 'Caught' = a candidate equals the name after title stripping (exact) or contains it (partial, e.g. 'Doctor Sinchipuma' for 'Sinchipuma')."
    missed = [r for r in rec["rows"] if not r["form1_caught"]]
    if missed:
        verdict += f"\n\nForm-1 names missed ({len(missed)}): " + ", ".join(f"{r['persona_id']} '{r['form1']}'" for r in missed[:40]) + (" ..." if len(missed) > 40 else "")
    report.add("Totals", verdict)

    # The informational second threshold (same spans at a lower score floor cannot be
    # derived from the 0.5 run, so it is a second pass; it changes nothing in the profile).
    if "--no-second-threshold" not in argv and kind == "gliner" and abs(threshold - SECOND_THRESHOLD) > 1e-9:
        net2 = build_candidate_net(dict(net_cfg, candidate_threshold=SECOND_THRESHOLD))
        net2._model, net2.device = net._model, net.device  # same weights
        results2, _t2 = run_net_on_chunks(net2, chunks)
        rec2 = recall_tables(personas_run, results2)
        tot2 = rec2["total"]
        report.add(
            f"Informational: the same model at threshold {SECOND_THRESHOLD}",
            f"Form 1: {pct(tot2.get('form1_caught', 0), tot2.get('form1_n', 0))}; form 2: {pct(tot2.get('form2_caught', 0), tot2.get('form2_n', 0))}; "
            f"candidates per chunk {statistics.mean(len(r['candidates']) for r in results2.values()):.1f} (against {statistics.mean(len(r['candidates']) for r in results.values()):.1f} at {threshold}). "
            "Not used; the profile keeps its threshold unless the gate fails.",
        )
        report.numbers["second_threshold"] = {"threshold": SECOND_THRESHOLD, "form1_rate": tot2.get("form1_caught", 0) / max(1, tot2.get("form1_n", 0)), "form2_rate": tot2.get("form2_caught", 0) / max(1, tot2.get("form2_n", 0))}

    # Candidates per chunk, labels, false-candidate proxy, timing.
    n_c = [len(r["candidates"]) for r in results.values()]
    labels = Counter(c["label"] for r in results.values() for c in r["candidates"])
    false_n = sum(1 for r in results.values() for c in r["candidates"] if false_proxy(c["text"], c["label"]))
    dropped = Counter(d["reason"] for r in results.values() for d in r["dropped"])
    secs = [r["seconds"] for r in results.values()]
    report.table(
        "Candidates per chunk",
        ["measure", "value"],
        [
            ["chunks", len(results)],
            ["candidates per chunk, mean / median / max", f"{statistics.mean(n_c):.1f} / {statistics.median(n_c):.0f} / {max(n_c)}"],
            ["raw spans per chunk before the filter, mean", f"{statistics.mean(r['raw_count'] for r in results.values()):.1f}"],
            ["chunks where the candidate cap was hit", sum(1 for r in results.values() if r["cap_hit"])],
            ["false-candidate proxy per chunk (common word / title / no letters)", f"{false_n / max(1, len(results)):.2f} ({false_n} candidates)"],
            ["candidates per label", dict(labels.most_common())],
            ["spans dropped by the filter, by reason", dict(dropped.most_common())],
            ["seconds per chunk on " + str(describe.get("device")) + ", mean / median / max", f"{statistics.mean(secs):.2f} / {statistics.median(secs):.2f} / {max(secs):.2f}"],
            ["total seconds for all chunks", f"{t_total:.0f}"],
            ["requirement (spec D): under one second per chunk", "met" if statistics.mean(secs) < 1.0 else "NOT met on this device"],
        ],
    )
    report.numbers.update({"seconds_per_chunk_mean": statistics.mean(secs), "candidates_per_chunk_mean": statistics.mean(n_c), "false_proxy_per_chunk": false_n / max(1, len(results)), "labels": dict(labels)})
    examples = [(r["chunk_id"], [c["text"] for c in r["candidates"]]) for r in list(results.values())[:3]]
    report.add("Three chunks and their candidates", "\n".join(f"- **{cid}**: {cands}" for cid, cands in examples))

    # Optional model comparison on the same chunks.
    compare = arg_value(argv, "--compare-models")
    if compare:
        rows = []
        for alt in [m.strip() for m in compare.split(",") if m.strip()]:
            try:
                alt_net = build_candidate_net(dict(net_cfg, candidate_model=alt))
                t1 = time.time()
                alt_net.load()
                alt_results, alt_t = run_net_on_chunks(alt_net, chunks)
                alt_rec = recall_tables(personas_run, alt_results)
                at = alt_rec["total"]
                rows.append([alt, pct(at.get("form1_caught", 0), at.get("form1_n", 0)), pct(at.get("form2_caught", 0), at.get("form2_n", 0)), f"{statistics.mean(len(r['candidates']) for r in alt_results.values()):.1f}", f"{alt_t / max(1, len(alt_results)):.2f}", f"{time.time() - t1:.0f}"])
                write_jsonl(GATE_DIR / f"candidates__{alt.replace('/', '__')}.jsonl", alt_results.values())
            except Exception as exc:
                rows.append([alt, "failed", str(exc)[:120], "-", "-", "-"])
        report.table("Other models on the same chunks (--compare-models)", ["model", "form 1 caught", "form 2 caught", "candidates/chunk", "s/chunk", "total s incl. load"], rows)

    # Models considered.
    report.table("Candidate-net models considered (Hugging Face, 8 Nov 2026)", ["model", "what it is", "zero-shot NER as the card states it", "assessment"], ALTERNATIVES)
    chosen = f"Chosen: `{describe['model']}`. " + ALTERNATIVES_NOTE
    report.add("Model choice", chosen)

    # The exact prompt.
    system, user = example_prompt(report, net, personas, chunks_all)
    (GATE_DIR / "system_prompt.txt").write_text(system + "\n", encoding="utf-8")
    (GATE_DIR / "example_user_prompt.txt").write_text(user + "\n", encoding="utf-8")
    report.files_written += [str(GATE_DIR / "system_prompt.txt"), str(GATE_DIR / "example_user_prompt.txt")]
    report.add("Exact system prompt of the LLM call (verbatim)", "```text\n" + system + "\n```")
    # The hierarchy JSON is long; it is shown shortened here and in full in the file.
    marker = "[HIERARCHY_CONTEXT_JSON]\n"
    shown = user
    if marker in user:
        head, rest = user.split(marker, 1)
        hier, tail = rest.split("\n\n[", 1)
        if len(hier) > 700:
            hier = hier[:700] + f" ... [{len(hier) - 700} more characters of history JSON; full text in example_user_prompt.txt]"
        shown = head + marker + hier + "\n\n[" + tail
    report.add(
        f"One filled user prompt (B31, gold chunk of persona c1_B31_000; recalled groups from {report.numbers.get('example_prompt_source')})",
        "```text\n" + shown + "\n```",
    )
    report.add(
        "Reading the prompt",
        "Interpretation recorded for the hand-off: the spec's output list ends with 'the chunk summary as now'. The current "
        "pipeline has no per-chunk summary in the entity call (the summaries are the separate fold calls, unchanged here), so the "
        "output is the decision block, the extraction JSON, then [END_JSON]; no summary is added to the call.",
    )

    # Verdict and hand-off.
    if f1_rate < RECALL_FLOOR and not limit:
        report.unexpected.append(
            f"**Form-1 recall is {100 * f1_rate:.0f}%, below the 80% floor of spec D.** Proposal before any GPU time: rerun this gate with "
            "`--compare-models knowledgator/gliner-multitask-large-v0.5,gliner-community/gliner_large-v2.5` and, if one is clearly better, "
            "set candidate_model in configs/profiles/p1_ledger.yaml to it; otherwise lower candidate_threshold to 0.3 (see the informational "
            "pass above) or combine two models (union of spans). Step 2 should not start until this is decided."
        )
        report.numbers["gate_passed"] = False
    else:
        report.numbers["gate_passed"] = True
    report.add("Hand-off", "Paste this report back. Step 2 (the GPU build) waits for approval of the model choice and the prompt text.")


if __name__ == "__main__":
    raise SystemExit(run_step(1, "Candidate-net gate (CPU)", main, requires=[]))
