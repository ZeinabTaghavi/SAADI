#!/usr/bin/env python3
"""Step 2 (the GPU step): index the 10 probe books with profile p1_ledger.

Runs the study runner (scripts/run_study.py --profile p1_ledger) exactly as run 0
ran it with --profile published: same suite (probe_bridge: temperature 0, seed
42), same dataset files, same model runtime, tensor parallel 4 on GPUs 0-3,
its own vLLM port, its own output root (outputs/method_update/p1_ledger).
Resumable per book: the construction cache saves as it goes and the ledger
event log is truncated to the regenerated step; rerun the script to continue.

Report: books finished; calls per book (must equal run 0's); mean input and
output tokens per call next to run 0; decision-block parse failures;
ledger-recall cap hits; records and stubs per book; promotions; wall clock.
Flags: --force, --dry-run (resolve the configuration without a GPU; keeps
step 3 locked), --tensor-parallel-size N, --gpus 0,1,2,3.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import (  # noqa: E402
    LEDGER_DIR,
    N_BOOKS,
    OUT_DIR,
    PROBE_CONFIG,
    PROFILE,
    ROOT,
    RUN0_OUT_DIR,
    StepReport,
    arg_value,
    find_run0_cache_index,
    read_json,
    run_step,
    verify_frozen_spec,
    write_json,
)

INVOCATIONS_JSON = OUT_DIR / "build_invocations.json"
RUN0_TOKENS_JSON = OUT_DIR / "run0_tokens.json"
LLM_MODEL_ID = os.environ.get("P1_LLM_MODEL_ID", "Qwen/Qwen3-30B-A3B-Instruct-2507")


def _hf_model_dir(hub: Path, model_id: str) -> Path:
    return hub / ("models--" + model_id.replace("/", "--"))


def resolve_configuration(report: StepReport) -> dict:
    """Resolve the settings exactly as run_study.py will and refuse anything but the fixed ones."""

    sys.path.insert(0, str(ROOT / "scripts"))
    import run_study
    from saadi.config.loader import load_config
    from saadi.config.validation import validate_experiment

    os.environ["SADI_OUTPUT_DIR"] = str(OUT_DIR)
    suite = load_config(ROOT / "configs" / "suites" / "probe_bridge.yaml")
    resolved = run_study.merge(run_study.profile_config(load_config(PROBE_CONFIG), PROFILE), suite["variants"][0]["overrides"])
    validate_experiment(resolved)
    decoding = (resolved["model"]["generate"]["temperature"], resolved["model"]["sampling"].get("seed"))
    if decoding != (0.0, 42):
        raise RuntimeError(f"Expected temperature 0 and seed 42, resolved {decoding}.")
    strategy = resolved["ingest"]["strategy_kwargs"]
    if strategy.get("ungrounded_entities") != "keep":
        raise RuntimeError(f"Expected ungrounded_entities=keep (run 0), resolved {strategy.get('ungrounded_entities')!r}.")
    ledger = strategy.get("ledger")
    if not isinstance(ledger, dict):
        raise RuntimeError("The resolved configuration has no ledger mapping; the profile was not applied.")
    if Path(ledger["store_dir"]).resolve() != LEDGER_DIR.resolve():
        raise RuntimeError(f"ledger.store_dir resolved to {ledger['store_dir']}, expected {LEDGER_DIR}.")
    if resolved["model"]["model_id"] != LLM_MODEL_ID:
        raise RuntimeError(f"The construction LLM must stay {LLM_MODEL_ID}; resolved {resolved['model']['model_id']}.")
    rows = [
        ["construction LLM", resolved["model"]["model_id"]],
        ["temperature / seed", f"{decoding[0]} / {decoding[1]}"],
        ["tensor parallel (config)", resolved["model"].get("tensor_parallel_size")],
        ["memory selector (overlap slots)", strategy.get("active_entity_selector")],
        ["history text / anchor matching / ungrounded", f"{strategy.get('history_l0_text')} / {strategy.get('anchor_matching')} / {strategy.get('ungrounded_entities')}"],
        ["budget shown (records, aliases, descriptions)", f"{strategy.get('max_entities')}, {strategy.get('max_aliases')}, {strategy.get('max_desc')}"],
        ["schedule / wavefront batch", f"{resolved['ingest'].get('schedule')} / {resolved['ingest'].get('wavefront_batch_size')}"],
    ] + [[f"ledger.{k}", v] for k, v in ledger.items()]
    report.table("Resolved construction settings (profile p1_ledger = published + ledger)", ["setting", "value"], rows)
    report.numbers["resolved"] = {"temperature": decoding[0], "seed": decoding[1], "strategy_kwargs": strategy, "model_id": resolved["model"]["model_id"]}
    return resolved


def environment_table(report: StepReport, ledger_cfg: dict, gpus: str | None) -> None:
    hub = Path(os.environ.get("HF_HUB_CACHE", ""))
    llm_dir = _hf_model_dir(hub, LLM_MODEL_ID)
    net_dir = _hf_model_dir(hub, str(ledger_cfg.get("candidate_model", "")))
    try:
        import gliner  # noqa: F401

        gliner_ok = f"yes ({gliner.__version__})"
    except Exception as exc:
        gliner_ok = f"NO ({exc})"
    rows = [
        ["SADI_OUTPUT_DIR", os.environ.get("SADI_OUTPUT_DIR")],
        ["HF_HOME / HF_HUB_CACHE", f"{os.environ.get('HF_HOME')} / {hub}"],
        ["construction LLM weights in the cache", llm_dir.is_dir()],
        ["candidate net weights in the cache", net_dir.is_dir()],
        ["gliner importable", gliner_ok],
        ["CUDA_VISIBLE_DEVICES", gpus or os.environ.get("CUDA_VISIBLE_DEVICES", "")],
        ["VLLM_PORT (this build)", os.environ.get("VLLM_PORT", "")],
        ["VLLM_USE_FLASHINFER_SAMPLER / HF_HUB_OFFLINE", f"{os.environ.get('VLLM_USE_FLASHINFER_SAMPLER', '')} / {os.environ.get('HF_HUB_OFFLINE', '')}"],
    ]
    report.table("Runtime environment", ["setting", "value"], rows)
    if not llm_dir.is_dir():
        report.unexpected.append(f"The construction LLM weights are not under {hub}; set PROBE_HF_HOME to the cache that holds them (the run would otherwise download).")
    if not net_dir.is_dir():
        report.unexpected.append(f"The candidate net is not in {hub}; run step 1 first (it downloads the model), or the build will fail offline.")
    if gliner_ok.startswith("NO"):
        report.unexpected.append("gliner is not importable in this environment; step 1 installs it into $P1_PYDEPS.")
    visible = gpus or os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible != "0,1,2,3":
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES is {visible!r}; this build may only use GPUs 0,1,2,3 (pass --gpus 0,1,2,3).")


def scan_cache(report: StepReport) -> tuple[int, dict[str, dict]]:
    from saadi.indexing.retrieval_methods.mbr import _resolve_cache_file_path

    books: dict[str, dict] = {}
    try:
        index_path = common.find_run1_cache_index()
    except FileNotFoundError:
        report.add("Construction records", "No cache index found yet.")
        return 0, books
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    finished = 0
    for doc in payload.get("docs", []):
        doc_id = str(doc.get("doc_id"))
        files = doc.get("files", {}) or {}
        steps_path = _resolve_cache_file_path(files.get("steps_index", ""), base_dir=index_path.parent)
        memory_path = _resolve_cache_file_path(files.get("memory", ""), base_dir=index_path.parent)
        n_steps = not_ok = 0
        out_chars: list[int] = []
        if Path(steps_path).exists():
            for line in open(steps_path, encoding="utf-8"):
                row = json.loads(line)
                n_steps += 1
                if row.get("parser_status") != "ok":
                    not_ok += 1
                out_chars.append(len(row.get("raw_model_text", "")))
        done = Path(memory_path).exists()
        finished += int(done)
        books[doc_id] = {"chunk_records": n_steps, "parser_not_ok": not_ok, "finished": done, "reply_chars": out_chars, "memory_path": str(memory_path), "steps_path": str(steps_path)}
    report.numbers["cache_index"] = str(index_path)
    return finished, books


def run0_chunk_records() -> dict[str, int]:
    index_path = find_run0_cache_index()
    if index_path is None:
        return {}
    from saadi.indexing.retrieval_methods.mbr import _resolve_cache_file_path

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    out = {}
    for doc in payload.get("docs", []):
        steps_path = _resolve_cache_file_path((doc.get("files") or {}).get("steps_index", ""), base_dir=index_path.parent)
        if Path(steps_path).exists():
            out[str(doc.get("doc_id"))] = sum(1 for _ in open(steps_path, encoding="utf-8"))
    return out


def load_tokenizer():
    try:
        from saadi.llm.hf_tokenizers import load_hf_tokenizer

        return load_hf_tokenizer(LLM_MODEL_ID, trust_remote_code=True)
    except Exception as exc:
        return exc


def count_tokens(tok, text: str) -> int:
    if tok is None or isinstance(tok, Exception):
        return max(1, len(text) // 4)  # a stated approximation when the tokenizer is unavailable
    return len(tok(text, add_special_tokens=False)["input_ids"])


def run1_tokens(tok, books: dict[str, dict]) -> dict[str, dict]:
    """Input tokens from the logged prompts (system + user text), output tokens from the replies."""

    from saadi.ledger.ledger import LedgerStore

    store = LedgerStore(LEDGER_DIR)
    system = (LEDGER_DIR / "system_prompt.txt").read_text(encoding="utf-8") if (LEDGER_DIR / "system_prompt.txt").exists() else ""
    sys_tokens = count_tokens(tok, system)
    out: dict[str, dict] = {}
    for doc_id in books:
        events = store.read_events(doc_id)
        inputs, outputs = [], []
        for ev in events:
            user = ev.get("prompt_user")
            inputs.append(sys_tokens + (count_tokens(tok, user) if user is not None else max(1, int(ev.get("prompt_user_chars", 0)) // 4)))
        truncated = 0
        for line in open(books[doc_id]["steps_path"], encoding="utf-8"):
            row = json.loads(line)
            n_out = count_tokens(tok, row.get("raw_model_text", ""))
            outputs.append(n_out)
            # The reply hit the fixed max_tokens (2048) and its JSON did not parse: the decision block
            # left no room for the extraction block. Counted so the cost of the format is visible.
            if n_out >= 2000 and row.get("parser_status") == "failed":
                truncated += 1
        out[doc_id] = {"input_mean": statistics.mean(inputs) if inputs else float("nan"), "output_mean": statistics.mean(outputs) if outputs else float("nan"), "n": len(events), "likely_truncated": truncated}
    return out


def run0_tokens(tok) -> dict[str, dict]:
    """Rebuild run 0's prompts from its saved records (published strategy, state of the previous step) and count."""

    if RUN0_TOKENS_JSON.exists():
        return read_json(RUN0_TOKENS_JSON)
    index_path = find_run0_cache_index()
    if index_path is None:
        return {}
    from saadi.context_strategies import get_strategy
    from saadi.core.state import init_discourse_state, state_from_snapshot
    from saadi.core.types import ChunkRecord
    from saadi.indexing.retrieval_methods.mbr import _load_doc_records, _read_index_payload

    payload = _read_index_payload(index_path)
    strategy = get_strategy("hierarchical", branching_factor=5, active_entity_selector="word", max_entities=5, max_aliases=2, max_desc=3, history_l0_text="anchored", anchor_matching="strict", ungrounded_entities="keep")
    out: dict[str, dict] = {}
    for doc in payload.get("docs", []):
        doc_id = str(doc.get("doc_id"))
        records = _load_doc_records(index_path=index_path, index_payload=payload, doc_entry=doc, doc_id=doc_id)
        inputs, outputs = [], []
        state = init_discourse_state(hierarchy_levels=[1])
        for i, rec in enumerate(records):
            if i > 0:
                state = state_from_snapshot(records[i - 1].state_snapshot)
            chunk = ChunkRecord(doc_id=doc_id, chunk_id=rec.chunk_id, chunk_index=rec.chunk_index, raw_text=rec.raw_chunk_text)
            prompt = strategy.build_prompt(chunk, state)
            inputs.append(count_tokens(tok, prompt.system) + count_tokens(tok, prompt.user))
            outputs.append(count_tokens(tok, rec.raw_model_text or ""))
        out[doc_id] = {"input_mean": statistics.mean(inputs), "output_mean": statistics.mean(outputs), "n": len(records)}
        print(f"  run 0 tokens rebuilt for {doc_id}: {len(records)} prompts", flush=True)
    write_json(RUN0_TOKENS_JSON, out)
    return out


def run0_wall_clock_hours() -> str:
    path = RUN0_OUT_DIR.parent.parent / "experiments" / "probe_bridge" / "reports" / "step_05.md"
    path = ROOT / "experiments" / "probe_bridge" / "reports" / "step_05.md"
    if path.exists():
        m = re.search(r"([\d.]+) hours \((\d+) s\)", path.read_text(encoding="utf-8"))
        if m:
            return f"{m.group(1)} h"
    return "n/a"


def ledger_stats(books: dict[str, dict]) -> dict[str, dict]:
    from saadi.ledger.ledger import LedgerStore
    from saadi.ledger.measure import memory_json_drift

    store = LedgerStore(LEDGER_DIR)
    out: dict[str, dict] = {}
    for doc_id, info in books.items():
        ledger = store.doc(doc_id)
        kinds = Counter(e.kind for e in ledger.entries.values())
        c = ledger.counters
        events = ledger.events
        drift = []
        if info.get("finished") and Path(info["memory_path"]).exists():
            drift = memory_json_drift(json.loads(Path(info["memory_path"]).read_text(encoding="utf-8")).get("entities", []), ledger)
        out[doc_id] = {
            "events": len(events),
            "records": kinds.get("record", 0),
            "stubs": kinds.get("stub", 0),
            "promotions": c.get("promotions", 0),
            "aliases_added": c.get("aliases_added", 0),
            "same_as": c.get("same_as", 0),
            "same_as_unresolved": c.get("same_as_unresolved", 0),
            "not_an_entity": c.get("not_an_entity", 0),
            "new_stub": c.get("new_stub", 0),
            "llm_additions": c.get("llm_additions", 0),
            "decision_parse_failures": c.get("decision_block_parse_failures", 0),
            "omitted_candidates": c.get("omitted_candidates", 0),
            "description_replacements": c.get("description_replacements", 0),
            "cut_by_budget": c.get("cut_by_budget", 0),
            "ledger_cap_hits": sum(1 for e in events if e.get("ledger_cap_hit")),
            "ledger_left_out": sum(len(e.get("ledger_left_out_ids", [])) for e in events),
            "candidates_per_chunk": statistics.mean(len(e.get("candidates", [])) for e in events) if events else float("nan"),
            "candidate_seconds_mean": statistics.mean(float(e.get("candidate_seconds", 0.0)) for e in events) if events else float("nan"),
            "candidate_device": Counter(str(e.get("candidate_device", "")) for e in events).most_common(1)[0][0] if events else "",
            "parser_json_not_ok": sum(1 for e in events if e.get("parser_status") != "ok"),
            "memory_json_drift": len(drift),
        }
    return out


def main(report: StepReport, argv: list[str]) -> None:
    dry = "--dry-run" in argv
    tp = arg_value(argv, "--tensor-parallel-size", os.environ.get("P1_TENSOR_PARALLEL", "4"))
    gpus = arg_value(argv, "--gpus", os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3"))
    spec_hash = verify_frozen_spec()
    report.add("Frozen spec", f"personas.jsonl SHA-256 verified: `{spec_hash}` (same spec as run 0).")
    if not (ROOT / "outputs" / "saadi" / "reference" / "manifest.json").exists():
        raise RuntimeError("outputs/saadi/reference/manifest.json is missing; the published base of the profile needs it.")
    resolved = resolve_configuration(report)
    ledger_cfg = resolved["ingest"]["strategy_kwargs"]["ledger"]
    environment_table(report, ledger_cfg, gpus)

    cmd = [sys.executable, str(ROOT / "scripts" / "run_study.py"), "--suite", "probe_bridge", "--profile", PROFILE, "--datasets", "novelhopqa_probe", "--retrievers", "bm25", "--stage", "ingest", "--tensor-parallel-size", str(tp)]
    if dry:
        cmd.append("--dry-run")
    env = dict(os.environ)
    env["SADI_OUTPUT_DIR"] = str(OUT_DIR)
    env["CUDA_VISIBLE_DEVICES"] = gpus
    report.commands.append(" ".join([f"SADI_OUTPUT_DIR={OUT_DIR}", f"CUDA_VISIBLE_DEVICES={gpus}", f"VLLM_PORT={env.get('VLLM_PORT', '')}"] + cmd))

    invocations = read_json(INVOCATIONS_JSON) if INVOCATIONS_JSON.exists() else []
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, env=env)
    elapsed = time.time() - t0
    if not dry:
        invocations.append({"started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)), "seconds": round(elapsed), "returncode": proc.returncode})
        write_json(INVOCATIONS_JSON, invocations)
        report.files_written.append(str(INVOCATIONS_JSON))
    total = sum(x["seconds"] for x in invocations)
    report.add("Wall clock", f"{elapsed / 3600:.2f} h ({elapsed:.0f} s) this invocation; {total / 3600:.2f} h over {len(invocations)} invocation(s). Run 0: {run0_wall_clock_hours()} in one invocation.")
    report.numbers.update({"wall_clock_s_this": elapsed, "wall_clock_s_total": total, "invocations": invocations})

    finished, books = scan_cache(report)
    if dry:
        report.skip_marker = True
        report.add("Dry run", "Configuration resolved only; step 3 stays locked.")
        return
    if proc.returncode != 0:
        report.unexpected.append(f"run_study.py exited with code {proc.returncode}. Rerun this script to resume; the construction cache and the ledger log are kept.")

    stats = ledger_stats(books)
    run0_calls = run0_chunk_records()
    rows = []
    calls_mismatch = []
    for doc_id in sorted(books, key=lambda d: books[d]["chunk_records"]):
        b, s = books[doc_id], stats[doc_id]
        r0 = run0_calls.get(doc_id)
        if b["finished"] and r0 is not None and r0 != b["chunk_records"]:
            calls_mismatch.append(f"{doc_id}: {b['chunk_records']} vs run 0 {r0}")
        rows.append([doc_id, "finished" if b["finished"] else "in progress", b["chunk_records"], r0 if r0 is not None else "n/a", s["records"], s["stubs"], s["promotions"], s["aliases_added"], s["decision_parse_failures"], s["ledger_cap_hits"], b["parser_not_ok"]])
    report.table(
        "Books: calls, records and stubs",
        ["book", "state", "calls (chunk records)", "run 0 calls", "records", "stubs", "promotions", "aliases added", "decision parse failures", "ledger cap hits", "JSON parser not ok"],
        rows,
    )
    if calls_mismatch:
        report.unexpected.append("Calls per book differ from run 0 (the chunking must be identical): " + "; ".join(calls_mismatch))
    tot = Counter()
    for s in stats.values():
        for k, v in s.items():
            if isinstance(v, (int, float)) and k not in ("candidates_per_chunk", "candidate_seconds_mean"):
                tot[k] += v
    n_events = sum(s["events"] for s in stats.values()) or 1
    report.table(
        "Ledger totals over the books built so far",
        ["measure", "value"],
        [
            ["calls (events)", n_events],
            ["records / stubs in the final ledgers", f"{tot['records']} / {tot['stubs']}"],
            ["promotions (stub -> record)", tot["promotions"]],
            ["SAME AS decisions / unresolved targets", f"{tot['same_as']} / {tot['same_as_unresolved']}"],
            ["NEW -> stub / LLM additions (records the net did not propose)", f"{tot['new_stub']} / {tot['llm_additions']}"],
            ["NOT AN ENTITY decisions", tot["not_an_entity"]],
            ["decision-block parse failures (all candidates -> stubs) / omitted candidates", f"{tot['decision_parse_failures']} / {tot['omitted_candidates']}"],
            ["ledger-recall cap hits (chunks) / entries left out", f"{tot['ledger_cap_hits']} / {tot['ledger_left_out']}"],
            ["description replacements (cap 5)", tot["description_replacements"]],
            ["records proposed beyond the budget of 5 (cut, as today)", tot["cut_by_budget"]],
            ["candidates per chunk (mean)", f"{statistics.mean(s['candidates_per_chunk'] for s in stats.values() if s['events']):.1f}" if any(s["events"] for s in stats.values()) else "-"],
            ["candidate-net seconds per chunk (mean) / device", f"{statistics.mean(s['candidate_seconds_mean'] for s in stats.values() if s['events']):.3f} / {Counter(s['candidate_device'] for s in stats.values()).most_common(1)[0][0]}" if any(s["events"] for s in stats.values()) else "-"],
            ["JSON parser status not ok (fallback or failed)", tot["parser_json_not_ok"]],
            ["memory.json entries holding a replaced description (should be 0)", tot["memory_json_drift"]],
        ],
    )
    report.numbers.update({"books": {d: {k: v for k, v in b.items() if k != "reply_chars"} for d, b in books.items()}, "ledger_stats": stats, "run0_calls": run0_calls, "books_finished": finished})

    # Tokens per call next to run 0.
    tok = load_tokenizer()
    if isinstance(tok, Exception):
        report.unexpected.append(f"The Qwen tokenizer could not be loaded offline ({tok}); token counts below are characters/4.")
    r1 = run1_tokens(tok, books)
    try:
        r0 = run0_tokens(tok)
    except Exception as exc:
        report.unexpected.append(f"Run 0 prompts could not be rebuilt for token counts: {exc}")
        r0 = {}
    rows = []
    for doc_id in sorted(books, key=lambda d: books[d]["chunk_records"]):
        a, b = r1.get(doc_id, {}), r0.get(doc_id, {})
        rows.append([doc_id, f"{a.get('input_mean', float('nan')):.0f}", f"{b.get('input_mean', float('nan')):.0f}" if b else "n/a", f"{a.get('output_mean', float('nan')):.0f}", f"{b.get('output_mean', float('nan')):.0f}" if b else "n/a"])
    if r1:
        all_in1 = statistics.mean(v["input_mean"] for v in r1.values() if v["n"])
        all_out1 = statistics.mean(v["output_mean"] for v in r1.values() if v["n"])
        all_in0 = statistics.mean(v["input_mean"] for v in r0.values()) if r0 else float("nan")
        all_out0 = statistics.mean(v["output_mean"] for v in r0.values()) if r0 else float("nan")
        rows.append(["**all books (mean of book means)**", f"**{all_in1:.0f}**", f"**{all_in0:.0f}**" if r0 else "n/a", f"**{all_out1:.0f}**", f"**{all_out0:.0f}**" if r0 else "n/a"])
        n_trunc = sum(v.get("likely_truncated", 0) for v in r1.values())
        report.add("Replies at the 2048-token cap with an unparsed JSON block (likely truncated)", f"{n_trunc} of {sum(v['n'] for v in r1.values())} calls. The output cap is the published one and was not changed.")
        report.numbers["likely_truncated_replies"] = n_trunc
        report.numbers["tokens"] = {"run1": r1, "run0": r0, "run1_input_mean": all_in1, "run1_output_mean": all_out1, "run0_input_mean": all_in0, "run0_output_mean": all_out0}
    report.table(
        "Tokens per call (prompt text system+user, and the reply; Qwen tokenizer, no chat template)",
        ["book", "input run 1", "input run 0", "output run 1", "output run 0"],
        rows,
    )
    report.add("Where run 0's token counts come from", "Run 0 did not log its prompts; they are rebuilt from its saved records (published strategy on the state after the previous step), cached in `run0_tokens.json`.")

    if finished < N_BOOKS or proc.returncode != 0:
        report.skip_marker = True
        report.unexpected.append(f"Only {finished}/{N_BOOKS} books have a final memory.json. Rerun `sh/02_build.sh` to resume; the done marker is withheld until all books finish.")
        return
    if (LEDGER_DIR / "system_prompt.txt").exists():
        report.files_written.append(str(LEDGER_DIR / "system_prompt.txt"))
    report.files_written += [report.numbers["cache_index"], str(LEDGER_DIR) + "/<book>/events.jsonl, ledger.json"]
    report.add("Hand-off", "All books are indexed. Paste this report back before step 3.")


if __name__ == "__main__":
    raise SystemExit(run_step(2, "Index with profile p1_ledger (GPU)", main, requires=[1]))
