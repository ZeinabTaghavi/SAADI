#!/usr/bin/env python3
"""C1: Contextual Retrieval baseline on SAADI's four benchmarks (ICLR rebuttal).

Started by experiments/rebuttal/C1_contextual_retrieval/run_gpus_0123.sh (--group A: LooGLE,
then NovelHopQA, GPUs 0-3) and run_gpus_4567.sh (--group B: MuSiQue-32K, then QASPER-64K,
GPUs 4-7). From the repository root:

    python scripts/rebuttal/c1_contextual_retrieval.py --group A
    python scripts/rebuttal/c1_contextual_retrieval.py --group A --dry-run       # CPU: 2 documents per dataset, BM25, stub contexts
    python scripts/rebuttal/c1_contextual_retrieval.py --group A --dry-run-full  # CPU: all documents, BM25, stub contexts
    python scripts/rebuttal/c1_contextual_retrieval.py --report-only             # rebuild every report from what exists

Per group: preflight (env, commit, weights, vLLM, disk, GPUs free) -> token census per dataset
-> run-time estimate -> memory check of every embedding model -> per dataset: contexts (vLLM,
TP 4, then the engine exits) -> retrieval jobs (dense encoders one per GPU, Qwen on two, BM25
on CPU) -> hybrid and complementarity check (CPU) -> per-dataset report. The final report is
written by whichever group finishes last (file lock). Resumable at every step.
Exit code 0 = everything done, 1 = a step failed (its report is still written), 2 = preflight failed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from saadi.rebuttal import contextual_retrieval as cr  # noqa: E402
from saadi.rebuttal import heldout_fusion as hf  # noqa: E402
from saadi.rebuttal.frozen_records import DATASETS, reference_dir  # noqa: E402


def _load_c2_runner():
    """C2's runner, for its GPU check, file lock and embedding cost model (imported, not copied)."""

    path = REPO_ROOT / "scripts" / "rebuttal" / "c2_heldout_fusion.py"
    spec = importlib.util.spec_from_file_location("c2_runner", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C2 = _load_c2_runner()
GPU_BUSY_MIB = C2.GPU_BUSY_MIB
DRY_MAX_DOCS = 2
VLLM_PORT = {"A": 29610, "B": 29710}
PLACEHOLDER_CONTEXT_TOKENS = cr.CONTEXT_MAX_TOKENS
MEMORY_MARGIN = 0.10
CONTEXT_TOKENS_ESTIMATE = 135  # B2's median context length (same model)
DISK_NEEDED_GB = 20


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def make_log(path: Path | None):
    def log(msg: str) -> None:
        line = f"[{now()}] {msg}"
        print(line, flush=True)

    return log


def fmt_s(seconds: float | None) -> str:
    return "—" if seconds is None else C2.fmt_s(seconds)


def paths_for(args) -> cr.C1Paths:
    tag = "dry_run_full" if args.dry_run_full else ("dry_run" if args.dry_run else "")
    return cr.C1Paths(REPO_ROOT, tag=tag)


def max_docs_for(args) -> int | None:
    if args.max_docs is not None:
        return args.max_docs or None
    return DRY_MAX_DOCS if args.dry_run else None


def retrievers_for(args) -> tuple[str, ...]:
    return ("bm25",) if (args.dry_run or args.dry_run_full) else cr.RETRIEVERS


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def git_state() -> dict[str, Any]:
    def run(*a: str) -> str:
        try:
            return subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception as exc:
            return f"<git failed: {exc}>"

    status = run("status", "--porcelain", "--", "src/saadi", "scripts/rebuttal", "experiments/rebuttal/C1_contextual_retrieval/env.sh",
                 "experiments/rebuttal/C1_contextual_retrieval/run_gpus_0123.sh", "experiments/rebuttal/C1_contextual_retrieval/run_gpus_4567.sh",
                 "experiments/probe_baselines/systems/b2_contextual/build.py")
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"), "code_changes_vs_commit": status.splitlines()}


def hf_model_present(model_id: str) -> str:
    hub = Path(os.environ.get("HF_HUB_CACHE") or (Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"))
    snap = hub / ("models--" + model_id.replace("/", "--")) / "snapshots"
    if not snap.exists():
        return "NOT in cache"
    files = [p.name for s in snap.iterdir() for p in s.iterdir()]
    has_weights = any(f.endswith((".safetensors", ".bin")) for f in files)
    return "in cache (weights present)" if has_weights else "in cache (tokenizer/config only, no weights)"


def preflight(group: str | None, args, paths: cr.C1Paths, log) -> dict[str, Any]:
    dry = paths.dry_run
    pf: dict[str, Any] = {"problems": [], "warnings": []}
    pf["python"] = {"executable": sys.executable, "version": platform.python_version()}
    mods = {}
    need = ["numpy", "pandas", "pyarrow", "transformers"] + ([] if dry else ["torch", "vllm", "accelerate"])
    for name in need:
        try:
            mod = __import__(name)
            mods[name] = getattr(mod, "__version__", "?")
        except Exception as exc:
            mods[name] = f"MISSING ({exc})"
            pf["problems"].append(f"python module {name} not importable: {exc}")
    pf["modules"] = mods
    pf["git"] = git_state()
    pf["hf_home"] = os.environ.get("HF_HOME", "")
    try:
        pf["prompt"] = cr.check_prompt()
    except Exception as exc:
        pf["problems"].append(f"prompt check failed: {exc}")
    models = {cr.LLM_MODEL_ID: hf_model_present(cr.LLM_MODEL_ID)}
    if not dry:
        for r in cr.DENSE_RETRIEVERS:
            models[C2.MODEL_INFO[r]["hf_id"]] = hf_model_present(C2.MODEL_INFO[r]["hf_id"])
    pf["models"] = models
    for mid, state in models.items():
        if state.startswith("NOT") or ("no weights" in state and not dry):
            (pf["warnings"] if dry else pf["problems"]).append(f"{mid}: {state}")
    # Inputs of the paper's runs.
    missing = []
    datasets = cr.GROUP_DATASETS[group] if group else cr.DATASET_KEYS
    for d in datasets:
        spec = DATASETS[d]
        for p in [REPO_ROOT / spec.index_json] + [REPO_ROOT / spec.saved_results_json(r) for r in retrievers_for(args)]:
            if not p.exists():
                missing.append(cr.rel(p))
    for extra in (
        "outputs/other_methods/vanilla/retrieval.csv",
        "results/tables/main/retrieval_comparison/table.csv",
        "results/tables/construction/cost/table.csv",
        "results/tables/construction/calls_per_document/table.csv",
    ):
        if not (REPO_ROOT / extra).exists():
            missing.append(extra)
    try:
        pf["expected_metrics"] = cr.rel(reference_dir(REPO_ROOT) / "expected_metrics.json")
    except FileNotFoundError as exc:
        missing.append(str(exc))
    pf["inputs_missing"] = missing
    if missing:
        pf["problems"].append(f"{len(missing)} input files missing (first: {missing[0]})")
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(paths.out_dir).free / 1024**3
    pf["disk"] = {"path": str(paths.out_dir), "free_gb": round(free, 1), "needed_gb_estimate": DISK_NEEDED_GB}
    if free < DISK_NEEDED_GB and not dry:
        pf["problems"].append(f"free disk {free:.1f} GB < {DISK_NEEDED_GB} GB")
    if dry:
        pf["gpus"] = "skipped (CPU dry run)"
    else:
        gpus, err = C2.nvidia_smi()
        if gpus is None:
            pf["problems"].append(err)
            pf["gpus"] = err
        else:
            mine = [g for g in gpus if g["index"] in cr.GROUP_GPUS[group]]
            pf["gpus"] = mine
            if len(mine) != 4:
                pf["problems"].append(f"expected GPUs {cr.GROUP_GPUS[group]}, nvidia-smi shows {[g['index'] for g in gpus]}")
            for g in mine:
                if g["memory_used_mib"] > GPU_BUSY_MIB:
                    pf["problems"].append(
                        f"GPU {g['index']} already uses {g['memory_used_mib']:.0f} MiB (> {GPU_BUSY_MIB}); another experiment is running there (C2?). Not sharing GPUs."
                    )
    c2 = {}
    for d in datasets:
        for r in retrievers_for(args):
            c2[f"{d}__{r}"] = "reused" if cr.c2_streams_available(d, r, REPO_ROOT) and not args.no_c2 else "recomputed (published path)"
    pf["c2_streams"] = c2
    pf["ok"] = not pf["problems"]
    return pf


# ---------------------------------------------------------------------------
# Estimates
# ---------------------------------------------------------------------------


def retrieval_estimates(dataset: str, census: dict[str, Any], retrievers, *, use_c2: bool, log) -> dict[str, Any]:
    """Seconds per retrieval job: C2's embedding cost model for the streams it would recompute, plus CR's chunk encodes."""

    out = {}
    counts = None
    chunk_tok = [t for r in census["docs"] for t in r["chunk_tokens"]]
    for r in retrievers:
        c2_ok = use_c2 and cr.c2_streams_available(dataset, r, REPO_ROOT) is not None
        if r == "bm25":
            if counts is None:
                counts = {dataset: C2.text_counts(dataset, max_docs=census.get("max_docs"))}
            base = C2.estimate_job(dataset, r, counts)["est_seconds"]
            out[r] = {"device": "cpu", "n_gpus": 0, "est_seconds": 3 * base, "basis": "C2's CPU rate x 3 (published streams, empty-context pass, CR pass)"}
            continue
        m = C2.MODEL_INFO[r]
        cap = m["max_length"]
        cr_tokens = sum(min(t + CONTEXT_TOKENS_ESTIMATE + 4, cap) + 2 for t in chunk_tok)
        plain_tokens = sum(min(t, cap) + 2 for t in chunk_tok)
        enc = lambda tok: tok * 2.0 * m["params_m"] * 1e6 / (C2.EFFECTIVE_TFLOPS * 1e12)  # noqa: E731
        q_s = 2 * census_queries(dataset, census) * m["query_call_s"]
        if c2_ok:
            secs = m["load_s"] + enc(cr_tokens) + enc(plain_tokens) + q_s
            basis = "C2 streams reused: CR chunks + plain chunks (empty-context check) encoded, 2 query encodings per query"
        else:
            if counts is None:
                counts = {dataset: C2.text_counts(dataset, max_docs=census.get("max_docs"))}
            secs = C2.estimate_job(dataset, r, counts)["est_seconds"] + enc(cr_tokens) + q_s
            basis = "C2's estimate for the published streams + CR chunks encoded"
        out[r] = {"device": "gpu", "n_gpus": int(m.get("min_gpus", 1)), "est_seconds": secs, "basis": basis}
    gpu_jobs = [{"device": "gpu", "n_gpus": v["n_gpus"], "est_seconds": v["est_seconds"]} for v in out.values() if v["device"] == "gpu"]
    wall = max(C2.simulate_makespan(gpu_jobs, 4) if gpu_jobs else 0.0, max((v["est_seconds"] for v in out.values() if v["device"] == "cpu"), default=0.0))
    return {"jobs": out, "wall_seconds": wall}


def census_queries(dataset: str, census: dict[str, Any]) -> int:
    rows = cr.load_saved_live_rows(DATASETS[dataset], "bm25", repo_root=REPO_ROOT)
    ids = {r["doc_id"] for r in census["docs"]}
    return sum(1 for x in rows if str(x.get("doc_id")) in ids)


# ---------------------------------------------------------------------------
# Memory check (GPU)
# ---------------------------------------------------------------------------


def longest_texts(datasets, *, max_docs: int | None) -> str:
    best = ""
    for d in datasets:
        for doc in cr.dataset_docs(d, max_docs=max_docs, repo_root=REPO_ROOT):
            for t in doc.texts:
                if len(t) > len(best):
                    best = t
    return best


def memory_check_worker(retriever: str, text: str, start_batch: int) -> dict[str, Any]:
    """Child process: encode one batch of the longest contextualized-length text, halve until it fits with a 10% margin."""

    import torch

    from saadi.indexing.retriever_adapters import get_retriever

    placeholder = " ".join(["context"] * PLACEHOLDER_CONTEXT_TOKENS)
    sample = cr.contextualize(placeholder, text)
    n_dev = torch.cuda.device_count()
    totals = [torch.cuda.get_device_properties(i).total_memory for i in range(n_dev)]
    tried = []
    batch = int(start_batch)
    Retriever = get_retriever(retriever)
    while batch >= 1:
        for i in range(n_dev):
            torch.cuda.reset_peak_memory_stats(i)
        torch.cuda.empty_cache()
        ok, err = True, None
        try:
            Retriever(corpus=[sample] * batch, chunk_ids=[str(i) for i in range(batch)], batch_size=batch)
        except torch.cuda.OutOfMemoryError as exc:
            ok, err = False, str(exc).splitlines()[0]
        peaks = [torch.cuda.max_memory_reserved(i) for i in range(n_dev)]
        fits = ok and all(p <= (1 - MEMORY_MARGIN) * t for p, t in zip(peaks, totals, strict=True))
        tried.append({"batch_size": batch, "peak_mib": [round(p / 2**20) for p in peaks], "fits": fits, "error": err})
        if fits:
            return {"retriever": retriever, "n_gpus": n_dev, "batch_size": batch, "default_batch_size": start_batch,
                    "peak_mib": [round(p / 2**20) for p in peaks], "total_mib": [round(t / 2**20) for t in totals], "tried": tried, "fits": True}
        batch //= 2
    return {"retriever": retriever, "n_gpus": n_dev, "batch_size": None, "default_batch_size": start_batch, "tried": tried, "fits": False,
            "total_mib": [round(t / 2**20) for t in totals]}


def default_batch(retriever: str) -> int:
    return int(C2.MODEL_INFO[retriever]["batch_size"])


def batch_env_name(retriever: str) -> str:
    return "SAADI_CONTRIEVER_BATCH_SIZE" if retriever == "contriever" else {"bge_m3": "SAADI_BGE_M3", "jina": "SAADI_JINA", "qwen": "SAADI_QWEN"}[retriever] + "_BATCH_SIZE"


def device_map_env_name(retriever: str) -> str | None:
    return {"bge_m3": "SAADI_BGE_M3_DEVICE_MAP", "jina": "SAADI_JINA_DEVICE_MAP", "qwen": "SAADI_QWEN_DEVICE_MAP"}.get(retriever)


def run_memory_checks(group: str, paths: cr.C1Paths, *, max_docs: int | None, log) -> dict[str, Any]:
    path = paths.memory_check_path(group)
    if path.exists():
        return cr.read_json(path)
    text = longest_texts(cr.GROUP_DATASETS[group], max_docs=max_docs)
    text_file = paths.out_dir / f"memory_check_text_group_{group}.txt"
    text_file.write_text(text, encoding="utf-8")
    results: dict[str, Any] = {"text_chars": len(text), "placeholder_context_tokens": PLACEHOLDER_CONTEXT_TOKENS, "margin": MEMORY_MARGIN, "models": {}}
    for r in cr.DENSE_RETRIEVERS:
        n_gpus = int(C2.MODEL_INFO[r].get("min_gpus", 1))
        for attempt_gpus in sorted({n_gpus, 2}):
            if attempt_gpus < n_gpus:
                continue
            gpus = cr.GROUP_GPUS[group][:attempt_gpus]
            env = child_env(gpus, r, n_gpus=attempt_gpus)
            env.pop(batch_env_name(r), None)
            out_file = paths.out_dir / f"memory_check_{group}_{r}_{attempt_gpus}gpu.json"
            cmd = [sys.executable, str(Path(__file__).resolve()), "--memory-check", r, "--text-file", str(text_file), "--out", str(out_file)]
            log(f"memory check {r} on GPU {','.join(map(str, gpus))} (batch from {default_batch(r)}) ...")
            with (paths.logs_dir / f"memory_check_{group}_{r}.log").open("a", encoding="utf-8") as handle:
                rc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT).returncode
            res = cr.read_json(out_file) if out_file.exists() else {"fits": False, "error": f"exit {rc}, no result"}
            res["gpus"] = gpus
            results["models"][r] = res
            if res.get("fits"):
                log(f"memory check {r}: batch {res['batch_size']} fits on {attempt_gpus} GPU(s), peak {res['peak_mib']} MiB")
                break
            log(f"memory check {r}: does not fit on {attempt_gpus} GPU(s) ({res.get('error') or res.get('tried', [])[-1:]})")
    results["ok"] = all(v.get("fits") for v in results["models"].values())
    cr.write_json(path, results)
    return results


def child_env(gpus: list[int], retriever: str | None = None, *, n_gpus: int = 1, memory: dict[str, Any] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    env.setdefault("SAADI_ENCODED_POOL_CACHE_SIZE", "32")
    env.setdefault("OMP_NUM_THREADS", os.environ.get("C1_THREADS", "8"))
    env.setdefault("MKL_NUM_THREADS", os.environ.get("C1_THREADS", "8"))
    env["TOKENIZERS_PARALLELISM"] = "false"
    if gpus:
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if retriever and retriever != "bm25":
        dm = device_map_env_name(retriever)
        if dm and n_gpus > 1:
            env[dm] = "auto"
        if memory and memory.get("models", {}).get(retriever, {}).get("batch_size"):
            env[batch_env_name(retriever)] = str(memory["models"][retriever]["batch_size"])
    return env


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generation_worker(dataset: str, args) -> int:
    paths = paths_for(args)
    log = make_log(None)
    census = cr.read_json(paths.census_path(dataset))
    t0 = time.perf_counter()
    try:
        gen = cr.VLLMGenerator(census["model"]["max_model_len"], log=log)
        info = cr.generate_contexts(dataset, paths=paths, census=census, generator=gen, max_docs=max_docs_for(args), log=log)
        settings, version = gen.settings, gen.version
    except Exception:
        log(traceback.format_exc())
        return 1
    record_generation_run(paths, dataset, info | {"generator": "vllm", "vllm_version": version, "vllm_settings": settings,
                                                  "wall_seconds": round(time.perf_counter() - t0, 1), "gpus": os.environ.get("CUDA_VISIBLE_DEVICES", "")})
    return 0


def record_generation_run(paths: cr.C1Paths, dataset: str, run: dict[str, Any]) -> None:
    p = paths.generation_summary_path(dataset)
    data = cr.read_json(p) if p.exists() else {"runs": []}
    data["runs"].append(run | {"finished_at": now()})
    cr.write_json(p, data)


def run_generation(group: str, dataset: str, census: dict[str, Any], paths: cr.C1Paths, args, log) -> dict[str, Any]:
    marker = paths.marker(dataset, "generation.DONE")
    if not marker.exists():
        if paths.dry_run:
            t0 = time.perf_counter()
            gen = cr.StubGenerator(cr.load_qwen_tokenizer())
            info = cr.generate_contexts(dataset, paths=paths, census=census, generator=gen, max_docs=max_docs_for(args), log=log)
            record_generation_run(paths, dataset, info | {"generator": "stub", "wall_seconds": round(time.perf_counter() - t0, 1), "gpus": ""})
        else:
            gpus = cr.GROUP_GPUS[group]
            env = child_env(gpus)
            env["VLLM_PORT"] = str(VLLM_PORT[group])
            env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
            env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
            log_path = paths.logs_dir / f"{dataset}__generation.log"
            cmd = [sys.executable, str(Path(__file__).resolve()), "--generate", dataset, "--group", group] + passthrough(args)
            log(f"{dataset}: generating contexts on GPUs {gpus} (vLLM TP 4); log {cr.rel(log_path)}")
            with log_path.open("a", encoding="utf-8") as handle:
                rc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT).returncode
            if rc != 0:
                log(f"{dataset}: generation exited {rc}; see {cr.rel(log_path)}")
        stats = cr.generation_stats(dataset, paths=paths, census=census)
        if stats["missing_chunks"] == 0:
            cr.write_json(marker, {"finished_at": now(), "calls": stats["calls"]})
    stats = cr.generation_stats(dataset, paths=paths, census=census)
    gen_runs = cr.read_json(paths.generation_summary_path(dataset)) if paths.generation_summary_path(dataset).exists() else {"runs": []}
    return {"done": marker.exists(), "stats": stats, "runs": gen_runs["runs"]}


def passthrough(args) -> list[str]:
    out = []
    if args.dry_run:
        out.append("--dry-run")
    if args.dry_run_full:
        out.append("--dry-run-full")
    if args.max_docs is not None:
        out += ["--max-docs", str(args.max_docs)]
    if args.no_c2:
        out.append("--no-c2")
    return out


# ---------------------------------------------------------------------------
# Retrieval jobs
# ---------------------------------------------------------------------------


def job_worker(job: str, args) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout)
    paths = paths_for(args)
    dataset, retriever = job.split("__")
    log = make_log(None)
    log(f"job {job} start; CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r}")
    try:
        s = cr.run_retrieval_job(dataset, retriever, paths=paths, max_docs=max_docs_for(args), use_c2=not args.no_c2, log=log)
    except Exception:
        tb = traceback.format_exc()
        log(tb)
        cr.write_json(paths.job_failed(dataset, retriever), {"dataset": dataset, "retriever": retriever, "status": "ERROR", "traceback": tb, "finished_at": now()})
        return 1
    log(f"job {job} {s['status']}: {json.dumps(s['gate'].get('problems'))}")
    return 0 if s["status"] == "DONE" else 1


def run_jobs(group: str, dataset: str, retrievers, paths: cr.C1Paths, args, memory: dict[str, Any] | None, est: dict[str, Any], log) -> dict[str, Any]:
    results: dict[str, Any] = {}
    queue = []
    for r in sorted(retrievers, key=lambda x: -est["jobs"].get(x, {}).get("est_seconds", 0)):
        if paths.job_done(dataset, r).exists():
            results[r] = {"status": "DONE (already finished)", "summary": cr.read_json(paths.job_done(dataset, r))}
            log(f"{dataset}__{r}: already DONE, skipped")
            continue
        if r != "bm25" and memory is not None and not memory.get("models", {}).get(r, {}).get("fits"):
            results[r] = {"status": "SKIPPED (memory check failed)"}
            continue
        queue.append(r)
    free = [] if paths.dry_run else list(cr.GROUP_GPUS[group])
    running: dict[str, tuple] = {}
    while queue or running:
        started = False
        for r in list(queue):
            if r == "bm25" or paths.dry_run:
                gpus: list[int] = []
            else:
                need = int((memory or {}).get("models", {}).get(r, {}).get("n_gpus") or C2.MODEL_INFO[r].get("min_gpus", 1))
                if len(free) < need:
                    continue
                gpus = [free.pop(0) for _ in range(need)]
            env = child_env(gpus, r, n_gpus=len(gpus), memory=memory)
            log_path = paths.logs_dir / f"{dataset}__{r}.log"
            handle = log_path.open("a", encoding="utf-8")
            cmd = [sys.executable, str(Path(__file__).resolve()), "--run-job", f"{dataset}__{r}", "--group", group] + passthrough(args)
            proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            running[r] = (proc, gpus, time.time(), handle)
            queue.remove(r)
            started = True
            where = f"GPU {','.join(map(str, gpus))}" if gpus else "CPU"
            log(f"{dataset}__{r}: started on {where} (estimate {fmt_s(est['jobs'].get(r, {}).get('est_seconds'))}); log {cr.rel(log_path)}")
        for r, (proc, gpus, t0, handle) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            handle.close()
            free = sorted(free + gpus)
            del running[r]
            secs = time.time() - t0
            if paths.job_done(dataset, r).exists():
                results[r] = {"status": "DONE", "seconds": secs, "gpus": gpus, "summary": cr.read_json(paths.job_done(dataset, r))}
            elif paths.job_failed(dataset, r).exists():
                fm = cr.read_json(paths.job_failed(dataset, r))
                results[r] = {"status": fm.get("status", "FAILED"), "seconds": secs, "gpus": gpus, "summary": fm}
            else:
                tail = (paths.logs_dir / f"{dataset}__{r}.log").read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
                results[r] = {"status": f"CRASHED (exit {rc})", "seconds": secs, "gpus": gpus, "log_tail": tail}
            log(f"{dataset}__{r}: {results[r]['status']} after {fmt_s(secs)}")
        if not started:
            time.sleep(2.0 if paths.dry_run else 10.0)
    return results


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def f1(x: Any, nd: int = 1) -> str:
    return "—" if x is None else f"{float(x):.{nd}f}"


def sgn(x: Any, nd: int = 1) -> str:
    return "—" if x is None else f"{float(x):+.{nd}f}"


def ci(c: dict[str, Any] | None, nd: int = 1) -> str:
    if not c:
        return "—"
    lo, hi = c["ci"]
    return f"{c['diff']:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


def verdict(c: dict[str, Any] | None) -> str:
    """For 'A minus B': A, B, or neither, from the 95% interval."""

    if not c:
        return "—"
    lo, hi = c["ci"]
    return "above" if lo > 0 else ("below" if hi < 0 else "n.s.")


def hybrid_label(d: str) -> str:
    if d == "bm25":
        return "BM25 + BM25 (dry-run stand-in for a dense retriever)"
    return cr.RETRIEVER_DISPLAY[d] + " + BM25"


def md_table(header: list[str], rows: list[list[Any]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def tok(n: Any) -> str:
    if n is None:
        return "—"
    n = float(n)
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e4:
        return f"{n / 1e3:.0f}K"
    return f"{n:.0f}"


# ---------------------------------------------------------------------------
# Per-dataset report
# ---------------------------------------------------------------------------


def cost_rows(dataset: str, census: dict[str, Any], gen: dict[str, Any], jobs: dict[str, Any], saadi: dict[str, float] | None) -> tuple[list[list[Any]], dict[str, Any]]:
    s = gen["stats"]
    runs = gen.get("runs", [])
    wall = sum(float(r.get("wall_seconds", 0)) for r in runs)
    n_gpu = 4 if any(r.get("generator") == "vllm" for r in runs) else 0
    chunks = max(1, census["n_chunks"])
    nominal = s["nominal_input_tokens"].get("total")
    computed = s["computed_input_tokens"].get("total")
    gen_total = s["generated_tokens"].get("total")
    c = {
        "calls": s["calls"],
        "calls_per_chunk": s["calls_per_chunk"],
        "calls_per_document": s["calls_per_document"],
        "nominal_input_total": nominal,
        "computed_input_total": computed,
        "nominal_input_per_chunk": (nominal or 0) / chunks,
        "computed_input_per_chunk": (computed / chunks) if computed is not None else None,
        "generated_total": gen_total,
        "generated_per_chunk_mean": s["generated_tokens"].get("mean"),
        "generated_per_chunk_median": s["generated_tokens"].get("median"),
        "cut_share": s.get("cut_share"),
        "generation_wall_seconds": wall,
        "generation_gpu_hours": wall * n_gpu / 3600.0,
        "cached_tokens_source": s.get("cached_tokens_source"),
    }
    sa = saadi or {}
    rows = [
        ["LLM calls (total)", f"{c['calls']}", f"{sa.get('calls_total', 0):.0f}" if sa else "—"],
        ["LLM calls per chunk", f1(c["calls_per_chunk"], 2), f1(sa.get("calls_per_chunk"), 2)],
        ["LLM calls per document", f1(c["calls_per_document"], 1), f1(sa.get("total_calls_per_document"), 1)],
        ["Input tokens per chunk, nominal (full prompt)", tok(c["nominal_input_per_chunk"]), tok(sa.get("input_tokens_per_chunk")) + " (no prefix cache)"],
        ["Input tokens per chunk, computed (after prefix-cache hits)", tok(c["computed_input_per_chunk"]), "same as nominal"],
        ["Input tokens total, nominal / computed", f"{tok(nominal)} / {tok(computed)}", tok(sa.get("input_tokens_total"))],
        ["Generated tokens per chunk, mean / median", f"{f1(c['generated_per_chunk_mean'])} / {f1(c['generated_per_chunk_median'])}", f1(sa.get("generated_tokens_per_chunk"))],
        ["Generated tokens total", tok(gen_total), tok(sa.get("generated_tokens_total"))],
        [f"Contexts cut at {cr.CONTEXT_MAX_TOKENS} tokens", f"{s['cut_at_max_tokens']} ({100 * (c['cut_share'] or 0):.1f}%)", "—"],
        ["Generation wall clock / GPU-hours", f"{fmt_s(wall)} / {c['generation_gpu_hours']:.2f}", "not in the paper's tables"],
    ]
    for r, j in jobs.items():
        summ = j.get("summary", {}) or {}
        t = (summ.get("timings") or {}).get("seconds_total")
        g = len(j.get("gpus") or []) or len([x for x in (summ.get("device", {}) or {}).get("cuda_visible_devices", "").split(",") if x])
        if t is not None:
            rows.append([f"Indexing + scoring, {cr.RETRIEVER_DISPLAY[r]} (wall / GPU-hours)", f"{fmt_s(t)} / {t * g / 3600:.2f}", "—"])
    if sa:
        # Per chunk, so that the ratio stays like for like when C1 covers a subset (dry run) or a different chunk count.
        def ratio(a, b):
            return (a / b) if (a is not None and b) else None

        c["ratio_nominal_input_vs_saadi"] = ratio(c["nominal_input_per_chunk"], sa.get("input_tokens_per_chunk"))
        c["ratio_computed_input_vs_saadi"] = ratio(c["computed_input_per_chunk"], sa.get("input_tokens_per_chunk"))
        c["ratio_calls_vs_saadi"] = ratio(c["calls_per_chunk"], sa.get("calls_per_chunk"))
        c["ratio_generated_vs_saadi"] = ratio((gen_total or 0) / chunks, sa.get("generated_tokens_per_chunk"))
        c["saadi_chunks"] = sa.get("actual_chunks")
        c["chunks"] = census["n_chunks"]
    return rows, c


def dataset_report(dataset: str, paths: cr.C1Paths, args, *, census, gen, memory, est, jobs_status, analysis, problems) -> dict[str, Any]:
    t1 = cr.table1_rows(REPO_ROOT)
    held = cr.c2_heldout(REPO_ROOT)
    saadi_cost = cr.saadi_costs(REPO_ROOT).get(dataset)
    jobs = {}
    for r in retrievers_for(args):
        if paths.job_done(dataset, r).exists():
            jobs[r] = {"status": "DONE", "summary": cr.read_json(paths.job_done(dataset, r)), **{k: v for k, v in (jobs_status.get(r) or {}).items() if k == "gpus"}}
        elif paths.job_failed(dataset, r).exists():
            jobs[r] = {"status": cr.read_json(paths.job_failed(dataset, r)).get("status", "FAILED"), "summary": cr.read_json(paths.job_failed(dataset, r))}
        else:
            jobs[r] = jobs_status.get(r) or {"status": "not run"}
    crows, cost = cost_rows(dataset, census, gen, {r: j for r, j in jobs.items() if j["status"] == "DONE"}, saadi_cost)
    disp = cr.DATASET_DISPLAY[dataset]
    tag = f" ({paths.tag.replace('_', ' ')}: stub contexts, BM25 only, placeholder numbers)" if paths.dry_run else ""
    L = [f"# C1 Contextual Retrieval: {disp}{tag}", ""]
    L.append(f"- Written {now()}; commit {git_state()['commit'][:12]}; documents {census['n_docs']}, chunks {census['n_chunks']}, queries {sum(j['summary'].get('queries', 0) for j in jobs.values() if j['status'] == 'DONE' and j['summary'].get('retriever') == 'bm25') or '—'} (expected {cr.EXPECTED_QUERIES[dataset]} on the full dataset).")
    L.append(f"- Status: generation {'done' if gen['done'] else 'NOT done'}; jobs " + ", ".join(f"{cr.RETRIEVER_DISPLAY[r]} {j['status']}" for r, j in jobs.items()) + ".")
    for p in problems:
        L.append(f"- PROBLEM: {p}")
    L.append("")

    L += ["## 1. Correctness check (empty context reproduces Vanilla)", ""]
    L.append("CR with every context empty indexes exactly the chunk text, so it must give Vanilla (SAADI's text channel alone) and the paper's saved results. "
             "BM25: identical rankings; dense: identical top 10 for at least 99% of queries and HR within 0.1 points (C2's Jina drift rule applies). "
             "The pool itself is checked by reproducing SAADI's saved rankings with the published weights.")
    L.append("")
    rows = []
    for r, j in jobs.items():
        g = (j.get("summary") or {}).get("gate")
        if not g:
            rows.append([cr.RETRIEVER_DISPLAY[r], j["status"], "—", "—", "—", "—", "—"])
            continue
        sr, ev = g["saadi_reproduction"], g["empty_vs_vanilla"]
        pq = g.get("saved_per_query_text_only")
        t1v = g.get("table1_vanilla", {})
        rows.append([
            cr.RETRIEVER_DISPLAY[r],
            "pass" if g["passed"] else "FAIL",
            f"{sr['identical_top10']}/{sr['queries']}",
            f"{ev['identical_top10']}/{ev['queries']} (top-150 {ev['identical_top150']})",
            f"{f1(ev['hr_empty']['hr_percent']['5'], 2)} / {f1(ev['hr_empty']['hr_percent']['10'], 2)} vs {f1(ev['hr_vanilla']['hr_percent']['5'], 2)} / {f1(ev['hr_vanilla']['hr_percent']['10'], 2)}",
            f"{pq['identical_top10']}/{pq['queries']}" if pq else "no file",
            f"{f1(t1v.get('5', {}).get('table1'))} / {f1(t1v.get('10', {}).get('table1'))}",
        ])
    L += md_table(["Retriever", "Gate", "SAADI top-10 = saved", "Empty = Vanilla top-10", "HR@5 / HR@10 empty vs Vanilla", "= saved text-only file", "Table 1 Vanilla"], rows)
    for r, j in jobs.items():
        g = (j.get("summary") or {}).get("gate") or {}
        for p in g.get("problems", []):
            L.append(f"- {cr.RETRIEVER_DISPLAY[r]} PROBLEM: {p}")
        for n_ in g.get("notes", []):
            L.append(f"- {cr.RETRIEVER_DISPLAY[r]} note: {n_}")
        for ex in g.get("mismatch_examples", [])[:2]:
            L.append(f"- {cr.RETRIEVER_DISPLAY[r]} mismatch example: `{json.dumps(ex)[:400]}`")
        if g:
            L.append(f"- {cr.RETRIEVER_DISPLAY[r]}: Vanilla reference for the dense check = {g['empty_vs_vanilla']['reference']}; memory and summary channels from {j['summary'].get('channel_source_docs')} documents.")
    if dataset not in cr.TABLE1_VANILLA_IS_TEXT_CHANNEL:
        L.append(f"- On {disp}, Table 1's Vanilla is not the text channel of the saved chunks (C2's server run: they differ by up to 13 points); C1's Vanilla is the text channel, and the gate uses the saved per-query text-only file where it exists.")
    L.append("")

    L += ["## 2. Token census (Qwen3 tokenizer)", ""]
    dt = census["doc_tokens"]
    L.append(f"- Window {census['model']['max_model_len']} tokens (model config, rope_scaling {census['model']['rope_scaling']}); prompt template {census['template_tokens']} tokens; output budget {cr.CONTEXT_MAX_TOKENS}.")
    L.append(f"- Document tokens: min {tok(dt['min'])}, median {tok(dt['median'])}, max {tok(dt['max'])}; chunk tokens median {f1(census['chunk_tokens']['median'], 0)}, max {f1(census['chunk_tokens']['max'], 0)}.")
    L.append(f"- Documents over the window: {census['docs_exceeding_window']} of {census['n_docs']} ({census['chunks_in_segmented_docs']} chunks); segments per document: {census['k_values']}.")
    L.append(f"- Split vs joint tokenization of the prompt identical on the checked samples: {census['joint_tokenization_equals_split']}.")
    L.append("")

    L += ["## 3. GPU memory check (embedding models)", ""]
    if memory is None:
        L.append("- Skipped (CPU dry run).")
    else:
        mrows = [[cr.RETRIEVER_DISPLAY[r], m.get("n_gpus"), m.get("default_batch_size"), m.get("batch_size"), m.get("peak_mib"), m.get("total_mib"), "yes" if m.get("fits") else "NO"]
                 for r, m in memory.get("models", {}).items()]
        L += md_table(["Model", "GPUs", "Vanilla batch", "Chosen batch", "Peak MiB", "Total MiB", "Fits (10% margin)"], mrows)
        L.append(f"- Longest plain chunk of this group ({memory.get('text_chars')} characters) with {PLACEHOLDER_CONTEXT_TOKENS} placeholder tokens in front, float32 as Vanilla.")
    L.append("")

    L += ["## 4. Long documents (segment rule)", ""]
    if census["segmented_docs"]:
        L += md_table(["Document", "Tokens", "Chunks", "Segments k"], [[s["doc_id"], tok(s["doc_tokens"]), s["chunks"], s["k"]] for s in census["segmented_docs"]])
        L.append("- These chunks saw only their own segment of the document, not the whole document: a limit of CR on book-length text.")
    else:
        L.append("- No document exceeds the window; every chunk saw its whole document.")
    L.append("")

    s = gen["stats"]
    L += ["## 5. Contexts", ""]
    g_ = s["generated_tokens"]
    L.append(f"- {s['calls']} contexts ({s['missing_chunks']} missing), generator {s['generator']}; empty {s['empty_contexts']}, failed {s['failed_contexts']}, cut at {cr.CONTEXT_MAX_TOKENS} tokens {s['cut_at_max_tokens']}.")
    if g_.get("n"):
        L.append(f"- Context length (tokens): min {f1(g_['min'], 0)}, p25 {f1(g_['p25'], 0)}, median {f1(g_['median'], 0)}, mean {f1(g_['mean'])}, p75 {f1(g_['p75'], 0)}, p95 {f1(g_['p95'], 0)}, max {f1(g_['max'], 0)}.")
    L.append(f"- Cached tokens: {s['cached_tokens_source']}.")
    L.append("")

    L += ["## 6. Cost next to SAADI", ""]
    L += md_table(["Quantity", "CR", "SAADI (memory + hierarchy calls)"], crows)
    if saadi_cost:
        L.append(f"- Ratios per chunk. SAADI's builds had no prefix caching, so its input is compared with both CR columns: CR nominal / SAADI ={f1(cost.get('ratio_nominal_input_vs_saadi'), 2)}x, CR computed / SAADI = {f1(cost.get('ratio_computed_input_vs_saadi'), 2)}x; calls {f1(cost.get('ratio_calls_vs_saadi'), 2)}x; generated tokens {f1(cost.get('ratio_generated_vs_saadi'), 2)}x. SAADI chunks in the cost table: {saadi_cost.get('actual_chunks'):.0f}; C1 chunks: {census['n_chunks']}.")
    L.append("")

    L += ["## 7. Retrieval: per retriever", ""]
    pr = analysis.get("per_retriever", {})
    rows = []
    for r in cr.RETRIEVERS:
        if r not in pr:
            continue
        a = pr[r]
        t1v = t1.get((dataset, r, "Vanilla"), {})
        lq = t1.get((dataset, r, "LQCA"), {})
        ho = (held or {}).get((dataset, r), {})
        rows.append([
            cr.RETRIEVER_DISPLAY[r],
            f"{f1(a['vanilla']['hr5'])} / {f1(a['vanilla']['hr10'])}",
            f"{f1(t1v.get('hr5'))} / {f1(t1v.get('hr10'))}",
            f"{f1(a['cr']['hr5'])} / {f1(a['cr']['hr10'])}",
            f"{f1(a['saadi']['hr5'])} / {f1(a['saadi']['hr10'])}",
            f"{f1(a['saadi_cr']['hr5'])} / {f1(a['saadi_cr']['hr10'])}",
            f"{f1(lq.get('hr5'))} / {f1(lq.get('hr10'))}",
        ] + ([f"{f1(ho.get('hr5'))} / {f1(ho.get('hr10'))}"] if held else []))
    head = ["Retriever", "Vanilla (text channel)", "Table 1 Vanilla", "CR", "SAADI (published)", "SAADI on CR text*", "LQCA (Table 1)"] + (["SAADI held-out (C2)"] if held else [])
    L.append("HR@5 / HR@10 (%). *Complementarity check, not a proposed method.")
    L.append("")
    L += md_table(head, rows)
    if not held:
        L.append("- SAADI held-out column left out: results/rebuttal/C2_heldout_fusion/REPORT.md does not exist yet.")
    L.append("")
    L.append("Paired differences (points) with 95% bootstrap intervals (10,000 resamples over queries, seed 42):")
    L.append("")
    rows = []
    for r in cr.RETRIEVERS:
        if r not in pr:
            continue
        a = pr[r]
        rows.append([cr.RETRIEVER_DISPLAY[r]] + [ci(a[f"{n}_hr{k}"]) for n in ("cr_minus_vanilla", "saadi_minus_cr", "saadi_cr_minus_cr") for k in ("5", "10")])
    L += md_table(["Retriever", "CR − Vanilla @5", "@10", "SAADI − CR @5", "@10", "SAADI-on-CR − CR @5", "@10"], rows)
    L.append("")

    L += ["## 8. Retrieval: hybrid (dense + BM25, cookbook fusion)", ""]
    hy = analysis.get("hybrid", {})
    if hy:
        rows = [[hybrid_label(d), f"{f1(h['vanilla']['hr5'])} / {f1(h['vanilla']['hr10'])}", f"{f1(h['cr']['hr5'])} / {f1(h['cr']['hr10'])}",
                 f"{f1(h['saadi']['hr5'])} / {f1(h['saadi']['hr10'])}", ci(h["cr_minus_vanilla_hr10"]), ci(h["saadi_minus_cr_hr10"]), ci(h["saadi_minus_cr_hr5"])]
                for d, h in hy.items()]
        L += md_table(["Hybrid", "Vanilla @5 / @10", "CR @5 / @10", "SAADI @5 / @10", "CR − Vanilla @10", "SAADI − CR @10", "SAADI − CR @5"], rows)
    else:
        L.append("- No dense retriever finished (dry run: BM25 only), so no hybrid rows. Placeholder layout:")
        L.append("")
        L += md_table(["Hybrid", "Vanilla @5 / @10", "CR @5 / @10", "SAADI @5 / @10", "CR − Vanilla @10", "SAADI − CR @10", "SAADI − CR @5"],
                      [[f"{cr.RETRIEVER_DISPLAY[d]} + BM25", "x / x", "x / x", "x / x", "+x [x, x]", "+x [x, x]", "+x [x, x]"] for d in cr.DENSE_RETRIEVERS])
    L.append(f"- Top {cr.RECALL_N} of each list, score {cr.SEMANTIC_WEIGHT} / rank_dense + {cr.BM25_WEIGHT} / rank_bm25 (1-based), the same merge for Vanilla, CR and SAADI (SAADI = SAADI with d + SAADI with BM25, published weights). No reranker.")
    L.append("")

    L += ["## 9. Dense truncation of the contextualized chunks", ""]
    rows = []
    for r, j in jobs.items():
        t = ((j.get("summary") or {}).get("truncation") or {})
        c_, p_ = t.get("contextualized") or {}, t.get("plain") or {}
        if c_:
            rows.append([cr.RETRIEVER_DISPLAY[r], c_.get("max_length"), f"{c_.get('over_max_length')}/{c_.get('texts')}", tok(c_.get("tokens_cut_total")), f"{p_.get('over_max_length')}/{p_.get('texts')}"])
    if rows:
        L += md_table(["Retriever", "Max input", "CR chunks over max", "CR tokens cut (end of chunk)", "Plain chunks over max"], rows)
    else:
        L.append("- No dense retriever in this run.")
    L.append("")

    L += ["## 10. Sanity sample (10 random contexts, seed 42)", ""]
    for x in cr.sanity_sample(dataset, paths=paths, max_docs=max_docs_for(args)):
        L.append(f"- `{x['chunk_id']}` (segment {x['segment']}, {x['generated_tokens']} tokens, {x['finish_reason']}): **{x['context'][:600]}**")
        L.append(f"  - chunk: {x['chunk'][:300].replace(chr(10), ' ')}…")
    L.append("")

    L += ["## 11. Estimates and timings", ""]
    eg = est.get("generation", {})
    L.append(f"- Generation estimate {fmt_s(eg.get('seconds'))} for {eg.get('calls')} calls ({eg.get('basis')}); measured {fmt_s(cost['generation_wall_seconds'])}.")
    for r, j in est.get("retrieval", {}).get("jobs", {}).items():
        meas = ((jobs.get(r, {}).get("summary") or {}).get("timings") or {}).get("seconds_total")
        where = f"{j['n_gpus']} GPU(s)" if j["n_gpus"] else "CPU"
        L.append(f"- {cr.RETRIEVER_DISPLAY[r]}: estimate {fmt_s(j['est_seconds'])} on {where} ({j['basis']}); measured {fmt_s(meas)}.")
    L.append("")
    report = {
        "dataset": dataset, "written_at": now(), "tag": paths.tag, "census": {k: v for k, v in census.items() if k != "docs"},
        "generation": gen, "memory_check": memory, "estimates": est, "jobs": jobs, "analysis": analysis, "cost": cost, "problems": problems,
    }
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    (paths.reports_dir / f"{dataset}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    cr.write_json(paths.reports_dir / f"{dataset}.json", report)
    return report


# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------


def _count(items: list[dict[str, Any] | None]) -> dict[str, int]:
    """For 'SAADI minus CR': SAADI above (CI > 0), CR above (CI < 0), neither."""

    out = {"saadi_above": 0, "cr_above": 0, "neither": 0, "pairs": 0}
    for c in items:
        if not c:
            continue
        out["pairs"] += 1
        v = verdict(c)
        out["saadi_above" if v == "above" else ("cr_above" if v == "below" else "neither")] += 1
    return out


def final_report(paths: cr.C1Paths, args, log) -> bool:
    reports = {}
    for d in cr.DATASET_KEYS:
        p = paths.reports_dir / f"{d}.json"
        if p.exists():
            reports[d] = cr.read_json(p)
    complete = all(d in reports and all(j.get("status") == "DONE" for j in reports[d]["jobs"].values()) and reports[d]["generation"]["done"] for d in cr.DATASET_KEYS)
    t1 = cr.table1_rows(REPO_ROOT)
    held = cr.c2_heldout(REPO_ROOT)
    pr_rows, hy_rows, cost_csv = [], [], []
    per_items = {k: [] for k in ("5", "10")}
    hy_items = {k: [] for k in ("5", "10")}
    cv_items = {k: [] for k in ("5", "10")}
    comp_items = {k: [] for k in ("5", "10")}
    comp_vs_saadi = {k: [] for k in ("5", "10")}
    for d, rep in reports.items():
        a = rep["analysis"]
        for r, x in a.get("per_retriever", {}).items():
            for k in ("5", "10"):
                per_items[k].append(x[f"saadi_minus_cr_hr{k}"])
                cv_items[k].append(x[f"cr_minus_vanilla_hr{k}"])
                comp_items[k].append(x[f"saadi_cr_minus_cr_hr{k}"])
                comp_vs_saadi[k].append(x[f"saadi_cr_minus_saadi_hr{k}"])
            pr_rows.append({"dataset": d, "retriever": r, **{f"{s}_hr{k}": x[s][f"hr{k}"] for s in cr.SYSTEMS for k in ("5", "10")},
                            "table1_vanilla_hr5": t1.get((d, r, "Vanilla"), {}).get("hr5"), "table1_vanilla_hr10": t1.get((d, r, "Vanilla"), {}).get("hr10"),
                            "lqca_hr5": t1.get((d, r, "LQCA"), {}).get("hr5"), "lqca_hr10": t1.get((d, r, "LQCA"), {}).get("hr10"),
                            **{f"{n}_hr{k}": x[f"{n}_hr{k}"]["diff"] for n in ("cr_minus_vanilla", "saadi_minus_cr", "saadi_cr_minus_cr") for k in ("5", "10")},
                            **{f"{n}_hr{k}_ci": x[f"{n}_hr{k}"]["ci"] for n in ("cr_minus_vanilla", "saadi_minus_cr", "saadi_cr_minus_cr") for k in ("5", "10")}})
        for dd, h in a.get("hybrid", {}).items():
            for k in ("5", "10"):
                hy_items[k].append(h[f"saadi_minus_cr_hr{k}"])
            hy_rows.append({"dataset": d, "dense": dd, **{f"{s}_hr{k}": h[s][f"hr{k}"] for s in ("vanilla", "cr", "saadi") for k in ("5", "10")},
                            **{f"{n}_hr{k}": h[f"{n}_hr{k}"]["diff"] for n in ("cr_minus_vanilla", "saadi_minus_cr") for k in ("5", "10")},
                            **{f"{n}_hr{k}_ci": h[f"{n}_hr{k}"]["ci"] for n in ("cr_minus_vanilla", "saadi_minus_cr") for k in ("5", "10")}})
        cost_csv.append({"dataset": d, **{k: v for k, v in rep["cost"].items() if not isinstance(v, (list, dict))}})
    cnt = {k: _count(per_items[k]) for k in ("5", "10")}
    hcnt = {k: _count(hy_items[k]) for k in ("5", "10")}
    ccnt = {k: _count(comp_items[k]) for k in ("5", "10")}  # 'saadi_above' here = SAADI-on-CR above CR
    cvc = {k: _count(cv_items[k]) for k in ("5", "10")}  # 'saadi_above' here = CR above Vanilla
    csc = {k: _count(comp_vs_saadi[k]) for k in ("5", "10")}  # 'saadi_above' here = SAADI-on-CR above SAADI
    sa_all = cr.saadi_costs(REPO_ROOT)
    tot = {"cr_nominal": 0.0, "cr_computed": 0.0, "cr_calls": 0, "cr_gen": 0.0, "saadi_in": 0.0, "saadi_calls": 0.0, "saadi_gen": 0.0, "computed_known": True}
    for d, rep in reports.items():
        c = rep["cost"]
        tot["cr_nominal"] += c.get("nominal_input_total") or 0
        if c.get("computed_input_total") is None:
            tot["computed_known"] = False
        tot["cr_computed"] += c.get("computed_input_total") or 0
        tot["cr_calls"] += c.get("calls") or 0
        tot["cr_gen"] += c.get("generated_total") or 0
        s = sa_all.get(d, {})
        n_ch = c.get("chunks") or 0  # SAADI's per-chunk rates x C1's chunks (= SAADI's totals when the chunk counts agree)
        tot["saadi_in"] += s.get("input_tokens_per_chunk", 0) * n_ch
        tot["saadi_calls"] += s.get("calls_per_chunk", 0) * n_ch
        tot["saadi_gen"] += s.get("generated_tokens_per_chunk", 0) * n_ch
    rn = tot["cr_nominal"] / tot["saadi_in"] if tot["saadi_in"] else None
    rc = (tot["cr_computed"] / tot["saadi_in"]) if (tot["saadi_in"] and tot["computed_known"]) else None

    out_dir = paths.results_dir
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    hf._write_csv(out_dir / "tables" / "per_retriever.csv", pr_rows)
    hf._write_csv(out_dir / "tables" / "hybrid.csv", hy_rows)
    hf._write_csv(out_dir / "tables" / "cost.csv", cost_csv)
    dry = f" ({paths.tag.replace('_', ' ')}: stub contexts, BM25 only; every number is a placeholder)" if paths.dry_run else ""
    L = [f"# C1: Contextual Retrieval on SAADI's benchmarks{dry}", ""]
    L.append(f"Written {now()}; commit {git_state()['commit'][:12]}. Datasets with a report: {', '.join(cr.DATASET_DISPLAY[d] for d in reports) or 'none'}" + ("" if complete else " — INCOMPLETE: not every dataset and retriever is done; numbers below cover what exists.") + "")
    L.append("")
    L += ["## Summary", ""]
    L.append(f"1. Per retriever (same retriever, same chunks, same queries), SAADI − CR at HR@10: SAADI above with the 95% interval excluding zero in {cnt['10']['saadi_above']}/{cnt['10']['pairs']} (dataset, retriever) pairs, CR above in {cnt['10']['cr_above']}, neither in {cnt['10']['neither']}; at HR@5: {cnt['5']['saadi_above']} / {cnt['5']['cr_above']} / {cnt['5']['neither']}.")
    L.append(f"2. Hybrid (dense + BM25, cookbook fusion, the configuration Anthropic reports): SAADI above in {hcnt['10']['saadi_above']}/{hcnt['10']['pairs']} pairs at HR@10, CR above in {hcnt['10']['cr_above']}, neither in {hcnt['10']['neither']}; at HR@5: {hcnt['5']['saadi_above']} / {hcnt['5']['cr_above']} / {hcnt['5']['neither']}.")
    L.append(f"3. CR vs Vanilla: CR above Vanilla with the interval excluding zero in {cvc['10']['saadi_above']}/{cvc['10']['pairs']} pairs at HR@10 (below in {cvc['10']['cr_above']}); at HR@5 {cvc['5']['saadi_above']} above, {cvc['5']['cr_above']} below.")
    L.append(f"4. Cost, all datasets: CR {tot['cr_calls']} LLM calls vs SAADI {tot['saadi_calls']:.0f}; CR input tokens {tok(tot['cr_nominal'])} nominal ({f1(rn, 1)}x SAADI's {tok(tot['saadi_in'])}) and {tok(tot['cr_computed']) if tot['computed_known'] else '—'} after prefix-cache hits ({f1(rc, 2)}x); generated tokens {tok(tot['cr_gen'])} vs {tok(tot['saadi_gen'])} (SAADI: its per-chunk rates from the paper's cost table x C1's chunks).")
    L.append(f"5. Complementarity check (SAADI's published fusion with the text channel on CR text; not a proposed method): above CR in {ccnt['10']['saadi_above']}/{ccnt['10']['pairs']} pairs at HR@10 (below in {ccnt['10']['cr_above']}); above SAADI in {csc['10']['saadi_above']}/{csc['10']['pairs']} (below in {csc['10']['cr_above']}).")
    L.append("")
    L += ["## Deliberate differences from Anthropic's setup", ""]
    L += [f"- {x}" for x in cr.DELIBERATE_DIFFERENCES]
    L.append("- Vanilla is SAADI's text channel on the saved chunks (decided with the user): on MuSiQue-32K and QASPER-64K it equals Table 1's Vanilla to the rounding; on LooGLE and NovelHopQA Table 1's Vanilla came from another run, and both are shown.")
    L.append("")
    L += ["## Per retriever (HR@5 / HR@10, %)", ""]
    rows = []
    for x in pr_rows:
        ho = (held or {}).get((x["dataset"], x["retriever"]), {})
        rows.append([cr.DATASET_DISPLAY[x["dataset"]], cr.RETRIEVER_DISPLAY[x["retriever"]], f"{f1(x['vanilla_hr5'])} / {f1(x['vanilla_hr10'])}", f"{f1(x['table1_vanilla_hr5'])} / {f1(x['table1_vanilla_hr10'])}",
                     f"{f1(x['cr_hr5'])} / {f1(x['cr_hr10'])}", f"{f1(x['saadi_hr5'])} / {f1(x['saadi_hr10'])}", f"{f1(x['saadi_cr_hr5'])} / {f1(x['saadi_cr_hr10'])}",
                     f"{f1(x['lqca_hr5'])} / {f1(x['lqca_hr10'])}", ci({"diff": x["cr_minus_vanilla_hr10"], "ci": x["cr_minus_vanilla_hr10_ci"]}),
                     ci({"diff": x["saadi_minus_cr_hr10"], "ci": x["saadi_minus_cr_hr10_ci"]})] + ([f"{f1(ho.get('hr5'))} / {f1(ho.get('hr10'))}"] if held else []))
    L += md_table(["Dataset", "Retriever", "Vanilla", "Table 1 Vanilla", "CR", "SAADI", "SAADI on CR text*", "LQCA", "CR − Vanilla @10 [CI]", "SAADI − CR @10 [CI]"] + (["SAADI held-out (C2)"] if held else []), rows)
    L.append("*Complementarity check, not a proposed method. HR@5 intervals are in the per-dataset reports and tables/per_retriever.csv." + ("" if held else " SAADI held-out column left out: C2's REPORT.md does not exist yet."))
    L.append("")
    L += ["## Hybrid (dense + BM25, HR@5 / HR@10, %)", ""]
    if hy_rows:
        rows = [[cr.DATASET_DISPLAY[x["dataset"]], hybrid_label(x["dense"]), f"{f1(x['vanilla_hr5'])} / {f1(x['vanilla_hr10'])}", f"{f1(x['cr_hr5'])} / {f1(x['cr_hr10'])}",
                 f"{f1(x['saadi_hr5'])} / {f1(x['saadi_hr10'])}", ci({"diff": x["cr_minus_vanilla_hr10"], "ci": x["cr_minus_vanilla_hr10_ci"]}),
                 ci({"diff": x["saadi_minus_cr_hr10"], "ci": x["saadi_minus_cr_hr10_ci"]}), ci({"diff": x["saadi_minus_cr_hr5"], "ci": x["saadi_minus_cr_hr5_ci"]})] for x in hy_rows]
    else:
        rows = [["x", f"{cr.RETRIEVER_DISPLAY[d]} + BM25", "x / x", "x / x", "x / x", "+x [x, x]", "+x [x, x]", "+x [x, x]"] for d in cr.DENSE_RETRIEVERS]
        L.append("(No dense retriever in this run: placeholder layout.)")
        L.append("")
    L += md_table(["Dataset", "Hybrid", "Vanilla", "CR", "SAADI", "CR − Vanilla @10", "SAADI − CR @10", "SAADI − CR @5"], rows)
    L.append("")
    L += ["## Cost per dataset", ""]
    rows = []
    for d, rep in reports.items():
        c = rep["cost"]
        s = sa_all.get(d, {})
        rows.append([cr.DATASET_DISPLAY[d], c["calls"], f"{s.get('calls_per_chunk', 0) * (c.get('chunks') or 0):.0f}", tok(c["nominal_input_per_chunk"]), tok(c.get("computed_input_per_chunk")), tok(s.get("input_tokens_per_chunk")),
                     f1(c.get("generated_per_chunk_mean")), f1(s.get("generated_tokens_per_chunk")), f"{100 * (c.get('cut_share') or 0):.1f}%", f"{c['generation_gpu_hours']:.1f}"])
    L += md_table(["Dataset", "CR calls", "SAADI calls (its rate per chunk x C1's chunks)", "CR input/chunk nominal", "CR input/chunk computed", "SAADI input/chunk", "CR generated/chunk", "SAADI generated/chunk", "CR cut at 1024", "CR generation GPU-h"], rows)
    L.append("- SAADI's builds had no prefix caching; its input tokens compare with CR's nominal column like for like and with the computed column as CR's best case.")
    L.append("")
    L += ["## Appending text to a document", ""]
    per_new = []
    for d, rep in reports.items():
        s = sa_all.get(d, {})
        per_new.append(f"{cr.DATASET_DISPLAY[d]}: CR {tok(rep['cost']['nominal_input_per_chunk'])} vs SAADI {tok(s.get('input_tokens_per_chunk'))}")
    L.append("Not a measurement. When text is appended, CR needs the whole updated document as context for every new chunk (and the contexts already written may become stale, since they were written without the new text), while SAADI processes a new chunk with its current state. "
             "Input tokens per new chunk from the numbers above (CR without cache hits; with a cached prefix CR computes about the new chunk and the template, but the prompt still carries the whole document): " + "; ".join(per_new) + ".")
    L.append("")
    L += ["## Limits", ""]
    seg = [(cr.DATASET_DISPLAY[d], rep["census"]["docs_exceeding_window"], rep["census"]["chunks_in_segmented_docs"], rep["census"]["k_values"]) for d, rep in reports.items()]
    L.append("- Long documents: " + "; ".join(f"{d}: {n} documents ({c} chunks) over the window, segments {k}" for d, n, c, k in seg) + ". In those documents a chunk sees its own segment, not the whole book.")
    L.append("- Dense truncation: the context comes first, so a contextualized chunk longer than a retriever's input limit loses the end of the chunk (counts per retriever in the per-dataset reports).")
    L.append("- One construction LLM and one prompt; no reranker; results on these four benchmarks only.")
    L.append("")
    L += ["## Files", ""]
    L.append(f"- Per-dataset reports: {cr.rel(paths.reports_dir)}/<dataset>.md and .json; tables: {cr.rel(out_dir / 'tables')}/per_retriever.csv, hybrid.csv, cost.csv; contexts and rankings: {cr.rel(paths.out_dir)}/<dataset>/.")
    (out_dir / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    G = [f"**C1 Contextual Retrieval (CR) vs SAADI{dry}**", ""]
    G.append(f"- Setup: Anthropic's CR prompt with SAADI's construction LLM (Qwen3-30B-A3B-Instruct-2507, greedy), the paper's chunks, queries and labels, the five Table 1 retrievers, HR@5/@10, paired bootstrap (10,000, seed 42). Nothing tuned.")
    G.append(f"- Per retriever, SAADI − CR @10: SAADI above in {cnt['10']['saadi_above']}/{cnt['10']['pairs']} pairs, CR above in {cnt['10']['cr_above']}, n.s. {cnt['10']['neither']} (@5: {cnt['5']['saadi_above']}/{cnt['5']['cr_above']}/{cnt['5']['neither']}).")
    G.append(f"- Hybrid dense + BM25 (Anthropic's configuration): SAADI above in {hcnt['10']['saadi_above']}/{hcnt['10']['pairs']}, CR above in {hcnt['10']['cr_above']}, n.s. {hcnt['10']['neither']} (@5: {hcnt['5']['saadi_above']}/{hcnt['5']['cr_above']}/{hcnt['5']['neither']}).")
    G.append(f"- CR vs Vanilla @10: above in {cvc['10']['saadi_above']}/{cvc['10']['pairs']}, below in {cvc['10']['cr_above']}.")
    for x in pr_rows:
        if x["retriever"] == "bm25":
            G.append(f"- {cr.DATASET_DISPLAY[x['dataset']]} BM25 HR@10: Vanilla {f1(x['vanilla_hr10'])}, CR {f1(x['cr_hr10'])}, SAADI {f1(x['saadi_hr10'])} (SAADI − CR {ci({'diff': x['saadi_minus_cr_hr10'], 'ci': x['saadi_minus_cr_hr10_ci']})}).")
    G.append(f"- Cost: CR {tot['cr_calls']} calls vs SAADI {tot['saadi_calls']:.0f}; input tokens {f1(rn, 1)}x SAADI nominal, {f1(rc, 2)}x after prefix-cache hits; generated {tok(tot['cr_gen'])} vs {tok(tot['saadi_gen'])}.")
    G.append(f"- Complementarity (SAADI fusion on CR text, not a method): above CR in {ccnt['10']['saadi_above']}/{ccnt['10']['pairs']}, above SAADI in {csc['10']['saadi_above']}/{csc['10']['pairs']} @10.")
    G.append("- Differences from Anthropic: Qwen instead of Claude 3 Haiku, temperature 0, no reranker, SAADI's BM25 (one field) instead of Elasticsearch (two fields), document rebuilt from the chunks, long documents split into segments.")
    G.append("- Long documents: " + "; ".join(f"{d} {n} docs split ({k})" for d, n, c, k in seg if n) + ("" if any(n for _, n, _, _ in seg) else "none") + ".")
    if not complete:
        G.append("- INCOMPLETE: not every dataset/retriever is done.")
    (out_dir / "REPORT_FOR_GDOC.md").write_text("\n".join(G[:25]) + "\n", encoding="utf-8")
    log(f"final report: {cr.rel(out_dir / 'REPORT.md')} ({'complete' if complete else 'incomplete'})")
    return complete


# ---------------------------------------------------------------------------
# Group driver
# ---------------------------------------------------------------------------


def write_group_report(group: str, report: dict[str, Any], paths: cr.C1Paths) -> None:
    cr.write_json(paths.reports_dir / f"group_{group}.json", report)
    pf = report.get("preflight", {})
    L = [f"# C1 group {group}" + (f" ({paths.tag.replace('_', ' ')})" if paths.dry_run else ""), ""]
    L.append(f"- Started {report['started_at']}, finished {report.get('finished_at', '—')}; host {platform.node()}; exit code {report.get('exit_code')}.")
    L.append(f"- Command: `{' '.join(report['argv'])}`")
    L += ["", "## Preflight", ""]
    L.append(f"- OK: {pf.get('ok')}. Python {pf.get('python', {}).get('version')} ({pf.get('python', {}).get('executable')}).")
    L.append(f"- Modules: {', '.join(f'{k} {v}' for k, v in pf.get('modules', {}).items())}.")
    g = pf.get("git", {})
    L.append(f"- Commit {g.get('commit')} on {g.get('branch')}; uncommitted code changes: {g.get('code_changes_vs_commit') or 'none'}.")
    gp = pf.get("gpus")
    L.append("- GPUs: " + ("; ".join(f"{x['index']} {x['name']} {x['memory_used_mib']:.0f}/{x['memory_total_mib']:.0f} MiB used" for x in gp) if isinstance(gp, list) else str(gp)) + ".")
    if "disk" in pf:
        L.append(f"- Disk: {pf['disk']['free_gb']} GB free at {pf['disk']['path']}.")
    L.append(f"- Models: {', '.join(f'{k}: {v}' for k, v in pf.get('models', {}).items())}.")
    L.append(f"- C2 streams (memory and summary channels): {pf.get('c2_streams')}.")
    for p in pf.get("problems", []):
        L.append(f"- PROBLEM: {p}")
    for w in pf.get("warnings", []):
        L.append(f"- Warning: {w}")
    L += ["", "## Census and expected run time (written before generation)", ""]
    rows = []
    for d, e in report.get("estimates", {}).items():
        c = report["census"][d]
        rows.append([cr.DATASET_DISPLAY[d], c["n_docs"], c["n_chunks"], f"{tok(c['doc_tokens']['min'])} / {tok(c['doc_tokens']['median'])} / {tok(c['doc_tokens']['max'])}",
                     f"{c['docs_exceeding_window']} ({c['k_values']})", tok(c["nominal_input_tokens"]), fmt_s(e["generation"]["seconds"]), fmt_s(e["retrieval"]["wall_seconds"]),
                     fmt_s(e["generation"]["seconds"] + e["retrieval"]["wall_seconds"] + 300)])
    L += md_table(["Dataset", "Docs", "Chunks", "Doc tokens min / median / max", "Docs over window (k)", "Nominal input tokens", "Generation", "Retrieval", "Dataset total"], rows)
    L.append(f"- Generation basis: {cr.estimate_generation_seconds({'docs': []})['basis']}. Retrieval: C2's embedding cost model; dense jobs run in parallel on the group's GPUs.")
    mem = report.get("memory_check")
    if mem:
        L += ["", "## Memory check", ""]
        L += md_table(["Model", "GPUs", "Batch", "Peak MiB", "Fits"], [[cr.RETRIEVER_DISPLAY[r], m.get("n_gpus"), m.get("batch_size"), m.get("peak_mib"), m.get("fits")] for r, m in mem.get("models", {}).items()])
    L += ["", "## Datasets", ""]
    for d, st in report.get("datasets", {}).items():
        L.append(f"- {cr.DATASET_DISPLAY[d]}: {st}")
    for p in report.get("problems", []):
        L.append(f"- PROBLEM: {p}")
    (paths.reports_dir / f"group_{group}.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def run_group(group: str, args) -> int:
    paths = paths_for(args)
    datasets = cr.GROUP_DATASETS[group]
    paths.ensure(datasets)
    log = make_log(None)
    max_docs = max_docs_for(args)
    retrievers = retrievers_for(args)
    report: dict[str, Any] = {"group": group, "started_at": now(), "argv": [Path(sys.argv[0]).name, *sys.argv[1:]], "datasets": {}, "problems": []}
    pf = preflight(group, args, paths, log)
    report["preflight"] = pf
    if not pf["ok"]:
        report["exit_code"] = 2
        report["finished_at"] = now()
        write_group_report(group, report, paths)
        for p in pf["problems"]:
            log(f"PREFLIGHT PROBLEM: {p}")
        return 2
    log("preflight ok")
    censuses, estimates = {}, {}
    for d in datasets:
        log(f"census {d} ...")
        censuses[d] = cr.run_census(d, paths=paths, max_docs=max_docs, log=log)
        estimates[d] = {"generation": cr.estimate_generation_seconds(censuses[d]),
                        "retrieval": retrieval_estimates(d, censuses[d], retrievers, use_c2=not args.no_c2, log=log)}
        log(f"{d}: {censuses[d]['n_docs']} docs, {censuses[d]['n_chunks']} chunks, {censuses[d]['docs_exceeding_window']} over the window; estimate generation {fmt_s(estimates[d]['generation']['seconds'])}, retrieval {fmt_s(estimates[d]['retrieval']['wall_seconds'])}")
    report["census"] = {d: {k: v for k, v in c.items() if k != "docs"} for d, c in censuses.items()}
    report["estimates"] = estimates
    write_group_report(group, report, paths)
    memory = None
    if not paths.dry_run:
        memory = run_memory_checks(group, paths, max_docs=max_docs, log=log)
        report["memory_check"] = memory
        if not memory.get("ok"):
            report["problems"].append("memory check: some embedding model does not fit even on two GPUs at batch 1; that retriever is skipped")
        write_group_report(group, report, paths)
    exit_code = 0
    for d in datasets:
        problems: list[str] = []
        if paths.marker(d, "dataset.DONE").exists():
            log(f"{d}: already done, report kept")
            report["datasets"][d] = "already done"
            continue
        gen = run_generation(group, d, censuses[d], paths, args, log)
        if not gen["done"]:
            problems.append(f"generation incomplete: {gen['stats']['missing_chunks']} chunks without a context")
            exit_code = 1
            jobs = {}
        else:
            jobs = run_jobs(group, d, retrievers, paths, args, memory, estimates[d]["retrieval"], log)
        for r, j in jobs.items():
            if not str(j.get("status", "")).startswith("DONE"):
                problems.append(f"{r}: {j.get('status')}")
                exit_code = 1
        try:
            analysis = cr.analyze_dataset(d, paths=paths, retrievers=retrievers)
        except Exception:
            analysis = {"per_retriever": {}, "hybrid": {}}
            problems.append("analysis failed: " + traceback.format_exc().splitlines()[-1])
            log(traceback.format_exc())
            exit_code = 1
        dataset_report(d, paths, args, census=censuses[d], gen=gen, memory=memory, est=estimates[d], jobs_status=jobs, analysis=analysis, problems=problems)
        if not problems:
            cr.write_json(paths.marker(d, "dataset.DONE"), {"finished_at": now()})
        report["datasets"][d] = "done" if not problems else "problems: " + "; ".join(problems)
        log(f"{d}: report {cr.rel(paths.reports_dir / (d + '.md'))}")
        write_group_report(group, report, paths)
    with C2.file_lock(paths.out_dir / "final_report.lock"):
        complete = final_report(paths, args, log)
    if not complete:
        other = "B" if group == "A" else "A"
        log(f"final report incomplete (waiting for group {other} or a failed step); the last group to finish rewrites it")
    report["exit_code"] = exit_code
    report["finished_at"] = now()
    write_group_report(group, report, paths)
    return exit_code


def report_only(args) -> int:
    paths = paths_for(args)
    log = make_log(None)
    for d in cr.DATASET_KEYS:
        if not paths.census_path(d).exists():
            continue
        census = cr.read_json(paths.census_path(d))
        gen_runs = cr.read_json(paths.generation_summary_path(d)) if paths.generation_summary_path(d).exists() else {"runs": []}
        gen = {"done": paths.marker(d, "generation.DONE").exists(), "stats": cr.generation_stats(d, paths=paths, census=census), "runs": gen_runs["runs"]}
        group = next(g for g, ds in cr.GROUP_DATASETS.items() if d in ds)
        memory = cr.read_json(paths.memory_check_path(group)) if paths.memory_check_path(group).exists() else None
        old = paths.reports_dir / f"{d}.json"
        est = cr.read_json(old).get("estimates", {}) if old.exists() else {}
        analysis = cr.analyze_dataset(d, paths=paths, retrievers=retrievers_for(args))
        dataset_report(d, paths, args, census=census, gen=gen, memory=memory, est=est, jobs_status={}, analysis=analysis, problems=[])
        log(f"{d}: report rebuilt")
    with C2.file_lock(paths.out_dir / "final_report.lock"):
        final_report(paths, args, log)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", choices=("A", "B"))
    ap.add_argument("--dry-run", action="store_true", help="CPU: 2 documents per dataset, BM25 only, stub contexts")
    ap.add_argument("--dry-run-full", action="store_true", help="CPU: all documents, BM25 only, stub contexts")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--max-docs", type=int, default=None, help="documents per dataset (0 = all)")
    ap.add_argument("--no-c2", action="store_true", help="recompute memory/summary channels instead of reusing C2's streams")
    ap.add_argument("--run-job", help=argparse.SUPPRESS)
    ap.add_argument("--generate", help=argparse.SUPPRESS)
    ap.add_argument("--memory-check", help=argparse.SUPPRESS)
    ap.add_argument("--text-file", help=argparse.SUPPRESS)
    ap.add_argument("--out", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.memory_check:
        res = memory_check_worker(args.memory_check, Path(args.text_file).read_text(encoding="utf-8"), default_batch(args.memory_check))
        cr.write_json(Path(args.out), res)
        return 0 if res.get("fits") else 1
    if args.run_job:
        return job_worker(args.run_job, args)
    if args.generate:
        return generation_worker(args.generate, args)
    if args.report_only:
        return report_only(args)
    if not args.group:
        ap.error("--group is required (or --report-only)")
    return run_group(args.group, args)


if __name__ == "__main__":
    raise SystemExit(main())
