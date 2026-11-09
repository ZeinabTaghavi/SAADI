#!/usr/bin/env python3
"""C2: SAADI fusion weights chosen on held-out queries (ICLR rebuttal).

Started by experiments/rebuttal/C2_heldout_fusion/run_gpus_0123.sh (--group A, GPUs 0-3)
and run_gpus_4567.sh (--group B, GPUs 4-7). From the repository root:

    python scripts/rebuttal/c2_heldout_fusion.py --group A            # this group's jobs, then the analysis gate
    python scripts/rebuttal/c2_heldout_fusion.py --group B --analysis-only
    python scripts/rebuttal/c2_heldout_fusion.py --group A --analysis-only --retrievers bm25,contriever,bge_m3
    python scripts/rebuttal/c2_heldout_fusion.py --group A --dry-run  # CPU: BM25, 2 documents per dataset, small grid

Steps per group: preflight -> plan (first group writes plan.json under a file lock, the other
reads it) -> jobs (up to four at a time, one GPU each; BM25 on CPU next to them) -> group report
-> if all 20 jobs are DONE, the CPU analysis (once, under a lock), else "waiting for group X".
Resumable: finished jobs leave a DONE marker and are skipped; unfinished jobs resume from
per-document checkpoints. Exit code 0 = no failure, 1 = a job failed, 2 = preflight failed.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
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

from saadi.rebuttal import heldout_fusion as hf  # noqa: E402
from saadi.rebuttal.frozen_records import DATASETS, load_saved_live_rows, reference_dir  # noqa: E402

GROUP_GPUS = {"A": [0, 1, 2, 3], "B": [4, 5, 6, 7]}
OTHER = {"A": "B", "B": "A"}
GPU_BUSY_MIB = 2048
A6000_MIB = 48 * 1024
DRY_MAX_DOCS = 2

# Cost model (basis written into plan.json). Model sizes from the model cards; the adapters
# (src/saadi/indexing/retriever_adapters) pass no dtype, so weights are assumed float32.
MODEL_INFO: dict[str, dict[str, Any]] = {
    "contriever": {
        "hf_id": "facebook/contriever",
        "params_m": 110,
        "max_length": 512,
        "batch_size": 128,
        "load_s": 20,
        "query_call_s": 0.008,
    },
    "bge_m3": {"hf_id": "BAAI/bge-m3", "params_m": 568, "max_length": 8192, "batch_size": 16, "load_s": 40, "query_call_s": 0.015},
    "jina": {
        "hf_id": "jinaai/jina-embeddings-v3",
        "params_m": 572,
        "max_length": 512,
        "batch_size": 64,
        "load_s": 60,
        "query_call_s": 0.02,
    },
    "qwen": {
        "hf_id": "Qwen/Qwen3-Embedding-8B",
        "params_m": 7570,
        "max_length": 512,
        "batch_size": 64,
        "load_s": 240,
        "query_call_s": 0.06,
        # Measured on the server (2026-11-09): one A6000 runs out of memory (float32 weights ~28 GB plus
        # batch-64 activations). Two GPUs of the group, layers split with device_map="auto" (same arithmetic).
        "min_gpus": 2,
        "device_map_env": "SAADI_QWEN_DEVICE_MAP",
    },
}
EFFECTIVE_TFLOPS = 10.0  # float32 on one A6000 (peak 38.7), allowing for padding and small batches
CPU_SECONDS_PER_QUERY_CHUNK = 1.5e-4  # BM25 scoring + two combinations: 0.6-1.0e-4 measured on the laptop, x 1.5
ANALYSIS_SECONDS_PER_QUERY_CHUNK = 3.5e-4  # per worker: grid + verification, 2.0-2.5e-4 measured on the laptop (BM25, full grid), x 1.5
BYTES_PER_STREAM_ROW = 150


class PreflightError(RuntimeError):
    """Inputs missing: stop before any job, report as a preflight problem."""


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def fmt_s(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


@contextlib.contextmanager
def file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception as exc:
            return f"<git failed: {exc}>"

    status = run(
        "status",
        "--porcelain",
        "--",
        "src/saadi",
        "scripts/rebuttal",
        "experiments/rebuttal/C2_heldout_fusion/run_gpus_0123.sh",
        "experiments/rebuttal/C2_heldout_fusion/run_gpus_4567.sh",
    )
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "code_changes_vs_commit": status.splitlines(),
    }


def nvidia_smi() -> tuple[list[dict[str, Any]] | None, str]:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None, "nvidia-smi not found"
    try:
        out = subprocess.run(
            [exe, "--query-gpu=index,name,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception as exc:
        return None, f"nvidia-smi failed: {exc}"
    if out.returncode != 0:
        return None, out.stderr.strip() or f"nvidia-smi exit {out.returncode}"
    gpus = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            gpus.append({"index": int(parts[0]), "name": parts[1], "memory_used_mib": float(parts[2]), "memory_total_mib": float(parts[3])})
    return gpus, ""


def preflight_base(group: str, *, dry_run: bool, need_dense: bool) -> dict[str, Any]:
    pf: dict[str, Any] = {"ok": True, "problems": [], "warnings": []}
    pf["python"] = {"executable": sys.executable, "version": platform.python_version()}
    mods = {}
    for name in ("numpy", "pandas", "pyarrow", "tqdm", *(("torch", "transformers", "accelerate") if need_dense else ())):
        try:
            mod = __import__(name)
            mods[name] = getattr(mod, "__version__", "?")
        except Exception as exc:
            mods[name] = f"MISSING ({exc})"
            pf["problems"].append(f"python module {name} not importable: {exc}")
    try:
        import matplotlib  # noqa: F401

        mods["matplotlib"] = matplotlib.__version__
    except Exception:
        mods["matplotlib"] = "missing (heatmap PNGs will be skipped; values still go to tables/f_grid_hr.csv)"
        pf["warnings"].append("matplotlib not importable: heatmap PNGs will be skipped")
    pf["modules"] = mods
    if need_dense and "MISSING" not in str(mods.get("torch")):
        import torch

        pf["torch_cuda"] = {"available": bool(torch.cuda.is_available()), "cuda_version": torch.version.cuda}
    pf["git"] = git_state()
    pf["hf_home"] = os.environ.get("HF_HOME", "")
    if dry_run:
        pf["gpus"] = "skipped (dry run on CPU)"
    else:
        gpus, err = nvidia_smi()
        if gpus is None:
            pf["problems"].append(err)
            pf["gpus"] = err
        else:
            mine = [g for g in gpus if g["index"] in GROUP_GPUS[group]]
            pf["gpus"] = mine
            if len(mine) != len(GROUP_GPUS[group]):
                pf["problems"].append(f"expected GPUs {GROUP_GPUS[group]}, nvidia-smi shows {[g['index'] for g in gpus]}")
            for g in mine:
                if g["memory_used_mib"] > GPU_BUSY_MIB:
                    pf["problems"].append(
                        f"GPU {g['index']} already uses {g['memory_used_mib']:.0f} MiB (> {GPU_BUSY_MIB}); another job is running there. Not sharing GPUs."
                    )
    pf["ok"] = not pf["problems"]
    return pf


def preflight_inputs(jobs: list[dict[str, Any]], paths: hf.C2Paths, pf: dict[str, Any], plan: dict[str, Any]) -> None:
    missing = []
    for j in jobs:
        spec = DATASETS[j["dataset"]]
        for p in (REPO_ROOT / spec.index_json, REPO_ROOT / spec.saved_results_json(j["retriever"])):
            if not p.exists():
                missing.append(str(p.relative_to(REPO_ROOT)))
    try:
        ref = reference_dir(REPO_ROOT) / "expected_metrics.json"
        pf["expected_metrics"] = str(ref.relative_to(REPO_ROOT))
    except FileNotFoundError as exc:
        missing.append(str(exc))
    for extra in ("outputs/other_methods/vanilla/retrieval.csv", "results/tables/main/retrieval_comparison/table.csv"):
        if not (REPO_ROOT / extra).exists():
            missing.append(extra)
    pf["inputs_missing"] = missing
    if missing:
        pf["problems"].append(f"{len(missing)} input files missing (first: {missing[0]})")
    hub = Path(os.environ.get("HF_HUB_CACHE") or (Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"))
    models = {}
    for r in sorted({j["retriever"] for j in jobs if j["retriever"] != "bm25"}):
        hf_id = MODEL_INFO[r]["hf_id"]
        present = (hub / ("models--" + hf_id.replace("/", "--"))).exists()
        models[hf_id] = "in cache" if present else "NOT in cache (will download)"
        if not present:
            pf["warnings"].append(f"{hf_id} not found under {hub}; it will be downloaded")
    pf["models"] = models
    need = sum(j["stream_rows"] for j in jobs) * BYTES_PER_STREAM_ROW * 2 + 5 * 1024**3
    free = shutil.disk_usage(paths.out_dir).free
    pf["disk"] = {"path": str(paths.out_dir), "free_gb": round(free / 1024**3, 1), "needed_gb_estimate": round(need / 1024**3, 1)}
    if free < need:
        pf["problems"].append(f"free disk {free / 1024**3:.1f} GB < estimate {need / 1024**3:.1f} GB")
    pf["ok"] = not pf["problems"]


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


def text_counts(dataset: str, *, max_docs: int | None) -> dict[str, Any]:
    """Texts to embed per dataset (chunks + names + aliases + descriptions + level-1 summaries) and queries."""

    from saadi.indexing.retrieval_methods.mbr import build_mbr_document_memory_index

    spec = DATASETS[dataset]
    rows = load_saved_live_rows(spec, "bm25", repo_root=REPO_ROOT)
    by_doc = hf.rows_by_document(rows)
    doc_ids = list(by_doc)[:max_docs] if max_docs else list(by_doc)
    out = {"n_docs": len(doc_ids), "queries": 0, "query_chunks": 0, "texts": 0, "tokens_512": 0.0, "tokens_8192": 0.0, "pools": {}}
    pools = {"chunks": 0, "names": 0, "aliases": 0, "descriptions": 0, "summaries": 0}
    for doc_id in doc_ids:
        idx = build_mbr_document_memory_index(REPO_ROOT / spec.index_json, doc_id=doc_id)
        texts = {
            "chunks": [c.text for c in idx.chunk_pool],
            "names": list(idx.name_pool),
            "aliases": list(idx.alias_pool),
            "descriptions": list(idx.description_pool),
            "summaries": [s.text for s in idx.summary_pool],
        }
        for k, v in texts.items():
            pools[k] += len(v)
            for t in v:
                est = len(str(t)) / 4.0
                out["tokens_512"] += min(est, 512) + 2
                out["tokens_8192"] += min(est, 8192) + 2
        out["queries"] += len(by_doc[doc_id])
        out["query_chunks"] += len(by_doc[doc_id]) * len(idx.chunk_pool)
    out["pools"] = pools
    out["texts"] = sum(pools.values())
    return out


def estimate_job(dataset: str, retriever: str, counts: dict[str, Any]) -> dict[str, Any]:
    c = counts[dataset]
    cpu_s = c["query_chunks"] * CPU_SECONDS_PER_QUERY_CHUNK
    job = {
        "job": hf.job_name(dataset, retriever),
        "dataset": dataset,
        "retriever": retriever,
        "texts_to_embed": c["texts"] + c["queries"],
        "doc_texts": c["texts"],
        "queries": c["queries"],
        "query_encodes": 5 * c["queries"],
        "stream_rows": c["query_chunks"],
    }
    if retriever == "bm25":
        job.update(
            {"device": "cpu", "n_gpus": 0, "est_seconds": cpu_s, "est_basis": "CPU: query x chunk pairs x CPU_SECONDS_PER_QUERY_CHUNK"}
        )
        return job
    m = MODEL_INFO[retriever]
    tokens = c["tokens_8192"] if m["max_length"] > 512 else c["tokens_512"]
    enc_s = tokens * 2.0 * m["params_m"] * 1e6 / (EFFECTIVE_TFLOPS * 1e12)
    q_s = 5 * c["queries"] * m["query_call_s"]
    weights_gb = m["params_m"] * 1e6 * 4 / 1024**3
    n_gpus = max(int(m.get("min_gpus", 1)), 1 if weights_gb * 1024 + 8 * 1024 < A6000_MIB else 2)
    job.update(
        {
            "device": "gpu",
            "n_gpus": n_gpus,
            "model": m["hf_id"],
            "weights_gb_fp32": round(weights_gb, 1),
            "fits_one_a6000": n_gpus == 1,
            "est_tokens": round(tokens),
            "est_seconds": m["load_s"] + enc_s + q_s + cpu_s,
            "est_parts_seconds": {
                "load": m["load_s"],
                "encode_documents": round(enc_s),
                "encode_queries": round(q_s),
                "cpu_combine": round(cpu_s),
            },
            "est_basis": "load + doc tokens x 2 x params / effective FLOP/s + 5 query calls per query x latency + CPU part",
        }
    )
    return job


def simulate_makespan(jobs: list[dict[str, Any]], n_slots: int) -> float:
    slots = [0.0] * n_slots
    for j in sorted([j for j in jobs if j["device"] == "gpu"], key=lambda x: -x["est_seconds"]):
        need = j["n_gpus"]
        slots.sort()
        start = slots[need - 1]
        for i in range(need):
            slots[i] = start + j["est_seconds"]
    gpu_end = max(slots) if slots else 0.0
    cpu_end = sum(j["est_seconds"] for j in jobs if j["device"] == "cpu") / 2.0  # two CPU jobs at a time
    return max(gpu_end, cpu_end)


def build_plan(paths: hf.C2Paths, group: str, *, dry_run: bool, log, analysis_workers: int = 8) -> dict[str, Any]:
    t0 = time.perf_counter()
    jobs_all = [(d, r) for d, r in hf.ALL_JOBS if (r == "bm25" or not dry_run)]
    missing = []
    for d, r in jobs_all:
        spec = DATASETS[d]
        for p in (REPO_ROOT / spec.index_json, REPO_ROOT / spec.saved_results_json(r)):
            if not p.exists() and str(p) not in missing:
                missing.append(str(p))
    if missing:
        raise PreflightError(
            f"{len(missing)} input files of the paper's runs are missing (cache index.json / saved rankings), e.g. {missing[0]}. "
            "Copy the outputs/<dataset>_ablation/ folders to this checkout first."
        )
    counts = {}
    for d in hf.DATASET_KEYS:
        log(f"plan: counting texts of {d} ...")
        counts[d] = text_counts(d, max_docs=DRY_MAX_DOCS if dry_run else None)
    jobs = [estimate_job(d, r, counts) for d, r in jobs_all]
    totals = {"A": 0.0, "B": 0.0}
    for j in sorted([j for j in jobs if j["device"] == "gpu"], key=lambda x: -x["est_seconds"]):
        g = "A" if totals["A"] <= totals["B"] else "B"
        j["group"] = g
        totals[g] += j["est_seconds"] * j["n_gpus"]
    n_cpu = {"A": 0, "B": 0}
    for j in sorted([j for j in jobs if j["device"] == "cpu"], key=lambda x: -x["est_seconds"]):
        g = "A" if (n_cpu["A"] < 2 and (totals["A"] <= totals["B"] or n_cpu["B"] >= 2)) else "B"
        j["group"] = g
        n_cpu[g] += 1
        totals[g] += 0.0
    groups = {}
    for g in ("A", "B"):
        gj = [j for j in jobs if j["group"] == g]
        groups[g] = {
            "gpus": GROUP_GPUS[g],
            "jobs": [j["job"] for j in sorted(gj, key=lambda x: -x["est_seconds"])],
            "gpu_seconds": round(sum(j["est_seconds"] * j["n_gpus"] for j in gj if j["device"] == "gpu")),
            "est_wall_seconds": round(simulate_makespan(gj, len(GROUP_GPUS[g]))),
        }
    total_qc = sum(c["query_chunks"] for c in counts.values()) * (len(hf.RETRIEVERS) if not dry_run else 1)
    plan = {
        "created_at": now(),
        "created_by_group": group,
        "commit": git_state()["commit"],
        "dry_run": dry_run,
        "max_docs": DRY_MAX_DOCS if dry_run else None,
        "cost_basis": {
            "how": "estimated from model sizes (no timing run): seconds = model load + (estimated document tokens x 2 x parameters) / effective FLOP/s + 5 query encodings per query x per-call latency + CPU combination time",
            "tokens": "characters / 4 per text, capped at the model's max_length, + 2 special tokens",
            "effective_tflops_fp32": EFFECTIVE_TFLOPS,
            "cpu_seconds_per_query_chunk": CPU_SECONDS_PER_QUERY_CHUNK,
            "models": MODEL_INFO,
            "memory": "weights in float32 (the adapters pass no dtype); Qwen3-Embedding-8B ran out of memory on one 48 GB A6000 (about 28 GB of weights plus batch-64 activations), so each Qwen job gets two GPUs of its group with device_map=auto; every other job gets one GPU",
            "note": "estimates, not measurements; per-job times measured in this run are in the group reports",
        },
        "counts": counts,
        "jobs": jobs,
        "groups": groups,
        "analysis_est_seconds": round(
            total_qc * ANALYSIS_SECONDS_PER_QUERY_CHUNK / min(analysis_workers, len(jobs_all)) + (0 if dry_run else 300)
        ),
        "seconds_to_build": round(time.perf_counter() - t0, 1),
    }
    return plan


def apply_gpu_requirements(plan: dict[str, Any]) -> bool:
    """Bring an existing plan up to the current per-model GPU needs; True if anything changed."""

    changed = False
    for j in plan["jobs"]:
        need = int(MODEL_INFO.get(j["retriever"], {}).get("min_gpus", 1)) if j["device"] == "gpu" else 0
        if j["device"] == "gpu" and j["n_gpus"] < need:
            j["n_gpus"] = need
            j["fits_one_a6000"] = False
            changed = True
    if changed:
        plan["cost_basis"]["memory"] = (
            "weights in float32 (the adapters pass no dtype); Qwen3-Embedding-8B ran out of memory on one 48 GB A6000 "
            "(about 28 GB of weights plus batch-64 activations), so each Qwen job gets two GPUs of its group with device_map=auto; "
            "every other job gets one GPU"
        )
        for g, info in plan["groups"].items():
            gj = [j for j in plan["jobs"] if j["group"] == g]
            info["gpu_seconds"] = round(sum(j["est_seconds"] * j["n_gpus"] for j in gj if j["device"] == "gpu"))
            info["est_wall_seconds"] = round(simulate_makespan(gj, len(GROUP_GPUS[g])))
        plan["updated_at"] = now()
    return changed


def get_plan(paths: hf.C2Paths, group: str, *, dry_run: bool, log) -> tuple[dict[str, Any], bool]:
    with file_lock(paths.out_dir / "plan.lock"):
        if paths.plan_path.exists():
            plan = json.loads(paths.plan_path.read_text(encoding="utf-8"))
            if bool(plan.get("dry_run")) == dry_run:
                if apply_gpu_requirements(plan):
                    hf.write_json(paths.plan_path, plan)
                    log("plan updated: Qwen jobs now use two GPUs each (out of memory on one A6000)")
                return plan, False
        plan = build_plan(paths, group, dry_run=dry_run, log=log)
        hf.write_json(paths.plan_path, plan)
        return plan, True


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


def run_worker(job: str, *, dry_run: bool) -> int:
    """Child process: one (dataset, retriever) job. Writes DONE or FAILED itself."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout)
    paths = hf.C2Paths(REPO_ROOT, dry_run=dry_run)
    dataset, retriever = job.split("__")
    print(f"[{now()}] job {job} start; CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r}", flush=True)
    try:
        summary = hf.run_stream_job(
            dataset, retriever, paths=paths, max_docs=DRY_MAX_DOCS if dry_run else None, log=lambda m: print(f"[{now()}] {m}", flush=True)
        )
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        hf.write_json(paths.failed_marker(job), {"job": job, "status": "ERROR", "traceback": tb, "finished_at": now()})
        return 1
    print(f"[{now()}] job {job} {summary['status']}: {json.dumps(summary['check'].get('problems'))}", flush=True)
    return 0 if summary["status"] == "DONE" else 1


def run_group_jobs(group: str, jobs: list[dict[str, Any]], paths: hf.C2Paths, *, dry_run: bool, log) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    queue = []
    for j in sorted(jobs, key=lambda x: -x["est_seconds"]):
        if paths.done_marker(j["job"]).exists():
            results[j["job"]] = {
                "status": "DONE (skipped, already finished)",
                "summary": json.loads(paths.done_marker(j["job"]).read_text(encoding="utf-8")),
            }
            log(f"{j['job']}: already DONE, skipped")
        else:
            queue.append(j)
    free_gpus = list(GROUP_GPUS[group])
    running: dict[str, tuple[subprocess.Popen, list[int], float, Any, dict[str, Any]]] = {}
    cpu_slots = 2
    threads = os.environ.get("C2_THREADS", "8")
    while queue or running:
        started_any = False
        for j in list(queue):
            if j["device"] == "cpu":
                if sum(1 for r in running.values() if not r[1]) >= cpu_slots:
                    continue
                gpus: list[int] = []
            else:
                if dry_run:
                    gpus = []
                elif len(free_gpus) >= j["n_gpus"]:
                    gpus = [free_gpus.pop(0) for _ in range(j["n_gpus"])]
                else:
                    continue
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
            env.setdefault("SAADI_ENCODED_POOL_CACHE_SIZE", "32")
            if gpus:
                env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            dm_env = MODEL_INFO.get(j.get("retriever", ""), {}).get("device_map_env")
            if dm_env and len(gpus) > 1:
                env[dm_env] = "auto"
            env.setdefault("OMP_NUM_THREADS", threads)
            env.setdefault("MKL_NUM_THREADS", threads)
            env["TOKENIZERS_PARALLELISM"] = "false"
            log_path = paths.logs_dir / f"{j['job']}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("a", encoding="utf-8")
            cmd = [sys.executable, str(Path(__file__).resolve()), "--run-job", j["job"], "--group", group] + (
                ["--dry-run"] if dry_run else []
            )
            proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            running[j["job"]] = (proc, gpus, time.time(), handle, j)
            queue.remove(j)
            started_any = True
            where = f"GPU {','.join(map(str, gpus))}" if gpus else "CPU"
            log(f"{j['job']}: started on {where} (estimate {fmt_s(j['est_seconds'])}); log {hf.rel(log_path, REPO_ROOT)}")
        for name, (proc, gpus, t_start, handle, _j) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            handle.close()
            free_gpus.extend(gpus)
            free_gpus.sort()
            del running[name]
            secs = time.time() - t_start
            if paths.done_marker(name).exists():
                results[name] = {
                    "status": "DONE",
                    "seconds": secs,
                    "summary": json.loads(paths.done_marker(name).read_text(encoding="utf-8")),
                }
            elif paths.failed_marker(name).exists():
                fm = json.loads(paths.failed_marker(name).read_text(encoding="utf-8"))
                results[name] = {"status": fm.get("status", "FAILED"), "seconds": secs, "summary": fm}
            else:
                tail = (paths.logs_dir / f"{name}.log").read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
                results[name] = {"status": f"CRASHED (exit {rc})", "seconds": secs, "log_tail": tail}
            log(f"{name}: {results[name]['status']} after {fmt_s(secs)}")
        if not started_any:
            time.sleep(2.0 if dry_run else 10.0)
    return results


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def report_stem(group: str, report: dict[str, Any]) -> str:
    # --analysis-only keeps the group's job report and writes its own.
    return f"group_{group}" + ("_analysis_only" if report.get("analysis_only") else "")


def write_group_report(group: str, report: dict[str, Any], paths: hf.C2Paths) -> None:
    hf.write_json(paths.reports_dir / f"{report_stem(group, report)}.json", report)
    L = [f"# C2 group {group} report" + (" (dry run)" if report.get("dry_run") else ""), ""]
    L.append(
        f"- Started {report['started_at']}, finished {report.get('finished_at', '—')}; host {report['host']}; exit code {report.get('exit_code')}."
    )
    L.append(f"- Command: `{' '.join(report['argv'])}`")
    L.append("")
    pf = report.get("preflight", {})
    L.append("## Preflight")
    L.append("")
    L.append(f"- OK: {pf.get('ok')}. Python {pf.get('python', {}).get('version')} ({pf.get('python', {}).get('executable')}).")
    L.append(f"- Modules: {', '.join(f'{k} {v}' for k, v in pf.get('modules', {}).items())}.")
    git = pf.get("git", {})
    L.append(
        f"- Commit {git.get('commit')} on {git.get('branch')}; uncommitted code changes: {git.get('code_changes_vs_commit') or 'none'}."
    )
    gp = pf.get("gpus")
    if isinstance(gp, list):
        L.append(
            "- GPUs: "
            + "; ".join(f"{g['index']} {g['name']} {g['memory_used_mib']:.0f}/{g['memory_total_mib']:.0f} MiB used" for g in gp)
            + "."
        )
    else:
        L.append(f"- GPUs: {gp}.")
    if "disk" in pf:
        L.append(
            f"- Disk: {pf['disk']['free_gb']} GB free at {pf['disk']['path']} (estimate needed {pf['disk']['needed_gb_estimate']} GB)."
        )
    if pf.get("models"):
        L.append(f"- Models: {', '.join(f'{k}: {v}' for k, v in pf['models'].items())}.")
    for p in pf.get("problems", []):
        L.append(f"- PROBLEM: {p}")
    for w in pf.get("warnings", []):
        L.append(f"- Warning: {w}")
    L.append("")
    plan = report.get("plan")
    if plan:
        L.append("## Plan")
        L.append("")
        L.append(
            f"- plan.json {'written by this group' if report.get('plan_written') else 'read (written by group ' + str(plan.get('created_by_group')) + ')'} at {plan.get('created_at')}."
        )
        L.append(f"- Cost basis: {plan['cost_basis']['how']}. {plan['cost_basis']['memory']}.")
        g = plan["groups"][group]
        L.append(
            f"- This group: GPUs {g['gpus']}; estimated wall time {fmt_s(g['est_wall_seconds'])} (other group: {fmt_s(plan['groups'][OTHER[group]]['est_wall_seconds'])}); analysis afterwards about {fmt_s(plan.get('analysis_est_seconds', 0))}."
        )
        L.append("")
        L.append("| Job | Device | Texts to embed | Queries | Estimate |")
        L.append("| --- | --- | --- | --- | --- |")
        for j in sorted([j for j in plan["jobs"] if j["group"] == group], key=lambda x: -x["est_seconds"]):
            L.append(
                f"| {j['job']} | {j['device']} x{max(1, j['n_gpus'])} | {j['texts_to_embed']} | {j['queries']} | {fmt_s(j['est_seconds'])} |"
            )
        L.append("")
    jobs = report.get("jobs", {})
    if jobs:
        L.append("## Jobs and reproduction checks")
        L.append("")
        L.append("| Job | Status | Time | Identical top-10 | HR@5 / HR@10 ours vs saved | text-only vs Table 1 Vanilla | Problems |")
        L.append("| --- | --- | --- | --- | --- | --- | --- |")
        for name, r in jobs.items():
            s = r.get("summary", {})
            c = s.get("check", {})
            if c:
                hr = f"{hf._f(c['hr_ours_published']['hr_percent']['5'], 2)} / {hf._f(c['hr_ours_published']['hr_percent']['10'], 2)} vs {hf._f(c['hr_saved']['hr_percent']['5'], 2)} / {hf._f(c['hr_saved']['hr_percent']['10'], 2)}"
                to = c["text_only"]
                van = (
                    f"{hf._f(to['hr']['hr_percent']['5'], 2)} / {hf._f(to['hr']['hr_percent']['10'], 2)} vs {hf._f(to['reported_percent']['5'])} / {hf._f(to['reported_percent']['10'])}"
                    + ("" if to["passed"] else " (differs)")
                )
                ident = f"{c['identical_top10']}/{c['n_queries']}"
                probs = "; ".join(c.get("problems", [])) or (
                    "drift accepted: " + "; ".join(c.get("drift_notes", [])[:-1]) if c.get("drift_accepted") else "none"
                )
            else:
                hr = van = ident = "—"
                probs = (s.get("traceback", "") or "\n".join(r.get("log_tail", [])))[-300:].replace("\n", " ").replace("|", "/") or "—"
            secs = r.get("seconds", s.get("seconds"))
            L.append(f"| {name} | {r['status']} | {fmt_s(secs) if secs else '—'} | {ident} | {hr} | {van} | {probs} |")
        L.append("")
        notes = [(n, r) for n, r in jobs.items() if r.get("summary", {}).get("check", {}).get("text_only", {}).get("notes")]
        for n, r in notes:
            L.append(f"- {n} text-only note: {'; '.join(r['summary']['check']['text_only']['notes'])}")
        for n, r in jobs.items():
            ex = r.get("summary", {}).get("check", {}).get("mismatch_examples")
            if ex and not r["status"].startswith("DONE"):
                L.append(f"- {n} mismatch example: `{json.dumps(ex[0])[:600]}`")
        L.append("")
    if report.get("unexpected"):
        L.append("## Unexpected")
        L.append("")
        for u in report["unexpected"]:
            L.append("```")
            L.append(str(u)[-3000:])
            L.append("```")
        L.append("")
    L.append("## Analysis")
    L.append("")
    L.append(f"- {report.get('analysis', 'not reached')}")
    L.append("")
    L.append("## Files written")
    L.append("")
    for f in report.get("files", []):
        L.append(f"- `{f}`")
    (paths.reports_dir / f"{report_stem(group, report)}.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def analysis_gate(
    group: str,
    paths: hf.C2Paths,
    *,
    dry_run: bool,
    force: bool,
    workers: int,
    report: dict[str, Any],
    log,
    retrievers: tuple[str, ...] | None = None,
) -> bool:
    """Run the analysis once all jobs (of the chosen retrievers) are DONE. Returns False if the analysis failed."""

    expected = [hf.job_name(d, r) for d, r in hf.analysis_jobs(retrievers, dry_run=dry_run)]
    with file_lock(paths.out_dir / "analysis.lock"):
        done = [j for j in expected if paths.done_marker(j).exists()]
        if len(done) < len(expected):
            plan = json.loads(paths.plan_path.read_text(encoding="utf-8")) if paths.plan_path.exists() else {"jobs": []}
            owner = {j["job"]: j.get("group") for j in plan["jobs"]}
            missing = [j for j in expected if j not in done]
            waiting = sorted({owner.get(j, "?") for j in missing})
            failed_here = [j for j in missing if owner.get(j) == group]
            msg = (
                f"{len(done)}/{len(expected)} jobs DONE; "
                + (f"this group's unfinished jobs: {failed_here}; " if failed_here else "")
                + f"waiting for group {' and '.join(w for w in waiting if w != group) or group}"
            )
            report["analysis"] = msg
            log(msg)
            return True
        marker = paths.out_dir / "ANALYSIS_DONE.json"
        if marker.exists() and not force:
            newest = max(paths.done_marker(j).stat().st_mtime for j in expected)
            if marker.stat().st_mtime >= newest:
                report["analysis"] = f"already run ({json.loads(marker.read_text())['finished_at']}); use --analysis-only to rerun"
                log(report["analysis"])
                return True
        scope = f" for {', '.join(retrievers)}" if retrievers else ""
        log(f"all {len(expected)} jobs{scope} DONE; running the analysis on CPU ({workers} workers)")
        t0 = time.time()
        try:
            a = hf.run_analysis(paths, workers=workers, log=log, retrievers=retrievers)
        except Exception:
            tb = traceback.format_exc()
            log(tb)
            report["unexpected"].append(tb)
            report["analysis"] = "FAILED (traceback above)"
            return False
        report["analysis"] = f"done in {fmt_s(time.time() - t0)}: {hf.rel(paths.results_dir / 'REPORT.md', REPO_ROOT)}"
        report["files"] += [hf.rel(p, REPO_ROOT) for p in sorted(paths.results_dir.rglob("*")) if p.is_file()]
        report["analysis_counts"] = a["counts"]
        return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", choices=["A", "B"], required=True)
    ap.add_argument("--analysis-only", action="store_true", help="rerun the CPU analysis only (all 20 jobs must be DONE)")
    ap.add_argument(
        "--retrievers",
        help="with --analysis-only: comma-separated retrievers to analyse (e.g. bm25,contriever,bge_m3); their jobs on all four datasets must be DONE",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="CPU test: BM25 only, 2 documents per dataset, reduced grid; separate dry_run/ folders"
    )
    ap.add_argument("--workers", type=int, default=int(os.environ.get("C2_ANALYSIS_WORKERS", "8")), help="CPU processes for the analysis")
    ap.add_argument("--run-job", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.run_job:
        return run_worker(args.run_job, dry_run=args.dry_run)
    retrievers = tuple(x.strip() for x in args.retrievers.split(",") if x.strip()) if args.retrievers else None
    if retrievers and not args.analysis_only:
        ap.error("--retrievers only works with --analysis-only")
    if retrievers:
        try:
            hf.analysis_jobs(retrievers)
        except ValueError as exc:
            ap.error(str(exc))

    paths = hf.C2Paths(REPO_ROOT, dry_run=args.dry_run)
    paths.ensure()
    group = args.group
    report: dict[str, Any] = {
        "group": group,
        "dry_run": args.dry_run,
        "started_at": now(),
        "host": platform.node(),
        "argv": [Path(sys.executable).name, *sys.argv],
        "analysis_only": bool(args.analysis_only),
        "retrievers": list(retrievers) if retrievers else list(hf.RETRIEVERS),
        "unexpected": [],
        "files": [],
    }

    def log(msg: str) -> None:
        print(f"[{now()}] [group {group}] {msg}", flush=True)

    exit_code = 0
    try:
        if args.analysis_only:
            report["preflight"] = preflight_base(group, dry_run=True, need_dense=False) | {"gpus": "not needed (--analysis-only)"}
            ok = analysis_gate(
                group, paths, dry_run=args.dry_run, force=True, workers=args.workers, report=report, log=log, retrievers=retrievers
            )
            exit_code = 0 if ok and not report["analysis"].startswith(tuple("0123456789")) else 1
        else:
            pf = preflight_base(group, dry_run=args.dry_run, need_dense=not args.dry_run)
            report["preflight"] = pf
            if not pf["ok"]:
                for p in pf["problems"]:
                    log(f"PREFLIGHT PROBLEM: {p}")
                exit_code = 2
            else:
                try:
                    plan, written = get_plan(paths, group, dry_run=args.dry_run, log=log)
                except PreflightError as exc:
                    pf["problems"].append(str(exc))
                    pf["ok"] = False
                    log(f"PREFLIGHT PROBLEM: {exc}")
                    raise
                report["plan"] = plan
                report["plan_written"] = written
                if written:
                    report["files"].append(hf.rel(paths.plan_path, REPO_ROOT))
                jobs = [j for j in plan["jobs"] if j["group"] == group]
                preflight_inputs(jobs, paths, pf, plan)
                for w in pf["warnings"]:
                    log(f"warning: {w}")
                if not pf["ok"]:
                    for p in pf["problems"]:
                        log(f"PREFLIGHT PROBLEM: {p}")
                    exit_code = 2
                else:
                    log(f"plan {'written' if written else 'read'}: {paths.plan_path.relative_to(REPO_ROOT)}")
                    for j in sorted(jobs, key=lambda x: -x["est_seconds"]):
                        log(
                            f"  {j['job']:<24} {j['device']} x{max(1, j['n_gpus'])}  texts {j['texts_to_embed']:>8}  queries {j['queries']:>5}  estimate {fmt_s(j['est_seconds'])}"
                        )
                    g = plan["groups"][group]
                    log(
                        f"expected wall time for group {group}: {fmt_s(g['est_wall_seconds'])} (group {OTHER[group]}: {fmt_s(plan['groups'][OTHER[group]]['est_wall_seconds'])}); analysis afterwards: about {fmt_s(plan.get('analysis_est_seconds', 0))}"
                    )
                    write_group_report(group, report, paths)
                    results = run_group_jobs(group, jobs, paths, dry_run=args.dry_run, log=log)
                    report["jobs"] = results
                    for name, r in results.items():
                        if r["status"].startswith("DONE"):
                            report["files"] += r.get("summary", {}).get("files", [])
                        else:
                            exit_code = 1
                            if "traceback" in r.get("summary", {}):
                                report["unexpected"].append(f"{name}:\n{r['summary']['traceback']}")
                            if "log_tail" in r:
                                report["unexpected"].append(f"{name} (log tail):\n" + "\n".join(r["log_tail"]))
                    if not analysis_gate(group, paths, dry_run=args.dry_run, force=False, workers=args.workers, report=report, log=log):
                        exit_code = 1
    except PreflightError:
        exit_code = 2
    except Exception:
        tb = traceback.format_exc()
        log(tb)
        report["unexpected"].append(tb)
        exit_code = 1
    report["finished_at"] = now()
    report["exit_code"] = exit_code
    report["files"] += [hf.rel(paths.reports_dir / f"{report_stem(group, report)}.{ext}", REPO_ROOT) for ext in ("md", "json")]
    write_group_report(group, report, paths)
    log(f"report: {hf.rel(paths.reports_dir / f'{report_stem(group, report)}.md', REPO_ROOT)}; exit code {exit_code}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
