"""C5 step 01 (GPU, MuSiQue-32K only): RAPTOR with its saved default settings on SAADI's corpus.

The saved RAPTOR MuSiQue-32K run is on RAPTOR's own 15-group corpus (300 questions), so the
C5 queries (SAADI's 45 groups, 900 questions) have no RAPTOR output. This step:
  1. writes a RAPTOR "prepared_expanded" bundle from SAADI's prepared files
     (outputs/musique_ablation/32K_expansion/prepared: documents.json, qa.json,
     group_membership.json, context_expansion_metadata.json), copied unchanged, plus the
     manifest.json with their checksums that RAPTOR's loader requires;
  2. writes a run config = RAPTOR's own configs/raptor/musique_32k_retrieval_ablation.yaml with
     ONLY the data location and the expected counts changed (45 documents, 900 questions);
     every tree-builder, retrieval and model setting is the saved one;
  3. runs RAPTOR's own runner (scripts/run_raptor_experiment.py, retrieval only, top_k 10),
     resuming a partial run, then RAPTOR's own evaluator (upper-bound labels and HR);
  4. reports. Nothing is written inside the RAPTOR repository (outputs go to C5_RAPTOR_RERUN_DIR).

RAPTOR summarises with its own in-process vLLM engine (same model and weights as the reader);
it cannot call an HTTP server, so it runs on the group's GPUs before the reader server starts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import c5_common as cc

DATASET = "musique_32k"
RUN_NAME = "musique_32k_retrieval_ablation_raptor"
BUNDLE_FILES = ("context_expansion_metadata.json", "documents.json", "group_membership.json", "qa.json")
STATUS = {0: "complete", 10: "missing", 11: "partial", 12: "incompatible"}


def log(msg: str) -> None:
    print(f"[c5-01] {cc.now()} {msg}", flush=True)


def write_if_changed(path: Path, text: str) -> bool:
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return True


def build_bundle(src: Path, dest: Path) -> dict[str, Any]:
    dest.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in BUNDLE_FILES:
        s, d = src / name, dest / name
        if not s.exists():
            raise FileNotFoundError(f"SAADI prepared file missing: {s}")
        if not d.exists() or cc.sha256_file(d) != cc.sha256_file(s):
            shutil.copyfile(s, d)
        files[name] = {"sha256": cc.sha256_file(d)}
    meta = cc.read_json(dest / "context_expansion_metadata.json")
    manifest = {
        "format_version": 1,
        "dataset_variant": DATASET,
        "dataset": meta["dataset"],
        "split": meta["split"],
        "target_context_tokens": int(meta["target_context_tokens"]),
        "selection": {
            "max_source_documents": meta.get("max_source_documents"),
            "selected_source_documents": meta.get("selected_source_documents"),
            "qa_entries": meta.get("qa_entries"),
            "prepared_documents": meta.get("prepared_documents"),
        },
        "source": "SAADI outputs/musique_ablation/32K_expansion/prepared (copied unchanged by C5)",
        "files": files,
    }
    write_if_changed(dest / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"manifest_sha256": cc.sha256_file(dest / "manifest.json"), "documents": int(meta["prepared_documents"]), "qa_entries": int(meta["qa_entries"]), "metadata": meta}


def build_config(raptor_dir: Path, bundle: Path, info: dict[str, Any], dest: Path) -> tuple[Path, list[str]]:
    import yaml  # RAPTOR's own environment has PyYAML

    base_path = raptor_dir / "configs/raptor/musique_32k_retrieval_ablation.yaml"
    cfg = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    run = cfg["raptor_run"]
    loader = run["dataset"]["loader"]
    changed = []
    for key, value in (("data_dir", str(bundle)), ("expected_qa_entries", info["qa_entries"]),
                       ("expected_prepared_documents", info["documents"]), ("expected_manifest_sha256", info["manifest_sha256"])):
        if loader.get(key) != value:
            changed.append(f"loader.{key}: {loader.get(key)!r} -> {value!r}")
            loader[key] = value
    text = "# C5: RAPTOR's configs/raptor/musique_32k_retrieval_ablation.yaml with only the data location and\n" \
           "# the expected counts changed (SAADI's 45-group MuSiQue-32K corpus). Every other setting is the saved one.\n" + \
           yaml.safe_dump(cfg, sort_keys=False)
    path = dest / "musique_32k_c5_retrieval.yaml"
    write_if_changed(path, text)  # byte-identical on a restart, so RAPTOR's resume check accepts it
    return path, changed


def run(cmd: list[str], cwd: Path, logf) -> int:
    log("$ " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    sys.stdout.write(proc.stdout)
    sys.stdout.flush()
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.parse_args(argv)
    paths = cc.C5Paths.from_env()
    env = dict(os.environ)
    rerun = Path(env["C5_RAPTOR_RERUN_DIR"])
    need = ("raptor/dataset_loaders/prepared_expanded.py", "scripts/check_raptor_run_status.py", "scripts/run_raptor_experiment.py",
            "scripts/evaluate_raptor_retrieval_upper_bound.py", "configs/raptor/musique_32k_retrieval_ablation.yaml")
    cands = [Path(x) for x in [env.get("C5_RAPTOR_CODE_DIR", ""), env.get("C5_RAPTOR_DIR", ""), *env.get("C5_RAPTOR_SEARCH", "").split(":")] if x]
    raptor_dir = next((c for c in cands if all((c / n).exists() for n in need)), cands[0] if cands else Path("."))
    py = env.get("C5_RAPTOR_PYTHON") or (str(raptor_dir / ".venv/bin/python") if (raptor_dir / ".venv/bin/python").exists() else sys.executable)
    runs_root = rerun / env.get("C5_RAPTOR_RUNS", "raptor_10_runs")
    evals_root = rerun / env.get("C5_RAPTOR_EVALS", "raptor_10_evaluations")
    run_dir = runs_root / DATASET / RUN_NAME
    eval_dir = evals_root / DATASET / RUN_NAME / "retrieval_upper_bound"
    info: dict[str, Any] = {"step": "01", "dataset": DATASET, "raptor_dir": str(raptor_dir), "run_dir": str(run_dir), "eval_dir": str(eval_dir),
                            "python": py, "gpus": env.get("CUDA_VISIBLE_DEVICES"), "commit": cc.git_commit(paths.root), "host": os.uname().nodename}
    status, code, t0 = "OK", 0, time.time()
    try:
        missing = [n for n in need if not (raptor_dir / n).exists()]
        if missing:
            raise RuntimeError(f"no RAPTOR copy with the MuSiQue runner among {[str(c) for c in cands]} (missing {missing}); set C5_RAPTOR_CODE_DIR")
        probe = subprocess.run([py, "-c", "import umap, sklearn, tiktoken, yaml, torch, transformers, vllm; print(torch.cuda.device_count())"],
                               cwd=raptor_dir, capture_output=True, text=True)
        info["python_check"] = (probe.stdout.strip() or probe.stderr.strip()[-300:])
        if probe.returncode != 0:
            raise RuntimeError(f"RAPTOR's Python {py} lacks a package RAPTOR needs: {probe.stderr.strip()[-300:]}; set C5_RAPTOR_PYTHON to RAPTOR's environment")
        if probe.stdout.strip() in ("", "0"):
            raise RuntimeError(f"RAPTOR's Python {py} sees no GPU (CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')})")
        src = Path(env["C5_MUSIQUE_QA"]).parent
        bundle = rerun / "data" / "musique_32k_c5"
        b = build_bundle(src, bundle)
        info["bundle"] = {"path": str(bundle), **{k: v for k, v in b.items() if k != "metadata"}}
        cfg, changed = build_config(raptor_dir, bundle, b, rerun / "configs")
        info["config"], info["config_changes"] = str(cfg), changed
        check = [py, "scripts/check_raptor_run_status.py", "--run-dir", str(run_dir), "--expected-config", str(cfg),
                 "--expected-queries", str(b["qa_entries"]), "--expected-dataset", DATASET, "--expected-top-k", "10"]
        st = run(check, raptor_dir, None)
        info["status_before"] = STATUS.get(st, st)
        if st == 12:
            raise RuntimeError(f"an incompatible RAPTOR run exists at {run_dir}; move it aside")
        if st != 0:
            cmd = [py, "scripts/run_raptor_experiment.py", "--dataset-name", DATASET, "--default-yaml", str(cfg),
                   "--output-root", str(runs_root), "--retrieval-top-k", "10"] + (["--resume"] if st == 11 else [])
            rc = run(cmd, raptor_dir, None)
            info["runner_exit"] = rc
            if rc != 0:
                raise RuntimeError(f"RAPTOR runner exited {rc}")
            st = run(check, raptor_dir, None)
            info["status_after"] = STATUS.get(st, st)
            if st != 0:
                raise RuntimeError(f"RAPTOR run not complete after the runner (status {STATUS.get(st, st)})")
        if not (eval_dir / "metrics_per_query.jsonl").exists() or (eval_dir / "metrics_per_query.jsonl").stat().st_mtime < (run_dir / "retrieval/retrieval_payloads.jsonl").stat().st_mtime:
            rc = run([py, "scripts/evaluate_raptor_retrieval_upper_bound.py", "--run-dir", str(run_dir), "--dataset-name", DATASET,
                      "--ks", "5", "10", "--output-dir", str(eval_dir)], raptor_dir, None)
            if rc != 0:
                raise RuntimeError(f"RAPTOR evaluator exited {rc}")
        summ = cc.read_json(eval_dir / "metrics_summary.json")
        info["evaluation"] = {k: summ.get(k) for k in ("num_queries", "num_queries_without_labels", "num_queries_unmatched")}
        info["evaluation"]["hits"] = summ.get("hits")
        bt = run_dir / "profiling/build_times.jsonl"
        if bt.exists():
            t = [float(r.get("build_time_ms", 0)) / 1000 for r in cc.read_jsonl(bt)]
            info["build_seconds"] = {"documents": len(t), "total": sum(t)}
        if info["evaluation"]["num_queries"] != b["qa_entries"]:
            raise RuntimeError(f"evaluation covers {info['evaluation']['num_queries']} queries, expected {b['qa_entries']}")
        info["one_line"] = f"RAPTOR re-run on 45 groups complete ({info['evaluation']['num_queries']} queries)"
    except Exception as exc:
        status, code = "FAILED", 1
        info["traceback"] = traceback.format_exc()
        info["one_line"] = f"FAILED: {type(exc).__name__}: {str(exc)[:110]}"
    info["seconds"] = round(time.time() - t0)
    rep = paths.reports / f"step_01_{DATASET}_raptor.md"
    L = [f"# C5 end-to-end QA - step 01 RAPTOR re-run, {cc.DATASET_DISPLAY[DATASET]}: {status}", "",
         f"Written {cc.now()}; commit {info['commit']}; host {info['host']}; GPUs {info['gpus']}; {info['seconds']} s.", "",
         "Command: `bash experiments/rebuttal/c5_end_to_end/sh/run_gpus_0123.sh` (stage raptor)", "",
         "## What was run", "",
         f"- RAPTOR repository `{raptor_dir}` (not modified), Python `{py}`.",
         f"- Input: SAADI's prepared MuSiQue-32K files copied unchanged into `{(info.get('bundle') or {}).get('path')}`, with the manifest RAPTOR's loader needs (sha256 {(info.get('bundle') or {}).get('manifest_sha256')}).",
         f"- Config `{info.get('config')}`: RAPTOR's saved musique_32k config; changed only: {'; '.join(info.get('config_changes') or []) or 'nothing'}.",
         "- Settings (saved = RAPTOR defaults except the models): leaf 100 cl100k tokens, 5 layers, threshold 0.5, top_k 5 per cluster, summaries up to 150 tokens, "
         "collapsed-tree retrieval top_k 10 with max_tokens 3500, Contriever embeddings, Qwen3-30B-A3B-Instruct-2507 summariser (vLLM in-process, TP 4, temperature 0).",
         f"- Run status before: {info.get('status_before')}; after: {info.get('status_after', info.get('status_before'))}.", "",
         "## Result", ""]
    ev = info.get("evaluation") or {}
    L += cc.md_table(["Queries", "Without labels", "Unmatched", "Build time (s, documents)"],
                     [[ev.get("num_queries"), ev.get("num_queries_without_labels"), ev.get("num_queries_unmatched"), json.dumps(info.get("build_seconds"))]])
    if ev.get("hits"):
        L += ["", f"RAPTOR's own upper-bound hit rates: `{json.dumps(ev['hits'])}`"]
    if info.get("traceback"):
        L += ["", "## Error", "", "```text", *info["traceback"].splitlines()[-60:], "```"]
    L += ["", "## Files written", "", f"- `{run_dir}` (raw RAPTOR run, not pushed)", f"- `{eval_dir}` (not pushed)", f"- `{paths.rel(rep)}`, `{paths.rel(rep.with_suffix('.json'))}`"]
    rep.write_text("\n".join(L) + "\n", encoding="utf-8")
    cc.write_json(rep.with_suffix(".json"), info)
    rep.with_suffix(".summary.txt").write_text(info["one_line"] + "\n", encoding="utf-8")
    log(f"done: {status}")
    return code


if __name__ == "__main__":
    sys.exit(main())
