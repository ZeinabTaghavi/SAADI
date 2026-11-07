#!/usr/bin/env python3
"""Step 5 (the GPU step): index the 10 modified books with the published profile.

Runs the standard study runner (scripts/run_study.py) so the construction is
exactly the paper's. Resumable: the construction cache saves as it goes, and a
rerun of this script continues where the last run stopped. Accepts
--tensor-parallel-size N, --gpus 0,1,2,3 and --dry-run (which resolves the
configuration without touching a GPU and does not unlock step 6).

This step writes the one declared file outside the probe folders:
configs/experiments/novelhopqa_probe_retrieval_ablation.yaml. The note named
configs/probe_bridge/novelhopqa_probe.yaml, but the runner only accepts
configs/experiments/<dataset>_retrieval_ablation.yaml, so that is where it goes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402
from common import (  # noqa: E402
    OUT_DIR,
    PERSONAS_JSONL,
    PROBE_CONFIG,
    ROOT,
    SPEC_FROZEN_JSON,
    StepReport,
    read_json,
    run_step,
    sha256_file,
)

CONFIG_TEXT = """\
# Bridge probe (experiments/probe_bridge): the single declared file outside the
# probe's own folders. It indexes the 10 modified books through the standard
# runner. See experiments/probe_bridge/README.md and reports/step_05.md.
extends: ../methods/sadi.yaml
output_dir: ${SADI_OUTPUT_DIR}/novelhopqa_probe
dataset:
  name: novelhopqa
  source: files
  documents_json: ../../experiments/probe_bridge/data/documents.json
  qa_json: ../../experiments/probe_bridge/data/qa.json
  split: test
  config_name: all
  qa_n: all
  qa_selection_method: first
preparation: null
ingest:
  chunk_size: 500
model:
  model_id: Qwen/Qwen3-30B-A3B-Instruct-2507
  alias: Qwen3-30B-A3B-Instruct-2507
  gpu_memory_utilization: 0.85
"""


def _arg_value(argv: list[str], name: str) -> str | None:
    if name in argv:
        i = argv.index(name)
        if i + 1 < len(argv):
            return argv[i + 1]
    return None


def scan_cache(report: StepReport) -> tuple[int, list[list]]:
    rows: list[list] = []
    finished = 0
    try:
        index_path = common.find_cache_index()
    except FileNotFoundError:
        report.add("Construction records", "No cache index found yet.")
        return 0, rows
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    from saadi.indexing.retrieval_methods.mbr import _resolve_cache_file_path

    for doc in payload.get("docs", []):
        doc_id = str(doc.get("doc_id"))
        files = doc.get("files", {}) or {}
        steps_path = _resolve_cache_file_path(files.get("steps_index", ""), base_dir=index_path.parent)
        memory_path = _resolve_cache_file_path(files.get("memory", ""), base_dir=index_path.parent)
        n_steps = parser_failures = 0
        if Path(steps_path).exists():
            for line in open(steps_path, encoding="utf-8"):
                n_steps += 1
                if '"parser_status": "ok"' not in line:
                    parser_failures += 1
        done = Path(memory_path).exists()
        finished += int(done)
        rows.append([doc_id, n_steps, parser_failures, "finished" if done else "in progress"])
    report.table("Books", ["book", "chunk records", "parser failures", "state"], rows)
    report.add("Location of the construction records", f"`{index_path}`")
    report.numbers["cache_index"] = str(index_path)
    report.numbers["books_finished"] = finished
    return finished, rows


def main(report: StepReport, argv: list[str]) -> None:
    dry = "--dry-run" in argv
    tp = _arg_value(argv, "--tensor-parallel-size")
    gpus = _arg_value(argv, "--gpus")

    # The spec is frozen; refuse to index anything else.
    frozen = read_json(SPEC_FROZEN_JSON)
    actual = sha256_file(PERSONAS_JSONL)
    if actual != frozen["personas_jsonl_sha256"]:
        raise RuntimeError(
            f"personas.jsonl hash {actual} differs from the frozen spec "
            f"{frozen['personas_jsonl_sha256']}. The spec must not change after step 4."
        )
    report.add("Frozen spec", f"personas.jsonl SHA-256 verified: `{actual}`")

    # --profile published reads outputs/saadi/reference at that exact path.
    if not (ROOT / "outputs" / "saadi" / "reference" / "manifest.json").exists():
        hint = ""
        if (ROOT / "outputs" / "sadi" / "reference" / "manifest.json").exists():
            hint = (
                " The reference bundle exists under the historical name outputs/sadi/reference - "
                "rename it to outputs/saadi (or pull the renamed folder) before running this step."
            )
        raise RuntimeError("outputs/saadi/reference/manifest.json is missing; --profile published needs it." + hint)

    # Write the declared config file (refuse to silently change an existing one).
    if PROBE_CONFIG.exists():
        if PROBE_CONFIG.read_text(encoding="utf-8") != CONFIG_TEXT:
            raise RuntimeError(f"{PROBE_CONFIG} exists with different content; refusing to overwrite.")
    else:
        PROBE_CONFIG.write_text(CONFIG_TEXT, encoding="utf-8")
    report.files_written.append(str(PROBE_CONFIG) + "  (the one declared file outside the probe folders)")

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_study.py"),
        "--suite",
        "main",
        "--profile",
        "published",
        "--datasets",
        "novelhopqa_probe",
        "--retrievers",
        "bm25",
        "--stage",
        "ingest",
    ]
    if tp:
        cmd += ["--tensor-parallel-size", tp]
    if dry:
        cmd += ["--dry-run"]
    env = dict(os.environ)
    env["SADI_OUTPUT_DIR"] = str(OUT_DIR)
    if gpus:
        env["CUDA_VISIBLE_DEVICES"] = gpus
    report.commands.append(" ".join(["SADI_OUTPUT_DIR=" + str(OUT_DIR)] + (["CUDA_VISIBLE_DEVICES=" + gpus] if gpus else []) + cmd))

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, env=env)
    elapsed = time.time() - t0
    report.add("Wall clock", f"{elapsed / 3600:.2f} hours ({elapsed:.0f} s) this invocation.")

    finished, _rows = scan_cache(report)
    if dry:
        report.skip_marker = True
        report.add("Dry run", "Configuration resolved only; step 6 stays locked.")
        return
    if proc.returncode != 0:
        raise RuntimeError(f"run_study.py exited with code {proc.returncode}. Rerun this script to resume; the construction cache is kept.")
    if finished < common.N_BOOKS:
        report.skip_marker = True
        report.unexpected.append(
            f"Only {finished}/{common.N_BOOKS} books have a final memory.json. "
            "Rerun this script to resume; the done marker is withheld until all books finish."
        )
        return
    report.add("Hand-off", "All books are indexed. Paste this report back before step 6.")


if __name__ == "__main__":
    raise SystemExit(run_step(5, "Index (GPU)", main, requires=[1, 2, 3, 4]))
