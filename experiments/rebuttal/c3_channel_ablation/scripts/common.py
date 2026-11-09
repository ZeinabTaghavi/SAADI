"""Shared code of C3: channel ablation by re-scoring (ICLR rebuttal, zXpi and iVU8).

Query-time only. The per-chunk channel scores (text, name, alias, description, summary)
come from C2's stream format: BM25 is recomputed on CPU from the construction caches with
C2's own code (``compute_job_streams``), the dense retrievers are read from a stream folder
given as a parameter (``C3_STREAMS_DIR``). Every variant is ranked by the published
Equation 2 (``combine_mbr_variant_chunk_scores``), called with the published weights read
from the config, one channel weight set to zero. Nothing else is changed: no re-tuning, and
no rescaling of the weights that remain (Equation 2's own per-chunk division by the sum of
the weights present stays exactly as published).

Imported, never edited: saadi.rebuttal.frozen_records (E2), saadi.rebuttal.heldout_fusion
(C2), saadi.indexing.retrieval_methods.mbr, saadi.evaluation.retrieval_metrics.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(os.environ.get("C3_ROOT_DIR") or Path(__file__).resolve().parents[4])
EXP_DIR = Path(os.environ.get("C3_EXP_DIR") or REPO_ROOT / "experiments" / "rebuttal" / "c3_channel_ablation")
RESULTS_DIR = Path(os.environ.get("C3_RESULTS_DIR") or REPO_ROOT / "results" / "rebuttal" / "c3_channel_ablation")
REPORT_DIR = Path(os.environ.get("C3_REPORT_DIR") or EXP_DIR / "reports")
WORK_DIR = Path(os.environ.get("C3_WORK_DIR") or EXP_DIR / "work")
STREAMS_DIR = Path(os.environ.get("C3_STREAMS_DIR") or REPO_ROOT / "outputs" / "rebuttal" / "C2_heldout_fusion" / "streams")
STREAMS_MARKERS_DIR = Path(os.environ.get("C3_STREAMS_MARKERS_DIR") or STREAMS_DIR.parent / "markers")
CONFIG_PATH = Path(os.environ.get("C3_CONFIG") or REPO_ROOT / "configs" / "methods" / "sadi.yaml")
SEED = int(os.environ.get("C3_SEED", "42"))
N_BOOT = int(os.environ.get("C3_BOOTSTRAP_RESAMPLES", "10000"))

if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from saadi.config.loader import load_config  # noqa: E402
from saadi.indexing.retrieval_methods.mbr import MBR_VARIANT_MBR2, combine_mbr_variant_chunk_scores  # noqa: E402
from saadi.rebuttal.frozen_records import (  # noqa: E402
    _METHOD_FILE_LABEL,
    DATASETS,
    PUBLISHED_SETTINGS,
    RETRIEVE_K,
    load_saved_live_rows,
)
from saadi.rebuttal.heldout_fusion import (  # noqa: E402
    BIG_RANK,
    DATASET_DISPLAY,
    RETRIEVER_DISPLAY,
    VANILLA_DATASET_NAME,
    C2Paths,
    QueryArrays,
    _hr,
    _rebuild_for_repo,
    compute_job_streams,
    expected_main_metrics,
    hit_rank_of_ranking,
    job_name,
    load_query_arrays,
    per_query_vanilla_file,
    published_vanilla,
    reproduction_check,
    save_streams,
)

ALL_DATASETS = ("loogle", "musique_32k", "qasper_64k", "novelhopqa")
ALL_RETRIEVERS = ("bm25", "contriever", "bge_m3", "jina", "qwen")
EXPECTED_TOTAL_QUERIES = 4973  # as in E2: 1832 + 900 + 1333 + 908


def selected(env_name: str, allowed: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(env_name, ",".join(allowed))
    chosen = tuple(x.strip() for x in raw.split(",") if x.strip())
    unknown = [x for x in chosen if x not in allowed]
    if unknown:
        raise ValueError(f"{env_name}: unknown values {unknown}; choose from {list(allowed)}")
    return tuple(x for x in allowed if x in chosen)


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(p)


# ---------------------------------------------------------------------------
# Published weights (read from the config) and the variants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Weights:
    lambda_t: float
    lambda_n: float
    lambda_h: float
    normalization_method: str
    mu_n: float
    mu_a: float
    mu_d: float
    memory_aggregation: str
    method: str

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def load_published_weights(config_path: Path = CONFIG_PATH) -> tuple[Weights, list[str]]:
    """The cached_retrieval block of the config, and every way it differs from what the
    saved paper rankings and C2's streams were computed with (an empty list = consistent)."""

    cfg = load_config(config_path).get("cached_retrieval", {}) or {}
    w = Weights(
        lambda_t=float(cfg["lambda_t"]),
        lambda_n=float(cfg["lambda_n"]),
        lambda_h=float(cfg["lambda_h"]),
        normalization_method=str(cfg["normalization_method"]),
        mu_n=float(cfg["memory_name_weight"]),
        mu_a=float(cfg["memory_alias_weight"]),
        mu_d=float(cfg["memory_description_weight"]),
        memory_aggregation=str(cfg["memory_aggregation"]),
        method=str(cfg.get("method", "")),
    )
    problems: list[str] = []
    if w.method != "mbr":
        problems.append(f"cached_retrieval.method is {w.method!r}, expected 'mbr'")
    pairs = {
        "lambda_t": w.lambda_t,
        "lambda_n": w.lambda_n,
        "lambda_h": w.lambda_h,
        "normalization_method": w.normalization_method,
        "memory_name_weight": w.mu_n,
        "memory_alias_weight": w.mu_a,
        "memory_description_weight": w.mu_d,
        "memory_aggregation": w.memory_aggregation,
    }
    for key, value in pairs.items():
        if PUBLISHED_SETTINGS[key] != value:
            problems.append(f"config {key}={value!r} but the saved rankings / C2 streams use {PUBLISHED_SETTINGS[key]!r}")
    label = (
        f"mbr_lt{w.lambda_t:g}_ln{w.lambda_n:g}_lh{w.lambda_h:g}_norm_{w.normalization_method}"
        f"_mnw{w.mu_n:g}_maw{w.mu_a:g}_mdw{w.mu_d:g}"
    )
    if label != _METHOD_FILE_LABEL:
        problems.append(f"config label {label!r} differs from the saved rankings' label {_METHOD_FILE_LABEL!r}")
    return w, problems


VARIANTS = ("text_only", "text_summary", "text_memory", "all_three", "summary_only", "memory_only")
MAIN_VARIANTS = VARIANTS[:4]
INFO_VARIANTS = VARIANTS[4:]
# Not re-scored: all three = the paper's saved rankings; text only = the text-only top-10
# saved in the streams (C2's code, lambda_M = lambda_S = 0). Only these are scored by C3:
NEW_VARIANTS = ("text_summary", "text_memory", "summary_only", "memory_only")
VARIANT_SOURCE = {
    "text_only": "saved text-only top-10 in the streams (C2)",
    "all_three": "paper's saved rankings",
    **{v: "scored by C3 (published Equation 2, one weight 0)" for v in NEW_VARIANTS},
}
VARIANT_LABEL = {
    "text_only": "Text only",
    "text_summary": "Text + summary",
    "text_memory": "Text + memory",
    "all_three": "All three (SAADI)",
    "summary_only": "Summary alone",
    "memory_only": "Memory alone",
}
# Which of (text, memory, summary) each variant keeps; a dropped channel gets weight 0.
_KEEP = {
    "text_only": (1, 0, 0),
    "text_summary": (1, 0, 1),
    "text_memory": (1, 1, 0),
    "all_three": (1, 1, 1),
    "summary_only": (0, 0, 1),
    "memory_only": (0, 1, 0),
}


def variant_lambdas(w: Weights, variant: str) -> tuple[float, float, float]:
    kt, km, ks = _KEEP[variant]
    return (w.lambda_t if kt else 0.0, w.lambda_n if km else 0.0, w.lambda_h if ks else 0.0)


def combine(index, streams, w: Weights, variant: str):
    """The published Equation 2 (repo function) with the variant's weights."""

    lt, ln, lh = variant_lambdas(w, variant)
    return combine_mbr_variant_chunk_scores(
        index,
        streams,
        variant=MBR_VARIANT_MBR2,
        lambda_t=lt,
        lambda_n=ln,
        lambda_h=lh,
        normalization_method=w.normalization_method,
        memory_name_weight=w.mu_n,
        memory_alias_weight=w.mu_a,
        memory_description_weight=w.mu_d,
    )


# ---------------------------------------------------------------------------
# Channel-score inputs (C2 stream format)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StreamFiles:
    parquet: Path
    queries: Path
    check: Path

    @property
    def present(self) -> bool:
        return self.parquet.exists() and self.queries.exists()

    def missing(self) -> list[str]:
        return [rel(p) for p in (self.parquet, self.queries) if not p.exists()]


def stream_files(folder: Path, dataset: str, retriever: str) -> StreamFiles:
    job = job_name(dataset, retriever)
    return StreamFiles(folder / f"{job}.parquet", folder / f"{job}.queries.json", folder / f"{job}.check.json")


BM25_STREAMS_DIR = WORK_DIR / "streams"


def input_streams(dataset: str, retriever: str) -> StreamFiles:
    """BM25: C3's own CPU recomputation; dense: the C3_STREAMS_DIR parameter."""

    return stream_files(BM25_STREAMS_DIR if retriever == "bm25" else STREAMS_DIR, dataset, retriever)


def compute_bm25_streams(dataset: str, *, force: bool = False, log: Callable[[str], None] = print) -> dict[str, Any]:
    """BM25 channel scores from the published construction caches (C2's code, CPU)."""

    files = stream_files(BM25_STREAMS_DIR, dataset, "bm25")
    if files.present and not force:
        return {"status": "reused", "files": [rel(files.parquet), rel(files.queries)]}
    t0 = time.perf_counter()
    paths = C2Paths(REPO_ROOT, out_base=WORK_DIR / "bm25_compute")
    docs, used_rows, timings = compute_job_streams(dataset, "bm25", paths=paths, log=log)
    n_rows = save_streams(dataset, "bm25", docs, used_rows, out_path=files.parquet, queries_path=files.queries)
    # C2's own gate on the same docs (saved SAADI scores, mismatch examples); C3 reads it, never re-scores.
    write_json(files.check, reproduction_check(dataset, "bm25", docs, used_rows, repo_root=REPO_ROOT, full_dataset=True))
    return {
        "status": "computed",
        "files": [rel(files.parquet), rel(files.queries), rel(files.check)],
        "stream_rows": n_rows,
        "n_queries": len(used_rows),
        "timings": timings,
        "seconds": time.perf_counter() - t0,
    }


def load_meta(files: StreamFiles) -> list[dict[str, Any]]:
    return json.loads(files.queries.read_text(encoding="utf-8"))["queries"]


def load_pair(dataset: str, retriever: str, files: StreamFiles) -> tuple[list[QueryArrays], list[dict[str, Any]], list[dict[str, Any]]]:
    """Per-query channel arrays, stream metadata, and the paper's saved row of each query."""

    queries = load_query_arrays(files.parquet, files.queries, retriever)
    meta = load_meta(files)
    return queries, meta, saved_rows_for(dataset, retriever, meta)


def saved_rows_for(dataset: str, retriever: str, meta: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The paper's saved row of each stream query (same query and document checked)."""

    saved = load_saved_live_rows(DATASETS[dataset], retriever, repo_root=REPO_ROOT)
    rows = []
    for m in meta:
        row = saved[int(m["row_index"])]
        if str(row.get("query_id", "")) != str(m["query_id"]) or str(row.get("doc_id", "")) != str(m["doc_id"]):
            raise RuntimeError(
                f"{job_name(dataset, retriever)}: stream query {m['qidx']} (row {m['row_index']}) does not match the saved row "
                f"({m['query_id']}/{m['doc_id']} vs {row.get('query_id')}/{row.get('doc_id')})"
            )
        rows.append(row)
    return rows


def n_saved_rows(dataset: str, retriever: str) -> int:
    return len(load_saved_live_rows(DATASETS[dataset], retriever, repo_root=REPO_ROOT))


def sha256_file(path: Path, *, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact binomial McNemar p-value from the discordant counts."""

    n = int(b) + int(c)
    if n == 0:
        return 1.0
    k = min(int(b), int(c))
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def hit_rank(q: QueryArrays, ranking: list[str]) -> int:
    """0-based rank at which the saved hit definition is first met (BIG_RANK: never)."""

    return int(min(hit_rank_of_ranking(q, ranking), BIG_RANK))


NOT_IN_TOP = RETRIEVE_K + 1  # stored rank when the hit is not met within the top-10


def top10_rank(gold: list[str], groups: list[list[str]], top: list[str]) -> int:
    """1-based rank within a top-10 at which the saved hit definition is met (a gold chunk, or
    every chunk of one evidence group); NOT_IN_TOP when it is not met within the top-10.
    The same definition as evaluate_retrieval_results, so rank <= k <=> Hit@k."""

    pos = {c: i + 1 for i, c in enumerate(top[:RETRIEVE_K])}
    best = NOT_IN_TOP
    for c in gold:
        if c in pos:
            best = min(best, pos[c])
    for g in groups:
        if g and all(c in pos for c in g):
            best = min(best, max(pos[c] for c in g))
    return best


# ---------------------------------------------------------------------------
# Step reports
# ---------------------------------------------------------------------------


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False).stdout.strip()
        except Exception:
            return ""

    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"), "dirty_files": len(run("status", "--porcelain").splitlines())}


def fmt(x: Any, nd: int = 2) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        if math.isnan(x):
            return "-"
        return f"{x:.{nd}f}"
    return str(x)


def table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    out += ["| " + " | ".join(fmt(c) for c in r) + " |" for r in rows]
    return out


def _json_default(obj: Any) -> Any:
    try:
        import numpy as np

        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    if isinstance(obj, Path):
        return rel(obj)
    raise TypeError(f"not JSON serializable: {type(obj)}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    os.replace(tmp, path)


class StepReport:
    """One self-contained Markdown report plus a JSON twin; written even when the step fails."""

    def __init__(self, step: int, title: str, command: str) -> None:
        self.step = step
        self.title = title
        self.command = command
        self.lines: list[str] = []
        self.data: dict[str, Any] = {}
        self.files: list[str] = []
        self.status = "RUNNING"
        self.t0 = time.time()

    @property
    def md_path(self) -> Path:
        return REPORT_DIR / f"step_{self.step:02d}.md"

    @property
    def json_path(self) -> Path:
        return REPORT_DIR / f"step_{self.step:02d}.json"

    def add(self, *lines: str) -> None:
        self.lines.extend(lines)

    def wrote(self, path: Path | str) -> None:
        r = rel(path)
        if r not in self.files:
            self.files.append(r)

    def write(self) -> None:
        head = [
            f"# C3 channel ablation - step {self.step:02d}: {self.title}",
            "",
            f"**Status: {self.status}**",
            "",
            f"- Command: `{self.command}`",
            f"- Started: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.t0))}; took {time.time() - self.t0:.0f} s",
            f"- Host: {platform.node()}; Python {platform.python_version()}; commit {git_state()['commit'][:12]} "
            f"({git_state()['branch']}, {git_state()['dirty_files']} changed files)",
            "",
        ]
        self.wrote(self.md_path)
        self.wrote(self.json_path)
        tail = ["", "## Files written", ""] + [f"- `{f}`" for f in self.files]
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        self.md_path.write_text("\n".join(head + self.lines + tail) + "\n", encoding="utf-8")
        write_json(
            self.json_path,
            {"step": self.step, "title": self.title, "status": self.status, "command": self.command, "git": git_state(), **self.data, "files": self.files},
        )


def run_step(report: StepReport, body: Callable[[StepReport], int]) -> int:
    """Run the step body; any exception still produces the report (with the traceback) and exit 1."""

    try:
        code = int(body(report))
        report.status = "PASSED" if code == 0 else "FAILED"
    except Exception:
        code = 1
        report.status = "FAILED (exception)"
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        report.add("", "## Exception", "", "```", tb.rstrip(), "```")
        report.data["traceback"] = tb
    report.write()
    return code


__all__ = [
    "ALL_DATASETS",
    "ALL_RETRIEVERS",
    "BIG_RANK",
    "DATASETS",
    "DATASET_DISPLAY",
    "EXPECTED_TOTAL_QUERIES",
    "INFO_VARIANTS",
    "MAIN_VARIANTS",
    "N_BOOT",
    "REPO_ROOT",
    "RETRIEVER_DISPLAY",
    "RETRIEVE_K",
    "SEED",
    "STREAMS_DIR",
    "STREAMS_MARKERS_DIR",
    "VANILLA_DATASET_NAME",
    "VARIANTS",
    "VARIANT_LABEL",
    "WORK_DIR",
    "StepReport",
    "Weights",
    "_hr",
    "_rebuild_for_repo",
    "combine",
    "compute_bm25_streams",
    "expected_main_metrics",
    "hit_rank",
    "input_streams",
    "job_name",
    "load_pair",
    "load_published_weights",
    "mcnemar_exact_p",
    "n_saved_rows",
    "per_query_vanilla_file",
    "published_vanilla",
    "rel",
    "run_step",
    "selected",
    "sha256_file",
    "stream_files",
    "table",
    "variant_lambdas",
    "write_json",
]
