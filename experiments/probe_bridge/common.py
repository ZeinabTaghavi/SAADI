"""Shared helpers for the bridge probe.

Writable locations: experiments/probe_bridge/ (spec, data, reports, logs),
outputs/probe_bridge/ and results/probe_bridge/. Nothing else is written,
except the single declared config file configs/experiments/
novelhopqa_probe_retrieval_ablation.yaml (created by step 5 and reported).

The probe reuses the repository's own code: the chunker
(src/saadi/core/chunking.py, which the ingest pipeline calls with
tokenizer=None, i.e. whitespace tokens packed at sentence boundaries), the
BM25 tokenizer, the HR evaluation, the MBR query-time scoring, and the E2
frozen-records machinery (src/saadi/rebuttal/frozen_records.py).

Note on stopwords: the note says "the repository's BM25 tokenizer and its
stopword list", but the repository's BM25 has no stopword list. The probe uses
the fixed list below - the same list as scripts/rebuttal/e2b_bridge_queries.py
- applied after the tokenizer's own normalization.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PB_DIR = Path(__file__).resolve().parent
ROOT = PB_DIR.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from saadi.core.chunking import chunk_text  # noqa: E402
from saadi.indexing.retriever_adapters.bm25 import _tokenize  # noqa: E402

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
SPEC_DIR = PB_DIR / "spec"
DATA_DIR = PB_DIR / "data"
REPORT_DIR = PB_DIR / "reports"
LOG_DIR = PB_DIR / "logs"
OUT_DIR = ROOT / "outputs" / "probe_bridge"
RESULTS_DIR = ROOT / "results" / "probe_bridge"
MARKER_DIR = OUT_DIR / "markers"

BOOKS_JSON = SPEC_DIR / "books.json"
TEMPLATES_JSON = SPEC_DIR / "templates.json"
PERSONAS_JSONL = SPEC_DIR / "personas.jsonl"
SPEC_FROZEN_JSON = SPEC_DIR / "spec_frozen.json"
INSERTIONS_JSON = DATA_DIR / "insertions.json"
DOCUMENTS_JSON = DATA_DIR / "documents.json"
QA_JSON = DATA_DIR / "qa.json"
PROBE_CONFIG = ROOT / "configs" / "experiments" / "novelhopqa_probe_retrieval_ablation.yaml"

SEED = 42
CHUNK_SIZE = 500
N_BOOKS = 10
PERSONAS_PER_BOOK = 24
PERSONAS_PER_CELL_PER_BOOK = 3
N_CELLS = 8

# Published query-time settings (configs/methods/sadi.yaml); never tuned here.
PUBLISHED_SETTINGS = {
    "lambda_t": 1.0,
    "lambda_n": 0.5,
    "lambda_h": 0.3,
    "normalization_method": "zscore",
    "memory_name_weight": 1.0,
    "memory_alias_weight": 1.0,
    "memory_description_weight": 1.0,
    "memory_aggregation": "max",
}

# ---------------------------------------------------------------------------
# Tokenization (repo BM25 tokenizer + fixed stopword list, as in E2b)
# ---------------------------------------------------------------------------
_STOPWORDS_RAW = """
a about above after again against all am an and any are aren as at be because
been before being below between both but by can cannot could couldn did didn
do does doesn doing don down during each few for from further had hadn has hasn
have haven having he her here hers herself him himself his how i if in into is
isn it its itself just me more most mustn my myself no nor not now of off on
once only or other our ours ourselves out over own same shan she should shouldn
so some such than that the their theirs them themselves then there these they
this those through to too under until up very was wasn we were weren what when
where which while who whom why will with won would wouldn you your yours
yourself yourselves
""".split()
STOPWORDS = {w for raw in _STOPWORDS_RAW for w in _tokenize(raw)} | set(_STOPWORDS_RAW)
# Tokenization artifacts of contractions and possessives ("Mitra's" -> "mitra", "s").
STOPWORDS |= {"s", "t", "d", "ll", "m", "re", "ve"}


def tokens(text: str) -> list[str]:
    """Repository BM25 tokens (lowercase, light stemming)."""
    return _tokenize(str(text or ""))


def content_token_set(text: str) -> set[str]:
    """BM25 tokens with stopwords removed."""
    return {t for t in tokens(text) if t not in STOPWORDS}


# ---------------------------------------------------------------------------
# Chunking (exactly the ingest pipeline's call: tokenizer=None)
# ---------------------------------------------------------------------------


def chunk_book(text: str) -> list[tuple[str, int, int]]:
    """(chunk_text, token_start, token_end) with whitespace tokens, size 500."""
    return chunk_text(text, tokenizer=None, chunk_size=CHUNK_SIZE, chunk_overlap=0)


def whitespace_token_offsets(text: str) -> list[int]:
    """Character offset of each whitespace token (matches str.split order)."""
    return [m.start() for m in re.finditer(r"\S+", text)]


# ---------------------------------------------------------------------------
# Small IO helpers
# ---------------------------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def read_jsonl(path: Path) -> list[dict]:
    out = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def software_versions() -> dict[str, str]:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "pandas", "yaml", "saadi"):
        try:
            m = __import__(mod)
            out[mod] = str(getattr(m, "__version__", "installed"))
        except Exception:
            out[mod] = "missing"
    try:
        out["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        out["git_commit"] = "unknown"
    return out


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------


def marker_path(step: int) -> Path:
    return MARKER_DIR / f"step_{step:02d}.done"


def require_marker(step: int) -> None:
    if not marker_path(step).exists():
        raise RuntimeError(f"Step {step:02d} has not finished: missing {marker_path(step)}. Run its script first.")


def write_marker(step: int, payload: dict | None = None) -> None:
    MARKER_DIR.mkdir(parents=True, exist_ok=True)
    write_json(marker_path(step), {"step": step, "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"), **(payload or {})})


# ---------------------------------------------------------------------------
# Step report (Markdown + JSON twin). A failing step still writes its report.
# ---------------------------------------------------------------------------


@dataclass
class StepReport:
    step: int
    title: str
    commands: list[str] = field(default_factory=list)
    sections: list[tuple[str, str]] = field(default_factory=list)
    numbers: dict[str, Any] = field(default_factory=dict)
    files_written: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    skip_marker: bool = False  # e.g. a --dry-run that must not unlock the next step

    def add(self, heading: str, body_markdown: str) -> None:
        self.sections.append((heading, body_markdown))

    def table(self, heading: str, headers: list[str], rows: list[list[Any]]) -> None:
        lines = ["| " + " | ".join(str(h) for h in headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
        for row in rows:
            lines.append("| " + " | ".join(str(v) for v in row) + " |")
        self.add(heading, "\n".join(lines))

    @property
    def md_path(self) -> Path:
        return REPORT_DIR / f"step_{self.step:02d}.md"

    @property
    def json_path(self) -> Path:
        return REPORT_DIR / f"step_{self.step:02d}.json"

    def write(self, *, status: str) -> Path:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# Bridge probe - step {self.step:02d}: {self.title}",
            "",
            f"Status: **{status}** - {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]
        lines += ["## What was done", ""]
        for cmd in self.commands:
            lines.append(f"- `{cmd}`")
        versions = software_versions()
        lines.append("- Versions: " + ", ".join(f"{k}={v}" for k, v in versions.items()))
        lines.append("")
        lines += ["## Numbers", ""]
        for heading, body in self.sections:
            lines += [f"### {heading}", "", body, ""]
        lines += ["## Anything unexpected", ""]
        if self.unexpected:
            for item in self.unexpected:
                lines.append(item)
                lines.append("")
        else:
            lines.append("Nothing unexpected.")
            lines.append("")
        lines += ["## Files written", ""]
        for path in self.files_written:
            lines.append(f"- `{path}`")
        if not self.files_written:
            lines.append("- (none)")
        self.md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        write_json(
            self.json_path,
            {
                "step": self.step,
                "title": self.title,
                "status": status,
                "commands": self.commands,
                "numbers": self.numbers,
                "files_written": self.files_written,
                "unexpected": self.unexpected,
                "versions": versions,
            },
        )
        return self.md_path


def _tail_of_log(n: int = 50) -> str:
    log_file = os.environ.get("PB_LOG_FILE", "")
    if not log_file or not Path(log_file).exists():
        return "(no log file)"
    try:
        lines = Path(log_file).read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception as exc:
        return f"(could not read log: {exc})"


# ---------------------------------------------------------------------------
# Scoring helpers shared by steps 6 and 7
# ---------------------------------------------------------------------------


def find_cache_index() -> Path:
    """The construction cache index.json written by step 5 under outputs/probe_bridge/."""
    candidates = sorted(OUT_DIR.glob("novelhopqa_probe/indexes/*/llm_cache/*/*/*/*/*/*/*/index.json"))
    if not candidates:
        candidates = sorted(p for p in OUT_DIR.rglob("index.json") if p.parent.name.endswith("Word"))
    if not candidates:
        raise FileNotFoundError(f"No construction cache index.json under {OUT_DIR}. Run step 5 first.")
    if len(candidates) > 1:
        raise RuntimeError(f"More than one cache index under {OUT_DIR}: {candidates}. Remove stale runs.")
    return candidates[0]


def memory_channel_ranks(streams, document_index) -> dict[str, int]:
    """1-based rank of every chunk on the memory channel alone (MBR2, z-scored).

    Adapted from scripts/rebuttal/e2b_bridge_queries.py (E2b); the original is
    left untouched.
    """
    from saadi.indexing.retrieval_methods.mbr import normalize_signal_scores

    recs = streams.chunk_scores
    norm = {
        kind: normalize_signal_scores(
            {cid: getattr(r, f"{kind}_score") for cid, r in recs.items() if getattr(r, f"{kind}_score") is not None},
            method=PUBLISHED_SETTINGS["normalization_method"],
            retriever_name="bm25",
        )
        for kind in ("name", "alias", "description")
    }
    memory: dict[str, float] = {}
    for cid in recs:
        parts = [norm[kind][cid] for kind in ("name", "alias", "description") if cid in norm[kind]]
        if parts:
            memory[cid] = sum(parts) / len(parts)
    order = sorted(
        document_index.chunk_pool,
        key=lambda ch: (-(memory.get(ch.chunk_id, float("-inf"))), int(ch.step or 0), str(ch.chunk_id)),
    )
    return {ch.chunk_id: i + 1 for i, ch in enumerate(order)}


def paired_bootstrap_ci(diffs, *, n_resamples: int = 10_000, seed: int = SEED):
    """Mean with a 95% bootstrap CI over personas (same as E2)."""
    import numpy as np

    arr = np.asarray(list(diffs), dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(int(n_resamples), arr.size))
    means = arr[idx].mean(axis=1)
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact binomial McNemar p-value from the discordant counts."""
    import math

    n = int(b) + int(c)
    if n == 0:
        return 1.0
    k = min(int(b), int(c))
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def run_step(step: int, title: str, main: Callable[[StepReport, list[str]], None], *, requires: list[int] = ()) -> int:
    """Run one step: markers, the step body, the report, the done marker.

    A failure still writes the report (with the traceback and the last 50 log
    lines) and exits non-zero. No half-written marker is left behind.
    """

    argv = sys.argv[1:]
    force = "--force" in argv
    report = StepReport(step=step, title=title)
    report.commands.append("bash experiments/probe_bridge/sh/" + f"{step:02d}_*.sh" + (" --force" if force else ""))
    try:
        if marker_path(step).exists() and not force:
            print(f"Step {step:02d} already finished ({marker_path(step)}). Use --force to rerun.")
            if not report.md_path.exists():
                prev = report.md_path.with_name(f"step_{step:02d}.prev.md")
                if prev.exists():
                    report.md_path.write_text(prev.read_text(encoding="utf-8"), encoding="utf-8")
                else:
                    report.add("Already finished", "This step finished in an earlier run; nothing was redone.")
                    report.write(status="ok (finished earlier)")
            print(report.md_path)
            return 0
        for prev in requires:
            require_marker(prev)
        main(report, argv)
        path = report.write(status="ok")
        if not report.skip_marker:
            write_marker(step, {"report": str(path)})
        print(path)
        return 0
    except Exception:
        tb = traceback.format_exc()
        report.unexpected.append("The step failed. Full traceback:\n\n```\n" + tb + "\n```")
        report.unexpected.append("Last 50 log lines:\n\n```\n" + _tail_of_log(50) + "\n```")
        path = report.write(status="FAILED")
        print(path)
        return 1
