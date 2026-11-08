"""Shared helpers of the p1_ledger steps.

The probe's own code (experiments/probe_bridge: common.py, banks.py, the step
modules) is reused by import, never copied. This module adds the run's paths,
markers and report class, and loads the probe modules under their own names
without shadowing (the probe's `common` is loaded as ``probe``).

Writable locations: experiments/method_update/p1_ledger/{reports,logs},
outputs/method_update/p1_ledger/ and results/method_update/p1_ledger/.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent
ROOT = EXP_DIR.parents[2]  # experiments/method_update/p1_ledger -> repository root
PROBE_DIR = Path(os.environ.get("P1_PROBE_DIR", ROOT / "experiments" / "probe_bridge"))
for p in (str(ROOT / "src"), str(PROBE_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)
pydeps = os.environ.get("P1_PYDEPS")
if pydeps and Path(pydeps).is_dir() and pydeps not in sys.path:
    sys.path.insert(0, pydeps)

PROFILE = os.environ.get("P1_PROFILE", "p1_ledger")
OUT_DIR = Path(os.environ.get("P1_OUT_DIR", ROOT / "outputs" / "method_update" / PROFILE))
RESULTS_DIR = Path(os.environ.get("P1_RESULTS_DIR", ROOT / "results" / "method_update" / PROFILE))
MARKER_DIR = Path(os.environ.get("P1_MARKER_DIR", OUT_DIR / "markers"))
REPORT_DIR = Path(os.environ.get("P1_REPORT_DIR", EXP_DIR / "reports"))
LOG_DIR = Path(os.environ.get("P1_LOG_DIR", EXP_DIR / "logs"))
LEDGER_DIR = Path(os.environ.get("P1_LEDGER_DIR", OUT_DIR / "ledger"))
RUN0_OUT_DIR = Path(os.environ.get("P1_RUN0_OUT_DIR", ROOT / "outputs" / "probe_bridge"))
PROFILE_YAML = ROOT / "configs" / "profiles" / f"{PROFILE}.yaml"
GATE_DIR = OUT_DIR / "gate"
SEED = int(os.environ.get("P1_SEED", 42))
N_BOOKS = int(os.environ.get("P1_N_BOOKS", 10))
N_RESAMPLES = int(os.environ.get("P1_BOOTSTRAP_RESAMPLES", 10_000))
# The probe's report helper reads PB_LOG_FILE for the log tail.
if os.environ.get("P1_LOG_FILE") and not os.environ.get("PB_LOG_FILE"):
    os.environ["PB_LOG_FILE"] = os.environ["P1_LOG_FILE"]


def _load_module_from(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# The probe's `common` under a non-clashing name. Its own `import common`
# lines (in the step modules) must still resolve to it, hence the swap below.
probe = _load_module_from(PROBE_DIR / "common.py", "probe_common")
banks = _load_module_from(PROBE_DIR / "banks.py", "probe_banks")


def load_probe_step(filename: str):
    """Import one of experiments/probe_bridge/steps/*.py so its functions can be reused."""

    saved = sys.modules.get("common")
    sys.modules["common"] = probe
    try:
        return _load_module_from(PROBE_DIR / "steps" / filename, "probe_" + Path(filename).stem)
    finally:
        if saved is not None:
            sys.modules["common"] = saved
        else:
            sys.modules.pop("common", None)


# Re-exported probe helpers (same tokenizer, same stopwords, same chunker).
tokens = probe.tokens
content_token_set = probe.content_token_set
chunk_book = probe.chunk_book
read_json = probe.read_json
write_json = probe.write_json
read_jsonl = probe.read_jsonl
write_jsonl = probe.write_jsonl
sha256_file = probe.sha256_file
paired_bootstrap_ci = probe.paired_bootstrap_ci
mcnemar_exact_p = probe.mcnemar_exact_p
PUBLISHED_SETTINGS = probe.PUBLISHED_SETTINGS
PERSONAS_JSONL = probe.PERSONAS_JSONL
SPEC_FROZEN_JSON = probe.SPEC_FROZEN_JSON
BOOKS_JSON = probe.BOOKS_JSON
PROBE_CONFIG = probe.PROBE_CONFIG
CELL_LABELS = {c["cell"]: c["label"] for c in banks.CELLS}
CELL_FAMILY = {c["cell"]: c["family"] for c in banks.CELLS}
REVEAL_CELLS = (1, 2, 3, 4, 6, 7, 8)
GATE_CELLS = (1, 2, 3)


def software_versions() -> dict[str, str]:
    out = probe.software_versions()
    for mod in ("torch", "transformers", "gliner", "vllm"):
        try:
            m = __import__(mod)
            out[mod] = str(getattr(m, "__version__", "installed"))
        except Exception:
            out[mod] = "missing"
    return out


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100.0 * n / d:.0f}%)" if d else "-"


def md_table(headers: list[Any], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(str(h) for h in headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Spec, books, chunks
# ---------------------------------------------------------------------------


def verify_frozen_spec() -> str:
    frozen = read_json(SPEC_FROZEN_JSON)
    actual = sha256_file(PERSONAS_JSONL)
    if actual != frozen["personas_jsonl_sha256"]:
        raise RuntimeError(
            f"personas.jsonl hash {actual} differs from the frozen spec {frozen['personas_jsonl_sha256']}. "
            "The probe spec must not change; refusing to continue."
        )
    return actual


def load_personas() -> list[dict]:
    return read_jsonl(PERSONAS_JSONL)


def book_ids() -> list[str]:
    return [row["doc_id"] for row in read_json(BOOKS_JSON)["selected"]]


def book_chunks(book_id: str) -> list[str]:
    """The chunks of the modified book, exactly as the ingest pipeline cuts them."""

    text = (PROBE_DIR / "data" / "books" / book_id / "modified.txt").read_text(encoding="utf-8")
    return [chunk for chunk, _s, _e in chunk_book(text)]


def ledger_profile_config() -> dict[str, Any]:
    """The `ledger` mapping of configs/profiles/p1_ledger.yaml with ${VARS} expanded."""

    from saadi.config.loader import load_config

    os.environ.setdefault("SADI_OUTPUT_DIR", str(OUT_DIR))
    spec = load_config(PROFILE_YAML)
    return dict(spec["overrides"]["ingest"]["strategy_kwargs"]["ledger"])


# ---------------------------------------------------------------------------
# Construction caches (run 1 here, run 0 under outputs/probe_bridge)
# ---------------------------------------------------------------------------


def find_run1_cache_index() -> Path:
    candidates = sorted(OUT_DIR.glob("novelhopqa_probe/indexes/*/llm_cache/**/index.json"))
    if not candidates:
        raise FileNotFoundError(f"No construction cache index.json under {OUT_DIR}/novelhopqa_probe. Run step 2 first.")
    if len(candidates) > 1:
        raise RuntimeError(f"More than one cache index under {OUT_DIR}: {candidates}. Remove stale runs.")
    return candidates[0]


def find_run0_cache_index() -> Path | None:
    try:
        return probe.find_cache_index()
    except Exception:
        return None


def run0_file(name: str) -> Path | None:
    path = RUN0_OUT_DIR / name
    return path if path.exists() else None


# ---------------------------------------------------------------------------
# Markers and reports
# ---------------------------------------------------------------------------


def marker_path(step: int) -> Path:
    return MARKER_DIR / f"step_{step:02d}.done"


def require_marker(step: int) -> None:
    if not marker_path(step).exists():
        raise RuntimeError(f"Step {step:02d} has not finished: missing {marker_path(step)}. Run its script first.")


def write_marker(step: int, payload: dict | None = None) -> None:
    MARKER_DIR.mkdir(parents=True, exist_ok=True)
    write_json(marker_path(step), {"step": step, "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"), **(payload or {})})


@dataclass
class StepReport:
    """Markdown report plus JSON twin, same four parts as the probe's reports."""

    step: int
    title: str
    commands: list[str] = field(default_factory=list)
    sections: list[tuple[str, str]] = field(default_factory=list)
    numbers: dict[str, Any] = field(default_factory=dict)
    files_written: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    skip_marker: bool = False

    def add(self, heading: str, body_markdown: str) -> None:
        self.sections.append((heading, body_markdown))

    def table(self, heading: str, headers: list[Any], rows: list[list[Any]]) -> None:
        self.add(heading, md_table(headers, rows))

    @property
    def md_path(self) -> Path:
        return REPORT_DIR / f"step_{self.step:02d}.md"

    @property
    def json_path(self) -> Path:
        return REPORT_DIR / f"step_{self.step:02d}.json"

    def write(self, *, status: str) -> Path:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        versions = software_versions()
        lines = [f"# {PROFILE} - step {self.step:02d}: {self.title}", "", f"Status: **{status}** - {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
        lines += ["## What was done", ""]
        for cmd in self.commands:
            lines.append(f"- `{cmd}`")
        lines.append("- Versions: " + ", ".join(f"{k}={v}" for k, v in versions.items()))
        lines.append("")
        lines += ["## Numbers", ""]
        for heading, body in self.sections:
            lines += [f"### {heading}", "", body, ""]
        lines += ["## Anything unexpected", ""]
        if self.unexpected:
            for item in self.unexpected:
                lines += [item, ""]
        else:
            lines += ["Nothing unexpected.", ""]
        lines += ["## Files written", ""]
        lines += [f"- `{path}`" for path in self.files_written] or ["- (none)"]
        self.md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        write_json(
            self.json_path,
            {
                "profile": PROFILE,
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
    log_file = os.environ.get("P1_LOG_FILE") or os.environ.get("PB_LOG_FILE", "")
    if not log_file or not Path(log_file).exists():
        return "(no log file)"
    try:
        return "\n".join(Path(log_file).read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
    except Exception as exc:
        return f"(could not read log: {exc})"


def run_step(step: int, title: str, main: Callable[[StepReport, list[str]], None], *, requires: Iterable[int] = ()) -> int:
    """Markers, the step body, the report, the done marker. A failure still writes the report."""

    argv = sys.argv[1:]
    force = "--force" in argv
    report = StepReport(step=step, title=title)
    report.commands.append(f"bash experiments/method_update/{PROFILE}/sh/{step:02d}_*.sh " + " ".join(argv))
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


def arg_value(argv: list[str], name: str, default: str | None = None) -> str | None:
    if name in argv:
        i = argv.index(name)
        if i + 1 < len(argv):
            return argv[i + 1]
    for item in argv:
        if item.startswith(name + "="):
            return item.split("=", 1)[1]
    return default
