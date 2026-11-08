"""Shared helpers of the p1b_ledger stages.

Run 1's measurement code (experiments/method_update/p1_ledger/scripts) is imported,
never copied. Its module ``common`` reads every path from P1_* environment
variables at import time, so this module sets them to run 1b's folders *before*
importing it, then checks that nothing points into run 1's or run 0's folders.

Writable: experiments/method_update/p1b_ledger/{reports,logs},
outputs/method_update/p1b_ledger/, results/method_update/p1b_ledger/.
Read only: run 1 (experiments|outputs/method_update/p1_ledger) and run 0
(outputs/probe_bridge).
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent
ROOT = EXP_DIR.parents[2]
PROFILE = "p1b_ledger"
P1_SCRIPTS = ROOT / "experiments" / "method_update" / "p1_ledger" / "scripts"


def _set(name: str, default: Path) -> Path:
    value = Path(os.environ.get(name.replace("P1_", "P1B_OVERRIDE_"), default))
    os.environ[name] = str(value)
    return value


# Run 1b's own paths (P1B_OVERRIDE_* allow a local dry run to write elsewhere).
os.environ["P1_PROFILE"] = PROFILE
OUT_DIR = _set("P1_OUT_DIR", ROOT / "outputs" / "method_update" / PROFILE)
RESULTS_DIR = _set("P1_RESULTS_DIR", ROOT / "results" / "method_update" / PROFILE)
MARKER_DIR = _set("P1_MARKER_DIR", OUT_DIR / "markers")
REPORT_DIR = _set("P1_REPORT_DIR", EXP_DIR / "reports")
LOG_DIR = _set("P1_LOG_DIR", EXP_DIR / "logs")
LEDGER_DIR = _set("P1_LEDGER_DIR", OUT_DIR / "ledger")
os.environ["SADI_OUTPUT_DIR"] = str(OUT_DIR)
os.environ.setdefault("P1_RUN0_OUT_DIR", str(ROOT / "outputs" / "probe_bridge"))

# Run 1, read only.
RUN1_OUT_DIR = Path(os.environ.get("P1B_RUN1_OUT_DIR", ROOT / "outputs" / "method_update" / "p1_ledger"))
RUN1_REPORT_DIR = Path(os.environ.get("P1B_RUN1_REPORT_DIR", ROOT / "experiments" / "method_update" / "p1_ledger" / "reports"))
RUN1_LEDGER_DIR = RUN1_OUT_DIR / "ledger"

_READ_ONLY = [ROOT / "experiments" / "method_update" / "p1_ledger", RUN1_OUT_DIR, ROOT / "results" / "method_update" / "p1_ledger", Path(os.environ["P1_RUN0_OUT_DIR"])]
for _path in (OUT_DIR, RESULTS_DIR, MARKER_DIR, REPORT_DIR, LOG_DIR, LEDGER_DIR):
    for _ro in _READ_ONLY:
        if _path.resolve() == _ro.resolve() or _ro.resolve() in _path.resolve().parents:
            raise RuntimeError(f"Run 1b would write into a read-only folder: {_path} (under {_ro}). Refusing.")

if str(P1_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(P1_SCRIPTS))
import common as p1  # noqa: E402  (run 1's common, now pointing at run 1b's folders)
import report_utils as RU  # noqa: E402

assert p1.PROFILE == PROFILE and p1.OUT_DIR == OUT_DIR and p1.LEDGER_DIR == LEDGER_DIR, "run 1's common did not pick up run 1b's paths"

StepReport = p1.StepReport
CELL_LABELS = p1.CELL_LABELS
REVEAL_CELLS = p1.REVEAL_CELLS
GATE_CELLS = p1.GATE_CELLS
SEED = p1.SEED
N_BOOKS = p1.N_BOOKS
pct = p1.pct
md_table = p1.md_table
read_json = p1.read_json
write_json = p1.write_json
read_jsonl = p1.read_jsonl
write_jsonl = p1.write_jsonl
tokens = p1.tokens
content_token_set = p1.content_token_set
probe = p1.probe
load_probe_step = p1.load_probe_step
verify_frozen_spec = p1.verify_frozen_spec
load_personas = p1.load_personas
arg_value = p1.arg_value
marker_path = p1.marker_path
write_marker = p1.write_marker
require_marker = p1.require_marker

STAGES = {1: "01_gate", 2: "02_build", 3: "03_memory_check", 4: "04_retrieve", 5: "05_report"}


def load_p1_step(module_name: str):
    """Import one of run 1's step modules (they import run 1's `common`, already loaded with run 1b's paths)."""

    import importlib

    return importlib.import_module(module_name)


class DetailReport(p1.StepReport):
    """Run 1's report class writing to reports/step_NN_run1_code.md: the tables run 1's own step code
    produces on run 1b's build (there, 'run 1' means run 1b)."""

    @property
    def md_path(self) -> Path:
        return REPORT_DIR / f"step_{self.step:02d}_run1_code.md"

    @property
    def json_path(self) -> Path:
        return REPORT_DIR / f"step_{self.step:02d}_run1_code.json"


def write_detail(detail: DetailReport) -> Path:
    detail.sections.insert(
        0,
        (
            "How to read this file",
            "Produced by run 1's own step code (experiments/method_update/p1_ledger/scripts, imported unchanged) on run 1b's "
            "build. In its tables **'run 1' means run 1b** and 'run 0' is run 0. The run 1b report with three columns is "
            f"step_{detail.step:02d}.md.",
        ),
    )
    return detail.write(status="ok")


# ---------------------------------------------------------------------------
# Run 1 and run 0 references
# ---------------------------------------------------------------------------


def run1_report_numbers(step: int) -> dict[str, Any]:
    path = RUN1_REPORT_DIR / f"step_{step:02d}.json"
    return read_json(path).get("numbers", {}) if path.exists() else {}


def run1_file(name: str) -> Path | None:
    path = RUN1_OUT_DIR / name
    return path if path.exists() else None


def find_cache_index(out_dir: Path) -> Path | None:
    candidates = sorted(out_dir.glob("novelhopqa_probe/indexes/*/llm_cache/**/index.json"))
    if len(candidates) > 1:
        raise RuntimeError(f"More than one cache index under {out_dir}: {candidates}")
    return candidates[0] if candidates else None


def run1_cache_index() -> Path | None:
    return find_cache_index(RUN1_OUT_DIR)


def run0_cache_index() -> Path | None:
    return p1.find_run0_cache_index()


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def three(a: Any, b: Any, c: Any) -> str:
    """'run 1b / run 1 / run 0' cell text."""

    return " / ".join("n/a" if v is None else str(v) for v in (a, b, c))


def rate(group: list[dict], key: str, *, getter: Callable[[dict], Any] | None = None) -> tuple[int, int] | None:
    vals = [(getter(p) if getter else p.get(key)) for p in group]
    known = [v for v in vals if v is not None]
    return (sum(bool(v) for v in known), len(known)) if known else None


def rate_text(r: tuple[int, int] | None) -> str | None:
    if r is None:
        return None
    n, d = r
    return f"{n}/{d} ({100.0 * n / d:.0f}%)" if d else "-"


def mean_or_nan(values: Iterable[float]) -> float:
    vals = [v for v in values if v is not None]
    return statistics.mean(vals) if vals else float("nan")


# ---------------------------------------------------------------------------
# The stage runner
# ---------------------------------------------------------------------------


def _tail_of_log(n: int = 50) -> str:
    log_file = os.environ.get("P1_LOG_FILE", "")
    if not log_file or not Path(log_file).exists():
        return "(no log file)"
    try:
        return "\n".join(Path(log_file).read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
    except Exception as exc:
        return f"(could not read log: {exc})"


def run_stage(step: int, title: str, main: Callable[[p1.StepReport, list[str]], None], *, requires: Iterable[int] = ()) -> int:
    """Markers, the stage body, the report, the done marker. A failure still writes its report (traceback + log tail)."""

    argv = sys.argv[1:]
    force = "--force" in argv
    report = p1.StepReport(step=step, title=title)
    report.commands.append(f"bash experiments/method_update/{PROFILE}/run_all.sh  (stage {STAGES[step]}; python {Path(sys.argv[0]).name} {' '.join(argv)})")
    report.commands.append(f"git commit {git_commit()}")
    t0 = time.time()
    try:
        if marker_path(step).exists() and not force:
            print(f"Stage {STAGES[step]} already finished ({marker_path(step)}); skipped. Use --force to rerun.", flush=True)
            if not report.md_path.exists():
                report.add("Already finished", "This stage finished in an earlier run; nothing was redone.")
                report.write(status="ok (finished earlier)")
            print(report.md_path)
            return 0
        for prev in requires:
            require_marker(prev)
        main(report, argv)
        report.numbers["stage_seconds"] = round(time.time() - t0)
        path = report.write(status="ok")
        if not report.skip_marker:
            write_marker(step, {"report": str(path), "seconds": round(time.time() - t0)})
        print(path)
        return 0
    except Exception:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr, flush=True)
        report.unexpected.append("The stage failed. Full traceback:\n\n```\n" + tb + "\n```")
        report.unexpected.append("Last 50 log lines:\n\n```\n" + _tail_of_log(50) + "\n```")
        path = report.write(status="FAILED")
        print(path)
        return 1


def dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)
