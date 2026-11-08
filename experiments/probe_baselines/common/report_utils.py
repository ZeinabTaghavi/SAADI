"""Stage reports in the fixed format, and the stage runner used by run_all.sh.

Pure standard library (the stage runner starts commands in any virtual environment).

Every stage writes reports/<stage>.md and .json with four parts: what was done (the
exact commands, versions, wall clock), the numbers (from the numbers_*.json and
report_*.md files the stage's scripts leave in outputs/probe_baselines/<system>/),
anything unexpected (with the traceback and the last 50 log lines on failure), and
the files written. A finished stage leaves markers/<stage>.done, a failed one
markers/<stage>.failed.

    python common/report_utils.py stage <stage> [--force] -- <command ...> [::: <command ...>]
"""

from __future__ import annotations

import collections
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent
ROOT = EXP_DIR.parents[1]
OUT_DIR = Path(os.environ.get("PBL_OUT_DIR", ROOT / "outputs" / "probe_baselines"))
REPORT_DIR = Path(os.environ.get("PBL_REPORT_DIR", EXP_DIR / "reports"))
MARKER_DIR = OUT_DIR / "markers"
STAGES = ["00_check", "01_b0_vanilla", "02_b1_oracle", "03_b2_contextual", "04_b3_hipporag", "05_b4_lqca", "06_score", "07_report"]
STAGE_SYSTEM = {
    "01_b0_vanilla": "b0_vanilla",
    "02_b1_oracle": "b1_oracle",
    "03_b2_contextual": "b2_contextual",
    "04_b3_hipporag": "b3_hipporag",
    "05_b4_lqca": "b4_lqca",
    "00_check": "check",
    "06_score": "score",
    "07_report": "report",
}
SEPARATOR = ":::"


def md_table(headers: list, rows: list[list]) -> str:
    lines = ["| " + " | ".join(str(h) for h in headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def marker(stage: str, kind: str) -> Path:
    return MARKER_DIR / f"{stage}.{kind}"


def stage_status(stage: str) -> str:
    if marker(stage, "done").exists():
        return "done"
    if marker(stage, "failed").exists():
        return "failed"
    return "not run"


def _numbers_and_fragments(system: str) -> tuple[dict, list[str], list[str]]:
    folder = OUT_DIR / system
    numbers, fragments, files = {}, [], []
    if folder.exists():
        for path in sorted(folder.glob("numbers_*.json")):
            try:
                numbers[path.stem[len("numbers_") :]] = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                numbers[path.stem] = f"unreadable: {exc}"
        for path in sorted(folder.glob("report_*.md")):
            fragments.append(path.read_text(encoding="utf-8").strip())
        files = [str(p) for p in sorted(folder.glob("files_*.txt"))]
    return numbers, fragments, files


def write_report(
    stage: str,
    *,
    status: str,
    commands: list[str],
    seconds: float,
    numbers: dict,
    fragments: list[str],
    unexpected: list[str],
    files: list[str],
) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    md = [
        f"# probe_baselines - stage {stage}",
        "",
        f"Status: **{status}** - {time.strftime('%Y-%m-%d %H:%M:%S')}; wall clock {seconds / 60:.1f} min",
        "",
    ]
    md += ["## What was done", ""] + [f"- `{c}`" for c in commands]
    md += [f"- git commit `{git_commit()}`; python {platform.python_version()} on {platform.platform()}", ""]
    md += ["## Numbers", ""]
    md += [f for f in fragments] or ["(no tables from this stage)"]
    md += [""]
    md += ["## Anything unexpected", ""] + (unexpected or ["Nothing unexpected."]) + [""]
    written = []
    for f in files:
        try:
            written += [line.strip() for line in Path(f).read_text(encoding="utf-8").splitlines() if line.strip()]
        except Exception:
            pass
    md += ["## Files written", ""] + ([f"- `{w}`" for w in written] or ["- (see the stage's output folder)"])
    path = REPORT_DIR / f"{stage}.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    (REPORT_DIR / f"{stage}.json").write_text(
        json.dumps(
            {
                "stage": stage,
                "status": status,
                "commands": commands,
                "seconds": round(seconds),
                "numbers": numbers,
                "unexpected": unexpected,
                "files_written": written,
                "git_commit": git_commit(),
            },
            indent=1,
            default=str,
        ),
        encoding="utf-8",
    )
    return path


def run_stage(stage: str, commands: list[list[str]], *, force: bool) -> int:
    MARKER_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"{stage}.md"
    if marker(stage, "done").exists() and not force:
        print(f"== {stage} already done ({marker(stage, 'done')}); skipped (use --force to rerun)", flush=True)
        print(report_path, flush=True)
        return 0
    marker(stage, "failed").unlink(missing_ok=True)
    system = STAGE_SYSTEM.get(stage, stage)
    t0 = time.time()
    tail: collections.deque[str] = collections.deque(maxlen=50)
    status, unexpected, code = "ok", [], 0
    shown = []
    for cmd in commands:
        shown.append(" ".join(shlex.quote(c) for c in cmd))
        print(f"== {stage}: {shown[-1]}", flush=True)
        try:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                tail.append(line.rstrip("\n"))
            code = proc.wait()
        except Exception as exc:  # the command could not even start
            code = 127
            tail.append(f"could not start: {exc}")
        if code != 0:
            status = "FAILED"
            unexpected.append(f"`{shown[-1]}` exited with code {code}. Its traceback is in the log lines below.")
            unexpected.append("Last 50 log lines:\n\n```\n" + "\n".join(tail) + "\n```")
            break
    numbers, fragments, files = _numbers_and_fragments(system)
    for key, value in numbers.items():
        if isinstance(value, dict):
            unexpected += [f"({key}) {u}" for u in value.get("unexpected", []) or []]
    path = write_report(
        stage,
        status=status,
        commands=shown,
        seconds=time.time() - t0,
        numbers=numbers,
        fragments=fragments,
        unexpected=unexpected,
        files=files,
    )
    marker(stage, "done" if status == "ok" else "failed").write_text(
        json.dumps({"stage": stage, "status": status, "at": time.strftime("%Y-%m-%d %H:%M:%S"), "report": str(path)}), encoding="utf-8"
    )
    print(path, flush=True)
    return 0 if status == "ok" else 1


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0] == "stage":
        stage = argv[1]
        if stage not in STAGES:
            print(f"unknown stage {stage}; expected one of {STAGES}")
            return 2
        rest = argv[2:]
        force = "--force" in rest[: rest.index("--")] if "--" in rest else "--force" in rest
        cmd_args = rest[rest.index("--") + 1 :] if "--" in rest else []
        commands, current = [], []
        for a in cmd_args:
            if a == SEPARATOR:
                if current:
                    commands.append(current)
                current = []
            else:
                current.append(a)
        if current:
            commands.append(current)
        if not commands:
            print("no command given")
            return 2
        return run_stage(stage, commands, force=force)
    if len(argv) >= 1 and argv[0] == "status":
        for s in STAGES:
            print(s, stage_status(s))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
