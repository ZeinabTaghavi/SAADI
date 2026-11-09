"""Shared helpers of C4 (sub-document control): settings, paper sources, step reports.

C4 re-ranks existing per-chunk scores of the concatenated MuSiQue / QASPER settings
(32K to 256K tokens) inside each query's gold sub-document. Nothing here changes the
published scoring path: repository code is imported, never edited.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import statistics
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(os.environ.get("C4_ROOT_DIR") or Path(__file__).resolve().parents[4]).resolve()
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from saadi.rebuttal.frozen_records import (  # noqa: E402
    _METHOD_FILE_LABEL,
    _REPORT_DIR_LABEL,
    PUBLISHED_SETTINGS,
)

EXPERIMENT_ID = "c4_subdoc_control"
SEED = int(os.environ.get("C4_SEED", "42"))
N_BOOT = int(os.environ.get("C4_BOOTSTRAP_RESAMPLES", "10000"))

EXP_DIR = Path(os.environ.get("C4_EXP_DIR") or ROOT / "experiments" / "rebuttal" / EXPERIMENT_ID)
RESULTS_DIR = Path(os.environ.get("C4_RESULTS_DIR") or ROOT / "results" / "rebuttal" / EXPERIMENT_ID)
REPORT_DIR = Path(os.environ.get("C4_REPORT_DIR") or EXP_DIR / "reports")
LOG_DIR = Path(os.environ.get("C4_LOG_DIR") or EXP_DIR / "logs")
WORK_DIR = Path(os.environ.get("C4_WORK_DIR") or EXP_DIR / "work")
MAP_DIR = Path(os.environ.get("C4_MAP_DIR") or RESULTS_DIR / "subdoc_map")
REFERENCE_DIR = Path(os.environ.get("C4_REFERENCE_DIR") or ROOT / "outputs" / "saadi" / "reference")
COMPARISON_DIR = Path(os.environ.get("C4_COMPARISON_DIR") or ROOT / "outputs" / "saadi" / "comparison_inputs")
DENSE_STREAMS_DIR = Path(os.environ.get("C4_DENSE_STREAMS_DIR") or ROOT / "outputs" / "rebuttal" / "C2_heldout_fusion" / "streams")
CONFIG_PATH = Path(os.environ.get("C4_CONFIG") or ROOT / "configs" / "methods" / "sadi.yaml")

DATASETS = ("musique", "qasper")
SIZES = ("32K", "64K", "128K", "256K")
RETRIEVERS = ("bm25", "contriever", "bge_m3", "jina", "qwen")
DATASET_DISPLAY = {"musique": "MuSiQue", "qasper": "QASPER"}
RETRIEVER_DISPLAY = {"bm25": "BM25", "contriever": "Contriever", "bge_m3": "BGE-M3", "jina": "Jina", "qwen": "Qwen"}
# Table 1's MuSiQue and QASPER columns are these expansion settings (reference manifest, study "main").
MAIN_CONDITION = {"musique": "32K_expansion", "qasper": "64K_expansion"}
_NS = "100_tokens/llm_cache/{ds}/hierarchical/branching_factor_5/chunk_100/AE_5_2_3/Qwen3-30B-A3B-Instruct-2507/AEWord"


@dataclass(frozen=True)
class Setting:
    key: str  # musique_32k
    dataset: str  # musique
    size: str  # 32K

    @property
    def condition(self) -> str:
        return f"{self.size}_expansion"

    @property
    def display(self) -> str:
        return f"{DATASET_DISPLAY[self.dataset]}-{self.size}"

    @property
    def target_tokens(self) -> int:
        return int(self.size[:-1]) * 1000

    @property
    def base_dir(self) -> Path:
        return ROOT / "outputs" / f"{self.dataset}_ablation" / self.condition

    @property
    def prepared_dir(self) -> Path:
        return self.base_dir / "prepared"

    @property
    def namespace(self) -> Path:
        return self.base_dir / _NS.format(ds=self.dataset)

    @property
    def index_json(self) -> Path:
        return self.namespace / "index.json"

    def saved_results_json(self, retriever: str) -> Path:
        return (
            self.namespace
            / "mbrB_ablation"
            / _REPORT_DIR_LABEL
            / retriever
            / "retrieval_reports"
            / f"retrieval_results__{retriever}__{_METHOD_FILE_LABEL}__per_document.json"
        )


ALL_SETTINGS: dict[str, Setting] = {
    f"{d}_{s.lower()}": Setting(f"{d}_{s.lower()}", d, s) for d in DATASETS for s in SIZES
}


def env_list(name: str, default: Iterable[str]) -> list[str]:
    raw = os.environ.get(name, "")
    vals = [v.strip() for v in raw.split(",") if v.strip()] if raw.strip() else list(default)
    return vals


def selected_settings(arg: str | None = None) -> list[Setting]:
    keys = [k.strip() for k in arg.split(",")] if arg else env_list("C4_SETTINGS", ALL_SETTINGS)
    unknown = [k for k in keys if k not in ALL_SETTINGS]
    if unknown:
        raise SystemExit(f"Unknown setting(s) {unknown}; known: {sorted(ALL_SETTINGS)}")
    return [ALL_SETTINGS[k] for k in keys]


def selected_retrievers(arg: str | None = None) -> list[str]:
    vals = [v.strip() for v in arg.split(",")] if arg else env_list("C4_RETRIEVERS", RETRIEVERS)
    unknown = [v for v in vals if v not in RETRIEVERS]
    if unknown:
        raise SystemExit(f"Unknown retriever(s) {unknown}; known: {list(RETRIEVERS)}")
    return vals


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except Exception:
        return str(p)


def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, ensure_ascii=False, default=_json_default)
    os.replace(tmp, path)


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
        return str(obj)
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def qid_sort_key(qid: str) -> tuple[int, Any]:
    return (0, int(qid)) if str(qid).isdigit() else (1, str(qid))


# ---------------------------------------------------------------------------
# The paper's saved rankings (reference + comparison bundles)
# ---------------------------------------------------------------------------


def paper_sources() -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Every saved per-query SAADI ranking of the paper on a concatenated setting.

    Keys (setting, retriever). Sources: the reference bundle (study "context_expansion",
    Table 26 BM25 rows; study "main", Table 1 on MuSiQue-32K and QASPER-64K) and the
    comparison bundle (Table 26 Contriever rows). Each entry carries the expected hit
    counts the paper reports for that ranking.
    """

    out: dict[tuple[str, str], list[dict[str, Any]]] = {}
    manifest = read_json(REFERENCE_DIR / "manifest.json")
    expected_rows = read_json(REFERENCE_DIR / "expected_metrics.json")
    exp_by_sid: dict[str, dict[int, dict[str, Any]]] = {}
    for row in expected_rows:
        exp_by_sid.setdefault(str(row["source_id"]), {})[int(row["cutoff"])] = row
    for s in manifest.get("sources", []):
        study = s.get("study")
        ds = s.get("dataset")
        if ds not in DATASETS:
            continue
        if study == "context_expansion":
            cond = str(s.get("condition"))
        elif study == "main" and s.get("condition") == "default":
            cond = MAIN_CONDITION[ds]
            if f"/{cond}/" not in str(s.get("original_source", "")):
                raise RuntimeError(f"reference main/{ds}/{s.get('retriever')} does not come from {cond}: {s.get('original_source')}")
        else:
            continue
        size = cond.split("_")[0]
        if size not in SIZES:
            continue
        key = f"{ds}_{size.lower()}"
        out.setdefault((key, str(s["retriever"])), []).append(
            {
                "bundle": "reference",
                "study": study,
                "source_id": s["source_id"],
                "ranking_path": str(REFERENCE_DIR / s["path"]),
                "original_source": s.get("original_source"),
                "saved_depth": s.get("saved_depth"),
                "expected": {
                    k: {"queries": int(v["queries"]), "eligible": int(v["eligible"]), "hits": int(v["hits"])}
                    for k, v in sorted(exp_by_sid.get(str(s["source_id"]), {}).items())
                },
            }
        )
    cmp_manifest_path = COMPARISON_DIR / "manifest.json"
    if cmp_manifest_path.exists():
        for r in read_json(cmp_manifest_path).get("rankings", []):
            ds = r.get("dataset")
            size = str(r.get("condition", "")).split("_")[0]
            if ds not in DATASETS or size not in SIZES:
                continue
            key = f"{ds}_{size.lower()}"
            out.setdefault((key, str(r["retriever"])), []).append(
                {
                    "bundle": "comparison_inputs",
                    "study": "context_expansion",
                    "source_id": r["source_id"],
                    "ranking_path": str(COMPARISON_DIR / r["path"]),
                    "original_source": r.get("original_source"),
                    "saved_depth": r.get("saved_depth"),
                    "selection": r.get("selection"),
                    "expected": {
                        int(e["cutoff"]): {"queries": int(e["queries"]), "eligible": int(e["eligible"]), "hits": int(e["hits"])}
                        for e in r.get("expected_metrics", [])
                    },
                }
            )
    return out


def load_ranking_rows(source: dict[str, Any]) -> list[dict[str, Any]]:
    with gzip.open(source["ranking_path"], "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evidence_from_row(row: dict[str, Any]) -> tuple[list[str], list[list[str]]]:
    """(gold chunk ids, evidence groups) of a saved ranking row, either field spelling."""

    gold = [str(x) for x in (row.get("relevant_chunk_ids", row.get("gold_chunk_ids", [])) or [])]
    groups_raw = row.get("evidence_groups", row.get("silver_chunk_groups", [])) or []
    groups = [[str(c) for c in g if str(c).strip()] for g in groups_raw if isinstance(g, list)]
    return gold, [g for g in groups if g]


def published_settings() -> dict[str, Any]:
    """Fusion and field weights from the config's cached_retrieval block (never hard-coded).

    They must equal the settings of the saved published run (frozen_records.PUBLISHED_SETTINGS),
    otherwise the reproduction gate would test something else; stop if they differ.
    """

    import yaml

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        cfg = (yaml.safe_load(handle) or {}).get("cached_retrieval", {}) or {}
    out: dict[str, Any] = {}
    for key, ref in PUBLISHED_SETTINGS.items():
        if key not in cfg:
            raise RuntimeError(f"{rel(CONFIG_PATH)}: cached_retrieval.{key} missing")
        val = cfg[key]
        if isinstance(ref, (int, float)) and not isinstance(ref, bool):
            if not math.isclose(float(val), float(ref), rel_tol=0, abs_tol=1e-12):
                raise RuntimeError(f"{rel(CONFIG_PATH)}: {key}={val} differs from the published run ({ref})")
            out[key] = float(val)
        else:
            if str(val) != str(ref):
                raise RuntimeError(f"{rel(CONFIG_PATH)}: {key}={val!r} differs from the published run ({ref!r})")
            out[key] = val
    return out


# ---------------------------------------------------------------------------
# Small statistics
# ---------------------------------------------------------------------------


def dist(values: Iterable[float]) -> dict[str, Any]:
    vals = [float(v) for v in values]
    if not vals:
        return {"n": 0, "min": None, "median": None, "max": None, "mean": None}
    return {
        "n": len(vals),
        "min": min(vals),
        "median": float(statistics.median(vals)),
        "max": max(vals),
        "mean": sum(vals) / len(vals),
    }


def fnum(x: Any, nd: int = 1) -> str:
    if x is None:
        return "-"
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if math.isnan(xf):
        return "-"
    if nd == 0 or (xf.is_integer() and nd <= 1 and abs(xf) >= 1):
        return f"{xf:.0f}"
    return f"{xf:.{nd}f}"


# ---------------------------------------------------------------------------
# Step reports
# ---------------------------------------------------------------------------


class StepReport:
    """Self-contained Markdown + JSON report; written even when the step fails."""

    def __init__(self, step: int, title: str, *, command: str) -> None:
        self.step = step
        self.title = title
        self.command = command
        self.lines: list[str] = []
        self.data: dict[str, Any] = {"step": step, "title": title, "command": command, "seed": SEED}
        self.problems: list[str] = []
        self.warnings: list[str] = []
        self.files: list[str] = []
        self.started = time.time()

    @property
    def md_path(self) -> Path:
        return REPORT_DIR / f"step_{self.step:02d}.md"

    @property
    def json_path(self) -> Path:
        return REPORT_DIR / f"step_{self.step:02d}.json"

    def h(self, text: str, level: int = 2) -> None:
        self.lines += ["", "#" * level + " " + text, ""]

    def p(self, text: str) -> None:
        self.lines.append(text)

    def bullet(self, text: str) -> None:
        self.lines.append(f"- {text}")

    def table(self, headers: list[str], rows: list[list[Any]]) -> None:
        self.lines.append("| " + " | ".join(headers) + " |")
        self.lines.append("| " + " | ".join("---" for _ in headers) + " |")
        for r in rows:
            self.lines.append("| " + " | ".join(str(c) for c in r) + " |")
        self.lines.append("")

    def code(self, text: str, lang: str = "") -> None:
        self.lines += [f"```{lang}", text.rstrip(), "```"]

    def fail(self, msg: str) -> None:
        self.problems.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def wrote(self, path: Path | str) -> None:
        self.files.append(rel(path))

    @property
    def ok(self) -> bool:
        return not self.problems

    def write(self) -> int:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        status = "OK" if self.ok else "FAILED"
        head = [
            f"# C4 sub-document control - step {self.step:02d}: {self.title} ({status})",
            "",
            "Command (from the repository root on the server):",
            "",
            "```bash",
            self.command,
            "```",
            "",
            f"Seed {SEED}. CPU only. Run time {time.time() - self.started:.0f} s.",
        ]
        tail: list[str] = []
        cap = 30  # keeps the report short; the JSON holds the full lists
        if self.problems:
            tail += ["", "## Problems (the step failed)", ""] + [f"- {p}" for p in self.problems[:cap]]
            if len(self.problems) > cap:
                tail.append(f"- ... {len(self.problems) - cap} more (see {rel(self.json_path)})")
        if self.warnings:
            tail += ["", "## Warnings", ""] + [f"- {w}" for w in self.warnings[:cap]]
            if len(self.warnings) > cap:
                tail.append(f"- ... {len(self.warnings) - cap} more (see {rel(self.json_path)})")
        files = sorted(set(self.files + [rel(self.md_path), rel(self.json_path)]))
        tail += ["", "## Files written", ""] + [f"- `{f}`" for f in files]
        self.md_path.write_text("\n".join(head + self.lines + tail) + "\n", encoding="utf-8")
        self.data.update(
            {
                "status": status,
                "problems": self.problems,
                "warnings": self.warnings,
                "files_written": files,
                "seconds": time.time() - self.started,
            }
        )
        write_json(self.json_path, self.data)
        return 0 if self.ok else 1


def run_step(report: StepReport, body: Callable[[StepReport], None]) -> int:
    try:
        body(report)
    except SystemExit:
        raise
    except BaseException as exc:  # the report must still be written
        report.fail(f"unexpected error: {type(exc).__name__}: {exc}")
        report.h("Traceback")
        report.code(traceback.format_exc())
    code = report.write()
    print(f"[c4] step {report.step:02d} {'OK' if code == 0 else 'FAILED'}; report: {rel(report.md_path)}", flush=True)
    return code


def require_step_ok(step: int) -> dict[str, Any]:
    """Later steps refuse to run on top of a failed or missing earlier step."""

    path = REPORT_DIR / f"step_{step:02d}.json"
    if not path.exists():
        raise RuntimeError(f"step {step:02d} has not run ({rel(path)} missing)")
    data = read_json(path)
    if data.get("status") != "OK":
        raise RuntimeError(f"step {step:02d} did not pass (status {data.get('status')}); fix it first")
    return data
