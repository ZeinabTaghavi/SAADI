"""Shared pieces of C5 (end-to-end QA): paths, conditions, the reader prompt, scoring, files.

Imported by every C5 script. Everything that decides a number is defined once here:
the reader prompt template, answer normalisation and scores (the repo's
``saadi.evaluation.rag_answer_metrics``), the evidence hit rule (the repo's
``evaluate_retrieval_results``) and the paired bootstrap (E2's ``paired_bootstrap_ci``).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from saadi.core.types import RetrievalResult
from saadi.evaluation.rag_answer_metrics import (
    answer_containment,
    best_over_references,
    exact_match,
    normalize_answer,
    token_f1,
)
from saadi.evaluation.retrieval_metrics import evaluate_retrieval_results
from saadi.rebuttal.frozen_records import paired_bootstrap_ci  # noqa: F401  (re-exported for the scripts)

SEED = 42
EXPERIMENT_ID = "c5_end_to_end"

DATASETS = ("musique_32k", "novelhopqa")
DATASET_DISPLAY = {"musique_32k": "MuSiQue-32K", "novelhopqa": "NovelHopQA"}
EXPECTED_QUERIES = {"musique_32k": 900, "novelhopqa": 908}
# Key of the dataset inside outputs/saadi/reference (manifest, expected_metrics) and other_methods CSVs.
METRICS_NAME = {"musique_32k": "musique", "novelhopqa": "novelhopqa"}

# Conditions read by the reader, in report order. "arag_own" is A-RAG's own saved final answer:
# scored like the others, no reader call.
CONDITIONS = (
    "closed_book",
    "oracle",
    "vanilla",
    "saadi",
    "cr",
    "raptor",
    "raptor_tm",
    "arag",
    "arag_tm",
    "saadi_at_raptor_len",
    "saadi_at_arag_len",
)
SCORED_ONLY = ("arag_own",)
ALL_CONDITIONS = CONDITIONS + SCORED_ONLY
CONDITION_DISPLAY = {
    "closed_book": "Closed-book",
    "oracle": "Oracle (gold evidence chunks)",
    "vanilla": "Vanilla BM25 top 10",
    "saadi": "SAADI top 10",
    "cr": "Contextual Retrieval top 10",
    "raptor": "RAPTOR (default, as retrieved)",
    "raptor_tm": "RAPTOR, token-matched to SAADI",
    "arag": "A-RAG read chunks (as read)",
    "arag_tm": "A-RAG read chunks, token-matched to SAADI",
    "arag_own": "A-RAG own final answer",
    "saadi_at_raptor_len": "SAADI cut to RAPTOR's length (per query)",
    "saadi_at_arag_len": "SAADI cut to A-RAG's length (per query)",
}
# Paired differences reported (SAADI minus each).
PAIRS = (("saadi", "vanilla"), ("saadi", "cr"), ("saadi", "raptor"), ("saadi", "arag"))
PAIRS_EXTRA = (("saadi_at_raptor_len", "raptor"), ("saadi_at_arag_len", "arag"), ("saadi", "raptor_tm"), ("saadi", "arag_tm"),
               ("saadi", "arag_own"), ("saadi", "closed_book"), ("oracle", "saadi"))
MEASURES = ("em", "f1", "contain")

# ---------------------------------------------------------------------------------------------
# Reader prompt: one template for every condition (closed-book gets the empty-context line).
# ---------------------------------------------------------------------------------------------
READER_TEMPLATE = (
    "Answer the question using the context passages below. "
    "If the passages do not contain the answer, answer from your own knowledge.\n"
    "\n"
    "Context:\n"
    "{context}\n"
    "\n"
    "Question: {question}\n"
    "Answer with a short phrase."
)
PASSAGE_TEMPLATE = "[{rank}] {text}"
PASSAGE_SEPARATOR = "\n\n"
EMPTY_CONTEXT = "(no passages)"


def format_context(passages: list[dict[str, Any]]) -> str:
    """Passages in rank order, numbered from 1; closed-book gets EMPTY_CONTEXT."""

    if not passages:
        return EMPTY_CONTEXT
    return PASSAGE_SEPARATOR.join(
        PASSAGE_TEMPLATE.format(rank=i, text=" ".join(str(p["text"]).split())) for i, p in enumerate(passages, start=1)
    )


def build_prompt(question: str, passages: list[dict[str, Any]]) -> str:
    return READER_TEMPLATE.format(context=format_context(passages), question=str(question).strip())


def build_messages(question: str, passages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """One user message, no system prompt (the model's chat template adds nothing else)."""

    return [{"role": "user", "content": build_prompt(question, passages)}]


# ---------------------------------------------------------------------------------------------
# Scores (standard SQuAD normalisation; best over the gold answers)
# ---------------------------------------------------------------------------------------------


def clean_output(text: Any) -> str:
    """The reader is asked for a short phrase; keep its first non-empty line, strip a leading 'Answer:'."""

    raw = str(text or "").strip()
    for line in raw.splitlines():
        line = line.strip()
        if line:
            raw = line
            break
    low = raw.lower()
    for prefix in ("final answer:", "answer:"):
        if low.startswith(prefix):
            raw = raw[len(prefix):].strip()
            break
    return raw


def score_answer(output: Any, gold: list[str]) -> dict[str, float]:
    pred = clean_output(output)
    refs = [str(g) for g in gold if str(g or "").strip()]
    if not refs:
        return {"em": float("nan"), "f1": float("nan"), "contain": float("nan")}
    return {
        "em": float(best_over_references(exact_match, pred, refs)),
        "f1": float(best_over_references(token_f1, pred, refs)),
        # Containment: a gold answer string (normalised) inside the normalised output.
        "contain": float(answer_containment(pred, refs)),
    }


# ---------------------------------------------------------------------------------------------
# Evidence: the paper's hit rule (a gold chunk, or a whole silver group, inside the context)
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Labels:
    query_id: str
    doc_id: str
    gold: tuple[str, ...]
    silver: tuple[str, ...]
    groups: tuple[tuple[str, ...], ...]

    @property
    def eligible(self) -> bool:
        return bool(self.gold) or any(len(g) > 0 for g in self.groups)

    def evidence_chunks(self) -> list[str]:
        """Gold chunks then the chunks of every silver group, first occurrence kept."""

        out = list(dict.fromkeys([*self.gold, *[c for g in self.groups for c in g]]))
        return out


def labels_from_row(row: dict[str, Any]) -> Labels:
    gold = tuple(str(x) for x in row.get("gold_chunk_ids", row.get("relevant_chunk_ids", [])) or [])
    silver = tuple(str(x) for x in row.get("silver_chunk_ids", []) or [])
    raw_groups = row.get("silver_chunk_groups", row.get("evidence_groups", [])) or []
    groups = tuple(tuple(str(c) for c in g) for g in raw_groups if isinstance(g, list) and g)
    if not silver:
        silver = tuple(dict.fromkeys(c for g in groups for c in g))
    return Labels(str(row.get("query_id", "")), str(row.get("doc_id", "")), gold, silver, groups)


def evidence_hits(labels: list[Labels], lists: list[list[str]], k: int | None = None) -> list[bool | None]:
    """Per query: True/False when the query has evidence labels, None otherwise.

    The repo's ``evaluate_retrieval_results`` (the function behind every HR of the paper), at
    cutoff k (default: the longest list, i.e. the whole context).
    """

    if len(labels) != len(lists):
        raise ValueError("labels and lists differ in length")
    if not labels:
        return []
    depth = int(k) if k is not None else max(1, max((len(x) for x in lists), default=1))
    results = [
        RetrievalResult(
            query_id=lab.query_id,
            doc_id=lab.doc_id,
            question="",
            retrieved_chunk_ids=list(lst),
            retrieved_indices=[],
            scores=[],
            gold_chunk_ids=list(lab.gold),
            silver_chunk_ids=list(lab.silver),
            silver_chunk_groups=[list(g) for g in lab.groups],
        )
        for lab, lst in zip(labels, lists, strict=True)
    ]
    report = evaluate_retrieval_results(results, eval_k=depth, retrieve_k=None, compute_per_query=True)
    return [bool(p["hit"]) if p["eligible"] else None for p in report.per_query]


def hit_rate(flags: Iterable[bool | None]) -> dict[str, Any]:
    vals = [f for f in flags if f is not None]
    return {"eligible": len(vals), "hits": int(sum(vals)), "hr_percent": (100.0 * sum(vals) / len(vals)) if vals else None}


# ---------------------------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------------------------


def env_path(name: str, default: str | Path | None = None) -> Path:
    value = os.environ.get(name)
    if value:
        return Path(value)
    if default is None:
        raise KeyError(f"environment variable {name} is not set (source experiments/rebuttal/c5_end_to_end/env.sh)")
    return Path(default)


@dataclass(frozen=True)
class C5Paths:
    root: Path
    exp: Path
    results: Path
    out: Path
    contexts: Path

    @classmethod
    def from_env(cls) -> "C5Paths":
        root = env_path("C5_ROOT_DIR", Path(__file__).resolve().parents[4])
        return cls(
            root=root,
            exp=env_path("C5_EXP_DIR", root / "experiments/rebuttal" / EXPERIMENT_ID),
            results=env_path("C5_RESULTS_DIR", root / "results/rebuttal" / EXPERIMENT_ID),
            out=env_path("C5_OUT_DIR", root / "outputs/rebuttal" / EXPERIMENT_ID),
            contexts=env_path("C5_CONTEXTS_DIR", root / "outputs/rebuttal" / EXPERIMENT_ID / "contexts"),
        )

    @property
    def reports(self) -> Path:
        return self.exp / "reports"

    @property
    def logs(self) -> Path:
        return self.exp / "logs"

    def context_file(self, dataset: str, condition: str) -> Path:
        return self.contexts / dataset / f"{condition}.jsonl.gz"

    def context_manifest(self, dataset: str) -> Path:
        return self.contexts / dataset / "manifest.json"

    def answers_file(self, dataset: str, condition: str) -> Path:
        return self.results / "answers" / dataset / f"{condition}.jsonl"

    def rel(self, path: Path | str) -> str:
        p = Path(path)
        try:
            return str(p.resolve().relative_to(self.root.resolve()))
        except Exception:
            return str(p)


# ---------------------------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if str(path).endswith(".gz") else open
    rows = []
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """Atomic write. Gzip files are deterministic (no time stamp, no file name in the header), so
    the same rows always give the same sha256 (the reader uses it to keep or redo answers)."""

    import io

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    n = 0
    with open(tmp, "wb") as raw:
        binary = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) if str(path).endswith(".gz") else raw
        with io.TextIOWrapper(binary, encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                n += 1
    os.replace(tmp, path)
    return n


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_commit(root: Path) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def md_table(header: list[str], rows: list[list[Any]]) -> list[str]:
    def cell(v: Any) -> str:
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.2f}"
        return str(v).replace("|", "\\|").replace("\n", " ")

    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    out += ["| " + " | ".join(cell(v) for v in r) + " |" for r in rows]
    return out


def normalize_question(text: Any) -> str:
    return " ".join(str(text or "").split()).strip()


__all__ = [
    "ALL_CONDITIONS",
    "C5Paths",
    "CONDITIONS",
    "CONDITION_DISPLAY",
    "DATASETS",
    "DATASET_DISPLAY",
    "EMPTY_CONTEXT",
    "EXPECTED_QUERIES",
    "Labels",
    "MEASURES",
    "METRICS_NAME",
    "PAIRS",
    "PAIRS_EXTRA",
    "PASSAGE_SEPARATOR",
    "PASSAGE_TEMPLATE",
    "READER_TEMPLATE",
    "SCORED_ONLY",
    "SEED",
    "build_messages",
    "build_prompt",
    "clean_output",
    "evidence_hits",
    "format_context",
    "hit_rate",
    "labels_from_row",
    "md_table",
    "normalize_answer",
    "normalize_question",
    "read_json",
    "read_jsonl",
    "score_answer",
    "sha256_file",
    "write_json",
    "write_jsonl",
]
