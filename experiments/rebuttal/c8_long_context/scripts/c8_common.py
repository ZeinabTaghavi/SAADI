"""Shared pieces of C8 (long-context reader baseline): paths, conditions, the prompt, files.

Everything that decides a number comes from C5 by import (experiments/rebuttal/c5_end_to_end/
scripts/c5_common.py): the reader prompt template, answer normalisation and scores, the paired
bootstrap. The full-document condition uses C5's template unchanged, with the whole document as
its single passage ("[1] <document>"), so the document comes first and the question last.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[4]
_C5_SCRIPTS = Path(os.environ.get("C5_EXP_DIR", _ROOT / "experiments/rebuttal/c5_end_to_end")) / "scripts"
if str(_C5_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_C5_SCRIPTS))

import c5_common as cc  # noqa: E402  (C5's shared module, read only)
from c5_common import (  # noqa: E402,F401  (re-exported for the C8 scripts)
    DATASET_DISPLAY,
    DATASETS,
    EXPECTED_QUERIES,
    MEASURES,
    READER_TEMPLATE,
    SEED,
    build_messages,
    clean_output,
    format_context,
    md_table,
    now,
    paired_bootstrap_ci,
    read_json,
    read_jsonl,
    score_answer,
    sha256_file,
    write_json,
    write_jsonl,
)

EXPERIMENT_ID = "c8_long_context"


def load_c5_step00() -> Any:
    """C5's step 00 module (Dataset, Tok, published-point and preserved-sources checks), loaded under
    its own name: C8's step 00 has the same file name."""

    import importlib.util

    name = "c5_step00_prepare"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _C5_SCRIPTS / "step00_prepare.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod

CONDITIONS = ("full_doc", "saadi", "vanilla", "closed_book")
# Conditions 2-4 are read from C5's context files (same queries, same passages).
C5_CONDITIONS = ("saadi", "vanilla", "closed_book")
CONDITION_DISPLAY = {
    "full_doc": "Full document",
    "saadi": "SAADI top 10",
    "vanilla": "Vanilla BM25 top 10",
    "closed_book": "Closed-book",
}
# Paired differences (first minus second), 10,000 bootstrap resamples over queries, seed 42.
PAIRS = (("full_doc", "saadi"), ("saadi", "vanilla"))
PAIRS_EXTRA = (("full_doc", "vanilla"), ("full_doc", "closed_book"), ("saadi", "closed_book"))
THIRDS = ("first", "middle", "last")


def document_passage_text(chunk_texts: list[str]) -> str:
    """The whole document from SAADI's chunks in document order. Chunks are cut at whitespace, so
    joining them with one space and collapsing whitespace (as C5's format_context does for every
    passage) gives the document text exactly as the other conditions see their passages."""

    return " ".join(" ".join(str(t).split()) for t in chunk_texts if str(t).strip())


def full_doc_messages(question: str, document_text: str) -> list[dict[str, str]]:
    """C5's prompt with the whole document as its one passage: instruction, document, question."""

    return build_messages(question, [{"text": document_text}])


@dataclass(frozen=True)
class C8Paths:
    root: Path
    exp: Path
    results: Path
    out: Path
    prepared: Path
    c5_contexts: Path
    c5_results: Path

    @classmethod
    def from_env(cls) -> "C8Paths":
        root = Path(os.environ.get("C8_ROOT_DIR", _ROOT))
        out = Path(os.environ.get("C8_OUT_DIR", root / "outputs/rebuttal" / EXPERIMENT_ID))
        return cls(
            root=root,
            exp=Path(os.environ.get("C8_EXP_DIR", root / "experiments/rebuttal" / EXPERIMENT_ID)),
            results=Path(os.environ.get("C8_RESULTS_DIR", root / "results/rebuttal" / EXPERIMENT_ID)),
            out=out,
            prepared=Path(os.environ.get("C8_PREPARED_DIR", out / "prepared")),
            c5_contexts=Path(os.environ.get("C8_C5_CONTEXTS_DIR", root / "outputs/rebuttal/c5_end_to_end/contexts")),
            c5_results=Path(os.environ.get("C8_C5_RESULTS_DIR", root / "results/rebuttal/c5_end_to_end")),
        )

    @property
    def reports(self) -> Path:
        return Path(os.environ.get("C8_REPORT_DIR", self.exp / "reports"))

    @property
    def settings_file(self) -> Path:
        return Path(os.environ.get("C8_SETTINGS_FILE", self.prepared / "vllm_settings.env"))

    # C5 inputs (read only)
    def c5_context_file(self, dataset: str, condition: str) -> Path:
        return self.c5_contexts / dataset / f"{condition}.jsonl.gz"

    def c5_manifest(self, dataset: str) -> Path:
        return self.c5_contexts / dataset / "manifest.json"

    def c5_para_labels(self, dataset: str) -> Path:
        return self.c5_results / "prepare" / f"labels_para_{dataset}.jsonl.gz"

    def c5_answers_file(self, dataset: str, condition: str) -> Path:
        return self.c5_results / "answers" / dataset / f"{condition}.jsonl"

    # C8 prepared inputs (git-ignored)
    def documents_file(self, dataset: str) -> Path:
        return self.prepared / dataset / "documents.jsonl.gz"

    def queries_file(self, dataset: str) -> Path:
        """One row per query: question, gold, document, evidence position, subset rank, and the prompt
        tokens of every condition (no text)."""
        return self.prepared / dataset / "queries.jsonl.gz"

    def manifest(self, dataset: str) -> Path:
        return self.prepared / dataset / "manifest.json"

    # C8 results (pushed)
    def answers_file(self, dataset: str, condition: str) -> Path:
        return self.results / "answers" / dataset / f"{condition}.jsonl"

    def rel(self, path: Path | str) -> str:
        p = Path(path)
        try:
            return str(p.resolve().relative_to(self.root.resolve()))
        except Exception:
            return str(p)


def subset_size(dataset: str) -> int | None:
    """C8_SUBSET_<DATASET> (number of queries) or None for all."""

    raw = os.environ.get(f"C8_SUBSET_{dataset.upper()}", "").strip()
    return int(raw) if raw else None


def fmt(v: Any, nd: int = 1) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if v != v:  # nan
            return "—"
        return f"{v:,.{nd}f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


__all__ = [
    "C5_CONDITIONS",
    "C8Paths",
    "CONDITIONS",
    "CONDITION_DISPLAY",
    "DATASETS",
    "DATASET_DISPLAY",
    "EXPECTED_QUERIES",
    "MEASURES",
    "PAIRS",
    "PAIRS_EXTRA",
    "READER_TEMPLATE",
    "SEED",
    "THIRDS",
    "build_messages",
    "cc",
    "clean_output",
    "document_passage_text",
    "fmt",
    "format_context",
    "full_doc_messages",
    "md_table",
    "now",
    "paired_bootstrap_ci",
    "read_json",
    "read_jsonl",
    "score_answer",
    "sha256_file",
    "subset_size",
    "write_json",
    "write_jsonl",
]
