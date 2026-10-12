"""Load upstream MBR retrieval artifacts for generation experiments.

Retrieval itself remains outside this package. Future generation runs should
consume MBR outputs through this loader instead of rerunning retrieval or
hard-coding retrieval-report paths inside context builders.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...core.types import RetrievalResult
from ...indexing.retrieval_methods.mbr import format_mbr_method_label
from ...io.hierarchy_cache import load_raw_chunk_records_from_cache_index
from ..context_builders.base import RetrievedContextItem
from .cross_document_mbr import CrossDocumentMBRConfig


@dataclass(frozen=True, slots=True)
class MBRRetrievalSelection:
    dataset: str
    retriever: str
    top_k: int
    summary_k: int
    method: str = "mbr"
    scope: str = "per_document"
    results_json: Path | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "retriever": self.retriever,
            "top_k": self.top_k,
            "summary_k": self.summary_k,
            "method": self.method,
            "scope": self.scope,
            "results_json": str(self.results_json) if self.results_json else None,
        }


def retrieval_result_from_dict(row: dict[str, Any]) -> RetrievalResult:
    return RetrievalResult(
        query_id=str(row.get("query_id", "")),
        doc_id=str(row.get("doc_id", "")),
        question=str(row.get("question", "")),
        retrieved_chunk_ids=[str(value) for value in row.get("retrieved_chunk_ids", []) or []],
        retrieved_indices=[int(value) for value in row.get("retrieved_indices", []) or []],
        scores=[float(value) for value in row.get("scores", []) or []],
        gold_chunk_ids=[str(value) for value in row.get("relevant_chunk_ids", row.get("gold_chunk_ids", [])) or []],
        silver_chunk_ids=[str(value) for value in row.get("silver_chunk_ids", []) or []],
        silver_chunk_groups=[
            [str(value) for value in group]
            for group in row.get("evidence_groups", row.get("silver_chunk_groups", [])) or []
            if isinstance(group, list)
        ],
    )


class MBRRetrievalLoader:
    """Boundary object for reading already-produced MBR retrieval results."""

    def __init__(self, selection: MBRRetrievalSelection) -> None:
        self.selection = selection

    @staticmethod
    def default_step_b_weights() -> dict[str, float]:
        """Default MBRB weights used by the Step-B retrieval ablation scripts."""

        return {
            "lambda_t": 1.0,
            "lambda_n": 0.5,
            "lambda_h": 0.3,
            "memory_name_weight": 1.0,
            "memory_alias_weight": 1.0,
            "memory_description_weight": 1.0,
        }

    @staticmethod
    def default_cross_document_config(
        retriever_name: str,
        top_k: int,
        *,
        target_token_budget: int | None = None,
        tolerance_tokens: int = 0,
    ) -> CrossDocumentMBRConfig:
        """Build a :class:`CrossDocumentMBRConfig` using the default Step-B weights."""
        weights = MBRRetrievalLoader.default_step_b_weights()
        return CrossDocumentMBRConfig(
            retriever_name=retriever_name,
            top_k=top_k,
            lambda_t=weights["lambda_t"],
            lambda_n=weights["lambda_n"],
            lambda_h=weights["lambda_h"],
            memory_name_weight=weights["memory_name_weight"],
            memory_alias_weight=weights["memory_alias_weight"],
            memory_description_weight=weights["memory_description_weight"],
            target_token_budget=target_token_budget,
            tolerance_tokens=tolerance_tokens,
        )

    @staticmethod
    def default_step_b_method_label() -> str:
        weights = MBRRetrievalLoader.default_step_b_weights()
        return format_mbr_method_label(
            lambda_t=weights["lambda_t"],
            lambda_n=weights["lambda_n"],
            lambda_h=weights["lambda_h"],
            normalization_method="none",
            memory_name_weight=weights["memory_name_weight"],
            memory_alias_weight=weights["memory_alias_weight"],
            memory_description_weight=weights["memory_description_weight"],
        )

    @staticmethod
    def default_step_b_run_name() -> str:
        weights = MBRRetrievalLoader.default_step_b_weights()
        return (
            f"mbr_B_lt{weights['lambda_t']:.1f}_"
            f"lm{weights['lambda_n']:.1f}_"
            f"lh{weights['lambda_h']:.1f}_"
            f"mn{weights['memory_name_weight']:.1f}_"
            f"ma{weights['memory_alias_weight']:.1f}_"
            f"md{weights['memory_description_weight']:.1f}"
        )

    @staticmethod
    def default_step_b_output_dir(dataset_experiment_dir: str | Path, *, retriever: str) -> Path:
        return (
            Path(dataset_experiment_dir).resolve()
            / "mbrB_ablation"
            / MBRRetrievalLoader.default_step_b_run_name()
            / str(retriever)
        )

    @staticmethod
    def old_all_signal_default_method_label() -> str:
        """Temporary compatibility for earlier SAADI runs that used 1/1/1 outer weights."""

        return format_mbr_method_label(
            lambda_t=1.0,
            lambda_n=1.0,
            lambda_h=1.0,
            normalization_method="none",
            memory_name_weight=1.0,
            memory_alias_weight=1.0,
            memory_description_weight=1.0,
        )

    @staticmethod
    def legacy_step_b_method_label() -> str:
        """Older Step-B artifacts sometimes omitted memory suffixes."""

        weights = MBRRetrievalLoader.default_step_b_weights()
        return format_mbr_method_label(
            lambda_t=weights["lambda_t"],
            lambda_n=weights["lambda_n"],
            lambda_h=weights["lambda_h"],
            normalization_method="none",
        )

    @staticmethod
    def legacy_all_signal_method_label() -> str:
        """Older MBR artifacts used the outer 1/1/1 weights without memory suffix."""

        return format_mbr_method_label(
            lambda_t=1.0,
            lambda_n=1.0,
            lambda_h=1.0,
            normalization_method="none",
        )

    @classmethod
    def default_method_candidates(cls) -> list[str]:
        """Preferred MBR labels for generation, ordered from MBRB to legacy."""

        candidates = [
            cls.default_step_b_method_label(),
            cls.old_all_signal_default_method_label(),
            cls.legacy_step_b_method_label(),
            cls.legacy_all_signal_method_label(),
        ]
        out: list[str] = []
        for candidate in candidates:
            if candidate not in out:
                out.append(candidate)
        return out

    @classmethod
    def for_step_b_default(
        cls,
        *,
        dataset: str,
        retriever: str,
        top_k: int,
        dataset_experiment_dir: str | Path | None = None,
        results_json: str | Path | None = None,
        scope: str = "per_document",
    ) -> "MBRRetrievalLoader":
        method = cls.default_step_b_method_label()
        resolved_results_json = Path(results_json).resolve() if results_json else None
        resolved_method = method
        if resolved_results_json is None and dataset_experiment_dir is not None:
            resolved_results_json, resolved_method = resolve_mbr_results_json_with_method(
                dataset_experiment_dir=dataset_experiment_dir,
                retriever=retriever,
                method=method,
                scope=scope,
                min_retrieve_k=top_k,
            )
        return cls(
            MBRRetrievalSelection(
                dataset=str(dataset),
                retriever=str(retriever),
                top_k=int(top_k),
                summary_k=0,
                method=resolved_method,
                scope=str(scope),
                results_json=resolved_results_json,
            )
        )

    def load_results(self) -> list[RetrievalResult]:
        if self.selection.results_json is None:
            raise ValueError(
                "MBR retrieval results path is not configured yet. "
                "Pass --mbr_results_json or --dataset_experiment_dir/--config so report-index resolution can find it."
            )
        path = self.selection.results_json
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON list of retrieval results: {path}")
        return [retrieval_result_from_dict(row) for row in payload if isinstance(row, dict)]


def _existing_or_report_local(path_text: str, *, report_dir: Path) -> Path | None:
    path = Path(path_text)
    if path.exists():
        return path.resolve()
    local = report_dir / path.name
    if local.exists():
        return local.resolve()
    return None


def _load_report_runs(report_index_json: Path) -> list[dict[str, Any]]:
    if not report_index_json.exists():
        return []
    with report_index_json.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    runs = payload.get("runs", []) if isinstance(payload, dict) else []
    return [row for row in runs if isinstance(row, dict)]


def _candidate_report_dirs(root: Path, *, retriever: str) -> list[Path]:
    dirs: list[Path] = []
    default_step_b = MBRRetrievalLoader.default_step_b_output_dir(root, retriever=retriever) / "retrieval_reports"
    dirs.append(default_step_b)
    mbr_b_root = root / "mbrB_ablation"
    if mbr_b_root.exists():
        dirs.extend(sorted(mbr_b_root.glob(f"*/{retriever}/retrieval_reports")))
    dirs.append(root / "retrieval_reports")

    out: list[Path] = []
    seen: set[str] = set()
    for path in dirs:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _resolve_mbr_results_json_in_report_dir(
    *,
    report_dir: Path,
    retriever: str,
    method_labels: list[str],
    scope: str,
    min_retrieve_k: int | None,
) -> tuple[Path, str] | None:
    runs = _load_report_runs(report_dir / "report_index.json")
    for method_label in method_labels:
        candidates: list[tuple[int, Path]] = []
        for run in runs:
            if str(run.get("retriever", "")) != str(retriever):
                continue
            if str(run.get("method", "")) != method_label:
                continue
            if str(run.get("scope", "per_document")) != str(scope):
                continue
            retrieve_k = run.get("retrieve_k")
            try:
                retrieve_k_int = int(retrieve_k)
            except Exception:
                retrieve_k_int = 0
            if min_retrieve_k is not None and retrieve_k_int and retrieve_k_int < int(min_retrieve_k):
                continue
            path_text = str(run.get("results_json", "") or "")
            if not path_text:
                continue
            resolved = _existing_or_report_local(path_text, report_dir=report_dir)
            if resolved is not None:
                candidates.append((retrieve_k_int, resolved))
        if candidates:
            candidates.sort(key=lambda item: (item[0] if item[0] else 10**9))
            return candidates[0][1], method_label

        expected = report_dir / f"retrieval_results__{retriever}__{method_label}__{scope}.json"
        if expected.exists():
            return expected.resolve(), method_label
    return None


def resolve_mbr_results_json(
    *,
    dataset_experiment_dir: str | Path,
    retriever: str,
    method: str | None = None,
    scope: str = "per_document",
    min_retrieve_k: int | None = None,
) -> Path:
    """Resolve an MBR retrieval-results artifact from a cache namespace.

    The default method is the Step-B memory setting with name/alias/description
    weights all set to 1.0.
    """

    path, _method = resolve_mbr_results_json_with_method(
        dataset_experiment_dir=dataset_experiment_dir,
        retriever=retriever,
        method=method,
        scope=scope,
        min_retrieve_k=min_retrieve_k,
    )
    return path


def resolve_mbr_results_json_with_method(
    *,
    dataset_experiment_dir: str | Path,
    retriever: str,
    method: str | None = None,
    scope: str = "per_document",
    min_retrieve_k: int | None = None,
) -> tuple[Path, str]:
    """Resolve an MBR retrieval artifact and report the matched method label."""

    root = Path(dataset_experiment_dir).resolve()
    requested = str(method or MBRRetrievalLoader.default_step_b_method_label())
    if requested == MBRRetrievalLoader.default_step_b_method_label():
        method_labels = MBRRetrievalLoader.default_method_candidates()
    else:
        method_labels = [requested]

    searched_dirs = _candidate_report_dirs(root, retriever=retriever)
    for report_dir in searched_dirs:
        resolved = _resolve_mbr_results_json_in_report_dir(
            report_dir=report_dir,
            retriever=retriever,
            method_labels=method_labels,
            scope=scope,
            min_retrieve_k=min_retrieve_k,
        )
        if resolved is not None:
            return resolved

    raise FileNotFoundError(
        "Could not find default MBRB retrieval results for generation. "
        f"Expected retriever={retriever!r}, methods={method_labels!r}, scope={scope!r}. "
        f"Searched report directories: {[str(path) for path in searched_dirs]}. "
        "Pass --mbr_results_json to select an explicit retrieval output."
    )


def load_raw_chunk_items_by_id(llm_cache_index_json: str | Path) -> dict[str, RetrievedContextItem]:
    chunks = load_raw_chunk_records_from_cache_index(llm_cache_index_json)
    out: dict[str, RetrievedContextItem] = {}
    for chunk in chunks:
        out[str(chunk.chunk_id)] = RetrievedContextItem(
            chunk_id=str(chunk.chunk_id),
            rank=0,
            score=None,
            raw_text=str(chunk.raw_text or ""),
            metadata={
                "doc_id": str(chunk.doc_id),
                "chunk_index": int(chunk.chunk_index),
                "token_start": chunk.token_start,
                "token_end": chunk.token_end,
            },
        )
    return out


def build_retrieved_context_items(
    *,
    result: RetrievalResult,
    chunk_items_by_id: dict[str, RetrievedContextItem],
    top_k: int,
) -> list[RetrievedContextItem]:
    items: list[RetrievedContextItem] = []
    for rank, chunk_id in enumerate(list(result.retrieved_chunk_ids)[: int(top_k)], start=1):
        base = chunk_items_by_id.get(str(chunk_id))
        score = None
        if rank - 1 < len(result.scores):
            try:
                score = float(result.scores[rank - 1])
            except Exception:
                score = None
        if base is None:
            items.append(
                RetrievedContextItem(
                    chunk_id=str(chunk_id),
                    rank=rank,
                    score=score,
                    raw_text="",
                    metadata={"missing_chunk_text": True},
                )
            )
            continue
        items.append(
            RetrievedContextItem(
                chunk_id=base.chunk_id,
                rank=rank,
                score=score,
                raw_text=base.raw_text,
                folded_text=base.folded_text,
                metadata=dict(base.metadata),
            )
        )
    return items
