"""Builds the retrieval corpus, runs the retriever, and computes retrieval metrics."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from ..core.types import CanonicalizedChunkRecord, ChunkRecord, CorpusArtifact, RetrievalExample, RetrievalResult
from ..evaluation.retrieval_metrics import evaluate_retrieval_results, write_eval_report_json
from ..indexing.retrieval_modes import (
    build_corpus_by_doc_for_mode,
    build_corpus_for_mode,
    resolve_retrieval_mode,
)
from ..indexing.retriever_adapters import get_retriever
from ..io.artifact_paths import default_artifact_paths
from ..io.manifests import GIT_PROVENANCE_FIELDS, git_provenance

logger = logging.getLogger(__name__)


def _safe_output_component(value: str, *, default: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "")).strip("._")
    return out or default


def build_retrieval_corpus(
    *,
    raw_chunks: list[ChunkRecord] | None = None,
    canonicalized_chunks: list[CanonicalizedChunkRecord] | None = None,
    retrieval_text_mode: str = "auto",
) -> tuple[CorpusArtifact, str]:
    mode = resolve_retrieval_mode(
        requested_mode=retrieval_text_mode,
        raw_chunks=raw_chunks,
        canonicalized_chunks=canonicalized_chunks,
    )
    if mode == "l0":
        logger.info(
            "Building retrieval corpus mode=%s from canonicalized chunks count=%d",
            mode,
            len(canonicalized_chunks or []),
        )
    else:
        logger.info(
            "Building retrieval corpus mode=%s from raw chunks count=%d",
            mode,
            len(raw_chunks or []),
        )
    corpus = build_corpus_for_mode(
        mode=mode,
        raw_chunks=raw_chunks,
        canonicalized_chunks=canonicalized_chunks,
    )
    return corpus, mode


def run_retrieval_pipeline(
    *,
    examples: list[RetrievalExample],
    corpus: CorpusArtifact,
    corpus_by_doc: dict[str, CorpusArtifact] | None = None,
    retriever_name: str = "bm25",
    retrieve_k: int = 10,
    retrieval_scope: str = "global",
) -> list[RetrievalResult]:
    if retrieval_scope not in {"global", "per_document"}:
        raise ValueError(
            "Invalid retrieval_scope. Expected one of: global, per_document "
            f"(got: {retrieval_scope!r})"
        )
    logger.info(
        "Running retriever=%s for queries=%d retrieve_k=%d corpus_chunks=%d scope=%s",
        retriever_name,
        len(examples),
        retrieve_k,
        len(corpus.chunk_ids),
        retrieval_scope,
    )
    Retriever = get_retriever(retriever_name)
    results: list[RetrievalResult] = []
    if retrieval_scope == "per_document":
        corpora = corpus_by_doc or {}
        retriever_by_doc: dict[str, object] = {}
        for ex in examples:
            doc_corpus = corpora.get(str(ex.doc_id))
            if doc_corpus is None or not doc_corpus.chunk_ids:
                retrieved_ids: list[str] = []
                scores: list[float] = []
                indices: list[int] = []
            else:
                retriever = retriever_by_doc.get(str(ex.doc_id))
                if retriever is None:
                    retriever = Retriever(corpus=doc_corpus.chunk_texts, chunk_ids=doc_corpus.chunk_ids, k=retrieve_k)
                    retriever_by_doc[str(ex.doc_id)] = retriever
                retrieved_ids, scores, indices = retriever.retrieve(ex.question, k=retrieve_k)
            results.append(
                RetrievalResult(
                    query_id=ex.query_id,
                    doc_id=ex.doc_id,
                    question=ex.question,
                    retrieved_chunk_ids=retrieved_ids,
                    retrieved_indices=indices,
                    scores=scores,
                    gold_chunk_ids=list(ex.gold_chunk_ids),
                    silver_chunk_ids=list(ex.silver_chunk_ids),
                    silver_chunk_groups=[list(group) for group in ex.silver_chunk_groups],
                )
            )
    else:
        retriever = Retriever(corpus=corpus.chunk_texts, chunk_ids=corpus.chunk_ids, k=retrieve_k)
        for ex in examples:
            retrieved_ids, scores, indices = retriever.retrieve(ex.question, k=retrieve_k)
            results.append(
                RetrievalResult(
                    query_id=ex.query_id,
                    doc_id=ex.doc_id,
                    question=ex.question,
                    retrieved_chunk_ids=retrieved_ids,
                    retrieved_indices=indices,
                    scores=scores,
                    gold_chunk_ids=list(ex.gold_chunk_ids),
                    silver_chunk_ids=list(ex.silver_chunk_ids),
                    silver_chunk_groups=[list(group) for group in ex.silver_chunk_groups],
                )
            )
    logger.info("Retriever completed produced_results=%d scope=%s", len(results), retrieval_scope)
    return results


def write_retrieval_results_json(
    path: str | Path,
    results: list[RetrievalResult],
    *,
    git_commit_hash: str | None = None,
    git_provenance_metadata: dict[str, object] | None = None,
) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    provenance = dict(
        git_provenance_metadata
        or git_provenance(cwd=str(Path(__file__).resolve().parents[3]))
    )
    if git_commit_hash is not None:
        provenance["git_commit"] = git_commit_hash
    rows = []
    for result in results:
        row = result.to_dict()
        for field in GIT_PROVENANCE_FIELDS:
            row[field] = provenance.get(field)
        rows.append(row)
    with p.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    return str(p)


def run_retrieval_eval(
    *,
    examples: list[RetrievalExample],
    raw_chunks: list[ChunkRecord] | None = None,
    canonicalized_chunks: list[CanonicalizedChunkRecord] | None = None,
    retrieval_text_mode: str = "auto",
    retriever_name: str = "bm25",
    retrieve_k: int = 10,
    eval_k: int = 5,
    retrieval_scope: str = "global",
    output_dir: str | None = None,
    provenance: dict[str, object] | None = None,
):
    retrieval_provenance = dict(
        provenance or git_provenance(cwd=str(Path(__file__).resolve().parents[3]))
    )
    if retrieval_scope not in {"global", "per_document"}:
        raise ValueError(
            "Invalid retrieval_scope. Expected one of: global, per_document "
            f"(got: {retrieval_scope!r})"
        )
    corpus, resolved_mode = build_retrieval_corpus(
        raw_chunks=raw_chunks,
        canonicalized_chunks=canonicalized_chunks,
        retrieval_text_mode=retrieval_text_mode,
    )
    corpus_by_doc: dict[str, CorpusArtifact] | None = None
    if retrieval_scope == "per_document":
        corpus_by_doc = build_corpus_by_doc_for_mode(
            mode=resolved_mode,
            raw_chunks=raw_chunks,
            canonicalized_chunks=canonicalized_chunks,
        )
    results = run_retrieval_pipeline(
        examples=examples,
        corpus=corpus,
        corpus_by_doc=corpus_by_doc,
        retriever_name=retriever_name,
        retrieve_k=retrieve_k,
        retrieval_scope=retrieval_scope,
    )
    report = evaluate_retrieval_results(
        results,
        eval_k=eval_k,
        retrieve_k=retrieve_k,
        metadata={
            **retrieval_provenance,
            "retriever": str(retriever_name),
            "retrieval_text_mode": str(resolved_mode),
            "retrieval_scope": str(retrieval_scope),
            "retrieve_k": int(retrieve_k),
            "eval_k": eval_k,
        },
    )
    logger.info(
        "Retrieval evaluation done mode=%s examples=%d strict_union_hr=%s eligible=%d scope=%s",
        resolved_mode,
        report.n_examples,
        report.hit_rate_at_k,
        report.n_union_eligible,
        retrieval_scope,
    )

    artifacts = None
    if output_dir:
        paths = default_artifact_paths(output_dir)
        reports_dir = Path(paths.root) / "retrieval_reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        retriever_label = _safe_output_component(str(retriever_name), default="retriever")
        mode_label = _safe_output_component(str(resolved_mode), default="mode")
        scope_label = _safe_output_component(str(retrieval_scope), default="scope")
        paths.retrieval_results_json = reports_dir / f"retrieval_results__{retriever_label}__{mode_label}__{scope_label}.json"
        paths.eval_report_json = reports_dir / f"eval_report__{retriever_label}__{mode_label}__{scope_label}.json"
        write_retrieval_results_json(
            paths.retrieval_results_json,
            results,
            git_provenance_metadata=retrieval_provenance,
        )
        write_eval_report_json(paths.eval_report_json, report)
        artifacts = paths
        logger.info("Retrieval artifacts written results=%s eval=%s", paths.retrieval_results_json, paths.eval_report_json)
    return results, report, corpus, artifacts
