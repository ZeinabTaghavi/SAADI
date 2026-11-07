"""E2 "frozen records": score the memory channel with records as of each chunk's step.

Three variants of the published MBR2 scoring (Equation 2):

- LIVE:      records of a chunk's linked names as they are after the last step
             (exactly the paper's method; must reproduce the saved rankings).
- FROZEN:    records as they were right AFTER the chunk's own step t; aliases and
             descriptions first added at steps > t are removed. Links unchanged.
- NO-MEMORY: lambda_n = 0. Lower reference.

Everything is recomputed from the published construction caches; no LLM calls.
Per-item pool scores are identical across variants (same texts, same retriever);
only the per-chunk field sets change, and every channel z-score is recomputed
per variant over the document's chunks, exactly as the live code path does.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.types import CanonicalizedChunkRecord, RetrievalResult
from ..evaluation.retrieval_metrics import evaluate_retrieval_results
from ..indexing.retrieval_methods.mbr import (
    MBR_VARIANT_MBR2,
    MBRChunkScoreRecord,
    MBRChunkScoreStreams,
    MBRDocumentMemoryIndex,
    _aggregate_memory_values,
    _dedupe_texts,
    _find_doc_entry,
    _load_doc_records,
    _normalize_memory_text_values,
    _read_index_payload,
    _resolve_cache_file_path,
    build_mbr_document_memory_index,
    combine_mbr_variant_chunk_scores,
    score_mbr_variant_document_pools,
)

# Published query-time settings (configs/methods/sadi.yaml, cached_retrieval).
PUBLISHED_SETTINGS: dict[str, Any] = {
    "lambda_t": 1.0,
    "lambda_n": 0.5,
    "lambda_h": 0.3,
    "normalization_method": "zscore",
    "memory_name_weight": 1.0,
    "memory_alias_weight": 1.0,
    "memory_description_weight": 1.0,
    "memory_aggregation": "max",
}
RETRIEVE_K = 10
VARIANTS = ("live", "frozen", "no_memory")

_METHOD_FILE_LABEL = "mbr_lt1_ln0.5_lh0.3_norm_zscore_mnw1_maw1_mdw1"
_REPORT_DIR_LABEL = "mbr_B_lt1.0_lm0.5_lh0.3_mn1.0_ma1.0_md1.0_norm_zscore"


@dataclass(frozen=True)
class DatasetSpec:
    key: str  # registry key used on the command line
    metrics_name: str  # dataset name inside outputs/saadi/reference/expected_metrics.json
    namespace: str  # cache namespace dir holding index.json and the per-doc dirs

    @property
    def index_json(self) -> str:
        return f"{self.namespace}/index.json"

    def saved_results_json(self, retriever: str) -> str:
        return (
            f"{self.namespace}/mbrB_ablation/{_REPORT_DIR_LABEL}/{retriever}/retrieval_reports/"
            f"retrieval_results__{retriever}__{_METHOD_FILE_LABEL}__per_document.json"
        )


_NS = "llm_cache/{ds}/hierarchical/branching_factor_5/chunk_{cs}/AE_5_2_3/Qwen3-30B-A3B-Instruct-2507/AEWord"

DATASETS: dict[str, DatasetSpec] = {
    "loogle": DatasetSpec(
        "loogle",
        "loogle",
        "outputs/loogle_ablation/500_tokens/" + _NS.format(ds="loogle", cs="500"),
    ),
    "musique_32k": DatasetSpec(
        "musique_32k",
        "musique",
        "outputs/musique_ablation/32K_expansion/100_tokens/" + _NS.format(ds="musique", cs="100"),
    ),
    "qasper_64k": DatasetSpec(
        "qasper_64k",
        "qasper",
        "outputs/qasper_ablation/64K_expansion/100_tokens/" + _NS.format(ds="qasper", cs="100"),
    ),
    "novelhopqa": DatasetSpec(
        "novelhopqa",
        "novelhopqa",
        "outputs/novelhopqa_ablation/500_tokens/" + _NS.format(ds="novelhopqa", cs="500"),
    ),
}

# The reference bundle lives at outputs/saadi/ (renamed from the historical
# outputs/sadi/); accept both spellings, current one first.
_REFERENCE_DIRS = ("outputs/saadi/reference", "outputs/sadi/reference")


def reference_dir(repo_root: Path | str = ".") -> Path:
    root = Path(repo_root)
    for candidate in _REFERENCE_DIRS:
        path = root / candidate
        if (path / "expected_metrics.json").exists():
            return path
    raise FileNotFoundError(f"expected_metrics.json not found under any of {_REFERENCE_DIRS} (root: {root})")


class ReplayCheckError(RuntimeError):
    """A saved cache does not support an exact per-step replay. Stop; do not guess."""


# ---------------------------------------------------------------------------
# Replay: when was each alias / description first added?
# ---------------------------------------------------------------------------


def _key(name: str) -> str:
    return str(name or "").strip().lower()


@dataclass
class ReplayedDocument:
    doc_id: str
    chunk_step: dict[str, int] = field(default_factory=dict)  # chunk_id -> step t
    steps: list[int] = field(default_factory=list)  # processed steps, ascending
    name_first: dict[str, int] = field(default_factory=dict)  # name key -> first step
    alias_first: dict[tuple[str, str], int] = field(default_factory=dict)
    desc_first: dict[tuple[str, str], int] = field(default_factory=dict)
    final_names: dict[str, str] = field(default_factory=dict)  # name key -> first casing
    final_aliases: dict[str, list[str]] = field(default_factory=dict)
    final_descriptions: dict[str, list[str]] = field(default_factory=dict)

    @property
    def last_step(self) -> int:
        return self.steps[-1]


def replay_first_added(doc_id: str, records: list[CanonicalizedChunkRecord]) -> ReplayedDocument:
    """Replay the per-step active records in order and date every field.

    The step rows store each step's recalled records AFTER that step's update,
    and the live loader rebuilds the full memory with merge_active_entities
    (name-keyed, first-seen order, no caps). This replay mirrors that merge and
    additionally records the step at which every name, alias and description
    was first added.
    """

    out = ReplayedDocument(doc_id=doc_id)
    final_aliases: dict[str, list[str]] = defaultdict(list)
    final_descriptions: dict[str, list[str]] = defaultdict(list)
    for rec in sorted(records, key=lambda r: (int(r.chunk_index), str(r.chunk_id))):
        t = int(rec.chunk_index)
        out.chunk_step[str(rec.chunk_id)] = t
        out.steps.append(t)
        active = (rec.state_snapshot or {}).get("active_entities", []) or []
        for item in active:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "") or "").strip()
            if not name:
                continue
            key = _key(name)
            if key not in out.name_first:
                out.name_first[key] = t
                out.final_names[key] = name
            for alias in _normalize_memory_text_values(item.get("aliases", [])):
                if (key, alias) not in out.alias_first:
                    out.alias_first[(key, alias)] = t
                    final_aliases[key].append(alias)
            raw_desc = item.get("description", item.get("descriptions", []))
            for desc in _normalize_memory_text_values(raw_desc):
                if (key, desc) not in out.desc_first:
                    out.desc_first[(key, desc)] = t
                    final_descriptions[key].append(desc)
    out.final_aliases = dict(final_aliases)
    out.final_descriptions = dict(final_descriptions)
    if not out.steps:
        raise ReplayCheckError(f"{doc_id}: no step records could be replayed")
    return out


def verify_replay_matches_final_memory(replay: ReplayedDocument, memory_entities: list[dict[str, Any]]) -> None:
    """Replaying to step N must give exactly the final saved memory (memory.json)."""

    problems: list[str] = []
    final_keys = set(replay.final_names)
    saved_keys = set()
    for entity in memory_entities:
        name = str(entity.get("name", "") or "").strip()
        key = _key(name)
        saved_keys.add(key)
        if key not in final_keys:
            problems.append(f"entity only in memory.json: {name!r}")
            continue
        if replay.final_names[key] != name:
            problems.append(f"name casing differs: replay {replay.final_names[key]!r} vs saved {name!r}")
        saved_aliases = _normalize_memory_text_values(entity.get("aliases", []))
        saved_descs = _normalize_memory_text_values(entity.get("description", entity.get("descriptions", [])))
        if replay.final_aliases.get(key, []) != saved_aliases:
            problems.append(f"aliases differ for {name!r}")
        if replay.final_descriptions.get(key, []) != saved_descs:
            problems.append(f"descriptions differ for {name!r}")
    for key in final_keys - saved_keys:
        problems.append(f"entity only in replay: {replay.final_names[key]!r}")
    if problems:
        head = "; ".join(problems[:3])
        raise ReplayCheckError(f"{replay.doc_id}: replayed memory != memory.json ({len(problems)} differences; first: {head})")


# ---------------------------------------------------------------------------
# Frozen per-chunk field sets
# ---------------------------------------------------------------------------


@dataclass
class LateField:
    entity_name: str
    kind: str  # "alias" | "description"
    value: str
    added_at_step: int


@dataclass
class FrozenChunk:
    chunk_id: str
    step: int
    linked_names: list[str]
    live_aliases: list[str]
    live_descriptions: list[str]
    frozen_aliases: list[str]
    frozen_descriptions: list[str]
    late_fields: list[LateField]

    @property
    def has_later_update(self) -> bool:
        return bool(self.late_fields)


def frozen_chunk_fields(
    document_index: MBRDocumentMemoryIndex,
    replay: ReplayedDocument,
) -> dict[str, FrozenChunk]:
    """For every chunk, split its live field sets into "by step t" and "later".

    Field sets come from the exact live structures (final-memory records of the
    chunk's linked names); the frozen subset keeps a field only when the replay
    dates it at a step <= the chunk's own step. Checks required by the plan:
    frozen is a subset of live, and the last chunk's frozen set equals live.
    """

    out: dict[str, FrozenChunk] = {}
    for chunk in document_index.chunk_pool:
        chunk_id = str(chunk.chunk_id)
        t = int(chunk.step) if chunk.step is not None else replay.chunk_step.get(chunk_id, -1)
        linked_names = list(document_index.chunk_to_names.get(chunk_id, []) or [])
        frozen_aliases: list[str] = []
        frozen_descriptions: list[str] = []
        late: list[LateField] = []
        for name in linked_names:
            key = _key(name)
            for alias in document_index.entity_to_aliases.get(name, []):
                first = replay.alias_first.get((key, alias))
                if first is None:
                    raise ReplayCheckError(f"{replay.doc_id}: live alias {alias!r} of {name!r} never appears in the step records")
                if first <= t:
                    frozen_aliases.append(alias)
                else:
                    late.append(LateField(name, "alias", alias, first))
            for desc in document_index.entity_to_descriptions.get(name, []):
                first = replay.desc_first.get((key, desc))
                if first is None:
                    raise ReplayCheckError(f"{replay.doc_id}: live description of {name!r} never appears in the step records")
                if first <= t:
                    frozen_descriptions.append(desc)
                else:
                    late.append(LateField(name, "description", desc, first))
        live_aliases = _dedupe_texts([a for name in linked_names for a in document_index.entity_to_aliases.get(name, [])])
        live_descriptions = _dedupe_texts([d for name in linked_names for d in document_index.entity_to_descriptions.get(name, [])])
        row = FrozenChunk(
            chunk_id=chunk_id,
            step=t,
            linked_names=linked_names,
            live_aliases=live_aliases,
            live_descriptions=live_descriptions,
            frozen_aliases=_dedupe_texts(frozen_aliases),
            frozen_descriptions=_dedupe_texts(frozen_descriptions),
            late_fields=late,
        )
        if not set(row.frozen_aliases) <= set(row.live_aliases):
            raise ReplayCheckError(f"{replay.doc_id}/{chunk_id}: frozen aliases not a subset of live")
        if not set(row.frozen_descriptions) <= set(row.live_descriptions):
            raise ReplayCheckError(f"{replay.doc_id}/{chunk_id}: frozen descriptions not a subset of live")
        if t == replay.last_step and (
            row.frozen_aliases != row.live_aliases or row.frozen_descriptions != row.live_descriptions or row.late_fields
        ):
            raise ReplayCheckError(f"{replay.doc_id}/{chunk_id}: last chunk frozen != live")
        out[chunk_id] = row
    return out


# ---------------------------------------------------------------------------
# Variant scoring (one pool-scoring pass per query; three fusions)
# ---------------------------------------------------------------------------


def _frozen_streams(
    streams: MBRChunkScoreStreams,
    frozen: dict[str, FrozenChunk],
    *,
    memory_aggregation: str,
) -> MBRChunkScoreStreams:
    """Clone the live streams, restricting each chunk's alias/description fields.

    Per-item pool scores stay identical; only the per-chunk max-aggregation and
    the raw memory blend are rebuilt over the frozen subsets. The downstream
    combine step recomputes every z-score over the frozen channel populations.
    """

    new_chunk_scores: dict[str, MBRChunkScoreRecord] = {}
    for chunk_id, rec in streams.chunk_scores.items():
        row = frozen[chunk_id]
        alias_vals = [float(streams.alias_scores[a]) for a in row.frozen_aliases if a in streams.alias_scores]
        desc_vals = [float(streams.description_scores[d]) for d in row.frozen_descriptions if d in streams.description_scores]
        alias_score = _aggregate_memory_values(alias_vals, aggregation=memory_aggregation)
        desc_score = _aggregate_memory_values(desc_vals, aggregation=memory_aggregation)
        available = [s for s in (rec.name_score, alias_score, desc_score) if s is not None]
        memory_score = (sum(available) / len(available)) if available else None
        new_chunk_scores[chunk_id] = MBRChunkScoreRecord(
            text_score=rec.text_score,
            linear_memory_score=None,
            name_score=rec.name_score,
            alias_score=alias_score,
            description_score=desc_score,
            memory_score=memory_score,
            summary_score=rec.summary_score,
            memory_effective_weights={},
        )
    return MBRChunkScoreStreams(
        doc_id=streams.doc_id,
        query=streams.query,
        retriever_name=streams.retriever_name,
        text_scores=dict(streams.text_scores),
        linear_memory_scores={},
        name_scores=dict(streams.name_scores),
        alias_scores=dict(streams.alias_scores),
        description_scores=dict(streams.description_scores),
        summary_scores=dict(streams.summary_scores),
        chunk_scores=new_chunk_scores,
    )


def score_query_variants(
    document_index: MBRDocumentMemoryIndex,
    frozen: dict[str, FrozenChunk],
    *,
    query: str,
    retriever_name: str,
    retrieve_k: int = RETRIEVE_K,
    settings: dict[str, Any] | None = None,
) -> dict[str, list[str]]:
    """Return the top-k chunk ids of the live, frozen and no-memory variants."""

    cfg = dict(PUBLISHED_SETTINGS if settings is None else settings)
    streams = score_mbr_variant_document_pools(
        document_index,
        query=query,
        retriever_name=retriever_name,
        variant=MBR_VARIANT_MBR2,
        memory_name_weight=cfg["memory_name_weight"],
        memory_alias_weight=cfg["memory_alias_weight"],
        memory_description_weight=cfg["memory_description_weight"],
        memory_aggregation=cfg["memory_aggregation"],
    )
    variant_streams = {
        "live": streams,
        "frozen": _frozen_streams(streams, frozen, memory_aggregation=cfg["memory_aggregation"]),
        "no_memory": streams,
    }
    out: dict[str, list[str]] = {}
    for variant, vstreams in variant_streams.items():
        lambda_n = 0.0 if variant == "no_memory" else cfg["lambda_n"]
        result = combine_mbr_variant_chunk_scores(
            document_index,
            vstreams,
            variant=MBR_VARIANT_MBR2,
            lambda_t=cfg["lambda_t"],
            lambda_n=lambda_n,
            lambda_h=cfg["lambda_h"],
            normalization_method=cfg["normalization_method"],
            memory_name_weight=cfg["memory_name_weight"],
            memory_alias_weight=cfg["memory_alias_weight"],
            memory_description_weight=cfg["memory_description_weight"],
        )
        out[variant] = [row.chunk_id for row in result.ranked_chunks[: int(retrieve_k)]]
    return out


# ---------------------------------------------------------------------------
# Dataset driver
# ---------------------------------------------------------------------------


@dataclass
class QueryOutcome:
    query_id: str
    doc_id: str
    question: str
    gold_chunk_ids: list[str]
    evidence_groups: list[list[str]]
    rankings: dict[str, list[str]]  # variant -> top-k chunk ids
    live_matches_saved: bool
    affected: bool
    evidence_third: int | None  # 0 / 1 / 2, position of earliest evidence
    late_evidence_fields: list[dict[str, Any]]


@dataclass
class ChunkOutcome:
    doc_id: str
    chunk_id: str
    step: int
    n_linked_names: int
    n_live_aliases: int
    n_live_descriptions: int
    n_late_aliases: int
    n_late_descriptions: int


@dataclass
class DatasetRun:
    dataset: str
    retriever: str
    queries: list[QueryOutcome]
    chunks: list[ChunkOutcome]
    n_docs: int

    def results_for(self, variant: str) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                query_id=q.query_id,
                doc_id=q.doc_id,
                question=q.question,
                retrieved_chunk_ids=list(q.rankings[variant]),
                retrieved_indices=[],
                scores=[],
                gold_chunk_ids=list(q.gold_chunk_ids),
                silver_chunk_ids=sorted({c for g in q.evidence_groups for c in g}),
                silver_chunk_groups=[list(g) for g in q.evidence_groups],
            )
            for q in self.queries
        ]

    def evaluate(self, variant: str, *, eval_k: int = 5):
        return evaluate_retrieval_results(self.results_for(variant), eval_k=eval_k, retrieve_k=RETRIEVE_K, compute_per_query=True)


def load_saved_live_rows(spec: DatasetSpec, retriever: str, *, repo_root: Path) -> list[dict[str, Any]]:
    path = repo_root / spec.saved_results_json(retriever)
    if not path.exists():
        raise FileNotFoundError(f"Saved live rankings not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Saved live rankings are empty or not a list: {path}")
    return rows


def _evidence_chunks(row: dict[str, Any]) -> list[str]:
    gold = [str(x) for x in row.get("gold_chunk_ids", []) or []]
    groups = row.get("silver_chunk_groups", []) or []
    silver = [str(c) for group in groups if isinstance(group, list) for c in group]
    return _dedupe_texts([*gold, *silver])


def run_dataset(
    spec: DatasetSpec,
    *,
    retriever: str = "bm25",
    repo_root: Path | str = ".",
    max_docs: int | None = None,
    progress: bool = True,
) -> DatasetRun:
    """Score every saved query of one dataset in the three variants."""

    repo_root = Path(repo_root)
    index_path = repo_root / spec.index_json
    payload = _read_index_payload(index_path)
    saved_rows = load_saved_live_rows(spec, retriever, repo_root=repo_root)

    rows_by_doc: dict[str, list[dict[str, Any]]] = {}
    for row in saved_rows:
        rows_by_doc.setdefault(str(row.get("doc_id", "")), []).append(row)
    doc_ids = list(rows_by_doc)
    if max_docs is not None:
        doc_ids = doc_ids[: int(max_docs)]

    iterator: Iterable[str] = doc_ids
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(doc_ids, desc=f"{spec.key}/{retriever}", unit="doc")
        except Exception:
            iterator = doc_ids

    query_outcomes: list[QueryOutcome] = []
    chunk_outcomes: list[ChunkOutcome] = []
    for doc_id in iterator:
        doc_entry = _find_doc_entry(payload, doc_id=doc_id)
        if doc_entry is None:
            raise KeyError(f"Document {doc_id!r} not found in cache index {index_path}")
        records = _load_doc_records(index_path=index_path, index_payload=payload, doc_entry=doc_entry, doc_id=doc_id)
        if not records:
            raise ReplayCheckError(f"{doc_id}: no records restored from the cache")
        replay = replay_first_added(doc_id, records)
        memory_ref = (doc_entry.get("files", {}) or {}).get("memory")
        if not memory_ref:
            raise ReplayCheckError(f"{doc_id}: cache index has no memory.json reference")
        memory_path = _resolve_cache_file_path(memory_ref, base_dir=index_path.parent)
        with Path(memory_path).open("r", encoding="utf-8") as handle:
            memory_entities = json.load(handle).get("entities", [])
        verify_replay_matches_final_memory(replay, memory_entities)

        document_index = build_mbr_document_memory_index(index_path, doc_id=doc_id)
        frozen = frozen_chunk_fields(document_index, replay)

        for row in frozen.values():
            chunk_outcomes.append(
                ChunkOutcome(
                    doc_id=doc_id,
                    chunk_id=row.chunk_id,
                    step=row.step,
                    n_linked_names=len(row.linked_names),
                    n_live_aliases=len(row.live_aliases),
                    n_live_descriptions=len(row.live_descriptions),
                    n_late_aliases=sum(1 for f in row.late_fields if f.kind == "alias"),
                    n_late_descriptions=sum(1 for f in row.late_fields if f.kind == "description"),
                )
            )

        steps_sorted = sorted(replay.chunk_step.values())
        step_rank = {step: i for i, step in enumerate(steps_sorted)}
        n_steps = len(steps_sorted)

        for row in rows_by_doc[doc_id]:
            question = str(row.get("question", "") or "")
            rankings = score_query_variants(document_index, frozen, query=question, retriever_name=retriever)
            evidence = _evidence_chunks(row)
            evidence_steps = [replay.chunk_step[c] for c in evidence if c in replay.chunk_step]
            third: int | None = None
            if evidence_steps and n_steps:
                rank = step_rank[min(evidence_steps)]
                third = min(2, (3 * rank) // n_steps)
            affected = any(frozen[c].has_later_update for c in evidence if c in frozen)
            late_evidence_fields = [
                {
                    "evidence_chunk_id": c,
                    "evidence_chunk_step": frozen[c].step,
                    "entity_name": f.entity_name,
                    "kind": f.kind,
                    "value": f.value,
                    "added_at_step": f.added_at_step,
                }
                for c in evidence
                if c in frozen
                for f in frozen[c].late_fields
            ]
            query_outcomes.append(
                QueryOutcome(
                    query_id=str(row.get("query_id", "")),
                    doc_id=doc_id,
                    question=question,
                    gold_chunk_ids=[str(x) for x in row.get("gold_chunk_ids", []) or []],
                    evidence_groups=[
                        [str(c) for c in group] for group in (row.get("silver_chunk_groups", []) or []) if isinstance(group, list)
                    ],
                    rankings=rankings,
                    live_matches_saved=rankings["live"] == [str(x) for x in row.get("retrieved_chunk_ids", [])],
                    affected=affected,
                    evidence_third=third,
                    late_evidence_fields=late_evidence_fields,
                )
            )

    return DatasetRun(
        dataset=spec.key,
        retriever=retriever,
        queries=query_outcomes,
        chunks=chunk_outcomes,
        n_docs=len(doc_ids),
    )


# ---------------------------------------------------------------------------
# Reproduction gate
# ---------------------------------------------------------------------------


def check_reproduction(
    run: DatasetRun,
    spec: DatasetSpec,
    *,
    repo_root: Path | str = ".",
    full_dataset: bool = True,
) -> dict[str, Any]:
    """LIVE must reproduce the saved HR@5/HR@10 before anything else is reported."""

    report = run.evaluate("live", eval_k=5)
    summary: dict[str, Any] = {
        "dataset": run.dataset,
        "retriever": run.retriever,
        "n_queries": report.n_examples,
        "n_eligible": report.n_union_eligible,
        "hits": {k: report.hits_at_k.get(k) for k in ("5", "10")},
        "hr": {k: report.hit_rate_at_k.get(k) for k in ("5", "10")},
        "rankings_equal_saved": sum(1 for q in run.queries if q.live_matches_saved),
        "rankings_total": len(run.queries),
        "expected": {},
        "matches_expected": None,
    }
    if not full_dataset:
        return summary

    expected_path = reference_dir(repo_root) / "expected_metrics.json"
    with expected_path.open("r", encoding="utf-8") as handle:
        expected_rows = json.load(handle)
    # expected_metrics.json holds one row per (study, condition); the published
    # Table-1 numbers are study="main", condition="default".
    expected = {
        int(row["cutoff"]): row
        for row in expected_rows
        if row.get("dataset") == spec.metrics_name
        and row.get("retriever") == run.retriever
        and row.get("study") == "main"
        and row.get("condition") == "default"
    }
    if not expected:
        raise ReplayCheckError(
            f"No main/default expected metrics for dataset={spec.metrics_name!r} retriever={run.retriever!r} in {expected_path}"
        )

    # Cross-check: the manifest's main/default source must be exactly the saved
    # per-document rankings this run reproduces (guards the namespace choice,
    # e.g. MuSiQue 32K and QASPER 64K as the paper's main settings).
    manifest_path = expected_path.parent / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    source = next(
        (
            s
            for s in manifest.get("sources", [])
            if s.get("study") == "main"
            and s.get("condition") == "default"
            and s.get("dataset") == spec.metrics_name
            and s.get("retriever") == run.retriever
        ),
        None,
    )
    if source is None:
        raise ReplayCheckError(f"manifest.json has no main/default source for {spec.metrics_name}/{run.retriever}")
    if Path(str(source.get("original_source", ""))).as_posix() != Path(spec.saved_results_json(run.retriever)).as_posix():
        raise ReplayCheckError(
            f"{spec.key}/{run.retriever}: manifest main/default source is "
            f"{source.get('original_source')!r}, but this run reproduces "
            f"{spec.saved_results_json(run.retriever)!r}; fix the dataset registry"
        )
    matches = True
    for cutoff in (5, 10):
        row = expected.get(cutoff)
        if row is None:
            continue
        summary["expected"][str(cutoff)] = {
            "eligible": row["eligible"],
            "hits": row["hits"],
            "hr": row["hr"],
        }
        if int(report.n_union_eligible) != int(row["eligible"]) or int(report.hits_at_k.get(str(cutoff)) or -1) != int(row["hits"]):
            matches = False
    summary["matches_expected"] = matches
    return summary


# ---------------------------------------------------------------------------
# Paired bootstrap
# ---------------------------------------------------------------------------


def paired_bootstrap_ci(
    diffs: Iterable[float],
    *,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Mean paired difference with a 95% bootstrap CI over queries (seed 42)."""

    import numpy as np

    arr = np.asarray(list(diffs), dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(int(n_resamples), arr.size))
    means = arr[idx].mean(axis=1)
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
