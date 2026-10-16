"""Single-document SAADI demo pipeline.

This module keeps the notebook and generated script thin: each visible demo
section calls one explicit function here, while the implementation reuses the
existing ingest, retrieval, folded-context, and generation primitives.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..core.parsers import extract_tagged_section
from ..core.types import CanonicalizedChunkRecord, ChunkRecord, RetrievalExample, RetrievalResult
from ..entities import active_entity_config_label, active_entity_selector_label
from ..indexing.retrieval_methods.mbr import (
    DEFAULT_MBR_MEMORY_AGGREGATION,
    MBR_VARIANT_MBR2,
    build_mbr_document_memory_index,
    run_mbr_variant_retrieval_pipeline,
)
from ..io.artifact_paths import ArtifactPaths
from ..io.jsonl import write_jsonl
from ..io.manifests import build_reproducibility_metadata, git_provenance
from ..llm.base import GenerateParams, describe_llm_provenance, render_prompt
from ..llm.factory import build_backend_from_config, resolve_backend
from ..saadi_generation.context_builders.base import ContextBuilderConfig, QueryContextRequest, RetrievedContextItem
from ..saadi_generation.context_builders.chunks_only import ChunksOnlyContextBuilder
from ..saadi_generation.context_builders.folded_context import FoldedContextBuilder, load_summary_blocks_by_doc
from .generation_helpers import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_TEMPLATE, _trim_context, build_long_context_prompt
from .ingest_pipeline import ingest_documents


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _default_model_config() -> dict[str, Any]:
    return {
        "backend": "vllm",
        "model_id": "Qwen/Qwen3.5-27B",
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "load_chat_tokenizer": True,
        "tensor_parallel_size": 4,
        "gpu_memory_utilization": 0.8,
        "max_model_len": 32768,
        "enforce_eager": False,
        "vllm_kwargs": {
            "disable_custom_all_reduce": True,
            "max_num_batched_tokens": 16384,
            "max_num_seqs": 16,
        },
        "sampling": {
            "seed": 42,
            "top_p": 1.0,
            "top_k": -1,
            "repetition_penalty": 1.0,
        },
    }


def _single_document_mock_generator(prompt: str) -> str:
    rollup_items = extract_tagged_section(prompt, "ROLLUP_ITEMS_JSON")
    if rollup_items:
        try:
            items = json.loads(rollup_items)
        except Exception:
            items = []
        if not isinstance(items, list):
            items = []
        joined = " ".join(str(item).strip() for item in items if str(item).strip())
        words = joined.split()
        summary = " ".join(words[:60]) if words else "summary"
        return summary + "\n[END_SUMMARY]"

    current_chunk = extract_tagged_section(prompt, "CURRENT_CHUNK")
    if current_chunk:
        return json.dumps(
            {
                "active_entities": [],
                "insertions": [],
            },
            ensure_ascii=False,
        ) + "\n[END_JSON]"

    return (
        "Mock answer: the retrieved context highlights the main themes and evidence "
        "that are most relevant to the query. Switch MODEL_CONFIG to vLLM/Qwen for "
        "a real generated answer."
    )


@dataclass(slots=True)
class SingleDocumentPipelineConfig:
    """User-facing configuration for the single-document demo."""

    demo_root: str = "single_document_demo"
    document_path: str = "single_document_demo/input/therapy_sample.txt"
    document_id: str = "therapy_sample"
    run_name: str = "therapy_sample_current_saadi"
    query: str = "What are the main themes discussed in this therapy text?"
    run_mode: str = "auto"

    model_config: dict[str, Any] = field(default_factory=_default_model_config)
    ingest_max_tokens: int = 2048
    ingest_temperature: float = 0.0
    ingest_stop: list[str] = field(default_factory=lambda: ["[END_JSON]", "<|im_end|>", "<|endoftext|>"])
    answer_max_tokens: int = 2048
    answer_temperature: float = 0.6
    answer_stop: list[str] = field(default_factory=lambda: ["[END_JSON]", "<|im_end|>", "<|endoftext|>"])

    chunk_size: int = 500
    chunk_overlap: int = 0
    strategy_name: str = "hierarchical"
    branching_factor: int = 5
    hierarchy_k: int | None = None
    active_entity_selector: str = "bm25"
    max_entities: int = 5
    max_aliases: int = 2
    max_desc: int = 3
    use_cache: bool = True
    cache_save_every_steps: int = 100
    cache_save_each_document: bool = True
    ingest_schedule: str = "wavefront"
    wavefront_batch_size: int = 32

    retriever: str = "bm25"
    retrieval_scope: str = "per_document"
    retrieve_k: int = 10
    mbr_variant: str = MBR_VARIANT_MBR2
    lambda_t: float = 1.0
    lambda_n: float = 0.5
    lambda_h: float = 0.3
    normalization_method: str = "zscore"
    memory_name_weight: float = 1.0
    memory_alias_weight: float = 1.0
    memory_description_weight: float = 1.0
    memory_aggregation: str = DEFAULT_MBR_MEMORY_AGGREGATION

    separator_phrases: list[str] = field(default_factory=list)
    # Compatibility alias for older demo notebooks. Prefer separator_phrases.
    session_delimiter: str = ""
    questions_path: str = ""

    context_method: str = "folded_context_block"
    generation_top_k: int | None = 5
    include_bridge: bool = True
    max_context_tokens: int | None = None
    overflow_policy: str = "fail"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    user_template: str = DEFAULT_USER_TEMPLATE
    prompt_version: str = "saadi_folded_context_block_v1"

    def run_dir(self) -> Path:
        return Path(self.demo_root) / "runs" / self.run_name

    def strategy_kwargs(self) -> dict[str, Any]:
        b = int(self.hierarchy_k if self.hierarchy_k is not None else self.branching_factor)
        return {
            "branching_factor": b,
            "active_entity_selector": str(self.active_entity_selector),
            "max_entities": int(self.max_entities),
            "max_aliases": int(self.max_aliases),
            "max_desc": int(self.max_desc),
        }

    def resolved_separator_phrases(self) -> list[str]:
        """Return de-duplicated literal separators, including the legacy alias."""

        raw_phrases = list(self.separator_phrases or [])
        legacy = str(self.session_delimiter or "").strip()
        if legacy:
            raw_phrases.append(legacy)
        phrases: list[str] = []
        for raw in raw_phrases:
            phrase = str(raw or "").strip()
            if phrase and phrase not in phrases:
                phrases.append(phrase)
        return phrases

    def ingest_generate_params(self) -> GenerateParams:
        return GenerateParams(
            max_tokens=int(self.ingest_max_tokens),
            temperature=float(self.ingest_temperature),
            stop=[str(item) for item in self.ingest_stop if str(item).strip()],
        )

    def answer_generate_params(self) -> GenerateParams:
        return GenerateParams(
            max_tokens=int(self.answer_max_tokens),
            temperature=float(self.answer_temperature),
            stop=[str(item) for item in self.answer_stop if str(item).strip()],
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["run_dir"] = str(self.run_dir())
        return payload


@dataclass(slots=True)
class LoadedDocument:
    documents_by_id: dict[str, str]
    document_path: Path
    document_chars: int
    document_words: int


@dataclass(slots=True)
class IngestStageResult:
    raw_chunks: list[ChunkRecord]
    canonicalized_chunks: list[CanonicalizedChunkRecord]
    artifacts: ArtifactPaths | None
    document_manifest_path: Path
    raw_chunks_path: Path
    canonicalized_chunks_path: Path
    memory_summary_path: Path | None = None


@dataclass(slots=True)
class RetrievalStageResult:
    retrieval_result: RetrievalResult
    resolved_mode: str
    method_label: str
    retrieval_results_path: Path
    retrieval_diagnostics_path: Path
    retrieved_chunks: list[dict[str, Any]]


@dataclass(slots=True)
class ContextStageResult:
    context_text: str
    debug_payload: dict[str, Any]
    context_path: Path
    debug_path: Path


@dataclass(slots=True)
class GenerationStageResult:
    generated_answer: str
    raw_output: str
    prompt: str
    user_prompt: str
    generation_record: dict[str, Any]
    prompt_path: Path
    answer_path: Path
    generation_record_path: Path


@dataclass(slots=True)
class SingleDocumentRunResult:
    loaded_document: LoadedDocument
    ingest: IngestStageResult
    retrieval: RetrievalStageResult
    context: ContextStageResult
    generation: GenerationStageResult
    run_manifest_path: Path


@dataclass(slots=True)
class MemoryOnlyRunResult:
    loaded_document: LoadedDocument
    ingest: IngestStageResult
    run_manifest_path: Path


def _write_json(path: str | Path, payload: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return p


def _write_text(path: str | Path, text: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(text or ""), encoding="utf-8")
    return p


def _write_memory_summary(
    path: Path,
    *,
    canonicalized_chunks: list[CanonicalizedChunkRecord],
    document_id: str,
) -> Path | None:
    """Extract the final entity memory from the last canonicalized chunk and write it as JSON.

    Returns the written path, or None if there are no chunks or no entity memory.
    """
    if not canonicalized_chunks:
        return None
    last_chunk = max(canonicalized_chunks, key=lambda c: c.chunk_index)
    snapshot = last_chunk.state_snapshot or {}
    entity_memory: list[dict[str, Any]] = snapshot.get("entity_memory", [])
    if not entity_memory:
        return None

    # Track first-seen chunk for each entity, alias, and description.
    first_seen_entity: dict[str, int] = {}
    first_seen_aliases: dict[str, dict[str, int]] = {}
    first_seen_descs: dict[str, dict[str, int]] = {}

    for chunk in sorted(canonicalized_chunks, key=lambda c: c.chunk_index):
        idx = chunk.chunk_index
        for ent in (chunk.state_snapshot or {}).get("entity_memory", []):
            name = ent.get("name", "")
            if not name:
                continue
            if name not in first_seen_entity:
                first_seen_entity[name] = idx
            alias_map = first_seen_aliases.setdefault(name, {})
            for alias in ent.get("aliases", []):
                if alias not in alias_map:
                    alias_map[alias] = idx
            desc_map = first_seen_descs.setdefault(name, {})
            for desc in ent.get("description", []):
                if desc not in desc_map:
                    desc_map[desc] = idx

    entities = []
    for ent in entity_memory:
        name = ent.get("name", "")
        alias_map = first_seen_aliases.get(name, {})
        desc_map = first_seen_descs.get(name, {})
        entities.append({
            "name": name,
            "aliases": [
                {"text": a, "first_seen_chunk": alias_map.get(a)}
                for a in ent.get("aliases", [])
            ],
            "descriptions": [
                {"text": d, "first_seen_chunk": desc_map.get(d)}
                for d in ent.get("description", [])
            ],
            "first_seen_chunk": first_seen_entity.get(name),
        })

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "document_id": document_id,
        "total_chunks": len(canonicalized_chunks),
        "total_entities": len(entities),
        "entities": entities,
    }
    return _write_json(path, payload)


def _infer_llm_label(model_config: dict[str, Any]) -> str:
    for key in ("alias", "name", "llm"):
        value = model_config.get(key)
        if value:
            return _safe_label(str(value))
    model_id = str(model_config.get("model_id", "") or "")
    if model_id:
        parts = [part for part in model_id.split("/") if part]
        if parts:
            return _safe_label(parts[-1])
    return _safe_label(str(model_config.get("backend", "unknown_llm") or "unknown_llm"))


def _safe_label(value: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(value or ""))
    return out.strip("._") or "value"


def _qa_entry(config: SingleDocumentPipelineConfig, *, query: str | None = None) -> dict[str, Any]:
    question = str(query if query is not None else config.query).strip()
    return {
        "id": "query_1",
        "document_id": str(config.document_id),
        "question": question,
        "answers": [],
    }


def _chunk_text_lookup(
    *,
    raw_chunks: list[ChunkRecord],
    canonicalized_chunks: list[CanonicalizedChunkRecord],
    retrieval_text_mode: str,
) -> tuple[str, dict[str, dict[str, Any]]]:
    resolved_mode = str(retrieval_text_mode).lower()
    if resolved_mode not in {"auto", "l0", "raw_chunks"}:
        raise ValueError(f"Unsupported chunk text mode: {resolved_mode}")
    if resolved_mode == "auto":
        resolved_mode = "l0" if canonicalized_chunks else "raw_chunks"
    lookup: dict[str, dict[str, Any]] = {}
    if resolved_mode == "l0":
        for chunk in canonicalized_chunks:
            lookup[str(chunk.chunk_id)] = {
                "chunk_id": str(chunk.chunk_id),
                "doc_id": str(chunk.doc_id),
                "chunk_index": int(chunk.chunk_index),
                "text": str(chunk.canonicalized_chunk_text or ""),
                "raw_text": str(chunk.raw_chunk_text or ""),
            }
    else:
        for chunk in raw_chunks:
            lookup[str(chunk.chunk_id)] = {
                "chunk_id": str(chunk.chunk_id),
                "doc_id": str(chunk.doc_id),
                "chunk_index": int(chunk.chunk_index),
                "text": str(chunk.raw_text or ""),
                "raw_text": str(chunk.raw_text or ""),
            }
    return resolved_mode, lookup


def _retrieved_chunk_rows(
    *,
    result: RetrievalResult,
    chunk_lookup: dict[str, dict[str, Any]],
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ids = list(result.retrieved_chunk_ids)
    if top_k is not None:
        ids = ids[: max(1, int(top_k))]
    for rank, chunk_id in enumerate(ids, start=1):
        score = result.scores[rank - 1] if rank - 1 < len(result.scores) else None
        meta = chunk_lookup.get(str(chunk_id), {})
        rows.append(
            {
                "rank": rank,
                "chunk_id": str(chunk_id),
                "doc_id": str(meta.get("doc_id", result.doc_id)),
                "chunk_index": meta.get("chunk_index"),
                "score": score,
                "text": str(meta.get("text", "")),
                "raw_text": str(meta.get("raw_text", "")),
            }
        )
    return rows


def _retrieved_items_from_rows(rows: list[dict[str, Any]]) -> list[RetrievedContextItem]:
    items: list[RetrievedContextItem] = []
    for row in rows:
        items.append(
            RetrievedContextItem(
                chunk_id=str(row.get("chunk_id", "")),
                rank=int(row.get("rank", len(items) + 1)),
                score=(float(row["score"]) if row.get("score") is not None else None),
                raw_text=str(row.get("text", "") or ""),
                metadata={
                    "doc_id": row.get("doc_id"),
                    "chunk_index": row.get("chunk_index"),
                    "retrieval_text_source": "raw",
                },
            )
        )
    return items


def _find_cache_index_path(config: SingleDocumentPipelineConfig, artifacts: ArtifactPaths | None) -> Path | None:
    if artifacts is not None and artifacts.llm_cache_index_json.exists():
        return artifacts.llm_cache_index_json
    root = config.run_dir() / "llm_cache"
    candidates = sorted(root.rglob("index.json")) if root.exists() else []
    return candidates[0] if candidates else None


def split_by_separator_phrases(
    text: str,
    doc_id: str,
    separator_phrases: list[str],
) -> list[ChunkRecord]:
    """Split on any literal separator phrase and omit the separators from chunks."""

    phrases: list[str] = []
    for raw in separator_phrases:
        phrase = str(raw or "").strip()
        if phrase and phrase not in phrases:
            phrases.append(phrase)
    if not phrases:
        raise ValueError("separator_phrases must contain at least one non-empty phrase")

    # Longest first prevents a shorter prefix phrase from consuming a longer one.
    pattern = re.compile("|".join(re.escape(phrase) for phrase in sorted(phrases, key=len, reverse=True)))
    records: list[ChunkRecord] = []
    word_offset = 0
    cursor = 0

    def append_part(raw_part: str) -> None:
        nonlocal word_offset
        stripped = raw_part.strip()
        if not stripped:
            word_offset += len(raw_part.split())
            return
        idx = len(records)
        word_count = len(stripped.split())
        records.append(
            ChunkRecord(
                doc_id=doc_id,
                chunk_id=f"{doc_id}:{idx}",
                chunk_index=idx,
                raw_text=stripped,
                token_start=word_offset,
                token_end=word_offset + word_count,
            )
        )
        word_offset += word_count

    for match in pattern.finditer(text):
        append_part(text[cursor : match.start()])
        word_offset += len(match.group(0).split())
        cursor = match.end()
    append_part(text[cursor:])
    return records


def _split_by_session_delimiter(text: str, doc_id: str, delimiter: str) -> list[ChunkRecord]:
    """Backward-compatible wrapper around :func:`split_by_separator_phrases`."""

    return split_by_separator_phrases(text, doc_id, [delimiter])


def load_questions_jsonl(path: str | Path) -> list[str]:
    """Read JSONL file and return the value of the 'question' key from each line."""
    questions: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                q = str(obj.get("question", "") or "").strip()
                if q:
                    questions.append(q)
            except json.JSONDecodeError:
                continue
    return questions


def build_backend(config: SingleDocumentPipelineConfig):
    """Build the configured LLM backend once for ingest and answer generation."""

    backend_name = str(config.model_config.get("backend", "mock") or "mock").strip().lower()
    if backend_name in {"mock", "dummy"}:
        from ..llm.mock_backend import MockBackend

        return MockBackend(generator=_single_document_mock_generator)
    return build_backend_from_config(config.model_config)


def load_single_document(config: SingleDocumentPipelineConfig) -> LoadedDocument:
    """Read the configured `.txt` document and return the single-doc mapping."""

    document_path = Path(config.document_path).expanduser()
    if not document_path.exists():
        raise FileNotFoundError(f"Single-document input file does not exist: {document_path}")
    text = document_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Single-document input file is empty: {document_path}")

    loaded = LoadedDocument(
        documents_by_id={str(config.document_id): text},
        document_path=document_path,
        document_chars=len(text),
        document_words=len(text.split()),
    )
    config.run_dir().mkdir(parents=True, exist_ok=True)
    _write_json(
        config.run_dir() / "document_manifest.json",
        {
            "document_id": config.document_id,
            "document_path": str(document_path),
            "document_chars": loaded.document_chars,
            "document_words": loaded.document_words,
        },
    )
    return loaded


def build_or_update_chunks_and_memory(
    config: SingleDocumentPipelineConfig,
    documents_by_id: dict[str, str],
    *,
    backend: Any | None = None,
) -> IngestStageResult:
    """Run SAADI ingest for the single document and write inspectable artifacts."""

    backend = resolve_backend(backend) if backend is not None else build_backend(config)
    run_dir = config.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)

    separator_phrases = config.resolved_separator_phrases()
    separator_cache_suffix = ""
    if separator_phrases:
        phrase_chunking_identity = {
            "separator_phrases": separator_phrases,
            "documents": {
                str(doc_id): hashlib.sha256(str(text).encode("utf-8")).hexdigest()
                for doc_id, text in sorted(documents_by_id.items())
            },
        }
        serialized_separators = json.dumps(
            phrase_chunking_identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        separator_cache_suffix = "_separators_" + hashlib.sha256(
            serialized_separators.encode("utf-8")
        ).hexdigest()[:12]
    precomputed: dict[str, list[ChunkRecord]] | None = None
    if separator_phrases:
        precomputed = {}
        for doc_id, text in documents_by_id.items():
            matched_phrases = [phrase for phrase in separator_phrases if phrase in text]
            if matched_phrases:
                chunks = split_by_separator_phrases(str(text), str(doc_id), matched_phrases)
                precomputed[str(doc_id)] = chunks
                print(
                    f"Separator-phrase chunking: doc={doc_id} "
                    f"phrases={matched_phrases!r} -> {len(chunks)} chunks"
                )
            else:
                import warnings
                warnings.warn(
                    f"None of the separator phrases {separator_phrases!r} were found in document {doc_id!r}; "
                    "falling back to token-window chunking for this document."
                )

    raw_chunks, canonicalized_chunks, artifacts = ingest_documents(
        documents_by_id={str(doc_id): str(text) for doc_id, text in documents_by_id.items()},
        backend=backend,
        strategy_name=config.strategy_name,
        strategy_kwargs=config.strategy_kwargs(),
        cache_namespace={
            # Phrase-defined chunking is outside the production token-window
            # namespace. Key it separately so changing phrases cannot reuse or
            # retain stale chunks from an incompatible split.
            "dataset": _safe_label(config.document_id) + separator_cache_suffix,
            "method": _safe_label(config.strategy_name),
            "llm": _infer_llm_label(config.model_config),
            "active_entity_config": active_entity_config_label(
                config.max_entities,
                config.max_aliases,
                config.max_desc,
            ),
            "active_entity_selector": active_entity_selector_label(config.active_entity_selector),
        },
        use_cache=bool(config.use_cache),
        cache_save_every_steps=int(config.cache_save_every_steps),
        cache_save_each_document=bool(config.cache_save_each_document),
        chunk_size=int(config.chunk_size),
        chunk_overlap=int(config.chunk_overlap),
        output_dir=str(run_dir),
        generate=config.ingest_generate_params(),
        write_raw_chunks=True,
        schedule=str(config.ingest_schedule),
        wavefront_batch_size=int(config.wavefront_batch_size),
        precomputed_raw_chunks_by_doc_id=precomputed,
    )

    document_manifest_path = _write_json(
        run_dir / "document_manifest.json",
        {
            "document_id": config.document_id,
            "document_chars": len(next(iter(documents_by_id.values()), "")),
            "raw_chunks": len(raw_chunks),
            "canonicalized_chunks": len(canonicalized_chunks),
            "chunking": {
                "mode": "separator_phrases" if separator_phrases else "token_window",
                "separator_phrases": separator_phrases,
                "chunk_size": int(config.chunk_size),
                "chunk_overlap": int(config.chunk_overlap),
            },
            "llm_cache_index_json": str(artifacts.llm_cache_index_json) if artifacts else None,
        },
    )
    raw_chunks_path = Path(write_jsonl(run_dir / "raw_chunks.jsonl", [chunk.to_dict() for chunk in raw_chunks]))
    canonicalized_chunks_path = Path(
        write_jsonl(run_dir / "canonicalized_chunks.jsonl", [chunk.to_dict() for chunk in canonicalized_chunks])
    )
    memory_summary_path = _write_memory_summary(
        run_dir / "memory_summary.json",
        canonicalized_chunks=canonicalized_chunks,
        document_id=config.document_id,
    )
    return IngestStageResult(
        raw_chunks=raw_chunks,
        canonicalized_chunks=canonicalized_chunks,
        artifacts=artifacts,
        document_manifest_path=document_manifest_path,
        raw_chunks_path=raw_chunks_path,
        canonicalized_chunks_path=canonicalized_chunks_path,
        memory_summary_path=memory_summary_path,
    )


def retrieve_for_query(
    config: SingleDocumentPipelineConfig,
    raw_chunks: list[ChunkRecord],
    canonicalized_chunks: list[CanonicalizedChunkRecord],
    *,
    artifacts: ArtifactPaths | None = None,
    query: str | None = None,
    out_dir: Path | None = None,
) -> RetrievalStageResult:
    """Run the current SAADI MBR2 retrieval method for one query."""

    retrieval_provenance = git_provenance(cwd=str(REPOSITORY_ROOT))

    question = str(query if query is not None else config.query).strip()
    if not question:
        raise ValueError("query must not be empty")
    retrieval_scope = str(config.retrieval_scope or "per_document").strip().lower()
    if retrieval_scope != "per_document":
        raise ValueError("Current SAADI MBR2 retrieval requires retrieval_scope='per_document'.")

    cache_index_path = _find_cache_index_path(config, artifacts)
    if cache_index_path is None:
        raise FileNotFoundError(
            "Current SAADI MBR2 retrieval requires the ingest cache index.json, but it could not be found."
        )
    document_index = build_mbr_document_memory_index(
        cache_index_path,
        doc_id=str(config.document_id),
    )

    example = RetrievalExample(
        query_id="query_1",
        doc_id=str(config.document_id),
        question=question,
    )
    results, detailed_results, method_label = run_mbr_variant_retrieval_pipeline(
        examples=[example],
        document_indices={str(config.document_id): document_index},
        retriever_name=str(config.retriever),
        retrieve_k=int(config.retrieve_k),
        variant=str(config.mbr_variant),
        lambda_t=float(config.lambda_t),
        lambda_n=float(config.lambda_n),
        lambda_h=float(config.lambda_h),
        normalization_method=str(config.normalization_method),
        memory_name_weight=float(config.memory_name_weight),
        memory_alias_weight=float(config.memory_alias_weight),
        memory_description_weight=float(config.memory_description_weight),
        memory_aggregation=str(config.memory_aggregation),
    )
    result = results[0]
    _, chunk_lookup = _chunk_text_lookup(
        raw_chunks=raw_chunks,
        canonicalized_chunks=canonicalized_chunks,
        # Production MBR ranks and returns the original raw chunks. Entity memory
        # and hierarchy summaries are scoring channels, not replacement chunk text.
        retrieval_text_mode="raw_chunks",
    )
    retrieved_chunks = _retrieved_chunk_rows(result=result, chunk_lookup=chunk_lookup)

    retrieval_dir = (out_dir / "retrieval") if out_dir is not None else (config.run_dir() / "retrieval")
    retrieval_results_path = retrieval_dir / "retrieval_result.json"
    retrieval_payload = {
        **retrieval_provenance,
        "query": question,
        "retriever": config.retriever,
        "retrieval_method": method_label,
        "mbr_variant": str(config.mbr_variant),
        "normalization_method": str(config.normalization_method),
        "outer_weights": {
            "text": float(config.lambda_t),
            "memory": float(config.lambda_n),
            "summary": float(config.lambda_h),
        },
        "memory_weights": {
            "name": float(config.memory_name_weight),
            "alias": float(config.memory_alias_weight),
            "description": float(config.memory_description_weight),
        },
        "memory_aggregation": str(config.memory_aggregation),
        "chunk_text_source": "raw",
        "retrieval_scope": retrieval_scope,
        "retrieve_k": int(config.retrieve_k),
        "result": result.to_dict(),
        "retrieved_chunks": retrieved_chunks,
    }
    _write_json(retrieval_results_path, retrieval_payload)
    detailed = detailed_results.get(str(example.query_id))
    diagnostics_payload = {
        **retrieval_payload,
        "llm_cache_index_json": str(cache_index_path),
        "mbr_details": asdict(detailed) if detailed is not None else None,
    }
    retrieval_diagnostics_path = _write_json(retrieval_dir / "retrieval_diagnostics.json", diagnostics_payload)
    return RetrievalStageResult(
        retrieval_result=result,
        resolved_mode=method_label,
        method_label=method_label,
        retrieval_results_path=retrieval_results_path,
        retrieval_diagnostics_path=retrieval_diagnostics_path,
        retrieved_chunks=retrieved_chunks,
    )


def prepare_context_for_generation(
    config: SingleDocumentPipelineConfig,
    retrieval_result: RetrievalResult,
    raw_chunks: list[ChunkRecord],
    canonicalized_chunks: list[CanonicalizedChunkRecord],
    *,
    artifacts: ArtifactPaths | None = None,
    query: str | None = None,
    out_dir: Path | None = None,
) -> ContextStageResult:
    """Build chunk-only or folded context for the single retrieved query."""

    top_k = int(config.generation_top_k or config.retrieve_k)
    branching_factor = int(config.hierarchy_k if config.hierarchy_k is not None else config.branching_factor)
    _, chunk_lookup = _chunk_text_lookup(
        raw_chunks=raw_chunks,
        canonicalized_chunks=canonicalized_chunks,
        retrieval_text_mode="raw_chunks",
    )
    rows = _retrieved_chunk_rows(result=retrieval_result, chunk_lookup=chunk_lookup, top_k=top_k)
    retrieved_items = _retrieved_items_from_rows(rows)
    qa_entry = _qa_entry(config, query=query)
    request = QueryContextRequest(
        retrieval_result=retrieval_result,
        retrieved_items=retrieved_items,
        question=str(query if query is not None else config.query),
        qa_entry=qa_entry,
    )

    method = str(config.context_method or "folded_context_block").strip().lower()
    if method == "chunks_only":
        built = ChunksOnlyContextBuilder(
            ContextBuilderConfig(
                experiment_name="chunks_only",
                top_k=top_k,
                summary_k=branching_factor,
                include_bridge=bool(config.include_bridge),
            )
        ).build(request)
    elif method in {"folded_context_block", "folded_context_nested"}:
        cache_index_path = _find_cache_index_path(config, artifacts)
        summaries_by_doc = (
            load_summary_blocks_by_doc(cache_index_path, summary_k=branching_factor)
            if cache_index_path is not None
            else {}
        )
        built = FoldedContextBuilder(
            ContextBuilderConfig(
                experiment_name=method,
                top_k=top_k,
                summary_k=branching_factor,
                include_bridge=bool(config.include_bridge),
                use_entity_memory=False,
            ),
            summaries_by_doc=summaries_by_doc,
        ).build(request)
        built.debug_payload["llm_cache_index_json"] = str(cache_index_path) if cache_index_path else None
    else:
        raise ValueError(
            "Invalid context_method. Expected one of: chunks_only, folded_context_block, "
            f"folded_context_nested (got: {config.context_method!r})."
        )

    context_dir = (out_dir / "context") if out_dir is not None else (config.run_dir() / "context")
    context_path = _write_text(context_dir / "context.txt", built.context_text)
    debug_path = _write_json(context_dir / "context_debug.json", built.debug_payload)
    return ContextStageResult(
        context_text=built.context_text,
        debug_payload=built.debug_payload,
        context_path=context_path,
        debug_path=debug_path,
    )


def generate_answer(
    config: SingleDocumentPipelineConfig,
    context_text: str,
    *,
    query: str | None = None,
    backend: Any | None = None,
    out_dir: Path | None = None,
) -> GenerationStageResult:
    """Generate the final answer from the prepared context."""

    backend = resolve_backend(backend) if backend is not None else build_backend(config)
    qa_entry = _qa_entry(config, query=query)
    question = str(qa_entry["question"])
    tokenizer = getattr(backend, "chat_tokenizer", None)
    trimmed_context, context_meta = _trim_context(
        context_text,
        tokenizer=tokenizer,
        max_context_tokens=config.max_context_tokens,
        overflow_policy=str(config.overflow_policy),
    )
    chat_prompt, user_prompt = build_long_context_prompt(
        context=trimmed_context,
        question=question,
        qa_entry=qa_entry,
        system_prompt=str(config.system_prompt),
        user_template=str(config.user_template),
    )
    # The rendered text is what this backend sends; it is saved for inspection.
    prompt = render_prompt(backend, chat_prompt)

    generate = config.answer_generate_params()
    generation_started = time.perf_counter()
    if context_meta.get("skip_reason"):
        raw_output = ""
        generated_answer = ""
        status = "skipped_context_overflow"
    else:
        raw_output = backend.generate_chat_batch(
            [chat_prompt],
            max_tokens=generate.max_tokens,
            temperature=generate.temperature,
            stop=generate.stop,
        )[0]
        generated_answer = str(raw_output).strip()
        status = "ok" if generated_answer else "empty_output"
    generation_latency_ms = (time.perf_counter() - generation_started) * 1000.0

    generation_record = {
        "query_id": qa_entry["id"],
        "doc_id": qa_entry["document_id"],
        "question": question,
        "generated_answer": generated_answer,
        "raw_output": raw_output,
        "status": status,
        "skip_reason": context_meta.get("skip_reason"),
        "prompt_version": config.prompt_version,
        "context_tokens": context_meta.get("context_tokens"),
        "original_context_tokens": context_meta.get("original_context_tokens"),
        "context_truncated": context_meta.get("context_truncated"),
        "generation_latency_ms": generation_latency_ms,
        "answer_max_tokens": generate.max_tokens,
        "answer_temperature": generate.temperature,
    }

    generation_dir = (out_dir / "generation") if out_dir is not None else (config.run_dir() / "generation")
    prompt_path = _write_text(generation_dir / "prompt.txt", prompt)
    _write_text(generation_dir / "user_prompt.txt", user_prompt)
    answer_path = _write_text(generation_dir / "answer.txt", generated_answer)
    generation_record_path = _write_json(generation_dir / "generation_record.json", generation_record)
    return GenerationStageResult(
        generated_answer=generated_answer,
        raw_output=raw_output,
        prompt=prompt,
        user_prompt=user_prompt,
        generation_record=generation_record,
        prompt_path=prompt_path,
        answer_path=answer_path,
        generation_record_path=generation_record_path,
    )


def save_demo_artifacts(
    config: SingleDocumentPipelineConfig,
    *,
    loaded_document: LoadedDocument | None = None,
    ingest: IngestStageResult | None = None,
    retrieval: RetrievalStageResult | None = None,
    context: ContextStageResult | None = None,
    generation: GenerationStageResult | None = None,
    backend: Any | None = None,
    provenance: dict[str, Any] | None = None,
) -> Path:
    """Write the run-level manifest that links all stage artifacts."""

    run_provenance = dict(
        provenance or git_provenance(cwd=str(REPOSITORY_ROOT))
    )
    model_config = dict(config.model_config or {})
    model_config["generate"] = {
        "max_tokens": int(config.ingest_max_tokens),
        "temperature": float(config.ingest_temperature),
        "stop": list(config.ingest_stop),
    }
    reproducibility = build_reproducibility_metadata(
        config={"model": model_config},
        backend=backend,
        generate=config.ingest_generate_params(),
        cwd=str(REPOSITORY_ROOT),
        provenance=run_provenance,
    )
    reproducibility = describe_llm_provenance(backend, reproducibility)
    manifest = {
        **run_provenance,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config.to_dict(),
        "reproducibility": reproducibility,
        "artifacts": {
            "document_manifest_json": str(ingest.document_manifest_path) if ingest else str(config.run_dir() / "document_manifest.json"),
            "raw_chunks_jsonl": str(ingest.raw_chunks_path) if ingest else None,
            "canonicalized_chunks_jsonl": str(ingest.canonicalized_chunks_path) if ingest else None,
            "memory_summary_json": str(ingest.memory_summary_path) if ingest and ingest.memory_summary_path else None,
            "llm_cache_dir": str(ingest.artifacts.llm_cache_dir) if ingest and ingest.artifacts else None,
            "llm_cache_index_json": str(ingest.artifacts.llm_cache_index_json) if ingest and ingest.artifacts else None,
            "retrieval_result_json": str(retrieval.retrieval_results_path) if retrieval else None,
            "retrieval_diagnostics_json": str(retrieval.retrieval_diagnostics_path) if retrieval else None,
            "context_txt": str(context.context_path) if context else None,
            "context_debug_json": str(context.debug_path) if context else None,
            "prompt_txt": str(generation.prompt_path) if generation else None,
            "answer_txt": str(generation.answer_path) if generation else None,
            "generation_record_json": str(generation.generation_record_path) if generation else None,
        },
        "summary": {
            "document_chars": loaded_document.document_chars if loaded_document else None,
            "document_words": loaded_document.document_words if loaded_document else None,
            "raw_chunks": len(ingest.raw_chunks) if ingest else None,
            "canonicalized_chunks": len(ingest.canonicalized_chunks) if ingest else None,
            "retrieved_chunks": len(retrieval.retrieved_chunks) if retrieval else None,
            "retrieval_method": retrieval.method_label if retrieval else None,
            "context_chars": len(context.context_text) if context else None,
            "answer_chars": len(generation.generated_answer) if generation else None,
        },
    }
    return _write_json(config.run_dir() / "run_manifest.json", manifest)


@dataclass(slots=True)
class QuestionRunResult:
    question_index: int
    question: str
    retrieval: RetrievalStageResult
    context: ContextStageResult
    generation: GenerationStageResult
    question_dir: Path


def _save_multi_question_summary(
    config: SingleDocumentPipelineConfig,
    loaded: LoadedDocument | None,
    ingest: IngestStageResult | None,
    results: list[QuestionRunResult],
) -> Path:
    question_records = []
    for r in results:
        gen_rec = r.generation.generation_record
        question_records.append({
            "question_index": r.question_index,
            "question": r.question,
            "context_tokens": gen_rec.get("context_tokens"),
            "context_words": len(r.context.context_text.split()),
            "context_chars": len(r.context.context_text),
            "answer_chars": len(r.generation.generated_answer),
            "generated_answer": r.generation.generated_answer,
            "generation_latency_ms": gen_rec.get("generation_latency_ms"),
            "status": gen_rec.get("status"),
            "artifacts": {
                "context_txt": str(r.context.context_path),
                "answer_txt": str(r.generation.answer_path),
                "generation_record_json": str(r.generation.generation_record_path),
                "retrieval_result_json": str(r.retrieval.retrieval_results_path),
                "retrieval_diagnostics_json": str(r.retrieval.retrieval_diagnostics_path),
            },
        })
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config.to_dict(),
        "n_questions": len(results),
        "ingest_summary": {
            "raw_chunks": len(ingest.raw_chunks) if ingest else None,
            "canonicalized_chunks": len(ingest.canonicalized_chunks) if ingest else None,
        },
        "document_summary": {
            "document_chars": loaded.document_chars if loaded else None,
            "document_words": loaded.document_words if loaded else None,
        },
        "questions": question_records,
    }
    return _write_json(config.run_dir() / "questions_summary.json", summary)


def run_multi_question_pipeline(
    config: SingleDocumentPipelineConfig,
    questions: list[str],
    *,
    backend: Any | None = None,
    loaded: LoadedDocument | None = None,
    ingest: IngestStageResult | None = None,
) -> tuple[list[QuestionRunResult], Path]:
    """Run retrieval + generation for each question over shared ingest; write per-question artifacts."""

    if not questions:
        raise ValueError("questions list must not be empty")
    if loaded is None:
        loaded = load_single_document(config)
    backend = resolve_backend(backend) if backend is not None else build_backend(config)
    if ingest is None:
        ingest = build_or_update_chunks_and_memory(config, loaded.documents_by_id, backend=backend)

    questions_dir = config.run_dir() / "questions"
    results: list[QuestionRunResult] = []

    for i, question in enumerate(questions):
        q_dir = questions_dir / f"q_{i:03d}"
        q_dir.mkdir(parents=True, exist_ok=True)

        retrieval = retrieve_for_query(
            config,
            ingest.raw_chunks,
            ingest.canonicalized_chunks,
            artifacts=ingest.artifacts,
            query=question,
            out_dir=q_dir,
        )
        context = prepare_context_for_generation(
            config,
            retrieval.retrieval_result,
            ingest.raw_chunks,
            ingest.canonicalized_chunks,
            artifacts=ingest.artifacts,
            query=question,
            out_dir=q_dir,
        )
        generation = generate_answer(
            config,
            context.context_text,
            query=question,
            backend=backend,
            out_dir=q_dir,
        )
        results.append(
            QuestionRunResult(
                question_index=i,
                question=question,
                retrieval=retrieval,
                context=context,
                generation=generation,
                question_dir=q_dir,
            )
        )

    summary_path = _save_multi_question_summary(config, loaded, ingest, results)
    return results, summary_path


def run_memory_only_pipeline(
    config: SingleDocumentPipelineConfig,
    *,
    backend: Any | None = None,
) -> MemoryOnlyRunResult:
    """Run only the SAADI ingest/memory construction stages for one document."""

    loaded = load_single_document(config)
    backend = resolve_backend(backend) if backend is not None else build_backend(config)
    ingest = build_or_update_chunks_and_memory(config, loaded.documents_by_id, backend=backend)
    manifest_path = save_demo_artifacts(
        config,
        loaded_document=loaded,
        ingest=ingest,
        backend=backend,
    )
    return MemoryOnlyRunResult(
        loaded_document=loaded,
        ingest=ingest,
        run_manifest_path=manifest_path,
    )


def run_single_document_pipeline(
    config: SingleDocumentPipelineConfig,
    *,
    backend: Any | None = None,
) -> SingleDocumentRunResult:
    """Run the complete single-document demo pipeline."""

    loaded = load_single_document(config)
    backend = resolve_backend(backend) if backend is not None else build_backend(config)
    ingest = build_or_update_chunks_and_memory(config, loaded.documents_by_id, backend=backend)
    retrieval = retrieve_for_query(
        config,
        ingest.raw_chunks,
        ingest.canonicalized_chunks,
        artifacts=ingest.artifacts,
    )
    context = prepare_context_for_generation(
        config,
        retrieval.retrieval_result,
        ingest.raw_chunks,
        ingest.canonicalized_chunks,
        artifacts=ingest.artifacts,
    )
    generation = generate_answer(config, context.context_text, backend=backend)
    manifest_path = save_demo_artifacts(
        config,
        loaded_document=loaded,
        ingest=ingest,
        retrieval=retrieval,
        context=context,
        generation=generation,
        backend=backend,
    )
    return SingleDocumentRunResult(
        loaded_document=loaded,
        ingest=ingest,
        retrieval=retrieval,
        context=context,
        generation=generation,
        run_manifest_path=manifest_path,
    )
