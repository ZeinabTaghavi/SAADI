"""The public single-document API: understand a document once, retrieve many times.

::

    from saadi import Saadi

    doc = Saadi(llm="openai:gpt-4.1-mini", state_dir="state/report")
    doc.index("report.txt")                 # one state-update call per chunk
    doc.append(more_text)                   # extends the state; earlier chunks are reused
    hits = doc.retrieve("Who leads the team?", k=5)
    reply = doc.answer("Who leads the team?")
    doc.memory, doc.history                 # entity records and the summary hierarchy

Everything lives in a flat state directory (see :data:`STATE_LAYOUT`), so a
state can be inspected with ordinary tools, copied, and loaded again with
:meth:`Saadi.load`.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
import shutil
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from ._version import __version__
from .core.types import CanonicalizedChunkRecord, ChunkRecord, EntityRecord, TextInsertion
from .exports import (
    HISTORY_DIR,
    MEMORY_FILE,
    SUMMARY_FILE,
    export_readable_state,
    find_cache_index,
    generate_document_summary,
    write_json,
)
from .inputs import PreparedDocument, prepare_document
from .io.artifact_paths import default_artifact_paths
from .llm.base import render_prompt
from .llm.factory import model_config_from_spec, resolve_backend
from .pipelines.ingest_pipeline import logger as _ingest_logger
from .pipelines.single_document_pipeline import (
    IngestStageResult,
    SingleDocumentPipelineConfig,
    build_or_update_chunks_and_memory,
    generate_answer,
    prepare_context_for_generation,
    retrieve_for_query,
    save_demo_artifacts,
)
from .prompts import PromptProfile, load_prompt_profile
from .retrievers import resolve_retriever

STATE_FORMAT_VERSION = 1
STATE_FILE = "saadi.json"
DOCUMENT_FILE = "document.txt"
CHUNKS_FILE = "chunks.jsonl"
QUERIES_DIR = "queries"
CACHE_DIR = "cache"
DEFAULT_STATE_ROOT = "saadi_state"

STATE_LAYOUT = {
    STATE_FILE: "this manifest: document id, settings, counts, and this layout",
    DOCUMENT_FILE: "the indexed text; append() extends it",
    CHUNKS_FILE: "one JSON object per chunk: chunk_id, chunk_index, text, token offsets",
    MEMORY_FILE: "entity memory: name, aliases, descriptions, first_seen_chunk",
    f"{HISTORY_DIR}/": "hierarchical history: L_0 chunks and L_n fold summaries, as .jsonl and .txt",
    SUMMARY_FILE: "whole-document summary, present after summarize()",
    f"{QUERIES_DIR}/": "one folder per retrieve() or answer() call: retrieval, context, answer",
    f"{CACHE_DIR}/": "the construction cache the pipeline resumes from; not meant for editing",
}


def _resolve_llm(llm: Any) -> Any:
    """Resolve the backend; the mock answers questions with a readable placeholder."""

    if isinstance(llm, str) and llm.strip().lower() in {"mock", "dummy"}:
        from .llm.mock_backend import MockBackend
        from .pipelines.single_document_pipeline import _single_document_mock_generator

        return MockBackend(generator=_single_document_mock_generator)
    return resolve_backend(llm)


class _SingleDocumentNoise(logging.Filter):
    """Drop the ingest pipeline's cross-document batching warning; it is moot for one document."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith("Wavefront schedule requested with 1 document")


def _safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip()).strip("._-")
    return cleaned or "document"


def _slug(value: str, limit: int = 40) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")
    return cleaned[:limit].rstrip("-") or "query"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class DocumentPipelineConfig(SingleDocumentPipelineConfig):
    """Pipeline configuration with an explicit output directory and the profile prompts."""

    output_dir: str = ""
    memory_entity_system_prompt: str = ""
    memory_rollup_system_prompt: str = ""

    def run_dir(self) -> Path:
        if self.output_dir:
            return Path(self.output_dir)
        return SingleDocumentPipelineConfig.run_dir(self)

    def strategy_kwargs(self) -> dict[str, Any]:
        kwargs = SingleDocumentPipelineConfig.strategy_kwargs(self)
        kwargs["system_prompt"] = self.memory_entity_system_prompt
        kwargs["rollup_system_prompt"] = self.memory_rollup_system_prompt
        return kwargs


class CountingBackend:
    """Wraps a backend and counts model calls; everything else is delegated."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.calls = 0
        self.prompts = 0

    @property
    def chat_tokenizer(self) -> Any:
        return getattr(self.inner, "chat_tokenizer", None)

    @property
    def runtime_metadata(self) -> dict[str, Any]:
        return dict(getattr(self.inner, "runtime_metadata", {}) or {})

    def render_prompt(self, prompt: Any) -> str:
        return render_prompt(self.inner, prompt)

    def generate_chat_batch(self, prompts: Sequence[Any], **options: Any) -> list[str]:
        self.calls += 1
        self.prompts += len(prompts)
        return self.inner.generate_chat_batch(prompts, **options)

    def generate_batch(self, prompts: list[str], **options: Any) -> list[str]:
        self.calls += 1
        self.prompts += len(prompts)
        return self.inner.generate_batch(prompts, **options)

    def close(self) -> None:
        close = getattr(self.inner, "close", None)
        if callable(close):
            close()


@dataclass(frozen=True)
class RetrievedChunk:
    rank: int
    chunk_id: str
    chunk_index: int | None
    score: float | None
    text: str


@dataclass(frozen=True)
class MemoryEntity:
    name: str
    aliases: tuple[str, ...]
    descriptions: tuple[str, ...]
    first_seen_chunk: int | None


@dataclass(frozen=True)
class HistoryItem:
    level: int
    index: int
    start_step: int
    end_step: int
    kind: str
    text: str


@dataclass(frozen=True)
class IndexReport:
    document_id: str
    state_dir: Path
    chunks: int
    new_chunks: int
    entities: int
    history_levels: tuple[int, ...]
    model_calls: int
    prompts_sent: int
    rejected_entities: int = 0  # proposals whose anchor text was not found verbatim in the chunk


@dataclass(frozen=True)
class Answer:
    question: str
    text: str
    chunks: tuple[RetrievedChunk, ...]
    context: str
    status: str
    directory: Path


class Saadi:
    """One document's SAADI state: build it once, then retrieve and answer many times.

    Args:
        llm: Anything :func:`saadi.llm.resolve_backend` accepts: ``"openai:gpt-4.1-mini"``,
            ``"ollama:qwen3:8b"``, ``"anthropic:claude-opus-5-5"``, ``"mock"``, a config
            mapping, a backend object, or a function. Needed to index, append, answer,
            and summarize; a loaded state can retrieve without one.
        retriever: A name such as ``bm25`` (default) or ``openai_embedding``, a
            retriever class, a reranker or embedding-model object, or an embedding
            function; see :mod:`saadi.retrievers`.
        prompts: A prompt profile: ``default``, ``conversation``, a directory, or a
            :class:`saadi.prompts.PromptProfile`.
        state_dir: Where the state lives. Defaults to ``saadi_state/<document_id>``.
        chunk_size, chunk_overlap, branching_factor, max_entities, max_aliases,
        max_desc, active_entity_selector, separator_phrases: construction settings
            (the paper's defaults).
        retrieve_k, answer_top_k: how many chunks retrieve() returns and answer() reads.
    """

    def __init__(
        self,
        llm: Any = None,
        retriever: Any = "bm25",
        prompts: Any = "default",
        *,
        state_dir: str | Path | None = None,
        chunk_size: int = 500,
        chunk_overlap: int = 0,
        branching_factor: int = 5,
        max_entities: int = 5,
        max_aliases: int = 2,
        max_desc: int = 3,
        active_entity_selector: str = "bm25",
        separator_phrases: Sequence[str] = (),
        retrieve_k: int = 10,
        answer_top_k: int = 5,
        ingest_max_tokens: int = 2048,
        answer_max_tokens: int = 2048,
        answer_temperature: float = 0.6,
        use_cache: bool = True,
    ) -> None:
        self._llm_spec = llm
        self._backend: CountingBackend | None = CountingBackend(_resolve_llm(llm)) if llm is not None else None
        self.retriever: str = resolve_retriever(retriever)
        self.profile: PromptProfile = load_prompt_profile(prompts)
        self._prompts_spec = str(prompts.directory) if isinstance(prompts, PromptProfile) else str(prompts)
        self.state_dir: Path | None = Path(state_dir).expanduser() if state_dir is not None else None
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)
        self.branching_factor = int(branching_factor)
        self.max_entities = int(max_entities)
        self.max_aliases = int(max_aliases)
        self.max_desc = int(max_desc)
        self.active_entity_selector = str(active_entity_selector)
        self.separator_phrases = tuple(str(p) for p in separator_phrases if str(p).strip())
        self.retrieve_k = int(retrieve_k)
        self.answer_top_k = int(answer_top_k)
        self.ingest_max_tokens = int(ingest_max_tokens)
        self.answer_max_tokens = int(answer_max_tokens)
        self.answer_temperature = float(answer_temperature)
        self.use_cache = bool(use_cache)
        self._document_id: str | None = None
        self._source: dict[str, Any] = {}
        self._config: DocumentPipelineConfig | None = None
        self._ingest: IngestStageResult | None = None
        self._created_at: str | None = None

    # ----------------------------------------------------------------- state --

    @property
    def document_id(self) -> str | None:
        return self._document_id

    @property
    def backend(self) -> Any:
        """The language-model backend in use, or None for a retrieval-only state."""

        return self._backend.inner if self._backend is not None else None

    @property
    def cache_dir(self) -> Path:
        self._require_state_dir()
        return self.state_dir / CACHE_DIR / f"prompts_{self.profile.content_hash[:10]}"

    def _require_state_dir(self) -> Path:
        if self.state_dir is None:
            raise RuntimeError("No state directory yet; call index() first or pass state_dir=")
        return self.state_dir

    def _require_backend(self, action: str) -> CountingBackend:
        if self._backend is None:
            raise RuntimeError(
                f"A language model is needed to {action}. Create the object with llm=..., "
                "for example llm='openai:gpt-4.1-mini' or llm='ollama:qwen3:8b'."
            )
        return self._backend

    def _require_indexed(self) -> IngestStageResult:
        if self._ingest is None and self.state_dir is not None and self._document_id is not None:
            self._ingest = self._load_ingest_from_cache()
        if self._ingest is None:
            raise RuntimeError("No indexed document. Call index() first, or load() a state directory.")
        return self._ingest

    def _model_config(self) -> dict[str, Any]:
        if isinstance(self._llm_spec, str):
            return model_config_from_spec(self._llm_spec)
        if isinstance(self._llm_spec, dict):
            return dict(self._llm_spec)
        if self._backend is None:
            return {"backend": "none"}
        runtime = self._backend.runtime_metadata
        return {"backend": str(runtime.get("backend") or "custom"), "model_id": str(runtime.get("model_name") or "custom")}

    def _pipeline_config(self) -> DocumentPipelineConfig:
        state_dir = self._require_state_dir()
        return DocumentPipelineConfig(
            demo_root=str(state_dir),
            document_path=str(state_dir / DOCUMENT_FILE),
            document_id=str(self._document_id),
            run_name=CACHE_DIR,
            query="",
            model_config=self._model_config(),
            ingest_max_tokens=self.ingest_max_tokens,
            answer_max_tokens=self.answer_max_tokens,
            answer_temperature=self.answer_temperature,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            branching_factor=self.branching_factor,
            active_entity_selector=self.active_entity_selector,
            max_entities=self.max_entities,
            max_aliases=self.max_aliases,
            max_desc=self.max_desc,
            use_cache=self.use_cache,
            retriever=self.retriever,
            retrieve_k=self.retrieve_k,
            separator_phrases=list(self.separator_phrases),
            generation_top_k=self.answer_top_k,
            output_dir=str(self.cache_dir),
            memory_entity_system_prompt=self.profile.entity_system_prompt,
            memory_rollup_system_prompt=self.profile.summary_system_prompt,
        )

    # ------------------------------------------------------------------ build --

    def index(self, source: str | Path | PreparedDocument, *, document_id: str | None = None, **input_options: Any) -> IndexReport:
        """Build the state for ``source``: inline text, a .txt/.md file, or a conversation .csv.

        Re-indexing the same document into the same state directory reuses the
        construction cache for every unchanged chunk.
        """

        document = prepare_document(source, document_id=document_id, **input_options)
        doc_id = _safe_id(document.document_id)
        if self.state_dir is None:
            self.state_dir = Path(DEFAULT_STATE_ROOT) / doc_id
        self.state_dir.mkdir(parents=True, exist_ok=True)
        previous = len(self._read_chunks_file()) if self._stored_document_id() == doc_id else 0
        self._document_id = doc_id
        self._source = {
            "path": str(document.source_path) if document.source_path else None,
            "input_format": document.input_format,
            "metadata": dict(document.metadata),
        }
        self._write_document(document.text)
        return self._ingest_document(document.text, previous_chunks=previous)

    def append(self, source: str | Path | PreparedDocument, *, separator: str = "\n\n", **input_options: Any) -> IndexReport:
        """Extend the indexed document and update the state without rebuilding it.

        Chunks whose text is unchanged are served from the construction cache,
        so only the chunks at the end of the document cost model calls.
        """

        self._require_indexed()
        addition = prepare_document(source, document_id=self._document_id, **input_options).text
        combined = self.document_text.rstrip("\n") + separator + addition.strip() + "\n"
        previous = len(self.chunks)
        self._write_document(combined)
        return self._ingest_document(combined, previous_chunks=previous)

    def _ingest_document(self, text: str, *, previous_chunks: int) -> IndexReport:
        backend = self._require_backend("index a document")
        config = self._pipeline_config()
        self._config = config
        calls_before, prompts_before = backend.calls, backend.prompts
        noise = _SingleDocumentNoise()
        _ingest_logger.addFilter(noise)
        try:
            ingest = build_or_update_chunks_and_memory(config, {str(self._document_id): text}, backend=backend)
        finally:
            _ingest_logger.removeFilter(noise)
        self._ingest = ingest
        readable = export_readable_state(
            self.state_dir, artifacts=ingest.artifacts, memory_summary_path=ingest.memory_summary_path
        )
        self._write_chunks(ingest.raw_chunks)
        stale_summary = self.state_dir / SUMMARY_FILE
        if stale_summary.exists():
            stale_summary.unlink()
        save_demo_artifacts(config, ingest=ingest, backend=backend)
        report = IndexReport(
            document_id=str(self._document_id),
            state_dir=self.state_dir,
            chunks=len(ingest.raw_chunks),
            new_chunks=max(0, len(ingest.raw_chunks) - int(previous_chunks)),
            entities=len(self.memory),
            history_levels=readable.levels,
            model_calls=backend.calls - calls_before,
            prompts_sent=backend.prompts - prompts_before,
            rejected_entities=sum(len(chunk.rejected_entities or []) for chunk in ingest.canonicalized_chunks),
        )
        self._write_state_file()
        return report

    # ------------------------------------------------------------------ query --

    def retrieve(self, query: str, *, k: int | None = None) -> list[RetrievedChunk]:
        """Rank the original chunks for ``query`` with the fused text, memory, and summary scores."""

        ingest = self._require_indexed()
        question = str(query or "").strip()
        if not question:
            raise ValueError("query must not be empty")
        top_k = int(k or self.retrieve_k)
        config = dataclasses.replace(self._config or self._pipeline_config(), retrieve_k=max(top_k, 1))
        query_dir = self._new_query_dir(question)
        retrieval = retrieve_for_query(
            config, ingest.raw_chunks, ingest.canonicalized_chunks, artifacts=ingest.artifacts, query=question, out_dir=query_dir
        )
        chunks = _retrieved_chunks(retrieval.retrieved_chunks, top_k)
        write_json(query_dir / "query.json", {"question": question, "mode": "retrieve", "k": top_k, "chunks": [dataclasses.asdict(c) for c in chunks]})
        return list(chunks)

    def answer(self, query: str, *, k: int | None = None) -> Answer:
        """Retrieve, build the folded context, and generate an answer with the language model."""

        backend = self._require_backend("answer questions")
        ingest = self._require_indexed()
        question = str(query or "").strip()
        if not question:
            raise ValueError("query must not be empty")
        top_k = int(k or self.answer_top_k)
        config = dataclasses.replace(
            self._config or self._pipeline_config(), retrieve_k=max(self.retrieve_k, top_k), generation_top_k=top_k
        )
        query_dir = self._new_query_dir(question)
        retrieval = retrieve_for_query(
            config, ingest.raw_chunks, ingest.canonicalized_chunks, artifacts=ingest.artifacts, query=question, out_dir=query_dir
        )
        context = prepare_context_for_generation(
            config, retrieval.retrieval_result, ingest.raw_chunks, ingest.canonicalized_chunks,
            artifacts=ingest.artifacts, query=question, out_dir=query_dir,
        )
        generation = generate_answer(config, context.context_text, query=question, backend=backend, out_dir=query_dir)
        chunks = _retrieved_chunks(retrieval.retrieved_chunks, top_k)
        result = Answer(
            question=question,
            text=generation.generated_answer,
            chunks=chunks,
            context=context.context_text,
            status=str(generation.generation_record.get("status", "ok")),
            directory=query_dir,
        )
        write_json(
            query_dir / "query.json",
            {"question": question, "mode": "answer", "k": top_k, "answer": result.text, "status": result.status, "chunks": [dataclasses.asdict(c) for c in chunks]},
        )
        return result

    def summarize(self, *, max_tokens: int = 384, temperature: float = 0.2) -> str:
        """Write one whole-document summary with the profile's document-summary prompt."""

        backend = self._require_backend("summarize the document")
        ingest = self._require_indexed()
        path = generate_document_summary(
            profile=self.profile,
            canonicalized_chunks=ingest.canonicalized_chunks,
            backend=backend,
            document_id=str(self._document_id),
            out_dir=self.cache_dir / "document_summary",
            max_tokens=max_tokens,
            temperature=temperature,
        )
        target = self._require_state_dir() / SUMMARY_FILE
        shutil.copyfile(path, target)
        self._write_state_file()
        return target.read_text(encoding="utf-8").strip()

    # ---------------------------------------------------------------- reading --

    @property
    def document_text(self) -> str:
        path = self._require_state_dir() / DOCUMENT_FILE
        return path.read_text(encoding="utf-8") if path.is_file() else ""

    @property
    def chunks(self) -> list[ChunkRecord]:
        if self._ingest is not None:
            return list(self._ingest.raw_chunks)
        return self._read_chunks_file()

    @property
    def memory(self) -> list[MemoryEntity]:
        """The entity memory, sanitized of role labels and pronouns."""

        if self.state_dir is None:
            return []
        path = self.state_dir / MEMORY_FILE
        if not path.is_file():
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        entities: list[MemoryEntity] = []
        for entity in payload.get("entities", []):
            entities.append(
                MemoryEntity(
                    name=str(entity.get("name", "")),
                    aliases=tuple(str(a.get("text", "") if isinstance(a, dict) else a) for a in entity.get("aliases", []) or []),
                    descriptions=tuple(str(d.get("text", "") if isinstance(d, dict) else d) for d in entity.get("descriptions", []) or []),
                    first_seen_chunk=entity.get("first_seen_chunk"),
                )
            )
        return entities

    @property
    def history(self) -> dict[int, list[HistoryItem]]:
        """The hierarchical history: level 0 holds the chunks, higher levels the fold summaries."""

        if self.state_dir is None:
            return {}
        history_dir = self.state_dir / HISTORY_DIR
        levels: dict[int, list[HistoryItem]] = {}
        for path in sorted(history_dir.glob("L_*.jsonl")) if history_dir.is_dir() else []:
            match = re.fullmatch(r"L_(\d+)\.jsonl", path.name)
            if not match:
                continue
            level = int(match.group(1))
            items = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    items.append(
                        HistoryItem(
                            level=level,
                            index=int(row.get("index", len(items))),
                            start_step=int(row.get("start_step", 0)),
                            end_step=int(row.get("end_step", 0)),
                            kind=str(row.get("kind", "chunk" if level == 0 else "summary")),
                            text=str(row.get("text", "")),
                        )
                    )
            levels[level] = items
        return dict(sorted(levels.items()))

    @property
    def summary(self) -> str | None:
        if self.state_dir is None:
            return None
        path = self.state_dir / SUMMARY_FILE
        return path.read_text(encoding="utf-8").strip() if path.is_file() else None

    def describe(self) -> dict[str, Any]:
        return {
            "document_id": self._document_id,
            "state_dir": str(self.state_dir) if self.state_dir else None,
            "llm": self._model_config() if self._backend is not None else None,
            "retriever": self.retriever,
            "prompts": self.profile.describe(),
            "chunks": len(self.chunks) if self.state_dir else 0,
            "entities": len(self.memory),
            "history_levels": sorted(self.history),
        }

    def __repr__(self) -> str:
        info = self.describe()
        return (
            f"Saadi(document_id={info['document_id']!r}, chunks={info['chunks']}, entities={info['entities']}, "
            f"retriever={info['retriever']!r}, prompts={self.profile.name!r}, state_dir={info['state_dir']!r})"
        )

    # ------------------------------------------------------------ persistence --

    def save(self, target: str | Path | None = None) -> Path:
        """Refresh the manifest; with ``target``, copy the whole state directory there."""

        state_dir = self._require_state_dir()
        self._write_state_file()
        if target is None:
            return state_dir
        destination = Path(target).expanduser()
        shutil.copytree(state_dir, destination, dirs_exist_ok=True)
        return destination

    @classmethod
    def load(
        cls,
        state_dir: str | Path,
        *,
        llm: Any = None,
        retriever: Any = None,
        prompts: Any = None,
        **overrides: Any,
    ) -> "Saadi":
        """Open an existing state directory.

        Construction settings come from the manifest. ``llm`` is only needed for
        append(), answer(), and summarize(); retrieve() works without one.
        """

        state_dir = Path(state_dir).expanduser()
        manifest_path = state_dir / STATE_FILE
        if not manifest_path.is_file():
            raise FileNotFoundError(f"{state_dir} is not a Saadi state directory (no {STATE_FILE})")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        settings = dict(payload.get("settings", {}))
        stored_prompts = dict(settings.get("prompts", {}))
        prompts_spec = prompts if prompts is not None else stored_prompts.get("spec", "default")
        kwargs: dict[str, Any] = {
            key: settings[key]
            for key in (
                "chunk_size", "chunk_overlap", "branching_factor", "max_entities", "max_aliases", "max_desc",
                "active_entity_selector", "separator_phrases", "retrieve_k", "answer_top_k",
                "ingest_max_tokens", "answer_max_tokens", "answer_temperature",
            )
            if key in settings
        }
        kwargs.update(overrides)
        doc = cls(llm, retriever if retriever is not None else settings.get("retriever", "bm25"), prompts_spec, state_dir=state_dir, **kwargs)
        if prompts is None and stored_prompts.get("content_hash") and stored_prompts["content_hash"] != doc.profile.content_hash:
            warnings.warn(
                f"The prompt profile {doc.profile.name!r} changed since this state was built; "
                "retrieval keeps using the stored construction, and append() will start a new cache.",
                stacklevel=2,
            )
        doc._document_id = str(payload.get("document_id") or "document")
        doc._source = dict(payload.get("source", {}))
        doc._created_at = payload.get("created_at_utc")
        doc._config = doc._pipeline_config()
        doc._ingest = doc._load_ingest_from_cache()
        stored_cache = payload.get("cache_dir")
        if doc._ingest is None and stored_cache and (state_dir / stored_cache).is_dir():
            # The profile changed, so this profile's cache is empty: read the stored
            # construction for retrieval; append() rebuilds under the new profile.
            doc._ingest = doc._load_ingest_from_cache(state_dir / stored_cache)
        return doc

    # ---------------------------------------------------------------- internals --

    def _new_query_dir(self, question: str) -> Path:
        queries_dir = self._require_state_dir() / QUERIES_DIR
        queries_dir.mkdir(parents=True, exist_ok=True)
        number = sum(1 for p in queries_dir.iterdir() if p.is_dir()) + 1
        while any(p.name.startswith(f"{number:03d}_") for p in queries_dir.iterdir()):
            number += 1
        query_dir = queries_dir / f"{number:03d}_{_slug(question)}"
        query_dir.mkdir(parents=True)
        return query_dir

    def _write_document(self, text: str) -> None:
        (self._require_state_dir() / DOCUMENT_FILE).write_text(text, encoding="utf-8")

    def _write_chunks(self, chunks: Sequence[ChunkRecord]) -> None:
        rows = [
            {"chunk_id": c.chunk_id, "chunk_index": c.chunk_index, "text": c.raw_text, "token_start": c.token_start, "token_end": c.token_end}
            for c in chunks
        ]
        (self._require_state_dir() / CHUNKS_FILE).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )

    def _read_chunks_file(self) -> list[ChunkRecord]:
        if self.state_dir is None:
            return []
        path = self.state_dir / CHUNKS_FILE
        if not path.is_file():
            return []
        doc_id = self._document_id or self._stored_document_id() or "document"
        chunks: list[ChunkRecord] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                chunks.append(
                    ChunkRecord(
                        doc_id=doc_id,
                        chunk_id=str(row["chunk_id"]),
                        chunk_index=int(row["chunk_index"]),
                        raw_text=str(row.get("text", "")),
                        token_start=row.get("token_start"),
                        token_end=row.get("token_end"),
                    )
                )
        return chunks

    def _stored_document_id(self) -> str | None:
        if self.state_dir is None:
            return None
        path = self.state_dir / STATE_FILE
        if not path.is_file():
            return None
        try:
            return str(json.loads(path.read_text(encoding="utf-8")).get("document_id") or "") or None
        except (OSError, ValueError):
            return None

    def _write_state_file(self) -> Path:
        state_dir = self._require_state_dir()
        if self._created_at is None:
            self._created_at = _now()
        queries_dir = state_dir / QUERIES_DIR
        payload = {
            "format_version": STATE_FORMAT_VERSION,
            "saadi_version": __version__,
            "document_id": self._document_id,
            "created_at_utc": self._created_at,
            "updated_at_utc": _now(),
            "source": self._source,
            "settings": {
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "branching_factor": self.branching_factor,
                "max_entities": self.max_entities,
                "max_aliases": self.max_aliases,
                "max_desc": self.max_desc,
                "active_entity_selector": self.active_entity_selector,
                "separator_phrases": list(self.separator_phrases),
                "retriever": self.retriever,
                "retrieve_k": self.retrieve_k,
                "answer_top_k": self.answer_top_k,
                "ingest_max_tokens": self.ingest_max_tokens,
                "answer_max_tokens": self.answer_max_tokens,
                "answer_temperature": self.answer_temperature,
                "prompts": {"spec": self._prompts_spec, **self.profile.describe()},
                "llm": self._model_config() if self._backend is not None else None,
            },
            "counts": {
                "chunks": len(self.chunks),
                "entities": len(self.memory),
                "history_levels": sorted(self.history),
                "queries": sum(1 for p in queries_dir.iterdir() if p.is_dir()) if queries_dir.is_dir() else 0,
                "has_summary": (state_dir / SUMMARY_FILE).is_file(),
            },
            "cache_dir": str(self.cache_dir.relative_to(state_dir)),
            "layout": STATE_LAYOUT,
        }
        return write_json(state_dir / STATE_FILE, payload)

    def _load_ingest_from_cache(self, cache_dir: Path | None = None) -> IngestStageResult | None:
        cache_dir = self.cache_dir if cache_dir is None else Path(cache_dir)
        raw_path = cache_dir / "raw_chunks.jsonl"
        canonical_path = cache_dir / "canonicalized_chunks.jsonl"
        if not (raw_path.is_file() and canonical_path.is_file()):
            return None
        raw = [ChunkRecord(**_chunk_fields(json.loads(line))) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        canonical = [_canonical_from_dict(json.loads(line)) for line in canonical_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        artifacts = default_artifact_paths(cache_dir)
        index_path = find_cache_index(artifacts)
        if index_path is not None:
            artifacts.llm_cache_index_json = index_path
        memory_path = cache_dir / "memory_summary.json"
        return IngestStageResult(
            raw_chunks=raw,
            canonicalized_chunks=canonical,
            artifacts=artifacts,
            document_manifest_path=cache_dir / "document_manifest.json",
            raw_chunks_path=raw_path,
            canonicalized_chunks_path=canonical_path,
            memory_summary_path=memory_path if memory_path.is_file() else None,
        )


def _chunk_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "doc_id": str(row.get("doc_id", "")),
        "chunk_id": str(row.get("chunk_id", "")),
        "chunk_index": int(row.get("chunk_index", 0)),
        "raw_text": str(row.get("raw_text", row.get("text", ""))),
        "token_start": row.get("token_start"),
        "token_end": row.get("token_end"),
    }


def _canonical_from_dict(row: dict[str, Any]) -> CanonicalizedChunkRecord:
    insertions = [
        TextInsertion(
            anchor_text=str(item.get("anchor_text", "")),
            name=str(item.get("name", "")),
            left_context=str(item.get("left_context", "")),
            right_context=str(item.get("right_context", "")),
            occurrence=int(item.get("occurrence", 1)),
        )
        for item in row.get("insertions", []) or []
        if isinstance(item, dict)
    ]
    rejected = [
        EntityRecord(name=str(e.get("name", "")), description=list(e.get("description", []) or []), aliases=list(e.get("aliases", []) or []))
        for e in row.get("rejected_entities", []) or []
        if isinstance(e, dict)
    ]
    return CanonicalizedChunkRecord(
        doc_id=str(row.get("doc_id", "")),
        chunk_id=str(row.get("chunk_id", "")),
        chunk_index=int(row.get("chunk_index", 0)),
        raw_chunk_text=str(row.get("raw_chunk_text", "")),
        canonicalized_chunk_text=str(row.get("canonicalized_chunk_text", "")),
        state_snapshot=dict(row.get("state_snapshot", {}) or {}),
        strategy_name=str(row.get("strategy_name", "hierarchical")),
        parser_status=row.get("parser_status", "ok"),
        raw_model_text=str(row.get("raw_model_text", "")),
        linking_sentence=str(row.get("linking_sentence", "")),
        insertions=insertions,
        rejected_entities=rejected,
    )


def _retrieved_chunks(rows: list[dict[str, Any]], top_k: int) -> tuple[RetrievedChunk, ...]:
    out = []
    for row in rows[: max(1, int(top_k))]:
        score = row.get("score")
        out.append(
            RetrievedChunk(
                rank=int(row.get("rank", len(out) + 1)),
                chunk_id=str(row.get("chunk_id", "")),
                chunk_index=row.get("chunk_index"),
                score=float(score) if score is not None else None,
                text=str(row.get("text", "")),
            )
        )
    return tuple(out)


__all__ = [
    "Answer",
    "CountingBackend",
    "DocumentPipelineConfig",
    "HistoryItem",
    "IndexReport",
    "MemoryEntity",
    "RetrievedChunk",
    "STATE_FILE",
    "STATE_FORMAT_VERSION",
    "STATE_LAYOUT",
    "Saadi",
]
