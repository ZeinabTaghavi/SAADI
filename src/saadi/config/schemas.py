"""Defines lightweight dataclass schemas for ingest, retrieval, and evaluation config."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class IngestConfig:
    chunk_size: int = 500
    chunk_overlap: int = 0
    strategy: str = "hierarchical"
    strategy_kwargs: dict[str, Any] = field(default_factory=dict)
    use_cache: bool = False
    cache_save_every_steps: int = 10
    cache_save_each_document: bool = False

    schedule: str = "document"

    wavefront_batch_size: int | None = None
    write_raw_chunks: bool = False
    enabled: bool = True


@dataclass(slots=True)
class RetrievalConfig:
    retriever: str = "bm25"

    retrieve_k: int = 10

    mode: str = "canonicalized"

    scope: str = "global"


@dataclass(slots=True)
class SampleConfig:

    max_documents: int | None = None

    max_qa_entries: int | None = None


@dataclass(slots=True)
class DatasetConfig:

    name: str | None = None
    source: str | None = None
    split: str = "test"
    config_name: str | None = "default"
    qa_n: int | str = "all"
    qa_selection_method: str = "first"


    books_root: str | None = None
    documents_json: str | None = None
    qa_json: str | None = None


@dataclass(slots=True)
class EvalConfig:
    compute_per_query: bool = True


@dataclass(slots=True)
class ExperimentConfig:
    output_dir: str = "outputs"
    ingest: IngestConfig = field(default_factory=IngestConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    sample: SampleConfig = field(default_factory=SampleConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    model: dict[str, Any] = field(default_factory=lambda: {"backend": "vllm"})
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
