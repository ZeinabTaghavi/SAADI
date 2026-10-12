"""Defines dataclasses for chunks, discourse state, retrieval outputs, and manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional

ParserStatus = Literal["ok", "fallback", "failed"]


@dataclass(slots=True)
class ChunkRecord:
    doc_id: str
    chunk_id: str
    chunk_index: int
    raw_text: str
    token_start: int | None = None
    token_end: int | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EntityRecord:
    """One canonical entity stored in SAADI's short- or long-term state."""

    name: str
    description: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


EntityObject = EntityRecord


@dataclass(slots=True)
class TextInsertion:
    anchor_text: str
    name: str
    left_context: str = ""
    right_context: str = ""
    occurrence: int = 1

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "anchor_text": self.anchor_text,
            "left_context": self.left_context,
            "right_context": self.right_context,
            "occurrence": self.occurrence,
        }
        if self.name:
            payload["name"] = self.name
        return payload


@dataclass(slots=True)
class DiscourseState:
    prev_chunk_text: str = ""
    entity_memory: list[EntityRecord] = field(default_factory=list)
    active_entities: list[EntityRecord] = field(default_factory=list)


    hierarchy_summaries: dict[int, dict[int, str]] = field(default_factory=dict)
    # Original text of each processed chunk, keyed by 0-based chunk step. The
    # hierarchical history shows these to the model as L0 context; the anchored
    # copies in ``hierarchy_summaries[1]`` remain the fold inputs.
    raw_chunk_texts: dict[int, str] = field(default_factory=dict)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "prev_chunk_text": self.prev_chunk_text,
            "entity_memory": [e.to_dict() for e in self.entity_memory],
            "active_entities": [e.to_dict() for e in self.active_entities],
            "hierarchy_summaries": {
                int(cycle): {int(step): str(summary) for step, summary in by_step.items()}
                for cycle, by_step in self.hierarchy_summaries.items()
            },
            "raw_chunk_texts": {int(step): str(text) for step, text in self.raw_chunk_texts.items()},
        }


@dataclass(slots=True)
class CanonicalizationOutput:
    updated_chunk_text: str
    active_entities: list[EntityRecord]
    raw_model_text: str
    parser_status: ParserStatus = "ok"
    linking_sentence: str = ""
    insertions: list[TextInsertion] = field(default_factory=list)
    rejected_entities: list[EntityRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "updated_chunk_text": self.updated_chunk_text,
            "active_entities": [e.to_dict() for e in self.active_entities],
            "raw_model_text": self.raw_model_text,
            "parser_status": self.parser_status,
            "insertions": [x.to_dict() for x in self.insertions],
            "rejected_entities": [e.to_dict() for e in self.rejected_entities],
            "rejected_entity_count": len(self.rejected_entities),
        }


@dataclass(slots=True)
class CanonicalizedChunkRecord:
    doc_id: str
    chunk_id: str
    chunk_index: int
    raw_chunk_text: str
    canonicalized_chunk_text: str
    state_snapshot: dict[str, Any]
    strategy_name: str
    parser_status: ParserStatus = "ok"
    raw_model_text: str = ""
    linking_sentence: str = ""
    insertions: list[TextInsertion] = field(default_factory=list)
    rejected_entities: list[EntityRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "raw_chunk_text": self.raw_chunk_text,
            "canonicalized_chunk_text": self.canonicalized_chunk_text,
            "state_snapshot": self.state_snapshot,
            "strategy_name": self.strategy_name,
            "parser_status": self.parser_status,
            "raw_model_text": self.raw_model_text,
            "insertions": [x.to_dict() for x in self.insertions],
            "rejected_entities": [e.to_dict() for e in self.rejected_entities],
            "rejected_entity_count": len(self.rejected_entities),
        }


@dataclass(slots=True)
class RetrievalExample:
    query_id: str
    doc_id: str
    question: str
    gold_chunk_ids: list[str] = field(default_factory=list)
    silver_chunk_ids: list[str] = field(default_factory=list)
    silver_chunk_groups: list[list[str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RetrievalResult:
    query_id: str
    doc_id: str
    question: str
    retrieved_chunk_ids: list[str]
    retrieved_indices: list[int]
    scores: list[float]
    gold_chunk_ids: list[str] = field(default_factory=list)
    silver_chunk_ids: list[str] = field(default_factory=list)
    silver_chunk_groups: list[list[str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["relevant_chunk_ids"] = payload.pop("gold_chunk_ids")
        payload["evidence_groups"] = payload.pop("silver_chunk_groups")
        payload.pop("silver_chunk_ids")
        return payload


@dataclass(slots=True)
class CorpusArtifact:
    chunk_ids: list[str]
    chunk_texts: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RetrievalEvalReport:
    n_examples: int
    n_gold_eligible: int
    n_gold_hits: int
    n_silver_eligible: int
    n_silver_hits_strict: int
    n_silver_hits_loose: int
    n_union_eligible: int
    n_union_hits: int
    gold_hit_rate: float
    silver_strict_hit_rate: float
    silver_loose_hit_rate: float
    union_hit_rate: float | None
    per_query: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    eval_k: int = 5
    hit_rate_at_k: dict[str, float | None] = field(default_factory=dict)
    hits_at_k: dict[str, int | None] = field(default_factory=dict)

    @property
    def hit_rate(self) -> float | None:
        return self.union_hit_rate

    @property
    def recall_gold(self) -> float:
        return self.gold_hit_rate

    @property
    def recall_silver_strict(self) -> float:
        return self.silver_strict_hit_rate

    @property
    def recall_silver_loose(self) -> float:
        return self.silver_loose_hit_rate

    @property
    def recall_union(self) -> float | None:
        return self.union_hit_rate

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "schema_version": 1,
            "metric": "HR",
            "n_examples": self.n_examples,
            "n_eligible": self.n_union_eligible,
            "n_hits": self.n_union_hits,
            "eval_k": self.eval_k,
            "hit_rate": self.hit_rate,
            "hit_rate_at_k": self.hit_rate_at_k,
            "hits_at_k": self.hits_at_k,
            "per_query": self.per_query,
            "metadata": self.metadata,
        }
        for field_name in (
            "git_commit",
            "git_dirty",
            "git_worktree_diff_sha256",
        ):
            if self.metadata.get(field_name) is not None:
                payload[field_name] = self.metadata.get(field_name)
        return payload


@dataclass(slots=True)
class ExperimentManifest:
    created_at_utc: str
    config: dict[str, Any]
    artifacts: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def now(cls, *, config: dict[str, Any], artifacts: dict[str, str], metadata: dict[str, Any] | None = None) -> "ExperimentManifest":
        return cls(
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            config=config,
            artifacts=artifacts,
            metadata=metadata or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
