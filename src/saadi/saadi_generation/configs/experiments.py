"""Saadi folded-context generation configurations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

ContextBuilderFamily = Literal["chunks_only", "folded_context"]
ContextOrdering = Literal["block", "nested"]
BridgeMode = Literal["prefix", "local", "generated"]


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    name: str
    context_builder_family: ContextBuilderFamily
    uses_entity_memory: bool
    include_bridge: bool
    description: str
    status: str = "implemented"
    ordering: ContextOrdering | None = None
    bridge_mode: BridgeMode | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        if self.bridge_mode == "generated":
            payload["context_strategy"] = "generated_bridge"
        else:
            payload["context_strategy"] = (
                "folded" if self.context_builder_family == "folded_context" else self.context_builder_family
            )
        payload["use_bridge"] = self.include_bridge
        payload["use_memory"] = self.uses_entity_memory
        return payload


EXPERIMENT_REGISTRY: dict[str, ExperimentSpec] = {
    "folded_context_block": ExperimentSpec(
        name="folded_context_block",
        context_builder_family="folded_context",
        uses_entity_memory=False,
        include_bridge=True,
        description="Use a flat folded-context block built around retrieved chunks.",
        status="implemented",
        ordering="block",
        bridge_mode="prefix",
    ),
    "folded_context_nested": ExperimentSpec(
        name="folded_context_nested",
        context_builder_family="folded_context",
        uses_entity_memory=False,
        include_bridge=True,
        description="Use a nested folded-context representation around retrieved chunks.",
        status="implemented",
        ordering="nested",
        bridge_mode="prefix",
    ),
    "folded_context_local_bridge": ExperimentSpec(
        name="folded_context_local_bridge",
        context_builder_family="folded_context",
        uses_entity_memory=False,
        include_bridge=True,
        description="Use nested folded context over only the local interval between retrieved chunks.",
        status="implemented",
        ordering="nested",
        bridge_mode="local",
    ),
    "folded_context_generated_bridge": ExperimentSpec(
        name="folded_context_generated_bridge",
        context_builder_family="folded_context",
        uses_entity_memory=False,
        include_bridge=True,
        description="Use retrieved chunks plus ultra-short generated continuity bridges between retrieved regions.",
        status="implemented",
        ordering="nested",
        bridge_mode="generated",
    ),

}


def experiment_names() -> list[str]:
    return sorted(EXPERIMENT_REGISTRY)


def get_experiment_spec(name: str) -> ExperimentSpec:
    key = str(name or "").strip()
    try:
        return EXPERIMENT_REGISTRY[key]
    except KeyError as exc:
        valid = ", ".join(experiment_names())
        raise ValueError(f"Unknown SAADI generation experiment {name!r}. Expected one of: {valid}") from exc
