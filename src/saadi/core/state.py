"""Maintains and restores discourse state, including hierarchical summaries across chunks."""

from __future__ import annotations

from copy import deepcopy
from typing import Callable

from .types import CanonicalizationOutput, DiscourseState, EntityRecord
from ..entities.merge import merge_active_entities


def init_discourse_state(*, hierarchy_levels: list[int] | None = None) -> DiscourseState:
    state = DiscourseState()
    if hierarchy_levels:
        state.hierarchy_summaries = {int(level): {} for level in hierarchy_levels}
    return state


def clone_state(state: DiscourseState) -> DiscourseState:
    return deepcopy(state)


def _entity_from_dict(obj: dict) -> EntityRecord:
    return EntityRecord(
        name=str(obj.get("name", "") or ""),
        description=[str(x) for x in (obj.get("description", []) or [])],
        aliases=[str(x) for x in (obj.get("aliases", []) or [])],
    )


def state_from_snapshot(snapshot: dict | None) -> DiscourseState:
    snapshot = snapshot or {}
    state = DiscourseState()
    state.prev_chunk_text = str(snapshot.get("prev_chunk_text", "") or "")

    entity_memory = snapshot.get("entity_memory", []) or []
    active_entities = snapshot.get("active_entities", []) or []
    if isinstance(entity_memory, list):
        state.entity_memory = [_entity_from_dict(e) for e in entity_memory if isinstance(e, dict)]
    if isinstance(active_entities, list):
        state.active_entities = [_entity_from_dict(e) for e in active_entities if isinstance(e, dict)]

    raw_hier = snapshot.get("hierarchy_summaries", {}) or {}
    if isinstance(raw_hier, dict):
        out: dict[int, dict[int, str]] = {}
        for k, vals in raw_hier.items():
            try:
                cycle = int(k)
            except Exception:
                continue
            # New format: {cycle: {step: summary}}
            if isinstance(vals, dict):
                by_step: dict[int, str] = {}
                for sk, sv in vals.items():
                    try:
                        step = int(sk)
                    except Exception:
                        continue
                    by_step[step] = str(sv)
                out[cycle] = by_step
                continue
            # Backward compatibility: legacy format {cycle: [summary, ...]}
            if isinstance(vals, list):
                by_step = {}
                for idx, sv in enumerate(vals):
                    step = (idx + 1) * cycle - 1
                    by_step[step] = str(sv)
                out[cycle] = by_step
        state.hierarchy_summaries = out

    raw_texts = snapshot.get("raw_chunk_texts", {}) or {}
    if isinstance(raw_texts, dict):
        restored: dict[int, str] = {}
        for k, v in raw_texts.items():
            try:
                restored[int(k)] = str(v)
            except Exception:
                continue
        state.raw_chunk_texts = restored
    return state


def apply_canonicalization_to_state(
    state: DiscourseState,
    output: CanonicalizationOutput,
    *,
    chunk_step: int | None = None,
    hierarchy_l0_text: str | None = None,
    hierarchy_rollup_fn: Callable[[list[str], int, int, int], str | None] | None = None,
    hierarchy_branching_factor: int | None = None,
    hierarchy_k: int | None = None,
    hierarchy_max_level: int | None = None,
    raw_chunk_text: str | None = None,
) -> DiscourseState:
    new_state, gen = begin_canonicalization_update(
        state,
        output,
        chunk_step=chunk_step,
        hierarchy_l0_text=hierarchy_l0_text,
        hierarchy_branching_factor=hierarchy_branching_factor,
        hierarchy_k=hierarchy_k,
        hierarchy_max_level=hierarchy_max_level,
        raw_chunk_text=raw_chunk_text,
    )
    if gen is not None:
        drive_rollup_generator(gen, hierarchy_rollup_fn)
    return new_state


def begin_canonicalization_update(
    state: DiscourseState,
    output: CanonicalizationOutput,
    *,
    chunk_step: int | None = None,
    hierarchy_l0_text: str | None = None,
    hierarchy_branching_factor: int | None = None,
    hierarchy_k: int | None = None,
    hierarchy_max_level: int | None = None,
    raw_chunk_text: str | None = None,
) -> tuple[DiscourseState, "Generator[tuple[list[str], int, int, int], str | None, None] | None"]:
    """Apply canonicalization but defer hierarchy rollups.

    Returns ``(new_state, gen)``. ``gen`` is a generator that yields the rollup
    requests needed for this step (or ``None`` when no rollups are required). The
    caller drives ``gen`` and sends back each rollup summary; this lets callers
    batch the rollup LLM calls across multiple documents instead of issuing them
    one at a time. For the synchronous path use ``apply_canonicalization_to_state``.

    ``hierarchy_l0_text`` (the anchored copy of the chunk) is what the folds
    summarize. ``raw_chunk_text`` is the original chunk text, kept alongside so
    the history can show the model unmodified text as L0 context.
    """
    new_state = clone_state(state)
    new_state.prev_chunk_text = output.updated_chunk_text or state.prev_chunk_text
    rejected_names = {entity.name for entity in output.rejected_entities}
    rejected_keys = {name.strip().casefold() for name in rejected_names if name.strip()}
    new_state.active_entities = [
        entity
        for entity in output.active_entities
        if entity.name.strip().casefold() not in rejected_keys
    ]
    new_state.entity_memory = merge_active_entities(
        new_state.entity_memory,
        new_state.active_entities,
        rejected_names=rejected_names,
    )

    raw_b = hierarchy_branching_factor if hierarchy_branching_factor is not None else hierarchy_k
    try:
        b = int(raw_b) if raw_b is not None else None
    except Exception:
        b = None

    if b and b > 1:
        l0_text = hierarchy_l0_text or output.updated_chunk_text
        if not l0_text:
            return new_state, None
        if chunk_step is None:
            l0 = new_state.hierarchy_summaries.get(1, {})
            chunk_step = (max(l0.keys()) + 1) if l0 else 0
        if raw_chunk_text is not None:
            new_state.raw_chunk_texts[int(chunk_step)] = str(raw_chunk_text)
        gen = begin_hierarchy_update(
            new_state,
            l0_text,
            chunk_step,
            b,
            max_hierarchy_level=hierarchy_max_level,
        )
        return new_state, gen
    return new_state, None


def drive_rollup_generator(
    gen: "Generator[tuple[list[str], int, int, int], str | None, None]",
    rollup_fn: Callable[[list[str], int, int, int], str | None] | None,
) -> None:
    """Drive a deferred-rollup generator synchronously using ``rollup_fn``."""
    try:
        group, cycle, next_cycle, step = next(gen)
        while True:
            summary: str | None = None
            if rollup_fn is not None:
                try:
                    summary = rollup_fn(group, cycle, next_cycle, step)
                except Exception:
                    summary = None
            group, cycle, next_cycle, step = gen.send(summary)
    except StopIteration:
        pass


def _update_hierarchy(
    state: DiscourseState,
    l0_text: str,
    chunk_step: int,
    branching_factor: int,
    *,
    rollup_fn: Callable[[list[str], int, int, int], str | None] | None = None,
    max_hierarchy_level: int | None = None,
) -> None:
    gen = begin_hierarchy_update(
        state, l0_text, chunk_step, branching_factor, max_hierarchy_level=max_hierarchy_level
    )
    drive_rollup_generator(gen, rollup_fn)


def begin_hierarchy_update(
    state: DiscourseState,
    l0_text: str,
    chunk_step: int,
    branching_factor: int,
    *,
    max_hierarchy_level: int | None = None,
):
    """Generator form of :func:`_update_hierarchy`.

    Stores the L0 text, then yields ``(group, lower_cycle, next_cycle, step)`` for
    each rollup that must be produced, lowest level first. The caller sends back the
    rollup summary via ``gen.send(summary)``; if the sent value is empty/None a local
    fallback summary is stored. Yielding lowest level first preserves the cascade
    ordering (a level's summary is stored before the next level's group is read).
    """
    # Cycle size 1 stores L0 (updated chunks) by step.
    # Cycle size b stores L1 summaries; b^2 stores L2; and so on.
    if 1 not in state.hierarchy_summaries:
        state.hierarchy_summaries[1] = {}
    state.hierarchy_summaries[1][int(chunk_step)] = l0_text

    b = int(branching_factor)
    cycle = 1
    while True:
        next_cycle = cycle * b
        # Create next-cycle summary only at exact boundaries (0-based steps).
        if (chunk_step + 1) % next_cycle != 0:
            break
        if max_hierarchy_level is not None and max_hierarchy_level >= 0:
            max_cycle = b**max_hierarchy_level
            if next_cycle > max_cycle:
                break
        lower = state.hierarchy_summaries.get(cycle, {})
        if not lower:
            break
        lower_steps = sorted(s for s in lower.keys() if s <= chunk_step)
        if len(lower_steps) < b:
            break
        group = [lower[s] for s in lower_steps[-b:]]
        summary = yield (group, cycle, next_cycle, int(chunk_step))
        if not isinstance(summary, str) or not summary.strip():
            summary = _local_rollup_summary(group)
        state.hierarchy_summaries.setdefault(next_cycle, {})[int(chunk_step)] = summary
        cycle = next_cycle


def _local_rollup_summary(items: list[str], max_words: int = 400) -> str:
    joined = " ".join(i.strip() for i in items if i.strip())
    words = joined.split()
    return " ".join(words[:max_words])
