"""Merges newly active entities into long-term entity memory."""

from __future__ import annotations

from ..core.types import EntityRecord


def _merge_unique(a: list[str], b: list[str], *, max_items: int = -1) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for seq in (a or [], b or []):
        for item in seq:
            if not isinstance(item, str):
                continue
            s = item.strip()
            if s and s not in seen:
                out.append(s)
                seen.add(s)
    if max_items >= 0:
        return out[:max_items]
    return out


def merge_active_entities(
    memory_entities: list[EntityRecord],
    new_entities: list[EntityRecord],
    *,
    max_memory: int = -1,
    max_aliases: int = -1,
    max_desc: int = -1,
    rejected_names: set[str] | None = None,
) -> list[EntityRecord]:
    memory = [EntityRecord(name=e.name, description=list(e.description), aliases=list(e.aliases)) for e in (memory_entities or [])]
    index = {o.name.strip().lower(): i for i, o in enumerate(memory) if o.name.strip()}
    rejected_keys = {
        str(name).strip().lower()
        for name in (rejected_names or set())
        if str(name).strip()
    }
    for n in (new_entities or []):
        key = n.name.strip().lower()
        if not key or key in rejected_keys:
            continue
        if key in index:
            cur = memory[index[key]]
            cur.aliases = _merge_unique(cur.aliases, n.aliases, max_items=max_aliases)
            cur.description = _merge_unique(cur.description, n.description, max_items=max_desc)
        else:
            aliases = _merge_unique([], list(n.aliases), max_items=max_aliases)
            desc = _merge_unique([], list(n.description), max_items=max_desc)
            memory.append(EntityRecord(name=n.name, aliases=aliases, description=desc))
            index[key] = len(memory) - 1
    if max_memory >= 0:
        memory = memory[-max_memory:]
    return memory


def merge_active_objects(
    memory_objects: list[EntityRecord],
    new_objects: list[EntityRecord],
    *,
    max_memory: int = -1,
    max_aliases: int = -1,
    max_desc: int = -1,
    rejected_names: set[str] | None = None,
) -> list[EntityRecord]:
    """Deprecated compatibility wrapper; use :func:`merge_active_entities`."""
    return merge_active_entities(
        memory_objects,
        new_objects,
        max_memory=max_memory,
        max_aliases=max_aliases,
        max_desc=max_desc,
        rejected_names=rejected_names,
    )
