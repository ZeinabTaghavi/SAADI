"""Implements hierarchical and limited-hierarchical summary strategies for stateful chunk canonicalization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..core.canonicalization import parse_canonicalization_response
from ..core.prompts import format_chat_prompt, json_schema_string
from ..core.state import apply_canonicalization_to_state, begin_canonicalization_update
from ..core.types import ChunkRecord, DiscourseState
from ..entities.active_selection import select_active_entities
from ..entities.active_entity_config import normalize_active_entity_min_relevance
from ..llm.base import GenerateParams, LLMBackend
from .base import StrategyCommonConfig

HIER_SCHEMA = {
    "active_entities": [{"name": "string", "description": ["string"], "aliases": ["string"]}],
    "insertions": [
        {
            "name": "same string as the active entity name",
            "anchor_text": "string",
            "left_context": "string",
            "right_context": "string",
        }
    ],
}

HIER_SYSTEM_PROMPT = """
You extract active entities from one current text chunk.

Goal:
- Return `active_entities` and one placement `insertion` for every active entity in CURRENT_CHUNK.

Inputs:
- HIERARCHY_CONTEXT_JSON is the already processed history before CURRENT_CHUNK. It is arranged from coarser summaries to finer L0 chunks, ending as close as possible to the immediate previous chunk.
- PREVIOUS_ACTIVE_ENTITIES_JSON contains known entity names, aliases, and descriptions from memory that might help for clarification of the current chunk.
- CURRENT_CHUNK is the only text you extract active entities from.

Output:
- Return JSON only with keys `active_entities` and `insertions`.
- Do not output CURRENT_CHUNK, rewritten chunk text, summaries, or any extra keys.

Grounding rules:
- Use HIERARCHY_CONTEXT_JSON and PREVIOUS_ACTIVE_ENTITIES_JSON only to resolve names, aliases, and coreference.
- Insertion anchors must be grounded in CURRENT_CHUNK.
- Do not invent facts, attributes, relationships, or events.

Span-copy rules (CRITICAL):
- `anchor_text`, `left_context`, and `right_context` MUST be exact substrings copied from CURRENT_CHUNK.
- Do NOT modify spelling, punctuation, or whitespace when copying spans.
- Use the minimal left/right context necessary to uniquely identify the anchor.

Naming rules:
- If an entity already exists in PREVIOUS_ACTIVE_ENTITIES_JSON, reuse its exact stored name; never rename it or mint a variant, even when a new description appears. Only treat a current mention as that entity when it genuinely refers to the same entity. The rules below govern only how to name a NEW entity.
- Name a new entity by its own standard designation -- the term that denotes the concept itself -- never by a citation or reference marker that merely points to it (e.g., "the model in [12]" is named by the model, never "[12]").
- Give each new entity a unique, stable name that does not collide with a different entity in PREVIOUS_ACTIVE_ENTITIES_JSON. The name need not be an exact substring of CURRENT_CHUNK; it must be a short, clear designation. By default use its standard designation (for named methods, datasets, or models often the acronym, e.g. "BERT", "BM25"; for general concepts the full term, e.g. "large language model").
- If the default name would collide with a different entity already in memory (e.g. two entities share the acronym "ML"), qualify it into a unique short form (e.g. "Meta Learning (ML)") and keep the bare form as an alias.
- Represent each entity once. Acronyms, expansions, and other surface forms of the same entity are aliases of that one entity, never separate entities.
- Always record both the standard full name and the acronym/short form in `aliases`, even when one of them is identical to the name.

Insertion rules:
- `insertions` must contain exactly one entry for each item in `active_entities`, with `name` equal to that active entity's `name`.
- `anchor_text` should be the mention, alias, pronoun, or referring phrase in CURRENT_CHUNK where that entity participates.
- The system places ` (name)` immediately after `anchor_text`: only the canonical name is inserted, and you do not output the inserted text.

`active_entities` rules:
- Return every entity that is active, present, or participating in CURRENT_CHUNK.
- For new or updated entities, include only aliases/descriptions newly supported by CURRENT_CHUNK.
- For entities that are only present with no new aliases/descriptions, include the name with empty aliases and description arrays.
- If no entities are active in CURRENT_CHUNK, return [].
- Descriptions state attributes, roles, or relations of the entity -- never an alternative name for it. Keep them short and factual. A citation or reference marker may be appended to a description that states real content, but must never be a description on its own.

Output contract:
- `insertions` must be [] only when `active_entities` is [].
- Each insertion must contain:
  - `name`: exact active entity name from `active_entities`
  - `anchor_text`: exact mention from CURRENT_CHUNK that receives the parenthetical
  - `left_context`: exact text immediately before `anchor_text` copied from CURRENT_CHUNK
  - `right_context`: exact text immediately after `anchor_text` copied from CURRENT_CHUNK
""".strip()

MEMORY_UPDATE_MODE_NO_HIS_NO_MEM = "no_his_no_mem"
MEMORY_UPDATE_MODE_HIS_NO_MEM = "his_no_mem"
MEMORY_UPDATE_MODE_NO_HIS_ONLY_MEM = "no_his_only_mem"
MEMORY_UPDATE_MODE_HIS_MEM = "his_mem"
MEMORY_UPDATE_MODES = (
    MEMORY_UPDATE_MODE_NO_HIS_NO_MEM,
    MEMORY_UPDATE_MODE_HIS_NO_MEM,
    MEMORY_UPDATE_MODE_NO_HIS_ONLY_MEM,
    MEMORY_UPDATE_MODE_HIS_MEM,
)

_MEMORY_UPDATE_MODE_ALIASES = {
    "nohisnomem": MEMORY_UPDATE_MODE_NO_HIS_NO_MEM,
    "no_history_no_memory": MEMORY_UPDATE_MODE_NO_HIS_NO_MEM,
    "mbr2memnohisnomem": MEMORY_UPDATE_MODE_NO_HIS_NO_MEM,
    "hisnomem": MEMORY_UPDATE_MODE_HIS_NO_MEM,
    "history_no_memory": MEMORY_UPDATE_MODE_HIS_NO_MEM,
    "mbr2memhisnomem": MEMORY_UPDATE_MODE_HIS_NO_MEM,
    "nohisonlymem": MEMORY_UPDATE_MODE_NO_HIS_ONLY_MEM,
    "no_history_only_memory": MEMORY_UPDATE_MODE_NO_HIS_ONLY_MEM,
    "mbr2memnohisonlymem": MEMORY_UPDATE_MODE_NO_HIS_ONLY_MEM,
    "hismem": MEMORY_UPDATE_MODE_HIS_MEM,
    "history_memory": MEMORY_UPDATE_MODE_HIS_MEM,
    "full": MEMORY_UPDATE_MODE_HIS_MEM,
    "default": MEMORY_UPDATE_MODE_HIS_MEM,
    "mbr2memhismem": MEMORY_UPDATE_MODE_HIS_MEM,
}


HISTORY_L0_TEXT_RAW = "raw"
HISTORY_L0_TEXT_ANCHORED = "anchored"
HISTORY_L0_TEXT_OPTIONS = (HISTORY_L0_TEXT_RAW, HISTORY_L0_TEXT_ANCHORED)


def normalize_history_l0_text(value: str | None) -> str:
    """Which text the history shows the model for recent (L0) chunks.

    ``raw``: the original chunk text (default). ``anchored``: the copy in which
    the canonical name is placed after each span anchor, which is also the fold
    input. The saved main-study constructions used ``anchored``.
    """
    raw = str(value or HISTORY_L0_TEXT_RAW).strip().lower().replace("-", "_")
    aliases = {
        "raw": HISTORY_L0_TEXT_RAW,
        "original": HISTORY_L0_TEXT_RAW,
        "anchored": HISTORY_L0_TEXT_ANCHORED,
        "materialized": HISTORY_L0_TEXT_ANCHORED,
        "inserted": HISTORY_L0_TEXT_ANCHORED,
    }
    normalized = aliases.get(raw, raw)
    if normalized not in HISTORY_L0_TEXT_OPTIONS:
        raise ValueError(
            f"Invalid history_l0_text: {value!r}. Expected one of: {', '.join(HISTORY_L0_TEXT_OPTIONS)}."
        )
    return normalized


def normalize_memory_update_mode(value: str | None) -> str:
    raw = str(value or MEMORY_UPDATE_MODE_HIS_MEM).strip().lower().replace("-", "_")
    normalized = _MEMORY_UPDATE_MODE_ALIASES.get(raw.replace("_", ""), _MEMORY_UPDATE_MODE_ALIASES.get(raw, raw))
    if normalized not in MEMORY_UPDATE_MODES:
        raise ValueError(
            f"Invalid memory_update_mode: {value!r}. Expected one of: {', '.join(MEMORY_UPDATE_MODES)}."
        )
    return normalized


_COMMON_ABLATION_OUTPUT_RULES = """
Output:
- Return JSON only with keys `active_entities` and `insertions`.
- Do not output CURRENT_CHUNK, rewritten chunk text, summaries, or any extra keys.

Grounding rules:
- Extract active entities only from CURRENT_CHUNK.
- Context, when supplied, may be used only to resolve names, aliases, and coreference.
- Insertion anchors must be grounded in CURRENT_CHUNK.
- Do not invent facts, attributes, relationships, or events.

Span-copy rules (CRITICAL):
- `anchor_text`, `left_context`, and `right_context` MUST be exact substrings copied from CURRENT_CHUNK.
- Do NOT modify spelling, punctuation, or whitespace when copying spans.
- Use the minimal left/right context necessary to uniquely identify the anchor.

Naming rules for new entities:
- Name a new entity by its own standard designation -- the term that denotes the concept itself -- never by a citation or reference marker that merely points to it (e.g., "the model in [12]" is named by the model, never "[12]").
- Give each new entity a unique, stable name. The name need not be an exact substring of CURRENT_CHUNK; it must be a short, clear designation. By default use its standard designation (for named methods, datasets, or models often the acronym, e.g. "BERT", "BM25"; for general concepts the full term, e.g. "large language model").
- Represent each entity once. Acronyms, expansions, and other surface forms of the same entity are aliases of that one entity, never separate entities.
- Always record both the standard full name and acronym/short form in `aliases`, even when one is identical to the name.

Insertion rules:
- `insertions` must contain exactly one entry for each item in `active_entities`, with the same `name`.
- `anchor_text` should be the mention, alias, pronoun, or referring phrase in CURRENT_CHUNK where that entity participates.
- The system places ` (name)` immediately after `anchor_text`: only the canonical name is inserted, and you do not output the inserted text.

`active_entities` rules:
- Return every entity active, present, or participating in CURRENT_CHUNK.
- For new or updated entities, include only aliases/descriptions newly supported by CURRENT_CHUNK.
- For entities that are only present with no new aliases/descriptions, include the name with empty aliases and description arrays.
- If no entities are active in CURRENT_CHUNK, return [].
- Descriptions state attributes, roles, or relations of the entity -- never an alternative name for it. Keep them short and factual. A citation or reference marker may be appended to a description that states real content, but must never be a description on its own.

Output contract:
- `insertions` must be [] only when `active_entities` is [].
- Each insertion must contain:
  - `name`: exact active entity name from `active_entities`
  - `anchor_text`: exact mention from CURRENT_CHUNK that receives the parenthetical
  - `left_context`: exact text immediately before `anchor_text` copied from CURRENT_CHUNK
  - `right_context`: exact text immediately after `anchor_text` copied from CURRENT_CHUNK
""".strip()

NO_HIS_NO_MEM_SYSTEM_PROMPT = f"""
You extract active entities from one current text chunk in complete isolation.

Memory-update arm: MBR2-MEM-NoHisNoMem.
- Goal: return `active_entities` and one placement `insertion` for every active entity in CURRENT_CHUNK.
- CURRENT_CHUNK is the only information available.
- There is no processed history, hierarchical summary, previous chunk, or previous memory.
- Build this chunk's memory update exactly as if CURRENT_CHUNK were the only chunk in the document.
- Because previous memory is unavailable, apply the new-entity naming rules independently within this chunk.

{_COMMON_ABLATION_OUTPUT_RULES}
""".strip()

HIS_NO_MEM_SYSTEM_PROMPT = f"""
You extract active entities from one current text chunk with processed-history context only.

Memory-update arm: MBR2-MEM-HisNoMem.
- Goal: return `active_entities` and one placement `insertion` for every active entity in CURRENT_CHUNK.
- HIERARCHY_CONTEXT_JSON is the already processed history before CURRENT_CHUNK, arranged from coarser summaries to finer L0 chunks.
- There is no previous entity memory and no retrieved memory entry in this arm.
- Use HIERARCHY_CONTEXT_JSON only to clarify references in CURRENT_CHUNK.
- Because previous entity memory is unavailable, apply the new-entity naming rules using CURRENT_CHUNK and the supplied history only.

{_COMMON_ABLATION_OUTPUT_RULES}
""".strip()

NO_HIS_ONLY_MEM_SYSTEM_PROMPT = f"""
You extract active entities from one current text chunk with retrieved entity-memory context only.

Memory-update arm: MBR2-MEM-NoHisOnlyMem.
- Goal: return `active_entities` and one placement `insertion` for every active entity in CURRENT_CHUNK.
- PREVIOUS_ACTIVE_ENTITIES_JSON contains the most relevant entity-memory entries retrieved from memory built before CURRENT_CHUNK.
- There is no processed history, hierarchical summary, previous chunk, or running summary in this arm.
- Use PREVIOUS_ACTIVE_ENTITIES_JSON only to clarify references in CURRENT_CHUNK.
- If an entity already exists in PREVIOUS_ACTIVE_ENTITIES_JSON, reuse its exact stored name. Never rename it or mint a variant unless the current mention is genuinely a different entity.
- Give a new entity a name that does not collide with a different entity in PREVIOUS_ACTIVE_ENTITIES_JSON. If a standard short name would collide (e.g. two entities share "ML"), qualify the new name (e.g. "Meta Learning (ML)") and keep the bare form as an alias.

{_COMMON_ABLATION_OUTPUT_RULES}
""".strip()

MEMORY_UPDATE_SYSTEM_PROMPTS = {
    MEMORY_UPDATE_MODE_NO_HIS_NO_MEM: NO_HIS_NO_MEM_SYSTEM_PROMPT,
    MEMORY_UPDATE_MODE_HIS_NO_MEM: HIS_NO_MEM_SYSTEM_PROMPT,
    MEMORY_UPDATE_MODE_NO_HIS_ONLY_MEM: NO_HIS_ONLY_MEM_SYSTEM_PROMPT,
    # Preserve the exact existing default prompt for the full/reference arm.
    MEMORY_UPDATE_MODE_HIS_MEM: HIER_SYSTEM_PROMPT,
}

ROLLUP_SYSTEM_PROMPT = """
You summarize one branching-factor group into one concise, factual summary.
Keep core entities and events. Do not add speculation.
Output plain text only, no bullets, no markdown, no JSON.
Terminate with [END_SUMMARY].
""".strip()




def _build_hierarchical_user_prompt(
    *,
    hierarchy_context: Any,
    filtered_entities: list[dict],
    chunk_text: str,
    schema: dict,
) -> str:
    return (
        "[HIERARCHY_CONTEXT_JSON]\n"
        f"{json.dumps(hierarchy_context, ensure_ascii=False)}\n\n"
        "[PREVIOUS_ACTIVE_ENTITIES_JSON]\n"
        f"{json.dumps(filtered_entities, ensure_ascii=False)}\n\n"
        "[CURRENT_CHUNK]\n"
        f"{chunk_text}\n\n"
        "[OUTPUT_SCHEMA]\n"
        f"{json_schema_string(schema)}\n\n"
        "[INSTRUCTIONS]\n"
        "Return JSON only, with keys active_entities and insertions and no other text. "
        "Include exactly one insertion for each active entity, and do not include any inserted-text field. "
        "Then [END_JSON]."
    )


def _build_memory_ablation_user_prompt(
    *,
    memory_update_mode: str,
    hierarchy_context: Any | None,
    filtered_entities: list[dict] | None,
    chunk_text: str,
    schema: dict,
) -> str:
    mode = normalize_memory_update_mode(memory_update_mode)
    mode_descriptions = {
        MEMORY_UPDATE_MODE_NO_HIS_NO_MEM: "Current chunk only; no history and no previous memory.",
        MEMORY_UPDATE_MODE_HIS_NO_MEM: "Current chunk plus hierarchical history; no previous memory.",
        MEMORY_UPDATE_MODE_NO_HIS_ONLY_MEM: "Current chunk plus retrieved previous memory; no history.",
    }
    sections = [
        "[MEMORY_UPDATE_MODE]\n"
        f"{mode}: {mode_descriptions[mode]}"
    ]
    if hierarchy_context is not None:
        sections.append(
            "[HIERARCHY_CONTEXT_JSON]\n"
            f"{json.dumps(hierarchy_context, ensure_ascii=False)}"
        )
    if filtered_entities is not None:
        sections.append(
            "[PREVIOUS_ACTIVE_ENTITIES_JSON]\n"
            f"{json.dumps(filtered_entities, ensure_ascii=False)}"
        )
    sections.extend(
        [
            "[CURRENT_CHUNK]\n" + chunk_text,
            "[OUTPUT_SCHEMA]\n" + json_schema_string(schema),
            (
                "[INSTRUCTIONS]\n"
                "Return JSON only, with keys active_entities and insertions and no other text. "
                "Include exactly one insertion for each active entity, and do not include any inserted-text field. "
                "Then [END_JSON]."
            ),
        ]
    )
    return "\n\n".join(sections)


@dataclass
class HierarchicalSummaryStrategy(StrategyCommonConfig):
    name: str = "hierarchical_summary"
    branching_factor: int = 5
    # Backward-compatible alias for older configs/tests. New configs should use branching_factor.
    k: int | None = None
    tokenizer_for_chat_template: object | None = None
    system_prompt: str = HIER_SYSTEM_PROMPT
    rollup_system_prompt: str = ROLLUP_SYSTEM_PROMPT
    memory_update_mode: str = MEMORY_UPDATE_MODE_HIS_MEM
    history_l0_text: str = HISTORY_L0_TEXT_RAW

    def __post_init__(self) -> None:
        raw_b = self.k if self.k is not None else self.branching_factor
        try:
            b = int(raw_b)
        except Exception:
            b = 5
        if b <= 0:
            b = 5
        self.branching_factor = b
        self.k = b
        self.memory_update_mode = normalize_memory_update_mode(self.memory_update_mode)
        self.history_l0_text = normalize_history_l0_text(self.history_l0_text)
        self.active_entity_min_relevance = normalize_active_entity_min_relevance(
            self.active_entity_min_relevance
        )

    @property
    def uses_history_context(self) -> bool:
        return self.memory_update_mode in {
            MEMORY_UPDATE_MODE_HIS_NO_MEM,
            MEMORY_UPDATE_MODE_HIS_MEM,
        }

    @property
    def uses_memory_context(self) -> bool:
        return self.memory_update_mode in {
            MEMORY_UPDATE_MODE_NO_HIS_ONLY_MEM,
            MEMORY_UPDATE_MODE_HIS_MEM,
        }

    @property
    def stores_hierarchy(self) -> bool:
        return self.uses_history_context

    def effective_system_prompt(self) -> str:
        if self.memory_update_mode == MEMORY_UPDATE_MODE_HIS_MEM:
            return self.system_prompt
        if self.system_prompt != HIER_SYSTEM_PROMPT:
            return self.system_prompt
        return MEMORY_UPDATE_SYSTEM_PROMPTS[self.memory_update_mode]

    def _postprocess_output(self, *, chunk: ChunkRecord, output):
        # Final chunk materialization is handled in canonicalization validation.
        return output

    def _select_entities(self, chunk: ChunkRecord, state: DiscourseState) -> list[dict]:
        return select_active_entities(
            state.entity_memory,
            chunk.raw_text,
            max_entities=self.max_entities,
            max_aliases=self.max_aliases,
            max_desc=self.max_desc,
            selector=self.active_entity_selector,
            min_relevance_score=self.active_entity_min_relevance,
        )

    def _cycle_to_level(self, cycle: int) -> int | None:
        if cycle < 1:
            return None
        if cycle == 1:
            return 0
        b = int(self.branching_factor)
        if b <= 1:
            return None
        cur = 1
        level = 0
        while cur < cycle:
            cur *= b
            level += 1
        return level if cur == cycle else None

    def _max_level_limit(self) -> int | None:
        return None

    def _max_available_level(self, state: DiscourseState, current_step: int) -> int:
        if current_step <= 0:
            return 0
        out = 0
        for cycle, by_step in state.hierarchy_summaries.items():
            if not by_step:
                continue
            level = self._cycle_to_level(int(cycle))
            if level is None:
                continue
            latest = max(by_step.keys())
            if latest <= current_step - 1:
                out = max(out, level)
        limit = self._max_level_limit()
        if limit is not None:
            out = min(out, int(limit))
        return max(0, out)

    def build_hierarchy_step_plan(
        self,
        *,
        state: DiscourseState,
        current_step: int,
    ) -> dict[str, list[int]]:
        """
        Build a top-down, non-overlapping step selection plan:
        - Start from highest available level and select up to b latest steps <= current_step-1.
        - Move to lower levels and select steps strictly greater than the last selected step.
        - Always include current_step-1 as L0 when available, even if a boundary summary also covers it.
        """
        if current_step <= 0:
            return {}
        upper = current_step - 1
        covered_until = -1
        max_level = self._max_available_level(state, current_step)
        plan: dict[str, list[int]] = {}
        b = int(self.branching_factor)
        for level in range(max_level, -1, -1):
            cycle = b**level
            by_step = state.hierarchy_summaries.get(cycle, {})
            if not by_step:
                continue
            candidates = [s for s in sorted(by_step.keys()) if covered_until < s <= upper]
            if not candidates:
                continue
            selected = candidates[-b:] if b > 0 else candidates
            if not selected:
                continue
            plan[f"L{level}"] = selected
            covered_until = max(selected)

        prev_step = current_step - 1
        l0 = state.hierarchy_summaries.get(1, {})
        if prev_step in l0:
            selected_l0 = list(plan.get("L0", []))
            if prev_step not in selected_l0:
                selected_l0.append(prev_step)
                selected_l0 = sorted(selected_l0)
            if selected_l0:
                plan["L0"] = selected_l0
        return plan

    @property
    def shows_raw_l0_history(self) -> bool:
        return self.history_l0_text == HISTORY_L0_TEXT_RAW

    def _l0_context_text(self, state: DiscourseState, step: int, anchored_text: str) -> str:
        """Text shown to the model for one recent chunk in the L0 history.

        The anchored copy (``hierarchy_summaries[1]``) is always what the folds
        summarize. In ``raw`` mode the model instead sees the original chunk text
        when the state carries it; a state restored from an older snapshot that
        lacks raw texts falls back to the anchored copy.
        """
        if self.shows_raw_l0_history:
            raw = state.raw_chunk_texts.get(int(step))
            if raw is not None:
                return str(raw)
        return str(anchored_text)

    def _inject_previous_l0_fallback(
        self,
        *,
        state: DiscourseState,
        current_step: int,
        step_plan: dict[str, list[int]],
        resolved: dict[str, list[dict[str, Any]]],
    ) -> None:
        """Keep history unified even if a restored state lacks L0 but still has prev_chunk_text."""
        if current_step <= 0 or not state.prev_chunk_text:
            return
        prev_step = current_step - 1
        if any(int(row.get("step", -1)) == prev_step for row in resolved.get("L0", [])):
            return
        selected_l0 = list(step_plan.get("L0", []))
        if prev_step not in selected_l0:
            selected_l0.append(prev_step)
            step_plan["L0"] = sorted(selected_l0)
        text = self._l0_context_text(state, prev_step, state.prev_chunk_text)
        resolved.setdefault("L0", []).append({"step": int(prev_step), "text": text})
        resolved["L0"] = sorted(resolved["L0"], key=lambda row: int(row.get("step", -1)))

    def resolve_hierarchy_step_plan(
        self,
        *,
        state: DiscourseState,
        step_plan: dict[str, list[int]],
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Replace planned steps with actual hierarchy text entries.
        Output shape: {"L2": [{"step": 24, "text": "..."}, ...], ...}
        """
        out: dict[str, list[dict[str, Any]]] = {}
        for level_label, steps in step_plan.items():
            try:
                level = int(level_label[1:])
            except Exception:
                continue
            cycle = int(self.branching_factor) ** level
            by_step = state.hierarchy_summaries.get(cycle, {})
            rows: list[dict[str, Any]] = []
            for s in steps:
                if s in by_step:
                    text = self._l0_context_text(state, s, by_step[s]) if level == 0 else str(by_step[s])
                    rows.append({"step": int(s), "text": text})
            if rows:
                out[level_label] = rows
        return out

    def _hierarchy_context(self, state: DiscourseState, current_step: int) -> dict[str, Any]:
        step_plan = self.build_hierarchy_step_plan(
            state=state,
            current_step=current_step,
        )
        resolved = self.resolve_hierarchy_step_plan(state=state, step_plan=step_plan)
        self._inject_previous_l0_fallback(
            state=state,
            current_step=current_step,
            step_plan=step_plan,
            resolved=resolved,
        )
        ctx: dict[str, Any] = {
            "mode": self.name,
            "branching_factor": int(self.branching_factor),
            "current_step": int(current_step),
            "description": (
                "Top-down processed history before CURRENT_CHUNK. Higher L values are coarser summaries; "
                + (
                    "L0 values are the original text of recent chunks. "
                    if self.shows_raw_l0_history
                    else "L0 values are recent materialized chunks. "
                )
                + "Boundary steps may include the immediate previous "
                "L0 even when a higher-level summary also covers it."
            ),
            "selected_steps_by_level": step_plan,
            "selected_context_by_level": resolved,
        }
        return ctx

    def build_prompt(self, chunk: ChunkRecord, state: DiscourseState) -> str:
        # Keep the legacy/default branch byte-for-byte compatible with the
        # existing MBR2 memory-construction prompt.
        if self.memory_update_mode == MEMORY_UPDATE_MODE_HIS_MEM:
            filtered_entities = self._select_entities(chunk, state)
            hierarchy_context = self._hierarchy_context(state, int(chunk.chunk_index))
            user_prompt = _build_hierarchical_user_prompt(
                hierarchy_context=hierarchy_context,
                filtered_entities=filtered_entities,
                chunk_text=chunk.raw_text,
                schema=HIER_SCHEMA,
            )
        else:
            filtered_entities = self._select_entities(chunk, state) if self.uses_memory_context else None
            hierarchy_context = (
                self._hierarchy_context(state, int(chunk.chunk_index))
                if self.uses_history_context
                else None
            )
            user_prompt = _build_memory_ablation_user_prompt(
                memory_update_mode=self.memory_update_mode,
                hierarchy_context=hierarchy_context,
                filtered_entities=filtered_entities,
                chunk_text=chunk.raw_text,
                schema=HIER_SCHEMA,
            )
        return format_chat_prompt(
            tokenizer=self.tokenizer_for_chat_template,
            system_prompt=self.effective_system_prompt(),
            user_prompt=user_prompt,
        )

    def _fallback_rollup(self, items: list[str], *, max_words: int = 120) -> str:
        joined = " ".join(i.strip() for i in items if isinstance(i, str) and i.strip())
        words = joined.split()
        return " ".join(words[:max_words])

    def _rollup_user_prompt(self, *, lower_cycle: int, next_cycle: int, step: int, items: list[str]) -> str:
        return (
            "[ROLLUP_META]\n"
            f'{json.dumps({"branching_factor": self.branching_factor, "lower_cycle": lower_cycle, "next_cycle": next_cycle, "step": step}, ensure_ascii=False)}\n\n'
            "[ROLLUP_ITEMS_JSON]\n"
            f"{json.dumps(items, ensure_ascii=False)}\n\n"
            "[INSTRUCTIONS]\n"
            "Summarize the items into one concise sentence (<= 60 words). "
            "Output plain text only, then [END_SUMMARY]."
        )

    def _rollup_with_backend(
        self,
        *,
        items: list[str],
        lower_cycle: int,
        next_cycle: int,
        step: int,
        backend: LLMBackend | None,
        generate: GenerateParams | None,
    ) -> str | None:
        if backend is None:
            return None
        g = generate or GenerateParams()
        user_prompt = self._rollup_user_prompt(lower_cycle=lower_cycle, next_cycle=next_cycle, step=step, items=items)
        prompt = format_chat_prompt(
            tokenizer=self.tokenizer_for_chat_template,
            system_prompt=self.rollup_system_prompt,
            user_prompt=user_prompt,
        )
        try:
            raw = backend.generate_batch(
                [prompt],
                max_tokens=min(int(g.max_tokens), 256),
                temperature=float(g.temperature),
                stop=["[END_SUMMARY]", "<|im_end|>", "<|endoftext|>"],
            )[0]
        except Exception:
            return None
        text = str(raw or "").strip()
        if not text:
            return None
        if "[END_SUMMARY]" in text:
            text = text.split("[END_SUMMARY]", 1)[0].strip()
        return text or None

    def build_rollup_prompt(self, *, items: list[str], lower_cycle: int, next_cycle: int, step: int) -> str:
        user_prompt = self._rollup_user_prompt(
            lower_cycle=lower_cycle, next_cycle=next_cycle, step=step, items=items
        )
        return format_chat_prompt(
            tokenizer=self.tokenizer_for_chat_template,
            system_prompt=self.rollup_system_prompt,
            user_prompt=user_prompt,
        )

    def run_rollup_prompts(
        self,
        *,
        prompts: list[str],
        backend: LLMBackend | None,
        generate: GenerateParams | None,
    ) -> list[str | None]:
        """Run a batch of rollup prompts; returns one cleaned summary (or None) per prompt."""
        if backend is None or not prompts:
            return [None] * len(prompts)
        g = generate or GenerateParams()
        try:
            raws = backend.generate_batch(
                prompts,
                max_tokens=min(int(g.max_tokens), 256),
                temperature=float(g.temperature),
                stop=["[END_SUMMARY]", "<|im_end|>", "<|endoftext|>"],
            )
        except Exception:
            return [None] * len(prompts)
        out: list[str | None] = []
        for raw in raws:
            text = str(raw or "").strip()
            if "[END_SUMMARY]" in text:
                text = text.split("[END_SUMMARY]", 1)[0].strip()
            out.append(text or None)
        return out

    def apply_output(
        self,
        chunk: ChunkRecord,
        state: DiscourseState,
        raw_model_text: str,
        *,
        backend: LLMBackend | None = None,
        generate: GenerateParams | None = None,
    ):
        output = parse_canonicalization_response(
            raw_model_text,
            fallback_chunk_text=chunk.raw_text,
            allow_linking_sentence=False,
            max_entities=self.max_entities,
            max_aliases=self.max_aliases,
            max_desc=self.max_desc,
        )
        output = self._postprocess_output(chunk=chunk, output=output)

        def _rollup_fn(items: list[str], lower_cycle: int, next_cycle: int, step: int) -> str | None:
            summary = self._rollup_with_backend(
                items=items,
                lower_cycle=lower_cycle,
                next_cycle=next_cycle,
                step=step,
                backend=backend,
                generate=generate,
            )
            if summary:
                return summary
            return self._fallback_rollup(items)

        new_state = apply_canonicalization_to_state(
            state,
            output,
            chunk_step=int(chunk.chunk_index),
            hierarchy_l0_text=output.updated_chunk_text,
            hierarchy_rollup_fn=_rollup_fn,
            hierarchy_branching_factor=(int(self.branching_factor) if self.stores_hierarchy else None),
            hierarchy_max_level=self._max_level_limit(),
            raw_chunk_text=chunk.raw_text,
        )
        return output, new_state

    def apply_output_deferred(
        self,
        chunk: ChunkRecord,
        state: DiscourseState,
        raw_model_text: str,
    ):
        """Like :meth:`apply_output` but defers hierarchy rollups for cross-document batching.

        Returns ``(output, new_state, rollup_gen)`` where ``rollup_gen`` is a generator
        (or ``None``) yielding rollup requests. Callers should advance the generators of
        many documents together so the rollup LLM calls for the same cascade level are
        issued in a single batch (see ``_drive_rollups_batched`` in the ingest pipeline).
        """
        output = parse_canonicalization_response(
            raw_model_text,
            fallback_chunk_text=chunk.raw_text,
            allow_linking_sentence=False,
            max_entities=self.max_entities,
            max_aliases=self.max_aliases,
            max_desc=self.max_desc,
        )
        output = self._postprocess_output(chunk=chunk, output=output)
        new_state, gen = begin_canonicalization_update(
            state,
            output,
            chunk_step=int(chunk.chunk_index),
            hierarchy_l0_text=output.updated_chunk_text,
            hierarchy_branching_factor=(int(self.branching_factor) if self.stores_hierarchy else None),
            hierarchy_max_level=self._max_level_limit(),
            raw_chunk_text=chunk.raw_text,
        )
        return output, new_state, gen


@dataclass
class LimitedHierarchicalSummaryStrategy(HierarchicalSummaryStrategy):
    name: str = "limited_hierarchical_summary"
    max_hierarchy_level: int = 2

    def _max_level_limit(self) -> int | None:
        return int(self.max_hierarchy_level)

    def apply_output(
        self,
        chunk: ChunkRecord,
        state: DiscourseState,
        raw_model_text: str,
        *,
        backend: LLMBackend | None = None,
        generate: GenerateParams | None = None,
    ):
        output = parse_canonicalization_response(
            raw_model_text,
            fallback_chunk_text=chunk.raw_text,
            allow_linking_sentence=False,
            max_entities=self.max_entities,
            max_aliases=self.max_aliases,
            max_desc=self.max_desc,
        )
        output = self._postprocess_output(chunk=chunk, output=output)

        def _rollup_fn(items: list[str], lower_cycle: int, next_cycle: int, step: int) -> str | None:
            summary = self._rollup_with_backend(
                items=items,
                lower_cycle=lower_cycle,
                next_cycle=next_cycle,
                step=step,
                backend=backend,
                generate=generate,
            )
            if summary:
                return summary
            return self._fallback_rollup(items)

        new_state = apply_canonicalization_to_state(
            state,
            output,
            chunk_step=int(chunk.chunk_index),
            hierarchy_l0_text=output.updated_chunk_text,
            hierarchy_rollup_fn=_rollup_fn,
            hierarchy_branching_factor=(int(self.branching_factor) if self.stores_hierarchy else None),
            hierarchy_max_level=self.max_hierarchy_level,
            raw_chunk_text=chunk.raw_text,
        )
        return output, new_state
