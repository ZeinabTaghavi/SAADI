"""The exact prompt of the ledger profile's one LLM call per chunk.

Order of the user prompt (spec, section E): the chunk text; the summary
history (as now); the recalled records in two labeled groups, "found by word
overlap" (the existing 5 slots) and "its name appears in this chunk" (ledger
recall, up to 5; a stub shows its name and stored sentence only); the numbered
candidate list from the small model. The output is the decision block, then the
extraction JSON in the existing format, then ``[END_JSON]``.

The step 1 report prints :data:`LEDGER_SYSTEM_PROMPT` and one filled user
prompt verbatim before any GPU time is spent.
"""

from __future__ import annotations

import json
from typing import Any

from ..core.prompts import json_schema_string
from .accounting import DECISIONS_CLOSE, DECISIONS_OPEN

# The same output schema as the published prompt (a copy, so this module does
# not import the strategy module).
LEDGER_SCHEMA = {
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

SECTION_CHUNK = "[CURRENT_CHUNK]"
SECTION_HISTORY = "[HIERARCHY_CONTEXT_JSON]"
SECTION_OVERLAP = "[RECALLED_RECORDS_FOUND_BY_WORD_OVERLAP_JSON]"
SECTION_LEDGER = "[RECALLED_RECORDS_WHOSE_NAME_APPEARS_IN_THIS_CHUNK_JSON]"
SECTION_CANDIDATES = "[CANDIDATES]"
SECTION_SCHEMA = "[OUTPUT_SCHEMA]"
SECTION_INSTRUCTIONS = "[INSTRUCTIONS]"

LEDGER_SYSTEM_PROMPT = f"""
You maintain the entity ledger of one long document while reading one current text chunk. Two jobs, in this order: account for every named candidate in the chunk, then describe the entities that matter.

Inputs:
- CURRENT_CHUNK is the only text you extract from.
- HIERARCHY_CONTEXT_JSON is the already processed history before CURRENT_CHUNK, arranged from coarser summaries to finer L0 chunks, ending as close as possible to the immediate previous chunk.
- RECALLED_RECORDS_FOUND_BY_WORD_OVERLAP_JSON holds records from memory selected by word overlap with CURRENT_CHUNK, with their known aliases and descriptions.
- RECALLED_RECORDS_WHOSE_NAME_APPEARS_IN_THIS_CHUNK_JSON holds records and stubs whose stored name or alias matches a candidate in CURRENT_CHUNK. A stub is a name that was seen earlier but never described; it shows only its name and the sentence it appeared in.
- CANDIDATES is the numbered list of named spans that a small NER model found in CURRENT_CHUNK, with the span type it guessed. The list is a floor, not a ceiling: you may describe entities it missed, such as descriptive names ("the Lantern Keeper").
- Every record and stub carries an id such as E0012. Ids are only used in the decision block.

Output, in this order and nothing else:
1. The decision block: one line per candidate, in the given order and numbering, between {DECISIONS_OPEN} and {DECISIONS_CLOSE}. Exactly one of these three forms:
   <n>. <candidate> -> NEW
   <n>. <candidate> -> SAME AS <id> (<one-line reason>)
   <n>. <candidate> -> NOT AN ENTITY
   - NEW: the candidate names a person, place, organization, group, or named object that has no record or stub yet.
   - SAME AS <id>: the candidate refers to a record or stub shown in one of the two recalled groups. Give its id and a one-line reason grounded in CURRENT_CHUNK or the history (for example: "the chunk says Maysunah's family name was Sinchipuma"). A shared surname alone is not enough to join two names. A different title (Mr. against Mrs.), a different given name, or a different gender count against joining. When in doubt, answer NEW.
   - NOT AN ENTITY: the span is not an entity: a common word, a sentence-initial word, a title without a name, a quotation, a misread span.
   Every candidate gets exactly one line. Do not skip, merge, reorder, or renumber candidates, and do not add lines for names that are not candidates.
2. The extraction block: a JSON object with keys `active_entities` and `insertions`.
   - Describe at most 5 records: the entities in CURRENT_CHUNK that matter most, whether new or recalled. A candidate marked NEW that you do not describe is kept as a stub automatically, and a candidate marked SAME AS is linked to its record automatically; you never need to list an entity only to keep it.
   - Naming: for a recalled record or stub, reuse its exact stored name; never rename it or mint a variant. Name a new entity by the form that appears in CURRENT_CHUNK, including a title when the chunk uses it (e.g. "Doctor Sinchipuma"). Represent each entity once.
   - Aliases: exact names only: another name of the entity, or a title plus its name, copied as it appears in the text. Never a descriptive phrase ("the cartographer" is a description, not an alias). Include an alias only when CURRENT_CHUNK or the history supports it.
   - Descriptions: short factual statements of distinguishing attributes only: occupation, family relation, residence, owned object, role, origin. Not passing events, which belong to the chunk history. Never an alternative name. Include only descriptions that CURRENT_CHUNK supports.
   - For an entity that is present but gains no new alias or description, include its name with empty `aliases` and `description` arrays, or leave it to the decision block.
3. Then [END_JSON].

Grounding rules:
- Use HIERARCHY_CONTEXT_JSON and the recalled records only to resolve names, aliases, and coreference.
- Insertion anchors must be grounded in CURRENT_CHUNK.
- Do not invent facts, attributes, relationships, or events.

Span-copy rules (CRITICAL):
- `anchor_text`, `left_context`, and `right_context` MUST be exact substrings copied from CURRENT_CHUNK.
- Do NOT modify spelling, punctuation, or whitespace when copying spans.
- Use the minimal left/right context necessary to uniquely identify the anchor.

Insertion rules:
- `insertions` must contain exactly one entry for each item in `active_entities`, with `name` equal to that active entity's `name`.
- `anchor_text` should be the mention, alias, pronoun, or referring phrase in CURRENT_CHUNK where that entity participates.
- The system places ` (name)` immediately after `anchor_text`: only the canonical name is inserted, and you do not output the inserted text.

Output contract:
- Do not output CURRENT_CHUNK, rewritten chunk text, summaries, or any extra keys.
- `insertions` must be [] only when `active_entities` is [].
- Each insertion must contain:
  - `name`: exact active entity name from `active_entities`
  - `anchor_text`: exact mention from CURRENT_CHUNK that receives the parenthetical
  - `left_context`: exact text immediately before `anchor_text` copied from CURRENT_CHUNK
  - `right_context`: exact text immediately after `anchor_text` copied from CURRENT_CHUNK
""".strip()

INSTRUCTIONS_TEXT = (
    f"First write the decision block: one line per candidate, in order, between {DECISIONS_OPEN} and {DECISIONS_CLOSE}, "
    "each line of the form `<n>. <candidate> -> NEW` or `<n>. <candidate> -> SAME AS <id> (<reason>)` or "
    "`<n>. <candidate> -> NOT AN ENTITY`. "
    "Then return the JSON object with keys active_entities and insertions for at most 5 records you describe, "
    "with exactly one insertion for each active entity and no inserted-text field. "
    "Then [END_JSON]."
)


def render_candidates(candidates: list[dict[str, Any]]) -> str:
    if not candidates:
        return "(none)"
    return "\n".join(f"{i + 1}. {c['text']} ({c.get('label', 'span')})" for i, c in enumerate(candidates))


def build_ledger_user_prompt(
    *,
    chunk_text: str,
    hierarchy_context: Any,
    overlap_entries: list[dict[str, Any]],
    ledger_entries: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    schema: dict[str, Any] | None = None,
) -> str:
    return (
        f"{SECTION_CHUNK}\n{chunk_text}\n\n"
        f"{SECTION_HISTORY}\n{json.dumps(hierarchy_context, ensure_ascii=False)}\n\n"
        f"{SECTION_OVERLAP}\n{json.dumps(overlap_entries, ensure_ascii=False)}\n\n"
        f"{SECTION_LEDGER}\n{json.dumps(ledger_entries, ensure_ascii=False)}\n\n"
        f"{SECTION_CANDIDATES}\n{render_candidates(candidates)}\n\n"
        f"{SECTION_SCHEMA}\n{json_schema_string(schema or LEDGER_SCHEMA)}\n\n"
        f"{SECTION_INSTRUCTIONS}\n{INSTRUCTIONS_TEXT}"
    )


__all__ = [
    "INSTRUCTIONS_TEXT",
    "LEDGER_SCHEMA",
    "LEDGER_SYSTEM_PROMPT",
    "build_ledger_user_prompt",
    "render_candidates",
]
