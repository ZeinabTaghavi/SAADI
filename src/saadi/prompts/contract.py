"""The fixed part of every SAADI prompt: what the parser and the memory rely on.

A prompt profile supplies *guidance*: what counts as an entity in a domain, how
to name and describe it, what a summary should keep. The library wraps that
guidance in the contract below, which fixes the inputs the model sees and the
output format the parsers expect (a JSON object with ``active_entities`` and
``insertions``, exact-substring anchors, and the end markers). A profile can
therefore change the extraction policy but cannot break parsing.

The contract text is sliced from the prompts used for the paper's experiments,
so the ``default`` profile composes to those prompts byte for byte; a test
asserts this.
"""

from __future__ import annotations

CONTRACT_VERSION = "1"

ENTITY_CONTRACT_PREAMBLE = 'You extract active entities from one current text chunk.\n\nGoal:\n- Return `active_entities` and one placement `insertion` for every active entity in CURRENT_CHUNK.\n\nInputs:\n- HIERARCHY_CONTEXT_JSON is the already processed history before CURRENT_CHUNK. It is arranged from coarser summaries to finer L0 chunks, ending as close as possible to the immediate previous chunk.\n- PREVIOUS_ACTIVE_ENTITIES_JSON contains known entity names, aliases, and descriptions from memory that might help for clarification of the current chunk.\n- CURRENT_CHUNK is the only text you extract active entities from.\n\nOutput:\n- Return JSON only with keys `active_entities` and `insertions`.\n- Do not output CURRENT_CHUNK, rewritten chunk text, summaries, or any extra keys.'

ENTITY_CONTRACT_POSTAMBLE = 'Output contract:\n- `insertions` must be [] only when `active_entities` is [].\n- Each insertion must contain:\n  - `name`: exact active entity name from `active_entities`\n  - `anchor_text`: exact mention from CURRENT_CHUNK that receives the parenthetical\n  - `left_context`: exact text immediately before `anchor_text` copied from CURRENT_CHUNK\n  - `right_context`: exact text immediately after `anchor_text` copied from CURRENT_CHUNK'

SUMMARY_CONTRACT_PREAMBLE = 'You summarize one branching-factor group into one concise, factual summary.'

SUMMARY_CONTRACT_POSTAMBLE = 'Output plain text only, no bullets, no markdown, no JSON.\nTerminate with [END_SUMMARY].'

DOCUMENT_SUMMARY_CONTRACT_POSTAMBLE = 'Output plain text only, with no heading, bullets, markdown, JSON, or prefatory language. Terminate with [END_SUMMARY].'

# Phrases a profile must not supply: the contract owns them. Catching them at
# load time keeps a profile from issuing a second, conflicting output format.
ENTITY_GUIDANCE_FORBIDDEN = ("[END_JSON]", "Output contract:", "Return JSON only", "[OUTPUT_SCHEMA]")
SUMMARY_GUIDANCE_FORBIDDEN = ("[END_SUMMARY]",)


def validate_entity_guidance(text: str) -> list[str]:
    """Return the contract phrases found in entity guidance (empty when it is clean)."""

    return [marker for marker in ENTITY_GUIDANCE_FORBIDDEN if marker in str(text or "")]


def validate_summary_guidance(text: str) -> list[str]:
    """Return the contract phrases found in summary guidance (empty when it is clean)."""

    return [marker for marker in SUMMARY_GUIDANCE_FORBIDDEN if marker in str(text or "")]


def compose_entity_system_prompt(guidance: str) -> str:
    """Wrap entity-extraction guidance in the fixed input and output contract."""

    return f"{ENTITY_CONTRACT_PREAMBLE}\n\n{str(guidance or '').strip()}\n\n{ENTITY_CONTRACT_POSTAMBLE}"


def compose_summary_system_prompt(guidance: str) -> str:
    """Wrap fold-summary guidance in the fixed plain-text contract."""

    return f"{SUMMARY_CONTRACT_PREAMBLE}\n{str(guidance or '').strip()}\n{SUMMARY_CONTRACT_POSTAMBLE}"


def compose_document_summary_prompt(guidance: str) -> str:
    """Wrap whole-document summary guidance in the fixed plain-text contract."""

    return f"{str(guidance or '').strip()}\n\n{DOCUMENT_SUMMARY_CONTRACT_POSTAMBLE}"


__all__ = [
    "CONTRACT_VERSION",
    "DOCUMENT_SUMMARY_CONTRACT_POSTAMBLE",
    "ENTITY_CONTRACT_POSTAMBLE",
    "ENTITY_CONTRACT_PREAMBLE",
    "ENTITY_GUIDANCE_FORBIDDEN",
    "SUMMARY_CONTRACT_POSTAMBLE",
    "SUMMARY_CONTRACT_PREAMBLE",
    "SUMMARY_GUIDANCE_FORBIDDEN",
    "compose_document_summary_prompt",
    "compose_entity_system_prompt",
    "compose_summary_system_prompt",
    "validate_entity_guidance",
    "validate_summary_guidance",
]
