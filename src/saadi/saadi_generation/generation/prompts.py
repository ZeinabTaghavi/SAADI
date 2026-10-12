"""Prompt templates for future SAADI generation experiments."""

from __future__ import annotations

DEFAULT_SAADI_GENERATION_SYSTEM_PROMPT = (
    "Answer the question using only the provided context. "
    "If the context does not contain the answer, say that you do not know. "
    "Keep the answer concise."
)

DEFAULT_SAADI_GENERATION_USER_TEMPLATE = """Context:
{context}

Question:
{question}{choices_block}

Answer:"""

SAADI_GENERATED_BRIDGE_PROMPT_TEMPLATE = """Question:
{query}
Previous retrieved chunk:
{chunk_a}
Intermediate summaries:
{summaries}
Next retrieved chunk:
{chunk_b}
Task:
Generate a very short factual bridge describing the key entities,
events, or transitions connecting the previous and next retrieved regions.
Constraints:
- Maximum 30 tokens
- Do not answer the question
- Do not repeat chunk content
- Keep only continuity-relevant information
- Prefer entities, actions, and transitions"""


def format_generated_bridge_prompt(
    *,
    query: str,
    chunk_a: str,
    summaries: str,
    chunk_b: str,
) -> str:
    return SAADI_GENERATED_BRIDGE_PROMPT_TEMPLATE.format(
        query=str(query or "").strip(),
        chunk_a=str(chunk_a or "").strip(),
        summaries=str(summaries or "").strip(),
        chunk_b=str(chunk_b or "").strip(),
    )


def format_choices_block(qa_entry: dict) -> str:
    choices = qa_entry.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    lines = ["", "", "Answer choices:"]
    for index, choice in enumerate(choices, start=1):
        text = str(choice or "").strip()
        if text:
            lines.append(f"{index}. {text}")
    if len(lines) <= 3:
        return ""
    lines.append("")
    lines.append("If answer choices are provided, select the best answer and include the answer text.")
    return "\n".join(lines)
