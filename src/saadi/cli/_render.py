"""Plain-text rendering of API results for the command line."""

from __future__ import annotations

from collections.abc import Iterable

from ..document import Answer, IndexReport, MemoryEntity, RetrievedChunk


def shorten(text: str, limit: int = 240) -> str:
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def format_index_report(report: IndexReport) -> str:
    levels = ", ".join(f"L{level}" for level in report.history_levels) or "none"
    return "\n".join(
        [
            f"Indexed {report.document_id}: {report.chunks} chunks ({report.new_chunks} new), "
            f"{report.entities} entities, history levels {levels}",
            f"Model calls: {report.model_calls} ({report.prompts_sent} prompts)",
            f"State: {report.state_dir}",
        ]
        + (
            [f"Rejected entity proposals: {report.rejected_entities} (anchor text not found verbatim in the chunk)"]
            if report.rejected_entities
            else []
        )
    )


def format_memory(entities: Iterable[MemoryEntity], limit: int = 10) -> str:
    items = list(entities)
    if not items:
        return "Entity memory: empty"
    lines = [f"Entity memory ({len(items)} entities" + (f", showing {limit}" if len(items) > limit else "") + "):"]
    for entity in items[:limit]:
        aliases = f" [{', '.join(entity.aliases)}]" if entity.aliases else ""
        descriptions = "; ".join(entity.descriptions)
        lines.append(f"- {entity.name}{aliases}" + (f": {shorten(descriptions, 160)}" if descriptions else ""))
    return "\n".join(lines)


def format_chunks(chunks: Iterable[RetrievedChunk], limit: int = 240) -> str:
    items = list(chunks)
    if not items:
        return "Top chunks: none"
    lines = ["Top chunks:"]
    for chunk in items:
        score = f"{chunk.score:.3f}" if chunk.score is not None else "-"
        index = f"chunk {chunk.chunk_index}" if chunk.chunk_index is not None else chunk.chunk_id
        lines.append(f"{chunk.rank}. {index} (score {score}): {shorten(chunk.text, limit)}")
    return "\n".join(lines)


def format_answer(answer: Answer) -> str:
    header = f"Answer: {answer.text}" if answer.text else f"Answer: (empty, status {answer.status})"
    return "\n".join([f"Question: {answer.question}", header, "", format_chunks(answer.chunks)])


__all__ = ["format_answer", "format_chunks", "format_index_report", "format_memory", "shorten"]
