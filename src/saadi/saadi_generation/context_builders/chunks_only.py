"""Context using the original chunks retrieved by Saadi."""

from __future__ import annotations

from dataclasses import dataclass

from .base import BuiltContext, ContextBuilderConfig, QueryContextRequest


@dataclass(slots=True)
class ChunksOnlyContextBuilder:
    """Format Saadi retrieved chunks without folded summaries."""

    config: ContextBuilderConfig
    name: str = "chunks_only"

    def build(self, request: QueryContextRequest) -> BuiltContext:
        if self.config.experiment_name != "chunks_only":
            raise NotImplementedError(
                "ChunksOnlyContextBuilder currently implements only chunks_only. "
                "Select chunks_only or use the folded-context builder."
            )
        items = list(request.retrieved_items)[: max(1, int(self.config.top_k))]
        blocks: list[str] = ["[Retrieved Chunks]"]
        debug_chunks: list[dict[str, object]] = []
        for item in items:
            display_id = item.metadata.get("chunk_index")
            if display_id is None or str(display_id).strip() == "":
                display_id = item.chunk_id
            text = str(item.raw_text or "").strip()
            blocks.append(f"[Chunk {display_id}]\n{text}")
            debug_chunks.append(
                {
                    "rank": int(item.rank),
                    "chunk_id": str(item.chunk_id),
                    "chunk_index": item.metadata.get("chunk_index"),
                    "score": item.score,
                    "text_chars": len(text),
                }
            )
        context_text = "\n\n".join(blocks).strip()
        return BuiltContext(
            context_text=context_text,
            debug_payload={
                "context_builder": self.name,
                "experiment_name": self.config.experiment_name,
                "uses_summaries": False,
                "uses_entity_memory": False,
                "uses_bridge": False,
                "retrieved_chunks": debug_chunks,
            },
        )
