"""Input adapters turn a source file into the plain text that SAADI indexes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(slots=True)
class PreparedDocument:
    """A source normalized into the plain-text form consumed by the pipeline."""

    document_id: str
    text: str
    source_path: Path | None
    input_format: str
    metadata: dict[str, Any] = field(default_factory=dict)


class InputAdapter(Protocol):
    def prepare(self, path: str | Path) -> PreparedDocument: ...
