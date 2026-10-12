from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(slots=True)
class PreparedDocument:
    """A source file normalized into the plain-text form consumed by SAADI."""

    document_id: str
    text: str
    source_path: Path
    input_format: str
    metadata: dict[str, Any] = field(default_factory=dict)


class InputAdapter(Protocol):
    def prepare(self, path: str | Path) -> PreparedDocument: ...
