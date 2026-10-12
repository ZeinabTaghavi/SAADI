from __future__ import annotations

from pathlib import Path

from .base import PreparedDocument


class TextAdapter:
    """Load an ordinary UTF-8 text document without changing its content."""

    suffix = ".txt"

    def prepare(self, path: str | Path) -> PreparedDocument:
        source = Path(path).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"Text input does not exist: {source}")
        if source.suffix.lower() != self.suffix:
            raise ValueError(f"Text workflow requires a .txt file: {source}")
        text = source.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError(f"Text input is empty: {source}")
        return PreparedDocument(
            document_id=source.stem,
            text=text,
            source_path=source,
            input_format="text",
            metadata={"source_suffix": source.suffix.lower()},
        )
