from __future__ import annotations

from pathlib import Path

from .base import PreparedDocument

TEXT_SUFFIXES = (".txt", ".md", ".text")


class TextAdapter:
    """Load a UTF-8 text document without changing its content."""

    suffixes = TEXT_SUFFIXES

    def prepare(self, path: str | Path) -> PreparedDocument:
        source = Path(path).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"Text input does not exist: {source}")
        if source.suffix.lower() not in self.suffixes:
            raise ValueError(f"Text input must be one of {', '.join(self.suffixes)}: {source}")
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
