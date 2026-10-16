"""Turn a source into the plain text SAADI indexes.

:func:`prepare_document` accepts inline text, a ``.txt``/``.md`` file, a
role-labelled conversation ``.csv``, or an already prepared document.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import InputAdapter, PreparedDocument
from .conversation_csv import ConversationCsvAdapter
from .text import TEXT_SUFFIXES, TextAdapter

KNOWN_SUFFIXES = (*TEXT_SUFFIXES, ".csv")


def looks_like_path(value: str) -> bool:
    """A single-line string ending in a known document suffix is treated as a file path."""

    text = str(value)
    if "\n" in text or len(text) > 4096:
        return False
    return Path(text.strip()).suffix.lower() in KNOWN_SUFFIXES


def adapter_for(path: Path, *, kind: str | None = None, **options: Any) -> InputAdapter:
    resolved = (kind or ("conversation" if path.suffix.lower() == ".csv" else "text")).strip().lower()
    if resolved == "conversation":
        return ConversationCsvAdapter(**options)
    if resolved == "text":
        if options:
            raise TypeError(f"Text input takes no options; got {', '.join(sorted(options))}")
        return TextAdapter()
    raise ValueError(f"Unknown input kind {kind!r}; expected 'text' or 'conversation'")


def prepare_document(
    source: str | Path | PreparedDocument,
    *,
    document_id: str | None = None,
    kind: str | None = None,
    **options: Any,
) -> PreparedDocument:
    """Normalize ``source`` into a :class:`PreparedDocument`.

    ``source`` is inline text, a path to a ``.txt``/``.md`` file, a path to a
    conversation ``.csv`` (keyword options go to
    :class:`ConversationCsvAdapter`), or a prepared document. ``document_id``
    overrides the id derived from the file name. Inline text is taken as is:
    ``kind`` and adapter options only apply to files.
    """

    if isinstance(source, PreparedDocument):
        document = source
    elif isinstance(source, Path) or looks_like_path(str(source)):
        path = Path(str(source).strip()).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Document file does not exist: {path}")
        document = adapter_for(path, kind=kind, **options).prepare(path)
    else:
        text = str(source)
        if not text.strip():
            raise ValueError("Document text is empty")
        document = PreparedDocument(
            document_id=document_id or "document",
            text=text if text.endswith("\n") else text + "\n",
            source_path=None,
            input_format="text",
            metadata={"source": "inline"},
        )
    if document_id:
        document.document_id = str(document_id)
    return document


__all__ = [
    "ConversationCsvAdapter",
    "InputAdapter",
    "KNOWN_SUFFIXES",
    "PreparedDocument",
    "TextAdapter",
    "adapter_for",
    "looks_like_path",
    "prepare_document",
]
