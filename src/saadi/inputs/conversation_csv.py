from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

from .base import PreparedDocument


def _normalize_utterance(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _order_key(value: str, fallback: int) -> tuple[int, float | str, int]:
    raw = str(value or "").strip()
    try:
        return (0, float(raw), fallback)
    except ValueError:
        return (1, raw, fallback)


@dataclass(slots=True)
class ConversationCsvAdapter:
    """Normalize ordered conversation turns into ``role: utterance`` plain text.

    The defaults match a therapist/patient transcript; ``allowed_roles`` and
    ``role_aliases`` adapt it to other speaker sets. An empty ``allowed_roles``
    accepts every role.
    """

    role_column: str = "role"
    text_column: str = "text"
    order_column: str = "turn_id"
    allowed_roles: tuple[str, ...] = ("therapist", "patient")
    role_aliases: dict[str, str] = field(default_factory=dict)
    include_timestamps: bool = False
    start_column: str = "start"

    def prepare(self, path: str | Path) -> PreparedDocument:
        source = Path(path).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"Conversation CSV does not exist: {source}")
        if source.suffix.lower() != ".csv":
            raise ValueError(f"Conversation input must be a .csv file: {source}")

        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            required = {self.role_column, self.text_column}
            missing = sorted(required - columns)
            if missing:
                raise ValueError(f"CSV {source} is missing required columns: {', '.join(missing)}")
            rows = list(reader)

        if self.order_column in columns:
            rows = [
                row
                for _, row in sorted(
                    enumerate(rows),
                    key=lambda pair: _order_key(pair[1].get(self.order_column, ""), pair[0]),
                )
            ]

        allowed = {role.casefold() for role in self.allowed_roles}
        aliases = {str(k).casefold(): str(v).casefold() for k, v in (self.role_aliases or {}).items()}
        lines: list[str] = []
        skipped_empty = 0
        role_counts: dict[str, int] = {}
        for row_number, row in enumerate(rows, start=2):
            raw_role = _normalize_utterance(row.get(self.role_column, "")).casefold()
            role = aliases.get(raw_role, raw_role)
            if not role:
                raise ValueError(f"CSV {source} has an empty role at row {row_number}")
            if allowed and role not in allowed:
                raise ValueError(
                    f"CSV {source} has unsupported role {role!r} at row {row_number}; allowed roles: {sorted(allowed)}"
                )
            utterance = _normalize_utterance(row.get(self.text_column, ""))
            if not utterance:
                skipped_empty += 1
                continue
            prefix = role
            if self.include_timestamps:
                timestamp = _normalize_utterance(row.get(self.start_column, ""))
                if timestamp:
                    prefix = f"{role} [{timestamp}]"
            lines.append(f"{prefix}: {utterance}")
            role_counts[role] = role_counts.get(role, 0) + 1

        if not lines:
            raise ValueError(f"Conversation CSV has no non-empty utterances: {source}")

        suffix = ".final_therapist_patient_turns.csv"
        document_id = source.name[: -len(suffix)] if source.name.endswith(suffix) else source.stem
        return PreparedDocument(
            document_id=document_id,
            text="\n".join(lines) + "\n",
            source_path=source,
            input_format="conversation_csv",
            metadata={
                "input_rows": len(rows),
                "output_turns": len(lines),
                "skipped_empty_turns": skipped_empty,
                "role_counts": role_counts,
                "include_timestamps": self.include_timestamps,
            },
        )
