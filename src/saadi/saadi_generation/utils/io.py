"""Small JSON/JSONL helpers for SAADI generation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def write_json(path: str | Path, payload: Any) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return str(p)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return str(p)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
