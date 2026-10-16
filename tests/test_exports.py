"""Readable exports: memory sanitization and the flat state files."""

from __future__ import annotations

import json
from pathlib import Path

from saadi.exports import export_readable_state, sanitize_memory_summary


def _memory_payload() -> dict:
    return {
        "entities": [
            {
                "name": "Doctor Brown",
                "aliases": [
                    {"text": "Dr Brown", "first_seen_chunk": 0},
                    {"text": "he", "first_seen_chunk": 1},
                    {"text": "therapist", "first_seen_chunk": 1},
                ],
                "descriptions": [{"text": "runs the clinic", "first_seen_chunk": 0}],
            }
        ]
    }


def test_memory_export_removes_role_labels_and_pronouns(tmp_path: Path) -> None:
    path = tmp_path / "memory_summary.json"
    path.write_text(json.dumps(_memory_payload()), encoding="utf-8")
    assert sanitize_memory_summary(path) == 2
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert [item["text"] for item in payload["entities"][0]["aliases"]] == ["Dr Brown"]
    assert payload["export_sanitization"]["removed_forbidden_aliases"] == 2
    assert sanitize_memory_summary(None) == 0


def test_readable_state_always_writes_a_memory_file(tmp_path: Path) -> None:
    state = tmp_path / "state"
    result = export_readable_state(state, artifacts=None, memory_summary_path=None)
    assert result.memory_path == state / "memory.json"
    assert json.loads((state / "memory.json").read_text(encoding="utf-8")) == {"total_entities": 0, "entities": []}


def test_readable_state_copies_memory_and_summary_without_touching_the_cache(tmp_path: Path) -> None:
    memory_path = tmp_path / "cache" / "memory_summary.json"
    memory_path.parent.mkdir()
    memory_path.write_text(json.dumps(_memory_payload()), encoding="utf-8")
    summary_path = tmp_path / "cache" / "summary.txt"
    summary_path.write_text("A complete summary.\n", encoding="utf-8")

    state = tmp_path / "state"
    result = export_readable_state(state, artifacts=None, memory_summary_path=memory_path, document_summary_path=summary_path)
    assert result.levels == ()  # no cache index given, so no history levels
    assert result.aliases_removed == 2
    assert (state / "history").is_dir()
    assert json.loads((state / "memory.json").read_text(encoding="utf-8"))["entities"][0]["aliases"] == [
        {"text": "Dr Brown", "first_seen_chunk": 0}
    ]
    assert (state / "summary.txt").read_text(encoding="utf-8") == "A complete summary.\n"
    original = json.loads(memory_path.read_text(encoding="utf-8"))
    assert len(original["entities"][0]["aliases"]) == 3  # the cache copy is untouched
