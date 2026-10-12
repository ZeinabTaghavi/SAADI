from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from single_document_demo.pipeline.artifacts import sanitize_memory_summary
from single_document_demo.pipeline.summaries import export_cleaned_outputs


def test_memory_export_removes_role_labels_and_pronouns(tmp_path: Path) -> None:
    path = tmp_path / "memory_summary.json"
    path.write_text(
        json.dumps(
            {
                "entities": [
                    {
                        "name": "Doctor Brown",
                        "aliases": [
                            {"text": "Dr Brown", "first_seen_chunk": 0},
                            {"text": "he", "first_seen_chunk": 1},
                            {"text": "therapist", "first_seen_chunk": 1},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    removed = sanitize_memory_summary(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert removed == 2
    assert [item["text"] for item in payload["entities"][0]["aliases"]] == [
        "Dr Brown"
    ]
    assert payload["export_sanitization"]["removed_forbidden_aliases"] == 2


def test_cleaned_outputs_collect_and_sanitize_consumer_files(tmp_path: Path) -> None:
    memory_path = tmp_path / "memory_summary.json"
    memory_path.write_text(
        json.dumps(
            {
                "entities": [
                    {
                        "name": "Doctor Brown",
                        "aliases": [
                            {"text": "Dr Brown", "first_seen_chunk": 0},
                            {"text": "he", "first_seen_chunk": 1},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    document_summary = tmp_path / "technical_summary.txt"
    document_summary.write_text("A complete session summary.\n", encoding="utf-8")

    result = export_cleaned_outputs(
        SimpleNamespace(artifacts=None),
        tmp_path,
        memory_summary_path=memory_path,
        document_summary_path=document_summary,
    )

    assert result is not None
    assert result.directory == tmp_path / "cleaned_outputs"
    assert result.aliases_removed == 1
    assert result.document_summary_path.read_text(encoding="utf-8") == (
        "A complete session summary.\n"
    )
    cleaned_memory = json.loads(result.memory_summary_path.read_text(encoding="utf-8"))
    assert [item["text"] for item in cleaned_memory["entities"][0]["aliases"]] == [
        "Dr Brown"
    ]
    original_memory = json.loads(memory_path.read_text(encoding="utf-8"))
    assert [item["text"] for item in original_memory["entities"][0]["aliases"]] == [
        "Dr Brown",
        "he",
    ]
