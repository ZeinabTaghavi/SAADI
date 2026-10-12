from __future__ import annotations

from pathlib import Path

import pytest

from single_document_demo.input_adapters import ConversationCsvAdapter, TextAdapter


def test_text_adapter_preserves_text(tmp_path: Path) -> None:
    source = tmp_path / "article.txt"
    source.write_text("First paragraph.\n\nSecond paragraph.\n", encoding="utf-8")
    document = TextAdapter().prepare(source)
    assert document.document_id == "article"
    assert document.text == "First paragraph.\n\nSecond paragraph.\n"


def test_csv_adapter_orders_turns_and_normalizes_multiline_text(tmp_path: Path) -> None:
    source = tmp_path / "session.final_therapist_patient_turns.csv"
    source.write_text(
        'turn_id,role,text,start\n'
        '2,patient,"I feel\n  worried.",4.2\n'
        '1,therapist,"How are you?",1.0\n',
        encoding="utf-8",
    )
    document = ConversationCsvAdapter().prepare(source)
    assert document.document_id == "session"
    assert document.text == (
        "therapist: How are you?\n"
        "patient: I feel worried.\n"
    )
    assert document.metadata["role_counts"] == {"therapist": 1, "patient": 1}


def test_csv_adapter_rejects_unknown_role(tmp_path: Path) -> None:
    source = tmp_path / "bad.csv"
    source.write_text("turn_id,role,text\n1,narrator,Hello\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported role"):
        ConversationCsvAdapter().prepare(source)
