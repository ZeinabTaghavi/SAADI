"""Input adapters and the prepare_document entry point."""

from __future__ import annotations

from pathlib import Path

import pytest

from saadi.inputs import ConversationCsvAdapter, PreparedDocument, TextAdapter, prepare_document


def test_text_adapter_preserves_text(tmp_path: Path) -> None:
    source = tmp_path / "article.txt"
    source.write_text("First paragraph.\n\nSecond paragraph.\n", encoding="utf-8")
    document = TextAdapter().prepare(source)
    assert document.document_id == "article"
    assert document.text == "First paragraph.\n\nSecond paragraph.\n"
    with pytest.raises(ValueError, match="empty"):
        (tmp_path / "empty.txt").write_text("  \n", encoding="utf-8")
        TextAdapter().prepare(tmp_path / "empty.txt")


def test_csv_adapter_orders_turns_and_normalizes_multiline_text(tmp_path: Path) -> None:
    source = tmp_path / "session.final_therapist_patient_turns.csv"
    source.write_text('turn_id,role,text,start\n2,patient,"I feel\n  worried.",4.2\n1,therapist,"How are you?",1.0\n', encoding="utf-8")
    document = ConversationCsvAdapter().prepare(source)
    assert document.document_id == "session"
    assert document.text == "therapist: How are you?\npatient: I feel worried.\n"
    assert document.metadata["role_counts"] == {"therapist": 1, "patient": 1}


def test_csv_adapter_roles_are_configurable(tmp_path: Path) -> None:
    source = tmp_path / "bad.csv"
    source.write_text("turn_id,role,text\n1,narrator,Hello\n", encoding="utf-8")
    assert ConversationCsvAdapter().prepare(source).text == "narrator: Hello\n"  # any role by default
    with pytest.raises(ValueError, match="unsupported role"):
        ConversationCsvAdapter(allowed_roles=("therapist", "patient")).prepare(source)
    assert ConversationCsvAdapter(role_aliases={"narrator": "therapist"}).prepare(source).text == "therapist: Hello\n"


def test_prepare_document_accepts_text_paths_and_prepared_documents(tmp_path: Path) -> None:
    inline = prepare_document("Just some text about Mira Chen.", document_id="note")
    assert (inline.document_id, inline.input_format, inline.source_path) == ("note", "text", None)
    assert inline.text.endswith("\n")

    article = tmp_path / "article.md"
    article.write_text("# Title\n\nBody.\n", encoding="utf-8")
    assert prepare_document(str(article)).document_id == "article"
    assert prepare_document(article, document_id="custom").document_id == "custom"

    transcript = tmp_path / "talk.csv"
    transcript.write_text("role,text\nhost,Welcome\nguest,Thanks\n", encoding="utf-8")
    assert prepare_document(transcript).input_format == "conversation_csv"
    with pytest.raises(ValueError, match="unsupported role"):
        prepare_document(transcript, allowed_roles=("host",))
    with pytest.raises(FileNotFoundError):
        prepare_document(str(tmp_path / "missing.txt"))
    with pytest.raises(ValueError, match="empty"):
        prepare_document("   ")

    ready = PreparedDocument(document_id="ready", text="x\n", source_path=None, input_format="text")
    assert prepare_document(ready) is ready
