from __future__ import annotations

import json
from pathlib import Path

import pytest

from single_document_demo.pipeline.config import REPO_ROOT, load_demo_config
from single_document_demo.pipeline.runner import (
    prepare_documents,
    resolve_run_mode,
    run_demo,
)


def test_conversation_config_settings() -> None:
    config = load_demo_config(
        "single_document_demo/configs/therapy_conversation.toml"
    )
    assert config.workflow == "therapy"
    assert config.input_format == "conversation_csv"
    assert config.prompt_profile == "conversation"
    assert config.saadi_options["max_entities"] == 8
    assert config.saadi_options["document_summary_enabled"] is True
    assert config.model_config["enforce_eager"] is True
    assert config.input_path == REPO_ROOT / "single_document_demo/input/sample_conversation/conversation.csv"
    assert config.output_root == REPO_ROOT / "single_document_demo" / "output"


def test_text_config_uses_default_saadi_profile() -> None:
    config = load_demo_config("single_document_demo/configs/text_document.toml")
    assert config.workflow == "text"
    assert config.input_format == "text"
    assert config.prompt_profile == "default"
    assert config.saadi_options["document_summary_enabled"] is False
    assert config.input_path == (
        REPO_ROOT / "single_document_demo" / "input" / "sample_text" / "document.txt"
    )


def test_text_mode_is_inferred_from_question_options(tmp_path: Path) -> None:
    config = load_demo_config("single_document_demo/configs/text_document.toml")
    assert resolve_run_mode(config) == ("memory_only", [])

    config.query = "What happened?"
    assert resolve_run_mode(config) == ("single_query", [])

    questions = tmp_path / "questions.jsonl"
    questions.write_text('{"question":"Who is involved?"}\n', encoding="utf-8")
    config.query = ""
    config.questions_path = str(questions)
    mode, loaded = resolve_run_mode(config)
    assert mode == "multi_question"
    assert loaded == ["Who is involved?"]


def test_therapy_rejects_question_answering() -> None:
    config = load_demo_config(
        "single_document_demo/configs/therapy_conversation.toml"
    )
    config.query = "What happened?"
    with pytest.raises(ValueError, match="memory-only"):
        resolve_run_mode(config)


def test_mock_single_text_run_uses_consistent_document_nesting(tmp_path: Path) -> None:
    source = tmp_path / "demo" / "input" / "sample_document" / "document.txt"
    source.parent.mkdir(parents=True)
    source.write_text("A short document about an important event.\n", encoding="utf-8")
    config = load_demo_config("single_document_demo/configs/text_document.toml")
    config.demo_root = tmp_path / "demo"
    config.output_root = config.demo_root / "output"
    config.input_path = source
    config.prompt_profile = "default"
    config.run_name = "mock_text"
    config.model_config = {"backend": "mock"}
    config.saadi_options.update({"use_cache": False, "chunk_size": 100})

    result = run_demo(config)
    assert not result.failures
    record = result.records[0]
    assert record.input_folder == "sample_document"
    assert Path(record.run_dir) == config.output_root / "sample_document"
    assert Path(record.normalized_document).is_file()
    assert Path(record.run_manifest or "").is_file()
    assert Path(record.cleaned_outputs_manifest or "").is_file()
    assert Path(record.cleaned_outputs_manifest or "").parent.name == "cleaned_outputs"
    assert record.document_summary is None

    payload = json.loads(result.batch_manifest_path.read_text(encoding="utf-8"))
    assert result.batch_manifest_path == (
        config.output_root / "_batches" / "mock_text" / "batch_manifest.json"
    )
    assert payload["workflow"] == "text"
    assert payload["run_mode"] == "memory_only"
    assert payload["documents_total"] == 1
    assert payload["documents_completed"] == 1


def test_recursive_input_discovery_preserves_folder_names(tmp_path: Path) -> None:
    input_root = tmp_path / "demo" / "input"
    first = input_root / "163.final_therapist_patient_turns"
    second = input_root / "conversation with spaces"
    for folder in (first, second):
        folder.mkdir(parents=True)
        (folder / f"{folder.name}.csv").write_text(
            "turn_id,role,text\n1,therapist,Hello\n2,patient,Hi\n",
            encoding="utf-8",
        )

    config = load_demo_config("single_document_demo/configs/therapy_conversation.toml")
    config.demo_root = tmp_path / "demo"
    config.input_path = input_root
    config.output_root = config.demo_root / "output"
    documents = prepare_documents(config)

    assert [item.metadata["input_folder"] for item in documents] == [
        "163.final_therapist_patient_turns",
        "conversation with spaces",
    ]
