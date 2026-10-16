"""Tests for prompt profiles: the fixed contract, the loader, the check command, and the wiring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from saadi.config.validation import validate_experiment
from saadi.context_strategies.hierarchical_summary import HIER_SYSTEM_PROMPT, ROLLUP_SYSTEM_PROMPT
from saadi.llm import CallableBackend
from saadi.prompts import (
    PromptProfile,
    apply_profile_to_strategy_kwargs,
    builtin_profile_names,
    check_prompt_profile,
    format_report,
    load_prompt_profile,
    profile_cache_suffix,
)
from saadi.prompts.contract import ENTITY_CONTRACT_POSTAMBLE, ENTITY_CONTRACT_PREAMBLE, SUMMARY_CONTRACT_POSTAMBLE

ROOT = Path(__file__).resolve().parents[1]


def write_profile(directory: Path, *, entity: str, summary: str, document: str | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "entity_extraction.txt").write_text(entity, encoding="utf-8")
    (directory / "hierarchy_summary.txt").write_text(summary, encoding="utf-8")
    if document is not None:
        (directory / "document_summary.txt").write_text(document, encoding="utf-8")
    return directory


# --- the contract ----------------------------------------------------------------


def test_default_profile_reproduces_the_paper_prompts_byte_for_byte() -> None:
    profile = load_prompt_profile("default")
    assert profile.entity_system_prompt == HIER_SYSTEM_PROMPT
    assert profile.summary_system_prompt == ROLLUP_SYSTEM_PROMPT
    assert profile.document_summary_prompt is None
    assert load_prompt_profile("Default").content_hash == profile.content_hash


def test_custom_guidance_is_wrapped_in_the_fixed_contract(tmp_path: Path) -> None:
    legal = write_profile(
        tmp_path / "legal",
        entity="Entities are parties, courts, judges, statutes, and contractual obligations. Name statutes by their citation.",
        summary="Keep the parties, the claims, and the procedural posture.",
    )
    profile = load_prompt_profile(legal)
    prompt = profile.entity_system_prompt
    assert prompt.startswith(ENTITY_CONTRACT_PREAMBLE)
    assert prompt.endswith(ENTITY_CONTRACT_POSTAMBLE)
    assert "parties, courts, judges" in prompt
    assert "Return JSON only with keys `active_entities` and `insertions`" in prompt
    assert profile.summary_system_prompt.endswith(SUMMARY_CONTRACT_POSTAMBLE)
    assert profile.summary_system_prompt.endswith("Terminate with [END_SUMMARY].")
    assert profile.name == "legal"


def test_guidance_may_not_redefine_the_output_contract(tmp_path: Path) -> None:
    bad_entity = write_profile(
        tmp_path / "bad_entity", entity="Return JSON only with my own keys. Then [END_JSON].", summary="Keep events."
    )
    with pytest.raises(ValueError, match="output contract is fixed"):
        load_prompt_profile(bad_entity)
    bad_summary = write_profile(tmp_path / "bad_summary", entity="Extract people.", summary="Write prose. Terminate with [END_SUMMARY].")
    with pytest.raises(ValueError, match="output contract is fixed"):
        load_prompt_profile(bad_summary)
    bad_document = write_profile(
        tmp_path / "bad_document", entity="Extract people.", summary="Keep events.", document="End with [END_SUMMARY]."
    )
    with pytest.raises(ValueError, match="output contract is fixed"):
        load_prompt_profile(bad_document)


# --- the loader ------------------------------------------------------------------


def test_builtin_names_aliases_and_errors(tmp_path: Path) -> None:
    assert builtin_profile_names() == ["conversation", "default"]
    conversation = load_prompt_profile("conversation")
    assert load_prompt_profile("therapy").content_hash == conversation.content_hash
    assert conversation.document_summary_prompt is not None
    assert "entire therapy session" in conversation.document_summary_prompt
    assert conversation.document_summary_prompt.endswith("Terminate with [END_SUMMARY].")
    assert "Return at most 8 active entities" in conversation.entity_system_prompt
    assert "NEVER store pronouns as aliases" in conversation.entity_system_prompt
    assert conversation.entity_system_prompt.startswith(ENTITY_CONTRACT_PREAMBLE)
    assert conversation.content_hash != load_prompt_profile("default").content_hash

    with pytest.raises(ValueError, match="Unknown prompt profile 'nope'. Built-in profiles: conversation, default"):
        load_prompt_profile("nope")
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "entity_extraction.txt").write_text("Extract people.", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="hierarchy_summary.txt"):
        load_prompt_profile(incomplete)
    (incomplete / "hierarchy_summary.txt").write_text("   \n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_prompt_profile(incomplete)
    assert load_prompt_profile(conversation) is conversation  # profile objects pass through


def test_content_hash_follows_the_composed_prompts(tmp_path: Path) -> None:
    first = load_prompt_profile(write_profile(tmp_path / "a", entity="Extract people.", summary="Keep events."))
    same = load_prompt_profile(write_profile(tmp_path / "b", entity="Extract people.", summary="Keep events."))
    changed = load_prompt_profile(write_profile(tmp_path / "c", entity="Extract people.", summary="Keep events and dates."))
    assert first.content_hash == same.content_hash
    assert first.content_hash != changed.content_hash
    with_document = load_prompt_profile(
        write_profile(tmp_path / "d", entity="Extract people.", summary="Keep events.", document="Summarize the whole file.")
    )
    assert with_document.content_hash != first.content_hash
    assert PromptProfile("inline", tmp_path, "Extract people.", "Keep events.").content_hash == first.content_hash


def test_package_data_is_installed_with_the_library() -> None:
    from importlib import resources

    root = resources.files("saadi.prompts").joinpath("profiles")
    for name in ("default", "conversation"):
        assert root.joinpath(name, "entity_extraction.txt").is_file()
        assert root.joinpath(name, "hierarchy_summary.txt").is_file()
    assert root.joinpath("conversation", "document_summary.txt").is_file()
    assert not root.joinpath("default", "document_summary.txt").is_file()


# --- the check command -----------------------------------------------------------


def test_check_prompt_profile_with_the_mock_backend_proves_the_plumbing() -> None:
    report = check_prompt_profile("default")
    assert report.ok
    assert report.entity_status == "ok" and report.entity_count == 0
    assert report.summary_status == "ok" and report.summary_text
    assert any("mock backend" in note for note in report.notes)
    text = format_report(report)
    assert "result: OK" in text and "entity extraction parse: ok" in text


def test_check_prompt_profile_reports_outputs_that_break_the_contract() -> None:
    def chatty_model(system: str, user: str, **options) -> str:
        if "[ROLLUP_ITEMS_JSON]" in user:
            return "A short summary.\n[END_SUMMARY]"
        return 'Sure! Here you go:\n{"active_entities": [{"name": "Mira Chen", "aliases": ["Mira"], "description": ["limnologist"]}], "insertions": [{"name": "Mira Chen", "anchor_text": "Mira Chen", "left_context": "founded in 2024 by ", "right_context": ", a limnologist"}, {"name": "Mira Chen", "anchor_text": "NOT IN TEXT", "left_context": "", "right_context": ""}]}\n[END_JSON]'

    report = check_prompt_profile("default", backend=CallableBackend(chatty_model))
    assert report.ok
    assert report.entity_status in {"ok", "fallback"}
    assert report.entity_names == ["Mira Chen"]
    assert report.entity_count == 1

    def broken_model(system: str, user: str, **options) -> str:
        return "" if "[ROLLUP_ITEMS_JSON]" in user else "I cannot do that."

    report = check_prompt_profile("default", backend=CallableBackend(broken_model))
    assert not report.ok
    assert report.entity_status == "failed"
    assert report.summary_status == "failed"
    assert any("not a JSON object" in problem for problem in report.problems)
    assert any("fold summary came back empty" in problem for problem in report.problems)
    assert "result: FAILED" in format_report(report)


def test_check_prompts_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from saadi.cli.main import main

    assert main(["check-prompts", "default"]) == 0
    out = capsys.readouterr().out
    assert "profile: default" in out and "result: OK" in out

    sample = tmp_path / "sample.txt"
    sample.write_text(" ".join(f"word{i}" for i in range(500)), encoding="utf-8")
    assert main(["check-prompts", "conversation", "--text", str(sample), "--words", "30", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["profile"] == "conversation" and payload["ok"] is True

    assert main(["check-prompts", "nope"]) == 1
    assert "Unknown prompt profile" in capsys.readouterr().out


# --- wiring into the pipelines ---------------------------------------------------


def test_documents_index_with_a_custom_profile_and_record_its_hash(tmp_path: Path) -> None:
    from saadi import Saadi

    legal = write_profile(
        tmp_path / "legal",
        entity="Entities are parties, courts, statutes, and obligations.",
        summary="Keep the parties and the claims.",
    )
    profile = load_prompt_profile(legal)
    doc = Saadi(llm="mock", prompts=legal, state_dir=tmp_path / "state", chunk_size=50)
    doc.index(ROOT / "examples" / "sample_text" / "document.txt")
    manifest = json.loads((tmp_path / "state" / "saadi.json").read_text(encoding="utf-8"))
    assert manifest["settings"]["prompts"]["name"] == "legal"
    assert manifest["settings"]["prompts"]["content_hash"] == profile.content_hash
    assert doc.cache_dir.name == f"prompts_{profile.content_hash[:10]}"  # a profile gets its own cache


def test_profile_aliases_in_the_api_and_the_cli() -> None:
    from saadi import Saadi
    from saadi.cli.index import build_arg_parser

    assert Saadi(llm="mock", prompts="therapy").profile.name == "conversation"
    assert build_arg_parser().parse_args(["session.csv", "--llm", "mock", "--prompts", "therapy"]).prompts == "therapy"
    assert load_prompt_profile("therapy").name == "conversation"


def test_experiment_configs_accept_a_prompt_profile() -> None:
    profile = load_prompt_profile("conversation")
    kwargs = apply_profile_to_strategy_kwargs(profile, {"branching_factor": 5})
    assert kwargs["branching_factor"] == 5
    assert kwargs["system_prompt"] == profile.entity_system_prompt
    assert kwargs["rollup_system_prompt"] == profile.summary_system_prompt
    pinned = apply_profile_to_strategy_kwargs(profile, {"system_prompt": "literal"})
    assert pinned["system_prompt"] == "literal"  # an explicit prompt wins
    assert profile_cache_suffix(profile) == f"prompt_conversation_{profile.content_hash[:10]}"

    config = {
        "dataset": {"name": "x"},
        "ingest": {"enabled": True, "prompt_profile": "conversation"},
        "retrieval": {},
        "model": {"backend": "mock"},
    }
    validate_experiment(config)


def test_experiment_runner_uses_the_profile_and_isolates_its_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from saadi.pipelines import experiment_runner

    seen: dict[str, object] = {}

    def fake_ingest(**kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop after capturing the ingest call")

    monkeypatch.setattr(experiment_runner, "ingest_documents", fake_ingest)
    config = {
        "output_dir": str(tmp_path),
        "dataset": {"name": "synthetic"},
        "ingest": {"enabled": True, "use_cache": False, "prompt_profile": "conversation", "strategy_kwargs": {"branching_factor": 3}},
        "retrieval": {},
        "model": {"backend": "mock"},
    }
    with pytest.raises(RuntimeError, match="stop after capturing"):
        experiment_runner.run_experiment(config, documents_by_id={"d": "text"}, qa_entries=[])
    profile = load_prompt_profile("conversation")
    assert seen["strategy_kwargs"]["system_prompt"] == profile.entity_system_prompt
    assert seen["strategy_kwargs"]["branching_factor"] == 3
    assert seen["cache_namespace"]["method"].endswith(profile_cache_suffix(profile))


def test_check_uses_the_profile_sample_when_present(tmp_path: Path) -> None:
    seen: list[str] = []

    def recording_model(system: str, user: str, **options) -> str:
        seen.append(user)
        if "[ROLLUP_ITEMS_JSON]" in user:
            return "Summary.\n[END_SUMMARY]"
        return '{"active_entities": [], "insertions": []}\n[END_JSON]'

    check_prompt_profile("conversation", backend=CallableBackend(recording_model))
    assert any("therapist:" in user for user in seen)  # the conversation profile ships a transcript sample
    seen.clear()
    custom = write_profile(tmp_path / "custom", entity="Extract people.", summary="Keep events.")
    (custom / "sample.txt").write_text("Custom sample about Zorblax.", encoding="utf-8")
    check_prompt_profile(custom, backend=CallableBackend(recording_model))
    assert any("Zorblax" in user for user in seen)
    seen.clear()
    check_prompt_profile(custom, backend=CallableBackend(recording_model), text="Explicit text wins.")
    assert any("Explicit text wins." in user for user in seen)
