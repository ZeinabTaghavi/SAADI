from __future__ import annotations

from pathlib import Path

from single_document_demo.pipeline.prompts import load_prompt_profile


def test_prompt_profile_loads_both_prompts_and_hashes_content(tmp_path: Path) -> None:
    profile_dir = tmp_path / "politics"
    profile_dir.mkdir()
    (profile_dir / "entity_extraction.txt").write_text("Extract political entities.", encoding="utf-8")
    summary = profile_dir / "hierarchy_summary.txt"
    summary.write_text("Summarize political events.", encoding="utf-8")
    first = load_prompt_profile(tmp_path, "politics")
    summary.write_text("Summarize events and policies.", encoding="utf-8")
    second = load_prompt_profile(tmp_path, "politics")
    assert first.name == "politics"
    assert first.entity_extraction_prompt == "Extract political entities."
    assert first.content_hash != second.content_hash


def test_therapy_prompt_enforces_compact_prioritized_memory() -> None:
    profile = load_prompt_profile(
        Path(__file__).resolve().parents[1] / "memory_prompts", "therapy"
    )
    prompt = profile.entity_extraction_prompt
    assert "Return at most 8 active entities" in prompt
    assert "Major adverse experiences" in prompt
    assert "NEVER store pronouns as aliases" in prompt
    assert "Prefer updating an existing broad entity" in prompt
    assert "future appointment is normally a description" in prompt
    assert profile.document_summary_prompt is not None
    assert "entire therapy session" in profile.document_summary_prompt
