"""Anchor matching: strict reproduces the saved constructions, tolerant accepts unique verbatim anchors."""

from __future__ import annotations

import json

import pytest

from saadi.config.validation import validate_experiment
from saadi.context_strategies import get_strategy
from saadi.core.canonicalization import parse_canonicalization_response
from saadi.core.text_edits import materialize_chunk_text
from saadi.core.types import TextInsertion
from saadi.llm import CallableBackend
from saadi.prompts import check_prompt_profile, format_report

CHUNK = (
    "This is a synthetic demonstration document. The Aurora research team was founded in 2024 by Mira Chen, "
    "a limnologist. Its first station opened beside Lake Alder in the spring of that year. "
    "Mira Chen later handed the station to Leon."
)

# The answer Qwen3-4B gave on Colab: every anchor is verbatim, every context is off by
# whitespace or repeats the anchor. Strict matching dropped all three entities.
QWEN_OUTPUT = json.dumps(
    {
        "active_entities": [
            {"name": "Aurora research team", "description": ["a research team"], "aliases": ["Aurora team"]},
            {"name": "Lake Alder", "description": ["a lake"], "aliases": []},
            {"name": "Leon", "description": ["takes over the station"], "aliases": []},
        ],
        "insertions": [
            {"name": "Aurora research team", "anchor_text": "Aurora research team", "left_context": "The", "right_context": "was founded"},
            {
                "name": "Lake Alder",
                "anchor_text": "Lake Alder",
                "left_context": "Lake Alder",
                "right_context": "in the spring of that year",
            },
            {"name": "Leon", "anchor_text": "Leon", "left_context": "handed the station to Leon", "right_context": "."},
        ],
    }
)


def _insertion(anchor: str, left: str = "", right: str = "", occurrence: int = 1) -> TextInsertion:
    return TextInsertion(anchor_text=anchor, left_context=left, right_context=right, occurrence=occurrence, name=anchor)


def _kept(insertion: TextInsertion, mode: str, text: str = CHUNK) -> str | None:
    result = materialize_chunk_text(text, insertions=[insertion], anchor_matching=mode)
    return result.text if result.insertions else None


@pytest.mark.parametrize(
    "insertion",
    [
        _insertion("Aurora research team", "The", "was founded"),  # spaces trimmed
        _insertion("Lake Alder", "Lake Alder", "in the spring"),  # anchor repeated in the left context
        _insertion("Lake Alder", "opened beside Lake Alder", "Lake Alder in the"),  # repeated on both sides
        _insertion("Leon", "unrelated words", "more unrelated words"),  # wrong context, unique anchor
    ],
)
def test_tolerant_keeps_unique_verbatim_anchors_that_strict_drops(insertion: TextInsertion) -> None:
    assert _kept(insertion, "strict") is None
    text = _kept(insertion, "tolerant")
    assert text is not None
    assert f"{insertion.anchor_text} ({insertion.name})" in text


def test_tolerant_is_strict_when_strict_succeeds() -> None:
    exact = _insertion("Mira Chen", "founded in 2024 by ", ", a limnologist")
    assert _kept(exact, "strict") == _kept(exact, "tolerant")
    assert "by Mira Chen (Mira Chen), a limnologist" in _kept(exact, "tolerant")


def test_anchor_missing_from_the_chunk_is_rejected_in_both_modes() -> None:
    for mode in ("strict", "tolerant"):
        assert _kept(_insertion("Sofia Marin", "", ""), mode) is None
        assert _kept(_insertion("Mira  Chen", "by", ","), mode) is None  # not verbatim


def test_tolerant_disambiguates_a_repeated_anchor_by_trimmed_context_and_refuses_otherwise() -> None:
    # "Mira Chen" occurs twice; trimmed contexts pick the second occurrence.
    exact = _insertion("Mira Chen", "that year. ", " later handed")
    kept = _kept(_insertion("Mira Chen", "that year.", "later handed"), "tolerant")
    assert kept is not None and _kept(exact, "strict") == kept
    assert kept.index("Mira Chen (Mira Chen) later") > kept.index("by Mira Chen,")
    # Contexts that fit neither occurrence leave the anchor ambiguous.
    assert _kept(_insertion("Mira Chen", "nowhere", "nothing"), "tolerant") is None


def test_parser_keeps_the_entities_from_the_real_model_output_only_when_tolerant() -> None:
    strict = parse_canonicalization_response(QWEN_OUTPUT, fallback_chunk_text=CHUNK, allow_linking_sentence=False, anchor_matching="strict")
    tolerant = parse_canonicalization_response(
        QWEN_OUTPUT, fallback_chunk_text=CHUNK, allow_linking_sentence=False, anchor_matching="tolerant"
    )
    assert strict.active_entities == [] and len(strict.rejected_entities) == 3
    assert [entity.name for entity in tolerant.active_entities] == ["Aurora research team", "Lake Alder", "Leon"]
    assert tolerant.rejected_entities == []
    assert tolerant.parser_status == "ok"
    # Retrieval never sees the anchored copy; it only feeds the history and the folds.
    assert "Aurora research team (Aurora research team)" in tolerant.updated_chunk_text


def test_strategy_default_published_profile_and_validation(tmp_path, monkeypatch) -> None:
    from test_release import ROOT, runner

    from saadi.config.loader import load_config

    assert get_strategy("hierarchical").anchor_matching == "tolerant"
    assert get_strategy("hierarchical", anchor_matching="strict").anchor_matching == "strict"
    with pytest.raises(ValueError):
        get_strategy("hierarchical", anchor_matching="loose")

    monkeypatch.setenv("SADI_OUTPUT_DIR", str(tmp_path))
    config_path = ROOT / "configs/experiments/loogle_retrieval_ablation.yaml"
    published = runner().published_config(load_config(config_path))
    validate_experiment(published)
    assert published["ingest"]["strategy_kwargs"]["anchor_matching"] == "strict"
    fresh = load_config(config_path)
    fresh["ingest"]["strategy_kwargs"]["anchor_matching"] = "loose"
    with pytest.raises(ValueError):
        validate_experiment(fresh)


def test_prompt_check_reports_dropped_proposals_and_fails_when_none_are_kept() -> None:
    def paraphrasing_model(system: str, user: str, **options) -> str:
        if "[ROLLUP_ITEMS_JSON]" in user:
            return "A short summary.\n[END_SUMMARY]"
        return (
            json.dumps(
                {
                    "active_entities": [{"name": "Mira Chen", "description": [], "aliases": []}],
                    "insertions": [{"name": "Mira Chen", "anchor_text": "Dr. Mira Chen", "left_context": "", "right_context": ""}],
                }
            )
            + "\n[END_JSON]"
        )

    report = check_prompt_profile("default", backend=CallableBackend(paraphrasing_model))
    assert not report.ok
    assert report.entity_count == 0 and report.rejected_count == 1
    text = format_report(report)
    assert "0 entities kept, 1 dropped" in text
    assert "entity memory would stay empty" in text and "result: FAILED" in text


def test_prompt_check_passes_the_real_model_output_shape() -> None:
    def spacing_model(system: str, user: str, **options) -> str:
        if "[ROLLUP_ITEMS_JSON]" in user:
            return "A short summary.\n[END_SUMMARY]"
        return (
            json.dumps(
                {
                    "active_entities": [{"name": "Mira Chen", "description": ["a limnologist"], "aliases": []}],
                    "insertions": [
                        {
                            "name": "Mira Chen",
                            "anchor_text": "Mira Chen",
                            "left_context": "by Mira Chen",
                            "right_context": ", a limnologist",
                        }
                    ],
                }
            )
            + "\n[END_JSON]"
        )

    report = check_prompt_profile("default", backend=CallableBackend(spacing_model))
    assert report.ok and report.entity_count == 1 and report.rejected_count == 0
