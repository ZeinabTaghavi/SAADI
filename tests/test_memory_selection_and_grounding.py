from __future__ import annotations

import json

from saadi.core.state import apply_canonicalization_to_state
from saadi.core.types import CanonicalizedChunkRecord, DiscourseState, EntityRecord
from saadi.core.validation import build_canonicalization_output
from saadi.entities.active_selection import (
    normalize_active_entity_selector,
    select_active_entities,
    select_active_entities_with_scores,
    tokenize_active_entity_text,
)
from saadi.io.hierarchy_cache import (
    export_llm_history_cache,
    load_canonicalized_chunk_records_from_cache_index,
    load_llm_history_cache_index,
    load_raw_chunk_records_from_cache_index,
    resolve_llm_cache_namespace_dir,
)


def test_active_memory_tokenizer_is_unicode_aware_and_removes_stopwords() -> None:
    tokens = tokenize_active_entity_text(
        "Cofradía de Pescadores and Zurbarán's work"
    )

    assert tokens == ["cofradía", "pescadores", "zurbarán", "work"]
    assert "cofrad" not in tokens
    assert "zurbar" not in tokens
    assert not {"a", "de", "of", "and", "s"} & set(tokens)


def test_default_bm25_selector_returns_empty_for_stopword_only_overlap() -> None:
    memory = [
        EntityRecord(
            name="Cofradía de Pescadores",
            aliases=["Fishermen's Brotherhood of Altea"],
            description=["A centre designed by Picardo"],
        ),
        EntityRecord(name="Zurbarán", aliases=[], description=["Spanish painter"]),
    ]

    assert normalize_active_entity_selector(None) == "bm25"
    assert select_active_entities(memory, "a de of and s") == []


def test_selector_diagnostics_expose_scores_without_changing_prompt_entries() -> None:
    memory = [
        EntityRecord(name="Cofradía de Pescadores"),
        EntityRecord(name="Zurbarán", description=["Spanish painter"]),
    ]
    diagnostics = select_active_entities_with_scores(
        memory,
        "The paintings by Zurbarán",
    )

    assert len(diagnostics) == 1
    assert diagnostics[0]["score"] > 0
    assert diagnostics[0]["entry"]["name"] == "Zurbarán"
    assert "score" not in select_active_entities(memory, "The paintings by Zurbarán")[0]

    score = diagnostics[0]["score"]
    assert (
        select_active_entities(
            memory,
            "The paintings by Zurbarán",
            min_relevance_score=score,
        )
        == []
    )


def test_ungrounded_active_entity_is_rejected_and_never_merged() -> None:
    output = build_canonicalization_output(
        parsed={
            "active_entities": [
                {
                    "name": "Cofradía de Pescadores",
                    "aliases": ["Fishermen's Brotherhood of Altea"],
                    "description": ["A centre designed by Picardo"],
                }
            ],
            "insertions": [
                {
                    "name": "Cofradía de Pescadores",
                    "anchor_text": "Cofradía de Pescadores",
                    "left_context": "",
                    "right_context": "",
                }
            ],
        },
        raw_model_text="{}",
        parser_status="ok",
        fallback_chunk_text="This chunk discusses a completely different building.",
    )

    assert output.parser_status == "fallback"
    assert output.active_entities == []
    assert [entity.name for entity in output.rejected_entities] == [
        "Cofradía de Pescadores"
    ]

    state = apply_canonicalization_to_state(DiscourseState(), output)
    assert state.active_entities == []
    assert state.entity_memory == []


def test_grounded_entity_is_retained() -> None:
    output = build_canonicalization_output(
        parsed={
            "active_entities": [
                {"name": "Zurbarán", "aliases": ["Zurbarán"], "description": []}
            ],
            "insertions": [
                {
                    "name": "Zurbarán",
                    "anchor_text": "Zurbarán",
                    "left_context": "Paintings by ",
                    "right_context": " were exhibited.",
                }
            ],
        },
        raw_model_text="{}",
        parser_status="ok",
        fallback_chunk_text="Paintings by Zurbarán were exhibited.",
    )

    assert [entity.name for entity in output.active_entities] == ["Zurbarán"]
    assert output.rejected_entities == []


def test_legacy_active_objects_payload_is_still_accepted() -> None:
    output = build_canonicalization_output(
        parsed={"active_objects": [], "insertions": []},
        raw_model_text="{}",
        parser_status="ok",
        fallback_chunk_text="Legacy cached chunk.",
    )

    assert output.active_entities == []
    assert output.updated_chunk_text == "Legacy cached chunk."


def test_canonical_cache_lookup_finds_legacy_ao_namespace(tmp_path) -> None:
    legacy_dir = resolve_llm_cache_namespace_dir(
        tmp_path,
        dataset_label="test",
        method_label="hierarchical",
        hierarchy_branching_factor=5,
        chunk_size=500,
        active_object_config_label="AO_5_2_3",
        llm_label="mock",
        active_object_selector_label="AOWord",
    )
    legacy_dir.mkdir(parents=True)
    legacy_doc_dir = legacy_dir / "doc"
    legacy_doc_dir.mkdir()
    legacy_steps_path = legacy_doc_dir / "steps_index.jsonl"
    legacy_steps_path.write_text(
        json.dumps(
            {
                "doc_id": "doc",
                "step": 0,
                "chunk_id": "doc:0",
                "raw_chunk_text": "Legacy cached chunk.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (legacy_dir / "index.json").write_text(
        json.dumps(
            {
                "format_version": 2,
                "cache_namespace_dir": str(legacy_dir),
                "docs": [
                    {
                        "doc_id": "doc",
                        "files": {"steps_index": str(legacy_steps_path)},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    loaded = load_llm_history_cache_index(
        output_dir=tmp_path,
        dataset_label="test",
        method_label="hierarchical",
        hierarchy_branching_factor=5,
        chunk_size=500,
        active_entity_config_label="AE_5_2_3",
        llm_label="mock",
        active_entity_selector_label="AEWord",
    )

    assert loaded is not None
    assert loaded["index_path"] == str(legacy_dir / "index.json")

    canonical_dir = resolve_llm_cache_namespace_dir(
        tmp_path,
        dataset_label="test",
        method_label="hierarchical",
        hierarchy_branching_factor=5,
        chunk_size=500,
        active_entity_config_label="AE_5_2_3",
        llm_label="mock",
        active_entity_selector_label="AEWord",
    )
    legacy_config_dir = legacy_dir.parents[1]
    legacy_dir.rename(legacy_dir.with_name("AEWord"))
    legacy_config_dir.rename(legacy_config_dir.with_name("AE_5_2_3"))

    raw_records = load_raw_chunk_records_from_cache_index(canonical_dir / "index.json")
    assert [record.raw_text for record in raw_records] == ["Legacy cached chunk."]


def test_rejected_entity_count_is_persisted_per_document(tmp_path) -> None:
    rejected = EntityRecord(name="Ungrounded")
    record = CanonicalizedChunkRecord(
        doc_id="doc",
        chunk_id="doc:0",
        chunk_index=0,
        raw_chunk_text="Current text.",
        canonicalized_chunk_text="Current text.",
        state_snapshot={
            "prev_chunk_text": "Current text.",
            "entity_memory": [],
            "active_entities": [],
            "hierarchy_summaries": {1: {0: "Current text."}},
        },
        strategy_name="hierarchical_summary",
        parser_status="fallback",
        rejected_entities=[rejected],
    )

    index = export_llm_history_cache(
        [record],
        output_dir=tmp_path,
        dataset_label="test",
        method_label="hierarchical",
        active_entity_selector_label="AEBM25",
        active_entity_min_relevance=0.0,
    )

    assert index["active_entity_min_relevance"] == 0.0
    assert index["active_entity_min_relevance"] == 0.0
    assert index["rejected_entity_count"] == 1
    assert index["docs"][0]["rejected_entity_count"] == 1

    memory_path = index["docs"][0]["files"]["memory"]
    with open(memory_path, encoding="utf-8") as handle:
        memory = json.load(handle)
    assert memory["rejected_entity_count"] == 1

    restored = load_canonicalized_chunk_records_from_cache_index(index["index_path"])
    assert [entity.name for entity in restored[0].rejected_entities] == ["Ungrounded"]
