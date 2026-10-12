"""The model sees original chunk text as L0 history; folds and grounding keep the anchored copy."""
from __future__ import annotations

import json

import pytest

from saadi.config.validation import validate_experiment
from saadi.context_strategies import get_strategy
from saadi.core.types import ChunkRecord
from saadi.core.state import state_from_snapshot
from saadi.core.types import DiscourseState
from saadi.io.hierarchy_cache import (
    export_llm_history_cache,
    load_canonicalized_chunk_records_from_cache_index,
)

from test_release import ROOT, runner

RAW = [
    "The old woman lived on the fourth floor. She lent money against pledges.",
    "Alyona Ivanovna handed him back the watch. Raskolnikov left the flat.",
    "He walked down the staircase and thought about the pledge.",
]


def _chunk(index: int) -> ChunkRecord:
    return ChunkRecord(doc_id="doc", chunk_id=f"doc:{index}", chunk_index=index, raw_text=RAW[index])


def _model_output(name: str, anchor: str, aliases: list[str] | None = None) -> str:
    return json.dumps(
        {
            "active_entities": [{"name": name, "description": [], "aliases": aliases or []}],
            "insertions": [{"name": name, "anchor_text": anchor, "left_context": "", "right_context": ""}],
        }
    )


def _history_l0_texts(strategy, state: DiscourseState, current_step: int) -> list[str]:
    context = strategy._hierarchy_context(state, current_step)
    return [row["text"] for row in context["selected_context_by_level"].get("L0", [])]


def _run_two_chunks(**strategy_kwargs):
    strategy = get_strategy("hierarchical", branching_factor=2, **strategy_kwargs)
    state = DiscourseState()
    _, state = strategy.apply_output(_chunk(0), state, _model_output("Alyona Ivanovna", "The old woman"))
    history_for_chunk_1 = _history_l0_texts(strategy, state, 1)
    _, state = strategy.apply_output(_chunk(1), state, _model_output("Alyona Ivanovna", "Alyona Ivanovna"))
    return strategy, state, history_for_chunk_1


def test_default_history_shows_raw_text_while_folds_use_anchored_copies():
    strategy, state, history = _run_two_chunks()

    assert strategy.history_l0_text == "raw"
    # The model processing chunk 1 sees chunk 0 exactly as written.
    assert history == [RAW[0]]
    # The anchored copy still exists and is what the fold summarizes (point 2).
    anchored = state.hierarchy_summaries[1][0]
    assert anchored == RAW[0].replace("The old woman", "The old woman (Alyona Ivanovna)")
    level_one = state.hierarchy_summaries[2][1]
    assert "(Alyona Ivanovna)" in level_one
    # The context never contains the inserted parenthetical, at any later step either.
    for text in _history_l0_texts(strategy, state, 2):
        assert "(Alyona Ivanovna)" not in text
    assert "original text of recent chunks" in strategy._hierarchy_context(state, 2)["description"]


def test_anchored_option_reproduces_the_saved_construction_context():
    strategy, state, history = _run_two_chunks(history_l0_text="anchored")

    assert history == [RAW[0].replace("The old woman", "The old woman (Alyona Ivanovna)")]
    assert state.hierarchy_summaries[1][0] == history[0]
    assert "recent materialized chunks" in strategy._hierarchy_context(state, 2)["description"]


def test_ungrounded_entity_is_still_rejected():
    strategy = get_strategy("hierarchical", branching_factor=2)
    output, state = strategy.apply_output(_chunk(0), DiscourseState(), _model_output("Ghost", "not in the chunk"))

    assert [entity.name for entity in output.rejected_entities] == ["Ghost"]
    assert state.entity_memory == []
    assert state.raw_chunk_texts == {0: RAW[0]}


def test_snapshot_and_cache_round_trips_keep_raw_texts(tmp_path):
    strategy, state, _ = _run_two_chunks()

    restored = state_from_snapshot(state.snapshot())
    assert restored.raw_chunk_texts == {0: RAW[0], 1: RAW[1]}
    assert _history_l0_texts(strategy, restored, 2) == _history_l0_texts(strategy, state, 2)

    # Snapshots written before raw texts existed fall back to the anchored copy.
    legacy = state.snapshot()
    legacy.pop("raw_chunk_texts")
    assert _history_l0_texts(strategy, state_from_snapshot(legacy), 2) == [state.hierarchy_summaries[1][1]]

    # A per-document cache rebuilt from disk restores the raw texts from steps_index.jsonl.
    from saadi.core.types import CanonicalizedChunkRecord

    records = []
    snapshot_state = DiscourseState()
    for index in range(2):
        output, snapshot_state = strategy.apply_output(
            _chunk(index), snapshot_state, _model_output("Alyona Ivanovna", RAW[index].split(".")[0])
        )
        records.append(
            CanonicalizedChunkRecord(
                doc_id="doc", chunk_id=f"doc:{index}", chunk_index=index, raw_chunk_text=RAW[index],
                canonicalized_chunk_text=output.updated_chunk_text, state_snapshot=snapshot_state.snapshot(),
                strategy_name=strategy.name, parser_status=output.parser_status, raw_model_text="",
                insertions=list(output.insertions),
            )
        )
    index = export_llm_history_cache(
        records, output_dir=tmp_path, dataset_label="d", method_label="hierarchical",
        hierarchy_branching_factor=2, chunk_size=500, active_entity_config_label="AE_5_2_3",
        llm_label="mock", active_entity_selector_label="AEBM25", history_l0_text=strategy.history_l0_text,
    )
    assert index["history_l0_text"] == "raw"
    reloaded = load_canonicalized_chunk_records_from_cache_index(index["index_path"])
    assert reloaded[-1].state_snapshot["raw_chunk_texts"] == {0: RAW[0], 1: RAW[1]}
    assert reloaded[-1].canonicalized_chunk_text != RAW[1]


def test_published_profile_restores_anchored_history(tmp_path, monkeypatch):
    from saadi.config.loader import load_config

    monkeypatch.setenv("SADI_OUTPUT_DIR", str(tmp_path))
    cfg = runner().published_config(load_config(ROOT / "configs/experiments/loogle_retrieval_ablation.yaml"))
    validate_experiment(cfg)
    assert cfg["ingest"]["strategy_kwargs"]["history_l0_text"] == "anchored"

    fresh = load_config(ROOT / "configs/experiments/loogle_retrieval_ablation.yaml")
    assert "history_l0_text" not in fresh["ingest"]["strategy_kwargs"]
    assert get_strategy("hierarchical", **fresh["ingest"]["strategy_kwargs"]).history_l0_text == "raw"
    fresh["ingest"]["strategy_kwargs"]["history_l0_text"] = "inserted"
    with pytest.raises(ValueError):
        validate_experiment(fresh)
