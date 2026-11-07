"""Synthetic check of the E2 frozen-record construction.

Two chunks. The entity "Acme" gains the alias "Apex Carbide" and one description
at step 1 (the second chunk). The frozen record of chunk 0 must not contain
them; the live record must; the last chunk's frozen record must equal live.
"""

from __future__ import annotations

import pytest

from saadi.core.types import CanonicalizedChunkRecord
from saadi.indexing.retrieval_methods.mbr import (
    MBRChunkPoolRecord,
    MBRDocumentMemoryIndex,
)
from saadi.rebuttal.frozen_records import (
    ReplayCheckError,
    frozen_chunk_fields,
    replay_first_added,
    score_query_variants,
    verify_replay_matches_final_memory,
)

DOC = "doc-1"
C0, C1 = f"{DOC}:0", f"{DOC}:1"


def _record(step: int, active: list[dict]) -> CanonicalizedChunkRecord:
    return CanonicalizedChunkRecord(
        doc_id=DOC,
        chunk_id=f"{DOC}:{step}",
        chunk_index=step,
        raw_chunk_text=f"chunk {step}",
        canonicalized_chunk_text=f"chunk {step}",
        state_snapshot={"active_entities": active},
        strategy_name="test",
    )


@pytest.fixture()
def records() -> list[CanonicalizedChunkRecord]:
    return [
        _record(0, [{"name": "Acme", "aliases": ["Acme"], "description": ["makes tools"]}]),
        _record(
            1,
            [
                {
                    "name": "Acme",
                    "aliases": ["Acme", "Apex Carbide"],
                    "description": ["makes tools", "renamed to Apex Carbide"],
                },
                {"name": "Beta", "aliases": ["Beta"], "description": []},
            ],
        ),
    ]


@pytest.fixture()
def document_index() -> MBRDocumentMemoryIndex:
    # Mirrors what build_mbr_document_memory_index derives from the final memory:
    # chunk 0 is linked to Acme; chunk 1 (whose step added the alias) to Acme and Beta.
    return MBRDocumentMemoryIndex(
        doc_id=DOC,
        chunk_pool=[
            MBRChunkPoolRecord(chunk_id=C0, text="alpha one two", step=0),
            MBRChunkPoolRecord(chunk_id=C1, text="delta five six", step=1),
        ],
        name_pool=["Acme", "Beta"],
        alias_pool=["Acme", "Apex Carbide", "Beta"],
        description_pool=["makes tools", "renamed to Apex Carbide"],
        chunk_to_names={C0: ["Acme"], C1: ["Acme", "Beta"]},
        entity_to_aliases={"Acme": ["Acme", "Apex Carbide"], "Beta": ["Beta"]},
        entity_to_descriptions={"Acme": ["makes tools", "renamed to Apex Carbide"]},
    )


def test_frozen_excludes_fields_added_later(records, document_index):
    replay = replay_first_added(DOC, records)
    assert replay.alias_first[("acme", "Apex Carbide")] == 1
    assert replay.desc_first[("acme", "renamed to Apex Carbide")] == 1

    frozen = frozen_chunk_fields(document_index, replay)
    first, last = frozen[C0], frozen[C1]

    # Frozen record of chunk 0 must not have the later alias/description.
    assert first.frozen_aliases == ["Acme"]
    assert first.frozen_descriptions == ["makes tools"]
    assert {(f.kind, f.value) for f in first.late_fields} == {
        ("alias", "Apex Carbide"),
        ("description", "renamed to Apex Carbide"),
    }
    assert all(f.added_at_step == 1 for f in first.late_fields)

    # The live record has them, and frozen stays a subset of live.
    assert "Apex Carbide" in first.live_aliases
    assert set(first.frozen_aliases) <= set(first.live_aliases)
    assert set(first.frozen_descriptions) <= set(first.live_descriptions)

    # For the last chunk of the document, frozen equals live.
    assert last.frozen_aliases == last.live_aliases
    assert last.frozen_descriptions == last.live_descriptions
    assert not last.late_fields


def test_replay_reproduces_final_memory(records):
    replay = replay_first_added(DOC, records)
    final_memory = [
        {
            "name": "Acme",
            "aliases": ["Acme", "Apex Carbide"],
            "description": ["makes tools", "renamed to Apex Carbide"],
        },
        {"name": "Beta", "aliases": ["Beta"], "description": []},
    ]
    verify_replay_matches_final_memory(replay, final_memory)

    with pytest.raises(ReplayCheckError):
        verify_replay_matches_final_memory(
            replay,
            [
                {"name": "Acme", "aliases": ["Acme", "Ghost Alias"], "description": ["makes tools"]},
                {"name": "Beta", "aliases": ["Beta"], "description": []},
            ],
        )


def test_variant_rankings_differ_only_through_the_late_fields(records, document_index):
    replay = replay_first_added(DOC, records)
    frozen = frozen_chunk_fields(document_index, replay)
    # The query matches only the late alias/description. Neither chunk text does.
    rankings = score_query_variants(document_index, frozen, query="what became of Apex Carbide", retriever_name="bm25", retrieve_k=2)
    # LIVE: both chunks carry the late fields through the shared record, so the
    # memory channel ties and the earlier chunk stays first (index tie-break).
    assert rankings["live"] == [C0, C1]
    # FROZEN: only chunk 1 (whose step added the fields) has them, so it moves up.
    assert rankings["frozen"] == [C1, C0]
    # NO-MEMORY ignores the memory channel and falls back to the text tie-break.
    assert rankings["no_memory"] == [C0, C1]
