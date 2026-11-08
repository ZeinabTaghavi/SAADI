"""End to end through ingest_documents with the ledger profile on: stubs, promotions, aliases,
links, NOT AN ENTITY, the parse-failure policy, the description cap, the cache export the probe
reads, the exact replay from the event log, and regeneration after a resume."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from saadi.context_strategies import get_strategy
from saadi.core.state import state_from_snapshot
from saadi.core.types import ChunkRecord
from saadi.indexing.retrieval_methods.mbr import build_mbr_document_memory_index
from saadi.ledger.ledger import LedgerStore
from saadi.ledger.measure import (
    apply_description_cap,
    ledger_frozen_chunk_fields,
    memory_json_drift,
    provenance_counts,
    recalled_at_step,
    verify_ledger_against_replay,
)
from saadi.ledger.prompt import SECTION_CANDIDATES, SECTION_LEDGER, SECTION_OVERLAP
from saadi.llm import CallableBackend
from saadi.llm.base import GenerateParams
from saadi.pipelines.ingest_pipeline import ingest_documents
from saadi.rebuttal.frozen_records import replay_first_added

DOC = "doc"
CHUNKS = [
    "Doctor Sinchipuma had lived in Gavelhall for many years. The river was quiet that night.",
    "Huck and Jim drifted past the town. Jim spoke of Miss Watson for a while.",
    "In those days, Maysunah's family name was Sinchipuma. Huck laughed at the story.",
    "Jim met the Lantern Keeper by the ferry. The Lantern Keeper was Tom Sawyer in disguise.",
    "Huck wrote to Tom Sawyer. Huck was a boy from Missouri, poor, clever, brave, tired and free.",
    "The ferry reached Gavelhall at dusk, and Huck slept.",
]
SENTENCE = "Doctor Sinchipuma had lived in Gavelhall for many years."


def _ids_in_prompt(user: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for section in (SECTION_OVERLAP, SECTION_LEDGER):
        m = re.search(re.escape(section) + r"\n(.*?)\n\n\[", user, flags=re.DOTALL)
        if not m:
            continue
        for row in json.loads(m.group(1)):
            if row.get("id"):
                out[row["name"]] = row["id"]
    return out


def _candidates_in_prompt(user: str) -> list[str]:
    m = re.search(re.escape(SECTION_CANDIDATES) + r"\n(.*?)\n\n\[", user, flags=re.DOTALL)
    assert m
    return [re.sub(r"\s\(\w[\w ]*\)$", "", line.split(". ", 1)[1]) for line in m.group(1).splitlines() if line[0].isdigit()]


def _json(entities, insertions) -> str:
    return json.dumps({"active_entities": entities, "insertions": insertions}) + "\n[END_JSON]"


def make_mock(calls: list[str]):
    def fn(system: str, user: str) -> str:
        if "[ROLLUP_ITEMS_JSON]" in user:
            return "Summary of the group. [END_SUMMARY]"
        chunk = re.search(r"\[CURRENT_CHUNK\]\n(.*?)\n\n\[HIERARCHY", user, flags=re.DOTALL).group(1)
        step = CHUNKS.index(chunk)
        calls.append(chunk)
        cands = _candidates_in_prompt(user)
        ids = _ids_in_prompt(user)

        def line(n: int, text: str, decision: str) -> str:
            return f"{n}. {text} -> {decision}"

        def same(name: str, reason: str) -> str:
            return f"SAME AS {ids.get(name, name)} ({reason})"

        if step == 0:
            assert cands == ["Doctor Sinchipuma", "Gavelhall"], cands
            decisions = [line(1, cands[0], "NEW"), line(2, cands[1], "NEW")]
            body = _json([], [])
        elif step == 1:
            assert cands == ["Huck", "Jim", "Miss Watson"], cands
            decisions = [line(1, "Huck", "NEW"), line(2, "Jim", "NEW"), line(3, "Miss Watson", "NOT AN ENTITY")]
            body = _json(
                [
                    {"name": "Huck", "description": ["a boy on a raft"], "aliases": ["Huck"]},
                    {"name": "Jim", "description": ["a man travelling with Huck"], "aliases": []},
                ],
                [
                    {"name": "Huck", "anchor_text": "Huck", "left_context": "", "right_context": " and Jim"},
                    {"name": "Jim", "anchor_text": "Jim", "left_context": "Huck and ", "right_context": " drifted"},
                ],
            )
        elif step == 2:
            assert cands == ["Maysunah", "Sinchipuma", "Huck"], cands
            assert "Doctor Sinchipuma" in ids, ids  # ledger recall brought the stub into the prompt
            decisions = [
                line(1, "Maysunah", same("Doctor Sinchipuma", "the chunk says Maysunah's family name was Sinchipuma")),
                line(2, "Sinchipuma", same("Doctor Sinchipuma", "same family name as the recalled stub")),
                line(3, "Huck", same("Huck", "same person")),
            ]
            body = _json([], [])
        elif step == 3:
            # The decision block is missing: every candidate falls back to the stub policy.
            decisions = []
            body = _json(
                [{"name": "Tom Sawyer", "description": ["a boy in disguise"], "aliases": []}],
                [{"name": "Tom Sawyer", "anchor_text": "Tom Sawyer", "left_context": "was ", "right_context": " in disguise"}],
            )
        elif step == 4:
            decisions = [
                line(1, "Huck", same("Huck", "same person")),
                line(2, "Tom Sawyer", same("Tom Sawyer", "same person")),
                line(3, "Missouri", "NEW"),
            ]
            body = _json(
                [{"name": "Huck", "description": ["from Missouri", "poor", "clever", "brave", "tired"], "aliases": []}],
                [{"name": "Huck", "anchor_text": "Huck wrote", "left_context": "", "right_context": " to Tom"}],
            )
        else:
            decisions = [line(1, "Gavelhall", same("Gavelhall", "same place")), line(2, "Huck", same("Huck", "same person"))]
            body = _json([], [])
        head = "[DECISIONS]\n" + "\n".join(decisions) + "\n[END_DECISIONS]\n" if decisions else ""
        return head + body

    return fn


def _chunks() -> list[ChunkRecord]:
    return [ChunkRecord(doc_id=DOC, chunk_id=f"{DOC}:{i}", chunk_index=i, raw_text=t) for i, t in enumerate(CHUNKS)]


def _kwargs(store_dir: Path) -> dict:
    return dict(
        branching_factor=5,
        active_entity_selector="word",
        max_entities=5,
        max_aliases=2,
        max_desc=3,
        history_l0_text="anchored",
        anchor_matching="strict",
        ungrounded_entities="keep",
        ledger={"store_dir": str(store_dir), "candidate_net": "regex", "description_cap": 5},
    )


def _ingest(tmp_path: Path, calls: list[str]):
    return ingest_documents(
        documents_by_id={DOC: " ".join(CHUNKS)},
        backend=CallableBackend(make_mock(calls)),
        strategy_name="hierarchical",
        strategy_kwargs=_kwargs(tmp_path / "ledger"),
        cache_namespace={
            "dataset": "test",
            "method": "hierarchical",
            "llm": "mock",
            "active_entity_config": "AE_5_2_3",
            "active_entity_selector": "AEWord",
        },
        use_cache=True,
        cache_save_every_steps=1,
        output_dir=str(tmp_path / "out"),
        generate=GenerateParams(max_tokens=512, temperature=0.0, stop=["[END_JSON]"]),
        schedule="wavefront",
        precomputed_raw_chunks_by_doc_id={DOC: _chunks()},
    )


@pytest.fixture()
def built(tmp_path: Path):
    calls: list[str] = []
    raw, canon, artifacts = _ingest(tmp_path, calls)
    return tmp_path, calls, canon, artifacts


def test_ledger_mechanics_end_to_end(built) -> None:
    tmp_path, calls, canon, artifacts = built
    assert len(calls) == 6 and len(canon) == 6
    assert [r.parser_status for r in canon] == ["ok"] * 6  # the JSON blocks all parse; the missing decision block is counted separately
    assert canon[0].strategy_name == "hierarchical_summary_p1_ledger"

    store = LedgerStore(tmp_path / "ledger")
    ledger = store.doc(DOC)
    by_name = {e.name: e for e in ledger.entries.values()}
    sinch = by_name["Doctor Sinchipuma"]
    assert sinch.kind == "record" and sinch.created_step == 0 and sinch.promoted_step == 2
    assert sinch.sentence == SENTENCE
    assert set(sinch.aliases) == {"Maysunah", "Sinchipuma"}
    assert sinch.linked_chunks == [f"{DOC}:0", f"{DOC}:2"]
    assert sinch.provenance == ["candidate_net"]
    gav = by_name["Gavelhall"]
    assert gav.kind == "record" and gav.promoted_step == 5 and gav.linked_chunks == [f"{DOC}:0", f"{DOC}:5"]
    huck = by_name["Huck"]
    assert huck.kind == "record" and huck.created_step == 1 and huck.promoted_step is None
    assert sorted(huck.provenance) == ["candidate_net", "llm"] and huck.described_steps == [1, 4]
    lantern = by_name["the Lantern Keeper"]
    assert lantern.kind == "stub" and "parse_failure" in lantern.provenance and lantern.sentence.startswith("Jim met the Lantern Keeper")
    assert by_name["Tom Sawyer"].kind == "record" and by_name["Missouri"].kind == "stub"
    assert "Miss Watson" not in by_name
    c = ledger.counters
    assert c["not_an_entity"] == 1 and c["decision_block_parse_failures"] == 1 and c["promotions"] == 2
    assert c["description_replacements"] == 1 and c["parse_failure_candidates"] == 3

    final = state_from_snapshot(canon[-1].state_snapshot)
    memory = {e.name: e for e in final.entity_memory}
    assert memory["Doctor Sinchipuma"].description[0] == SENTENCE
    # Six descriptions arrived for Huck; the weakest (fewest content tokens; ties go to the oldest) was replaced.
    assert len(memory["Huck"].description) == 5 and "from Missouri" not in memory["Huck"].description
    assert memory["the Lantern Keeper"].description == [] and memory["the Lantern Keeper"].aliases == []
    assert huck.dropped_descriptions == ["from Missouri"]

    # What the probe's measurement reads: memory.json has the stubs, chunks are linked to every NEW/SAME AS name.
    index = build_mbr_document_memory_index(artifacts.llm_cache_index_json, doc_id=DOC)
    names = {e.name for e in index.memory_entities}
    assert {"the Lantern Keeper", "Missouri", "Doctor Sinchipuma"} <= names
    assert set(index.chunk_to_names[f"{DOC}:0"]) == {"Doctor Sinchipuma", "Gavelhall"}
    assert set(index.chunk_to_names[f"{DOC}:2"]) == {"Doctor Sinchipuma", "Huck"}
    assert set(index.chunk_to_names[f"{DOC}:3"]) == {"Jim", "the Lantern Keeper", "Tom Sawyer"}
    assert "Maysunah" in index.entity_to_aliases["Doctor Sinchipuma"]
    assert index.entity_to_descriptions["the Lantern Keeper"] == []

    # Frozen records: the replay dates the promotion; the annotation says chunk 0 saw a stub.
    from saadi.indexing.retrieval_methods.mbr import _find_doc_entry, _load_doc_records, _read_index_payload

    payload = _read_index_payload(Path(artifacts.llm_cache_index_json))
    records = _load_doc_records(
        index_path=Path(artifacts.llm_cache_index_json), index_payload=payload, doc_entry=_find_doc_entry(payload, doc_id=DOC), doc_id=DOC
    )
    replay = replay_first_added(DOC, records)
    assert verify_ledger_against_replay(ledger, replay) == []
    frozen, notes = ledger_frozen_chunk_fields(index, replay, ledger)
    note = {n.name: n for n in notes[f"{DOC}:0"]}
    assert note["Doctor Sinchipuma"].stub_at_chunk_step and note["Doctor Sinchipuma"].promoted_step == 2
    assert frozen[f"{DOC}:0"].frozen_descriptions == [] and SENTENCE in frozen[f"{DOC}:0"].live_descriptions
    assert "Maysunah" in frozen[f"{DOC}:0"].live_aliases and "Maysunah" not in frozen[f"{DOC}:0"].frozen_aliases

    # Prompt bookkeeping and provenance views.
    at2 = recalled_at_step(ledger, 2)
    assert (
        "Doctor Sinchipuma" in at2["ledger"] + at2["overlap"] and at2["decision_status"] == "ok"
    )  # the stub was in the prompt (overlap slot here)
    prov = provenance_counts(ledger)["entries"]
    assert prov["record:both"] >= 2 and prov["stub:candidate_net"] >= 2
    assert (tmp_path / "ledger" / "system_prompt.txt").exists()


def test_event_log_replays_to_the_live_ledger_and_resume_uses_the_cache(built) -> None:
    tmp_path, calls, canon, artifacts = built
    live = LedgerStore(tmp_path / "ledger").doc(DOC).snapshot()
    replayed = LedgerStore(tmp_path / "ledger").doc(DOC).snapshot()  # a fresh store reads events.jsonl
    assert replayed == live
    # Second run: everything comes from the cache, no LLM call, ledger untouched.
    calls2: list[str] = []
    _, canon2, art2 = _ingest(tmp_path, calls2)
    assert calls2 == []
    assert LedgerStore(tmp_path / "ledger").doc(DOC).snapshot() == live
    assert [r.raw_model_text for r in canon2] == [r.raw_model_text for r in canon]
    assert [[e["name"] for e in r.state_snapshot["active_entities"]] for r in canon2] == [
        [e["name"] for e in r.state_snapshot["active_entities"]] for r in canon
    ]
    # The cache loader rebuilds memory by merging the step rows (uncapped), so the replaced
    # description is back in the rebuilt snapshots and in the re-exported memory.json ...
    huck2 = next(e for e in canon2[-1].state_snapshot["entity_memory"] if e["name"] == "Huck")
    assert "from Missouri" in huck2["description"]
    ledger = LedgerStore(tmp_path / "ledger").doc(DOC)
    memory_json = json.loads(Path(json.loads(Path(art2.llm_cache_index_json).read_text())["docs"][0]["files"]["memory"]).read_text())
    assert memory_json_drift(memory_json["entities"], ledger) == [
        {"id": "E0003", "name": "Huck", "replaced_but_present": ["from Missouri"]}
    ]
    # ... and the measurement applies the ledger's cap before any scoring.
    index2 = build_mbr_document_memory_index(art2.llm_cache_index_json, doc_id=DOC)
    counts = apply_description_cap(index2, ledger)
    assert counts == {"records_with_replacements": 1, "descriptions_removed_from_index": 1}
    assert index2.entity_to_descriptions["Huck"] == ["a boy on a raft", "poor", "clever", "brave", "tired"]
    assert "from Missouri" not in index2.description_pool
    assert all("from Missouri" not in d for d in index2.chunk_to_descriptions.values())


def test_regenerating_a_step_truncates_the_log_and_reproduces_it(built) -> None:
    tmp_path, calls, canon, artifacts = built
    store = LedgerStore(tmp_path / "ledger")
    before = store.doc(DOC).snapshot()
    event3 = store.read_events(DOC)[3]
    # A resume that lost steps 3..5 of the cache: the state of step 2 is restored and step 3 is rebuilt.
    strategy = get_strategy("hierarchical", **_kwargs(tmp_path / "ledger"))
    state = state_from_snapshot(canon[2].state_snapshot)
    chunk = _chunks()[3]
    prompt = strategy.build_prompt(chunk, state)
    assert LedgerStore(tmp_path / "ledger").doc(DOC).last_step == 2 and len(store.read_events(DOC)) == 3
    # The restored memory still has the description that step 4 replaced? No: step 4 is gone with its event.
    assert "Doctor Sinchipuma" in prompt.user
    reply = make_mock([])(prompt.system, prompt.user)
    out, new_state, gen = strategy.apply_output_deferred(chunk, state, reply)
    event3_again = store.read_events(DOC)[3]
    for key in ("ops", "decisions", "counts", "candidates", "links", "new_entries"):
        assert event3_again[key] == event3[key], key
    assert [e["step"] for e in store.read_events(DOC)] == [0, 1, 2, 3]
    assert LedgerStore(tmp_path / "ledger").doc(DOC).snapshot() != before  # steps 4 and 5 are forgotten until rebuilt


def test_reconcile_removes_replaced_descriptions_after_a_resume(built) -> None:
    tmp_path, calls, canon, artifacts = built
    strategy = get_strategy("hierarchical", **_kwargs(tmp_path / "ledger"))
    # The cache loader rebuilds memory by merging every step's active entities, so the replaced
    # description comes back; build_prompt must remove it again before the prompt is built.
    state = state_from_snapshot(canon[4].state_snapshot)
    huck = next(e for e in state.entity_memory if e.name == "Huck")
    huck.description.append("from Missouri")
    assert len(huck.description) == 6
    strategy.build_prompt(_chunks()[5], state)
    huck = next(e for e in state.entity_memory if e.name == "Huck")
    assert "from Missouri" not in huck.description and len(huck.description) == 5
