"""Profile p1b_ledger: the three changes of run 1b, each against the case it targets.

1. Join direction: a SAME AS never creates a record; a JSON record named like a
   SAME AS candidate is redirected into the target; auto-join of a NEW candidate
   equal to a shown entry, with and without a title conflict.
2. Recall by chunk tokens: priority full name > full alias > single token,
   ties by the most recent mention, under the cap.
3. SAME AS acceptance: (a) shared token, (b) same sentence, title conflict,
   type conflict; alias hygiene.
Plus: the profile differs from p1_ledger only in the stated keys, and the
probe's reveal sentence end to end through ingest_documents.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import yaml

from saadi.core.types import EntityRecord
from saadi.ledger.accounting import NEW, parse_decisions
from saadi.ledger.candidate_net import Candidate, split_sentences
from saadi.ledger.ledger import DocumentLedger, LedgerStore
from saadi.ledger.p1b import (
    P1bConfig,
    P1bStep,
    accept_same_as,
    filter_aliases,
    recall_by_chunk_tokens,
    title_conflict,
    title_stripped_form,
    type_conflict,
)
from saadi.ledger.promotion import StepPlan, apply_decisions

ROOT = Path(__file__).resolve().parents[2]
ALL_ON = P1bConfig(join_direction=True, chunk_token_recall=True, same_as_rule=True, aliases_from_spans=True)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def ledger_with(*rows: tuple[str, str, int, str | None, list[str]]) -> DocumentLedger:
    """rows: (name, kind, step, label, aliases)."""

    ledger = DocumentLedger("d")
    for name, kind, step, label, aliases in rows:
        entry_id = ledger.new_id()
        ledger.apply_op(
            {
                "op": "create",
                "id": entry_id,
                "name": name,
                "kind": kind,
                "step": step,
                "chunk_id": f"d:{step}",
                "sentence": f"{name} was here.",
                "provenance": ["candidate_net"],
            }
        )
        if label:
            ledger.apply_op({"op": "label", "id": entry_id, "label": label})
        for alias in aliases:
            ledger.apply_op({"op": "alias", "id": entry_id, "alias": alias, "step": step})
    return ledger


def cand(text: str, chunk: str, label: str = "person") -> Candidate:
    start = chunk.find(text)
    sentence = next((chunk[s:e].strip() for s, e in split_sentences(chunk) if s <= start < e), chunk)
    return Candidate(text=text, label=label, score=0.9, start=start, end=start + len(text), sentence=sentence)


def plan_for(
    ledger: DocumentLedger, chunk: str, cands: list[Candidate], *, step: int, shown: list[str], cfg: P1bConfig | None = ALL_ON
) -> StepPlan:
    return StepPlan(
        doc_id="d",
        chunk_id=f"d:{step}",
        step=step,
        candidates=cands,
        dropped_candidates=[],
        candidate_raw_count=len(cands),
        candidate_seconds=0.0,
        candidate_cap_hit=False,
        candidate_device="cpu",
        overlap_ids=[],
        overlap_names=[],
        ledger_ids=list(shown),
        left_out_ids=[],
        cap_hit=False,
        matched_candidates={},
        reconcile_ops=[],
        recovered_names=[],
        dropped_descriptions_removed=0,
        prompt_system="",
        prompt_user="",
        p1b=P1bStep(config=cfg, chunk_text=chunk, shown_ids=list(shown)) if cfg is not None else None,
    )


def decide(ledger, chunk, cands, lines, described=None, *, step=5, shown=None, cfg=ALL_ON):
    block = parse_decisions("[DECISIONS]\n" + "\n".join(lines) + "\n[END_DECISIONS]\n{}", [c.text for c in cands])
    plan = plan_for(ledger, chunk, cands, step=step, shown=shown if shown is not None else list(ledger.entries), cfg=cfg)
    return apply_decisions(ledger=ledger, plan=plan, block=block, described=list(described or []), cut_names=[])


def by_name(ledger: DocumentLedger) -> dict:
    return {e.name: e for e in ledger.entries.values()}


# The probe's family A reveal: form 1 in the gold chunk, form 2 revealed later.
REVEAL = "In those days, Maysunah's family name was Sinchipuma. Huck laughed at the story."


# ---------------------------------------------------------------------------
# 1. Join direction
# ---------------------------------------------------------------------------


def test_same_as_never_creates_a_record_and_redirects_the_json_record_into_the_target() -> None:
    ledger = ledger_with(("Doctor Sinchipuma", "record", 0, "person", []))
    target = ledger.get("E0001")
    cands = [cand("Maysunah", REVEAL), cand("Sinchipuma", REVEAL)]
    # The model joins Maysunah to the shown record, then describes it anyway as a new
    # record under a title variant of the SAME AS candidate.
    described = [EntityRecord(name="Mrs. Maysunah", description=["her family name is Sinchipuma"], aliases=["Maysunah"])]
    out = decide(
        ledger,
        REVEAL,
        cands,
        [
            "1. Maysunah -> SAME AS E0001 (the chunk says Maysunah's family name was Sinchipuma)",
            "2. Sinchipuma -> SAME AS E0001 (same name)",
        ],
        described,
    )
    assert len(ledger.entries) == 1, "a SAME AS candidate or its JSON record created a record"
    assert out.counts["redirected"] == 1 and out.redirects == [{"from": "Mrs. Maysunah", "to": "Doctor Sinchipuma", "id": "E0001"}]
    assert "Maysunah" in target.aliases
    rec = next(r for r in out.active_entities if r.name == "Doctor Sinchipuma")
    assert "her family name is Sinchipuma" in rec.description  # the descriptions merged into the target
    assert "Mrs. Maysunah" not in rec.aliases  # not a span of the chunk: dropped by alias hygiene
    assert all(r.name != "Mrs. Maysunah" for r in out.active_entities)


def test_without_p1b_the_same_reply_creates_the_extra_record() -> None:
    """The p1 path is unchanged: the title-variant JSON record becomes its own record."""

    ledger = ledger_with(("Doctor Sinchipuma", "record", 0, None, []))
    cands = [cand("Maysunah", REVEAL), cand("Sinchipuma", REVEAL)]
    described = [EntityRecord(name="Mrs. Maysunah", description=["her family name is Sinchipuma"], aliases=["Maysunah"])]
    decide(ledger, REVEAL, cands, ["1. Maysunah -> SAME AS E0001 (x)", "2. Sinchipuma -> SAME AS E0001 (x)"], described, cfg=None)
    assert "Mrs. Maysunah" in by_name(ledger)


def test_auto_join_of_a_new_candidate_equal_to_a_shown_entry() -> None:
    chunk = "Sinchipuma walked to the ferry."
    ledger = ledger_with(("Doctor Sinchipuma", "stub", 0, "person", []))
    out = decide(ledger, chunk, [cand("Sinchipuma", chunk)], ["1. Sinchipuma -> NEW"])
    assert out.counts["auto_join"] == 1 and len(ledger.entries) == 1
    entry = ledger.get("E0001")
    assert entry.kind == "record" and entry.promoted_step == 5  # a joined stub is promoted
    assert out.decisions[0]["auto_join"] == {"id": "E0001", "name": "Doctor Sinchipuma"} and out.decisions[0]["model_decision"] == NEW


def test_auto_join_only_for_entries_shown_in_the_prompt() -> None:
    chunk = "Sinchipuma walked to the ferry."
    ledger = ledger_with(("Doctor Sinchipuma", "record", 0, "person", []))
    out = decide(ledger, chunk, [cand("Sinchipuma", chunk)], ["1. Sinchipuma -> NEW"], shown=[])
    assert out.counts["auto_join"] == 0 and len(ledger.entries) == 2


def test_auto_join_is_blocked_by_a_title_conflict() -> None:
    chunk = "Mrs. Bates poured the tea."
    ledger = ledger_with(("Miss Bates", "record", 0, "person", []))
    out = decide(ledger, chunk, [cand("Mrs. Bates", chunk)], ["1. Mrs. Bates -> NEW"])
    assert out.counts["auto_join"] == 0 and out.counts["auto_join_blocked_by_conflict"] == 1
    assert {"Miss Bates", "Mrs. Bates"} <= set(by_name(ledger)) and ledger.get("E0001").aliases == []


def test_auto_join_is_blocked_by_a_type_conflict() -> None:
    chunk = "They reached Hartfield at noon."
    ledger = ledger_with(("Hartfield", "record", 0, "person", []))
    out = decide(ledger, chunk, [cand("Hartfield", chunk, label="place")], ["1. Hartfield -> NEW"])
    assert out.counts["auto_join"] == 0 and out.counts["auto_join_blocked_by_conflict"] == 1


def test_json_alias_join_is_off_by_default_and_joins_when_switched_on() -> None:
    """Run 1's wrong direction: form 2 answered NEW and described with form 1 as its alias."""

    lines = ["1. Maysunah -> NEW", "2. Sinchipuma -> SAME AS E0001 (same name)"]

    def described():
        return [EntityRecord(name="Maysunah", description=["family name Sinchipuma"], aliases=["Sinchipuma"])]

    ledger = ledger_with(("Doctor Sinchipuma", "record", 0, "person", []))
    cands = [cand("Maysunah", REVEAL), cand("Sinchipuma", REVEAL)]
    decide(ledger, REVEAL, cands, lines, described())
    assert "Maysunah" in by_name(ledger), "with the switch off, the separate record is kept (spec as written)"

    on = P1bConfig(join_direction=True, chunk_token_recall=True, same_as_rule=True, aliases_from_spans=True, json_alias_join=True)
    ledger = ledger_with(("Doctor Sinchipuma", "record", 0, "person", []))
    out = decide(ledger, REVEAL, cands, lines, described(), cfg=on)
    assert out.counts["json_alias_join"] == 1 and len(ledger.entries) == 1
    assert "Maysunah" in ledger.get("E0001").aliases
    assert out.decisions[0]["json_alias_join"]["acceptance"] == "same_sentence"


# ---------------------------------------------------------------------------
# 3. SAME AS acceptance rule and alias hygiene
# ---------------------------------------------------------------------------


def test_acceptance_a_shared_name_token() -> None:
    chunk = "Edgar came home late."
    ledger = ledger_with(("Mr. Edgar Linton", "record", 0, "person", []))
    v = accept_same_as(cand("Edgar", chunk), ledger.get("E0001"), ledger, chunk)
    assert v.accepted and v.reason == "shared_token"


def test_acceptance_b_same_sentence() -> None:
    ledger = ledger_with(("Doctor Sinchipuma", "record", 0, "person", []))
    v = accept_same_as(cand("Maysunah", REVEAL), ledger.get("E0001"), ledger, REVEAL)
    assert v.accepted and v.reason == "same_sentence" and v.detail.startswith("In those days")


def test_acceptance_rejects_without_shared_token_or_sentence_and_the_candidate_becomes_new() -> None:
    chunk = "Maysunah sang. Later, far away, Sinchipuma slept."
    ledger = ledger_with(("Doctor Sinchipuma", "record", 0, "person", []))
    c = cand("Maysunah", chunk)
    assert accept_same_as(c, ledger.get("E0001"), ledger, chunk).reason == "no_shared_token_or_sentence"
    out = decide(ledger, chunk, [c], ["1. Maysunah -> SAME AS E0001 (guess)"])
    assert out.counts["same_as_rejected_no_shared_token_or_sentence"] == 1
    assert by_name(ledger)["Maysunah"].kind == "stub" and ledger.get("E0001").aliases == []
    assert out.decisions[0]["rejected_target_id"] == "E0001" and out.decisions[0]["outcome"] == "stub"


def test_acceptance_rejects_a_title_conflict_even_with_a_shared_token() -> None:
    chunk = "Mrs. Bumble scolded the boy."
    ledger = ledger_with(("Mr. Bumble", "record", 0, "person", ["Bumble"]))
    v = accept_same_as(cand("Mrs. Bumble", chunk), ledger.get("E0001"), ledger, chunk)
    assert not v.accepted and v.reason == "title_conflict"
    out = decide(ledger, chunk, [cand("Mrs. Bumble", chunk)], ["1. Mrs. Bumble -> SAME AS E0001 (same family)"])
    assert out.counts["same_as_rejected_title_conflict"] == 1 and "Mrs. Bumble" in by_name(ledger)
    assert "Mrs. Bumble" not in ledger.get("E0001").aliases


def test_acceptance_rejects_a_type_conflict() -> None:
    chunk = "Sikes ran towards London that night."
    ledger = ledger_with(("London", "record", 0, "place", []))
    v = accept_same_as(cand("Sikes", chunk), ledger.get("E0001"), ledger, chunk)
    assert not v.accepted and v.reason == "type_conflict"


def test_title_and_type_rules() -> None:
    assert title_conflict("Mrs. Bates", "Miss Bates") and title_conflict("Mrs. Linton", "Mr. Edgar Linton")
    assert not title_conflict("Bates", "Miss Bates") and not title_conflict("Dr. Losberne", "Doctor Losberne")
    assert not title_conflict("Mrs. Bates", "Mrs. Elton")
    assert type_conflict("person", "place") and type_conflict("organization", "person")
    assert not type_conflict("person", "named object") and not type_conflict("person", "") and not type_conflict("span", "place")
    assert title_stripped_form("Doctor Sinchipuma") == "Sinchipuma" and title_stripped_form("the Firkin Cooper") == "the Firkin Cooper"


def test_alias_hygiene_keeps_spans_and_drops_the_rest() -> None:
    chunk = "Doctor Sinchipuma and Mrs. Linton spoke of the cartographer."
    cands = [cand("Doctor Sinchipuma", chunk), cand("Mrs. Linton", chunk)]
    ledger = ledger_with(("Mr. Edgar Linton", "record", 0, "person", []))
    recs = [
        EntityRecord(name="Maysunah", aliases=["Doctor Sinchipuma", "Sinchipuma", "the cartographer", "Mrs. Maysunah"]),
        EntityRecord(name="Mr. Edgar Linton", aliases=["Mrs. Linton"]),
    ]
    counts: Counter = Counter()
    log = filter_aliases(recs, cands, ledger, counts=counts)
    assert recs[0].aliases == ["Doctor Sinchipuma", "Sinchipuma"]  # a span, and its title-stripped form
    assert recs[1].aliases == []  # a span, but Mrs. against Mr. on the same stripped name
    assert counts["alias_rejected_not_a_span"] == 2 and counts["alias_rejected_title_conflict"] == 1
    assert {"record": "Mr. Edgar Linton", "alias": "Mrs. Linton", "why": "title_conflict"} in log


# ---------------------------------------------------------------------------
# 2. Recall by chunk tokens
# ---------------------------------------------------------------------------


def test_recall_priority_full_name_then_alias_then_single_token_then_recency_under_the_cap() -> None:
    chunk = "Edgar Linton spoke with Nelly about Heathcliff and the Grange. Linton nodded."
    ledger = ledger_with(
        ("Mrs. Dean", "record", 0, "person", ["Nelly"]),  # E0001: full alias
        ("Edgar Linton", "record", 1, "person", []),  # E0002: full name
        ("Isabella Linton", "record", 2, "person", []),  # E0003: single token, older
        ("Heathcliff", "stub", 3, "person", []),  # E0004: full name, more recent than E0002
        ("Thrushcross Grange", "record", 4, "place", []),  # E0005: single token, most recent
        ("Joseph", "record", 5, "person", []),  # E0006: no match
        ("Mr. Lockwood", "record", 6, None, ["Lockwood"]),  # E0007: no match (title stripped)
    )
    recalled, left_out, cap_hit, matches = recall_by_chunk_tokens(ledger, chunk, exclude_keys=set(), cap=3)
    assert [e.id for e in recalled] == ["E0004", "E0002", "E0001"]
    assert [e.id for e in left_out] == ["E0005", "E0003"] and cap_hit
    assert matches["E0001"]["class"] == "full_alias" and matches["E0001"]["via"] == "Nelly"
    assert matches["E0003"] == {"class": "single_token", "via": "Isabella Linton", "tokens": ["linton"]}
    assert "E0006" not in matches and "E0007" not in matches
    # Excluded names (the overlap group) never count, and short tokens never match.
    recalled, _, _, _ = recall_by_chunk_tokens(ledger, chunk, exclude_keys={"heathcliff"}, cap=8)
    assert "E0004" not in [e.id for e in recalled]
    short = ledger_with(("Al", "record", 0, None, []))
    assert recall_by_chunk_tokens(short, "Al ran.", cap=8)[0] == []


def test_recall_is_independent_of_the_candidate_net() -> None:
    """The reveal's earlier record is recalled even when the net proposed nothing there."""

    ledger = ledger_with(("Doctor Sinchipuma", "record", 0, "person", []))
    recalled, *_ = recall_by_chunk_tokens(ledger, REVEAL, cap=8)
    assert [e.name for e in recalled] == ["Doctor Sinchipuma"]
    assert ledger.recall([], cap=5)[0] == []  # p1's recall needs a candidate


# ---------------------------------------------------------------------------
# The profile and the run-1 path
# ---------------------------------------------------------------------------


P1B_KEYS = {
    "candidate_threshold": 0.3,
    "recall": "chunk_tokens",
    "recall_cap": 8,
    "join_direction": "enforced",
    "json_alias_join": False,
    "same_as_rule": "p1b",
    "aliases_from_spans": True,
}


def test_p1b_profile_differs_from_p1_only_where_stated() -> None:
    p1 = yaml.safe_load((ROOT / "configs/profiles/p1_ledger.yaml").read_text())
    p1b = yaml.safe_load((ROOT / "configs/profiles/p1b_ledger.yaml").read_text())
    assert p1b["base"] == p1["base"] == "published"
    l1, l1b = p1["overrides"]["ingest"]["strategy_kwargs"]["ledger"], p1b["overrides"]["ingest"]["strategy_kwargs"]["ledger"]
    assert set(p1["overrides"]) == set(p1b["overrides"]) == {"ingest"}
    diff = {k: l1b.get(k) for k in set(l1) | set(l1b) if l1.get(k) != l1b.get(k)}
    assert diff == P1B_KEYS, diff
    assert P1bConfig.from_cfg(l1) == P1bConfig()  # p1: every rule off
    assert P1bConfig.from_cfg(l1b) == ALL_ON


# ---------------------------------------------------------------------------
# End to end: the probe's reveal through ingest_documents with the p1b profile
# ---------------------------------------------------------------------------

CHUNKS = [
    "Doctor Sinchipuma had lived in Gavelhall for many years. The river was quiet that night.",
    "Huck and Jim drifted past the town for a long while.",
    REVEAL,
]


def _ids(user: str) -> dict[str, str]:
    from saadi.ledger.prompt import SECTION_LEDGER, SECTION_OVERLAP

    out = {}
    for section in (SECTION_OVERLAP, SECTION_LEDGER):
        m = re.search(re.escape(section) + r"\n(.*?)\n\n\[", user, flags=re.DOTALL)
        for row in json.loads(m.group(1)) if m else []:
            if row.get("id"):
                out[row["name"]] = row["id"]
    return out


def _mock(system: str, user: str) -> str:
    if "[ROLLUP_ITEMS_JSON]" in user:
        return "Summary. [END_SUMMARY]"
    chunk = re.search(r"\[CURRENT_CHUNK\]\n(.*?)\n\n\[HIERARCHY", user, flags=re.DOTALL).group(1)
    step = CHUNKS.index(chunk)
    body = json.dumps({"active_entities": [], "insertions": []}) + "\n[END_JSON]"
    if step == 0:
        lines = ["1. Doctor Sinchipuma -> NEW", "2. Gavelhall -> NEW"]
    elif step == 1:
        lines = ["1. Huck -> NEW", "2. Jim -> NEW"]
    else:
        ids = _ids(user)
        assert "Doctor Sinchipuma" in ids, "the earlier record must be in the prompt at the reveal"
        # Run 1's failure: form 2 answered NEW and described as its own record with form 1 as alias;
        # form 1 answered NEW too (the model did not see it as the shown stub).
        lines = ["1. Maysunah -> NEW", "2. Sinchipuma -> NEW", "3. Huck -> SAME AS " + ids.get("Huck", "Huck") + " (same boy)"]
        body = (
            json.dumps(
                {
                    "active_entities": [{"name": "Maysunah", "description": ["her family name is Sinchipuma"], "aliases": ["Sinchipuma"]}],
                    "insertions": [
                        {"name": "Maysunah", "anchor_text": "Maysunah", "left_context": "In those days, ", "right_context": "'s family"}
                    ],
                }
            )
            + "\n[END_JSON]"
        )
    return "[DECISIONS]\n" + "\n".join(lines) + "\n[END_DECISIONS]\n" + body


def _ingest(tmp_path: Path, **ledger_extra):
    from saadi.core.types import ChunkRecord
    from saadi.llm import CallableBackend
    from saadi.llm.base import GenerateParams
    from saadi.pipelines.ingest_pipeline import ingest_documents

    ledger_cfg = {
        "store_dir": str(tmp_path / "ledger"),
        "candidate_net": "regex",
        "description_cap": 5,
        "recall_cap": 8,
        "recall": "chunk_tokens",
        "join_direction": "enforced",
        "same_as_rule": "p1b",
        "aliases_from_spans": True,
        **ledger_extra,
    }
    chunks = [ChunkRecord(doc_id="doc", chunk_id=f"doc:{i}", chunk_index=i, raw_text=t) for i, t in enumerate(CHUNKS)]
    return ingest_documents(
        documents_by_id={"doc": " ".join(CHUNKS)},
        backend=CallableBackend(_mock),
        strategy_name="hierarchical",
        strategy_kwargs=dict(
            branching_factor=5,
            active_entity_selector="word",
            max_entities=5,
            max_aliases=2,
            max_desc=3,
            history_l0_text="anchored",
            anchor_matching="strict",
            ungrounded_entities="keep",
            ledger=ledger_cfg,
        ),
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
        precomputed_raw_chunks_by_doc_id={"doc": chunks},
    )


def test_reveal_end_to_end_auto_join_of_form_1_and_the_switch_for_form_2(tmp_path: Path) -> None:
    from saadi.indexing.retrieval_methods.mbr import build_mbr_document_memory_index

    _, _, art = _ingest(tmp_path / "spec")
    ledger = LedgerStore(tmp_path / "spec" / "ledger").doc("doc")
    names = by_name(ledger)
    sinch = names["Doctor Sinchipuma"]
    # Form 1 at the reveal ("Sinchipuma", answered NEW) is auto-joined into the shown stub.
    assert ledger.counters["auto_join"] == 1 and "Sinchipuma" in sinch.aliases and "doc:2" in sinch.linked_chunks
    # Spec as written: form 2 keeps its own record (the wrong direction of run 1).
    assert "Maysunah" in names and "Maysunah" not in sinch.aliases
    event = ledger.events[2]
    assert event["p1b"]["recall"] == "chunk_tokens" and sinch.id in event["p1b"]["shown_ids"]

    _, _, art2 = _ingest(tmp_path / "on", json_alias_join=True)
    ledger2 = LedgerStore(tmp_path / "on" / "ledger").doc("doc")
    sinch2 = by_name(ledger2)["Doctor Sinchipuma"]
    assert ledger2.counters["json_alias_join"] == 1 and "Maysunah" in sinch2.aliases and "Maysunah" not in by_name(ledger2)
    index = build_mbr_document_memory_index(art2.llm_cache_index_json, doc_id="doc")
    assert "Maysunah" in index.entity_to_aliases["Doctor Sinchipuma"]  # the bridge the probe measures
    assert "her family name is Sinchipuma" in index.entity_to_descriptions["Doctor Sinchipuma"]
    assert "Doctor Sinchipuma" in index.chunk_to_names["doc:0"]
