"""Forced-accounting parser: lenient about form, strict about meaning."""

from __future__ import annotations

from saadi.ledger.accounting import NEW, NOT_AN_ENTITY, SAME_AS, parse_decisions, split_reply

CANDS = ["Maysunah", "Sinchipuma", "Orleans", "Miss Watson"]

REPLY = """[DECISIONS]
1. Maysunah -> SAME AS E0012 (the chunk says Maysunah's family name was Sinchipuma, the stub's name)
2. Sinchipuma -> SAME AS E0012 (same family name as the recalled stub Doctor Sinchipuma)
3. Orleans -> NEW
4. Miss Watson -> NOT AN ENTITY
[END_DECISIONS]
{"active_entities": [{"name": "Doctor Sinchipuma", "description": ["a doctor"], "aliases": ["Maysunah"]}], "insertions": [{"name": "Doctor Sinchipuma", "anchor_text": "Sinchipuma", "left_context": "name was ", "right_context": "."}]}
[END_JSON]"""


def test_clean_block_parses_every_candidate() -> None:
    block = parse_decisions(REPLY, CANDS)
    assert block.status == "ok"
    assert block.omitted == []
    kinds = [d.kind for d in block.decisions]
    assert kinds == [SAME_AS, SAME_AS, NEW, NOT_AN_ENTITY]
    assert block.decisions[0].target == "E0012"
    assert "family name" in block.decisions[0].reason
    assert all(d.matched_by == "number" for d in block.decisions)
    decisions, json_text = split_reply(REPLY)
    assert json_text.startswith('{"active_entities"')
    assert "[END_DECISIONS]" not in decisions


def test_variants_of_form_are_accepted() -> None:
    reply = """1) Maysunah → same as E0012 — her family name was Sinchipuma
2. Sinchipuma => SAME AS "Doctor Sinchipuma" (same surname and the chunk names her)
3: Orleans - NEW
4. Miss Watson -> Not an entity
{"active_entities": [], "insertions": []}
[END_JSON]"""
    block = parse_decisions(reply, CANDS)
    assert block.status == "ok", block
    d = block.by_index()
    assert d[0].kind == SAME_AS and d[0].target == "E0012" and "family name" in d[0].reason
    assert d[1].kind == SAME_AS and d[1].target == "Doctor Sinchipuma" and "same surname" in d[1].reason
    assert d[2].kind == NEW
    assert d[3].kind == NOT_AN_ENTITY


def test_missing_numbers_match_by_text_and_omissions_are_counted() -> None:
    reply = """Maysunah -> NEW
Orleans -> NOT AN ENTITY
{"active_entities": [], "insertions": []}"""
    block = parse_decisions(reply, CANDS)
    assert block.status == "partial"
    assert block.omitted == [1, 3]
    assert {d.candidate_index: d.matched_by for d in block.decisions} == {0: "text", 2: "text"}


def test_unparseable_block_is_a_failure_and_json_only_reply_too() -> None:
    assert parse_decisions('no decisions at all\n{"active_entities": [], "insertions": []}', CANDS).status == "failed"
    assert parse_decisions('{"active_entities": [], "insertions": []}\n[END_JSON]', CANDS).status == "failed"
    assert parse_decisions("", []).status == "empty"


def test_duplicate_and_out_of_range_lines_are_reported_not_applied() -> None:
    reply = """1. Maysunah -> NEW
1. Maysunah -> NOT AN ENTITY
9. Nobody -> NEW
2. Sinchipuma -> NEW
3. Orleans -> NEW
4. Miss Watson -> NEW
{"active_entities": [], "insertions": []}"""
    block = parse_decisions(reply, CANDS)
    assert block.status == "ok"
    assert block.by_index()[0].kind == NEW
    assert len(block.unmatched_lines) == 2
