#!/usr/bin/env python3
"""Read-only check on run 1, before starting run 1b (CPU, under a minute; writes nothing).

How did run 1's wrong-direction joins happen? For the personas of cells 1-3, 6
and 8 it prints the model's decision on form 2 at the reveal and the outcome, and,
for the joins that went into a separate form-2 record, whether form 1 sits there as
an alias, a description or in the name. Change 1 as specified (redirect +
auto-join) acts only when form 2 was answered SAME AS, or when a NEW name equals a
shown one; the optional JSON-alias join (ledger.json_alias_join) acts when form 2
was answered NEW and its own record lists form 1 as an alias.

It also replays run 1's ledger up to each link step (the reveal; the gold chunk in
cell 4) and asks whether run 1b's recall (chunk tokens, cap 8, ties by the most
recent mention) would have shown the earlier record there, next to what run 1's
recall did. Run 1b's own ledger will differ, so this is an estimate.

    bash -c 'source experiments/method_update/p1b_ledger/env.sh && python experiments/method_update/p1b_ledger/scripts/check_run1_joins.py'
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_p1b as C  # noqa: E402
import stage03_memory_check as S3  # noqa: E402


def counterfactual_recall(rows: list[dict], cap: int = 8) -> tuple[Counter, list[dict]]:
    from saadi.ledger.ledger import DocumentLedger, LedgerStore, name_key, stripped_tokens
    from saadi.ledger.p1b import recall_by_chunk_tokens

    store = LedgerStore(C.RUN1_LEDGER_DIR)
    out: Counter = Counter()
    cut: list[dict] = []
    by_book: dict[str, list[dict]] = defaultdict(list)
    for p in rows:
        step = p.get("gold_chunk_index") if p["cell"] == 4 else p.get("reveal_chunk_index")
        if p["cell"] != 5 and step is not None:
            by_book[p["book_id"]].append({**p, "_step": int(step)})
    for book, group in sorted(by_book.items()):
        events = sorted(store.read_events(book), key=lambda e: int(e["step"]))
        at = {int(e["step"]): e for e in events}
        chunks = C.p1.book_chunks(book)
        ledger, i = DocumentLedger(book), 0
        for p in sorted(group, key=lambda x: x["_step"]):
            s = p["_step"]
            while i < len(events) and int(events[i]["step"]) < s:
                ledger.apply_event(events[i])
                i += 1
            want = set(stripped_tokens(p["form2"] if p["cell"] == 4 else p["form1"]))
            earlier = [e for e in ledger.entries.values() if want and any(want <= set(stripped_tokens(n)) for n in e.all_names())]
            if not earlier:
                out["no earlier entry in the ledger"] += 1
                continue
            overlap = {name_key(n) for n in (at.get(s) or {}).get("recalled_overlap_names", [])}
            if any(name_key(e.name) in overlap for e in earlier):
                out["in the overlap group"] += 1
                continue
            recalled, left_out, _hit, matches = recall_by_chunk_tokens(ledger, chunks[s], exclude_keys=overlap, cap=cap)
            shown = {e.id for e in recalled}
            order = [e.id for e in recalled + left_out]
            if any(e.id in shown for e in earlier):
                out[f"shown by chunk-token recall (cap {cap})"] += 1
            elif any(e.id in order for e in earlier):
                out[f"matched, cut by the cap of {cap}"] += 1
                e = next(e for e in earlier if e.id in order)
                cut.append({"persona": p["persona_id"], "cell": p["cell"], "rank": order.index(e.id) + 1, "class": matches[e.id]["class"], "full_name_matches": sum(1 for m in matches.values() if m["class"] == "full_name"), "matches": len(matches)})
            else:
                out["not matched by chunk tokens"] += 1
    return out, cut


def main() -> int:
    from saadi.ledger.ledger import LedgerStore

    path = C.run1_file("personas_memory.jsonl")
    if path is None or not C.RUN1_LEDGER_DIR.exists():
        print(f"Run 1's personas_memory.jsonl or ledger is missing under {C.RUN1_OUT_DIR}.")
        return 1
    rows = C.read_jsonl(path)
    store = LedgerStore(C.RUN1_LEDGER_DIR)
    step06 = C.load_probe_step("step06_memory_check.py")
    grp = [p for p in rows if p["cell"] in (1, 2, 3, 6, 8)]
    print(f"Run 1, cells 1-3, 6, 8 (n={len(grp)}): form 2 at the reveal")
    for k, v in Counter(S3.decision_on_form2(store, p) for p in grp).most_common():
        print(f"  {v:4d}  {k}")
    c123 = [p for p in rows if p["cell"] in (1, 2, 3)]
    print(f"\nRun 1, cells 1-3 (n={len(c123)}): join direction {dict(Counter(p.get('merge_direction') for p in c123))}")
    books = sorted({p["book_id"] for p in rows})
    idx = S3.indexes_for(C.run1_cache_index(), C.RUN1_LEDGER_DIR, books)
    loc = Counter(x for x in (S3.form1_location(idx[p["book_id"]], p, step06) for p in rows if p["cell"] != 5) if x)
    print(f"Run 1, all cells with a reveal: where form 1 sits in the separate form-2 record {dict(loc)}")
    wrong = [p for p in c123 if p.get("merge_direction") == "form1_into_form2"]
    print(f"\nRun 1, cells 1-3, wrong direction ({len(wrong)}): decision on form 2 at the reveal")
    for k, v in Counter(S3.decision_on_form2(store, p) for p in wrong).most_common():
        print(f"  {v:4d}  {k}")

    reveal = [p for p in rows if p["cell"] != 5]
    print(f"\nEarlier record at the link step, all cells with a reveal (n={len(reveal)})")
    print(f"  run 1 as built (candidate recall, cap 5): {dict(Counter(S3.why_not_in_prompt(p) for p in reveal))}")
    est, cut = counterfactual_recall(rows)
    print(f"  run 1b recall on run 1's ledger (estimate): {dict(est)}")
    if cut:
        print(f"  cut by the cap: the earlier record's rank among the matches, median {sorted(r['rank'] for r in cut)[len(cut) // 2]}; its match class {dict(Counter(r['class'] for r in cut))}")
        print(f"  full-name matches in those chunks, median {sorted(r['full_name_matches'] for r in cut)[len(cut) // 2]}; by cell {dict(Counter(r['cell'] for r in cut))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
