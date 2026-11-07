#!/usr/bin/env python3
"""Step 3: run checks 1-6, regenerate failures from the reserves, draw positions.

A persona that fails a check is regenerated (a different phrasing first, then a
different name from the unused reserve), never patched by hand. Every
regeneration is recorded on the persona. Positions are drawn in units of the
original chunk sequence with numpy.random.default_rng(42) and recorded.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import banks  # noqa: E402
import common  # noqa: E402
from common import (  # noqa: E402
    BOOKS_JSON,
    PERSONAS_JSONL,
    SEED,
    StepReport,
    chunk_book,
    content_token_set,
    read_json,
    read_jsonl,
    run_step,
    write_jsonl,
)
from step02_generate_personas import materialize_sentences  # noqa: E402

QUESTION_SET_TXT = common.SPEC_DIR / "question_set.txt"
MAX_ATTEMPTS = 60
MIN_SEPARATION = 3  # the note's rule; checked again after re-chunking (step 4).
# The crowded region [0.10N, 0.35N] must hold 21 golds, 3 near-reveals and
# 3 cell-4 reveals = 27 sentences 3 apart, so books need roughly N >= 340
# (312 theoretical plus headroom). Step 1 reports this per book.


# ---------------------------------------------------------------------------
# Invented items of a persona (the things checks 1, 2 and 6 look at)
# ---------------------------------------------------------------------------


def invented_items(p: dict) -> dict[str, str]:
    items = {"form2": p["form2"], "answer": p["answer"]}
    if p["family"] == "A":
        items["form1_surname"] = p["surname"]
        items["given"] = p["given_name"]
    elif p["family"] == "B":
        items["form1_descriptive"] = p["descriptive_name"]
        items["given"] = p["given_name"]
    elif p["family"] == "C":
        items["form1_given"] = p["given_name"]
        items["occupation"] = p["occupation"]
    else:
        items["form1_object"] = p["object_name"]
        items["given"] = p["given_name"]
    return items


def rare_components(p: dict) -> list[str]:
    """Common nouns used inside names/descriptions that must be rare (check 6)."""
    words: list[str] = []
    if p["family"] == "B":
        words += sorted(content_token_set(p["descriptive_name"]))
    elif p["family"] == "C":
        words += sorted(content_token_set(p["occupation"]))
    elif p["family"] == "D":
        words += sorted(content_token_set(p["object_name"]))
    return words


# ---------------------------------------------------------------------------
# Checks 1-6
# ---------------------------------------------------------------------------


class Checker:
    def __init__(self, book_texts: dict[str, str], question_set: str, chunk_token_sets: dict[str, list[set[str]]]):
        self.absence_corpus = [t.lower() for t in book_texts.values()] + [question_set.lower()]
        self.chunk_token_sets = chunk_token_sets
        self.used_words: dict[str, str] = {}  # invented word -> persona_id

    def failures(self, p: dict) -> list[str]:
        out: list[str] = []
        # 1. Invented items absent from every original book and the question set.
        # The whole item is matched as a case-insensitive substring ("Mitra"
        # also rejects "Mitral"); the rarity of single components inside a
        # multi-word item is check 6's job.
        for label, item in invented_items(p).items():
            phrase = str(item).lower().strip()
            if phrase.startswith("the "):
                phrase = phrase[4:]
            if phrase and any(phrase in corpus for corpus in self.absence_corpus):
                out.append(f"check1:{label}:{phrase}")
        # 2. No reuse across personas
        for label, item in invented_items(p).items():
            for word in str(item).split():
                w = word.strip(".,'").lower()
                if not w or w in common.STOPWORDS or w == "the":
                    continue
                owner = self.used_words.get(w)
                if owner is not None and owner != p["persona_id"]:
                    out.append(f"check2:{label}:{w}:{owner}")
        # 3. Query and gold share no content token (stems included)
        if content_token_set(p["query"]) & content_token_set(p["gold_sentence"]):
            shared = content_token_set(p["query"]) & content_token_set(p["gold_sentence"])
            out.append(f"check3:{sorted(shared)}")
        if p["reveal_sentence"]:
            reveal_t = content_token_set(p["reveal_sentence"])
            gold_t = content_token_set(p["gold_sentence"])
            form1_t = content_token_set(p["form1"])
            form2_t = content_token_set(p["form2"])
            # 4. Reveal and gold share only the link (form 1)
            if (reveal_t & gold_t) != (form1_t & gold_t) or not form1_t <= reveal_t:
                out.append(f"check4:{sorted((reveal_t & gold_t) ^ (form1_t & gold_t))}")
            # 5. Query and reveal share only form 2
            query_t = content_token_set(p["query"])
            if (query_t & reveal_t) != form2_t:
                out.append(f"check5:{sorted((query_t & reveal_t) ^ form2_t)}")
        # 6. Rare common nouns: each component in at most 2 chunks of the book
        token_sets = self.chunk_token_sets[p["book_id"]]
        for word in rare_components(p):
            n = sum(1 for toks in token_sets if word in toks)
            if n > 2:
                out.append(f"check6:{word}:{n}chunks")
        return out

    def claim_words(self, p: dict) -> None:
        for item in invented_items(p).values():
            for word in str(item).split():
                w = word.strip(".,'").lower()
                if w and w not in common.STOPWORDS and w != "the":
                    self.used_words.setdefault(w, p["persona_id"])


# ---------------------------------------------------------------------------
# Regeneration from reserves
# ---------------------------------------------------------------------------


class Reserves:
    def __init__(self, personas: list[dict], rng) -> None:
        def unused(bank, used: set[str]):
            pool = [x for x in bank if (x["name"] if isinstance(x, dict) else (x[1] if isinstance(x, tuple) else x)) not in used]
            order = rng.permutation(len(pool))
            return [pool[i] for i in order]

        used_given = {p["given_name"] for p in personas}
        used_sur = {p.get("surname") for p in personas}
        used_desc = {p.get("descriptive_name") for p in personas}
        used_occ = {p.get("occupation") for p in personas}
        used_obj = {p.get("object_name") for p in personas}
        used_ans = {p["answer"] for p in personas}
        self.given = unused(banks.GIVEN_NAMES, used_given)
        self.surname = unused(banks.SURNAMES, used_sur)
        self.descriptive = unused(banks.DESCRIPTIVE_NAMES, used_desc)
        self.occupation = unused(banks.OCCUPATIONS, used_occ)
        self.object = unused(banks.OBJECTS, used_obj)
        self.answer = unused(banks.ANSWERS, used_ans)
        self.exhausted: list[str] = []

    def pop(self, kind: str):
        pool = getattr(self, kind)
        if not pool:
            self.exhausted.append(kind)
            raise RuntimeError(
                f"The reserve list for {kind!r} is exhausted. Extend the bank in "
                f"experiments/probe_bridge/banks.py and rerun this step with --force."
            )
        return pool.pop(0)


def regenerate(p: dict, failures: list[str], reserves: Reserves) -> None:
    reasons = {f.split(":", 1)[0] for f in failures}
    p["regenerations"].append({"failures": failures})
    if "check3" in reasons:
        raise RuntimeError(
            f"{p['persona_id']}: the fixed templates leak a query token into the gold sentence "
            f"({failures}). This is a template bug; stop and fix banks.py."
        )
    if ("check4" in reasons or "check5" in reasons) and len(p["regenerations"]) <= 4:
        p["phrasing_id"] = (p["phrasing_id"] + 1) % 4
    else:
        # Replace the offending material.
        bad = " ".join(failures)
        if p["family"] == "A" and ("surname" in bad or "check4" in reasons or "check5" in reasons):
            p["surname"] = reserves.pop("surname")
        if p["family"] == "B" and ("descriptive" in bad or "check6" in reasons):
            p["descriptive_name"] = reserves.pop("descriptive")
        if p["family"] == "C" and ("occupation" in bad or "form2" in bad or "check6" in reasons):
            p["occupation"] = reserves.pop("occupation")
        if p["family"] == "D" and ("object" in bad or "check6" in reasons):
            kind = p["object_kind"]
            pool = [o for o in reserves.object if o[0] == kind] or reserves.object
            item = pool[0]
            reserves.object.remove(item)
            p["object_name"] = item[1]
            p["object_kind"] = item[0]
            p["fact_type"] = f"home_object:{item[0]}"
        if "given" in bad or "form2" in bad and p["family"] != "C":
            person = reserves.pop("given")
            p["given_name"], p["gender"], p["language_background"] = person["name"], person["gender"], person["language_background"]
        if "answer" in bad:
            p["answer"] = reserves.pop("answer")
    materialize_sentences(p)


# ---------------------------------------------------------------------------
# Position draws
# ---------------------------------------------------------------------------


def _min_gap_layout(rng, start: int, end: int, widths: list[int], gap: int) -> list[int]:
    """Left-to-right positions with a minimum gap; uniform given the constraint.

    Item i occupies [p_i, p_i + widths[i]]; successive items keep
    p_{i+1} >= p_i + widths[i] + gap, all inside [start, end].
    """
    n = len(widths)
    total = sum(widths) + gap * (n - 1)
    slack = (end - start) - total
    if slack < 0:
        raise RuntimeError(f"window [{start}, {end}] cannot hold {n} items (needs {total}, has {end - start})")
    cuts = sorted(float(rng.uniform(0, slack)) for _ in range(n))
    positions, acc = [], 0
    for i, width in enumerate(widths):
        positions.append(start + int(cuts[i]) + acc)
        acc += width + gap
    return positions


def _assign(p: dict, g: int, r, n: int, note=None) -> list[int]:
    for pos in [g] + ([r] if r is not None else []):
        if pos < int(0.02 * n) or pos > int(0.95 * n):
            raise RuntimeError(f"{p['persona_id']}: position {pos} outside [0.02N, 0.95N] (N={n})")
    p["target_gold_pos"], p["target_reveal_pos"] = g, r
    p["placement_note"] = note
    return [g] + ([r] if r is not None else [])


def _reveal_for(p: dict, g: int, n: int):
    placement = p["placement"]
    if placement == "near":
        return g + MIN_SEPARATION
    if placement == "middle":
        return g + int(0.25 * n)
    if placement == "far":
        return g + int(0.55 * n)
    if placement == "before":
        return g - int(0.20 * n)
    return None


def _place_book_plain(main_group, cell4, n, rng) -> list[int]:
    """The note's draws: golds laid in [0.10N, 0.30N], cell-4 golds uniform in
    [0.35N, 0.55N] by rejection. Works when the book is long enough."""
    order = [main_group[i] for i in rng.permutation(len(main_group))]
    widths = [MIN_SEPARATION if p["placement"] == "near" else 0 for p in order]
    golds = _min_gap_layout(rng, int(0.10 * n) + 1, int(0.30 * n), widths, MIN_SEPARATION)
    taken: list[int] = []
    for p, g in zip(order, golds, strict=True):
        r = _reveal_for(p, g, n)
        p["rng_draws"] = [float(g)]
        taken += _assign(p, g, r, n)
    for p in cell4:
        draws: list[float] = []
        for _ in range(5000):
            u = float(rng.uniform(0.35 * n, 0.55 * n))
            draws.append(u)
            g = int(u)
            r = g - int(0.20 * n)
            if r < int(0.02 * n) or g > int(0.95 * n):
                continue
            if any(abs(g - t) < MIN_SEPARATION for t in taken) or any(abs(r - t) < MIN_SEPARATION for t in taken):
                continue
            if abs(g - r) < MIN_SEPARATION:
                continue
            p["rng_draws"] = draws
            taken += _assign(p, g, r, n)
            break
        else:
            raise RuntimeError(f"cell-4 rejection failed for {p['persona_id']} (N={n})")
    return taken


def _place_book_pinned(main_group, cell4, n, rng) -> list[int]:
    """Fallback for tight books: one layout of all 24 items in the gold window,
    with the cell-4 items pinned to its top quarter [0.25N, 0.30N]. Their gold
    then lands at q + 0.25N (inside the note's [0.35N, 0.55N]) and their reveal
    at q + 0.25N - 0.20N, in the empty band just above the gold window; every
    pairwise separation is inherited from the layout."""
    # The pinned band: low enough that the derived reveal stays MIN_SEPARATION
    # below the lowest possible middle reveal, high enough that it stays above
    # the gold window (the layout's own gaps cover the rest).
    q_lo = int(0.25 * n) + 1
    q_hi = int(0.10 * n) + int(0.20 * n) - MIN_SEPARATION + 1
    if q_hi < q_lo + MIN_SEPARATION * (len(cell4) - 1):
        raise RuntimeError(f"pinned band [{q_lo}, {q_hi}] too small for {len(cell4)} cell-4 items (N={n})")
    for _ in range(400):
        order = [main_group[i] for i in rng.permutation(len(main_group))]
        items = order + [cell4[i] for i in rng.permutation(len(cell4))]
        widths = [MIN_SEPARATION if p["placement"] == "near" else 0 for p in items]
        laid = _min_gap_layout(rng, int(0.10 * n) + 1, int(0.30 * n), widths, MIN_SEPARATION)
        qs = laid[len(order) :]
        if any(q < q_lo or q > q_hi for q in qs):
            continue
        taken: list[int] = []
        for p, pos in zip(items, laid, strict=True):
            if p["cell"] == 4:
                g = pos + int(0.25 * n)
                r = g - int(0.20 * n)
                p["rng_draws"] = [float(pos)]
                taken += _assign(p, g, r, n, note="banded_cell4")
            else:
                r = _reveal_for(p, pos, n)
                p["rng_draws"] = [float(pos)]
                taken += _assign(p, pos, r, n)
        return taken
    raise RuntimeError(f"pinned layout failed after 400 rounds (N={n})")


def draw_positions(personas: list[dict], n_chunks: dict[str, int], rng) -> None:
    """Draw every target position, honouring the note's windows and check 7.

    The gold window is a packing problem (21 golds plus the cell-1 reveals at
    g+3, all >= 3 apart), so golds are laid out with a sorted-uniform
    minimum-gap draw. Cell 4 uses the note's uniform draw when the book has
    room, and the pinned fallback otherwise (recorded and reported)."""
    for book, n in sorted(n_chunks.items()):
        group = [q for q in personas if q["book_id"] == book]
        main_group = [p for p in group if p["cell"] != 4]
        cell4 = [p for p in group if p["cell"] == 4]
        taken = None
        errors = []
        for strategy in (_place_book_plain, _place_book_pinned):
            try:
                taken = strategy(main_group, cell4, n, rng)
                break
            except RuntimeError as exc:
                errors.append(str(exc))
        if taken is None:
            raise RuntimeError(
                f"Could not place the {len(group)} personas of book {book} (N={n}): "
                + " | ".join(errors)
                + " - stop and decide at the hand-off (see step 1's feasibility column)."
            )
        ordered = sorted(taken)
        bad = [(a, b) for a, b in zip(ordered, ordered[1:], strict=False) if b - a < MIN_SEPARATION]
        if bad:
            raise RuntimeError(f"Internal error: separation violated in {book}: {bad[:5]}")


# ---------------------------------------------------------------------------


def main(report: StepReport, argv: list[str]) -> None:
    import numpy as np

    personas = read_jsonl(PERSONAS_JSONL)
    books = read_json(BOOKS_JSON)["selected"]
    book_ids = [b["doc_id"] for b in books]
    from saadi.indexing.dataset_loaders.novelhopqa import load_novelhopqa_documents

    books_root = os.environ.get("NOVELHOPQA_BOOKS_ROOT", "") or None
    docs = load_novelhopqa_documents(config_name="all", books_root=books_root)
    book_texts = {bid: docs[bid] for bid in book_ids}
    question_set = QUESTION_SET_TXT.read_text(encoding="utf-8")

    chunk_token_sets: dict[str, list[set[str]]] = {}
    n_chunks: dict[str, int] = {}
    for bid, text in book_texts.items():
        chunks = chunk_book(text)
        n_chunks[bid] = len(chunks)
        chunk_token_sets[bid] = [set(common.tokens(c)) for c, _, _ in chunks]

    checker = Checker(book_texts, question_set, chunk_token_sets)
    rng = np.random.default_rng(SEED + 3)
    reserves = Reserves(personas, rng)

    regen_count = 0
    reason_hist: dict[str, int] = {}
    pass_counts = {f"check{i}": 0 for i in range(1, 7)}
    for p in personas:
        attempts = 0
        failures = checker.failures(p)
        while failures and attempts < MAX_ATTEMPTS:
            attempts += 1
            regen_count += 1
            for f in failures:
                key = f.split(":", 1)[0]
                reason_hist[key] = reason_hist.get(key, 0) + 1
            regenerate(p, failures, reserves)
            failures = checker.failures(p)
        if failures:
            raise RuntimeError(f"{p['persona_id']} still failing after {MAX_ATTEMPTS} regenerations: {failures}")
        checker.claim_words(p)
        for i in range(1, 7):
            pass_counts[f"check{i}"] += 1

    rng_pos = np.random.default_rng(SEED)
    draw_positions(personas, n_chunks, rng_pos)
    banded = [p["persona_id"] for p in personas if p.get("placement_note") == "banded_cell4"]
    if banded:
        report.unexpected.append(
            f"{len(banded)} cell-4 personas used the banded fallback (gold drawn in [0.50N, 0.55N] "
            f"inside the note's window, so the reveal clears the packed gold region): {banded}"
        )
    write_jsonl(PERSONAS_JSONL, personas)
    report.files_written.append(str(PERSONAS_JSONL))
    report.commands.append("checks 1-6 over the 10 originals + question set; positions with default_rng(42)")

    report.table("Checks", ["check", "personas passing"], [[k, v] for k, v in pass_counts.items()])
    report.add(
        "Regenerations",
        f"Total regeneration rounds: **{regen_count}**. Failure reasons: {reason_hist}. "
        f"Exhausted reserves: {reserves.exhausted or 'none'}.",
    )
    pos_rows = []
    for cell in sorted({p["cell"] for p in personas}):
        sub = [p for p in personas if p["cell"] == cell]
        g = [p["target_gold_pos"] for p in sub]
        r = [p["target_reveal_pos"] for p in sub if p["target_reveal_pos"] is not None]
        pos_rows.append([cell, len(sub), f"{min(g)}-{max(g)}", (f"{min(r)}-{max(r)}" if r else "-")])
    report.table("Target positions (original chunk units)", ["cell", "personas", "gold range", "reveal range"], pos_rows)

    picks = sorted(int(x) for x in np.random.default_rng(SEED).choice(len(personas), size=10, replace=False))
    lines = []
    for i in picks:
        p = personas[i]
        lines.append(
            f"- **{p['persona_id']}** (cell {p['cell']}, {p['fact_type']}, g={p['target_gold_pos']}, r={p['target_reveal_pos']})\n"
            f"  - gold: {p['gold_sentence']}\n  - reveal: {p['reveal_sentence']}\n"
            f"  - query: {p['query']}  -> {p['answer']}"
        )
    report.add("Ten complete personas (seeded draw)", "\n".join(lines))
    report.numbers.update({"regenerations": regen_count, "reasons": reason_hist, "n_chunks": n_chunks, "pass_counts": pass_counts})


if __name__ == "__main__":
    raise SystemExit(run_step(3, "Check and place", main, requires=[1, 2]))
