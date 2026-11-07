#!/usr/bin/env python3
"""Step 2: build templates.json and the 240 personas under the diversity rules.

Positions are drawn in step 3; chunk indices are filled in step 4. All random
draws use numpy.random.default_rng(42) and are recorded.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import banks  # noqa: E402
import common  # noqa: E402
from common import (  # noqa: E402
    BOOKS_JSON,
    PERSONAS_JSONL,
    SEED,
    TEMPLATES_JSON,
    StepReport,
    content_token_set,
    read_json,
    run_step,
    write_json,
    write_jsonl,
)


def _interleaved_given_names(rng) -> list[dict]:
    """Round-robin across (background, gender) pools, each pool shuffled."""
    pools: dict[tuple[str, str], list[dict]] = {}
    for item in banks.GIVEN_NAMES:
        pools.setdefault((item["language_background"], item["gender"]), []).append(dict(item))
    keys = sorted(pools)
    for key in keys:
        pool = pools[key]
        order = rng.permutation(len(pool))
        pools[key] = [pool[i] for i in order]
    out: list[dict] = []
    i = 0
    while any(pools[k] for k in keys):
        key = keys[i % len(keys)]
        if pools[key]:
            out.append(pools[key].pop())
        i += 1
    return out


def _shuffled(rng, items: list) -> list:
    order = rng.permutation(len(items))
    return [items[i] for i in order]


def build_personas() -> tuple[list[dict], dict]:
    import numpy as np

    rng = np.random.default_rng(SEED)
    books = [row["doc_id"] for row in read_json(BOOKS_JSON)["selected"]]
    if len(books) != common.N_BOOKS:
        raise RuntimeError(f"Expected {common.N_BOOKS} selected books, found {len(books)}")

    given = _interleaved_given_names(rng)
    surnames = _shuffled(rng, list(banks.SURNAMES))
    descriptive = _shuffled(rng, list(banks.DESCRIPTIVE_NAMES))
    occupations = _shuffled(rng, list(banks.OCCUPATIONS))
    objects = _shuffled(rng, list(banks.OBJECTS))
    objects_by_kind: dict[str, list[tuple[str, str]]] = {}
    for kind, name in objects:
        objects_by_kind.setdefault(kind, []).append((kind, name))
    object_kinds = sorted(objects_by_kind)
    answers = _shuffled(rng, list(banks.ANSWERS))
    titles = list(banks.TITLES)

    reserves = {
        "given": given,
        "surname": surnames,
        "descriptive": descriptive,
        "occupation": occupations,
        "object": objects,
        "answer": answers,
    }

    personas: list[dict] = []
    cell_counters = {cell["cell"]: 0 for cell in banks.CELLS}
    for book in books:
        for cell in banks.CELLS:
            family = cell["family"]
            for _ in range(common.PERSONAS_PER_CELL_PER_BOOK):
                k = cell_counters[cell["cell"]]
                cell_counters[cell["cell"]] += 1
                person = reserves["given"].pop(0)
                answer = reserves["answer"].pop(0)
                phrasing_id = k % 4
                persona: dict = {
                    "persona_id": f"c{cell['cell']}_{book}_{k:03d}",
                    "book_id": book,
                    "cell": cell["cell"],
                    "family": family,
                    "placement": cell["placement"],
                    "phrasing_id": phrasing_id,
                    "gender": person["gender"],
                    "language_background": person["language_background"],
                    "given_name": person["name"],
                    "answer": answer,
                    "title": None,
                    "fact_type": None,
                    "form1": None,
                    "form2": None,
                    "object_kind": None,
                    "gold_sentence": None,
                    "reveal_sentence": None,
                    "query": None,
                    "target_gold_pos": None,
                    "target_reveal_pos": None,
                    "rng_draws": [],
                    "regenerations": [],
                    "gold_chunk_index": None,
                    "reveal_chunk_index": None,
                    "gold_chunk_id": None,
                    "reveal_chunk_id": None,
                }
                if family == "A":
                    persona["fact_type"] = banks.FACT_ROTATION[k % 5]
                    persona["title"] = titles[k % len(titles)]
                    persona["surname"] = reserves["surname"].pop(0)
                elif family == "B":
                    persona["fact_type"] = banks.FACT_ROTATION[k % 5]
                    persona["descriptive_name"] = reserves["descriptive"].pop(0)
                elif family == "C":
                    persona["fact_type"] = banks.FACT_ROTATION[k % 5]
                    persona["occupation"] = reserves["occupation"].pop(0)
                else:  # D: rotate the six object kinds instead of fact types
                    kind = object_kinds[k % len(object_kinds)]
                    pool = objects_by_kind[kind] or reserves["object"]
                    kind, obj_name = pool.pop(0)
                    persona["fact_type"] = f"home_object:{kind}"
                    persona["object_kind"] = kind
                    persona["object_name"] = obj_name
                materialize_sentences(persona)
                personas.append(persona)
    templates = {
        "fact_types": banks.FACT_TYPES,
        "object_facts": banks.OBJECT_FACTS,
        "reveal_phrasings": banks.REVEAL_PHRASINGS,
        "cells": banks.CELLS,
        "titles": banks.TITLES,
        "seed": SEED,
        "notes": banks.__doc__,
    }
    return personas, templates


def materialize_sentences(p: dict) -> None:
    """Fill form1/form2, the gold/reveal sentences and the query from templates."""
    family = p["family"]
    given = p["given_name"]
    if family == "A":
        p["form1"] = p["surname"]
        p["form2"] = given
        x_gold = f"{p['title']} {p['surname']}"
        y_query = given
    elif family == "B":
        p["form1"] = p["descriptive_name"]
        p["form2"] = given
        x_gold = p["descriptive_name"][0].upper() + p["descriptive_name"][1:]
        y_query = given
    elif family == "C":
        p["form1"] = given
        p["form2"] = p["occupation"]
        x_gold = given
        y_query = f"the {p['occupation']}"
    else:
        p["form1"] = p["object_name"]
        p["form2"] = given
        x_gold = p["object_name"][0].upper() + p["object_name"][1:]
        y_query = given

    if family == "D":
        fact = banks.OBJECT_FACTS[p["object_kind"]]
    else:
        fact = banks.FACT_TYPES[p["fact_type"]]
    p["gold_sentence"] = fact["gold"].format(X=x_gold, P=p["answer"])
    p["query"] = fact["query"].format(Y=y_query)

    if p["placement"] == "none":
        p["reveal_sentence"] = None
        return
    phrasing = banks.REVEAL_PHRASINGS[family][p["phrasing_id"]]
    if family == "C":
        p["reveal_sentence"] = phrasing.format(X=given, OCC=p["occupation"])
    elif family == "D":
        p["reveal_sentence"] = phrasing.format(X=p["object_name"], Y=given)
    else:  # A, B: X = bare form 1 (no title), Y = given name
        x_reveal = p["form1"] if family == "A" else p["descriptive_name"]
        p["reveal_sentence"] = phrasing.format(X=x_reveal, Y=given)
    # Capitalize a sentence that starts with a lowercase "the ..." phrase.
    for key in ("reveal_sentence",):
        s = p[key]
        if s and s[0].islower():
            p[key] = s[0].upper() + s[1:]


def main(report: StepReport, argv: list[str]) -> None:
    personas, templates = build_personas()
    write_json(TEMPLATES_JSON, templates)
    write_jsonl(PERSONAS_JSONL, personas)
    report.files_written += [str(TEMPLATES_JSON), str(PERSONAS_JSONL)]
    report.commands.append(f"numpy.random.default_rng({SEED}); 240 personas, banks from experiments/probe_bridge/banks.py")

    def count_by(key) -> dict:
        out: dict = {}
        for p in personas:
            out[p.get(key)] = out.get(p.get(key), 0) + 1
        return dict(sorted(out.items(), key=lambda kv: str(kv[0])))

    for key, title in (
        ("cell", "Personas per cell"),
        ("family", "Per family"),
        ("fact_type", "Per fact type"),
        ("gender", "Per gender"),
        ("language_background", "Per language background"),
    ):
        counts = count_by(key)
        report.table(title, [key, "count"], [[k, v] for k, v in counts.items()])
        report.numbers[f"count_by_{key}"] = counts

    # Five full example personas drawn with the seed, not chosen.
    import numpy as np

    rng = np.random.default_rng(SEED)
    picks = rng.choice(len(personas), size=5, replace=False)
    examples = []
    for i in sorted(int(x) for x in picks):
        p = personas[i]
        examples.append(
            f"- **{p['persona_id']}** (cell {p['cell']}, {p['family']}, {p['fact_type']})\n"
            f"  - gold: {p['gold_sentence']}\n"
            f"  - reveal: {p['reveal_sentence']}\n"
            f"  - query: {p['query']}  -> answer: {p['answer']}"
        )
    report.add("Five example personas (seeded draw)", "\n".join(examples))
    report.numbers["n_personas"] = len(personas)

    # Sentence-length rule: 8-18 words for every inserted sentence.
    bad = [p["persona_id"] for p in personas for s in (p["gold_sentence"], p["reveal_sentence"]) if s and not (8 <= len(s.split()) <= 18)]
    if bad:
        report.unexpected.append(f"Sentences outside 8-18 words for: {bad[:10]} ... fix the templates before step 3.")
    # Self-check of the fixed templates: query and gold share no content token.
    clashes = []
    for p in personas:
        if content_token_set(p["query"]) & content_token_set(p["gold_sentence"]):
            clashes.append(p["persona_id"])
    if clashes:
        report.unexpected.append(f"Template-level query/gold token overlap for: {clashes[:10]}. Step 3 will refuse these.")


if __name__ == "__main__":
    raise SystemExit(run_step(2, "Generate personas", main, requires=[1]))
