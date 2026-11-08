#!/usr/bin/env python3
"""Step 9: where are the names lost? Four read-only questions on the keep build.

Spec: "Probe Diagnosis: Four Questions" (claude.ai artifact S5hKQedMJHPhzcFnwWqNLJ).
Reads the construction records of step 5, the memory check of step 6 and the
rankings of step 7; writes reports/step_09.{md,json} and per-character detail to
outputs/probe_bridge/diagnose/. No model calls, nothing under outputs/ or spec/
is changed.

Two inputs the spec names are not stored as such and are rebuilt exactly:
- the memory after every step: the cache loader replays the accepted records of
  each step with merge_active_entities (the construction's own merge); the
  replay of the last step equals the book's final memory;
- the records recalled into the prompt: the recall rule
  (select_active_entities, word overlap, budget 5/2/3) is re-run on the memory
  before the step and the chunk's raw text, with the run's own settings.

Name matching: a form is present in a field when all its content tokens (the
repository's BM25 tokens, stopwords removed) occur in that field, the rule of
step 6. Titles are stripped before two record names are compared.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402
from common import (  # noqa: E402
    OUT_DIR,
    PERSONAS_JSONL,
    PUBLISHED_SETTINGS,
    ROOT,
    SEED,
    StepReport,
    content_token_set,
    memory_channel_ranks,
    read_json,
    read_jsonl,
    run_step,
    tokens,
    write_json,
    write_jsonl,
)

DIAG_DIR = OUT_DIR / "diagnose"
MEMORY_JSONL = OUT_DIR / "personas_memory.jsonl"
TITLES = ["Dr.", "Doctor", "Captain", "Mrs.", "Mr.", "Professor", "Father", "Judge", "Madame", "Sergeant", "Colonel"]
TITLE_TOKENS = {t for title in TITLES for t in tokens(title)}
REVEAL_CELLS = (1, 2, 3, 4, 6, 7, 8)
LAMBDA_GRID = [round(0.05 * i, 2) for i in range(0, 401)]  # 0 to 20


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _name(e) -> str:
    return str((e.get("name") if isinstance(e, dict) else getattr(e, "name", "")) or "").strip()


def _aliases(e) -> list[str]:
    raw = e.get("aliases", []) if isinstance(e, dict) else getattr(e, "aliases", [])
    return [str(a) for a in (raw or []) if str(a).strip()]


def _descs(e) -> list[str]:
    if isinstance(e, dict):
        raw = e.get("description", e.get("descriptions", []))
    else:
        raw = getattr(e, "description", None) or getattr(e, "descriptions", None)
    if isinstance(raw, str):
        raw = [raw]
    return [str(d) for d in (raw or []) if str(d).strip()]


def _key(name: str) -> str:
    return str(name or "").strip().lower()


def _in(text: str, form_t: set[str]) -> bool:
    return bool(form_t) and form_t <= set(tokens(text))


def in_name_alias(e, form_t: set[str]) -> bool:
    return _in(_name(e), form_t) or any(_in(a, form_t) for a in _aliases(e))


def in_any_field(e, form_t: set[str]) -> bool:
    return in_name_alias(e, form_t) or any(_in(d, form_t) for d in _descs(e))


def stripped(name: str) -> tuple[str, ...]:
    return tuple(t for t in tokens(name) if t not in TITLE_TOKENS and t not in common.STOPWORDS)


def word_in_text(text: str, form: str) -> bool:
    """The loose test of tools/diag_extraction.py (whole word, case-insensitive)."""
    return bool(form) and re.search(r"\b" + re.escape(form.lower()) + r"\b", str(text or "").lower()) is not None


def proposed_records(raw_model_text: str) -> tuple[list[dict], str]:
    from saadi.core.parsers import safe_json_from_text

    parsed, status = safe_json_from_text(raw_model_text or "")
    parsed = parsed or {}
    raw = parsed.get("active_entities")
    if raw is None:
        raw = parsed.get("active_objects", parsed.get("objects_active", []))
    return [x for x in (raw or []) if isinstance(x, dict)], str(status)


def reply_key_holding(raw_model_text: str, form: str) -> str:
    """Which JSON key of the reply carries the word, for names outside any record."""
    from saadi.core.parsers import safe_json_from_text

    parsed, _ = safe_json_from_text(raw_model_text or "")
    if not isinstance(parsed, dict):
        return "unparsed reply"
    hits = [k for k, v in parsed.items() if word_in_text(json.dumps(v, ensure_ascii=False), form)]
    if "insertions" in hits:
        for ins in parsed.get("insertions") or []:
            if isinstance(ins, dict):
                for field in ("anchor_text", "left_context", "right_context", "name"):
                    if word_in_text(str(ins.get(field, "")), form):
                        return f"insertions.{field}"
    return ", ".join(hits) or "outside the JSON"


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100.0 * n / d:.0f}%)" if d else "-"


# ---------------------------------------------------------------------------
# Book state
# ---------------------------------------------------------------------------


class Book:
    def __init__(self, index_path: Path, payload: dict, book: str, recall_cfg: dict):
        from saadi.indexing.retrieval_methods.mbr import (
            _find_doc_entry,
            _load_doc_records,
            _resolve_cache_file_path,
            build_mbr_document_memory_index,
        )

        doc_entry = _find_doc_entry(payload, doc_id=book)
        if doc_entry is None:
            raise RuntimeError(f"Book {book} is not in {index_path}")
        records = _load_doc_records(index_path=index_path, index_payload=payload, doc_entry=doc_entry, doc_id=book)
        self.by_step = {int(r.chunk_index): r for r in records}
        self.steps = sorted(self.by_step)
        self.index = build_mbr_document_memory_index(index_path, doc_id=book)
        mem_path = Path(_resolve_cache_file_path((doc_entry.get("files") or {}).get("memory", ""), base_dir=index_path.parent))
        self.memory_text = mem_path.read_text(encoding="utf-8") if mem_path.exists() else ""
        self.recall_cfg = recall_cfg
        self._proposals: dict[int, list[dict]] = {}

    def proposals(self, step: int) -> list[dict]:
        if step not in self._proposals:
            self._proposals[step] = proposed_records(self.by_step[step].raw_model_text)[0]
        return self._proposals[step]

    def record(self, step):
        return self.by_step.get(int(step)) if step is not None else None

    def memory_after(self, step: int) -> list[dict]:
        rec = self.record(step)
        return list((rec.state_snapshot or {}).get("entity_memory", []) or []) if rec else []

    def memory_before(self, step: int) -> list[dict]:
        earlier = [s for s in self.steps if s < int(step)]
        return self.memory_after(earlier[-1]) if earlier else []

    def accepted(self, step: int) -> list[dict]:
        rec = self.record(step)
        return list((rec.state_snapshot or {}).get("active_entities", []) or []) if rec else []

    def recall(self, step: int) -> tuple[list[str], dict[str, float]]:
        """Names recalled into the prompt at `step`, and every record's recall score."""
        from saadi.entities.active_selection import select_active_entities_with_scores, string_token_overlap_similarity

        rec = self.record(step)
        before = self.memory_before(step)
        cfg = self.recall_cfg
        chosen = select_active_entities_with_scores(
            before,
            rec.raw_chunk_text,
            max_entities=cfg["max_entities"],
            max_aliases=cfg["max_aliases"],
            max_desc=cfg["max_desc"],
            selector=cfg["selector"],
            min_relevance_score=cfg["min_relevance"],
        )
        scores = {}
        if cfg["selector"] == "word":
            for e in before:
                scores[_key(_name(e))] = float(string_token_overlap_similarity(e, rec.raw_chunk_text))
        else:
            for item in select_active_entities_with_scores(before, rec.raw_chunk_text, max_entities=10**6, selector=cfg["selector"]):
                scores[_key(item["entry"]["name"])] = float(item["score"])
        return [_key(item["entry"]["name"]) for item in chosen], scores

    def linked(self, chunk_id: str | None) -> set[str]:
        return {_key(n) for n in self.index.chunk_to_names.get(chunk_id or "", [])}


def recall_settings(index_path: Path) -> tuple[dict, str]:
    """The run's own recall settings, read from its runtime config when present."""
    import yaml

    cfg = {"selector": "word", "max_entities": 5, "max_aliases": 2, "max_desc": 3, "min_relevance": 0.0}
    hash_dir = next((p.name for p in index_path.parents if p.parent.name == "indexes"), None)
    for path in sorted((OUT_DIR / "novelhopqa_probe" / "runs").glob("*/*/*/runtime_config.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if hash_dir and hash_dir not in str(data.get("output_dir", "")):
            continue
        kw = (data.get("ingest") or {}).get("strategy_kwargs") or {}
        cfg.update(
            selector=str(kw.get("active_entity_selector", cfg["selector"])),
            max_entities=int(kw.get("max_entities", cfg["max_entities"])),
            max_aliases=int(kw.get("max_aliases", cfg["max_aliases"])),
            max_desc=int(kw.get("max_desc", cfg["max_desc"])),
            min_relevance=float(kw.get("active_entity_min_relevance", cfg["min_relevance"])),
        )
        return cfg, str(path)
    return cfg, "runtime config not found; defaults of the published profile used"


# ---------------------------------------------------------------------------
# Question 1
# ---------------------------------------------------------------------------


def q1_classify(book: Book, step: int, form: str) -> dict:
    form_t = content_token_set(form)
    rec = book.record(step)
    out = {"step": step, "class": "unknown", "detail": ""}
    if rec is None:
        out["detail"] = "step record missing"
        return out
    proposed, status = proposed_records(rec.raw_model_text)
    out["parser_status"] = rec.parser_status
    out["n_proposed"] = len(proposed)
    before_keys = {_key(_name(e)) for e in book.memory_before(step)}
    accepted = book.accepted(step)
    out["kept_names"] = [_name(e) for e in accepted]
    out["kept_carried_over"] = sum(1 for e in accepted if _key(_name(e)) in before_keys)
    if not proposed and status != "ok":
        out.update({"class": "parser loss", "detail": f"reply not parsed (status {status})"})
        return out
    hits = [(i, e) for i, e in enumerate(proposed) if in_any_field(e, form_t)]
    if not hits:
        out.update({"class": "not proposed as a record", "detail": f"word only in {reply_key_holding(rec.raw_model_text, form)}"})
        return out
    acc_by_key = {_key(_name(e)): e for e in accepted}
    outcomes = []
    for i, e in hits:
        kept = acc_by_key.get(_key(_name(e)))
        if kept is None:
            outcomes.append(("budget cut", i) if i >= book.recall_cfg["max_entities"] else ("parser loss", i))
        elif in_any_field(kept, form_t):
            outcomes.append(("merge loss", i))
        else:
            outcomes.append(("field cap cut", i))
    order = ["merge loss", "field cap cut", "parser loss", "budget cut"]
    cls, pos = min(outcomes, key=lambda o: order.index(o[0]))
    out.update({"class": cls, "position": pos, "record_name": _name(hits[[i for i, _ in hits].index(pos)][1])})
    if cls == "merge loss":
        key = _key(out["record_name"])
        gone = next(
            (
                s
                for s in book.steps
                if s >= step and not any(_key(_name(e)) == key and in_any_field(e, form_t) for e in book.memory_after(s))
            ),
            None,
        )
        if gone is None:
            # The earlier count matched the whole phrase ("the Firkin Cooper") against
            # memory.json; the record stores it without the article ("Firkin Cooper").
            out["class"] = "not lost (phrase-match artifact)"
            out["detail"] = "present in every snapshot and in the final memory under the token rule"
        else:
            out["detail"] = f"disappeared at step {gone}"
    elif cls == "field cap cut":
        e = hits[[i for i, _ in hits].index(pos)][1]
        where = "alias" if any(_in(a, form_t) for a in _aliases(e)) else "description"
        out["detail"] = (
            f"record kept, the {where} carrying the name was beyond the cap ({book.recall_cfg['max_aliases']} aliases, {book.recall_cfg['max_desc']} descriptions)"
        )
    elif cls == "parser loss":
        out["detail"] = f"record at position {pos} proposed, no record produced (parser status {rec.parser_status})"
    else:
        out["detail"] = f"record at position {pos} of {len(proposed)} proposed; {book.recall_cfg['max_entities']} kept"
    return out


# ---------------------------------------------------------------------------
# Question 2
# ---------------------------------------------------------------------------


def q2_link_step(book: Book, step, earlier_form: str, new_form: str) -> dict:
    """At `step`, was a record of `earlier_form` in memory, recalled, and joined with `new_form`?"""
    e_t, n_t = content_token_set(earlier_form), content_token_set(new_form)
    out = {"step": step}
    rec = book.record(step)
    if rec is None:
        out["class"] = "unknown"
        return out
    earlier = [e for e in book.memory_before(step) if in_name_alias(e, e_t)]
    out["earlier_records"] = [_name(e) for e in earlier]
    if not earlier:
        out["class"] = "no earlier record"
        return out
    recalled, scores = book.recall(step)
    out["recalled"] = recalled
    hit = [e for e in earlier if _key(_name(e)) in recalled]
    if not hit:
        best = max(scores.get(_key(_name(e)), 0.0) for e in earlier)
        out["class"] = "not recalled"
        out["earlier_score"] = best
        if best <= book.recall_cfg["min_relevance"]:
            out["reason"] = "word-overlap score zero"
        elif len(recalled) >= book.recall_cfg["max_entities"]:
            out["reason"] = "recall budget full with other records"
        else:
            out["reason"] = "other"
        return out
    out["class"] = "recalled"
    proposed = book.proposals(int(step))
    hit_keys = {_key(_name(e)) for e in hit}
    joined = any(
        (_key(_name(p)) in hit_keys and in_any_field(p, n_t))
        or (in_name_alias(p, e_t) and in_any_field(p, n_t))
        or (in_name_alias(p, n_t) and in_any_field(p, e_t))
        for p in proposed
    )
    if joined:
        out["outcome"] = "joined"
    elif any(in_any_field(p, n_t) for p in proposed):
        out["outcome"] = "new separate record"
    elif any(in_any_field(p, e_t) for p in proposed):
        out["outcome"] = "earlier record written, new name not added"
    else:
        out["outcome"] = "nothing written"
    return out


# ---------------------------------------------------------------------------
# Question 3
# ---------------------------------------------------------------------------


def pair_class(a, b, f1_t: set[str], f2_t: set[str]) -> str:
    sa, sb = stripped(_name(a)), stripped(_name(b))
    if sa and sa == sb:
        return "title variant"
    if sa and sb and (sa == sb[-len(sa) :] or sb == sa[-len(sb) :]):
        return "surname only against full name"
    a1, a2 = in_name_alias(a, f1_t), in_name_alias(a, f2_t)
    b1, b2 = in_name_alias(b, f1_t), in_name_alias(b, f2_t)
    if (a1 and not a2 and b2 and not b1) or (a2 and not a1 and b1 and not b2):
        return "given name against surname"
    return "other"


def published_index() -> Path | None:
    import os

    env = os.environ.get("PROBE_PUBLISHED_INDEX")
    if env:
        return Path(env) if Path(env).exists() else None
    preferred = (
        ROOT
        / "outputs/novelhopqa_ablation/500_tokens/llm_cache/novelhopqa/hierarchical/branching_factor_5/chunk_500/AE_5_2_3/Qwen3-30B-A3B-Instruct-2507/AEWord/index.json"
    )
    if preferred.exists():
        return preferred
    for path in sorted((ROOT / "outputs").glob("**/Qwen3-30B-A3B-Instruct-2507/AEWord/index.json")):
        s = str(path)
        if "novelhopqa" in s and "probe_bridge" not in s and "memory_ablation" not in s:
            return path
    return None


def q3_published(book_ids: list[str]) -> dict:
    from saadi.indexing.retrieval_methods.mbr import build_mbr_document_memory_index

    path = published_index()
    if path is None:
        return {
            "missing": "no published NovelHopQA Qwen3-30B construction cache (AEWord/index.json) under outputs/; set PROBE_PUBLISHED_INDEX"
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    have = {str(d.get("doc_id")) for d in payload.get("docs", [])}
    out = {"index": str(path), "books": {}, "missing_books": [b for b in book_ids if b not in have]}
    groups_all: list[tuple[str, tuple, list[str]]] = []
    shared_surname_pairs = 0
    title_conflict_groups = 0
    n_records = 0
    for book in [b for b in book_ids if b in have]:
        di = build_mbr_document_memory_index(path, doc_id=book)
        by_stripped: dict[tuple, list[str]] = defaultdict(list)
        for e in di.memory_entities:
            s = stripped(e.name)
            if s:
                by_stripped[s].append(e.name)
        n_records += len(di.memory_entities)
        dup = {s: names for s, names in by_stripped.items() if len(names) >= 2}
        for s, names in dup.items():
            groups_all.append((book, s, names))
            titles = {tuple(t for t in tokens(n) if t in TITLE_TOKENS) for n in names}
            if len(titles) > 1:
                title_conflict_groups += 1
        # Different people a surname rule would join: multi-word names sharing the last word only.
        multi = [s for s in by_stripped if len(s) >= 2]
        by_last: dict[str, set] = defaultdict(set)
        for s in multi:
            by_last[s[-1]].add(s)
        shared_surname_pairs += sum(len(v) * (len(v) - 1) // 2 for v in by_last.values() if len(v) >= 2)
        out["books"][book] = {"records": len(di.memory_entities), "stripped_names_with_2_or_more_records": len(dup)}
    groups_all.sort(key=lambda g: -len(g[2]))
    out.update(
        {
            "records": n_records,
            "stripped_names_with_2_or_more_records": len(groups_all),
            "records_in_such_groups": sum(len(g[2]) for g in groups_all),
            "groups_whose_titles_differ": title_conflict_groups,
            "pairs_sharing_only_a_surname": shared_surname_pairs,
            "top10": [{"book": b, "stripped": " ".join(s), "names": n} for b, s, n in groups_all[:10]],
        }
    )
    return out


# ---------------------------------------------------------------------------
# Question 4
# ---------------------------------------------------------------------------


def q4_scores(book: Book, p: dict) -> dict:
    from saadi.indexing.retrieval_methods.mbr import (
        MBR_VARIANT_MBR2,
        combine_mbr_variant_chunk_scores,
        score_mbr_variant_document_pools,
    )

    cfg = PUBLISHED_SETTINGS
    di = book.index
    streams = score_mbr_variant_document_pools(
        di,
        query=p["query"],
        retriever_name="bm25",
        variant=MBR_VARIANT_MBR2,
        memory_name_weight=cfg["memory_name_weight"],
        memory_alias_weight=cfg["memory_alias_weight"],
        memory_description_weight=cfg["memory_description_weight"],
        memory_aggregation=cfg["memory_aggregation"],
    )

    def ranked(lambda_n: float):
        return combine_mbr_variant_chunk_scores(
            di,
            streams,
            variant=MBR_VARIANT_MBR2,
            lambda_t=cfg["lambda_t"],
            lambda_n=lambda_n,
            lambda_h=cfg["lambda_h"],
            normalization_method=cfg["normalization_method"],
            memory_name_weight=cfg["memory_name_weight"],
            memory_alias_weight=cfg["memory_alias_weight"],
            memory_description_weight=cfg["memory_description_weight"],
        ).ranked_chunks

    gold = p["gold_chunk_id"]

    def gold_rank(rows) -> int:
        return next((i + 1 for i, r in enumerate(rows) if r.chunk_id == gold), len(rows) + 1)

    rows = ranked(cfg["lambda_n"])
    g_rank = gold_rank(rows)

    def row_dict(i, r) -> dict:
        w = r.effective_weights or {}
        contrib = {
            "text": w.get("text", 0.0) * (r.normalized_text_score or 0.0),
            "memory": w.get("memory", 0.0) * (r.normalized_memory_score or 0.0),
            "summary": w.get("summary", 0.0) * (r.normalized_summary_score or 0.0),
        }
        fields = {"name": r.name_score, "alias": r.alias_score, "description": r.description_score}
        best_field = max((k for k in fields if fields[k] is not None), key=lambda k: fields[k], default=None)
        return {
            "rank": i + 1,
            "chunk": r.chunk_id,
            "gold": r.chunk_id == gold,
            "text_raw": r.text_score,
            "text_z": r.normalized_text_score,
            "memory_raw": r.memory_score,
            "memory_field": best_field,
            "memory_z": r.normalized_memory_score,
            "summary_z": r.normalized_summary_score,
            "final": r.final_score,
            "carried_by": max(contrib, key=contrib.get),
        }

    table = [row_dict(i, r) for i, r in enumerate(rows[:10])]
    if g_rank > 10:
        table.append(row_dict(g_rank - 1, rows[g_rank - 1]))
    needed = next((lam for lam in LAMBDA_GRID if gold_rank(ranked(lam)) <= 10), None)
    mem_ranks = memory_channel_ranks(streams, di)
    gold_mem = streams.chunk_scores[gold].memory_score if gold in streams.chunk_scores else None
    ties = sum(1 for r in streams.chunk_scores.values() if gold_mem is not None and r.memory_score == gold_mem) - 1
    record = p.get("form1_record_name")
    shared = sorted(cid for cid, names in di.chunk_to_names.items() if record and record in names)
    return {
        "persona_id": p["persona_id"],
        "cell": p["cell"],
        "query": p["query"],
        "gold": gold,
        "record": record,
        "gold_rank_published": g_rank,
        "gold_rank_lambda_n_1.0": gold_rank(ranked(1.0)),
        "gold_rank_lambda_n_2.0": gold_rank(ranked(2.0)),
        "smallest_lambda_n_for_top10": needed,
        "gold_memory_rank": mem_ranks.get(gold),
        "chunks_tied_with_gold_memory_score": max(ties, 0),
        "gold_has_top_memory_score": gold_mem is not None
        and gold_mem >= max(r.memory_score for r in streams.chunk_scores.values() if r.memory_score is not None),
        "chunks_linked_to_record": len(shared),
        "table": table,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(report: StepReport, argv: list[str]) -> None:
    import numpy as np

    from saadi.indexing.retrieval_methods.mbr import _read_index_payload

    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    personas = read_jsonl(PERSONAS_JSONL)
    memory_rows = {p["persona_id"]: p for p in read_jsonl(MEMORY_JSONL)}
    index_path = common.find_cache_index()
    payload = _read_index_payload(index_path)
    recall_cfg, recall_src = recall_settings(index_path)
    books = {b: Book(index_path, payload, b, recall_cfg) for b in sorted({p["book_id"] for p in personas})}
    rng = np.random.default_rng(SEED)
    report.commands.append(
        f"read {index_path}; memory per step replayed by the cache loader; recall re-run with {recall_cfg} ({recall_src})"
    )
    report.add(
        "Matching and reconstruction",
        f"Titles stripped before names are compared: {', '.join(TITLES)}. A form is present in a field when all its content "
        "tokens occur there (step 6's rule). The memory before each step is the loader's replay of the earlier steps; the "
        "recalled records are the recall rule re-run on that memory and the chunk text.",
    )
    with_reveal = [p for p in personas if p["cell"] in REVEAL_CELLS]

    # ---------------- Q1 ----------------
    q1 = []
    for p in with_reveal:
        b = books[p["book_id"]]
        for kind, form, step in (("form1", p["form1"], p["gold_chunk_index"]), ("form2", p["form2"], p.get("reveal_chunk_index"))):
            rec = b.record(step)
            if rec is None:
                q1.append(
                    {"persona_id": p["persona_id"], "cell": p["cell"], "form": kind, "class": "unknown", "detail": "step record missing"}
                )
                continue
            if word_in_text(rec.raw_model_text, form) and not word_in_text(b.memory_text, form):
                row = q1_classify(b, int(step), form)
                row.update({"persona_id": p["persona_id"], "cell": p["cell"], "form": kind, "name": form, "raw_reply": rec.raw_model_text})
                q1.append(row)
    write_jsonl(DIAG_DIR / "q1_lost_names.jsonl", q1)
    classes = [
        "budget cut",
        "field cap cut",
        "merge loss",
        "parser loss",
        "not proposed as a record",
        "not lost (phrase-match artifact)",
        "unknown",
    ]
    cells = sorted({p["cell"] for p in with_reveal})
    rows = []
    for c in classes:
        grp = [r for r in q1 if r["class"] == c]
        per_cell = Counter(r["cell"] for r in grp)
        rows.append(
            [c, sum(r["form"] == "form1" for r in grp), sum(r["form"] == "form2" for r in grp), len(grp)]
            + [per_cell.get(x, 0) for x in cells]
        )
    report.table(
        f"Q1. Names in the model's reply but absent from the final memory ({len(q1)})",
        ["class", "form 1", "form 2", "total"] + [f"cell {x}" for x in cells],
        rows,
    )
    cuts = [r for r in q1 if r["class"] == "budget cut"]
    if cuts:
        mean_prop = sum(r["n_proposed"] for r in cuts) / len(cuts)
        slots = sum(len(r["kept_names"]) for r in cuts)
        carried = sum(r["kept_carried_over"] for r in cuts)
        report.add(
            "Q1. Budget cuts",
            f"Mean records proposed at those chunks: {mean_prop:.1f} (budget {recall_cfg['max_entities']}). "
            f"Kept slots taken by carried-over records: {pct(carried, slots)}.",
        )
    outside = Counter(r["detail"] for r in q1 if r["class"] == "not proposed as a record")
    if outside:
        report.add(
            "Q1. Names not proposed as a record",
            "The word appears in the reply only outside any proposed record (the earlier 'written in the reply' count "
            f"used a whole-text match): {dict(outside.most_common())}.",
        )
    ex_lines = []
    for c in classes:
        grp = [r for r in q1 if r["class"] == c]
        if not grp:
            continue
        picks = sorted(int(i) for i in rng.choice(len(grp), size=min(3, len(grp)), replace=False))
        for i in picks:
            r = grp[i]
            reply = " ".join(str(r.get("raw_reply", "")).split())[:400]
            ex_lines.append(
                f"- **{c}**, {r['persona_id']} ({r['form']} '{r.get('name')}'), step {r.get('step')}: {r.get('detail')}. "
                f"Kept: {r.get('kept_names')}. Reply (first 400 characters): `{reply}`"
            )
    report.add("Q1. Examples (seed 42; full replies in outputs/probe_bridge/diagnose/q1_lost_names.jsonl)", "\n".join(ex_lines) or "None.")

    # ---------------- Q2 ----------------
    q2 = []
    for p in with_reveal:
        b = books[p["book_id"]]
        row = {"persona_id": p["persona_id"], "cell": p["cell"]}
        row["at_reveal"] = q2_link_step(b, p.get("reveal_chunk_index"), p["form1"], p["form2"])
        if p["cell"] == 4:
            row["at_gold"] = q2_link_step(b, p["gold_chunk_index"], p["form2"], p["form1"])
        f1_t, f2_t = content_token_set(p["form1"]), content_token_set(p["form2"])
        wrote1 = any(in_any_field(e, f1_t) for s in b.steps for e in b.proposals(s))
        wrote2 = any(in_any_field(e, f2_t) for s in b.steps for e in b.proposals(s))
        row["both_written"] = bool(wrote1 and wrote2)
        q2.append(row)
    write_jsonl(DIAG_DIR / "q2_recall.jsonl", q2)

    def q2_label(x: dict) -> str:
        if x.get("class") == "recalled":
            return f"recalled: {x['outcome']}"
        if x.get("class") == "not recalled":
            return f"not recalled: {x['reason']}"
        return x.get("class", "unknown")

    labels = sorted({q2_label(r["at_reveal"]) for r in q2})
    report.table(
        "Q2. At the reveal chunk: was the earlier record of form 1 in the prompt? (210 characters)",
        ["class", "total"] + [f"cell {x}" for x in cells],
        [
            [lab, sum(q2_label(r["at_reveal"]) == lab for r in q2)]
            + [sum(q2_label(r["at_reveal"]) == lab and r["cell"] == x for r in q2) for x in cells]
            for lab in labels
        ],
    )
    c4 = [r for r in q2 if r["cell"] == 4]
    c4_labels = sorted({q2_label(r["at_gold"]) for r in c4})
    report.table(
        "Q2. Cell 4 at the gold chunk: was the earlier record of form 2 recalled, and did the model link form 1 to it?",
        ["class", "cell 4"],
        [[lab, sum(q2_label(r["at_gold"]) == lab for r in c4)] for lab in c4_labels],
    )
    both = [r for r in q2 if r["both_written"]]
    present = [r for r in both if (r["at_gold"] if r["cell"] == 4 else r["at_reveal"]).get("class") == "recalled"]
    report.add(
        "Q2. Earlier record in the prompt where the link could form",
        f"Of the {len(both)} characters whose two names were both written in some proposed record, the earlier record was in "
        f"the prompt at the step where the link could form (the reveal; the gold chunk in cell 4) in {pct(len(present), len(both))}.",
    )

    # ---------------- Q3 ----------------
    q3 = []
    pair_counts: Counter = Counter()
    for p in personas:
        b = books[p["book_id"]]
        f1_t, f2_t = content_token_set(p["form1"]), content_token_set(p["form2"])
        recs = [e for e in b.index.memory_entities if in_name_alias(e, f1_t) or in_name_alias(e, f2_t)]
        desc_only = [
            e
            for e in b.index.memory_entities
            if e not in recs and (any(_in(d, f1_t) for d in _descs(e)) or any(_in(d, f2_t) for d in _descs(e)))
        ]
        gold_l, rev_l = b.linked(p["gold_chunk_id"]), b.linked(p.get("reveal_chunk_id"))
        pairs = [pair_class(recs[i], recs[j], f1_t, f2_t) for i in range(len(recs)) for j in range(i + 1, len(recs))]
        pair_counts.update(pairs)
        q3.append(
            {
                "persona_id": p["persona_id"],
                "cell": p["cell"],
                "form1": p["form1"],
                "form2": p["form2"],
                "n_records": len(recs),
                "n_description_only": len(desc_only),
                "pair_classes": pairs,
                "records": [
                    {
                        "name": e.name,
                        "aliases": list(e.aliases),
                        "descriptions": list(e.descriptions)[:3],
                        "gold_linked": _key(e.name) in gold_l,
                        "reveal_linked": _key(e.name) in rev_l,
                    }
                    for e in recs
                ],
            }
        )
    write_jsonl(DIAG_DIR / "q3_duplicates.jsonl", q3)

    def bucket(n: int) -> str:
        return "3+" if n >= 3 else str(n)

    all_cells = sorted({p["cell"] for p in personas})
    report.table(
        "Q3. Records per character in the final memory (name or alias holds form 1 or form 2)",
        ["records", "total"] + [f"cell {x}" for x in all_cells],
        [
            [k, sum(bucket(r["n_records"]) == k for r in q3)]
            + [sum(bucket(r["n_records"]) == k and r["cell"] == x for r in q3) for x in all_cells]
            for k in ("0", "1", "2", "3+")
        ],
    )
    report.add(
        "Q3. Pair classes (characters with 2 or more records)",
        f"Characters with 2 or more records: {sum(r['n_records'] >= 2 for r in q3)}. Pairs: {dict(pair_counts.most_common())}. "
        f"Characters with a further record holding a form only in its description: {sum(r['n_description_only'] > 0 for r in q3)}.",
    )
    multi = [r for r in q3 if r["n_records"] >= 2]
    lines = []
    if multi:
        for i in sorted(int(i) for i in rng.choice(len(multi), size=min(5, len(multi)), replace=False)):
            r = multi[i]
            side = "; ".join(
                f"'{x['name']}' aliases={x['aliases']} gold={'yes' if x['gold_linked'] else 'no'} reveal={'yes' if x['reveal_linked'] else 'no'}"
                for x in r["records"]
            )
            lines.append(f"- **{r['persona_id']}** (cell {r['cell']}, {r['form1']} / {r['form2']}): {side}")
    report.add("Q3. Five examples (seed 42)", "\n".join(lines) or "No character has 2 or more records.")
    book_ids = [row["doc_id"] for row in read_json(common.BOOKS_JSON)["selected"]]
    pub = q3_published(book_ids or sorted(books))
    write_json(DIAG_DIR / "q3_published.json", pub)
    if "missing" in pub:
        report.unexpected.append(f"Q3, published memories: {pub['missing']}. That part was skipped; the rest ran.")
    else:
        report.add(
            "Q3. The paper's own memories of the 10 unmodified books",
            f"From `{pub['index']}`: {pub['records']} records; {pub['stripped_names_with_2_or_more_records']} stripped names hold 2 or more "
            f"records ({pub['records_in_such_groups']} records). Groups whose titles differ (possibly two people, e.g. Mr./Mrs.): "
            f"{pub['groups_whose_titles_differ']}. Pairs of distinct multi-word names sharing only a surname (different people a "
            f"surname rule would join): {pub['pairs_sharing_only_a_surname']}."
            + (f" Books missing from that cache: {pub['missing_books']}." if pub["missing_books"] else ""),
        )
        report.table(
            "Q3. Ten most duplicated stripped names (unmodified books)",
            ["book", "stripped name", "records"],
            [[g["book"], g["stripped"], "; ".join(g["names"])] for g in pub["top10"]],
        )

    # ---------------- Q4 ----------------
    bridged = [
        p
        for p in personas
        if memory_rows.get(p["persona_id"], {}).get("form2_in_that_record_final")
        and memory_rows[p["persona_id"]].get("gold_chunk_linked_to_that_record")
    ]
    q4 = [q4_scores(books[p["book_id"]], {**p, **memory_rows[p["persona_id"]]}) for p in bridged]
    write_json(DIAG_DIR / "q4_scores.json", q4)
    for r in q4:
        report.table(
            f"Q4. {r['persona_id']} (cell {r['cell']}): \"{r['query']}\", record '{r['record']}', gold rank {r['gold_rank_published']}",
            ["rank", "chunk", "text raw", "text z", "memory raw (field)", "memory z", "summary z", "final", "carried by"],
            [
                [
                    x["rank"],
                    x["chunk"] + (" (gold)" if x["gold"] else ""),
                    f"{x['text_raw']:.2f}" if x["text_raw"] is not None else "-",
                    f"{x['text_z']:.2f}" if x["text_z"] is not None else "-",
                    f"{x['memory_raw']:.2f} ({x['memory_field']})" if x["memory_raw"] is not None else "-",
                    f"{x['memory_z']:.2f}" if x["memory_z"] is not None else "-",
                    f"{x['summary_z']:.2f}" if x["summary_z"] is not None else "-",
                    f"{x['final']:.3f}",
                    x["carried_by"],
                ]
                for x in r["table"]
            ],
        )
    if q4:
        report.table(
            "Q4. Summary",
            [
                "character",
                "gold rank",
                "rank at memory weight 1.0",
                "rank at 2.0",
                "smallest memory weight for top 10",
                "gold memory rank",
                "chunks tied on memory score",
                "chunks linked to the record",
            ],
            [
                [
                    r["persona_id"],
                    r["gold_rank_published"],
                    r["gold_rank_lambda_n_1.0"],
                    r["gold_rank_lambda_n_2.0"],
                    r["smallest_lambda_n_for_top10"] if r["smallest_lambda_n_for_top10"] is not None else "> 20",
                    r["gold_memory_rank"],
                    r["chunks_tied_with_gold_memory_score"],
                    r["chunks_linked_to_record"],
                ]
                for r in q4
            ],
        )
        report.add(
            "Q4. Note",
            f"Memory weight is lambda_n (published {PUBLISHED_SETTINGS['lambda_n']}), other weights fixed, searched on a grid of 0.05 up to 20. "
            "This describes the landscape only; no weight is changed anywhere. The gold chunk had the top memory score in the book "
            f"in {sum(r['gold_has_top_memory_score'] for r in q4)} of {len(q4)} cases.",
        )
    else:
        report.add("Q4", "No character had the bridge word in the gold chunk's live record.")

    # ---------------- Decision table ----------------
    def q1n(*cls):
        return sum(r["class"] in cls for r in q1)

    report.table(
        "Counts against the candidate fixes",
        ["Finding", "Count", "Fix it points to"],
        [
            ["Names cut by the 5-record budget", q1n("budget cut"), "Separate slots for new names, or a larger budget"],
            ["Names cut by the alias/description caps", q1n("field cap cut"), "Larger field caps"],
            ["Names lost in merge or parsing", q1n("merge loss", "parser loss"), "Update step or output format"],
            ["Names never proposed as a record", q1n("not proposed as a record"), "Extraction instructions"],
            [
                "Earlier record not recalled at the reveal",
                sum(r["at_reveal"].get("class") == "not recalled" for r in q2),
                "Recall by name tokens, not only by word overlap",
            ],
            [
                "Recalled, but the model wrote a separate record",
                sum(r["at_reveal"].get("outcome") == "new separate record" for r in q2),
                "Extraction instructions",
            ],
            [
                "Duplicate records that differ only by a title",
                pair_counts.get("title variant", 0),
                "Merge on the stripped name, with the surname caution",
            ],
            [
                "Link formed, text channel still wins",
                sum(r["gold_rank_published"] > 10 and r["gold_has_top_memory_score"] for r in q4),
                "Fusion or record design, to be decided later",
            ],
        ],
    )
    unknown = sum(r["class"] == "unknown" for r in q1) + sum(r["at_reveal"].get("class") == "unknown" for r in q2)
    if unknown:
        report.unexpected.append(f"{unknown} cases could not be classified from the records (counted as unknown).")
    report.numbers.update(
        {
            "q1": dict(Counter(r["class"] for r in q1)),
            "q2_at_reveal": dict(Counter(q2_label(r["at_reveal"]) for r in q2)),
            "q2_cell4_at_gold": dict(Counter(q2_label(r["at_gold"]) for r in c4)),
            "q2_both_written": len(both),
            "q2_present_where_link_could_form": len(present),
            "q3_records_hist": dict(Counter(bucket(r["n_records"]) for r in q3)),
            "q3_pairs": dict(pair_counts),
            "q3_published": {k: v for k, v in pub.items() if k != "top10"},
            "q4": [{k: v for k, v in r.items() if k != "table"} for r in q4],
            "recall_settings": recall_cfg,
        }
    )
    for name in ("q1_lost_names.jsonl", "q2_recall.jsonl", "q3_duplicates.jsonl", "q3_published.json", "q4_scores.json"):
        report.files_written.append(str(DIAG_DIR / name))


if __name__ == "__main__":
    raise SystemExit(run_step(9, "Diagnose: where are the names lost", main, requires=[6, 7]))
