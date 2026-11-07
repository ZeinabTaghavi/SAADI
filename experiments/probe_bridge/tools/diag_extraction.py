#!/usr/bin/env python3
"""Read-only diagnostic after step 6: where does each persona's name get lost?

For every persona it looks at the construction records of the gold chunk (form 1)
and the reveal chunk (form 2) and asks, in order:
  1. did the model's raw reply mention the form at all,
  2. was it kept as an accepted record of that chunk,
  3. was it rejected by the anchor check,
  4. was the chunk's record budget (5) already full,
  5. does the form appear anywhere in the book's final memory.
Writes reports/diag_extraction.md and .json. Changes nothing else; not a numbered
step and no marker.

    python3 experiments/probe_bridge/tools/diag_extraction.py [--cache-index PATH]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
import common  # noqa: E402
from common import PERSONAS_JSONL, REPORT_DIR, read_jsonl  # noqa: E402

BUDGET = 5


def _has(text: str, form: str) -> bool:
    return bool(form) and re.search(r"\b" + re.escape(form.lower()) + r"\b", str(text or "").lower()) is not None


def _entity_text(entity: dict) -> str:
    return " | ".join(
        [str(entity.get("name", ""))]
        + [str(a) for a in entity.get("aliases", []) or []]
        + [str(d) for d in entity.get("description", entity.get("descriptions", [])) or []]
    )


def _names(entities: list[dict]) -> str:
    return " | ".join(str(e.get("name", "")) + " " + " ".join(str(a) for a in e.get("aliases", []) or []) for e in entities)


def check(rows: dict[int, dict], memory_text: str, all_active: list[tuple[int, str]], idx, form: str) -> dict:
    out = {
        "raw_reply": None,
        "accepted": None,
        "rejected": None,
        "budget_full": None,
        "in_final_memory": _has(memory_text, form),
        "first_step_in_memory": None,
    }
    for step, text in all_active:
        if _has(text, form):
            out["first_step_in_memory"] = step
            break
    if idx is None or int(idx) not in rows:
        return out
    r = rows[int(idx)]
    active = r.get("active_entities", []) or []
    out["raw_reply"] = _has(r.get("raw_model_text", ""), form)
    out["accepted"] = _has(_names(active), form) or any(_has(_entity_text(e), form) for e in active)
    out["rejected"] = any(_has(_entity_text(e), form) for e in (r.get("rejected_entities", []) or []))
    out["budget_full"] = len(active) >= BUDGET
    return out


def main(argv: list[str]) -> int:
    from saadi.indexing.retrieval_methods.mbr import _resolve_cache_file_path

    index_path = Path(argv[argv.index("--cache-index") + 1]) if "--cache-index" in argv else common.find_cache_index()
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    docs = {str(d["doc_id"]): d.get("files", {}) for d in payload.get("docs", [])}
    personas = read_jsonl(PERSONAS_JSONL)

    cache: dict[str, tuple] = {}
    results = []
    for p in personas:
        book = p["book_id"]
        if book not in cache:
            files = docs[book]
            steps = Path(_resolve_cache_file_path(files["steps_index"], base_dir=index_path.parent))
            mem = Path(_resolve_cache_file_path(files["memory"], base_dir=index_path.parent))
            rows = {int(r["step"]): r for r in map(json.loads, open(steps, encoding="utf-8"))}
            all_active = [(s, " || ".join(_entity_text(e) for e in r.get("active_entities", []) or [])) for s, r in sorted(rows.items())]
            cache[book] = (rows, mem.read_text(encoding="utf-8") if mem.exists() else "", all_active)
        rows, memory_text, all_active = cache[book]
        g = check(rows, memory_text, all_active, p.get("gold_chunk_index"), p["form1"])
        rv = (
            check(rows, memory_text, all_active, p.get("reveal_chunk_index"), p["form2"])
            if p.get("reveal_chunk_index") is not None
            else None
        )
        results.append(
            {
                "persona_id": p["persona_id"],
                "cell": p["cell"],
                "family": p["family"],
                "form1": p["form1"],
                "form2": p["form2"],
                "gold": g,
                "reveal": rv,
            }
        )

    def pct(vals) -> str:
        vals = [bool(v) for v in vals if v is not None]
        return f"{sum(vals)}/{len(vals)} ({100 * sum(vals) / len(vals):.0f}%)" if vals else "-"

    lines = [
        "# Bridge probe - extraction diagnostic (after step 6)",
        "",
        f"Read-only. Cache: `{index_path}`. Word match, case-insensitive. Budget = {BUDGET} records per chunk.",
        "",
    ]
    for label, key in (("Form 1 at the gold chunk", "gold"), ("Form 2 at the reveal chunk", "reveal")):
        lines += [
            f"## {label}",
            "",
            "| cell | n | in model's raw reply | kept as a record | rejected by anchor check | budget full (5) | anywhere in final memory |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for cell in sorted({r["cell"] for r in results}):
            grp = [r[key] for r in results if r["cell"] == cell and r[key] is not None]
            if not grp:
                lines.append(f"| {cell} | 0 | - | - | - | - | - |")
                continue
            lines.append(
                f"| {cell} | {len(grp)} | {pct(x['raw_reply'] for x in grp)} | {pct(x['accepted'] for x in grp)} | "
                f"{pct(x['rejected'] for x in grp)} | {pct(x['budget_full'] for x in grp)} | {pct(x['in_final_memory'] for x in grp)} |"
            )
        lines.append("")
    budget_lost = [r for r in results if r["gold"]["raw_reply"] is False and r["gold"]["budget_full"]]
    lines += [
        "## Reading",
        "",
        f"- Gold chunks where the model did not mention form 1 and the 5 slots were full: {len(budget_lost)}/{len(results)}.",
        "- 'in raw reply' but not 'kept' means the pipeline dropped it after the model wrote it (anchor check or budget trim).",
        "",
    ]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "diag_extraction.md").write_text("\n".join(lines), encoding="utf-8")
    (REPORT_DIR / "diag_extraction.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("\n".join(lines))
    print(REPORT_DIR / "diag_extraction.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
