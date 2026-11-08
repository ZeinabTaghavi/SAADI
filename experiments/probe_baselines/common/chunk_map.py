"""Maps any system's output units back to the paper's chunk ids, and reads/writes the shared rankings.

Pure standard library on purpose: HippoRAG v2 and LQCA run in their own virtual
environments (their pins clash with the study's torch/vLLM), so they cannot import
SAADI. They read the probe exported once by 00_check
(outputs/probe_baselines/inputs/<book_id>.json, written from common/load_probe.py)
and write rankings through this module.

Shared ranking format, one file per system and book,
outputs/probe_baselines/rankings/<system>/<book_id>.jsonl, one line per persona:

    {"persona_id", "book_id", "cell", "ranked_chunk_ids": [top 50 original chunk ids],
     "gold_rank", "n_chunks", "model_calls", "input_tokens", "generated_tokens", ...}

``gold_rank`` (1-based rank of the gold chunk in the system's full ranking; n_chunks + 1
when absent) and ``n_chunks`` are added to the requested fields: the median gold rank is
around 110-200 on this probe, far below a top-50 list, so it cannot be read from the list.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from collections.abc import Iterable
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent
ROOT = EXP_DIR.parents[1]
OUT_DIR = Path(os.environ.get("PBL_OUT_DIR", ROOT / "outputs" / "probe_baselines"))
INPUTS_DIR = OUT_DIR / "inputs"
RANKINGS_DIR = OUT_DIR / "rankings"
TOP_N = 50


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def book_ids() -> list[str]:
    manifest = INPUTS_DIR / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"{manifest} is missing; 00_check exports the probe there first.")
    return list(json.loads(manifest.read_text(encoding="utf-8"))["books"])


def load_inputs(book_id: str) -> dict:
    """{book_id, chunk_ids, chunks, personas: [{persona_id, cell, query, gold_chunk_id, ...}]}"""

    return json.loads((INPUTS_DIR / f"{book_id}.json").read_text(encoding="utf-8"))


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text or ""))).strip()


class ChunkMap:
    """Original chunk ids of one book, found from a unit's text (exact, then whitespace/Unicode-normalized)."""

    def __init__(self, chunk_ids: list[str], chunks: list[str]) -> None:
        if len(chunk_ids) != len(chunks):
            raise ValueError("chunk_ids and chunks differ in length")
        self.chunk_ids = list(chunk_ids)
        self.exact: dict[str, str] = {}
        self.normalized: dict[str, str] = {}
        self.duplicates: list[str] = []
        for cid, text in zip(chunk_ids, chunks, strict=True):
            if text in self.exact:
                self.duplicates.append(cid)
            self.exact.setdefault(text, cid)
            self.normalized.setdefault(_norm(text), cid)
        self.unmapped: list[str] = []

    @classmethod
    def for_book(cls, book_id: str) -> ChunkMap:
        inp = load_inputs(book_id)
        return cls(inp["chunk_ids"], inp["chunks"])

    def to_id(self, text: str) -> str | None:
        cid = self.exact.get(text)
        if cid is None:
            cid = self.normalized.get(_norm(text))
        if cid is None:
            self.unmapped.append(str(text)[:120])
        return cid

    def map_ranked_texts(self, texts: list[str]) -> list[str]:
        """A system's ranked passage texts -> ranked original chunk ids (unmapped units dropped, duplicates kept once)."""

        out: list[str] = []
        seen: set[str] = set()
        for text in texts:
            cid = self.to_id(text)
            if cid is not None and cid not in seen:
                seen.add(cid)
                out.append(cid)
        return out


def ranking_row(
    persona: dict,
    ranked_ids: list[str],
    *,
    n_chunks: int,
    model_calls: int = 0,
    input_tokens: int = 0,
    generated_tokens: int = 0,
    **extra: Any,
) -> dict:
    gold = persona["gold_chunk_id"]
    rank = ranked_ids.index(gold) + 1 if gold in ranked_ids else int(n_chunks) + 1
    row = {
        "persona_id": persona["persona_id"],
        "book_id": persona["book_id"],
        "cell": int(persona["cell"]),
        "ranked_chunk_ids": list(ranked_ids[:TOP_N]),
        "gold_rank": rank,
        "n_chunks": int(n_chunks),
        "model_calls": int(model_calls),
        "input_tokens": int(input_tokens),
        "generated_tokens": int(generated_tokens),
    }
    row.update(extra)
    return row


def rankings_path(system: str, book_id: str) -> Path:
    return RANKINGS_DIR / system / f"{book_id}.jsonl"


def write_rankings(system: str, book_id: str, rows: list[dict]) -> Path:
    path = rankings_path(system, book_id)
    write_jsonl(path, rows)
    return path


def read_rankings(system: str, rankings_dir: Path | None = None) -> list[dict]:
    folder = (rankings_dir or RANKINGS_DIR) / system
    rows: list[dict] = []
    for path in sorted(folder.glob("*.jsonl")):
        rows += read_jsonl(path)
    return rows


def system_dir(system: str) -> Path:
    path = OUT_DIR / system
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_numbers(system: str, name: str, numbers: dict) -> Path:
    """Numbers a system script hands to its stage report (merged by report_utils)."""

    path = system_dir(system) / f"numbers_{name}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(numbers, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)
    return path


# ---------------------------------------------------------------------------
# vLLM server counters (exact calls and tokens per book, for every LLM stage)
# ---------------------------------------------------------------------------

_COUNTERS = ("prompt_tokens_total", "generation_tokens_total", "request_success_total", "prefix_cache_queries", "prefix_cache_hits")


def vllm_counters(base_url: str) -> dict[str, float]:
    """Sum of the server's Prometheus counters whose name contains one of _COUNTERS (all labels)."""

    import urllib.request

    url = base_url.rstrip("/")
    url = url[: -len("/v1")] if url.endswith("/v1") else url
    try:
        with urllib.request.urlopen(url + "/metrics", timeout=30) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return {}
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        for key in _COUNTERS:
            if key in name and not name.endswith("_created"):
                try:
                    out[key] = out.get(key, 0.0) + float(line.rsplit(" ", 1)[1])
                except ValueError:
                    pass
    return out


def counter_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {k: after.get(k, 0.0) - before.get(k, 0.0) for k in set(before) | set(after)}
