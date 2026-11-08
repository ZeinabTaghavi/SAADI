"""The persona probe as every baseline sees it: frozen spec, modified books, chunks, chunk ids, personas.

Everything is imported from the probe's own code (experiments/probe_bridge/common.py),
never copied: the chunker (SAADI's, 500 whitespace tokens at sentence boundaries),
the spec files, the SHA-256 check of the frozen spec. Chunk ids are the probe's
``<book_id>:<chunk_index>``, the ids the gold and reveal chunks carry in personas.jsonl.

Run as a script for the spec part of 00_check:  python common/load_probe.py --check
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import dataclass, field
from functools import cache, lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent
ROOT = EXP_DIR.parents[1]
PROBE_DIR = Path(os.environ.get("PBL_PROBE_DIR", ROOT / "experiments" / "probe_bridge"))
for _p in (str(ROOT / "src"), str(PROBE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# The probe's common module, under a name that cannot shadow anything.
probe = sys.modules.get("probe_common") or _load_module(PROBE_DIR / "common.py", "probe_common")

SEED = probe.SEED
PERSONAS_JSONL = probe.PERSONAS_JSONL
SPEC_FROZEN_JSON = probe.SPEC_FROZEN_JSON
BOOKS_JSON = probe.BOOKS_JSON
DATA_DIR = probe.DATA_DIR
read_json, write_json, read_jsonl, write_jsonl, sha256_file = (
    probe.read_json,
    probe.write_json,
    probe.read_jsonl,
    probe.write_jsonl,
    probe.sha256_file,
)
REVEAL_CELLS = (1, 2, 3, 4, 6, 7, 8)


def verify_frozen_spec() -> str:
    """SHA-256 of personas.jsonl against the frozen spec; raises when it differs."""

    frozen = read_json(SPEC_FROZEN_JSON)
    actual = sha256_file(PERSONAS_JSONL)
    if actual != frozen["personas_jsonl_sha256"]:
        raise RuntimeError(f"personas.jsonl hash {actual} differs from the frozen spec {frozen['personas_jsonl_sha256']}; refusing to run.")
    return actual


@lru_cache(maxsize=1)
def personas() -> tuple[dict, ...]:
    return tuple(read_jsonl(PERSONAS_JSONL))


def book_ids() -> list[str]:
    return [row["doc_id"] for row in read_json(BOOKS_JSON)["selected"]]


def chunk_id(book_id: str, index: int) -> str:
    return f"{book_id}:{int(index)}"


@dataclass
class Book:
    book_id: str
    chunks: list[str]  # chunk texts, index = chunk index
    text: str  # modified.txt
    personas: list[dict] = field(default_factory=list)

    @property
    def chunk_ids(self) -> list[str]:
        return [chunk_id(self.book_id, i) for i in range(len(self.chunks))]

    @property
    def n_chunks(self) -> int:
        return len(self.chunks)


@cache
def load_book(book_id: str) -> Book:
    """The modified book, cut exactly as SAADI's ingest cut it (the probe's chunk_book)."""

    text = (DATA_DIR / "books" / book_id / "modified.txt").read_text(encoding="utf-8")
    chunks = [c for c, _s, _e in probe.chunk_book(text)]
    group = [dict(p) for p in personas() if p["book_id"] == book_id]
    return Book(book_id=book_id, chunks=chunks, text=text, personas=group)


def check_spec() -> dict:
    """The spec part of 00_check: hash, 10 books, chunk counts, gold/reveal sentences in their chunks."""

    out = {"spec_sha256": verify_frozen_spec(), "books": {}, "problems": []}
    # books.json holds the ORIGINAL books' chunk counts; the modified books have one or two more
    # (24 inserted sentences). SAADI's builds made one call per modified chunk: run 0 and run 1
    # recorded 244 ... 385, the counts these chunks must match.
    original = {row["doc_id"]: row["n_chunks"] for row in read_json(BOOKS_JSON)["selected"]}
    saadi = {}
    run1_step2 = ROOT / "experiments" / "method_update" / "p1_ledger" / "reports" / "step_02.json"
    if run1_step2.exists():
        saadi = {b: v["chunk_records"] for b, v in read_json(run1_step2)["numbers"]["books"].items()}
    out["chunk_count_reference"] = "SAADI run 0/1 construction records (p1_ledger reports/step_02.json)" if saadi else "none found"
    for book_id in book_ids():
        book = load_book(book_id)
        out["books"][book_id] = {
            "chunks": book.n_chunks,
            "saadi_chunks": saadi.get(book_id),
            "original_book_chunks": original.get(book_id),
            "personas": len(book.personas),
        }
        if saadi and saadi.get(book_id) != book.n_chunks:
            out["problems"].append(f"{book_id}: {book.n_chunks} chunks here, SAADI's builds used {saadi.get(book_id)}")
        if book.n_chunks < int(original.get(book_id) or 0):
            out["problems"].append(f"{book_id}: fewer chunks than the original book")
        for p in book.personas:
            if p["gold_sentence"] not in book.chunks[int(p["gold_chunk_index"])]:
                out["problems"].append(f"{p['persona_id']}: gold sentence not in chunk {p['gold_chunk_index']}")
            if (
                p.get("reveal_chunk_index") is not None
                and p.get("reveal_sentence")
                and p["reveal_sentence"] not in book.chunks[int(p["reveal_chunk_index"])]
            ):
                out["problems"].append(f"{p['persona_id']}: reveal sentence not in chunk {p['reveal_chunk_index']}")
    out["personas"] = len(personas())
    out["cells"] = sorted({p["cell"] for p in personas()})
    return out


PERSONA_FIELDS = (
    "persona_id",
    "book_id",
    "cell",
    "family",
    "query",
    "form1",
    "form2",
    "gold_chunk_id",
    "gold_chunk_index",
    "reveal_chunk_id",
    "reveal_chunk_index",
    "gold_sentence",
    "reveal_sentence",
)


def export_inputs(out_dir: Path) -> Path:
    """Write <out_dir>/<book_id>.json for every book plus manifest.json (read by chunk_map in any venv)."""

    out_dir.mkdir(parents=True, exist_ok=True)
    books = {}
    for book_id in book_ids():
        book = load_book(book_id)
        payload = {
            "book_id": book_id,
            "chunk_ids": book.chunk_ids,
            "chunks": book.chunks,
            "personas": [{k: p.get(k) for k in PERSONA_FIELDS} for p in book.personas],
        }
        (out_dir / f"{book_id}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        books[book_id] = {"chunks": book.n_chunks, "personas": len(book.personas)}
    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps({"spec_sha256": verify_frozen_spec(), "books": books}, indent=1), encoding="utf-8")
    return manifest


def bm25_scores(texts: list[str], ids: list[str], query: str) -> dict[str, float]:
    """BM25 over the given texts, through the very function the probe's Vanilla uses (MBR's text pool)."""

    from saadi.indexing.retrieval_methods.mbr import _score_pool

    return _score_pool(list(texts), ids=list(ids), query=query, retriever_name="bm25")


def bm25_ranking(texts: list[str], ids: list[str], query: str) -> list[str]:
    """All ids ranked by BM25, ties broken by chunk position: step 7's Vanilla ranking."""

    scores = bm25_scores(texts, ids, query)
    position = {cid: i for i, cid in enumerate(ids)}
    return sorted(ids, key=lambda cid: (-float(scores.get(cid, 0.0)), position[cid]))


if __name__ == "__main__":
    if "--export" in sys.argv:
        target = Path(sys.argv[sys.argv.index("--export") + 1])
        print(export_inputs(target))
        raise SystemExit(0)
    if "--check" in sys.argv:
        result = check_spec()
        print(json.dumps(result, indent=1))
        raise SystemExit(1 if result["problems"] else 0)
    print(__doc__)
