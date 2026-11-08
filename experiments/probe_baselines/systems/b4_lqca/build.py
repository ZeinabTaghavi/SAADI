#!/usr/bin/env python3
"""B4 LQCA: coreference-normalized rewrite of each book with the official code, on segments.

Official code: github.com/OceannTwT/LQCA at a pinned commit, cloned by run_all.sh into
third_party/LQCA (the repository has no license file, so it is not committed here). Runs in
third_party/venv_lqca with maverick-coref (the coreference model the code imports as
`from maverick import Maverick`; its requirements.txt names "maverick", which on PyPI is an
unrelated poker library).

The official pipeline per text, called exactly as main.py's loop does: build_matrix (Maverick
on sliding 128-token windows) -> floyd_max_product -> get_coref_clusters(threshold 0.85, the
config default) -> replace_to_golden_form. It cannot run on a whole book (an n-by-n matrix over
every mention of the book plus a Floyd pass), so, as decided with the user, it runs on
consecutive segments of 8 probe chunks (~4,000 words); coreference across segments is lost.
Chunks in a segment are joined with a marker line, and the rewritten segment is split back on
it. Two departures from the code, both unavoidable: the Maverick checkpoint is loaded from its
Hugging Face id instead of the author's local path, on a GPU instead of the CPU; and a segment
whose pipeline raises (main.py silently skips it) keeps its original text and is counted.

Resumable per segment: outputs/probe_baselines/b4_lqca/<book_id>/segments.jsonl.
Environment: PBL_LQCA_DIR (the clone), PBL_B4_SEGMENT_CHUNKS (8), PBL_B4_WORKERS (10),
PBL_B4_GPUS (number of visible GPUs, 4).
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common"))
import chunk_map as CM  # noqa: E402
import report_utils as RPT  # noqa: E402

SYSTEM = "b4_lqca"
LQCA_DIR = Path(os.environ.get("PBL_LQCA_DIR", HERE.parents[1] / "third_party" / "LQCA"))
SEGMENT_CHUNKS = int(os.environ.get("PBL_B4_SEGMENT_CHUNKS", "8"))
WORKERS = int(os.environ.get("PBL_B4_WORKERS", "10"))
N_GPUS = int(os.environ.get("PBL_B4_GPUS", "4"))
MAVERICK = os.environ.get("PBL_MAVERICK", "sapienzanlp/maverick-mes-ontonotes")
MARK = "[[[PBL-CHUNK-BREAK]]]"
JOIN = f"\n\n{MARK}\n\n"


def load_lqca(device: str):
    """Import the official main.py as a module (its config parses sys.argv at import time)."""

    os.chdir(LQCA_DIR)  # get_coref_clusters opens pron.txt relative to the working directory
    sys.argv = ["main.py"]
    if str(LQCA_DIR) not in sys.path:
        sys.path.insert(0, str(LQCA_DIR))
    import main as lqca  # noqa: E402  (the official module)
    from maverick import Maverick

    lqca.model = Maverick(hf_name_or_path=MAVERICK, device=device)
    return lqca


def rewrite(lqca, text: str) -> str:
    """main.py's loop body for one text, unchanged."""

    arr, score_graph, link_graph, subseq_graph, token_list, all_tokens, pair_mp = lqca.build_matrix(text)
    # main.py's loop assigns these at module level in its __main__ block; get_coref_clusters reads them as globals.
    lqca.all_tokens, lqca.token_list = all_tokens, token_list
    new_arr = lqca.floyd_max_product(arr)
    coref_list, index_map = lqca.get_coref_clusters(new_arr, lqca.args.thredhold_value)
    return lqca.replace_to_golden_form(coref_list, token_list, all_tokens, pair_mp, index_map)


def work(job: tuple[str, int]) -> dict:
    book_id, gpu = job
    import torch

    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    out_dir = CM.system_dir(SYSTEM) / book_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "segments.jsonl"
    done = {r["segment"] for r in CM.read_jsonl(path)} if path.exists() else set()
    inp = CM.load_inputs(book_id)
    ids, chunks = inp["chunk_ids"], inp["chunks"]
    segments = [list(range(s, min(s + SEGMENT_CHUNKS, len(chunks)))) for s in range(0, len(chunks), SEGMENT_CHUNKS)]
    todo = [i for i in range(len(segments)) if i not in done]
    t_book = time.time()
    if todo:
        lqca = load_lqca(device)
        print(f"B4 {book_id}: {len(segments)} segments of {SEGMENT_CHUNKS} chunks, {len(todo)} to do, Maverick on {device}", flush=True)
        for n, i in enumerate(todo, 1):
            idx = segments[i]
            text = JOIN.join(chunks[j] for j in idx)
            t0 = time.time()
            status, error = "ok", None
            try:
                new = rewrite(lqca, text)
                parts = [p.strip() for p in new.split(MARK)]
                if len(parts) != len(idx):
                    status, parts = "split_failed", [chunks[j] for j in idx]
            except Exception:
                status, error, parts = "pipeline_error", traceback.format_exc(limit=3), [chunks[j] for j in idx]
            row = {
                "segment": i,
                "chunk_ids": [ids[j] for j in idx],
                "rewritten": parts,
                "status": status,
                "error": error,
                "seconds": round(time.time() - t0, 1),
            }
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            if n % 5 == 0:
                print(f"  B4 {book_id}: {n}/{len(todo)} segments, {time.time() - t_book:.0f} s", flush=True)
    rows = CM.read_jsonl(path)
    changed = sum(1 for r in rows for cid, t in zip(r["chunk_ids"], r["rewritten"], strict=True) if t != chunks[ids.index(cid)])
    return {
        "book_id": book_id,
        "segments": len(segments),
        "segments_done": len(rows),
        "ok": sum(r["status"] == "ok" for r in rows),
        "split_failed": sum(r["status"] == "split_failed" for r in rows),
        "pipeline_error": sum(r["status"] == "pipeline_error" for r in rows),
        "chunks_changed": changed,
        "seconds_this_invocation": round(time.time() - t_book, 1),
        "segment_seconds_total": round(sum(r["seconds"] for r in rows), 1),
        "index_chars": sum(len(t) for r in rows for t in r["rewritten"]),
    }


def main() -> int:
    if not (LQCA_DIR / "main.py").exists():
        raise FileNotFoundError(f"The LQCA clone is missing at {LQCA_DIR}; run_all.sh clones it in 00_check.")
    books = CM.book_ids()
    jobs = [(b, i % max(1, N_GPUS)) for i, b in enumerate(books)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=min(WORKERS, len(jobs))) as pool:
        results = pool.map(work, jobs, chunksize=1)
    per_book = {r["book_id"]: r for r in results}
    unexpected = [
        f"{b}: {v['segments_done']} of {v['segments']} segments" for b, v in per_book.items() if v["segments_done"] != v["segments"]
    ]
    errors = sum(v["pipeline_error"] + v["split_failed"] for v in per_book.values())
    if errors:
        unexpected.append(f"{errors} segments kept their original text (pipeline error or split failure); see segments.jsonl")
    CM.write_numbers(
        SYSTEM,
        "build",
        {
            "lqca_dir": str(LQCA_DIR),
            "segment_chunks": SEGMENT_CHUNKS,
            "maverick": MAVERICK,
            "threshold": 0.85,
            "per_book": per_book,
            "unexpected": unexpected,
        },
    )
    frag = f"### B4 LQCA rewrite per book (official code on segments of {SEGMENT_CHUNKS} chunks)\n\n" + RPT.md_table(
        ["book", "segments", "rewritten ok", "split failed", "pipeline error", "chunks changed", "seconds (all segments)"],
        [
            [b, v["segments"], v["ok"], v["split_failed"], v["pipeline_error"], v["chunks_changed"], v["segment_seconds_total"]]
            for b, v in per_book.items()
        ],
    )
    (CM.system_dir(SYSTEM) / "report_build.md").write_text(frag + "\n", encoding="utf-8")
    (CM.system_dir(SYSTEM) / "files_build.txt").write_text(str(CM.system_dir(SYSTEM)) + "/<book>/segments.jsonl\n", encoding="utf-8")
    return 1 if any(v["segments_done"] != v["segments"] for v in per_book.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
