#!/usr/bin/env python3
"""C6 step 2b: SAADI's full ranking (every chunk of the document) for each Table 2 query.

Table 2 compares SAADI (BM25, published fusion point lambda_M=0.5, lambda_S=0.3, mu=1) at
fixed k. The saved SAADI rankings stop at depth 20, and some agents see more than 20 chunks
for a query (UMA averages ~20 on MuSiQue), so the per-query comparison needs SAADI's ranking
past 20. This script produces it with the repository's own scorer, on CPU:

  1. per-chunk channel scores ("streams") for the setting. Reused from C2
     (outputs/rebuttal/C2_heldout_fusion/streams) or C4 (experiments/rebuttal/c4_subdoc_control/
     work/streams) when present; otherwise computed here with C2's run_stream_job, which
     re-scores BM25 over the saved construction cache (nothing is rebuilt) and passes C2's
     reproduction gate before saving. Computed streams go to <out>/c2_streams/, never into
     C2's or C4's folders.
  2. the full ranking per query = combine_mbr_variant_chunk_scores at the published point
     (heldout_fusion.repo_ranking), i.e. exactly the paper's retrieval, just not cut at 20.
  3. a gate: the first 20 of every ranking must equal the saved Table 2 ranking
     (outputs/saadi/reference/rankings/<id>.jsonl.gz) for every query. Any difference aborts.

Output: <out>/saadi_rankings/<setting>.jsonl.gz ({query_id, doc_id, ranking}) and
<out>/saadi_rankings/<setting>.check.json.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "src"))

SETTINGS = ("loogle", "musique_32k", "qasper_64k", "novelhopqa")
# Table 2's SAADI source per setting (outputs/saadi/reference/manifest.json, study retrieval_depth, bm25, top20).
REFERENCE_DATASET = {"loogle": "loogle", "musique_32k": "musique", "qasper_64k": "qasper", "novelhopqa": "novelhopqa"}


def reference_rankings(setting: str) -> tuple[dict, dict]:
    manifest = json.loads((ROOT / "outputs/saadi/reference/manifest.json").read_text(encoding="utf-8"))
    src = [s for s in manifest["sources"]
           if s.get("study") == "retrieval_depth" and s.get("retriever") == "bm25"
           and s.get("dataset") == REFERENCE_DATASET[setting]]
    if len(src) != 1:
        raise RuntimeError(f"expected one retrieval_depth bm25 source for {setting}, found {len(src)}")
    src = src[0]
    path = ROOT / "outputs/saadi/reference" / src["path"]
    rows = {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            rows[str(r["query_id"])] = r
    return src, rows


def find_streams(setting: str, extra: list[Path]) -> tuple[Path, Path] | None:
    job = f"{setting}__bm25"
    dirs = [*extra, ROOT / "outputs/rebuttal/C2_heldout_fusion/streams",
            ROOT / "experiments/rebuttal/c4_subdoc_control/work/streams"]
    for d in dirs:
        p, q = d / f"{job}.parquet", d / f"{job}.queries.json"
        if p.exists() and q.exists():
            return p, q
    return None


def compute_streams(setting: str, out: Path, log) -> tuple[Path, Path]:
    from saadi.rebuttal.heldout_fusion import C2Paths, run_stream_job

    base = out / "c2_streams"
    paths = C2Paths(ROOT, out_base=base, results_base=base / "results", exp_base=base / "exp")
    summary = run_stream_job(setting, "bm25", paths=paths, log=log)
    if summary.get("status") not in ("DONE", None) and not summary.get("skipped"):
        raise RuntimeError(f"stream job {setting} failed its reproduction gate: {json.dumps(summary.get('check'))[:2000]}")
    return paths.stream_path(f"{setting}__bm25"), paths.queries_path(f"{setting}__bm25")


def run_setting(setting: str, out: Path, extra: list[Path], compute: bool, log) -> dict:
    from saadi.rebuttal.heldout_fusion import PUBLISHED_POINT, load_query_arrays, repo_ranking

    t0 = time.perf_counter()
    src, ref = reference_rankings(setting)
    found = find_streams(setting, extra)
    if found is None:
        if not compute:
            raise FileNotFoundError(f"no BM25 streams for {setting}; rerun with --compute-missing")
        log(f"[c6] {setting}: no saved streams, computing them (BM25 on CPU, saved construction cache)")
        found = compute_streams(setting, out, log)
    stream, queries = found
    log(f"[c6] {setting}: streams {stream}")
    qs = load_query_arrays(stream, queries, "bm25")
    dest_dir = out / "saadi_rankings"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{setting}.jsonl.gz"
    tmp = dest.with_suffix(".tmp")
    same = mismatched = 0
    examples = []
    written = set()
    pools = []
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        for q in qs:
            if q.query_id not in ref:
                continue  # the stream holds a few queries outside the paper population
            ranking = repo_ranking(q, "bm25", PUBLISHED_POINT)
            if ranking[:20] == list(ref[q.query_id]["retrieved_chunk_ids"]):
                same += 1
            else:
                mismatched += 1
                if len(examples) < 5:
                    examples.append({"query_id": q.query_id, "ours": ranking[:20], "paper": ref[q.query_id]["retrieved_chunk_ids"]})
            pools.append(len(ranking))
            written.add(q.query_id)
            fh.write(json.dumps({"query_id": q.query_id, "doc_id": q.doc_id, "ranking": ranking}, separators=(",", ":")) + "\n")
    missing = sorted(set(ref) - written)
    check = {
        "setting": setting, "reference_source_id": src["source_id"], "reference_queries": len(ref),
        "streams": str(stream), "published_point": list(PUBLISHED_POINT),
        "rankings_written": len(written), "top20_identical": same, "top20_different": mismatched,
        "missing_queries": len(missing), "missing_examples": missing[:10], "mismatch_examples": examples,
        "pool_min": min(pools) if pools else None, "pool_max": max(pools) if pools else None,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    check["passed"] = mismatched == 0 and not missing
    (dest_dir / f"{setting}.check.json").write_text(json.dumps(check, indent=1), encoding="utf-8")
    if not check["passed"]:
        raise RuntimeError(f"{setting}: full ranking does not reproduce Table 2 (top-20 different {mismatched}, missing {len(missing)}); see {setting}.check.json")
    tmp.replace(dest)
    log(f"[c6] {setting}: {len(written)} rankings, top-20 identical for all, pool {check['pool_min']}-{check['pool_max']} chunks, {check['seconds']} s")
    return check


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/rebuttal/c6_agent_exposure/per_query")
    ap.add_argument("--settings", default=",".join(SETTINGS))
    ap.add_argument("--streams-dir", type=Path, action="append", default=[], help="extra folder to look for <setting>__bm25 streams in (first)")
    ap.add_argument("--compute-missing", action="store_true", help="compute streams that are not saved anywhere (CPU, can take hours for NovelHopQA)")
    args = ap.parse_args(argv)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rc, results = 0, []
    for setting in [s.strip() for s in args.settings.split(",") if s.strip()]:
        if setting not in SETTINGS:
            print(f"unknown setting {setting}", file=sys.stderr)
            return 2
        try:
            results.append(run_setting(setting, out, args.streams_dir, args.compute_missing, lambda m: print(m, file=sys.stderr, flush=True)))
        except Exception as exc:
            rc = 1
            results.append({"setting": setting, "passed": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
            print(traceback.format_exc(), file=sys.stderr)
    (out / "saadi_rankings").mkdir(parents=True, exist_ok=True)
    (out / "saadi_rankings" / "summary.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"[c6] SAADI rankings: {sum(r.get('passed', False) for r in results)}/{len(results)} settings passed", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
