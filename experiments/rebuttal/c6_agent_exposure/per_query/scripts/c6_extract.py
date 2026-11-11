#!/usr/bin/env python3
"""C6 step 2a: extract each agent's per-query exposure from its saved outputs (read-only).

Standard library only, so it runs on any server that holds an agent repository
(gpu03: A-RAG, RAG-R1, Agentic-R, GRIP; gpu01: UMA) without SAADI installed.

For the four Table 2 settings (LooGLE, MuSiQue-32K, QASPER-64K, NovelHopQA) it finds every
saved per-query source of one agent and writes one compact gzip JSONL per source:

    <out>/<method>/<setting>/<source-slug>.jsonl.gz      one line per query
    <out>/<method>/sources.json                          what was read, record counts

Each compact line keeps the query id, the agent's own labels and status, and every
top-level list of chunk ids and every chunk/exposure count in the record. It also keeps
two derived lists:
  - A-RAG: ``preview_union`` = chunks read in full, then every chunk shown as a search
    preview (``trajectory[].candidates[].chunk_id``), in order of first appearance.
  - UMA: ``memory_read_source_union`` and ``memory_read_contextual_union``, the union of
    ``memory_reads[].source_chunk_ids`` (and ``contextual_source_chunk_ids``).
Which list reproduces the paper's exposure is decided later by c6_match.py, against the
paper's own numbers; nothing is chosen here.

It never writes inside --repo, never prints values of keys that look like secrets, and
streams JSONL line by line (some per-query files are 0.8 GB).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import json
import os
import re
import sys
import traceback
from pathlib import Path

SETTINGS = ("loogle", "musique_32k", "qasper_64k", "novelhopqa")
METHODS = ("a_rag", "rag_r1", "agentic_r", "grip", "uma")
SKIP_PARTS = {
    ".git", ".venv", "venv", "__pycache__", "node_modules", "exports", "saadi_snapshot", "saadi_mbr",
    "retrieval_snapshots", "c6_inventory", "c6_per_query", "streams_failed",
}
LABEL_KEYS = ("gold_chunk_ids", "silver_chunk_ids", "silver_chunk_groups", "relevant_chunk_ids", "evidence_groups")
LIST_EXCLUDE = re.compile(r"gold|silver|relevant|evidence_group|candidate_document|document_ids", re.I)
COUNT_RE = re.compile(r"chunk|exposure|read", re.I)
SECRET_RE = re.compile(r"(api[_-]?key|token(?!s?_)|secret|password|passwd|credential|bearer)", re.I)
TOKENS_RE = re.compile(r"tokens", re.I)


def norm_qid(rec: dict) -> str | None:
    q = rec.get("canonical_query_id")
    if q in (None, ""):
        q = rec.get("query_id", rec.get("id"))
    if q in (None, ""):
        return None
    q = str(q)
    return q.split("::", 1)[1] if "::" in q else q


def is_chunk_id_list(v) -> bool:
    return isinstance(v, list) and len(v) > 0 and all(isinstance(x, str) and ":" in x for x in v)


def ordered_union(*seqs) -> list[str]:
    seen, out = set(), []
    for seq in seqs:
        for x in seq or ():
            if isinstance(x, str) and x not in seen:
                seen.add(x)
                out.append(x)
    return out


def compact(rec: dict, method: str) -> dict:
    out: dict = {"query_id": norm_qid(rec)}
    lists: dict[str, list[str]] = {}
    counts: dict[str, int] = {}
    flags: dict[str, bool] = {}
    for k, v in rec.items():
        if SECRET_RE.search(k) and not TOKENS_RE.search(k):
            continue
        if k in LABEL_KEYS:
            continue
        if isinstance(v, bool):
            if k.endswith("hit"):
                flags[k] = v
        elif isinstance(v, int):
            if COUNT_RE.search(k):
                counts[k] = v
        elif is_chunk_id_list(v) and not LIST_EXCLUDE.search(k):
            lists[k] = v
    if method == "a_rag":
        previews = []
        for ev in rec.get("trajectory") or []:
            if isinstance(ev, dict):
                for c in ev.get("candidates") or []:
                    if isinstance(c, dict) and isinstance(c.get("chunk_id"), str):
                        previews.append(c["chunk_id"])
        reads = rec.get("retrieved_chunk_ids") or rec.get("ordered_full_read_chunk_ids") or []
        lists["preview_union"] = ordered_union(reads, previews)
        lists["preview_only"] = [c for c in ordered_union(previews) if c not in set(reads)]
    if method == "uma":
        src, ctx = [], []
        for m in rec.get("memory_reads") or []:
            if isinstance(m, dict):
                src.append(m.get("source_chunk_ids") or [])
                ctx.append(m.get("contextual_source_chunk_ids") or [])
        if src:
            lists["memory_read_source_union"] = ordered_union(*src)
            lists["memory_read_contextual_union"] = ordered_union(*ctx)
    out["lists"] = lists
    out["counts"] = counts
    out["flags"] = flags
    out["labels"] = {k: rec[k] for k in LABEL_KEYS if k in rec}
    for k in ("status", "termination_reason", "error", "doc_id", "document_id", "dataset", "hop"):
        if k in rec and not isinstance(rec[k], (dict, list)):
            out[k] = rec[k]
    return out


def setting_of(rel_parts: tuple[str, ...]) -> str | None:
    for part in rel_parts:
        if part in SETTINGS:
            return part
    return None


def skip(rel_parts: tuple[str, ...]) -> bool:
    return any(p in SKIP_PARTS or p.startswith(".") and p not in (".",) for p in rel_parts)


def find_sources(repo: Path, method: str) -> list[tuple[str, str, list[Path]]]:
    """(setting, slug, files). One source = one per_query.jsonl, or one A-RAG checkpoints/ dir."""

    found: list[tuple[str, str, list[Path]]] = []
    outputs = repo / "outputs"
    if not outputs.is_dir():
        return found
    for dirpath, dirnames, filenames in os.walk(outputs):
        d = Path(dirpath)
        rel = d.relative_to(repo).parts
        dirnames[:] = sorted(x for x in dirnames if not skip(rel + (x,)))
        st = setting_of(rel)
        if st is None:
            continue
        if method == "a_rag":
            if d.name == "checkpoints" and "arag" in rel:
                files = sorted(d / f for f in filenames if f.endswith(".json"))
                if files:
                    found.append((st, "__".join(rel[1:-1]), files))
                dirnames[:] = []
        elif "per_query.jsonl" in filenames:
            found.append((st, "__".join(rel[1:]) + "__per_query", [d / "per_query.jsonl"]))
    return found


def iter_records(files: list[Path]):
    for f in files:
        if f.suffix == ".jsonl":
            with f.open("rb") as fh:
                for n, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line), None
                    except json.JSONDecodeError as exc:
                        yield None, f"{f.name}:{n}: {exc}"
        else:
            try:
                with f.open("rb") as fh:
                    yield json.load(fh), None
            except (json.JSONDecodeError, OSError) as exc:
                yield None, f"{f.name}: {exc}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--repo", required=True, type=Path, help="the agent's repository (read only)")
    ap.add_argument("--out", required=True, type=Path, help="output folder (outside --repo)")
    args = ap.parse_args(argv)
    repo, out = args.repo.resolve(), args.out.resolve()
    if out == repo or repo in out.parents:
        print(f"c6_extract: --out must be outside --repo ({out})", file=sys.stderr)
        return 2
    mdir = out / args.method
    mdir.mkdir(parents=True, exist_ok=True)
    manifest = {"tool": "c6_extract", "version": 1, "method": args.method, "repo": str(repo),
                "host": os.uname().nodename, "started": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                "sources": [], "status": "running"}
    rc = 0
    try:
        sources = find_sources(repo, args.method)
        if not sources:
            raise RuntimeError(f"no per-query sources for {args.method} under {repo}/outputs for {SETTINGS}")
        for setting, slug, files in sources:
            dest = mdir / setting / f"{slug}.jsonl.gz"
            dest.parent.mkdir(parents=True, exist_ok=True)
            n = bad = dup = 0
            seen: set[str] = set()
            errors: list[str] = []
            tmp = dest.with_suffix(".tmp")
            with gzip.open(tmp, "wt", encoding="utf-8") as fh:
                for rec, err in iter_records(files):
                    if err:
                        bad += 1
                        if len(errors) < 5:
                            errors.append(err)
                        continue
                    if not isinstance(rec, dict):
                        bad += 1
                        continue
                    c = compact(rec, args.method)
                    if c["query_id"] is None:
                        bad += 1
                        continue
                    if c["query_id"] in seen:
                        dup += 1
                    seen.add(c["query_id"])
                    fh.write(json.dumps(c, ensure_ascii=False, separators=(",", ":")) + "\n")
                    n += 1
            tmp.replace(dest)
            entry = {"setting": setting, "slug": slug, "file": str(dest.relative_to(out)),
                     "source": str(files[0].parent.relative_to(repo) if len(files) > 1 else files[0].relative_to(repo)),
                     "input_files": len(files), "input_bytes": sum(f.stat().st_size for f in files),
                     "records": n, "distinct_queries": len(seen), "duplicates": dup, "unreadable": bad, "errors": errors}
            manifest["sources"].append(entry)
            print(f"[c6_extract] {args.method} {setting} {slug}: {n} records ({len(seen)} queries, {bad} unreadable)", file=sys.stderr)
        manifest["status"] = "ok"
    except Exception as exc:  # the manifest is still written
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["traceback"] = traceback.format_exc()
        print(manifest["traceback"], file=sys.stderr)
        rc = 1
    manifest["finished"] = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    (mdir / "sources.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"[c6_extract] wrote {mdir}/sources.json (status {manifest['status']})", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
