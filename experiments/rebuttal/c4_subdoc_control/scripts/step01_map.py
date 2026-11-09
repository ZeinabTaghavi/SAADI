#!/usr/bin/env python3
"""C4 step 01: map every chunk to its sub-document(s); find and check each query's gold sub-document.

How sub-documents are identified. The concatenated settings are built by
src/saadi/context_expansion.py (_format_group_context): each source document is written as

    [Document i/N | doc_id=<source id>]
    <source text>

and the blocks are joined with blank lines. group_membership.json gives the order of the
N source documents of a group. A group whose text exceeded the token window was trimmed
per target document (``__variant_<hash>`` ids, QASPER-64K): the text may then start or
end inside a sub-document, and the leading part without a header belongs to the member
just before the first header.

Two independent methods, which must agree:
  A. alignment: every chunk's raw text (steps_index.jsonl, in step order) is located in
     the whitespace-normalised document text; a chunk belongs to every sub-document whose
     span it overlaps with non-whitespace text;
  B. header walk: a chunk belongs to the sub-document open when it starts (if it has text
     before its first header) and to every sub-document whose header it contains.
A chunk that crosses a boundary belongs to both sub-documents.

Gold sub-document: qa.json's ``source_document_id`` (the source document the question was
written for). Every evidence chunk (sufficient chunks and evidence-group chunks, from the
paper's saved rankings) must belong to it; otherwise the query is left out of C4.

Reads only; writes results/rebuttal/c4_subdoc_control/subdoc_map/<setting>.json and the
step report.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from c4_common import (
    MAP_DIR,
    ROOT,
    StepReport,
    Setting,
    dist,
    evidence_from_row,
    fnum,
    load_ranking_rows,
    paper_sources,
    qid_sort_key,
    read_json,
    rel,
    run_step,
    selected_retrievers,
    selected_settings,
    write_json,
)

from saadi.io.hierarchy_cache import _resolve_cache_file_path  # noqa: E402
from saadi.io.jsonl import read_jsonl  # noqa: E402
from saadi.rebuttal.frozen_records import DATASETS as E2_DATASETS  # noqa: E402

HEADER_RE = re.compile(r"\[Document (\d+)/(\d+) \| doc_id=([^\]]+)\]")
_WS = re.compile(r"\s+")
SMALL_K = (1, 3, 5, 10)


def norm(text: str) -> str:
    return _WS.sub(" ", str(text or "")).strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _dedupe(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ---------------------------------------------------------------------------
# One document
# ---------------------------------------------------------------------------


def map_document(doc_id: str, text: str, members: list[str], chunks: list[tuple[str, int, str]]) -> dict[str, Any]:
    """chunks: (chunk_id, step, raw_chunk_text) in step order."""

    problems: list[str] = []
    D = norm(text)
    headers = [(m.start(), int(m.group(1)), int(m.group(2)), m.group(3).strip()) for m in HEADER_RE.finditer(D)]
    total = len(members)
    if not headers:
        problems.append(f"{doc_id}: no [Document i/N] header in the text")
    for _pos, i, n, sid in headers:
        if n != total or not (1 <= i <= total) or members[i - 1] != sid:
            problems.append(f"{doc_id}: header [Document {i}/{n} | doc_id={sid}] does not match group_membership.json (N={total})")
            break
    idx = [h[1] for h in headers]
    if idx and idx != list(range(idx[0], idx[0] + len(idx))):
        problems.append(f"{doc_id}: header indices are not contiguous ({idx[:5]}...)")

    leading: str | None = None
    if headers and D[: headers[0][0]].strip():
        if headers[0][1] >= 2:
            leading = members[headers[0][1] - 2]
        else:
            problems.append(f"{doc_id}: text before the [Document 1/N] header")
    trimmed_end = bool(headers) and headers[-1][1] < total
    regions: list[tuple[int, int, str]] = []
    if leading is not None:
        regions.append((0, headers[0][0], leading))
    for k, (pos, _i, _n, sid) in enumerate(headers):
        end = headers[k + 1][0] if k + 1 < len(headers) else len(D)
        regions.append((pos, end, sid))
    present = [r[2] for r in regions]
    missing = [m for m in members if m not in set(present)]
    if len(set(present)) != len(present):
        problems.append(f"{doc_id}: a sub-document appears twice in the text")

    out_chunks: dict[str, list[str]] = {}
    unaligned: list[str] = []
    disagree: list[dict[str, Any]] = []
    walk_seq: list[str] = []
    current = leading
    cursor = 0
    covered = 0
    out_of_order = 0
    for cid, _step, raw in chunks:
        C = norm(raw)
        # B. header walk
        hs = list(HEADER_RE.finditer(C))
        walk: list[str] = []
        if not hs:
            if current is not None:
                walk.append(current)
        else:
            if C[: hs[0].start()].strip() and current is not None:
                walk.append(current)
            for m in hs:
                walk.append(m.group(3).strip())
                walk_seq.append(m.group(3).strip())
            current = hs[-1].group(3).strip()
        walk = _dedupe(walk)
        # A. alignment
        aligned: list[str] | None = None
        pos = D.find(C, cursor) if C else -1
        if pos < 0 and C:
            pos = D.find(C)
            if pos >= 0:
                out_of_order += 1
        if pos >= 0:
            s, e = pos, pos + len(C)
            cursor = max(cursor, e)
            covered += len(C)
            aligned = [rid for (rs, re_, rid) in regions if rs < e and s < re_ and D[max(rs, s) : min(re_, e)].strip()]
        else:
            unaligned.append(cid)
        if aligned is not None and aligned != walk:
            disagree.append({"chunk_id": cid, "alignment": aligned, "header_walk": walk})
        final = aligned if aligned is not None else walk
        if not final:
            problems.append(f"{doc_id}: chunk {cid} could not be given a sub-document")
        out_chunks[cid] = final
    header_ids = [h[3] for h in headers]
    if walk_seq != header_ids:
        # A header split across two chunks; the alignment (A) is still exact then.
        disagree.append({"chunk_id": None, "note": "header sequence seen in chunks != headers in the text"})
    if unaligned and disagree:
        problems.append(f"{doc_id}: {len(unaligned)} chunk(s) not found in the text and the header walk is not reliable")

    per_sub = Counter(s for m in out_chunks.values() for s in m)
    boundary = [cid for cid, m in out_chunks.items() if len(m) > 1]
    return {
        "doc_id": doc_id,
        "members": members,
        "subdocs": present,
        "leading_partial": leading,
        "trimmed_end": trimmed_end,
        "missing_members": missing,
        "n_chunks": len(chunks),
        "chunks": out_chunks,
        "chunks_per_subdoc": dict(per_sub),
        "boundary_chunks": boundary,
        "unaligned": unaligned,
        "out_of_order": out_of_order,
        "disagree": disagree,
        "coverage": covered / len(D) if D else 0.0,
        "problems": problems,
    }


def load_doc_chunks(index_path: Path, doc_entry: dict[str, Any]) -> list[tuple[str, int, str]]:
    steps_ref = (doc_entry.get("files", {}) or {}).get("steps_index")
    if not steps_ref:
        raise RuntimeError(f"{doc_entry.get('doc_id')}: cache index has no steps_index file")
    path = _resolve_cache_file_path(steps_ref, base_dir=index_path.parent)
    rows = [r for r in read_jsonl(Path(path)) if isinstance(r, dict)]
    out = []
    for r in rows:
        out.append((str(r["chunk_id"]), int(r.get("step", r.get("chunk_index", -1))), str(r.get("raw_chunk_text", ""))))
    out.sort(key=lambda t: (t[1], t[0]))
    ids = [c for c, _, _ in out]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"{doc_entry.get('doc_id')}: duplicate chunk ids in {rel(path)}")
    return out


# ---------------------------------------------------------------------------
# One setting
# ---------------------------------------------------------------------------


def map_setting(setting: Setting, retrievers: list[str], sources: dict[tuple[str, str], list[dict[str, Any]]]) -> dict[str, Any]:
    t0 = time.time()
    res: dict[str, Any] = {"setting": setting.key, "display": setting.display, "problems": [], "warnings": []}
    P = res["problems"]
    prep = setting.prepared_dir
    documents = read_json(prep / "documents.json")
    membership = read_json(prep / "group_membership.json")
    qa = read_json(prep / "qa.json")
    meta = read_json(prep / "context_expansion_metadata.json")
    res["metadata"] = {k: meta.get(k) for k in ("target_context_tokens", "tolerance_tokens", "expansion_seed", "prepared_documents", "groups_with_trimmed_variants", "token_counter", "prepared_token_lengths")}

    # The index must have been built from exactly these prepared files.
    binding_path = setting.namespace / "prepared_dataset_binding.json"
    if binding_path.exists():
        binding = read_json(binding_path)
        for key, fname in (("documents_sha256", "documents.json"), ("qa_sha256", "qa.json")):
            if binding.get(key) and binding[key] != sha256_file(prep / fname):
                P.append(f"{fname} sha256 differs from the index's prepared_dataset_binding.json")
    else:
        res["warnings"].append("no prepared_dataset_binding.json next to index.json (hash check skipped)")
    # Guard the namespace choice against E2/C2's registry on the two Table-1 settings.
    if setting.key in E2_DATASETS and (ROOT / E2_DATASETS[setting.key].namespace).resolve() != setting.namespace.resolve():
        P.append(f"namespace differs from E2's registry: {E2_DATASETS[setting.key].namespace}")

    index_path = setting.index_json
    payload = read_json(index_path)
    docs_out: dict[str, Any] = {}
    chunk_map: dict[str, list[str]] = {}
    for entry in payload.get("docs", []):
        doc_id = str(entry.get("doc_id"))
        if doc_id not in documents or doc_id not in membership:
            P.append(f"{doc_id}: in index.json but not in documents.json / group_membership.json")
            continue
        chunks = load_doc_chunks(index_path, entry)
        d = map_document(doc_id, documents[doc_id], [str(m) for m in membership[doc_id]], chunks)
        P.extend(d.pop("problems"))
        for cid, m in d["chunks"].items():
            if cid in chunk_map:
                P.append(f"chunk id {cid} appears in two documents")
            chunk_map[cid] = m
        docs_out[doc_id] = d

    # ---- queries: the union of the paper's query sets over the retrievers
    qa_by_id = {str(e["id"]): e for e in qa}
    per_ret: dict[str, Any] = {}
    evidence: dict[str, tuple[list[str], list[list[str]]]] = {}
    in_sets: dict[str, list[str]] = {}
    ev_conflicts: list[str] = []
    for r in retrievers:
        srcs = sources.get((setting.key, r), [])
        if not srcs:
            per_ret[r] = {"paper_sources": 0}
            continue
        per_ret[r] = {"paper_sources": len(srcs), "sources": []}
        qsets = []
        for src in srcs:
            rows = load_ranking_rows(src)
            qids = [str(row["query_id"]) for row in rows]
            qsets.append(set(qids))
            per_ret[r]["sources"].append(
                {"bundle": src["bundle"], "study": src["study"], "source_id": src["source_id"], "n_queries": len(qids), "expected": src["expected"]}
            )
            for row in rows:
                qid = str(row["query_id"])
                g, groups = evidence_from_row(row)
                key = (sorted(g), sorted(sorted(x) for x in groups))
                if qid in evidence:
                    prev = evidence[qid]
                    if (sorted(prev[0]), sorted(sorted(x) for x in prev[1])) != key:
                        ev_conflicts.append(f"{qid} ({r}, {src['source_id']})")
                else:
                    evidence[qid] = (g, groups)
                in_sets.setdefault(qid, [])
                if r not in in_sets[qid]:
                    in_sets[qid].append(r)
        if len({frozenset(s) for s in qsets}) > 1:
            P.append(f"{r}: the paper's sources for {setting.key} hold different query sets")
        per_ret[r]["n_queries"] = len(qsets[0]) if qsets else 0
    if ev_conflicts:
        P.append(f"evidence labels differ between saved rankings for {len(ev_conflicts)} queries (first: {ev_conflicts[:3]})")

    queries: dict[str, Any] = {}
    status_counts: Counter[str] = Counter()
    excluded_examples: list[dict[str, Any]] = []
    cross_check_disagree = 0
    for qid in sorted(evidence, key=qid_sort_key):
        gold, groups = evidence[qid]
        e = qa_by_id.get(qid)
        if e is None:
            P.append(f"query {qid}: not in qa.json")
            continue
        doc_id = str(e["document_id"])
        gold_sub = str(e.get("source_document_id", ""))
        ev = _dedupe(gold + [c for g in groups for c in g])
        eligible = bool(gold or groups)
        doc = docs_out.get(doc_id)
        q: dict[str, Any] = {"doc_id": doc_id, "gold_subdoc": gold_sub, "eligible": eligible, "retrievers": in_sets.get(qid, [])}
        if doc is None:
            P.append(f"query {qid}: document {doc_id} not mapped")
            continue
        bad_prefix = [c for c in ev if not c.startswith(doc_id + ":")]
        unknown = [c for c in ev if c not in chunk_map]
        if bad_prefix or unknown:
            P.append(f"query {qid}: evidence chunk(s) outside its document or unknown: {(bad_prefix + unknown)[:3]}")
            continue
        outside = [{"chunk_id": c, "subdocs": chunk_map[c]} for c in ev if gold_sub not in chunk_map[c]]
        common = set(chunk_map[ev[0]]) if ev else set()
        for c in ev[1:]:
            common &= set(chunk_map[c])
        if ev and gold_sub not in common and len(common) >= 1:
            cross_check_disagree += 1
        q["n_gold_chunks"] = int(doc["chunks_per_subdoc"].get(gold_sub, 0))
        q["evidence_chunks"] = ev
        if not eligible:
            status = "ineligible"
        elif gold_sub not in doc["subdocs"]:
            status = "excluded_gold_subdoc_not_in_text"
        elif outside:
            status = "excluded_evidence_outside"
        else:
            status = "kept"
        q["status"] = status
        if outside:
            q["evidence_outside"] = outside
        if status.startswith("excluded") and len(excluded_examples) < 5:
            excluded_examples.append({"query_id": qid, "doc_id": doc_id, "gold_subdoc": gold_sub, "status": status, "evidence_outside": outside[:3], "question": str(e.get("question", ""))[:160]})
        status_counts[status] += 1
        queries[qid] = q

    # ---- statistics
    n_chunks = len(chunk_map)
    n_boundary = sum(1 for m in chunk_map.values() if len(m) > 1)
    subs_per_doc = [len(d["subdocs"]) for d in docs_out.values()]
    distinct_subs = {s for d in docs_out.values() for s in d["subdocs"]}
    all_sizes = [n for d in docs_out.values() for n in d["chunks_per_subdoc"].values()]
    kept = {qid: q for qid, q in queries.items() if q["status"] == "kept"}
    gold_pairs = {(q["doc_id"], q["gold_subdoc"]): q["n_gold_chunks"] for q in kept.values()}
    q_sizes = [q["n_gold_chunks"] for q in kept.values()]
    res["stats"] = {
        "documents": len(docs_out),
        "subdocs_distinct": len(distinct_subs),
        "subdocs_per_document": dist(subs_per_doc),
        "chunks": n_chunks,
        "boundary_chunks": n_boundary,
        "boundary_share": n_boundary / n_chunks if n_chunks else None,
        "boundary_per_document": {d: len(v["boundary_chunks"]) for d, v in docs_out.items()},
        "chunks_per_subdoc_all": dist(all_sizes),
        "leading_partial_docs": sum(1 for d in docs_out.values() if d["leading_partial"]),
        "trimmed_end_docs": sum(1 for d in docs_out.values() if d["trimmed_end"]),
        "missing_members": sum(len(d["missing_members"]) for d in docs_out.values()),
        "unaligned_chunks": sum(len(d["unaligned"]) for d in docs_out.values()),
        "out_of_order_chunks": sum(d["out_of_order"] for d in docs_out.values()),
        "method_disagreements": sum(len([x for x in d["disagree"] if x.get("chunk_id")]) for d in docs_out.values()),
        "header_sequence_breaks": sum(len([x for x in d["disagree"] if not x.get("chunk_id")]) for d in docs_out.values()),
        "min_text_coverage": min((d["coverage"] for d in docs_out.values()), default=None),
        "queries_union": len(queries),
        "status_counts": dict(status_counts),
        "evidence_common_subdoc_not_gold": cross_check_disagree,
        "gold_subdocs_distinct": len(gold_pairs),
        "chunks_per_gold_subdoc": dist(gold_pairs.values()),
        "chunks_per_gold_subdoc_over_queries": dist(q_sizes),
        "kept_with_gold_subdoc_at_most_k": {str(k): sum(1 for n in q_sizes if n <= k) for k in SMALL_K},
    }
    res["per_retriever"] = per_ret
    res["excluded_examples"] = excluded_examples
    res["disagreement_examples"] = [x for d in docs_out.values() for x in d["disagree"]][:5]
    res["seconds"] = time.time() - t0

    # ---- the map file (one per setting)
    map_payload = {
        "setting": setting.key,
        "display": setting.display,
        "ok": not P,
        "source": {
            "documents_json": rel(prep / "documents.json"),
            "group_membership_json": rel(prep / "group_membership.json"),
            "qa_json": rel(prep / "qa.json"),
            "index_json": rel(index_path),
        },
        "rule": "chunk -> every sub-document whose span it overlaps with non-whitespace text (boundary chunks belong to both)",
        "documents": {
            d: {k: v[k] for k in ("members", "subdocs", "leading_partial", "trimmed_end", "missing_members", "n_chunks", "chunks_per_subdoc", "boundary_chunks")}
            for d, v in docs_out.items()
        },
        "chunks": chunk_map,
        "queries": queries,
    }
    out_path = MAP_DIR / f"{setting.key}.json"
    write_json(out_path, map_payload)
    res["map_file"] = rel(out_path)
    return res


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def body(report: StepReport, args: argparse.Namespace) -> None:
    settings = selected_settings(args.settings)
    retrievers = selected_retrievers(args.retrievers)
    sources = paper_sources()
    results: dict[str, Any] = {}
    for s in settings:
        print(f"[c4] step 01: {s.key} ...", flush=True)
        try:
            results[s.key] = map_setting(s, retrievers, sources)
        except Exception as exc:
            import traceback

            results[s.key] = {"setting": s.key, "display": s.display, "problems": [f"{type(exc).__name__}: {exc}"], "traceback": traceback.format_exc()}
        for p in results[s.key]["problems"]:
            report.fail(f"{s.key}: {p}")
        for w in results[s.key].get("warnings", []):
            report.warn(f"{s.key}: {w}")
        if results[s.key].get("map_file"):
            report.wrote(ROOT / results[s.key]["map_file"])
        print(f"[c4] step 01: {s.key} done ({len(results[s.key]['problems'])} problems)", flush=True)
    report.data["settings"] = results
    ok = {k: v for k, v in results.items() if "stats" in v}

    report.p("Every chunk of every concatenated setting gets the id(s) of the source document(s) it comes from, "
             "from the `[Document i/N | doc_id=...]` headers that `context_expansion.py` writes before each source "
             "document. Two methods (text alignment and a header walk) must agree. A chunk that crosses a boundary "
             "belongs to both sub-documents. The gold sub-document of a query is qa.json's `source_document_id`.")

    report.h("1. Settings and query sets of the paper")
    rows = []
    for s in settings:
        v = results[s.key]
        for r in retrievers:
            pr = (v.get("per_retriever") or {}).get(r, {})
            if not pr.get("paper_sources"):
                continue
            src = pr["sources"][0]
            ex = src["expected"].get(10, {})
            rows.append([s.display, r, "; ".join(f"{x['bundle']}/{x['study']} {x['source_id'][:8]}" for x in pr["sources"]), pr.get("n_queries"), ex.get("eligible", "-")])
    report.table(["Setting", "Retriever", "Saved ranking(s) of the paper", "Queries", "Eligible (paper)"], rows)
    missing = [f"{s.display}/{r}" for s in settings for r in retrievers if not (results[s.key].get("per_retriever") or {}).get(r, {}).get("paper_sources")]
    if missing:
        report.p(f"No saved paper ranking (the paper does not report these pairs): {', '.join(missing)}.")

    report.h("2. Sub-documents and chunks")
    rows = []
    for k, v in ok.items():
        st = v["stats"]
        spd = st["subdocs_per_document"]
        rows.append([
            v["display"], st["documents"], st["subdocs_distinct"], f"{fnum(spd['min'],0)} / {fnum(spd['median'],1)} / {fnum(spd['max'],0)}",
            st["chunks"], f"{st['boundary_chunks']} ({100 * (st['boundary_share'] or 0):.1f}%)",
            st["leading_partial_docs"], st["trimmed_end_docs"], st["unaligned_chunks"], st["method_disagreements"], st["header_sequence_breaks"],
            fnum(st["min_text_coverage"], 3),
        ])
    report.table(["Setting", "Indexed documents", "Sub-documents (distinct)", "Sub-docs per document min/med/max", "Chunks", "Boundary chunks",
                  "Docs starting mid-sub-doc", "Docs ending mid-sub-doc", "Chunks not found in text", "A vs B disagreements", "Header split across chunks", "Min text coverage"], rows)
    report.p("Boundary chunks belong to two sub-documents. \"Docs starting / ending mid-sub-doc\" are trimmed variants; "
             "their partial sub-document is still named from group_membership.json. Text coverage = characters of the "
             "document text matched by its chunks (whitespace-normalised).")

    report.h("3. Gold sub-document checks")
    rows = []
    for k, v in ok.items():
        st = v["stats"]
        sc = st["status_counts"]
        rows.append([v["display"], st["queries_union"], sc.get("ineligible", 0), sc.get("excluded_evidence_outside", 0),
                     sc.get("excluded_gold_subdoc_not_in_text", 0), sc.get("kept", 0), st["evidence_common_subdoc_not_gold"]])
    report.table(["Setting", "Queries (union over retrievers)", "Ineligible (no evidence)", "Excluded: evidence outside gold sub-doc",
                  "Excluded: gold sub-doc not in text", "Kept for C4", "Evidence shares another sub-doc, not the gold one"], rows)
    ex_rows = [[v["display"], e["query_id"], e["doc_id"], e["gold_subdoc"], e["status"],
                "; ".join(f"{o['chunk_id']} -> {','.join(o['subdocs'])}" for o in e["evidence_outside"])] for v in ok.values() for e in v["excluded_examples"]]
    if ex_rows:
        report.p("First excluded queries per setting (at most 5 each):")
        report.p("")
        report.table(["Setting", "Query", "Document", "Gold sub-doc", "Status", "Evidence chunk -> its sub-doc(s)"], ex_rows[:20])

    report.h("4. Chunks per gold sub-document (kept queries)")
    rows = []
    for k, v in ok.items():
        st = v["stats"]
        g = st["chunks_per_gold_subdoc"]
        gq = st["chunks_per_gold_subdoc_over_queries"]
        a = st["chunks_per_subdoc_all"]
        kept_n = st["status_counts"].get("kept", 0) or 1
        small = st["kept_with_gold_subdoc_at_most_k"]
        rows.append([v["display"], st["gold_subdocs_distinct"], f"{fnum(g['min'],0)} / {fnum(g['median'],1)} / {fnum(g['max'],0)}",
                     f"{fnum(gq['min'],0)} / {fnum(gq['median'],1)} / {fnum(gq['max'],0)}",
                     f"{fnum(a['min'],0)} / {fnum(a['median'],1)} / {fnum(a['max'],0)}"]
                    + [f"{small[str(kk)]} ({100 * small[str(kk)] / kept_n:.0f}%)" for kk in SMALL_K])
    report.table(["Setting", "Gold sub-docs (distinct)", "Chunks per gold sub-doc min/med/max", "Same, over queries", "All sub-docs min/med/max"]
                 + [f"Queries with <= {kk} chunks" for kk in SMALL_K], rows)
    report.p("HR@k inside the sub-document is trivially 100% for queries whose gold sub-document has k chunks or fewer; "
             "later steps report each HR@k on all kept queries and on those with more than k chunks.")

    dis = [(v["display"], x) for v in ok.values() for x in v.get("disagreement_examples", [])]
    if dis:
        report.h("5. Method disagreements (first 5)")
        report.code("\n".join(f"{d}: {x}" for d, x in dis[:5]))
    for k, v in results.items():
        if v.get("traceback"):
            report.h(f"Traceback ({v['display']})")
            report.code(v["traceback"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--settings", default=None, help="comma list (default: $C4_SETTINGS)")
    ap.add_argument("--retrievers", default=None, help="comma list (default: $C4_RETRIEVERS)")
    args = ap.parse_args()
    cmd = "bash experiments/rebuttal/c4_subdoc_control/sh/01_map.sh" + ("" if len(sys.argv) == 1 else " " + " ".join(sys.argv[1:]))
    report = StepReport(1, "chunk to sub-document map and gold checks", command=cmd)
    return run_step(report, lambda r: body(r, args))


if __name__ == "__main__":
    raise SystemExit(main())
