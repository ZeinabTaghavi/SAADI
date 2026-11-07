#!/usr/bin/env python3
"""E2b "bridge queries": why did frozen match live in E2?

For every (query, evidence chunk) pair, split the query tokens by where the
chunk can match them: in its own text, in the frozen records of its linked
names, or only in a record field that arrived at a LATER step (a "bridge").

Run from the repository root:

    .venv/bin/python scripts/rebuttal/e2b_bridge_queries.py

Reuses the E2 machinery and outputs/rebuttal/E2_frozen_records/per_query.parquet.
No LLM calls, no GPU, no network. BM25 only. Definitions are fixed in this file
and are not adjusted after seeing results.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from saadi.indexing.retriever_adapters.bm25 import _tokenize  # noqa: E402
from saadi.indexing.retrieval_methods.mbr import (  # noqa: E402
    MBR_VARIANT_MBR2,
    _find_doc_entry,
    _load_doc_records,
    _read_index_payload,
    build_mbr_document_memory_index,
    normalize_signal_scores,
    score_mbr_variant_document_pools,
)
from saadi.rebuttal.frozen_records import (  # noqa: E402
    DATASETS,
    PUBLISHED_SETTINGS,
    _frozen_streams,
    frozen_chunk_fields,
    load_saved_live_rows,
    paired_bootstrap_ci,
    replay_first_added,
)

EXPERIMENT_ID = "E2b_bridge_queries"
RARE_DF_SHARE = 0.05  # a token is rare in a document when it appears in <5% of its chunks
MIN_GROUP = 30  # groups smaller than this are reported but flagged as too small

# Standard English stopword list (fixed before running; matched after the same
# normalization the repo's BM25 tokenizer applies).
_STOPWORDS_RAW = """
a about above after again against all am an and any are aren as at be because
been before being below between both but by can cannot could couldn did didn
do does doesn doing don down during each few for from further had hadn has hasn
have haven having he her here hers herself him himself his how i if in into is
isn it its itself just me more most mustn my myself no nor not now of off on
once only or other our ours ourselves out over own same shan she should shouldn
so some such than that the their theirs them themselves then there these they
this those through to too under until up very was wasn we were weren what when
where which while who whom why will with won would wouldn you your yours
yourself yourselves
""".split()
STOPWORDS = {w for raw in _STOPWORDS_RAW for w in _tokenize(raw)} | set(_STOPWORDS_RAW)


def query_tokens(text: str) -> set[str]:
    return {t for t in _tokenize(text) if t not in STOPWORDS}


def tokens_of(values: list[str]) -> set[str]:
    out: set[str] = set()
    for value in values:
        out.update(_tokenize(value))
    return out


def memory_channel_ranks(streams, document_index) -> dict[str, int]:
    """1-based rank of every chunk on the memory channel alone (MBR2, z-scored)."""

    recs = streams.chunk_scores
    norm = {
        kind: normalize_signal_scores(
            {cid: getattr(r, f"{kind}_score") for cid, r in recs.items() if getattr(r, f"{kind}_score") is not None},
            method=PUBLISHED_SETTINGS["normalization_method"],
            retriever_name="bm25",
        )
        for kind in ("name", "alias", "description")
    }
    memory: dict[str, float] = {}
    for cid in recs:
        parts = [norm[kind][cid] for kind in ("name", "alias", "description") if cid in norm[kind]]
        if parts:
            memory[cid] = sum(parts) / len(parts)
    order = sorted(
        document_index.chunk_pool,
        key=lambda ch: (-(memory.get(ch.chunk_id, float("-inf"))), int(ch.step or 0), str(ch.chunk_id)),
    )
    return {ch.chunk_id: i + 1 for i, ch in enumerate(order)}


def analyze_dataset(key: str, *, retriever: str = "bm25") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per-query frame, bridge-token frame) for one dataset."""

    spec = DATASETS[key]
    index_path = REPO_ROOT / spec.index_json
    payload = _read_index_payload(index_path)
    saved_rows = load_saved_live_rows(spec, retriever, repo_root=REPO_ROOT)
    rows_by_doc: dict[str, list[dict]] = {}
    for row in saved_rows:
        rows_by_doc.setdefault(str(row["doc_id"]), []).append(row)

    try:
        from tqdm import tqdm

        doc_iter = tqdm(list(rows_by_doc), desc=f"{key}/{retriever}", unit="doc")
    except Exception:
        doc_iter = list(rows_by_doc)

    query_rows: list[dict] = []
    token_rows: list[dict] = []
    for doc_id in doc_iter:
        doc_entry = _find_doc_entry(payload, doc_id=doc_id)
        records = _load_doc_records(index_path=index_path, index_payload=payload, doc_entry=doc_entry, doc_id=doc_id)
        replay = replay_first_added(doc_id, records)
        dindex = build_mbr_document_memory_index(index_path, doc_id=doc_id)
        frozen = frozen_chunk_fields(dindex, replay)

        chunk_text_tokens = {ch.chunk_id: tokens_of([ch.text]) for ch in dindex.chunk_pool}
        n_chunks = len(dindex.chunk_pool)
        doc_freq: dict[str, int] = defaultdict(int)
        for toks in chunk_text_tokens.values():
            for t in toks:
                doc_freq[t] += 1

        def is_rare(token: str) -> bool:
            return doc_freq.get(token, 0) < RARE_DF_SHARE * n_chunks

        frozen_tokens: dict[str, set[str]] = {}
        live_tokens: dict[str, set[str]] = {}
        late_prov: dict[str, dict[str, tuple[str, str, str, int]]] = {}
        for cid, fc in frozen.items():
            frozen_tokens[cid] = tokens_of(fc.linked_names + fc.frozen_aliases + fc.frozen_descriptions)
            live_tokens[cid] = tokens_of(fc.linked_names + fc.live_aliases + fc.live_descriptions)
            prov: dict[str, tuple[str, str, str, int]] = {}
            for f in sorted(fc.late_fields, key=lambda f: (f.added_at_step, f.kind, f.value)):
                for t in tokens_of([f.value]):
                    prov.setdefault(t, (f.entity_name, f.kind, f.value, f.added_at_step))
            late_prov[cid] = prov

        for row in rows_by_doc[doc_id]:
            qid = str(row["query_id"])
            question = str(row.get("question", "") or "")
            q_tok = query_tokens(question)
            gold = [str(x) for x in row.get("gold_chunk_ids", []) or []]
            groups = row.get("silver_chunk_groups", []) or []
            evidence = sorted({*gold, *(str(c) for g in groups if isinstance(g, list) for c in g)})
            evidence = [c for c in evidence if c in frozen]

            overlaps: list[float] = []
            overlaps_rare: list[float] = []
            memory_all = memory_rare = bridge_all = bridge_rare = alias_bridge_all = alias_bridge_rare = False
            bridge_evidence: list[str] = []
            for cid in evidence:
                t_tok = chunk_text_tokens[cid]
                if q_tok:
                    overlaps.append(len(q_tok & t_tok) / len(q_tok))
                q_rare = {t for t in q_tok if is_rare(t)}
                if q_rare:
                    overlaps_rare.append(len(q_rare & t_tok) / len(q_rare))
                m = (q_tok & live_tokens[cid]) - t_tok
                b = m - frozen_tokens[cid]
                memory_all |= bool(m)
                memory_rare |= bool({t for t in m if is_rare(t)})
                if b:
                    bridge_all = True
                    bridge_evidence.append(cid)
                bridge_rare |= bool({t for t in b if is_rare(t)})
                for t in sorted(b):
                    name, kind, value, step = late_prov[cid].get(t, ("", "", "", -1))
                    if kind == "alias":
                        alias_bridge_all = True
                        if is_rare(t):
                            alias_bridge_rare = True
                    token_rows.append(
                        {
                            "dataset": key,
                            "retriever": retriever,
                            "query_id": qid,
                            "chunk_id": cid,
                            "chunk_step": frozen[cid].step,
                            "token": t,
                            "rare": is_rare(t),
                            "entity_name": name,
                            "field_kind": kind,
                            "field_value": value,
                            "added_at_step": step,
                            "n_chunks_sharing_record": len(dindex.name_to_chunks.get(name, [])),
                        }
                    )

            rank_live = rank_frozen = None
            if bridge_evidence:
                streams = score_mbr_variant_document_pools(
                    dindex,
                    query=question,
                    retriever_name=retriever,
                    variant=MBR_VARIANT_MBR2,
                    memory_name_weight=PUBLISHED_SETTINGS["memory_name_weight"],
                    memory_alias_weight=PUBLISHED_SETTINGS["memory_alias_weight"],
                    memory_description_weight=PUBLISHED_SETTINGS["memory_description_weight"],
                    memory_aggregation=PUBLISHED_SETTINGS["memory_aggregation"],
                )
                fstreams = _frozen_streams(streams, frozen, memory_aggregation=PUBLISHED_SETTINGS["memory_aggregation"])
                live_ranks = memory_channel_ranks(streams, dindex)
                frozen_ranks = memory_channel_ranks(fstreams, dindex)
                rank_live = min(live_ranks[c] for c in bridge_evidence)
                rank_frozen = min(frozen_ranks[c] for c in bridge_evidence)

            query_rows.append(
                {
                    "dataset": key,
                    "retriever": retriever,
                    "query_id": qid,
                    "doc_id": doc_id,
                    "question": question,
                    "n_evidence": len(evidence),
                    "overlap_text": statistics.mean(overlaps) if overlaps else None,
                    "overlap_text_rare": statistics.mean(overlaps_rare) if overlaps_rare else None,
                    "memory_query": memory_all,
                    "memory_query_rare": memory_rare,
                    "bridge_query": bridge_all,
                    "bridge_query_rare": bridge_rare,
                    "alias_bridge_query": alias_bridge_all,
                    "alias_bridge_query_rare": alias_bridge_rare,
                    "memory_rank_live": rank_live,
                    "memory_rank_frozen": rank_frozen,
                }
            )

    return pd.DataFrame(query_rows), pd.DataFrame(token_rows)


def fmt(v, digits=1):
    return "-" if v is None or pd.isna(v) else f"{v:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    args = parser.parse_args()

    e2_parquet = REPO_ROOT / "outputs" / "rebuttal" / "E2_frozen_records" / "per_query.parquet"
    hits = pd.read_parquet(e2_parquet)

    out_dir = REPO_ROOT / "outputs" / "rebuttal" / EXPERIMENT_ID
    results_dir = REPO_ROOT / "results" / "rebuttal" / EXPERIMENT_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_queries: list[pd.DataFrame] = []
    all_tokens: list[pd.DataFrame] = []
    for key in args.datasets:
        qf, tf = analyze_dataset(key)
        all_queries.append(qf)
        all_tokens.append(tf)
    queries = pd.concat(all_queries, ignore_index=True)
    tokens = pd.concat(all_tokens, ignore_index=True) if any(len(t) for t in all_tokens) else pd.DataFrame()

    queries = queries.merge(
        hits[["dataset", "query_id", "eligible",
              "hit5_live", "hit10_live", "hit5_frozen", "hit10_frozen", "hit5_no_memory", "hit10_no_memory"]],
        on=["dataset", "query_id"],
        how="left",
        validate="one_to_one",
    )
    if queries["eligible"].isna().any():
        raise RuntimeError("Some queries did not join to the E2 per-query results; stopping.")

    queries.to_parquet(out_dir / "per_query.parquet", index=False)
    if len(tokens):
        tokens.to_parquet(out_dir / "bridge_tokens.parquet", index=False)

    # ---- report tables -----------------------------------------------------
    csv_rows: list[dict] = []
    lines: list[str] = [
        "# E2b. Bridge queries: why frozen matched live",
        "",
        "Tokenization: the repository's BM25 tokenizer, stopwords dropped. A token is rare in a "
        f"document when it appears in fewer than {int(RARE_DF_SHARE*100)}% of that document's chunks. "
        "A memory query has an evidence chunk that matches a query token only through its linked "
        "records; a bridge query needs a field that arrived at a LATER step than the chunk.",
        "",
        "## 1-2. Lexical overlap and query counts",
        "",
        "| Dataset | Queries | Mean Q-in-text overlap | (rare tokens) | Memory queries | (rare) | Bridge queries | (rare) | Alias-bridge | (rare) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key, sub in queries.groupby("dataset", sort=False):
        n = len(sub)
        def pc(col):
            return f"{int(sub[col].sum())} ({100.0*sub[col].mean():.1f}%)"
        lines.append(
            f"| {key} | {n} | {100.0*sub['overlap_text'].mean():.1f}% | {100.0*sub['overlap_text_rare'].mean():.1f}% "
            f"| {pc('memory_query')} | {pc('memory_query_rare')} | {pc('bridge_query')} | {pc('bridge_query_rare')} "
            f"| {pc('alias_bridge_query')} | {pc('alias_bridge_query_rare')} |"
        )
        csv_rows.append(
            {
                "section": "counts", "dataset": key, "n": n,
                "overlap_text_pct": 100.0 * sub["overlap_text"].mean(),
                "overlap_text_rare_pct": 100.0 * sub["overlap_text_rare"].mean(),
                "memory_queries": int(sub["memory_query"].sum()),
                "memory_queries_rare": int(sub["memory_query_rare"].sum()),
                "bridge_queries": int(sub["bridge_query"].sum()),
                "bridge_queries_rare": int(sub["bridge_query_rare"].sum()),
                "alias_bridge_queries": int(sub["alias_bridge_query"].sum()),
                "alias_bridge_queries_rare": int(sub["alias_bridge_query_rare"].sum()),
            }
        )

    lines += [
        "",
        "## 3. Retrieval on bridge queries only",
        "",
        f"HR in percent; paired live - frozen in pp with 95% bootstrap CI (10,000 resamples, seed 42). Groups under {MIN_GROUP} queries are too small to conclude.",
        "",
        "| Dataset | k | n | Live | Frozen | No memory | Live-Frozen (pp) | 95% CI | Note |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key, sub in queries.groupby("dataset", sort=False):
        bridge = sub[sub["bridge_query"] & sub["eligible"].astype(bool)]
        for k in ("5", "10"):
            live = bridge[f"hit{k}_live"].astype(float)
            froz = bridge[f"hit{k}_frozen"].astype(float)
            mean_pp, lo, hi = paired_bootstrap_ci(((live - froz) * 100.0).tolist())
            note = "too small to conclude" if len(bridge) < MIN_GROUP else ""
            lines.append(
                f"| {key} | {k} | {len(bridge)} | {fmt(100.0*live.mean())} | {fmt(100.0*froz.mean())} "
                f"| {fmt(100.0*bridge[f'hit{k}_no_memory'].astype(float).mean())} "
                f"| {fmt(mean_pp,2)} | [{fmt(lo,2)}, {fmt(hi,2)}] | {note} |"
            )
            csv_rows.append(
                {
                    "section": "bridge_hr", "dataset": key, "k": int(k), "n": int(len(bridge)),
                    "hr_live": 100.0 * live.mean() if len(bridge) else None,
                    "hr_frozen": 100.0 * froz.mean() if len(bridge) else None,
                    "hr_no_memory": 100.0 * bridge[f"hit{k}_no_memory"].astype(float).mean() if len(bridge) else None,
                    "diff_pp": mean_pp, "ci_lo_pp": lo, "ci_hi_pp": hi,
                    "too_small": len(bridge) < MIN_GROUP,
                }
            )

    lines += [
        "",
        "## 4-5. Record sharing and memory-channel ranks on bridge queries",
        "",
        "Sharing: chunks of the document linked to the record carrying each bridge token (all of them receive the same boost). Rank: best memory-channel-only rank of a bridge evidence chunk.",
        "",
        "| Dataset | Bridge tokens | Sharing median | Sharing p90 | Memory rank live (median) | Memory rank frozen (median) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for key in args.datasets:
        sub_q = queries[queries["dataset"] == key]
        sub_t = tokens[tokens["dataset"] == key] if len(tokens) else pd.DataFrame()
        bridged = sub_q[sub_q["bridge_query"]]
        share_med = share_p90 = None
        if len(sub_t):
            share_med = float(sub_t["n_chunks_sharing_record"].median())
            share_p90 = float(sub_t["n_chunks_sharing_record"].quantile(0.9))
        rank_live = float(bridged["memory_rank_live"].median()) if len(bridged) else None
        rank_frozen = float(bridged["memory_rank_frozen"].median()) if len(bridged) else None
        lines.append(
            f"| {key} | {len(sub_t)} | {fmt(share_med,0)} | {fmt(share_p90,0)} "
            f"| {fmt(rank_live,0)} | {fmt(rank_frozen,0)} |"
        )
        csv_rows.append(
            {
                "section": "sharing_ranks", "dataset": key, "n": int(len(sub_t)),
                "sharing_median": share_med, "sharing_p90": share_p90,
                "memory_rank_live_median": rank_live, "memory_rank_frozen_median": rank_frozen,
            }
        )

    lines += ["", "## 6. Example bridge queries (random, seed 42)", ""]
    rng = random.Random(42)
    for key in args.datasets:
        sub = queries[(queries["dataset"] == key) & queries["bridge_query"]]
        ids = sorted(sub["query_id"].tolist())
        picks = rng.sample(ids, min(5, len(ids)))
        lines.append(f"**{key}** ({len(ids)} bridge queries)")
        lines.append("")
        for qid in picks:
            row = sub[sub["query_id"] == qid].iloc[0]
            toks = tokens[(tokens["dataset"] == key) & (tokens["query_id"] == qid)]
            t0 = toks.iloc[0]
            value = str(t0["field_value"])[:90]
            lines.append(
                f"- `{qid}`: {str(row['question'])[:140]}\n"
                f"  - bridge token \"{t0['token']}\" from the {t0['field_kind']} \"{value}\" of "
                f"\"{t0['entity_name']}\" (chunk step {t0['chunk_step']}, arrived step {t0['added_at_step']}); "
                f"hit@10 live={bool(row['hit10_live'])}, frozen={bool(row['hit10_frozen'])}"
            )
        lines.append("")

    (results_dir / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.DataFrame(csv_rows).to_csv(results_dir / "table.csv", index=False)

    # ---- README ------------------------------------------------------------
    counts = pd.DataFrame([r for r in csv_rows if r["section"] == "counts"])
    bridge_hr = pd.DataFrame([r for r in csv_rows if r["section"] == "bridge_hr" and r["k"] == 10])
    sharing = pd.DataFrame([r for r in csv_rows if r["section"] == "sharing_ranks"])
    bridge_share = counts["bridge_queries"] / counts["n"] * 100.0
    rare_share = counts["bridge_queries_rare"] / counts["n"] * 100.0
    lo_ds = counts.loc[bridge_share.idxmin(), "dataset"]
    hi_ds = counts.loc[bridge_share.idxmax(), "dataset"]
    readme = f"""# E2b. Bridge queries

**Command.** `.venv/bin/python scripts/rebuttal/e2b_bridge_queries.py` (repository root; no LLM calls, no GPU).

**Definitions.** Repo BM25 tokenizer, stopwords dropped; fixed before running. For a query q and evidence
chunk c: memory-only tokens M = (Q intersect L(c)) minus T(c); bridge tokens B = M minus F(c), where F/L are
the frozen/live record tokens of c's linked names. Rare = document frequency under {int(RARE_DF_SHARE*100)}%
of the document's chunks. Queries: {json.dumps({r['dataset']: r['n'] for r in csv_rows if r['section']=='counts'})}.
HR hits come from E2's per_query.parquet (same hr.py implementation); groups under {MIN_GROUP} queries are
flagged as too small to conclude.

**Summary.**
How often the benchmarks nominally need a later update varies widely: bridge queries are
{bridge_share.min():.1f}% of queries on {lo_ds} but up to {bridge_share.max():.1f}% on {hi_ds}
({rare_share.min():.1f}-{rare_share.max():.1f}% on rare tokens), so lack of exposure explains the E2 null only
where bridges are rare. Even when a query does need a later update, live does not reliably beat frozen:
on bridge queries the paired live-minus-frozen difference at k=10 spans {bridge_hr['diff_pp'].min():+.2f} to
{bridge_hr['diff_pp'].max():+.2f} pp with every 95% CI crossing or touching zero, although on the memory
channel alone live ranks the bridge evidence better than frozen (median ranks in table.md). The boost also
lands broadly: the record carrying a bridge token is shared by a median of
{sharing['sharing_median'].min():.0f}-{sharing['sharing_median'].max():.0f} chunks per dataset (90th percentile
up to {sharing['sharing_p90'].max():.0f}), so many chunks receive the same memory-channel lift at once.
"""
    (results_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"Wrote {out_dir} and {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
