# E2b. Bridge queries

**Command.** `.venv/bin/python scripts/rebuttal/e2b_bridge_queries.py` (repository root; no LLM calls, no GPU).

**Definitions.** Repo BM25 tokenizer, stopwords dropped; fixed before running. For a query q and evidence
chunk c: memory-only tokens M = (Q intersect L(c)) minus T(c); bridge tokens B = M minus F(c), where F/L are
the frozen/live record tokens of c's linked names. Rare = document frequency under 5%
of the document's chunks. Queries: {"loogle": 1832, "musique_32k": 900, "qasper_64k": 1333, "novelhopqa": 908}.
HR hits come from E2's per_query.parquet (same hr.py implementation); groups under 30 queries are
flagged as too small to conclude.

**Summary.**
How often the benchmarks nominally need a later update varies widely: bridge queries are
1.6% of queries on loogle but up to 66.0% on novelhopqa
(0.1-48.1% on rare tokens), so lack of exposure explains the E2 null only
where bridges are rare. Even when a query does need a later update, live does not reliably beat frozen:
on bridge queries the paired live-minus-frozen difference at k=10 spans +0.00 to
+0.65 pp with every 95% CI crossing or touching zero, although on the memory
channel alone live ranks the bridge evidence better than frozen (median ranks in table.md). The boost also
lands broadly: the record carrying a bridge token is shared by a median of
4-108 chunks per dataset (90th percentile
up to 251), so many chunks receive the same memory-channel lift at once.
