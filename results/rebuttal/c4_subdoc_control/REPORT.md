# C4: SAADI vs Vanilla inside the gold sub-document (reviewer iVU8, Q2)

## Summary for the rebuttal

To separate SAADI's within-document ranking from sub-document selection, we re-ranked only the chunks of each query's gold sub-document in every concatenated MuSiQue and QASPER setting (32K to 256K tokens), for 16 setting-retriever pairs (BM25, Contriever), comparing SAADI with Vanilla (the text channel alone) on identical chunk scores. On the whole document both methods first reproduce the paper's saved rankings and hit rates exactly (identical top-k lists for every query and identical hit rates). On all queries, with the evidence labels re-derived inside the gold sub-document by the paper's own labeler, SAADI's MRR inside the gold sub-document is higher than Vanilla's in 15 of 16 pairs, and the 95% bootstrap CI of the gain lies fully above zero in 12. The in-sub-document MRR gain ranges from -0.2 to +4.8 points (median +2.5); inside HR@5 rises in 13 of 16 pairs. SAADI does not reach the gold sub-document more often than Vanilla: whether at least one of its top-10 chunks comes from it is higher in 9 of 16 pairs, with the CI above zero in 0; but it fills more of its top 10 with that sub-document's chunks in 16 of 16 pairs (+3.3 to +8.7 points, CI above zero in 16), so the full-document gain comes both from favouring the right sub-document and from ranking better inside it. From 32K to 256K the in-sub-document MRR gain is, for BM25: MuSiQue +4.4 at 32K, +4.1 at 64K, +3.6 at 128K, +4.1 at 256K; QASPER +1.0 at 32K, +1.3 at 64K, +0.6 at 128K, +1.1 at 256K / for Contriever: MuSiQue +4.5 at 32K, +4.8 at 64K, +3.9 at 128K, +4.8 at 256K; QASPER -0.2 at 32K, +0.6 at 64K, +0.2 at 128K, +1.0 at 256K. Restricting to the queries whose original evidence lies entirely inside the gold sub-document (no relabeling) gives the same picture: an MRR gain in 15 of 16 pairs (CI above zero in 9). Renormalising the scores over the gold sub-document only, instead of keeping the full-document scores, gives a gain in 15 of 16 pairs (CI above zero in 13).

Not included (full per-chunk scores not available here; C2 computed them on the server for MuSiQue-32K and QASPER-64K only, not copied): musique_32k__bge_m3, musique_32k__jina, musique_32k__qwen, qasper_64k__bge_m3, qasper_64k__jina, qasper_64k__qwen.

## Setup

- Main query set: set R (labels re-derived inside the gold sub-document); the other set (S) is reported alongside.
- Restriction (main): full-document scores kept, chunks outside the gold sub-document dropped. Second version: every channel z-scored again over the gold sub-document's chunks (SAADI normalises per query).
- Hit = a sufficient chunk, or every chunk of one evidence group, in the top k (the paper's definition). Full-ranking measures use the paper's labels on the whole document.
- Sub-document hit@10: at least one of the full-document top-10 chunks belongs to the gold sub-document (a chunk crossing a boundary belongs to both sub-documents).
- Paired differences SAADI - Vanilla, percent points, 95% CI from 10,000 bootstrap resamples over queries, seed 42. Nothing tuned.

## 1. Is the full-document gain explained by sub-document selection, by the ranking inside it, or both? (set R)

| Pair | n | Full HR@10 gain | Sub-doc hit@10 gain | Top-10 share gain | Inside HR@5 gain | Inside MRR gain | Reading |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MuSiQue-32K BM25 | 900 | +3.7 [+1.7, +5.7] | +0.0 [-0.3, +0.3] | +4.4 [+3.8, +5.0] | +3.2 [+1.3, +5.0] | +4.4 [+3.3, +5.6] | full gain; both |
| MuSiQue-64K BM25 | 900 | +5.1 [+3.2, +7.1] | -0.2 [-0.6, +0.0] | +4.5 [+3.9, +5.1] | +4.1 [+2.2, +6.0] | +4.1 [+3.0, +5.2] | full gain; both |
| MuSiQue-128K BM25 | 900 | +3.9 [+2.0, +5.8] | -0.4 [-0.9, -0.1] | +5.1 [+4.5, +5.8] | +3.3 [+1.4, +5.2] | +3.6 [+2.5, +4.7] | full gain; both |
| MuSiQue-256K BM25 | 900 | +3.8 [+1.8, +5.8] | +0.1 [-0.6, +0.8] | +4.7 [+4.0, +5.4] | +3.3 [+1.4, +5.2] | +4.1 [+3.1, +5.2] | full gain; both |
| QASPER-32K BM25 | 1321 | +4.5 [+2.8, +6.3] | +0.4 [-1.0, +1.8] | +6.7 [+6.0, +7.4] | +0.3 [-1.4, +2.0] | +1.0 [+0.2, +1.7] | full gain; both |
| QASPER-64K BM25 | 1322 | +4.2 [+2.7, +5.7] | -0.8 [-2.4, +0.8] | +5.2 [+4.6, +5.9] | +1.0 [-0.6, +2.6] | +1.3 [+0.6, +2.0] | full gain; both |
| QASPER-128K BM25 | 1321 | +4.1 [+2.6, +5.6] | +0.2 [-1.4, +1.7] | +5.2 [+4.5, +5.8] | +1.7 [-0.1, +3.6] | +0.6 [-0.1, +1.3] | full gain; sub-document selection |
| QASPER-256K BM25 | 1321 | +2.6 [+1.4, +3.9] | +1.4 [-0.2, +3.0] | +4.4 [+3.8, +5.0] | +1.4 [-0.2, +3.0] | +1.1 [+0.4, +1.8] | full gain; both |
| MuSiQue-32K Contriever | 900 | +7.6 [+5.4, +9.7] | +0.2 [+0.0, +0.6] | +8.7 [+8.0, +9.5] | +6.4 [+4.3, +8.7] | +4.5 [+3.3, +5.7] | full gain; both |
| MuSiQue-64K Contriever | 900 | +9.1 [+6.9, +11.3] | +0.3 [-0.2, +0.9] | +8.3 [+7.6, +9.1] | +6.4 [+4.2, +8.7] | +4.8 [+3.6, +5.9] | full gain; both |
| MuSiQue-128K Contriever | 900 | +7.8 [+5.7, +10.0] | +0.6 [-0.3, +1.3] | +8.3 [+7.5, +9.1] | +5.0 [+2.8, +7.2] | +3.9 [+2.8, +5.1] | full gain; both |
| MuSiQue-256K Contriever | 900 | +10.0 [+7.7, +12.3] | +1.1 [-0.1, +2.4] | +7.7 [+6.9, +8.6] | +5.3 [+3.1, +7.7] | +4.8 [+3.5, +6.1] | full gain; both |
| QASPER-32K Contriever | 1321 | +2.3 [+0.7, +3.8] | -1.1 [-2.6, +0.3] | +3.7 [+3.1, +4.3] | -0.2 [-2.2, +1.7] | -0.2 [-0.9, +0.6] | full gain; sub-document selection |
| QASPER-64K Contriever | 1322 | +2.6 [+1.1, +4.2] | -3.5 [-5.1, -1.8] | +3.4 [+2.8, +4.0] | -0.7 [-2.6, +1.2] | +0.6 [-0.2, +1.4] | full gain; sub-document selection |
| QASPER-128K Contriever | 1321 | +2.3 [+0.8, +3.7] | -0.8 [-2.3, +0.8] | +3.3 [+2.8, +3.9] | -0.6 [-2.5, +1.2] | +0.2 [-0.6, +1.0] | full gain; sub-document selection |
| QASPER-256K Contriever | 1321 | +2.0 [+0.7, +3.4] | +0.4 [-1.3, +2.0] | +3.5 [+2.8, +4.1] | +1.4 [-0.5, +3.2] | +1.0 [+0.2, +1.8] | full gain; both |

Reading: "sub-document selection" when the CI of the sub-document hit@10 or of the top-10 share is above zero, "inside ranking" when the inside-MRR CI is above zero, "both" when both are. A fixed rule for reading the table, not a test of a decomposition.

### Means over the four lengths (%, Vanilla → SAADI; full-document HR@10 on all eligible queries, the rest on set R)

| Dataset, retriever | Full-document HR@10 | At least one top-10 chunk from the gold document | Share of top 10 from the gold document | MRR inside the gold document |
| --- | --- | --- | --- | --- |
| MuSiQue, BM25 (4 lengths) | 67.2 → 71.3 | 99.4 → 99.3 | 66.0 → 70.7 | 41.9 → 46.0 |
| QASPER, BM25 (4 lengths) | 21.0 → 24.8 | 72.7 → 72.9 | 32.1 → 37.5 | 22.8 → 23.8 |
| MuSiQue, Contriever (4 lengths) | 64.8 → 73.4 | 98.3 → 98.8 | 50.5 → 58.8 | 45.5 → 50.0 |
| QASPER, Contriever (4 lengths) | 21.7 → 24.0 | 69.9 → 68.6 | 34.1 → 37.6 | 23.7 → 24.2 |

## 2. Measures per pair

### Set R (labels re-derived inside the gold sub-document)

| Pair | Method | n | Full HR@5 | Full HR@10 | Full med. rank | Sub-doc hit@10 | Top-10 share | In HR@1 | In HR@3 | In HR@5 | In HR@10 | In HR@10 (>10 ch.) | In MRR | In med. rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MuSiQue-32K BM25 | Vanilla | 900 | 57.6 | 74 | 4 | 99.9 | 80.0 | 23 | 48 | 63.8 | 82.3 | 82.2 (n=894) | 40.9 | 4 |
| MuSiQue-32K BM25 | SAADI | 900 | 62.8 | 77.7 | 4 | 99.9 | 84.4 | 27.8 | 53.1 | 67 | 84.3 | 84.2 (n=894) | 45.3 | 3 |
| MuSiQue-64K BM25 | Vanilla | 900 | 54 | 69.2 | 5 | 99.8 | 71.7 | 23.7 | 48.7 | 63.2 | 81.3 | 81.2 (n=894) | 41.6 | 4 |
| MuSiQue-64K BM25 | SAADI | 900 | 59.2 | 74.3 | 4 | 99.6 | 76.2 | 28.1 | 54.3 | 67.3 | 84.1 | 84.0 (n=894) | 45.7 | 3 |
| MuSiQue-128K BM25 | Vanilla | 900 | 51.6 | 64.9 | 5 | 99.4 | 61.2 | 24.4 | 49.7 | 63.6 | 81.7 | 81.5 (n=894) | 42.4 | 4 |
| MuSiQue-128K BM25 | SAADI | 900 | 55.7 | 68.8 | 5 | 99 | 66.3 | 28.2 | 53.9 | 66.9 | 83.9 | 83.8 (n=894) | 46.0 | 3 |
| MuSiQue-256K BM25 | Vanilla | 900 | 48.4 | 60.6 | 6 | 98.6 | 51.0 | 25.2 | 50.2 | 63.6 | 81.3 | 81.2 (n=894) | 42.9 | 3 |
| MuSiQue-256K BM25 | SAADI | 900 | 52.3 | 64.3 | 5 | 98.7 | 55.8 | 29.2 | 55.6 | 66.9 | 83.4 | 83.3 (n=894) | 47.0 | 3 |
| QASPER-32K BM25 | Vanilla | 1321 | 17.9 | 27.6 | 39 | 85.6 | 40.0 | 8.2 | 24.0 | 36.4 | 56.0 | 55.8 (n=1314) | 22.5 | 9 |
| QASPER-32K BM25 | SAADI | 1321 | 19.6 | 32.1 | 28 | 86.0 | 46.7 | 8.9 | 24.7 | 36.7 | 59.7 | 59.5 (n=1314) | 23.4 | 8 |
| QASPER-64K BM25 | Vanilla | 1322 | 14.4 | 22.0 | 77 | 76.3 | 34.0 | 8.5 | 23.9 | 35.4 | 56.5 | 56.3 (n=1315) | 22.7 | 9 |
| QASPER-64K BM25 | SAADI | 1322 | 17.0 | 26.2 | 52 | 75.5 | 39.2 | 9.2 | 25.4 | 36.4 | 59.1 | 58.9 (n=1315) | 24.0 | 8 |
| QASPER-128K BM25 | Vanilla | 1321 | 12.8 | 18.5 | 169 | 68.2 | 29.3 | 8.9 | 24.5 | 35.9 | 56.5 | 56.2 (n=1314) | 23.0 | 9 |
| QASPER-128K BM25 | SAADI | 1321 | 15.0 | 22.6 | 117 | 68.4 | 34.4 | 9.0 | 24.7 | 37.6 | 59.3 | 59.1 (n=1314) | 23.7 | 8 |
| QASPER-256K BM25 | Vanilla | 1321 | 11.5 | 15.7 | 408 | 60.5 | 25.3 | 8.9 | 24.9 | 35.9 | 56.9 | 56.6 (n=1314) | 23.1 | 9 |
| QASPER-256K BM25 | SAADI | 1321 | 12.2 | 18.4 | 284 | 61.9 | 29.7 | 9.4 | 25.4 | 37.3 | 60.0 | 59.7 (n=1314) | 24.2 | 8 |
| MuSiQue-32K Contriever | Vanilla | 900 | 59.9 | 76.1 | 4 | 99.6 | 66.5 | 25.1 | 56.8 | 71.3 | 91 | 90.9 (n=894) | 45.6 | 3 |
| MuSiQue-32K Contriever | SAADI | 900 | 68.8 | 83.7 | 3 | 99.8 | 75.2 | 29.7 | 61.8 | 77.8 | 93.2 | 93.2 (n=894) | 50.1 | 2 |
| MuSiQue-64K Contriever | Vanilla | 900 | 54 | 68.3 | 5 | 99.1 | 56.2 | 24.9 | 56.7 | 71.2 | 90.8 | 90.7 (n=894) | 45.4 | 3 |
| MuSiQue-64K Contriever | SAADI | 900 | 60.9 | 77.4 | 3 | 99.4 | 64.6 | 29.8 | 61 | 77.7 | 92.8 | 92.7 (n=894) | 50.2 | 2 |
| MuSiQue-128K Contriever | Vanilla | 900 | 47.9 | 60.7 | 6 | 98.2 | 44.7 | 25.1 | 56.8 | 71.4 | 90.8 | 90.7 (n=894) | 45.6 | 3 |
| MuSiQue-128K Contriever | SAADI | 900 | 56.3 | 68.4 | 4 | 98.8 | 53.0 | 28.7 | 62.1 | 76.4 | 92.7 | 92.6 (n=894) | 49.5 | 2 |
| MuSiQue-256K Contriever | Vanilla | 900 | 44.3 | 54 | 8 | 96.2 | 34.6 | 24.9 | 56.6 | 71.2 | 90.8 | 90.7 (n=894) | 45.4 | 3 |
| MuSiQue-256K Contriever | SAADI | 900 | 53.6 | 64 | 5 | 97.3 | 42.4 | 30.1 | 61.7 | 76.6 | 92.7 | 92.6 (n=894) | 50.2 | 2 |
| QASPER-32K Contriever | Vanilla | 1321 | 18.5 | 29.5 | 33 | 83.1 | 44.2 | 8.6 | 25.8 | 37.7 | 60.1 | 59.9 (n=1314) | 23.7 | 8 |
| QASPER-32K Contriever | SAADI | 1321 | 19.6 | 31.8 | 27 | 82.0 | 47.8 | 8.3 | 24.5 | 37.5 | 60.3 | 60.1 (n=1314) | 23.5 | 8 |
| QASPER-64K Contriever | Vanilla | 1322 | 15.3 | 23.4 | 68 | 74.1 | 36.5 | 8.5 | 26.1 | 38.8 | 60.3 | 60.1 (n=1315) | 23.8 | 8 |
| QASPER-64K Contriever | SAADI | 1322 | 16.3 | 25.9 | 51 | 70.6 | 39.9 | 9.4 | 25.4 | 38.1 | 62.0 | 61.7 (n=1315) | 24.5 | 8 |
| QASPER-128K Contriever | Vanilla | 1321 | 13.2 | 19.2 | 143 | 65.1 | 30.6 | 8.6 | 26.1 | 38.5 | 60.4 | 60.2 (n=1314) | 23.8 | 8 |
| QASPER-128K Contriever | SAADI | 1321 | 13.1 | 21.5 | 111 | 64.3 | 34.0 | 8.9 | 25.7 | 37.9 | 61.2 | 61.0 (n=1314) | 24.0 | 8 |
| QASPER-256K Contriever | Vanilla | 1321 | 10.1 | 14.8 | 307 | 57.2 | 25.2 | 8.6 | 25.8 | 38.1 | 60.5 | 60.3 (n=1314) | 23.7 | 8 |
| QASPER-256K Contriever | SAADI | 1321 | 11.4 | 16.9 | 235 | 57.6 | 28.6 | 9.3 | 26.6 | 39.4 | 62.1 | 61.9 (n=1314) | 24.7 | 7 |

### Set S (all paper evidence inside the gold sub-document)

| Pair | Method | n | Full HR@5 | Full HR@10 | Full med. rank | Sub-doc hit@10 | Top-10 share | In HR@1 | In HR@3 | In HR@5 | In HR@10 | In HR@10 (>10 ch.) | In MRR | In med. rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MuSiQue-32K BM25 | Vanilla | 575 | 61.4 | 78.1 | 4 | 99.8 | 84.9 | 24.7 | 50.3 | 64.5 | 83.3 | 83.1 (n=569) | 42.3 | 3 |
| MuSiQue-32K BM25 | SAADI | 575 | 66.1 | 81.9 | 3 | 99.8 | 89.4 | 29.4 | 53.7 | 68.2 | 84.7 | 84.5 (n=569) | 46.6 | 3 |
| MuSiQue-64K BM25 | Vanilla | 385 | 58.7 | 74.3 | 4 | 99.7 | 79.5 | 27.3 | 50.4 | 64.4 | 81.8 | 81.6 (n=380) | 43.7 | 3 |
| MuSiQue-64K BM25 | SAADI | 385 | 64.9 | 82.1 | 3 | 99.5 | 84.6 | 33.5 | 55.8 | 68.3 | 85.5 | 85.3 (n=380) | 49.4 | 3 |
| MuSiQue-128K BM25 | Vanilla | 250 | 60.4 | 74.8 | 3.5 | 99.6 | 73.0 | 31.2 | 54 | 65.2 | 84.4 | 84.1 (n=245) | 47.4 | 3 |
| MuSiQue-128K BM25 | SAADI | 250 | 66.8 | 81.6 | 2.5 | 99.6 | 79.5 | 38 | 60.4 | 71.6 | 88 | 87.8 (n=245) | 53.6 | 2 |
| MuSiQue-256K BM25 | Vanilla | 175 | 57.7 | 73.1 | 3 | 98.9 | 68.9 | 35.4 | 54.9 | 64 | 84 | 83.5 (n=170) | 49.7 | 3 |
| MuSiQue-256K BM25 | SAADI | 175 | 62.9 | 83.4 | 2 | 98.3 | 74.9 | 40 | 59.4 | 70.3 | 88.6 | 88.2 (n=170) | 54.8 | 2 |
| QASPER-32K BM25 | Vanilla | 979 | 18.0 | 27.8 | 36 | 85.0 | 39.7 | 8.5 | 23.5 | 36.6 | 55.2 | 54.8 (n=972) | 22.4 | 9 |
| QASPER-32K BM25 | SAADI | 979 | 20.3 | 33.4 | 26 | 85.0 | 46.5 | 8.6 | 24.1 | 36.5 | 59.2 | 59.0 (n=972) | 23.0 | 8 |
| QASPER-64K BM25 | Vanilla | 794 | 15.6 | 23.7 | 57 | 74.8 | 33.0 | 8.7 | 25.1 | 36.6 | 55.8 | 55.5 (n=789) | 23.0 | 9 |
| QASPER-64K BM25 | SAADI | 794 | 17.6 | 27.6 | 39.5 | 74.7 | 37.9 | 9.8 | 26.6 | 37.7 | 59.4 | 59.2 (n=789) | 24.7 | 8 |
| QASPER-128K BM25 | Vanilla | 660 | 15.3 | 21.4 | 89 | 65.2 | 28.1 | 10.6 | 28.6 | 40.3 | 59.2 | 58.9 (n=655) | 25.4 | 8 |
| QASPER-128K BM25 | SAADI | 660 | 18.3 | 26.1 | 60.5 | 66.7 | 32.6 | 10.5 | 28.8 | 39.8 | 61.4 | 61.1 (n=655) | 25.6 | 7 |
| QASPER-256K BM25 | Vanilla | 505 | 14.5 | 19.6 | 160 | 57.4 | 23.4 | 12.3 | 30.5 | 41.8 | 60 | 59.6 (n=500) | 26.8 | 8 |
| QASPER-256K BM25 | SAADI | 505 | 15.0 | 23.6 | 111 | 57.4 | 27.6 | 12.7 | 31.1 | 42.4 | 62.4 | 62 (n=500) | 27.9 | 7 |
| MuSiQue-32K Contriever | Vanilla | 575 | 65.6 | 80.3 | 3 | 99.3 | 69.3 | 26.8 | 58.6 | 72.9 | 91.3 | 91.2 (n=569) | 47.4 | 3 |
| MuSiQue-32K Contriever | SAADI | 575 | 75.8 | 90.1 | 2 | 99.7 | 79.5 | 31.7 | 63.1 | 79.5 | 94.6 | 94.6 (n=569) | 52.0 | 2 |
| MuSiQue-64K Contriever | Vanilla | 385 | 63.9 | 76.6 | 3 | 99.0 | 60.8 | 29.4 | 60.5 | 73.5 | 89.9 | 89.7 (n=380) | 49.0 | 3 |
| MuSiQue-64K Contriever | SAADI | 385 | 70.9 | 87.0 | 2 | 99.2 | 70.9 | 35.8 | 64.4 | 78.4 | 93.2 | 93.2 (n=380) | 54.6 | 2 |
| MuSiQue-128K Contriever | Vanilla | 250 | 62 | 74.4 | 3 | 97.6 | 51.3 | 36 | 65.2 | 75.2 | 90.8 | 90.6 (n=245) | 53.7 | 2 |
| MuSiQue-128K Contriever | SAADI | 250 | 75.6 | 84.8 | 2 | 98.4 | 63.3 | 39.6 | 71.6 | 82 | 92.8 | 92.7 (n=245) | 58.1 | 2 |
| MuSiQue-256K Contriever | Vanilla | 175 | 60.6 | 68 | 3 | 96 | 43.3 | 40.6 | 66.9 | 76.6 | 90.9 | 90.6 (n=170) | 56.7 | 2 |
| MuSiQue-256K Contriever | SAADI | 175 | 70.9 | 80.6 | 2 | 97.7 | 56.2 | 45.1 | 69.7 | 78.3 | 92.6 | 92.4 (n=170) | 60.4 | 2 |
| QASPER-32K Contriever | Vanilla | 979 | 18.3 | 29.9 | 30 | 82.8 | 43.9 | 8.7 | 26.0 | 37.2 | 60.0 | 59.7 (n=972) | 23.7 | 8 |
| QASPER-32K Contriever | SAADI | 979 | 20.1 | 32.0 | 25 | 81.7 | 47.7 | 8.3 | 25.0 | 37.2 | 59.0 | 58.7 (n=972) | 23.4 | 8 |
| QASPER-64K Contriever | Vanilla | 794 | 16.1 | 25.3 | 50.5 | 72.4 | 35.5 | 9.3 | 27.8 | 39.4 | 59.8 | 59.6 (n=789) | 24.7 | 8 |
| QASPER-64K Contriever | SAADI | 794 | 17.4 | 28.1 | 37 | 69.4 | 38.4 | 10.3 | 26.8 | 38.9 | 61.7 | 61.5 (n=789) | 25.3 | 8 |
| QASPER-128K Contriever | Vanilla | 660 | 15 | 22.6 | 75.5 | 63.3 | 29.0 | 10.5 | 29.1 | 41.4 | 62.6 | 62.3 (n=655) | 26.1 | 7.5 |
| QASPER-128K Contriever | SAADI | 660 | 14.5 | 25 | 65.5 | 63.0 | 31.9 | 10.9 | 28.0 | 41.5 | 62.3 | 62.0 (n=655) | 26.1 | 7 |
| QASPER-256K Contriever | Vanilla | 505 | 12.3 | 18.4 | 153 | 53.1 | 22.8 | 12.3 | 30.3 | 42.8 | 63.4 | 63 (n=500) | 27.6 | 7 |
| QASPER-256K Contriever | SAADI | 505 | 13.5 | 20.8 | 101 | 55.0 | 26.3 | 12.3 | 30.3 | 43.4 | 65.0 | 64.6 (n=500) | 28.0 | 7 |

## 3. Paired differences (SAADI - Vanilla, 95% CI)

### Set R (labels re-derived inside the gold sub-document)

| Pair | Full HR@10 | Sub-doc hit@10 | Top-10 share | In HR@3 keep / renorm | In HR@5 keep / renorm | In MRR keep / renorm | In HR@10 (>10 ch.) keep / renorm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MuSiQue-32K BM25 | +3.7 [+1.7, +5.7] | +0.0 [-0.3, +0.3] | +4.4 [+3.8, +5.0] | +5.1 [+2.9, +7.4] / +6.7 [+4.4, +9.0] | +3.2 [+1.3, +5.0] / +2.8 [+1.0, +4.7] | +4.4 [+3.3, +5.6] / +5.2 [+4.0, +6.4] | +2.0 [+0.3, +3.7] / +2.1 [+0.4, +3.9] |
| MuSiQue-64K BM25 | +5.1 [+3.2, +7.1] | -0.2 [-0.6, +0.0] | +4.5 [+3.9, +5.1] | +5.7 [+3.8, +7.6] / +6.1 [+4.2, +8.1] | +4.1 [+2.2, +6.0] / +3.3 [+1.4, +5.2] | +4.1 [+3.0, +5.2] / +4.5 [+3.3, +5.6] | +2.8 [+1.2, +4.4] / +2.6 [+0.8, +4.4] |
| MuSiQue-128K BM25 | +3.9 [+2.0, +5.8] | -0.4 [-0.9, -0.1] | +5.1 [+4.5, +5.8] | +4.2 [+2.3, +6.1] / +5.6 [+3.6, +7.6] | +3.3 [+1.4, +5.2] / +2.7 [+0.8, +4.6] | +3.6 [+2.5, +4.7] / +4.5 [+3.3, +5.6] | +2.2 [+0.6, +3.9] / +2.5 [+0.7, +4.3] |
| MuSiQue-256K BM25 | +3.8 [+1.8, +5.8] | +0.1 [-0.6, +0.8] | +4.7 [+4.0, +5.4] | +5.3 [+3.4, +7.2] / +6.3 [+4.2, +8.4] | +3.3 [+1.4, +5.2] / +4.0 [+2.0, +6.1] | +4.1 [+3.1, +5.2] / +4.3 [+3.2, +5.4] | +2.1 [+0.3, +4.0] / +1.8 [-0.1, +3.7] |
| QASPER-32K BM25 | +4.5 [+2.8, +6.3] | +0.4 [-1.0, +1.8] | +6.7 [+6.0, +7.4] | +0.7 [-0.8, +2.2] / +0.7 [-0.8, +2.1] | +0.3 [-1.4, +2.0] / +0.4 [-1.3, +2.1] | +1.0 [+0.2, +1.7] / +1.1 [+0.4, +1.8] | +3.7 [+1.9, +5.6] / +3.8 [+2.0, +5.6] |
| QASPER-64K BM25 | +4.2 [+2.7, +5.7] | -0.8 [-2.4, +0.8] | +5.2 [+4.6, +5.9] | +1.5 [-0.1, +3.2] / +1.6 [+0.0, +3.2] | +1.0 [-0.6, +2.6] / +1.4 [-0.2, +3.1] | +1.3 [+0.6, +2.0] / +1.4 [+0.7, +2.1] | +2.6 [+0.7, +4.4] / +2.6 [+0.7, +4.4] |
| QASPER-128K BM25 | +4.1 [+2.6, +5.6] | +0.2 [-1.4, +1.7] | +5.2 [+4.5, +5.8] | +0.2 [-1.4, +1.8] / +0.8 [-0.8, +2.3] | +1.7 [-0.1, +3.6] / +2.9 [+1.1, +4.7] | +0.6 [-0.1, +1.3] / +1.1 [+0.4, +1.8] | +2.9 [+1.2, +4.6] / +4.0 [+2.3, +5.7] |
| QASPER-256K BM25 | +2.6 [+1.4, +3.9] | +1.4 [-0.2, +3.0] | +4.4 [+3.8, +5.0] | +0.5 [-1.0, +2.0] / +0.8 [-0.6, +2.3] | +1.4 [-0.2, +3.0] / +2.0 [+0.5, +3.6] | +1.1 [+0.4, +1.8] / +1.6 [+0.9, +2.4] | +3.1 [+1.4, +4.9] / +3.3 [+1.4, +5.2] |
| MuSiQue-32K Contriever | +7.6 [+5.4, +9.7] | +0.2 [+0.0, +0.6] | +8.7 [+8.0, +9.5] | +5.0 [+2.7, +7.3] / +3.3 [+1.1, +5.6] | +6.4 [+4.3, +8.7] / +5.4 [+3.2, +7.7] | +4.5 [+3.3, +5.7] / +2.9 [+1.8, +3.9] | +2.2 [+0.8, +3.7] / +2.0 [+0.6, +3.5] |
| MuSiQue-64K Contriever | +9.1 [+6.9, +11.3] | +0.3 [-0.2, +0.9] | +8.3 [+7.6, +9.1] | +4.3 [+2.0, +6.6] / +3.2 [+1.0, +5.4] | +6.4 [+4.2, +8.7] / +5.0 [+2.8, +7.2] | +4.8 [+3.6, +5.9] / +3.4 [+2.3, +4.4] | +2.0 [+0.6, +3.5] / +2.1 [+0.8, +3.5] |
| MuSiQue-128K Contriever | +7.8 [+5.7, +10.0] | +0.6 [-0.3, +1.3] | +8.3 [+7.5, +9.1] | +5.3 [+3.1, +7.6] / +3.3 [+1.1, +5.6] | +5.0 [+2.8, +7.2] / +3.7 [+1.4, +5.9] | +3.9 [+2.8, +5.1] / +2.1 [+1.1, +3.1] | +1.9 [+0.6, +3.2] / +2.0 [+0.7, +3.4] |
| MuSiQue-256K Contriever | +10.0 [+7.7, +12.3] | +1.1 [-0.1, +2.4] | +7.7 [+6.9, +8.6] | +5.1 [+2.7, +7.4] / +3.7 [+1.3, +6.0] | +5.3 [+3.1, +7.7] / +4.6 [+2.3, +6.9] | +4.8 [+3.5, +6.1] / +3.5 [+2.3, +4.6] | +1.9 [+0.6, +3.2] / +2.1 [+0.8, +3.6] |
| QASPER-32K Contriever | +2.3 [+0.7, +3.8] | -1.1 [-2.6, +0.3] | +3.7 [+3.1, +4.3] | -1.3 [-2.8, +0.2] / -1.1 [-2.6, +0.5] | -0.2 [-2.2, +1.7] / -0.1 [-2.0, +1.8] | -0.2 [-0.9, +0.6] / -0.0 [-0.8, +0.7] | +0.2 [-1.4, +1.9] / +0.2 [-1.6, +1.9] |
| QASPER-64K Contriever | +2.6 [+1.1, +4.2] | -3.5 [-5.1, -1.8] | +3.4 [+2.8, +4.0] | -0.7 [-2.3, +1.0] / -0.3 [-2.0, +1.4] | -0.7 [-2.6, +1.2] / -0.5 [-2.4, +1.3] | +0.6 [-0.2, +1.4] / +0.6 [-0.2, +1.3] | +1.7 [-0.1, +3.4] / +1.2 [-0.5, +2.9] |
| QASPER-128K Contriever | +2.3 [+0.8, +3.7] | -0.8 [-2.3, +0.8] | +3.3 [+2.8, +3.9] | -0.5 [-2.2, +1.2] / -0.5 [-2.3, +1.3] | -0.6 [-2.5, +1.2] / -0.3 [-2.2, +1.6] | +0.2 [-0.6, +1.0] / +0.3 [-0.4, +1.1] | +0.8 [-1.0, +2.7] / +0.4 [-1.4, +2.2] |
| QASPER-256K Contriever | +2.0 [+0.7, +3.4] | +0.4 [-1.3, +2.0] | +3.5 [+2.8, +4.1] | +0.8 [-1.0, +2.6] / +0.6 [-1.1, +2.3] | +1.4 [-0.5, +3.2] / +1.9 [+0.1, +3.6] | +1.0 [+0.2, +1.8] / +1.0 [+0.2, +1.8] | +1.7 [-0.2, +3.4] / +1.8 [+0.0, +3.5] |

### Set S (all paper evidence inside the gold sub-document)

| Pair | Full HR@10 | Sub-doc hit@10 | Top-10 share | In HR@3 keep / renorm | In HR@5 keep / renorm | In MRR keep / renorm | In HR@10 (>10 ch.) keep / renorm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MuSiQue-32K BM25 | +3.8 [+1.2, +6.3] | +0.0 [-0.5, +0.5] | +4.5 [+3.7, +5.2] | +3.5 [+0.7, +6.3] / +5.4 [+2.6, +8.2] | +3.7 [+1.6, +5.7] / +3.1 [+1.0, +5.2] | +4.3 [+2.8, +5.7] / +5.2 [+3.7, +6.7] | +1.4 [-0.7, +3.5] / +2.1 [+0.0, +4.2] |
| MuSiQue-64K BM25 | +7.8 [+4.7, +10.9] | -0.3 [-0.8, +0.0] | +5.1 [+4.2, +6.1] | +5.5 [+2.6, +8.6] / +6.5 [+3.4, +9.6] | +3.9 [+1.0, +6.8] / +3.1 [+0.5, +5.7] | +5.7 [+3.9, +7.7] / +5.6 [+3.7, +7.7] | +3.7 [+1.1, +6.3] / +4.2 [+1.1, +7.4] |
| MuSiQue-128K BM25 | +6.8 [+3.2, +10.4] | +0.0 [+0.0, +0.0] | +6.6 [+5.4, +7.8] | +6.4 [+3.2, +10.0] / +7.6 [+3.6, +11.6] | +6.4 [+2.8, +10.0] / +6.8 [+3.2, +10.8] | +6.3 [+4.1, +8.5] / +6.6 [+4.5, +8.9] | +3.7 [+0.0, +7.3] / +4.9 [+1.2, +8.6] |
| MuSiQue-256K BM25 | +10.3 [+5.1, +15.4] | -0.6 [-1.7, +0.0] | +5.9 [+4.5, +7.5] | +4.6 [+1.1, +8.6] / +6.9 [+2.3, +12.0] | +6.3 [+2.3, +10.9] / +9.1 [+4.6, +14.3] | +5.1 [+2.6, +7.7] / +4.9 [+2.6, +7.4] | +4.7 [+0.6, +9.4] / +3.5 [-1.2, +8.2] |
| QASPER-32K BM25 | +5.6 [+3.6, +7.7] | +0.0 [-1.5, +1.5] | +6.7 [+5.9, +7.6] | +0.6 [-1.2, +2.5] / +0.6 [-1.1, +2.3] | -0.1 [-2.0, +1.9] / +0.2 [-1.7, +2.0] | +0.6 [-0.2, +1.5] / +0.8 [+0.0, +1.7] | +4.1 [+2.0, +6.3] / +4.0 [+1.9, +6.2] |
| QASPER-64K BM25 | +3.9 [+2.0, +5.8] | -0.1 [-2.1, +1.9] | +4.9 [+4.1, +5.8] | +1.5 [-0.6, +3.7] / +1.5 [-0.6, +3.7] | +1.0 [-1.0, +3.0] / +1.4 [-0.8, +3.5] | +1.7 [+0.7, +2.7] / +1.8 [+0.8, +2.7] | +3.7 [+1.4, +6.0] / +3.4 [+1.1, +5.7] |
| QASPER-128K BM25 | +4.7 [+2.6, +7.0] | +1.5 [-0.8, +3.8] | +4.5 [+3.6, +5.4] | +0.2 [-2.3, +2.6] / +0.6 [-1.8, +3.0] | -0.5 [-3.0, +2.0] / +0.9 [-1.5, +3.5] | +0.3 [-0.8, +1.3] / +1.2 [+0.2, +2.3] | +2.1 [-0.2, +4.4] / +4.0 [+1.7, +6.3] |
| QASPER-256K BM25 | +4.0 [+1.6, +6.3] | +0.0 [-2.6, +2.8] | +4.3 [+3.2, +5.3] | +0.6 [-2.0, +3.0] / +1.2 [-1.4, +3.8] | +0.6 [-2.0, +3.2] / +1.0 [-1.6, +3.6] | +1.1 [-0.4, +2.5] / +2.0 [+0.5, +3.6] | +2.4 [-0.4, +5.2] / +2.0 [-1.0, +5.0] |
| MuSiQue-32K Contriever | +9.7 [+7.1, +12.3] | +0.3 [+0.0, +0.9] | +10.3 [+9.3, +11.3] | +4.5 [+1.7, +7.3] / +3.1 [+0.5, +5.7] | +6.6 [+3.8, +9.4] / +5.9 [+3.0, +8.7] | +4.7 [+3.1, +6.3] / +3.0 [+1.6, +4.3] | +3.3 [+1.6, +5.3] / +3.0 [+1.2, +4.7] |
| MuSiQue-64K Contriever | +10.4 [+7.3, +13.8] | +0.3 [-0.8, +1.6] | +10.0 [+8.8, +11.2] | +3.9 [+0.8, +7.0] / +2.6 [-0.3, +5.5] | +4.9 [+1.8, +8.1] / +4.2 [+1.0, +7.3] | +5.6 [+3.7, +7.5] / +4.4 [+2.7, +6.2] | +3.4 [+1.1, +5.8] / +3.4 [+1.1, +5.8] |
| MuSiQue-128K Contriever | +10.4 [+6.4, +14.4] | +0.8 [-1.2, +2.8] | +12.0 [+10.4, +13.6] | +6.4 [+2.4, +10.8] / +2.8 [-1.2, +6.8] | +6.8 [+3.2, +10.8] / +4.0 [+0.8, +7.6] | +4.3 [+2.2, +6.6] / +2.1 [+0.2, +4.1] | +2.0 [-0.8, +4.9] / +2.9 [+0.0, +5.7] |
| MuSiQue-256K Contriever | +12.6 [+7.4, +18.3] | +1.7 [-1.1, +4.6] | +12.9 [+10.8, +14.9] | +2.9 [-1.7, +8.0] / -1.1 [-5.1, +2.9] | +1.7 [-2.9, +6.3] / +1.7 [-1.7, +5.7] | +3.7 [+0.8, +6.7] / +1.6 [-0.6, +3.9] | +1.8 [-1.2, +4.7] / +1.8 [-1.2, +4.7] |
| QASPER-32K Contriever | +2.0 [+0.2, +3.9] | -1.1 [-2.8, +0.5] | +3.8 [+3.1, +4.5] | -1.0 [-2.8, +0.7] / -0.7 [-2.5, +1.0] | +0.0 [-2.1, +2.1] / -0.1 [-2.1, +2.0] | -0.3 [-1.2, +0.6] / -0.2 [-1.1, +0.7] | -0.9 [-3.0, +1.1] / -1.2 [-3.3, +0.8] |
| QASPER-64K Contriever | +2.8 [+0.9, +4.8] | -3.0 [-5.0, -1.1] | +3.0 [+2.2, +3.7] | -1.0 [-3.1, +1.0] / -0.4 [-2.5, +1.8] | -0.5 [-2.8, +1.8] / -0.9 [-3.1, +1.4] | +0.7 [-0.4, +1.7] / +0.6 [-0.5, +1.6] | +1.9 [-0.3, +4.2] / +1.3 [-0.9, +3.4] |
| QASPER-128K Contriever | +2.4 [+0.2, +4.7] | -0.3 [-2.7, +2.1] | +2.8 [+2.0, +3.7] | -1.1 [-3.5, +1.4] / -0.5 [-3.0, +2.1] | +0.2 [-2.6, +2.9] / +0.0 [-2.6, +2.6] | +0.0 [-1.1, +1.2] / +0.1 [-1.0, +1.2] | -0.3 [-2.7, +2.3] / -0.2 [-2.6, +2.3] |
| QASPER-256K Contriever | +2.4 [+0.0, +4.8] | +2.0 [-1.0, +5.0] | +3.5 [+2.6, +4.4] | +0.0 [-3.0, +3.0] / -0.2 [-3.0, +2.6] | +0.6 [-2.2, +3.4] / +1.4 [-1.4, +4.2] | +0.4 [-1.1, +2.0] / +0.6 [-0.9, +2.1] | +1.6 [-1.4, +4.6] / +1.6 [-1.2, +4.4] |

### All eligible queries (full document only)

| Pair | n | Full HR@10 | Sub-doc hit@10 | Top-10 share |
| --- | --- | --- | --- | --- |
| MuSiQue-32K BM25 | 900 | +3.7 [+1.7, +5.7] | +0.0 [-0.3, +0.3] | +4.4 [+3.8, +5.0] |
| MuSiQue-64K BM25 | 900 | +5.1 [+3.2, +7.1] | -0.2 [-0.6, +0.0] | +4.5 [+3.9, +5.1] |
| MuSiQue-128K BM25 | 900 | +3.9 [+2.0, +5.8] | -0.4 [-0.9, -0.1] | +5.1 [+4.5, +5.8] |
| MuSiQue-256K BM25 | 900 | +3.8 [+1.8, +5.8] | +0.1 [-0.6, +0.8] | +4.7 [+4.0, +5.4] |
| QASPER-32K BM25 | 1321 | +4.5 [+2.8, +6.3] | +0.4 [-1.0, +1.8] | +6.7 [+6.0, +7.4] |
| QASPER-64K BM25 | 1333 | +4.1 [+2.6, +5.6] | -0.8 [-2.4, +0.8] | +5.2 [+4.5, +5.9] |
| QASPER-128K BM25 | 1321 | +4.1 [+2.6, +5.6] | +0.2 [-1.4, +1.7] | +5.2 [+4.5, +5.8] |
| QASPER-256K BM25 | 1321 | +2.6 [+1.4, +3.9] | +1.4 [-0.2, +3.0] | +4.4 [+3.8, +5.0] |
| MuSiQue-32K Contriever | 900 | +7.6 [+5.4, +9.7] | +0.2 [+0.0, +0.6] | +8.7 [+8.0, +9.5] |
| MuSiQue-64K Contriever | 900 | +9.1 [+6.9, +11.3] | +0.3 [-0.2, +0.9] | +8.3 [+7.6, +9.1] |
| MuSiQue-128K Contriever | 900 | +7.8 [+5.7, +10.0] | +0.6 [-0.3, +1.3] | +8.3 [+7.5, +9.1] |
| MuSiQue-256K Contriever | 900 | +10.0 [+7.7, +12.3] | +1.1 [-0.1, +2.4] | +7.7 [+6.9, +8.6] |
| QASPER-32K Contriever | 1321 | +2.3 [+0.7, +3.8] | -1.1 [-2.6, +0.3] | +3.7 [+3.1, +4.3] |
| QASPER-64K Contriever | 1333 | +2.5 [+0.9, +4.1] | -3.5 [-5.1, -1.9] | +3.4 [+2.8, +4.0] |
| QASPER-128K Contriever | 1321 | +2.3 [+0.8, +3.7] | -0.8 [-2.3, +0.8] | +3.3 [+2.8, +3.9] |
| QASPER-256K Contriever | 1321 | +2.0 [+0.7, +3.4] | +0.4 [-1.3, +2.0] | +3.5 [+2.8, +4.1] |

## 4. Across pairs

| Query set | Restriction | Pairs | Inside MRR gain > 0 | CI fully above 0 | CI fully below 0 |
| --- | --- | --- | --- | --- | --- |
| R | keep | 16 | 15 | 12 | 0 |
| R | renorm | 16 | 15 | 13 | 0 |
| S | keep | 16 | 15 | 9 | 0 |
| S | renorm | 16 | 15 | 11 | 0 |

Trend from 32K to 256K (set R, keep; SAADI - Vanilla):

| Retriever | Dataset | Measure | 32K | 64K | 128K | 256K |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 | MuSiQue | Full HR@10 | +3.7 [+1.7, +5.7] | +5.1 [+3.2, +7.1] | +3.9 [+2.0, +5.8] | +3.8 [+1.8, +5.8] |
| BM25 | MuSiQue | Sub-doc hit@10 | +0.0 [-0.3, +0.3] | -0.2 [-0.6, +0.0] | -0.4 [-0.9, -0.1] | +0.1 [-0.6, +0.8] |
| BM25 | MuSiQue | Top-10 share | +4.4 [+3.8, +5.0] | +4.5 [+3.9, +5.1] | +5.1 [+4.5, +5.8] | +4.7 [+4.0, +5.4] |
| BM25 | MuSiQue | Inside MRR | +4.4 [+3.3, +5.6] | +4.1 [+3.0, +5.2] | +3.6 [+2.5, +4.7] | +4.1 [+3.1, +5.2] |
| BM25 | QASPER | Full HR@10 | +4.5 [+2.8, +6.3] | +4.2 [+2.7, +5.7] | +4.1 [+2.6, +5.6] | +2.6 [+1.4, +3.9] |
| BM25 | QASPER | Sub-doc hit@10 | +0.4 [-1.0, +1.8] | -0.8 [-2.4, +0.8] | +0.2 [-1.4, +1.7] | +1.4 [-0.2, +3.0] |
| BM25 | QASPER | Top-10 share | +6.7 [+6.0, +7.4] | +5.2 [+4.6, +5.9] | +5.2 [+4.5, +5.8] | +4.4 [+3.8, +5.0] |
| BM25 | QASPER | Inside MRR | +1.0 [+0.2, +1.7] | +1.3 [+0.6, +2.0] | +0.6 [-0.1, +1.3] | +1.1 [+0.4, +1.8] |
| Contriever | MuSiQue | Full HR@10 | +7.6 [+5.4, +9.7] | +9.1 [+6.9, +11.3] | +7.8 [+5.7, +10.0] | +10.0 [+7.7, +12.3] |
| Contriever | MuSiQue | Sub-doc hit@10 | +0.2 [+0.0, +0.6] | +0.3 [-0.2, +0.9] | +0.6 [-0.3, +1.3] | +1.1 [-0.1, +2.4] |
| Contriever | MuSiQue | Top-10 share | +8.7 [+8.0, +9.5] | +8.3 [+7.6, +9.1] | +8.3 [+7.5, +9.1] | +7.7 [+6.9, +8.6] |
| Contriever | MuSiQue | Inside MRR | +4.5 [+3.3, +5.7] | +4.8 [+3.6, +5.9] | +3.9 [+2.8, +5.1] | +4.8 [+3.5, +6.1] |
| Contriever | QASPER | Full HR@10 | +2.3 [+0.7, +3.8] | +2.6 [+1.1, +4.2] | +2.3 [+0.8, +3.7] | +2.0 [+0.7, +3.4] |
| Contriever | QASPER | Sub-doc hit@10 | -1.1 [-2.6, +0.3] | -3.5 [-5.1, -1.8] | -0.8 [-2.3, +0.8] | +0.4 [-1.3, +2.0] |
| Contriever | QASPER | Top-10 share | +3.7 [+3.1, +4.3] | +3.4 [+2.8, +4.0] | +3.3 [+2.8, +3.9] | +3.5 [+2.8, +4.1] |
| Contriever | QASPER | Inside MRR | -0.2 [-0.9, +0.6] | +0.6 [-0.2, +1.4] | +0.2 [-0.6, +1.0] | +1.0 [+0.2, +1.8] |

Only retrievers with scores at the longer settings appear in the trend table; the dense retrievers' full scores exist for MuSiQue-32K and QASPER-64K (C2), plus whatever was recomputed into `$C4_DENSE_STREAMS_DIR`.

## 5. Notes

- Query sets. The paper's labels come from the repository labeler run over the whole concatenated document. A span repeated in another source document gives a second sufficient chunk outside the gold sub-document, and a span that fits in no single chunk becomes an evidence group of every chunk that shares a boundary token run with it, anywhere in the document. Set S keeps only queries with all such evidence inside; set R re-derives the labels with the same labeler on the gold sub-document's chunks only (see step 01).
- The share of queries with evidence outside the gold sub-document grows with the setting length (step 01, section 3), so the paper's full-document hit rates at the longer settings also count partial-match groups that cross sub-documents.
- Vanilla's ranking is the same under both restrictions: z-scoring a single channel keeps its order.
- Files: `table.csv` (every measure), `paired.csv` (every CI), `table_paper.tex`, per-query ranks under `experiments/rebuttal/c4_subdoc_control/work/perquery/`.
