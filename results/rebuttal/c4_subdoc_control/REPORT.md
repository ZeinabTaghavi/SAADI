# C4: SAADI vs Vanilla inside the gold sub-document (reviewer iVU8, Q2)

## Summary for the rebuttal

To separate SAADI's within-document ranking from sub-document selection, we re-ranked only the chunks of each query's gold sub-document in every concatenated MuSiQue and QASPER setting (32K to 256K tokens), for 8 setting-retriever pairs (BM25), comparing SAADI with Vanilla (the text channel alone) on identical chunk scores. On the whole document both methods first reproduce the paper's saved rankings and hit rates exactly. On all queries, with the evidence labels re-derived inside the gold sub-document by the paper's own labeler, SAADI's MRR inside the gold sub-document is higher than Vanilla's in 8 of 8 pairs, and the 95% bootstrap CI of the gain lies fully above zero in 7. The in-sub-document MRR gain ranges from +0.6 to +4.4 points (median +2.5); inside HR@5 rises in 8 of 8 pairs. SAADI does not reach the gold sub-document more often than Vanilla: whether at least one of its top-10 chunks comes from it is higher in 4 of 8 pairs, with the CI above zero in 0; but it fills more of its top 10 with that sub-document's chunks in 8 of 8 pairs (+4.4 to +6.7 points, CI above zero in 8), so the full-document gain comes both from favouring the right sub-document and from ranking better inside it. For BM25 the in-sub-document MRR gain is MuSiQue +4.4 at 32K, +4.1 at 64K, +3.6 at 128K, +4.1 at 256K; QASPER +1.0 at 32K, +1.3 at 64K, +0.6 at 128K, +1.1 at 256K. Restricting to the queries whose original evidence lies entirely inside the gold sub-document (no relabeling) gives the same picture: an MRR gain in 8 of 8 pairs (CI above zero in 5). Renormalising the scores over the gold sub-document only, instead of keeping the full-document scores, gives a gain in 8 of 8 pairs (CI above zero in 8).

Not included: (step 04 not run: no dense retriever included).

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

Reading: "sub-document selection" when the CI of the sub-document hit@10 or of the top-10 share is above zero, "inside ranking" when the inside-MRR CI is above zero, "both" when both are. A fixed rule for reading the table, not a test of a decomposition.

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

## 4. Across pairs

| Query set | Restriction | Pairs | Inside MRR gain > 0 | CI fully above 0 | CI fully below 0 |
| --- | --- | --- | --- | --- | --- |
| R | keep | 8 | 8 | 7 | 0 |
| R | renorm | 8 | 8 | 8 | 0 |
| S | keep | 8 | 8 | 5 | 0 |
| S | renorm | 8 | 8 | 8 | 0 |

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

Only retrievers with scores at the longer settings appear in the trend table; the dense retrievers' full scores exist for MuSiQue-32K and QASPER-64K (C2), plus whatever was recomputed into `$C4_DENSE_STREAMS_DIR`.

## 5. Notes

- Query sets. The paper's labels come from the repository labeler run over the whole concatenated document. A span repeated in another source document gives a second sufficient chunk outside the gold sub-document, and a span that fits in no single chunk becomes an evidence group of every chunk that shares a boundary token run with it, anywhere in the document. Set S keeps only queries with all such evidence inside; set R re-derives the labels with the same labeler on the gold sub-document's chunks only (see step 01).
- The share of queries with evidence outside the gold sub-document grows with the setting length (step 01, section 3), so the paper's full-document hit rates at the longer settings also count partial-match groups that cross sub-documents.
- Vanilla's ranking is the same under both restrictions: z-scoring a single channel keeps its order.
- Files: `table.csv` (every measure), `paired.csv` (every CI), `table_paper.tex`, per-query ranks under `experiments/rebuttal/c4_subdoc_control/work/perquery/`.
