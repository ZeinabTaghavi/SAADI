# C4 sub-document control - step 05: paired bootstrap and final report (OK)

Command (from the repository root on the server):

```bash
bash experiments/rebuttal/c4_subdoc_control/sh/05_report.sh
```

Seed 42. CPU only. Run time 10 s.
Pairs: 16 (musique_32k__bm25, musique_64k__bm25, musique_128k__bm25, musique_256k__bm25, qasper_32k__bm25, qasper_64k__bm25, qasper_128k__bm25, qasper_256k__bm25, musique_32k__contriever, musique_64k__contriever, musique_128k__contriever, musique_256k__contriever, qasper_32k__contriever, qasper_64k__contriever, qasper_128k__contriever, qasper_256k__contriever). Main set R. Full tables: `results/rebuttal/c4_subdoc_control/REPORT.md`.

## Summary (as written at the top of REPORT.md)

To separate SAADI's within-document ranking from sub-document selection, we re-ranked only the chunks of each query's gold sub-document in every concatenated MuSiQue and QASPER setting (32K to 256K tokens), for 16 setting-retriever pairs (BM25, Contriever), comparing SAADI with Vanilla (the text channel alone) on identical chunk scores. On the whole document both methods first reproduce the paper's saved rankings and hit rates exactly (identical top-k lists for every query and identical hit rates). On all queries, with the evidence labels re-derived inside the gold sub-document by the paper's own labeler, SAADI's MRR inside the gold sub-document is higher than Vanilla's in 15 of 16 pairs, and the 95% bootstrap CI of the gain lies fully above zero in 12. The in-sub-document MRR gain ranges from -0.2 to +4.8 points (median +2.5); inside HR@5 rises in 13 of 16 pairs. SAADI does not reach the gold sub-document more often than Vanilla: whether at least one of its top-10 chunks comes from it is higher in 9 of 16 pairs, with the CI above zero in 0; but it fills more of its top 10 with that sub-document's chunks in 16 of 16 pairs (+3.3 to +8.7 points, CI above zero in 16), so the full-document gain comes both from favouring the right sub-document and from ranking better inside it. From 32K to 256K the in-sub-document MRR gain is, for BM25: MuSiQue +4.4 at 32K, +4.1 at 64K, +3.6 at 128K, +4.1 at 256K; QASPER +1.0 at 32K, +1.3 at 64K, +0.6 at 128K, +1.1 at 256K / for Contriever: MuSiQue +4.5 at 32K, +4.8 at 64K, +3.9 at 128K, +4.8 at 256K; QASPER -0.2 at 32K, +0.6 at 64K, +0.2 at 128K, +1.0 at 256K. Restricting to the queries whose original evidence lies entirely inside the gold sub-document (no relabeling) gives the same picture: an MRR gain in 15 of 16 pairs (CI above zero in 9). Renormalising the scores over the gold sub-document only, instead of keeping the full-document scores, gives a gain in 15 of 16 pairs (CI above zero in 13).

## Decomposition (set R)

| Pair | Full HR@10 gain | Sub-doc hit@10 gain | Top-10 share gain | Inside MRR gain | Reading |
| --- | --- | --- | --- | --- | --- |
| musique_32k__bm25 | +3.7 [+1.7, +5.7] | +0.0 [-0.3, +0.3] | +4.4 [+3.8, +5.0] | +4.4 [+3.3, +5.6] | full gain; both |
| musique_64k__bm25 | +5.1 [+3.2, +7.1] | -0.2 [-0.6, +0.0] | +4.5 [+3.9, +5.1] | +4.1 [+3.0, +5.2] | full gain; both |
| musique_128k__bm25 | +3.9 [+2.0, +5.8] | -0.4 [-0.9, -0.1] | +5.1 [+4.5, +5.8] | +3.6 [+2.5, +4.7] | full gain; both |
| musique_256k__bm25 | +3.8 [+1.8, +5.8] | +0.1 [-0.6, +0.8] | +4.7 [+4.0, +5.4] | +4.1 [+3.1, +5.2] | full gain; both |
| qasper_32k__bm25 | +4.5 [+2.8, +6.3] | +0.4 [-1.0, +1.8] | +6.7 [+6.0, +7.4] | +1.0 [+0.2, +1.7] | full gain; both |
| qasper_64k__bm25 | +4.2 [+2.7, +5.7] | -0.8 [-2.4, +0.8] | +5.2 [+4.6, +5.9] | +1.3 [+0.6, +2.0] | full gain; both |
| qasper_128k__bm25 | +4.1 [+2.6, +5.6] | +0.2 [-1.4, +1.7] | +5.2 [+4.5, +5.8] | +0.6 [-0.1, +1.3] | full gain; sub-document selection |
| qasper_256k__bm25 | +2.6 [+1.4, +3.9] | +1.4 [-0.2, +3.0] | +4.4 [+3.8, +5.0] | +1.1 [+0.4, +1.8] | full gain; both |
| musique_32k__contriever | +7.6 [+5.4, +9.7] | +0.2 [+0.0, +0.6] | +8.7 [+8.0, +9.5] | +4.5 [+3.3, +5.7] | full gain; both |
| musique_64k__contriever | +9.1 [+6.9, +11.3] | +0.3 [-0.2, +0.9] | +8.3 [+7.6, +9.1] | +4.8 [+3.6, +5.9] | full gain; both |
| musique_128k__contriever | +7.8 [+5.7, +10.0] | +0.6 [-0.3, +1.3] | +8.3 [+7.5, +9.1] | +3.9 [+2.8, +5.1] | full gain; both |
| musique_256k__contriever | +10.0 [+7.7, +12.3] | +1.1 [-0.1, +2.4] | +7.7 [+6.9, +8.6] | +4.8 [+3.5, +6.1] | full gain; both |
| qasper_32k__contriever | +2.3 [+0.7, +3.8] | -1.1 [-2.6, +0.3] | +3.7 [+3.1, +4.3] | -0.2 [-0.9, +0.6] | full gain; sub-document selection |
| qasper_64k__contriever | +2.6 [+1.1, +4.2] | -3.5 [-5.1, -1.8] | +3.4 [+2.8, +4.0] | +0.6 [-0.2, +1.4] | full gain; sub-document selection |
| qasper_128k__contriever | +2.3 [+0.8, +3.7] | -0.8 [-2.3, +0.8] | +3.3 [+2.8, +3.9] | +0.2 [-0.6, +1.0] | full gain; sub-document selection |
| qasper_256k__contriever | +2.0 [+0.7, +3.4] | +0.4 [-1.3, +2.0] | +3.5 [+2.8, +4.1] | +1.0 [+0.2, +1.8] | full gain; both |


## Across pairs

| Set | Restriction | Pairs | MRR gain > 0 | CI above 0 | CI below 0 |
| --- | --- | --- | --- | --- | --- |
| R | keep | 16 | 15 | 12 | 0 |
| R | renorm | 16 | 15 | 13 | 0 |
| S | keep | 16 | 15 | 9 | 0 |
| S | renorm | 16 | 15 | 11 | 0 |

Not included: musique_32k__bge_m3, musique_32k__jina, musique_32k__qwen, qasper_64k__bge_m3, qasper_64k__jina, qasper_64k__qwen.

## Files written

- `experiments/rebuttal/c4_subdoc_control/reports/step_05.json`
- `experiments/rebuttal/c4_subdoc_control/reports/step_05.md`
- `results/rebuttal/c4_subdoc_control/REPORT.md`
- `results/rebuttal/c4_subdoc_control/paired.csv`
- `results/rebuttal/c4_subdoc_control/table.csv`
- `results/rebuttal/c4_subdoc_control/table_paper.tex`
