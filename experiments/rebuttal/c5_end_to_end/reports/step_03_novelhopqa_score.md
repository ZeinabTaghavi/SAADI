# C5 end-to-end QA - step 03 scores, NovelHopQA: OK

Written 2026-11-10 23:39:55; commit cfe1d4ff30de; host eee-s132-infsys-gpu03.

Scores in percent. EM and token F1 after SQuAD normalisation (best over the gold answers); Contain = a gold answer inside the output. Evidence yes/no = the paper's hit rule (a gold chunk or a whole silver group) applied to the chunks in the context. HR@10 = the condition's retrieval hit (RAPTOR: its upper-bound HR@10; A-RAG: hit over the chunks it read; none for the cut SAADI rows, whose retrieval measure is the evidence in the cut context).

| Condition | n | EM | F1 | Contain | F1 evidence yes (n) | F1 evidence no (n) | F1 no labels (n) | HR@10 | Evidence in context | Context tokens mean / max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Closed-book | 908 | 0.9 | 12.6 | 3.0 | — (0) | 12.6 (908) | — (0) | — | 0.0 | 0 / 0 |
| Oracle (gold evidence chunks) | 908 | 7.4 | 30.3 | 21.8 | 30.3 (908) | — (0) | — (0) | — | 100.0 | 2302 / 30097 |
| Oracle (chunks of every evidence paragraph) | 908 | 11.2 | 40.7 | 30.6 | 39.4 (439) | 41.9 (469) | — (0) | — | 48.3 | 1793 / 7600 |
| Vanilla BM25 top 10 | 908 | 7.7 | 30.8 | 21.9 | 36.2 (188) | 29.4 (720) | — (0) | 20.7 | 20.7 | 6442 / 7445 |
| SAADI top 10 | 908 | 7.6 | 31.5 | 22.6 | 36.3 (200) | 30.2 (708) | — (0) | 22.0 | 22.0 | 6447 / 7452 |
| Contextual Retrieval top 10 | 908 | 7.2 | 29.9 | 21.9 | 36.5 (197) | 28.0 (711) | — (0) | 21.7 | 21.7 | 6426 / 7441 |
| RAPTOR (default, as retrieved) | 908 | 2.2 | 17.6 | 9.1 | 20.7 (138) | 16.9 (647) | 18.3 (123) | 8.7 | 17.6 | 1701 / 3399 |
| RAPTOR, token-matched to SAADI | 908 | 2.4 | 17.6 | 9.1 | 20.5 (138) | 16.8 (647) | 18.6 (123) | 8.7 | 17.6 | 1701 / 3399 |
| A-RAG read chunks (as read) | 908 | 4.5 | 21.4 | 13.2 | 34.9 (82) | 20.1 (826) | — (0) | 9.0 | 9.0 | 2548 / 9130 |
| A-RAG read chunks, token-matched to SAADI | 908 | 4.3 | 21.3 | 13.3 | 35.0 (82) | 19.9 (826) | — (0) | 9.0 | 9.0 | 2544 / 6446 |
| SAADI cut to RAPTOR's length (per query) | 908 | 5.8 | 27.7 | 17.2 | 38.4 (122) | 26.0 (786) | — (0) | — | 13.4 | 1700 / 3398 |
| SAADI cut to A-RAG's length (per query) | 908 | 6.9 | 28.5 | 18.4 | 37.8 (146) | 26.7 (762) | — (0) | — | 16.1 | 2544 / 6916 |
| A-RAG own final answer | 908 | 0.1 | 9.7 | 12.0 | 13.6 (82) | 9.3 (826) | — (0) | 9.0 | 9.0 | 2548 / 9130 |

Token-matched rows are cut to 6447 tokens (mean SAADI top-10 context).

Paired differences (points; 95% bootstrap interval, 10,000 resamples over queries, seed 42):

| Pair | n | EM | F1 | Contain | HR@10 | Evidence in context | Evidence, paragraph labels (all) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SAADI top 10 − Vanilla BM25 top 10 | 908 | -0.1 [-1.1, +0.9] | +0.7 [-0.5, +1.8] | +0.7 [-0.8, +2.1] | +1.3 [+0.2, +2.4] | +1.3 [+0.2, +2.4] | +2.1 [+0.7, +3.6] |
| SAADI top 10 − Contextual Retrieval top 10 | 908 | +0.4 [-0.6, +1.4] | +1.7 [+0.5, +2.9] | +0.7 [-1.1, +2.4] | +0.3 [-1.1, +1.8] | +0.3 [-1.1, +1.8] | +1.8 [-0.1, +3.6] |
| SAADI top 10 − RAPTOR (default, as retrieved) | 908 | +5.4 [+3.6, +7.2] | +13.9 [+11.9, +15.9] | +13.4 [+10.8, +16.1] | +11.1 [+8.0, +14.1] | +2.2 [-1.3, +5.5] | +35.0 [+31.9, +38.2] |
| SAADI top 10 − A-RAG read chunks (as read) | 908 | +3.1 [+1.5, +4.6] | +10.1 [+8.2, +12.1] | +9.4 [+6.9, +11.8] | +13.0 [+10.6, +15.4] | +13.0 [+10.6, +15.4] | +22.3 [+19.3, +25.4] |
| SAADI cut to RAPTOR's length (per query) − RAPTOR (default, as retrieved) | 908 | +3.6 [+2.1, +5.3] | +10.1 [+8.1, +12.0] | +8.0 [+5.7, +10.4] | — | -5.0 [-8.3, -1.8] | +19.0 [+16.4, +21.5] |
| SAADI cut to A-RAG's length (per query) − A-RAG read chunks (as read) | 908 | +2.4 [+1.0, +3.9] | +7.0 [+5.2, +8.9] | +5.2 [+2.9, +7.5] | — | +7.0 [+4.7, +9.4] | +10.3 [+7.4, +13.2] |
| SAADI top 10 − RAPTOR, token-matched to SAADI | 908 | +5.2 [+3.5, +6.9] | +13.9 [+11.9, +15.9] | +13.4 [+10.9, +16.1] | +11.1 [+8.0, +14.1] | +2.2 [-1.3, +5.5] | +35.0 [+31.9, +38.2] |
| SAADI top 10 − A-RAG read chunks, token-matched to SAADI | 908 | +3.3 [+1.8, +4.8] | +10.2 [+8.3, +12.2] | +9.3 [+6.8, +11.6] | +13.0 [+10.6, +15.4] | +13.0 [+10.6, +15.4] | +22.3 [+19.3, +25.4] |
| SAADI top 10 − A-RAG own final answer | 908 | +7.5 [+5.7, +9.3] | +21.8 [+20.0, +23.7] | +10.6 [+7.9, +13.2] | +13.0 [+10.6, +15.4] | +13.0 [+10.6, +15.4] | +22.3 [+19.3, +25.4] |
| SAADI top 10 − Closed-book | 908 | +6.7 [+5.0, +8.6] | +18.9 [+16.7, +21.0] | +19.6 [+17.0, +22.4] | — | +22.0 [+19.4, +24.8] | +40.6 [+37.4, +43.9] |
| Oracle (gold evidence chunks) − SAADI top 10 | 908 | -0.2 [-1.8, +1.4] | -1.2 [-3.1, +0.8] | -0.8 [-3.3, +1.8] | — | +78.0 [+75.2, +80.6] | +1.2 [-1.7, +4.2] |
| Oracle (chunks of every evidence paragraph) − SAADI top 10 | 908 | +3.6 [+2.0, +5.3] | +9.2 [+7.3, +11.0] | +8.0 [+5.6, +10.5] | — | +26.3 [+23.5, +29.3] | +59.4 [+56.2, +62.5] |

Paragraph labels (each evidence paragraph of the dataset located on its own in SAADI's chunks; see step 00). Evidence any / all = at least one / every located paragraph is in the context (full chunks or RAPTOR leaf text; summaries and cut passages do not count); for the top-10 rows this is HR@10 with these labels.

| Condition | n | Evidence any % | Evidence all % | F1 all paragraphs in context (n) | F1 not all (n) | F1 1-hop (n) | F1 2-4 hops (n) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Closed-book | 908 | 0.0 | 0.0 | — (0) | 12.6 (891) | 8.1 (377) | 15.9 (531) |
| Oracle (gold evidence chunks) | 908 | 82.0 | 41.9 | 47.5 (373) | 18.7 (518) | 46.6 (377) | 18.8 (531) |
| Oracle (chunks of every evidence paragraph) | 908 | 100.0 | 100.0 | 41.2 (891) | — (0) | 48.5 (377) | 35.1 (531) |
| Vanilla BM25 top 10 | 908 | 80.4 | 38.5 | 41.3 (343) | 24.4 (548) | 34.3 (377) | 28.3 (531) |
| SAADI top 10 | 908 | 82.8 | 40.6 | 41.9 (362) | 24.7 (529) | 35.7 (377) | 28.6 (531) |
| Contextual Retrieval top 10 | 908 | 80.4 | 38.8 | 41.3 (346) | 22.9 (545) | 33.5 (377) | 27.2 (531) |
| RAPTOR (default, as retrieved) | 908 | 9.9 | 5.6 | 51.2 (50) | 15.6 (841) | 16.3 (377) | 18.6 (531) |
| RAPTOR, token-matched to SAADI | 908 | 9.9 | 5.6 | 50.6 (50) | 15.5 (841) | 16.4 (377) | 18.5 (531) |
| A-RAG read chunks (as read) | 908 | 39.3 | 18.3 | 46.2 (163) | 16.0 (728) | 23.6 (377) | 19.9 (531) |
| A-RAG read chunks, token-matched to SAADI | 908 | 39.3 | 18.3 | 45.9 (163) | 15.9 (728) | 23.5 (377) | 19.7 (531) |
| SAADI cut to RAPTOR's length (per query) | 908 | 61.6 | 24.6 | 46.5 (219) | 21.8 (672) | 29.4 (377) | 26.5 (531) |
| SAADI cut to A-RAG's length (per query) | 908 | 66.9 | 28.6 | 44.4 (255) | 22.3 (636) | 30.7 (377) | 26.9 (531) |
| A-RAG own final answer | 908 | 39.3 | 18.3 | 17.6 (163) | 7.9 (728) | 10.3 (377) | 9.3 (531) |

## Files written

- `results/rebuttal/c5_end_to_end/per_dataset/novelhopqa/table.csv`
- `results/rebuttal/c5_end_to_end/per_dataset/novelhopqa/paired.csv`
- `results/rebuttal/c5_end_to_end/per_dataset/novelhopqa/summary.json`
- `experiments/rebuttal/c5_end_to_end/reports/step_03_novelhopqa_score.md`

## Push

```
pushed dcc0f9b to origin/fix/notebook-key-and-install: C5 novelhopqa 03_score: F1 SAADI 31.5, SAADI-Vanilla +0.7 [-0.5, +1.8]
```
