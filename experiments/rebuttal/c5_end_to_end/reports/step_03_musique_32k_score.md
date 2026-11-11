# C5 end-to-end QA - step 03 scores, MuSiQue-32K: OK

Written 2026-11-11 02:01:16; commit 619580cf910d; host eee-s132-infsys-gpu03.

Scores in percent. EM and token F1 after SQuAD normalisation (best over the gold answers); Contain = a gold answer inside the output. Evidence yes/no = the paper's hit rule (a gold chunk or a whole silver group) applied to the chunks in the context. HR@10 = the condition's retrieval hit (RAPTOR: its upper-bound HR@10; A-RAG: hit over the chunks it read; none for the cut SAADI rows, whose retrieval measure is the evidence in the cut context).

| Condition | n | EM | F1 | Contain | F1 evidence yes (n) | F1 evidence no (n) | F1 no labels (n) | HR@10 | Evidence in context | Context tokens mean / max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Closed-book | 900 | 2.4 | 9.3 | 9.2 | — (0) | 9.3 (900) | — (0) | — | 0.0 | 0 / 0 |
| Oracle (gold evidence chunks) | 900 | 24.3 | 39.7 | 50.2 | 39.7 (900) | — (0) | — (0) | — | 100.0 | 834 / 4434 |
| Vanilla BM25 top 10 | 900 | 10.3 | 21.9 | 29.7 | 24.1 (666) | 15.5 (234) | — (0) | 74.0 | 74.0 | 1327 / 1765 |
| SAADI top 10 | 900 | 11.4 | 22.8 | 30.3 | 24.5 (699) | 16.9 (201) | — (0) | 77.7 | 77.7 | 1326 / 2016 |
| Contextual Retrieval top 10 | 900 | 10.8 | 22.8 | 30.8 | 24.4 (740) | 15.8 (160) | — (0) | 82.2 | 82.2 | 1332 / 2041 |
| RAPTOR (default, as retrieved) | 900 | 9.1 | 21.7 | 27.8 | 24.7 (643) | 14.2 (257) | — (0) | 65.6 | 71.4 | 923 / 1218 |
| RAPTOR, token-matched to SAADI | 900 | 9.2 | 21.6 | 27.2 | 24.4 (643) | 14.7 (257) | — (0) | 65.6 | 71.4 | 923 / 1218 |
| A-RAG read chunks (as read) | 900 | 11.6 | 22.2 | 24.2 | 30.4 (406) | 15.4 (494) | — (0) | 45.1 | 45.1 | 360 / 1676 |
| A-RAG read chunks, token-matched to SAADI | 900 | 11.6 | 22.0 | 24.2 | 30.4 (406) | 15.1 (494) | — (0) | 45.1 | 45.1 | 359 / 1325 |
| SAADI cut to RAPTOR's length (per query) | 900 | 9.4 | 21.0 | 27.0 | 23.5 (604) | 16.0 (296) | — (0) | — | 67.1 | 922 / 1215 |
| SAADI cut to A-RAG's length (per query) | 900 | 5.8 | 15.6 | 17.7 | 19.6 (368) | 12.8 (532) | — (0) | — | 40.9 | 358 / 1389 |
| A-RAG own final answer | 900 | 2.9 | 12.7 | 32.8 | 16.0 (406) | 9.9 (494) | — (0) | 45.1 | 45.1 | 360 / 1676 |

Token-matched rows are cut to 1326 tokens (mean SAADI top-10 context).

Paired differences (points; 95% bootstrap interval, 10,000 resamples over queries, seed 42):

| Pair | n | EM | F1 | Contain | HR@10 | Evidence in context | Evidence, paragraph labels (all) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SAADI top 10 − Vanilla BM25 top 10 | 900 | +1.1 [-0.4, +2.8] | +1.0 [-0.4, +2.3] | +0.7 [-1.3, +2.8] | +3.7 [+1.7, +5.7] | +3.7 [+1.7, +5.7] | — |
| SAADI top 10 − Contextual Retrieval top 10 | 900 | +0.7 [-1.0, +2.3] | -0.0 [-1.5, +1.5] | -0.4 [-2.8, +1.9] | -4.6 [-6.8, -2.4] | -4.6 [-6.8, -2.4] | — |
| SAADI top 10 − RAPTOR (default, as retrieved) | 900 | +2.3 [+0.3, +4.4] | +1.1 [-0.9, +3.2] | +2.6 [-0.4, +5.7] | +12.1 [+8.4, +15.7] | +6.2 [+2.7, +9.8] | — |
| SAADI top 10 − A-RAG read chunks (as read) | 900 | -0.1 [-2.4, +2.1] | +0.7 [-1.6, +2.9] | +6.1 [+2.8, +9.3] | +32.6 [+29.0, +36.0] | +32.6 [+29.0, +36.0] | — |
| SAADI cut to RAPTOR's length (per query) − RAPTOR (default, as retrieved) | 900 | +0.3 [-1.6, +2.2] | -0.7 [-2.6, +1.3] | -0.8 [-3.7, +2.2] | — | -4.3 [-7.9, -0.7] | — |
| SAADI cut to A-RAG's length (per query) − A-RAG read chunks (as read) | 900 | -5.8 [-7.7, -3.9] | -6.6 [-8.6, -4.7] | -6.6 [-9.3, -3.8] | — | -4.2 [-7.4, -1.1] | — |
| SAADI top 10 − RAPTOR, token-matched to SAADI | 900 | +2.2 [+0.1, +4.3] | +1.2 [-0.8, +3.3] | +3.1 [+0.1, +6.1] | +12.1 [+8.4, +15.7] | +6.2 [+2.7, +9.8] | — |
| SAADI top 10 − A-RAG read chunks, token-matched to SAADI | 900 | -0.1 [-2.3, +2.1] | +0.8 [-1.4, +3.1] | +6.1 [+2.8, +9.3] | +32.6 [+29.0, +36.0] | +32.6 [+29.0, +36.0] | — |
| SAADI top 10 − A-RAG own final answer | 900 | +8.6 [+6.3, +10.8] | +10.2 [+8.1, +12.3] | -2.4 [-6.0, +1.1] | +32.6 [+29.0, +36.0] | +32.6 [+29.0, +36.0] | — |
| SAADI top 10 − Closed-book | 900 | +9.0 [+6.9, +11.1] | +13.5 [+11.4, +15.6] | +21.1 [+18.1, +24.2] | — | +77.7 [+74.8, +80.3] | — |
| Oracle (gold evidence chunks) − SAADI top 10 | 900 | +12.9 [+10.3, +15.4] | +16.9 [+14.6, +19.2] | +19.9 [+16.7, +23.1] | — | +22.3 [+19.7, +25.1] | — |

## Files written

- `results/rebuttal/c5_end_to_end/per_dataset/musique_32k/table.csv`
- `results/rebuttal/c5_end_to_end/per_dataset/musique_32k/paired.csv`
- `results/rebuttal/c5_end_to_end/per_dataset/musique_32k/summary.json`
- `experiments/rebuttal/c5_end_to_end/reports/step_03_musique_32k_score.md`

## Push

```
pushed 64775a9 to origin/fix/notebook-key-and-install: C5 musique_32k 03_score: F1 SAADI 22.8, SAADI-Vanilla +1.0 [-0.4, +2.3]
```
