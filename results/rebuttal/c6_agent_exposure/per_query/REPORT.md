# C6: SAADI vs each agent at that query's own exposure

For every query, SAADI gets exactly as many chunks as the agent showed its LLM for that query (k_q), taken from the top of SAADI's ranking (BM25, published fusion point; identical to Table 2's ranking for the first 20, extended past 20 with the same scorer). Same labels and hit rule as Table 2; HR over all Table 2 queries. Re-scoring only: no agent or SAADI model was re-run.

| Dataset | Method | Budget | Agent HR | SAADI HR at k_q | Diff (95% CI) | SAADI only / agent only | mean k_q (max) | k_q > 20 | Table 2: SAADI at floor / ceil |
|---|---|---|---|---|---|---|---|---|---|
| LooGLE | RAG-R1 | exposure | 71.1 | 90.7 | +19.7 (+17.6, +21.8) | 416 / 56 | 4.49 (12) | 0 | k=4: 90.6 / k=5: 92.3 |
| LooGLE | Agentic-R | exposure | 54.1 | 88.6 | +34.5 (+31.9, +37.0) | 699 / 67 | 3.37 (8) | 0 | k=3: 87.9 / k=4: 90.6 |
| LooGLE | A-RAG | exposure | 68.6 | 69.4 | +0.8 (-1.3, +3.0) | 213 / 198 | 1.95 (21) | 1 | k=1: 70.1 / k=2: 83.1 |
| LooGLE | A-RAG | exposure + previews | 68.6 | 91.8 | +23.3 (+21.0, +25.5) | 485 / 59 | 4.82 (21) | 1 | k=1: 70.1 / k=2: 83.1 |
| LooGLE | A-RAG | exposure + previews, both sides | 83.7 | 91.8 | +8.1 (+6.3, +9.9) | 220 / 71 | 4.82 (21) | 1 | k=1: 70.1 / k=2: 83.1 |
| LooGLE | GRIP | exposure | 86.7 | 91.4 | +4.6 (+3.4, +5.9) | 113 / 28 | 4.57 (9) | 0 | k=4: 90.6 / k=5: 92.3 |
| LooGLE | UMA | exposure | 87.8 | 87.9 | +0.1 (-1.1, +1.3) | 66 / 64 | 3.06 (7) | 0 | k=3: 87.9 / k=4: 90.6 |
| MuSiQue-32K | RAG-R1 | exposure | 86.0 | 71.2 | -14.8 (-17.8, -11.8) | 35 / 168 | 8.09 (22) | 1 | k=8: 73.1 / k=9: 75.3 |
| MuSiQue-32K | Agentic-R | exposure | 75.2 | 63.2 | -12.0 (-15.2, -8.8) | 59 / 167 | 5.54 (13) | 0 | k=5: 62.8 / k=6: 65.8 |
| MuSiQue-32K | A-RAG | exposure | 45.1 | 44.7 | -0.4 (-3.8, +2.9) | 112 / 116 | 2.66 (13) | 0 | k=2: 41.1 / k=3: 49.2 |
| MuSiQue-32K | A-RAG | exposure + previews | 45.1 | 64.8 | +19.7 (+16.3, +23.0) | 224 / 47 | 6.47 (24) | 5 | k=2: 41.1 / k=3: 49.2 |
| MuSiQue-32K | A-RAG | exposure + previews, both sides | 57.4 | 64.8 | +7.3 (+4.2, +10.4) | 137 / 71 | 6.47 (24) | 5 | k=2: 41.1 / k=3: 49.2 |
| MuSiQue-32K | GRIP | exposure | 50.7 | 55.7 | +5.0 (+2.4, +7.7) | 95 / 50 | 3.99 (9) | 0 | k=3: 49.2 / k=4: 57.6 |
| MuSiQue-32K | UMA | exposure | 90.3 | 91.0 | +0.7 (-1.3, +2.8) | 46 / 40 | 19.59 (40) | 404 | k=19: 91.4 / k=20: 91.9 |
| QASPER-64K | RAG-R1 | exposure | 20.0 | 19.7 | -0.3 (-2.6, +2.0) | 119 / 123 | 7.64 (32) | 6 | k=7: 21.4 / k=8: 23.3 |
| QASPER-64K | Agentic-R | exposure | 16.1 | 13.1 | -3.0 (-5.2, -0.9) | 81 / 121 | 3.76 (13) | 0 | k=3: 11.7 / k=4: 14.3 |
| QASPER-64K | A-RAG | exposure | 12.6 | 12.2 | -0.5 (-2.4, +1.6) | 87 / 93 | 3.48 (6) | 0 | k=3: 11.7 / k=4: 14.3 |
| QASPER-64K | A-RAG | exposure + previews | 12.6 | 16.7 | +4.1 (+2.1, +6.2) | 126 / 71 | 4.91 (10) | 0 | k=3: 11.7 / k=4: 14.3 |
| QASPER-64K | A-RAG | exposure + previews, both sides | 13.7 | 16.7 | +3.0 (+1.0, +5.1) | 119 / 79 | 4.91 (10) | 0 | k=3: 11.7 / k=4: 14.3 |
| QASPER-64K | GRIP | exposure | 10.3 | 14.1 | +3.8 (+2.3, +5.3) | 78 / 27 | 3.91 (9) | 0 | k=3: 11.7 / k=4: 14.3 |
| QASPER-64K | UMA | exposure | 27.3 | 31.1 | +3.8 (+2.0, +5.8) | 115 / 64 | 14.79 (29) | 157 | k=14: 31.1 / k=15: 32.3 |
| NovelHopQA | RAG-R1 | exposure | 14.1 | 18.8 | +4.7 (+2.2, +7.3) | 93 / 50 | 5.93 (15) | 0 | k=5: 18.1 / k=6: 18.7 |
| NovelHopQA | Agentic-R | exposure | 12.6 | 16.3 | +3.7 (+1.2, +6.3) | 86 / 52 | 3.91 (9) | 0 | k=3: 15.4 / k=4: 16.7 |
| NovelHopQA | A-RAG | exposure | 9.0 | 16.5 | +7.5 (+5.2, +10.0) | 97 / 29 | 3.89 (14) | 0 | k=3: 15.4 / k=4: 16.7 |
| NovelHopQA | A-RAG | exposure + previews | 9.0 | 18.4 | +9.4 (+7.0, +11.8) | 107 / 22 | 5.22 (15) | 0 | k=3: 15.4 / k=4: 16.7 |
| NovelHopQA | A-RAG | exposure + previews, both sides | 10.1 | 18.4 | +8.3 (+5.8, +10.7) | 103 / 28 | 5.22 (15) | 0 | k=3: 15.4 / k=4: 16.7 |
| NovelHopQA | GRIP | exposure | 14.2 | 17.7 | +3.5 (+2.1, +5.0) | 39 / 7 | 4.95 (9) | 0 | k=4: 16.7 / k=5: 18.1 |
| NovelHopQA | UMA | exposure | 14.0 | 16.3 | +2.3 (+0.6, +4.1) | 45 / 24 | 3.72 (8) | 0 | k=3: 15.4 / k=4: 16.7 |

## Queries where the agent showed its LLM no chunk (k_q = 0)

On these queries the matched comparison gives SAADI zero chunks too, so both miss. This table shows how often that happens and how SAADI would do there with one or three chunks. Those columns are a different budget from the agent's and are not part of the comparison above; they only show what the zero budget costs SAADI. Budget = exposure (chunks the agent fully read); for A-RAG, search previews are not counted here.

| Dataset | Method | Queries | k_q = 0 (%) | of which no agent record | agent hits there | SAADI top 1 hits there | SAADI top 2 hits there | SAADI top 3 hits there | SAADI HR at k_q | SAADI HR if k_q = 0 got 1 chunk |
|---|---|---|---|---|---|---|---|---|---|---|
| LooGLE | RAG-R1 | 1832 | 6 (0.3%) | 0 | 0 | 3 | 4 | 4 | 90.7 | 90.9 |
| LooGLE | Agentic-R | 1832 | 5 (0.3%) | 0 | 0 | 4 | 5 | 5 | 88.6 | 88.8 |
| LooGLE | A-RAG | 1832 | 221 (12.1%) | 2 | 0 | 192 | 208 | 212 | 69.4 | 79.9 |
| LooGLE | GRIP | 1832 | 2 (0.1%) | 0 | 0 | 1 | 1 | 2 | 91.4 | 91.4 |
| LooGLE | UMA | 1832 | 2 (0.1%) | 0 | 0 | 0 | 1 | 2 | 87.9 | 87.9 |
| MuSiQue-32K | RAG-R1 | 900 | 2 (0.2%) | 0 | 0 | 2 | 2 | 2 | 71.2 | 71.4 |
| MuSiQue-32K | Agentic-R | 900 | 3 (0.3%) | 0 | 0 | 0 | 1 | 1 | 63.2 | 63.2 |
| MuSiQue-32K | A-RAG | 900 | 26 (2.9%) | 0 | 0 | 6 | 8 | 10 | 44.7 | 45.3 |
| MuSiQue-32K | GRIP | 900 | 2 (0.2%) | 0 | 0 | 0 | 0 | 0 | 55.7 | 55.7 |
| MuSiQue-32K | UMA | 900 | 0 (0.0%) | 0 | 0 | 0 | 0 | 0 | 91.0 | 91.0 |
| QASPER-64K | RAG-R1 | 1333 | 3 (0.2%) | 0 | 0 | 0 | 0 | 0 | 19.7 | 19.7 |
| QASPER-64K | Agentic-R | 1333 | 8 (0.6%) | 0 | 0 | 1 | 1 | 1 | 13.1 | 13.2 |
| QASPER-64K | A-RAG | 1333 | 21 (1.6%) | 0 | 0 | 2 | 3 | 4 | 12.2 | 12.3 |
| QASPER-64K | GRIP | 1333 | 23 (1.7%) | 0 | 0 | 2 | 2 | 3 | 14.1 | 14.3 |
| QASPER-64K | UMA | 1333 | 12 (0.9%) | 12 | 0 | 1 | 1 | 1 | 31.1 | 31.2 |
| NovelHopQA | RAG-R1 | 908 | 2 (0.2%) | 0 | 0 | 0 | 0 | 0 | 18.8 | 18.8 |
| NovelHopQA | Agentic-R | 908 | 12 (1.3%) | 0 | 0 | 0 | 0 | 0 | 16.3 | 16.3 |
| NovelHopQA | A-RAG | 908 | 0 (0.0%) | 0 | 0 | 0 | 0 | 0 | 16.5 | 16.5 |
| NovelHopQA | GRIP | 908 | 0 (0.0%) | 0 | 0 | 0 | 0 | 0 | 17.7 | 17.7 |
| NovelHopQA | UMA | 908 | 0 (0.0%) | 0 | 0 | 0 | 0 | 0 | 16.3 | 16.3 |

## A-RAG search previews

A-RAG's search tool first returns previews (a few matched sentences per chunk); the agent then reads some chunks in full. The main budget counts only full reads. *Budget only*: previewed chunks are added to SAADI's budget, A-RAG is still scored on its full reads. *Both sides*: previewed chunks are added to SAADI's budget and also count as chunks A-RAG has seen. Diff = SAADI HR minus A-RAG HR (95% CI).

| Dataset | Read in full per query | Previewed only per query | Diff, full reads | Diff, previews budget only | Diff, previews both sides |
|---|---|---|---|---|---|
| LooGLE | 1.95 | 2.87 | +0.8 (-1.3, +3.0) | +23.3 (+21.0, +25.5) | +8.1 (+6.3, +9.9) |
| MuSiQue-32K | 2.66 | 3.81 | -0.4 (-3.8, +2.9) | +19.7 (+16.3, +23.0) | +7.3 (+4.2, +10.4) |
| QASPER-64K | 3.48 | 1.43 | -0.5 (-2.4, +1.6) | +4.1 (+2.1, +6.2) | +3.0 (+1.0, +5.1) |
| NovelHopQA | 3.89 | 1.34 | +7.5 (+5.2, +10.0) | +9.4 (+7.0, +11.8) | +8.3 (+5.8, +10.7) |

## Checks

Each agent's exposure list is the saved list that reproduces the paper's own Table 2 numbers (mean chunks and HR, over the same queries).

| Dataset | Method | Source | List | mean |list| (paper) | agent HR (paper) | selection |
|---|---|---|---|---|---|---|
| LooGLE | RAG-R1 | `rag_r1_eval_normal__loogle__per_query` | `ordered_exposed_chunk_ids` | 4.4885 (4.4885) | 71.070 (71.070) | exact |
| LooGLE | Agentic-R | `agentic_r_eval__loogle__per_query` | `ordered_exposed_chunk_ids` | 3.3695 (3.3695) | 54.094 (54.094) | exact |
| LooGLE | A-RAG | `loogle__arag__loogle_a_rag_raw_ea3f88232a625841` | `retrieved_chunk_ids` | 1.9460 (1.9460) | 68.559 (68.559) | exact |
| LooGLE | GRIP | `grip_eval_normal__loogle__per_query` | `ordered_exposed_chunk_ids` | 4.5688 (4.5688) | 86.736 (86.736) | exact |
| LooGLE | UMA | `uma_eval_normal__loogle__per_query` | `ordered_exposed_chunk_ids` | 3.0562 (3.0562) | 87.828 (87.828) | exact |
| MuSiQue-32K | RAG-R1 | `rag_r1_eval_normal__musique_32k__per_query` | `ordered_exposed_chunk_ids` | 8.0878 (8.0878) | 86.000 (86.000) | exact |
| MuSiQue-32K | Agentic-R | `agentic_r_eval__musique_32k__per_query` | `ordered_exposed_chunk_ids` | 5.5444 (5.5444) | 75.222 (75.222) | exact |
| MuSiQue-32K | A-RAG | `musique_32k__arag__musique_32k_a_rag_raw_f06714eda6267987` | `retrieved_chunk_ids` | 2.6622 (2.6622) | 45.111 (45.111) | exact |
| MuSiQue-32K | GRIP | `grip_eval_normal__musique_32k__per_query` | `ordered_exposed_chunk_ids` | 3.9856 (3.9856) | 50.667 (50.667) | exact |
| MuSiQue-32K | UMA | `uma_context_matrix__musique_32k__per_query` | `ordered_exposed_chunk_ids` | 19.5878 (19.5878) | 90.333 (90.333) | exact |
| QASPER-64K | RAG-R1 | `rag_r1_eval_normal__qasper_64k__per_query` | `ordered_exposed_chunk_ids` | 7.6377 (7.6377) | 19.955 (19.955) | exact |
| QASPER-64K | Agentic-R | `agentic_r_eval__qasper_64k__per_query` | `ordered_exposed_chunk_ids` | 3.7592 (3.7592) | 16.129 (16.129) | exact |
| QASPER-64K | A-RAG | `qasper_64k__arag__qasper_64k_a_rag_raw_c289b860993c714a` | `retrieved_chunk_ids` | 3.4816 (3.4816) | 12.603 (12.603) | exact |
| QASPER-64K | GRIP | `grip_eval_normal__qasper_64k__per_query` | `ordered_exposed_chunk_ids` | 3.9145 (3.9145) | 10.278 (10.278) | exact |
| QASPER-64K | UMA | `uma_context_matrix__qasper_64k__per_query` | `ordered_exposed_chunk_ids` | 14.7899 (14.7899) | 27.307 (27.307) | exact |
| NovelHopQA | RAG-R1 | `rag_r1_eval_normal__novelhopqa__per_query` | `ordered_exposed_chunk_ids` | 5.9295 (5.9295) | 14.097 (14.097) | exact |
| NovelHopQA | Agentic-R | `agentic_r_eval__novelhopqa__per_query` | `ordered_exposed_chunk_ids` | 3.9086 (3.9086) | 12.555 (12.555) | exact |
| NovelHopQA | A-RAG | `novelhopqa__arag__novelhopqa_a_rag_raw_479355014fce5066` | `retrieved_chunk_ids` | 3.8855 (3.8855) | 9.031 (9.031) | exact |
| NovelHopQA | GRIP | `grip_eval_normal__novelhopqa__per_query` | `ordered_exposed_chunk_ids` | 4.9548 (4.9548) | 14.207 (14.207) | exact |
| NovelHopQA | UMA | `uma_eval_normal__novelhopqa__per_query` | `ordered_exposed_chunk_ids` | 3.7203 (3.7203) | 13.987 (13.987) | exact |

Columns: *Diff* = SAADI HR minus agent HR, points, paired bootstrap over queries (10,000 resamples, seed 42). *SAADI only / agent only* = queries only one of the two hits. *k_q > 20* = queries that needed SAADI's ranking past the saved depth 20. *exposure + previews* (A-RAG) = the budget also counts chunks A-RAG only saw as search previews; A-RAG's hit is still from the chunks it read, as in the paper. *exposure + previews, both sides* (A-RAG) = the same budget, and A-RAG's hit also counts the previewed chunks.
