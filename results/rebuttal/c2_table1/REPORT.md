# Table 1 with held-out fusion weights

## Summary

Mean rank: published 1.68, held-out 1.62. First or second in 35 of 40 cells (published) and 37 of 40 cells (held-out). With held-out weights SAADI improves over Vanilla in 18 of 20 pairs at HR@10, significantly in 10; at HR@5 in 14 of 20, significantly in 6.

First in 18 of 40 cells with the published weights and 18 of 40 with the held-out weights; 4 cell(s) change rank. Retrievers: BM25, Contriever, BGE-M3, Jina, Qwen (20 dataset-retriever pairs, 40 cells: HR@5 and HR@10 per pair).

Check: SAADI with the published weights prints the Table 1 value in 33 of 40 cells. The 7 other cell(s) belong to pairs where C2 accepted embedding drift (listed in part 1); for them both versions use the recomputed values.
The held-out, published and Vanilla values equal C2's `analysis.json` in 120 of 120 comparisons, and the selected weights match in 20 of 20 pairs.

## Setup

- Retrievers included: BM25, Contriever, BGE-M3, Jina, Qwen.
- Inputs: C2 grid arrays `outputs/rebuttal/C2_heldout_fusion/grid/<dataset>__<retriever>.npz` (per-query hit ranks for all 1296 weight settings) and Table 1 `results/tables/main/retrieval_comparison/table.csv`. Nothing is rebuilt; no LLM or GPU is used.
- Held-out weights: C2's protocol (`heldout_fusion.cross_validated`): for each dataset and retriever separately, 5 folds grouped by document (seed 42); in each fold the weights with the highest HR@10 on the other four folds score the held-out fold.
- Rank: 1 + number of Table 1 methods of the same retriever with a strictly higher printed value (one decimal); SAADI's recomputed value is rounded to one decimal first, and a tie gives SAADI the better rank.
- Vanilla in part 4: the text channel alone (lambda_M = lambda_S = 0) on the same chunks and queries, so the comparison is paired. Its value can differ from the Vanilla row of Table 1, which is kept unchanged in the rank tables.
- Bootstrap: 95% paired interval over queries, 10,000 resamples, seed 42 (`frozen_records.paired_bootstrap_ci`).
- Commit 810b5083f49129ef502ee49fd363d0a80e1abb3c; host eee-s132-infsys-gpu03; 4 s.

## 1. Check: SAADI with the published weights vs Table 1

| Pair | HR@5 Table 1 / recomputed | HR@10 Table 1 / recomputed | Matches | C2 drift accepted |
| --- | --- | --- | --- | --- |
| LooGLE / BM25 | 92.3 / 92.30 | 96.5 / 96.45 | yes | no |
| MuSiQue-32K / BM25 | 62.8 / 62.78 | 77.7 / 77.67 | yes | no |
| QASPER-64K / BM25 | 17.0 / 16.95 | 26.1 / 26.11 | yes | no |
| NovelHopQA / BM25 | 18.1 / 18.06 | 22.0 / 22.03 | yes | no |
| LooGLE / Contriever | 75.4 / 75.44 | 85.7 / 85.70 | yes | no |
| MuSiQue-32K / Contriever | 68.8 / 68.78 | 83.7 / 83.67 | yes | no |
| QASPER-64K / Contriever | 16.2 / 16.20 | 25.8 / 25.81 | yes | no |
| NovelHopQA / Contriever | 13.5 / 13.55 | 18.2 / 18.17 | yes | no |
| LooGLE / BGE-M3 | 81.1 / 81.06 | 90.1 / 90.12 | yes | no |
| MuSiQue-32K / BGE-M3 | 76.9 / 76.89 | 88.3 / 88.33 | yes | no |
| QASPER-64K / BGE-M3 | 20.9 / 20.93 | 30.9 / 30.91 | yes | no |
| NovelHopQA / BGE-M3 | 16.5 / 16.52 | 22.1 / 22.14 | yes | no |
| LooGLE / Jina | 70.1 / 70.20 | 82.0 / 81.99 | no | yes |
| MuSiQue-32K / Jina | 73.2 / 73.56 | 88.1 / 88.11 | no | yes |
| QASPER-64K / Jina | 18.9 / 18.83 | 27.2 / 27.08 | no | yes |
| NovelHopQA / Jina | 14.8 / 14.76 | 21.1 / 20.93 | no | yes |
| LooGLE / Qwen | 77.0 / 77.13 | 86.7 / 86.41 | no | yes |
| MuSiQue-32K / Qwen | 76.4 / 76.44 | 90.0 / 90.00 | yes | no |
| QASPER-64K / Qwen | 23.3 / 23.33 | 34.1 / 34.13 | yes | no |
| NovelHopQA / Qwen | 15.2 / 15.20 | 20.6 / 20.59 | yes | no |

## 2. Table 1 with SAADI's published and held-out weights (HR %, other methods from Table 1)

| Retriever | Method | LooGLE @5 | LooGLE @10 | MuSiQue-32K @5 | MuSiQue-32K @10 | QASPER-64K @5 | QASPER-64K @10 | NovelHopQA @5 | NovelHopQA @10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BM25 | Vanilla | 91.5 | 95.4 | 57.6 | 74.0 | 14.4 | 22.1 | 16.9 | 20.0 |
| BM25 | LlamaIndex | 91.0 | 95.5 | 55.4 | 72.7 | 20.4 | 27.0 | 3.9 | 4.5 |
| BM25 | LQCA | 88.8 | 94.2 | 83.6 | 91.8 | 36.9 | 47.0 | 1.4 | 1.9 |
| BM25 | SAADI, published weights | **92.3** (1) | **96.5** (1) | **62.8** (2) | **77.7** (2) | **17.0** (3) | **26.1** (3) | **18.1** (1) | **22.0** (1) |
| BM25 | SAADI, held-out weights | **92.3** (1) | **96.2** (1) | **64.7** (2) | **79.9** (2) | **17.3** (3) | **27.3** (2) | **17.6** (1) | **21.1** (1) |
| Contriever | Vanilla | 72.4 | 73.3 | 59.9 | 76.1 | 15.2 | 23.3 | 11.8 | 11.8 |
| Contriever | LlamaIndex | 36.3 | 52.2 | 50.9 | 67.7 | 5.9 | 11.3 | 1.9 | 3.1 |
| Contriever | LQCA | 73.3 | 84.5 | 80.8 | 89.0 | 34.4 | 46.0 | 0.9 | 1.7 |
| Contriever | SAADI, published weights | **75.4** (1) | **85.7** (1) | **68.8** (2) | **83.7** (2) | **16.2** (2) | **25.8** (2) | **13.5** (1) | **18.2** (1) |
| Contriever | SAADI, held-out weights | **76.3** (1) | **86.2** (1) | **68.8** (2) | **84.9** (2) | **15.8** (2) | **25.2** (2) | **13.2** (1) | **18.1** (1) |
| BGE-M3 | Vanilla | 80.6 | 89.8 | 73.2 | 87.4 | 21.0 | 29.3 | 14.1 | 20.7 |
| BGE-M3 | LlamaIndex | 80.1 | 87.3 | 68.8 | 82.4 | 19.2 | 30.3 | 13.5 | 18.2 |
| BGE-M3 | LQCA | 74.8 | 84.1 | 76.7 | 90.4 | 32.9 | 45.8 | 1.4 | 2.2 |
| BGE-M3 | SAADI, published weights | **81.1** (1) | **90.1** (1) | **76.9** (1) | **88.3** (2) | **20.9** (3) | **30.9** (2) | **16.5** (1) | **22.1** (1) |
| BGE-M3 | SAADI, held-out weights | **82.3** (1) | **90.6** (1) | **76.1** (2) | **90.8** (1) | **22.1** (2) | **31.9** (2) | **17.1** (1) | **22.7** (1) |
| Jina | Vanilla | 70.1 | 80.1 | 71.6 | 85.8 | 15.7 | 24.8 | 14.8 | 19.1 |
| Jina | Late Chunking (no overlap) | 63.4 | 76.8 | 42.0 | 59.8 | 7.9 | 12.4 | 14.8 | 17.7 |
| Jina | Late Chunking (low overlap) | 67.5 | 79.3 | 41.7 | 58.1 | 7.9 | 11.1 | 14.3 | 17.7 |
| Jina | Late Chunking (more overlap) | 67.7 | 80.3 | 36.2 | 49.4 | 7.4 | 10.3 | 14.6 | 17.3 |
| Jina | LlamaIndex | 71.6 | 81.5 | 67.3 | 81.9 | 21.2 | 29.2 | 3.1 | 4.2 |
| Jina | LQCA | 68.6 | 79.1 | 79.4 | 89.0 | 31.7 | 41.8 | 2.1 | 2.2 |
| Jina | SAADI, published weights | **70.2** (2) | **82.0** (1) | **73.6** (2) | **88.1** (2) | **18.8** (3) | **27.1** (3) | **14.8** (1) | **20.9** (1) |
| Jina | SAADI, held-out weights | **70.6** (2) | **81.8** (1) | **72.4** (2) | **88.0** (2) | **18.1** (3) | **27.9** (3) | **14.9** (1) | **20.3** (1) |
| Qwen | Vanilla | 76.9 | 85.9 | 75.3 | 89.0 | 22.6 | 32.6 | 14.6 | 19.8 |
| Qwen | LlamaIndex | 50.8 | 57.3 | 52.4 | 58.1 | 22.6 | 28.6 | 2.4 | 2.8 |
| Qwen | LQCA | 81.4 | 86.8 | 82.2 | 90.4 | 39.7 | 53.6 | 2.2 | 2.8 |
| Qwen | SAADI, published weights | **77.1** (2) | **86.4** (2) | **76.4** (2) | **90.0** (2) | **23.3** (2) | **34.1** (2) | **15.2** (1) | **20.6** (1) |
| Qwen | SAADI, held-out weights | **77.2** (2) | **86.4** (2) | **75.3** (2) | **89.6** (2) | **23.8** (2) | **34.2** (2) | **15.4** (1) | **19.8** (1) |

SAADI's rank among the methods of its retriever is in parentheses.

## 3. SAADI's position

| Scope | Cells | Mean rank published / held-out | First published / held-out | First or second published / held-out | Cells that change rank |
| --- | --- | --- | --- | --- | --- |
| All cells | 40 | 1.68 / 1.62 | 18 / 18 | 35 / 37 | 4 |
| LooGLE | 10 | 1.30 / 1.30 | 7 / 7 | 10 / 10 | 0 |
| MuSiQue-32K | 10 | 1.90 / 1.90 | 1 / 1 | 10 / 10 | 2 |
| QASPER-64K | 10 | 2.50 / 2.30 | 0 / 0 | 5 / 7 | 2 |
| NovelHopQA | 10 | 1.00 / 1.00 | 10 / 10 | 10 / 10 | 0 |
| BM25 (4 methods) | 8 | 1.75 / 1.62 | 4 / 4 | 6 / 7 | 1 |
| Contriever (4 methods) | 8 | 1.50 / 1.50 | 4 / 4 | 8 / 8 | 0 |
| BGE-M3 (4 methods) | 8 | 1.50 / 1.38 | 5 / 5 | 7 / 8 | 3 |
| Jina (7 methods) | 8 | 1.88 / 1.88 | 3 / 3 | 6 / 6 | 0 |
| Qwen (4 methods) | 8 | 1.75 / 1.75 | 2 / 2 | 8 / 8 | 0 |
| HR@5 | 20 | 1.70 / 1.70 | 9 / 8 | 17 / 18 | 2 |
| HR@10 | 20 | 1.65 / 1.55 | 9 / 10 | 18 / 19 | 2 |

Rank in every cell (published -> held-out; best other method of Table 1):

| Retriever | Dataset | HR@k | SAADI published | SAADI held-out | Rank published -> held-out | Best other method |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 | LooGLE | 5 | 92.3 | 92.3 | 1 -> 1 | Vanilla 91.5 |
| BM25 | LooGLE | 10 | 96.5 | 96.2 | 1 -> 1 | LlamaIndex 95.5 |
| BM25 | MuSiQue-32K | 5 | 62.8 | 64.7 | 2 -> 2 | LQCA 83.6 |
| BM25 | MuSiQue-32K | 10 | 77.7 | 79.9 | 2 -> 2 | LQCA 91.8 |
| BM25 | QASPER-64K | 5 | 17.0 | 17.3 | 3 -> 3 | LQCA 36.9 |
| BM25 | QASPER-64K | 10 | 26.1 | 27.3 | 3 -> 2 (changed) | LQCA 47.0 |
| BM25 | NovelHopQA | 5 | 18.1 | 17.6 | 1 -> 1 | Vanilla 16.9 |
| BM25 | NovelHopQA | 10 | 22.0 | 21.1 | 1 -> 1 | Vanilla 20.0 |
| Contriever | LooGLE | 5 | 75.4 | 76.3 | 1 -> 1 | LQCA 73.3 |
| Contriever | LooGLE | 10 | 85.7 | 86.2 | 1 -> 1 | LQCA 84.5 |
| Contriever | MuSiQue-32K | 5 | 68.8 | 68.8 | 2 -> 2 | LQCA 80.8 |
| Contriever | MuSiQue-32K | 10 | 83.7 | 84.9 | 2 -> 2 | LQCA 89.0 |
| Contriever | QASPER-64K | 5 | 16.2 | 15.8 | 2 -> 2 | LQCA 34.4 |
| Contriever | QASPER-64K | 10 | 25.8 | 25.2 | 2 -> 2 | LQCA 46.0 |
| Contriever | NovelHopQA | 5 | 13.5 | 13.2 | 1 -> 1 | Vanilla 11.8 |
| Contriever | NovelHopQA | 10 | 18.2 | 18.1 | 1 -> 1 | Vanilla 11.8 |
| BGE-M3 | LooGLE | 5 | 81.1 | 82.3 | 1 -> 1 | Vanilla 80.6 |
| BGE-M3 | LooGLE | 10 | 90.1 | 90.6 | 1 -> 1 | Vanilla 89.8 |
| BGE-M3 | MuSiQue-32K | 5 | 76.9 | 76.1 | 1 -> 2 (changed) | LQCA 76.7 |
| BGE-M3 | MuSiQue-32K | 10 | 88.3 | 90.8 | 2 -> 1 (changed) | LQCA 90.4 |
| BGE-M3 | QASPER-64K | 5 | 20.9 | 22.1 | 3 -> 2 (changed) | LQCA 32.9 |
| BGE-M3 | QASPER-64K | 10 | 30.9 | 31.9 | 2 -> 2 | LQCA 45.8 |
| BGE-M3 | NovelHopQA | 5 | 16.5 | 17.1 | 1 -> 1 | Vanilla 14.1 |
| BGE-M3 | NovelHopQA | 10 | 22.1 | 22.7 | 1 -> 1 | Vanilla 20.7 |
| Jina | LooGLE | 5 | 70.2 | 70.6 | 2 -> 2 | LlamaIndex 71.6 |
| Jina | LooGLE | 10 | 82.0 | 81.8 | 1 -> 1 | LlamaIndex 81.5 |
| Jina | MuSiQue-32K | 5 | 73.6 | 72.4 | 2 -> 2 | LQCA 79.4 |
| Jina | MuSiQue-32K | 10 | 88.1 | 88.0 | 2 -> 2 | LQCA 89.0 |
| Jina | QASPER-64K | 5 | 18.8 | 18.1 | 3 -> 3 | LQCA 31.7 |
| Jina | QASPER-64K | 10 | 27.1 | 27.9 | 3 -> 3 | LQCA 41.8 |
| Jina | NovelHopQA | 5 | 14.8 | 14.9 | 1 -> 1 | Vanilla 14.8 |
| Jina | NovelHopQA | 10 | 20.9 | 20.3 | 1 -> 1 | Vanilla 19.1 |
| Qwen | LooGLE | 5 | 77.1 | 77.2 | 2 -> 2 | LQCA 81.4 |
| Qwen | LooGLE | 10 | 86.4 | 86.4 | 2 -> 2 | LQCA 86.8 |
| Qwen | MuSiQue-32K | 5 | 76.4 | 75.3 | 2 -> 2 | LQCA 82.2 |
| Qwen | MuSiQue-32K | 10 | 90.0 | 89.6 | 2 -> 2 | LQCA 90.4 |
| Qwen | QASPER-64K | 5 | 23.3 | 23.8 | 2 -> 2 | LQCA 39.7 |
| Qwen | QASPER-64K | 10 | 34.1 | 34.2 | 2 -> 2 | LQCA 53.6 |
| Qwen | NovelHopQA | 5 | 15.2 | 15.4 | 1 -> 1 | Vanilla 14.6 |
| Qwen | NovelHopQA | 10 | 20.6 | 19.8 | 1 -> 1 | Vanilla 19.8 |

## 4. Held-out SAADI minus Vanilla (text channel on the same chunks), HR points, 95% paired bootstrap

| Pair | Queries | Vanilla @5 / @10 | Held-out @5 / @10 | Diff @5 [95% CI] | Diff @10 [95% CI] | Table 1 Vanilla @5 / @10 |
| --- | --- | --- | --- | --- | --- | --- |
| LooGLE / BM25 | 1832 | 91.9 / 95.7 | 92.3 / 96.2 | +0.4 [-0.3, +1.1] | +0.4 [-0.2, +1.0] | 91.5 / 95.4 |
| MuSiQue-32K / BM25 | 900 | 57.6 / 74.0 | 64.7 / 79.9 | +7.1 [+4.3, +9.9] | +5.9 [+3.3, +8.4] | 57.6 / 74.0 |
| QASPER-64K / BM25 | 1333 | 14.4 / 22.1 | 17.3 / 27.3 | +2.9 [+1.4, +4.5] | +5.3 [+3.5, +7.0] | 14.4 / 22.1 |
| NovelHopQA / BM25 | 908 | 17.5 / 20.7 | 17.6 / 21.1 | +0.1 [-1.1, +1.3] | +0.4 [-0.8, +1.8] | 16.9 / 20.0 |
| LooGLE / Contriever | 1832 | 75.2 / 86.0 | 76.3 / 86.2 | +1.1 [+0.1, +2.2] | +0.2 [-0.5, +1.0] | 72.4 / 73.3 |
| MuSiQue-32K / Contriever | 900 | 59.9 / 76.1 | 68.8 / 84.9 | +8.9 [+5.9, +12.0] | +8.8 [+6.3, +11.3] | 59.9 / 76.1 |
| QASPER-64K / Contriever | 1333 | 15.2 / 23.3 | 15.8 / 25.2 | +0.6 [-0.9, +2.2] | +1.9 [+0.4, +3.4] | 15.2 / 23.3 |
| NovelHopQA / Contriever | 908 | 12.4 / 17.8 | 13.2 / 18.1 | +0.8 [-0.3, +1.9] | +0.2 [-1.0, +1.4] | 11.8 / 11.8 |
| LooGLE / BGE-M3 | 1832 | 82.5 / 90.7 | 82.3 / 90.6 | -0.2 [-1.0, +0.7] | -0.1 [-0.8, +0.5] | 80.6 / 89.8 |
| MuSiQue-32K / BGE-M3 | 900 | 73.2 / 87.4 | 76.1 / 90.8 | +2.9 [+0.6, +5.2] | +3.3 [+1.7, +5.0] | 73.2 / 87.4 |
| QASPER-64K / BGE-M3 | 1333 | 21.0 / 29.3 | 22.1 / 31.9 | +1.1 [-0.6, +2.7] | +2.6 [+0.8, +4.4] | 21.0 / 29.3 |
| NovelHopQA / BGE-M3 | 908 | 17.3 / 21.9 | 17.1 / 22.7 | -0.2 [-1.3, +0.9] | +0.8 [-0.7, +2.2] | 14.1 / 20.7 |
| LooGLE / Jina | 1832 | 69.9 / 80.7 | 70.6 / 81.8 | +0.7 [-0.6, +1.9] | +1.1 [+0.1, +2.2] | 70.1 / 80.1 |
| MuSiQue-32K / Jina | 900 | 71.7 / 85.8 | 72.4 / 88.0 | +0.8 [-1.6, +3.0] | +2.2 [+0.3, +4.2] | 71.6 / 85.8 |
| QASPER-64K / Jina | 1333 | 16.0 / 25.0 | 18.1 / 27.9 | +2.1 [+0.5, +3.8] | +2.9 [+1.4, +4.5] | 15.7 / 24.8 |
| NovelHopQA / Jina | 908 | 16.0 / 20.2 | 14.9 / 20.3 | -1.1 [-2.3, +0.1] | +0.1 [-1.1, +1.3] | 14.8 / 19.1 |
| LooGLE / Qwen | 1832 | 77.9 / 86.2 | 77.2 / 86.4 | -0.7 [-1.9, +0.5] | +0.2 [-0.7, +1.0] | 76.9 / 85.9 |
| MuSiQue-32K / Qwen | 900 | 75.3 / 89.0 | 75.3 / 89.6 | +0.0 [-2.4, +2.3] | +0.6 [-1.0, +2.2] | 75.3 / 89.0 |
| QASPER-64K / Qwen | 1333 | 22.6 / 32.6 | 23.8 / 34.2 | +1.2 [-0.2, +2.6] | +1.7 [+0.2, +3.1] | 22.6 / 32.6 |
| NovelHopQA / Qwen | 908 | 16.7 / 20.5 | 15.4 / 19.8 | -1.3 [-2.5, -0.1] | -0.7 [-2.1, +0.8] | 14.6 / 19.8 |

- HR@5: gain positive in 14 of 20 pairs, interval above zero in 6, below zero in 1; mean gain +1.36 points.
- HR@10: gain positive in 18 of 20 pairs, interval above zero in 10, below zero in 0; mean gain +1.89 points.

## Selected weights per fold (lambda_M, lambda_S | mu_A, mu_D)

| Pair | Folds |
| --- | --- |
| LooGLE / BM25 | (0.3, 0 | 1, 1); (0.3, 0.2 | 0, 0); (0.3, 0.1 | 0.3, 0); (0.1, 0.3 | 1, 0); (0.3, 0 | 0.5, 1) |
| MuSiQue-32K / BM25 | (1, 1.5 | 0, 0.5); (0.75, 1 | 0, 0.3); (0.75, 2 | 1, 0.5); (1, 1.5 | 0, 0.5); (0.75, 1 | 0, 0.3) |
| QASPER-64K / BM25 | (0.5, 0.75 | 1, 1); (0.5, 0.75 | 1, 1); (0.5, 0.75 | 1, 1); (0.5, 0.75 | 1, 1); (0.5, 0.75 | 1, 1) |
| NovelHopQA / BM25 | (0.2, 0.3 | 0, 0); (1.5, 0.3 | 0.5, 1); (0.5, 0.5 | 1, 0.5); (1.5, 0.3 | 0.5, 1); (0.5, 0.5 | 0, 0.3) |
| LooGLE / Contriever | (0.3, 0 | 0, 1); (0.3, 0 | 0, 1); (0.3, 0.2 | 0.5, 1); (0.3, 0 | 0.3, 1); (0.3, 0 | 0, 1) |
| MuSiQue-32K / Contriever | (0.75, 2 | 0.5, 0.3); (0.75, 2 | 0.5, 0.3); (0.75, 2 | 0.5, 0.3); (1.5, 1.5 | 1, 1); (1.5, 1.5 | 1, 1) |
| QASPER-64K / Contriever | (0.5, 0.75 | 0.3, 1); (0.5, 0.1 | 0.3, 1); (0.5, 0.75 | 0.3, 1); (0.3, 0.2 | 0, 1); (0.3, 0.2 | 0, 1) |
| NovelHopQA / Contriever | (0.1, 0.1 | 0, 0.5); (0.2, 0.1 | 0.3, 0.5); (0.5, 0.1 | 0, 0.5); (0.5, 0.1 | 0.3, 0.3); (0.5, 0.2 | 0, 0) |
| LooGLE / BGE-M3 | (0.3, 0 | 0, 1); (0.2, 0 | 0.3, 1); (0.1, 0.1 | 0, 1); (0.3, 0 | 0, 1); (0.2, 0 | 1, 0.5) |
| MuSiQue-32K / BGE-M3 | (0.2, 0.75 | 0.3, 1); (0.2, 0.75 | 0.3, 1); (0.2, 0.75 | 0.3, 1); (0.2, 0.75 | 0, 1); (0.2, 0.75 | 0.3, 1) |
| QASPER-64K / BGE-M3 | (0, 0.75 | 1, 1); (0, 0.75 | 1, 1); (0.2, 0.75 | 1, 1); (0.2, 0.75 | 1, 1); (0.2, 0.75 | 1, 1) |
| NovelHopQA / BGE-M3 | (0.1, 0.3 | 0, 0.5); (0, 0.3 | 1, 1); (0, 0.3 | 1, 1); (0, 0.3 | 1, 1); (0.3, 0.3 | 1, 0.3) |
| LooGLE / Jina | (0.3, 0 | 0.5, 0); (0.3, 0 | 0.5, 0); (0.5, 0.1 | 0.3, 0.5); (0.3, 0 | 1, 0); (0.3, 0 | 1, 0) |
| MuSiQue-32K / Jina | (0.2, 1.5 | 0, 1); (0.5, 1.5 | 0, 1); (0.1, 0.5 | 0, 1); (0.1, 0.75 | 0, 1); (0.1, 0.75 | 0, 1) |
| QASPER-64K / Jina | (0.3, 0.75 | 0.3, 0.5); (0.3, 0.75 | 0.3, 1); (0.3, 0.75 | 0.3, 0.5); (0.3, 0.5 | 1, 1); (0.3, 0.5 | 0.3, 1) |
| NovelHopQA / Jina | (0.3, 0.2 | 0.5, 0.5); (0.2, 0.2 | 0.5, 0.3); (0.3, 0.5 | 0.5, 0); (0.3, 0.5 | 1, 0); (0.3, 0.2 | 0.5, 0.5) |
| LooGLE / Qwen | (0.5, 0.5 | 0, 1); (0.2, 0.3 | 0.3, 1); (0.3, 0.3 | 1, 1); (0.3, 0.2 | 0.5, 0.5); (0.5, 0.5 | 0, 1) |
| MuSiQue-32K / Qwen | (0.2, 1 | 0.3, 0.3); (1, 2 | 0, 0.3); (0.5, 1 | 1, 0); (0.3, 1 | 0, 1); (0.1, 0.3 | 0, 1) |
| QASPER-64K / Qwen | (0.2, 0.5 | 1, 1); (0.3, 0.3 | 1, 1); (0.3, 0.3 | 1, 1); (0.3, 0.3 | 1, 1); (0.3, 0.3 | 1, 1) |
| NovelHopQA / Qwen | (1, 0.1 | 0, 0.3); (0.3, 0 | 0, 0); (0.75, 0 | 0.3, 0.3); (0.75, 0 | 0.3, 0.3); (1, 0.3 | 0, 0) |

Files: `results/rebuttal/c2_table1/results.json` (every number above), `results/rebuttal/c2_table1/table1_published.csv`, `results/rebuttal/c2_table1/table1_heldout.csv`.
