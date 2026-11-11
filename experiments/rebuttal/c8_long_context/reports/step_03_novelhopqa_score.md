# C8 long-context reader - step 03 scores, NovelHopQA: OK

Written 2026-11-11 14:23:40; commit 368135a048c0; host eee-s132-infsys-gpu03; 2 s. Server mode native.

Common queries (answered in all four conditions): 526 of 908 selected. Answered per condition: {'full_doc': 526, 'saadi': 908, 'vanilla': 908, 'closed_book': 908}. C5 reader answers: results/rebuttal/c5_end_to_end/answers/novelhopqa/saadi.jsonl (526 on the common queries).

Accuracy (percent, 95% bootstrap interval over queries, 10,000 resamples, seed 42):

| Condition | Reader | n | EM | F1 | Containment |
| --- | --- | --- | --- | --- | --- |
| Full document | Qwen2.5-7B-Instruct-1M | 526 | 3.6 [2.1, 5.3] | 23.3 [21.0, 25.6] | 17.1 [13.9, 20.3] |
| SAADI top 10 | Qwen2.5-7B-Instruct-1M | 526 | 6.7 [4.6, 8.9] | 30.2 [27.6, 32.9] | 23.0 [19.6, 26.6] |
| Vanilla BM25 top 10 | Qwen2.5-7B-Instruct-1M | 526 | 6.3 [4.2, 8.4] | 30.5 [27.9, 33.1] | 22.8 [19.4, 26.4] |
| Closed-book | Qwen2.5-7B-Instruct-1M | 526 | 1.0 [0.2, 1.9] | 8.1 [6.8, 9.5] | 1.7 [0.8, 2.9] |
| SAADI top 10, C5 reader (Qwen3-30B-A3B) | Qwen3-30B-A3B (C5) | 526 | 7.6 [5.5, 9.9] | 31.8 [29.2, 34.5] | 24.3 [20.9, 28.1] |

Paired differences (points, same queries):

| Pair | n | EM | F1 | Containment |
| --- | --- | --- | --- | --- |
| **Full document - SAADI top 10** | 526 | -3.0 [-4.9, -1.1] | -6.9 [-9.6, -4.3] | -5.9 [-9.1, -2.7] |
| **SAADI top 10 - Vanilla BM25 top 10** | 526 | +0.4 [-1.1, +1.9] | -0.2 [-1.9, +1.4] | +0.2 [-1.7, +2.1] |
| Full document - Vanilla BM25 top 10 | 526 | -2.7 [-4.8, -0.8] | -7.2 [-9.8, -4.6] | -5.7 [-9.1, -2.3] |
| Full document - Closed-book | 526 | +2.7 [+1.0, +4.4] | +15.2 [+12.7, +17.7] | +15.4 [+12.2, +18.6] |
| SAADI top 10 - Closed-book | 526 | +5.7 [+3.6, +8.0] | +22.1 [+19.3, +24.9] | +21.3 [+17.7, +24.9] |
| Full document - SAADI top 10, C5 reader (Qwen3-30B-A3B) | 526 | -4.0 [-6.3, -1.7] | -8.5 [-11.2, -5.9] | -7.2 [-10.5, -4.0] |
| SAADI top 10 - SAADI top 10, C5 reader (Qwen3-30B-A3B) | 526 | -1.0 [-3.2, +1.1] | -1.6 [-3.5, +0.4] | -1.3 [-4.2, +1.5] |

Cost per query:

| Condition | n | Input tokens (mean) | Output tokens (mean) | Request latency in the run (s) | Single request alone, timing (s) | Amortised wall s / query | Full doc without cache (s / query) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Full document | 526 | 195,298 | 11.8 | 6.89 | 63.2 uncached / 1.03 cached | 1.97 | 62.4 |
| SAADI top 10 | 526 | 6,455 | 12.3 | 10.01 | 1.32 | 1.32 | — |
| Vanilla BM25 top 10 | 526 | 6,449 | 12.3 | 9.97 | 1.58 | 1.33 | — |
| Closed-book | 526 | 80 | 10.9 | 0.24 | 0.19 | 0.03 | — |
| SAADI top 10, C5 reader (Qwen3-30B-A3B) | 526 | 6,444 | 15.6 | — | — | — | — |

F1 by the position of the gold evidence in the document (all common queries, and those whose evidence lies in one third):

| Condition | Subset | Third | n | EM | F1 | Containment |
| --- | --- | --- | --- | --- | --- | --- |
| Full document | all | first | 139 | 3.6 | 22.5 | 19.4 |
| Full document | all | middle | 228 | 3.5 | 23.0 | 14.9 |
| Full document | all | last | 142 | 4.2 | 26.1 | 19.7 |
| Full document | evidence_in_one_third | first | 107 | 4.7 | 23.8 | 22.4 |
| Full document | evidence_in_one_third | middle | 92 | 4.3 | 24.2 | 22.8 |
| Full document | evidence_in_one_third | last | 109 | 3.7 | 27.0 | 20.2 |
| Full document | all | no located evidence | 17 | 0.0 | 10.6 | 5.9 |
| SAADI top 10 | all | first | 139 | 9.4 | 33.7 | 28.8 |
| SAADI top 10 | all | middle | 228 | 6.1 | 29.0 | 20.6 |
| SAADI top 10 | all | last | 142 | 5.6 | 30.9 | 23.2 |
| SAADI top 10 | evidence_in_one_third | first | 107 | 10.3 | 35.2 | 32.7 |
| SAADI top 10 | evidence_in_one_third | middle | 92 | 7.6 | 32.2 | 32.6 |
| SAADI top 10 | evidence_in_one_third | last | 109 | 5.5 | 31.0 | 23.9 |
| SAADI top 10 | all | no located evidence | 17 | 0.0 | 13.1 | 5.9 |
| Vanilla BM25 top 10 | all | first | 139 | 10.1 | 35.3 | 28.1 |
| Vanilla BM25 top 10 | all | middle | 228 | 4.8 | 27.8 | 20.2 |
| Vanilla BM25 top 10 | all | last | 142 | 4.9 | 31.2 | 23.2 |
| Vanilla BM25 top 10 | evidence_in_one_third | first | 107 | 10.3 | 37.4 | 32.7 |
| Vanilla BM25 top 10 | evidence_in_one_third | middle | 92 | 4.3 | 29.0 | 30.4 |
| Vanilla BM25 top 10 | evidence_in_one_third | last | 109 | 5.5 | 32.1 | 24.8 |
| Vanilla BM25 top 10 | all | no located evidence | 17 | 5.9 | 20.3 | 11.8 |

Left out: skipped (prompt longer than the served window) {'full_doc': 382, 'saadi': 0, 'vanilla': 0, 'closed_book': 0}; full-document queries not started at the deadline: 0.

## Files written

- `results/rebuttal/c8_long_context/per_dataset/novelhopqa/table.csv`
- `results/rebuttal/c8_long_context/per_dataset/novelhopqa/paired.csv`
- `results/rebuttal/c8_long_context/per_dataset/novelhopqa/cost.csv`
- `results/rebuttal/c8_long_context/per_dataset/novelhopqa/position.csv`
- `results/rebuttal/c8_long_context/per_dataset/novelhopqa/summary.json`
- `experiments/rebuttal/c8_long_context/reports/step_03_novelhopqa_score.md`
