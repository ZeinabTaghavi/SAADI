# C8 long-context reader - step 03 scores, MuSiQue-32K: OK

Written 2026-11-11 13:25:43; commit f66bb89d536d; host eee-s132-infsys-gpu03; 3 s. Server mode native.

Common queries (answered in all four conditions): 900 of 900 selected. Answered per condition: {'full_doc': 900, 'saadi': 900, 'vanilla': 900, 'closed_book': 900}. C5 reader answers: results/rebuttal/c5_end_to_end/answers/musique_32k/saadi.jsonl (900 on the common queries).

Accuracy (percent, 95% bootstrap interval over queries, 10,000 resamples, seed 42):

| Condition | Reader | n | EM | F1 | Containment |
| --- | --- | --- | --- | --- | --- |
| Full document | Qwen2.5-7B-Instruct-1M | 900 | 13.8 [11.6, 16.1] | 28.6 [26.3, 30.9] | 31.0 [28.0, 34.0] |
| SAADI top 10 | Qwen2.5-7B-Instruct-1M | 900 | 12.3 [10.2, 14.6] | 26.3 [23.9, 28.6] | 26.3 [23.3, 29.2] |
| Vanilla BM25 top 10 | Qwen2.5-7B-Instruct-1M | 900 | 10.8 [8.8, 12.9] | 24.8 [22.6, 27.1] | 24.1 [21.3, 27.0] |
| Closed-book | Qwen2.5-7B-Instruct-1M | 900 | 1.3 [0.7, 2.1] | 7.1 [6.0, 8.3] | 4.8 [3.4, 6.2] |
| SAADI top 10, C5 reader (Qwen3-30B-A3B) | Qwen3-30B-A3B (C5) | 900 | 11.4 [9.4, 13.7] | 22.8 [20.7, 25.0] | 30.3 [27.3, 33.3] |

Paired differences (points, same queries):

| Pair | n | EM | F1 | Containment |
| --- | --- | --- | --- | --- |
| **Full document - SAADI top 10** | 900 | +1.4 [-1.2, +4.1] | +2.3 [-0.4, +5.1] | +4.7 [+1.3, +8.1] |
| **SAADI top 10 - Vanilla BM25 top 10** | 900 | +1.6 [-0.1, +3.2] | +1.4 [-0.3, +3.2] | +2.2 [-0.1, +4.6] |
| Full document - Vanilla BM25 top 10 | 900 | +3.0 [+0.3, +5.7] | +3.7 [+1.1, +6.4] | +6.9 [+3.6, +10.3] |
| Full document - Closed-book | 900 | +12.4 [+10.2, +14.8] | +21.4 [+19.1, +23.7] | +26.2 [+23.1, +29.3] |
| SAADI top 10 - Closed-book | 900 | +11.0 [+8.8, +13.3] | +19.1 [+16.7, +21.6] | +21.6 [+18.6, +24.6] |
| Full document - SAADI top 10, C5 reader (Qwen3-30B-A3B) | 900 | +2.3 [-0.4, +5.1] | +5.7 [+3.0, +8.4] | +0.7 [-2.9, +4.1] |
| SAADI top 10 - SAADI top 10, C5 reader (Qwen3-30B-A3B) | 900 | +0.9 [-1.6, +3.3] | +3.4 [+1.2, +5.7] | -4.0 [-6.8, -1.2] |

Cost per query:

| Condition | n | Input tokens (mean) | Output tokens (mean) | Request latency in the run (s) | Single request alone, timing (s) | Amortised wall s / query | Full doc without cache (s / query) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Full document | 900 | 48,904 | 10.2 | 1.73 | 11.4 uncached / 0.41 cached | 0.74 | 11.4 |
| SAADI top 10 | 900 | 1,403 | 11.3 | 1.97 | 0.44 | 0.25 | — |
| Vanilla BM25 top 10 | 900 | 1,404 | 11.8 | 2.31 | 0.48 | 0.29 | — |
| Closed-book | 900 | 80 | 15.7 | 0.33 | 0.34 | 0.04 | — |
| SAADI top 10, C5 reader (Qwen3-30B-A3B) | 900 | 1,392 | 23.5 | — | — | — | — |

F1 by the position of the gold evidence in the document (all common queries, and those whose evidence lies in one third):

| Condition | Subset | Third | n | EM | F1 | Containment |
| --- | --- | --- | --- | --- | --- | --- |
| Full document | all | first | 269 | 13.8 | 28.3 | 36.1 |
| Full document | all | middle | 369 | 11.4 | 26.5 | 24.4 |
| Full document | all | last | 262 | 17.2 | 31.8 | 35.1 |
| Full document | evidence_in_one_third | first | 201 | 15.4 | 29.7 | 37.3 |
| Full document | evidence_in_one_third | middle | 203 | 8.9 | 25.3 | 22.2 |
| Full document | evidence_in_one_third | last | 206 | 17.0 | 31.9 | 35.0 |
| Full document | all | no located evidence | 0 | — | — | — |
| SAADI top 10 | all | first | 269 | 11.5 | 26.9 | 29.4 |
| SAADI top 10 | all | middle | 369 | 11.4 | 24.7 | 22.8 |
| SAADI top 10 | all | last | 262 | 14.5 | 27.8 | 28.2 |
| SAADI top 10 | evidence_in_one_third | first | 201 | 14.4 | 27.9 | 28.4 |
| SAADI top 10 | evidence_in_one_third | middle | 203 | 12.3 | 24.4 | 22.7 |
| SAADI top 10 | evidence_in_one_third | last | 206 | 14.6 | 27.8 | 27.2 |
| SAADI top 10 | all | no located evidence | 0 | — | — | — |
| Vanilla BM25 top 10 | all | first | 269 | 11.2 | 25.3 | 27.9 |
| Vanilla BM25 top 10 | all | middle | 369 | 10.0 | 23.5 | 19.5 |
| Vanilla BM25 top 10 | all | last | 262 | 11.5 | 26.2 | 26.7 |
| Vanilla BM25 top 10 | evidence_in_one_third | first | 201 | 14.4 | 27.1 | 27.4 |
| Vanilla BM25 top 10 | evidence_in_one_third | middle | 203 | 9.9 | 23.3 | 19.2 |
| Vanilla BM25 top 10 | evidence_in_one_third | last | 206 | 11.2 | 25.7 | 24.8 |
| Vanilla BM25 top 10 | all | no located evidence | 0 | — | — | — |

## Files written

- `results/rebuttal/c8_long_context/per_dataset/musique_32k/table.csv`
- `results/rebuttal/c8_long_context/per_dataset/musique_32k/paired.csv`
- `results/rebuttal/c8_long_context/per_dataset/musique_32k/cost.csv`
- `results/rebuttal/c8_long_context/per_dataset/musique_32k/position.csv`
- `results/rebuttal/c8_long_context/per_dataset/musique_32k/summary.json`
- `experiments/rebuttal/c8_long_context/reports/step_03_musique_32k_score.md`
