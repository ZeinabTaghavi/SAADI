# C8 step 04 - NovelHopQA, every book read whole with dual chunk attention: OK

Written 2026-11-12 16:45:34; commit 93057fb5f449; host eee-s132-infsys-gpu03; 6 s.

Every NovelHopQA book read whole, each in the mode its length needs (the main run's native read for the books inside 262,144 tokens, dual chunk attention beyond; 16 books, 908 questions): F1 21.7 [20.1, 23.4] against 30.5 [28.6, 32.5] for SAADI's top 10 with the same 7B reader; paired difference -8.8 [-10.8, -6.8] (EM -3.2 [-4.7, -1.7], containment -8.1 [-10.8, -5.6]). The 7B 1M-context reader (Qwen2.5-7B-Instruct-1M) read every NovelHopQA book whole with Qwen's dual chunk attention (vLLM 0.10.1.1, V0 engine, as on the model card): 16 books of 133,420-759,549 tokens, 908 of 908 questions answered. Reading the whole book gives F1 21.2 [19.5, 22.8]; the same reader with SAADI's top 10 gives 30.5 [28.6, 32.5]; paired difference full book - SAADI -9.3 [-11.4, -7.4] (EM -3.4 [-5.1, -1.9], containment -8.3 [-10.9, -5.7]). On the 6 books beyond the native 262,144-token window (264,945-759,549 tokens, 382 questions), the ones the main run had to skip: full book 19.6 [17.3, 22.0], SAADI 30.8 [28.0, 33.8], difference -11.2 [-14.4, -8.2]. On the 10 books inside the native window, dual chunk attention gives F1 22.3 against 23.3 for the main run's native read of the same books (difference -1.0 [-2.3, +0.3]). Input per question: 295,083 tokens for the whole book against 6,525 for SAADI (45x); a fresh read of a book takes 85.7 s per question on average, 2.98 s with the book kept in the prefix cache, against 1.32 s for SAADI.

Accuracy (percent, 95% bootstrap interval over questions, 10,000 resamples, seed 42). SAADI, Vanilla and closed-book are the main run's answers (their prompts are below 8K tokens, where dual chunk attention equals standard attention).

| Books | Condition | n | EM | F1 | Containment |
| --- | --- | --- | --- | --- | --- |
| all_mixed | Full book: native read up to 262K, dual chunk attention beyond (7B) | 908 | 3.3 [2.2, 4.5] | 21.7 [20.1, 23.4] | 13.3 [11.2, 15.6] |
| all_mixed | SAADI top 10 (7B) | 908 | 6.5 [5.0, 8.1] | 30.5 [28.6, 32.5] | 21.5 [18.8, 24.2] |
| all_mixed | Vanilla BM25 top 10 (7B) | 908 | 6.4 [4.8, 8.0] | 30.5 [28.6, 32.6] | 21.6 [18.9, 24.3] |
| all_mixed | Closed-book (7B) | 908 | 0.7 [0.2, 1.2] | 8.5 [7.5, 9.6] | 1.7 [0.9, 2.5] |
| all_mixed | SAADI top 10, C5 reader (Qwen3-30B-A3B) | 908 | 7.6 [5.9, 9.4] | 31.5 [29.6, 33.6] | 22.6 [19.9, 25.3] |
| all | Full book, dual chunk attention (7B) | 908 | 3.1 [2.0, 4.3] | 21.2 [19.5, 22.8] | 13.2 [11.0, 15.5] |
| all | SAADI top 10 (7B) | 908 | 6.5 [5.0, 8.1] | 30.5 [28.6, 32.5] | 21.5 [18.8, 24.2] |
| all | Vanilla BM25 top 10 (7B) | 908 | 6.4 [4.8, 8.0] | 30.5 [28.6, 32.6] | 21.6 [18.9, 24.3] |
| all | Closed-book (7B) | 908 | 0.7 [0.2, 1.2] | 8.5 [7.5, 9.6] | 1.7 [0.9, 2.5] |
| all | SAADI top 10, C5 reader (Qwen3-30B-A3B) | 908 | 7.6 [5.9, 9.4] | 31.5 [29.6, 33.6] | 22.6 [19.9, 25.3] |
| beyond_native | Full book, dual chunk attention (7B) | 382 | 2.9 [1.3, 4.7] | 19.6 [17.3, 22.0] | 8.1 [5.5, 11.0] |
| beyond_native | SAADI top 10 (7B) | 382 | 6.3 [3.9, 8.9] | 30.8 [28.0, 33.8] | 19.4 [15.4, 23.6] |
| beyond_native | Vanilla BM25 top 10 (7B) | 382 | 6.5 [4.2, 9.2] | 30.6 [27.7, 33.5] | 19.9 [16.0, 24.1] |
| beyond_native | Closed-book (7B) | 382 | 0.3 [0.0, 0.8] | 9.1 [7.6, 10.7] | 1.6 [0.5, 2.9] |
| beyond_native | SAADI top 10, C5 reader (Qwen3-30B-A3B) | 382 | 7.6 [5.0, 10.5] | 31.1 [28.1, 34.1] | 20.2 [16.2, 24.3] |
| within_native | Full book, dual chunk attention (7B) | 526 | 3.2 [1.9, 4.8] | 22.3 [20.1, 24.6] | 16.9 [13.9, 20.2] |
| within_native | Full book, native window (7B, main run) | 526 | 3.6 [2.1, 5.3] | 23.3 [21.0, 25.6] | 17.1 [13.9, 20.3] |
| within_native | SAADI top 10 (7B) | 526 | 6.7 [4.6, 8.9] | 30.2 [27.6, 32.9] | 23.0 [19.6, 26.6] |
| within_native | Vanilla BM25 top 10 (7B) | 526 | 6.3 [4.2, 8.4] | 30.5 [27.9, 33.1] | 22.8 [19.4, 26.4] |
| within_native | Closed-book (7B) | 526 | 1.0 [0.2, 1.9] | 8.1 [6.8, 9.5] | 1.7 [0.8, 2.9] |
| within_native | SAADI top 10, C5 reader (Qwen3-30B-A3B) | 526 | 7.6 [5.5, 9.9] | 31.8 [29.2, 34.5] | 24.3 [20.9, 28.1] |

Paired differences (points, same questions):

| Books | Pair | n | EM | F1 | Containment |
| --- | --- | --- | --- | --- | --- |
| all_mixed | Full book: native read up to 262K, dual chunk attention beyond (7B) - SAADI top 10 (7B) | 908 | -3.2 [-4.7, -1.7] | -8.8 [-10.8, -6.8] | -8.1 [-10.8, -5.6] |
| all_mixed | Full book: native read up to 262K, dual chunk attention beyond (7B) - SAADI top 10, C5 reader (Qwen3-30B-A3B) | 908 | -4.3 [-6.1, -2.5] | -9.8 [-11.8, -7.8] | -9.3 [-11.8, -6.7] |
| all_mixed | SAADI top 10 (7B) - Vanilla BM25 top 10 (7B) | 908 | +0.1 [-1.0, +1.2] | -0.0 [-1.3, +1.2] | -0.1 [-1.5, +1.3] |
| all_mixed | Full book: native read up to 262K, dual chunk attention beyond (7B) - Closed-book (7B) | 908 | +2.6 [+1.5, +4.0] | +13.2 [+11.5, +15.0] | +11.7 [+9.6, +13.9] |
| all | Full book, dual chunk attention (7B) - SAADI top 10 (7B) | 908 | -3.4 [-5.1, -1.9] | -9.3 [-11.4, -7.4] | -8.3 [-10.9, -5.7] |
| all | Full book, dual chunk attention (7B) - SAADI top 10, C5 reader (Qwen3-30B-A3B) | 908 | -4.5 [-6.3, -2.8] | -10.3 [-12.4, -8.4] | -9.4 [-11.9, -6.9] |
| all | SAADI top 10 (7B) - Vanilla BM25 top 10 (7B) | 908 | +0.1 [-1.0, +1.2] | -0.0 [-1.3, +1.2] | -0.1 [-1.5, +1.3] |
| all | Full book, dual chunk attention (7B) - Vanilla BM25 top 10 (7B) | 908 | -3.3 [-4.8, -1.8] | -9.4 [-11.4, -7.4] | -8.4 [-10.9, -5.9] |
| all | Full book, dual chunk attention (7B) - Closed-book (7B) | 908 | +2.4 [+1.3, +3.6] | +12.6 [+11.0, +14.4] | +11.6 [+9.4, +13.8] |
| beyond_native | Full book, dual chunk attention (7B) - SAADI top 10 (7B) | 382 | -3.4 [-5.8, -1.0] | -11.2 [-14.4, -8.2] | -11.3 [-15.4, -7.1] |
| beyond_native | Full book, dual chunk attention (7B) - SAADI top 10, C5 reader (Qwen3-30B-A3B) | 382 | -4.7 [-7.3, -2.1] | -11.5 [-14.5, -8.5] | -12.0 [-16.0, -8.1] |
| beyond_native | SAADI top 10 (7B) - Vanilla BM25 top 10 (7B) | 382 | -0.3 [-1.8, +1.3] | +0.2 [-1.7, +2.1] | -0.5 [-2.9, +1.6] |
| beyond_native | Full book, dual chunk attention (7B) - Vanilla BM25 top 10 (7B) | 382 | -3.7 [-6.0, -1.3] | -11.0 [-14.0, -8.2] | -11.8 [-15.7, -8.1] |
| beyond_native | Full book, dual chunk attention (7B) - Closed-book (7B) | 382 | +2.6 [+1.0, +4.5] | +10.5 [+8.1, +13.0] | +6.5 [+3.9, +9.4] |
| within_native | Full book, dual chunk attention (7B) - SAADI top 10 (7B) | 526 | -3.4 [-5.5, -1.5] | -7.9 [-10.7, -5.3] | -6.1 [-9.3, -3.0] |
| within_native | Full book, dual chunk attention (7B) - SAADI top 10, C5 reader (Qwen3-30B-A3B) | 526 | -4.4 [-6.8, -2.1] | -9.5 [-12.2, -6.9] | -7.4 [-10.6, -4.4] |
| within_native | SAADI top 10 (7B) - Vanilla BM25 top 10 (7B) | 526 | +0.4 [-1.1, +1.9] | -0.2 [-1.9, +1.4] | +0.2 [-1.7, +2.1] |
| within_native | Full book, dual chunk attention (7B) - Vanilla BM25 top 10 (7B) | 526 | -3.0 [-5.1, -1.1] | -8.2 [-10.9, -5.5] | -5.9 [-9.1, -2.7] |
| within_native | Full book, dual chunk attention (7B) - Closed-book (7B) | 526 | +2.3 [+0.8, +4.0] | +14.2 [+11.8, +16.6] | +15.2 [+12.0, +18.4] |
| within_native | Full book, dual chunk attention (7B) - Full book, native window (7B, main run) | 526 | -0.4 [-1.1, +0.4] | -1.0 [-2.3, +0.3] | -0.2 [-2.1, +1.7] |
| within_native | Full book, native window (7B, main run) - SAADI top 10 (7B) | 526 | -3.0 [-4.9, -1.1] | -6.9 [-9.6, -4.3] | -5.9 [-9.1, -2.7] |

Cost per question:

| Books | n | Input tokens, full book | Input tokens, SAADI | Full book without cache (s) | Full book with cache, amortised (s) | SAADI, amortised (s, main run) |
| --- | --- | --- | --- | --- | --- | --- |
| all | 908 | 295,083 | 6,525 | 85.7 | 2.98 | 1.32 |
| beyond_native | 382 | 432,485 | 6,621 | 131.1 | 4.25 | 1.32 |
| within_native | 526 | 195,298 | 6,455 | 52.7 | 2.06 | 1.32 |

Per book (shortest first):

| Book | Tokens | Beyond 262K | Questions | Answered | F1 full (DCA) | F1 SAADI | Full - SAADI | F1 full (native, main run) | Uncached read (s) | Cached question latency (s) | Cache hits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B24 | 133,420 | False | 65 | 65 | 28.2 | 34.4 | -6.3 | 28.9 | 34.4 | 5.96 | 64/64 |
| B43 | 159,866 | False | 60 | 60 | 18.3 | 24.5 | -6.2 | 26.0 | 41.9 | 6.67 | 59/59 |
| B37 | 167,704 | False | 62 | 62 | 25.6 | 30.5 | -4.9 | 25.6 | 45.3 | 6.97 | 61/61 |
| B18 | 194,220 | False | 44 | 44 | 24.2 | 30.3 | -6.1 | 27.7 | 52.5 | 8.64 | 43/43 |
| B27 | 201,836 | False | 60 | 60 | 24.6 | 28.3 | -3.7 | 22.8 | 54.1 | 8.26 | 59/59 |
| B34 | 206,833 | False | 65 | 65 | 20.5 | 30.2 | -9.7 | 20.4 | 55.6 | 7.95 | 64/64 |
| B04 | 209,488 | False | 45 | 45 | 23.0 | 32.5 | -9.5 | 20.6 | 56.7 | 8.52 | 44/44 |
| B61 | 225,064 | False | 14 | 14 | 10.2 | 12.6 | -2.4 | 12.1 | 61.2 | 8.59 | 13/13 |
| B12 | 240,542 | False | 61 | 61 | 16.1 | 34.3 | -18.2 | 18.6 | 67.3 | 9.95 | 60/60 |
| B26 | 253,466 | False | 50 | 50 | 23.6 | 31.6 | -8.0 | 21.7 | 69.5 | 10.56 | 49/49 |
| B44 | 264,945 | True | 57 | 57 | 16.0 | 27.6 | -11.6 | — | 75.1 | 10.95 | 56/56 |
| B15 | 273,811 | True | 75 | 75 | 21.2 | 31.2 | -10.0 | — | 76.6 | 11.29 | 74/74 |
| B05 | 324,835 | True | 22 | 22 | 23.8 | 25.2 | -1.4 | — | 93.3 | 12.59 | 21/21 |
| B54 | 411,934 | True | 74 | 74 | 22.9 | 25.2 | -2.3 | — | 122.8 | 16.93 | 73/73 |
| B62 | 461,227 | True | 84 | 84 | 20.4 | 36.6 | -16.3 | — | 139.5 | 18.19 | 83/83 |
| B42 | 759,549 | True | 70 | 70 | 15.0 | 33.9 | -18.9 | — | 245.9 | 32.99 | 69/69 |

Prefix cache: 892 of 892 questions asked after their book's first question found at least 90% of their prompt cached.

Servers (attention backend lines from the vLLM logs):

- shard_0: INFO 11-12 16:11:55 [cuda.py:367] Using DualChunkFlashAttention backend.
[1;36m(VllmWorkerProcess pid=83010)[0;0m INFO 11-12 16:12:00 [cuda.py:367] Using DualChunkFlashAttention backend.
[1;36m(VllmWorkerProcess pid=83012)[0;0m INFO 11-12 16:12:01 [cuda.py:367] Using DualChunkFlashAttention backend.
- shard_1: INFO 11-12 16:40:31 [cuda.py:367] Using DualChunkFlashAttention backend.
[1;36m(VllmWorkerProcess pid=93602)[0;0m INFO 11-12 16:40:36 [cuda.py:367] Using DualChunkFlashAttention backend.
[1;36m(VllmWorkerProcess pid=93603)[0;0m INFO 11-12 16:40:36 [cuda.py:367] Using DualChunkFlashAttention backend.

## Files written

- `results/rebuttal/c8_long_context/dca/table.csv`
- `results/rebuttal/c8_long_context/dca/paired.csv`
- `results/rebuttal/c8_long_context/dca/per_book.csv`
- `results/rebuttal/c8_long_context/dca/cost.csv`
- `results/rebuttal/c8_long_context/dca/summary.json`
- `experiments/rebuttal/c8_long_context/reports/step_04_dca_score.md`

## Push

```
pushed 25457ba to origin/fix/notebook-key-and-install: C8 novelhopqa 04_dca_score: all books read whole: F1 full book 21.2, SAADI 30.5 (7B, n=908 of 908); full-SAADI -9.3 [-11.4, -7.4]; mixed full read - SAADI -8.8 [-10.8, -6.8] (n=9
```
