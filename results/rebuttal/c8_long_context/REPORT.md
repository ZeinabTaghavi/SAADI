# C8: long-context reader vs SAADI top 10 (reviewer iVU8, Q1)

On MuSiQue-32K (900 queries over 45 documents of 42,257 to 51,692 tokens), Qwen2.5-7B-Instruct-1M reading the whole document reaches F1 28.6 [26.3, 30.9], not distinguishable from the same model reading SAADI's top 10 (26.3 [23.9, 28.6]; difference +2.3 [-0.4, +5.1] points); SAADI's top 10 read by C5's reader (Qwen3-30B-A3B) gives 22.8 [20.7, 25.0].
With the 7B reader, SAADI minus Vanilla BM25 top 10 is +1.4 [-0.3, +3.2] F1 points (EM +1.6 [-0.1, +3.2]).
The full document costs 48,904 input tokens per query against 1,403 for SAADI's top 10 (35x); in wall-clock time it took 0.74 s per query when the questions on one document share its prefix cache and 11.4 s per query without the cache, against 0.25 s for SAADI's top 10 (single request alone: 11.4 s uncached full document, 0.44 s SAADI; 4 RTX A6000).
By the position of the gold evidence, full-document F1 is 28.3 / 26.5 / 31.8 in the first / middle / last third (n = 269 / 369 / 262; range 5.3 points), and SAADI's 26.9 / 24.7 / 27.8.
On NovelHopQA (526 queries over 10 documents of 133,420 to 253,466 tokens), Qwen2.5-7B-Instruct-1M reading the whole document reaches F1 23.3 [21.0, 25.6], lower than the same model reading SAADI's top 10 (30.2 [27.6, 32.9]; difference -6.9 [-9.6, -4.3] points); SAADI's top 10 read by C5's reader (Qwen3-30B-A3B) gives 31.8 [29.2, 34.5].
With the 7B reader, SAADI minus Vanilla BM25 top 10 is -0.2 [-1.9, +1.4] F1 points (EM +0.4 [-1.1, +1.9]).
The full document costs 195,298 input tokens per query against 6,455 for SAADI's top 10 (30x); in wall-clock time it took 1.97 s per query when the questions on one document share its prefix cache and 62.4 s per query without the cache, against 1.32 s for SAADI's top 10 (single request alone: 63.2 s uncached full document, 1.32 s SAADI; 4 RTX A6000).
By the position of the gold evidence, full-document F1 is 22.5 / 23.0 / 26.1 in the first / middle / last third (n = 139 / 228 / 142; range 3.6 points), and SAADI's 33.7 / 29.0 / 30.9.
Caveats: NovelHopQA: the server's vLLM has no dual chunk attention, so the window is the model's native 262,144 tokens; the 6 longer books (264,945 to 759,549 tokens, 382 queries) were not read in full and their queries are left out of every condition, so the full-document comparison covers the 10 shorter books (526 queries). NovelHopQA's evidence positions use C5's paragraph labels (the paper's window labels accept one-word matches); answer scores do not depend on labels.

Written 2026-11-11 19:59:34; commit 2f88145e547a. Reader Qwen/Qwen2.5-7B-Instruct-1M (vLLM, 4 RTX A6000, temperature 0, seed 42, C5's prompt and scoring); the full document is SAADI's chunks in order, given first, the question last.

## Accuracy and cost per query

| Dataset | Condition | n | EM | F1 | Containment | Input tokens / query | Wall s / query | Full doc without cache, s / query |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MuSiQue-32K | Full document, 7B | 900 | 13.8 [11.6, 16.1] | 28.6 [26.3, 30.9] | 31.0 [28.0, 34.0] | 48,904 | 0.74 | 11.4 |
| MuSiQue-32K | SAADI top 10, 7B | 900 | 12.3 [10.2, 14.6] | 26.3 [23.9, 28.6] | 26.3 [23.3, 29.2] | 1,403 | 0.25 | — |
| MuSiQue-32K | SAADI top 10, C5 reader (30B-A3B) | 900 | 11.4 [9.4, 13.7] | 22.8 [20.7, 25.0] | 30.3 [27.3, 33.3] | 1,392 | — | — |
| MuSiQue-32K | Vanilla top 10, 7B | 900 | 10.8 [8.8, 12.9] | 24.8 [22.6, 27.1] | 24.1 [21.3, 27.0] | 1,404 | 0.29 | — |
| MuSiQue-32K | Closed-book, 7B | 900 | 1.3 [0.7, 2.1] | 7.1 [6.0, 8.3] | 4.8 [3.4, 6.2] | 80 | 0.04 | — |
| NovelHopQA | Full document, 7B | 526 | 3.6 [2.1, 5.3] | 23.3 [21.0, 25.6] | 17.1 [13.9, 20.3] | 195,298 | 1.97 | 62.4 |
| NovelHopQA | SAADI top 10, 7B | 526 | 6.7 [4.6, 8.9] | 30.2 [27.6, 32.9] | 23.0 [19.6, 26.6] | 6,455 | 1.32 | — |
| NovelHopQA | SAADI top 10, C5 reader (30B-A3B) | 526 | 7.6 [5.5, 9.9] | 31.8 [29.2, 34.5] | 24.3 [20.9, 28.1] | 6,444 | — | — |
| NovelHopQA | Vanilla top 10, 7B | 526 | 6.3 [4.2, 8.4] | 30.5 [27.9, 33.1] | 22.8 [19.4, 26.4] | 6,449 | 1.33 | — |
| NovelHopQA | Closed-book, 7B | 526 | 1.0 [0.2, 1.9] | 8.1 [6.8, 9.5] | 1.7 [0.8, 2.9] | 80 | 0.03 | — |

Intervals: 95% bootstrap over queries (10,000 resamples, seed 42). Wall s / query: amortised over the run (full document: questions on one document share its prefix cache); 'without cache' = the document's uncached read, paid by every question.

## Paired differences (points, same queries)

| Dataset | Pair | n | EM | F1 | Containment |
| --- | --- | --- | --- | --- | --- |
| MuSiQue-32K | Full document - SAADI top 10 | 900 | +1.4 [-1.2, +4.1] | +2.3 [-0.4, +5.1] | +4.7 [+1.3, +8.1] |
| MuSiQue-32K | SAADI top 10 - Vanilla BM25 top 10 | 900 | +1.6 [-0.1, +3.2] | +1.4 [-0.3, +3.2] | +2.2 [-0.1, +4.6] |
| MuSiQue-32K | Full document - Vanilla BM25 top 10 | 900 | +3.0 [+0.3, +5.7] | +3.7 [+1.1, +6.4] | +6.9 [+3.6, +10.3] |
| MuSiQue-32K | Full document - Closed-book | 900 | +12.4 [+10.2, +14.8] | +21.4 [+19.1, +23.7] | +26.2 [+23.1, +29.3] |
| MuSiQue-32K | SAADI top 10 - Closed-book | 900 | +11.0 [+8.8, +13.3] | +19.1 [+16.7, +21.6] | +21.6 [+18.6, +24.6] |
| MuSiQue-32K | Full document - SAADI top 10, C5 reader (Qwen3-30B-A3B) | 900 | +2.3 [-0.4, +5.1] | +5.7 [+3.0, +8.4] | +0.7 [-2.9, +4.1] |
| MuSiQue-32K | SAADI top 10 - SAADI top 10, C5 reader (Qwen3-30B-A3B) | 900 | +0.9 [-1.6, +3.3] | +3.4 [+1.2, +5.7] | -4.0 [-6.8, -1.2] |
| NovelHopQA | Full document - SAADI top 10 | 526 | -3.0 [-4.9, -1.1] | -6.9 [-9.6, -4.3] | -5.9 [-9.1, -2.7] |
| NovelHopQA | SAADI top 10 - Vanilla BM25 top 10 | 526 | +0.4 [-1.1, +1.9] | -0.2 [-1.9, +1.4] | +0.2 [-1.7, +2.1] |
| NovelHopQA | Full document - Vanilla BM25 top 10 | 526 | -2.7 [-4.8, -0.8] | -7.2 [-9.8, -4.6] | -5.7 [-9.1, -2.3] |
| NovelHopQA | Full document - Closed-book | 526 | +2.7 [+1.0, +4.4] | +15.2 [+12.7, +17.7] | +15.4 [+12.2, +18.6] |
| NovelHopQA | SAADI top 10 - Closed-book | 526 | +5.7 [+3.6, +8.0] | +22.1 [+19.3, +24.9] | +21.3 [+17.7, +24.9] |
| NovelHopQA | Full document - SAADI top 10, C5 reader (Qwen3-30B-A3B) | 526 | -4.0 [-6.3, -1.7] | -8.5 [-11.2, -5.9] | -7.2 [-10.5, -4.0] |
| NovelHopQA | SAADI top 10 - SAADI top 10, C5 reader (Qwen3-30B-A3B) | 526 | -1.0 [-3.2, +1.1] | -1.6 [-3.5, +0.4] | -1.3 [-4.2, +1.5] |

## F1 by the position of the gold evidence

| Dataset | Condition | Subset | First third | Middle third | Last third |
| --- | --- | --- | --- | --- | --- |
| MuSiQue-32K | Full document | all | 28.3 (n=269) | 26.5 (n=369) | 31.8 (n=262) |
| MuSiQue-32K | Full document | evidence_in_one_third | 29.7 (n=201) | 25.3 (n=203) | 31.9 (n=206) |
| MuSiQue-32K | SAADI top 10 | all | 26.9 (n=269) | 24.7 (n=369) | 27.8 (n=262) |
| MuSiQue-32K | SAADI top 10 | evidence_in_one_third | 27.9 (n=201) | 24.4 (n=203) | 27.8 (n=206) |
| MuSiQue-32K | Vanilla BM25 top 10 | all | 25.3 (n=269) | 23.5 (n=369) | 26.2 (n=262) |
| MuSiQue-32K | Vanilla BM25 top 10 | evidence_in_one_third | 27.1 (n=201) | 23.3 (n=203) | 25.7 (n=206) |
| NovelHopQA | Full document | all | 22.5 (n=139) | 23.0 (n=228) | 26.1 (n=142) |
| NovelHopQA | Full document | evidence_in_one_third | 23.8 (n=107) | 24.2 (n=92) | 27.0 (n=109) |
| NovelHopQA | SAADI top 10 | all | 33.7 (n=139) | 29.0 (n=228) | 30.9 (n=142) |
| NovelHopQA | SAADI top 10 | evidence_in_one_third | 35.2 (n=107) | 32.2 (n=92) | 31.0 (n=109) |
| NovelHopQA | Vanilla BM25 top 10 | all | 35.3 (n=139) | 27.8 (n=228) | 31.2 (n=142) |
| NovelHopQA | Vanilla BM25 top 10 | evidence_in_one_third | 37.4 (n=107) | 29.0 (n=92) | 32.1 (n=109) |

## Cost detail

| Dataset | Condition | Input tokens | Output tokens | Request latency in the run (s) | Single request alone (s) | Amortised wall s | Without cache s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MuSiQue-32K | Full document | 48,904 | 10.2 | 1.73 | 11.4 / 0.41 | 0.74 | 11.4 |
| MuSiQue-32K | SAADI top 10 | 1,403 | 11.3 | 1.97 | 0.44 | 0.25 | — |
| MuSiQue-32K | Vanilla BM25 top 10 | 1,404 | 11.8 | 2.31 | 0.48 | 0.29 | — |
| MuSiQue-32K | Closed-book | 80 | 15.7 | 0.33 | 0.34 | 0.04 | — |
| MuSiQue-32K | SAADI top 10, C5 reader (Qwen3-30B-A3B) | 1,392 | 23.5 | — | — | — | — |
| NovelHopQA | Full document | 195,298 | 11.8 | 6.89 | 63.2 / 1.03 | 1.97 | 62.4 |
| NovelHopQA | SAADI top 10 | 6,455 | 12.3 | 10.01 | 1.32 | 1.32 | — |
| NovelHopQA | Vanilla BM25 top 10 | 6,449 | 12.3 | 9.97 | 1.58 | 1.33 | — |
| NovelHopQA | Closed-book | 80 | 10.9 | 0.24 | 0.19 | 0.03 | — |
| NovelHopQA | SAADI top 10, C5 reader (Qwen3-30B-A3B) | 6,444 | 15.6 | — | — | — | — |

## Documents

| Dataset | Distinct documents | Tokens min | median | max | Read in full (documents, tokens) | Not read in full |
| --- | --- | --- | --- | --- | --- | --- |
| MuSiQue-32K | 45 | 42,257 | 48,851 | 51,692 | 45 (42,257 to 51,692) | — |
| NovelHopQA | 16 | 133,420 | 232,803 | 759,549 | 10 (133,420 to 253,466) | B05, B15, B42, B44, B54, B62 (382 queries) |

## Files

- `results/rebuttal/c8_long_context/{table,paired,cost,position}.csv`, `answers/<dataset>/<condition>.jsonl`, `table_paper.tex`
- step reports: `experiments/rebuttal/c8_long_context/reports/`

<!-- c8-window-split:begin -->

## NovelHopQA beyond the native window

The full-document condition reads only the 10 books inside the native 262,144-token window. The retrieval conditions answered every question, so their accuracy is also given on the 6 longer books (264,945-759,549 tokens, Qwen2.5-7B-Instruct-1M tokenizer, 382 questions): with the same 7B reader, SAADI's top 10 gives F1 30.8 [28.0, 33.8] there, against 30.2 [27.6, 32.9] inside the window.

F1 in percent, 95% bootstrap interval over questions (10,000 resamples, seed 42; same order as step 03). File: `results/rebuttal/c8_long_context/per_dataset/novelhopqa/window_split.csv`.

| Books | Condition | n | Books (tokens) | EM | F1 | Containment |
| --- | --- | --- | --- | --- | --- | --- |
| inside_window | SAADI top 10 | 526 | 10 (133,420-253,466) | 6.7 [4.6, 8.9] | 30.2 [27.6, 32.9] | 23.0 [19.6, 26.6] |
| inside_window | Vanilla BM25 top 10 | 526 | 10 (133,420-253,466) | 6.3 [4.2, 8.4] | 30.5 [27.9, 33.1] | 22.8 [19.4, 26.4] |
| inside_window | Closed-book | 526 | 10 (133,420-253,466) | 1.0 [0.2, 1.9] | 8.1 [6.8, 9.5] | 1.7 [0.8, 2.9] |
| beyond_window | SAADI top 10 | 382 | 6 (264,945-759,549) | 6.3 [3.9, 8.9] | 30.8 [28.0, 33.8] | 19.4 [15.4, 23.6] |
| beyond_window | Vanilla BM25 top 10 | 382 | 6 (264,945-759,549) | 6.5 [4.2, 9.2] | 30.6 [27.7, 33.5] | 19.9 [16.0, 24.1] |
| beyond_window | Closed-book | 382 | 6 (264,945-759,549) | 0.3 [0.0, 0.8] | 9.1 [7.6, 10.7] | 1.6 [0.5, 2.9] |
| all_books | SAADI top 10 | 908 | 16 (133,420-759,549) | 6.5 [5.0, 8.1] | 30.5 [28.6, 32.5] | 21.5 [18.8, 24.2] |
| all_books | Vanilla BM25 top 10 | 908 | 16 (133,420-759,549) | 6.4 [4.8, 8.0] | 30.5 [28.6, 32.6] | 21.6 [18.9, 24.3] |
| all_books | Closed-book | 908 | 16 (133,420-759,549) | 0.7 [0.2, 1.2] | 8.5 [7.5, 9.6] | 1.7 [0.9, 2.5] |

<!-- c8-window-split:end -->
