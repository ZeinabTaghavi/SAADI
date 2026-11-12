# C8 long-context reader - step 05 NovelHopQA split by the native window: OK

Written 2026-11-12 14:03:27; commit b431bc33ef44.


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


Self-check: inside-window rows equal per_dataset/novelhopqa/table.csv.

## Files written

- `results/rebuttal/c8_long_context/per_dataset/novelhopqa/window_split.csv`
- `results/rebuttal/c8_long_context/REPORT.md`
- `experiments/rebuttal/c8_long_context/reports/step_05_window_split.md`
