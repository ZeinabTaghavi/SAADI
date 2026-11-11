# C8 long-context reader - step 99 final report: OK

Written 2026-11-11 19:59:34; commit 2f88145e547a; 0 s.

Command:

```bash
bash experiments/rebuttal/c8_long_context/sh/99_report.sh
```

## Opening of REPORT.md

On MuSiQue-32K (900 queries over 45 documents of 42,257 to 51,692 tokens), Qwen2.5-7B-Instruct-1M reading the whole document reaches F1 28.6 [26.3, 30.9], not distinguishable from the same model reading SAADI's top 10 (26.3 [23.9, 28.6]; difference +2.3 [-0.4, +5.1] points); SAADI's top 10 read by C5's reader (Qwen3-30B-A3B) gives 22.8 [20.7, 25.0].
With the 7B reader, SAADI minus Vanilla BM25 top 10 is +1.4 [-0.3, +3.2] F1 points (EM +1.6 [-0.1, +3.2]).
The full document costs 48,904 input tokens per query against 1,403 for SAADI's top 10 (35x); in wall-clock time it took 0.74 s per query when the questions on one document share its prefix cache and 11.4 s per query without the cache, against 0.25 s for SAADI's top 10 (single request alone: 11.4 s uncached full document, 0.44 s SAADI; 4 RTX A6000).
By the position of the gold evidence, full-document F1 is 28.3 / 26.5 / 31.8 in the first / middle / last third (n = 269 / 369 / 262; range 5.3 points), and SAADI's 26.9 / 24.7 / 27.8.
On NovelHopQA (526 queries over 10 documents of 133,420 to 253,466 tokens), Qwen2.5-7B-Instruct-1M reading the whole document reaches F1 23.3 [21.0, 25.6], lower than the same model reading SAADI's top 10 (30.2 [27.6, 32.9]; difference -6.9 [-9.6, -4.3] points); SAADI's top 10 read by C5's reader (Qwen3-30B-A3B) gives 31.8 [29.2, 34.5].
With the 7B reader, SAADI minus Vanilla BM25 top 10 is -0.2 [-1.9, +1.4] F1 points (EM +0.4 [-1.1, +1.9]).
The full document costs 195,298 input tokens per query against 6,455 for SAADI's top 10 (30x); in wall-clock time it took 1.97 s per query when the questions on one document share its prefix cache and 62.4 s per query without the cache, against 1.32 s for SAADI's top 10 (single request alone: 63.2 s uncached full document, 1.32 s SAADI; 4 RTX A6000).
By the position of the gold evidence, full-document F1 is 22.5 / 23.0 / 26.1 in the first / middle / last third (n = 139 / 228 / 142; range 3.6 points), and SAADI's 33.7 / 29.0 / 30.9.
Caveats: NovelHopQA: the server's vLLM has no dual chunk attention, so the window is the model's native 262,144 tokens; the 6 longer books (264,945 to 759,549 tokens, 382 queries) were not read in full and their queries are left out of every condition, so the full-document comparison covers the 10 shorter books (526 queries). NovelHopQA's evidence positions use C5's paragraph labels (the paper's window labels accept one-word matches); answer scores do not depend on labels.

## Files written

- `results/rebuttal/c8_long_context/table.csv`
- `results/rebuttal/c8_long_context/paired.csv`
- `results/rebuttal/c8_long_context/cost.csv`
- `results/rebuttal/c8_long_context/position.csv`
- `results/rebuttal/c8_long_context/REPORT.md`
- `results/rebuttal/c8_long_context/table_paper.tex`
- `experiments/rebuttal/c8_long_context/reports/step_99_report.md`
