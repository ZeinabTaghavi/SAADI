# C8 long-context reader - step 02 reader, NovelHopQA: OK

Written 2026-11-11 14:23:34; commit ddcda651af37; host eee-s132-infsys-gpu03; GPUs 4,5,6,7; 3468 s.

Reader `Qwen/Qwen2.5-7B-Instruct-1M` (vLLM, tensor parallel 4, mode native, --max-model-len 262,144), temperature 0, top_p 1, seed 42, max output tokens 64, 8 requests in flight (full document: the first question of each document alone). Queries: 908 (all).

| Condition | Selected | Done before | Asked | Answered | Skipped (too long) | Errors | Not started (deadline) | Cut at max tokens | Prompt tokens != step 00 | Wall s | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Closed-book | 908 | 0 | 908 | 908 | 0 | 0 | 0 | 1 | 0 | 28.20 | complete |
| Vanilla BM25 top 10 | 908 | 0 | 908 | 908 | 0 | 0 | 0 | 0 | 0 | 1203.80 | complete |
| SAADI top 10 | 908 | 0 | 908 | 908 | 0 | 0 | 0 | 0 | 0 | 1199.70 | complete |
| Full document | 908 | 0 | 526 | 526 | 382 | 0 | 0 | 0 | 0 | 1036.40 | complete |

Full document: 382 queries skipped on documents B05, B15, B42, B44, B54, B62: document beyond the native 262,144-token window and the server runs without dual chunk attention.

Full document, per document (this run):

| Document | Tokens | Uncached first question (s) | Cache reset | Other questions | Their mean latency (s) | Document wall (s) |
| --- | --- | --- | --- | --- | --- | --- |
| B27 | 201,836 | 64.8 | True | 59 | 6.08 | 111.9 |
| B34 | 206,833 | 66.7 | True | 64 | 6.15 | 118.1 |
| B04 | 209,488 | 67.7 | True | 44 | 6.17 | 103.9 |
| B37 | 167,704 | 50.4 | True | 61 | 4.84 | 89.0 |
| B24 | 133,420 | 37.6 | True | 64 | 4.02 | 71.6 |
| B18 | 194,220 | 61.2 | True | 43 | 5.93 | 94.8 |
| B26 | 253,466 | 88.2 | True | 49 | 7.63 | 138.4 |
| B61 | 225,064 | 74.9 | True | 13 | 5.82 | 86.9 |
| B12 | 240,542 | 82.1 | True | 60 | 7.01 | 137.1 |
| B43 | 159,866 | 47.3 | True | 59 | 4.81 | 84.5 |

## Files written

- `results/rebuttal/c8_long_context/answers/novelhopqa/closed_book.jsonl`
- `results/rebuttal/c8_long_context/answers/novelhopqa/vanilla.jsonl`
- `results/rebuttal/c8_long_context/answers/novelhopqa/saadi.jsonl`
- `results/rebuttal/c8_long_context/answers/novelhopqa/full_doc_documents.jsonl`
- `results/rebuttal/c8_long_context/answers/novelhopqa/full_doc.jsonl`
- `results/rebuttal/c8_long_context/answers/novelhopqa/full_doc.skipped.json`
- `experiments/rebuttal/c8_long_context/reports/step_02_novelhopqa_reader.md`
