# C8 long-context reader - step 02 reader, NovelHopQA: OK

Written 2026-11-12 16:35:54; commit 9732e142077c; host eee-s132-infsys-gpu03; GPUs 0,1,2,3; 1237 s.

Reader `Qwen/Qwen2.5-7B-Instruct-1M` (vLLM, tensor parallel 4, mode dca, --max-model-len 778,240), temperature 0, top_p 1, seed 42, max output tokens 64, 8 requests in flight (full document: the first question of each document alone). Queries: 324 (all).

| Condition | Selected | Done before | Asked | Answered | Skipped (too long) | Errors | Not started (deadline) | Cut at max tokens | Prompt tokens != step 00 | Wall s | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Full document | 324 | 0 | 324 | 324 | 0 | 0 | 0 | 0 | 0 | 1236.60 | complete |

Full document, per document (this run):

| Document | Tokens | Uncached first question (s) | Cache reset | Other questions | Their mean latency (s) | Document wall (s) |
| --- | --- | --- | --- | --- | --- | --- |
| B05 | 324,835 | 93.3 | True | 21 | 12.59 | 127.1 |
| B42 | 759,549 | 245.9 | True | 69 | 32.99 | 543.0 |
| B37 | 167,704 | 45.3 | True | 61 | 6.97 | 99.5 |
| B04 | 209,488 | 56.7 | True | 44 | 8.52 | 106.8 |
| B61 | 225,064 | 61.2 | True | 13 | 8.59 | 78.8 |
| B12 | 240,542 | 67.3 | True | 60 | 9.95 | 145.3 |
| B26 | 253,466 | 69.5 | True | 49 | 10.56 | 136.0 |

## Files written

- `results/rebuttal/c8_long_context/dca/shard_0/answers/novelhopqa/full_doc_documents.jsonl`
- `results/rebuttal/c8_long_context/dca/shard_0/answers/novelhopqa/full_doc.jsonl`
- `experiments/rebuttal/c8_long_context/reports/dca/shard_0/step_02_novelhopqa_reader.md`

## Push

```
pushed d5629c0 to origin/fix/notebook-key-and-install: C8 novelhopqa 04_dca_unit0: answered full_doc 324
```
