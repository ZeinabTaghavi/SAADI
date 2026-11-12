# C8 long-context reader - step 02 reader, NovelHopQA: FAILED

Written 2026-11-12 16:40:00; commit 9732e142077c; host eee-s132-infsys-gpu03; GPUs 4,5,6,7; 1483 s.

Reader `Qwen/Qwen2.5-7B-Instruct-1M` (vLLM, tensor parallel 4, mode dca, --max-model-len 778,240), temperature 0, top_p 1, seed 42, max output tokens 64, 8 requests in flight (full document: the first question of each document alone). Queries: 584 (all).

| Condition | Selected | Done before | Asked | Answered | Skipped (too long) | Errors | Not started (deadline) | Cut at max tokens | Prompt tokens != step 00 | Wall s | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Full document | 584 | 0 | 584 | 459 | 0 | 0 | 0 | 0 | 0 | 1483.10 | INCOMPLETE |

Full document, per document (this run):

| Document | Tokens | Uncached first question (s) | Cache reset | Other questions | Their mean latency (s) | Document wall (s) |
| --- | --- | --- | --- | --- | --- | --- |
| B44 | 264,945 | 75.1 | True | 56 | 10.95 | 153.9 |
| B15 | 273,811 | 76.6 | True | 74 | 11.29 | 185.6 |
| B54 | 411,934 | 122.8 | True | 73 | 16.93 | 280.8 |
| B62 | 461,227 | 139.5 | True | 83 | 18.19 | 333.1 |
| B24 | 133,420 | 34.4 | True | 64 | 5.96 | 84.3 |
| B43 | 159,866 | 41.9 | True | 59 | 6.67 | 93.4 |
| B18 | 194,220 | 52.5 | True | 43 | 8.64 | 101.9 |
| B27 | 201,836 | 54.5 | True | 59 | 8.90 | 121.0 |
| B34 | 206,833 | 56.2 | True | 64 | 8.71 | 128.9 |

## Error

```text
Traceback (most recent call last):
  File "/home/iataghav/data/SAADI/experiments/rebuttal/c8_long_context/scripts/step02_reader.py", line 190, in main
    raise RuntimeError(f"unanswered queries remain in {bad} (errors in the report); start the GPU script again to resume")
RuntimeError: unanswered queries remain in ['full_doc'] (errors in the report); start the GPU script again to resume
```

## Files written

- `results/rebuttal/c8_long_context/dca/shard_1/answers/novelhopqa/full_doc_documents.jsonl`
- `results/rebuttal/c8_long_context/dca/shard_1/answers/novelhopqa/full_doc.jsonl`
- `experiments/rebuttal/c8_long_context/reports/dca/shard_1/step_02_novelhopqa_reader.md`
