# C8 long-context reader - step 02 reader, NovelHopQA: OK

Written 2026-11-12 16:45:20; commit da21d34868c5; host eee-s132-infsys-gpu03; GPUs 0,1,2,3; 240 s.

Reader `Qwen/Qwen2.5-7B-Instruct-1M` (vLLM, tensor parallel 4, mode dca, --max-model-len 778,240), temperature 0, top_p 1, seed 42, max output tokens 64, 8 requests in flight (full document: the first question of each document alone). Queries: 584 (all).

| Condition | Selected | Done before | Asked | Answered | Skipped (too long) | Errors | Not started (deadline) | Cut at max tokens | Prompt tokens != step 00 | Wall s | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Full document | 584 | 459 | 125 | 584 | 0 | 0 | 0 | 0 | 0 | 239.40 | complete |

Full document, per document (this run):

| Document | Tokens | Uncached first question (s) | Cache reset | Other questions | Their mean latency (s) | Document wall (s) |
| --- | --- | --- | --- | --- | --- | --- |
| B27 | 201,836 | 54.1 | True | 59 | 8.26 | 116.5 |
| B34 | 206,833 | 55.6 | True | 64 | 7.95 | 122.8 |

## Files written

- `results/rebuttal/c8_long_context/dca/shard_1/answers/novelhopqa/full_doc_documents.jsonl`
- `results/rebuttal/c8_long_context/dca/shard_1/answers/novelhopqa/full_doc.jsonl`
- `experiments/rebuttal/c8_long_context/reports/dca/shard_1/step_02_novelhopqa_reader.md`

## Push

```
pushed 93057fb to origin/fix/notebook-key-and-install: C8 novelhopqa 04_dca_unit1: answered full_doc 584
```
