# C5 end-to-end QA - step 02 reader, NovelHopQA: OK

Written 2026-11-10 23:39:45; commit dc923429fa20; host eee-s132-infsys-gpu03; GPUs 4,5,6,7; port 18532; 335 s.

Reader `Qwen/Qwen3-30B-A3B-Instruct-2507` (vLLM, tensor parallel 4), temperature 0, top_p 1, seed 42, max output tokens 64, 48 requests in flight. Prompt template: the one in step 00's report (c5_common.READER_TEMPLATE).

| Condition | Queries | Done before | Asked | Answered | Errors | Cut at max tokens | Prompt tokens != step 00 | Seconds | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| closed_book | 908 | 908 | 0 | 908 | 0 | 0 | 0 | 0.00 | complete |
| oracle | 908 | 908 | 0 | 908 | 0 | 0 | 0 | 0.00 | complete |
| oracle_para | 908 | 0 | 908 | 908 | 0 | 9 | 0 | 333.40 | complete |
| vanilla | 908 | 908 | 0 | 908 | 0 | 0 | 0 | 0.00 | complete |
| saadi | 908 | 908 | 0 | 908 | 0 | 0 | 0 | 0.00 | complete |
| cr | 908 | 908 | 0 | 908 | 0 | 0 | 0 | 0.00 | complete |
| raptor | 908 | 908 | 0 | 908 | 0 | 0 | 0 | 0.00 | complete |
| raptor_tm | 908 | 908 | 0 | 908 | 0 | 0 | 0 | 0.00 | complete |
| arag | 908 | 908 | 0 | 908 | 0 | 0 | 0 | 0.00 | complete |
| arag_tm | 908 | 908 | 0 | 908 | 0 | 0 | 0 | 0.00 | complete |
| saadi_at_raptor_len | 908 | 908 | 0 | 908 | 0 | 0 | 0 | 0.00 | complete |
| saadi_at_arag_len | 908 | 908 | 0 | 908 | 0 | 0 | 0 | 0.00 | complete |
| arag_own | 908 | 908 | 0 | 908 | 0 | 0 | 0 | 0.00 | complete |

## Files written

- `results/rebuttal/c5_end_to_end/answers/novelhopqa/closed_book.jsonl`
- `results/rebuttal/c5_end_to_end/answers/novelhopqa/oracle.jsonl`
- `results/rebuttal/c5_end_to_end/answers/novelhopqa/oracle_para.jsonl`
- `results/rebuttal/c5_end_to_end/answers/novelhopqa/vanilla.jsonl`
- `results/rebuttal/c5_end_to_end/answers/novelhopqa/saadi.jsonl`
- `results/rebuttal/c5_end_to_end/answers/novelhopqa/cr.jsonl`
- `results/rebuttal/c5_end_to_end/answers/novelhopqa/raptor.jsonl`
- `results/rebuttal/c5_end_to_end/answers/novelhopqa/raptor_tm.jsonl`
- `results/rebuttal/c5_end_to_end/answers/novelhopqa/arag.jsonl`
- `results/rebuttal/c5_end_to_end/answers/novelhopqa/arag_tm.jsonl`
- `results/rebuttal/c5_end_to_end/answers/novelhopqa/saadi_at_raptor_len.jsonl`
- `results/rebuttal/c5_end_to_end/answers/novelhopqa/saadi_at_arag_len.jsonl`
- `results/rebuttal/c5_end_to_end/answers/novelhopqa/arag_own.jsonl`
- `experiments/rebuttal/c5_end_to_end/reports/step_02_novelhopqa_reader.md`

## Push

```
pushed cfe1d4f to origin/fix/notebook-key-and-install: C5 novelhopqa 02_reader: reader answered 11804 queries over 13 conditions
```
