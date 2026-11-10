# C5 end-to-end QA - step 02 reader, NovelHopQA: OK

Written 2026-11-10 21:51:54; commit c492f48ab0c5; host eee-s132-infsys-gpu03; GPUs 4,5,6,7; port 18532; 6468 s.

Reader `Qwen/Qwen3-30B-A3B-Instruct-2507` (vLLM, tensor parallel 4), temperature 0, top_p 1, seed 42, max output tokens 64, 48 requests in flight. Prompt template: the one in step 00's report (c5_common.READER_TEMPLATE).

| Condition | Queries | Done before | Asked | Answered | Errors | Cut at max tokens | Prompt tokens != step 00 | Seconds | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| closed_book | 908 | 0 | 908 | 908 | 0 | 5 | 0 | 28.00 | complete |
| oracle | 908 | 0 | 908 | 908 | 0 | 10 | 0 | 367.40 | complete |
| vanilla | 908 | 0 | 908 | 908 | 0 | 9 | 0 | 1259.60 | complete |
| saadi | 908 | 0 | 908 | 908 | 0 | 12 | 0 | 1256.60 | complete |
| cr | 908 | 0 | 908 | 908 | 0 | 12 | 0 | 1257.90 | complete |
| raptor | 908 | 0 | 908 | 908 | 0 | 5 | 0 | 312.20 | complete |
| raptor_tm | 908 | 0 | 908 | 908 | 0 | 6 | 0 | 315.60 | complete |
| arag | 908 | 0 | 908 | 908 | 0 | 4 | 0 | 438.40 | complete |
| arag_tm | 908 | 0 | 908 | 908 | 0 | 4 | 0 | 443.30 | complete |
| saadi_at_raptor_len | 908 | 0 | 908 | 908 | 0 | 5 | 0 | 313.20 | complete |
| saadi_at_arag_len | 908 | 0 | 908 | 908 | 0 | 6 | 0 | 473.40 | complete |
| arag_own | 908 | 0 | 908 | 908 | 0 | 0 | 0 | 0.20 | complete |

## Files written

- `results/rebuttal/c5_end_to_end/answers/novelhopqa/closed_book.jsonl`
- `results/rebuttal/c5_end_to_end/answers/novelhopqa/oracle.jsonl`
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
