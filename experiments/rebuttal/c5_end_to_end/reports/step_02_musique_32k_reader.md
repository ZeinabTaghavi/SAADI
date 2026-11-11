# C5 end-to-end QA - step 02 reader, MuSiQue-32K: OK

Written 2026-11-11 02:01:07; commit c5e4bf426667; host eee-s132-infsys-gpu03; GPUs 0,1,2,3; port 18531; 1432 s.

Reader `Qwen/Qwen3-30B-A3B-Instruct-2507` (vLLM, tensor parallel 4), temperature 0, top_p 1, seed 42, max output tokens 64, 48 requests in flight. Prompt template: the one in step 00's report (c5_common.READER_TEMPLATE).

| Condition | Queries | Done before | Asked | Answered | Errors | Cut at max tokens | Prompt tokens != step 00 | Seconds | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| closed_book | 900 | 0 | 900 | 900 | 0 | 124 | 0 | 25.70 | complete |
| oracle | 900 | 0 | 900 | 900 | 0 | 58 | 0 | 158.40 | complete |
| oracle_para | — | — | — | — | — | — | — | — | no contexts (missing in step 00) |
| vanilla | 900 | 0 | 900 | 900 | 0 | 98 | 0 | 248.80 | complete |
| saadi | 900 | 0 | 900 | 900 | 0 | 98 | 0 | 251.70 | complete |
| cr | 900 | 0 | 900 | 900 | 0 | 109 | 0 | 253.20 | complete |
| raptor | 900 | 0 | 900 | 900 | 0 | 106 | 0 | 177.70 | complete |
| raptor_tm | 900 | 0 | 900 | 900 | 0 | 101 | 0 | 20.50 | complete |
| arag | 900 | 0 | 900 | 900 | 0 | 93 | 0 | 79.90 | complete |
| arag_tm | 900 | 0 | 900 | 900 | 0 | 97 | 0 | 18.70 | complete |
| saadi_at_raptor_len | 900 | 0 | 900 | 900 | 0 | 91 | 0 | 169.70 | complete |
| saadi_at_arag_len | 900 | 0 | 900 | 900 | 0 | 112 | 0 | 26.20 | complete |
| arag_own | 900 | 0 | 900 | 900 | 0 | 0 | 0 | 0.10 | complete |

## Files written

- `results/rebuttal/c5_end_to_end/answers/musique_32k/closed_book.jsonl`
- `results/rebuttal/c5_end_to_end/answers/musique_32k/oracle.jsonl`
- `results/rebuttal/c5_end_to_end/answers/musique_32k/vanilla.jsonl`
- `results/rebuttal/c5_end_to_end/answers/musique_32k/saadi.jsonl`
- `results/rebuttal/c5_end_to_end/answers/musique_32k/cr.jsonl`
- `results/rebuttal/c5_end_to_end/answers/musique_32k/raptor.jsonl`
- `results/rebuttal/c5_end_to_end/answers/musique_32k/raptor_tm.jsonl`
- `results/rebuttal/c5_end_to_end/answers/musique_32k/arag.jsonl`
- `results/rebuttal/c5_end_to_end/answers/musique_32k/arag_tm.jsonl`
- `results/rebuttal/c5_end_to_end/answers/musique_32k/saadi_at_raptor_len.jsonl`
- `results/rebuttal/c5_end_to_end/answers/musique_32k/saadi_at_arag_len.jsonl`
- `results/rebuttal/c5_end_to_end/answers/musique_32k/arag_own.jsonl`
- `experiments/rebuttal/c5_end_to_end/reports/step_02_musique_32k_reader.md`
