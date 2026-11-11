# C8 long-context reader - step 02 reader, MuSiQue-32K: OK

Written 2026-11-11 13:25:37; commit dea82d8cef87; host eee-s132-infsys-gpu03; GPUs 4,5,6,7; 1187 s.

Reader `Qwen/Qwen2.5-7B-Instruct-1M` (vLLM, tensor parallel 4, mode native, --max-model-len 262,144), temperature 0, top_p 1, seed 42, max output tokens 64, 8 requests in flight (full document: the first question of each document alone). Queries: 900 (all).

| Condition | Selected | Done before | Asked | Answered | Skipped (too long) | Errors | Not started (deadline) | Cut at max tokens | Prompt tokens != step 00 | Wall s | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Closed-book | 900 | 0 | 900 | 900 | 0 | 0 | 0 | 22 | 0 | 37.20 | complete |
| Vanilla BM25 top 10 | 900 | 0 | 900 | 900 | 0 | 0 | 0 | 20 | 0 | 261.80 | complete |
| SAADI top 10 | 900 | 0 | 900 | 900 | 0 | 0 | 0 | 21 | 0 | 223.20 | complete |
| Full document | 900 | 0 | 900 | 900 | 0 | 0 | 0 | 1 | 0 | 664.70 | complete |

Full document, per document (this run):

| Document | Tokens | Uncached first question (s) | Cache reset | Other questions | Their mean latency (s) | Document wall (s) |
| --- | --- | --- | --- | --- | --- | --- |
| group_030 | 49,287 | 11.4 | True | 19 | 1.45 | 15.7 |
| group_035 | 48,527 | 11.4 | True | 19 | 1.42 | 15.2 |
| group_045 | 42,257 | 9.7 | True | 18 | 1.04 | 12.4 |
| group_024 | 48,555 | 11.2 | True | 18 | 1.22 | 14.2 |
| group_027 | 49,601 | 11.6 | True | 19 | 1.27 | 15.0 |
| group_022 | 50,140 | 11.6 | True | 20 | 1.24 | 15.4 |
| group_043 | 47,764 | 11.1 | True | 17 | 1.40 | 14.4 |
| group_033 | 48,851 | 11.7 | True | 17 | 1.16 | 14.7 |
| group_005 | 50,785 | 11.8 | True | 19 | 1.20 | 15.2 |
| group_025 | 48,271 | 11.2 | True | 20 | 1.11 | 14.4 |
| group_036 | 49,610 | 11.6 | True | 19 | 1.23 | 14.9 |
| group_008 | 48,604 | 11.3 | True | 19 | 1.26 | 14.6 |
| group_029 | 49,548 | 11.5 | True | 19 | 1.31 | 15.2 |
| group_010 | 48,383 | 11.3 | True | 21 | 1.11 | 14.5 |
| group_021 | 47,009 | 10.8 | True | 18 | 1.25 | 14.1 |
| group_006 | 47,459 | 11.0 | True | 19 | 1.15 | 14.5 |
| group_019 | 47,765 | 11.0 | True | 18 | 1.17 | 14.3 |
| group_041 | 50,450 | 11.9 | True | 19 | 1.31 | 15.3 |
| group_017 | 48,958 | 11.5 | True | 19 | 1.31 | 15.0 |
| group_038 | 49,881 | 11.5 | True | 21 | 1.21 | 15.2 |
| group_028 | 46,497 | 10.8 | True | 17 | 1.15 | 13.6 |
| group_032 | 49,564 | 11.7 | True | 19 | 1.15 | 14.9 |
| group_026 | 49,139 | 11.4 | True | 21 | 1.13 | 14.9 |
| group_040 | 49,226 | 11.4 | True | 18 | 1.20 | 14.9 |
| group_011 | 47,764 | 11.5 | True | 17 | 1.19 | 14.3 |
| group_004 | 47,379 | 10.9 | True | 20 | 1.20 | 14.2 |
| group_039 | 46,991 | 10.8 | True | 18 | 1.32 | 14.4 |
| group_042 | 51,692 | 12.1 | True | 19 | 1.28 | 15.6 |
| group_018 | 49,538 | 11.5 | True | 19 | 1.17 | 14.6 |
| group_031 | 49,578 | 12.1 | True | 20 | 1.22 | 15.7 |
| group_007 | 49,358 | 11.6 | True | 20 | 1.20 | 14.9 |
| group_016 | 50,435 | 11.9 | True | 21 | 1.22 | 15.6 |
| group_012 | 50,680 | 11.8 | True | 18 | 1.20 | 14.7 |
| group_023 | 48,363 | 11.2 | True | 20 | 1.27 | 14.9 |
| group_020 | 48,975 | 11.3 | True | 21 | 1.22 | 15.0 |
| group_001 | 48,333 | 11.4 | True | 18 | 1.18 | 14.6 |
| group_013 | 51,117 | 11.9 | True | 20 | 1.22 | 15.6 |
| group_015 | 48,201 | 11.2 | True | 19 | 1.21 | 14.4 |
| group_044 | 50,078 | 11.7 | True | 20 | 1.19 | 14.9 |
| group_003 | 48,518 | 11.4 | True | 18 | 1.21 | 14.9 |
| group_037 | 50,260 | 11.7 | True | 20 | 1.31 | 15.2 |
| group_034 | 48,050 | 11.1 | True | 18 | 1.23 | 14.2 |
| group_002 | 48,586 | 11.5 | True | 18 | 1.24 | 14.8 |
| group_014 | 47,677 | 11.0 | True | 19 | 1.26 | 14.2 |
| group_009 | 47,983 | 11.5 | True | 19 | 1.13 | 14.5 |

## Files written

- `results/rebuttal/c8_long_context/answers/musique_32k/closed_book.jsonl`
- `results/rebuttal/c8_long_context/answers/musique_32k/vanilla.jsonl`
- `results/rebuttal/c8_long_context/answers/musique_32k/saadi.jsonl`
- `results/rebuttal/c8_long_context/answers/musique_32k/full_doc_documents.jsonl`
- `results/rebuttal/c8_long_context/answers/musique_32k/full_doc.jsonl`
- `experiments/rebuttal/c8_long_context/reports/step_02_musique_32k_reader.md`
