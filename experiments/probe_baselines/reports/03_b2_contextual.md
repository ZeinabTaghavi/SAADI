# probe_baselines - stage 03_b2_contextual

Status: **ok** - 2026-11-09 00:40:23; wall clock 106.0 min

## What was done

- `python3 /home/iataghav/data/SAADI/experiments/probe_baselines/systems/b2_contextual/build.py`
- `python3 /home/iataghav/data/SAADI/experiments/probe_baselines/systems/b2_contextual/retrieve.py`
- git commit `28eed3a1ee31487f3bf23dcb3470aa0e164e8cfd`; python 3.12.3 on Linux-6.17.0-23-generic-x86_64-with-glibc2.39

## Numbers

### B2 contexts per book (served window 262144 tokens)

| book | chunks | book tokens | visible | calls | input tokens | generated tokens | median context tokens | cut at max tokens | seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B31 | 244 | 166636 | whole | 244 | 40783611 | 25469 | 103.0 | 2 | 358.3 |
| B43 | 246 | 170429 | whole | 246 | 42052668 | 34948 | 140.0 | 12 | 415.6 |
| B37 | 262 | 169933 | whole | 262 | 44643392 | 35550 | 135.0 | 8 | 431.9 |
| B18 | 312 | 196267 | whole | 312 | 61356069 | 43298 | 136.0 | 12 | 573.1 |
| B04 | 321 | 220615 | whole | 321 | 70962606 | 41941 | 129 | 9 | 653.5 |
| B34 | 335 | 222077 | whole | 335 | 74529434 | 41416 | 124 | 6 | 661.8 |
| B27 | 338 | 217411 | whole | 338 | 73612424 | 47747 | 140.5 | 15 | 683.4 |
| B61 | 357 | 227130 | whole | 357 | 81210798 | 51756 | 146 | 41 | 752.7 |
| B26 | 384 | 267977 | prefix to chunk 370 | 384 | 99920587 | 51959 | 134.5 | 14 | 895.1 |
| B44 | 385 | 268771 | prefix to chunk 371 | 385 | 100127378 | 54853 | 142 | 21 | 914.3 |

Input tokens are the logical prompt tokens of every call (the whole visible book each time); the server's prefix cache recomputes only the chunk part after the first call of a book (server counters per book are in the JSON).
### B2: reveal visible when the gold chunk's context was written, and the written context

| cell | n | reveal visible | context holds form 2 | context holds form 1 |
| --- | --- | --- | --- | --- |
| 1 | 30 | 30/30 | 0 | 0 |
| 2 | 30 | 30/30 | 0 | 0 |
| 3 | 30 | 30/30 | 0 | 0 |
| 4 | 30 | 30/30 | 0 | 0 |
| 5 | 30 | 0/0 | 0 | 0 |
| 6 | 30 | 30/30 | 0 | 0 |
| 7 | 30 | 30/30 | 0 | 0 |
| 8 | 30 | 30/30 | 0 | 0 |

## Anything unexpected

Nothing unexpected.

## Files written

- `/home/iataghav/data/SAADI/outputs/probe_baselines/b2_contextual/contexts/<book>.jsonl`
- `/home/iataghav/data/SAADI/outputs/probe_baselines/rankings/b2_contextual/<book>.jsonl`
