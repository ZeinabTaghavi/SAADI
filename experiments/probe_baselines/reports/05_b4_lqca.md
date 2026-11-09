# probe_baselines - stage 05_b4_lqca

Status: **ok** - 2026-11-09 10:24:46; wall clock 52.6 min

## What was done

- `/home/iataghav/data/SAADI/experiments/probe_baselines/third_party/venv_lqca/bin/python /home/iataghav/data/SAADI/experiments/probe_baselines/systems/b4_lqca/build.py`
- `python3 /home/iataghav/data/SAADI/experiments/probe_baselines/systems/b4_lqca/retrieve.py`
- git commit `aba11eead6837a7fae635e18972a849822a81381`; python 3.12.3 on Linux-6.17.0-23-generic-x86_64-with-glibc2.39

## Numbers

### B4 LQCA rewrite per book (official code on segments of 8 chunks)

| book | segments | rewritten ok | split failed | pipeline error | chunks changed | seconds (all segments) |
| --- | --- | --- | --- | --- | --- | --- |
| B31 | 31 | 20 | 11 | 0 | 156 | 1760.4 |
| B43 | 31 | 22 | 9 | 0 | 172 | 2446.8 |
| B37 | 33 | 17 | 16 | 0 | 136 | 1664.7 |
| B18 | 39 | 26 | 13 | 0 | 208 | 2136.2 |
| B04 | 41 | 24 | 17 | 0 | 185 | 2383.7 |
| B34 | 42 | 24 | 18 | 0 | 189 | 2785.3 |
| B27 | 43 | 25 | 17 | 1 | 193 | 2167.6 |
| B61 | 45 | 22 | 22 | 1 | 172 | 2262.1 |
| B26 | 48 | 34 | 14 | 0 | 271 | 3093.2 |
| B44 | 49 | 28 | 19 | 2 | 224 | 2936.2 |
### B4: the rewritten gold chunk

| cell | n | gold chunk rewritten at all | rewritten gold chunk holds form 2 |
| --- | --- | --- | --- |
| 1 | 30 | 17 | 0 |
| 2 | 30 | 17 | 0 |
| 3 | 30 | 19 | 0 |
| 4 | 30 | 17 | 0 |
| 5 | 30 | 21 | 0 |
| 6 | 30 | 20 | 0 |
| 7 | 30 | 14 | 0 |
| 8 | 30 | 21 | 0 |

## Anything unexpected

(build) 160 segments kept their original text (pipeline error or split failure); see segments.jsonl

## Files written

- `/home/iataghav/data/SAADI/outputs/probe_baselines/b4_lqca/<book>/segments.jsonl`
- `/home/iataghav/data/SAADI/outputs/probe_baselines/rankings/b4_lqca/<book>.jsonl`
