# probe_baselines - stage 01_b0_vanilla

Status: **ok** - 2026-11-08 22:54:17; wall clock 0.1 min

## What was done

- `python3 /home/iataghav/data/SAADI/experiments/probe_baselines/systems/b0_vanilla/build.py`
- `python3 /home/iataghav/data/SAADI/experiments/probe_baselines/systems/b0_vanilla/retrieve.py`
- git commit `28eed3a1ee31487f3bf23dcb3470aa0e164e8cfd`; python 3.12.3 on Linux-6.17.0-23-generic-x86_64-with-glibc2.39

## Numbers

### B0 Vanilla BM25 against the probe's own Vanilla

Per persona against run 0's Vanilla rank (outputs/probe_bridge/per_persona.jsonl): 240 of 240 identical.

Per cell against results/probe_bridge/table.csv (B0 / probe Vanilla):

| cell | HR@5 | HR@10 | median rank | check |
| --- | --- | --- | --- | --- |
| 1 | 0.0 / 0.0 | 3.3 / 3.3 | 143.5 / 143.5 | identical |
| 2 | 3.3 / 3.3 | 3.3 / 3.3 | 124.0 / 124.0 | identical |
| 3 | 0.0 / 0.0 | 3.3 / 3.3 | 162.0 / 162.0 | identical |
| 4 | 0.0 / 0.0 | 6.7 / 6.7 | 118.5 / 118.5 | identical |
| 5 | 0.0 / 0.0 | 0.0 / 0.0 | 152.5 / 152.5 | identical |
| 6 | 0.0 / 0.0 | 0.0 / 0.0 | 167.5 / 167.5 | identical |
| 7 | 0.0 / 0.0 | 0.0 / 0.0 | 174.5 / 174.5 | identical |
| 8 | 0.0 / 0.0 | 3.3 / 3.3 | 128.0 / 128.0 | identical |

### Cost per book

| book | chunks | index size (chars) | seconds |
| --- | --- | --- | --- |
| B31 | 244 | 593067 | 0.49 |
| B43 | 246 | 667227 | 0.26 |
| B37 | 262 | 717260 | 0.27 |
| B18 | 312 | 840509 | 0.34 |
| B04 | 321 | 882386 | 0.34 |
| B34 | 335 | 901193 | 0.36 |
| B27 | 338 | 902949 | 0.35 |
| B61 | 357 | 958986 | 0.38 |
| B26 | 384 | 1028207 | 0.45 |
| B44 | 385 | 1034251 | 0.39 |

## Anything unexpected

Nothing unexpected.

## Files written

- `/home/iataghav/data/SAADI/outputs/probe_baselines/rankings/b0_vanilla/<book>.jsonl`
