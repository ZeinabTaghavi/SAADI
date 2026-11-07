# Bridge probe - step 05: Index (GPU)

Status: **ok** - 2026-11-07 23:11:30

## What was done

- `bash experiments/probe_bridge/sh/05_*.sh`
- `SADI_OUTPUT_DIR=/mnt/data/zeinab/SAADI/outputs/probe_bridge /mnt/data/zeinab/raptor/.venv/bin/python3 /mnt/data/zeinab/SAADI/scripts/run_study.py --suite probe_bridge --profile published --datasets novelhopqa_probe --retrievers bm25 --stage ingest`
- Versions: python=3.12.3, platform=Linux-6.17.0-23-generic-x86_64-with-glibc2.39, numpy=2.3.5, pandas=3.0.1, yaml=6.0.3, saadi=0.1.0, git_commit=90380fadb3b5b557fb5b6d032baf74f04bc6188f

## Numbers

### Frozen spec

personas.jsonl SHA-256 verified: `11c4c1696c67171f78821d3ba8311ccd28c851315ca1c2a404a9a422593897bb`

### Runtime environment

| setting | value |
| --- | --- |
| HF_HOME | /mnt/cache/taghavi |
| HF_HUB_CACHE | /mnt/cache/taghavi/hub |
| Model weights already in the cache | True |
| HF_HOME writable | True |
| CUDA_VISIBLE_DEVICES | 0,1,2,3 |

### Resolved construction settings

| setting | value |
| --- | --- |
| temperature | 0.0 |
| seed | 42 |
| memory selector | word |
| history text | anchored |
| anchor matching | strict |
| recall budget (records, aliases, descriptions) | 5, 2, 3 |
| cached model snapshots | 0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe |
| paper's checkpoint | 0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe |

### Wall clock

3.23 hours (11619 s) this invocation.

### Books

| book | chunk records | parser failures | state |
| --- | --- | --- | --- |
| B31 | 244 | 244 | finished |
| B43 | 246 | 246 | finished |
| B37 | 262 | 260 | finished |
| B18 | 312 | 300 | finished |
| B04 | 321 | 320 | finished |
| B34 | 335 | 335 | finished |
| B27 | 338 | 337 | finished |
| B61 | 357 | 341 | finished |
| B26 | 384 | 382 | finished |
| B44 | 385 | 369 | finished |

### Location of the construction records

`/mnt/data/zeinab/SAADI/outputs/probe_bridge/novelhopqa_probe/indexes/7f585bd63886659e/llm_cache/novelhopqa/hierarchical/branching_factor_5/chunk_500/AE_5_2_3/Qwen3-30B-A3B-Instruct-2507/AEWord/index.json`

### Hand-off

All books are indexed. Paste this report back before step 6.

## Anything unexpected

Nothing unexpected.

## Files written

- `/mnt/data/zeinab/SAADI/configs/experiments/novelhopqa_probe_retrieval_ablation.yaml  (declared file outside the probe folders)`
