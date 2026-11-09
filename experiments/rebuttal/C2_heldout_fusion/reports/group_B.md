# C2 group B report

- Started 2026-11-09 14:01:34, finished —; host eee-s132-infsys-gpu03; exit code None.
- Command: `python3 scripts/rebuttal/c2_heldout_fusion.py --group B`

## Preflight

- OK: True. Python 3.12.3 (/mnt/data/zeinab/raptor/.venv/bin/python3).
- Modules: numpy 2.3.5, pandas 3.0.1, pyarrow 23.0.1, tqdm 4.68.2, torch 2.11.0+cu129, transformers 4.57.6, accelerate 1.13.0, matplotlib 3.10.8.
- Commit e2dbaa178fa07c25806c32e8fce787455bae9ce3 on fix/notebook-key-and-install; uncommitted code changes: none.
- GPUs: 4 NVIDIA RTX A6000 4/49140 MiB used; 5 NVIDIA RTX A6000 4/49140 MiB used; 6 NVIDIA RTX A6000 4/49140 MiB used; 7 NVIDIA RTX A6000 4/49140 MiB used.
- Disk: 4285.6 GB free at /mnt/data/zeinab/SAADI/outputs/rebuttal/C2_heldout_fusion (estimate needed 5.9 GB).
- Models: BAAI/bge-m3: in cache, facebook/contriever: in cache, jinaai/jina-embeddings-v3: in cache, Qwen/Qwen3-Embedding-8B: in cache.

## Plan

- plan.json read (written by group A) at 2026-11-09 12:34:30.
- Cost basis: estimated from model sizes (no timing run): seconds = model load + (estimated document tokens x 2 x parameters) / effective FLOP/s + 5 query encodings per query x per-call latency + CPU combination time. weights in float32 (the adapters pass no dtype); Qwen3-Embedding-8B ran out of memory on one 48 GB A6000 (about 28 GB of weights plus batch-64 activations), so each Qwen job gets two GPUs of its group with device_map=auto; every other job gets one GPU.
- This group: GPUs [4, 5, 6, 7]; estimated wall time 2.3 h (other group: 2.3 h); analysis afterwards about 12 min.

| Job | Device | Texts to embed | Queries | Estimate |
| --- | --- | --- | --- | --- |
| musique_32k__qwen | gpu x2 | 221446 | 900 | 2.2 h |
| novelhopqa__qwen | gpu x2 | 80518 | 908 | 2.1 h |
| novelhopqa__jina | gpu x1 | 80518 | 908 | 12 min |
| loogle__bge_m3 | gpu x1 | 55069 | 1832 | 9 min |
| qasper_64k__contriever | gpu x1 | 168248 | 1333 | 5 min |
| novelhopqa__contriever | gpu x1 | 80518 | 908 | 4 min |
| loogle__contriever | gpu x1 | 55069 | 1832 | 3 min |
| musique_32k__bm25 | cpu x1 | 221446 | 900 | 51s |
| loogle__bm25 | cpu x1 | 55069 | 1832 | 9s |

## Analysis

- not reached

## Files written

