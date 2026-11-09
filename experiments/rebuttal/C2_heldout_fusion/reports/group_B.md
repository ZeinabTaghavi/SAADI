# C2 group B report

- Started 2026-11-09 14:01:34, finished 2026-11-09 15:52:17; host eee-s132-infsys-gpu03; exit code 0.
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

## Jobs and reproduction checks

| Job | Status | Time | Identical top-10 | HR@5 / HR@10 ours vs saved | text-only vs Table 1 Vanilla | Problems |
| --- | --- | --- | --- | --- | --- | --- |
| loogle__bge_m3 | DONE (skipped, already finished) | 9 min | 1832/1832 | 81.06 / 90.12 vs 81.06 / 90.12 | 82.48 / 90.72 vs 80.6 / 89.8 (differs) | none |
| qasper_64k__contriever | DONE (skipped, already finished) | 8 min | 1333/1333 | 16.20 / 25.81 vs 16.20 / 25.81 | 15.15 / 23.33 vs 15.2 / 23.3 | none |
| novelhopqa__contriever | DONE (skipped, already finished) | 5 min | 908/908 | 13.55 / 18.17 vs 13.55 / 18.17 | 12.44 / 17.84 vs 11.8 / 11.8 (differs) | none |
| loogle__contriever | DONE (skipped, already finished) | 4 min | 1832/1832 | 75.44 / 85.70 vs 75.44 / 85.70 | 75.16 / 86.03 vs 72.4 / 73.3 (differs) | none |
| musique_32k__bm25 | DONE (skipped, already finished) | 2 min | 900/900 | 62.78 / 77.67 vs 62.78 / 77.67 | 57.56 / 74.00 vs 57.6 / 74.0 | none |
| loogle__bm25 | DONE (skipped, already finished) | 16s | 1832/1832 | 92.30 / 96.45 vs 92.30 / 96.45 | 91.87 / 95.74 vs 91.5 / 95.4 (differs) | none |
| novelhopqa__qwen | DONE | 79 min | 908/908 | 15.20 / 20.59 vs 15.20 / 20.59 | 16.74 / 20.48 vs 14.6 / 19.8 (differs) | none |
| novelhopqa__jina | DONE | 10s | 624/908 | 14.76 / 20.93 vs 14.76 / 21.15 | 15.97 / 20.15 vs 14.8 / 19.1 (differs) | drift accepted: identical top-10 for 624/908 queries (< 99%); HR@10 20.93 vs saved 21.15 (tolerance 0.1 points); HR@10 20.93 vs expected 21.15 |
| musique_32k__qwen | DONE | 1.8 h | 899/900 | 76.44 / 90.00 vs 76.44 / 90.00 | 75.33 / 89.00 vs 75.3 / 89.0 | none |

- loogle__bge_m3 text-only note: text-only HR@5 82.48 vs Table 1 Vanilla 80.6 (tolerance 0.15); text-only HR@10 90.72 vs Table 1 Vanilla 89.8 (tolerance 0.15)
- novelhopqa__contriever text-only note: text-only HR@5 12.44 vs Table 1 Vanilla 11.8 (tolerance 0.15); text-only HR@10 17.84 vs Table 1 Vanilla 11.8 (tolerance 0.15)
- loogle__contriever text-only note: text-only HR@5 75.16 vs Table 1 Vanilla 72.4 (tolerance 0.15); text-only HR@10 86.03 vs Table 1 Vanilla 73.3 (tolerance 0.15)
- loogle__bm25 text-only note: text-only HR@5 91.87 vs Table 1 Vanilla 91.5 (tolerance 0.05); text-only HR@10 95.74 vs Table 1 Vanilla 95.4 (tolerance 0.05)
- novelhopqa__qwen text-only note: text-only HR@5 16.74 vs Table 1 Vanilla 14.6 (tolerance 0.15); text-only HR@10 20.48 vs Table 1 Vanilla 19.8 (tolerance 0.15)
- novelhopqa__jina text-only note: text-only HR@5 15.97 vs Table 1 Vanilla 14.8 (tolerance 0.15); text-only HR@10 20.15 vs Table 1 Vanilla 19.1 (tolerance 0.15)

## Analysis

- 19/20 jobs DONE; waiting for group A

## Files written

- `outputs/rebuttal/C2_heldout_fusion/streams/loogle__bge_m3.parquet`
- `outputs/rebuttal/C2_heldout_fusion/streams/loogle__bge_m3.queries.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/loogle__bge_m3.check.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/qasper_64k__contriever.parquet`
- `outputs/rebuttal/C2_heldout_fusion/streams/qasper_64k__contriever.queries.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/qasper_64k__contriever.check.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/novelhopqa__contriever.parquet`
- `outputs/rebuttal/C2_heldout_fusion/streams/novelhopqa__contriever.queries.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/novelhopqa__contriever.check.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/loogle__contriever.parquet`
- `outputs/rebuttal/C2_heldout_fusion/streams/loogle__contriever.queries.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/loogle__contriever.check.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/musique_32k__bm25.parquet`
- `outputs/rebuttal/C2_heldout_fusion/streams/musique_32k__bm25.queries.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/musique_32k__bm25.check.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/loogle__bm25.parquet`
- `outputs/rebuttal/C2_heldout_fusion/streams/loogle__bm25.queries.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/loogle__bm25.check.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/novelhopqa__qwen.parquet`
- `outputs/rebuttal/C2_heldout_fusion/streams/novelhopqa__qwen.queries.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/novelhopqa__qwen.check.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/novelhopqa__jina.parquet`
- `outputs/rebuttal/C2_heldout_fusion/streams/novelhopqa__jina.queries.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/novelhopqa__jina.check.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/musique_32k__qwen.parquet`
- `outputs/rebuttal/C2_heldout_fusion/streams/musique_32k__qwen.queries.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/musique_32k__qwen.check.json`
- `experiments/rebuttal/C2_heldout_fusion/reports/group_B.md`
- `experiments/rebuttal/C2_heldout_fusion/reports/group_B.json`
