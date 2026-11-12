# C2 group A report

- Started 2026-11-12 15:27:25, finished 2026-11-12 15:41:29; host eee-s132-infsys-gpu03; exit code 0.
- Command: `python3 scripts/rebuttal/c2_heldout_fusion.py --group A`

## Preflight

- OK: True. Python 3.12.3 (/mnt/data/zeinab/raptor/.venv/bin/python3).
- Modules: numpy 2.3.5, pandas 3.0.1, pyarrow 23.0.1, tqdm 4.68.2, torch 2.11.0+cu129, transformers 4.57.6, accelerate 1.13.0, matplotlib 3.10.8.
- Commit cffbe0da1325c961a5aaf4fffa74785261ac8a03 on fix/notebook-key-and-install; uncommitted code changes: none.
- GPUs: 0 NVIDIA RTX A6000 4/49140 MiB used; 1 NVIDIA RTX A6000 4/49140 MiB used; 2 NVIDIA RTX A6000 4/49140 MiB used; 3 NVIDIA RTX A6000 4/49140 MiB used.
- Disk: 4423.8 GB free at /mnt/data/zeinab/SAADI/outputs/rebuttal/C2_heldout_fusion (estimate needed 6.7 GB).
- Models: BAAI/bge-m3: in cache, facebook/contriever: in cache, jinaai/jina-embeddings-v3: in cache, Qwen/Qwen3-Embedding-8B: in cache.

## Plan

- plan.json read (written by group A) at 2026-11-09 12:34:30.
- Cost basis: estimated from model sizes (no timing run): seconds = model load + (estimated document tokens x 2 x parameters) / effective FLOP/s + 5 query encodings per query x per-call latency + CPU combination time. weights in float32 (the adapters pass no dtype); Qwen3-Embedding-8B ran out of memory on one 48 GB A6000 (about 28 GB of weights plus batch-64 activations), so each Qwen job gets two GPUs of its group with device_map=auto; every other job gets one GPU.
- This group: GPUs [0, 1, 2, 3]; estimated wall time 2.3 h (other group: 2.3 h); analysis afterwards about 12 min.

| Job | Device | Texts to embed | Queries | Estimate |
| --- | --- | --- | --- | --- |
| qasper_64k__qwen | gpu x2 | 168248 | 1333 | 2.3 h |
| loogle__qwen | gpu x2 | 55069 | 1832 | 77 min |
| qasper_64k__jina | gpu x1 | 168248 | 1333 | 15 min |
| qasper_64k__bge_m3 | gpu x1 | 168248 | 1333 | 14 min |
| novelhopqa__bge_m3 | gpu x1 | 80518 | 908 | 14 min |
| musique_32k__jina | gpu x1 | 221446 | 900 | 13 min |
| musique_32k__bge_m3 | gpu x1 | 221446 | 900 | 12 min |
| loogle__jina | gpu x1 | 55069 | 1832 | 9 min |
| musique_32k__contriever | gpu x1 | 221446 | 900 | 4 min |
| qasper_64k__bm25 | cpu x1 | 168248 | 1333 | 2 min |
| novelhopqa__bm25 | cpu x1 | 80518 | 908 | 63s |

## Jobs and reproduction checks

| Job | Status | Time | Identical top-10 | HR@5 / HR@10 ours vs saved | text-only vs Table 1 Vanilla | Problems |
| --- | --- | --- | --- | --- | --- | --- |
| qasper_64k__qwen | DONE (skipped, already finished) | 1.6 h | 1333/1333 | 23.33 / 34.13 vs 23.33 / 34.13 | 22.58 / 32.56 vs 22.6 / 32.6 | none |
| qasper_64k__jina | DONE (skipped, already finished) | 5s | 927/1333 | 18.83 / 27.08 vs 18.90 / 27.16 | 15.98 / 24.98 vs 15.7 / 24.8 (differs) | drift accepted: identical top-10 for 927/1333 queries (< 99%) |
| qasper_64k__bge_m3 | DONE (skipped, already finished) | 16 min | 1333/1333 | 20.93 / 30.91 vs 20.93 / 30.91 | 21.01 / 29.33 vs 21.0 / 29.3 | none |
| novelhopqa__bge_m3 | DONE (skipped, already finished) | 11 min | 908/908 | 16.52 / 22.14 vs 16.52 / 22.14 | 17.29 / 21.92 vs 14.1 / 20.7 (differs) | none |
| musique_32k__jina | DONE (skipped, already finished) | 2s | 705/900 | 73.56 / 88.11 vs 73.22 / 88.11 | 71.67 / 85.78 vs 71.6 / 85.8 | drift accepted: identical top-10 for 705/900 queries (< 99%); HR@5 73.56 vs saved 73.22 (tolerance 0.1 points); HR@5 73.56 vs expected 73.22 |
| musique_32k__bge_m3 | DONE (skipped, already finished) | 9 min | 900/900 | 76.89 / 88.33 vs 76.89 / 88.33 | 73.22 / 87.44 vs 73.2 / 87.4 | none |
| loogle__jina | DONE (skipped, already finished) | 1s | 1519/1832 | 70.20 / 81.99 vs 70.14 / 81.99 | 69.92 / 80.68 vs 70.1 / 80.1 (differs) | drift accepted: identical top-10 for 1519/1832 queries (< 99%) |
| musique_32k__contriever | DONE (skipped, already finished) | 6 min | 900/900 | 68.78 / 83.67 vs 68.78 / 83.67 | 59.89 / 76.11 vs 59.9 / 76.1 | none |
| qasper_64k__bm25 | DONE (skipped, already finished) | 3 min | 1333/1333 | 16.95 / 26.11 vs 16.95 / 26.11 | 14.40 / 22.06 vs 14.4 / 22.1 | none |
| novelhopqa__bm25 | DONE (skipped, already finished) | 2 min | 908/908 | 18.06 / 22.03 vs 18.06 / 22.03 | 17.51 / 20.70 vs 16.9 / 20.0 (differs) | none |
| loogle__qwen | DONE | 10s | 733/1832 | 77.13 / 86.41 vs 77.02 / 86.68 | 77.89 / 86.19 vs 76.9 / 85.9 (differs) | drift accepted: identical top-10 for 733/1832 queries (< 99%); HR@5 77.13 vs saved 77.02 (tolerance 0.1 points); HR@10 86.41 vs saved 86.68 (tolerance 0.1 points); HR@5 77.13 vs expected 77.02; HR@10 86.41 vs expected 86.68 |

- qasper_64k__jina text-only note: text-only HR@5 15.98 vs Table 1 Vanilla 15.7 (tolerance 0.15); text-only HR@10 24.98 vs Table 1 Vanilla 24.8 (tolerance 0.15)
- novelhopqa__bge_m3 text-only note: text-only HR@5 17.29 vs Table 1 Vanilla 14.1 (tolerance 0.15); text-only HR@10 21.92 vs Table 1 Vanilla 20.7 (tolerance 0.15)
- loogle__jina text-only note: text-only HR@5 69.92 vs Table 1 Vanilla 70.1 (tolerance 0.15); text-only HR@10 80.68 vs Table 1 Vanilla 80.1 (tolerance 0.15)
- novelhopqa__bm25 text-only note: text-only HR@5 17.51 vs Table 1 Vanilla 16.9 (tolerance 0.05); text-only HR@10 20.70 vs Table 1 Vanilla 20.0 (tolerance 0.05)
- loogle__qwen text-only note: text-only HR@5 77.89 vs Table 1 Vanilla 76.9 (tolerance 0.15); text-only HR@10 86.19 vs Table 1 Vanilla 85.9 (tolerance 0.15)

## Analysis

- done in 13 min: results/rebuttal/C2_heldout_fusion/REPORT.md

## Files written

- `outputs/rebuttal/C2_heldout_fusion/streams/qasper_64k__qwen.parquet`
- `outputs/rebuttal/C2_heldout_fusion/streams/qasper_64k__qwen.queries.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/qasper_64k__qwen.check.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/qasper_64k__jina.parquet`
- `outputs/rebuttal/C2_heldout_fusion/streams/qasper_64k__jina.queries.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/qasper_64k__jina.check.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/qasper_64k__bge_m3.parquet`
- `outputs/rebuttal/C2_heldout_fusion/streams/qasper_64k__bge_m3.queries.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/qasper_64k__bge_m3.check.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/novelhopqa__bge_m3.parquet`
- `outputs/rebuttal/C2_heldout_fusion/streams/novelhopqa__bge_m3.queries.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/novelhopqa__bge_m3.check.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/musique_32k__jina.parquet`
- `outputs/rebuttal/C2_heldout_fusion/streams/musique_32k__jina.queries.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/musique_32k__jina.check.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/musique_32k__bge_m3.parquet`
- `outputs/rebuttal/C2_heldout_fusion/streams/musique_32k__bge_m3.queries.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/musique_32k__bge_m3.check.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/loogle__jina.parquet`
- `outputs/rebuttal/C2_heldout_fusion/streams/loogle__jina.queries.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/loogle__jina.check.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/musique_32k__contriever.parquet`
- `outputs/rebuttal/C2_heldout_fusion/streams/musique_32k__contriever.queries.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/musique_32k__contriever.check.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/qasper_64k__bm25.parquet`
- `outputs/rebuttal/C2_heldout_fusion/streams/qasper_64k__bm25.queries.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/qasper_64k__bm25.check.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/novelhopqa__bm25.parquet`
- `outputs/rebuttal/C2_heldout_fusion/streams/novelhopqa__bm25.queries.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/novelhopqa__bm25.check.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/loogle__qwen.parquet`
- `outputs/rebuttal/C2_heldout_fusion/streams/loogle__qwen.queries.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/loogle__qwen.check.json`
- `results/rebuttal/C2_heldout_fusion/REPORT.md`
- `results/rebuttal/C2_heldout_fusion/REPORT_FOR_GDOC.md`
- `results/rebuttal/C2_heldout_fusion/analysis.json`
- `results/rebuttal/C2_heldout_fusion/dry_run/REPORT.md`
- `results/rebuttal/C2_heldout_fusion/dry_run/REPORT_FOR_GDOC.md`
- `results/rebuttal/C2_heldout_fusion/dry_run/analysis.json`
- `results/rebuttal/C2_heldout_fusion/dry_run/figures/heatmap_hr10_bm25.png`
- `results/rebuttal/C2_heldout_fusion/dry_run/figures/heatmap_hr10_mean_over_retrievers.png`
- `results/rebuttal/C2_heldout_fusion/dry_run/tables/a_chosen_per_fold.csv`
- `results/rebuttal/C2_heldout_fusion/dry_run/tables/b_lodo.csv`
- `results/rebuttal/C2_heldout_fusion/dry_run/tables/c_table9_reconstruction.csv`
- `results/rebuttal/C2_heldout_fusion/dry_run/tables/d_per_pair.csv`
- `results/rebuttal/C2_heldout_fusion/dry_run/tables/e_ranking_cells.csv`
- `results/rebuttal/C2_heldout_fusion/dry_run/tables/f_grid_hr.csv`
- `results/rebuttal/C2_heldout_fusion/dry_run/tables/f_sensitivity.csv`
- `results/rebuttal/C2_heldout_fusion/dry_run/tables/g_learned_weights.csv`
- `results/rebuttal/C2_heldout_fusion/dry_run/tables/verification.csv`
- `results/rebuttal/C2_heldout_fusion/figures/heatmap_hr10_bm25.png`
- `results/rebuttal/C2_heldout_fusion/figures/heatmap_hr10_mean_over_retrievers.png`
- `results/rebuttal/C2_heldout_fusion/tables/a_chosen_per_fold.csv`
- `results/rebuttal/C2_heldout_fusion/tables/b_lodo.csv`
- `results/rebuttal/C2_heldout_fusion/tables/c_table9_reconstruction.csv`
- `results/rebuttal/C2_heldout_fusion/tables/d_per_pair.csv`
- `results/rebuttal/C2_heldout_fusion/tables/e_ranking_cells.csv`
- `results/rebuttal/C2_heldout_fusion/tables/f_grid_hr.csv`
- `results/rebuttal/C2_heldout_fusion/tables/f_sensitivity.csv`
- `results/rebuttal/C2_heldout_fusion/tables/g_learned_weights.csv`
- `results/rebuttal/C2_heldout_fusion/tables/verification.csv`
- `experiments/rebuttal/C2_heldout_fusion/reports/group_A.md`
- `experiments/rebuttal/C2_heldout_fusion/reports/group_A.json`
