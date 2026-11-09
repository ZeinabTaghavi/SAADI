# C2 group B report (dry run)

- Started 2026-11-09 11:40:40, finished 2026-11-09 11:41:01; host LIGHT-C026225M.local; exit code 0.
- Command: `python3 scripts/rebuttal/c2_heldout_fusion.py --group B --dry-run`

## Preflight

- OK: True. Python 3.14.8 (/Users/uem-stag-lap-ic/Desktop/SAADI/venv.nosync/bin/python3).
- Modules: numpy 2.5.4, pandas 3.1.1, pyarrow 26.0.1, tqdm 4.70.1, matplotlib 3.11.2.
- Commit f1a54482c68d91eb797d75b11362f1973bc202a5 on fix/notebook-key-and-install; uncommitted code changes: ['?? experiments/rebuttal/C2_heldout_fusion/run_gpus_0123.sh', '?? experiments/rebuttal/C2_heldout_fusion/run_gpus_4567.sh', '?? scripts/rebuttal/c2_heldout_fusion.py', '?? src/saadi/rebuttal/heldout_fusion.py'].
- GPUs: skipped (dry run on CPU).
- Disk: 240.6 GB free at /Users/uem-stag-lap-ic/Desktop/SAADI/outputs/rebuttal/C2_heldout_fusion/dry_run (estimate needed 5.0 GB).

## Plan

- plan.json read (written by group A) at 2026-11-09 11:40:28.
- Cost basis: estimated from model sizes (no timing run): seconds = model load + (estimated document tokens x 2 x parameters) / effective FLOP/s + 5 query encodings per query x per-call latency + CPU combination time. weights in float32 (the adapters pass no dtype); every model fits on one 48 GB A6000 (largest: Qwen3-Embedding-8B, about 28 GB of weights), so every job gets one GPU.
- This group: GPUs [4, 5, 6, 7]; estimated wall time 1s (other group: 14s); analysis afterwards about 17s.

| Job | Device | Texts to embed | Queries | Estimate |
| --- | --- | --- | --- | --- |
| musique_32k__bm25 | cpu x1 | 10200 | 42 | 2s |
| loogle__bm25 | cpu x1 | 971 | 21 | 0s |

## Jobs and reproduction checks

| Job | Status | Time | Identical top-10 | HR@5 / HR@10 ours vs saved | text-only vs Table 1 Vanilla | Problems |
| --- | --- | --- | --- | --- | --- | --- |
| musique_32k__bm25 | DONE | 2s | 42/42 | 69.05 / 85.71 vs 69.05 / 85.71 | 64.29 / 80.95 vs 57.6 / 74.0 | none |
| loogle__bm25 | DONE | 2s | 21/21 | 90.48 / 95.24 vs 90.48 / 95.24 | 85.71 / 90.48 vs 91.5 / 95.4 | none |


## Analysis

- done in 16s: results/rebuttal/C2_heldout_fusion/dry_run/REPORT.md

## Files written

- `outputs/rebuttal/C2_heldout_fusion/dry_run/streams/musique_32k__bm25.parquet`
- `outputs/rebuttal/C2_heldout_fusion/dry_run/streams/musique_32k__bm25.queries.json`
- `outputs/rebuttal/C2_heldout_fusion/dry_run/streams/musique_32k__bm25.check.json`
- `outputs/rebuttal/C2_heldout_fusion/dry_run/streams/loogle__bm25.parquet`
- `outputs/rebuttal/C2_heldout_fusion/dry_run/streams/loogle__bm25.queries.json`
- `outputs/rebuttal/C2_heldout_fusion/dry_run/streams/loogle__bm25.check.json`
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
- `experiments/rebuttal/C2_heldout_fusion/reports/dry_run/group_B.md`
- `experiments/rebuttal/C2_heldout_fusion/reports/dry_run/group_B.json`
