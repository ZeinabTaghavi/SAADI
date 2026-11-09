# C2 group A report (dry run)

- Started 2026-11-09 11:40:24, finished 2026-11-09 11:40:40; host LIGHT-C026225M.local; exit code 0.
- Command: `python3 scripts/rebuttal/c2_heldout_fusion.py --group A --dry-run`

## Preflight

- OK: True. Python 3.14.8 (/Users/uem-stag-lap-ic/Desktop/SAADI/venv.nosync/bin/python3).
- Modules: numpy 2.5.4, pandas 3.1.1, pyarrow 26.0.1, tqdm 4.70.1, matplotlib 3.11.2.
- Commit f1a54482c68d91eb797d75b11362f1973bc202a5 on fix/notebook-key-and-install; uncommitted code changes: ['?? experiments/rebuttal/C2_heldout_fusion/run_gpus_0123.sh', '?? experiments/rebuttal/C2_heldout_fusion/run_gpus_4567.sh', '?? scripts/rebuttal/c2_heldout_fusion.py', '?? src/saadi/rebuttal/heldout_fusion.py'].
- GPUs: skipped (dry run on CPU).
- Disk: 240.6 GB free at /Users/uem-stag-lap-ic/Desktop/SAADI/outputs/rebuttal/C2_heldout_fusion/dry_run (estimate needed 5.1 GB).

## Plan

- plan.json written by this group at 2026-11-09 11:40:28.
- Cost basis: estimated from model sizes (no timing run): seconds = model load + (estimated document tokens x 2 x parameters) / effective FLOP/s + 5 query encodings per query x per-call latency + CPU combination time. weights in float32 (the adapters pass no dtype); every model fits on one 48 GB A6000 (largest: Qwen3-Embedding-8B, about 28 GB of weights), so every job gets one GPU.
- This group: GPUs [0, 1, 2, 3]; estimated wall time 14s (other group: 1s); analysis afterwards about 17s.

| Job | Device | Texts to embed | Queries | Estimate |
| --- | --- | --- | --- | --- |
| qasper_64k__bm25 | cpu x1 | 13796 | 129 | 14s |
| novelhopqa__bm25 | cpu x1 | 11367 | 145 | 13s |

## Jobs and reproduction checks

| Job | Status | Time | Identical top-10 | HR@5 / HR@10 ours vs saved | text-only vs Table 1 Vanilla | Problems |
| --- | --- | --- | --- | --- | --- | --- |
| qasper_64k__bm25 | DONE | 6s | 129/129 | 20.93 / 25.58 vs 20.93 / 25.58 | 14.73 / 20.16 vs 14.4 / 22.1 | none |
| novelhopqa__bm25 | DONE | 10s | 145/145 | 22.76 / 25.52 vs 22.76 / 25.52 | 21.38 / 25.52 vs 16.9 / 20.0 | none |


## Analysis

- 2/4 jobs DONE; waiting for group B

## Files written

- `outputs/rebuttal/C2_heldout_fusion/dry_run/plan.json`
- `outputs/rebuttal/C2_heldout_fusion/dry_run/streams/qasper_64k__bm25.parquet`
- `outputs/rebuttal/C2_heldout_fusion/dry_run/streams/qasper_64k__bm25.queries.json`
- `outputs/rebuttal/C2_heldout_fusion/dry_run/streams/qasper_64k__bm25.check.json`
- `outputs/rebuttal/C2_heldout_fusion/dry_run/streams/novelhopqa__bm25.parquet`
- `outputs/rebuttal/C2_heldout_fusion/dry_run/streams/novelhopqa__bm25.queries.json`
- `outputs/rebuttal/C2_heldout_fusion/dry_run/streams/novelhopqa__bm25.check.json`
- `experiments/rebuttal/C2_heldout_fusion/reports/dry_run/group_A.md`
- `experiments/rebuttal/C2_heldout_fusion/reports/dry_run/group_A.json`
