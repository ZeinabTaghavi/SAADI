# C3 channel ablation - step 03: new variants for the dense retrievers

**Status: FAILED**

- Command: `C3_RETRIEVERS=contriever,bge_m3 bash experiments/rebuttal/c3_channel_ablation/sh/03_dense.sh`
- Started: 2026-11-09 17:19:14; took 0 s
- Host: eee-s132-infsys-gpu03; Python 3.12.3; commit 1e7b1fb6b31c (fix/notebook-key-and-install, 3 changed files)

## Settings

- Weights from `configs/methods/sadi.yaml`: (lambda_T, lambda_M, lambda_S) = (1, 0.5, 0.3), field weights (1, 1, 1), zscore. A new variant sets the dropped channel's weight to 0 in the published Equation 2; nothing else changes.
- Where each variant comes from: Text only: saved text-only top-10 in the streams (C2); Text + summary: scored by C3 (published Equation 2, one weight 0); Text + memory: scored by C3 (published Equation 2, one weight 0); All three (SAADI): paper's saved rankings; Summary alone: scored by C3 (published Equation 2, one weight 0); Memory alone: scored by C3 (published Equation 2, one weight 0).
- Rank = 1-based position within the top-10 at which the saved hit definition is met (gold chunk, or every chunk of one evidence group); median over eligible queries, '>10' when fewer than half are hit within the top-10.
- Paired differences in HR points: 95% bootstrap over queries (10000 resamples, seed 42); McNemar exact two-sided p.
- Pairs from `experiments/rebuttal/c3_channel_ablation/work/ready_pairs.json` (step 01, check 1 passed): none.

Not scored (did not pass step 01): loogle__contriever (check 1 failed or exception); loogle__bge_m3 (check 1 failed or exception); musique_32k__contriever (check 1 failed or exception); musique_32k__bge_m3 (check 1 failed or exception); qasper_64k__contriever (check 1 failed or exception); qasper_64k__bge_m3 (check 1 failed or exception); novelhopqa__contriever (check 1 failed or exception); novelhopqa__bge_m3 (check 1 failed or exception)

No ready pair for this step. Run step 01 for these retrievers first.

## Files written

- `experiments/rebuttal/c3_channel_ablation/reports/step_03.md`
- `experiments/rebuttal/c3_channel_ablation/reports/step_03.json`
