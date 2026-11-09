# C3 channel ablation - step 03: new variants for the dense retrievers

**Status: PASSED**

- Command: `C3_RETRIEVERS=contriever,bge_m3 bash experiments/rebuttal/c3_channel_ablation/sh/03_dense.sh`
- Started: 2026-11-09 17:37:02; took 226 s
- Host: eee-s132-infsys-gpu03; Python 3.12.3; commit 98f4f789a04e (fix/notebook-key-and-install, 6 changed files)

## Settings

- Weights from `configs/methods/sadi.yaml`: (lambda_T, lambda_M, lambda_S) = (1, 0.5, 0.3), field weights (1, 1, 1), zscore. A new variant sets the dropped channel's weight to 0 in the published Equation 2; nothing else changes.
- Where each variant comes from: Text only: saved text-only top-10 in the streams (C2); Text + summary: scored by C3 (published Equation 2, one weight 0); Text + memory: scored by C3 (published Equation 2, one weight 0); All three (SAADI): paper's saved rankings; Summary alone: scored by C3 (published Equation 2, one weight 0); Memory alone: scored by C3 (published Equation 2, one weight 0).
- Rank = 1-based position within the top-10 at which the saved hit definition is met (gold chunk, or every chunk of one evidence group); median over eligible queries, '>10' when fewer than half are hit within the top-10.
- Paired differences in HR points: 95% bootstrap over queries (10000 resamples, seed 42); McNemar exact two-sided p.
- Pairs from `experiments/rebuttal/c3_channel_ablation/work/ready_pairs.json` (step 01, check 1 passed): loogle__contriever, musique_32k__contriever, qasper_64k__contriever, novelhopqa__contriever, loogle__bge_m3, musique_32k__bge_m3, qasper_64k__bge_m3, novelhopqa__bge_m3.

## Vanilla and SAADI against the paper

| Pair | Text only HR@5 / HR@10 (here) | Table 1 Vanilla | All three HR@5 / HR@10 (here) | Paper SAADI |
| --- | --- | --- | --- | --- |
| LooGLE / Contriever | 75.16 / 86.03 | 72.4 / 73.3 | 75.44 / 85.70 | 75.44 / 85.70 |
| MuSiQue-32K / Contriever | 59.89 / 76.11 | 59.9 / 76.1 | 68.78 / 83.67 | 68.78 / 83.67 |
| QASPER-64K / Contriever | 15.15 / 23.33 | 15.2 / 23.3 | 16.20 / 25.81 | 16.20 / 25.81 |
| NovelHopQA / Contriever | 12.44 / 17.84 | 11.8 / 11.8 | 13.55 / 18.17 | 13.55 / 18.17 |
| LooGLE / BGE-M3 | 82.48 / 90.72 | 80.6 / 89.8 | 81.06 / 90.12 | 81.06 / 90.12 |
| MuSiQue-32K / BGE-M3 | 73.22 / 87.44 | 73.2 / 87.4 | 76.89 / 88.33 | 76.89 / 88.33 |
| QASPER-64K / BGE-M3 | 21.01 / 29.33 | 21.0 / 29.3 | 20.93 / 30.91 | 20.93 / 30.91 |
| NovelHopQA / BGE-M3 | 17.29 / 21.92 | 14.1 / 20.7 | 16.52 / 22.14 | 16.52 / 22.14 |

Text only differs from Table 1's Vanilla on LooGLE and NovelHopQA (a separate run in the paper; step 01 explains it). All three is the paper's saved run itself, so it equals the paper's SAADI numbers.

## Guards

| Pair | Rank HR = repo HR (all variants, k=5,10) | Eligible queries | Time |
| --- | --- | --- | --- |
| LooGLE / Contriever | yes | 1832 | 5 s |
| MuSiQue-32K / Contriever | yes | 900 | 21 s |
| QASPER-64K / Contriever | yes | 1333 | 62 s |
| NovelHopQA / Contriever | yes | 908 | 25 s |
| LooGLE / BGE-M3 | yes | 1832 | 5 s |
| MuSiQue-32K / BGE-M3 | yes | 900 | 21 s |
| QASPER-64K / BGE-M3 | yes | 1333 | 61 s |
| NovelHopQA / BGE-M3 | yes | 908 | 25 s |

## HR@5 / HR@10 (%) and median rank, all variants

| Pair | Variant | HR@5 | HR@10 | Median rank |
| --- | --- | --- | --- | --- |
| LooGLE / Contriever | Text only | 75.16 | 86.03 | 2 |
| LooGLE / Contriever | Text + summary | 74.02 | 85.59 | 2 |
| LooGLE / Contriever | Text + memory | 75.98 | 85.70 | 2 |
| LooGLE / Contriever | All three (SAADI) | 75.44 | 85.70 | 2 |
| LooGLE / Contriever | Summary alone | 34.12 | 55.51 | 8 |
| LooGLE / Contriever | Memory alone | 51.04 | 65.12 | 5 |
| MuSiQue-32K / Contriever | Text only | 59.89 | 76.11 | 4 |
| MuSiQue-32K / Contriever | Text + summary | 61.67 | 79.11 | 4 |
| MuSiQue-32K / Contriever | Text + memory | 66.78 | 82.11 | 3 |
| MuSiQue-32K / Contriever | All three (SAADI) | 68.78 | 83.67 | 3 |
| MuSiQue-32K / Contriever | Summary alone | 52.67 | 73.78 | 5 |
| MuSiQue-32K / Contriever | Memory alone | 64.89 | 77.67 | 3 |
| QASPER-64K / Contriever | Text only | 15.15 | 23.33 | >10 |
| QASPER-64K / Contriever | Text + summary | 15.45 | 24.46 | >10 |
| QASPER-64K / Contriever | Text + memory | 16.05 | 25.13 | >10 |
| QASPER-64K / Contriever | All three (SAADI) | 16.20 | 25.81 | >10 |
| QASPER-64K / Contriever | Summary alone | 10.13 | 18.00 | >10 |
| QASPER-64K / Contriever | Memory alone | 11.70 | 18.00 | >10 |
| NovelHopQA / Contriever | Text only | 12.44 | 17.84 | >10 |
| NovelHopQA / Contriever | Text + summary | 13.33 | 18.83 | >10 |
| NovelHopQA / Contriever | Text + memory | 12.67 | 18.61 | >10 |
| NovelHopQA / Contriever | All three (SAADI) | 13.55 | 18.17 | >10 |
| NovelHopQA / Contriever | Summary alone | 4.74 | 8.59 | >10 |
| NovelHopQA / Contriever | Memory alone | 6.72 | 9.69 | >10 |
| LooGLE / BGE-M3 | Text only | 82.48 | 90.72 | 1 |
| LooGLE / BGE-M3 | Text + summary | 81.55 | 89.57 | 1 |
| LooGLE / BGE-M3 | Text + memory | 81.17 | 90.39 | 1 |
| LooGLE / BGE-M3 | All three (SAADI) | 81.06 | 90.12 | 2 |
| LooGLE / BGE-M3 | Summary alone | 40.67 | 59.01 | 6 |
| LooGLE / BGE-M3 | Memory alone | 53.60 | 67.69 | 5 |
| MuSiQue-32K / BGE-M3 | Text only | 73.22 | 87.44 | 2 |
| MuSiQue-32K / BGE-M3 | Text + summary | 75.22 | 89.33 | 2 |
| MuSiQue-32K / BGE-M3 | Text + memory | 76.11 | 86.89 | 2 |
| MuSiQue-32K / BGE-M3 | All three (SAADI) | 76.89 | 88.33 | 2 |
| MuSiQue-32K / BGE-M3 | Summary alone | 59.67 | 80.44 | 3.5 |
| MuSiQue-32K / BGE-M3 | Memory alone | 65.56 | 78.11 | 3 |
| QASPER-64K / BGE-M3 | Text only | 21.01 | 29.33 | >10 |
| QASPER-64K / BGE-M3 | Text + summary | 21.01 | 31.36 | >10 |
| QASPER-64K / BGE-M3 | Text + memory | 20.26 | 28.81 | >10 |
| QASPER-64K / BGE-M3 | All three (SAADI) | 20.93 | 30.91 | >10 |
| QASPER-64K / BGE-M3 | Summary alone | 15.00 | 24.38 | >10 |
| QASPER-64K / BGE-M3 | Memory alone | 13.35 | 18.75 | >10 |
| NovelHopQA / BGE-M3 | Text only | 17.29 | 21.92 | >10 |
| NovelHopQA / BGE-M3 | Text + summary | 17.40 | 23.13 | >10 |
| NovelHopQA / BGE-M3 | Text + memory | 16.08 | 20.93 | >10 |
| NovelHopQA / BGE-M3 | All three (SAADI) | 16.52 | 22.14 | >10 |
| NovelHopQA / BGE-M3 | Summary alone | 7.16 | 10.24 | >10 |
| NovelHopQA / BGE-M3 | Memory alone | 5.40 | 9.47 | >10 |

Summary alone and memory alone are for information only: a chunk without that channel falls back to its text score (counts in step 01).

## Paired differences at HR@10 (first look; step 04 has the full set)

| Pair | Comparison | Diff (pp) | 95% CI | Hit only in first / only in second | McNemar p |
| --- | --- | --- | --- | --- | --- |
| LooGLE / Contriever | All three (SAADI) - Text + summary | +0.11 | [-0.93, +1.15] | 51 / 49 | 0.92 |
| LooGLE / Contriever | All three (SAADI) - Text + memory | +0.00 | [-0.76, +0.76] | 26 / 26 | 1 |
| LooGLE / Contriever | Text + summary - Text only | -0.44 | [-1.31, +0.49] | 31 / 39 | 0.403 |
| LooGLE / Contriever | Text + memory - Text only | -0.33 | [-1.36, +0.71] | 41 / 47 | 0.594 |
| LooGLE / Contriever | All three (SAADI) - Text only | -0.33 | [-1.47, +0.82] | 52 / 58 | 0.634 |
| MuSiQue-32K / Contriever | All three (SAADI) - Text + summary | +4.56 | [+2.78, +6.33] | 57 / 16 | 1.53e-06 |
| MuSiQue-32K / Contriever | All three (SAADI) - Text + memory | +1.56 | [+0.44, +2.78] | 22 / 8 | 0.0161 |
| MuSiQue-32K / Contriever | Text + summary - Text only | +3.00 | [+1.33, +4.67] | 42 / 15 | 0.00046 |
| MuSiQue-32K / Contriever | Text + memory - Text only | +6.00 | [+4.00, +8.00] | 71 / 17 | 5.15e-09 |
| MuSiQue-32K / Contriever | All three (SAADI) - Text only | +7.56 | [+5.44, +9.78] | 86 / 18 | 8.22e-12 |
| QASPER-64K / Contriever | All three (SAADI) - Text + summary | +1.35 | [+0.00, +2.70] | 49 / 31 | 0.0567 |
| QASPER-64K / Contriever | All three (SAADI) - Text + memory | +0.68 | [-0.45, +1.80] | 34 / 25 | 0.298 |
| QASPER-64K / Contriever | Text + summary - Text only | +1.13 | [-0.08, +2.25] | 39 / 24 | 0.0769 |
| QASPER-64K / Contriever | Text + memory - Text only | +1.80 | [+0.23, +3.38] | 69 / 45 | 0.0308 |
| QASPER-64K / Contriever | All three (SAADI) - Text only | +2.48 | [+0.90, +4.05] | 72 / 39 | 0.00224 |
| NovelHopQA / Contriever | All three (SAADI) - Text + summary | -0.66 | [-1.76, +0.33] | 9 / 15 | 0.307 |
| NovelHopQA / Contriever | All three (SAADI) - Text + memory | -0.44 | [-1.65, +0.77] | 13 / 17 | 0.585 |
| NovelHopQA / Contriever | Text + summary - Text only | +0.99 | [-0.22, +2.31] | 21 / 12 | 0.163 |
| NovelHopQA / Contriever | Text + memory - Text only | +0.77 | [-0.22, +1.87] | 15 / 8 | 0.21 |
| NovelHopQA / Contriever | All three (SAADI) - Text only | +0.33 | [-0.99, +1.65] | 20 / 17 | 0.743 |
| LooGLE / BGE-M3 | All three (SAADI) - Text + summary | +0.55 | [-0.44, +1.47] | 44 / 34 | 0.308 |
| LooGLE / BGE-M3 | All three (SAADI) - Text + memory | -0.27 | [-1.04, +0.49] | 25 / 30 | 0.59 |
| LooGLE / BGE-M3 | Text + summary - Text only | -1.15 | [-2.02, -0.27] | 22 / 43 | 0.0125 |
| LooGLE / BGE-M3 | Text + memory - Text only | -0.33 | [-1.31, +0.66] | 38 / 44 | 0.581 |
| LooGLE / BGE-M3 | All three (SAADI) - Text only | -0.60 | [-1.58, +0.38] | 36 / 47 | 0.272 |
| MuSiQue-32K / BGE-M3 | All three (SAADI) - Text + summary | -1.00 | [-2.44, +0.44] | 19 / 28 | 0.243 |
| MuSiQue-32K / BGE-M3 | All three (SAADI) - Text + memory | +1.44 | [+0.44, +2.56] | 19 / 6 | 0.0146 |
| MuSiQue-32K / BGE-M3 | Text + summary - Text only | +1.89 | [+0.67, +3.11] | 24 / 7 | 0.00333 |
| MuSiQue-32K / BGE-M3 | Text + memory - Text only | -0.56 | [-2.00, +0.78] | 19 / 24 | 0.542 |
| MuSiQue-32K / BGE-M3 | All three (SAADI) - Text only | +0.89 | [-0.67, +2.44] | 30 / 22 | 0.332 |
| QASPER-64K / BGE-M3 | All three (SAADI) - Text + summary | -0.45 | [-1.95, +1.05] | 49 / 55 | 0.624 |
| QASPER-64K / BGE-M3 | All three (SAADI) - Text + memory | +2.10 | [+1.05, +3.23] | 43 / 15 | 0.000307 |
| QASPER-64K / BGE-M3 | Text + summary - Text only | +2.03 | [+0.83, +3.23] | 49 / 22 | 0.00182 |
| QASPER-64K / BGE-M3 | Text + memory - Text only | -0.53 | [-2.18, +1.20] | 62 / 69 | 0.6 |
| QASPER-64K / BGE-M3 | All three (SAADI) - Text only | +1.58 | [-0.08, +3.30] | 76 / 55 | 0.0802 |
| NovelHopQA / BGE-M3 | All three (SAADI) - Text + summary | -0.99 | [-1.98, +0.00] | 6 / 15 | 0.0784 |
| NovelHopQA / BGE-M3 | All three (SAADI) - Text + memory | +1.21 | [+0.00, +2.42] | 23 / 12 | 0.0895 |
| NovelHopQA / BGE-M3 | Text + summary - Text only | +1.21 | [-0.11, +2.64] | 26 / 15 | 0.117 |
| NovelHopQA / BGE-M3 | Text + memory - Text only | -0.99 | [-2.09, +0.11] | 9 / 18 | 0.122 |
| NovelHopQA / BGE-M3 | All three (SAADI) - Text only | +0.22 | [-1.32, +1.76] | 26 / 24 | 0.888 |


## Files written

- `results/rebuttal/c3_channel_ablation/per_query/loogle__contriever.csv.gz`
- `results/rebuttal/c3_channel_ablation/per_query/musique_32k__contriever.csv.gz`
- `results/rebuttal/c3_channel_ablation/per_query/qasper_64k__contriever.csv.gz`
- `results/rebuttal/c3_channel_ablation/per_query/novelhopqa__contriever.csv.gz`
- `results/rebuttal/c3_channel_ablation/per_query/loogle__bge_m3.csv.gz`
- `results/rebuttal/c3_channel_ablation/per_query/musique_32k__bge_m3.csv.gz`
- `results/rebuttal/c3_channel_ablation/per_query/qasper_64k__bge_m3.csv.gz`
- `results/rebuttal/c3_channel_ablation/per_query/novelhopqa__bge_m3.csv.gz`
- `experiments/rebuttal/c3_channel_ablation/reports/step_03.md`
- `experiments/rebuttal/c3_channel_ablation/reports/step_03.json`
