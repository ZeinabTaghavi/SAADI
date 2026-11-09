# C3 channel ablation - step 02: new variants for BM25

**Status: PASSED**

- Command: `bash experiments/rebuttal/c3_channel_ablation/sh/02_bm25.sh`
- Started: 2026-11-09 19:44:50; took 113 s
- Host: eee-s132-infsys-gpu03; Python 3.12.3; commit 98f4f789a04e (fix/notebook-key-and-install, 7 changed files)

## Settings

- Weights from `configs/methods/sadi.yaml`: (lambda_T, lambda_M, lambda_S) = (1, 0.5, 0.3), field weights (1, 1, 1), zscore. A new variant sets the dropped channel's weight to 0 in the published Equation 2; nothing else changes.
- Where each variant comes from: Text only: saved text-only top-10 in the streams (C2); Text + summary: scored by C3 (published Equation 2, one weight 0); Text + memory: scored by C3 (published Equation 2, one weight 0); All three (SAADI): paper's saved rankings; Summary alone: scored by C3 (published Equation 2, one weight 0); Memory alone: scored by C3 (published Equation 2, one weight 0).
- Rank = 1-based position within the top-10 at which the saved hit definition is met (gold chunk, or every chunk of one evidence group); median over eligible queries, '>10' when fewer than half are hit within the top-10.
- Paired differences in HR points: 95% bootstrap over queries (10000 resamples, seed 42); McNemar exact two-sided p.
- Pairs from `experiments/rebuttal/c3_channel_ablation/work/ready_pairs.json` (step 01, check 1 passed): loogle__bm25, musique_32k__bm25, qasper_64k__bm25, novelhopqa__bm25.

## Vanilla and SAADI against the paper

| Pair | Text only HR@5 / HR@10 (here) | Table 1 Vanilla | All three HR@5 / HR@10 (here) | Paper SAADI |
| --- | --- | --- | --- | --- |
| LooGLE / BM25 | 91.87 / 95.74 | 91.5 / 95.4 | 92.30 / 96.45 | 92.30 / 96.45 |
| MuSiQue-32K / BM25 | 57.56 / 74.00 | 57.6 / 74.0 | 62.78 / 77.67 | 62.78 / 77.67 |
| QASPER-64K / BM25 | 14.40 / 22.06 | 14.4 / 22.1 | 16.95 / 26.11 | 16.95 / 26.11 |
| NovelHopQA / BM25 | 17.51 / 20.70 | 16.9 / 20.0 | 18.06 / 22.03 | 18.06 / 22.03 |

Text only differs from Table 1's Vanilla on LooGLE and NovelHopQA (a separate run in the paper; step 01 explains it). All three is the paper's saved run itself, so it equals the paper's SAADI numbers.

## Guards

| Pair | Rank HR = repo HR (all variants, k=5,10) | Eligible queries | Time |
| --- | --- | --- | --- |
| LooGLE / BM25 | yes | 1832 | 5 s |
| MuSiQue-32K / BM25 | yes | 900 | 21 s |
| QASPER-64K / BM25 | yes | 1333 | 62 s |
| NovelHopQA / BM25 | yes | 908 | 25 s |

## HR@5 / HR@10 (%) and median rank, all variants

| Pair | Variant | HR@5 | HR@10 | Median rank |
| --- | --- | --- | --- | --- |
| LooGLE / BM25 | Text only | 91.87 | 95.74 | 1 |
| LooGLE / BM25 | Text + summary | 92.09 | 96.07 | 1 |
| LooGLE / BM25 | Text + memory | 92.14 | 96.29 | 1 |
| LooGLE / BM25 | All three (SAADI) | 92.30 | 96.45 | 1 |
| LooGLE / BM25 | Summary alone | 38.54 | 58.30 | 6 |
| LooGLE / BM25 | Memory alone | 53.77 | 67.25 | 5 |
| MuSiQue-32K / BM25 | Text only | 57.56 | 74.00 | 4 |
| MuSiQue-32K / BM25 | Text + summary | 59.56 | 76.56 | 4 |
| MuSiQue-32K / BM25 | Text + memory | 61.22 | 75.44 | 4 |
| MuSiQue-32K / BM25 | All three (SAADI) | 62.78 | 77.67 | 4 |
| MuSiQue-32K / BM25 | Summary alone | 52.44 | 73.44 | 5 |
| MuSiQue-32K / BM25 | Memory alone | 51.11 | 61.56 | 5 |
| QASPER-64K / BM25 | Text only | 14.40 | 22.06 | >10 |
| QASPER-64K / BM25 | Text + summary | 16.05 | 24.38 | >10 |
| QASPER-64K / BM25 | Text + memory | 15.98 | 24.46 | >10 |
| QASPER-64K / BM25 | All three (SAADI) | 16.95 | 26.11 | >10 |
| QASPER-64K / BM25 | Summary alone | 10.88 | 19.05 | >10 |
| QASPER-64K / BM25 | Memory alone | 11.10 | 16.80 | >10 |
| NovelHopQA / BM25 | Text only | 17.51 | 20.70 | >10 |
| NovelHopQA / BM25 | Text + summary | 17.62 | 21.81 | >10 |
| NovelHopQA / BM25 | Text + memory | 17.40 | 20.93 | >10 |
| NovelHopQA / BM25 | All three (SAADI) | 18.06 | 22.03 | >10 |
| NovelHopQA / BM25 | Summary alone | 4.19 | 7.49 | >10 |
| NovelHopQA / BM25 | Memory alone | 7.38 | 11.01 | >10 |

Summary alone and memory alone are for information only: a chunk without that channel falls back to its text score (counts in step 01).

## Paired differences at HR@10 (first look; step 04 has the full set)

| Pair | Comparison | Diff (pp) | 95% CI | Hit only in first / only in second | McNemar p |
| --- | --- | --- | --- | --- | --- |
| LooGLE / BM25 | All three (SAADI) - Text + summary | +0.38 | [-0.22, +1.04] | 20 / 13 | 0.296 |
| LooGLE / BM25 | All three (SAADI) - Text + memory | +0.16 | [-0.27, +0.66] | 11 / 8 | 0.648 |
| LooGLE / BM25 | Text + summary - Text only | +0.33 | [-0.22, +0.87] | 15 / 9 | 0.307 |
| LooGLE / BM25 | Text + memory - Text only | +0.55 | [-0.11, +1.20] | 25 / 15 | 0.154 |
| LooGLE / BM25 | All three (SAADI) - Text only | +0.71 | [+0.00, +1.42] | 27 / 14 | 0.0596 |
| MuSiQue-32K / BM25 | All three (SAADI) - Text + summary | +1.11 | [-0.33, +2.56] | 27 / 17 | 0.174 |
| MuSiQue-32K / BM25 | All three (SAADI) - Text + memory | +2.22 | [+0.78, +3.67] | 32 / 12 | 0.00366 |
| MuSiQue-32K / BM25 | Text + summary - Text only | +2.56 | [+0.78, +4.33] | 45 / 22 | 0.00674 |
| MuSiQue-32K / BM25 | Text + memory - Text only | +1.44 | [-0.33, +3.22] | 42 / 29 | 0.154 |
| MuSiQue-32K / BM25 | All three (SAADI) - Text only | +3.67 | [+1.67, +5.67] | 59 / 26 | 0.000447 |
| QASPER-64K / BM25 | All three (SAADI) - Text + summary | +1.73 | [+0.45, +3.00] | 50 / 27 | 0.0117 |
| QASPER-64K / BM25 | All three (SAADI) - Text + memory | +1.65 | [+0.53, +2.78] | 40 / 18 | 0.00535 |
| QASPER-64K / BM25 | Text + summary - Text only | +2.33 | [+1.13, +3.53] | 50 / 19 | 0.000244 |
| QASPER-64K / BM25 | Text + memory - Text only | +2.40 | [+1.05, +3.75] | 60 / 28 | 0.000846 |
| QASPER-64K / BM25 | All three (SAADI) - Text only | +4.05 | [+2.55, +5.48] | 77 / 23 | 5.51e-08 |
| NovelHopQA / BM25 | All three (SAADI) - Text + summary | +0.22 | [-0.55, +1.10] | 8 / 6 | 0.791 |
| NovelHopQA / BM25 | All three (SAADI) - Text + memory | +1.10 | [+0.33, +1.98] | 13 / 3 | 0.0213 |
| NovelHopQA / BM25 | Text + summary - Text only | +1.10 | [+0.11, +2.09] | 16 / 6 | 0.0525 |
| NovelHopQA / BM25 | Text + memory - Text only | +0.22 | [-0.55, +0.99] | 8 / 6 | 0.791 |
| NovelHopQA / BM25 | All three (SAADI) - Text only | +1.32 | [+0.22, +2.42] | 19 / 7 | 0.029 |


## Files written

- `results/rebuttal/c3_channel_ablation/per_query/loogle__bm25.csv.gz`
- `results/rebuttal/c3_channel_ablation/per_query/musique_32k__bm25.csv.gz`
- `results/rebuttal/c3_channel_ablation/per_query/qasper_64k__bm25.csv.gz`
- `results/rebuttal/c3_channel_ablation/per_query/novelhopqa__bm25.csv.gz`
- `experiments/rebuttal/c3_channel_ablation/reports/step_02.md`
- `experiments/rebuttal/c3_channel_ablation/reports/step_02.json`
