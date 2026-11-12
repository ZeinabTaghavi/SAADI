# C2: SAADI fusion weights chosen on held-out queries

## Summary

Retrievers: BM25, Contriever, BGE-M3; not included in this analysis: Jina, Qwen.

1. mokb Q1. With all four weights chosen without the evaluated query (5-fold CV grouped by document, 1296-point grid; A1), held-out SAADI is above the text-only Vanilla in 11/12 (dataset, retriever) pairs at HR@10 and 10/12 at HR@5; the 95% paired interval is above zero in 6/12 (HR@10) and 5/12 (HR@5). With the published weights the interval is above zero in 5/12 (HR@10) and 4/12 (HR@5).
2. Held-out minus published weights, HR@10: mean +0.65 points (range -0.88 to +2.44); the interval is below zero in 1/12 pairs. HR@5: mean +0.34.
3. Ranking (Table 1): SAADI is first or second in 23/24 cells with held-out (A1) weights vs 21/24 with published weights (24 cells recomputed); 4 cell(s) changed rank.
4. iVU8 Q3. Weights transferred from the other three datasets (B) beat Vanilla at HR@10 in 9/12 pairs (interval above zero: 6); one global setting per held-out dataset: 8/12 (5). Median share of channel settings within 1 point of the best HR@10: 0.19.
5. A learned pairwise logistic fusion (demonstration) gives mean HR@10 56.5 vs 56.2 for A1 and beats Vanilla in 12/12 pairs. Part C: the 100-query tuning sample could not be reconstructed, so C is skipped.

## Definitions and commands

- Hit@k: a gold chunk in the top k, or every chunk of one evidence group in the top k (`evaluate_retrieval_results`). HR = hits / eligible queries. Hit rank = the 0-based rank at which this is first met; MRR = mean of 1/(hit rank + 1).
- Score: Equation 2 as published (`combine_mbr_variant_chunk_scores`, variant MBR2, z-score per channel over the document's chunks, max aggregation, per-chunk missing-signal renormalization). lambda_T = mu_N = 1. Grid: lambda_M, lambda_S in [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]; mu_A, mu_D in [0.0, 0.3, 0.5, 1.0] (1296 points). Published point (lambda_M, lambda_S | mu_A, mu_D) = (0.5, 0.3 | 1, 1). Vanilla = the text channel alone (lambda_M = lambda_S = 0).
- A: 5 outer folds over queries grouped by document (groups shuffled with seed 42, each group to the fold with fewest queries); per training part the point with the highest mean HR@10 (ties: HR@5, MRR, smallest L1 distance to the published point). A1 = all four weights; A2 = channel weights only, mu = (1, 1, 1). B: per retriever, the point maximizing the mean of per-dataset HR@10 over the other three datasets; B-global: the same over all retrievers of the other three datasets.
- D: 95% paired bootstrap over queries (10,000 resamples, seed 42; `paired_bootstrap_ci`) and a document-level cluster bootstrap (10,000, seed 42). Differences in HR points. Commands: `bash experiments/rebuttal/C2_heldout_fusion/run_gpus_0123.sh`, `bash experiments/rebuttal/C2_heldout_fusion/run_gpus_4567.sh`; analysis only: `... --analysis-only`.

## Reproduction and checks

| Pair | Queries | Identical top-10 vs saved | HR@5 / HR@10 ours (published weights) | saved | text-only vs Table 1 Vanilla (HR@5 / HR@10) |
| --- | --- | --- | --- | --- | --- |
| loogle__bm25 | 1832 | 1832 | 92.30 / 96.45 | 92.30 / 96.45 | 91.87 / 95.74 vs 91.5 / 95.4 (differs) |
| loogle__contriever | 1832 | 1832 | 75.44 / 85.70 | 75.44 / 85.70 | 75.16 / 86.03 vs 72.4 / 73.3 (differs) |
| loogle__bge_m3 | 1832 | 1832 | 81.06 / 90.12 | 81.06 / 90.12 | 82.48 / 90.72 vs 80.6 / 89.8 (differs) |
| musique_32k__bm25 | 900 | 900 | 62.78 / 77.67 | 62.78 / 77.67 | 57.56 / 74.00 vs 57.6 / 74.0 |
| musique_32k__contriever | 900 | 900 | 68.78 / 83.67 | 68.78 / 83.67 | 59.89 / 76.11 vs 59.9 / 76.1 |
| musique_32k__bge_m3 | 900 | 900 | 76.89 / 88.33 | 76.89 / 88.33 | 73.22 / 87.44 vs 73.2 / 87.4 |
| qasper_64k__bm25 | 1333 | 1333 | 16.95 / 26.11 | 16.95 / 26.11 | 14.40 / 22.06 vs 14.4 / 22.1 |
| qasper_64k__contriever | 1333 | 1333 | 16.20 / 25.81 | 16.20 / 25.81 | 15.15 / 23.33 vs 15.2 / 23.3 |
| qasper_64k__bge_m3 | 1333 | 1333 | 20.93 / 30.91 | 20.93 / 30.91 | 21.01 / 29.33 vs 21.0 / 29.3 |
| novelhopqa__bm25 | 908 | 908 | 18.06 / 22.03 | 18.06 / 22.03 | 17.51 / 20.70 vs 16.9 / 20.0 (differs) |
| novelhopqa__contriever | 908 | 908 | 13.55 / 18.17 | 13.55 / 18.17 | 12.44 / 17.84 vs 11.8 / 11.8 (differs) |
| novelhopqa__bge_m3 | 908 | 908 | 16.52 / 22.14 | 16.52 / 22.14 | 17.29 / 21.92 vs 14.1 / 20.7 (differs) |

Vectorized grid vs the repo function (full rankings, published point + 20 random grid points, seed 42): 0 different rankings and 0 different hit ranks out of 313299 compared. Vanilla grid point vs saved text-only top-10: 0 mismatches.

## A. Cross-validated weight selection (HR %, every query scored with weights chosen without it)

| Dataset | Retriever | Vanilla @5 / @10 | Published @5 / @10 | A1 held-out @5 / @10 | A2 (channels only) @5 / @10 | folds |
| --- | --- | --- | --- | --- | --- | --- |
| LooGLE | BM25 | 91.9 / 95.7 | 92.3 / 96.5 | 92.3 / 96.2 | 92.1 / 96.5 | 5 |
| LooGLE | Contriever | 75.2 / 86.0 | 75.4 / 85.7 | 76.3 / 86.2 | 75.9 / 85.9 | 5 |
| LooGLE | BGE-M3 | 82.5 / 90.7 | 81.1 / 90.1 | 82.3 / 90.6 | 82.3 / 91.0 | 5 |
| MuSiQue-32K | BM25 | 57.6 / 74.0 | 62.8 / 77.7 | 64.7 / 79.9 | 64.2 / 80.3 | 5 |
| MuSiQue-32K | Contriever | 59.9 / 76.1 | 68.8 / 83.7 | 68.8 / 84.9 | 69.8 / 85.9 | 5 |
| MuSiQue-32K | BGE-M3 | 73.2 / 87.4 | 76.9 / 88.3 | 76.1 / 90.8 | 75.0 / 90.0 | 5 |
| QASPER-64K | BM25 | 14.4 / 22.1 | 17.0 / 26.1 | 17.3 / 27.3 | 17.3 / 27.3 | 5 |
| QASPER-64K | Contriever | 15.2 / 23.3 | 16.2 / 25.8 | 15.8 / 25.2 | 15.6 / 25.0 | 5 |
| QASPER-64K | BGE-M3 | 21.0 / 29.3 | 20.9 / 30.9 | 22.1 / 31.9 | 22.1 / 31.9 | 5 |
| NovelHopQA | BM25 | 17.5 / 20.7 | 18.1 / 22.0 | 17.6 / 21.1 | 18.0 / 21.1 | 5 |
| NovelHopQA | Contriever | 12.4 / 17.8 | 13.5 / 18.2 | 13.2 / 18.1 | 13.2 / 19.3 | 5 |
| NovelHopQA | BGE-M3 | 17.3 / 21.9 | 16.5 / 22.1 | 17.1 / 22.7 | 17.2 / 22.7 | 5 |

## B. Leave-one-dataset-out transfer

| Held-out dataset | Retriever | Point chosen on the other datasets | HR@5 / HR@10 | minus Vanilla @10 [95% CI] | Global point | Global HR@5 / HR@10 |
| --- | --- | --- | --- | --- | --- | --- |
| LooGLE | BM25 | (0.75, 0.5 | 0.3, 1) | 91.5 / 96.3 | +0.5 [-0.2, +1.3] | (0.2, 0.75 | 0, 1) | 90.0 / 95.6 |
| LooGLE | Contriever | (0.75, 0.5 | 0.3, 1) | 73.4 / 85.1 | -0.9 [-2.3, +0.4] | (0.2, 0.75 | 0, 1) | 70.6 / 83.5 |
| LooGLE | BGE-M3 | (0.1, 0.75 | 0.5, 1) | 77.0 / 86.9 | -3.8 [-5.0, -2.6] | (0.2, 0.75 | 0, 1) | 77.2 / 87.2 |
| MuSiQue-32K | BM25 | (0.5, 0.5 | 0.3, 1) | 63.1 / 79.1 | +5.1 [+2.9, +7.3] | (0.3, 0.2 | 0.5, 1) | 60.9 / 77.0 |
| MuSiQue-32K | Contriever | (0.5, 0.2 | 0, 1) | 67.1 / 82.9 | +6.8 [+4.8, +8.8] | (0.3, 0.2 | 0.5, 1) | 66.4 / 82.8 |
| MuSiQue-32K | BGE-M3 | (0.1, 0.3 | 0, 1) | 74.8 / 89.6 | +2.1 [+0.9, +3.3] | (0.3, 0.2 | 0.5, 1) | 76.0 / 88.7 |
| QASPER-64K | BM25 | (0.75, 0.5 | 0.3, 1) | 16.7 / 26.6 | +4.6 [+2.9, +6.2] | (0.3, 0.5 | 0, 1) | 17.5 / 26.4 |
| QASPER-64K | Contriever | (0.75, 0.5 | 0.3, 1) | 15.4 / 25.7 | +2.4 [+0.7, +4.1] | (0.3, 0.5 | 0, 1) | 16.2 / 25.7 |
| QASPER-64K | BGE-M3 | (0.1, 0.3 | 0.3, 1) | 22.2 / 31.4 | +2.1 [+0.8, +3.4] | (0.3, 0.5 | 0, 1) | 22.0 / 32.3 |
| NovelHopQA | BM25 | (0.75, 0.5 | 0.3, 1) | 17.3 / 21.6 | +0.9 [-0.4, +2.2] | (0.5, 0.5 | 0, 1) | 17.2 / 21.9 |
| NovelHopQA | Contriever | (0.75, 0.5 | 0.3, 1) | 12.9 / 18.4 | +0.6 [-1.1, +2.2] | (0.5, 0.5 | 0, 1) | 13.0 / 18.3 |
| NovelHopQA | BGE-M3 | (0.1, 0.5 | 0.5, 1) | 16.9 / 21.9 | +0.0 [-1.5, +1.5] | (0.5, 0.5 | 0, 1) | 16.4 / 21.6 |

## C. Excluding the original tuning sample

- How the sample was drawn: load_*_qa_entries(..., selection_method='random') calls random.sample(idxs, n) on the global RNG; no seed is set anywhere in src/. The query ids of the sample are not saved, so they cannot be read back.
- Test: recompute the Table 9 cells (40: LooGLE and NovelHopQA x BM25, Contriever x 5 channel settings x HR@5/HR@10, mu = (1, 1, 1)) on candidate query sets and compare with the reported integer percentages (outputs/saadi/reference/summaries/43a8b999722a1192ad32.json).

| Candidate query set | Cells compared | Cells matching |
| --- | --- | --- |
| full test set, normalization=zscore | 40 | 17 |
| first 100 saved rows, normalization=zscore | 40 | 5 |
| 100 smallest query ids, normalization=zscore | 20 | 3 |
| full test set, normalization=none | 40 | 10 |
| first 100 saved rows, normalization=none | 40 | 3 |
| 100 smallest query ids, normalization=none | 20 | 1 |

No candidate 100-query set matches every cell, so the tuning sample is not reconstructed and C is skipped. A and B do not depend on it. Per-cell values: `tables/c_table9_reconstruction.csv`.

## D. Paired differences (HR points; 95% paired bootstrap over queries; cluster = by document)

HR@10:

| Pair | A1 minus Vanilla [CI] | cluster CI | A1 minus published [CI] | cluster CI | A2 minus Vanilla [CI] | B minus Vanilla [CI] |
| --- | --- | --- | --- | --- | --- | --- |
| LooGLE / BM25 | +0.4 [-0.2, +1.0] | [-0.2, +1.0] | -0.3 [-0.8, +0.2] | [-0.8, +0.3] | +0.7 [+0.2, +1.3] | +0.5 [-0.2, +1.3] |
| LooGLE / Contriever | +0.2 [-0.5, +1.0] | [-0.6, +1.0] | +0.5 [-0.4, +1.5] | [-0.4, +1.5] | -0.2 [-0.9, +0.6] | -0.9 [-2.3, +0.4] |
| LooGLE / BGE-M3 | -0.1 [-0.8, +0.5] | [-0.7, +0.4] | +0.5 [-0.3, +1.3] | [-0.3, +1.4] | +0.3 [-0.4, +0.9] | -3.8 [-5.0, -2.6] |
| MuSiQue-32K / BM25 | +5.9 [+3.3, +8.4] | [+3.4, +8.5] | +2.2 [+0.3, +4.1] | [+0.3, +4.1] | +6.3 [+3.8, +8.9] | +5.1 [+2.9, +7.3] |
| MuSiQue-32K / Contriever | +8.8 [+6.3, +11.3] | [+6.2, +11.3] | +1.2 [-0.6, +3.0] | [-0.1, +2.6] | +9.8 [+7.2, +12.3] | +6.8 [+4.8, +8.8] |
| MuSiQue-32K / BGE-M3 | +3.3 [+1.7, +5.0] | [+1.5, +5.2] | +2.4 [+1.0, +4.0] | [+0.8, +4.1] | +2.6 [+0.9, +4.2] | +2.1 [+0.9, +3.3] |
| QASPER-64K / BM25 | +5.3 [+3.5, +7.0] | [+3.6, +6.8] | +1.2 [-0.2, +2.6] | [+0.2, +2.2] | +5.3 [+3.5, +7.0] | +4.6 [+2.9, +6.2] |
| QASPER-64K / Contriever | +1.9 [+0.4, +3.4] | [+0.6, +3.1] | -0.6 [-1.6, +0.4] | [-1.6, +0.3] | +1.7 [+0.2, +3.1] | +2.4 [+0.7, +4.1] |
| QASPER-64K / BGE-M3 | +2.6 [+0.8, +4.4] | [+0.7, +4.5] | +1.0 [-0.6, +2.6] | [-0.6, +2.4] | +2.6 [+0.8, +4.4] | +2.1 [+0.8, +3.4] |
| NovelHopQA / BM25 | +0.4 [-0.8, +1.8] | [-0.5, +1.5] | -0.9 [-1.7, -0.1] | [-1.6, -0.1] | +0.4 [-0.7, +1.5] | +0.9 [-0.4, +2.2] |
| NovelHopQA / Contriever | +0.2 [-1.0, +1.4] | [-1.3, +1.7] | -0.1 [-1.1, +0.9] | [-1.0, +0.7] | +1.4 [+0.6, +2.3] | +0.6 [-1.1, +2.2] |
| NovelHopQA / BGE-M3 | +0.8 [-0.7, +2.2] | [-0.6, +2.1] | +0.6 [-0.3, +1.4] | [-0.2, +1.4] | +0.8 [-0.7, +2.2] | +0.0 [-1.5, +1.5] |

HR@5:

| Pair | A1 minus Vanilla [CI] | cluster CI | A1 minus published [CI] | cluster CI | A2 minus Vanilla [CI] | B minus Vanilla [CI] |
| --- | --- | --- | --- | --- | --- | --- |
| LooGLE / BM25 | +0.4 [-0.3, +1.1] | [-0.3, +1.1] | +0.0 [-0.7, +0.7] | [-0.7, +0.6] | +0.2 [-0.5, +1.0] | -0.4 [-1.4, +0.7] |
| LooGLE / Contriever | +1.1 [+0.1, +2.2] | [-0.1, +2.3] | +0.9 [-0.2, +2.0] | [-0.3, +2.0] | +0.8 [-0.4, +1.9] | -1.7 [-3.4, -0.1] |
| LooGLE / BGE-M3 | -0.2 [-1.0, +0.7] | [-1.0, +0.7] | +1.3 [+0.2, +2.4] | [+0.0, +2.5] | -0.2 [-1.0, +0.7] | -5.5 [-7.0, -4.0] |
| MuSiQue-32K / BM25 | +7.1 [+4.3, +9.9] | [+4.8, +9.5] | +1.9 [-0.2, +4.1] | [-0.3, +4.2] | +6.7 [+4.0, +9.4] | +5.6 [+3.4, +7.7] |
| MuSiQue-32K / Contriever | +8.9 [+5.9, +12.0] | [+5.9, +12.0] | +0.0 [-2.1, +2.2] | [-2.3, +2.5] | +9.9 [+7.0, +13.0] | +7.2 [+4.9, +9.6] |
| MuSiQue-32K / BGE-M3 | +2.9 [+0.6, +5.2] | [-0.1, +5.8] | -0.8 [-2.8, +1.2] | [-3.1, +1.6] | +1.8 [-0.4, +4.0] | +1.6 [-0.1, +3.2] |
| QASPER-64K / BM25 | +2.9 [+1.4, +4.5] | [+1.5, +4.4] | +0.4 [-0.7, +1.4] | [-0.6, +1.4] | +2.9 [+1.4, +4.5] | +2.3 [+0.8, +3.9] |
| QASPER-64K / Contriever | +0.6 [-0.9, +2.2] | [-1.1, +2.2] | -0.5 [-1.4, +0.5] | [-1.5, +0.6] | +0.5 [-1.1, +2.0] | +0.2 [-1.4, +1.8] |
| QASPER-64K / BGE-M3 | +1.1 [-0.6, +2.7] | [-0.4, +2.5] | +1.1 [-0.3, +2.6] | [+0.2, +2.0] | +1.1 [-0.6, +2.7] | +1.2 [+0.1, +2.4] |
| NovelHopQA / BM25 | +0.1 [-1.1, +1.3] | [-0.6, +0.8] | -0.4 [-1.2, +0.3] | [-1.3, +0.4] | +0.4 [-0.7, +1.5] | -0.2 [-1.5, +1.1] |
| NovelHopQA / Contriever | +0.8 [-0.3, +1.9] | [-0.4, +2.0] | -0.3 [-1.5, +0.8] | [-1.8, +0.8] | +0.8 [+0.1, +1.5] | +0.4 [-1.2, +2.1] |
| NovelHopQA / BGE-M3 | -0.2 [-1.3, +0.9] | [-1.6, +1.1] | +0.6 [-0.2, +1.4] | [-0.2, +1.3] | -0.1 [-1.2, +1.0] | -0.4 [-1.8, +1.0] |

| System | HR@k | above Vanilla | CI above 0 | CI below 0 | cluster CI above 0 | above published | CI below published |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A1 held-out | 5 | 10/12 | 5 | 0 | 3 | 6 | 0 |
| A1 held-out | 10 | 11/12 | 6 | 0 | 6 | 8 | 1 |
| A2 held-out | 5 | 10/12 | 4 | 0 | 4 | 7 | 0 |
| A2 held-out | 10 | 11/12 | 8 | 0 | 8 | 9 | 1 |
| B transfer | 5 | 7/12 | 4 | 2 | 4 | 3 | 5 |
| B transfer | 10 | 9/12 | 6 | 1 | 6 | 5 | 1 |
| B global | 5 | 7/12 | 4 | 3 | 4 | 2 | 5 |
| B global | 10 | 8/12 | 5 | 2 | 6 | 4 | 3 |
| G learned | 5 | 10/12 | 5 | 0 | 5 | 6 | 0 |
| G learned | 10 | 12/12 | 6 | 0 | 7 | 10 | 0 |
| Published | 5 | 9/12 | 4 | | | | |
| Published | 10 | 10/12 | 5 | | | | |

## E. Ranking stability in Table 1

SAADI first or second: 21/24 with published weights, 23/24 with A1 held-out weights (other methods unchanged; held-out value rounded to one decimal as printed; 24 cells recomputed). First: 13 -> 13. Source: `results/tables/main/retrieval_comparison/table.csv`.

| Retriever | Dataset | HR@k | SAADI published | SAADI A1 | rank published -> held-out | best other |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 | QASPER-64K | 10 | 26.1 | 27.3 | 3 -> 2 | LQCA 47.0 |
| BGE-M3 | MuSiQue-32K | 5 | 76.9 | 76.1 | 1 -> 2 | LQCA 76.7 |
| BGE-M3 | MuSiQue-32K | 10 | 88.3 | 90.8 | 2 -> 1 | LQCA 90.4 |
| BGE-M3 | QASPER-64K | 5 | 20.9 | 22.1 | 3 -> 2 | LQCA 32.9 |

## F. Sensitivity (channel grid with mu = (1, 1, 1); chosen points per outer fold)

| Pair | HR@10 published | best (point) | worst | share within 1 pt of best | A1 points chosen in the folds (lambda_M, lambda_S | mu_A, mu_D) |
| --- | --- | --- | --- | --- | --- |
| LooGLE / BM25 | 96.5 | 96.7 (0.3, 0 | 1, 1) | 87.2 | 0.41 | (0.3, 0 | 1, 1); (0.3, 0.2 | 0, 0); (0.3, 0.1 | 0.3, 0); (0.1, 0.3 | 1, 0); (0.3, 0 | 0.5, 1) |
| LooGLE / Contriever | 85.7 | 86.4 (0.3, 0 | 1, 1) | 73.8 | 0.25 | (0.3, 0 | 0, 1); (0.3, 0 | 0, 1); (0.3, 0.2 | 0.5, 1); (0.3, 0 | 0.3, 1); (0.3, 0 | 0, 1) |
| LooGLE / BGE-M3 | 90.1 | 91.2 (0.2, 0.1 | 1, 1) | 77.0 | 0.19 | (0.3, 0 | 0, 1); (0.2, 0 | 0.3, 1); (0.1, 0.1 | 0, 1); (0.3, 0 | 0, 1); (0.2, 0 | 1, 0.5) |
| MuSiQue-32K / BM25 | 77.7 | 81.0 (1, 1.5 | 1, 1) | 74.0 | 0.15 | (1, 1.5 | 0, 0.5); (0.75, 1 | 0, 0.3); (0.75, 2 | 1, 0.5); (1, 1.5 | 0, 0.5); (0.75, 1 | 0, 0.3) |
| MuSiQue-32K / Contriever | 83.7 | 85.9 (1.5, 1.5 | 1, 1) | 76.1 | 0.19 | (0.75, 2 | 0.5, 0.3); (0.75, 2 | 0.5, 0.3); (0.75, 2 | 0.5, 0.3); (1.5, 1.5 | 1, 1); (1.5, 1.5 | 1, 1) |
| MuSiQue-32K / BGE-M3 | 88.3 | 90.6 (0.1, 0.5 | 1, 1) | 83.4 | 0.20 | (0.2, 0.75 | 0.3, 1); (0.2, 0.75 | 0.3, 1); (0.2, 0.75 | 0.3, 1); (0.2, 0.75 | 0, 1); (0.2, 0.75 | 0.3, 1) |
| QASPER-64K / BM25 | 26.1 | 27.3 (0.5, 0.75 | 1, 1) | 22.1 | 0.17 | (0.5, 0.75 | 1, 1); (0.5, 0.75 | 1, 1); (0.5, 0.75 | 1, 1); (0.5, 0.75 | 1, 1); (0.5, 0.75 | 1, 1) |
| QASPER-64K / Contriever | 25.8 | 26.1 (0.3, 0.1 | 1, 1) | 22.3 | 0.35 | (0.5, 0.75 | 0.3, 1); (0.5, 0.1 | 0.3, 1); (0.5, 0.75 | 0.3, 1); (0.3, 0.2 | 0, 1); (0.3, 0.2 | 0, 1) |
| QASPER-64K / BGE-M3 | 30.9 | 32.7 (0.2, 0.75 | 1, 1) | 24.0 | 0.11 | (0, 0.75 | 1, 1); (0, 0.75 | 1, 1); (0.2, 0.75 | 1, 1); (0.2, 0.75 | 1, 1); (0.2, 0.75 | 1, 1) |
| NovelHopQA / BM25 | 22.0 | 22.0 (0.5, 0.3 | 1, 1) | 16.0 | 0.49 | (0.2, 0.3 | 0, 0); (1.5, 0.3 | 0.5, 1); (0.5, 0.5 | 1, 0.5); (1.5, 0.3 | 0.5, 1); (0.5, 0.5 | 0, 0.3) |
| NovelHopQA / Contriever | 18.2 | 19.3 (0.1, 0.1 | 1, 1) | 14.1 | 0.33 | (0.1, 0.1 | 0, 0.5); (0.2, 0.1 | 0.3, 0.5); (0.5, 0.1 | 0, 0.5); (0.5, 0.1 | 0.3, 0.3); (0.5, 0.2 | 0, 0) |
| NovelHopQA / BGE-M3 | 22.1 | 23.1 (0, 0.3 | 1, 1) | 18.0 | 0.16 | (0.1, 0.3 | 0, 0.5); (0, 0.3 | 1, 1); (0, 0.3 | 1, 1); (0, 0.3 | 1, 1); (0.3, 0.3 | 1, 0.3) |

Heatmaps: `results/rebuttal/C2_heldout_fusion/figures/heatmap_hr10_bm25.png`, `results/rebuttal/C2_heldout_fusion/figures/heatmap_hr10_mean_over_retrievers.png`.

## G. Learned fusion (demonstration, not the main result)

Pairwise logistic regression in the same outer folds: each evidence chunk against every non-evidence chunk of the same training query; features = z-scored text, name, alias, description, summary (0 when missing) and a missing indicator per memory/summary channel; L2, C = 1.0, no intercept (Newton's method; seed 42 has no effect without an intercept). Weights per fold: `tables/g_learned_weights.csv`.

| Pair | A1 @5 / @10 | Learned @5 / @10 | Learned minus A1 @10 |
| --- | --- | --- | --- |
| LooGLE / BM25 | 92.3 / 96.2 | 92.7 / 96.2 | +0.0 |
| LooGLE / Contriever | 76.3 / 86.2 | 75.9 / 86.1 | -0.2 |
| LooGLE / BGE-M3 | 82.3 / 90.6 | 82.4 / 90.9 | +0.3 |
| MuSiQue-32K / BM25 | 64.7 / 79.9 | 61.8 / 80.3 | +0.4 |
| MuSiQue-32K / Contriever | 68.8 / 84.9 | 67.6 / 85.1 | +0.2 |
| MuSiQue-32K / BGE-M3 | 76.1 / 90.8 | 75.6 / 89.8 | -1.0 |
| QASPER-64K / BM25 | 17.3 / 27.3 | 18.0 / 27.3 | +0.0 |
| QASPER-64K / Contriever | 15.8 / 25.2 | 17.0 / 26.5 | +1.3 |
| QASPER-64K / BGE-M3 | 22.1 / 31.9 | 22.6 / 32.8 | +0.9 |
| NovelHopQA / BM25 | 17.6 / 21.1 | 18.1 / 21.7 | +0.6 |
| NovelHopQA / Contriever | 13.2 / 18.1 | 13.4 / 19.1 | +1.0 |
| NovelHopQA / BGE-M3 | 17.1 / 22.7 | 16.4 / 22.8 | +0.1 |

## H. Full grid: rank of the published setting and the selected weights (1296 points)

Mean HR@10 over the 12 analysed pairs for every grid point (`tables/f_grid_hr.csv`); rank = 1 + number of points with a strictly higher mean. The best single point is chosen with all evaluated queries in hand, so it is an upper bound for any one fixed setting, not a held-out result.

- Published point (0.5, 0.3 | 1, 1): mean HR@10 55.59, rank 26 of 1296 (top 2.0%).
- Best single point (0.5, 0.5 | 0.3, 1): mean HR@10 55.83, 0.24 points above the published point.

A1-selected weights over all outer folds (`tables/a_chosen_per_fold.csv`):

| Weight | Published | Median selected | Folds with a non-zero weight |
| --- | --- | --- | --- |
| lambda_M | 0.5 | 0.3 | 55/60 (91.7%) |
| lambda_S | 0.3 | 0.3 | 50/60 (83.3%) |
| mu_A | 1 | 0.3 | 40/60 (66.7%) |
| mu_D | 1 | 1 | 55/60 (91.7%) |

Mean HR@10 per dataset over the analysed retrievers (`tables/d_per_pair.csv`):

| Dataset | Retrievers | Published | A1 held-out | A1 minus published |
| --- | --- | --- | --- | --- |
| LooGLE | 3 | 90.76 | 91.01 | +0.25 |
| MuSiQue-32K | 3 | 83.22 | 85.19 | +1.96 |
| QASPER-64K | 3 | 27.61 | 28.13 | +0.53 |
| NovelHopQA | 3 | 20.78 | 20.63 | -0.15 |
| All | | 55.59 | 56.24 | +0.65 |

## Not part of C2 (index-time settings; need new LLM builds)

- folding factor f (branching): `results/tables/ablations/hierarchy_branching/`
- recall budget E/A/D: `results/tables/ablations/memory_budget/`
- memory selector: `results/tables/ablations/memory_selector/`
- field weights (Table of mu, saved summaries): `results/tables/ablations/memory_field_weights/`
- channel weights (Table 9, saved summaries): `results/tables/ablations/score_fusion/`
- chunk size: suite `configs/suites/chunk_size.yaml`; no saved results found in the repository (only the 500-token LooGLE/NovelHopQA and 100-token MuSiQue/QASPER builds are present).

## Files

- Tables: `results/rebuttal/C2_heldout_fusion/tables/`; full numbers: `results/rebuttal/C2_heldout_fusion/analysis.json`.
- Streams and grid arrays (not committed): `outputs/rebuttal/C2_heldout_fusion/streams/`, `grid/`.
