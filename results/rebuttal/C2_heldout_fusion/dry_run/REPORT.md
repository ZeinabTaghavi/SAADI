# C2: SAADI fusion weights chosen on held-out queries (DRY RUN: BM25, 2 documents per dataset, reduced grid)

## Summary

1. mokb Q1. With all four weights chosen without the evaluated query (2-fold CV grouped by document, 64-point grid; A1), held-out SAADI is above the text-only Vanilla in 2/4 (dataset, retriever) pairs at HR@10 and 2/4 at HR@5; the 95% paired interval is above zero in 0/4 (HR@10) and 1/4 (HR@5). With the published weights the interval is above zero in 1/4 (HR@10) and 1/4 (HR@5).
2. Held-out minus published weights, HR@10: mean -2.54 points (range -7.14 to +0.00); the interval is below zero in 0/4 pairs. HR@5: mean -2.52.
3. Ranking (Table 1): SAADI is first or second in 32/40 cells with held-out (A1) weights vs 34/40 with published weights (8 cells recomputed); 2 cell(s) changed rank.
4. iVU8 Q3. Weights transferred from the other three datasets (B) beat Vanilla at HR@10 in 3/4 pairs (interval above zero: 0); one global setting per held-out dataset: 3/4 (0). Median share of channel settings within 1 point of the best HR@10: 0.47.
5. A learned pairwise logistic fusion (demonstration) gives mean HR@10 57.4 vs 55.5 for A1 and beats Vanilla in 2/4 pairs. Part C: the 100-query tuning sample could not be reconstructed, so C is skipped.

## Definitions and commands

- Hit@k: a gold chunk in the top k, or every chunk of one evidence group in the top k (`evaluate_retrieval_results`). HR = hits / eligible queries. Hit rank = the 0-based rank at which this is first met; MRR = mean of 1/(hit rank + 1).
- Score: Equation 2 as published (`combine_mbr_variant_chunk_scores`, variant MBR2, z-score per channel over the document's chunks, max aggregation, per-chunk missing-signal renormalization). lambda_T = mu_N = 1. Grid: lambda_M, lambda_S in [0.0, 0.3, 0.5, 1.0]; mu_A, mu_D in [0.0, 1.0] (64 points). Published point (lambda_M, lambda_S | mu_A, mu_D) = (0.5, 0.3 | 1, 1). Vanilla = the text channel alone (lambda_M = lambda_S = 0).
- A: 5 outer folds over queries grouped by document (groups shuffled with seed 42, each group to the fold with fewest queries); per training part the point with the highest mean HR@10 (ties: HR@5, MRR, smallest L1 distance to the published point). A1 = all four weights; A2 = channel weights only, mu = (1, 1, 1). B: per retriever, the point maximizing the mean of per-dataset HR@10 over the other three datasets; B-global: the same over all retrievers of the other three datasets.
- D: 95% paired bootstrap over queries (10,000 resamples, seed 42; `paired_bootstrap_ci`) and a document-level cluster bootstrap (10,000, seed 42). Differences in HR points. Commands: `bash experiments/rebuttal/C2_heldout_fusion/run_gpus_0123.sh`, `bash experiments/rebuttal/C2_heldout_fusion/run_gpus_4567.sh`; analysis only: `... --analysis-only`.

## Reproduction and checks

| Pair | Queries | Identical top-10 vs saved | HR@5 / HR@10 ours (published weights) | saved | text-only vs Table 1 Vanilla (HR@5 / HR@10) |
| --- | --- | --- | --- | --- | --- |
| loogle__bm25 | 21 | 21 | 90.48 / 95.24 | 90.48 / 95.24 | 85.71 / 90.48 vs 91.5 / 95.4 |
| musique_32k__bm25 | 42 | 42 | 69.05 / 85.71 | 69.05 / 85.71 | 64.29 / 80.95 vs 57.6 / 74.0 |
| qasper_64k__bm25 | 129 | 129 | 20.93 / 25.58 | 20.93 / 25.58 | 14.73 / 20.16 vs 14.4 / 22.1 |
| novelhopqa__bm25 | 145 | 145 | 22.76 / 25.52 | 22.76 / 25.52 | 21.38 / 25.52 vs 16.9 / 20.0 |

Vectorized grid vs the repo function (full rankings, published point + 20 random grid points, seed 42): 0 different rankings and 0 different hit ranks out of 7077 compared. Vanilla grid point vs saved text-only top-10: 0 mismatches.

## A. Cross-validated weight selection (HR %, every query scored with weights chosen without it)

| Dataset | Retriever | Vanilla @5 / @10 | Published @5 / @10 | A1 held-out @5 / @10 | A2 (channels only) @5 / @10 | folds |
| --- | --- | --- | --- | --- | --- | --- |
| LooGLE | BM25 | 85.7 / 90.5 | 90.5 / 95.2 | 85.7 / 95.2 | 85.7 / 95.2 | 2 |
| MuSiQue-32K | BM25 | 64.3 / 81.0 | 69.0 / 85.7 | 66.7 / 78.6 | 71.4 / 85.7 | 2 |
| QASPER-64K | BM25 | 14.7 / 20.2 | 20.9 / 25.6 | 19.4 / 23.3 | 18.6 / 23.3 | 2 |
| NovelHopQA | BM25 | 21.4 / 25.5 | 22.8 / 25.5 | 21.4 / 24.8 | 21.4 / 24.8 | 2 |

## B. Leave-one-dataset-out transfer

| Held-out dataset | Retriever | Point chosen on the other datasets | HR@5 / HR@10 | minus Vanilla @10 [95% CI] | Global point | Global HR@5 / HR@10 |
| --- | --- | --- | --- | --- | --- | --- |
| LooGLE | BM25 | (0.5, 0.3 | 0, 1) | 90.5 / 95.2 | +4.8 [+0.0, +14.3] | (0.5, 0.3 | 0, 1) | 90.5 / 95.2 |
| MuSiQue-32K | BM25 | (0.5, 0.3 | 0, 1) | 69.0 / 85.7 | +4.8 [+0.0, +11.9] | (0.5, 0.3 | 0, 1) | 69.0 / 85.7 |
| QASPER-64K | BM25 | (0.3, 0.3 | 0, 0) | 16.3 / 24.0 | +3.9 [+0.0, +7.8] | (0.3, 0.3 | 0, 0) | 16.3 / 24.0 |
| NovelHopQA | BM25 | (1, 1 | 0, 1) | 17.2 / 22.1 | -3.4 [-6.9, +0.0] | (1, 1 | 0, 1) | 17.2 / 22.1 |

## C. Excluding the original tuning sample

- How the sample was drawn: load_*_qa_entries(..., selection_method='random') calls random.sample(idxs, n) on the global RNG; no seed is set anywhere in src/. The query ids of the sample are not saved, so they cannot be read back.
- Test: recompute the Table 9 cells (60: LooGLE and NovelHopQA x BM25, Contriever, Jina x 5 channel settings x HR@5/HR@10, mu = (1, 1, 1)) on candidate query sets and compare with the reported integer percentages (outputs/saadi/reference/summaries/43a8b999722a1192ad32.json).

| Candidate query set | Cells compared | Cells matching |
| --- | --- | --- |
| full test set, normalization=zscore | 20 | 3 |
| first 100 saved rows, normalization=zscore | 20 | 3 |
| 100 smallest query ids, normalization=zscore | 10 | 3 |
| full test set, normalization=none | 20 | 2 |
| first 100 saved rows, normalization=none | 20 | 3 |
| 100 smallest query ids, normalization=none | 10 | 2 |

No candidate 100-query set matches every cell, so the tuning sample is not reconstructed and C is skipped. A and B do not depend on it. Per-cell values: `tables/c_table9_reconstruction.csv`.

## D. Paired differences (HR points; 95% paired bootstrap over queries; cluster = by document)

HR@10:

| Pair | A1 minus Vanilla [CI] | cluster CI | A1 minus published [CI] | cluster CI | A2 minus Vanilla [CI] | B minus Vanilla [CI] |
| --- | --- | --- | --- | --- | --- | --- |
| LooGLE / BM25 | +4.8 [+0.0, +14.3] | [+0.0, +8.3] | +0.0 [+0.0, +0.0] | [+0.0, +0.0] | +4.8 [+0.0, +14.3] | +4.8 [+0.0, +14.3] |
| MuSiQue-32K / BM25 | -2.4 [-11.9, +4.8] | [-4.5, +0.0] | -7.1 [-16.7, +0.0] | [-10.0, -4.5] | +4.8 [-4.8, +14.3] | +4.8 [+0.0, +11.9] |
| QASPER-64K / BM25 | +3.1 [-0.8, +7.8] | [+2.9, +3.4] | -2.3 [-6.2, +1.6] | [-3.4, -1.4] | +3.1 [+0.0, +7.0] | +3.9 [+0.0, +7.8] |
| NovelHopQA / BM25 | -0.7 [-2.1, +0.0] | [-1.6, +0.0] | -0.7 [-2.1, +0.0] | [-1.6, +0.0] | -0.7 [-2.1, +0.0] | -3.4 [-6.9, +0.0] |

HR@5:

| Pair | A1 minus Vanilla [CI] | cluster CI | A1 minus published [CI] | cluster CI | A2 minus Vanilla [CI] | B minus Vanilla [CI] |
| --- | --- | --- | --- | --- | --- | --- |
| LooGLE / BM25 | +0.0 [+0.0, +0.0] | [+0.0, +0.0] | -4.8 [-14.3, +0.0] | [-8.3, +0.0] | +0.0 [+0.0, +0.0] | +4.8 [+0.0, +14.3] |
| MuSiQue-32K / BM25 | +2.4 [-9.5, +14.3] | [+0.0, +4.5] | -2.4 [-11.9, +7.1] | [-5.0, +0.0] | +7.1 [-2.4, +19.0] | +4.8 [+0.0, +11.9] |
| QASPER-64K / BM25 | +4.7 [+0.8, +9.3] | [+1.7, +7.1] | -1.6 [-6.2, +2.3] | [-3.4, +0.0] | +3.9 [+0.0, +7.8] | +1.6 [-2.3, +5.4] |
| NovelHopQA / BM25 | +0.0 [-2.1, +2.1] | [+0.0, +0.0] | -1.4 [-3.4, +0.0] | [-1.6, -1.2] | +0.0 [-2.8, +2.8] | -4.1 [-8.3, +0.0] |

| System | HR@k | above Vanilla | CI above 0 | CI below 0 | cluster CI above 0 | above published | CI below published |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A1 held-out | 5 | 2/4 | 1 | 0 | 1 | 0 | 0 |
| A1 held-out | 10 | 2/4 | 0 | 0 | 1 | 0 | 0 |
| A2 held-out | 5 | 2/4 | 0 | 0 | 1 | 1 | 0 |
| A2 held-out | 10 | 3/4 | 0 | 0 | 1 | 0 | 0 |
| B transfer | 5 | 3/4 | 0 | 0 | 2 | 0 | 2 |
| B transfer | 10 | 3/4 | 0 | 0 | 2 | 0 | 0 |
| B global | 5 | 3/4 | 0 | 0 | 2 | 0 | 2 |
| B global | 10 | 3/4 | 0 | 0 | 2 | 0 | 0 |
| G learned | 5 | 2/4 | 1 | 0 | 2 | 1 | 0 |
| G learned | 10 | 2/4 | 1 | 0 | 1 | 1 | 0 |
| Published | 5 | 4/4 | 1 | | | | |
| Published | 10 | 3/4 | 1 | | | | |

## E. Ranking stability in Table 1

SAADI first or second: 34/40 with published weights, 32/40 with A1 held-out weights (other methods unchanged; held-out value rounded to one decimal as printed; 8 cells recomputed). First: 17 -> 15. Source: `results/tables/main/retrieval_comparison/table.csv`.

| Retriever | Dataset | HR@k | SAADI published | SAADI A1 | rank published -> held-out | best other |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 | LooGLE | 5 | 92.3 | 85.7 | 1 -> 4 | Vanilla 91.5 |
| BM25 | LooGLE | 10 | 96.5 | 95.2 | 1 -> 3 | LlamaIndex 95.5 |

## F. Sensitivity (channel grid with mu = (1, 1, 1); chosen points per outer fold)

| Pair | HR@10 published | best (point) | worst | share within 1 pt of best | A1 points chosen in the folds (lambda_M, lambda_S | mu_A, mu_D) |
| --- | --- | --- | --- | --- | --- |
| LooGLE / BM25 | 95.2 | 95.2 (0, 0.5 | 1, 1) | 90.5 | 0.69 | (0.3, 1 | 1, 1); (0.5, 1 | 1, 0) |
| MuSiQue-32K / BM25 | 85.7 | 88.1 (0, 1 | 1, 1) | 81.0 | 0.06 | (1, 0 | 0, 1); (1, 1 | 0, 0) |
| QASPER-64K / BM25 | 25.6 | 27.1 (1, 1 | 1, 1) | 20.2 | 0.38 | (0.3, 1 | 0, 1); (1, 0.5 | 1, 1) |
| NovelHopQA / BM25 | 25.5 | 25.5 (0, 0 | 1, 1) | 22.1 | 0.56 | (0.3, 0.3 | 0, 1); (0.3, 0 | 1, 1) |

Heatmaps: `results/rebuttal/C2_heldout_fusion/dry_run/figures/heatmap_hr10_bm25.png`, `results/rebuttal/C2_heldout_fusion/dry_run/figures/heatmap_hr10_mean_over_retrievers.png`.

## G. Learned fusion (demonstration, not the main result)

Pairwise logistic regression in the same outer folds: each evidence chunk against every non-evidence chunk of the same training query; features = z-scored text, name, alias, description, summary (0 when missing) and a missing indicator per memory/summary channel; L2, C = 1.0, no intercept (Newton's method; seed 42 has no effect without an intercept). Weights per fold: `tables/g_learned_weights.csv`.

| Pair | A1 @5 / @10 | Learned @5 / @10 | Learned minus A1 @10 |
| --- | --- | --- | --- |
| LooGLE / BM25 | 85.7 / 95.2 | 85.7 / 95.2 | +0.0 |
| MuSiQue-32K / BM25 | 66.7 / 78.6 | 64.3 / 81.0 | +2.4 |
| QASPER-64K / BM25 | 19.4 / 23.3 | 22.5 / 28.7 | +5.4 |
| NovelHopQA / BM25 | 21.4 / 24.8 | 22.8 / 24.8 | +0.0 |

## Not part of C2 (index-time settings; need new LLM builds)

- folding factor f (branching): `results/tables/ablations/hierarchy_branching/`
- recall budget E/A/D: `results/tables/ablations/memory_budget/`
- memory selector: `results/tables/ablations/memory_selector/`
- field weights (Table of mu, saved summaries): `results/tables/ablations/memory_field_weights/`
- channel weights (Table 9, saved summaries): `results/tables/ablations/score_fusion/`
- chunk size: suite `configs/suites/chunk_size.yaml`; no saved results found in the repository (only the 500-token LooGLE/NovelHopQA and 100-token MuSiQue/QASPER builds are present).

## Files

- Tables: `results/rebuttal/C2_heldout_fusion/dry_run/tables/`; full numbers: `results/rebuttal/C2_heldout_fusion/dry_run/analysis.json`.
- Streams and grid arrays (not committed): `outputs/rebuttal/C2_heldout_fusion/dry_run/streams/`, `grid/`.
