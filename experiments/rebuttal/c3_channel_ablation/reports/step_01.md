# C3 channel ablation - step 01: inputs and checks

**Status: PASSED**

- Command: `C3_STREAMS_DIR=/Users/uem-stag-lap-ic/Desktop/SAADI/outputs/rebuttal/C2_heldout_fusion/streams bash experiments/rebuttal/c3_channel_ablation/sh/01_check.sh`
- Started: 2026-11-09 16:35:33; took 219 s
- Host: LIGHT-C026225M.local; Python 3.14.8; commit c09a4e10f7ab (fix/notebook-key-and-install, 0 changed files)

## Settings

- Config: `configs/methods/sadi.yaml` (block `cached_retrieval`). Fusion weights (lambda_T, lambda_M, lambda_S) = (1, 0.5, 0.3); field weights (mu_N, mu_A, mu_D) = (1, 1, 1); normalization zscore; memory aggregation max.
- A variant drops a channel by setting its weight to 0 in the published Equation 2 (`combine_mbr_variant_chunk_scores`, MBR2). The remaining weights are not re-tuned or rescaled; Equation 2's own per-chunk division by the sum of the weights present is unchanged.
- Dense channel scores read from `C3_STREAMS_DIR` = `outputs/rebuttal/C2_heldout_fusion/streams`; C2 markers from `outputs/rebuttal/C2_heldout_fusion/markers`.
- BM25 channel scores written to `experiments/rebuttal/c3_channel_ablation/work/streams`.
- Datasets: loogle, musique_32k, qasper_64k, novelhopqa. Retrievers: bm25.

| Variant | lambda_T | lambda_M | lambda_S |
| --- | --- | --- | --- |
| Text only | 1.00 | 0.00 | 0.00 |
| Text + summary | 1.00 | 0.00 | 0.30 |
| Text + memory | 1.00 | 0.50 | 0.00 |
| All three (SAADI) | 1.00 | 0.50 | 0.30 |
| Summary alone | 0.00 | 0.00 | 0.30 |
| Memory alone | 0.00 | 0.50 | 0.00 |

Summary alone and memory alone are for information only (steps 02-04).

- Preserved sources (`docs/preserved_sources.json`): 93 files, 0 changed, 0 missing.

## 1. Inputs

| Pair | Saved rankings (queries) | Channel scores | Status |
| --- | --- | --- | --- |
| LooGLE / BM25 | 1832 | recomputed here (CPU) from the construction cache | to compute |
| MuSiQue-32K / BM25 | 900 | recomputed here (CPU) from the construction cache | to compute |
| QASPER-64K / BM25 | 1333 | recomputed here (CPU) from the construction cache | to compute |
| NovelHopQA / BM25 | 908 | recomputed here (CPU) from the construction cache | to compute |

Queries per dataset (saved rankings): LooGLE 1832, MuSiQue-32K 900, QASPER-64K 1333, NovelHopQA 908; total 4973 (E2: 4973).

## 2. BM25 channel scores (CPU)

| Dataset | Status | Queries | Time | Same as C2's BM25 streams |
| --- | --- | --- | --- | --- |
| LooGLE | computed | 1832 | 5 s | - |
| MuSiQue-32K | computed | 900 | 40 s | - |
| QASPER-64K | computed | 1333 | 84 s | - |
| NovelHopQA | computed | 908 | 60 s | - |

## 3. Check 1: all three reproduces the paper's saved rankings

| Pair | Queries scored | Identical top-10 | HR@5 ours / saved | HR@10 ours / saved | Hits = expected_metrics | Max abs score diff | Check 1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LooGLE / BM25 | 1832/1832 | 1832/1832 | 92.30 / 92.30 | 96.45 / 96.45 | yes | 8.9e-16 | PASS |
| MuSiQue-32K / BM25 | 900/900 | 900/900 | 62.78 / 62.78 | 77.67 / 77.67 | yes | 8.9e-16 | PASS |
| QASPER-64K / BM25 | 1333/1333 | 1333/1333 | 16.95 / 16.95 | 26.11 / 26.11 | yes | 8.9e-16 | PASS |
| NovelHopQA / BM25 | 908/908 | 908/908 | 18.06 / 18.06 | 22.03 / 22.03 | yes | 4.4e-16 | PASS |

Matching queries in total: 4973 of 4973 (4 of 4 pairs pass).

## 4. Check 2: text only against the paper's Vanilla

| Pair | Text only HR@5 / HR@10 | Table 1 Vanilla HR@5 / HR@10 | Equal to 1 decimal | Saved per-query text-only file identical | Same as C2 text-only |
| --- | --- | --- | --- | --- | --- |
| LooGLE / BM25 | 91.87 / 95.74 | 91.5 / 95.4 | NO | 1832/1832 | 1832/1832 |
| MuSiQue-32K / BM25 | 57.56 / 74.00 | 57.6 / 74.0 | yes | no file | 900/900 |
| QASPER-64K / BM25 | 14.40 / 22.06 | 14.4 / 22.1 | yes | no file | 1333/1333 |
| NovelHopQA / BM25 | 17.51 / 20.70 | 16.9 / 20.0 | NO | 908/908 | 908/908 |

Text only differs from Table 1's Vanilla on: LooGLE, NovelHopQA. Why: Table 1's Vanilla is a reported summary (`outputs/other_methods/vanilla/retrieval.csv`, source `main_fixed_depth` = `outputs/Tables/Final_table.txt`) with no per-query rankings. On LooGLE and NovelHopQA (500-token chunks) it comes from a separate run, not from the text channel of SAADI's run on the same chunks; where the saved per-query text-only rankings on SAADI's chunks exist (BM25, LooGLE and NovelHopQA: `retrieval_results__bm25__mbr_lt1_ln0_lh0_norm_none__per_document.json`), they equal C3's text only query by query (column 5). C3 uses the text channel on SAADI's own chunks as 'text only', so every comparison is paired on the same chunks; Table 1's Vanilla is shown next to it in the final report.

Table 1 reports LooGLE/Contriever Vanilla as 72.4 / 73.3 and NovelHopQA/Contriever as 11.8 / 11.8 (HR@5 about equal to HR@10), which suggests a transcription problem in that table row; it does not affect C3.

## 5. Chunks without a summary or memory score

Equation 2 renormalizes over the channels a chunk has. In 'summary alone' and 'memory alone' a chunk without that channel has no weight left, and the published code then falls back to its text score. Counts over (query, chunk) pairs:

| Pair | Median chunks per query | No summary score | No memory field | (query, chunk) pairs |
| --- | --- | --- | --- | --- |
| LooGLE / BM25 | 31.00 | 4286 | 2230 | 62471 |
| MuSiQue-32K / BM25 | 381.00 | 1726 | 2274 | 342176 |
| QASPER-64K / BM25 | 745.00 | 2790 | 5339 | 992410 |
| NovelHopQA / BM25 | 383.00 | 1536 | 2406 | 418611 |

## 6. Next steps

- Step 02 (BM25) can run on: LooGLE / BM25, MuSiQue-32K / BM25, QASPER-64K / BM25, NovelHopQA / BM25.
- Step 03 (dense) can run on: none.
- Not used: none.


## Files written

- `experiments/rebuttal/c3_channel_ablation/work/streams/loogle__bm25.parquet`
- `experiments/rebuttal/c3_channel_ablation/work/streams/loogle__bm25.queries.json`
- `experiments/rebuttal/c3_channel_ablation/work/streams/musique_32k__bm25.parquet`
- `experiments/rebuttal/c3_channel_ablation/work/streams/musique_32k__bm25.queries.json`
- `experiments/rebuttal/c3_channel_ablation/work/streams/qasper_64k__bm25.parquet`
- `experiments/rebuttal/c3_channel_ablation/work/streams/qasper_64k__bm25.queries.json`
- `experiments/rebuttal/c3_channel_ablation/work/streams/novelhopqa__bm25.parquet`
- `experiments/rebuttal/c3_channel_ablation/work/streams/novelhopqa__bm25.queries.json`
- `experiments/rebuttal/c3_channel_ablation/work/ready_pairs.json`
- `experiments/rebuttal/c3_channel_ablation/reports/step_01.md`
- `experiments/rebuttal/c3_channel_ablation/reports/step_01.json`
