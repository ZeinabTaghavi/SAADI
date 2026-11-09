# C2 held-out fusion weights: key numbers (DRY RUN)

- Setup: 4 (dataset, retriever) pairs; all weights re-chosen on a 64-point grid without the evaluated queries (2-fold CV grouped by document, seed 42).
- Mean HR@10 over pairs: Vanilla 54.3, SAADI published 58.0, SAADI held-out (A1) 55.5, channels-only held-out (A2) 57.3.
- Mean HR@5 over pairs: Vanilla 46.5, published 50.8, A1 48.3, A2 49.3.
- A1 above Vanilla: 2/4 pairs at HR@10 (0 with the 95% CI above zero; 1 with the document-cluster CI); 2/4 at HR@5 (1 CI above zero).
- Published weights for comparison: CI above zero in 1/4 (HR@10) and 1/4 (HR@5).
- A1 minus published, HR@10: mean -2.54 points (range -7.14 to +0.00); significantly below published in 0/4 pairs.
- Table 1 ranking: SAADI first or second in 32/40 cells with held-out weights (published: 34/40).
- Transfer from the other three datasets: above Vanilla at HR@10 in 3/4 pairs per retriever, 3/4 with one global setting.
- Stability: mean number of distinct points chosen across the outer folds 2.0; folds choosing exactly the published point: 0/8; mean L1 distance to the published point 1.49.
- Sensitivity: median share of the 16 channel settings within 1 HR@10 point of the best: 0.47; median HR@10 gap best minus published 0.78 points.
- Learned pairwise logistic fusion (demonstration): mean HR@10 57.4 vs A1 55.5.
- Original 100-query tuning sample: drawn with an unseeded random.sample; not reconstructable from the code or saved outputs, so it is not excluded separately (A and B do not depend on it).
- Full tables, intervals and definitions: results/rebuttal/C2_heldout_fusion/REPORT.md.
