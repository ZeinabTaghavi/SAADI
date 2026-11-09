# C2 held-out fusion weights: key numbers

- Setup: 12 (dataset, retriever) pairs (Retrievers: BM25, Contriever, BGE-M3; not included in this analysis: Jina, Qwen); all weights re-chosen on a 1296-point grid without the evaluated queries (5-fold CV grouped by document, seed 42).
- Mean HR@10 over pairs: Vanilla 53.8, SAADI published 55.6, SAADI held-out (A1) 56.2, channels-only held-out (A2) 56.4.
- Mean HR@5 over pairs: Vanilla 44.8, published 46.6, A1 47.0, A2 46.9.
- A1 above Vanilla: 11/12 pairs at HR@10 (6 with the 95% CI above zero; 6 with the document-cluster CI); 10/12 at HR@5 (5 CI above zero).
- Published weights for comparison: CI above zero in 5/12 (HR@10) and 4/12 (HR@5).
- A1 minus published, HR@10: mean +0.65 points (range -0.88 to +2.44); significantly below published in 1/12 pairs.
- Table 1 ranking: SAADI first or second in 23/24 cells with held-out weights (published: 21/24).
- Transfer from the other three datasets: above Vanilla at HR@10 in 9/12 pairs per retriever, 8/12 with one global setting.
- Stability: mean number of distinct points chosen across the outer folds 3.1; folds choosing exactly the published point: 0/60; mean L1 distance to the published point 1.50.
- Sensitivity: median share of the 81 channel settings within 1 HR@10 point of the best: 0.19; median HR@10 gap best minus published 1.10 points.
- Learned pairwise logistic fusion (demonstration): mean HR@10 56.5 vs A1 56.2.
- Original 100-query tuning sample: drawn with an unseeded random.sample; not reconstructable from the code or saved outputs, so it is not excluded separately (A and B do not depend on it).
- Full tables, intervals and definitions: results/rebuttal/C2_heldout_fusion/REPORT.md.
