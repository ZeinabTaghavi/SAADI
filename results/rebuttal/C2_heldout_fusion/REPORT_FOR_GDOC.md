# C2 held-out fusion weights: key numbers

- Setup: 20 (dataset, retriever) pairs (Retrievers: BM25, Contriever, BGE-M3, Jina, Qwen); all weights re-chosen on a 1296-point grid without the evaluated queries (5-fold CV grouped by document, seed 42).
- Mean HR@10 over pairs: Vanilla 54.3, SAADI published 55.8, SAADI held-out (A1) 56.1, channels-only held-out (A2) 56.4.
- Mean HR@5 over pairs: Vanilla 45.2, published 46.4, A1 46.6, A2 46.6.
- A1 above Vanilla: 18/20 pairs at HR@10 (10 with the 95% CI above zero; 9 with the document-cluster CI); 14/20 at HR@5 (6 CI above zero).
- Published weights for comparison: CI above zero in 8/20 (HR@10) and 6/20 (HR@5).
- A1 minus published, HR@10: mean +0.32 points (range -0.88 to +2.44); significantly below published in 1/20 pairs.
- Table 1 ranking: SAADI first or second in 37/40 cells with held-out weights (published: 34/40).
- Transfer from the other three datasets: above Vanilla at HR@10 in 15/20 pairs per retriever, 16/20 with one global setting.
- Stability: mean number of distinct points chosen across the outer folds 3.4; folds choosing exactly the published point: 0/100; mean L1 distance to the published point 1.51.
- Sensitivity: median share of the 81 channel settings within 1 HR@10 point of the best: 0.26; median HR@10 gap best minus published 1.00 points.
- Full grid (1296 points, mean HR@10 over the 20 pairs): published point rank 36 (top 2.8%), 55.82; best single point (0.3, 0.5 | 0.3, 1) 56.02 (+0.20), chosen with all queries in hand (upper bound, not held-out).
- A1-selected weights over all outer folds, median (non-zero folds): lambda_M 0.3 (95/100, 95%); lambda_S 0.3 (83/100, 83%); mu_A 0.3 (67/100, 67%); mu_D 1 (86/100, 86%).
- Mean HR@10 per dataset, published -> A1: LooGLE 88.13 -> 88.24 (+0.11); MuSiQue-32K 85.56 -> 86.62 (+1.07); QASPER-64K 28.81 -> 29.30 (+0.50); NovelHopQA 20.77 -> 20.40 (-0.37).
- Learned pairwise logistic fusion (demonstration): mean HR@10 56.5 vs A1 56.1.
- Embedding drift: loogle__jina, loogle__qwen, musique_32k__jina, qasper_64k__jina, novelhopqa__jina recomputed with scores about 1e-3 off the paper's run (HR within 0.5 points); all their systems use the same recomputed scores.
- Original 100-query tuning sample: drawn with an unseeded random.sample; not reconstructable from the code or saved outputs, so it is not excluded separately (A and B do not depend on it).
- Full tables, intervals and definitions: results/rebuttal/C2_heldout_fusion/REPORT.md.
