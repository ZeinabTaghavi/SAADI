# C3 channel ablation - step 01: inputs and checks

**Status: FAILED**

- Command: `C3_STREAMS_DIR=/home/iataghav/data/SAADI/outputs/rebuttal/C2_heldout_fusion/streams bash experiments/rebuttal/c3_channel_ablation/sh/01_check.sh`
- Started: 2026-11-09 17:19:10; took 1 s
- Host: eee-s132-infsys-gpu03; Python 3.12.3; commit 1e7b1fb6b31c (fix/notebook-key-and-install, 2 changed files)

## Settings

- Config: `configs/methods/sadi.yaml` (block `cached_retrieval`). Fusion weights (lambda_T, lambda_M, lambda_S) = (1, 0.5, 0.3); field weights (mu_N, mu_A, mu_D) = (1, 1, 1); normalization zscore; memory aggregation max.
- A variant drops a channel by setting its weight to 0 in the published Equation 2 (`combine_mbr_variant_chunk_scores`, MBR2). The remaining weights are not re-tuned or rescaled; Equation 2's own per-chunk division by the sum of the weights present is unchanged.
- Dense channel scores read from `C3_STREAMS_DIR` = `outputs/rebuttal/C2_heldout_fusion/streams`; C2 markers from `outputs/rebuttal/C2_heldout_fusion/markers`.
- BM25 channel scores written to `experiments/rebuttal/c3_channel_ablation/work/streams`.
- Datasets: loogle, musique_32k, qasper_64k, novelhopqa. Retrievers: contriever, bge_m3.

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
| LooGLE / Contriever | 1832 | read from outputs/rebuttal/C2_heldout_fusion/streams | channel scores found |
| LooGLE / BGE-M3 | 1832 | read from outputs/rebuttal/C2_heldout_fusion/streams | channel scores found |
| MuSiQue-32K / Contriever | 900 | read from outputs/rebuttal/C2_heldout_fusion/streams | channel scores found |
| MuSiQue-32K / BGE-M3 | 900 | read from outputs/rebuttal/C2_heldout_fusion/streams | channel scores found |
| QASPER-64K / Contriever | 1333 | read from outputs/rebuttal/C2_heldout_fusion/streams | channel scores found |
| QASPER-64K / BGE-M3 | 1333 | read from outputs/rebuttal/C2_heldout_fusion/streams | channel scores found |
| NovelHopQA / Contriever | 908 | read from outputs/rebuttal/C2_heldout_fusion/streams | channel scores found |
| NovelHopQA / BGE-M3 | 908 | read from outputs/rebuttal/C2_heldout_fusion/streams | channel scores found |

Queries per dataset (saved rankings): LooGLE 1832, MuSiQue-32K 900, QASPER-64K 1333, NovelHopQA 908; total 4973 (E2: 4973).

## 3. Check 1: all three reproduces the paper's saved rankings

| Pair | Queries scored | Identical top-10 | HR@5 ours / saved | HR@10 ours / saved | Hits = expected_metrics | Max abs score diff | Check 1 |
| --- | --- | --- | --- | --- | --- | --- | --- |

Matching queries in total: 0 of 0 (0 of 0 pairs pass).

### loogle__contriever: exception

```
Traceback (most recent call last):
  File "/home/iataghav/data/SAADI/experiments/rebuttal/c3_channel_ablation/scripts/step01_check.py", line 430, in body
    res = check_pair(d, r, w)
          ^^^^^^^^^^^^^^^^^^^
  File "/home/iataghav/data/SAADI/experiments/rebuttal/c3_channel_ablation/scripts/step01_check.py", line 192, in check_pair
    rows = saved_rows_for(d, r, meta)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/SAADI/experiments/rebuttal/c3_channel_ablation/scripts/common.py", line 291, in saved_rows_for
    return queries, meta, rows
           ^^^^^^^
NameError: name 'queries' is not defined
```

### loogle__bge_m3: exception

```
Traceback (most recent call last):
  File "/home/iataghav/data/SAADI/experiments/rebuttal/c3_channel_ablation/scripts/step01_check.py", line 430, in body
    res = check_pair(d, r, w)
          ^^^^^^^^^^^^^^^^^^^
  File "/home/iataghav/data/SAADI/experiments/rebuttal/c3_channel_ablation/scripts/step01_check.py", line 192, in check_pair
    rows = saved_rows_for(d, r, meta)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/SAADI/experiments/rebuttal/c3_channel_ablation/scripts/common.py", line 291, in saved_rows_for
    return queries, meta, rows
           ^^^^^^^
NameError: name 'queries' is not defined
```

### musique_32k__contriever: exception

```
Traceback (most recent call last):
  File "/home/iataghav/data/SAADI/experiments/rebuttal/c3_channel_ablation/scripts/step01_check.py", line 430, in body
    res = check_pair(d, r, w)
          ^^^^^^^^^^^^^^^^^^^
  File "/home/iataghav/data/SAADI/experiments/rebuttal/c3_channel_ablation/scripts/step01_check.py", line 192, in check_pair
    rows = saved_rows_for(d, r, meta)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/SAADI/experiments/rebuttal/c3_channel_ablation/scripts/common.py", line 291, in saved_rows_for
    return queries, meta, rows
           ^^^^^^^
NameError: name 'queries' is not defined
```

### musique_32k__bge_m3: exception

```
Traceback (most recent call last):
  File "/home/iataghav/data/SAADI/experiments/rebuttal/c3_channel_ablation/scripts/step01_check.py", line 430, in body
    res = check_pair(d, r, w)
          ^^^^^^^^^^^^^^^^^^^
  File "/home/iataghav/data/SAADI/experiments/rebuttal/c3_channel_ablation/scripts/step01_check.py", line 192, in check_pair
    rows = saved_rows_for(d, r, meta)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/SAADI/experiments/rebuttal/c3_channel_ablation/scripts/common.py", line 291, in saved_rows_for
    return queries, meta, rows
           ^^^^^^^
NameError: name 'queries' is not defined
```

### qasper_64k__contriever: exception

```
Traceback (most recent call last):
  File "/home/iataghav/data/SAADI/experiments/rebuttal/c3_channel_ablation/scripts/step01_check.py", line 430, in body
    res = check_pair(d, r, w)
          ^^^^^^^^^^^^^^^^^^^
  File "/home/iataghav/data/SAADI/experiments/rebuttal/c3_channel_ablation/scripts/step01_check.py", line 192, in check_pair
    rows = saved_rows_for(d, r, meta)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/SAADI/experiments/rebuttal/c3_channel_ablation/scripts/common.py", line 291, in saved_rows_for
    return queries, meta, rows
           ^^^^^^^
NameError: name 'queries' is not defined
```

### qasper_64k__bge_m3: exception

```
Traceback (most recent call last):
  File "/home/iataghav/data/SAADI/experiments/rebuttal/c3_channel_ablation/scripts/step01_check.py", line 430, in body
    res = check_pair(d, r, w)
          ^^^^^^^^^^^^^^^^^^^
  File "/home/iataghav/data/SAADI/experiments/rebuttal/c3_channel_ablation/scripts/step01_check.py", line 192, in check_pair
    rows = saved_rows_for(d, r, meta)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/SAADI/experiments/rebuttal/c3_channel_ablation/scripts/common.py", line 291, in saved_rows_for
    return queries, meta, rows
           ^^^^^^^
NameError: name 'queries' is not defined
```

### novelhopqa__contriever: exception

```
Traceback (most recent call last):
  File "/home/iataghav/data/SAADI/experiments/rebuttal/c3_channel_ablation/scripts/step01_check.py", line 430, in body
    res = check_pair(d, r, w)
          ^^^^^^^^^^^^^^^^^^^
  File "/home/iataghav/data/SAADI/experiments/rebuttal/c3_channel_ablation/scripts/step01_check.py", line 192, in check_pair
    rows = saved_rows_for(d, r, meta)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/SAADI/experiments/rebuttal/c3_channel_ablation/scripts/common.py", line 291, in saved_rows_for
    return queries, meta, rows
           ^^^^^^^
NameError: name 'queries' is not defined
```

### novelhopqa__bge_m3: exception

```
Traceback (most recent call last):
  File "/home/iataghav/data/SAADI/experiments/rebuttal/c3_channel_ablation/scripts/step01_check.py", line 430, in body
    res = check_pair(d, r, w)
          ^^^^^^^^^^^^^^^^^^^
  File "/home/iataghav/data/SAADI/experiments/rebuttal/c3_channel_ablation/scripts/step01_check.py", line 192, in check_pair
    rows = saved_rows_for(d, r, meta)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/SAADI/experiments/rebuttal/c3_channel_ablation/scripts/common.py", line 291, in saved_rows_for
    return queries, meta, rows
           ^^^^^^^
NameError: name 'queries' is not defined
```

## 4. Check 2: text only against the paper's Vanilla

| Pair | Text only HR@5 / HR@10 | Table 1 Vanilla HR@5 / HR@10 | Equal to 1 decimal | Saved per-query text-only file identical |
| --- | --- | --- | --- | --- |

Text only equals Table 1's Vanilla (to one decimal) on every pair checked.

## 5. Chunks without a summary or memory score

Equation 2 renormalizes over the channels a chunk has. In 'summary alone' and 'memory alone' a chunk without that channel has no weight left, and the published code then falls back to its text score. Counts over (query, chunk) pairs:

| Pair | Median chunks per query | No summary score | No memory field | (query, chunk) pairs |
| --- | --- | --- | --- | --- |

## 6. Next steps

- Step 02 (BM25) can run on: none.
- Step 03 (dense) can run on: none.
- Not used: LooGLE / Contriever (check 1 failed or exception); LooGLE / BGE-M3 (check 1 failed or exception); MuSiQue-32K / Contriever (check 1 failed or exception); MuSiQue-32K / BGE-M3 (check 1 failed or exception); QASPER-64K / Contriever (check 1 failed or exception); QASPER-64K / BGE-M3 (check 1 failed or exception); NovelHopQA / Contriever (check 1 failed or exception); NovelHopQA / BGE-M3 (check 1 failed or exception).


## Files written

- `experiments/rebuttal/c3_channel_ablation/work/ready_pairs.json`
- `experiments/rebuttal/c3_channel_ablation/reports/step_01.md`
- `experiments/rebuttal/c3_channel_ablation/reports/step_01.json`
