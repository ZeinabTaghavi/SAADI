# C4 sub-document control - step 03: BM25 ranks inside the gold sub-document (OK)

Command (from the repository root on the server):

```bash
bash experiments/rebuttal/c4_subdoc_control/sh/03_bm25.sh
```

Seed 42. CPU only. Run time 11 s.
Pairs measured: 8. Main query set: R (set with `C4_MAIN_SET`; both sets are always computed). Vanilla = the text channel alone; SAADI = the published fusion (lambda_M=0.5, lambda_S=0.3, mu=(1.0, 1.0, 1.0), from `configs/methods/sadi.yaml`).

## 1. Query sets

| Setting | Retriever | Eligible (paper) | Set S | Set R | S with > 10 chunks | R with > 10 chunks | S: inside relabel != paper labels | Chunk-count mismatches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MuSiQue-32K | BM25 | 900 | 575 | 900 | 569 | 894 | 0 | 0 |
| MuSiQue-64K | BM25 | 900 | 385 | 900 | 380 | 894 | 0 | 0 |
| MuSiQue-128K | BM25 | 900 | 250 | 900 | 245 | 894 | 0 | 0 |
| MuSiQue-256K | BM25 | 900 | 175 | 900 | 170 | 894 | 0 | 0 |
| QASPER-32K | BM25 | 1321 | 979 | 1321 | 972 | 1314 | 0 | 0 |
| QASPER-64K | BM25 | 1333 | 794 | 1322 | 789 | 1315 | 0 | 0 |
| QASPER-128K | BM25 | 1321 | 660 | 1321 | 655 | 1314 | 0 | 0 |
| QASPER-256K | BM25 | 1321 | 505 | 1321 | 500 | 1314 | 0 | 0 |


## 2a. Set R: full ranking (A), sub-document (C), inside, keep scores (B)

| Setting | Retriever | Method | n | Full HR@5 | Full HR@10 | Full median rank | Sub-doc hit@10 | Top-10 share from gold sub-doc | In HR@1 | In HR@3 | In HR@5 | In HR@10 | In HR@10 (>10 chunks) | In MRR | In median rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MuSiQue-32K | BM25 | Vanilla | 900 | 57.6 | 74 | 4 | 99.9 | 80.0 | 23 | 48 | 63.8 | 82.3 | 82.2 (n=894) | 40.9 | 4 |
| MuSiQue-32K | BM25 | SAADI | 900 | 62.8 | 77.7 | 4 | 99.9 | 84.4 | 27.8 | 53.1 | 67 | 84.3 | 84.2 (n=894) | 45.3 | 3 |
| MuSiQue-64K | BM25 | Vanilla | 900 | 54 | 69.2 | 5 | 99.8 | 71.7 | 23.7 | 48.7 | 63.2 | 81.3 | 81.2 (n=894) | 41.6 | 4 |
| MuSiQue-64K | BM25 | SAADI | 900 | 59.2 | 74.3 | 4 | 99.6 | 76.2 | 28.1 | 54.3 | 67.3 | 84.1 | 84.0 (n=894) | 45.7 | 3 |
| MuSiQue-128K | BM25 | Vanilla | 900 | 51.6 | 64.9 | 5 | 99.4 | 61.2 | 24.4 | 49.7 | 63.6 | 81.7 | 81.5 (n=894) | 42.4 | 4 |
| MuSiQue-128K | BM25 | SAADI | 900 | 55.7 | 68.8 | 5 | 99 | 66.3 | 28.2 | 53.9 | 66.9 | 83.9 | 83.8 (n=894) | 46.0 | 3 |
| MuSiQue-256K | BM25 | Vanilla | 900 | 48.4 | 60.6 | 6 | 98.6 | 51.0 | 25.2 | 50.2 | 63.6 | 81.3 | 81.2 (n=894) | 42.9 | 3 |
| MuSiQue-256K | BM25 | SAADI | 900 | 52.3 | 64.3 | 5 | 98.7 | 55.8 | 29.2 | 55.6 | 66.9 | 83.4 | 83.3 (n=894) | 47.0 | 3 |
| QASPER-32K | BM25 | Vanilla | 1321 | 17.9 | 27.6 | 39 | 85.6 | 40.0 | 8.2 | 24.0 | 36.4 | 56.0 | 55.8 (n=1314) | 22.5 | 9 |
| QASPER-32K | BM25 | SAADI | 1321 | 19.6 | 32.1 | 28 | 86.0 | 46.7 | 8.9 | 24.7 | 36.7 | 59.7 | 59.5 (n=1314) | 23.4 | 8 |
| QASPER-64K | BM25 | Vanilla | 1322 | 14.4 | 22.0 | 77 | 76.3 | 34.0 | 8.5 | 23.9 | 35.4 | 56.5 | 56.3 (n=1315) | 22.7 | 9 |
| QASPER-64K | BM25 | SAADI | 1322 | 17.0 | 26.2 | 52 | 75.5 | 39.2 | 9.2 | 25.4 | 36.4 | 59.1 | 58.9 (n=1315) | 24.0 | 8 |
| QASPER-128K | BM25 | Vanilla | 1321 | 12.8 | 18.5 | 169 | 68.2 | 29.3 | 8.9 | 24.5 | 35.9 | 56.5 | 56.2 (n=1314) | 23.0 | 9 |
| QASPER-128K | BM25 | SAADI | 1321 | 15.0 | 22.6 | 117 | 68.4 | 34.4 | 9.0 | 24.7 | 37.6 | 59.3 | 59.1 (n=1314) | 23.7 | 8 |
| QASPER-256K | BM25 | Vanilla | 1321 | 11.5 | 15.7 | 408 | 60.5 | 25.3 | 8.9 | 24.9 | 35.9 | 56.9 | 56.6 (n=1314) | 23.1 | 9 |
| QASPER-256K | BM25 | SAADI | 1321 | 12.2 | 18.4 | 284 | 61.9 | 29.7 | 9.4 | 25.4 | 37.3 | 60.0 | 59.7 (n=1314) | 24.2 | 8 |


## 2b. Set S: full ranking (A), sub-document (C), inside, keep scores (B)

| Setting | Retriever | Method | n | Full HR@5 | Full HR@10 | Full median rank | Sub-doc hit@10 | Top-10 share from gold sub-doc | In HR@1 | In HR@3 | In HR@5 | In HR@10 | In HR@10 (>10 chunks) | In MRR | In median rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MuSiQue-32K | BM25 | Vanilla | 575 | 61.4 | 78.1 | 4 | 99.8 | 84.9 | 24.7 | 50.3 | 64.5 | 83.3 | 83.1 (n=569) | 42.3 | 3 |
| MuSiQue-32K | BM25 | SAADI | 575 | 66.1 | 81.9 | 3 | 99.8 | 89.4 | 29.4 | 53.7 | 68.2 | 84.7 | 84.5 (n=569) | 46.6 | 3 |
| MuSiQue-64K | BM25 | Vanilla | 385 | 58.7 | 74.3 | 4 | 99.7 | 79.5 | 27.3 | 50.4 | 64.4 | 81.8 | 81.6 (n=380) | 43.7 | 3 |
| MuSiQue-64K | BM25 | SAADI | 385 | 64.9 | 82.1 | 3 | 99.5 | 84.6 | 33.5 | 55.8 | 68.3 | 85.5 | 85.3 (n=380) | 49.4 | 3 |
| MuSiQue-128K | BM25 | Vanilla | 250 | 60.4 | 74.8 | 3.5 | 99.6 | 73.0 | 31.2 | 54 | 65.2 | 84.4 | 84.1 (n=245) | 47.4 | 3 |
| MuSiQue-128K | BM25 | SAADI | 250 | 66.8 | 81.6 | 2.5 | 99.6 | 79.5 | 38 | 60.4 | 71.6 | 88 | 87.8 (n=245) | 53.6 | 2 |
| MuSiQue-256K | BM25 | Vanilla | 175 | 57.7 | 73.1 | 3 | 98.9 | 68.9 | 35.4 | 54.9 | 64 | 84 | 83.5 (n=170) | 49.7 | 3 |
| MuSiQue-256K | BM25 | SAADI | 175 | 62.9 | 83.4 | 2 | 98.3 | 74.9 | 40 | 59.4 | 70.3 | 88.6 | 88.2 (n=170) | 54.8 | 2 |
| QASPER-32K | BM25 | Vanilla | 979 | 18.0 | 27.8 | 36 | 85.0 | 39.7 | 8.5 | 23.5 | 36.6 | 55.2 | 54.8 (n=972) | 22.4 | 9 |
| QASPER-32K | BM25 | SAADI | 979 | 20.3 | 33.4 | 26 | 85.0 | 46.5 | 8.6 | 24.1 | 36.5 | 59.2 | 59.0 (n=972) | 23.0 | 8 |
| QASPER-64K | BM25 | Vanilla | 794 | 15.6 | 23.7 | 57 | 74.8 | 33.0 | 8.7 | 25.1 | 36.6 | 55.8 | 55.5 (n=789) | 23.0 | 9 |
| QASPER-64K | BM25 | SAADI | 794 | 17.6 | 27.6 | 39.5 | 74.7 | 37.9 | 9.8 | 26.6 | 37.7 | 59.4 | 59.2 (n=789) | 24.7 | 8 |
| QASPER-128K | BM25 | Vanilla | 660 | 15.3 | 21.4 | 89 | 65.2 | 28.1 | 10.6 | 28.6 | 40.3 | 59.2 | 58.9 (n=655) | 25.4 | 8 |
| QASPER-128K | BM25 | SAADI | 660 | 18.3 | 26.1 | 60.5 | 66.7 | 32.6 | 10.5 | 28.8 | 39.8 | 61.4 | 61.1 (n=655) | 25.6 | 7 |
| QASPER-256K | BM25 | Vanilla | 505 | 14.5 | 19.6 | 160 | 57.4 | 23.4 | 12.3 | 30.5 | 41.8 | 60 | 59.6 (n=500) | 26.8 | 8 |
| QASPER-256K | BM25 | SAADI | 505 | 15.0 | 23.6 | 111 | 57.4 | 27.6 | 12.7 | 31.1 | 42.4 | 62.4 | 62 (n=500) | 27.9 | 7 |


## 3. Set R: HR@k on queries whose gold sub-document has more than k chunks (keep / renorm)

| Setting | Retriever | Method | HR@1 (>1) | HR@3 (>3) | HR@5 (>5) | HR@10 (>10) | MRR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MuSiQue-32K | BM25 | Vanilla | 23 / 23 (n=900) | 48 / 48 (n=900) | 63.8 / 63.8 (n=900) | 82.2 / 82.2 (n=894) | 40.9 / 40.9 |
| MuSiQue-32K | BM25 | SAADI | 27.8 / 28.4 (n=900) | 53.1 / 54.7 (n=900) | 67 / 66.6 (n=900) | 84.2 / 84.3 (n=894) | 45.3 / 46.0 |
| MuSiQue-64K | BM25 | Vanilla | 23.7 / 23.7 (n=900) | 48.7 / 48.7 (n=900) | 63.2 / 63.2 (n=900) | 81.2 / 81.2 (n=894) | 41.6 / 41.6 |
| MuSiQue-64K | BM25 | SAADI | 28.1 / 28.7 (n=900) | 54.3 / 54.8 (n=900) | 67.3 / 66.6 (n=900) | 84.0 / 83.8 (n=894) | 45.7 / 46.1 |
| MuSiQue-128K | BM25 | Vanilla | 24.4 / 24.4 (n=900) | 49.7 / 49.7 (n=900) | 63.6 / 63.6 (n=900) | 81.5 / 81.5 (n=894) | 42.4 / 42.4 |
| MuSiQue-128K | BM25 | SAADI | 28.2 / 29.4 (n=900) | 53.9 / 55.2 (n=900) | 66.9 / 66.2 (n=900) | 83.8 / 84.0 (n=894) | 46.0 / 46.9 |
| MuSiQue-256K | BM25 | Vanilla | 25.2 / 25.2 (n=900) | 50.2 / 50.2 (n=900) | 63.6 / 63.6 (n=900) | 81.2 / 81.2 (n=894) | 42.9 / 42.9 |
| MuSiQue-256K | BM25 | SAADI | 29.2 / 29.6 (n=900) | 55.6 / 56.6 (n=900) | 66.9 / 67.6 (n=900) | 83.3 / 83.0 (n=894) | 47.0 / 47.2 |
| QASPER-32K | BM25 | Vanilla | 8.2 / 8.2 (n=1321) | 24.0 / 24.0 (n=1321) | 36.4 / 36.4 (n=1321) | 55.8 / 55.8 (n=1314) | 22.5 / 22.5 |
| QASPER-32K | BM25 | SAADI | 8.9 / 9.2 (n=1321) | 24.7 / 24.7 (n=1321) | 36.7 / 36.8 (n=1321) | 59.5 / 59.6 (n=1314) | 23.4 / 23.5 |
| QASPER-64K | BM25 | Vanilla | 8.5 / 8.5 (n=1322) | 23.9 / 23.9 (n=1322) | 35.4 / 35.4 (n=1322) | 56.3 / 56.3 (n=1315) | 22.7 / 22.7 |
| QASPER-64K | BM25 | SAADI | 9.2 / 9.3 (n=1322) | 25.4 / 25.5 (n=1322) | 36.4 / 36.8 (n=1322) | 58.9 / 58.9 (n=1315) | 24.0 / 24.1 |
| QASPER-128K | BM25 | Vanilla | 8.9 / 8.9 (n=1321) | 24.5 / 24.5 (n=1321) | 35.9 / 35.9 (n=1321) | 56.2 / 56.2 (n=1314) | 23.0 / 23.0 |
| QASPER-128K | BM25 | SAADI | 9.0 / 9.4 (n=1321) | 24.7 / 25.2 (n=1321) | 37.6 / 38.8 (n=1321) | 59.1 / 60.3 (n=1314) | 23.7 / 24.1 |
| QASPER-256K | BM25 | Vanilla | 8.9 / 8.9 (n=1321) | 24.9 / 24.9 (n=1321) | 35.9 / 35.9 (n=1321) | 56.6 / 56.6 (n=1314) | 23.1 / 23.1 |
| QASPER-256K | BM25 | SAADI | 9.4 / 10.1 (n=1321) | 25.4 / 25.7 (n=1321) | 37.3 / 37.9 (n=1321) | 59.7 / 59.9 (n=1314) | 24.2 / 24.7 |

keep = full-document scores, chunks outside the gold sub-document dropped (main). renorm = every channel z-scored again over the gold sub-document's chunks only. Vanilla's ranking is the same under both (z-scoring one channel does not change its order). All values in percent except ranks (1-based). The full-ranking columns use the paper's labels on the whole document.

## Files written

- `experiments/rebuttal/c4_subdoc_control/reports/step_03.json`
- `experiments/rebuttal/c4_subdoc_control/reports/step_03.md`
- `experiments/rebuttal/c4_subdoc_control/work/perquery/musique_128k__bm25.csv.gz`
- `experiments/rebuttal/c4_subdoc_control/work/perquery/musique_256k__bm25.csv.gz`
- `experiments/rebuttal/c4_subdoc_control/work/perquery/musique_32k__bm25.csv.gz`
- `experiments/rebuttal/c4_subdoc_control/work/perquery/musique_64k__bm25.csv.gz`
- `experiments/rebuttal/c4_subdoc_control/work/perquery/qasper_128k__bm25.csv.gz`
- `experiments/rebuttal/c4_subdoc_control/work/perquery/qasper_256k__bm25.csv.gz`
- `experiments/rebuttal/c4_subdoc_control/work/perquery/qasper_32k__bm25.csv.gz`
- `experiments/rebuttal/c4_subdoc_control/work/perquery/qasper_64k__bm25.csv.gz`
