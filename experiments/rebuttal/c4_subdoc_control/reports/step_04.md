# C4 sub-document control - step 04: dense-retriever ranks inside the gold sub-document (OK)

Command (from the repository root on the server):

```bash
bash experiments/rebuttal/c4_subdoc_control/sh/04_dense.sh
```

Seed 42. CPU only. Run time 11 s.
Pairs measured: 8. Main query set: R (set with `C4_MAIN_SET`; both sets are always computed). Vanilla = the text channel alone; SAADI = the published fusion (lambda_M=0.5, lambda_S=0.3, mu=(1.0, 1.0, 1.0), from `configs/methods/sadi.yaml`).

## Pairs not measured

| Setting | Retriever | Reason |
| --- | --- | --- |
| MuSiQue-32K | bge_m3 | waiting for scores: `outputs/rebuttal/C2_heldout_fusion/streams/musique_32k__bge_m3.parquet`, `outputs/rebuttal/C2_heldout_fusion/streams/musique_32k__bge_m3.queries.json` |
| MuSiQue-32K | jina | waiting for scores: `outputs/rebuttal/C2_heldout_fusion/streams/musique_32k__jina.parquet`, `outputs/rebuttal/C2_heldout_fusion/streams/musique_32k__jina.queries.json` |
| MuSiQue-32K | qwen | waiting for scores: `outputs/rebuttal/C2_heldout_fusion/streams/musique_32k__qwen.parquet`, `outputs/rebuttal/C2_heldout_fusion/streams/musique_32k__qwen.queries.json` |
| QASPER-64K | bge_m3 | waiting for scores: `outputs/rebuttal/C2_heldout_fusion/streams/qasper_64k__bge_m3.parquet`, `outputs/rebuttal/C2_heldout_fusion/streams/qasper_64k__bge_m3.queries.json` |
| QASPER-64K | jina | waiting for scores: `outputs/rebuttal/C2_heldout_fusion/streams/qasper_64k__jina.parquet`, `outputs/rebuttal/C2_heldout_fusion/streams/qasper_64k__jina.queries.json` |
| QASPER-64K | qwen | waiting for scores: `outputs/rebuttal/C2_heldout_fusion/streams/qasper_64k__qwen.parquet`, `outputs/rebuttal/C2_heldout_fusion/streams/qasper_64k__qwen.queries.json` |


## 1. Query sets

| Setting | Retriever | Eligible (paper) | Set S | Set R | S with > 10 chunks | R with > 10 chunks | S: inside relabel != paper labels | Chunk-count mismatches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MuSiQue-32K | Contriever | 900 | 575 | 900 | 569 | 894 | 0 | 0 |
| MuSiQue-64K | Contriever | 900 | 385 | 900 | 380 | 894 | 0 | 0 |
| MuSiQue-128K | Contriever | 900 | 250 | 900 | 245 | 894 | 0 | 0 |
| MuSiQue-256K | Contriever | 900 | 175 | 900 | 170 | 894 | 0 | 0 |
| QASPER-32K | Contriever | 1321 | 979 | 1321 | 972 | 1314 | 0 | 0 |
| QASPER-64K | Contriever | 1333 | 794 | 1322 | 789 | 1315 | 0 | 0 |
| QASPER-128K | Contriever | 1321 | 660 | 1321 | 655 | 1314 | 0 | 0 |
| QASPER-256K | Contriever | 1321 | 505 | 1321 | 500 | 1314 | 0 | 0 |


## 2a. Set R: full ranking (A), sub-document (C), inside, keep scores (B)

| Setting | Retriever | Method | n | Full HR@5 | Full HR@10 | Full median rank | Sub-doc hit@10 | Top-10 share from gold sub-doc | In HR@1 | In HR@3 | In HR@5 | In HR@10 | In HR@10 (>10 chunks) | In MRR | In median rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MuSiQue-32K | Contriever | Vanilla | 900 | 59.9 | 76.1 | 4 | 99.6 | 66.5 | 25.1 | 56.8 | 71.3 | 91 | 90.9 (n=894) | 45.6 | 3 |
| MuSiQue-32K | Contriever | SAADI | 900 | 68.8 | 83.7 | 3 | 99.8 | 75.2 | 29.7 | 61.8 | 77.8 | 93.2 | 93.2 (n=894) | 50.1 | 2 |
| MuSiQue-64K | Contriever | Vanilla | 900 | 54 | 68.3 | 5 | 99.1 | 56.2 | 24.9 | 56.7 | 71.2 | 90.8 | 90.7 (n=894) | 45.4 | 3 |
| MuSiQue-64K | Contriever | SAADI | 900 | 60.9 | 77.4 | 3 | 99.4 | 64.6 | 29.8 | 61 | 77.7 | 92.8 | 92.7 (n=894) | 50.2 | 2 |
| MuSiQue-128K | Contriever | Vanilla | 900 | 47.9 | 60.7 | 6 | 98.2 | 44.7 | 25.1 | 56.8 | 71.4 | 90.8 | 90.7 (n=894) | 45.6 | 3 |
| MuSiQue-128K | Contriever | SAADI | 900 | 56.3 | 68.4 | 4 | 98.8 | 53.0 | 28.7 | 62.1 | 76.4 | 92.7 | 92.6 (n=894) | 49.5 | 2 |
| MuSiQue-256K | Contriever | Vanilla | 900 | 44.3 | 54 | 8 | 96.2 | 34.6 | 24.9 | 56.6 | 71.2 | 90.8 | 90.7 (n=894) | 45.4 | 3 |
| MuSiQue-256K | Contriever | SAADI | 900 | 53.6 | 64 | 5 | 97.3 | 42.4 | 30.1 | 61.7 | 76.6 | 92.7 | 92.6 (n=894) | 50.2 | 2 |
| QASPER-32K | Contriever | Vanilla | 1321 | 18.5 | 29.5 | 33 | 83.1 | 44.2 | 8.6 | 25.8 | 37.7 | 60.1 | 59.9 (n=1314) | 23.7 | 8 |
| QASPER-32K | Contriever | SAADI | 1321 | 19.6 | 31.8 | 27 | 82.0 | 47.8 | 8.3 | 24.5 | 37.5 | 60.3 | 60.1 (n=1314) | 23.5 | 8 |
| QASPER-64K | Contriever | Vanilla | 1322 | 15.3 | 23.4 | 68 | 74.1 | 36.5 | 8.5 | 26.1 | 38.8 | 60.3 | 60.1 (n=1315) | 23.8 | 8 |
| QASPER-64K | Contriever | SAADI | 1322 | 16.3 | 25.9 | 51 | 70.6 | 39.9 | 9.4 | 25.4 | 38.1 | 62.0 | 61.7 (n=1315) | 24.5 | 8 |
| QASPER-128K | Contriever | Vanilla | 1321 | 13.2 | 19.2 | 143 | 65.1 | 30.6 | 8.6 | 26.1 | 38.5 | 60.4 | 60.2 (n=1314) | 23.8 | 8 |
| QASPER-128K | Contriever | SAADI | 1321 | 13.1 | 21.5 | 111 | 64.3 | 34.0 | 8.9 | 25.7 | 37.9 | 61.2 | 61.0 (n=1314) | 24.0 | 8 |
| QASPER-256K | Contriever | Vanilla | 1321 | 10.1 | 14.8 | 307 | 57.2 | 25.2 | 8.6 | 25.8 | 38.1 | 60.5 | 60.3 (n=1314) | 23.7 | 8 |
| QASPER-256K | Contriever | SAADI | 1321 | 11.4 | 16.9 | 235 | 57.6 | 28.6 | 9.3 | 26.6 | 39.4 | 62.1 | 61.9 (n=1314) | 24.7 | 7 |


## 2b. Set S: full ranking (A), sub-document (C), inside, keep scores (B)

| Setting | Retriever | Method | n | Full HR@5 | Full HR@10 | Full median rank | Sub-doc hit@10 | Top-10 share from gold sub-doc | In HR@1 | In HR@3 | In HR@5 | In HR@10 | In HR@10 (>10 chunks) | In MRR | In median rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MuSiQue-32K | Contriever | Vanilla | 575 | 65.6 | 80.3 | 3 | 99.3 | 69.3 | 26.8 | 58.6 | 72.9 | 91.3 | 91.2 (n=569) | 47.4 | 3 |
| MuSiQue-32K | Contriever | SAADI | 575 | 75.8 | 90.1 | 2 | 99.7 | 79.5 | 31.7 | 63.1 | 79.5 | 94.6 | 94.6 (n=569) | 52.0 | 2 |
| MuSiQue-64K | Contriever | Vanilla | 385 | 63.9 | 76.6 | 3 | 99.0 | 60.8 | 29.4 | 60.5 | 73.5 | 89.9 | 89.7 (n=380) | 49.0 | 3 |
| MuSiQue-64K | Contriever | SAADI | 385 | 70.9 | 87.0 | 2 | 99.2 | 70.9 | 35.8 | 64.4 | 78.4 | 93.2 | 93.2 (n=380) | 54.6 | 2 |
| MuSiQue-128K | Contriever | Vanilla | 250 | 62 | 74.4 | 3 | 97.6 | 51.3 | 36 | 65.2 | 75.2 | 90.8 | 90.6 (n=245) | 53.7 | 2 |
| MuSiQue-128K | Contriever | SAADI | 250 | 75.6 | 84.8 | 2 | 98.4 | 63.3 | 39.6 | 71.6 | 82 | 92.8 | 92.7 (n=245) | 58.1 | 2 |
| MuSiQue-256K | Contriever | Vanilla | 175 | 60.6 | 68 | 3 | 96 | 43.3 | 40.6 | 66.9 | 76.6 | 90.9 | 90.6 (n=170) | 56.7 | 2 |
| MuSiQue-256K | Contriever | SAADI | 175 | 70.9 | 80.6 | 2 | 97.7 | 56.2 | 45.1 | 69.7 | 78.3 | 92.6 | 92.4 (n=170) | 60.4 | 2 |
| QASPER-32K | Contriever | Vanilla | 979 | 18.3 | 29.9 | 30 | 82.8 | 43.9 | 8.7 | 26.0 | 37.2 | 60.0 | 59.7 (n=972) | 23.7 | 8 |
| QASPER-32K | Contriever | SAADI | 979 | 20.1 | 32.0 | 25 | 81.7 | 47.7 | 8.3 | 25.0 | 37.2 | 59.0 | 58.7 (n=972) | 23.4 | 8 |
| QASPER-64K | Contriever | Vanilla | 794 | 16.1 | 25.3 | 50.5 | 72.4 | 35.5 | 9.3 | 27.8 | 39.4 | 59.8 | 59.6 (n=789) | 24.7 | 8 |
| QASPER-64K | Contriever | SAADI | 794 | 17.4 | 28.1 | 37 | 69.4 | 38.4 | 10.3 | 26.8 | 38.9 | 61.7 | 61.5 (n=789) | 25.3 | 8 |
| QASPER-128K | Contriever | Vanilla | 660 | 15 | 22.6 | 75.5 | 63.3 | 29.0 | 10.5 | 29.1 | 41.4 | 62.6 | 62.3 (n=655) | 26.1 | 7.5 |
| QASPER-128K | Contriever | SAADI | 660 | 14.5 | 25 | 65.5 | 63.0 | 31.9 | 10.9 | 28.0 | 41.5 | 62.3 | 62.0 (n=655) | 26.1 | 7 |
| QASPER-256K | Contriever | Vanilla | 505 | 12.3 | 18.4 | 153 | 53.1 | 22.8 | 12.3 | 30.3 | 42.8 | 63.4 | 63 (n=500) | 27.6 | 7 |
| QASPER-256K | Contriever | SAADI | 505 | 13.5 | 20.8 | 101 | 55.0 | 26.3 | 12.3 | 30.3 | 43.4 | 65.0 | 64.6 (n=500) | 28.0 | 7 |


## 3. Set R: HR@k on queries whose gold sub-document has more than k chunks (keep / renorm)

| Setting | Retriever | Method | HR@1 (>1) | HR@3 (>3) | HR@5 (>5) | HR@10 (>10) | MRR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MuSiQue-32K | Contriever | Vanilla | 25.1 / 25.1 (n=900) | 56.8 / 56.8 (n=900) | 71.3 / 71.3 (n=900) | 90.9 / 90.9 (n=894) | 45.6 / 45.6 |
| MuSiQue-32K | Contriever | SAADI | 29.7 / 27.9 (n=900) | 61.8 / 60.1 (n=900) | 77.8 / 76.8 (n=900) | 93.2 / 93.0 (n=894) | 50.1 / 48.5 |
| MuSiQue-64K | Contriever | Vanilla | 24.9 / 24.9 (n=900) | 56.7 / 56.7 (n=900) | 71.2 / 71.2 (n=900) | 90.7 / 90.7 (n=894) | 45.4 / 45.4 |
| MuSiQue-64K | Contriever | SAADI | 29.8 / 28.1 (n=900) | 61 / 59.9 (n=900) | 77.7 / 76.2 (n=900) | 92.7 / 92.8 (n=894) | 50.2 / 48.8 |
| MuSiQue-128K | Contriever | Vanilla | 25.1 / 25.1 (n=900) | 56.8 / 56.8 (n=900) | 71.4 / 71.4 (n=900) | 90.7 / 90.7 (n=894) | 45.6 / 45.6 |
| MuSiQue-128K | Contriever | SAADI | 28.7 / 26.3 (n=900) | 62.1 / 60.1 (n=900) | 76.4 / 75.1 (n=900) | 92.6 / 92.7 (n=894) | 49.5 / 47.7 |
| MuSiQue-256K | Contriever | Vanilla | 24.9 / 24.9 (n=900) | 56.6 / 56.6 (n=900) | 71.2 / 71.2 (n=900) | 90.7 / 90.7 (n=894) | 45.4 / 45.4 |
| MuSiQue-256K | Contriever | SAADI | 30.1 / 28.6 (n=900) | 61.7 / 60.2 (n=900) | 76.6 / 75.8 (n=900) | 92.6 / 92.8 (n=894) | 50.2 / 48.8 |
| QASPER-32K | Contriever | Vanilla | 8.6 / 8.6 (n=1321) | 25.8 / 25.8 (n=1321) | 37.7 / 37.7 (n=1321) | 59.9 / 59.9 (n=1314) | 23.7 / 23.7 |
| QASPER-32K | Contriever | SAADI | 8.3 / 8.8 (n=1321) | 24.5 / 24.8 (n=1321) | 37.5 / 37.6 (n=1321) | 60.1 / 60.0 (n=1314) | 23.5 / 23.6 |
| QASPER-64K | Contriever | Vanilla | 8.5 / 8.5 (n=1322) | 26.1 / 26.1 (n=1322) | 38.8 / 38.8 (n=1322) | 60.1 / 60.1 (n=1315) | 23.8 / 23.8 |
| QASPER-64K | Contriever | SAADI | 9.4 / 9.2 (n=1322) | 25.4 / 25.8 (n=1322) | 38.1 / 38.3 (n=1322) | 61.7 / 61.3 (n=1315) | 24.5 / 24.4 |
| QASPER-128K | Contriever | Vanilla | 8.6 / 8.6 (n=1321) | 26.1 / 26.1 (n=1321) | 38.5 / 38.5 (n=1321) | 60.2 / 60.2 (n=1314) | 23.8 / 23.8 |
| QASPER-128K | Contriever | SAADI | 8.9 / 9.0 (n=1321) | 25.7 / 25.7 (n=1321) | 37.9 / 38.2 (n=1321) | 61.0 / 60.6 (n=1314) | 24.0 / 24.1 |
| QASPER-256K | Contriever | Vanilla | 8.6 / 8.6 (n=1321) | 25.8 / 25.8 (n=1321) | 38.1 / 38.1 (n=1321) | 60.3 / 60.3 (n=1314) | 23.7 / 23.7 |
| QASPER-256K | Contriever | SAADI | 9.3 / 9.3 (n=1321) | 26.6 / 26.4 (n=1321) | 39.4 / 40.0 (n=1321) | 61.9 / 62.0 (n=1314) | 24.7 / 24.8 |

keep = full-document scores, chunks outside the gold sub-document dropped (main). renorm = every channel z-scored again over the gold sub-document's chunks only. Vanilla's ranking is the same under both (z-scoring one channel does not change its order). All values in percent except ranks (1-based). The full-ranking columns use the paper's labels on the whole document.

## Warnings

- musique_32k__bge_m3: waiting for scores
- musique_32k__jina: waiting for scores
- musique_32k__qwen: waiting for scores
- qasper_64k__bge_m3: waiting for scores
- qasper_64k__jina: waiting for scores
- qasper_64k__qwen: waiting for scores

## Files written

- `experiments/rebuttal/c4_subdoc_control/reports/step_04.json`
- `experiments/rebuttal/c4_subdoc_control/reports/step_04.md`
- `experiments/rebuttal/c4_subdoc_control/work/perquery/musique_128k__contriever.csv.gz`
- `experiments/rebuttal/c4_subdoc_control/work/perquery/musique_256k__contriever.csv.gz`
- `experiments/rebuttal/c4_subdoc_control/work/perquery/musique_32k__contriever.csv.gz`
- `experiments/rebuttal/c4_subdoc_control/work/perquery/musique_64k__contriever.csv.gz`
- `experiments/rebuttal/c4_subdoc_control/work/perquery/qasper_128k__contriever.csv.gz`
- `experiments/rebuttal/c4_subdoc_control/work/perquery/qasper_256k__contriever.csv.gz`
- `experiments/rebuttal/c4_subdoc_control/work/perquery/qasper_32k__contriever.csv.gz`
- `experiments/rebuttal/c4_subdoc_control/work/perquery/qasper_64k__contriever.csv.gz`
