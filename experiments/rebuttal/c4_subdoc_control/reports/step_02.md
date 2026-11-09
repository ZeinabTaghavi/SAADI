# C4 sub-document control - step 02: input completeness and reproduction of the paper (OK)

Command (from the repository root on the server):

```bash
bash experiments/rebuttal/c4_subdoc_control/sh/02_check.sh
```

Seed 42. CPU only. Run time 628 s.
Published weights read from `configs/methods/sadi.yaml` (cached_retrieval): lambda_t=1.0, lambda_n=0.5, lambda_h=0.3, normalization_method=zscore, memory_name_weight=1.0, memory_alias_weight=1.0, memory_description_weight=1.0, memory_aggregation=max.

## 1. What the paper saved

No full ranking and no per-chunk score is saved for any pair: the paper keeps SAADI's top-k list per query (depth 10 or 20) and Vanilla only as aggregate HR. Full per-chunk channel scores therefore come from BM25 re-scoring here (CPU) or, for the dense retrievers, from C2's recomputed streams.
| Setting | Retriever | Paper top-k depth | Saved per-document top-10 file | Full per-chunk scores | Source of full scores |
| --- | --- | --- | --- | --- | --- |
| MuSiQue-32K | bm25 | 10/20 | yes | available | re-scored here |
| MuSiQue-32K | contriever | 10 | yes | waiting for scores | outputs/rebuttal/C2_heldout_fusion/streams |
| MuSiQue-32K | bge_m3 | 10 | yes | waiting for scores | outputs/rebuttal/C2_heldout_fusion/streams |
| MuSiQue-32K | jina | 10 | yes | waiting for scores | outputs/rebuttal/C2_heldout_fusion/streams |
| MuSiQue-32K | qwen | 10 | yes | waiting for scores | outputs/rebuttal/C2_heldout_fusion/streams |
| MuSiQue-64K | bm25 | 20 | yes | available | re-scored here |
| MuSiQue-64K | contriever | 10 | yes | waiting for scores | outputs/rebuttal/C2_heldout_fusion/streams |
| MuSiQue-128K | bm25 | 20 | yes | available | re-scored here |
| MuSiQue-128K | contriever | 10 | yes | waiting for scores | outputs/rebuttal/C2_heldout_fusion/streams |
| MuSiQue-256K | bm25 | 20 | yes | available | re-scored here |
| MuSiQue-256K | contriever | 10 | yes | waiting for scores | outputs/rebuttal/C2_heldout_fusion/streams |
| QASPER-32K | bm25 | 20 | yes | available | re-scored here |
| QASPER-32K | contriever | 10 | yes | waiting for scores | outputs/rebuttal/C2_heldout_fusion/streams |
| QASPER-64K | bm25 | 10/20 | yes | available | re-scored here |
| QASPER-64K | contriever | 10 | yes | waiting for scores | outputs/rebuttal/C2_heldout_fusion/streams |
| QASPER-64K | bge_m3 | 10 | yes | waiting for scores | outputs/rebuttal/C2_heldout_fusion/streams |
| QASPER-64K | jina | 10 | yes | waiting for scores | outputs/rebuttal/C2_heldout_fusion/streams |
| QASPER-64K | qwen | 10 | yes | waiting for scores | outputs/rebuttal/C2_heldout_fusion/streams |
| QASPER-128K | bm25 | 20 | yes | available | re-scored here |
| QASPER-128K | contriever | 10 | yes | waiting for scores | outputs/rebuttal/C2_heldout_fusion/streams |
| QASPER-256K | bm25 | 20 | yes | available | re-scored here |
| QASPER-256K | contriever | 10 | yes | waiting for scores | outputs/rebuttal/C2_heldout_fusion/streams |

Missing full-score files (pairs marked "waiting for scores"; C4 reads them later from `$C4_DENSE_STREAMS_DIR` with no code change):

- MuSiQue-32K / contriever: `outputs/rebuttal/C2_heldout_fusion/streams/musique_32k__contriever.parquet`, `outputs/rebuttal/C2_heldout_fusion/streams/musique_32k__contriever.queries.json`
- MuSiQue-32K / bge_m3: `outputs/rebuttal/C2_heldout_fusion/streams/musique_32k__bge_m3.parquet`, `outputs/rebuttal/C2_heldout_fusion/streams/musique_32k__bge_m3.queries.json`
- MuSiQue-32K / jina: `outputs/rebuttal/C2_heldout_fusion/streams/musique_32k__jina.parquet`, `outputs/rebuttal/C2_heldout_fusion/streams/musique_32k__jina.queries.json`
- MuSiQue-32K / qwen: `outputs/rebuttal/C2_heldout_fusion/streams/musique_32k__qwen.parquet`, `outputs/rebuttal/C2_heldout_fusion/streams/musique_32k__qwen.queries.json`
- MuSiQue-64K / contriever: `outputs/rebuttal/C2_heldout_fusion/streams/musique_64k__contriever.parquet`, `outputs/rebuttal/C2_heldout_fusion/streams/musique_64k__contriever.queries.json`
- MuSiQue-128K / contriever: `outputs/rebuttal/C2_heldout_fusion/streams/musique_128k__contriever.parquet`, `outputs/rebuttal/C2_heldout_fusion/streams/musique_128k__contriever.queries.json`
- MuSiQue-256K / contriever: `outputs/rebuttal/C2_heldout_fusion/streams/musique_256k__contriever.parquet`, `outputs/rebuttal/C2_heldout_fusion/streams/musique_256k__contriever.queries.json`
- QASPER-32K / contriever: `outputs/rebuttal/C2_heldout_fusion/streams/qasper_32k__contriever.parquet`, `outputs/rebuttal/C2_heldout_fusion/streams/qasper_32k__contriever.queries.json`
- QASPER-64K / contriever: `outputs/rebuttal/C2_heldout_fusion/streams/qasper_64k__contriever.parquet`, `outputs/rebuttal/C2_heldout_fusion/streams/qasper_64k__contriever.queries.json`
- QASPER-64K / bge_m3: `outputs/rebuttal/C2_heldout_fusion/streams/qasper_64k__bge_m3.parquet`, `outputs/rebuttal/C2_heldout_fusion/streams/qasper_64k__bge_m3.queries.json`
- QASPER-64K / jina: `outputs/rebuttal/C2_heldout_fusion/streams/qasper_64k__jina.parquet`, `outputs/rebuttal/C2_heldout_fusion/streams/qasper_64k__jina.queries.json`
- QASPER-64K / qwen: `outputs/rebuttal/C2_heldout_fusion/streams/qasper_64k__qwen.parquet`, `outputs/rebuttal/C2_heldout_fusion/streams/qasper_64k__qwen.queries.json`
- QASPER-128K / contriever: `outputs/rebuttal/C2_heldout_fusion/streams/qasper_128k__contriever.parquet`, `outputs/rebuttal/C2_heldout_fusion/streams/qasper_128k__contriever.queries.json`
- QASPER-256K / contriever: `outputs/rebuttal/C2_heldout_fusion/streams/qasper_256k__contriever.parquet`, `outputs/rebuttal/C2_heldout_fusion/streams/qasper_256k__contriever.queries.json`

BM25 scoring: musique_128k: 12 documents (0 from checkpoints), 323 s; musique_256k: 6 documents (0 from checkpoints), 513 s; musique_32k: 45 documents (0 from checkpoints), 52 s; musique_64k: 23 documents (0 from checkpoints), 148 s; qasper_128k: 11 documents (0 from checkpoints), 265 s; qasper_256k: 6 documents (0 from checkpoints), 280 s; qasper_32k: 46 documents (0 from checkpoints), 112 s; qasper_64k: 23 documents (0 from checkpoints), 184 s.

## 2. Reproduction on the full ranking

| Setting | Retriever | Queries | Eligible | SAADI top-k identical to each paper ranking | SAADI HR@5 / HR@10 | Max HR gap vs paper (points) | Label mismatches | Repo fusion check (bad/rankings) | Vanilla != stream-time top-10 | Gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MuSiQue-32K | bm25 | 900 | 900 | 900/900 / 900/900 | 62.8 / 77.7 | 0.00 | 0 | 0/40 | 0 | PASS |
| MuSiQue-64K | bm25 | 900 | 900 | 900/900 | 59.2 / 74.3 | 0.00 | 0 | 0/40 | 0 | PASS |
| MuSiQue-128K | bm25 | 900 | 900 | 900/900 | 55.7 / 68.8 | 0.00 | 0 | 0/40 | 0 | PASS |
| MuSiQue-256K | bm25 | 900 | 900 | 900/900 | 52.3 / 64.3 | 0.00 | 0 | 0/40 | 0 | PASS |
| QASPER-32K | bm25 | 1321 | 1321 | 1321/1321 | 19.6 / 32.1 | 0.00 | 0 | 0/40 | 0 | PASS |
| QASPER-64K | bm25 | 1333 | 1333 | 1333/1333 / 1333/1333 | 17.0 / 26.1 | 0.00 | 0 | 0/40 | 0 | PASS |
| QASPER-128K | bm25 | 1321 | 1321 | 1321/1321 | 15.0 / 22.6 | 0.00 | 0 | 0/40 | 0 | PASS |
| QASPER-256K | bm25 | 1321 | 1321 | 1321/1321 | 12.2 / 18.4 | 0.00 | 0 | 0/40 | 0 | PASS |


## 3. Vanilla (text channel) against the paper's Vanilla (aggregate, not gated)

| Setting | Retriever | Text channel HR@5 / HR@10 (ours) | Table 26 Vanilla (ours - paper) | Table 1 Vanilla (ours - paper) |
| --- | --- | --- | --- | --- |
| MuSiQue-32K | bm25 | 57.6 / 74 | 57.6 (+0.00) / 74 (+0.00) | 57.6 (-0.04) / 74 (+0.00) |
| MuSiQue-64K | bm25 | 54 / 69.2 | 54 (+0.00) / 69.2 (+0.00) | 57.6 (-3.60) / 74 (-4.78) |
| MuSiQue-128K | bm25 | 51.6 / 64.9 | 51.6 (+0.00) / 64.9 (+0.00) | 57.6 (-6.04) / 74 (-9.11) |
| MuSiQue-256K | bm25 | 48.4 / 60.6 | 48.4 (+0.00) / 60.6 (+0.00) | 57.6 (-9.16) / 74 (-13.44) |
| QASPER-32K | bm25 | 17.9 / 27.6 | 17.9 (+0.00) / 27.6 (+0.00) | 14.4 (+3.47) / 22.1 (+5.45) |
| QASPER-64K | bm25 | 14.4 / 22.1 | 14.4 (+0.00) / 22.1 (+0.00) | 14.4 (+0.00) / 22.1 (-0.04) |
| QASPER-128K | bm25 | 12.8 / 18.5 | 12.8 (+0.00) / 18.5 (+0.00) | 14.4 (-1.61) / 22.1 (-3.55) |
| QASPER-256K | bm25 | 11.5 / 15.7 | 11.5 (-0.00) / 15.7 (-0.00) | 14.4 (-2.89) / 22.1 (-6.35) |

Rebuttal Vanilla = SAADI's text channel on the same chunks (decided in C1). The paper's Vanilla rows are aggregate numbers from a separate run, so they are shown here and not used as a gate.

## Warnings

- musique_32k/contriever: waiting for scores
- musique_32k/bge_m3: waiting for scores
- musique_32k/jina: waiting for scores
- musique_32k/qwen: waiting for scores
- musique_64k/contriever: waiting for scores
- musique_128k/contriever: waiting for scores
- musique_256k/contriever: waiting for scores
- qasper_32k/contriever: waiting for scores
- qasper_64k/contriever: waiting for scores
- qasper_64k/bge_m3: waiting for scores
- qasper_64k/jina: waiting for scores
- qasper_64k/qwen: waiting for scores
- qasper_128k/contriever: waiting for scores
- qasper_256k/contriever: waiting for scores

## Files written

- `experiments/rebuttal/c4_subdoc_control/reports/step_02.json`
- `experiments/rebuttal/c4_subdoc_control/reports/step_02.md`
