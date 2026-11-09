# C3: channel ablation by re-scoring

For reviewers zXpi and iVU8: which of SAADI's three query-time channels (text, memory, summary) carries the gain?
Query-time only: nothing is rebuilt, no LLM is called, no GPU is used. This is not the index-time ablation of the
paper's Table 12 (history-conditioned extraction, `results/tables/ablations/state_conditioning/`).

What the paper already has: the score-fusion table (`results/tables/ablations/score_fusion/`) varies the channel
weights but never sets one to zero, and only as rounded summaries; the memory-field table varies mu_N/mu_A/mu_D
inside the memory channel. No saved per-query run drops a whole channel. C3 adds that, on all four datasets, paired
per query, with bootstrap intervals.

## Variants

Published Equation 2 (`combine_mbr_variant_chunk_scores`, MBR2), published weights read from
`configs/methods/sadi.yaml` (`cached_retrieval`), one channel weight set to 0. The remaining weights are not
re-tuned or rescaled; Equation 2's own per-chunk division by the sum of the weights present is unchanged.

| Variant | lambda_T | lambda_M | lambda_S |
| --- | --- | --- | --- |
| Text only | 1.0 | 0 | 0 |
| Text + summary | 1.0 | 0 | 0.3 |
| Text + memory | 1.0 | 0.5 | 0 |
| All three (SAADI) | 1.0 | 0.5 | 0.3 |
| Summary alone (information) | 0 | 0 | 0.3 |
| Memory alone (information) | 0 | 0.5 | 0 |

## Inputs

- The paper's saved rankings (one file per dataset x retriever, `DATASETS[...].saved_results_json`), for the
  query set, evidence labels and the check against "all three".
- Per-chunk channel scores in C2's stream format (`<dataset>__<retriever>.parquet` + `.queries.json`):
  - BM25: recomputed by step 01 on CPU from the construction caches with C2's `compute_job_streams`, into `work/streams/`.
  - Dense retrievers: read only, from `C3_STREAMS_DIR` (default `outputs/rebuttal/C2_heldout_fusion/streams`).
    The paper's saved outputs hold only fused top-10 rankings, not channel scores, so a pair without streams is
    reported as "waiting for channel scores" with the missing files.

## Steps (server, from the repository root, one at a time)

```bash
bash experiments/rebuttal/c3_channel_ablation/sh/01_check.sh
```

| Step | Script | What it does |
| --- | --- | --- |
| 01 | `sh/01_check.sh` | Inputs; BM25 channel scores; check 1 (all three = saved rankings, every query); check 2 (text only vs Vanilla) |
| 02 | `sh/02_bm25.sh` | All six variants for BM25: per-query ranks, HR@5/HR@10, median rank, first paired differences |
| 03 | `sh/03_dense.sh` | The same for the dense pairs that passed check 1 (only from saved channel scores) |
| 04 | `sh/04_report.sh` | Tables, paired comparisons, `REPORT.md`, `table_paper.tex` (not written yet) |

Steps 02 and 03 run the same script (`scripts/step02_variants.py`) and write one small per-query file per pair to
`results/rebuttal/c3_channel_ablation/per_query/` (committed), so step 04 can combine a laptop BM25 run with a server
dense run. `work/ready_pairs.json` is merged across step 01 runs on the same machine.

Dense pairs on the server, after `git pull`:

```bash
C3_RETRIEVERS=contriever,bge_m3 bash experiments/rebuttal/c3_channel_ablation/sh/01_check.sh
C3_RETRIEVERS=contriever,bge_m3 bash experiments/rebuttal/c3_channel_ablation/sh/03_dense.sh
```

Each step writes `reports/step_NN.md` and `.json` (also when it fails, with the traceback) and a log in `logs/`.
Results go to `results/rebuttal/c3_channel_ablation/`. Seed 42; 10,000 bootstrap resamples over queries.
