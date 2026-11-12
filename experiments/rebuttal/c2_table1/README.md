# Table 1 with held-out fusion weights (C2 follow-up)

Where SAADI stands in the paper's Table 1 when its fusion weights are the held-out weights of C2
(5-fold cross-validation grouped by document, seed 42, chosen separately for each dataset and retriever),
compared with the published weights. Re-scoring only: no index is rebuilt, no LLM is called, no GPU is used.

```bash
bash experiments/rebuttal/c2_table1/run.sh
```

## Inputs

- C2's per-query grid arrays `outputs/rebuttal/C2_heldout_fusion/grid/<dataset>__<retriever>.npz` (hit ranks
  for every one of the 1,296 weight settings), written by C2's analysis, and C2's `analysis.json` (fold count
  per pair and a consistency check).
- The paper's Table 1: `results/tables/main/retrieval_comparison/table.csv`.

A retriever is included when all four of its datasets have a C2 DONE marker and a grid array that is newer
than the job's streams. Jina and Qwen are included automatically once C2's analysis has run with them:

```bash
bash -c 'source experiments/rebuttal/C2_heldout_fusion/env.sh && python scripts/rebuttal/c2_heldout_fusion.py --group A --analysis-only'
bash experiments/rebuttal/c2_table1/run.sh
```

## Steps

1. **Check.** SAADI with the published weights must print the Table 1 value (one decimal) in every cell. A
   difference in a pair that passed C2's strict reproduction gate stops the run (exit code 1; `REPORT.md` lists
   the differences). For pairs where C2 accepted embedding drift (Jina, Qwen; HR within 0.5 points) the
   differences are listed and the run continues; both Table 1 versions then use the recomputed values.
2. **Table 1 twice**: SAADI with the published and with the held-out weights; every other method keeps its
   Table 1 value.
3. **Rank** of SAADI among the methods of its retriever in every cell: 1 + number of methods whose Table 1
   value is strictly higher than SAADI's unrounded value, the rule behind the paper's "first or second in 34 of
   40 cells". Mean rank (three decimals), first, first or second; overall, per dataset, per retriever and per cutoff.
4. **Held-out SAADI minus Vanilla** per pair at HR@5 and HR@10, 95% paired bootstrap over queries (10,000
   resamples, seed 42). Vanilla is the text channel alone (lambda_M = lambda_S = 0) on the same chunks and
   queries; on LooGLE and NovelHopQA it differs from the Table 1 Vanilla row, which came from another run and is
   kept unchanged in the rank tables.

C2's code is imported, not changed: `saadi.rebuttal.heldout_fusion` (grid arrays, held-out selection) and
`saadi.rebuttal.frozen_records.paired_bootstrap_ci`.

## Outputs

`results/rebuttal/c2_table1/`: `REPORT.md` (summary, then the tables), `results.json` (every number in the
report), `table1_published.csv`, `table1_heldout.csv`. Logs: `experiments/rebuttal/c2_table1/logs/`.
