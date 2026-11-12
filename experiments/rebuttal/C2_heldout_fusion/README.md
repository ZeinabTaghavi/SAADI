# C2: SAADI fusion weights chosen on held-out queries

Answers mokb Q1 and iVU8 Q3: are SAADI's gains and its ranking still there when every fusion weight is
chosen without seeing the evaluated queries? Code: `src/saadi/rebuttal/heldout_fusion.py` (library) and
`scripts/rebuttal/c2_heldout_fusion.py` (entry point).

## Run (server, from the repository root)

Group A uses GPUs 0-3 only, group B GPUs 4-7 only. Start both; the order does not matter.

```bash
nohup bash experiments/rebuttal/C2_heldout_fusion/run_gpus_0123.sh > /dev/null 2>&1 &
nohup bash experiments/rebuttal/C2_heldout_fusion/run_gpus_4567.sh > /dev/null 2>&1 &
```

Or run each one in its own tmux window without `nohup`. Each script writes its own log (it tees it), so
nothing is lost when the terminal closes:

- group logs: `logs/group_A_<date>.log`, `logs/group_B_<date>.log`; per-job logs: `logs/<dataset>__<retriever>.log`
- group reports: `reports/group_A.md` / `.json`, `reports/group_B.md` / `.json`
- final report (written by whichever group finishes last): `results/rebuttal/C2_heldout_fusion/REPORT.md`
  and `REPORT_FOR_GDOC.md`

Options: `--dry-run` (CPU test: BM25, 2 documents per dataset, small grid; writes to `dry_run/` folders),
`--analysis-only` (rerun the CPU analysis once all 20 jobs are DONE; report `reports/group_X_analysis_only.md`).
`--analysis-only --retrievers bm25,contriever,bge_m3` analyses only those retrievers (their jobs on all four
datasets must be DONE); the reports name the retrievers left out, and the Table 1 ranking counts only their cells.
`--report-only` rebuilds `REPORT.md` and `REPORT_FOR_GDOC.md` from the saved `analysis.json` and `tables/`
(CPU, seconds, no streams needed); part H gives the full-grid rank of the published setting, the best single
setting, the selected-weight medians and non-zero shares, and the per-dataset means.

```bash
bash -c 'source experiments/rebuttal/C2_heldout_fusion/env.sh && python scripts/rebuttal/c2_heldout_fusion.py --group A --analysis-only --retrievers bm25,contriever,bge_m3'
```

## What happens

1. Preflight: Python env and modules, commit, input files, free disk, `nvidia-smi` (stops if any of the
   group's GPUs already uses more than 2 GB).
2. Plan: the first group writes `outputs/rebuttal/C2_heldout_fusion/plan.json` under a file lock; the
   other reads it. 16 dense jobs split longest-first between the groups, two BM25 jobs per group (CPU).
3. Jobs: up to four at once, one GPU each (`CUDA_VISIBLE_DEVICES=<id>`). Each job scores every saved query
   with the published pipeline, checks the published weights against the paper's saved rankings
   (BM25: identical; dense: HR@5/HR@10 within 0.1 points and at least 99% identical top-10), and only then
   saves its streams (`outputs/rebuttal/C2_heldout_fusion/streams/`) and a `DONE` marker. A failed check
   writes a `FAILED` marker and a quarantined copy under `streams_failed/` that the analysis never reads.
4. Analysis (CPU, once): after the last job, the group that finishes second runs it; the other one
   reports "waiting for group X" and exits 0.

Rerunning a script skips DONE jobs and resumes unfinished ones from per-document checkpoints
(`outputs/rebuttal/C2_heldout_fusion/work/`). Exit codes: 0 fine, 1 a job or the analysis failed,
2 preflight failed.
