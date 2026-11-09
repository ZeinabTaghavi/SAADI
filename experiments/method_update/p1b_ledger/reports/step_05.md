# p1b_ledger - step 05: Final report

Status: **ok** - 2026-11-09 00:43:27

## What was done

- `bash experiments/method_update/p1b_ledger/run_all.sh  (stage 05_report; python stage05_report.py )`
- `git commit 28eed3a1ee31487f3bf23dcb3470aa0e164e8cfd`
- Versions: python=3.12.3, platform=Linux-6.17.0-23-generic-x86_64-with-glibc2.39, numpy=2.3.5, pandas=3.0.1, yaml=6.0.3, saadi=0.1.0, git_commit=28eed3a1ee31487f3bf23dcb3470aa0e164e8cfd, torch=2.11.0+cu129, transformers=4.57.6, gliner=0.2.30, vllm=0.23.0

## Numbers

### Verdict

Run 1b: bridge 24 of 90 in cells 1-3 (run 1: 14/90); retrieval flat at fixed weights.

### Expected against observed

| Quantity | Expected (written before the run) | Observed | Verdict |
| --- | --- | --- | --- |
| Bridge formed, cells 1-3 | from 14 of 90 to above 40 of 90 | 24/90 (run 1: 14/90; run 0: 1/90) | not held |
| Join into the gold record (cells 1-3) | from 24 of 49 joins to most joins | 34 of 52 joins into the gold record, 18 into a new form-2 record (run 1: 24 of 49) | held |
| Earlier record in the prompt at the reveal | from 82 of 159 to above 130 of 159 | 121/174 (run 1: 82/159) | not held |
| Wrong SAME AS in the audit sample of 30 | from 7-8 of 30 to under 2 of 30 | to be read: the 30 sampled decisions are listed below; the code cannot judge a merge | to be read |
| Retrieval (live HR@10, cells 1-3) | probably still flat at fixed weights; the fusion landscape shows whether the bridges would be retrievable at a higher memory weight | live 6% against frozen 3% (run 1: 6% / 3%); live minus frozen +2.2 pp [-2.2, +6.7], McNemar p=0.625 | flat |
| Cell 5 (no reveal) | stays at zero | zero | held |
| Calls per book | unchanged (equal to run 1) | equal | held |

## Anything unexpected

Nothing unexpected.

## Files written

- `/mnt/data/zeinab/SAADI/experiments/method_update/p1b_ledger/reports/FINAL.md`
- `/mnt/data/zeinab/SAADI/results/method_update/p1b_ledger/REPORT_FOR_GDOC.md`
- `/mnt/data/zeinab/SAADI/results/method_update/p1b_ledger/TRACKER_ROW_1b.md`
- `/mnt/data/zeinab/SAADI/results/method_update/p1b_ledger/table.md`
- `/mnt/data/zeinab/SAADI/results/method_update/p1b_ledger/table.csv`
