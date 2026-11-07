# Bridge probe - step 04: Insert and re-chunk

Status: **FAILED** - 2026-11-07 17:29:29

## What was done

- `bash experiments/probe_bridge/sh/04_*.sh`
- Versions: python=3.12.3, platform=Linux-6.17.0-23-generic-x86_64-with-glibc2.39, numpy=2.3.5, pandas=3.0.1, yaml=6.0.3, saadi=0.1.0, git_commit=2b22afc634c7bf24a0250e0c247598a2f31eac77

## Numbers

## Anything unexpected

The step failed. Full traceback:

```
Traceback (most recent call last):
  File "/mnt/data/zeinab/SAADI/experiments/probe_bridge/common.py", line 393, in run_step
    main(report, argv)
  File "/mnt/data/zeinab/SAADI/experiments/probe_bridge/steps/step04_insert_and_rechunk.py", line 128, in main
    raise RuntimeError(
RuntimeError: Check 7 failed in B31: inserted sentences at chunks 96 and 98 (< 3 apart) - [('c6_B31_001', 'gold'), ('c6_B31_002', 'gold')]. Rerun step 3 with --force after deciding at the hand-off.

```

Last 50 log lines:

```
NovelHopQA subset mode enabled; skipped 26 unresolved title(s) under /mnt/data/zeinab/SAADI/outputs/novelhopqa_materials/book-corpus-root. Example title(s): A Connecticut Yankee in King Arthur's Court, Billy Budd, David Copperfield, Don Quixote, Dracula. Reports written to /mnt/data/zeinab/SAADI/experiments/probe_bridge/logs/dataset_reports
```

## Files written

- (none)
