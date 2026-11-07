# Bridge probe - step 05: Index (GPU)

Status: **FAILED** - 2026-11-08 00:20:58

## What was done

- `bash experiments/probe_bridge/sh/05_*.sh --force`
- `SADI_OUTPUT_DIR=/mnt/data/zeinab/SAADI/outputs/probe_bridge /mnt/data/zeinab/raptor/.venv/bin/python3 /mnt/data/zeinab/SAADI/scripts/run_study.py --suite probe_bridge --profile published --datasets novelhopqa_probe --retrievers bm25 --stage ingest`
- Versions: python=3.12.3, platform=Linux-6.17.0-23-generic-x86_64-with-glibc2.39, numpy=2.3.5, pandas=3.0.1, yaml=6.0.3, saadi=0.1.0, git_commit=a9eea072e5d0577849d28e4c81a0ee7448a1a5e0

## Numbers

### Frozen spec

personas.jsonl SHA-256 verified: `11c4c1696c67171f78821d3ba8311ccd28c851315ca1c2a404a9a422593897bb`

### Runtime environment

| setting | value |
| --- | --- |
| HF_HOME | /mnt/cache/taghavi |
| HF_HUB_CACHE | /mnt/cache/taghavi/hub |
| Model weights already in the cache | True |
| HF_HOME writable | True |
| CUDA_VISIBLE_DEVICES | 0,1,2,3 |
| VLLM_USE_FLASHINFER_SAMPLER | 0 |
| HF_HUB_OFFLINE | 1 |

### Resolved construction settings

| setting | value |
| --- | --- |
| temperature | 0.0 |
| seed | 42 |
| memory selector | word |
| history text | anchored |
| anchor matching | strict |
| entities whose anchor is not found | keep |
| recall budget (records, aliases, descriptions) | 5, 2, 3 |
| cached model snapshots | 0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe |
| paper's checkpoint | 0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe |

### Wall clock

0.00 hours (0 s) this invocation.

### Construction records

No cache index found yet.

## Anything unexpected

The step failed. Full traceback:

```
Traceback (most recent call last):
  File "/mnt/data/zeinab/SAADI/experiments/probe_bridge/common.py", line 393, in run_step
    main(report, argv)
  File "/mnt/data/zeinab/SAADI/experiments/probe_bridge/steps/step05_index.py", line 236, in main
    raise RuntimeError(f"run_study.py exited with code {proc.returncode}. Rerun this script to resume; the construction cache is kept.")
RuntimeError: run_study.py exited with code 1. Rerun this script to resume; the construction cache is kept.

```

Last 50 log lines:

```
Saadi: novelhopqa_probe / probe_bridge / greedy_seed42
Traceback (most recent call last):
  File "/mnt/data/zeinab/SAADI/scripts/run_study.py", line 99, in stage_lock
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
BlockingIOError: [Errno 11] Resource temporarily unavailable

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/mnt/data/zeinab/SAADI/scripts/run_study.py", line 313, in <module>
    raise SystemExit(main())
                     ^^^^^^
  File "/mnt/data/zeinab/SAADI/scripts/run_study.py", line 309, in main
    return run(args)
           ^^^^^^^^^
  File "/mnt/data/zeinab/SAADI/scripts/run_study.py", line 243, in run
    execute_stage([sys.executable, "-m", "saadi.cli.main", "run-experiment", "--config", config_path,
  File "/mnt/data/zeinab/SAADI/scripts/run_study.py", line 112, in execute_stage
    with stage_lock(directory):
  File "/usr/lib/python3.12/contextlib.py", line 137, in __enter__
    return next(self.gen)
           ^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/SAADI/scripts/run_study.py", line 101, in stage_lock
    raise RuntimeError(f"Another process is writing {directory}") from exc
RuntimeError: Another process is writing /mnt/data/zeinab/SAADI/outputs/probe_bridge/novelhopqa_probe/indexes/59bb6acf00d222d8
```

## Files written

- `/mnt/data/zeinab/SAADI/configs/experiments/novelhopqa_probe_retrieval_ablation.yaml  (declared file outside the probe folders)`
