# Bridge probe - step 06: Memory check

Status: **FAILED** - 2026-11-08 00:16:11

## What was done

- `bash experiments/probe_bridge/sh/06_*.sh --force`
- Versions: python=3.12.3, platform=Linux-6.17.0-23-generic-x86_64-with-glibc2.39, numpy=2.3.5, pandas=3.0.1, yaml=6.0.3, saadi=0.1.0, git_commit=a9eea072e5d0577849d28e4c81a0ee7448a1a5e0

## Numbers

## Anything unexpected

The step failed. Full traceback:

```
Traceback (most recent call last):
  File "/mnt/data/zeinab/SAADI/experiments/probe_bridge/common.py", line 392, in run_step
    require_marker(prev)
  File "/mnt/data/zeinab/SAADI/experiments/probe_bridge/common.py", line 198, in require_marker
    raise RuntimeError(f"Step {step:02d} has not finished: missing {marker_path(step)}. Run its script first.")
RuntimeError: Step 05 has not finished: missing /mnt/data/zeinab/SAADI/outputs/probe_bridge/markers/step_05.done. Run its script first.

```

Last 50 log lines:

```

```

## Files written

- (none)
