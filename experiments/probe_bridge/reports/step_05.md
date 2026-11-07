# Bridge probe - step 05: Index (GPU)

Status: **FAILED** - 2026-11-07 19:27:12

## What was done

- `bash experiments/probe_bridge/sh/05_*.sh`
- Versions: python=3.12.3, platform=Linux-6.17.0-23-generic-x86_64-with-glibc2.39, numpy=2.3.5, pandas=3.0.1, yaml=6.0.3, saadi=0.1.0, git_commit=677ee67084b34e34a7158772ee7870bdaaf4c3c8

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

## Anything unexpected

The step failed. Full traceback:

```
Traceback (most recent call last):
  File "/mnt/data/zeinab/SAADI/experiments/probe_bridge/common.py", line 393, in run_step
    main(report, argv)
  File "/mnt/data/zeinab/SAADI/experiments/probe_bridge/steps/step05_index.py", line 169, in main
    resolved = run_study.merge(run_study.published_config(load_config(PROBE_CONFIG)), suite["variants"][0]["overrides"])
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/SAADI/scripts/run_study.py", line 58, in published_config
    manifest = verify_bundle(ROOT / "outputs/saadi/reference")
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/SAADI/src/saadi/reporting/tables.py", line 37, in verify_bundle
    raise ValueError("Reference bundle contains missing or unregistered files")
ValueError: Reference bundle contains missing or unregistered files

```

Last 50 log lines:

```

```

## Files written

- `/mnt/data/zeinab/SAADI/configs/experiments/novelhopqa_probe_retrieval_ablation.yaml  (declared file outside the probe folders)`
