# probe_baselines - stage 04_b3_hipporag

Status: **FAILED** - 2026-11-09 00:40:32; wall clock 0.0 min

## What was done

- `/home/iataghav/data/SAADI/experiments/probe_baselines/third_party/venv_hipporag/bin/python /home/iataghav/data/SAADI/experiments/probe_baselines/systems/b3_hipporag/build.py`
- git commit `28eed3a1ee31487f3bf23dcb3470aa0e164e8cfd`; python 3.12.3 on Linux-6.17.0-23-generic-x86_64-with-glibc2.39

## Numbers

(no tables from this stage)

## Anything unexpected

`/home/iataghav/data/SAADI/experiments/probe_baselines/third_party/venv_hipporag/bin/python /home/iataghav/data/SAADI/experiments/probe_baselines/systems/b3_hipporag/build.py` exited with code 1. Its traceback is in the log lines below.
Last 50 log lines:

```
Traceback (most recent call last):
  File "/home/iataghav/data/SAADI/experiments/probe_baselines/systems/b3_hipporag/build.py", line 126, in <module>
    raise SystemExit(main())
                     ^^^^^^
  File "/home/iataghav/data/SAADI/experiments/probe_baselines/systems/b3_hipporag/build.py", line 53, in main
    import hipporag
ModuleNotFoundError: No module named 'hipporag'
```

## Files written

- (see the stage's output folder)
