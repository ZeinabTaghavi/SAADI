# probe_baselines - stage 05_b4_lqca

Status: **FAILED** - 2026-11-09 00:40:35; wall clock 0.0 min

## What was done

- `/home/iataghav/data/SAADI/experiments/probe_baselines/third_party/venv_lqca/bin/python /home/iataghav/data/SAADI/experiments/probe_baselines/systems/b4_lqca/build.py`
- git commit `28eed3a1ee31487f3bf23dcb3470aa0e164e8cfd`; python 3.12.3 on Linux-6.17.0-23-generic-x86_64-with-glibc2.39

## Numbers

(no tables from this stage)

## Anything unexpected

`/home/iataghav/data/SAADI/experiments/probe_baselines/third_party/venv_lqca/bin/python /home/iataghav/data/SAADI/experiments/probe_baselines/systems/b4_lqca/build.py` exited with code 1. Its traceback is in the log lines below.
Last 50 log lines:

```
multiprocessing.pool.RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/lib/python3.12/multiprocessing/pool.py", line 125, in worker
    result = (True, func(*args, **kwds))
                    ^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/multiprocessing/pool.py", line 48, in mapstar
    return list(map(*args))
           ^^^^^^^^^^^^^^^^
  File "/home/iataghav/data/SAADI/experiments/probe_baselines/systems/b4_lqca/build.py", line 77, in work
    import torch
ModuleNotFoundError: No module named 'torch'
"""

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/iataghav/data/SAADI/experiments/probe_baselines/systems/b4_lqca/build.py", line 172, in <module>
    raise SystemExit(main())
                     ^^^^^^
  File "/home/iataghav/data/SAADI/experiments/probe_baselines/systems/b4_lqca/build.py", line 139, in main
    results = pool.map(work, jobs, chunksize=1)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/multiprocessing/pool.py", line 367, in map
    return self._map_async(func, iterable, mapstar, chunksize).get()
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/multiprocessing/pool.py", line 774, in get
    raise self._value
ModuleNotFoundError: No module named 'torch'
```

## Files written

- (see the stage's output folder)
