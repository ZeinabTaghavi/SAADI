# Bridge probe - step 05: Index (GPU)

Status: **FAILED** - 2026-11-07 19:47:52

## What was done

- `bash experiments/probe_bridge/sh/05_*.sh`
- `SADI_OUTPUT_DIR=/mnt/data/zeinab/SAADI/outputs/probe_bridge /mnt/data/zeinab/raptor/.venv/bin/python3 /mnt/data/zeinab/SAADI/scripts/run_study.py --suite probe_bridge --profile published --datasets novelhopqa_probe --retrievers bm25 --stage ingest`
- Versions: python=3.12.3, platform=Linux-6.17.0-23-generic-x86_64-with-glibc2.39, numpy=2.3.5, pandas=3.0.1, yaml=6.0.3, saadi=0.1.0, git_commit=12630a46504ac8a020f57fdd99e1e9358e3092b5

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

### Resolved construction settings

| setting | value |
| --- | --- |
| temperature | 0.0 |
| seed | 42 |
| memory selector | word |
| history text | anchored |
| anchor matching | strict |
| recall budget (records, aliases, descriptions) | 5, 2, 3 |
| cached model snapshots | 0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe |
| paper's checkpoint | 0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe |

### Wall clock

0.26 hours (920 s) this invocation.

### Construction records

No cache index found yet.

## Anything unexpected

The step failed. Full traceback:

```
Traceback (most recent call last):
  File "/mnt/data/zeinab/SAADI/experiments/probe_bridge/common.py", line 393, in run_step
    main(report, argv)
  File "/mnt/data/zeinab/SAADI/experiments/probe_bridge/steps/step05_index.py", line 229, in main
    raise RuntimeError(f"run_study.py exited with code {proc.returncode}. Rerun this script to resume; the construction cache is kept.")
RuntimeError: run_study.py exited with code 1. Rerun this script to resume; the construction cache is kept.

```

Last 50 log lines:

```
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/SAADI/src/saadi/llm/factory.py", line 376, in _build_vllm_backend
    model = LLM(**llm_kwargs)
            ^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/vllm/entrypoints/llm.py", line 349, in __init__
    self.llm_engine = LLMEngine.from_engine_args(
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 170, in from_engine_args
    return cls(
           ^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 104, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 102, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 771, in __init__
    super().__init__(
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 570, in __init__
    with launch_core_engines(
  File "/usr/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1190, in launch_core_engines
    wait_for_engine_startup(
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1249, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {'EngineCore': 1}
/usr/lib/python3.12/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 3 leaked shared_memory objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '
Traceback (most recent call last):
  File "/mnt/data/zeinab/SAADI/scripts/run_study.py", line 310, in <module>
    raise SystemExit(main())
                     ^^^^^^
  File "/mnt/data/zeinab/SAADI/scripts/run_study.py", line 306, in main
    return run(args)
           ^^^^^^^^^
  File "/mnt/data/zeinab/SAADI/scripts/run_study.py", line 240, in run
    execute_stage([sys.executable, "-m", "saadi.cli.main", "run-experiment", "--config", config_path,
  File "/mnt/data/zeinab/SAADI/scripts/run_study.py", line 110, in execute_stage
    _execute_stage(command, directory, inputs=inputs)
  File "/mnt/data/zeinab/SAADI/scripts/run_study.py", line 130, in _execute_stage
    execute(command, False)
  File "/mnt/data/zeinab/SAADI/scripts/run_study.py", line 39, in execute
    subprocess.run(list(map(str, command)), cwd=ROOT, check=True)
  File "/usr/lib/python3.12/subprocess.py", line 571, in run
    raise CalledProcessError(retcode, process.args,
subprocess.CalledProcessError: Command '['/mnt/data/zeinab/raptor/.venv/bin/python3', '-m', 'saadi.cli.main', 'run-experiment', '--config', '/mnt/data/zeinab/SAADI/outputs/probe_bridge/novelhopqa_probe/runs/probe_bridge/greedy_seed42/af50299000d1f31f/runtime_config.yaml', '--ingest_only']' returned non-zero exit status 1.
```

## Files written

- `/mnt/data/zeinab/SAADI/configs/experiments/novelhopqa_probe_retrieval_ablation.yaml  (declared file outside the probe folders)`
