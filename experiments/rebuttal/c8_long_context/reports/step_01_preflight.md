# C8 long-context reader - GPU run (vLLM start): FAILED

Written 2026-11-11 00:09:12 on eee-s132-infsys-gpu03; GPUs 4,5,6,7; port 18541.

The vLLM server did not start (logs in outputs/rebuttal/c8_long_context/vllm_logs/).

Last 60 lines of experiments/rebuttal/c8_long_context/logs/run_gpus_4567_20261111_000820.log:

```
[c8] 2026-11-11 00:08:20 host eee-s132-infsys-gpu03; GPUs 4,5,6,7; port 18541; model Qwen/Qwen2.5-7B-Instruct-1M; vLLM python /mnt/data/zeinab/raptor/.venv/bin/python3
[c8] 2026-11-11 00:08:26 vLLM 0.23.0; step 00: max_model_len 778240, dual chunk attention available 0, 382 queries beyond the native window
[c8] 2026-11-11 00:08:42 starting vLLM (native) on GPUs 4,5,6,7, port 18541, --max-model-len 262144; log outputs/rebuttal/c8_long_context/vllm_logs/server_native_20261111_000820.log
[c8] 2026-11-11 00:09:12 vLLM (native) exited during start-up; last lines:
(APIServer pid=528140)   File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/vllm/entrypoints/openai/api_server.py", line 705, in <module>
(APIServer pid=528140)     uvloop.run(run_server(args))
(APIServer pid=528140)   File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/uvloop/__init__.py", line 96, in run
(APIServer pid=528140)     return __asyncio.run(
(APIServer pid=528140)            ^^^^^^^^^^^^^^
(APIServer pid=528140)   File "/usr/lib/python3.12/asyncio/runners.py", line 194, in run
(APIServer pid=528140)     return runner.run(main)
(APIServer pid=528140)            ^^^^^^^^^^^^^^^^
(APIServer pid=528140)   File "/usr/lib/python3.12/asyncio/runners.py", line 118, in run
(APIServer pid=528140)     return self._loop.run_until_complete(task)
(APIServer pid=528140)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=528140)   File "uvloop/loop.pyx", line 1518, in uvloop.loop.Loop.run_until_complete
(APIServer pid=528140)   File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/uvloop/__init__.py", line 48, in wrapper
(APIServer pid=528140)     return await main
(APIServer pid=528140)            ^^^^^^^^^^
(APIServer pid=528140)   File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/vllm/entrypoints/openai/api_server.py", line 665, in run_server
(APIServer pid=528140)     await run_server_worker(listen_address, sock, args, **uvicorn_kwargs)
(APIServer pid=528140)   File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/vllm/entrypoints/openai/api_server.py", line 679, in run_server_worker
(APIServer pid=528140)     async with build_async_engine_client(
(APIServer pid=528140)   File "/usr/lib/python3.12/contextlib.py", line 210, in __aenter__
(APIServer pid=528140)     return await anext(self.gen)
(APIServer pid=528140)            ^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=528140)   File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/vllm/entrypoints/openai/api_server.py", line 99, in build_async_engine_client
(APIServer pid=528140)     async with build_async_engine_client_from_engine_args(
(APIServer pid=528140)   File "/usr/lib/python3.12/contextlib.py", line 210, in __aenter__
(APIServer pid=528140)     return await anext(self.gen)
(APIServer pid=528140)            ^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=528140)   File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/vllm/entrypoints/openai/api_server.py", line 123, in build_async_engine_client_from_engine_args
(APIServer pid=528140)     vllm_config = engine_args.create_engine_config(usage_context=usage_context)
(APIServer pid=528140)                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=528140)   File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/vllm/engine/arg_utils.py", line 2229, in create_engine_config
(APIServer pid=528140)     config = VllmConfig(
(APIServer pid=528140)              ^^^^^^^^^^^
(APIServer pid=528140)   File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/pydantic/_internal/_dataclasses.py", line 121, in __init__
(APIServer pid=528140)     s.__pydantic_validator__.validate_python(ArgsKwargs(args, kwargs), self_instance=s)
(APIServer pid=528140)   File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/vllm/config/vllm.py", line 860, in __post_init__
(APIServer pid=528140)     self.model_config.verify_dual_chunk_attention_config(self.load_config)
(APIServer pid=528140)   File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/vllm/config/model.py", line 1173, in verify_dual_chunk_attention_config
(APIServer pid=528140)     self.hf_config.dual_chunk_attention_config[
(APIServer pid=528140) TypeError: 'NoneType' object does not support item assignment
```
