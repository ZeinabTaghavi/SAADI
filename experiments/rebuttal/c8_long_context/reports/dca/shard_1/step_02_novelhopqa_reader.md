# C8 step 04 (dual chunk attention) - unit 1, vLLM start: FAILED

Written 2026-11-11 22:37:19 on eee-s132-infsys-gpu03. The main C8 results are unchanged.

The vLLM 0.10.1.1 server with dual chunk attention did not start on GPUs 4,5,6,7 (log outputs/rebuttal/c8_long_context/vllm_logs/server_dca_g1_20261111_222946_223648.log).

Last 60 lines of experiments/rebuttal/c8_long_context/logs/run_dca_long_books_20261111_222946.log:

```
                     ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/iataghav/data/SAADI/outputs/rebuttal/c8_long_context/venv_vllm_0_10_1_1/lib/python3.12/site-packages/vllm/engine/llm_engine.py", line 522, in _init_tokenizer
    return init_tokenizer_from_configs(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/iataghav/data/SAADI/outputs/rebuttal/c8_long_context/venv_vllm_0_10_1_1/lib/python3.12/site-packages/vllm/transformers_utils/tokenizer_group.py", line 121, in init_tokenizer_from_configs
    return TokenizerGroup(
           ^^^^^^^^^^^^^^^
  File "/home/iataghav/data/SAADI/outputs/rebuttal/c8_long_context/venv_vllm_0_10_1_1/lib/python3.12/site-packages/vllm/transformers_utils/tokenizer_group.py", line 26, in __init__
    self.tokenizer = get_tokenizer(self.tokenizer_id, **tokenizer_config)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/iataghav/data/SAADI/outputs/rebuttal/c8_long_context/venv_vllm_0_10_1_1/lib/python3.12/site-packages/vllm/transformers_utils/tokenizer.py", line 284, in get_tokenizer
    tokenizer = get_cached_tokenizer(tokenizer)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/iataghav/data/SAADI/outputs/rebuttal/c8_long_context/venv_vllm_0_10_1_1/lib/python3.12/site-packages/vllm/transformers_utils/tokenizer.py", line 101, in get_cached_tokenizer
    tokenizer.all_special_tokens_extended)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/iataghav/data/SAADI/outputs/rebuttal/c8_long_context/venv_vllm_0_10_1_1/lib/python3.12/site-packages/transformers/tokenization_utils_base.py", line 1335, in __getattr__
    raise AttributeError(f"{self.__class__.__name__} has no attribute {key}")
AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended. Did you mean: 'num_special_tokens_to_add'?
[c8-dca] 2026-11-11 22:37:19 [group 1] vLLM exited during start-up; last lines:
Traceback (most recent call last):
  File "/usr/lib/python3.12/multiprocessing/process.py", line 314, in _bootstrap
    self.run()
  File "/usr/lib/python3.12/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/home/iataghav/data/SAADI/outputs/rebuttal/c8_long_context/venv_vllm_0_10_1_1/lib/python3.12/site-packages/vllm/engine/multiprocessing/engine.py", line 469, in run_mp_engine
    raise e from None
  File "/home/iataghav/data/SAADI/outputs/rebuttal/c8_long_context/venv_vllm_0_10_1_1/lib/python3.12/site-packages/vllm/engine/multiprocessing/engine.py", line 455, in run_mp_engine
    engine = MQLLMEngine.from_vllm_config(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/iataghav/data/SAADI/outputs/rebuttal/c8_long_context/venv_vllm_0_10_1_1/lib/python3.12/site-packages/vllm/utils/__init__.py", line 1557, in inner
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/home/iataghav/data/SAADI/outputs/rebuttal/c8_long_context/venv_vllm_0_10_1_1/lib/python3.12/site-packages/vllm/engine/multiprocessing/engine.py", line 144, in from_vllm_config
    return cls(
           ^^^^
  File "/home/iataghav/data/SAADI/outputs/rebuttal/c8_long_context/venv_vllm_0_10_1_1/lib/python3.12/site-packages/vllm/engine/multiprocessing/engine.py", line 88, in __init__
    self.engine = LLMEngine(*args, **kwargs)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/iataghav/data/SAADI/outputs/rebuttal/c8_long_context/venv_vllm_0_10_1_1/lib/python3.12/site-packages/vllm/engine/llm_engine.py", line 238, in __init__
    self.tokenizer = self._init_tokenizer()
                     ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/iataghav/data/SAADI/outputs/rebuttal/c8_long_context/venv_vllm_0_10_1_1/lib/python3.12/site-packages/vllm/engine/llm_engine.py", line 522, in _init_tokenizer
    return init_tokenizer_from_configs(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/iataghav/data/SAADI/outputs/rebuttal/c8_long_context/venv_vllm_0_10_1_1/lib/python3.12/site-packages/vllm/transformers_utils/tokenizer_group.py", line 121, in init_tokenizer_from_configs
    return TokenizerGroup(
           ^^^^^^^^^^^^^^^
  File "/home/iataghav/data/SAADI/outputs/rebuttal/c8_long_context/venv_vllm_0_10_1_1/lib/python3.12/site-packages/vllm/transformers_utils/tokenizer_group.py", line 26, in __init__
    self.tokenizer = get_tokenizer(self.tokenizer_id, **tokenizer_config)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/iataghav/data/SAADI/outputs/rebuttal/c8_long_context/venv_vllm_0_10_1_1/lib/python3.12/site-packages/vllm/transformers_utils/tokenizer.py", line 284, in get_tokenizer
    tokenizer = get_cached_tokenizer(tokenizer)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/iataghav/data/SAADI/outputs/rebuttal/c8_long_context/venv_vllm_0_10_1_1/lib/python3.12/site-packages/vllm/transformers_utils/tokenizer.py", line 101, in get_cached_tokenizer
    tokenizer.all_special_tokens_extended)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/iataghav/data/SAADI/outputs/rebuttal/c8_long_context/venv_vllm_0_10_1_1/lib/python3.12/site-packages/transformers/tokenization_utils_base.py", line 1335, in __getattr__
    raise AttributeError(f"{self.__class__.__name__} has no attribute {key}")
AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended. Did you mean: 'num_special_tokens_to_add'?
```
