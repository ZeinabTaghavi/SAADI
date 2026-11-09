# probe_baselines - stage 04_b3_hipporag

Status: **FAILED** - 2026-11-09 11:12:23; wall clock 0.1 min

## What was done

- `/home/iataghav/data/SAADI/experiments/probe_baselines/third_party/venv_hipporag/bin/python /home/iataghav/data/SAADI/experiments/probe_baselines/systems/b3_hipporag/build.py`
- git commit `f1a54482c68d91eb797d75b11362f1973bc202a5`; python 3.12.3 on Linux-6.17.0-23-generic-x86_64-with-glibc2.39

## Numbers

(no tables from this stage)

## Anything unexpected

`/home/iataghav/data/SAADI/experiments/probe_baselines/third_party/venv_hipporag/bin/python /home/iataghav/data/SAADI/experiments/probe_baselines/systems/b3_hipporag/build.py` exited with code 1. Its traceback is in the log lines below.
Last 50 log lines:

```
/home/iataghav/data/SAADI/experiments/probe_baselines/third_party/venv_hipporag/lib/python3.12/site-packages/transformers/utils/hub.py:128: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
A new version of the following files was downloaded from https://huggingface.co/nvidia/NV-Embed-v2:
- configuration_nvembed.py
. Make sure to double-check they do not contain any added malicious code. To avoid downloading new versions of the code file, you can pin a revision.
Encountered exception while importing datasets: pyarrow requires NumPy 2.0 or newer, found 1.26.4
Traceback (most recent call last):
  File "/home/iataghav/data/SAADI/experiments/probe_baselines/systems/b3_hipporag/build.py", line 223, in <module>
    raise SystemExit(main())
                     ^^^^^^
  File "/home/iataghav/data/SAADI/experiments/probe_baselines/systems/b3_hipporag/build.py", line 149, in main
    h = make_hipporag(book_id)
        ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/iataghav/data/SAADI/experiments/probe_baselines/systems/b3_hipporag/build.py", line 98, in make_hipporag
    h = HippoRAG(save_dir=str(save_dir(book_id)), llm_model_name=MODEL, llm_base_url=URL, embedding_model_name=EMBEDDING)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/iataghav/data/SAADI/experiments/probe_baselines/third_party/venv_hipporag/lib/python3.12/site-packages/hipporag/HippoRAG.py", line 136, in __init__
    self.embedding_model: BaseEmbeddingModel = _get_embedding_model_class(
                                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/iataghav/data/SAADI/experiments/probe_baselines/third_party/venv_hipporag/lib/python3.12/site-packages/hipporag/embedding_model/NVEmbedV2.py", line 30, in __init__
    self.embedding_model = AutoModel.from_pretrained(**self.embedding_config.model_init_params)
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/iataghav/data/SAADI/experiments/probe_baselines/third_party/venv_hipporag/lib/python3.12/site-packages/transformers/models/auto/auto_factory.py", line 553, in from_pretrained
    model_class = get_class_from_dynamic_module(
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/iataghav/data/SAADI/experiments/probe_baselines/third_party/venv_hipporag/lib/python3.12/site-packages/transformers/dynamic_module_utils.py", line 540, in get_class_from_dynamic_module
    final_module = get_cached_module_file(
                   ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/iataghav/data/SAADI/experiments/probe_baselines/third_party/venv_hipporag/lib/python3.12/site-packages/transformers/dynamic_module_utils.py", line 365, in get_cached_module_file
    modules_needed = check_imports(resolved_module_file)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/iataghav/data/SAADI/experiments/probe_baselines/third_party/venv_hipporag/lib/python3.12/site-packages/transformers/dynamic_module_utils.py", line 185, in check_imports
    importlib.import_module(imp)
  File "/usr/lib/python3.12/importlib/__init__.py", line 90, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen importlib._bootstrap>", line 1387, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1331, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 935, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 995, in exec_module
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "/home/iataghav/data/SAADI/experiments/probe_baselines/third_party/venv_hipporag/lib/python3.12/site-packages/datasets/__init__.py", line 17, in <module>
    from .arrow_dataset import Dataset
  File "/home/iataghav/data/SAADI/experiments/probe_baselines/third_party/venv_hipporag/lib/python3.12/site-packages/datasets/arrow_dataset.py", line 57, in <module>
    import pyarrow as pa
  File "/home/iataghav/data/SAADI/experiments/probe_baselines/third_party/venv_hipporag/lib/python3.12/site-packages/pyarrow/__init__.py", line 59, in <module>
    from pyarrow.lib import (BuildInfo, CppBuildInfo, RuntimeInfo, set_timezone_db_path,
  File "pyarrow/lib.pyx", line 42, in init pyarrow.lib
ImportError: pyarrow requires NumPy 2.0 or newer, found 1.26.4
```

## Files written

- (see the stage's output folder)
