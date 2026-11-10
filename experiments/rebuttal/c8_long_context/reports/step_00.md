# C8 long-context reader - step 00 (prepare): FAILED (unexpected error)

Written 2026-11-10 23:54:32; commit 479ce2c426bd; host eee-s132-infsys-gpu03; 4 s. CPU only, no model was called.

Command:

```bash
bash experiments/rebuttal/c8_long_context/sh/00_prepare.sh 
```

## Summary

- Unexpected error: OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.

## Long-context model

`Qwen/Qwen2.5-7B-Instruct-1M`: in the Hugging Face cache: True (/mnt/cache/taghavi/hub/models--Qwen--Qwen2.5-7B-Instruct-1M/snapshots/e28526f7bb80e2a9c8af03b831a9af3812f18fba).
Tokenizer used for every count: `Qwen/Qwen2.5-7B-Instruct-1M`.

Shape: 28 layers, 28 query heads, 4 KV heads, head dim 128, weights 15.2 GB (bfloat16); KV cache 56 KiB per token; max position 1,010,000; native window 262,144 tokens (longer inputs need dual chunk attention: {"chunk_size": 262144, "local_size": 8192, "original_max_position_embeddings": 262144}).

## Error

```text
Traceback (most recent call last):
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/transformers/utils/hub.py", line 479, in cached_files
    hf_hub_download(
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/huggingface_hub/utils/_validators.py", line 114, in _inner_fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/huggingface_hub/file_download.py", line 1014, in hf_hub_download
    return _hf_hub_download_to_cache_dir(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/huggingface_hub/file_download.py", line 1121, in _hf_hub_download_to_cache_dir
    _raise_on_head_call_error(head_call_error, force_download, local_files_only)
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/huggingface_hub/file_download.py", line 1653, in _raise_on_head_call_error
    raise LocalEntryNotFoundError(
huggingface_hub.errors.LocalEntryNotFoundError: Cannot find the requested files in the disk cache and outgoing traffic has been disabled. To enable hf.co look-ups and downloads online, set 'local_files_only' to False.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/iataghav/data/SAADI/experiments/rebuttal/c8_long_context/scripts/step00_prepare.py", line 566, in main
    tok = s0.Tok(tok_name)
          ^^^^^^^^^^^^^^^^
  File "/home/iataghav/data/SAADI/experiments/rebuttal/c5_end_to_end/scripts/step00_prepare.py", line 644, in __init__
    self.tok = AutoTokenizer.from_pretrained(model, local_files_only=os.environ.get("HF_HUB_OFFLINE", "1") == "1")
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/transformers/models/auto/tokenization_auto.py", line 1109, in from_pretrained
    config = AutoConfig.from_pretrained(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/transformers/models/auto/configuration_auto.py", line 1332, in from_pretrained
    config_dict, unused_kwargs = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/transformers/configuration_utils.py", line 662, in get_config_dict
    config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/transformers/configuration_utils.py", line 721, in _get_config_dict
    resolved_config_file = cached_file(
                           ^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/transformers/utils/hub.py", line 322, in cached_file
    file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/transformers/utils/hub.py", line 553, in cached_files
    raise OSError(
OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.
```

## Next

```bash
nohup bash experiments/rebuttal/c8_long_context/sh/run_gpus_4567.sh > /dev/null 2>&1 &
```

(after C5's run on GPUs 4-7 has finished)

## Files written

- `experiments/rebuttal/c8_long_context/reports/step_00.md`
- `experiments/rebuttal/c8_long_context/reports/step_00.json`
