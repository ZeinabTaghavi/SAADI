# C8 long-context reader - step 00 (prepare): FAILED

Written 2026-11-10 23:49:02; commit efc463a0d957; host eee-s132-infsys-gpu03; 0 s. CPU only, no model was called.

Command:

```bash
bash experiments/rebuttal/c8_long_context/sh/00_prepare.sh 
```

## Summary

- Stopped: Qwen/Qwen2.5-7B-Instruct-1M is not in the Hugging Face cache (/mnt/cache/taghavi/hub); download it first, e.g. HF_HUB_OFFLINE=0 HF_HOME=/mnt/cache/taghavi huggingface-cli download Qwen/Qwen2.5-7B-Instruct-1M

## Long-context model

`Qwen/Qwen2.5-7B-Instruct-1M`: in the Hugging Face cache: False; error: LocalEntryNotFoundError: Cannot find an appropriate cached snapshot folder for the specified revision on the local disk and outgoing traffic has been disabled. To enable repo look-ups and downloads online, pass 'local_files_o.
Tokenizer used for every count: `Qwen/Qwen2.5-7B-Instruct-1M`.

Shape (assumed from the model card: config not in the cache): 28 layers, 28 query heads, 4 KV heads, head dim 128, weights 15.2 GB (bfloat16); KV cache 56 KiB per token; max position 1,010,000; native window 262,144 tokens (longer inputs need dual chunk attention: {}).

## Error

```text
Traceback (most recent call last):
  File "/home/iataghav/data/SAADI/experiments/rebuttal/c8_long_context/scripts/step00_prepare.py", line 563, in main
    raise GateFailed(f"{model} is not in the Hugging Face cache ({env.get('HF_HUB_CACHE')}); download it first, e.g. "
GateFailed: Qwen/Qwen2.5-7B-Instruct-1M is not in the Hugging Face cache (/mnt/cache/taghavi/hub); download it first, e.g. HF_HUB_OFFLINE=0 HF_HOME=/mnt/cache/taghavi huggingface-cli download Qwen/Qwen2.5-7B-Instruct-1M
```

## Next

```bash
nohup bash experiments/rebuttal/c8_long_context/sh/run_gpus_4567.sh > /dev/null 2>&1 &
```

(after C5's run on GPUs 4-7 has finished)

## Files written

- `experiments/rebuttal/c8_long_context/reports/step_00.md`
- `experiments/rebuttal/c8_long_context/reports/step_00.json`

## Push

```
pushed 479ce2c to origin/fix/notebook-key-and-install: C8 all 00_prepare: FAILED: Qwen/Qwen2.5-7B-Instruct-1M is not in the Hugging Face cache (/mnt/cache/taghavi/hub); download it first, e.g. HF_HUB_OFFLINE=0 HF
```
