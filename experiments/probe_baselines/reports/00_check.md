# probe_baselines - stage 00_check

Status: **ok** - 2026-11-08 22:54:13; wall clock 0.0 min

## What was done

- `python3 /home/iataghav/data/SAADI/experiments/probe_baselines/common/check.py`
- git commit `28eed3a1ee31487f3bf23dcb3470aa0e164e8cfd`; python 3.12.3 on Linux-6.17.0-23-generic-x86_64-with-glibc2.39

## Numbers

### Checks

| check | value | result |
| --- | --- | --- |
| frozen spec (personas.jsonl SHA-256) | `11c4c1696c67171f78821d3ba8311ccd28c851315ca1c2a404a9a422593897bb` | ok |
| personas / cells / books | 240 / [1, 2, 3, 4, 5, 6, 7, 8] / 10 | ok |
| chunks per book equal SAADI's builds | B31 244, B43 246, B37 262, B18 312, B04 321, B34 335, B27 338, B61 357, B26 384, B44 385 | ok |
| probe exported for the third-party venvs | /home/iataghav/data/SAADI/outputs/probe_baselines/inputs | ok |
| vLLM server | http://127.0.0.1:8047/v1, model Qwen/Qwen3-30B-A3B-Instruct-2507, vLLM 0.23.0 | ok |
| served window (max_model_len) | 262144 | ok |
| test call | 'ready' | ok |
| scorer self-test: run 1's per_persona.jsonl against results/method_update/p1_ledger/table.csv | 32 cell-method rows | PASSED |
| B3 hipporag venv imports | ModuleNotFoundError: No module named 'hipporag' | FAILED: B3 will fail |
| NV-Embed-v2 weights in the HF cache | ModuleNotFoundError: No module named 'huggingface_hub' | FAILED: B3 will fail |
| LQCA clone at the pinned commit | /home/iataghav/data/SAADI/experiments/probe_baselines/third_party/LQCA @ 2d05899479d5 | ok |
| B4 venv: maverick-coref, Maverick checkpoint, spaCy, tiktoken | ModuleNotFoundError: No module named 'maverick' | FAILED: B4 will fail |

## Anything unexpected

(check) hipporag import failed (/home/iataghav/data/SAADI/experiments/probe_baselines/third_party/venv_hipporag/bin/python):
```
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'hipporag'
```
(check) NV-Embed-v2 download failed:
```
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'huggingface_hub'
```
(check) LQCA venv check failed (/home/iataghav/data/SAADI/experiments/probe_baselines/third_party/venv_lqca/bin/python):
```
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'maverick'
```

## Files written

- `/home/iataghav/data/SAADI/outputs/probe_baselines/inputs/<book>.json, manifest.json`
