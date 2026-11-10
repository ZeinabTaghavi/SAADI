# C5 end-to-end QA - step 01 RAPTOR re-run, MuSiQue-32K: FAILED

Written 2026-11-10 19:59:07; commit 3fd3c0dca1b3; host eee-s132-infsys-gpu03; GPUs 0,1,2,3; 0 s.

Command: `bash experiments/rebuttal/c5_end_to_end/sh/run_gpus_0123.sh` (stage raptor)

## What was run

- RAPTOR repository `/mnt/data/zeinab/SAADI_Raptor` (not modified), Python `/mnt/data/zeinab/raptor/.venv/bin/python3`.
- Input: SAADI's prepared MuSiQue-32K files copied unchanged into `None`, with the manifest RAPTOR's loader needs (sha256 None).
- Config `None`: RAPTOR's saved musique_32k config; changed only: nothing.
- Settings (saved = RAPTOR defaults except the models): leaf 100 cl100k tokens, 5 layers, threshold 0.5, top_k 5 per cluster, summaries up to 150 tokens, collapsed-tree retrieval top_k 10 with max_tokens 3500, Contriever embeddings, Qwen3-30B-A3B-Instruct-2507 summariser (vLLM in-process, TP 4, temperature 0).
- Run status before: None; after: None.

## Result

| Queries | Without labels | Unmatched | Build time (s, documents) |
| --- | --- | --- | --- |
| — | — | — | null |

## Error

```text
Traceback (most recent call last):
  File "/home/iataghav/data/SAADI/experiments/rebuttal/c5_end_to_end/scripts/step01_raptor_rerun.py", line 134, in main
    raise RuntimeError(f"no RAPTOR copy with the MuSiQue runner among {[str(c) for c in cands]} (missing {missing}); set C5_RAPTOR_CODE_DIR")
RuntimeError: no RAPTOR copy with the MuSiQue runner among ['/mnt/data/zeinab/SAADI_Raptor', '/mnt/data/zeinab/SAADI_Raptor', '/home/iataghav/SAADI_Raptor', '/home/iataghav/data/SAADI_Raptor', '/home/iataghav/data/SAADI/../SAADI_Raptor', '/home/iataghav/data/SAADI/outputs/other_methods_method/SAADI_Raptor'] (missing ['raptor/dataset_loaders/prepared_expanded.py', 'scripts/check_raptor_run_status.py', 'configs/raptor/musique_32k_retrieval_ablation.yaml']); set C5_RAPTOR_CODE_DIR
```

## Files written

- `/home/iataghav/data/SAADI/outputs/rebuttal/c5_end_to_end/raptor_rerun/raptor_10_runs/musique_32k/musique_32k_retrieval_ablation_raptor` (raw RAPTOR run, not pushed)
- `/home/iataghav/data/SAADI/outputs/rebuttal/c5_end_to_end/raptor_rerun/raptor_10_evaluations/musique_32k/musique_32k_retrieval_ablation_raptor/retrieval_upper_bound` (not pushed)
- `experiments/rebuttal/c5_end_to_end/reports/step_01_musique_32k_raptor.md`, `experiments/rebuttal/c5_end_to_end/reports/step_01_musique_32k_raptor.json`

## Push

```
pushed 38a4c4e to origin/fix/notebook-key-and-install: C5 musique_32k 01_raptor: FAILED: RuntimeError: no RAPTOR copy with the MuSiQue runner among ['/mnt/data/zeinab/SAADI_Raptor', '/mnt/data/zeinab/SAADI_Raptor'
```
