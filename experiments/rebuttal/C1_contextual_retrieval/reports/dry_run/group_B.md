# C1 group B (dry run)

- Started 2026-11-09 14:44:38, finished 2026-11-09 14:45:15; host LIGHT-C026225M.local; exit code 0.
- Command: `c1_contextual_retrieval.py --group B --dry-run`

## Preflight

- OK: True. Python 3.14.8 (/Users/uem-stag-lap-ic/Desktop/SAADI/venv.nosync/bin/python3).
- Modules: numpy 2.5.4, pandas 3.1.1, pyarrow 26.0.1, transformers 4.57.6.
- Commit e2dbaa178fa07c25806c32e8fce787455bae9ce3 on fix/notebook-key-and-install; uncommitted code changes: ['?? experiments/rebuttal/C1_contextual_retrieval/env.sh', '?? experiments/rebuttal/C1_contextual_retrieval/run_gpus_0123.sh', '?? experiments/rebuttal/C1_contextual_retrieval/run_gpus_4567.sh', '?? scripts/rebuttal/c1_contextual_retrieval.py', '?? src/saadi/rebuttal/contextual_retrieval.py'].
- GPUs: skipped (CPU dry run).
- Disk: 238.6 GB free at /Users/uem-stag-lap-ic/Desktop/SAADI/outputs/rebuttal/C1_contextual_retrieval/dry_run.
- Models: Qwen/Qwen3-30B-A3B-Instruct-2507: in cache (tokenizer/config only, no weights).
- C2 streams (memory and summary channels): {'musique_32k__bm25': 'recomputed (published path)', 'qasper_64k__bm25': 'recomputed (published path)'}.

## Census and expected run time (written before generation)

| Dataset | Docs | Chunks | Doc tokens min / median / max | Docs over window (k) | Nominal input tokens | Generation | Retrieval | Dataset total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MuSiQue-32K | 2 | 771 | 49K / 49K / 50K | 0 ({'1': 2}) | 38.2M | 15 min | 7s | 20 min |
| QASPER-64K | 2 | 1488 | 89K / 89K / 89K | 0 ({'1': 2}) | 133.2M | 31 min | 43s | 37 min |
- Generation basis: per call 9.3e-06 s x visible document tokens + 0.15 s, plus 420 s engine start (B2's measured rate on the same model and runtime). Retrieval: C2's embedding cost model; dense jobs run in parallel on the group's GPUs.

## Datasets

- MuSiQue-32K: done
- QASPER-64K: done
