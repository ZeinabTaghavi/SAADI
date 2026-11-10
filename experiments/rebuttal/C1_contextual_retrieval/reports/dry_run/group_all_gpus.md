# C1 group all_gpus (dry run)

- Started 2026-11-10 10:08:09, finished 2026-11-10 10:08:53; host LIGHT-C026225M.local; exit code 0.
- Command: `c1_contextual_retrieval.py --all-gpus --dry-run`

## Preflight

- OK: True. Python 3.14.8 (/Users/uem-stag-lap-ic/Desktop/SAADI/venv.nosync/bin/python3).
- Modules: numpy 2.5.4, pandas 3.1.1, pyarrow 26.0.1, transformers 5.22.2.
- Commit 7274ab6a2a49d9444aaa3597c1ebb9378773d22a on fix/notebook-key-and-install; uncommitted code changes: ['M scripts/rebuttal/c1_contextual_retrieval.py', ' M src/saadi/rebuttal/contextual_retrieval.py'].
- GPUs: skipped (CPU dry run).
- Disk: 219.7 GB free at /Users/uem-stag-lap-ic/Desktop/SAADI/outputs/rebuttal/C1_contextual_retrieval/dry_run.
- Models: Qwen/Qwen3-30B-A3B-Instruct-2507: in cache (tokenizer/config only, no weights).
- C2 streams (memory and summary channels): {'loogle__bm25': 'recomputed (published path)', 'musique_32k__bm25': 'recomputed (published path)', 'qasper_64k__bm25': 'recomputed (published path)', 'novelhopqa__bm25': 'recomputed (published path)'}.

## Census and expected run time (written before generation)

| Dataset | Docs | Chunks | Doc tokens min / median / max | Docs over window (k) | Nominal input tokens | Generation | Retrieval | Dataset total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LooGLE | 2 | 56 | 15K / 17K / 19K | 0 ({'1': 2}) | 1.0M | 7 min | 0s | 12 min |
| QASPER-64K | 2 | 1488 | 89K / 89K / 89K | 0 ({'1': 2}) | 133.2M | 31 min | 43s | 37 min |
| NovelHopQA | 2 | 1114 | 242K / 352K / 463K | 1 ({'1': 1, '2': 1}) | 262.5M | 50 min | 38s | 56 min |
- Generation basis: per call 9.3e-06 s x visible document tokens + 0.15 s, plus 420 s engine start (B2's measured rate on the same model and runtime). Retrieval: C2's embedding cost model; dense jobs run in parallel on the group's GPUs.

## Contexts: remaining segments over two engines

| Engine | GPUs | Datasets | Segments | Contexts | Estimate | Measured (s) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | [0, 1, 2, 3] | NovelHopQA | 1 | 385 | 23 min | 4.1 |
| 1 | [4, 5, 6, 7] | NovelHopQA, QASPER-64K | 2 | 916 | 26 min | 6.6 |

## Retrieval tasks (all eight GPUs)

| Task | Status | GPUs | Time |
| --- | --- | --- | --- |
| qasper_64k__bm25__shard1of2 | DONE | CPU | 4s |
| qasper_64k__bm25__shard0of2 | DONE | CPU | 6s |
| qasper_64k__bm25 | DONE | CPU | 4s |
| novelhopqa__bm25__shard0of2 | DONE | CPU | 10s |
| novelhopqa__bm25__shard1of2 | DONE | CPU | 4s |
| novelhopqa__bm25 | DONE | CPU | 2s |
| loogle__bm25__shard0of2 | DONE | CPU | 2s |
| loogle__bm25__shard1of2 | DONE | CPU | 2s |
| loogle__bm25 | DONE | CPU | 2s |

## Datasets

- LooGLE: done
- QASPER-64K: done
- NovelHopQA: done
