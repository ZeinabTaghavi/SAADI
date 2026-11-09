# C1 group A

- Started 2026-11-09 15:55:30, finished —; host eee-s132-infsys-gpu03; exit code None.
- Command: `c1_contextual_retrieval.py --group A`

## Preflight

- OK: True. Python 3.12.3 (/mnt/data/zeinab/raptor/.venv/bin/python3).
- Modules: numpy 2.3.5, pandas 3.0.1, pyarrow 23.0.1, transformers 4.57.6, torch 2.11.0+cu129, vllm 0.23.0, accelerate 1.13.0.
- Commit 270da36f7ad219cdbc6dd90df76ef2d4763f9275 on fix/notebook-key-and-install; uncommitted code changes: none.
- GPUs: 0 NVIDIA RTX A6000 1/49140 MiB used; 1 NVIDIA RTX A6000 1/49140 MiB used; 2 NVIDIA RTX A6000 1/49140 MiB used; 3 NVIDIA RTX A6000 1/49140 MiB used.
- Disk: 4285.4 GB free at /mnt/data/zeinab/SAADI/outputs/rebuttal/C1_contextual_retrieval.
- Models: Qwen/Qwen3-30B-A3B-Instruct-2507: in cache (weights present), facebook/contriever: in cache (weights present), BAAI/bge-m3: in cache (weights present), jinaai/jina-embeddings-v3: in cache (weights present), Qwen/Qwen3-Embedding-8B: in cache (weights present).
- C2 streams (memory and summary channels): {'loogle__bm25': 'reused', 'loogle__contriever': 'reused', 'loogle__bge_m3': 'reused', 'loogle__jina': 'reused', 'loogle__qwen': 'recomputed (published path)', 'novelhopqa__bm25': 'reused', 'novelhopqa__contriever': 'reused', 'novelhopqa__bge_m3': 'reused', 'novelhopqa__jina': 'reused', 'novelhopqa__qwen': 'reused'}.

## Census and expected run time (written before generation)

| Dataset | Docs | Chunks | Doc tokens min / median / max | Docs over window (k) | Nominal input tokens | Generation | Retrieval | Dataset total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LooGLE | 105 | 3346 | 14K / 19K / 60K | 0 ({'1': 105}) | 86.0M | 28 min | 2.1 h | 2.6 h |
| NovelHopQA | 16 | 7011 | 134K / 234K / 763K | 7 ({'1': 9, '2': 6, '3': 1}) | 1.43B | 4.1 h | 3.1 h | 7.3 h |
- Generation basis: per call 9.3e-06 s x visible document tokens + 0.15 s, plus 420 s engine start (B2's measured rate on the same model and runtime). Retrieval: C2's embedding cost model; dense jobs run in parallel on the group's GPUs.

## Memory check

| Model | GPUs | Batch | Peak MiB | Fits |
| --- | --- | --- | --- | --- |
| Contriever | 1 | 128 | [3306] | True |
| BGE-M3 | 1 | 16 | [4004] | True |
| Jina | 1 | 64 | [2082] | True |
| Qwen | 2 | 64 | [27106, 28446] | True |

## Datasets

