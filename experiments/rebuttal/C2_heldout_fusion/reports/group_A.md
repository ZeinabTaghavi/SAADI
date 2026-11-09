# C2 group A report

- Started 2026-11-09 12:30:24, finished 2026-11-09 12:57:30; host eee-s132-infsys-gpu03; exit code 1.
- Command: `python3 scripts/rebuttal/c2_heldout_fusion.py --group A`

## Preflight

- OK: True. Python 3.12.3 (/mnt/data/zeinab/raptor/.venv/bin/python3).
- Modules: numpy 2.3.5, pandas 3.0.1, pyarrow 23.0.1, tqdm 4.68.2, torch 2.11.0+cu129, transformers 4.57.6, matplotlib 3.10.8.
- Commit 1ecb54c278bdfd82c731f1aea140d25768497d63 on fix/notebook-key-and-install; uncommitted code changes: none.
- GPUs: 0 NVIDIA RTX A6000 4/49140 MiB used; 1 NVIDIA RTX A6000 4/49140 MiB used; 2 NVIDIA RTX A6000 4/49140 MiB used; 3 NVIDIA RTX A6000 4/49140 MiB used.
- Disk: 4286.0 GB free at /mnt/data/zeinab/SAADI/outputs/rebuttal/C2_heldout_fusion (estimate needed 6.7 GB).
- Models: BAAI/bge-m3: in cache, facebook/contriever: in cache, jinaai/jina-embeddings-v3: in cache, Qwen/Qwen3-Embedding-8B: in cache.

## Plan

- plan.json written by this group at 2026-11-09 12:34:30.
- Cost basis: estimated from model sizes (no timing run): seconds = model load + (estimated document tokens x 2 x parameters) / effective FLOP/s + 5 query encodings per query x per-call latency + CPU combination time. weights in float32 (the adapters pass no dtype); every model fits on one 48 GB A6000 (largest: Qwen3-Embedding-8B, about 28 GB of weights), so every job gets one GPU.
- This group: GPUs [0, 1, 2, 3]; estimated wall time 2.3 h (other group: 2.2 h); analysis afterwards about 12 min.

| Job | Device | Texts to embed | Queries | Estimate |
| --- | --- | --- | --- | --- |
| qasper_64k__qwen | gpu x1 | 168248 | 1333 | 2.3 h |
| loogle__qwen | gpu x1 | 55069 | 1832 | 77 min |
| qasper_64k__jina | gpu x1 | 168248 | 1333 | 15 min |
| qasper_64k__bge_m3 | gpu x1 | 168248 | 1333 | 14 min |
| novelhopqa__bge_m3 | gpu x1 | 80518 | 908 | 14 min |
| musique_32k__jina | gpu x1 | 221446 | 900 | 13 min |
| musique_32k__bge_m3 | gpu x1 | 221446 | 900 | 12 min |
| loogle__jina | gpu x1 | 55069 | 1832 | 9 min |
| musique_32k__contriever | gpu x1 | 221446 | 900 | 4 min |
| qasper_64k__bm25 | cpu x1 | 168248 | 1333 | 2 min |
| novelhopqa__bm25 | cpu x1 | 80518 | 908 | 63s |

## Jobs and reproduction checks

| Job | Status | Time | Identical top-10 | HR@5 / HR@10 ours vs saved | text-only vs Table 1 Vanilla | Problems |
| --- | --- | --- | --- | --- | --- | --- |
| novelhopqa__bm25 | DONE | 2 min | 908/908 | 18.06 / 22.03 vs 18.06 / 22.03 | 17.51 / 20.70 vs 16.9 / 20.0 (differs) | none |
| qasper_64k__bm25 | DONE | 4 min | 1333/1333 | 16.95 / 26.11 vs 16.95 / 26.11 | 14.40 / 22.06 vs 14.4 / 22.1 | none |
| loogle__qwen | ERROR | 7 min | — | — | — | rch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)  |
| qasper_64k__jina | FAILED | 9 min | 927/1333 | 18.83 / 27.08 vs 18.90 / 27.16 | 15.98 / 24.98 vs 15.7 / 24.8 (differs) | identical top-10 for 927/1333 queries (< 99%) |
| qasper_64k__qwen | ERROR | 14 min | — | — | — | rch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)  |
| qasper_64k__bge_m3 | DONE | 16 min | 1333/1333 | 20.93 / 30.91 vs 20.93 / 30.91 | 21.01 / 29.33 vs 21.0 / 29.3 | none |
| musique_32k__jina | FAILED | 7 min | 705/900 | 73.56 / 88.11 vs 73.22 / 88.11 | 71.67 / 85.78 vs 71.6 / 85.8 | identical top-10 for 705/900 queries (< 99%); HR@5 73.55555555555556 vs saved 73.22222222222223 (tolerance 0.1 points); HR@5 73.56 vs expected 73.22 |
| novelhopqa__bge_m3 | DONE | 11 min | 908/908 | 16.52 / 22.14 vs 16.52 / 22.14 | 17.29 / 21.92 vs 14.1 / 20.7 (differs) | none |
| loogle__jina | FAILED | 4 min | 1519/1832 | 70.20 / 81.99 vs 70.14 / 81.99 | 69.92 / 80.68 vs 70.1 / 80.1 (differs) | identical top-10 for 1519/1832 queries (< 99%) |
| musique_32k__contriever | DONE | 6 min | 900/900 | 68.78 / 83.67 vs 68.78 / 83.67 | 59.89 / 76.11 vs 59.9 / 76.1 | none |
| musique_32k__bge_m3 | DONE | 9 min | 900/900 | 76.89 / 88.33 vs 76.89 / 88.33 | 73.22 / 87.44 vs 73.2 / 87.4 | none |

- novelhopqa__bm25 text-only note: text-only HR@5 17.51 vs Table 1 Vanilla 16.9 (tolerance 0.05); text-only HR@10 20.70 vs Table 1 Vanilla 20.0 (tolerance 0.05)
- qasper_64k__jina text-only note: text-only HR@5 15.98 vs Table 1 Vanilla 15.7 (tolerance 0.15); text-only HR@10 24.98 vs Table 1 Vanilla 24.8 (tolerance 0.15)
- novelhopqa__bge_m3 text-only note: text-only HR@5 17.29 vs Table 1 Vanilla 14.1 (tolerance 0.15); text-only HR@10 21.92 vs Table 1 Vanilla 20.7 (tolerance 0.15)
- loogle__jina text-only note: text-only HR@5 69.92 vs Table 1 Vanilla 70.1 (tolerance 0.15); text-only HR@10 80.68 vs Table 1 Vanilla 80.1 (tolerance 0.15)
- qasper_64k__jina mismatch example: `{"query_id": "0", "doc_id": "group_013", "saved_top10": ["group_013:393", "group_013:666", "group_013:588", "group_013:620", "group_013:646", "group_013:649", "group_013:667", "group_013:674", "group_013:648", "group_013:587"], "saved_scores": [3.1575028777292404, 2.4763050977650116, 2.437243459976009, 2.401405414195527, 2.2543814981402908, 2.2211353726584315, 2.0261689462579158, 2.020767488795535, 1.9959292766750774, 1.9760035275122634], "ours_top10": ["group_013:393", "group_013:666", "group_013:588", "group_013:620", "group_013:646", "group_013:649", "group_013:674", "group_013:667", "group`
- musique_32k__jina mismatch example: `{"query_id": "1210", "doc_id": "group_036", "saved_top10": ["group_036:52", "group_036:53", "group_036:71", "group_036:47", "group_036:69", "group_036:46", "group_036:48", "group_036:51", "group_036:54", "group_036:68"], "saved_scores": [3.7107016045278547, 3.6867404738227525, 3.4323317243572427, 3.3528860335514223, 3.2612530159506496, 3.2570415107310136, 3.0707236839503023, 3.0121729572153293, 2.9640100468739394, 2.944635767534626], "ours_top10": ["group_036:52", "group_036:53", "group_036:71", "group_036:47", "group_036:46", "group_036:69", "group_036:48", "group_036:51", "group_036:54", "gr`
- loogle__jina mismatch example: `{"query_id": "13", "doc_id": "5984f759-6d02-4c94-858c-b4ca561ca5e5", "saved_top10": ["5984f759-6d02-4c94-858c-b4ca561ca5e5:20", "5984f759-6d02-4c94-858c-b4ca561ca5e5:7", "5984f759-6d02-4c94-858c-b4ca561ca5e5:9", "5984f759-6d02-4c94-858c-b4ca561ca5e5:1", "5984f759-6d02-4c94-858c-b4ca561ca5e5:17", "5984f759-6d02-4c94-858c-b4ca561ca5e5:3", "5984f759-6d02-4c94-858c-b4ca561ca5e5:8", "5984f759-6d02-4c94-858c-b4ca561ca5e5:18", "5984f759-6d02-4c94-858c-b4ca561ca5e5:15", "5984f759-6d02-4c94-858c-b4ca561ca5e5:14"], "saved_scores": [1.549544564247661, 1.1095411211860697, 0.8306690736239161, 0.63507694955`

## Unexpected

```
^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/transformers/utils/generic.py", line 1072, in wrapper
    outputs = func(self, *args, **kwargs)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py", line 410, in forward
    hidden_states = decoder_layer(
                    ^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/transformers/modeling_layers.py", line 94, in __call__
    return super().__call__(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/transformers/utils/deprecation.py", line 172, in wrapped_func
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py", line 275, in forward
    hidden_states = self.mlp(hidden_states)
                    ^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py", line 82, in forward
    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
                               ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.50 GiB. GPU 0 has a total capacity of 47.40 GiB of which 294.25 MiB is free. Including non-PyTorch memory, this process has 47.11 GiB memory in use. Of the allocated memory 41.70 GiB is allocated by PyTorch, and 5.10 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)

```
```
^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/transformers/utils/generic.py", line 1072, in wrapper
    outputs = func(self, *args, **kwargs)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py", line 410, in forward
    hidden_states = decoder_layer(
                    ^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/transformers/modeling_layers.py", line 94, in __call__
    return super().__call__(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/transformers/utils/deprecation.py", line 172, in wrapped_func
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py", line 275, in forward
    hidden_states = self.mlp(hidden_states)
                    ^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py", line 82, in forward
    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
                               ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.38 GiB. GPU 0 has a total capacity of 47.40 GiB of which 478.25 MiB is free. Including non-PyTorch memory, this process has 46.93 GiB memory in use. Of the allocated memory 43.01 GiB is allocated by PyTorch, and 3.61 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)

```

## Analysis

- 12/20 jobs DONE; this group's unfinished jobs: ['loogle__jina', 'loogle__qwen', 'musique_32k__jina', 'qasper_64k__jina', 'qasper_64k__qwen']; waiting for group B

## Files written

- `outputs/rebuttal/C2_heldout_fusion/plan.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/novelhopqa__bm25.parquet`
- `outputs/rebuttal/C2_heldout_fusion/streams/novelhopqa__bm25.queries.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/novelhopqa__bm25.check.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/qasper_64k__bm25.parquet`
- `outputs/rebuttal/C2_heldout_fusion/streams/qasper_64k__bm25.queries.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/qasper_64k__bm25.check.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/qasper_64k__bge_m3.parquet`
- `outputs/rebuttal/C2_heldout_fusion/streams/qasper_64k__bge_m3.queries.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/qasper_64k__bge_m3.check.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/novelhopqa__bge_m3.parquet`
- `outputs/rebuttal/C2_heldout_fusion/streams/novelhopqa__bge_m3.queries.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/novelhopqa__bge_m3.check.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/musique_32k__contriever.parquet`
- `outputs/rebuttal/C2_heldout_fusion/streams/musique_32k__contriever.queries.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/musique_32k__contriever.check.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/musique_32k__bge_m3.parquet`
- `outputs/rebuttal/C2_heldout_fusion/streams/musique_32k__bge_m3.queries.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/musique_32k__bge_m3.check.json`
- `experiments/rebuttal/C2_heldout_fusion/reports/group_A.md`
- `experiments/rebuttal/C2_heldout_fusion/reports/group_A.json`
