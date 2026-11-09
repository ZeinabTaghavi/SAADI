# C2 group B report

- Started 2026-11-09 12:30:45, finished 2026-11-09 12:57:20; host eee-s132-infsys-gpu03; exit code 1.
- Command: `python3 scripts/rebuttal/c2_heldout_fusion.py --group B`

## Preflight

- OK: True. Python 3.12.3 (/mnt/data/zeinab/raptor/.venv/bin/python3).
- Modules: numpy 2.3.5, pandas 3.0.1, pyarrow 23.0.1, tqdm 4.68.2, torch 2.11.0+cu129, transformers 4.57.6, matplotlib 3.10.8.
- Commit 1ecb54c278bdfd82c731f1aea140d25768497d63 on fix/notebook-key-and-install; uncommitted code changes: none.
- GPUs: 4 NVIDIA RTX A6000 4/49140 MiB used; 5 NVIDIA RTX A6000 4/49140 MiB used; 6 NVIDIA RTX A6000 4/49140 MiB used; 7 NVIDIA RTX A6000 4/49140 MiB used.
- Disk: 4286.0 GB free at /mnt/data/zeinab/SAADI/outputs/rebuttal/C2_heldout_fusion (estimate needed 5.9 GB).
- Models: BAAI/bge-m3: in cache, facebook/contriever: in cache, jinaai/jina-embeddings-v3: in cache, Qwen/Qwen3-Embedding-8B: in cache.

## Plan

- plan.json read (written by group A) at 2026-11-09 12:34:30.
- Cost basis: estimated from model sizes (no timing run): seconds = model load + (estimated document tokens x 2 x parameters) / effective FLOP/s + 5 query encodings per query x per-call latency + CPU combination time. weights in float32 (the adapters pass no dtype); every model fits on one 48 GB A6000 (largest: Qwen3-Embedding-8B, about 28 GB of weights), so every job gets one GPU.
- This group: GPUs [4, 5, 6, 7]; estimated wall time 2.2 h (other group: 2.3 h); analysis afterwards about 12 min.

| Job | Device | Texts to embed | Queries | Estimate |
| --- | --- | --- | --- | --- |
| musique_32k__qwen | gpu x1 | 221446 | 900 | 2.2 h |
| novelhopqa__qwen | gpu x1 | 80518 | 908 | 2.1 h |
| novelhopqa__jina | gpu x1 | 80518 | 908 | 12 min |
| loogle__bge_m3 | gpu x1 | 55069 | 1832 | 9 min |
| qasper_64k__contriever | gpu x1 | 168248 | 1333 | 5 min |
| novelhopqa__contriever | gpu x1 | 80518 | 908 | 4 min |
| loogle__contriever | gpu x1 | 55069 | 1832 | 3 min |
| musique_32k__bm25 | cpu x1 | 221446 | 900 | 51s |
| loogle__bm25 | cpu x1 | 55069 | 1832 | 9s |

## Jobs and reproduction checks

| Job | Status | Time | Identical top-10 | HR@5 / HR@10 ours vs saved | text-only vs Table 1 Vanilla | Problems |
| --- | --- | --- | --- | --- | --- | --- |
| loogle__bm25 | DONE | 20s | 1832/1832 | 92.30 / 96.45 vs 92.30 / 96.45 | 91.87 / 95.74 vs 91.5 / 95.4 (differs) | none |
| musique_32k__bm25 | DONE | 3 min | 900/900 | 62.78 / 77.67 vs 62.78 / 77.67 | 57.56 / 74.00 vs 57.6 / 74.0 | none |
| novelhopqa__qwen | ERROR | 5 min | — | — | — | rch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)  |
| novelhopqa__jina | FAILED | 6 min | 624/908 | 14.76 / 20.93 vs 14.76 / 21.15 | 15.97 / 20.15 vs 14.8 / 19.1 (differs) | identical top-10 for 624/908 queries (< 99%); HR@10 20.92511013215859 vs saved 21.145374449339208 (tolerance 0.1 points); HR@10 20.93 vs expected 21.15 |
| loogle__bge_m3 | DONE | 9 min | 1832/1832 | 81.06 / 90.12 vs 81.06 / 90.12 | 82.48 / 90.72 vs 80.6 / 89.8 (differs) | none |
| novelhopqa__contriever | DONE | 5 min | 908/908 | 13.55 / 18.17 vs 13.55 / 18.17 | 12.44 / 17.84 vs 11.8 / 11.8 (differs) | none |
| qasper_64k__contriever | DONE | 8 min | 1333/1333 | 16.20 / 25.81 vs 16.20 / 25.81 | 15.15 / 23.33 vs 15.2 / 23.3 | none |
| loogle__contriever | DONE | 4 min | 1832/1832 | 75.44 / 85.70 vs 75.44 / 85.70 | 75.16 / 86.03 vs 72.4 / 73.3 (differs) | none |
| musique_32k__qwen | ERROR | 23 min | — | — | — | rch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)  |

- loogle__bm25 text-only note: text-only HR@5 91.87 vs Table 1 Vanilla 91.5 (tolerance 0.05); text-only HR@10 95.74 vs Table 1 Vanilla 95.4 (tolerance 0.05)
- novelhopqa__jina text-only note: text-only HR@5 15.97 vs Table 1 Vanilla 14.8 (tolerance 0.15); text-only HR@10 20.15 vs Table 1 Vanilla 19.1 (tolerance 0.15)
- loogle__bge_m3 text-only note: text-only HR@5 82.48 vs Table 1 Vanilla 80.6 (tolerance 0.15); text-only HR@10 90.72 vs Table 1 Vanilla 89.8 (tolerance 0.15)
- novelhopqa__contriever text-only note: text-only HR@5 12.44 vs Table 1 Vanilla 11.8 (tolerance 0.15); text-only HR@10 17.84 vs Table 1 Vanilla 11.8 (tolerance 0.15)
- loogle__contriever text-only note: text-only HR@5 75.16 vs Table 1 Vanilla 72.4 (tolerance 0.15); text-only HR@10 86.03 vs Table 1 Vanilla 73.3 (tolerance 0.15)
- novelhopqa__jina mismatch example: `{"query_id": "hop_1:2", "doc_id": "B62", "saved_top10": ["B62:336", "B62:649", "B62:3", "B62:344", "B62:11", "B62:690", "B62:346", "B62:338", "B62:345", "B62:14"], "saved_scores": [2.341083751968238, 2.0811383163428148, 1.7719866725061864, 1.7071806589929204, 1.6861227045783842, 1.6847346295408885, 1.656603890653753, 1.6563039237784971, 1.640331206942304, 1.6021902755230513], "ours_top10": ["B62:336", "B62:649", "B62:3", "B62:344", "B62:690", "B62:11", "B62:346", "B62:338", "B62:345", "B62:14"], "ours_scores": [2.3407122299752934, 2.08076679434987, 1.76127897329541, 1.7068625185301212, 1.69139`

## Unexpected

```
^^^^^^^^^^^^^^^
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
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py", line 260, in forward
    hidden_states, _ = self.self_attn(
                       ^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/transformers/utils/deprecation.py", line 172, in wrapped_func
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py", line 216, in forward
    attn_output, attn_weights = attention_interface(
                                ^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py", line 96, in sdpa_attention_forward
    attn_output = torch.nn.functional.scaled_dot_product_attention(
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total capacity of 47.40 GiB of which 1.37 GiB is free. Including non-PyTorch memory, this process has 46.02 GiB memory in use. Of the allocated memory 43.70 GiB is allocated by PyTorch, and 2.01 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)

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
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.26 GiB. GPU 0 has a total capacity of 47.40 GiB of which 502.25 MiB is free. Including non-PyTorch memory, this process has 46.90 GiB memory in use. Of the allocated memory 43.34 GiB is allocated by PyTorch, and 3.25 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)

```

## Analysis

- 12/20 jobs DONE; this group's unfinished jobs: ['musique_32k__qwen', 'novelhopqa__jina', 'novelhopqa__qwen']; waiting for group A

## Files written

- `outputs/rebuttal/C2_heldout_fusion/streams/loogle__bm25.parquet`
- `outputs/rebuttal/C2_heldout_fusion/streams/loogle__bm25.queries.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/loogle__bm25.check.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/musique_32k__bm25.parquet`
- `outputs/rebuttal/C2_heldout_fusion/streams/musique_32k__bm25.queries.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/musique_32k__bm25.check.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/loogle__bge_m3.parquet`
- `outputs/rebuttal/C2_heldout_fusion/streams/loogle__bge_m3.queries.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/loogle__bge_m3.check.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/novelhopqa__contriever.parquet`
- `outputs/rebuttal/C2_heldout_fusion/streams/novelhopqa__contriever.queries.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/novelhopqa__contriever.check.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/qasper_64k__contriever.parquet`
- `outputs/rebuttal/C2_heldout_fusion/streams/qasper_64k__contriever.queries.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/qasper_64k__contriever.check.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/loogle__contriever.parquet`
- `outputs/rebuttal/C2_heldout_fusion/streams/loogle__contriever.queries.json`
- `outputs/rebuttal/C2_heldout_fusion/streams/loogle__contriever.check.json`
- `experiments/rebuttal/C2_heldout_fusion/reports/group_B.md`
- `experiments/rebuttal/C2_heldout_fusion/reports/group_B.json`
