# C8 long-context reader - step 00 (prepare): OK (vLLM needs the dual chunk attention backend)

Written 2026-11-11 00:04:17; commit e93fe008b35e; host eee-s132-infsys-gpu03; 355 s. CPU only, no model was called.

Command:

```bash
bash experiments/rebuttal/c8_long_context/sh/00_prepare.sh 
```

## Summary

- MuSiQue-32K: 900 queries on 45 distinct documents; document length 42,257 / 48,851 / 51,692 tokens (min / median / max); 0 queries on 0 documents beyond the native 262,144-token window. Rough estimate 1.9 GPU h with the prefix cache (10 GPU h for the full documents without it).
- NovelHopQA: 908 queries on 16 distinct documents; document length 133,420 / 232,803 / 759,549 tokens (min / median / max); 382 queries on 6 documents beyond the native 262,144-token window. Rough estimate 5.4 GPU h with the prefix cache (231 GPU h for the full documents without it).
- --max-model-len 778,240 (longest prompt 759,644 + 64 output tokens + margin). Estimated KV capacity on 4 GPUs: 2,035,315 tokens, enough for the longest document; every document fits, none is cut.
- The installed vLLM (0.23.0+cu129, `/mnt/data/zeinab/raptor/.venv/bin/python3`) has no dual chunk attention backend (V0 engine False); 382 queries need it. Install vLLM 0.10.1.1 in a separate environment and set C8_VLLM_PYTHON to its python before the GPU script.

## Inputs from C5 (read only)

| Dataset | Condition | Rows | Complete | sha256 | File |
| --- | --- | --- | --- | --- | --- |
| MuSiQue-32K | saadi | 900 | True | 7fee9785ef01 | outputs/rebuttal/c5_end_to_end/contexts/musique_32k/saadi.jsonl.gz |
| MuSiQue-32K | vanilla | 900 | True | daa98169c38c | outputs/rebuttal/c5_end_to_end/contexts/musique_32k/vanilla.jsonl.gz |
| MuSiQue-32K | closed_book | 900 | True | 262c652ce5f1 | outputs/rebuttal/c5_end_to_end/contexts/musique_32k/closed_book.jsonl.gz |
| NovelHopQA | saadi | 908 | True | 72b34e7323e9 | outputs/rebuttal/c5_end_to_end/contexts/novelhopqa/saadi.jsonl.gz |
| NovelHopQA | vanilla | 908 | True | 28d3a697899c | outputs/rebuttal/c5_end_to_end/contexts/novelhopqa/vanilla.jsonl.gz |
| NovelHopQA | closed_book | 908 | True | 4232955baf9d | outputs/rebuttal/c5_end_to_end/contexts/novelhopqa/closed_book.jsonl.gz |

Published SAADI point (0.5, 0.3, 1.0, 1.0) (configs/methods/sadi.yaml); docs/preserved_sources.json: 93 files, changed 0, missing 0.

## Long-context model

`Qwen/Qwen2.5-7B-Instruct-1M`: in the Hugging Face cache: True (/mnt/cache/taghavi/hub/models--Qwen--Qwen2.5-7B-Instruct-1M/snapshots/e28526f7bb80e2a9c8af03b831a9af3812f18fba).
Tokenizer used for every count: `Qwen/Qwen2.5-7B-Instruct-1M (/mnt/cache/taghavi/hub/models--Qwen--Qwen2.5-7B-Instruct-1M/snapshots/e28526f7bb80e2a9c8af03b831a9af3812f18fba)`.

Shape: 28 layers, 28 query heads, 4 KV heads, head dim 128, weights 15.2 GB (bfloat16); KV cache 56 KiB per token; max position 1,010,000; native window 262,144 tokens (longer inputs need dual chunk attention: {"chunk_size": 262144, "local_size": 8192, "original_max_position_embeddings": 262144}).

## Document lengths (tokens of the long-context model's tokenizer)

| Dataset | Queries | Distinct documents | Queries per document (min / median / max) | Tokens min | Tokens median | Tokens mean | Tokens max | Chunks per document (mean) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MuSiQue-32K | 900 | 45 | 18 / 20 / 22 | 42,257 | 48,851 | 48,793 | 51,692 | 380 |
| NovelHopQA | 908 | 16 | 14 / 60.5 / 84 | 133,420 | 232,803 | 280,546 | 759,549 | 438 |

Document = SAADI's chunks of that document in order, whitespace collapsed and joined by one space (the chunks are cut at whitespace), given to the reader as C5's single passage `[1] <document>`. Per-document lengths: `results/rebuttal/c8_long_context/prepare/doc_lengths.csv`.

NovelHopQA books:

| Book | Tokens | Chunks | Queries | Prompt fits the window | Beyond the native window |
| --- | --- | --- | --- | --- | --- |
| B42 | 759,549 | 1165 | 70 | True | True |
| B62 | 461,227 | 729 | 84 | True | True |
| B54 | 411,934 | 660 | 74 | True | True |
| B05 | 324,835 | 508 | 22 | True | True |
| B15 | 273,811 | 421 | 75 | True | True |
| B44 | 264,945 | 384 | 57 | True | True |
| B26 | 253,466 | 383 | 50 | True | False |
| B12 | 240,542 | 385 | 61 | True | False |
| B61 | 225,064 | 355 | 14 | True | False |
| B04 | 209,488 | 320 | 45 | True | False |
| B34 | 206,833 | 334 | 65 | True | False |
| B27 | 201,836 | 337 | 60 | True | False |
| B18 | 194,220 | 311 | 44 | True | False |
| B37 | 167,704 | 261 | 62 | True | False |
| B43 | 159,866 | 245 | 60 | True | False |
| B24 | 133,420 | 213 | 65 | True | False |

## Prompt tokens per condition (chat template included)

| Dataset | Condition | Queries | Mean | Median | Max | Total (M) |
| --- | --- | --- | --- | --- | --- | --- |
| MuSiQue-32K | Full document | 900 | 48,904 | 49,024 | 51,793 | 44.01 |
| MuSiQue-32K | SAADI top 10 | 900 | 1,403 | 1,398 | 2,098 | 1.26 |
| MuSiQue-32K | Vanilla BM25 top 10 | 900 | 1,404 | 1,398 | 1,847 | 1.26 |
| MuSiQue-32K | Closed-book | 900 | 80 | 79 | 113 | 0.07 |
| NovelHopQA | Full document | 908 | 295,083 | 240,622 | 759,644 | 267.94 |
| NovelHopQA | SAADI top 10 | 908 | 6,525 | 6,496 | 7,531 | 5.92 |
| NovelHopQA | Vanilla BM25 top 10 | 908 | 6,520 | 6,492 | 7,518 | 5.92 |
| NovelHopQA | Closed-book | 908 | 81 | 81 | 113 | 0.07 |

Full-document counts: the split count equals the full tokenisation and the chat template's count on the first query of all 61 documents (max difference 0).

## vLLM settings

| Setting | Value | Why |
| --- | --- | --- |
| --max-model-len | 778,240 | longest prompt + 64 output tokens = 759,708; + margin 15,195 (max of 8192 tokens and 2%), rounded up to a multiple of 8192 |
| --tensor-parallel-size | 4 | 4 GPUs (4,5,6,7); the 4 KV heads split one per GPU |
| VLLM_ATTENTION_BACKEND | DUAL_CHUNK_FLASH_ATTN | dual chunk attention with sparse prefill, Qwen's setup for inputs beyond 262,144 tokens; identical to full attention inside one chunk |
| VLLM_USE_V1 | 0 | the dual chunk attention backend exists only in vLLM's V0 engine |
| --enable-chunked-prefill / --max-num-batched-tokens | on / 131,072 | Qwen's setting; bounds the prefill activations of a 760K-token book |
| --enforce-eager | True | Qwen's setting for this backend (no CUDA graphs) |
| --enable-prefix-caching | True | queries on the same document reuse its KV cache; the timing step checks it works |
| --gpu-memory-utilization | 0.90 | leaves ~4.8 GB per GPU for CUDA and NCCL |
| --max-num-seqs / client concurrency | 8 / 8 | one uncached document at a time (sent alone), then that document's questions together |
| temperature / top_p / seed / max_tokens | 0 / 1 / 42 / 64 | same as C5's reader |

Memory fit per GPU (estimate; vLLM's own profile at start-up is the real check):

| GPU memory | x utilisation | Weights | Activations + CUDA (est.) | Left for KV | KV per token | KV capacity | Longest sequence needs | Fits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 49,140 MiB | 0.90 | 3.5 GiB | 12.5 GiB | 27.2 GiB | 14 KiB | 2,035,315 tokens (2.6 x the window) | 10.4 GiB | yes |

GPU sizes from nvidia-smi: GPU 4 NVIDIA RTX A6000 49,140 MiB (1 MiB in use now); GPU 5 NVIDIA RTX A6000 49,140 MiB (1 MiB in use now); GPU 6 NVIDIA RTX A6000 49,140 MiB (1 MiB in use now); GPU 7 NVIDIA RTX A6000 49,140 MiB (1 MiB in use now).

Installed vLLM (`/mnt/data/zeinab/raptor/.venv/bin/python3`): version 0.23.0+cu129; V0 engine present: False; dual chunk attention backend file: False.

NovelHopQA: 6 documents (382 queries) are longer than the native window of 262,144 tokens: B42, B62, B54, B05, B15, B44. They need the dual chunk attention backend (NOT found in the installed vLLM: see Summary).

## Position of the gold evidence in the document

Mean position (in tokens) of the evidence chunks' midpoints; MuSiQue-32K: the paper's gold and silver chunks; NovelHopQA: C5's paragraph labels (each evidence paragraph located in SAADI's chunks). 'Spread' = the evidence chunks fall in more than one third.

| Dataset | First third | Middle third | Last third | No located evidence | Spread over thirds |
| --- | --- | --- | --- | --- | --- |
| MuSiQue-32K | 269 | 369 | 262 | 0 | 290 |
| NovelHopQA | 236 | 384 | 271 | 17 | 362 |

## Rough time estimate (a priori; the GPU script's timing run replaces it)

Assumes 120 TFLOP/s effective prefill on 4 GPUs (dense attention, no sparse speed-up) and 24 output tokens per answer.

| Dataset | Cold prefill, once per document (h) | Questions on a cached document (h) | SAADI + Vanilla + closed-book (h) | Wall with cache (h) | GPU h with cache | Full document WITHOUT cache, wall (h) |
| --- | --- | --- | --- | --- | --- | --- |
| MuSiQue-32K | 0.13 | 0.07 | 0.03 | 0.48 | 1.9 | 2.6 |
| NovelHopQA | 0.91 | 0.12 | 0.08 | 1.36 | 5.4 | 57.6 |

## Next

```bash
nohup bash experiments/rebuttal/c8_long_context/sh/run_gpus_4567.sh > /dev/null 2>&1 &
```

(after C5's run on GPUs 4-7 has finished)

## Files written

- `outputs/rebuttal/c8_long_context/prepared/musique_32k/documents.jsonl.gz`
- `outputs/rebuttal/c8_long_context/prepared/musique_32k/queries.jsonl.gz`
- `outputs/rebuttal/c8_long_context/prepared/musique_32k/manifest.json`
- `outputs/rebuttal/c8_long_context/prepared/novelhopqa/documents.jsonl.gz`
- `outputs/rebuttal/c8_long_context/prepared/novelhopqa/queries.jsonl.gz`
- `outputs/rebuttal/c8_long_context/prepared/novelhopqa/manifest.json`
- `outputs/rebuttal/c8_long_context/prepared/vllm_settings.env`
- `results/rebuttal/c8_long_context/prepare/doc_lengths.csv`
- `results/rebuttal/c8_long_context/prepare/prompt_tokens.csv`
- `results/rebuttal/c8_long_context/prepare/evidence_position.csv`
- `experiments/rebuttal/c8_long_context/reports/step_00.md`
- `experiments/rebuttal/c8_long_context/reports/step_00.json`
