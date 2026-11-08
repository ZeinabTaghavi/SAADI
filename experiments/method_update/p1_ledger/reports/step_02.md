# p1_ledger - step 02: Index with profile p1_ledger (GPU)

Status: **ok** - 2026-11-08 18:03:57

## What was done

- `bash experiments/method_update/p1_ledger/sh/02_*.sh `
- `SADI_OUTPUT_DIR=/home/iataghav/data/SAADI/outputs/method_update/p1_ledger CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_PORT=29517 /mnt/data/zeinab/raptor/.venv/bin/python3 /mnt/data/zeinab/SAADI/scripts/run_study.py --suite probe_bridge --profile p1_ledger --datasets novelhopqa_probe --retrievers bm25 --stage ingest --tensor-parallel-size 4`
- Versions: python=3.12.3, platform=Linux-6.17.0-23-generic-x86_64-with-glibc2.39, numpy=2.3.5, pandas=3.0.1, yaml=6.0.3, saadi=0.1.0, git_commit=71f66a3ac3a5853b584420736adb439fae992d91, torch=2.11.0+cu129, transformers=4.57.6, gliner=0.2.30, vllm=0.23.0

## Numbers

### Frozen spec

personas.jsonl SHA-256 verified: `11c4c1696c67171f78821d3ba8311ccd28c851315ca1c2a404a9a422593897bb` (same spec as run 0).

### Resolved construction settings (profile p1_ledger = published + ledger)

| setting | value |
| --- | --- |
| construction LLM | Qwen/Qwen3-30B-A3B-Instruct-2507 |
| temperature / seed | 0.0 / 42 |
| tensor parallel (config) | 4 |
| memory selector (overlap slots) | word |
| history text / anchor matching / ungrounded | anchored / strict / keep |
| budget shown (records, aliases, descriptions) | 5, 2, 3 |
| schedule / wavefront batch | wavefront / 32 |
| ledger.store_dir | /home/iataghav/data/SAADI/outputs/method_update/p1_ledger/ledger |
| ledger.candidate_net | gliner |
| ledger.candidate_model | urchade/gliner_large-v2.1 |
| ledger.candidate_labels | ['person', 'place', 'organization', 'named object', 'title'] |
| ledger.candidate_threshold | 0.5 |
| ledger.candidate_device | auto |
| ledger.candidate_window_words | 200 |
| ledger.candidate_max_per_chunk | 40 |
| ledger.recall_cap | 5 |
| ledger.description_cap | 5 |
| ledger.extract_max_aliases | -1 |
| ledger.log_prompts | True |

### Runtime environment

| setting | value |
| --- | --- |
| SADI_OUTPUT_DIR | /home/iataghav/data/SAADI/outputs/method_update/p1_ledger |
| HF_HOME / HF_HUB_CACHE | /mnt/cache/taghavi / /mnt/cache/taghavi/hub |
| construction LLM weights in the cache | True |
| candidate net weights in the cache | True |
| gliner importable | yes (0.2.30) |
| CUDA_VISIBLE_DEVICES | 0,1,2,3 |
| VLLM_PORT (this build) | 29517 |
| VLLM_USE_FLASHINFER_SAMPLER / HF_HUB_OFFLINE | 0 / 1 |

### Wall clock

3.69 h (13280 s) this invocation; 3.69 h over 1 invocation(s). Run 0: 3.36 h in one invocation.

### Books: calls, records and stubs

| book | state | calls (chunk records) | run 0 calls | records | stubs | promotions | aliases added | decision parse failures | ledger cap hits | JSON parser not ok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B31 | finished | 244 | 244 | 253 | 33 | 18 | 181 | 0 | 35 | 240 |
| B43 | finished | 246 | 246 | 104 | 5 | 2 | 192 | 0 | 98 | 245 |
| B37 | finished | 262 | 262 | 165 | 41 | 16 | 270 | 0 | 205 | 251 |
| B18 | finished | 312 | 312 | 244 | 36 | 9 | 313 | 0 | 79 | 300 |
| B04 | finished | 321 | 321 | 211 | 29 | 7 | 204 | 0 | 18 | 314 |
| B34 | finished | 335 | 335 | 187 | 35 | 11 | 246 | 0 | 225 | 334 |
| B27 | finished | 338 | 338 | 199 | 28 | 11 | 240 | 0 | 252 | 337 |
| B61 | finished | 357 | 357 | 446 | 144 | 44 | 262 | 0 | 72 | 324 |
| B26 | finished | 384 | 384 | 247 | 36 | 26 | 376 | 0 | 182 | 382 |
| B44 | finished | 385 | 385 | 267 | 41 | 20 | 273 | 0 | 126 | 368 |

### Ledger totals over the books built so far

| measure | value |
| --- | --- |
| calls (events) | 3184 |
| records / stubs in the final ledgers | 2323 / 428 |
| promotions (stub -> record) | 164 |
| SAME AS decisions / unresolved targets | 17094 / 33 |
| NEW -> stub / LLM additions (records the net did not propose) | 592 / 50 |
| NOT AN ENTITY decisions | 1674 |
| decision-block parse failures (all candidates -> stubs) / omitted candidates | 0 / 7 |
| ledger-recall cap hits (chunks) / entries left out | 1292 / 6638 |
| description replacements (cap 5) | 6359 |
| records proposed beyond the budget of 5 (cut, as today) | 0 |
| candidates per chunk (mean) | 7.0 |
| candidate-net seconds per chunk (mean) / device | 0.082 / cuda |
| JSON parser status not ok (fallback or failed) | 3095 |
| memory.json entries holding a replaced description (should be 0) | 0 |

### Replies at the 2048-token cap with an unparsed JSON block (likely truncated)

3 of 3184 calls. The output cap is the published one and was not changed.

### Tokens per call (prompt text system+user, and the reply; Qwen tokenizer, no chat template)

| book | input run 1 | input run 0 | output run 1 | output run 0 |
| --- | --- | --- | --- | --- |
| B31 | 4713 | 4094 | 835 | 1097 |
| B43 | 5190 | 4117 | 1324 | 1013 |
| B37 | 5271 | 3941 | 1292 | 1320 |
| B18 | 4814 | 3937 | 1166 | 1101 |
| B04 | 4725 | 4028 | 960 | 1008 |
| B34 | 5049 | 3962 | 1261 | 1131 |
| B27 | 5053 | 4074 | 1320 | 1214 |
| B61 | 4707 | 4153 | 1057 | 1090 |
| B26 | 5426 | 4203 | 1144 | 880 |
| B44 | 4966 | 4140 | 922 | 935 |
| **all books (mean of book means)** | **4991** | **4065** | **1128** | **1079** |

### Where run 0's token counts come from

Run 0 did not log its prompts; they are rebuilt from its saved records (published strategy on the state after the previous step), cached in `run0_tokens.json`.

### Hand-off

All books are indexed. Paste this report back before step 3.

## Anything unexpected

Nothing unexpected.

## Files written

- `/home/iataghav/data/SAADI/outputs/method_update/p1_ledger/build_invocations.json`
- `/home/iataghav/data/SAADI/outputs/method_update/p1_ledger/ledger/system_prompt.txt`
- `/home/iataghav/data/SAADI/outputs/method_update/p1_ledger/novelhopqa_probe/indexes/68ecf44999173223/llm_cache/novelhopqa/hierarchical/branching_factor_5/chunk_500/AE_5_2_3/Qwen3-30B-A3B-Instruct-2507/AEWord/index.json`
- `/home/iataghav/data/SAADI/outputs/method_update/p1_ledger/ledger/<book>/events.jsonl, ledger.json`
