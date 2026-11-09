# p1b_ledger - step 02: Index with profile p1b_ledger (GPUs 0-3)

Status: **ok** - 2026-11-09 00:40:16

## What was done

- `bash experiments/method_update/p1b_ledger/run_all.sh  (stage 02_build; python stage02_build.py )`
- `git commit e4373b889608b1a08574bc5205c75cf7f863b5a0`
- `SADI_OUTPUT_DIR=/mnt/data/zeinab/SAADI/outputs/method_update/p1b_ledger CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_PORT=29537 /mnt/data/zeinab/raptor/.venv/bin/python3 /mnt/data/zeinab/SAADI/scripts/run_study.py --suite probe_bridge --profile p1b_ledger --datasets novelhopqa_probe --retrievers bm25 --stage ingest --tensor-parallel-size 4`
- Versions: python=3.12.3, platform=Linux-6.17.0-23-generic-x86_64-with-glibc2.39, numpy=2.3.5, pandas=3.0.1, yaml=6.0.3, saadi=0.1.0, git_commit=28eed3a1ee31487f3bf23dcb3470aa0e164e8cfd, torch=2.11.0+cu129, transformers=4.57.6, gliner=0.2.30, vllm=0.23.0

## Numbers

### Frozen spec

personas.jsonl SHA-256 verified: `11c4c1696c67171f78821d3ba8311ccd28c851315ca1c2a404a9a422593897bb` (same spec as runs 0 and 1).

### Resolved construction settings (profile p1b_ledger)

| setting | value |
| --- | --- |
| construction LLM | Qwen/Qwen3-30B-A3B-Instruct-2507 |
| temperature / seed | 0.0 / 42 |
| max output tokens | 2048 |
| budget shown (records, aliases, descriptions) | 5, 2, 3 |
| history text / anchor matching / ungrounded | anchored / strict / keep |
| schedule / wavefront batch | wavefront / 32 |
| ledger.store_dir | /mnt/data/zeinab/SAADI/outputs/method_update/p1b_ledger/ledger |
| ledger.candidate_net | gliner |
| ledger.candidate_model | urchade/gliner_large-v2.1 |
| ledger.candidate_labels | ['person', 'place', 'organization', 'named object', 'title'] |
| ledger.candidate_device | auto |
| ledger.candidate_window_words | 200 |
| ledger.candidate_max_per_chunk | 40 |
| ledger.description_cap | 5 |
| ledger.extract_max_aliases | -1 |
| ledger.log_prompts | True |
| ledger.candidate_threshold | 0.3 |
| ledger.recall | chunk_tokens |
| ledger.recall_cap | 8 |
| ledger.join_direction | enforced |
| ledger.json_alias_join | False |
| ledger.same_as_rule | p1b |
| ledger.aliases_from_spans | True |

### Runtime environment

| setting | value |
| --- | --- |
| SADI_OUTPUT_DIR | /mnt/data/zeinab/SAADI/outputs/method_update/p1b_ledger |
| HF_HOME / HF_HUB_CACHE | /mnt/cache/taghavi / /mnt/cache/taghavi/hub |
| construction LLM weights in the cache | True |
| candidate net weights in the cache | True |
| gliner importable | yes (0.2.30) |
| CUDA_VISIBLE_DEVICES | 0,1,2,3 |
| VLLM_PORT (this build) | 29537 |
| VLLM_USE_FLASHINFER_SAMPLER / HF_HUB_OFFLINE | 0 / 1 |

### Books: calls (1b / 1 / 0), records, stubs, recall cap hits and candidates per chunk (1b / 1)

| book | state | calls | records | stubs | recall cap hits | decision parse failures | candidates per chunk |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B31 | finished | 244 / 244 / 244 | 325 / 253 | 58 / 33 | 162 / 35 | 0 | 6.2 / 5.5 |
| B43 | finished | 246 / 246 / 246 | 130 / 104 | 44 / 5 | 163 / 98 | 0 | 7.9 / 7.4 |
| B37 | finished | 262 / 262 / 262 | 189 / 165 | 57 / 41 | 200 / 205 | 0 | 9.9 / 9.2 |
| B18 | finished | 312 / 312 / 312 | 325 / 244 | 151 / 36 | 270 / 79 | 0 | 7.1 / 6.4 |
| B04 | finished | 321 / 321 / 321 | 266 / 211 | 62 / 29 | 224 / 18 | 0 | 6.6 / 6.0 |
| B34 | finished | 335 / 335 / 335 | 225 / 187 | 48 / 35 | 237 / 225 | 0 | 9.1 / 8.6 |
| B27 | finished | 338 / 338 / 338 | 247 / 199 | 73 / 28 | 272 / 252 | 0 | 8.9 / 8.4 |
| B61 | finished | 357 / 357 / 357 | 526 / 446 | 275 / 144 | 318 / 72 | 0 | 7.2 / 6.2 |
| B26 | finished | 384 / 384 / 384 | 340 / 247 | 136 / 36 | 338 / 182 | 0 | 7.4 / 6.7 |
| B44 | finished | 385 / 385 / 385 | 365 / 267 | 101 / 41 | 259 / 126 | 0 | 6.2 / 5.5 |

### Ledger totals (run 1b / run 1)

| measure | run 1b | run 1 |
| --- | --- | --- |
| calls (events) | 3184 | 3184 |
| records / stubs | 2938 / 1005 | 2323 / 428 |
| promotions | 396 | 164 |
| final SAME AS (model accepted + auto-join + JSON-alias join) / unresolved | 17099 / 79 | 17094 / 33 |
| NEW -> stub / NOT AN ENTITY | 1401 / 2232 | 592 / 1674 |
| decision-block parse failures / omitted candidates | 0 / 3 | 0 / 7 |
| recall cap hits (chunks) / entries left out | 2443 (77%) / 33176 | 1292 / 6638 |
| description replacements (cap 5) | 6262 | 6359 |
| candidate-net seconds per chunk (mean) | 0.083 | 0.082 |
| JSON parser status not ok (fallback or failed) | 3111 | 3095 |
| memory.json entries holding a replaced description (should be 0) | 0 | 0 |

### The three changes at work (all books)

| measure | value |
| --- | --- |
| model SAME AS accepted, by evidence | {'shared_token': 16164, 'same_sentence': 302} |
| model SAME AS rejected (became NEW), by reason | {'no_shared_token_or_sentence': 1288, 'type_conflict': 559, 'title_conflict': 345} |
| auto-join (NEW equal to a shown entry) / blocked by a conflict / ambiguous | 633 / 1099 / 95 |
| JSON-alias join (optional switch) / ambiguous | 0 / 0 (switch off) |
| JSON records redirected into a SAME AS target / duplicates merged | 2991 / 385 |
| NEW folded into a described record, rejected for a conflict | 292 |
| aliases rejected, by reason | {'not_a_span': 9811, 'title_conflict': 322, 'type_conflict': 36, 'descriptive': 7} |
| recall: entries shown, by match class | {'single_token': 7032, 'full_name': 14596, 'full_alias': 1081} |
| recall: entries left out by the cap of 8, by match class | {'single_token': 28067, 'full_name': 3268, 'full_alias': 1841} |
| entries with a candidate-net label | 3921 |

### Tokens per call, input and output (1b / 1 / 0; Qwen tokenizer), and replies at the output cap with no JSON

| book | input | output | likely truncated (1b) |
| --- | --- | --- | --- |
| B31 | 4964 / 4713 / 4094 | 820 / 835 / 1097 | 0 |
| B43 | 5224 / 5190 / 4117 | 1287 / 1324 / 1013 | 0 |
| B37 | 5131 / 5271 / 3941 | 1329 / 1292 / 1320 | 0 |
| B18 | 5147 / 4814 / 3937 | 1231 / 1166 / 1101 | 1 |
| B04 | 5079 / 4725 / 4028 | 997 / 960 / 1008 | 1 |
| B34 | 5071 / 5049 / 3962 | 1225 / 1261 / 1131 | 0 |
| B27 | 5116 / 5053 / 4074 | 1240 / 1320 / 1214 | 0 |
| B61 | 4983 / 4707 / 4153 | 1107 / 1057 / 1090 | 0 |
| B26 | 5691 / 5426 / 4203 | 1159 / 1144 / 880 | 0 |
| B44 | 5180 / 4966 / 4140 | 949 / 922 / 935 | 0 |
| **all books (mean of book means)** | 5159 / 4991 / 4065 | 1134 / 1128 / 1079 | 2 |

### Wall clock

3.84 h this invocation; 3.84 h over 1 invocation(s). Run 1: 3.69 h; run 0: 3.36 h.

## Anything unexpected

Nothing unexpected.

## Files written

- `/mnt/data/zeinab/SAADI/outputs/method_update/p1b_ledger/build_invocations.json`
- `/mnt/data/zeinab/SAADI/outputs/method_update/p1b_ledger/novelhopqa_probe/indexes/eb4d1079603b2760/llm_cache/novelhopqa/hierarchical/branching_factor_5/chunk_500/AE_5_2_3/Qwen3-30B-A3B-Instruct-2507/AEWord/index.json`
- `/mnt/data/zeinab/SAADI/outputs/method_update/p1b_ledger/ledger/<book>/events.jsonl, ledger.json`
