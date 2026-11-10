# C5 end-to-end QA - step 00 (prepare): OK (some comparison outputs missing)

Written 2026-11-10 19:44:20; commit 2c864247f598; host eee-s132-infsys-gpu03; exit code 0.

Command: `bash experiments/rebuttal/c5_end_to_end/sh/00_prepare.sh `

## Summary

- MuSiQue-32K: gate passed (SAADI 900/900 identical to the paper, HR@10 77.67; Vanilla HR@10 74.00). Contexts built: closed_book, oracle, vanilla, saadi, cr, arag, arag_tm, saadi_at_arag_len, arag_own. Missing: raptor, raptor_tm, saadi_at_raptor_len.
- NovelHopQA: gate passed (SAADI 908/908 identical to the paper, HR@10 22.03; Vanilla HR@10 20.70). Contexts built: closed_book, oracle, vanilla, saadi, cr, raptor, raptor_tm, arag, arag_tm, saadi_at_raptor_len, saadi_at_arag_len, arag_own.

## Fixed settings

- Base retriever: BM25 (the retriever of the paper's agent comparison, Table 2; its SAADI inputs are the bm25 `retrieval_depth` rankings of outputs/saadi/reference).
- SAADI: published point (lambda_M, lambda_S, mu_A, mu_D) = (0.5, 0.3, 1.0, 1.0) from `/home/iataghav/data/SAADI/configs/methods/sadi.yaml` (lambda_T = 1, z-score, max aggregation).
- Reader: `Qwen/Qwen3-30B-A3B-Instruct-2507`, vLLM, temperature 0, seed 42, max output tokens 64, max model length 32768. Tokenizer for all counts: the reader's.
- One prompt for every condition: a single user message, no system prompt, chat template of the model. The exact template:

```text
Answer the question using the context passages below. If the passages do not contain the answer, answer from your own knowledge.

Context:
{context}

Question: {question}
Answer with a short phrase.
```

`{context}` is the passages in rank order, each `[i] <text>` (whitespace collapsed), separated by a blank line; closed-book gets `(no passages)`. Scoring: SQuAD normalisation (lower case, no punctuation, no articles), EM and token F1 as the max over the gold answers, containment = a normalised gold answer inside the normalised output. The output is first cut to its first non-empty line and a leading `Answer:` is dropped.

Example (MuSiQue-32K, SAADI, first query; passages shortened here to 120 characters):

```text
Answer the question using the context passages below. If the passages do not contain the answer, answer from your own knowledge.

Context:
[1] [Document 5/20 | doc_id=2hop__460946_294723] Grant's First Stand is the debut album by American jazz guitarist Grant Gre...

[2] State Upper House Seats 0 / 1,972 State Lower House Seats 2 / 5,411 Territorial Governorships 0 / 6 Territorial Upper Ch...

[3] The Latin Bit is an album by American jazz guitarist Grant Green featuring performances recorded in 1962 and released on...

[4] 17 years ago (2001 - 04) Split from Greens / Green Party USA Preceded by Citizens Party Association of State Green Parti...

[5] Tombs also often contained small green amulets in the shape of scarab beetles made of malachite, which would protect and...

[6] The Main Attraction is an album by American jazz guitarist Grant Green featuring performances recorded in 1976 and relea...

[7] The series' release led to record subscriptions for All Access, and positive reviews from critics who highlighted Martin...

[8] Animals typically use the color green as camouflage, blending in with the chlorophyll green of the surrounding environme...

[9] Written in spring 1977 at the same time as his previous album, the funk-inflected "Motivation Radio" (1977), "Green" was...

[10] Phylicia and Ahmad Rashād are the parents of actress Condola Rashād. For the ancient Egyptians, green had very positive ...

Question: Who is the spouse of the Green performer?
Answer with a short phrase.
```

## MuSiQue-32K

Queries 900, documents 45, chunks 17102; gold answers: outputs/musique_ablation/32K_expansion/prepared/qa.json + answer aliases from dgslibisey/MuSiQue.
MuSiQue answer aliases: {"used": true, "source": "Hugging Face dgslibisey/MuSiQue validation (offline cache)", "matched_questions": 900, "answer_mismatch": 0, "questions_with_aliases_added": 198}.

Reproduction gate: **PASSED** (streams `outputs/rebuttal/C2_heldout_fusion/streams/musique_32k__bm25.parquet`).

| List | Identical | Ours | Paper |
| --- | --- | --- | --- |
| SAADI top 10 vs paper's saved ranking | 900/900 identical | HR@5 62.78, HR@10 77.67 | hits {5: 565, 10: 699} vs paper {5: 565, 10: 699} |
| Vanilla top 10 (text channel) | no per-query file | HR@5 57.56, HR@10 74.00 | Table 1: {5: 57.6, 10: 74.0}; reference: Table 1 Vanilla HR@5/HR@10 (outputs/other_methods/vanilla/retrieval.csv; no per-query file exists) |

Saved comparison outputs:

| Method | Status | C5 queries covered | Checks | Source |
| --- | --- | --- | --- | --- |
| Contextual Retrieval (C1) | ok | 900/900 | CR HR@10 82.22 (C1 report 82.22222222222223); C1 Vanilla/SAADI = ours: 900/900;  | /home/iataghav/data/SAADI/outputs/rebuttal/C1_contextual_retrieval/musique_32k/retrieval/bm25/rankings.pkl.gz |
| RAPTOR | missing | 0/900 | saved run; document text check None; records None; own-queries HR@10 (upper bound) — vs paper None; same question None, other document None | /home/iataghav/data/SAADI/outputs/other_methods_method/SAADI_Raptor/raptor_10_runs/musique_32k/musique_32k_retrieval_ablation_raptor |
| A-RAG | ok | 900/900 | HR 45.11 vs Table 2 45.111111111111114; mean read 2.6622 vs 2.6622222222222223; unknown ids 0; terminations {'model_final_answer': 899, 'max_loops': 1};  | /mnt/data/zeinab/SAADI_A_RAG/outputs/musique_32k/arag/musique_32k_a_rag_raw_f06714eda6267987 |

Contexts (token budget of the token-matched rows = mean SAADI top-10 context = 1326 tokens):

| Condition | n | Passages | Ctx tok mean | Ctx tok max | Prompt tok mean | Prompt tok max | Evidence in context % (eligible) | HR@10 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| closed_book | 900 | 0.00 | 0 | 0 | 69 | 102 | 0.00 (900) | — |
| oracle | 900 | 6.18 | 834 | 4434 | 900 | 4505 | 100.00 (900) | — |
| vanilla | 900 | 10.00 | 1327 | 1765 | 1393 | 1836 | 74.00 (900) | 74.00 |
| saadi | 900 | 10.00 | 1326 | 2016 | 1392 | 2087 | 77.67 (900) | 77.67 |
| cr | 900 | 10.00 | 1332 | 2041 | 1397 | 2113 | 82.22 (900) | 82.22 |
| raptor | 0 | — | — | — | — | — | — | missing |
| raptor_tm | 0 | — | — | — | — | — | — | missing |
| arag | 900 | 2.66 | 360 | 1676 | 426 | 1747 | 45.11 (900) | 45.11 |
| arag_tm | 900 | 2.66 | 359 | 1325 | 425 | 1405 | 45.11 (900) | 45.11; cut 4 |
| saadi_at_raptor_len | 0 | — | — | — | — | — | — | missing |
| saadi_at_arag_len | 900 | 3.08 | 358 | 1389 | 424 | 1463 | 40.89 (900) | —; cut 892 |
| arag_own | 900 | 2.66 | 360 | 1676 | 426 | 1747 | 45.11 (900) | 45.11 |

## NovelHopQA

Queries 908, documents 16, chunks 7011; gold answers: Hugging Face abhaygupta1266/novelhopqa (repo loader, offline cache).

Reproduction gate: **PASSED** (streams `outputs/rebuttal/C2_heldout_fusion/streams/novelhopqa__bm25.parquet`).

| List | Identical | Ours | Paper |
| --- | --- | --- | --- |
| SAADI top 10 vs paper's saved ranking | 908/908 identical | HR@5 18.06, HR@10 22.03 | hits {5: 164, 10: 200} vs paper {5: 164, 10: 200} |
| Vanilla top 10 (text channel) | 908/908 identical | HR@5 17.51, HR@10 20.70 | Table 1: {5: 16.9, 10: 20.0}; reference: saved per-query text-only rankings (outputs/novelhopqa_ablation/500_tokens/llm_cache/novelhopqa/hierarchical/branching_factor_5/chunk_500/AE_5_2_3/Qwen3-30B-A3B-Instruct-2507/AEWord/retrieval_reports/retrieval_results__bm25__mbr_lt1_ln0_lh0_norm_none__per_document.json) |

Saved comparison outputs:

| Method | Status | C5 queries covered | Checks | Source |
| --- | --- | --- | --- | --- |
| Contextual Retrieval (C1) | ok | 908/908 | CR HR@10 21.70 (C1 report 21.696035242290748); C1 Vanilla/SAADI = ours: 908/908;  | /home/iataghav/data/SAADI/outputs/rebuttal/C1_contextual_retrieval/novelhopqa/retrieval/bm25/rankings.pkl.gz |
| RAPTOR | ok | 908/908 | saved run; document text check ids only (whole books, loaded by the same NovelHopQA loader); records 1841; own-queries HR@10 (upper bound) 10.13 vs paper 10.1; same question 908, other document 0 | /mnt/data/zeinab/SAADI_Raptor/raptor_10_runs/novelhopqa/novelhopqa_retrieval_ablation_raptor |
| A-RAG | ok | 908/908 | HR 9.03 vs Table 2 9.030837004405287; mean read 3.8855 vs 3.8854625550660793; unknown ids 0; terminations {'model_final_answer': 908};  | /mnt/data/zeinab/SAADI_A_RAG/outputs/novelhopqa/arag/novelhopqa_a_rag_raw_479355014fce5066 |

Contexts (token budget of the token-matched rows = mean SAADI top-10 context = 6447 tokens):

| Condition | n | Passages | Ctx tok mean | Ctx tok max | Prompt tok mean | Prompt tok max | Evidence in context % (eligible) | HR@10 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| closed_book | 908 | 0.00 | 0 | 0 | 70 | 102 | 0.00 (908) | — |
| oracle | 908 | 3.59 | 2302 | 30097 | 2369 | 30187 | 100.00 (908) | — |
| vanilla | 908 | 10.00 | 6442 | 7445 | 6509 | 7507 | 20.70 (908) | 20.70 |
| saadi | 908 | 10.00 | 6447 | 7452 | 6514 | 7520 | 22.03 (908) | 22.03 |
| cr | 908 | 10.00 | 6426 | 7441 | 6493 | 7501 | 21.70 (908) | 21.70 |
| raptor | 908 | 9.94 | 1701 | 3399 | 1769 | 3460 | 17.58 (785) | 8.66 |
| raptor_tm | 908 | 9.94 | 1701 | 3399 | 1769 | 3460 | 17.58 (785) | 8.66 |
| arag | 908 | 3.89 | 2548 | 9130 | 2615 | 9213 | 9.03 (908) | 9.03 |
| arag_tm | 908 | 3.88 | 2544 | 6446 | 2611 | 6530 | 9.03 (908) | 9.03; cut 2 |
| saadi_at_raptor_len | 908 | 3.21 | 1700 | 3398 | 1768 | 3460 | 13.44 (908) | —; cut 908 |
| saadi_at_arag_len | 908 | 4.48 | 2544 | 6916 | 2611 | 6994 | 16.08 (908) | —; cut 907 |
| arag_own | 908 | 3.89 | 2548 | 9130 | 2615 | 9213 | 9.03 (908) | 9.03 |

## Comparison-method settings (defaults; nothing tuned)

- MuSiQue-32K cr: construction LLM = Qwen/Qwen3-30B-A3B-Instruct-2507; context max tokens = 1024; sampling = {'temperature': 0.0, 'top_p': 1.0, 'top_k': -1, 'repetition_penalty': 1.0, 'seed': 42}; prompt = Anthropic's published contextualizer prompt (C1 ANTHROPIC_PROMPT); index = SAADI's BM25 on context + '\n\n' + chunk; no reranker; reader input = the ORIGINAL chunk text of the CR top 10 (CR changes what is indexed, not the chunk)
- MuSiQue-32K arag: model = Qwen/Qwen3-30B-A3B-Instruct-2507; max_loops = 15; max_context_tokens = 32768; message_token_guard = 30720; max_completion_tokens = 2048; embedding_model = Qwen/Qwen3-Embedding-0.6B; workers = 4; upstream = https://github.com/Ayanami0730/arag; chunks = 17102
- NovelHopQA cr: construction LLM = Qwen/Qwen3-30B-A3B-Instruct-2507; context max tokens = 1024; sampling = {'temperature': 0.0, 'top_p': 1.0, 'top_k': -1, 'repetition_penalty': 1.0, 'seed': 42}; prompt = Anthropic's published contextualizer prompt (C1 ANTHROPIC_PROMPT); index = SAADI's BM25 on context + '\n\n' + chunk; no reranker; reader input = the ORIGINAL chunk text of the CR top 10 (CR changes what is indexed, not the chunk)
- NovelHopQA raptor: tree_builder.max_tokens (leaf size) = 500; tree_builder.num_layers = 5; tree_builder.threshold = 0.5; tree_builder.top_k = 5; tree_builder.selection_mode = top_k; tree_builder.summarization_length = 150; retrieve.top_k = 10; retrieve.max_tokens = 3500; retrieve.collapse_tree = True; embedding model = facebook/contriever; summarization model = Qwen/Qwen3-30B-A3B-Instruct-2507
  - differs from official RAPTOR defaults [saved, official]: tree_builder.max_tokens (leaf size): [500, 100]; tree_builder.summarization_length: [150, 100]; embedding model: ['facebook/contriever', 'OpenAI text-embedding-ada-002']; summarization model: ['Qwen/Qwen3-30B-A3B-Instruct-2507', 'gpt-3.5-turbo']
- NovelHopQA arag: model = Qwen/Qwen3-30B-A3B-Instruct-2507; max_loops = 15; max_context_tokens = 32768; message_token_guard = 30720; max_completion_tokens = 2048; embedding_model = Qwen/Qwen3-Embedding-0.6B; workers = 4; upstream = https://github.com/Ayanami0730/arag; chunks = 7736

## What is missing and what running it would cost

| Dataset | Item | Status | Estimated wall hours (4 GPUs) | GPU hours | Basis |
| --- | --- | --- | --- | --- | --- |
| musique_32k | RAPTOR (default settings, build trees for the C5 documents + retrieve) | missing | — | — | no saved profiling |
| musique_32k | Reader, 7200 prompts (5.8M prompt tokens) over the available conditions | to run in the GPU script | 0.4 | 1.7 | rough: 8000 prompt tok/s, 1500 output tok/s, 0.15 h start |
| novelhopqa | Reader, 9988 prompts (31.9M prompt tokens) over the available conditions | to run in the GPU script | 1.4 | 5.5 | rough: 8000 prompt tok/s, 1500 output tok/s, 0.15 h start |

## Checks of unchanged inputs

- docs/preserved_sources.json: 93 files checked, changed 0, missing 0 [] 
- configs/methods/sadi.yaml sha256 18846ef24b6075586cc4ed53e1e8b6f7a916f0253e5932aa203b0ec741bbfdf0

## Files written

- `outputs/rebuttal/c5_end_to_end/contexts/musique_32k/closed_book.jsonl.gz`
- `outputs/rebuttal/c5_end_to_end/contexts/musique_32k/oracle.jsonl.gz`
- `outputs/rebuttal/c5_end_to_end/contexts/musique_32k/vanilla.jsonl.gz`
- `outputs/rebuttal/c5_end_to_end/contexts/musique_32k/saadi.jsonl.gz`
- `outputs/rebuttal/c5_end_to_end/contexts/musique_32k/cr.jsonl.gz`
- `outputs/rebuttal/c5_end_to_end/contexts/musique_32k/arag.jsonl.gz`
- `outputs/rebuttal/c5_end_to_end/contexts/musique_32k/arag_tm.jsonl.gz`
- `outputs/rebuttal/c5_end_to_end/contexts/musique_32k/saadi_at_arag_len.jsonl.gz`
- `outputs/rebuttal/c5_end_to_end/contexts/musique_32k/arag_own.jsonl.gz`
- `outputs/rebuttal/c5_end_to_end/contexts/musique_32k/manifest.json`
- `outputs/rebuttal/c5_end_to_end/contexts/novelhopqa/closed_book.jsonl.gz`
- `outputs/rebuttal/c5_end_to_end/contexts/novelhopqa/oracle.jsonl.gz`
- `outputs/rebuttal/c5_end_to_end/contexts/novelhopqa/vanilla.jsonl.gz`
- `outputs/rebuttal/c5_end_to_end/contexts/novelhopqa/saadi.jsonl.gz`
- `outputs/rebuttal/c5_end_to_end/contexts/novelhopqa/cr.jsonl.gz`
- `outputs/rebuttal/c5_end_to_end/contexts/novelhopqa/raptor.jsonl.gz`
- `outputs/rebuttal/c5_end_to_end/contexts/novelhopqa/raptor_tm.jsonl.gz`
- `outputs/rebuttal/c5_end_to_end/contexts/novelhopqa/arag.jsonl.gz`
- `outputs/rebuttal/c5_end_to_end/contexts/novelhopqa/arag_tm.jsonl.gz`
- `outputs/rebuttal/c5_end_to_end/contexts/novelhopqa/saadi_at_raptor_len.jsonl.gz`
- `outputs/rebuttal/c5_end_to_end/contexts/novelhopqa/saadi_at_arag_len.jsonl.gz`
- `outputs/rebuttal/c5_end_to_end/contexts/novelhopqa/arag_own.jsonl.gz`
- `outputs/rebuttal/c5_end_to_end/contexts/novelhopqa/manifest.json`
- `results/rebuttal/c5_end_to_end/prepare/context_stats_musique_32k.csv`
- `results/rebuttal/c5_end_to_end/prepare/context_stats_novelhopqa.csv`
- `experiments/rebuttal/c5_end_to_end/reports/step_00.md`
- `experiments/rebuttal/c5_end_to_end/reports/step_00.json`
