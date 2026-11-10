# C5 end-to-end QA - step 00 (prepare): OK

Written 2026-11-10 23:32:12; commit 53426a5e71b0; host eee-s132-infsys-gpu03; exit code 0.

Command: `bash experiments/rebuttal/c5_end_to_end/sh/00_prepare.sh --datasets novelhopqa --tag novelhopqa`

## Summary

- NovelHopQA: gate passed (SAADI 908/908 identical to the paper, HR@10 22.03; Vanilla HR@10 20.70). Contexts built: closed_book, oracle, oracle_para, vanilla, saadi, cr, raptor, raptor_tm, arag, arag_tm, saadi_at_raptor_len, saadi_at_arag_len, arag_own.

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
| oracle_para | 908 | 2.80 | 1793 | 7600 | 1861 | 7664 | 48.35 (908) | — |
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

Paragraph labels (`results/rebuttal/c5_end_to_end/prepare/labels_para_novelhopqa.jsonl.gz`): each 'Paragraph i:' piece of the dataset's context is located on its own in SAADI's chunks of the same book (the chunk contains it, it contains the chunk, or they overlap at a boundary by >= min(8, paragraph length) tokens; same token normalisation as the paper's labeler). Evidence 'all' = every located paragraph is in the context; 'any' = at least one.

| Hops | Queries | Paragraphs | Located % | Queries with none located | Chunks per query: paragraph labels | Chunks per query: paper labels | Paper-label chunks inside a paragraph % | Paper labels cover every paragraph |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 377 | 377 | 97.1 | 11 | 1.14 | 3.94 | 28.8 | 366 |
| 2 | 171 | 342 | 98.0 | 2 | 2.82 | 2.75 | 28.5 | 7 |
| 3 | 169 | 507 | 98.6 | 2 | 3.92 | 3.75 | 20.8 | 0 |
| 4 | 191 | 764 | 97.5 | 2 | 5.09 | 3.49 | 24.3 | 0 |

Evidence in context with the paragraph labels (any / every located paragraph; for the top-10 rows this is HR@10):

| Condition | Any % | All % | Eligible |
| --- | --- | --- | --- |
| closed_book | 0.0 | 0.0 | 891 |
| oracle | 82.0 | 41.9 | 891 |
| oracle_para | 100.0 | 100.0 | 891 |
| vanilla | 80.4 | 38.5 | 891 |
| saadi | 82.8 | 40.6 | 891 |
| cr | 80.4 | 38.8 | 891 |
| raptor | 9.9 | 5.6 | 891 |
| raptor_tm | 9.9 | 5.6 | 891 |
| arag | 39.3 | 18.3 | 891 |
| arag_tm | 39.3 | 18.3 | 891 |
| saadi_at_raptor_len | 61.6 | 24.6 | 891 |
| saadi_at_arag_len | 66.9 | 28.6 | 891 |
| arag_own | 39.3 | 18.3 | 891 |

## Comparison-method settings (defaults; nothing tuned)

- NovelHopQA cr: construction LLM = Qwen/Qwen3-30B-A3B-Instruct-2507; context max tokens = 1024; sampling = {'temperature': 0.0, 'top_p': 1.0, 'top_k': -1, 'repetition_penalty': 1.0, 'seed': 42}; prompt = Anthropic's published contextualizer prompt (C1 ANTHROPIC_PROMPT); index = SAADI's BM25 on context + '\n\n' + chunk; no reranker; reader input = the ORIGINAL chunk text of the CR top 10 (CR changes what is indexed, not the chunk)
- NovelHopQA raptor: tree_builder.max_tokens (leaf size) = 500; tree_builder.num_layers = 5; tree_builder.threshold = 0.5; tree_builder.top_k = 5; tree_builder.selection_mode = top_k; tree_builder.summarization_length = 150; retrieve.top_k = 10; retrieve.max_tokens = 3500; retrieve.collapse_tree = True; embedding model = facebook/contriever; summarization model = Qwen/Qwen3-30B-A3B-Instruct-2507
  - differs from official RAPTOR defaults [saved, official]: tree_builder.max_tokens (leaf size): [500, 100]; tree_builder.summarization_length: [150, 100]; embedding model: ['facebook/contriever', 'OpenAI text-embedding-ada-002']; summarization model: ['Qwen/Qwen3-30B-A3B-Instruct-2507', 'gpt-3.5-turbo']
- NovelHopQA arag: model = Qwen/Qwen3-30B-A3B-Instruct-2507; max_loops = 15; max_context_tokens = 32768; message_token_guard = 30720; max_completion_tokens = 2048; embedding_model = Qwen/Qwen3-Embedding-0.6B; workers = 4; upstream = https://github.com/Ayanami0730/arag; chunks = 7736

## What is missing and what running it would cost

| Dataset | Item | Status | Estimated wall hours (4 GPUs) | GPU hours | Basis |
| --- | --- | --- | --- | --- | --- |
| novelhopqa | Reader, 10896 prompts (33.6M prompt tokens) over the available conditions | to run in the GPU script | 1.4 | 5.8 | rough: 8000 prompt tok/s, 1500 output tok/s, 0.15 h start |

## Checks of unchanged inputs

- docs/preserved_sources.json: 93 files checked, changed 0, missing 0 [] 
- configs/methods/sadi.yaml sha256 18846ef24b6075586cc4ed53e1e8b6f7a916f0253e5932aa203b0ec741bbfdf0

## Files written

- `results/rebuttal/c5_end_to_end/prepare/labels_para_novelhopqa.jsonl.gz`
- `outputs/rebuttal/c5_end_to_end/contexts/novelhopqa/closed_book.jsonl.gz`
- `outputs/rebuttal/c5_end_to_end/contexts/novelhopqa/oracle.jsonl.gz`
- `outputs/rebuttal/c5_end_to_end/contexts/novelhopqa/oracle_para.jsonl.gz`
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
- `outputs/rebuttal/c5_end_to_end/contexts/novelhopqa/evidence_para.jsonl.gz`
- `outputs/rebuttal/c5_end_to_end/contexts/novelhopqa/manifest.json`
- `results/rebuttal/c5_end_to_end/prepare/context_stats_novelhopqa.csv`
- `experiments/rebuttal/c5_end_to_end/reports/step_00_novelhopqa.md`
- `experiments/rebuttal/c5_end_to_end/reports/step_00_novelhopqa.json`

## Push

```
pushed dc92342 to origin/fix/notebook-key-and-install: C5 all 00_prepare_novelhopqa: gate passed on 1 datasets; contexts written; nothing missing
```
