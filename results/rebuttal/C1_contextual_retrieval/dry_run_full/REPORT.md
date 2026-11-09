# C1: Contextual Retrieval on SAADI's benchmarks (dry run full: stub contexts, BM25 only; every number is a placeholder)

Written 2026-11-09 15:00:46; commit e2dbaa178fa0. Datasets with a report: LooGLE, MuSiQue-32K, QASPER-64K, NovelHopQA

## Summary

1. Per retriever (same retriever, same chunks, same queries), SAADI − CR at HR@10: SAADI above with the 95% interval excluding zero in 2/4 (dataset, retriever) pairs, CR above in 0, neither in 2; at HR@5: 4 / 0 / 0.
2. Hybrid (dense + BM25, cookbook fusion, the configuration Anthropic reports): SAADI above in 2/4 pairs at HR@10, CR above in 0, neither in 2; at HR@5: 4 / 0 / 0.
3. CR vs Vanilla: CR above Vanilla with the interval excluding zero in 1/4 pairs at HR@10 (below in 0); at HR@5 0 above, 2 below.
4. Cost, all datasets: CR 44618 LLM calls vs SAADI 55475; CR input tokens 3.95B nominal (28.5x SAADI's 138.6M) and 25.6M after prefix-cache hits (0.18x); generated tokens 2.0M vs 37.1M (SAADI: its per-chunk rates from the paper's cost table x C1's chunks).
5. Complementarity check (SAADI's published fusion with the text channel on CR text; not a proposed method): above CR in 3/4 pairs at HR@10 (below in 0); above SAADI in 1/4 (below in 1).

## Deliberate differences from Anthropic's setup

- Construction LLM: Qwen3-30B-A3B-Instruct-2507 (SAADI's construction model) instead of Claude 3 Haiku; greedy decoding (temperature 0), max_tokens 1024 (the cookbook allows 2048), vLLM with SAADI's saved runtime (bf16, TP 4, max_num_seqs 16) and the model's native window.
- No reranker: SAADI uses none, and a reranker could sit on top of either method.
- BM25: SAADI's own BM25 (the paper's Vanilla) on the contextualized text as one field, instead of Elasticsearch with the English analyzer searching chunk and context as two fields (multi_match); only the indexed text changes.
- Document text: rebuilt from the saved chunks joined by a single space (' '). The chunker's sentence splitter already changed some whitespace inside chunks (e.g. '1912. 02866'), so no separator restores the source exactly; the rebuilt text equals the source up to whitespace. It is the text SAADI read.
- Long documents: when document + prompt + chunk + 1024 output tokens exceed the window, the document is split at chunk boundaries into the smallest number k of contiguous segments of near-equal token length that each fit, and each chunk sees its own segment.
- Prompt sent as one user message with the document first (Anthropic sends two content blocks so the document can be cached); vLLM's prefix cache plays the role of Anthropic's prompt caching.
- Dense retrievers: the paper's adapters with their own truncation (Contriever, Jina and Qwen 512 tokens, BGE-M3 8192); the context comes first, so on long chunks the end of the chunk is what gets cut (counted per retriever).
- Hybrid: the cookbook's fusion (top 150 of each list, 0.8/rank_dense + 0.2/rank_bm25, ties as the cookbook sorts them: higher chunk position first), applied identically to Vanilla, CR and SAADI.
- Vanilla is SAADI's text channel on the saved chunks (decided with the user): on MuSiQue-32K and QASPER-64K it equals Table 1's Vanilla to the rounding; on LooGLE and NovelHopQA Table 1's Vanilla came from another run, and both are shown.

## Per retriever (HR@5 / HR@10, %)

| Dataset | Retriever | Vanilla | Table 1 Vanilla | CR | SAADI | SAADI on CR text* | LQCA | CR − Vanilla @10 [CI] | SAADI − CR @10 [CI] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LooGLE | BM25 | 91.9 / 95.7 | 91.5 / 95.4 | 91.3 / 95.7 | 92.3 / 96.5 | 91.8 / 96.0 | 88.8 / 94.2 | -0.1 [-0.4, +0.3] | +0.8 [+0.0, +1.5] |
| MuSiQue-32K | BM25 | 57.6 / 74.0 | 57.6 / 74.0 | 57.3 / 76.4 | 62.8 / 77.7 | 63.4 / 80.2 | 83.6 / 91.8 | +2.4 [+0.1, +4.7] | +1.2 [-1.1, +3.6] |
| QASPER-64K | BM25 | 14.4 / 22.1 | 14.4 / 22.1 | 14.6 / 23.0 | 17.0 / 26.1 | 17.1 / 27.2 | 36.9 / 47.0 | +0.9 [-0.6, +2.5] | +3.2 [+1.4, +4.9] |
| NovelHopQA | BM25 | 17.5 / 20.7 | 16.9 / 20.0 | 16.7 / 20.7 | 18.1 / 22.0 | 17.5 / 21.9 | 1.4 / 1.9 | +0.0 [-0.6, +0.6] | +1.3 [+0.1, +2.5] |
*Complementarity check, not a proposed method. HR@5 intervals are in the per-dataset reports and tables/per_retriever.csv. SAADI held-out column left out: C2's REPORT.md does not exist yet.

## Hybrid (dense + BM25, HR@5 / HR@10, %)

| Dataset | Hybrid | Vanilla | CR | SAADI | CR − Vanilla @10 | SAADI − CR @10 | SAADI − CR @5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LooGLE | BM25 + BM25 (dry-run stand-in for a dense retriever) | 91.9 / 95.7 | 91.3 / 95.7 | 92.3 / 96.5 | -0.1 [-0.4, +0.3] | +0.8 [+0.0, +1.5] | +1.0 [+0.2, +1.9] |
| MuSiQue-32K | BM25 + BM25 (dry-run stand-in for a dense retriever) | 57.6 / 74.0 | 57.3 / 76.4 | 62.8 / 77.7 | +2.4 [+0.1, +4.7] | +1.2 [-1.1, +3.6] | +5.4 [+2.8, +8.2] |
| QASPER-64K | BM25 + BM25 (dry-run stand-in for a dense retriever) | 14.4 / 22.1 | 14.6 / 23.0 | 17.0 / 26.1 | +0.9 [-0.6, +2.5] | +3.2 [+1.4, +4.9] | +2.4 [+0.8, +4.0] |
| NovelHopQA | BM25 + BM25 (dry-run stand-in for a dense retriever) | 17.5 / 20.7 | 16.7 / 20.7 | 18.1 / 22.0 | +0.0 [-0.6, +0.6] | +1.3 [+0.1, +2.5] | +1.3 [+0.2, +2.4] |

## Cost per dataset

| Dataset | CR calls | SAADI calls (its rate per chunk x C1's chunks) | CR input/chunk nominal | CR input/chunk computed | SAADI input/chunk | CR generated/chunk | SAADI generated/chunk | CR cut at 1024 | CR generation GPU-h |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LooGLE | 3346 | 4046 | 26K | 1453 | 4608 | 45.5 | 1755.7 | 0.0% | 0.0 |
| MuSiQue-32K | 17102 | 21294 | 49K | 326 | 2697 | 46.3 | 697.2 | 0.0% | 0.0 |
| QASPER-64K | 17159 | 21404 | 92K | 316 | 2530 | 43.9 | 633.4 | 0.0% | 0.0 |
| NovelHopQA | 7011 | 8731 | 204K | 1392 | 4797 | 41.8 | 1199.0 | 0.0% | 0.0 |
- SAADI's builds had no prefix caching; its input tokens compare with CR's nominal column like for like and with the computed column as CR's best case.

## Appending text to a document

Not a measurement. When text is appended, CR needs the whole updated document as context for every new chunk (and the contexts already written may become stale, since they were written without the new text), while SAADI processes a new chunk with its current state. Input tokens per new chunk from the numbers above (CR without cache hits; with a cached prefix CR computes about the new chunk and the template, but the prompt still carries the whole document): LooGLE: CR 26K vs SAADI 4608; MuSiQue-32K: CR 49K vs SAADI 2697; QASPER-64K: CR 92K vs SAADI 2530; NovelHopQA: CR 204K vs SAADI 4797.

## Limits

- Long documents: LooGLE: 0 documents (0 chunks) over the window, segments {'1': 105}; MuSiQue-32K: 0 documents (0 chunks) over the window, segments {'1': 45}; QASPER-64K: 0 documents (0 chunks) over the window, segments {'1': 23}; NovelHopQA: 7 documents (4250 chunks) over the window, segments {'1': 9, '2': 6, '3': 1}. In those documents a chunk sees its own segment, not the whole book.
- Dense truncation: the context comes first, so a contextualized chunk longer than a retriever's input limit loses the end of the chunk (counts per retriever in the per-dataset reports).
- One construction LLM and one prompt; no reranker; results on these four benchmarks only.

## Files

- Per-dataset reports: experiments/rebuttal/C1_contextual_retrieval/reports/dry_run_full/<dataset>.md and .json; tables: results/rebuttal/C1_contextual_retrieval/dry_run_full/tables/per_retriever.csv, hybrid.csv, cost.csv; contexts and rankings: outputs/rebuttal/C1_contextual_retrieval/dry_run_full/<dataset>/.
