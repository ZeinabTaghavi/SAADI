# C1: Contextual Retrieval on SAADI's benchmarks

Written 2026-11-10 15:02:01; commit 0df9d38e4fd5. Datasets with a report: LooGLE, MuSiQue-32K, QASPER-64K, NovelHopQA

## Summary

1. Per retriever (same retriever, same chunks, same queries), SAADI − CR at HR@10: SAADI above with the 95% interval excluding zero in 4/20 (dataset, retriever) pairs, CR above in 3, neither in 13; at HR@5: 4 / 2 / 14.
2. Hybrid (dense + BM25, cookbook fusion, the configuration Anthropic reports): SAADI above in 2/16 pairs at HR@10, CR above in 4, neither in 10; at HR@5: 1 / 2 / 13.
3. CR vs Vanilla: CR above Vanilla with the interval excluding zero in 8/20 pairs at HR@10 (below in 4); at HR@5 7 above, 5 below.
4. Cost, all datasets: CR 44618 LLM calls vs SAADI 55475; CR input tokens 3.95B nominal (28.5x SAADI's 138.6M) and 26.9M after prefix-cache hits (0.19x); generated tokens 4.7M vs 37.1M (SAADI: its per-chunk rates from the paper's cost table x C1's chunks).
5. Complementarity check (SAADI's published fusion with the text channel on CR text; not a proposed method): above CR in 3/20 pairs at HR@10 (below in 0); above SAADI in 7/20 (below in 4).

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

| Dataset | Retriever | Vanilla | Table 1 Vanilla | CR | SAADI | SAADI on CR text* | LQCA | CR − Vanilla @10 [CI] | SAADI − CR @10 [CI] | SAADI held-out (C2) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LooGLE | BM25 | 91.9 / 95.7 | 91.5 / 95.4 | 92.0 / 96.0 | 92.3 / 96.5 | 92.4 / 96.1 | 88.8 / 94.2 | +0.2 [-0.3, +0.8] | +0.5 [-0.2, +1.2] | 92.3 / 96.2 |
| LooGLE | Contriever | 75.2 / 86.0 | 72.4 / 73.3 | 72.5 / 84.3 | 75.4 / 85.7 | 72.4 / 84.2 | 73.3 / 84.5 | -1.7 [-3.2, -0.2] | +1.4 [-0.2, +2.9] | 76.3 / 86.2 |
| LooGLE | BGE-M3 | 82.5 / 90.7 | 80.6 / 89.8 | 82.2 / 90.1 | 81.1 / 90.1 | 79.9 / 89.6 | 74.8 / 84.1 | -0.7 [-2.0, +0.7] | +0.1 [-1.3, +1.4] | 82.3 / 90.6 |
| LooGLE | Jina | 69.9 / 80.7 | 70.1 / 80.1 | 65.8 / 79.7 | 70.2 / 82.0 | 67.5 / 80.6 | 68.6 / 79.1 | -1.0 [-2.7, +0.7] | +2.3 [+0.6, +4.0] | — / — |
| LooGLE | Qwen | 77.9 / 86.2 | 76.9 / 85.9 | 75.4 / 84.7 | 77.1 / 86.4 | 75.0 / 85.3 | 81.4 / 86.8 | -1.5 [-3.1, -0.1] | +1.7 [+0.2, +3.3] | — / — |
| MuSiQue-32K | BM25 | 57.6 / 74.0 | 57.6 / 74.0 | 65.0 / 82.2 | 62.8 / 77.7 | 68.3 / 83.4 | 83.6 / 91.8 | +8.2 [+6.0, +10.4] | -4.6 [-6.8, -2.3] | 64.7 / 79.9 |
| MuSiQue-32K | Contriever | 59.9 / 76.1 | 59.9 / 76.1 | 69.3 / 86.1 | 68.8 / 83.7 | 73.6 / 87.9 | 80.8 / 89.0 | +10.0 [+7.6, +12.4] | -2.4 [-4.6, -0.3] | 68.8 / 84.9 |
| MuSiQue-32K | BGE-M3 | 73.2 / 87.4 | 73.2 / 87.4 | 77.4 / 89.7 | 76.9 / 88.3 | 79.4 / 90.6 | 76.7 / 90.4 | +2.2 [+0.6, +4.0] | -1.3 [-3.1, +0.4] | 76.1 / 90.8 |
| MuSiQue-32K | Jina | 71.7 / 85.8 | 71.6 / 85.8 | 77.1 / 89.7 | 73.6 / 88.1 | 76.7 / 90.0 | 79.4 / 89.0 | +3.9 [+2.2, +5.7] | -1.6 [-3.3, +0.2] | — / — |
| MuSiQue-32K | Qwen | 75.3 / 89.0 | 75.3 / 89.0 | 75.3 / 88.6 | 76.4 / 90.0 | 75.6 / 89.8 | 82.2 / 90.4 | -0.4 [-2.4, +1.6] | +1.4 [-0.6, +3.4] | — / — |
| QASPER-64K | BM25 | 14.4 / 22.1 | 14.4 / 22.1 | 17.6 / 26.9 | 17.0 / 26.1 | 18.7 / 29.4 | 36.9 / 47.0 | +4.9 [+3.2, +6.6] | -0.8 [-2.7, +1.1] | 17.3 / 27.3 |
| QASPER-64K | Contriever | 15.2 / 23.3 | 15.2 / 23.3 | 16.7 / 26.0 | 16.2 / 25.8 | 17.6 / 27.6 | 34.4 / 46.0 | +2.7 [+0.9, +4.5] | -0.2 [-2.1, +1.7] | 15.8 / 25.2 |
| QASPER-64K | BGE-M3 | 21.0 / 29.3 | 21.0 / 29.3 | 23.9 / 33.9 | 20.9 / 30.9 | 22.6 / 33.4 | 32.9 / 45.8 | +4.6 [+2.7, +6.5] | -3.0 [-5.0, -1.1] | 22.1 / 31.9 |
| QASPER-64K | Jina | 16.0 / 25.0 | 15.7 / 24.8 | 17.9 / 28.1 | 18.8 / 27.1 | 18.1 / 27.8 | 31.7 / 41.8 | +3.1 [+1.2, +5.0] | -1.0 [-2.9, +1.0] | — / — |
| QASPER-64K | Qwen | 22.6 / 32.6 | 22.6 / 32.6 | 15.8 / 25.5 | 23.3 / 34.1 | 20.9 / 31.4 | 39.7 / 53.6 | -7.1 [-9.2, -4.8] | +8.6 [+6.4, +11.0] | — / — |
| NovelHopQA | BM25 | 17.5 / 20.7 | 16.9 / 20.0 | 18.4 / 21.7 | 18.1 / 22.0 | 18.4 / 23.2 | 1.4 / 1.9 | +1.0 [-0.4, +2.4] | +0.3 [-1.0, +1.8] | 17.6 / 21.1 |
| NovelHopQA | Contriever | 12.4 / 17.8 | 11.8 / 11.8 | 14.1 / 18.5 | 13.5 / 18.2 | 14.6 / 18.7 | 0.9 / 1.7 | +0.7 [-1.2, +2.6] | -0.3 [-2.1, +1.4] | 13.2 / 18.1 |
| NovelHopQA | BGE-M3 | 17.3 / 21.9 | 14.1 / 20.7 | 15.5 / 21.4 | 16.5 / 22.1 | 15.1 / 19.8 | 1.4 / 2.2 | -0.6 [-2.6, +1.4] | +0.8 [-1.2, +2.8] | 17.1 / 22.7 |
| NovelHopQA | Jina | 16.0 / 20.2 | 14.8 / 19.1 | 15.0 / 19.4 | 14.8 / 20.9 | 14.0 / 19.9 | 2.1 / 2.2 | -0.8 [-2.9, +1.3] | +1.5 [-0.4, +3.5] | — / — |
| NovelHopQA | Qwen | 16.7 / 20.5 | 14.6 / 19.8 | 11.8 / 16.7 | 15.2 / 20.6 | 12.7 / 18.2 | 2.2 / 2.8 | -3.7 [-6.1, -1.4] | +3.9 [+1.7, +6.2] | — / — |
*Complementarity check, not a proposed method. HR@5 intervals are in the per-dataset reports and tables/per_retriever.csv.

## Hybrid (dense + BM25, HR@5 / HR@10, %)

| Dataset | Hybrid | Vanilla | CR | SAADI | CR − Vanilla @10 | SAADI − CR @10 | SAADI − CR @5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LooGLE | Contriever + BM25 | 89.3 / 95.6 | 88.3 / 95.6 | 88.3 / 95.2 | +0.0 [-0.8, +0.8] | -0.4 [-1.3, +0.5] | +0.1 [-1.3, +1.4] |
| LooGLE | BGE-M3 + BM25 | 91.4 / 96.1 | 91.6 / 96.5 | 90.4 / 96.0 | +0.4 [-0.3, +1.1] | -0.5 [-1.3, +0.3] | -1.1 [-2.3, +0.1] |
| LooGLE | Jina + BM25 | 88.2 / 94.7 | 87.4 / 94.9 | 87.1 / 94.8 | +0.2 [-0.6, +1.0] | -0.1 [-1.0, +0.8] | -0.3 [-1.6, +1.1] |
| LooGLE | Qwen + BM25 | 90.4 / 96.4 | 89.8 / 95.6 | 89.5 / 95.9 | -0.8 [-1.6, +0.0] | +0.3 [-0.5, +1.1] | -0.4 [-1.6, +0.9] |
| MuSiQue-32K | Contriever + BM25 | 61.3 / 78.9 | 72.0 / 88.0 | 69.8 / 83.9 | +9.1 [+6.9, +11.3] | -4.1 [-6.1, -2.1] | -2.2 [-4.8, +0.3] |
| MuSiQue-32K | BGE-M3 + BM25 | 71.7 / 87.4 | 77.6 / 90.4 | 75.1 / 87.7 | +3.0 [+1.4, +4.7] | -2.8 [-4.6, -1.1] | -2.4 [-4.9, +0.0] |
| MuSiQue-32K | Jina + BM25 | 71.9 / 85.4 | 76.7 / 89.9 | 73.8 / 87.7 | +4.4 [+2.7, +6.3] | -2.2 [-4.0, -0.4] | -2.9 [-5.3, -0.6] |
| MuSiQue-32K | Qwen + BM25 | 73.3 / 88.9 | 76.7 / 89.7 | 75.0 / 89.7 | +0.8 [-1.2, +2.7] | +0.0 [-2.0, +2.0] | -1.7 [-4.2, +0.9] |
| QASPER-64K | Contriever + BM25 | 15.8 / 25.3 | 17.6 / 28.3 | 17.3 / 26.9 | +3.0 [+1.3, +4.7] | -1.4 [-3.2, +0.6] | -0.2 [-1.9, +1.5] |
| QASPER-64K | BGE-M3 + BM25 | 21.4 / 29.4 | 24.4 / 34.1 | 21.4 / 30.6 | +4.7 [+2.8, +6.8] | -3.5 [-5.5, -1.6] | -3.0 [-4.9, -1.1] |
| QASPER-64K | Jina + BM25 | 17.1 / 26.8 | 18.8 / 29.9 | 19.3 / 28.4 | +3.2 [+1.2, +5.1] | -1.5 [-3.5, +0.5] | +0.5 [-1.4, +2.4] |
| QASPER-64K | Qwen + BM25 | 22.4 / 32.5 | 17.3 / 29.2 | 23.1 / 34.5 | -3.3 [-5.5, -1.1] | +5.3 [+3.2, +7.7] | +5.9 [+3.7, +8.1] |
| NovelHopQA | Contriever + BM25 | 16.6 / 22.1 | 18.1 / 22.9 | 16.9 / 22.6 | +0.8 [-1.1, +2.6] | -0.3 [-2.1, +1.4] | -1.2 [-2.9, +0.4] |
| NovelHopQA | BGE-M3 + BM25 | 19.9 / 25.2 | 18.5 / 24.4 | 19.6 / 24.9 | -0.8 [-2.5, +1.0] | +0.4 [-1.3, +2.3] | +1.1 [-0.7, +2.9] |
| NovelHopQA | Jina + BM25 | 19.4 / 24.3 | 19.1 / 24.4 | 19.1 / 25.2 | +0.1 [-1.9, +2.1] | +0.8 [-1.2, +2.8] | +0.0 [-1.8, +1.8] |
| NovelHopQA | Qwen + BM25 | 20.3 / 23.9 | 16.9 / 23.1 | 18.6 / 25.3 | -0.8 [-2.8, +1.2] | +2.2 [+0.2, +4.2] | +1.8 [-0.2, +3.7] |

## Cost per dataset

| Dataset | CR calls | SAADI calls (its rate per chunk x C1's chunks) | CR input/chunk nominal | CR input/chunk computed | SAADI input/chunk | CR generated/chunk | SAADI generated/chunk | CR cut at 1024 | CR generation GPU-h |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LooGLE | 3346 | 4046 | 26K | 1478 | 4608 | 130.1 | 1755.7 | 0.0% | 2.4 |
| MuSiQue-32K | 17102 | 21294 | 49K | 349 | 2697 | 95.0 | 697.2 | 0.0% | 8.2 |
| QASPER-64K | 17159 | 21404 | 92K | 351 | 2530 | 98.8 | 633.4 | 0.0% | 11.7 |
| NovelHopQA | 7011 | 8731 | 204K | 1427 | 4797 | 138.0 | 1199.0 | 0.0% | 2.7 |
- SAADI's builds had no prefix caching; its input tokens compare with CR's nominal column like for like and with the computed column as CR's best case.

## Appending text to a document

Not a measurement. When text is appended, CR needs the whole updated document as context for every new chunk (and the contexts already written may become stale, since they were written without the new text), while SAADI processes a new chunk with its current state. Input tokens per new chunk from the numbers above (CR without cache hits; with a cached prefix CR computes about the new chunk and the template, but the prompt still carries the whole document): LooGLE: CR 26K vs SAADI 4608; MuSiQue-32K: CR 49K vs SAADI 2697; QASPER-64K: CR 92K vs SAADI 2530; NovelHopQA: CR 204K vs SAADI 4797.

## Limits

- Long documents: LooGLE: 0 documents (0 chunks) over the window, segments {'1': 105}; MuSiQue-32K: 0 documents (0 chunks) over the window, segments {'1': 45}; QASPER-64K: 0 documents (0 chunks) over the window, segments {'1': 23}; NovelHopQA: 7 documents (4250 chunks) over the window, segments {'1': 9, '2': 6, '3': 1}. In those documents a chunk sees its own segment, not the whole book.
- Dense truncation: the context comes first, so a contextualized chunk longer than a retriever's input limit loses the end of the chunk. Affected: LooGLE/Contriever 98% of chunks, 123 extra tokens cut per chunk; LooGLE/Jina 99% of chunks, 147 extra tokens cut per chunk; LooGLE/Qwen 99% of chunks, 127 extra tokens cut per chunk; NovelHopQA/Contriever 100% of chunks, 139 extra tokens cut per chunk; NovelHopQA/Jina 100% of chunks, 164 extra tokens cut per chunk; NovelHopQA/Qwen 100% of chunks, 137 extra tokens cut per chunk. On these pairs BGE-M3 (8192 tokens) and BM25 are the comparisons without the effect.
- One construction LLM and one prompt; no reranker; results on these four benchmarks only.

## Files

- Per-dataset reports: experiments/rebuttal/C1_contextual_retrieval/reports/<dataset>.md and .json; tables: results/rebuttal/C1_contextual_retrieval/tables/per_retriever.csv, hybrid.csv, cost.csv; contexts and rankings: outputs/rebuttal/C1_contextual_retrieval/<dataset>/.
