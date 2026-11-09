**C1 Contextual Retrieval (CR) vs SAADI (dry run: stub contexts, BM25 only; every number is a placeholder)**

- Setup: Anthropic's CR prompt with SAADI's construction LLM (Qwen3-30B-A3B-Instruct-2507, greedy), the paper's chunks, queries and labels, the five Table 1 retrievers, HR@5/@10, paired bootstrap (10,000, seed 42). Nothing tuned.
- Per retriever, SAADI − CR @10: SAADI above in 0/4 pairs, CR above in 0, n.s. 4 (@5: 0/0/4).
- Hybrid dense + BM25 (Anthropic's configuration): SAADI above in 0/4, CR above in 0, n.s. 4 (@5: 0/0/4).
- CR vs Vanilla @10: above in 0/4, below in 0.
- LooGLE BM25 HR@10: Vanilla 90.5, CR 90.5, SAADI 95.2 (SAADI − CR +4.8 [+0.0, +14.3]).
- MuSiQue-32K BM25 HR@10: Vanilla 81.0, CR 76.2, SAADI 85.7 (SAADI − CR +9.5 [+0.0, +21.4]).
- QASPER-64K BM25 HR@10: Vanilla 20.2, CR 24.0, SAADI 25.6 (SAADI − CR +1.6 [-4.7, +7.8]).
- NovelHopQA BM25 HR@10: Vanilla 25.5, CR 25.5, SAADI 25.5 (SAADI − CR +0.0 [-2.1, +2.1]).
- Cost: CR 3429 calls vs SAADI 4271; input tokens 38.0x SAADI nominal, 0.22x after prefix-cache hits; generated 148K vs 2.9M.
- Complementarity (SAADI fusion on CR text, not a method): above CR in 2/4, above SAADI in 1/4 @10.
- Differences from Anthropic: Qwen instead of Claude 3 Haiku, temperature 0, no reranker, SAADI's BM25 (one field) instead of Elasticsearch (two fields), document rebuilt from the chunks, long documents split into segments.
- Long documents: NovelHopQA 1 docs split ({'1': 1, '2': 1}).
