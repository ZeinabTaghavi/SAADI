**C1 Contextual Retrieval (CR) vs SAADI**

- Setup: Anthropic's CR prompt with SAADI's construction LLM (Qwen3-30B-A3B-Instruct-2507, greedy), the paper's chunks, queries and labels, the five Table 1 retrievers, HR@5/@10, paired bootstrap (10,000, seed 42). Nothing tuned.
- Per retriever, SAADI − CR @10: SAADI above in 4/20 pairs, CR above in 3, n.s. 13 (@5: 4/2/14).
- Hybrid dense + BM25 (Anthropic's configuration): SAADI above in 2/16, CR above in 4, n.s. 10 (@5: 1/2/13).
- CR vs Vanilla @10: above in 8/20, below in 4.
- LooGLE BM25 HR@10: Vanilla 95.7, CR 96.0, SAADI 96.5 (SAADI − CR +0.5 [-0.2, +1.2]).
- MuSiQue-32K BM25 HR@10: Vanilla 74.0, CR 82.2, SAADI 77.7 (SAADI − CR -4.6 [-6.8, -2.3]).
- QASPER-64K BM25 HR@10: Vanilla 22.1, CR 26.9, SAADI 26.1 (SAADI − CR -0.8 [-2.7, +1.1]).
- NovelHopQA BM25 HR@10: Vanilla 20.7, CR 21.7, SAADI 22.0 (SAADI − CR +0.3 [-1.0, +1.8]).
- Cost: CR 44618 calls vs SAADI 55475; input tokens 28.5x SAADI nominal, 0.19x after prefix-cache hits; generated 4.7M vs 37.1M.
- Complementarity (SAADI fusion on CR text, not a method): above CR in 3/20, above SAADI in 7/20 @10.
- Differences from Anthropic: Qwen instead of Claude 3 Haiku, temperature 0, no reranker, SAADI's BM25 (one field) instead of Elasticsearch (two fields), document rebuilt from the chunks, long documents split into segments.
- Long documents: NovelHopQA 7 docs split ({'1': 9, '2': 6, '3': 1}).
