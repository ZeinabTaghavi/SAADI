**C1 Contextual Retrieval (CR) vs SAADI (dry run full: stub contexts, BM25 only; every number is a placeholder)**

- Setup: Anthropic's CR prompt with SAADI's construction LLM (Qwen3-30B-A3B-Instruct-2507, greedy), the paper's chunks, queries and labels, the five Table 1 retrievers, HR@5/@10, paired bootstrap (10,000, seed 42). Nothing tuned.
- Per retriever, SAADI − CR @10: SAADI above in 2/4 pairs, CR above in 0, n.s. 2 (@5: 4/0/0).
- Hybrid dense + BM25 (Anthropic's configuration): SAADI above in 2/4, CR above in 0, n.s. 2 (@5: 4/0/0).
- CR vs Vanilla @10: above in 1/4, below in 0.
- LooGLE BM25 HR@10: Vanilla 95.7, CR 95.7, SAADI 96.5 (SAADI − CR +0.8 [+0.0, +1.5]).
- MuSiQue-32K BM25 HR@10: Vanilla 74.0, CR 76.4, SAADI 77.7 (SAADI − CR +1.2 [-1.1, +3.6]).
- QASPER-64K BM25 HR@10: Vanilla 22.1, CR 23.0, SAADI 26.1 (SAADI − CR +3.2 [+1.4, +4.9]).
- NovelHopQA BM25 HR@10: Vanilla 20.7, CR 20.7, SAADI 22.0 (SAADI − CR +1.3 [+0.1, +2.5]).
- Cost: CR 44618 calls vs SAADI 55475; input tokens 28.5x SAADI nominal, 0.18x after prefix-cache hits; generated 2.0M vs 37.1M.
- Complementarity (SAADI fusion on CR text, not a method): above CR in 3/4, above SAADI in 1/4 @10.
- Differences from Anthropic: Qwen instead of Claude 3 Haiku, temperature 0, no reranker, SAADI's BM25 (one field) instead of Elasticsearch (two fields), document rebuilt from the chunks, long documents split into segments.
- Long documents: NovelHopQA 7 docs split ({'1': 9, '2': 6, '3': 1}).
