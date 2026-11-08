# B0 Vanilla BM25

The probe's own Vanilla column, recomputed through the shared ranking format so that every
system goes through one scorer. `retrieve.py` ranks all chunks of a book with the same
function the probe's step 7 uses (`mbr._score_pool`, BM25 over the raw chunk texts, ties by
chunk position) and checks the result against run 0's per-persona Vanilla ranks and against
`results/probe_bridge/table.csv`. `build.py` only records the index size: the BM25 index is
built in memory per query. No LLM.
