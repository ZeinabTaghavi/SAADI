# B1 Oracle upper bound

The ceiling for any bridging method: `build.py` copies each book and appends form 2 (the
name the query uses) to the gold chunk of every persona whose form 2 is revealed in the book;
`retrieve.py` ranks the copy with the same BM25 as B0. Cell 5 has no reveal, so a perfect link
adds nothing there and its gold chunks stay unchanged (otherwise the cell-5 control would be
broken by construction). The gap between any system and B1 is what the link itself, and the
scoring after it, still costs. No LLM.
