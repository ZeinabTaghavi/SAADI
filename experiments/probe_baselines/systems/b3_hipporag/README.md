# B3 HippoRAG v2

The official implementation (`hipporag` 2.0.0a4, OSU-NLP-Group) as a dependency, in its own
virtual environment under `third_party/` because it pins torch 2.5.1, transformers 4.45.2 and
vLLM 0.6.6. `build.py` builds one index per book from the probe's chunks: OpenIE with the
construction LLM through the local vLLM server's OpenAI-compatible endpoint, NV-Embed-v2
embeddings, the knowledge graph with passage nodes. `retrieve.py` runs the package's
retrieval (fact retrieval, LLM recognition-memory filter, personalized PageRank) for the
book's 24 queries, asks for every chunk so the gold rank is exact, and maps the returned
passages back to chunk ids by their text. It also checks whether the extracted triples hold
one that connects form 1 and form 2, which is what HippoRAG would need to bridge the names.

Settings are the package defaults, decided with the user because the paper's Appendix E.6
was not available; the report states that they were not checked against it. HippoRAG caches
every LLM reply and its OpenIE results, so an interrupted book resumes without new calls.
