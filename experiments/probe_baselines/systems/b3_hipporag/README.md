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

Installation and parsing follow the user's SAADI_Hipporag repository (pinned commit, cloned by
`run_all.sh` into `third_party/SAADI_Hipporag`). `pip install hipporag==2.0.0a4` cannot resolve,
because the package's metadata pins `openai==1.91.1`, which is not on PyPI. SAADI_Hipporag's
`requirements.txt` is its tested set (openai 1.58.1 and litellm 1.55.8, the rest as the
package pins), and hipporag then goes in with `--no-deps`; `check_env.py` checks the versions.
B3 also uses that repository's two parser fixes, as its runner applies them: `RobustOpenIE`
replaces the package's OpenIE object (same prompts; Qwen's list and typed-object replies
are parsed safely instead of with `eval`; one retry, then up to two schema-constrained
rescue requests; a per-chunk audit in `openie_audit.jsonl`), and the fact filter's reply
goes through the same parser. The adapter's rescue request field is renamed from vLLM 0.6.6's
`guided_json` to vLLM 0.23's `structured_outputs` (the server here ignores the old name).

Other settings are the package defaults, decided with the user because the paper's Appendix E.6
was not available; the report states that they were not checked against it. HippoRAG caches
every LLM reply and its OpenIE results, so an interrupted book resumes without new calls.
