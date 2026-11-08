# B2 Contextual Retrieval (whole document)

For every chunk, one LLM call reads the book and the chunk and writes a short context that
places the chunk in the book, with the prompt of Anthropic's Contextual Retrieval post. The
index is "context + chunk", searched with the same BM25 as B0; original chunk ids are
returned. This is the one baseline that can bridge the probe's names: if the model saw the
reveal ("Maysunah's family name was Sinchipuma") while writing the gold chunk's context, it
can put "Maysunah" into that context.

The model sees as much of the book as the served window allows. The vLLM server for this run
uses Qwen3-30B-A3B-Instruct-2507's native window of 262,144 tokens (decided with the user;
SAADI's builds use 32,768), so most books fit whole and the longest ones get their longest
prefix. `build.py` records for every chunk how much of the book was visible; `retrieve.py`
marks per persona whether the reveal was visible and whether the written context contains
form 2. The book is the same prefix for every chunk of a book, so the server's prefix cache
computes it once per book. Resumable per chunk.
