# Library usage

Saadi keeps one state per document. You build it once with a language model, then retrieve and answer as often as you like; retrieval alone needs no model. This page covers the `Saadi` object, the inputs it accepts, the state directory it writes, and the command line. For plugging in your own model, retriever, or prompts see [customizing Saadi](customizing.md).

## Install

```bash
pip install 'saadi[openai]'      # OpenAI and OpenAI-compatible servers
pip install 'saadi[anthropic]'   # Anthropic
pip install 'saadi[vllm]'        # local vLLM, as in the paper
```

Python 3.11 or 3.12. Set the provider's key in the environment (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`); keys are never written into a state.

## The Saadi object

```python
from saadi import Saadi

doc = Saadi(
    llm="openai:gpt-4.1-mini",  # provider:model, a config mapping, a backend object, or a function
    retriever="bm25",  # bm25, a dense retriever, openai_embedding, ollama_embedding, or your own
    prompts="default",  # default, conversation, or a profile directory
    state_dir="saadi_state/report",
    chunk_size=500,  # tokens per chunk; the paper's default
    branching_factor=5,  # chunks per fold summary
    max_entities=5,  # entities recalled per state update
)
```

| Argument | Default | Meaning |
|---|---|---|
| `llm` | `None` | Needed for `index`, `append`, `answer`, `summarize`. A loaded state retrieves without one. |
| `retriever` | `"bm25"` | Scores the question against chunk text, entity memory, and summaries; the three scores are fused. |
| `prompts` | `"default"` | The guidance the model gets for entity extraction and fold summaries. |
| `state_dir` | `saadi_state/<document_id>` | Where the state lives. One directory holds one document. |
| `chunk_size`, `chunk_overlap` | 500, 0 | Chunking in tokens. |
| `branching_factor` | 5 | Items folded into one summary at each level. |
| `max_entities`, `max_aliases`, `max_desc` | 5, 2, 3 | Entity recall per update and caps on aliases and descriptions. |
| `retrieve_k`, `answer_top_k` | 10, 5 | Chunks returned by `retrieve` and read by `answer`. |
| `answer_temperature`, `answer_max_tokens` | 0.6, 2048 | Sampling for the answer. |

### index

```python
report = doc.index("report.txt")  # a path
report = doc.index(text, document_id="memo")  # inline text
report = doc.index("session.csv", kind="conversation", allowed_roles=("therapist", "patient"))
```

`index` accepts a `.txt` or `.md` file, inline text, or a conversation `.csv` with `role` and `text` columns (turns are ordered by `turn_id` when present). Keyword options go to the CSV adapter: `allowed_roles`, `role_aliases` (for example `{"clinician": "therapist"}`), `include_timestamps`, and the column names. They are remembered for `append`.

The returned `IndexReport` tells you what the pass cost:

```python
IndexReport(
    document_id="report",
    state_dir=...,
    chunks=10,
    new_chunks=10,
    entities=13,
    history_levels=(0, 1),
    model_calls=12,
    prompts_sent=12,
    rejected_entities=3,
)
```

`rejected_entities` counts proposals the grounding check dropped because their anchor text was not found verbatim in the chunk; it is a quality signal for the model, not an error. Re-indexing the same document into the same state reuses every unchanged chunk. Nothing is written unless the model run succeeds, and a state directory refuses a different document id.

### append

```python
update = doc.append("More pages ...")
update = doc.append("more_pages.txt")
```

The text is appended to the document and the state is updated. Chunks whose text is unchanged come from the construction cache, so only the tail costs model calls, plus the fold summaries that include a changed chunk. `update.new_chunks` counts chunks whose text is new to the state. If you append with a different model than the one that built the state, Saadi warns and reads the whole document again, because the cache is kept per model.

### retrieve and answer

```python
hits = doc.retrieve("Who keeps the Corvin station?", k=5)
hits[0].rank, hits[0].chunk_index, hits[0].score, hits[0].text

reply = doc.answer("Who keeps the Corvin station?", k=5)
reply.text  # the generated answer
reply.chunks  # the chunks it read, same shape as retrieve()
reply.context  # the folded context given to the model
reply.status  # "ok", or why generation was skipped
reply.directory  # the query folder with every intermediate file
```

`retrieve` is retrieval only and works on a loaded state without a model. `answer` retrieves, builds the folded context (the top chunks with the summaries and entities around them), and makes one generation call. Each call writes a numbered folder under `queries/`.

### summarize

```python
summary = doc.summarize()
```

One whole-document summary with the profile's `document_summary.txt` prompt; the `conversation` profile has one, `default` does not. The summary is written to `summary.txt` and cleared when the document changes. The prompt receives every chunk at once, so this suits documents of up to a few thousand words.

### memory and history

```python
for entity in doc.memory:
    entity.name, entity.aliases, entity.descriptions, entity.first_seen_chunk

doc.history[0]  # the chunks as the model read them, with entity anchors
doc.history[1]  # fold summaries, each covering five chunks
item = doc.history[1][0]
item.start_step, item.end_step, item.text
```

`memory` is a readable view: pronoun and role-label aliases are removed from it. Retrieval scores the raw memory kept in the cache.

### save and load

```python
copy = doc.save("backups/report")  # a self-contained copy
again = Saadi.load("saadi_state/report")  # retrieval only
again = Saadi.load("saadi_state/report", llm="openai:gpt-4.1-mini")  # answer and append too
```

`load` takes the construction settings, the retriever, and the prompt profile from the manifest. A state copied or moved by other means is relocated on load. If the prompt profile changed since the state was built, `load` warns; retrieval keeps the stored construction and the next `append` rebuilds with the new prompts. If a custom profile directory is gone, the guidance recorded in the manifest stands in for it.

## The state directory

```text
saadi_state/report/
├── saadi.json        manifest: document id, settings, the model and prompts that built the state, counts, layout
├── document.txt      the indexed text; append() extends it
├── chunks.jsonl      one object per chunk: chunk_id, chunk_index, text, token offsets
├── memory.json       the entity memory (readable view)
├── history/          L_0.jsonl, L_0.txt, L_1.jsonl, L_1.txt, …
├── summary.txt       after summarize()
├── queries/          001_who-keeps-the-corvin-station/ … retrieval, context, generation
└── cache/            the construction cache and the retrieval index, one folder per prompt profile
```

Everything except `cache/` is meant to be read; `cache/` is required for `retrieve` and `answer`, so keep it, but do not edit it. `saadi.json` records the model, the retriever, and the prompts that built the state, never credentials.

## Command line

```bash
saadi demo  [--llm SPEC] [--state DIR] [--query TEXT]
saadi index SOURCE --llm SPEC [--state DIR] [--retriever NAME] [--prompts PROFILE] [--append]
                   [--document-id ID] [--kind text|conversation] [--roles ROLE,ROLE] [--role-alias OLD=NEW]
                   [--timestamps] [--chunk-size N] [--branching-factor N] [--max-entities N] [--summary] [--json]
saadi query QUESTION --state DIR [--llm SPEC] [--k N] [--retriever NAME] [--memory] [--json]
saadi check-prompts PROFILE [--llm SPEC]
saadi --version
```

`saadi index` on a directory creates one state per file under `--state`. With `--append`, the stored profile, retriever, and construction settings apply, and flags that would change them are refused. Commands print one `error:` line and exit 1 on failure; `--verbose` shows the pipeline log and the traceback. The demo, the index, and the query commands print the answer, the top chunks with their scores, and the entity records.

## Cost and speed

A document is processed one chunk at a time by design, since each state update depends on the previous one, so indexing takes the per-call latency times the number of chunks (plus one call per fold). Answering costs retrieval over the saved state plus one generation call. Hosted backends send the prompts of a batch concurrently and retry rate limits with backoff. With Anthropic models, `effort: low` in a config mapping is the economical setting for the many short extraction calls. With Ollama, raise the server's context length if outputs are cut off.
