# Customizing Saadi

Three things are pluggable, each through one argument of `Saadi` or one flag of the command line: the language model, the retriever, and the prompt profile. A fourth, the input format, is a small adapter.

## Language models

Every model call goes through one backend interface, so construction and answer generation work with any provider. Pass one of these as `llm=` (or `--llm`):

**A `provider:model` string.**

| Spec | Needs |
|---|---|
| `openai:gpt-4.1-mini` | `pip install 'saadi[openai]'` and `OPENAI_API_KEY`. Any OpenAI-compatible server works through `OPENAI_BASE_URL`: LM Studio, llama.cpp, Groq, Together, `vllm serve`. |
| `ollama:qwen3:8b` | `pip install 'saadi[openai]'` and a local Ollama server. No key. |
| `anthropic:claude-opus-5-5` | `pip install 'saadi[anthropic]'` and `ANTHROPIC_API_KEY`, or a profile from `ant auth login`. |
| `vllm:Qwen/Qwen3.5-27B` | `pip install 'saadi[vllm]'` and local GPUs. The backend that produced the paper's results. |
| `mock` | Nothing. Placeholder outputs for checking a pipeline. |

**A config mapping**, for anything the string cannot say:

```python
doc = Saadi(
    llm={
        "backend": "openai",
        "model_id": "llama-3.3-70b-versatile",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",  # the variable that holds the key; the key itself is never stored
        "max_workers": 4,
        "timeout": 60,
    }
)
```

Hosted backends send the prompts of a batch concurrently and retry rate limits with backoff. The same mapping is the `model` section of an experiment configuration.

**A backend object or a function.** `saadi.llm.resolve_backend` accepts a backend object, or a function `(system, user, **options) -> str`, so any other client can be plugged in:

```python
from saadi import Saadi


def my_model(system: str, user: str, **options) -> str:
    return client.generate(system=system, prompt=user)


doc = Saadi(llm=my_model)
```

The function receives the system prompt and the user prompt of each call and returns the model's text. It is the simplest way to use a provider that has no backend yet.

What the model must do: for each chunk it returns a JSON object with the active entities and the text spans that anchor them, ending with `[END_JSON]`; for each fold it returns a summary ending with `[END_SUMMARY]`. The fixed part of the prompts explains this contract; `saadi check-prompts default --llm your:model` sends one of each and reports whether the answers follow it. An anchor must appear verbatim in the chunk; a proposal whose anchor does not is dropped and counted in `IndexReport.rejected_entities`. The context words around the anchor only locate it, so small differences in spacing are accepted when the anchor occurs once. The check fails when the model keeps no entities from its sample text.

## Retrievers

The retriever scores the question against three pools, the original chunks, the entity memory, and the fold summaries, and Saadi fuses the three scores per chunk. Scores are z-normalized per pool, so any scorer whose higher value means more relevant works.

| Name | Needs |
|---|---|
| `bm25` (default) | Nothing. |
| `contriever`, `bge_m3`, `e5`, `jina`, `qwen` | `pip install 'saadi[dense]'`. The dense retrievers of the paper. |
| `openai_embedding` | `pip install 'saadi[openai]'` and `OPENAI_API_KEY`. Uses `text-embedding-3-small`; `SAADI_OPENAI_EMBEDDING_MODEL` changes it and `OPENAI_BASE_URL` points it at another server. |
| `ollama_embedding` | A local Ollama server with an embedding model such as `nomic-embed-text`. |

Your own retriever is a class with a `score(query)` method that returns one number per text in `self.corpus`, registered under a name:

```python
from saadi.retrievers import BaseRetriever, register_retriever


@register_retriever("keyword_overlap")
class KeywordOverlap(BaseRetriever):
    def score(self, query):
        words = set(query.lower().split())
        return [len(words & set(text.lower().split())) for text in self.corpus]


doc = Saadi(llm="openai:gpt-4.1-mini", retriever="keyword_overlap")
```

Instances work too. `resolve_retriever` accepts a name, a class, a reranker object with `score(query, texts)`, an embedding model object with `encode(texts)` such as a sentence-transformers model, or a plain function of either shape, and registers it under a name:

```python
from sentence_transformers import SentenceTransformer

doc = Saadi(llm="openai:gpt-4.1-mini", retriever=SentenceTransformer("all-MiniLM-L6-v2"))
```

`saadi.register_retriever` and `saadi.resolve_retriever` are the same entry points at the package top level. The retriever's name is recorded in the state, so `Saadi.load` uses it again; a retriever registered in code must be registered before the state is loaded.

## Prompt profiles

The index-time model calls use two prompts: entity extraction for each chunk and a fold summary for each group of chunks. A prompt profile is a folder with `entity_extraction.txt`, `hierarchy_summary.txt`, and optionally `document_summary.txt` (which enables `summarize()`) and `sample.txt` (which `check-prompts` uses as its sample chunk). The files hold domain guidance only: what counts as an entity, how to name and describe it, what a summary should keep. The library wraps that guidance in a fixed contract that states the inputs and the output format the parsers expect, so a profile can change the extraction policy but cannot break parsing.

| Profile | Use |
|---|---|
| `default` | The paper's prompts, byte for byte. |
| `conversation` | Role-labelled transcripts; builds a compact case-level memory (the person, the presenting problem, important relationships, a plan) instead of a list of every name. `therapy` is an alias. |
| a directory | Your own guidance. Start by copying the packaged `default` files and editing the rules. |

```python
doc = Saadi(llm="openai:gpt-4.1-mini", prompts="my_profile/")
```

Validate a profile before a long run:

```bash
saadi check-prompts my_profile/ --llm openai:gpt-4.1-mini
```

It sends one extraction prompt and one fold prompt built from the profile and reports whether the answers follow the contract, which entities came back, and why an insertion was dropped. Each state records the profile's name, a hash of the composed prompts, and the guidance texts themselves, and every profile gets its own construction cache, so changing the prompts never mixes two constructions.

The packaged profiles live in `src/saadi/prompts/profiles/`; a profile for legal documents, meeting notes, or scientific articles is a copy of `default` with its own `Focus` and `Selection` rules.

## Input formats

`Saadi.index` and `append` take inline text, `.txt` and `.md` files, and conversation `.csv` files. Anything else is a small adapter that returns a `PreparedDocument`:

```python
from saadi import Saadi
from saadi.inputs import PreparedDocument


def from_pdf(path: str) -> PreparedDocument:
    text = extract_text(path)  # your extractor
    return PreparedDocument(document_id="contract", text=text, source_path=path, input_format="pdf")


doc = Saadi(llm="openai:gpt-4.1-mini")
doc.index(from_pdf("contract.pdf"))
```

The prepared document is indexed as is; the manifest records its id, format, and source path.
