<p align="center">
  <img src="https://raw.githubusercontent.com/ZeinabTaghavi/SAADI/main/docs/assets/logo.png" alt="Saadi" width="320">
</p>

<p align="center">
  <a href="https://github.com/ZeinabTaghavi/SAADI/actions/workflows/ci.yml"><img src="https://github.com/ZeinabTaghavi/SAADI/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/saadi/"><img src="https://img.shields.io/pypi/v/saadi.svg" alt="PyPI"></a>
  <a href="https://colab.research.google.com/github/ZeinabTaghavi/SAADI/blob/main/examples/quickstart.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
</p>

Index a long document once, then answer questions about it many times. Saadi reads the document with a language model one chunk at a time, keeps an entity memory and a summary hierarchy, and uses them to find the right passages for each question. It is the code of the paper *Understand Once, Retrieve Many Times*, packaged as a library.

## Install

```bash
pip install 'saadi[openai]'
```

Other providers (Anthropic, Ollama, local vLLM) and the paper's dense retrievers are listed under [Models, retrievers, prompts](#models-retrievers-prompts).

## Sixty seconds

```python
from saadi import Saadi

doc = Saadi(llm="openai:gpt-4.1-mini", state_dir="saadi_state/report")
doc.index("report.txt")  # any .txt, .md, or conversation .csv
reply = doc.answer("Who keeps the Corvin station?")
print(reply.text)
```

No document at hand? `saadi demo --llm openai:gpt-4.1-mini` runs the same thing on a bundled synthetic report about a fictional research team. This is what it printed, with `OPENAI_API_KEY` set:

```text
Indexed aurora_report: 10 chunks (10 new), 13 entities, history levels L0, L1
Model calls: 12 (12 prompts)

Question: Who keeps the Corvin station?
Answer: Tomas Egger keeps the Corvin station.

Top chunks:
1. chunk 4 (score 2.103): The Corvin station was built in May 2025 by Leon and a new technician, Tomas Egger, who grew up in the valley and knows which paths are safe in early spring. Tomas keeps the Corvin station, and Leon keeps the Alder station, but each covers…
2. chunk 7 (score 0.220): Tomas wrote the field guide that explains how to reach each station and what to check first when a reading looks wrong. Mira insists that every correction be reversible, so the archive keeps the raw readings next to the corrected ones. Fun…
```

Ten chunks cost ten model calls plus two for the summary folds. Every question after that costs one.

## What you get back

The index-time pass leaves behind more than a vector store would. All of it is readable, and all of it is used at query time.

**An entity memory**, one record per entity with aliases and accumulated descriptions:

```python
>>> [entity for entity in doc.memory if entity.name == "Tomas Egger"][0]
MemoryEntity(name='Tomas Egger', aliases=('Tomas',), descriptions=('new technician who maintains the Corvin station', 'grew up in the valley and knows safe paths in early spring', 'wrote the field guide explaining station access and troubleshooting', ...), first_seen_chunk=4)
```

On the command line the same memory prints as one line per entity, as in `saadi demo`:

```text
Entity memory (13 entities, showing 8):
- Mira Chen [Mira]: limnologist who founded the Aurora research team; appointed Leon Park to maintain the station; installed a second sensor at Lake Alder in June 2024; wanted a l…
- Leon Park [Leon]: electrical engineer appointed to maintain the station; records water temperature, checks solar panel, cleans turbidity sensor; installed radio link to Aurora l…
- Lake Corvin: second lake introduced in the second field season; lies two valleys to the east of Lake Alder; at a slightly higher altitude; morning temperature difference wi…
```

**A summary hierarchy**, level 0 holding the chunks as the model read them and each level above folding five items of the one below:

```python
>>> doc.history[1][0].text
'The Aurora research team, founded in 2024 by limnologist Mira Chen, established monitoring stations at Lake Alder and Lake Corvin …'
```

**A state that grows without being rebuilt.** Appending text reuses every unchanged chunk from the cache, so only the new tail costs model calls:

```python
>>> update = doc.append(open("more_pages.txt").read())
>>> update.chunks, update.new_chunks, update.model_calls
(11, 2, 3)
```

**Retrieval that needs no model.** The state is a plain directory, and a loaded state answers `retrieve` from the chunk text, the memory, and the summaries at once:

```python
>>> again = Saadi.load("saadi_state/report")
>>> again.retrieve("Who keeps the Corvin station?", k=3)[0].chunk_index
4
```

## Command line

The three commands mirror the Python calls. A state directory is created by `index` and read by `query`.

```bash
saadi demo  --llm openai:gpt-4.1-mini
saadi index report.txt --llm openai:gpt-4.1-mini --state saadi_state/report
saadi query "Who keeps the Corvin station?" --state saadi_state/report --llm openai:gpt-4.1-mini
saadi index more_pages.txt --append --state saadi_state/report --llm openai:gpt-4.1-mini
```

`saadi query` without `--llm` prints the retrieved chunks only, `--memory` adds the entity records, and `--json` gives everything as JSON. Conversation transcripts, a `.csv` with `role` and `text` columns, index the same way with `--prompts conversation`.

## Models, retrievers, prompts

Three choices, each one argument. Details and the Python forms of each are in [customizing Saadi](docs/customizing.md).

| Language model (`llm=` or `--llm`) | Needs |
|---|---|
| `openai:gpt-4.1-mini` | `pip install 'saadi[openai]'` and `OPENAI_API_KEY`. Any OpenAI-compatible server (LM Studio, llama.cpp, Groq, Together, `vllm serve`) works through `OPENAI_BASE_URL`. |
| `ollama:qwen3:8b` | `pip install 'saadi[openai]'` and a local Ollama server. No key. |
| `anthropic:claude-opus-5-5` | `pip install 'saadi[anthropic]'` and `ANTHROPIC_API_KEY`. |
| `vllm:Qwen/Qwen3.5-27B` | `pip install 'saadi[vllm]'` and local GPUs. The backend that produced the paper's results. |
| `mock` | Nothing. Placeholder outputs for checking a pipeline. |

| Retriever (`retriever=` or `--retriever`) | Needs |
|---|---|
| `bm25` (default) | Nothing. |
| `contriever`, `bge_m3`, `e5`, `jina`, `qwen` | `pip install 'saadi[dense]'`. The dense retrievers of the paper. |
| `openai_embedding`, `ollama_embedding` | The matching server; uses `text-embedding-3-small` or `nomic-embed-text`. |
| your own | A class with `score(query)`, an embedding function, or a reranker object, registered under a name. |

| Prompt profile (`prompts=` or `--prompts`) | Use |
|---|---|
| `default` | The paper's prompts, byte for byte. |
| `conversation` | Role-labelled transcripts; a case-level memory instead of a list of names. |
| a directory | Your own guidance for what counts as an entity. Validate it with `saadi check-prompts my_profile/ --llm …`. |

## Try it in Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ZeinabTaghavi/SAADI/blob/main/examples/quickstart.ipynb) runs the sixty-second example end to end on an API key. [A second notebook](examples/colab_local_model.ipynb) does the same on Colab's free GPU with an open 8B model through Ollama, and [a third](examples/paper_results.ipynb) rebuilds every table of the paper from the saved results on a plain CPU runtime. [A fourth](examples/release_check.ipynb) is the pre-release check: it installs a candidate build on a GPU runtime, prints the resolved library versions, loads a Qwen model through vLLM, and runs the whole path with assertions. [A fifth](examples/model_family_check.ipynb) runs light models from several families (Qwen, Llama, Gemma, SmolLM, Phi) and several retrievers over one small document and tabulates entities, call counts, and retrieval hits per model and retriever.

## The paper

<p align="center">
  <img src="https://raw.githubusercontent.com/ZeinabTaghavi/SAADI/main/docs/assets/overview.svg" alt="Overview: the document is read once into an entity memory and a summary hierarchy; each question is answered by fusing three scores over the saved state" width="900">
</p>

A long document rarely keeps the evidence for a question in one place. Saadi (SAADI in the paper: Stateful Accumulation of Document Information) reads the document once in order, and at each chunk a language model updates a small entity memory and emits a summary that folds into a hierarchy. At query time the question is scored against the original chunk text, the memory, and the summaries, and the three scores are fused, so a chunk is found through the entities and the summaries that explain it, not only through its own words. The document state is built once and serves every later question.

*Understand Once, Retrieve Many Times: SAADI for Single-Pass Retrieval of Distributed Evidence in Long Documents.* Zeinab Sadat Taghavi, Lars Klein, Mary-Anne Hartley, and Andreas Marfurt. Preprint, 2026.

```bibtex
@misc{taghavi2026saadi,
  title  = {Understand Once, Retrieve Many Times: {SAADI} for Single-Pass Retrieval of Distributed Evidence in Long Documents},
  author = {Taghavi, Zeinab Sadat and Klein, Lars and Hartley, Mary-Anne and Marfurt, Andreas},
  year   = {2026},
  note   = {Preprint}
}
```

GitHub's "Cite this repository" button uses [CITATION.cff](CITATION.cff), which carries the same entry.

## Reproducing the paper

Every table of the paper rebuilds from the saved, hash-verified study values in this repository, without datasets, GPUs, or a model: `python scripts/build_tables.py`, or the [results notebook](examples/paper_results.ipynb). The [paper consistency check](results/paper_consistency/README.md) matches 1,307 of 1,310 cells. Rerunning the study itself needs the datasets and GPUs; [running the experiments](docs/experiments.md) covers the study runner, the configuration suites, and the archived outputs, and the [paper-to-code guide](docs/paper_to_code.md) maps every equation to the code. The code that produced the saved results is kept byte for byte, listed in [docs/preserved_sources.json](docs/preserved_sources.json) and checked by a test.

## Documentation

- [Library usage](docs/library_usage.md): the `Saadi` object, inputs, the state directory, and the command line.
- [Customizing Saadi](docs/customizing.md): your own language model, retriever, prompt profile, or input format.
- [Running the experiments](docs/experiments.md), [datasets and artifacts](docs/data_and_artifacts.md), [saved results](docs/results.md), [releasing](docs/release.md).
- [Changelog](CHANGELOG.md), [contributing](CONTRIBUTING.md), and [discussions](https://github.com/ZeinabTaghavi/SAADI/discussions) for questions.

Licensed under [MIT](LICENSE). Dataset content remains subject to its original terms.
