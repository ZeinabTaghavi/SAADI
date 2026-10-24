# Changelog

All notable changes to Saadi are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Documentation pages for library usage and customization, a page for the
  paper's experiments, and three notebooks: the quickstart on an API key, the
  same on Colab's free GPU with an open model through Ollama, and a rebuild of
  the paper's tables from the saved results.
- `examples/release_check.ipynb`, a pre-release check for a GPU runtime: it
  installs a candidate build, prints the resolved library versions, verifies
  the files bundled in the wheel, loads a Qwen model through the vLLM backend,
  checks the prompt contract, and runs indexing, retrieval, answering,
  appending, and reloading with assertions. Referenced from the release guide.
- `examples/model_family_check.ipynb`, a matrix over light instruct models
  from several families, each loaded through vLLM in its own subprocess, and
  over several retrievers on one small document: per model it records loading,
  chat template, prompt contract, entities, dropped proposals, and extraction
  versus fold call counts; per retriever a hit table.
- A README that starts with the library (install, a sixty-second example with
  real output, what the state contains) and moves the experiment instructions
  to the docs; the logo, an overview figure, issue templates, and a
  contributing guide.

## [0.1.0] - 2026-10-16

First release of Saadi as a library. The reproduction code of the paper
(`scripts/`, `configs/`, the saved results, and the files listed in
`docs/preserved_sources.json`) ships unchanged next to it.

### Added

- `saadi.Saadi`, one object per document with `index`, `append`, `retrieve`,
  `answer`, `summarize`, `memory`, `history`, `save`, and `load`. The state is
  a flat directory (`saadi.json`, `document.txt`, `chunks.jsonl`,
  `memory.json`, `history/`, `summary.txt`, `queries/`, `cache/`) that can be
  inspected, copied, moved, and loaded again; retrieval needs no model.
  `append` reuses every unchanged chunk from the construction cache.
- Command line: `saadi index`, `saadi query`, and `saadi demo`, which runs on
  a bundled synthetic report; `saadi check-prompts` validates a prompt profile;
  `saadi --version`.
- Language-model backends behind one `provider:model` spec: OpenAI and every
  OpenAI-compatible server (Ollama, LM Studio, Groq, vLLM serve), Anthropic,
  local vLLM, a mock, and any Python callable.
- Retrievers behind one name: BM25, the dense retrievers of the paper, OpenAI
  and Ollama embeddings, plus a registry for scorer classes, embedding
  functions, and reranker objects.
- Prompt profiles: a fixed output contract plus swappable guidance; packaged
  `default` (the paper's prompt, byte for byte) and `conversation` profiles,
  and custom profile directories.
- Input adapters for plain text, Markdown, and role-labelled conversation CSVs.
- Packaging: extras `openai`, `anthropic`, `dense`, `vllm`, `repro`, and
  `dev`; releases published to PyPI through GitHub Actions trusted publishing;
  `CITATION.cff` with the paper's preprint entry.

[Unreleased]: https://github.com/ZeinabTaghavi/SAADI/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ZeinabTaghavi/SAADI/releases/tag/v0.1.0
