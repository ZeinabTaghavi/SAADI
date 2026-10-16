# Changelog

All notable changes to Saadi are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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
