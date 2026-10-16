# Saadi

Saadi (SAADI in the paper, Stateful Accumulation of Document Information) builds a query-agnostic document state at index time, an entity memory and a hierarchical history, and uses it to improve single-pass top-k retrieval over the original chunks of long documents. This repository contains the dataset preparation, construction, retrieval, evaluation, and ablation code, together with compact saved results and a single-document example.

## Paper

*Understand Once, Retrieve Many Times: SAADI for Single-Pass Retrieval of Distributed Evidence in Long Documents.* Zeinab Sadat Taghavi, Lars Klein, Mary-Anne Hartley, and Andreas Marfurt. Preprint, 2026.

```bibtex
@misc{taghavi2026saadi,
  title  = {Understand Once, Retrieve Many Times: {SAADI} for Single-Pass Retrieval of Distributed Evidence in Long Documents},
  author = {Taghavi, Zeinab Sadat and Klein, Lars and Hartley, Mary-Anne and Marfurt, Andreas},
  year   = {2026},
  note   = {Preprint}
}
```

The project, Python package, and command are **Saadi**, `saadi`, and `saadi`, respectively. Historical artifact paths such as `outputs/sadi/`, method IDs such as `sadi`/`mbr`, and `SADI_*` environment variables are retained for compatibility with the saved experiments. Public tables and evaluation reports use **HR@5** and **HR@10**.

See the [paper-to-code guide](docs/paper_to_code.md) for equations, notation, and table commands, and [datasets and artifacts](docs/data_and_artifacts.md) for data preparation, cache restoration, and the exact publication contents.

## Install

Use Python 3.11 or 3.12 on Linux or macOS (the study runner uses POSIX file locks). Run these commands from this repository's root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python scripts/smoke_test.py
python -m pytest -q
```

The smoke test constructs a synthetic document index, performs Saadi BM25 retrieval, and checks HR on CPU using a mock model. Its scores are software checks, not experimental results.

For dataset downloads, plots, dense retrieval, and model inference:

```bash
python -m pip install -e '.[datasets,figures,llm]'
```

Model inference uses vLLM on a supported GPU environment. The construction configuration defaults to four GPUs; set `--tensor-parallel-size` for your hardware. Exact GPU package versions and model revisions should be recorded with each new experiment. The CPU validation environment is recorded in `docs/validation_environment.json`.

## View the saved results

```bash
python scripts/build_tables.py
# Optional retrieval-depth curve (PNG, PDF, SVG):
python scripts/build_tables.py --figures
# Generate just the final main comparison:
python results/tables/main/retrieval_comparison/generate.py
# Optional local mirror:
python scripts/build_tables.py --output-dir outputs/tables
```

This rebuilds `results/tables/` without downloading data or running a language model. A fresh checkout uses the compact `outputs/sadi/reference/` and `outputs/sadi/comparison_inputs/` bundles. When the optional full output archive has been restored and catalogued in `outputs/output_sources.json` (see [datasets and artifacts](docs/data_and_artifacts.md#large-output-archives)), the generators read those archived files instead. Saved CSVs under `outputs/other_methods/<method>/` supply the other methods without rerunning them. Both routes verify input hashes and check the same saved study values. Use `--input-source outputs` or `--input-source reference` to require a particular route. Missing or changed inputs in an existing output catalog cause an error.

Table directories use `main/`, `comparisons/`, `ablations/`, `robustness/`, and `construction/` categories, with a separate folder for each study. Each folder contains its `generate.py`, tables, source paths and hashes, and verification results. Tables are available as CSV, Markdown, and LaTeX; presentation HR values are percentages. `combined/all_metrics.csv` collects all available HR configurations.

[Final main comparison](results/tables/main/retrieval_comparison/table.md), [Saadi-only results](results/tables/main/sadi_retrieval/table.md), and the [table index](results/tables/README.md) can be browsed directly. [Saved results and provenance](docs/results.md) lists the evidence level of every table and the provenance notes.

The [paper consistency check](results/paper_consistency/README.md) compares the generated tables with the paper: 1,307 of 1,310 checked cells match, including all 184 values of Table 1. The three differences are second-decimal rounding of construction statistics.

## Run Saadi

```bash
# Inspect the commands and resolved scope without writing files:
python scripts/run_study.py --suite main --datasets loogle --retrievers bm25 --dry-run

# Construct the index, retrieve, and evaluate:
python scripts/run_study.py --suite main --datasets loogle --retrievers bm25
```

The full `main` suite covers LooGLE, QASPER, NovelHopQA, and MuSiQue with BM25, Contriever, BGE-M3, Jina, and Qwen retrieval. Comparison methods are not run; their saved results are in `outputs/other_methods/`. Dataset preparation and evidence labeling are the same code that produced the saved results.

New runs use BM25 memory selection and show the model the original text of recent chunks as history. The saved results were produced with word-overlap selection and the anchored chunk copies; see the [provenance notes](docs/results.md#provenance-notes). To construct with the saved settings:

```bash
python scripts/run_study.py --suite main --profile published --datasets loogle --retrievers bm25
```

This profile restores the recorded selector, history text, model configuration, temperature, and sampling settings. Model revisions and runtime versions of the original runs are not pinned, so regenerated model text can differ. The table rebuild checks the saved results exactly.

The default directories are `data/`, `.cache/`, and `outputs/`. Override them with `SADI_DATA_DIR`, `SADI_CACHE_DIR`, and `SADI_OUTPUT_DIR`. New run data, caches, and large outputs are excluded from Git; the small saved table-input bundles are included. NovelHopQA requires the book files described in [the dataset guide](docs/data_and_artifacts.md#novelhopqa-book-corpus).

## Configuration and ablations

Dataset overrides in `configs/experiments/` extend `configs/methods/sadi.yaml`, which extends the construction settings in `configs/models/construction.yaml`. Suites in `configs/suites/` define experimental variations. Configuration keys are validated before loading data or models.

```bash
python scripts/run_study.py --suite hierarchy --datasets loogle --retrievers bm25
python scripts/run_study.py --suite memory_conditioning --datasets loogle --retrievers bm25
python scripts/run_study.py --suite construction_models --datasets loogle --retrievers bm25
```

Use `--stage prepare`, `ingest`, `retrieve`, or `generate` to run a particular stage. `--max-documents 2` creates a separate small-run identity. See [reproduction instructions](docs/reproduction.md) for other suites, configuration fields, stage reuse, and generation.

## Single-document example

```bash
python single_document_demo/run.py text --mock --query 'Who leads the Aurora team?'
python single_document_demo/run.py text --llm openai:gpt-4.1-mini --query 'Who leads the Aurora team?'
```

The [single-document example](single_document_demo/README.md) accepts plain text or a role-labeled conversation CSV. Both included inputs are synthetic. `--mock` checks the workflow on CPU without a model. `--llm provider:model` picks the language model; the providers are listed below.

## Language model backends

Every model call goes through one backend interface, so construction and answer generation work with any of these providers. The vLLM backend produced the paper's results.

| `--llm` or `model.backend` | Example | Needs |
|---|---|---|
| `openai` | `openai:gpt-4.1-mini` | `pip install 'saadi[openai]'` and `OPENAI_API_KEY`. Any OpenAI-compatible server works through `OPENAI_BASE_URL` or `model.base_url`: LM Studio, llama.cpp, Groq, Together, `vllm serve`. |
| `ollama` | `ollama:qwen3:8b` | `pip install 'saadi[openai]'` and a local Ollama server. No key. |
| `anthropic` | `anthropic:claude-opus-5-5` | `pip install 'saadi[anthropic]'` and `ANTHROPIC_API_KEY`, or a profile from `ant auth login`. |
| `vllm` | `vllm:Qwen/Qwen3.5-27B` | `pip install 'saadi[llm]'` and local GPUs. |
| `mock` | `mock` | Nothing. Placeholder outputs for checking the workflow. |

A single document is processed one chunk at a time by design, since each state update depends on the previous one, so the time for a document is the per-call latency times the number of chunks; concurrency speeds up the summary folds and multi-document runs. With Anthropic models, `effort: low` in the `model` section is the economical setting for the many short extraction calls. With Ollama, raise the server's context length if outputs are cut off.

In experiment and demo configurations the same choice is the `model` section: `backend`, `model_id`, and for hosted APIs the optional `base_url`, `api_key_env`, `max_workers`, `max_retries`, and `timeout`. Hosted backends send the prompts of a batch concurrently and retry rate limits with backoff. From Python, `saadi.llm.resolve_backend` accepts the same `provider:model` string, a config mapping, a backend object, or a function `(system, user, **options) -> str`, so any other client can be plugged in.

## Retrievers

The base retriever scores the query against the original chunks, the entity memory, and the summaries, and Saadi fuses the three scores. Select it with `--retriever` in the example or with `retriever` in a configuration. Scores are z-normalized per channel, so any scorer whose higher value means more relevant works.

| Name | Needs |
|---|---|
| `bm25` (default) | Nothing. |
| `contriever`, `bge_m3`, `e5`, `jina`, `qwen` | `pip install 'saadi[dense]'`. The dense retrievers of the paper. |
| `openai_embedding` | `pip install 'saadi[openai]'` and `OPENAI_API_KEY`. Uses `text-embedding-3-small`; set `SAADI_OPENAI_EMBEDDING_MODEL` to change it, and `OPENAI_BASE_URL` for any other OpenAI-compatible embeddings endpoint. |
| `ollama_embedding` | A local Ollama server with an embedding model such as `nomic-embed-text`. No key. |

Your own retriever is a class with a `score(query)` method that returns one number per text, registered under a name:

```python
from saadi.retrievers import BaseRetriever, register_retriever

@register_retriever("keyword_overlap")
class KeywordOverlap(BaseRetriever):
    def score(self, query):
        words = set(query.lower().split())
        return [len(words & set(text.lower().split())) for text in self.corpus]
```

Instances work too. `resolve_retriever` accepts a name, a class, a reranker object with `score(query, texts)`, an embedding model object with `encode(texts)` such as a sentence-transformers model, or a plain function of either shape, and registers it under a name. `saadi.register_retriever` and `saadi.resolve_retriever` are the same entry points at the package top level.

## Repository contents

| Directory | Purpose |
|---|---|
| `src/saadi/` | Construction, memory selection, retrieval, generation, and evaluation |
| `configs/` | Shared settings, dataset overrides, and study suites |
| `scripts/` | Study runner, grouped-data preparation, evaluation, and result export tools |
| `outputs/sadi/reference/` | Checksummed Saadi rankings and saved summary values |
| `outputs/sadi/comparison_inputs/` | Additional Saadi expansion rankings and comparison-specific cost/exposure inputs |
| `outputs/other_methods/` | Structured saved retrieval/resource CSVs and provenance for comparison methods |
| `results/tables/` | Final comparisons and Saadi study tables, each with its generator and provenance |
| `results/paper_consistency/` | Cell-by-cell comparison of the generated tables with the paper |
| `docs/` | Paper-to-code guide, dataset and artifact guide, reproduction guide, result coverage |
| `single_document_demo/` | Self-contained examples using the same Saadi implementation |
| `tests/` | Scientific behavior, provenance, standalone workflow, and result regression checks |

This repository is self-contained: everything needed to run the code, the tests, and the table rebuild is in Git. Not in Git are the full construction caches and archived per-run outputs of the saved experiments (about 9.6 GB). They are distributed as a separate archive; when restored they occupy `outputs/<dataset>_ablation/` and `outputs/study_sources/` and are catalogued in `outputs/output_sources.json`. `outputs/tables/` is an optional local mirror of regenerated tables. Model checkpoints are not bundled. See [the archive procedure](docs/data_and_artifacts.md#large-output-archives) for packaging, verification, and restoration. Dataset content remains subject to its original terms. Source licensing is in [LICENSE](LICENSE).
