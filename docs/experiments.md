# Running the experiments of the paper

This page is for reproducing the study. Using the library on your own documents is covered by [library usage](library_usage.md); the saved results can be rebuilt without any of the hardware below, see [Reproducing the tables](#view-the-saved-results).

The project, Python package, and command are **Saadi**, `saadi`, and `saadi`, respectively. Historical artifact paths such as `outputs/sadi/`, method IDs such as `sadi`/`mbr`, and `SADI_*` environment variables are retained for compatibility with the saved experiments. Public tables and evaluation reports use **HR@5** and **HR@10**.

See the [paper-to-code guide](paper_to_code.md) for equations, notation, and table commands, and [datasets and artifacts](data_and_artifacts.md) for data preparation, cache restoration, and the exact publication contents.

## Development install

Clone the repository and use Python 3.11 or 3.12 on Linux or macOS (the study runner uses POSIX file locks). Run these commands from the repository's root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pre-commit install
python scripts/smoke_test.py
python -m pytest -q
```

The smoke test constructs a synthetic document index, performs Saadi BM25 retrieval, and checks HR on CPU using a mock model. Its scores are software checks, not experimental results.

For dataset downloads, plots, dense retrieval, and model inference:

```bash
python -m pip install -e '.[repro]'
```

Model inference uses vLLM on a supported GPU environment. The construction configuration defaults to four GPUs; set `--tensor-parallel-size` for your hardware. Exact GPU package versions and model revisions should be recorded with each new experiment. The CPU validation environment is recorded in `validation_environment.json`.

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

This rebuilds `results/tables/` without downloading data or running a language model. A fresh checkout uses the compact `outputs/sadi/reference/` and `outputs/sadi/comparison_inputs/` bundles. When the optional full output archive has been restored and catalogued in `outputs/output_sources.json` (see [datasets and artifacts](data_and_artifacts.md#large-output-archives)), the generators read those archived files instead. Saved CSVs under `outputs/other_methods/<method>/` supply the other methods without rerunning them. Both routes verify input hashes and check the same saved study values. Use `--input-source outputs` or `--input-source reference` to require a particular route. Missing or changed inputs in an existing output catalog cause an error.

Table directories use `main/`, `comparisons/`, `ablations/`, `robustness/`, and `construction/` categories, with a separate folder for each study. Each folder contains its `generate.py`, tables, source paths and hashes, and verification results. Tables are available as CSV, Markdown, and LaTeX; presentation HR values are percentages. `combined/all_metrics.csv` collects all available HR configurations.

[Final main comparison](../results/tables/main/retrieval_comparison/table.md), [Saadi-only results](../results/tables/main/sadi_retrieval/table.md), and the [table index](../results/tables/README.md) can be browsed directly. [Saved results and provenance](results.md) lists the evidence level of every table and the provenance notes.

The [paper consistency check](../results/paper_consistency/README.md) compares the generated tables with the paper: 1,307 of 1,310 checked cells match, including all 184 values of Table 1. The three differences are second-decimal rounding of construction statistics.

The notebook [examples/paper_results.ipynb](../examples/paper_results.ipynb) performs this rebuild on a plain CPU runtime, checks that the rebuilt tables are byte-identical to the committed ones, reruns the paper consistency check, and displays the tables.

## Run Saadi

```bash
# Inspect the commands and resolved scope without writing files:
python scripts/run_study.py --suite main --datasets loogle --retrievers bm25 --dry-run

# Construct the index, retrieve, and evaluate:
python scripts/run_study.py --suite main --datasets loogle --retrievers bm25
```

The full `main` suite covers LooGLE, QASPER, NovelHopQA, and MuSiQue with BM25, Contriever, BGE-M3, Jina, and Qwen retrieval. Comparison methods are not run; their saved results are in `outputs/other_methods/`. Dataset preparation and evidence labeling are the same code that produced the saved results.

New runs use BM25 memory selection and show the model the original text of recent chunks as history. The saved results were produced with word-overlap selection and the anchored chunk copies; see the [provenance notes](results.md#provenance-notes). To construct with the saved settings:

```bash
python scripts/run_study.py --suite main --profile published --datasets loogle --retrievers bm25
```

This profile restores the recorded selector, history text, model configuration, temperature, and sampling settings. Model revisions and runtime versions of the original runs are not pinned, so regenerated model text can differ. The table rebuild checks the saved results exactly.

The default directories are `data/`, `.cache/`, and `outputs/`. Override them with `SADI_DATA_DIR`, `SADI_CACHE_DIR`, and `SADI_OUTPUT_DIR`. New run data, caches, and large outputs are excluded from Git; the small saved table-input bundles are included. NovelHopQA requires the book files described in [the dataset guide](data_and_artifacts.md#novelhopqa-book-corpus).

## Configuration and ablations

Dataset overrides in `configs/experiments/` extend `configs/methods/sadi.yaml`, which extends the construction settings in `configs/models/construction.yaml`. Suites in `configs/suites/` define experimental variations. Configuration keys are validated before loading data or models.

```bash
python scripts/run_study.py --suite hierarchy --datasets loogle --retrievers bm25
python scripts/run_study.py --suite memory_conditioning --datasets loogle --retrievers bm25
python scripts/run_study.py --suite construction_models --datasets loogle --retrievers bm25
```

Use `--stage prepare`, `ingest`, `retrieve`, or `generate` to run a particular stage. `--max-documents 2` creates a separate small-run identity. See [reproduction instructions](reproduction.md) for other suites, configuration fields, stage reuse, and generation.

## Repository contents

| Directory | Purpose |
|---|---|
| `src/saadi/` | The library (`document.py`, `llm/`, `retrievers/`, `prompts/`, `inputs/`, `cli/`) and the construction, memory selection, retrieval, generation, and evaluation code of the paper |
| `configs/` | Shared settings, dataset overrides, and study suites |
| `scripts/` | Study runner, grouped-data preparation, evaluation, and result export tools |
| `outputs/sadi/reference/` | Checksummed Saadi rankings and saved summary values |
| `outputs/sadi/comparison_inputs/` | Additional Saadi expansion rankings and comparison-specific cost/exposure inputs |
| `outputs/other_methods/` | Structured saved retrieval/resource CSVs and provenance for comparison methods |
| `results/tables/` | Final comparisons and Saadi study tables, each with its generator and provenance |
| `results/paper_consistency/` | Cell-by-cell comparison of the generated tables with the paper |
| `docs/` | Library usage, customization, paper-to-code guide, dataset and artifact guide, reproduction guide, result coverage, release process |
| `examples/` | Sample inputs and the notebooks |
| `tests/` | Scientific behavior, provenance, library, and result regression checks |

The repository is self-contained: everything needed to run the code, the tests, and the table rebuild is in Git. Not in Git are the full construction caches and archived per-run outputs of the saved experiments (about 9.6 GB). They are distributed as a separate archive; when restored they occupy `outputs/<dataset>_ablation/` and `outputs/study_sources/` and are catalogued in `outputs/output_sources.json`. `outputs/tables/` is an optional local mirror of regenerated tables. Model checkpoints are not bundled. See [the archive procedure](data_and_artifacts.md#large-output-archives) for packaging, verification, and restoration. Dataset content remains subject to its original terms. Source licensing is in [LICENSE](../LICENSE).

The files listed in `preserved_sources.json` are byte-identical to the code that produced the saved results; `tests/test_release.py` checks this. The library layer around them (backends, retrievers, prompt profiles, the `Saadi` class, the CLI) was added afterwards and is excluded from that list.
