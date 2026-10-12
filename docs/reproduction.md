# Reproducing Saadi

Run commands from the repository root after installation. The shell launchers source `scripts/env.sh`; the Python study runner supplies the same local directory defaults.

Dataset sources, exact preparation commands, expected saved-study populations, and the NovelHopQA corpus layout are in [datasets and artifacts](data_and_artifacts.md). Paper symbols and table commands are mapped in [paper to code](paper_to_code.md).

## Data and construction

The loaders support LooGLE, QASPER, MuSiQue, NovelHopQA, NovelQA, NarrativeQA, and QuALITY. The main study suite selects the first four. Configurations preserve their evidence-span construction and grouping rules. MuSiQue and QASPER are prepared into local document/QA files before indexing. The grouped-data preparation script is the one that produced the saved populations; it is listed in `docs/preserved_sources.json`.

NovelHopQA needs its source book corpus. Place the files in `data/novelhopqa/book-corpus-root`, or set `NOVELHOPQA_BOOKS_ROOT` to the corpus directory. Dataset downloads and model weights are not bundled. New runs resolve remote dataset versions at load time; archive the prepared inputs and resolved run manifests with each experiment.

The default construction model is `Qwen/Qwen3-30B-A3B-Instruct-2507`. Shared settings specify a branching factor of 5, a budget of 5 entities, 2 aliases and 3 descriptions, and wavefront scheduling. Each dataset config supplies its chunk size. Defaults for a new run use seed 42, temperature 0, BM25 active-memory selection, and the original chunk text as L0 history (`history_l0_text: raw`). The `published` profile reads the saved main construction settings from the verified reference manifest before applying suite overrides, and sets `history_l0_text: anchored`, the behavior under which the archived caches were built.

| Configuration field | Meaning |
|---|---|
| `active_entity_selector` | Selector used to choose active memory for construction |
| `history_l0_text` | Text shown to the model for recent chunks in the hierarchical history: `raw` (original chunk text, default) or `anchored` (the copy with canonical names placed after span anchors, as in the saved constructions). The folds always summarize the anchored copies. |
| `max_entities`, `max_aliases`, `max_desc` | Memory/prompt budgets |
| `branching_factor` | Hierarchical summary branching factor |
| `lambda_t` | Chunk-text score weight |
| `lambda_n` | Memory score weight; retained internal parameter name |
| `lambda_h` | Hierarchy score weight |
| `memory_name_weight`, `memory_alias_weight`, `memory_description_weight` | Weights within the memory score |
| `retrieve_k` | Saved ranking depth |
| `eval_k` | Primary HR cutoff |

## HR

A query is a hit at cutoff k when the retrieved chunks contain a sufficient single evidence chunk or every chunk in at least one complete evidence group. The denominator contains only queries with a sufficient chunk or a nonempty evidence group. A partial group does not count as a hit. No eligible queries produces an unavailable value, not zero. Evaluation rejects cutoffs beyond the known saved ranking depth.

Result records contain `relevant_chunk_ids` and `evidence_groups`. Evaluation JSON contains `metric: HR`, `n_examples`, `n_eligible`, `n_hits`, `hit_rate`, and `hit_rate_at_k`. JSON rates are fractions; tables show percentages. Readers also accept the field names used in the archived caches.

For example, replacing the placeholder paths with a ranking file and an output path:

```bash
saadi eval --results_json path/to/retrieval_results.json --out path/to/hr.json
```

For a standalone ranking file without a neighboring `report_index.json`, supply `--retrieve_k` with its saved depth.

## Study suites

| Suite | Controlled variation |
|---|---|
| `main`, `retrievers` | Main dataset/retriever matrix |
| `chunk_size`, `context_expansion` | Chunk and context sizes |
| `entity_capacity`, `entity_selector` | Active memory budgets and selection |
| `hierarchy` | Hierarchy branching factor |
| `memory_conditioning` | History and memory conditioning |
| `construction_models` | Construction language model |
| `components`, `retrieval_methods` | Saadi component and scoring ablations |
| `memory_fields`, `normalization` | Memory fields and score normalization |
| `mbr_step_a`, `mbr_step_b` | Retrieval scoring configurations |
| `multihop` | Retrieval grouped by question hop count |
| `generation` | Saadi folded-context generation at k=5 and k=10 |

The configuration files define the exact matrix. A saved table may cover fewer configurations than a suite can execute; the reference manifest lists the available results. Repeated conditioning runs have separate identities; the saved conditioning table reports single runs.

## Output identity and reuse

The runner writes a resolved `runtime_config.yaml` under each run. Input file hashes, construction settings, and source-code hashes identify the index. Retrieval settings and the selected profile identify each run directory. Changing a selector, model, sample cap, or scoring setting does not overwrite an incompatible run.

Completed construction/retrieval stages have a `.stage.json` containing input identities and artifact hashes. A matching stage is verified and reused; a changed artifact is rejected. A file lock prevents simultaneous writes to a stage. Remove neither the marker nor its artifacts to work around a mismatch: choose another `SADI_OUTPUT_DIR` when intentionally starting afresh. Auxiliary evaluation and generation directories are excluded from construction-stage hashes.

Retrieval-only and generation-only stages also require verified completed inputs. Relative local dataset paths are resolved against the dataset experiment config before the runtime config is written. `sample.max_qa_entries` caps evaluation after questions are matched to the selected documents.

`--dry-run` prints commands and writes no configs, indexes, or outputs. For grouped datasets without prepared files, a dry run cannot know their eventual content hashes; paths printed before preparation are provisional. Model inference, GPU kernels, and unpinned model revisions can introduce differences between independent executions.

## Additional evaluations

Numbered scripts under `scripts/evaluation/updated_chunks/` compute structural statistics, cumulative growth, and model-judged faithfulness. Pass the resolved configuration for the run to evaluate:

```bash
DATASET=loogle bash scripts/evaluation/updated_chunks/2_1_Chunk_Level_Changes.sh \
  --experiment_config /absolute/path/to/runtime_config.yaml
```

Cross-dataset plotting configs use `SADI_LOOGLE_CONFIG`, `SADI_QASPER_CONFIG`, `SADI_NOVELHOPQA_CONFIG`, and `SADI_MUSIQUE_CONFIG` to identify four explicit runtime configs. Faithfulness judging requires a model; structural statistics can operate on saved construction outputs. These evaluations retain the original algorithms.

## Archived study outputs

The full experiment archive, distributed separately from this repository, contains the four dataset ablation folders, including the construction caches. It was assembled from the authors' experiment outputs with:

```bash
python scripts/import_study_outputs.py --source-root /absolute/path/to/experiment/outputs --include-caches
```

The source directory must contain the four dataset ablation folders and the archived table inputs identified by `outputs/sadi/reference/manifest.json`. The import verifies source and destination hashes, copies rankings and caches unchanged, keeps only Saadi entries, refuses to overwrite a different existing file, and writes `outputs/output_sources.json`. Omit `--include-caches` for a smaller copy that is sufficient for table generation.

Rebuild from the copied outputs:

```bash
python scripts/build_tables.py --input-source outputs --figures
python scripts/build_tables.py --input-source outputs --output-dir outputs/tables --figures
# Rebuild only the main table beside this script:
python results/tables/main/sadi_retrieval/generate.py --input-source outputs
# Rebuild only the language-model ablation in the local output tables:
python outputs/tables/ablations/construction_model/generate.py --input-source outputs
```

A fresh checkout needs no archive to rebuild the saved tables: run `python scripts/build_tables.py --input-source reference`. Both routes check the same saved values, and the per-table `sources.json` and `verification.json` record the route used. The generators reproduce the saved study and reject changed inputs; new runs are evaluated by the study runner and have separate run identities.

## Large output bundles

See [packaging, verification, and restoration](data_and_artifacts.md#large-output-archives) for the archive and per-file checksum workflow. Large caches are optional for rebuilding the saved tables. A public download location has not yet been configured.

## Saved comparison results

This repository includes structured CSVs in `outputs/other_methods/`. They feed the final comparison tables without running other methods. The compact Saadi inputs live under `outputs/sadi/`, and each final table has its own generator under `results/tables/`.

```bash
python scripts/validate_table_inputs.py
python scripts/build_tables.py
python results/tables/main/matched_exposure/generate.py
```

The saved comparison CSVs and supplementary Saadi inputs were produced once with `python scripts/import_comparison_inputs.py --source-root /absolute/path/to/experiment/workspace`, which expects the authors' experiment layout. A fresh checkout already contains its output and does not need it. See [the comparison input schema](../outputs/other_methods/README.md) for intentional CSV updates.
