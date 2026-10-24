# Saadi: paper to code

This guide maps *Understand Once, Retrieve Many Times: SAADI for Single-Pass Retrieval of Distributed Evidence in Long Documents* to the release. Table numbers refer to the 53-page `SAADI_ICLR.pdf` identified by SHA-256 in [paper.json](../results/paper_consistency/paper.json). The public project name is **Saadi**. The paper uses the uppercase acronym SAADI.

## Method and notation

Paths below are relative to the repository root. The source files identical to the code that produced the saved experiments are listed in [preserved_sources.json](preserved_sources.json).

| Paper concept | Implementation | Configuration / output |
|---|---|---|
| Section 4, dataset loading and evidence alignment | [dataset loaders](../src/saadi/indexing/dataset_loaders/), [grouped preparation](../scripts/ablations/retrieval/prepare_grouped_retrieval_dataset.py), [chunking](../src/saadi/core/chunking.py) | `configs/experiments/*_retrieval_ablation.yaml`; [data instructions](data_and_artifacts.md) |
| Section 3.1, chunks `C_t` | [chunking.py](../src/saadi/core/chunking.py) | `ingest.chunk_size`, `ingest.chunk_overlap: 0`; original chunk text is returned at retrieval time |
| Recall `R_t` from memory `M_(t-1)` | [active_selection.py](../src/saadi/entities/active_selection.py), `select_active_entities` | `ingest.strategy_kwargs.active_entity_selector`, `max_entities`, `max_aliases`, `max_desc` |
| Equation (1), extraction `A_t` and memory update | [hierarchical_summary.py](../src/saadi/context_strategies/hierarchical_summary.py), [merge.py](../src/saadi/entities/merge.py), [experiment_runner.py](../src/saadi/pipelines/experiment_runner.py) | construction model in `configs/models/construction.yaml`; entity records and per-chunk name links in the index |
| History `H_t`, folding factor `f` | [hierarchical_summary.py](../src/saadi/context_strategies/hierarchical_summary.py) | `ingest.strategy_kwargs.branching_factor`; `history_l0_text` selects the text shown for recent chunks (see below); hierarchy caches retain the construction history |
| Equation (2), text / memory / summary channels | [mbr.py](../src/saadi/indexing/retrieval_methods/mbr.py), `build_mbr_document_memory_index`, `score_mbr_variant_document_pools`, `combine_mbr_variant_chunk_scores` | `cached_retrieval.method: mbr`; summaries used for retrieval are level one |
| `lambda_T`, `lambda_M`, `lambda_S` | `MBRWeights` in `mbr.py` | `cached_retrieval.lambda_t`, **`lambda_n`**, **`lambda_h`**, respectively; defaults `1`, `0.5`, `0.3` |
| Per-channel normalized score, Equation (2) | `normalize_signal_scores` in `mbr.py` | `cached_retrieval.normalization_method: zscore` |
| Memory field weights, Appendix C.5.2 | memory scoring in `mbr.py` | `memory_name_weight`, `memory_alias_weight`, `memory_description_weight`; `memory_aggregation: max` |
| Equation (3), construction call count | folding in `hierarchical_summary.py`; `construction_table` in [comparison_tables.py](../src/saadi/reporting/comparison_tables.py) | `N + sum(floor(N / f**i))`; Table 25 combines saved document-level statistics, not a floor applied to a mean document |
| HR at cutoff `k` | [hr.py](../src/saadi/evaluation/hr.py), used by evaluation and the table builder | `retrieve_k` is ranking depth; `eval_k` is the primary evaluation cutoff; saved results use HR@5 / HR@10 |

`mbr`, `lambda_n`, `lambda_h`, `outputs/sadi/`, the CSV method ID `sadi`, and the `SADI_*` environment variables are internal identifiers kept for compatibility with the saved outputs. Tables call the method **Saadi** and the metric **HR**. The memory-recall selector used at construction time (`active_entity_selector`) is independent of the query-time base retriever; the saved constructions used word overlap and new runs default to BM25, see the [provenance notes](results.md#provenance-notes).

**Span anchors.** Each state-update call returns one span anchor per active entity (Appendix B.2). An entity whose anchor is not found in the chunk is discarded. `ingest.strategy_kwargs.anchor_matching` sets how an anchor is located: `strict` requires the model's left and right context to match character for character (the saved constructions), and `tolerant` (default for new runs) also accepts an anchor that occurs verbatim and exactly once in the chunk. For matched anchors the canonical name is placed after the mention in a working copy of the chunk, and the level-one folds summarize these copies. Retrieval scores and returns the original chunks; anchors are not used at query time. `ingest.strategy_kwargs.history_l0_text` sets which text the model sees as recent history while processing the next chunk: `raw`, the original chunk text (default for new runs), or `anchored`, the copies with inserted names (the saved constructions). A cache records the setting it was built with and is not reused under the other one.

## Source flow and evidence levels

```text
outputs/sadi/reference/          rankings + saved Saadi summaries
outputs/sadi/comparison_inputs/  extra rankings + cost/exposure inputs
outputs/other_methods/<method>/  saved comparison CSVs
                    |
                    v
             scripts/build_tables.py
                    |
                    v
results/tables/<category>/<study>/  table.csv / .md / .tex
                                  generate.py, sources.json, verification.json
```

With `--input-source reference` only the compact inputs in Git are read; with `--input-source outputs` the restored full archive is read through `outputs/output_sources.json`. Final comparison tables read inputs directly, never another generated table, and no comparison method is executed. The evidence level of every cell (per-query, reported summary, or paper-reported) is defined in [saved results and provenance](results.md#evidence-levels). `verification.json` records agreement with the saved inputs; the [paper audit](../results/paper_consistency/README.md) records agreement with the PDF.

## Table and configuration map

Every table directory below contains its own `generate.py`. From the repository root, run `python results/tables/<directory>/generate.py --input-source reference`. To generate all tables, use:

```bash
python scripts/build_tables.py --input-source reference
python scripts/audit_paper.py
```

The suite column identifies controls for **new experiments**, not a promise that today's default suite recreates every historical configuration. The exact saved conditions and model settings are in [the reference manifest](../outputs/sadi/reference/manifest.json). Suites live in [configs/suites](../configs/suites/); `scripts/run_study.py --suite NAME --dry-run` shows their current scope. The table generators use saved inputs, regardless of current suite defaults.

| Paper table | Table directory under `results/tables/` | New-run suite / implementation | Saved inputs and evidence |
|---|---|---|---|
| 1, Saadi rows | [main/sadi_retrieval](../results/tables/main/sadi_retrieval/) | `main`; Eq. (2) scoring | Reference `main`: per-query |
| 1, full comparison | [main/retrieval_comparison](../results/tables/main/retrieval_comparison/) | `main`; `comparison_tables.py` | Saadi main rankings + other-method CSVs |
| 2 | [main/matched_exposure](../results/tables/main/matched_exposure/) | `comparison_tables.py`, `matched_table` | Saadi depth rankings + exposure and cost inputs + other-method CSVs |
| 6A | [ablations/memory_budget](../results/tables/ablations/memory_budget/) | `entity_capacity` | Reference `active_entity_budget`: paper-reported |
| 6B | [ablations/memory_selector](../results/tables/ablations/memory_selector/) | `entity_selector` | Reference `active_entity_selector`: paper-reported |
| 7 | [ablations/construction_model](../results/tables/ablations/construction_model/) | `construction_models` | Reference `llm_ablation`: per-query; LooGLE and NovelHopQA |
| 8 | [ablations/hierarchy_branching](../results/tables/ablations/hierarchy_branching/) | `hierarchy` | Reference `branching_factor`: per-query |
| 9 | [ablations/score_fusion](../results/tables/ablations/score_fusion/) | `mbr_step_a`; `lambda_t/n/h` | Reference `saadi_score_fusion_weights`: reported summary |
| 10 | [ablations/memory_field_weights](../results/tables/ablations/memory_field_weights/) | `mbr_step_b`, `memory_fields` | Reference `saadi_score_level_2_memory_weights`: reported summary |
| 11 | [ablations/score_formulation](../results/tables/ablations/score_formulation/) | `retrieval_methods`, `normalization` | Reference `saadi_score_formulation_and_normalization`: per-query |
| 12, 27(a) | [ablations/state_conditioning](../results/tables/ablations/state_conditioning/) | `memory_conditioning` | Reference `state_conditioning_ablation`: per-query |
| 15 | [construction/memory_statistics](../results/tables/construction/memory_statistics/) | `scripts/evaluation/updated_chunks/2_2_Active_Entity_Statistics.sh` | Reference `active_entities_statistics`: reported statistics |
| 16 | [construction/memory_expansion](../results/tables/construction/memory_expansion/) | `context_expansion`; active-entity statistics evaluation | Reference `active_entities_expansion_statistics`: reported statistics |
| 17 | [construction/hierarchy_statistics](../results/tables/construction/hierarchy_statistics/) | `scripts/evaluation/updated_chunks/2_4_Hierarchical_Summary_Statistics.sh` | Reference `hierarchical_summary_statistics`: reported statistics + growth series |
| 21, Saadi rows | [robustness/multihop](../results/tables/robustness/multihop/) | `multihop`; `scripts/evaluation/retrieval/evaluate_multihop.py` | Reference `multihop`: reported statistics |
| 21, comparison | [comparisons/multihop](../results/tables/comparisons/multihop/) | `comparison_tables.py` | Saadi summary + exposure inputs + other-method CSVs |
| 22 | [comparisons/fixed_and_agentic](../results/tables/comparisons/fixed_and_agentic/) | `comparison_tables.py` | Saadi depth rankings + other-method CSVs |
| 23 | [robustness/retrieval_depth](../results/tables/robustness/retrieval_depth/) | `main` with sufficient `retrieve_k`; Eq. (2) scoring | Reference `retrieval_depth`: per-query |
| 24 | [comparisons/resource_use](../results/tables/comparisons/resource_use/) | `comparison_tables.py` | Saadi comparison-specific cost inputs + other-method resource CSVs |
| 25 | [construction/calls_per_document](../results/tables/construction/calls_per_document/) | Eq. (3); `comparison_tables.py` | Saadi reported memory/hierarchy statistics |
| 26, BM25 Saadi rows | [robustness/context_expansion](../results/tables/robustness/context_expansion/) | `context_expansion` | Reference `context_expansion`: per-query |
| 26, comparison including Contriever | [comparisons/context_expansion](../results/tables/comparisons/context_expansion/) | `context_expansion`; `comparison_tables.py` | Reference + supplementary Saadi rankings + other-method CSVs |
| Supporting construction data | [construction/memory_growth](../results/tables/construction/memory_growth/) | `scripts/evaluation/updated_chunks/3_1_Per_Document_Cumulative_Growth_Logging.sh` and related growth scripts | Reference `active_entities_cumulative_growth`: reported series/statistics |
| Supporting construction data | [construction/cost](../results/tables/construction/cost/) | construction logs; `tables.py` | Reference `construction_cost`: reported statistics |

Saadi-only table formatting is implemented in [tables.py](../src/saadi/reporting/tables.py); final comparison formulas are in [comparison_tables.py](../src/saadi/reporting/comparison_tables.py). For example:

```bash
python results/tables/ablations/memory_budget/generate.py --input-source reference
python results/tables/main/retrieval_comparison/generate.py --input-source reference
python scripts/build_tables.py --input-source reference --output-dir outputs/tables
```

## Coverage and limits

`--figures` generates the retrieval-depth curve (`robustness/retrieval_depth/curve.png`, `.pdf`, `.svg`). The other paper figures, the held-out parameter selection, the model-judged faithfulness tables, the per-channel retrieval ablations, the paired outcomes, and the qualitative cases are not generated from saved inputs; the evaluation code for the faithfulness and statistics studies is under `scripts/evaluation/`.

The audit checks 1,310 cells of Tables 1, 2, 6 to 12, 15 to 17, 21 to 26, and 27a: 1,307 match and three construction-statistics cells differ in the second decimal (see the [provenance notes](results.md#provenance-notes)). For Tables 21 and 26 the Saadi cells are checked; the comparison-method cells of those two tables come from saved CSVs. A fresh GPU construction is a new experiment with its own run identity.
