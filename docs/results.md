# Saved results and provenance

## What is included

`outputs/sadi/reference/manifest.json` inventories 150 saved result sources with their file hashes, query-population hashes, ranking depths, and construction settings. Together with `outputs/sadi/comparison_inputs/` and `outputs/other_methods/`, these compact inputs rebuild every table under `results/tables/` in a fresh checkout. Rankings store chunk identifiers and evidence labels, not document text or model checkpoints.

The main-study query populations are the same for all five retrievers: LooGLE 1,832; QASPER-64K 1,333; NovelHopQA 908; MuSiQue-32K 900.

## Evidence levels

Every Saadi table cell carries one of three evidence levels, recorded in the folder's `sources.json` and `metrics.csv`:

- **Per-query** (276 cells): HR recomputed from saved rankings and evidence labels. Query identities and file hashes are checked against the manifest.
- **Reported summary** (168 cells): aggregate values preserved from saved outputs, with their source hashes. This covers construction statistics, construction cost, multihop summaries, and the fusion-weight studies.
- **Paper-reported** (24 cells): the recall-budget and selector values of Table 6, transcribed from the paper. The PDF hash, page, and panel are recorded. See the notes below.

Comparison-method values come from the saved CSVs in `outputs/other_methods/` (371 retrieval rows and 20 resource rows). No comparison method is executed.

## Provenance notes

- **Memory selector.** The saved main constructions recalled memory with word overlap (`active_entity_selector: word`). New runs default to BM25 recall, the setting the paper describes and which Table 6(b) reports as slightly better. `--profile published` restores the saved setting.
- **History text.** The saved constructions showed the model the anchored chunk copies as recent history (`history_l0_text: anchored`). New runs default to the original chunk text (`raw`). See [the paper guide](paper_to_code.md#method-and-notation).
- **Table 6.** The budget and selector values are taken from the paper because the summaries stored with the experiment archive came from a different construction run. The query-level outputs of the paper's run are not included.
- **QASPER expansions.** The 64K condition uses the full 1,333-query population. The 32K, 128K, and 256K snapshots contain the 1,321 queries common to all expansion sizes; the Contriever expansion snapshots use the same intersection.
- **Construction-model ablation** (Table 7) covers LooGLE and NovelHopQA, as in the paper.
- **Conditioning ablation** (Tables 12 and 27a) reports single runs.
- **Rounding.** Three construction-statistics cells (Tables 15 and 25) differ from the paper in the second decimal. LooGLE has 3,346 chunks over 105 documents, 31.867 per document, which the paper prints as 31.86.

## Rebuilding the tables

`python scripts/build_tables.py` rebuilds `results/tables/` from the compact inputs. When the separate output archive has been restored and `outputs/output_sources.json` exists, the default route reads the archived files instead; `--input-source reference` or `--input-source outputs` selects a route explicitly. Both routes verify input hashes and check the same saved values, and a missing or changed input is an error rather than a fallback. The committed tables are generated with the reference route.

Each study folder contains `generate.py`, `table.csv`, `table.md`, `table.tex`, `sources.json`, and `verification.json`; HR studies also contain `metrics.csv`. Growth series and the optional retrieval-depth figures sit beside their tables. `results/tables/combined/all_metrics.csv` is the consolidated machine-readable HR table with source identifiers, units (fractions), cutoffs, eligible counts, and evidence levels. An individual `generate.py` can be run from any directory; its defaults resolve against the repository.

Tables are generated artifacts. To change a value, update the saved inputs through the documented workflow, then rebuild; do not edit table cells.

## Final comparisons

`outputs/sadi/` supplies the Saadi rankings and summaries; `outputs/other_methods/<method>/` supplies the saved results of the other methods. The final generators read these inputs directly and never read another generated table. Matched exposure (Table 2) evaluates Saadi at the floor and ceiling of each agent's unrounded mean chunk exposure. Resource crossovers (Table 24) use `(Saadi construction calls per document - method construction calls per document) / method calls per query`, with Saadi's construction cost taken over the same document population as the method, including UMA's 16-document NovelHopQA scope.

## The separate output archive

The full construction caches, archived rankings, evaluation reports, and run configurations of the saved experiments (about 9.6 GB) are distributed separately; see [large output archives](data_and_artifacts.md#large-output-archives). When restored they occupy `outputs/<dataset>_ablation/` and `outputs/study_sources/`, with `outputs/output_sources.json` mapping each saved source to its archived file. Archived files are preserved byte-for-byte, including their original field names and paths.

## Code provenance

`docs/preserved_sources.json` lists the source files that are identical to the code that produced the saved experiments. Four construction files (`hierarchical_summary.py`, `state.py`, `hierarchy_cache.py`, `ingest_pipeline.py`) were modified afterwards to add the `history_l0_text` option, whose `anchored` value reproduces the saved behavior, and are therefore not listed. Orchestration, configuration loading, output serialization, and table generation were written for this repository.
