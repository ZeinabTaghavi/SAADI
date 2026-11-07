# Saadi table inputs

- `reference/`: compact per-query rankings, evidence labels, saved study summaries, and the source manifest with checksums. These inputs rebuild the 17 Saadi study tables.
- `comparison_inputs/`: ten additional Contriever expansion ranking snapshots, saved multihop exposure values, and Saadi construction costs for the document populations used in the comparisons. Its manifest records source hashes, retained query counts, and filtering rules.

Both directories are in Git and, together with `../other_methods/`, are sufficient to regenerate every table offline. Rankings contain chunk identifiers and evidence labels, not document text or model checkpoints.

The full output archive (about 9.6 GB, including the memory and hierarchy caches) is distributed separately and restores under `../<dataset>_ablation/` and `../study_sources/`; `../output_sources.json`, written when it is imported, maps each saved source to its archived file. `python scripts/build_tables.py` reads the restored archive when that catalog exists and these compact inputs otherwise; `--input-source reference` or `--input-source outputs` selects a route explicitly, and both verify hashes.

Evidence levels and provenance notes, including the memory selector and history text of the saved constructions, are in [docs/results.md](../../docs/results.md).
