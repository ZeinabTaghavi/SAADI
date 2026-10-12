# Tables

Regenerate all tables with `python scripts/build_tables.py` from the repository root. Each folder also contains a standalone `generate.py`. Inputs live under `outputs/`; this directory contains generated tables, code, and provenance.

## Final comparisons

Saadi measurements are joined to saved CSVs in `outputs/other_methods/`. No other method is executed.

- [Retrieval comparison (Table 1)](main/retrieval_comparison/README.md)
- [Retrieval at matched evidence exposure (Table 2)](main/matched_exposure/README.md)
- [Fixed-depth and agentic retrieval (Table 22)](comparisons/fixed_and_agentic/README.md)
- [Retrieval by hop count (Table 21)](comparisons/multihop/README.md)
- [Context expansion comparison (Table 26)](comparisons/context_expansion/README.md)
- [Model calls and tokens (Table 24)](comparisons/resource_use/README.md)
- [Saadi construction calls per document (Table 25)](construction/calls_per_document/README.md)

## Saadi studies

- [Saadi retrieval across datasets and retrievers](main/sadi_retrieval/README.md)
- [Active memory budget](ablations/memory_budget/README.md)
- [Active memory selection](ablations/memory_selector/README.md)
- [Hierarchy branching factor](ablations/hierarchy_branching/README.md)
- [Construction language model](ablations/construction_model/README.md)
- [Score formulation and normalization](ablations/score_formulation/README.md)
- [Score fusion weights](ablations/score_fusion/README.md)
- [Memory field weights](ablations/memory_field_weights/README.md)
- [History and memory conditioning](ablations/state_conditioning/README.md)
- [Context expansion](robustness/context_expansion/README.md)
- [Retrieval by question hop count](robustness/multihop/README.md)
- [Retrieval depth](robustness/retrieval_depth/README.md)
- [Cumulative memory growth](construction/memory_growth/README.md)
- [Memory expansion statistics](construction/memory_expansion/README.md)
- [Entity memory statistics](construction/memory_statistics/README.md)
- [Hierarchical summary statistics and growth](construction/hierarchy_statistics/README.md)
- [Saadi construction cost](construction/cost/README.md)

See [the paper audit](../../results/paper_consistency/README.md) for the cell-by-cell comparison with the paper. The Saadi-only [combined metrics](combined/all_metrics.csv) retain all saved study configurations.
