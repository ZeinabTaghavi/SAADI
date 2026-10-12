# Saved comparison-method outputs

These small CSVs supply the final paper tables. The repository does not execute the comparison methods. Each method has a `retrieval.csv` and, when applicable, a `resources.csv`. `manifest.json` records their SHA-256 hashes and the original saved sources.

Included methods: Vanilla, LlamaIndex, LQCA, Late Chunking, HippoRAG v2, RAPTOR, RAG-R1, Agentic-R, A-RAG, GRIP, and UMA. Saadi values are read independently from its own outputs.

## Retrieval schema

| Fields | Meaning |
|---|---|
| `study` | `main`, `agentic`, `multihop`, or `context_expansion` |
| `method`, `dataset` | Stable identifiers; the method matches its folder name |
| `condition` | `standard`, `no_expansion`, or `32K_expansion` / `64K_expansion` / `128K_expansion` / `256K_expansion` |
| `retriever`, `variant` | Backend and configuration; `native` for method-specific retrieval, overlap variant for Late Chunking |
| `cutoff` | Positive integer for fixed-depth retrieval; empty for an agent trajectory |
| `hop` | `overall`, `1-hop`, `2-hop`, `3-hop`, or `4-hop` |
| `hr_percent` | HR in **percent**, between 0 and 100 |
| `mean_chunks` | Observed mean distinct chunks admitted before answering, or mean fixed-depth exposure; empty when unavailable in the main summary |
| `queries`, `eligible_queries`, `hits` | Integer counts when recorded; empty means unavailable, never zero by assumption |
| `provenance`, `source_id` | `reported_summary` and a source identifier registered in the manifest |

The row identity is `(study, method, dataset, condition, retriever, variant, cutoff, hop)`. The main table uses standard MuSiQue-32K and QASPER-64K; expansion rows identify their target separately. The final multihop table uses BM25; additional saved BGE-M3 Vanilla rows are retained as inputs but are not displayed there.

## Resource schema

One row per `(method, dataset, condition)` records query/document counts, mean chunks, model calls/input tokens/generated tokens per query, and the method's offline construction use per document. `input_tokens_basis` and `generated_tokens_basis` preserve whether token values were measured or reconstructed. Final-answer generation is excluded. Saadi's corresponding construction rates are stored separately in `../sadi/comparison_inputs/costs.csv`, matched to the method's query/document scope. In particular, UMA's NovelHopQA comparison covers 16 documents.

## Validate and regenerate

From the repository root:

```bash
python scripts/validate_table_inputs.py
python scripts/build_tables.py
python results/tables/main/retrieval_comparison/generate.py
```

To supply revised results, edit the CSV values and record the new source identifier, original source path, and SHA-256 under `sources` in `manifest.json`. Then explicitly accept the schema-valid input changes and regenerate:

```bash
python scripts/validate_table_inputs.py --update-checksums
python scripts/build_tables.py
python scripts/audit_paper.py
```

Changed inputs cannot pass the default hash checks without this explicit update. The table generators also reject missing required rows and duplicate identities. Updating hashes establishes the intended input version; it does not establish agreement with the paper. Do not edit generated table cells.

`scripts/import_comparison_inputs.py --source-root /path/to/experiment/workspace` is the one-time importer that produced these files from the authors' experiment layout. A checkout already contains its portable output and does not need that directory.
