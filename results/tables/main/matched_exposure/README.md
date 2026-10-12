# Retrieval at matched evidence exposure (Table 2)

Run `python generate.py` here, or invoke this script by its path. Saadi inputs come from `outputs/sadi/` (or the restored full experiment archive); comparison methods come from `outputs/other_methods/<method>/*.csv`. No comparison method is rerun.

[table.md](table.md), [table.csv](table.csv), and [table.tex](table.tex) are generated together. CSV values retain full precision; HR columns are percentages. [metrics.csv](metrics.csv) records the inputs used, including per-query versus reported-summary evidence. [sources.json](sources.json) records input hashes and [verification.json](verification.json) records coverage.

Use `--input-source reference` to require the compact Saadi inputs, or `--input-source outputs` to require the restored archive. Use `--other-methods-root` for a separately validated set of comparison CSVs. Agreement with the paper is checked separately by the paper-consistency report.
