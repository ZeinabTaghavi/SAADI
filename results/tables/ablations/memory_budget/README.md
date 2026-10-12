# Active memory budget

This folder contains the Saadi table, its generator, and its input provenance.

```bash
python generate.py
# Require the restored full output archive:
python generate.py --input-source outputs
```

Run from this folder, or invoke the script by its path from the repository root. The default reads the restored output archive when it is present and the bundled reference inputs otherwise.

These values are transcribed from Table 6 of the paper and labeled `paper_reported_summary`; the PDF hash, page, and panel are recorded in `sources.json`. They are a transcription, not a recomputation: the query-level outputs of that run are not included. Presentation HR values are percentages; `metrics.csv`, when present, stores fractions. See [table.md](table.md), [sources.json](sources.json), and [verification.json](verification.json).
