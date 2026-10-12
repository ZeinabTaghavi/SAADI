# Construction language model

This folder contains the Saadi table, its generator, and its input provenance.

```bash
python generate.py
# Require the restored full output archive:
python generate.py --input-source outputs
```

Run from this folder, or invoke the script by its path from the repository root. The default reads the restored output archive when it is present and the bundled reference inputs otherwise.

HR is recomputed from per-query rankings and checked against the saved study values. Presentation HR values are percentages; `metrics.csv`, when present, stores fractions. See [table.md](table.md), [sources.json](sources.json), and [verification.json](verification.json).
