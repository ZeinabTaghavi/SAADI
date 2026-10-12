# Paper consistency check

Numerical comparison with the supplied 53-page `SAADI_ICLR.pdf`, identified by its SHA-256 in `paper.json`. Expected values were extracted independently from the paper. Generated values are rounded to the paper's stated precision before comparison.

Checked **1310 cells**: **1307 match**, **3 differ**, and **0 are missing from the generated tables**.

| Paper table | Matching cells | Different cells | Missing cells |
| --- | ---: | ---: | ---: |
| 1 | 184 | 0 | 0 |
| 2 | 160 | 0 | 0 |
| 6A | 16 | 0 | 0 |
| 6B | 8 | 0 | 0 |
| 7 | 32 | 0 | 0 |
| 8 | 64 | 0 | 0 |
| 9 | 60 | 0 | 0 |
| 10 | 108 | 0 | 0 |
| 11 | 48 | 0 | 0 |
| 12 | 64 | 0 | 0 |
| 15 | 19 | 1 | 0 |
| 16 | 50 | 0 | 0 |
| 17 | 24 | 0 | 0 |
| 21 | 40 | 0 | 0 |
| 22 | 120 | 0 | 0 |
| 23 | 80 | 0 | 0 |
| 24 | 140 | 0 | 0 |
| 25 | 14 | 2 | 0 |
| 26 | 60 | 0 | 0 |
| 27a | 16 | 0 | 0 |

The [cell comparison](comparison.csv) lists each paper value, its generated value, row selector, and status. [The machine-readable report](report.json) records the table hashes and coverage.

## Scope and notes

- Tables 1, 2, 22, and 24 are checked in full: 184 Table 1 cells (40 Saadi cells recomputed from rankings, 144 saved comparison-method values), 160 Table 2 cells, 120 Table 22 cells, and 140 Table 24 cells.
- Table 6: the 24 cells are transcribed from the paper (`paper_reported_summary`) because the summaries stored with the experiment archive came from a different construction run; agreement checks the transcription.
- Table 26: the 60 Saadi HR cells are checked (BM25 at k=5/10/15/20, Contriever at k=5/10). The 64K condition uses the 1,333-query population; the other QASPER expansion snapshots contain the 1,321 queries common to all sizes. Comparison-method cells of Tables 21 and 26 come from saved CSVs and are not part of this check.
- Tables 15 and 25: three cells differ in the second decimal. LooGLE has 3,346 chunks over 105 documents, 31.867 per document, printed as 31.86 in the paper.
- Table 7 covers LooGLE and NovelHopQA, as in the paper.
- Held-out selection, faithfulness, per-channel retrieval ablations, paired outcomes, and qualitative cases are outside this check.

## Reproduce this check

Run from the repository root:

```bash
python scripts/build_tables.py --output-dir results/tables
python scripts/audit_paper.py
```

The audit exits with code 1 when any checked cell differs or is missing. Agreement with the saved inputs (`verification.json`) and agreement with the paper (this check) are separate checks.
