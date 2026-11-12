# C4 addendum: which channel carries the gain inside the gold sub-document

Same scores, queries and restriction as the main C4 analysis (labels re-derived inside the gold document, full-document scores kept). Each SAADI channel is added to the text channel alone with its published weight; nothing is tuned. Values: gain over text alone in points, mean over the four lengths; in brackets, the number of lengths where the 95% CI (10,000 paired bootstrap resamples over queries) is above zero.

**MRR inside the gold document**

| | Text + summary | Text + memory | Full SAADI |
| --- | --- | --- | --- |
| MuSiQue, BM25 | +1.0 (4/4) | +3.5 (4/4) | +4.1 (4/4) |
| QASPER, BM25 | +0.6 (3/4) | +0.4 (1/4) | +1.0 (3/4) |
| MuSiQue, Contriever | +0.4 (0/4) | +4.4 (4/4) | +4.5 (4/4) |
| QASPER, Contriever | +0.3 (0/4) | +0.0 (0/4) | +0.4 (1/4) |

**Share of the full-document top 10 from the gold document**

| | Text + summary | Text + memory | Full SAADI |
| --- | --- | --- | --- |
| MuSiQue, BM25 | +5.0 (4/4) | +0.4 (1/4) | +4.7 (4/4) |
| QASPER, BM25 | +3.6 (4/4) | +3.1 (4/4) | +5.4 (4/4) |
| MuSiQue, Contriever | +6.1 (4/4) | +4.5 (4/4) | +8.3 (4/4) |
| QASPER, Contriever | +2.2 (4/4) | +2.0 (4/4) | +3.5 (4/4) |

**At least one full-document top-10 chunk from the gold document**

| | Text + summary | Text + memory | Full SAADI |
| --- | --- | --- | --- |
| MuSiQue, BM25 | -0.2 (0/4) | +0.1 (0/4) | -0.1 (0/4, 1 lower) |
| QASPER, BM25 | +0.3 (0/4) | +0.1 (0/4) | +0.3 (0/4) |
| MuSiQue, Contriever | -0.1 (0/4) | +0.7 (1/4) | +0.6 (0/4) |
| QASPER, Contriever | -1.4 (0/4, 3 lower) | -0.3 (0/4, 1 lower) | -1.2 (0/4, 1 lower) |

**Absolute values (%, mean over the four lengths; in brackets, lengths with the 95% CI of the gain over text only above zero)**

*MRR inside the gold document*

| | Text only (Vanilla) | Text + summary | Text + memory | Full SAADI |
| --- | --- | --- | --- | --- |
| MuSiQue, BM25 | 41.9 | 42.9 (4/4) | 45.5 (4/4) | 46.0 (4/4) |
| QASPER, BM25 | 22.8 | 23.5 (3/4) | 23.3 (1/4) | 23.8 (3/4) |
| MuSiQue, Contriever | 45.5 | 45.9 (0/4) | 49.9 (4/4) | 50.0 (4/4) |
| QASPER, Contriever | 23.7 | 24.0 (0/4) | 23.8 (0/4) | 24.2 (1/4) |

*Share of the full-document top 10 from the gold document*

| | Text only (Vanilla) | Text + summary | Text + memory | Full SAADI |
| --- | --- | --- | --- | --- |
| MuSiQue, BM25 | 66.0 | 71.0 (4/4) | 66.4 (1/4) | 70.7 (4/4) |
| QASPER, BM25 | 32.1 | 35.8 (4/4) | 35.2 (4/4) | 37.5 (4/4) |
| MuSiQue, Contriever | 50.5 | 56.6 (4/4) | 55.0 (4/4) | 58.8 (4/4) |
| QASPER, Contriever | 34.1 | 36.3 (4/4) | 36.1 (4/4) | 37.6 (4/4) |

*At least one full-document top-10 chunk from the gold document*

| | Text only (Vanilla) | Text + summary | Text + memory | Full SAADI |
| --- | --- | --- | --- | --- |
| MuSiQue, BM25 | 99.4 | 99.2 (0/4) | 99.5 (0/4) | 99.3 (0/4) |
| QASPER, BM25 | 72.7 | 72.9 (0/4) | 72.8 (0/4) | 72.9 (0/4) |
| MuSiQue, Contriever | 98.3 | 98.2 (0/4) | 98.9 (1/4) | 98.8 (0/4) |
| QASPER, Contriever | 69.9 | 68.4 (0/4) | 69.5 (0/4) | 68.6 (0/4) |

Across all pairs (gain > 0 / CI above zero / CI below zero, out of N):

| | Text + summary | Text + memory | Full SAADI |
| --- | --- | --- | --- |
| MRR inside the gold document | 16 / 7 / 0 of 16 | 13 / 9 / 0 of 16 | 15 / 12 / 0 of 16 |
| Top-10 share from the gold document | 16 / 16 / 0 of 16 | 16 / 13 / 0 of 16 | 16 / 16 / 0 of 16 |
| At least one top-10 chunk from the gold document | 5 / 0 / 3 of 16 | 11 / 1 / 1 of 16 | 9 / 0 / 2 of 16 |

Per-setting values: `channels.csv`.
