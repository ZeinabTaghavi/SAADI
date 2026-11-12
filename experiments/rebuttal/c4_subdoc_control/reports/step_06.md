# C4 sub-document control - step 06: channel breakdown inside the gold sub-document (OK)

Command (from the repository root on the server):

```bash
bash experiments/rebuttal/c4_subdoc_control/sh/06_channels.sh
```

Seed 42. CPU only. Run time 78 s.
Pairs: 16. Set R, full-document scores kept (the main C4 analysis). Channel weights: lambda_M=0.5, lambda_S=0.3 (published) or 0.
Check: the text-only and full rankings equal those of steps 03-04 for every pair (else a problem is listed).

## Gain over text alone, mean over the four lengths (lengths with CI above zero)

**MRR inside the gold document**

|  | Text + summary | Text + memory | Full SAADI |
| --- | --- | --- | --- |
| MuSiQue, BM25 | +1.0 (4/4) | +3.5 (4/4) | +4.1 (4/4) |
| QASPER, BM25 | +0.6 (3/4) | +0.4 (1/4) | +1.0 (3/4) |
| MuSiQue, Contriever | +0.4 (0/4) | +4.4 (4/4) | +4.5 (4/4) |
| QASPER, Contriever | +0.3 (0/4) | +0.0 (0/4) | +0.4 (1/4) |

**HR@5 inside the gold document**

|  | Text + summary | Text + memory | Full SAADI |
| --- | --- | --- | --- |
| MuSiQue, BM25 | +0.9 (0/4) | +2.5 (4/4) | +3.5 (4/4) |
| QASPER, BM25 | +1.4 (2/4) | +0.4 (0/4) | +1.1 (0/4) |
| MuSiQue, Contriever | +1.0 (0/4) | +4.2 (4/4) | +5.8 (4/4) |
| QASPER, Contriever | +0.1 (0/4) | -0.3 (0/4) | -0.0 (0/4) |

**Share of the full-document top 10 from the gold document**

|  | Text + summary | Text + memory | Full SAADI |
| --- | --- | --- | --- |
| MuSiQue, BM25 | +5.0 (4/4) | +0.4 (1/4) | +4.7 (4/4) |
| QASPER, BM25 | +3.6 (4/4) | +3.1 (4/4) | +5.4 (4/4) |
| MuSiQue, Contriever | +6.1 (4/4) | +4.5 (4/4) | +8.3 (4/4) |
| QASPER, Contriever | +2.2 (4/4) | +2.0 (4/4) | +3.5 (4/4) |

**At least one full-document top-10 chunk from the gold document**

|  | Text + summary | Text + memory | Full SAADI |
| --- | --- | --- | --- |
| MuSiQue, BM25 | -0.2 (0/4) | +0.1 (0/4) | -0.1 (0/4, 1 lower) |
| QASPER, BM25 | +0.3 (0/4) | +0.1 (0/4) | +0.3 (0/4) |
| MuSiQue, Contriever | -0.1 (0/4) | +0.7 (1/4) | +0.6 (0/4) |
| QASPER, Contriever | -1.4 (0/4, 3 lower) | -0.3 (0/4, 1 lower) | -1.2 (0/4, 1 lower) |


## Per setting: MRR inside the gold document (text alone, and gain of each variant)

| Setting | Retriever | Text only | + summary | + memory | Full |
| --- | --- | --- | --- | --- | --- |
| MuSiQue-32K | BM25 | 40.9 | +1.2* | +3.7* | +4.4* |
| MuSiQue-64K | BM25 | 41.6 | +1.1* | +3.5* | +4.1* |
| MuSiQue-128K | BM25 | 42.4 | +0.9* | +3.4* | +3.6* |
| MuSiQue-256K | BM25 | 42.9 | +0.8* | +3.6* | +4.1* |
| QASPER-32K | BM25 | 22.5 | +0.5 | +0.5 | +1.0* |
| QASPER-64K | BM25 | 22.7 | +0.6* | +0.9* | +1.3* |
| QASPER-128K | BM25 | 23.0 | +1.0* | -0.1 | +0.6 |
| QASPER-256K | BM25 | 23.1 | +0.5* | +0.4 | +1.1* |
| MuSiQue-32K | Contriever | 45.6 | +0.5 | +4.1* | +4.5* |
| MuSiQue-64K | Contriever | 45.4 | +0.2 | +4.4* | +4.8* |
| MuSiQue-128K | Contriever | 45.6 | +0.3 | +3.9* | +3.9* |
| MuSiQue-256K | Contriever | 45.4 | +0.4 | +5.2* | +4.8* |
| QASPER-32K | Contriever | 23.7 | +0.2 | -0.5 | -0.2 |
| QASPER-64K | Contriever | 23.8 | +0.4 | +0.3 | +0.6 |
| QASPER-128K | Contriever | 23.8 | +0.0 | -0.3 | +0.2 |
| QASPER-256K | Contriever | 23.7 | +0.6 | +0.5 | +1.0* |

* = 95% CI above zero.

## Files written

- `experiments/rebuttal/c4_subdoc_control/reports/step_06.json`
- `experiments/rebuttal/c4_subdoc_control/reports/step_06.md`
- `experiments/rebuttal/c4_subdoc_control/work/channels/musique_128k__bm25.csv.gz`
- `experiments/rebuttal/c4_subdoc_control/work/channels/musique_128k__contriever.csv.gz`
- `experiments/rebuttal/c4_subdoc_control/work/channels/musique_256k__bm25.csv.gz`
- `experiments/rebuttal/c4_subdoc_control/work/channels/musique_256k__contriever.csv.gz`
- `experiments/rebuttal/c4_subdoc_control/work/channels/musique_32k__bm25.csv.gz`
- `experiments/rebuttal/c4_subdoc_control/work/channels/musique_32k__contriever.csv.gz`
- `experiments/rebuttal/c4_subdoc_control/work/channels/musique_64k__bm25.csv.gz`
- `experiments/rebuttal/c4_subdoc_control/work/channels/musique_64k__contriever.csv.gz`
- `experiments/rebuttal/c4_subdoc_control/work/channels/qasper_128k__bm25.csv.gz`
- `experiments/rebuttal/c4_subdoc_control/work/channels/qasper_128k__contriever.csv.gz`
- `experiments/rebuttal/c4_subdoc_control/work/channels/qasper_256k__bm25.csv.gz`
- `experiments/rebuttal/c4_subdoc_control/work/channels/qasper_256k__contriever.csv.gz`
- `experiments/rebuttal/c4_subdoc_control/work/channels/qasper_32k__bm25.csv.gz`
- `experiments/rebuttal/c4_subdoc_control/work/channels/qasper_32k__contriever.csv.gz`
- `experiments/rebuttal/c4_subdoc_control/work/channels/qasper_64k__bm25.csv.gz`
- `experiments/rebuttal/c4_subdoc_control/work/channels/qasper_64k__contriever.csv.gz`
- `results/rebuttal/c4_subdoc_control/CHANNELS.md`
- `results/rebuttal/c4_subdoc_control/channels.csv`
