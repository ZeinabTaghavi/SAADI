# C5 reader limit, MuSiQue-32K

Labels and SAADI top 10: `/Users/uem-stag-lap-ic/Desktop/SAADI/experiments/rebuttal/c4_subdoc_control/work/streams/musique_32k__bm25.queries.json`. Answers: `results/rebuttal/c5_end_to_end/answers/musique_32k/saadi.jsonl`, `results/rebuttal/c5_end_to_end/answers/musique_32k/oracle.jsonl`.
Labels file sha256 34c916348a0a8f600d871a2bae7dbdcdc1d04f149172b999168d37987ae36608. Self-check: any-gold flags equal the answer files' evidence_in_context for all 900 questions. Questions without gold chunks (silver groups only): 212.

| Subset | n | SAADI F1 | Oracle F1, same questions |
| --- | --- | --- | --- |
| all questions | 900 | 22.8 | 39.7 |
| any gold in SAADI's top 10 (paper hit rule) | 699 | 24.5 | 40.6 |
| all gold chunks in SAADI's top 10 | 359 | 30.0 | 45.5 |
| gold chunks, but not all in SAADI's top 10 | 329 | 16.2 | 38.4 |
| no gold in SAADI's top 10 (paper hit rule fails) | 201 | 16.9 | 36.9 |
