# C4 sub-document control - step 01: chunk to sub-document map and gold checks (OK)

Command (from the repository root on the server):

```bash
bash experiments/rebuttal/c4_subdoc_control/sh/01_map.sh
```

Seed 42. CPU only. Run time 1504 s.
Every chunk of every concatenated setting gets the id(s) of the source document(s) it comes from, from the `[Document i/N | doc_id=...]` headers that `context_expansion.py` writes before each source document. Two methods (text alignment and a header walk) must agree. A chunk that crosses a boundary belongs to both sub-documents. The gold sub-document of a query is qa.json's `source_document_id`.

## 1. Settings and query sets of the paper

| Setting | Retriever | Saved ranking(s) of the paper | Queries | Eligible (paper) |
| --- | --- | --- | --- | --- |
| MuSiQue-32K | bm25 | reference/main 9c6fadcc; reference/context_expansion 1b10f597 | 900 | 900 |
| MuSiQue-32K | contriever | reference/main d9967926; comparison_inputs/context_expansion afd46747 | 900 | 900 |
| MuSiQue-32K | bge_m3 | reference/main 17f74abd | 900 | 900 |
| MuSiQue-32K | jina | reference/main 2c68eb54 | 900 | 900 |
| MuSiQue-32K | qwen | reference/main c5249add | 900 | 900 |
| MuSiQue-64K | bm25 | reference/context_expansion 90bc9b3c | 900 | 900 |
| MuSiQue-64K | contriever | comparison_inputs/context_expansion 7c71b79f | 900 | 900 |
| MuSiQue-128K | bm25 | reference/context_expansion ad7c33ad | 900 | 900 |
| MuSiQue-128K | contriever | comparison_inputs/context_expansion 8513847e | 900 | 900 |
| MuSiQue-256K | bm25 | reference/context_expansion 93eabe32 | 900 | 900 |
| MuSiQue-256K | contriever | comparison_inputs/context_expansion 361938a1 | 900 | 900 |
| QASPER-32K | bm25 | reference/context_expansion 300a3066 | 1321 | 1321 |
| QASPER-32K | contriever | comparison_inputs/context_expansion ad8333c3 | 1321 | 1321 |
| QASPER-64K | bm25 | reference/main 83745de6; reference/context_expansion 4bdc8b93 | 1333 | 1333 |
| QASPER-64K | contriever | reference/main 4a14b0b0; comparison_inputs/context_expansion bd7eb2fd | 1333 | 1333 |
| QASPER-64K | bge_m3 | reference/main 71085a49 | 1333 | 1333 |
| QASPER-64K | jina | reference/main 8e3d319b | 1333 | 1333 |
| QASPER-64K | qwen | reference/main b25e0f14 | 1333 | 1333 |
| QASPER-128K | bm25 | reference/context_expansion 5cb4b256 | 1321 | 1321 |
| QASPER-128K | contriever | comparison_inputs/context_expansion e06f9c44 | 1321 | 1321 |
| QASPER-256K | bm25 | reference/context_expansion e2bb4f40 | 1321 | 1321 |
| QASPER-256K | contriever | comparison_inputs/context_expansion d50f833d | 1321 | 1321 |

No saved paper ranking (the paper does not report these pairs): MuSiQue-64K/bge_m3, MuSiQue-64K/jina, MuSiQue-64K/qwen, MuSiQue-128K/bge_m3, MuSiQue-128K/jina, MuSiQue-128K/qwen, MuSiQue-256K/bge_m3, MuSiQue-256K/jina, MuSiQue-256K/qwen, QASPER-32K/bge_m3, QASPER-32K/jina, QASPER-32K/qwen, QASPER-128K/bge_m3, QASPER-128K/jina, QASPER-128K/qwen, QASPER-256K/bge_m3, QASPER-256K/jina, QASPER-256K/qwen.

## 2. Sub-documents and chunks

| Setting | Indexed documents | Sub-documents (distinct) | Sub-docs per document min/med/max | Chunks | Boundary chunks | Docs starting mid-sub-doc | Docs ending mid-sub-doc | Chunks not found in text | A vs B disagreements | Header split across chunks | Min text coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MuSiQue-32K | 45 | 900 | 18 / 20 / 22 | 17102 | 617 (3.6%) | 0 | 0 | 0 | 0 | 0 | 1.000 |
| MuSiQue-64K | 23 | 900 | 36 / 39 / 41 | 17089 | 635 (3.7%) | 0 | 0 | 0 | 0 | 0 | 1.000 |
| MuSiQue-128K | 12 | 900 | 44 / 78 / 80 | 17091 | 638 (3.7%) | 0 | 0 | 0 | 0 | 0 | 1.000 |
| MuSiQue-256K | 6 | 900 | 120 / 155.5 / 158 | 17088 | 643 (3.8%) | 0 | 0 | 0 | 0 | 0 | 1.000 |
| QASPER-32K | 46 | 416 | 2 / 10 / 13 | 17474 | 264 (1.5%) | 4 | 0 | 0 | 0 | 0 | 1.000 |
| QASPER-64K | 23 | 416 | 17 / 20 / 23 | 17159 | 276 (1.6%) | 2 | 0 | 0 | 0 | 0 | 1.000 |
| QASPER-128K | 11 | 416 | 32 / 38 / 42 | 15886 | 261 (1.6%) | 0 | 0 | 0 | 0 | 0 | 1.000 |
| QASPER-256K | 6 | 416 | 35 / 74 / 83 | 15886 | 262 (1.6%) | 0 | 0 | 0 | 0 | 0 | 1.000 |

Boundary chunks belong to two sub-documents. "Docs starting / ending mid-sub-doc" are trimmed variants; their partial sub-document is still named from group_membership.json. Text coverage = characters of the document text matched by its chunks (all whitespace removed).

## 3. Gold sub-document checks

| Setting | Queries | Paper labels reproduced by the repo labeler | Ineligible | Evidence outside gold sub-doc | ... only duplicate copies | ... only partial-match groups | ... both | Gold sub-doc not in text | Set S (strict: all evidence inside) | Set R (labels re-derived inside) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MuSiQue-32K | 900 | 900/900 | 0 | 325 | 28 | 234 | 63 | 0 | 575 | 900 |
| MuSiQue-64K | 900 | 900/900 | 0 | 515 | 52 | 336 | 127 | 0 | 385 | 900 |
| MuSiQue-128K | 900 | 900/900 | 0 | 650 | 74 | 354 | 222 | 0 | 250 | 900 |
| MuSiQue-256K | 900 | 900/900 | 0 | 725 | 108 | 318 | 299 | 0 | 175 | 900 |
| QASPER-32K | 1321 | 1321/1321 | 0 | 342 | 28 | 307 | 7 | 0 | 979 | 1321 |
| QASPER-64K | 1333 | 1333/1333 | 0 | 539 | 38 | 490 | 11 | 0 | 794 | 1322 |
| QASPER-128K | 1321 | 1321/1321 | 0 | 661 | 29 | 614 | 18 | 0 | 660 | 1321 |
| QASPER-256K | 1321 | 1321/1321 | 0 | 816 | 25 | 766 | 25 | 0 | 505 | 1321 |

Why evidence lies outside the gold sub-document. The paper's labels come from `indexing/labeling.py`, run over the whole concatenated document: (a) every chunk anywhere that contains a span counts as a sufficient chunk, so a paragraph repeated in another source document (common in MuSiQue) adds a copy outside; (b) a span that fits in no single chunk becomes an evidence group of every chunk that shares a boundary token run with it, anywhere in the document, and a hit needs the whole group in the top k. Set S follows the task as written (such queries left out). Set R keeps every query whose spans also match inside the gold sub-document, with labels from the same labeler run on the gold sub-document's chunks only. The full-ranking measures always use the paper's labels.

Examples (first 2 per setting):

| Setting | Query | Document | Gold sub-doc | Evidence chunk outside -> its sub-doc(s) |
| --- | --- | --- | --- | --- |
| MuSiQue-32K | 6 | group_044 | 2hop__207476_625987 | group_044:271 -> 2hop__609313_20273 |
| MuSiQue-32K | 8 | group_027 | 2hop__354635_174222 | group_027:95 -> 3hop1__144439_443779_52195; group_027:100 -> 3hop1__144439_443779_52195,4hop3__426860_88460_30152_20999 |
| MuSiQue-64K | 2 | group_015 | 2hop__701895_752697 | group_015:144 -> 4hop2__161602_426860_88460_20985 |
| MuSiQue-64K | 5 | group_007 | 2hop__697790_864352 | group_007:379 -> 2hop__197470_271394; group_007:383 -> 2hop__197470_271394 |
| MuSiQue-128K | 2 | group_008 | 2hop__701895_752697 | group_008:64 -> 3hop2__132957_433432_40768,2hop__657913_88628; group_008:209 -> 4hop2__161602_426860_88460_20985; group_008:1371 -> 2hop__96062_159673 |
| MuSiQue-128K | 5 | group_004 | 2hop__697790_864352 | group_004:379 -> 2hop__197470_271394; group_004:383 -> 2hop__197470_271394 |
| MuSiQue-256K | 2 | group_004 | 2hop__701895_752697 | group_004:1226 -> 4hop2__161602_474028_88460_20985; group_004:1533 -> 3hop2__132957_433432_40768,2hop__657913_88628; group_004:1678 -> 4hop2__161602_426860_88460_20985 |
| MuSiQue-256K | 5 | group_002 | 2hop__697790_864352 | group_002:1869 -> 2hop__197470_271394; group_002:1873 -> 2hop__197470_271394 |
| QASPER-32K | 1 | group_026 | 1911.10742 | group_026:280 -> 1912.10162; group_026:363 -> 1709.05700; group_026:184 -> 1911.10401 |
| QASPER-32K | 2 | group_026 | 1911.10742 | group_026:169 -> 1803.04579; group_026:304 -> 1912.10162 |
| QASPER-64K | 2 | group_013 | 1911.10742 | group_013:115 -> 1610.06510 |
| QASPER-64K | 3 | group_013 | 1911.10742 | group_013:402 -> 1910.10487 |
| QASPER-128K | 1 | group_007 | 1911.10742 | group_007:1126 -> 1912.10162; group_007:1209 -> 1709.05700; group_007:1030 -> 1911.10401 |
| QASPER-128K | 2 | group_007 | 1911.10742 | group_007:383 -> 1610.06510; group_007:1015 -> 1803.04579; group_007:1150 -> 1912.10162 |
| QASPER-256K | 1 | group_004 | 1911.10742 | group_004:1179 -> 1912.10162; group_004:1262 -> 1709.05700; group_004:1083 -> 1911.10401 |
| QASPER-256K | 2 | group_004 | 1911.10742 | group_004:436 -> 1610.06510; group_004:1068 -> 1803.04579; group_004:1203 -> 1912.10162 |


## 4. Chunks per gold sub-document

| Setting | Set | Gold sub-docs (distinct) | Chunks per gold sub-doc min/med/max | Same, over queries | All sub-docs min/med/max | Queries with <= 1 chunks | Queries with <= 3 chunks | Queries with <= 5 chunks | Queries with <= 10 chunks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MuSiQue-32K | S | 575 | 8 / 19 / 33 | 8 / 19 / 33 | 8 / 19 / 37 | 0 (0%) | 0 (0%) | 0 (0%) | 6 (1%) |
| MuSiQue-32K | R | 900 | 8 / 19 / 37 | 8 / 19 / 37 | 8 / 19 / 37 | 0 (0%) | 0 (0%) | 0 (0%) | 6 (1%) |
| MuSiQue-64K | S | 385 | 8 / 19 / 32 | 8 / 19 / 32 | 8 / 19 / 37 | 0 (0%) | 0 (0%) | 0 (0%) | 5 (1%) |
| MuSiQue-64K | R | 900 | 8 / 19 / 37 | 8 / 19 / 37 | 8 / 19 / 37 | 0 (0%) | 0 (0%) | 0 (0%) | 6 (1%) |
| MuSiQue-128K | S | 250 | 8 / 18 / 30 | 8 / 18 / 30 | 8 / 19 / 37 | 0 (0%) | 0 (0%) | 0 (0%) | 5 (2%) |
| MuSiQue-128K | R | 900 | 8 / 19 / 37 | 8 / 19 / 37 | 8 / 19 / 37 | 0 (0%) | 0 (0%) | 0 (0%) | 6 (1%) |
| MuSiQue-256K | S | 175 | 8 / 17 / 30 | 8 / 17 / 30 | 8 / 19 / 37 | 0 (0%) | 0 (0%) | 0 (0%) | 5 (3%) |
| MuSiQue-256K | R | 900 | 8 / 19 / 37 | 8 / 19 / 37 | 8 / 19 / 37 | 0 (0%) | 0 (0%) | 0 (0%) | 6 (1%) |
| QASPER-32K | S | 381 | 8 / 37 / 242 | 8 / 38 / 242 | 8 / 36 / 242 | 0 (0%) | 0 (0%) | 0 (0%) | 7 (1%) |
| QASPER-32K | R | 408 | 8 / 37 / 242 | 8 / 38 / 242 | 8 / 36 / 242 | 0 (0%) | 0 (0%) | 0 (0%) | 7 (1%) |
| QASPER-64K | S | 353 | 8 / 37 / 242 | 8 / 37 / 242 | 8 / 36 / 242 | 0 (0%) | 0 (0%) | 0 (0%) | 5 (1%) |
| QASPER-64K | R | 408 | 8 / 37 / 242 | 8 / 38 / 242 | 8 / 36 / 242 | 0 (0%) | 0 (0%) | 0 (0%) | 7 (1%) |
| QASPER-128K | S | 325 | 7 / 37 / 242 | 7 / 37 / 242 | 7 / 37 / 242 | 0 (0%) | 0 (0%) | 0 (0%) | 5 (1%) |
| QASPER-128K | R | 408 | 7 / 37 / 242 | 7 / 38 / 242 | 7 / 37 / 242 | 0 (0%) | 0 (0%) | 0 (0%) | 7 (1%) |
| QASPER-256K | S | 281 | 8 / 36 / 242 | 8 / 37 / 242 | 8 / 37 / 242 | 0 (0%) | 0 (0%) | 0 (0%) | 5 (1%) |
| QASPER-256K | R | 408 | 8 / 37 / 242 | 8 / 38 / 242 | 8 / 37 / 242 | 0 (0%) | 0 (0%) | 0 (0%) | 7 (1%) |

HR@k inside the sub-document is trivially 100% for queries whose gold sub-document has k chunks or fewer; later steps report each HR@k on all queries of a set and on those with more than k chunks.

## Files written

- `experiments/rebuttal/c4_subdoc_control/reports/step_01.json`
- `experiments/rebuttal/c4_subdoc_control/reports/step_01.md`
- `results/rebuttal/c4_subdoc_control/subdoc_map/musique_128k.json`
- `results/rebuttal/c4_subdoc_control/subdoc_map/musique_256k.json`
- `results/rebuttal/c4_subdoc_control/subdoc_map/musique_32k.json`
- `results/rebuttal/c4_subdoc_control/subdoc_map/musique_64k.json`
- `results/rebuttal/c4_subdoc_control/subdoc_map/qasper_128k.json`
- `results/rebuttal/c4_subdoc_control/subdoc_map/qasper_256k.json`
- `results/rebuttal/c4_subdoc_control/subdoc_map/qasper_32k.json`
- `results/rebuttal/c4_subdoc_control/subdoc_map/qasper_64k.json`
