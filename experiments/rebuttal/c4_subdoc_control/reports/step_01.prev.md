# C4 sub-document control - step 01: chunk to sub-document map and gold checks (FAILED)

Command (from the repository root on the server):

```bash
bash experiments/rebuttal/c4_subdoc_control/sh/01_map.sh
```

Seed 42. CPU only. Run time 17 s.
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
| MuSiQue-32K | 45 | 900 | 18 / 20 / 22 | 17102 | 618 (3.6%) | 0 | 0 | 3398 | 13 | 1 | 0.724 |
| MuSiQue-64K | 23 | 900 | 36 / 39 / 41 | 17089 | 636 (3.7%) | 0 | 0 | 3405 | 13 | 1 | 0.739 |
| MuSiQue-128K | 12 | 900 | 44 / 78 / 80 | 17091 | 639 (3.7%) | 0 | 0 | 3400 | 13 | 1 | 0.772 |
| MuSiQue-256K | 6 | 900 | 120 / 155.5 / 158 | 17088 | 644 (3.8%) | 0 | 0 | 3399 | 13 | 1 | 0.775 |
| QASPER-32K | 46 | 416 | 2 / 10 / 13 | 17474 | 367 (2.1%) | 4 | 0 | 5141 | 12264 | 46 | 0.549 |
| QASPER-64K | 23 | 416 | 17 / 20 / 23 | 17159 | 384 (2.2%) | 2 | 0 | 5184 | 11923 | 23 | 0.590 |
| QASPER-128K | 11 | 416 | 32 / 38 / 42 | 15886 | 362 (2.3%) | 0 | 0 | 4704 | 11182 | 11 | 0.656 |
| QASPER-256K | 6 | 416 | 35 / 74 / 83 | 15886 | 367 (2.3%) | 0 | 0 | 4698 | 11188 | 6 | 0.677 |

Boundary chunks belong to two sub-documents. "Docs starting / ending mid-sub-doc" are trimmed variants; their partial sub-document is still named from group_membership.json. Text coverage = characters of the document text matched by its chunks (whitespace-normalised).

## 3. Gold sub-document checks

| Setting | Queries (union over retrievers) | Ineligible (no evidence) | Excluded: evidence outside gold sub-doc | Excluded: gold sub-doc not in text | Kept for C4 | Evidence shares another sub-doc, not the gold one |
| --- | --- | --- | --- | --- | --- | --- |
| MuSiQue-32K | 900 | 0 | 326 | 0 | 574 | 0 |
| MuSiQue-64K | 900 | 0 | 515 | 0 | 385 | 0 |
| MuSiQue-128K | 900 | 0 | 650 | 0 | 250 | 0 |
| MuSiQue-256K | 900 | 0 | 725 | 0 | 175 | 0 |
| QASPER-32K | 1321 | 0 | 869 | 0 | 452 | 100 |
| QASPER-64K | 1333 | 0 | 958 | 0 | 375 | 95 |
| QASPER-128K | 1321 | 0 | 990 | 0 | 331 | 83 |
| QASPER-256K | 1321 | 0 | 1050 | 0 | 271 | 69 |

First excluded queries per setting (at most 5 each):

| Setting | Query | Document | Gold sub-doc | Status | Evidence chunk -> its sub-doc(s) |
| --- | --- | --- | --- | --- | --- |
| MuSiQue-32K | 6 | group_044 | 2hop__207476_625987 | excluded_evidence_outside | group_044:271 -> 2hop__609313_20273 |
| MuSiQue-32K | 8 | group_027 | 2hop__354635_174222 | excluded_evidence_outside | group_027:95 -> 3hop1__144439_443779_52195; group_027:100 -> 3hop1__144439_443779_52195,4hop3__426860_88460_30152_20999 |
| MuSiQue-32K | 29 | group_008 | 2hop__403712_793210 | excluded_evidence_outside | group_008:244 -> 2hop__86458_20273 |
| MuSiQue-32K | 72 | group_033 | 2hop__471509_136043 | excluded_evidence_outside | group_033:188 -> 2hop__373759_136043 |
| MuSiQue-32K | 77 | group_031 | 2hop__42027_42004 | excluded_evidence_outside | group_031:256 -> 4hop1__105401_17130_70784_61381; group_031:343 -> 2hop__96062_159673 |
| MuSiQue-64K | 2 | group_015 | 2hop__701895_752697 | excluded_evidence_outside | group_015:144 -> 4hop2__161602_426860_88460_20985 |
| MuSiQue-64K | 5 | group_007 | 2hop__697790_864352 | excluded_evidence_outside | group_007:379 -> 2hop__197470_271394; group_007:383 -> 2hop__197470_271394 |
| MuSiQue-64K | 6 | group_023 | 2hop__207476_625987 | excluded_evidence_outside | group_023:252 -> 2hop__609313_20273 |
| MuSiQue-64K | 8 | group_014 | 2hop__354635_174222 | excluded_evidence_outside | group_014:265 -> 3hop1__144439_443779_52195; group_014:270 -> 3hop1__144439_443779_52195,4hop3__426860_88460_30152_20999 |
| MuSiQue-64K | 27 | group_023 | 2hop__673655_316668 | excluded_evidence_outside | group_023:449 -> 2hop__728887_159109 |
| MuSiQue-128K | 2 | group_008 | 2hop__701895_752697 | excluded_evidence_outside | group_008:64 -> 3hop2__132957_433432_40768,2hop__657913_88628; group_008:209 -> 4hop2__161602_426860_88460_20985; group_008:1371 -> 2hop__96062_159673 |
| MuSiQue-128K | 5 | group_004 | 2hop__697790_864352 | excluded_evidence_outside | group_004:379 -> 2hop__197470_271394; group_004:383 -> 2hop__197470_271394 |
| MuSiQue-128K | 6 | group_012 | 2hop__207476_625987 | excluded_evidence_outside | group_012:353 -> 2hop__609313_20273 |
| MuSiQue-128K | 8 | group_007 | 2hop__354635_174222 | excluded_evidence_outside | group_007:1055 -> 3hop1__144439_443779_52195; group_007:1060 -> 3hop1__144439_443779_52195,4hop3__426860_88460_30152_20999 |
| MuSiQue-128K | 27 | group_012 | 2hop__673655_316668 | excluded_evidence_outside | group_012:550 -> 2hop__728887_159109 |
| MuSiQue-256K | 2 | group_004 | 2hop__701895_752697 | excluded_evidence_outside | group_004:1226 -> 4hop2__161602_474028_88460_20985; group_004:1533 -> 3hop2__132957_433432_40768,2hop__657913_88628; group_004:1678 -> 4hop2__161602_426860_88460_20985 |
| MuSiQue-256K | 5 | group_002 | 2hop__697790_864352 | excluded_evidence_outside | group_002:1869 -> 2hop__197470_271394; group_002:1873 -> 2hop__197470_271394 |
| MuSiQue-256K | 6 | group_006 | 2hop__207476_625987 | excluded_evidence_outside | group_006:1866 -> 2hop__609313_20273 |
| MuSiQue-256K | 8 | group_004 | 2hop__354635_174222 | excluded_evidence_outside | group_004:1055 -> 3hop1__144439_443779_52195; group_004:1060 -> 3hop1__144439_443779_52195,4hop3__426860_88460_30152_20999; group_004:2693 -> 4hop3__638064_88460_30152_20999 |
| MuSiQue-256K | 12 | group_003 | 2hop__362107_652514 | excluded_evidence_outside | group_003:177 -> 2hop__747894_135844; group_003:627 -> 4hop1__89632_32392_823060_610794,4hop3__275416_24325_125104_10557 |


## 4. Chunks per gold sub-document (kept queries)

| Setting | Gold sub-docs (distinct) | Chunks per gold sub-doc min/med/max | Same, over queries | All sub-docs min/med/max | Queries with <= 1 chunks | Queries with <= 3 chunks | Queries with <= 5 chunks | Queries with <= 10 chunks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MuSiQue-32K | 574 | 8 / 19 / 33 | 8 / 19 / 33 | 8 / 19 / 37 | 0 (0%) | 0 (0%) | 0 (0%) | 6 (1%) |
| MuSiQue-64K | 385 | 8 / 19 / 32 | 8 / 19 / 32 | 8 / 19 / 37 | 0 (0%) | 0 (0%) | 0 (0%) | 5 (1%) |
| MuSiQue-128K | 250 | 8 / 18 / 30 | 8 / 18 / 30 | 8 / 19 / 37 | 0 (0%) | 0 (0%) | 0 (0%) | 5 (2%) |
| MuSiQue-256K | 175 | 8 / 17 / 30 | 8 / 17 / 30 | 8 / 19 / 37 | 0 (0%) | 0 (0%) | 0 (0%) | 5 (3%) |
| QASPER-32K | 243 | 5 / 27 / 133 | 5 / 28 / 133 | 2 / 20 / 133 | 0 (0%) | 0 (0%) | 5 (1%) | 12 (3%) |
| QASPER-64K | 224 | 5 / 25.5 / 133 | 5 / 27 / 133 | 2 / 20 / 133 | 0 (0%) | 0 (0%) | 5 (1%) | 10 (3%) |
| QASPER-128K | 206 | 4 / 26 / 132 | 4 / 27 / 132 | 2 / 20 / 142 | 0 (0%) | 0 (0%) | 4 (1%) | 7 (2%) |
| QASPER-256K | 179 | 5 / 25 / 133 | 5 / 27 / 133 | 2 / 20 / 156 | 0 (0%) | 0 (0%) | 4 (1%) | 7 (3%) |

HR@k inside the sub-document is trivially 100% for queries whose gold sub-document has k chunks or fewer; later steps report each HR@k on all kept queries and on those with more than k chunks.

## 5. Method disagreements (first 5)

```
MuSiQue-32K: {'chunk_id': 'group_012:197', 'alignment': ['2hop__79875_84469', '3hop1__458916_651917_47686'], 'header_walk': ['2hop__79875_84469']}
MuSiQue-32K: {'chunk_id': 'group_012:199', 'alignment': ['3hop1__458916_651917_47686'], 'header_walk': ['2hop__79875_84469']}
MuSiQue-32K: {'chunk_id': 'group_012:200', 'alignment': ['3hop1__458916_651917_47686'], 'header_walk': ['2hop__79875_84469']}
MuSiQue-32K: {'chunk_id': 'group_012:201', 'alignment': ['3hop1__458916_651917_47686'], 'header_walk': ['2hop__79875_84469']}
MuSiQue-32K: {'chunk_id': 'group_012:203', 'alignment': ['3hop1__458916_651917_47686'], 'header_walk': ['2hop__79875_84469']}
```

## Problems (the step failed)

- musique_32k: group_012: 67 chunk(s) not found in the text and the header walk is not reliable
- musique_64k: group_006: 133 chunk(s) not found in the text and the header walk is not reliable
- musique_128k: group_003: 280 chunk(s) not found in the text and the header walk is not reliable
- musique_256k: group_002: 559 chunk(s) not found in the text and the header walk is not reliable
- qasper_32k: group_001: 107 chunk(s) not found in the text and the header walk is not reliable
- qasper_32k: group_002: 107 chunk(s) not found in the text and the header walk is not reliable
- qasper_32k: group_003: 118 chunk(s) not found in the text and the header walk is not reliable
- qasper_32k: group_004: 108 chunk(s) not found in the text and the header walk is not reliable
- qasper_32k: group_005: 136 chunk(s) not found in the text and the header walk is not reliable
- qasper_32k: group_006: 94 chunk(s) not found in the text and the header walk is not reliable
- qasper_32k: group_007: 104 chunk(s) not found in the text and the header walk is not reliable
- qasper_32k: group_008: 92 chunk(s) not found in the text and the header walk is not reliable
- qasper_32k: group_009__variant_59d548cab7: 124 chunk(s) not found in the text and the header walk is not reliable
- qasper_32k: group_009__variant_6ee7556429: 130 chunk(s) not found in the text and the header walk is not reliable
- qasper_32k: group_010: 124 chunk(s) not found in the text and the header walk is not reliable
- qasper_32k: group_011: 112 chunk(s) not found in the text and the header walk is not reliable
- qasper_32k: group_012: 148 chunk(s) not found in the text and the header walk is not reliable
- qasper_32k: group_013: 109 chunk(s) not found in the text and the header walk is not reliable
- qasper_32k: group_014: 119 chunk(s) not found in the text and the header walk is not reliable
- qasper_32k: group_015: 133 chunk(s) not found in the text and the header walk is not reliable
- qasper_32k: group_016: 125 chunk(s) not found in the text and the header walk is not reliable
- qasper_32k: group_017: 113 chunk(s) not found in the text and the header walk is not reliable
- qasper_32k: group_018: 112 chunk(s) not found in the text and the header walk is not reliable
- qasper_32k: group_019: 168 chunk(s) not found in the text and the header walk is not reliable
- qasper_32k: group_020__variant_06ecc81089: 125 chunk(s) not found in the text and the header walk is not reliable
- qasper_32k: group_020__variant_3361b8013d: 127 chunk(s) not found in the text and the header walk is not reliable
- qasper_32k: group_021: 87 chunk(s) not found in the text and the header walk is not reliable
- qasper_32k: group_022: 145 chunk(s) not found in the text and the header walk is not reliable
- qasper_32k: group_023: 102 chunk(s) not found in the text and the header walk is not reliable
- qasper_32k: group_024: 104 chunk(s) not found in the text and the header walk is not reliable
- ... 60 more (see experiments/rebuttal/c4_subdoc_control/reports/step_01.json)

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
