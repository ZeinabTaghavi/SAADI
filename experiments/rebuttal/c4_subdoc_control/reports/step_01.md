# C4 sub-document control - step 01: chunk to sub-document map and gold checks (FAILED)

Command (from the repository root on the server):

```bash
bash experiments/rebuttal/c4_subdoc_control/sh/01_map.sh
```

Seed 42. CPU only. Run time 5 s.
Every chunk of every concatenated setting gets the id(s) of the source document(s) it comes from, from the `[Document i/N | doc_id=...]` headers that `context_expansion.py` writes before each source document. Two methods (text alignment and a header walk) must agree. A chunk that crosses a boundary belongs to both sub-documents. The gold sub-document of a query is qa.json's `source_document_id`.

## 1. Settings and query sets of the paper

| Setting | Retriever | Saved ranking(s) of the paper | Queries | Eligible (paper) |
| --- | --- | --- | --- | --- |
| MuSiQue-32K | bm25 | reference/main 9c6fadcc; reference/context_expansion 1b10f597 | 900 | 900 |
| MuSiQue-32K | contriever | reference/main d9967926; comparison_inputs/context_expansion afd46747 | 900 | 900 |
| MuSiQue-32K | bge_m3 | reference/main 17f74abd | 900 | 900 |
| MuSiQue-32K | jina | reference/main 2c68eb54 | 900 | 900 |
| MuSiQue-32K | qwen | reference/main c5249add | 900 | 900 |
| QASPER-64K | bm25 | reference/main 83745de6; reference/context_expansion 4bdc8b93 | 1333 | 1333 |
| QASPER-64K | contriever | reference/main 4a14b0b0; comparison_inputs/context_expansion bd7eb2fd | 1333 | 1333 |
| QASPER-64K | bge_m3 | reference/main 71085a49 | 1333 | 1333 |
| QASPER-64K | jina | reference/main 8e3d319b | 1333 | 1333 |
| QASPER-64K | qwen | reference/main b25e0f14 | 1333 | 1333 |

No saved paper ranking (the paper does not report these pairs): MuSiQue-64K/bm25, MuSiQue-64K/contriever, MuSiQue-64K/bge_m3, MuSiQue-64K/jina, MuSiQue-64K/qwen, MuSiQue-128K/bm25, MuSiQue-128K/contriever, MuSiQue-128K/bge_m3, MuSiQue-128K/jina, MuSiQue-128K/qwen, MuSiQue-256K/bm25, MuSiQue-256K/contriever, MuSiQue-256K/bge_m3, MuSiQue-256K/jina, MuSiQue-256K/qwen, QASPER-32K/bm25, QASPER-32K/contriever, QASPER-32K/bge_m3, QASPER-32K/jina, QASPER-32K/qwen, QASPER-128K/bm25, QASPER-128K/contriever, QASPER-128K/bge_m3, QASPER-128K/jina, QASPER-128K/qwen, QASPER-256K/bm25, QASPER-256K/contriever, QASPER-256K/bge_m3, QASPER-256K/jina, QASPER-256K/qwen.

## 2. Sub-documents and chunks

| Setting | Indexed documents | Sub-documents (distinct) | Sub-docs per document min/med/max | Chunks | Boundary chunks | Docs starting mid-sub-doc | Docs ending mid-sub-doc | Chunks not found in text | A vs B disagreements | Header split across chunks | Min text coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MuSiQue-32K | 45 | 900 | 18 / 20 / 22 | 17102 | 618 (3.6%) | 0 | 0 | 3398 | 13 | 1 | 0.724 |
| QASPER-64K | 23 | 416 | 17 / 20 / 23 | 17159 | 384 (2.2%) | 2 | 0 | 5184 | 11923 | 23 | 0.590 |

Boundary chunks belong to two sub-documents. "Docs starting / ending mid-sub-doc" are trimmed variants; their partial sub-document is still named from group_membership.json. Text coverage = characters of the document text matched by its chunks (whitespace-normalised).

## 3. Gold sub-document checks

| Setting | Queries (union over retrievers) | Ineligible (no evidence) | Excluded: evidence outside gold sub-doc | Excluded: gold sub-doc not in text | Kept for C4 | Evidence shares another sub-doc, not the gold one |
| --- | --- | --- | --- | --- | --- | --- |
| MuSiQue-32K | 900 | 0 | 326 | 0 | 574 | 0 |
| QASPER-64K | 1333 | 0 | 958 | 0 | 375 | 95 |

First excluded queries per setting (at most 5 each):

| Setting | Query | Document | Gold sub-doc | Status | Evidence chunk -> its sub-doc(s) |
| --- | --- | --- | --- | --- | --- |
| MuSiQue-32K | 6 | group_044 | 2hop__207476_625987 | excluded_evidence_outside | group_044:271 -> 2hop__609313_20273 |
| MuSiQue-32K | 8 | group_027 | 2hop__354635_174222 | excluded_evidence_outside | group_027:95 -> 3hop1__144439_443779_52195; group_027:100 -> 3hop1__144439_443779_52195,4hop3__426860_88460_30152_20999 |
| MuSiQue-32K | 29 | group_008 | 2hop__403712_793210 | excluded_evidence_outside | group_008:244 -> 2hop__86458_20273 |
| MuSiQue-32K | 72 | group_033 | 2hop__471509_136043 | excluded_evidence_outside | group_033:188 -> 2hop__373759_136043 |
| MuSiQue-32K | 77 | group_031 | 2hop__42027_42004 | excluded_evidence_outside | group_031:256 -> 4hop1__105401_17130_70784_61381; group_031:343 -> 2hop__96062_159673 |
| QASPER-64K | 0 | group_013 | 1911.10742 | excluded_evidence_outside | group_013:598 -> 2004. 02105; group_013:633 -> 2004. 02105 |
| QASPER-64K | 1 | group_013 | 1911.10742 | excluded_evidence_outside | group_013:598 -> 2004. 02105; group_013:633 -> 2004. 02105; group_013:599 -> 2004. 02105 |
| QASPER-64K | 2 | group_013 | 1911.10742 | excluded_evidence_outside | group_013:115 -> 1610.06510; group_013:621 -> 2004. 02105; group_013:624 -> 2004. 02105 |
| QASPER-64K | 3 | group_013 | 1911.10742 | excluded_evidence_outside | group_013:402 -> 1910.10487 |
| QASPER-64K | 10 | group_001 | 1611.06322 | excluded_evidence_outside | group_001:302 -> 1611. 06322; group_001:288 -> 1909. 09070,1611. 06322 |


## 4. Chunks per gold sub-document (kept queries)

| Setting | Gold sub-docs (distinct) | Chunks per gold sub-doc min/med/max | Same, over queries | All sub-docs min/med/max | Queries with <= 1 chunks | Queries with <= 3 chunks | Queries with <= 5 chunks | Queries with <= 10 chunks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MuSiQue-32K | 574 | 8 / 19 / 33 | 8 / 19 / 33 | 8 / 19 / 37 | 0 (0%) | 0 (0%) | 0 (0%) | 6 (1%) |
| QASPER-64K | 224 | 5 / 25.5 / 133 | 5 / 27 / 133 | 2 / 20 / 133 | 0 (0%) | 0 (0%) | 5 (1%) | 10 (3%) |

HR@k inside the sub-document is trivially 100% for queries whose gold sub-document has k chunks or fewer; later steps report each HR@k on all kept queries and on those with more than k chunks.

## 5. Method disagreements (first 5)

```
MuSiQue-32K: {'chunk_id': 'group_012:197', 'alignment': ['2hop__79875_84469', '3hop1__458916_651917_47686'], 'header_walk': ['2hop__79875_84469']}
MuSiQue-32K: {'chunk_id': 'group_012:199', 'alignment': ['3hop1__458916_651917_47686'], 'header_walk': ['2hop__79875_84469']}
MuSiQue-32K: {'chunk_id': 'group_012:200', 'alignment': ['3hop1__458916_651917_47686'], 'header_walk': ['2hop__79875_84469']}
MuSiQue-32K: {'chunk_id': 'group_012:201', 'alignment': ['3hop1__458916_651917_47686'], 'header_walk': ['2hop__79875_84469']}
MuSiQue-32K: {'chunk_id': 'group_012:203', 'alignment': ['3hop1__458916_651917_47686'], 'header_walk': ['2hop__79875_84469']}
```

## Traceback (MuSiQue-64K)

```
Traceback (most recent call last):
  File "/home/iataghav/data/SAADI/experiments/rebuttal/c4_subdoc_control/scripts/step01_map.py", line 432, in body
    results[s.key] = map_setting(s, retrievers, sources)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/iataghav/data/SAADI/experiments/rebuttal/c4_subdoc_control/scripts/step01_map.py", line 231, in map_setting
    documents = read_json(prep / "documents.json")
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/SAADI/experiments/rebuttal/c4_subdoc_control/scripts/c4_common.py", line 138, in read_json
    with Path(path).open("r", encoding="utf-8") as handle:
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/pathlib.py", line 1015, in open
    return io.open(self, mode, buffering, encoding, errors, newline)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/mnt/data/zeinab/SAADI/outputs/musique_ablation/64K_expansion/prepared/documents.json'
```

## Traceback (MuSiQue-128K)

```
Traceback (most recent call last):
  File "/home/iataghav/data/SAADI/experiments/rebuttal/c4_subdoc_control/scripts/step01_map.py", line 432, in body
    results[s.key] = map_setting(s, retrievers, sources)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/iataghav/data/SAADI/experiments/rebuttal/c4_subdoc_control/scripts/step01_map.py", line 231, in map_setting
    documents = read_json(prep / "documents.json")
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/SAADI/experiments/rebuttal/c4_subdoc_control/scripts/c4_common.py", line 138, in read_json
    with Path(path).open("r", encoding="utf-8") as handle:
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/pathlib.py", line 1015, in open
    return io.open(self, mode, buffering, encoding, errors, newline)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/mnt/data/zeinab/SAADI/outputs/musique_ablation/128K_expansion/prepared/documents.json'
```

## Traceback (MuSiQue-256K)

```
Traceback (most recent call last):
  File "/home/iataghav/data/SAADI/experiments/rebuttal/c4_subdoc_control/scripts/step01_map.py", line 432, in body
    results[s.key] = map_setting(s, retrievers, sources)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/iataghav/data/SAADI/experiments/rebuttal/c4_subdoc_control/scripts/step01_map.py", line 231, in map_setting
    documents = read_json(prep / "documents.json")
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/SAADI/experiments/rebuttal/c4_subdoc_control/scripts/c4_common.py", line 138, in read_json
    with Path(path).open("r", encoding="utf-8") as handle:
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/pathlib.py", line 1015, in open
    return io.open(self, mode, buffering, encoding, errors, newline)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/mnt/data/zeinab/SAADI/outputs/musique_ablation/256K_expansion/prepared/documents.json'
```

## Traceback (QASPER-32K)

```
Traceback (most recent call last):
  File "/home/iataghav/data/SAADI/experiments/rebuttal/c4_subdoc_control/scripts/step01_map.py", line 432, in body
    results[s.key] = map_setting(s, retrievers, sources)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/iataghav/data/SAADI/experiments/rebuttal/c4_subdoc_control/scripts/step01_map.py", line 231, in map_setting
    documents = read_json(prep / "documents.json")
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/SAADI/experiments/rebuttal/c4_subdoc_control/scripts/c4_common.py", line 138, in read_json
    with Path(path).open("r", encoding="utf-8") as handle:
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/pathlib.py", line 1015, in open
    return io.open(self, mode, buffering, encoding, errors, newline)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/mnt/data/zeinab/SAADI/outputs/qasper_ablation/32K_expansion/prepared/documents.json'
```

## Traceback (QASPER-128K)

```
Traceback (most recent call last):
  File "/home/iataghav/data/SAADI/experiments/rebuttal/c4_subdoc_control/scripts/step01_map.py", line 432, in body
    results[s.key] = map_setting(s, retrievers, sources)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/iataghav/data/SAADI/experiments/rebuttal/c4_subdoc_control/scripts/step01_map.py", line 231, in map_setting
    documents = read_json(prep / "documents.json")
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/SAADI/experiments/rebuttal/c4_subdoc_control/scripts/c4_common.py", line 138, in read_json
    with Path(path).open("r", encoding="utf-8") as handle:
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/pathlib.py", line 1015, in open
    return io.open(self, mode, buffering, encoding, errors, newline)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/mnt/data/zeinab/SAADI/outputs/qasper_ablation/128K_expansion/prepared/documents.json'
```

## Traceback (QASPER-256K)

```
Traceback (most recent call last):
  File "/home/iataghav/data/SAADI/experiments/rebuttal/c4_subdoc_control/scripts/step01_map.py", line 432, in body
    results[s.key] = map_setting(s, retrievers, sources)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/iataghav/data/SAADI/experiments/rebuttal/c4_subdoc_control/scripts/step01_map.py", line 231, in map_setting
    documents = read_json(prep / "documents.json")
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/SAADI/experiments/rebuttal/c4_subdoc_control/scripts/c4_common.py", line 138, in read_json
    with Path(path).open("r", encoding="utf-8") as handle:
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/pathlib.py", line 1015, in open
    return io.open(self, mode, buffering, encoding, errors, newline)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/mnt/data/zeinab/SAADI/outputs/qasper_ablation/256K_expansion/prepared/documents.json'
```

## Problems (the step failed)

- musique_32k: group_012: 67 chunk(s) not found in the text and the header walk is not reliable
- musique_64k: FileNotFoundError: [Errno 2] No such file or directory: '/mnt/data/zeinab/SAADI/outputs/musique_ablation/64K_expansion/prepared/documents.json'
- musique_128k: FileNotFoundError: [Errno 2] No such file or directory: '/mnt/data/zeinab/SAADI/outputs/musique_ablation/128K_expansion/prepared/documents.json'
- musique_256k: FileNotFoundError: [Errno 2] No such file or directory: '/mnt/data/zeinab/SAADI/outputs/musique_ablation/256K_expansion/prepared/documents.json'
- qasper_32k: FileNotFoundError: [Errno 2] No such file or directory: '/mnt/data/zeinab/SAADI/outputs/qasper_ablation/32K_expansion/prepared/documents.json'
- qasper_64k: group_001: 206 chunk(s) not found in the text and the header walk is not reliable
- qasper_64k: group_002: 227 chunk(s) not found in the text and the header walk is not reliable
- qasper_64k: group_003: 224 chunk(s) not found in the text and the header walk is not reliable
- qasper_64k: group_004: 205 chunk(s) not found in the text and the header walk is not reliable
- qasper_64k: group_005: 209 chunk(s) not found in the text and the header walk is not reliable
- qasper_64k: group_006__variant_ab84194533: 272 chunk(s) not found in the text and the header walk is not reliable
- qasper_64k: group_006__variant_8d880e5af9: 258 chunk(s) not found in the text and the header walk is not reliable
- qasper_64k: group_007: 213 chunk(s) not found in the text and the header walk is not reliable
- qasper_64k: group_008: 263 chunk(s) not found in the text and the header walk is not reliable
- qasper_64k: group_009: 211 chunk(s) not found in the text and the header walk is not reliable
- qasper_64k: group_010: 281 chunk(s) not found in the text and the header walk is not reliable
- qasper_64k: group_011: 202 chunk(s) not found in the text and the header walk is not reliable
- qasper_64k: group_012: 231 chunk(s) not found in the text and the header walk is not reliable
- qasper_64k: group_013: 195 chunk(s) not found in the text and the header walk is not reliable
- qasper_64k: group_014: 203 chunk(s) not found in the text and the header walk is not reliable
- qasper_64k: group_015: 172 chunk(s) not found in the text and the header walk is not reliable
- qasper_64k: group_016: 176 chunk(s) not found in the text and the header walk is not reliable
- qasper_64k: group_017: 249 chunk(s) not found in the text and the header walk is not reliable
- qasper_64k: group_018__variant_ea0eb825e2: 303 chunk(s) not found in the text and the header walk is not reliable
- qasper_64k: group_018__variant_62176dfa97: 313 chunk(s) not found in the text and the header walk is not reliable
- qasper_64k: group_019: 179 chunk(s) not found in the text and the header walk is not reliable
- qasper_64k: group_020: 228 chunk(s) not found in the text and the header walk is not reliable
- qasper_64k: group_021: 164 chunk(s) not found in the text and the header walk is not reliable
- qasper_128k: FileNotFoundError: [Errno 2] No such file or directory: '/mnt/data/zeinab/SAADI/outputs/qasper_ablation/128K_expansion/prepared/documents.json'
- qasper_256k: FileNotFoundError: [Errno 2] No such file or directory: '/mnt/data/zeinab/SAADI/outputs/qasper_ablation/256K_expansion/prepared/documents.json'

## Files written

- `experiments/rebuttal/c4_subdoc_control/reports/step_01.json`
- `experiments/rebuttal/c4_subdoc_control/reports/step_01.md`
- `results/rebuttal/c4_subdoc_control/subdoc_map/musique_32k.json`
- `results/rebuttal/c4_subdoc_control/subdoc_map/qasper_64k.json`
