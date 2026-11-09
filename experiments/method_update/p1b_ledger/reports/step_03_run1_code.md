# p1b_ledger - step 03: Memory check with stubs (run 1's step 3 code on run 1b)

Status: **ok** - 2026-11-09 00:41:01

## What was done

- `inspect /mnt/data/zeinab/SAADI/outputs/method_update/p1b_ledger/novelhopqa_probe/indexes/eb4d1079603b2760/llm_cache/novelhopqa/hierarchical/branching_factor_5/chunk_500/AE_5_2_3/Qwen3-30B-A3B-Instruct-2507/AEWord/index.json + ledger /mnt/data/zeinab/SAADI/outputs/method_update/p1b_ledger/ledger with the probe's step 6 (imported) and saadi.ledger.measure`
- Versions: python=3.12.3, platform=Linux-6.17.0-23-generic-x86_64-with-glibc2.39, numpy=2.3.5, pandas=3.0.1, yaml=6.0.3, saadi=0.1.0, git_commit=28eed3a1ee31487f3bf23dcb3470aa0e164e8cfd, torch=2.11.0+cu129, transformers=4.57.6, gliner=0.2.30, vllm=0.23.0

## Numbers

### How to read this file

Produced by run 1's own step code (experiments/method_update/p1_ledger/scripts, imported unchanged) on run 1b's build. In its tables **'run 1' means run 1b** and 'run 0' is run 0. The run 1b report with three columns is step_03.md.

### Books: chunk records, records, stubs, promotions, events

| book | chunk records | records | stubs | promotions | ledger events |
| --- | --- | --- | --- | --- | --- |
| B04 | 321 | 266 | 62 | 29 | 321 |
| B18 | 312 | 325 | 151 | 45 | 312 |
| B26 | 384 | 340 | 136 | 60 | 384 |
| B27 | 338 | 247 | 73 | 33 | 338 |
| B31 | 244 | 325 | 58 | 34 | 244 |
| B34 | 335 | 225 | 48 | 30 | 335 |
| B37 | 262 | 189 | 57 | 32 | 262 |
| B43 | 246 | 130 | 44 | 13 | 246 |
| B44 | 385 | 365 | 101 | 39 | 385 |
| B61 | 357 | 526 | 275 | 81 | 357 |

### Description cap applied to the loaded index (ledger replacements removed before scoring)

| book | records with replacements | descriptions removed |
| --- | --- | --- |
| B04 | 33 | 0 |
| B18 | 25 | 0 |
| B26 | 48 | 0 |
| B27 | 39 | 0 |
| B31 | 14 | 0 |
| B34 | 44 | 0 |
| B37 | 37 | 0 |
| B43 | 21 | 0 |
| B44 | 45 | 0 |
| B61 | 31 | 0 |

### JSON parser status per call: ok / fallback / failed (fallback = fenced JSON recovered by the lenient extractor, the published behaviour; failed = no JSON recovered)

| book | run 1 | run 0 |
| --- | --- | --- |
| B04 | 3 / 317 / 1 | 3 / 318 / 0 |
| B18 | 13 / 298 / 1 | 11 / 301 / 0 |
| B26 | 2 / 382 / 0 | 0 / 384 / 0 |
| B27 | 1 / 337 / 0 | 0 / 338 / 0 |
| B31 | 3 / 241 / 0 | 0 / 244 / 0 |
| B34 | 2 / 333 / 0 | 0 / 335 / 0 |
| B37 | 9 / 253 / 0 | 5 / 257 / 0 |
| B43 | 1 / 245 / 0 | 0 / 246 / 0 |
| B44 | 19 / 366 / 0 | 10 / 375 / 0 |
| B61 | 20 / 337 / 0 | 16 / 341 / 0 |
| **all books** | **73 / 3109 / 2** | **45 / 3139 / 0** |

### Memory outcomes per cell, run 1 (run 0 in parentheses)

| cell | n | form 1 present after the gold chunk | form 2 present after the reveal | both present (final) | earlier record in the prompt at the reveal (gold chunk for cell 4) | joined when both present | bridge formed (form 2 in the gold chunk's live record) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 (A, reveal 3 chunks after the gold) | 30 | 27/30 (90%) (run 0: 18/30 (60%)) | 24/30 (80%) (run 0: 6/30 (20%)) | 24/30 (80%) (run 0: 19/30 (63%)) | 27/30 (90%) (run 0: 0/30 (0%)) | 20/30 (67%) (run 0: 12/30 (40%)) | 13/30 (43%) (run 0: 1/30 (3%)) |
| 2 (A, reveal 25% of the book after) | 30 | 27/30 (90%) (run 0: 19/30 (63%)) | 24/30 (80%) (run 0: 3/30 (10%)) | 24/30 (80%) (run 0: 23/30 (77%)) | 24/30 (80%) (run 0: 1/30 (3%)) | 17/30 (57%) (run 0: 10/30 (33%)) | 8/30 (27%) (run 0: 0/30 (0%)) |
| 3 (A, reveal 55% of the book after) | 30 | 29/30 (97%) (run 0: 14/30 (47%)) | 24/30 (80%) (run 0: 4/30 (13%)) | 24/30 (80%) (run 0: 15/30 (50%)) | 17/30 (57%) (run 0: 1/30 (3%)) | 14/30 (47%) (run 0: 6/30 (20%)) | 3/30 (10%) (run 0: 0/30 (0%)) |
| 4 (A, reveal 20% before the gold (control)) | 30 | 28/30 (93%) (run 0: 14/30 (47%)) | 21/30 (70%) (run 0: 5/30 (17%)) | 21/30 (70%) (run 0: 16/30 (53%)) | 0/30 (0%) (run 0: 2/30 (7%)) | 13/30 (43%) (run 0: 6/30 (20%)) | 8/30 (27%) (run 0: 0/30 (0%)) |
| 5 (A, no reveal (control)) | 30 | 27/30 (90%) (run 0: 15/30 (50%)) | - | - | - | - | - |
| 6 (B, nickname, reveal 25% after) | 30 | 23/30 (77%) (run 0: 9/30 (30%)) | 26/30 (87%) (run 0: 3/30 (10%)) | 22/30 (73%) (run 0: 14/30 (47%)) | 18/30 (60%) (run 0: 0/30 (0%)) | 18/30 (60%) (run 0: 8/30 (27%)) | 7/30 (23%) (run 0: 0/30 (0%)) |
| 7 (C, description, reveal 25% after) | 30 | 25/30 (83%) (run 0: 23/30 (77%)) | 0/30 (0%) (run 0: 15/30 (50%)) | 0/30 (0%) (run 0: 23/30 (77%)) | 20/30 (67%) (run 0: 2/30 (7%)) | 0/30 (0%) (run 0: 15/30 (50%)) | 17/30 (57%) (run 0: 5/30 (17%)) |
| 8 (D, object, reveal 25% after) | 30 | 22/30 (73%) (run 0: 10/30 (33%)) | 26/30 (87%) (run 0: 3/30 (10%)) | 24/30 (80%) (run 0: 16/30 (53%)) | 15/30 (50%) (run 0: 2/30 (7%)) | 24/30 (80%) (run 0: 11/30 (37%)) | 11/30 (37%) (run 0: 0/30 (0%)) |

### Reading the columns

Run 0's 'form 1 present' is its step-6 'form1 record found' (final memory); run 0's 'both present' counts personas whose merge direction was not 'no_record' (both names in some record); run 0's 'in the prompt' is step 9's recall class at the reveal (the gold chunk for cell 4). Stubs count as records throughout.

### Merge directions per cell

| cell | run 1 | run 0 |
| --- | --- | --- |
| 1 | {'form1_into_form2': 4, 'form2_into_form1': 16, 'never_joined': 10} | {'form1_into_form2': 6, 'form2_into_form1': 6, 'never_joined': 7, 'no_record': 11} |
| 2 | {'form1_into_form2': 6, 'form2_into_form1': 11, 'never_joined': 13} | {'form1_into_form2': 7, 'form2_into_form1': 3, 'never_joined': 13, 'no_record': 7} |
| 3 | {'form1_into_form2': 8, 'form2_into_form1': 7, 'never_joined': 15} | {'form1_into_form2': 2, 'form2_into_form1': 4, 'never_joined': 9, 'no_record': 15} |
| 4 | {'form1_into_form2': 3, 'form2_into_form1': 11, 'never_joined': 14, 'no_record': 2} | {'form1_into_form2': 1, 'form2_into_form1': 5, 'never_joined': 10, 'no_record': 14} |
| 5 | {'never_joined': 27, 'no_record': 3} | {'never_joined': 15, 'no_record': 15} |
| 6 | {'form1_into_form2': 10, 'form2_into_form1': 9, 'never_joined': 10, 'no_record': 1} | {'form1_into_form2': 5, 'form2_into_form1': 3, 'never_joined': 6, 'no_record': 16} |
| 7 | {'form2_into_form1': 21, 'never_joined': 8, 'no_record': 1} | {'form2_into_form1': 15, 'never_joined': 8, 'no_record': 7} |
| 8 | {'form1_into_form2': 8, 'form2_into_form1': 18, 'never_joined': 4} | {'form1_into_form2': 8, 'form2_into_form1': 3, 'never_joined': 5, 'no_record': 14} |

### Where form 2 sits in the form-1 record (run 1)

| cell | name | alias | description | none | form 2 in gold FROZEN record |
| --- | --- | --- | --- | --- | --- |
| 1 | 0 | 11 | 5 | 14 | 0/30 (0%) |
| 2 | 0 | 5 | 6 | 19 | 0/30 (0%) |
| 3 | 0 | 3 | 4 | 23 | 0/30 (0%) |
| 4 | 0 | 2 | 9 | 19 | 8/30 (27%) |
| 5 | 0 | 0 | 0 | 0 | - |
| 6 | 1 | 7 | 1 | 21 | 0/30 (0%) |
| 7 | 0 | 0 | 21 | 9 | 0/30 (0%) |
| 8 | 0 | 0 | 18 | 12 | 0/30 (0%) |

### Earlier record in the prompt where the link could form

Earlier record existed before the link step: 174/210 (83%); of those, in the prompt: 121/174 (70%) (run 0: 6 of 72); existed but left out by the ledger-recall cap: 32. Decision block status at the link step: {'ok': 210}.

### Gate

The bridge word reached the gold chunk's record in 24 of 90 personas of cells 1-3 (27%); run 0: 1 of 90. **Below half: the retrieval numbers of step 4 must be read with that in mind.**

### Provenance of records and stubs (all books)

| measure | value |
| --- | --- |
| records: from the candidate net only / from the LLM only / from both | 89 / 22 / 2827 |
| stubs: from the candidate net (incl. parse-failure stubs) | 1005 |
| entries recovered (memory name unknown to the ledger; should be 0) | 0 |
| NOT AN ENTITY decisions | 2232 |
| SAME AS decisions / unresolved | 17099 / 79 |
| decision-block parse failures / candidates stubbed by that policy / omitted candidates | 0 / 0 / 3 |
| promotions | 396 |
| aliases added / rejected as descriptive | 1418 / 7 |
| described records named by a stored alias (kept under the stored name) | 288 |
| description replacements | 6262 |

### Merge audit: 30 of 17099 SAME AS decisions (seed 42; all in same_as_decisions.jsonl)

- B04 step 304: 'Oliver' -> 'Oliver Twist' [E0001], linked: the chunk refers to him as the boy who is being addressed and whose tears fall, consistent with prior records
- B04 step 316: 'Oliver' -> 'Oliver Twist' [E0001], linked: the chunk refers to him as "Oliver" and he is the boy who was brought to the prison cell
- B18 step 18: 'Tringham' -> 'Parson Tringham' [E0019], linked: the antiquary who previously mentioned the d’Urbervilles, referred to as "Pa’son Tringham" in the chunk and previously recorded
- B18 step 143: 'The Gilded Antler' -> 'The Gilded Antler' [E0145], linked: the chunk states "The Gilded Antler belonged to Wamanpuma", and the record E0145 identifies it as a public house on the high road at Kelmgate, which matches the context
- B26 step 30: 'Eliza' -> 'Eliza' [E0006], linked: the chunk confirms she is the mother of Harry and is fleeing with him
- B26 step 89: 'George' -> 'George Harris' [E0044], linked: auto-join: the title-stripped name equals a shown record's name or alias
- B27 step 198: 'London' -> 'London' [E0018], linked: the chunk mentions London as the destination for Fanny's departure, matching the record's description
- B27 step 283: 'Crown' -> 'Crown' [E0289], linked: the chunk mentions Mr. Crawford meeting an acquaintance at the Crown, matching the recalled record
- B31 step 72: 'free States' -> 'The river' [E0049], linked: the free States are referenced as the destination on the river, consistent with the river's role as a path to freedom
- B31 step 108: 'Jim' -> 'Jim' [E0015], linked: the chunk refers to Jim as "Jim" and describes his actions and relationship to Huck, matching the record
- B31 step 139: 'Peter' -> 'Peter' [E0269], linked: the chunk confirms Peter was the dying brother, left a letter, and had a family connection to Harvey and William
- B31 step 176: 'Jim' -> 'Jim' [E0015], linked: the chunk uses "Jim" as a direct reference to the same character previously established
- B34 step 92: 'Mr. Elton' -> 'Mr. Elton' [E0017], linked: the chunk describes his recent confession and current isolation, matching his prior record
- B34 step 121: 'Mr. Elton' -> 'Mr. Elton' [E0017], linked: the chunk confirms Mr. Elton’s engagement to Miss Hawkins and his changed demeanor, matching his prior role as the one who wrote the charade
- B34 step 149: 'Mrs. Weston' -> 'Mrs. Weston' [E0006], linked: the chunk shows her speaking about the carriage and forming suspicions, matching her role as a thoughtful observer
- B34 step 188: 'Miss Woodhouse' -> 'Emma' [E0001], linked: the chunk refers to Emma as "Miss Woodhouse" and describes her reactions, matching the record
- B37 step 75: 'Darcy' -> 'Darcy' [E0021], linked: recalled record with same name and description of having silent contempt for the Bennet family and being the subject of Mr. Collins's intended introduction
- B37 step 94: 'Jane' -> 'Jane' [E0042], linked: the chunk refers to Jane as the sister whose happiness is at stake, and she is already recorded
- B37 step 187: 'Jane' -> 'Jane' [E0042], linked: the chunk mentions Jane’s letter and her role in conveying the news about Lydia, consistent with her prior record
- B37 step 230: 'Elizabeth' -> 'Elizabeth' [E0022], linked: the chunk refers to Elizabeth as the one hoping to be joined by Darcy, consistent with her role in the history
- B43 step 21: 'Hareton Earnshaw' -> 'Hareton Earnshaw' [E0007], linked: he is the young man at Wuthering Heights and the last of the Earnshaws
- B43 step 81: 'Cathy' -> 'Miss Cathy' [E0019], linked: the chunk uses "Cathy" as a nickname for Miss Cathy, and she is described as Catherine's daughter, matching E0019's record
- B43 step 115: 'Catherine' -> 'Catherine Earnshaw' [E0016], linked: Catherine Earnshaw is the central figure in the emotional reunion, and her name is used interchangeably with "Cathy" and "Catherine Linton"
- B43 step 139: 'Hareton' -> 'Hareton Earnshaw' [E0007], linked: the chunk refers to Hareton as a strong lad of eighteen, matching the record's description
- B43 step 149: 'Mr. Edgar' -> 'Master Linton' [E0053], linked: the chunk refers to Mr. Edgar as the father of Cathy and the one who says "if I can get him" about Linton's return
- B44 step 6: 'Bessie' -> 'Bessie' [E0019], linked: the chunk uses "Bessie" and the record E0019 identifies her as the nurse
- B44 step 74: 'John Reed' -> 'John' [E0021], linked: John Reed is the cousin of Jane Eyre, previously mentioned in the history and now described in detail
- B44 step 147: 'AdÃ¨le' -> 'AdÃ¨le' [E0163], linked: the chunk refers to her as Mr. Rochester’s ward and describes her interaction with Henry Lynn, matching the record
- B61 step 15: 'Nozdryov' -> 'Nozdryov' [E0004], linked: the chunk describes him as a flamboyant, carefree man who exposes Tchitchikov, matching the prior record's description
- B61 step 274: 'Alexasha' -> 'Alexasha' [E0574], promoted: The chunk mentions Alexasha as the one who announces Platon Mihailovitch’s arrival, matching the stub

### Mr./Mrs. check on the books' real characters

Records whose names/aliases share a stripped name but carry different titles (persona names excluded): **6** (list in title_conflicts.jsonl). SAME AS decisions without a reason: 0 of 17099. Run 0 had no SAME AS mechanism; step 9 counted 10 title-variant duplicate pairs among persona records.

### Up to ten title conflicts

| book | record | aliases | stripped name | titles |
| --- | --- | --- | --- | --- |
| B04 | The surgeon | ['Mr. Losberne', 'Doctor Losberne', 'Losberne'] | losberne | ['', 'doctor', 'mr'] |
| B18 | Angel Clare | ['Angel', 'Mr Angel Clare', 'Clare', 'Mr Clare', 'Dairyman Crick', 'Mrs Crick'] | crick | ['mr', 'mrs'] |
| B31 | Dr. Robinson | ['Robinson', 'Doctor Robinson'] | robinson | ['', 'doctor', 'dr'] |
| B43 | Catherine Earnshaw | ['Catherine', 'Catherine Linton', 'Mrs. Linton', 'Mr. Linton', 'Miss Catherine', 'Edgar'] | linton | ['', 'mr', 'mrs'] |
| B43 | Miss Cathy | ['Cathy', 'little Cathy', 'My little mistress', 'Ellen', 'My young mistress', 'My young lady'] | my young | ['lady', 'mistress'] |
| B44 | Mrs. Colonel Dent | ['Colonel and Mrs. Dent'] | dent | ['colonel mrs', 'mrs colonel'] |

### Five full ledger entries (seed 42)

- **B18 E0024 'William the Conqueror'** kind=stub created step 3 (B18:3) promoted=None provenance=['candidate_net'] aliases=[] linked chunks=1 described at=[] same_as=0 sentence='Don’t you really know, Durbeyfield, that you are the lineal representative of the ancient and knightly family of the d’Urbervilles, who derive their descent fro'
- **B31 E0108 'America'** kind=record created step 67 (B31:67) promoted=None provenance=['candidate_net', 'llm'] aliases=[] linked chunks=1 described at=[67] same_as=0 sentence=''
- **B31 E0131 'Saul'** kind=record created step 82 (B31:82) promoted=None provenance=['candidate_net', 'llm'] aliases=['Mars Saul'] linked chunks=2 described at=[82, 96] same_as=1 sentence=''
- **B43 E0078 'Providence'** kind=stub created step 69 (B43:69) promoted=None provenance=['candidate_net'] aliases=[] linked chunks=1 described at=[] same_as=0 sentence='I had risen in angry\nrebellion against Providence.'
- **B44 E0374 'Messalina'** kind=record created step 262 (B44:262) promoted=None provenance=['candidate_net', 'llm'] aliases=[] linked chunks=1 described at=[262] same_as=0 sentence=''

### JSON parser status of chunks holding inserted sentences

{'fallback': 441, 'ok': 9}

### Hand-off

Paste this report back. Step 4 (retrieval) waits for it.

## Anything unexpected

Nothing unexpected.

## Files written

- `/mnt/data/zeinab/SAADI/outputs/method_update/p1b_ledger/personas_memory.jsonl`
- `/mnt/data/zeinab/SAADI/outputs/method_update/p1b_ledger/same_as_decisions.jsonl`
- `/mnt/data/zeinab/SAADI/outputs/method_update/p1b_ledger/title_conflicts.jsonl`
