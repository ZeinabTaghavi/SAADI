# p1_ledger - step 03: Memory check with stubs

Status: **ok** - 2026-11-08 18:16:13

## What was done

- `bash experiments/method_update/p1_ledger/sh/03_*.sh `
- `inspect /home/iataghav/data/SAADI/outputs/method_update/p1_ledger/novelhopqa_probe/indexes/68ecf44999173223/llm_cache/novelhopqa/hierarchical/branching_factor_5/chunk_500/AE_5_2_3/Qwen3-30B-A3B-Instruct-2507/AEWord/index.json + ledger /home/iataghav/data/SAADI/outputs/method_update/p1_ledger/ledger with the probe's step 6 (imported) and saadi.ledger.measure`
- Versions: python=3.12.3, platform=Linux-6.17.0-23-generic-x86_64-with-glibc2.39, numpy=2.3.5, pandas=3.0.1, yaml=6.0.3, saadi=0.1.0, git_commit=4e7bbd36476e82f4d55a5f98fa1641bfe13898eb, torch=2.11.0+cu129, transformers=4.57.6, gliner=0.2.30, vllm=0.23.0

## Numbers

### Books: chunk records, records, stubs, promotions, events

| book | chunk records | records | stubs | promotions | ledger events |
| --- | --- | --- | --- | --- | --- |
| B04 | 321 | 211 | 29 | 7 | 321 |
| B18 | 312 | 244 | 36 | 9 | 312 |
| B26 | 384 | 247 | 36 | 26 | 384 |
| B27 | 338 | 199 | 28 | 11 | 338 |
| B31 | 244 | 253 | 33 | 18 | 244 |
| B34 | 335 | 187 | 35 | 11 | 335 |
| B37 | 262 | 165 | 41 | 16 | 262 |
| B43 | 246 | 104 | 5 | 2 | 246 |
| B44 | 385 | 267 | 41 | 20 | 385 |
| B61 | 357 | 446 | 144 | 44 | 357 |

### Description cap applied to the loaded index (ledger replacements removed before scoring)

| book | records with replacements | descriptions removed |
| --- | --- | --- |
| B04 | 23 | 0 |
| B18 | 24 | 0 |
| B26 | 41 | 0 |
| B27 | 33 | 0 |
| B31 | 11 | 0 |
| B34 | 41 | 0 |
| B37 | 29 | 0 |
| B43 | 19 | 0 |
| B44 | 33 | 0 |
| B61 | 23 | 0 |

### Memory outcomes per cell, run 1 (run 0 in parentheses)

| cell | n | form 1 present after the gold chunk | form 2 present after the reveal | both present (final) | earlier record in the prompt at the reveal (gold chunk for cell 4) | joined when both present | bridge formed (form 2 in the gold chunk's live record) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 (A, reveal 3 chunks after the gold) | 30 | 26/30 (87%) (run 0: 18/30 (60%)) | 23/30 (77%) (run 0: 6/30 (20%)) | 22/30 (73%) (run 0: 19/30 (63%)) | 16/30 (53%) (run 0: 0/30 (0%)) | 17/30 (57%) (run 0: 12/30 (40%)) | 4/30 (13%) (run 0: 1/30 (3%)) |
| 2 (A, reveal 25% of the book after) | 30 | 27/30 (90%) (run 0: 19/30 (63%)) | 23/30 (77%) (run 0: 3/30 (10%)) | 21/30 (70%) (run 0: 23/30 (77%)) | 15/30 (50%) (run 0: 1/30 (3%)) | 14/30 (47%) (run 0: 10/30 (33%)) | 5/30 (17%) (run 0: 0/30 (0%)) |
| 3 (A, reveal 55% of the book after) | 30 | 28/30 (93%) (run 0: 14/30 (47%)) | 23/30 (77%) (run 0: 4/30 (13%)) | 22/30 (73%) (run 0: 15/30 (50%)) | 10/30 (33%) (run 0: 1/30 (3%)) | 15/30 (50%) (run 0: 6/30 (20%)) | 5/30 (17%) (run 0: 0/30 (0%)) |
| 4 (A, reveal 20% before the gold (control)) | 30 | 26/30 (87%) (run 0: 14/30 (47%)) | 21/30 (70%) (run 0: 5/30 (17%)) | 19/30 (63%) (run 0: 16/30 (53%)) | 3/30 (10%) (run 0: 2/30 (7%)) | 16/30 (53%) (run 0: 6/30 (20%)) | 1/30 (3%) (run 0: 0/30 (0%)) |
| 5 (A, no reveal (control)) | 30 | 25/30 (83%) (run 0: 15/30 (50%)) | - | - | - | - | - |
| 6 (B, nickname, reveal 25% after) | 30 | 16/30 (53%) (run 0: 9/30 (30%)) | 22/30 (73%) (run 0: 3/30 (10%)) | 16/30 (53%) (run 0: 14/30 (47%)) | 9/30 (30%) (run 0: 0/30 (0%)) | 13/30 (43%) (run 0: 8/30 (27%)) | 4/30 (13%) (run 0: 0/30 (0%)) |
| 7 (C, description, reveal 25% after) | 30 | 25/30 (83%) (run 0: 23/30 (77%)) | 0/30 (0%) (run 0: 15/30 (50%)) | 0/30 (0%) (run 0: 23/30 (77%)) | 18/30 (60%) (run 0: 2/30 (7%)) | 0/30 (0%) (run 0: 15/30 (50%)) | 20/30 (67%) (run 0: 5/30 (17%)) |
| 8 (D, object, reveal 25% after) | 30 | 16/30 (53%) (run 0: 10/30 (33%)) | 24/30 (80%) (run 0: 3/30 (10%)) | 21/30 (70%) (run 0: 16/30 (53%)) | 11/30 (37%) (run 0: 2/30 (7%)) | 15/30 (50%) (run 0: 11/30 (37%)) | 3/30 (10%) (run 0: 0/30 (0%)) |

### Reading the columns

Run 0's 'form 1 present' is its step-6 'form1 record found' (final memory); run 0's 'both present' counts personas whose merge direction was not 'no_record' (both names in some record); run 0's 'in the prompt' is step 9's recall class at the reveal (the gold chunk for cell 4). Stubs count as records throughout.

### Merge directions per cell

| cell | run 1 | run 0 |
| --- | --- | --- |
| 1 | {'form1_into_form2': 9, 'form2_into_form1': 8, 'never_joined': 12, 'no_record': 1} | {'form1_into_form2': 6, 'form2_into_form1': 6, 'never_joined': 7, 'no_record': 11} |
| 2 | {'form1_into_form2': 8, 'form2_into_form1': 8, 'never_joined': 14} | {'form1_into_form2': 7, 'form2_into_form1': 3, 'never_joined': 13, 'no_record': 7} |
| 3 | {'form1_into_form2': 8, 'form2_into_form1': 8, 'never_joined': 13, 'no_record': 1} | {'form1_into_form2': 2, 'form2_into_form1': 4, 'never_joined': 9, 'no_record': 15} |
| 4 | {'form1_into_form2': 6, 'form2_into_form1': 12, 'never_joined': 10, 'no_record': 2} | {'form1_into_form2': 1, 'form2_into_form1': 5, 'never_joined': 10, 'no_record': 14} |
| 5 | {'never_joined': 25, 'no_record': 5} | {'never_joined': 15, 'no_record': 15} |
| 6 | {'form1_into_form2': 9, 'form2_into_form1': 7, 'never_joined': 11, 'no_record': 3} | {'form1_into_form2': 5, 'form2_into_form1': 3, 'never_joined': 6, 'no_record': 16} |
| 7 | {'form2_into_form1': 24, 'never_joined': 6} | {'form2_into_form1': 15, 'never_joined': 8, 'no_record': 7} |
| 8 | {'form1_into_form2': 7, 'form2_into_form1': 12, 'never_joined': 10, 'no_record': 1} | {'form1_into_form2': 8, 'form2_into_form1': 3, 'never_joined': 5, 'no_record': 14} |

### Where form 2 sits in the form-1 record (run 1)

| cell | name | alias | description | none | form 2 in gold FROZEN record |
| --- | --- | --- | --- | --- | --- |
| 1 | 0 | 7 | 1 | 22 | 0/30 (0%) |
| 2 | 0 | 5 | 3 | 22 | 0/30 (0%) |
| 3 | 0 | 6 | 2 | 22 | 0/30 (0%) |
| 4 | 3 | 2 | 7 | 18 | 5/30 (17%) |
| 5 | 0 | 0 | 0 | 0 | - |
| 6 | 1 | 6 | 0 | 23 | 0/30 (0%) |
| 7 | 0 | 0 | 24 | 6 | 0/30 (0%) |
| 8 | 0 | 0 | 12 | 18 | 0/30 (0%) |

### Earlier record in the prompt where the link could form

Earlier record existed before the link step: 159/210 (76%); of those, in the prompt: 82/159 (52%) (run 0: 6 of 72); existed but left out by the ledger-recall cap: 27. Decision block status at the link step: {'ok': 210}.

### Gate

The bridge word reached the gold chunk's record in 14 of 90 personas of cells 1-3 (16%); run 0: 1 of 90. **Below half: the retrieval numbers of step 4 must be read with that in mind.**

### Provenance of records and stubs (all books)

| measure | value |
| --- | --- |
| records: from the candidate net only / from the LLM only / from both | 38 / 25 / 2260 |
| stubs: from the candidate net (incl. parse-failure stubs) | 428 |
| entries recovered (memory name unknown to the ledger; should be 0) | 0 |
| NOT AN ENTITY decisions | 1674 |
| SAME AS decisions / unresolved | 17094 / 33 |
| decision-block parse failures / candidates stubbed by that policy / omitted candidates | 0 / 0 / 7 |
| promotions | 164 |
| aliases added / rejected as descriptive | 2557 / 3 |
| described records named by a stored alias (kept under the stored name) | 3823 |
| description replacements | 6359 |

### Merge audit: 30 of 17127 SAME AS decisions (seed 42; all in same_as_decisions.jsonl)

- B04 step 307: 'Brownlow' -> 'Mr. Brownlow' [E0046], linked: a man who has known Monks's father since youth and is now confronting Monks about his brother's identity
- B04 step 319: 'Sikes' -> 'London' [E0026], linked: the chunk refers to Sikes as having committed a crime and fled, and the history confirms he died in prison, matching E0026's record
- B18 step 19: 'Sir John' -> 'Sir Pagan d’Urberville' [E0018], linked: the chunk refers to him as "Sir John" and describes his noble claims, matching the recalled record
- B18 step 139: 'Marian' -> 'Marian' [E0153], linked: she is a dairymaid at Talbothays Dairy, and the chunk refers to her by name with no new details
- B26 step 25: 'Shelby' -> 'Mr. Shelby' [E0020], linked: the chunk refers to him as "Mr. Shelby" and he is the same person described in the history as Mr. Shelby, who is in debt and sells Tom
- B26 step 83: 'Chloe' -> 'Aunt Chloe' [E0032], linked: this is a common name for Aunt Chloe, and the chunk uses it in context with her actions
- B27 step 189: 'Edmund' -> 'Edmund Bertram' [E0021], linked: the chunk refers to him as "Edmund" and describes his emotional state and interactions with Fanny
- B27 step 277: 'Fanny' -> 'Fanny Price' [E0019], linked: the chunk refers to her as "Fanny" and she is the central figure in the narrative
- B31 step 45: 'Jim' -> 'Jim' [E0015], linked: the chunk refers to Jim as the person who was bitten by the snake and is described as a big nigger belonging to Miss Watson
- B31 step 75: 'Jim' -> 'Jim' [E0015], linked: the chunk describes Jim as the enslaved man owned by Miss Watson, consistent with the recalled record
- B31 step 97: 'Miss Sophia' -> 'Miss Sophia' [E0123], linked: the chunk confirms she ran off with Harney and matches prior description
- B31 step 135: 'George' -> 'George Peters' [E0069], linked: the chunk identifies George as the married brother who died last year with his wife, and the father of Mary Jane, Susan, and Joanna, matching E0069's description
- B34 step 71: 'Mr. Woodhouse' -> 'Mr. Woodhouse' [E0003], linked: the chunk refers to him as the father whose habits are consulted, matching the record
- B34 step 101: 'Emma' -> 'Emma' [E0001], linked: the chunk refers to her as "Emma" and matches the known record
- B34 step 131: 'Highbury' -> 'Highbury road' [E0075], linked: the chunk refers to Highbury as a place with many good-looking houses, and E0075 is described as a road leading toward Hartfield, consistent with the context
- B34 step 171: 'Miss Bates' -> 'Mrs. Bates' [E0022], linked: the chunk refers to her as "Miss Bates" and she is already recorded with that name
- B37 step 65: 'Collins' -> 'Mr. Collins' [E0032], linked: the chunk refers to Mr. Collins as the one who sought Elizabeth’s hand, matching his prior record
- B37 step 87: 'Charlotte' -> 'Charlotte Lucas' [E0040], linked: the chunk refers to Charlotte Lucas as the one who receives Mr. Collins's proposal, confirming she is the same person as Miss Lucas
- B37 step 183: 'Lydia' -> 'Lydia Bennet' [E0023], linked: the chunk explicitly names Lydia as the sister who eloped with Wickham, matching her record
- B37 step 229: 'Jane' -> 'Jane Bennet' [E0066], linked: the chunk refers to Jane Bennet as Elizabeth's sister and the object of Bingley's affection, matching the record
- B43 step 21: 'Hareton Earnshaw' -> 'Hareton Earnshaw' [E0006], linked: he is the nephew of Mrs. Linton and the last of the Earnshaws, and his name is carved on the house
- B43 step 81: 'Savasan' -> 'Frances' [E0014], linked: The chunk states "Ellsworthen, whose given name was Savasan", linking the name to the record E0014, which includes 'Savasan' as an alias
- B43 step 114: 'Linton' -> 'Gimmerton Chapel' [E0049], linked: the chunk refers to "Linton" as a person in the context of Catherine's love and marriage, and the record identifies Linton as Edgar Linton, who is married to Catherine
- B43 step 137: 'Ellen' -> 'Frances' [E0014], linked: the chunk refers to the narrator as "Ellen" and "Nelly", and E0014 is the record for Mrs. Dean, also known as Nelly, Ellen, or Frances, who is the narrator and caretaker
- B43 step 146: 'Miss Cathy' -> 'Catherine Earnshaw' [E0012], linked: the chunk uses "Miss Cathy" to refer to Catherine Earnshaw, who is already recorded as E0012
- B43 step 243: 'Project Gutenberg' -> 'United States' [E0104], linked: the chunk uses "Project Gutenberg" interchangeably with "Project Gutenberg-tm" as a registered trademark, and E0104's aliases include "Project Gutenberg-tm"
- B44 step 64: 'Mr. Bates' -> 'Mr. Bates' [E0078], linked: the chunk confirms he is a surgeon sent for at night and came out with the nurse, matching the record's description
- B44 step 143: 'Mary' -> 'Mary Ingram' [E0147], linked: the chunk identifies her as one of the daughters of Dowager Lady Ingram, and the record E0147 confirms her as Mary Ingram
- B61 step 14: 'Nozdryov' -> 'Nozdryov' [E0004], linked: the chunk describes him as a hedonistic bully who refuses to sell souls and tries to cheat Tchitchikov
- B61 step 274: 'Tchitchikov' -> 'Tchitchikov' [E0003], linked: same as candidate 1, already resolved

### Mr./Mrs. check on the books' real characters

Records whose names/aliases share a stripped name but carry different titles (persona names excluded): **120** (list in title_conflicts.jsonl). SAME AS decisions without a reason: 0 of 17127. Run 0 had no SAME AS mechanism; step 9 counted 10 title-variant duplicate pairs among persona records.

### Up to ten title conflicts

| book | record | aliases | stripped name | titles |
| --- | --- | --- | --- | --- |
| B04 | The surgeon | ['Mr. Losberne', 'Doctor Losberne', 'The honest gentleman', 'Blathers', 'Losberne'] | losberne | ['', 'doctor', 'mr'] |
| B04 | Mr. Bumble | ['Bumble', 'Beadle', 'Mr Bumble', 'Mrs. Sowerberry', 'The beadle', 'Mrs. Bumble'] | bumble | ['', 'mr', 'mr mrs', 'mrs'] |
| B04 | Mr. Sowerberry | ['Sowerberry', 'Mrs. Sowerberry'] | sowerberry | ['', 'mr', 'mrs'] |
| B04 | Mrs. Corney | ['Mr. Corney', 'Corney'] | corney | ['', 'mr', 'mrs'] |
| B04 | Mrs. Maylie | ['Rose', 'Miss Maylie', 'The young lady', 'The old lady', 'Mrs. Maylies', 'Rose Maylie'] | maylie | ['miss', 'mrs'] |
| B18 | Tess of the d’Urbervilles | ['Tess Durbeyfield', 'Tess', 'Mrs Durbeyfield', 'Joan Durbeyfield', 'Simple Tess Durbeyfield', 'Sissy'] | durbeyfield | ['miss', 'mrs'] |
| B18 | Jack Durbeyfield | ['Durbeyfield', 'Mrs Durbeyfield', 'Joan Durbeyfield', 'Mr Durbeyfield', 'Joan', 'Durbeyfield household'] | durbeyfield | ['', 'mr', 'mrs'] |
| B18 | Jack Durbeyfield | ['Durbeyfield', 'Mrs Durbeyfield', 'Joan Durbeyfield', 'Mr Durbeyfield', 'Joan', 'Durbeyfield household'] | tess | ['father', 'father mother'] |
| B18 | Angel | ['Angel Clare', 'Mr Clare', 'Mr Angel Clare', 'Cambridge', 'Evangelical school', 'London'] | clare | ['', 'mr', 'mrs'] |
| B18 | Lady Durbeyfield | ['Mrs Durbeyfield', 'Joan', 'Joan Durbeyfield', 'The harvesters', 'J. Durbeyfield', 'Widow Joan'] | durbeyfield | ['lady', 'mrs'] |

### Five full ledger entries (seed 42)

- **B18 E0006 'the Devil’s pitchfork'** kind=record created step 1 (B18:1) promoted=None provenance=['candidate_net', 'llm'] aliases=[] linked chunks=1 described at=[1] same_as=0 sentence=''
- **B31 E0162 'Hamlet'** kind=stub created step 115 (B31:115) promoted=None provenance=['candidate_net'] aliases=[] linked chunks=1 described at=[] same_as=0 sentence='and you — well, let me see — oh, I’ve got it —\nyou can do Hamlet’s soliloquy.'
- **B31 E0177 'Ben Thompson'** kind=record created step 117 (B31:117) promoted=None provenance=['candidate_net', 'llm'] aliases=[] linked chunks=1 described at=[117] same_as=0 sentence=''
- **B43 E0056 'Falderwick'** kind=record created step 75 (B43:75) promoted=None provenance=['candidate_net', 'llm'] aliases=[] linked chunks=1 described at=[75] same_as=0 sentence=''
- **B44 E0274 'East'** kind=record created step 307 (B44:307) promoted=None provenance=['candidate_net', 'llm'] aliases=['Ganges'] linked chunks=2 described at=[307, 340] same_as=1 sentence=''

### JSON parser status of chunks holding inserted sentences

{'fallback': 439, 'ok': 11}

### Hand-off

Paste this report back. Step 4 (retrieval) waits for it.

## Anything unexpected

Nothing unexpected.

## Files written

- `/home/iataghav/data/SAADI/outputs/method_update/p1_ledger/personas_memory.jsonl`
- `/home/iataghav/data/SAADI/outputs/method_update/p1_ledger/same_as_decisions.jsonl`
- `/home/iataghav/data/SAADI/outputs/method_update/p1_ledger/title_conflicts.jsonl`
