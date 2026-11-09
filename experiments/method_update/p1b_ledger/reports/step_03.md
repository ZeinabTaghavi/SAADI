# p1b_ledger - step 03: Memory check (run 1b / run 1 / run 0)

Status: **ok** - 2026-11-09 00:41:22

## What was done

- `bash experiments/method_update/p1b_ledger/run_all.sh  (stage 03_memory_check; python stage03_memory_check.py )`
- `git commit 28eed3a1ee31487f3bf23dcb3470aa0e164e8cfd`
- Versions: python=3.12.3, platform=Linux-6.17.0-23-generic-x86_64-with-glibc2.39, numpy=2.3.5, pandas=3.0.1, yaml=6.0.3, saadi=0.1.0, git_commit=28eed3a1ee31487f3bf23dcb3470aa0e164e8cfd, torch=2.11.0+cu129, transformers=4.57.6, gliner=0.2.30, vllm=0.23.0

## Numbers

### Memory check per cell: run 1b / run 1 / run 0

| cell | n | form 1 after gold | form 2 after reveal | both present (final) | earlier record in prompt at the reveal | joined when both present | bridge formed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 (A, reveal 3 chunks after the gold) | 30 | 27/30 (90%) / 26/30 (87%) / 18/30 (60%) | 24/30 (80%) / 23/30 (77%) / 6/30 (20%) | 24/30 (80%) / 22/30 (73%) / 19/30 (63%) | 27/30 (90%) / 16/30 (53%) / 0/30 (0%) | 20/30 (67%) / 17/30 (57%) / 12/30 (40%) | 13/30 (43%) / 4/30 (13%) / 1/30 (3%) |
| 2 (A, reveal 25% of the book after) | 30 | 27/30 (90%) / 27/30 (90%) / 19/30 (63%) | 24/30 (80%) / 23/30 (77%) / 3/30 (10%) | 24/30 (80%) / 21/30 (70%) / 23/30 (77%) | 24/30 (80%) / 15/30 (50%) / 1/30 (3%) | 17/30 (57%) / 14/30 (47%) / 10/30 (33%) | 8/30 (27%) / 5/30 (17%) / 0/30 (0%) |
| 3 (A, reveal 55% of the book after) | 30 | 29/30 (97%) / 28/30 (93%) / 14/30 (47%) | 24/30 (80%) / 23/30 (77%) / 4/30 (13%) | 24/30 (80%) / 22/30 (73%) / 15/30 (50%) | 17/30 (57%) / 10/30 (33%) / 1/30 (3%) | 14/30 (47%) / 15/30 (50%) / 6/30 (20%) | 3/30 (10%) / 5/30 (17%) / 0/30 (0%) |
| 4 (A, reveal 20% before the gold (control)) | 30 | 28/30 (93%) / 26/30 (87%) / 14/30 (47%) | 21/30 (70%) / 21/30 (70%) / 5/30 (17%) | 21/30 (70%) / 19/30 (63%) / 16/30 (53%) | 0/30 (0%) / 3/30 (10%) / 2/30 (7%) | 13/30 (43%) / 16/30 (53%) / 6/30 (20%) | 8/30 (27%) / 1/30 (3%) / 0/30 (0%) |
| 5 (A, no reveal (control)) | 30 | 27/30 (90%) / 25/30 (83%) / 15/30 (50%) | - | - | - | - | - |
| 6 (B, nickname, reveal 25% after) | 30 | 23/30 (77%) / 16/30 (53%) / 9/30 (30%) | 26/30 (87%) / 22/30 (73%) / 3/30 (10%) | 22/30 (73%) / 16/30 (53%) / 14/30 (47%) | 18/30 (60%) / 9/30 (30%) / 0/30 (0%) | 18/30 (60%) / 13/30 (43%) / 8/30 (27%) | 7/30 (23%) / 4/30 (13%) / 0/30 (0%) |
| 7 (C, description, reveal 25% after) | 30 | 25/30 (83%) / 25/30 (83%) / 23/30 (77%) | 0/30 (0%) / 0/30 (0%) / 15/30 (50%) | 0/30 (0%) / 0/30 (0%) / 23/30 (77%) | 20/30 (67%) / 18/30 (60%) / 2/30 (7%) | 0/30 (0%) / 0/30 (0%) / 15/30 (50%) | 17/30 (57%) / 20/30 (67%) / 5/30 (17%) |
| 8 (D, object, reveal 25% after) | 30 | 22/30 (73%) / 16/30 (53%) / 10/30 (33%) | 26/30 (87%) / 24/30 (80%) / 3/30 (10%) | 24/30 (80%) / 21/30 (70%) / 16/30 (53%) | 15/30 (50%) / 11/30 (37%) / 2/30 (7%) | 24/30 (80%) / 15/30 (50%) / 11/30 (37%) | 11/30 (37%) / 3/30 (10%) / 0/30 (0%) |

### Reading the table

Stubs count as records. Run 0 columns use the proxies of run 1's step 3 (form 2 after reveal = form 2 in the form-1 record; both present = some record holds both names; in prompt = step 9's recall class). Cell 7's form 2 is an occupation phrase: it lands in descriptions, so 'both present' is 0 there and the bridge column is the measure that counts.

### Join direction (run 1b / run 1 / run 0)

| group | into the gold record | into a new form-2 record | never joined | no record |
| --- | --- | --- | --- | --- |
| cells 1-3 | 34 / 24 / 13 | 18 / 25 / 15 | 38 / 39 / 29 | 0 / 2 / 33 |
| cell 1 | 16 / 8 / 6 | 4 / 9 / 6 | 10 / 12 / 7 | 0 / 1 / 11 |
| cell 2 | 11 / 8 / 3 | 6 / 8 / 7 | 13 / 14 / 13 | 0 / 0 / 7 |
| cell 3 | 7 / 8 / 4 | 8 / 8 / 2 | 15 / 13 / 9 | 0 / 1 / 15 |
| cell 4 | 11 / 12 / 5 | 3 / 6 / 1 | 14 / 10 / 10 | 2 / 2 / 14 |
| cell 6 | 9 / 7 / 3 | 10 / 9 / 5 | 10 / 11 / 6 | 1 / 3 / 16 |
| cell 7 | 21 / 24 / 15 | 0 / 0 / 0 | 8 / 6 / 8 | 1 / 0 / 7 |
| cell 8 | 18 / 12 / 3 | 8 / 7 / 8 | 4 / 10 / 5 | 0 / 1 / 14 |

### Earlier record at the link step (the reveal; the gold chunk in cell 4), all cells with a reveal

| outcome | run 1b | run 1 | run 0 |
| --- | --- | --- | --- |
| in the prompt | 121 | 82 | 8 |
| cut by the cap | 32 | 27 | 0 |
| not matched by recall | 21 | 50 | 71 |
| no earlier record | 36 | 51 | 131 |

### Form 2 at the reveal, cells 1-3, 6 and 8 (n = 150): the model's decision and the final outcome

| outcome | run 1b | run 1 |
| --- | --- | --- |
| model NEW -> record | 84 | 80 |
| model NOT_AN_ENTITY -> not_an_entity | 23 | 29 |
| model SAME_AS -> joined (model) into the form-1 record | 17 | 18 |
| model NEW -> stub | 5 | 7 |
| model SAME_AS rejected (no_shared_token_or_sentence) -> record | 11 | 0 |
| form 2 not a candidate | 3 | 7 |
| model SAME_AS -> joined (model) into another record | 0 | 9 |
| model SAME_AS rejected (no_shared_token_or_sentence) -> stub | 4 | 0 |
| model SAME_AS rejected (type_conflict) -> record | 2 | 0 |
| model SAME_AS rejected (type_conflict) -> stub | 1 | 0 |

### Wrong direction: where form 1 sits in the separate form-2 record (all cells with a reveal)

| field | run 1b | run 1 |
| --- | --- | --- |
| name | 0 | 0 |
| alias | 5 | 5 |
| description | 34 | 42 |
| none | 0 | 0 |
| record not found | 0 | 0 |

### The three changes at work (counts over all books)

| measure | value |
| --- | --- |
| JSON records redirected into a SAME AS target | 2991 |
| auto-join (NEW equal to a shown entry) / blocked by a conflict / ambiguous | 633 / 1099 / 95 |
| JSON-alias join (optional switch) | 0 |
| model SAME AS rejected, by reason | {'no_shared_token_or_sentence': 1288, 'type_conflict': 559, 'title_conflict': 345} |
| model SAME AS accepted, by evidence | {'shared_token': 16164, 'same_sentence': 302} |
| NEW folded into a described record, rejected for a conflict | 292 |
| aliases rejected, by reason | {'not_a_span': 9811, 'title_conflict': 322, 'type_conflict': 36, 'descriptive': 7} |

### Merge audit: 30 of 17099 final SAME AS decisions (seed 42)

- B04 step 304: 'Oliver' -> 'Oliver Twist' [E0001] (model, shared_token): the chunk refers to him as the boy who is being addressed and whose tears fall, consistent with prior records
- B04 step 316: 'Oliver' -> 'Oliver Twist' [E0001] (model, shared_token): the chunk refers to him as "Oliver" and he is the boy who was brought to the prison cell
- B18 step 18: 'Tringham' -> 'Parson Tringham' [E0019] (model, shared_token): the antiquary who previously mentioned the d’Urbervilles, referred to as "Pa’son Tringham" in the chunk and previously recorded
- B18 step 143: 'The Gilded Antler' -> 'The Gilded Antler' [E0145] (model, shared_token): the chunk states "The Gilded Antler belonged to Wamanpuma", and the record E0145 identifies it as a public house on the high road at Kelmgate, which matches the context
- B26 step 30: 'Eliza' -> 'Eliza' [E0006] (model, shared_token): the chunk confirms she is the mother of Harry and is fleeing with him
- B26 step 89: 'George' -> 'George Harris' [E0044] (auto-join): auto-join: the title-stripped name equals a shown record's name or alias
- B27 step 198: 'London' -> 'London' [E0018] (model, shared_token): the chunk mentions London as the destination for Fanny's departure, matching the record's description
- B27 step 283: 'Crown' -> 'Crown' [E0289] (model, shared_token): the chunk mentions Mr. Crawford meeting an acquaintance at the Crown, matching the recalled record
- B31 step 72: 'free States' -> 'The river' [E0049] (model, same_sentence): the free States are referenced as the destination on the river, consistent with the river's role as a path to freedom
- B31 step 108: 'Jim' -> 'Jim' [E0015] (model, shared_token): the chunk refers to Jim as "Jim" and describes his actions and relationship to Huck, matching the record
- B31 step 139: 'Peter' -> 'Peter' [E0269] (model, shared_token): the chunk confirms Peter was the dying brother, left a letter, and had a family connection to Harvey and William
- B31 step 176: 'Jim' -> 'Jim' [E0015] (model, shared_token): the chunk uses "Jim" as a direct reference to the same character previously established
- B34 step 92: 'Mr. Elton' -> 'Mr. Elton' [E0017] (model, shared_token): the chunk describes his recent confession and current isolation, matching his prior record
- B34 step 121: 'Mr. Elton' -> 'Mr. Elton' [E0017] (model, shared_token): the chunk confirms Mr. Elton’s engagement to Miss Hawkins and his changed demeanor, matching his prior role as the one who wrote the charade
- B34 step 149: 'Mrs. Weston' -> 'Mrs. Weston' [E0006] (model, shared_token): the chunk shows her speaking about the carriage and forming suspicions, matching her role as a thoughtful observer
- B34 step 188: 'Miss Woodhouse' -> 'Emma' [E0001] (model, shared_token): the chunk refers to Emma as "Miss Woodhouse" and describes her reactions, matching the record
- B37 step 75: 'Darcy' -> 'Darcy' [E0021] (model, shared_token): recalled record with same name and description of having silent contempt for the Bennet family and being the subject of Mr. Collins's intended introduction
- B37 step 94: 'Jane' -> 'Jane' [E0042] (model, shared_token): the chunk refers to Jane as the sister whose happiness is at stake, and she is already recorded
- B37 step 187: 'Jane' -> 'Jane' [E0042] (model, shared_token): the chunk mentions Jane’s letter and her role in conveying the news about Lydia, consistent with her prior record
- B37 step 230: 'Elizabeth' -> 'Elizabeth' [E0022] (model, shared_token): the chunk refers to Elizabeth as the one hoping to be joined by Darcy, consistent with her role in the history
- B43 step 21: 'Hareton Earnshaw' -> 'Hareton Earnshaw' [E0007] (model, shared_token): he is the young man at Wuthering Heights and the last of the Earnshaws
- B43 step 81: 'Cathy' -> 'Miss Cathy' [E0019] (model, shared_token): the chunk uses "Cathy" as a nickname for Miss Cathy, and she is described as Catherine's daughter, matching E0019's record
- B43 step 115: 'Catherine' -> 'Catherine Earnshaw' [E0016] (model, shared_token): Catherine Earnshaw is the central figure in the emotional reunion, and her name is used interchangeably with "Cathy" and "Catherine Linton"
- B43 step 139: 'Hareton' -> 'Hareton Earnshaw' [E0007] (model, shared_token): the chunk refers to Hareton as a strong lad of eighteen, matching the record's description
- B43 step 149: 'Mr. Edgar' -> 'Master Linton' [E0053] (model, shared_token): the chunk refers to Mr. Edgar as the father of Cathy and the one who says "if I can get him" about Linton's return
- B44 step 6: 'Bessie' -> 'Bessie' [E0019] (model, shared_token): the chunk uses "Bessie" and the record E0019 identifies her as the nurse
- B44 step 74: 'John Reed' -> 'John' [E0021] (model, shared_token): John Reed is the cousin of Jane Eyre, previously mentioned in the history and now described in detail
- B44 step 147: 'AdÃ¨le' -> 'AdÃ¨le' [E0163] (model, shared_token): the chunk refers to her as Mr. Rochester’s ward and describes her interaction with Henry Lynn, matching the record
- B61 step 15: 'Nozdryov' -> 'Nozdryov' [E0004] (model, shared_token): the chunk describes him as a flamboyant, carefree man who exposes Tchitchikov, matching the prior record's description
- B61 step 274: 'Alexasha' -> 'Alexasha' [E0574] (model, shared_token): The chunk mentions Alexasha as the one who announces Platon Mihailovitch’s arrival, matching the stub

### Ten of 2192 rejected SAME AS (seed 42)

- B04 step 300: 'The man' -> 'Bill Sikes' rejected: no_shared_token_or_sentence ; model's reason: auto-join: the title-stripped name equals a shown record's name or alias
- B04 step 308: 'Monks' -> 'Monks' rejected: type_conflict person against organization; model's reason: auto-join: the title-stripped name equals a shown record's name or alias
- B18 step 2: 'T. H' -> 'Thomas Hardy' rejected: no_shared_token_or_sentence ; model's reason: the chunk identifies T. H as the author, matching the recalled record
- B26 step 17: 'General Washington' -> 'Uncle Tom' rejected: no_shared_token_or_sentence ; model's reason: the chunk describes a portrait of General Washington, and he is known from prior context as a historical figure; no new record needed
- B27 step 267: 'Mr. Campbell' -> 'Turner' rejected: type_conflict person against place; model's reason: the chunk identifies him as the surgeon of the Thrush, and he is not previously recorded but matches the context of William's friend and the ship's surgeon
- B27 step 273: 'H. C' -> 'Henry Crawford' rejected: no_shared_token_or_sentence ; model's reason: the chunk identifies H. C. as Henry Crawford, who is described in the history as the suitor and brother of Miss Crawford
- B37 step 140: 'Pemberley' -> 'Miss Darcy' rejected: type_conflict place against person; model's reason: auto-join: the title-stripped name equals a shown record's name or alias
- B43 step 51: 'Ellen' -> 'Nelly' rejected: no_shared_token_or_sentence ; model's reason: auto-join: the title-stripped name equals a shown record's name or alias
- B43 step 222: 'Catherine' -> 'Miss Cathy' rejected: no_shared_token_or_sentence ; model's reason: the chunk refers to her as "Cathy" and describes her emotional state and actions, matching the record's description of Miss Cathy
- B44 step 305: 'Marseilles' -> 'Lowood school' rejected: no_shared_token_or_sentence ; model's reason: the chunk contrasts Marseilles with England, and E0065 is a place in a different context but not relevant; however, Marseilles is not in the recalled records, and no prior mention exists—this is a new location

### Mr./Mrs. check and wrong merges

Records joining one stripped name under different titles (books' real characters): **6** (run 1: 120). Whether a sampled SAME AS is wrong is a reading, not a count the code can make: run 1's sample of 30 was read as 7 to 8 wrong; the sample above needs the same reading.

### Aliases in memory

Aliases in total / most on one record / mean per record: run 1b 1846 / 22 / 0.47; run 1 3955 / 80 / 1.44. The ten records with the most aliases in run 1b (for a reader to judge pollution):

- B61 'NIKOLAI GOGOL': ['Gogol', 'Nikolai Vasilyevich Gogol', 'Nikolai', 'Ivan Turgenev', 'Poe', 'Simon Karlinsky', 'Vsevolod Setchkarev', 'Joseph Frank', 'Victor Erlich', 'Robert Maguire', 'James Woodward', 'George Gibian', 'Diary of a Madman', 'Caesar', 'Dante', 'Mark Chagall', 'Chagall', 'Robert A. Maguire', 'Setchkarev, Vsevolod', 'Nikolai Gogol', 'Virgil', 'Nikolai Karamzin']
- B18 'd’Urbervilles': ['d’Urberville family', 'Urbervilles', 'Stoke-d’Urbervilles', 'The d’Urbervilles', 'Mr d’Urberville', 'old d’Urberville family', 'd’Urberville', 'knightly d’Urbervilles', 'Purbeck-marble and alabaster people', 'Knight of the Conqueror', 'ancient Norman family', 'd’Urberville dames', 'mansion of the d’Urbervilles', 'D’Urberville', 'd’Urberville mansion', 'Egdon Heath', 'd’Urberville Aisle', 'd’Urberville Window', 'd’Urberville knights and dames']
- B18 'Tess of the d’Urbervilles': ['Tess Durbeyfield', 'Tess', 'Simple Tess Durbeyfield', 'Queen of Spades', 'Queen of Diamonds', 'Car', 'Tess d’Urberville', 'otherwise d’Urberville', 'The girl-mother', 'Tessy', 'Durbeyfield household', 'Nazareth', 'Mrs Angel Clare', 'Jesus Christ', 'Tess’s father and mother', 'T', 'Durbeyfield', 'Miss Durbeyfield']
- B04 'Oliver Twist': ['Oliver', 'young Oliver Twist', 'The boy', 'The Artful Dodger', 'Dodger', 'the Dodger', 'The Dodger', 'Brownlow', 'Mr. Brownlow', 'Little Oliver Twist', 'Little Oliver', 'Nolly', 'Ned', 'Hampton', 'Young Twist', 'young Oliver', 'Master Oliver']
- B26 'George Harris': ['George', 'Mr. Harris', 'young Mas’r Tom', 'Mas’r', 'Tom', 'young Master George', 'Mas’r George', 'Mas’r Georgey', 'Master George', 'Missis', 'Mr. Wilson', 'The old gentleman', 'This young man', 'Jim', 'George Harris', 'Harris']
- B43 'Master Linton': ['Edgar', 'Master Linton', 'Edgar Linton', 'Mr. Edgar', 'Linton', 'Maister Linton', 'Linton Heathcliff', 'Pretty Linton', 'Heathcliff', 'Master Heathcliff', 'Young master', 'My master', 'His nephew', 'Papa']
- B43 'Catherine Earnshaw': ['Catherine', 'Catherine Linton', 'Miss Catherine', 'Edgar', 'Edgar Linton', 'Linton', 'Miss Earnshaw', 'Miss Cathy', 'Cathy', 'Mr. Edgar', 'Catherine Heathcliff', 'My father', 'Catherine Earnshaw']
- B61 'Tchitchikov': ['Chichikov', 'Pavel Ivanovitch Tchitchikov', 'Pavel Ivanovitch', 'Madame Manilov', 'Manilov', 'Our hero', 'Gremin', 'Zvonsky', 'Lidin', 'Tchitchikov’s father', 'Tchitchikov’s horses', 'Tchitchikov', 'Platonov']
- B04 'Jack Dawkins': ['The Artful Dodger', 'Mr. Dawkins', 'Mr. John Dawkins', 'Artful Dodger', 'Dodger', 'Artful', 'The Artful', 'the Dodger', 'Dawkins', 'Jack', 'Jack Dawkins']
- B04 'Fagin': ['the Jew', 'the Dodger', 'Fagin', 'The Jew', 'Jew', 'Mr. Fagin', 'the old Jew', 'Faguey', 'old Fagin', 'young Jew', 'the crafty Jew']

## Anything unexpected

Nothing unexpected.

## Files written

- `/mnt/data/zeinab/SAADI/experiments/method_update/p1b_ledger/reports/step_03_run1_code.md`
- `/mnt/data/zeinab/SAADI/outputs/method_update/p1b_ledger/personas_memory.jsonl`
- `/mnt/data/zeinab/SAADI/outputs/method_update/p1b_ledger/same_as_decisions.jsonl`
- `/mnt/data/zeinab/SAADI/outputs/method_update/p1b_ledger/title_conflicts.jsonl`
