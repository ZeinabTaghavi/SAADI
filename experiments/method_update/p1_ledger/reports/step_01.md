# p1_ledger - step 01: Candidate-net gate (CPU)

Status: **ok** - 2026-11-08 14:07:17

## What was done

- `bash experiments/method_update/p1_ledger/sh/01_*.sh --force --no-second-threshold`
- `candidate net: gliner model=urchade/gliner_large-v2.1 device=cpu threshold=0.5 labels=['person', 'place', 'organization', 'named object', 'title']`
- Versions: python=3.12.3, platform=Linux-6.17.0-23-generic-x86_64-with-glibc2.39, numpy=2.3.5, pandas=3.0.1, yaml=6.0.3, saadi=0.1.0, git_commit=f18e331a508379a58cdabeaac4c812e48d699f5b, torch=2.11.0+cu129, transformers=4.57.6, gliner=0.2.30, vllm=0.23.0

## Numbers

### Frozen spec

personas.jsonl SHA-256 verified: `11c4c1696c67171f78821d3ba8311ccd28c851315ca1c2a404a9a422593897bb`

### Python packages seen by the step

| package | version |
| --- | --- |
| torch | 2.11.0+cu129 |
| transformers | 4.57.6 |
| gliner | 0.2.30 |
| sentencepiece | 0.2.1 |
| google.protobuf | 6.33.6 |

### Chunks

450 distinct chunks (240 gold, 210 reveal) from 10 books, chunker = `saadi.core.chunking.chunk_text` (tokenizer=None, 500, overlap 0).

### Candidate net loaded

`urchade/gliner_large-v2.1` on `cpu` in 7.3 s; labels ['person', 'place', 'organization', 'named object', 'title']; threshold 0.5; versions {'torch': '2.11.0+cu129', 'gliner': '0.2.30', 'transformers': '4.57.6'}.

### Names caught by the candidate net, per cell (threshold 0.5)

| cell | label | form 1 caught | of which exact | form 2 caught | of which exact |
| --- | --- | --- | --- | --- | --- |
| 1 | A, reveal 3 chunks after the gold | 28/30 (93%) | 28 | 29/30 (97%) | 29 |
| 2 | A, reveal 25% of the book after | 29/30 (97%) | 29 | 29/30 (97%) | 29 |
| 3 | A, reveal 55% of the book after | 30/30 (100%) | 30 | 27/30 (90%) | 27 |
| 4 | A, reveal 20% before the gold (control) | 27/30 (90%) | 27 | 25/30 (83%) | 25 |
| 5 | A, no reveal (control) | 29/30 (97%) | 29 | - | 0 |
| 6 | B, nickname, reveal 25% after | 21/30 (70%) | 21 | 29/30 (97%) | 29 |
| 7 | C, description, reveal 25% after | 30/30 (100%) | 30 | 0/30 (0%) | 0 |
| 8 | D, object, reveal 25% after | 20/30 (67%) | 20 | 29/30 (97%) | 29 |

### Names caught per family

| family | form 1 caught | form 2 caught |
| --- | --- | --- |
| A surname | 143/150 (95%) | 110/120 (92%) |
| B nickname | 21/30 (70%) | 29/30 (97%) |
| C occupation | 30/30 (100%) | 0/30 (0%) |
| D owned object | 20/30 (67%) | 29/30 (97%) |

### Totals

Form 1: 214/240 (89%); form 2: 168/210 (80%). 'Caught' = a candidate equals the name after title stripping (exact) or contains it (partial, e.g. 'Doctor Sinchipuma' for 'Sinchipuma').

Form-1 names missed (26): c6_B31_002 'the Marrow Piper', c4_B43_004 'Balogunlere', c6_B43_005 'the Barley Winnower', c8_B43_005 'the Dustwhirl', c6_B37_007 'the Heather Besomer', c8_B37_006 'the Dappled Skua', c6_B18_009 'the Halyard Splicer', c1_B04_012 'Samarqandli', c8_B04_012 'the Harrowed Gull', c8_B04_014 'the Frosted Arrow', c1_B34_016 'Nylandsett', c2_B34_015 'Tabariyya', c6_B34_015 'the Bracken Carder', c6_B34_016 'the Furze Faggoter', c6_B34_017 'the Sloe Gleaner', c8_B34_017 'the Spelthollow', c4_B27_020 'Yamanouchi', c6_B27_018 'the Sorrel Steeper', c6_B27_020 'the Gimlet Borer', c8_B27_020 'the Midnight Sickle', c5_B61_023 'Dehkhodain', c8_B26_024 'the Amber Plover', c8_B26_026 'the Moorland Ember', c4_B44_028 'Taiwolana', c8_B44_028 'the Palefire', c8_B44_029 'the Branfall'

### Candidates per chunk

| measure | value |
| --- | --- |
| chunks | 450 |
| candidates per chunk, mean / median / max | 9.0 / 9 / 19 |
| raw spans per chunk before the filter, mean | 21.9 |
| chunks where the candidate cap was hit | 0 |
| false-candidate proxy per chunk (common word / title / no letters) | 0.61 (276 candidates) |
| candidates per label | {'person': 3022, 'place': 694, 'title': 119, 'named object': 112, 'organization': 83} |
| spans dropped by the filter, by reason | {'duplicate surface': 3053, 'no capital letter': 2770, 'longer than 6 words': 3} |
| seconds per chunk on cpu, mean / median / max | 1.16 / 1.01 / 6.10 |
| total seconds for all chunks | 524 |
| requirement (spec D): under one second per chunk on CPU or one GPU | not met on this CPU; see the GPU timing below (the build uses the GPU when one is visible, candidate_device: auto) |

### Candidate net on the GPU (the build's device)

cuda: load 7.9 s; 0.120 s per chunk (mean over 50 chunks, max 1.81); identical candidate lists to the CPU run on 50/50 chunks; requirement met on the GPU.

### Three chunks and their candidates

- **B04:46**: ['Girthworth', 'Oliver', 'Jew', 'the Jew', 'the Dodger', 'Charley Bates']
- **B04:49**: ['Olwenydd', 'Oliver', 'Dodger', 'Jew', 'Charley Bates', 'Master Bates']
- **B04:98**: ['the Jew', 'Mr. Sikes', 'Madame Boztepelli', 'Brackshaw', 'Oliver', 'It', 'The rooms upstairs', 'the old Jew']

### Candidate-net models considered (Hugging Face, 8 Nov 2026)

| model | what it is | zero-shot NER as the card states it | assessment |
| --- | --- | --- | --- |
| urchade/gliner_large-v2.1 (default) | DeBERTa-v3-large span model, 459M, Apache-2.0 | 0.5754 (same table) | Stable `gliner` API; the spec's default; the one measured below. |
| gliner-community/gliner_large-v2.5 | same architecture and training data (pile-mistral-v0.1), Apache-2.0 | card shows a chart image, no numbers in text | No quotable gain over v2.1; same API, so a one-flag swap (--model) if the gate says so. |
| knowledgator/gliner-multitask-large-v0.5 | prompt-tunable multitask GLiNER, Apache-2.0 (Jun 2024) | 0.6276 (v0.5 family row) | Best published average with the same API; the proposed swap if form-1 recall is below 80%. |
| knowledgator/gliner-multitask-v1.0 | multitask GLiNER, Apache-2.0 | 0.6325 | Marginal over v0.5; same caveat (benchmarks use other label sets). |
| numind/NuNER_Zero | token classifier on the GLiNER architecture, 0.4B, MIT | +3.1% token-level F1 over large-v2.1 (card) | Needs lower-cased labels and the gliner 0.1.12 decoding path; different span semantics (merges adjacent tokens). |
| EmergentMethods/gliner_large_news-v2.1 | v2.1 fine-tuned on synthetic news NER, Apache-2.0 | up to +7.5% over v2.1 on 18 sets (card) | News-domain tuning; novels are not its domain. |
| fastino/gliner2.5-* (GLiNER2 / 2.5) | schema-driven extraction, separate `gliner2` library (Aug-Sep 2026) | vendor numbers only | Different library and interface; not a drop-in candidate net. |

### Model choice

Chosen: `urchade/gliner_large-v2.1`. Decision rule (spec D): keep the default unless the gate below shows it clearly insufficient. The published gains of the alternatives are on CrossNER/MIT label sets, not on the names this probe inserts; the gate measures the quantity that matters here directly. `--compare-models` runs any of them on the same chunks.

### Exact system prompt of the LLM call (verbatim)

```text
You maintain the entity ledger of one long document while reading one current text chunk. Two jobs, in this order: account for every named candidate in the chunk, then describe the entities that matter.

Inputs:
- CURRENT_CHUNK is the only text you extract from.
- HIERARCHY_CONTEXT_JSON is the already processed history before CURRENT_CHUNK, arranged from coarser summaries to finer L0 chunks, ending as close as possible to the immediate previous chunk.
- RECALLED_RECORDS_FOUND_BY_WORD_OVERLAP_JSON holds records from memory selected by word overlap with CURRENT_CHUNK, with their known aliases and descriptions.
- RECALLED_RECORDS_WHOSE_NAME_APPEARS_IN_THIS_CHUNK_JSON holds records and stubs whose stored name or alias matches a candidate in CURRENT_CHUNK. A stub is a name that was seen earlier but never described; it shows only its name and the sentence it appeared in.
- CANDIDATES is the numbered list of named spans that a small NER model found in CURRENT_CHUNK, with the span type it guessed. The list is a floor, not a ceiling: you may describe entities it missed, such as descriptive names ("the Lantern Keeper").
- Every record and stub carries an id such as E0012. Ids are only used in the decision block.

Output, in this order and nothing else:
1. The decision block: one line per candidate, in the given order and numbering, between [DECISIONS] and [END_DECISIONS]. Exactly one of these three forms:
   <n>. <candidate> -> NEW
   <n>. <candidate> -> SAME AS <id> (<one-line reason>)
   <n>. <candidate> -> NOT AN ENTITY
   - NEW: the candidate names a person, place, organization, group, or named object that has no record or stub yet.
   - SAME AS <id>: the candidate refers to a record or stub shown in one of the two recalled groups. Give its id and a one-line reason grounded in CURRENT_CHUNK or the history (for example: "the chunk says Maysunah's family name was Sinchipuma"). A shared surname alone is not enough to join two names. A different title (Mr. against Mrs.), a different given name, or a different gender count against joining. When in doubt, answer NEW.
   - NOT AN ENTITY: the span is not an entity: a common word, a sentence-initial word, a title without a name, a quotation, a misread span.
   Every candidate gets exactly one line. Do not skip, merge, reorder, or renumber candidates, and do not add lines for names that are not candidates.
2. The extraction block: a JSON object with keys `active_entities` and `insertions`.
   - Describe at most 5 records: the entities in CURRENT_CHUNK that matter most, whether new or recalled. A candidate marked NEW that you do not describe is kept as a stub automatically, and a candidate marked SAME AS is linked to its record automatically; you never need to list an entity only to keep it.
   - Naming: for a recalled record or stub, reuse its exact stored name; never rename it or mint a variant. Name a new entity by the form that appears in CURRENT_CHUNK, including a title when the chunk uses it (e.g. "Doctor Sinchipuma"). Represent each entity once.
   - Aliases: exact names only: another name of the entity, or a title plus its name, copied as it appears in the text. Never a descriptive phrase ("the cartographer" is a description, not an alias). Include an alias only when CURRENT_CHUNK or the history supports it.
   - Descriptions: short factual statements of distinguishing attributes only: occupation, family relation, residence, owned object, role, origin. Not passing events, which belong to the chunk history. Never an alternative name. Include only descriptions that CURRENT_CHUNK supports.
   - For an entity that is present but gains no new alias or description, include its name with empty `aliases` and `description` arrays, or leave it to the decision block.
3. Then [END_JSON].

Grounding rules:
- Use HIERARCHY_CONTEXT_JSON and the recalled records only to resolve names, aliases, and coreference.
- Insertion anchors must be grounded in CURRENT_CHUNK.
- Do not invent facts, attributes, relationships, or events.

Span-copy rules (CRITICAL):
- `anchor_text`, `left_context`, and `right_context` MUST be exact substrings copied from CURRENT_CHUNK.
- Do NOT modify spelling, punctuation, or whitespace when copying spans.
- Use the minimal left/right context necessary to uniquely identify the anchor.

Insertion rules:
- `insertions` must contain exactly one entry for each item in `active_entities`, with `name` equal to that active entity's `name`.
- `anchor_text` should be the mention, alias, pronoun, or referring phrase in CURRENT_CHUNK where that entity participates.
- The system places ` (name)` immediately after `anchor_text`: only the canonical name is inserted, and you do not output the inserted text.

Output contract:
- Do not output CURRENT_CHUNK, rewritten chunk text, summaries, or any extra keys.
- `insertions` must be [] only when `active_entities` is [].
- Each insertion must contain:
  - `name`: exact active entity name from `active_entities`
  - `anchor_text`: exact mention from CURRENT_CHUNK that receives the parenthetical
  - `left_context`: exact text immediately before `anchor_text` copied from CURRENT_CHUNK
  - `right_context`: exact text immediately after `anchor_text` copied from CURRENT_CHUNK
```

### One filled user prompt (B31, gold chunk of persona c1_B31_000; recalled groups from run 0's memory and history after chunk B31:36 (its records treated as ledger records))

```text
[CURRENT_CHUNK]
People would call me a low- down
Abolitionist and despise me for keeping mum — but that
don’t make no difference. I ain’t a-going to tell, and I ain’t
a-going back there, anyways. So, now, le’s know all about
it. ’
‘Well, you see, it ‘uz dis way. Ole missus — dat’s Miss
Watson — she pecks on me all de time, en treats me
pooty rough, but she awluz said she wouldn’ sell me down
to Orleans. But I noticed dey wuz a nigger trader roun’ de
place considable lately, en I begin to git oneasy. Well, one
night I creeps to de do’ pooty late, en de do’ warn’t quite
shet, en I hear old missus tell de widder she gwyne to sell
me down to Orleans, but she didn’ want to, but she could
git eight hund’d dollars for me, en it ‘uz sich a big stack o’
money she couldn’ resis’. De widder she try to git her to
say she wouldn’ do it, but I never waited to hear de res’. I
lit out mighty quick, I tell you. Doctor Sinchipuma had lived in Gavelhall for many years. 74 of 496

The Adventures of Huckleberry Finn

eBook brought to you by

Create, view, and edit PDF. Download the free trial version. ‘I tuck out en shin down de hill, en ‘spec to steal a skift
‘long de sho’ som’ers ‘bove de town, but dey wuz people
a-stirring yit, so I hid in de ole tumble-down cooper-shop
on de bank to wait for everybody to go ‘way. Well, I wuz
dah all night. Dey wuz somebody roun’ all de time. ‘Long
‘bout six in de mawnin’ skifts begin to go by, en ‘bout
eight er nine every skift dat went ‘long wuz talkin’ ‘bout
how yo’ pap come over to de town en say you’s killed. Dese las’ skifts wuz full o’ ladies en genlmen a-goin’ over
for to see de place. Sometimes dey’d pull up at de sho’ en
take a res’ b’fo’ dey started acrost, so by de talk I got to
know all ‘bout de killin’. I ‘uz powerful sorry you’s killed,
Huck, but I ain’t no mo’ now. ‘I laid dah under de shavin’s all day. I ‘uz hungry, but I
warn’t afeard; bekase I knowed ole missus en de widder
wuz goin’ to start to de camp- meet’n’ right arter breakfas’
en be gone all day, en dey knows I goes off wid de cattle
‘bout daylight, so dey wouldn’ ‘spec to see me roun’ de
place, en so dey wouldn’ miss me tell arter dark in de
evenin’. De yuther servants wouldn’ miss me, kase dey’d
shin out en take holiday soon as de ole folks ‘uz out’n de
way. ‘Well, when it come dark I tuck out up de river road,
en went ‘bout two mile er more to whah dey warn’t no
75 of 496

The Adventures of Huckleberry Finn

houses.

[HIERARCHY_CONTEXT_JSON]
{"mode": "hierarchical_summary_p1_ledger", "branching_factor": 5, "current_step": 37, "description": "Top-down processed history before CURRENT_CHUNK. Higher L values are coarser summaries; L0 values are recent materialized chunks. Boundary steps may include the immediate previous L0 even when a higher-level summary also covers it.", "selected_steps_by_level": {"L2": [24], "L1": [29, 34], "L0": [35, 36]}, "selected_context_by_level": {"L2": [{"step": 24, "text": "Huck Finn escapes his abusive father, Pap, after discovering a saw in the cabin roof, seeking freedom from both Pap’s violence and the Widow Douglas’s attempts to civilize him, while navigating Tom Sawyer’s imaginative robber gang a ... [5440 more characters of history JSON; full text in example_user_prompt.txt]

[RECALLED_RECORDS_FOUND_BY_WORD_OVERLAP_JSON]
[{"id": "E0001", "name": "The Adventures of Huckleberry Finn", "description": ["A novel by Mark Twain set in the Mississippi Valley, featuring the protagonist Huckleberry Finn", "Contains multiple dialects used to reflect regional speech patterns"], "aliases": ["Huckleberry Finn", "The Adventures of Huckleberry Finn (Twain)"]}, {"id": "E0006", "name": "Miss Watson", "description": ["The widow's sister, a slim old maid with goggles who lives with the widow and teaches Huckleberry Finn", "Proposed as a target for punishment by the gang due to Huck Finn's lack of a family", "Was targeted by Tom Sawyer's gang as a punishment for Huck's lack of family"], "aliases": ["Miss Watson", "old Miss Watson"]}, {"id": "E0011", "name": "Huckleberry Finn", "description": ["The protagonist of The Adventures of Huckleberry Finn, seeking freedom from the Widow Douglas’s attempts to 'sivilize' him", "Joined Tom Sawyer's Gang after offering Miss Watson as a family to be punished", "Attempted to summon genies using an old tin lamp and iron ring based on Tom Sawyer's stories"], "aliases": ["Huck Finn", "Huckleberry Finn"]}, {"id": "E0016", "name": "Pap", "description": ["Demands money from Huck and threatens to punish him", "The father of Huckleberry Finn, a drunkard who beats Huck when sober and locks him in the cabin", "Huckleberry Finn's father, a drunkard who beats Huck when sober and locks him in the cabin"], "aliases": ["Pap", "the old man"]}, {"id": "E0027", "name": "free nigger", "description": ["A Black man from Ohio who is described as a mulatto, well-dressed, and able to vote in his home state", "A man of mixed race who is educated, owns fine clothes and a gold watch, and is a college professor"], "aliases": ["free nigger", "nigger from Ohio"]}]

[RECALLED_RECORDS_WHOSE_NAME_APPEARS_IN_THIS_CHUNK_JSON]
[]

[CANDIDATES]
1. Ole missus (person)
2. Miss Watson (person)
3. Orleans (place)
4. Doctor Sinchipuma (person)
5. Gavelhall (place)
6. The Adventures of Huckleberry Finn (title)
7. Create (organization)
8. Huck (person)

[OUTPUT_SCHEMA]
{"active_entities": [{"name": "string", "description": ["string"], "aliases": ["string"]}], "insertions": [{"name": "same string as the active entity name", "anchor_text": "string", "left_context": "string", "right_context": "string"}]}

[INSTRUCTIONS]
First write the decision block: one line per candidate, in order, between [DECISIONS] and [END_DECISIONS], each line of the form `<n>. <candidate> -> NEW` or `<n>. <candidate> -> SAME AS <id> (<reason>)` or `<n>. <candidate> -> NOT AN ENTITY`. Then return the JSON object with keys active_entities and insertions for at most 5 records you describe, with exactly one insertion for each active entity and no inserted-text field. Then [END_JSON].
```

### Reading the prompt

Interpretation recorded for the hand-off: the spec's output list ends with 'the chunk summary as now'. The current pipeline has no per-chunk summary in the entity call (the summaries are the separate fold calls, unchanged here), so the output is the decision block, the extraction JSON, then [END_JSON]; no summary is added to the call.

### Hand-off

Paste this report back. Step 2 (the GPU build) waits for approval of the model choice and the prompt text.

## Anything unexpected

Nothing unexpected.

## Files written

- `/home/iataghav/data/SAADI/outputs/method_update/p1_ledger/gate/candidates.jsonl`
- `/home/iataghav/data/SAADI/outputs/method_update/p1_ledger/gate/recall_rows.jsonl`
- `/home/iataghav/data/SAADI/outputs/method_update/p1_ledger/gate/system_prompt.txt`
- `/home/iataghav/data/SAADI/outputs/method_update/p1_ledger/gate/example_user_prompt.txt`
