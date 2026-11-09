# p1b_ledger - step 01: Candidate net at threshold 0.3 (CPU)

Status: **ok** - 2026-11-08 20:48:54

## What was done

- `bash experiments/method_update/p1b_ledger/run_all.sh  (stage 01_gate; python stage01_gate.py )`
- `git commit e4373b889608b1a08574bc5205c75cf7f863b5a0`
- `candidate net: gliner model=urchade/gliner_large-v2.1 device=cpu threshold=0.3 labels=['person', 'place', 'organization', 'named object', 'title']`
- Versions: python=3.12.3, platform=Linux-6.17.0-23-generic-x86_64-with-glibc2.39, numpy=2.3.5, pandas=3.0.1, yaml=6.0.3, saadi=0.1.0, git_commit=e4373b889608b1a08574bc5205c75cf7f863b5a0, torch=2.11.0+cu129, transformers=4.57.6, gliner=0.2.30, vllm=0.23.0

## Numbers

### Frozen spec

personas.jsonl SHA-256 verified: `11c4c1696c67171f78821d3ba8311ccd28c851315ca1c2a404a9a422593897bb` (same spec as runs 0 and 1).

### Candidate net

`urchade/gliner_large-v2.1` on `cpu`, loaded in 12.3 s; labels ['person', 'place', 'organization', 'named object', 'title']; threshold 0.3; versions {'torch': '2.11.0+cu129', 'gliner': '0.2.30', 'transformers': '4.57.6'}.

### Names caught per family: threshold 0.3 (run 1b) against 0.5 (run 1)

| family | form 1 at 0.3 | form 1 at 0.5 (run 1) | form 2 at 0.3 | form 2 at 0.5 (run 1) |
| --- | --- | --- | --- | --- |
| A surname | 150/150 (100%) | 143/150 (95%) | 114/120 (95%) | 110/120 (92%) |
| B nickname | 28/30 (93%) | 21/30 (70%) | 30/30 (100%) | 29/30 (97%) |
| C occupation | 30/30 (100%) | 30/30 (100%) | 0/30 (0%) | 0/30 (0%) |
| D owned object | 25/30 (83%) | 20/30 (67%) | 30/30 (100%) | 29/30 (97%) |

### Names caught per cell: threshold 0.3 (run 1b) against 0.5 (run 1)

| cell | label | form 1 at 0.3 | form 1 at 0.5 | form 2 at 0.3 | form 2 at 0.5 |
| --- | --- | --- | --- | --- | --- |
| 1 | A, reveal 3 chunks after the gold | 30/30 (100%) | 28/30 (93%) | 30/30 (100%) | 29/30 (97%) |
| 2 | A, reveal 25% of the book after | 30/30 (100%) | 29/30 (97%) | 29/30 (97%) | 29/30 (97%) |
| 3 | A, reveal 55% of the book after | 30/30 (100%) | 30/30 (100%) | 28/30 (93%) | 27/30 (90%) |
| 4 | A, reveal 20% before the gold (control) | 30/30 (100%) | 27/30 (90%) | 27/30 (90%) | 25/30 (83%) |
| 5 | A, no reveal (control) | 30/30 (100%) | 29/30 (97%) | - | - |
| 6 | B, nickname, reveal 25% after | 28/30 (93%) | 21/30 (70%) | 30/30 (100%) | 29/30 (97%) |
| 7 | C, description, reveal 25% after | 30/30 (100%) | 30/30 (100%) | 0/30 (0%) | 0/30 (0%) |
| 8 | D, object, reveal 25% after | 25/30 (83%) | 20/30 (67%) | 30/30 (100%) | 29/30 (97%) |

### Totals and cost: threshold 0.3 (run 1b) against 0.5 (run 1)

| measure | run 1b (0.3) | run 1 (0.5) |
| --- | --- | --- |
| form 1 caught | 233/240 (97%) | 214/240 (89%) |
| form 2 caught | 174/210 (83%) | 168/210 (80%) |
| candidates per chunk (mean) | 10.0 | 9.0 |
| false-candidate proxy per chunk (common word / title / no letters) | 0.85 | 0.61 |
| chunks at the cap of 40 candidates | 0 | - |
| seconds per chunk on CPU (mean) | 1.53 | 1.16 |
| candidates per label | {'person': 3297, 'place': 763, 'named object': 153, 'title': 146, 'organization': 125} | {'person': 3022, 'place': 694, 'title': 119, 'named object': 112, 'organization': 83} |

### Form 2 missed at 0.3

36 names; by family {'A': 6, 'C': 30}. Family C form 2 is an occupation phrase, which a name tagger does not list (as in run 1); the recall of run 1b no longer depends on the candidate list, so a missed name can still bring its record back by its tokens.

### System prompt of the LLM call

Identical to run 1's (`saadi.ledger.prompt.LEDGER_SYSTEM_PROMPT`, unchanged): yes. Full text: `/mnt/data/zeinab/SAADI/outputs/method_update/p1b_ledger/gate/system_prompt.txt`.

### One filled user prompt with run 1b's recall (B31, gold chunk of persona c1_B31_000; run 0's memory and history after chunk B31:36 (its records treated as ledger records))

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
{"mode": "hierarchical_summary_p1_ledger", "branching_factor": 5, "current_step": 37, "description": "Top-down processed history before CURRENT_CHUNK. Higher L values are coarser summaries; L0 values are recent materialized chunks. Boundary steps may include the immediate previous L0 even when a higher-level summary also covers it.", "selected_steps_by_level": {"L2": [24], "L1": [29, 34], "L0": [35, 36]}, "selected_context_by_level": {"L2": [{"step": 24, "text": "Huck Finn escapes his abusive fa ... [5640 more characters; full text in example_user_prompt.txt]

[RECALLED_RECORDS_FOUND_BY_WORD_OVERLAP_JSON]
[{"id": "E0001", "name": "The Adventures of Huckleberry Finn", "description": ["A novel by Mark Twain set in the Mississippi Valley, featuring the protagonist Huckleberry Finn", "Contains multiple dialects used to reflect regional speech patterns"], "aliases": ["Huckleberry Finn", "The Adventures of Huckleberry Finn (Twain)"]}, {"id": "E0006", "name": "Miss Watson", "description": ["The widow's sister, a slim old maid with goggles who lives with the widow and teaches Huckleberry Finn", "Proposed as a target for punishment by the gang due to Huck Finn's lack of a family", "Was targeted by Tom Sawyer's gang as a punishment for Huck's lack of family"], "aliases": ["Miss Watson", "old Miss Watson"]}, {"id": "E0011", "name": "Huckleberry Finn", "description": ["The protagonist of The Adventures of Huckleberry Finn, seeking freedom from the Widow Douglas’s attempts to 'sivilize' him", "Joined Tom Sawyer's Gang after offering Miss Watson as a family to be punished", "Attempted to summon genies using an old tin lamp and iron ring based on Tom Sawyer's stories"], "aliases": ["Huck Finn", "Huckleberry Finn"]}, {"id": "E0016", "name": "Pap", "description": ["Demands money from Huck and threatens to punish him", "The father of Huckleberry Finn, a drunkard who beats Huck when sober and locks him in the cabin", "Huckleberry Finn's father, a drunkard who beats Huck when sober and locks him in the cabin"], "aliases": ["Pap", "the old man"]}, {"id": "E0027", "name": "free nigger", "description": ["A Black man from Ohio who is described as a mulatto, well-dressed, and able to vote in his home state", "A man of mixed race who is educated, owns fine clothes and a gold watch, and is a college professor"], "aliases": ["free nigger", "nigger from Ohio"]}]

[RECALLED_RECORDS_WHOSE_NAME_APPEARS_IN_THIS_CHUNK_JSON]
[{"id": "E0008", "name": "good place", "kind": "record", "description": ["A religious afterlife destination described as a place where people sing with harps forever"], "aliases": ["the good place"]}, {"id": "E0021", "name": "old tin lamp", "kind": "record", "description": ["An object Huckleberry Finn used to try to summon genies, based on Tom Sawyer's stories about magic"], "aliases": ["tin lamp", "old tin lamp"]}, {"id": "E0041", "name": "quick silver", "kind": "record", "description": ["A metal used in bread to make it float toward a drowned body", "Believed to attract the body of a drowned person"], "aliases": ["quick silver", "mercury"]}, {"id": "E0045", "name": "camp fire", "kind": "record", "description": ["A fire built by Huck Finn to cook food and provide warmth", "Used for cooking supper and later for evening relaxation"], "aliases": ["campfire", "fire"]}]

[CANDIDATES]
1. Abolitionist (place)
2. Ole missus (person)
3. Miss Watson (person)
4. Orleans (place)
5. Doctor Sinchipuma (person)
6. Gavelhall (place)
7. The Adventures of Huckleberry Finn (title)
8. Create (organization)
9. Huck (person)
10. De yuther servants (person)

[OUTPUT_SCHEMA]
{"active_entities": [{"name": "string", "description": ["string"], "aliases": ["string"]}], "insertions": [{"name": "same string as the active entity name", "anchor_text": "string", "left_context": "string", "right_context": "string"}]}

[INSTRUCTIONS]
First write the decision block: one line per candidate, in order, between [DECISIONS] and [END_DECISIONS], each line of the form `<n>. <candidate> -> NEW` or `<n>. <candidate> -> SAME AS <id> (<reason>)` or `<n>. <candidate> -> NOT AN ENTITY`. Then return the JSON object with keys active_entities and insertions for at most 5 records you describe, with exactly one insertion for each active entity and no inserted-text field. Then [END_JSON].
```

### Hand-off

Reported only; run_all.sh starts the build next.

## Anything unexpected

Nothing unexpected.

## Files written

- `/mnt/data/zeinab/SAADI/outputs/method_update/p1b_ledger/gate/candidates.jsonl`
- `/mnt/data/zeinab/SAADI/outputs/method_update/p1b_ledger/gate/recall_rows.jsonl`
- `/mnt/data/zeinab/SAADI/outputs/method_update/p1b_ledger/gate/system_prompt.txt`
- `/mnt/data/zeinab/SAADI/outputs/method_update/p1b_ledger/gate/example_user_prompt.txt`
