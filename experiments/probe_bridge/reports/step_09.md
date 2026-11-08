# Bridge probe - step 09: Diagnose: where are the names lost

Status: **ok** - 2026-11-08 10:16:22

## What was done

- `bash experiments/probe_bridge/sh/09_*.sh`
- `read /mnt/data/zeinab/SAADI/outputs/probe_bridge/novelhopqa_probe/indexes/59bb6acf00d222d8/llm_cache/novelhopqa/hierarchical/branching_factor_5/chunk_500/AE_5_2_3/Qwen3-30B-A3B-Instruct-2507/AEWord/index.json; memory per step replayed by the cache loader; recall re-run with {'selector': 'word', 'max_entities': 5, 'max_aliases': 2, 'max_desc': 3, 'min_relevance': 0.0} (/mnt/data/zeinab/SAADI/outputs/probe_bridge/novelhopqa_probe/runs/probe_bridge/greedy_seed42/8792ffaf376e7d89/runtime_config.yaml)`
- Versions: python=3.12.3, platform=Linux-6.17.0-23-generic-x86_64-with-glibc2.39, numpy=2.3.5, pandas=3.0.1, yaml=6.0.3, saadi=0.1.0, git_commit=57626498a8010643d79adda06e5376480d6284df

## Numbers

### Matching and reconstruction

Titles stripped before names are compared: Dr., Doctor, Captain, Mrs., Mr., Professor, Father, Judge, Madame, Sergeant, Colonel. A form is present in a field when all its content tokens occur there (step 6's rule). The memory before each step is the loader's replay of the earlier steps; the recalled records are the recall rule re-run on that memory and the chunk text.

### Q1. Names in the model's reply but absent from the final memory (119)

| class | form 1 | form 2 | total | cell 1 | cell 2 | cell 3 | cell 4 | cell 6 | cell 7 | cell 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| budget cut | 25 | 19 | 44 | 7 | 5 | 5 | 11 | 9 | 2 | 5 |
| field cap cut | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| merge loss | 4 | 0 | 4 | 0 | 0 | 0 | 0 | 2 | 0 | 2 |
| parser loss | 1 | 2 | 3 | 0 | 0 | 1 | 0 | 0 | 0 | 2 |
| not proposed as a record | 18 | 50 | 68 | 9 | 10 | 12 | 6 | 13 | 8 | 10 |
| unknown | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### Q1. Budget cuts

Mean records proposed at those chunks: 7.5 (budget 5). Kept slots taken by carried-over records: 196/220 (89%).

### Q1. Names not proposed as a record

The word appears in the reply only outside any proposed record (the earlier 'written in the reply' count used a whole-text match): {'word only in insertions.left_context': 30, 'word only in insertions.right_context': 19, 'word only in outside the JSON': 19}.

### Q1. Examples (seed 42; full replies in outputs/probe_bridge/diagnose/q1_lost_names.jsonl)

- **budget cut**, c4_B43_004 (form1 'Balogunlere'), step 135: record at position 7 of 9 proposed; 5 kept. Kept: ['Miss Cathy', 'Penistone Crags', 'Fairy Cave', 'Mr. Edgar Linton', 'Ellen']. Reply (first 400 characters): `{"active_entities": [{"name": "Miss Cathy", "description": ["the daughter of Mr. Edgar Linton and Catherine Earnshaw", "the young girl who lives at Thrushcross Grange and is raised by her father", "the child who shows curiosity about the landscape beyond the park"], "aliases": ["Miss Cathy", "Cathy", "the little lady"]}, {"name": "Penistone Crags", "description": ["a rocky escarpment with steep cl`
- **budget cut**, c3_B34_017 (form2 'Golandam'), step 221: record at position 8 of 9 proposed; 5 kept. Kept: ['Emma Woodhouse', 'Mr. Knightley', 'Mr. Elton', 'Mrs. Elton', 'Miss Bates']. Reply (first 400 characters): `{"active_entities": [{"name": "Emma Woodhouse", "description": ["felt that she could not do too much for Harriet", "feels that she could not do too much for Harriet", "received little encouragement from Mrs. Elton", "had no opportunity of speaking to Mr. Knightley till after supper", "wanted Mr. Elton to marry Harriet", "was completely mistaken in Mr. Elton", "did not want to be flattered by Mr. E`
- **budget cut**, c1_B27_019 (form1 'Aydinlikel'), step 47: record at position 11 of 12 proposed; 5 kept. Kept: ['Mansfield Park', 'Fanny Price', 'Edmund Bertram', 'Miss Crawford', 'Mansfield Common']. Reply (first 400 characters): `{"active_entities": [{"name": "Mansfield Park", "description": ["The estate of Sir Thomas Bertram in the county of Northampton", "Home of Fanny Price for the past five years, where she has lived under the care of her Bertram relatives"], "aliases": ["Mansfield Park"]}, {"name": "Fanny Price", "description": ["Niece of Sir Thomas and Lady Bertram, daughter of Mrs. Price, who has lived at Mansfield `
- **merge loss**, c6_B31_000 (form1 'the Firkin Cooper'), step 109: present in every snapshot; absent only from memory.json. Kept: ['Armand de Montalban', 'Garrick the Younger', 'Richard III', 'Romeo and Juliet', 'Firkin Cooper']. Reply (first 400 characters): `{"active_entities": [{"name": "Armand de Montalban", "description": ["A character claimed by the duke in the current chunk, presenting himself as a lecturer on the Science of Phrenology", "A persona adopted by the duke in the current chunk, also claiming to be the world-renowned Shakespearian tragedian Garrick the Younger"], "aliases": ["Armand de Montalban", "the lecturer on the Science of Phreno`
- **merge loss**, c6_B18_010 (form1 'the Mistle Thatcher'), step 71: present in every snapshot; absent only from memory.json. Kept: ['Tess Durbeyfield', 'reaping-machine', 'buff gloves', 'Mistle Thatcher']. Reply (first 400 characters): `{"active_entities": [{"name": "Tess Durbeyfield", "description": ["a young mother who is nursing her child in the field", "the central character who is left alone in The Chase after a night ride with Alec", "the eldest daughter of the Durbeyfield family"], "aliases": ["Tess", "Tess Durbeyfield"]}, {"name": "reaping-machine", "description": ["a large, rickety machine pulled by three horses that cut`
- **merge loss**, c8_B61_023 (form1 'the Querncroft'), step 152: present in every snapshot; absent only from memory.json. Kept: ['Tchitchikov', 'millionaire', 'Querncroft']. Reply (first 400 characters): `{"active_entities": [{"name": "Tchitchikov", "description": ["Main character in Dead Souls who travels to collect deceased serfs for financial gain", "Shady figure with a history of swindles, including a smuggling fraud, who escapes criminal charges by portraying himself as a victim of injustice", "Protagonist of Dead Souls, a landowner and schemer who travels through provincial Russia to purchase`
- **parser loss**, c3_B37_008 (form2 'Lowriwena'), step 197: reply not parsed (status fallback). Kept: []. Reply (first 400 characters): `{"active_entities": [{"name": "Mr. Bennet", "description": ["is the father of Elizabeth and her sisters", "is the husband of Mrs. Bennet", "was hard on by Miss Austen, Mr. Darcy, and Miss Elizabeth for the 'impropriety' of his conduct"], "aliases": ["Mr. Bennet", "the father"]}, {"name": "Elizabeth", "description": ["is the second daughter of Mr. Bennet", "is the daughter of Mr. and Mrs. Bennet", `
- **parser loss**, c8_B34_015 (form1 'the Weary Pilgrim'), step 132: reply not parsed (status fallback). Kept: []. Reply (first 400 characters): `{"active_entities": [{"name": "Mr. Frank Churchill", "description": ["arrived early at Hartfield", "impressed Emma with his good looks and charm", "visited Miss Jane Fairfax", "showed interest in Highbury and its residents", "expressed enthusiasm for reviving ball events at the Crown Inn", "visited the Bateses' lodgings", "discussed Miss Fairfax's complexion with Emma", "expressed desire to buy so`
- **parser loss**, c8_B34_017 (form2 'Eerikalev'), step 225: reply not parsed (status fallback). Kept: []. Reply (first 400 characters): `{"active_entities": [{"name": "Emma Woodhouse", "description": ["aware of the anxiety and alarm" caused by the gipsy incident, "resolved to keep her father from knowing", "would not interfere with her father's perception of Harriet's health"], "aliases": ["Emma", "Miss Woodhouse"]}, {"name": "Harriet Smith", "description": ["was frightened by the gipsies", "clung to Frank Churchill", "recovered he`
- **not proposed as a record**, c6_B18_011 (form2 'Faramarzin'), step 214: word only in outside the JSON. Kept: ['Tess Durbeyfield', 'Angel Clare', 'Talbothays', 'Marian', 'Izz Huett']. Reply (first 400 characters): `{ "active_entities": [ { "name": "Tess Durbeyfield", "description": [ "the former dairymaid at Talbothays", "the former wife of Angel Clare who is not to be replaced by Izz", "a girl of simple life, not yet one-and-twenty, who has been caught during her days of immaturity like a bird in a springe", "the woman who is now on her way to an upland farm in the centre of the county", "the woman whose id`
- **not proposed as a record**, c3_B27_018 (form1 'Rhosllaner'), step 66: word only in insertions.right_context. Kept: ['Fanny Price', 'Edmund Bertram', 'Miss Crawford', 'Julia Bertram', 'Henry Crawford']. Reply (first 400 characters): `{ "active_entities": [ { "name": "Fanny Price", "description": [ "Shy and observant girl who watches Edmund and Miss Crawford ride together with growing unease, feeling slighted and anxious about her cousin's attention", "Quiet and sensitive niece of Lady Bertram who feels discomfort at seeing Edmund and Miss Crawford together, despite her civil response", "Shy, timid, and observant girl who watch`
- **not proposed as a record**, c6_B27_020 (form2 'Juwaynid'), step 164: word only in insertions.left_context. Kept: ['Henry Crawford', 'Fanny Price', 'Lady Bertram', 'Mrs. Grant', 'Sir Thomas Bertram']. Reply (first 400 characters): `{"active_entities": [{"name": "Henry Crawford", "description": ["Young man who arrives at Mansfield Park and becomes increasingly interested in Fanny Price, seeking to win her affection despite her reserve", "Captivated by Fanny's transformation and moral integrity, he resolves to court her during his extended stay, showing admiration for her sensibility and affection for her brother William", "Ex`

### Q2. At the reveal chunk: was the earlier record of form 1 in the prompt? (210 characters)

| class | total | cell 1 | cell 2 | cell 3 | cell 4 | cell 6 | cell 7 | cell 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no earlier record | 138 | 16 | 11 | 17 | 30 | 24 | 17 | 23 |
| not recalled: recall budget full with other records | 66 | 14 | 18 | 12 | 0 | 6 | 11 | 5 |
| recalled: earlier record written, new name not added | 2 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| recalled: joined | 3 | 0 | 1 | 1 | 0 | 0 | 1 | 0 |
| recalled: nothing written | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |

### Q2. Cell 4 at the gold chunk: was the earlier record of form 2 recalled, and did the model link form 1 to it?

| class | cell 4 |
| --- | --- |
| no earlier record | 23 |
| not recalled: recall budget full with other records | 3 |
| not recalled: word-overlap score zero | 2 |
| recalled: nothing written | 2 |

### Q2. Earlier record in the prompt where the link could form

Of the 90 characters whose two names were both written in some proposed record, the earlier record was in the prompt at the step where the link could form (the reveal; the gold chunk in cell 4) in 5/90 (6%).

### Q3. Records per character in the final memory (name or alias holds form 1 or form 2)

| records | total | cell 1 | cell 2 | cell 3 | cell 4 | cell 5 | cell 6 | cell 7 | cell 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 99 | 11 | 7 | 15 | 14 | 15 | 16 | 7 | 14 |
| 1 | 116 | 12 | 15 | 11 | 13 | 15 | 14 | 23 | 13 |
| 2 | 25 | 7 | 8 | 4 | 3 | 0 | 0 | 0 | 3 |
| 3+ | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### Q3. Pair classes (characters with 2 or more records)

Characters with 2 or more records: 25. Pairs: {'title variant': 10, 'other': 8, 'given name against surname': 6, 'surname only against full name': 1}. Characters with a further record holding a form only in its description: 20.

### Q3. Five examples (seed 42)

- **c4_B43_005** (cell 4, Ellsworthen / Savasan): 'Ellsworthen' aliases=['Ellsworthen', 'Savasan'] gold=no reveal=yes; 'Madame Ellsworthen' aliases=['Madame Ellsworthen'] gold=yes reveal=no
- **c3_B18_009** (cell 3, Vasconcelar / Modupeola): 'Captain Vasconcelar' aliases=['Captain Vasconcelar'] gold=yes reveal=no; 'Vasconcelar' aliases=['Modupeola', 'Vasconcelar'] gold=no reveal=yes
- **c4_B18_009** (cell 4, Sakuragiya / Ubaydallan): 'Sakuragiya' aliases=['Sakuragiya', 'Ubaydallan'] gold=no reveal=yes; 'Captain Sakuragiya' aliases=['Captain Sakuragiya'] gold=yes reveal=no
- **c2_B04_012** (cell 2, Chukiwanka / Olumidera): 'Judge Chukiwanka' aliases=['the judge', 'Judge Chukiwanka'] gold=yes reveal=no; 'Olumidera' aliases=['Olumidera', 'Chukiwanka'] gold=no reveal=yes
- **c8_B26_024** (cell 8, the Amber Plover / Rumisunqu): 'Amber Plover' aliases=['Amber Plover'] gold=yes reveal=no; 'Rumisunqu' aliases=['Rumisunqu', 'the owner of the Amber Plover'] gold=no reveal=yes

### Q4. c7_B37_008 (cell 7): "Where did the herbalist come from?", record 'Mitrassa', gold rank 118

| rank | chunk | text raw | text z | memory raw (field) | memory z | summary z | final | carried by |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | B37:175 | 7.23 | 5.24 | 3.11 (description) | 1.81 | 0.71 | 3.530 | text |
| 2 | B37:197 | 3.60 | 1.96 | - | - | 2.12 | 2.000 | text |
| 3 | B37:118 | 4.24 | 2.54 | 1.88 (description) | 0.16 | 1.08 | 1.635 | text |
| 4 | B37:18 | 4.52 | 2.79 | 1.70 (description) | -0.07 | -0.64 | 1.426 | text |
| 5 | B37:81 | 3.60 | 1.96 | 1.88 (description) | 0.16 | 1.31 | 1.355 | text |
| 6 | B37:102 | 4.38 | 2.67 | 1.88 (description) | 0.16 | -1.14 | 1.337 | text |
| 7 | B37:47 | 3.35 | 1.74 | 1.64 (description) | -0.15 | 2.43 | 1.330 | text |
| 8 | B37:226 | 4.48 | 2.75 | 1.59 (description) | -0.22 | -1.14 | 1.279 | text |
| 9 | B37:232 | 3.98 | 2.31 | 1.88 (description) | 0.16 | -0.65 | 1.218 | text |
| 10 | B37:198 | 2.97 | 1.39 | 1.88 (description) | 0.16 | 2.12 | 1.172 | text |
| 118 | B37:109 (gold) | 0.74 | -0.62 | 3.11 (description) | 1.81 | -1.14 | -0.034 | memory |

### Q4. c7_B18_011 (cell 7): "What was the name of the sailmaker's vessel?", record 'Nustakilla', gold rank 146

| rank | chunk | text raw | text z | memory raw (field) | memory z | summary z | final | carried by |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | B18:257 | 7.80 | 6.79 | 3.96 (description) | 0.12 | -0.20 | 3.773 | text |
| 2 | B18:158 | 5.96 | 4.97 | 3.96 (description) | 0.12 | 0.29 | 2.843 | text |
| 3 | B18:9 | 5.15 | 4.17 | 3.96 (description) | 0.12 | 1.09 | 2.533 | text |
| 4 | B18:294 | 3.92 | 2.96 | 5.83 (description) | 1.16 | 2.87 | 2.447 | text |
| 5 | B18:51 | 5.05 | 4.07 | 3.96 (description) | 0.12 | -0.65 | 2.186 | text |
| 6 | B18:172 | 3.46 | 2.51 | 3.96 (description) | 0.12 | 2.96 | 1.917 | text |
| 7 | B18:146 | 3.92 | 2.96 | 5.83 (description) | 1.16 | -0.78 | 1.836 | text |
| 8 | B18:28 | 3.93 | 2.97 | 3.96 (description) | 0.12 | 0.27 | 1.727 | text |
| 9 | B18:290 | 3.01 | 2.06 | 3.96 (description) | 0.12 | 2.87 | 1.654 | text |
| 10 | B18:295 | 3.41 | 2.45 | 5.83 (description) | 1.16 | -0.47 | 1.607 | text |
| 146 | B18:80 (gold) | 0.20 | -0.72 | 5.51 (description) | 0.80 | 0.36 | -0.121 | memory |

### Q4. c7_B34_016 (cell 7): "What was the name of the shinglewright's vessel?", record 'Ladipojul', gold rank 183

| rank | chunk | text raw | text z | memory raw (field) | memory z | summary z | final | carried by |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | B34:209 | 5.74 | 5.92 | 4.75 (description) | 0.88 | -0.43 | 3.462 | text |
| 2 | B34:145 | 6.00 | 6.24 | 3.55 (description) | -0.48 | 0.35 | 3.390 | text |
| 3 | B34:128 | 3.70 | 3.41 | 4.64 (description) | 0.75 | 0.43 | 2.174 | text |
| 4 | B34:199 | 3.16 | 2.75 | 4.44 (description) | 0.52 | 0.22 | 1.712 | text |
| 5 | B34:16 | 2.40 | 1.81 | 5.76 (name) | 1.86 | 0.49 | 1.605 | text |
| 6 | B34:273 | 2.63 | 2.10 | 4.08 (description) | 0.11 | 2.22 | 1.568 | text |
| 7 | B34:221 | 2.39 | 1.81 | 4.19 (description) | 0.24 | 2.59 | 1.502 | text |
| 8 | B34:237 | 3.30 | 2.93 | 4.03 (description) | 0.06 | -1.17 | 1.450 | text |
| 9 | B34:265 | 2.54 | 1.99 | 4.44 (description) | 0.52 | 0.55 | 1.342 | text |
| 10 | B34:213 | 2.09 | 1.44 | 5.55 (description) | 1.69 | 0.25 | 1.308 | text |
| 183 | B34:124 (gold) | 0.13 | -0.97 | 4.71 (description) | 0.83 | 0.71 | -0.186 | memory |

### Q4. c1_B61_021 (cell 1): "What was the name of Simindokht's vessel?", record 'Madame Bhattarain', gold rank 6

| rank | chunk | text raw | text z | memory raw (field) | memory z | summary z | final | carried by |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | B61:103 | 6.22 | 5.73 | 7.19 (description) | 3.90 | -0.61 | 4.164 | text |
| 2 | B61:261 | 5.49 | 4.93 | 1.41 (description) | -0.48 | 1.26 | 2.816 | text |
| 3 | B61:114 | 5.46 | 4.90 | 2.26 (description) | 0.09 | -0.21 | 2.716 | text |
| 4 | B61:58 | 4.21 | 3.55 | 2.11 (description) | 0.10 | 1.17 | 2.195 | text |
| 5 | B61:177 | 3.38 | 2.66 | 3.53 (description) | 1.29 | 1.26 | 2.045 | text |
| 6 | B61:100 (gold) | 2.59 | 1.80 | 7.19 (description) | 3.90 | -0.61 | 1.981 | memory |
| 7 | B61:169 | 3.38 | 2.65 | 2.57 (alias) | 0.58 | 0.55 | 1.725 | text |
| 8 | B61:256 | 3.67 | 2.96 | 1.50 (description) | -0.42 | 0.71 | 1.646 | text |
| 9 | B61:131 | 3.26 | 2.53 | 2.19 (description) | 0.02 | 0.83 | 1.547 | text |
| 10 | B61:263 | 3.37 | 2.64 | 1.42 (description) | -0.48 | 1.26 | 1.543 | text |

### Q4. c7_B61_023 (cell 7): "Where did the engraver come from?", record 'Roudabeh', gold rank 135

| rank | chunk | text raw | text z | memory raw (field) | memory z | summary z | final | carried by |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | B61:229 | 5.88 | 4.23 | 3.47 (description) | 1.07 | -0.58 | 2.552 | text |
| 2 | B61:71 | 5.20 | 3.56 | 1.42 (description) | -0.13 | 0.94 | 2.100 | text |
| 3 | B61:146 | 2.58 | 1.01 | 4.01 (alias) | 4.63 | -0.65 | 1.739 | memory |
| 4 | B61:292 | 2.94 | 1.36 | 3.32 (description) | 0.98 | 3.95 | 1.687 | text |
| 5 | B61:54 | 3.88 | 2.28 | 1.40 (description) | -0.15 | 1.91 | 1.546 | text |
| 6 | B61:114 | 3.36 | 1.77 | 1.79 (description) | 0.08 | 2.63 | 1.443 | text |
| 7 | B61:158 | 4.39 | 2.78 | 1.42 (description) | -0.13 | -0.52 | 1.421 | text |
| 8 | B61:160 | 4.40 | 2.79 | 1.33 (description) | -0.18 | -0.62 | 1.393 | text |
| 9 | B61:87 | 4.31 | 2.70 | 1.42 (description) | -0.13 | -0.64 | 1.357 | text |
| 10 | B61:246 | 4.04 | 2.44 | 0.96 (name) | 0.10 | -0.63 | 1.279 | text |
| 135 | B61:140 (gold) | 1.42 | -0.12 | 3.46 (description) | 1.05 | -0.51 | 0.139 | memory |

### Q4. c7_B44_028 (cell 7): "Where did the ropemaker come from?", record 'Umutcan', gold rank 22

| rank | chunk | text raw | text z | memory raw (field) | memory z | summary z | final | carried by |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | B44:196 | 7.44 | 6.31 | 3.77 (description) | 0.81 | 1.18 | 3.929 | text |
| 2 | B44:384 | 2.21 | 0.73 | 3.23 (name) | 4.90 | 0.54 | 1.855 | memory |
| 3 | B44:356 | 4.53 | 3.20 | 1.63 (description) | -0.27 | 0.62 | 1.810 | text |
| 4 | B44:239 | 2.29 | 0.82 | 4.04 (description) | 4.73 | -0.26 | 1.725 | memory |
| 5 | B44:291 | 3.66 | 2.28 | 1.66 (description) | -0.25 | 2.65 | 1.636 | text |
| 6 | B44:145 | 3.07 | 1.65 | 2.73 (description) | 0.29 | 3.01 | 1.496 | text |
| 7 | B44:224 | 3.91 | 2.55 | 2.73 (description) | 0.29 | -0.55 | 1.406 | text |
| 8 | B44:368 | 3.52 | 2.13 | 2.73 (description) | 0.29 | 0.76 | 1.388 | text |
| 9 | B44:141 | 3.25 | 1.84 | 2.73 (description) | 0.29 | 1.35 | 1.326 | text |
| 10 | B44:321 | 2.37 | 0.90 | 2.31 (description) | 2.31 | 1.03 | 1.315 | memory |
| 22 | B44:98 (gold) | 3.01 | 1.59 | 3.77 (description) | 0.81 | -0.25 | 1.065 | text |

### Q4. Summary

| character | gold rank | rank at memory weight 1.0 | rank at 2.0 | smallest memory weight for top 10 | gold memory rank | chunks tied on memory score | chunks linked to the record |
| --- | --- | --- | --- | --- | --- | --- | --- |
| c7_B37_008 | 118 | 67 | 7 | 1.9 | 1 | 1 | 2 |
| c7_B18_011 | 146 | 100 | 75 | > 20 | 26 | 0 | 2 |
| c7_B34_016 | 183 | 111 | 76 | > 20 | 12 | 0 | 2 |
| c1_B61_021 | 6 | 2 | 2 | 0.3 | 1 | 1 | 2 |
| c7_B61_023 | 135 | 79 | 36 | > 20 | 11 | 0 | 2 |
| c7_B44_028 | 22 | 14 | 7 | 1.65 | 7 | 1 | 2 |

### Q4. Note

Memory weight is lambda_n (published 0.5), other weights fixed, searched on a grid of 0.05 up to 20. This describes the landscape only; no weight is changed anywhere. The gold chunk had the top memory score in the book in 2 of 6 cases.

### Counts against the candidate fixes

| Finding | Count | Fix it points to |
| --- | --- | --- |
| Names cut by the 5-record budget | 44 | Separate slots for new names, or a larger budget |
| Names cut by the alias/description caps | 0 | Larger field caps |
| Names lost in merge or parsing | 7 | Update step or output format |
| Names never proposed as a record | 68 | Extraction instructions |
| Earlier record not recalled at the reveal | 66 | Recall by name tokens, not only by word overlap |
| Recalled, but the model wrote a separate record | 0 | Extraction instructions |
| Duplicate records that differ only by a title | 10 | Merge on the stripped name, with the surname caution |
| Link formed, text channel still wins | 1 | Fusion or record design, to be decided later |

## Anything unexpected

Q3, published memories: no published NovelHopQA Qwen3-30B construction cache (AEWord/index.json) under outputs/; set PROBE_PUBLISHED_INDEX. That part was skipped; the rest ran.

## Files written

- `/mnt/data/zeinab/SAADI/outputs/probe_bridge/diagnose/q1_lost_names.jsonl`
- `/mnt/data/zeinab/SAADI/outputs/probe_bridge/diagnose/q2_recall.jsonl`
- `/mnt/data/zeinab/SAADI/outputs/probe_bridge/diagnose/q3_duplicates.jsonl`
- `/mnt/data/zeinab/SAADI/outputs/probe_bridge/diagnose/q3_published.json`
- `/mnt/data/zeinab/SAADI/outputs/probe_bridge/diagnose/q4_scores.json`
