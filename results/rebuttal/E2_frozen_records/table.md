# E2. Frozen records against live records

HR in percent. Diff = paired (live - frozen) in percentage points with a 95% bootstrap CI (10,000 resamples over queries, seed 42). Variants share every setting of the published run; z-scores are recomputed per variant.

## Main table: all eligible queries

| Dataset | Retriever | Subset | k | n | Live | Frozen | No memory | Live-Frozen (pp) | 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| loogle | bm25 | all | 5 | 1832 | 92.3 | 92.3 | 92.1 | 0.00 | [-0.33, 0.33] |
| loogle | bm25 | all | 10 | 1832 | 96.5 | 96.5 | 96.1 | -0.05 | [-0.27, 0.11] |
| musique_32k | bm25 | all | 5 | 900 | 62.8 | 62.4 | 59.6 | 0.33 | [-0.33, 1.00] |
| musique_32k | bm25 | all | 10 | 900 | 77.7 | 78.0 | 76.6 | -0.33 | [-0.89, 0.11] |
| qasper_64k | bm25 | all | 5 | 1333 | 17.0 | 16.8 | 16.1 | 0.15 | [-0.30, 0.60] |
| qasper_64k | bm25 | all | 10 | 1333 | 26.1 | 25.9 | 24.4 | 0.23 | [-0.38, 0.83] |
| novelhopqa | bm25 | all | 5 | 908 | 18.1 | 18.1 | 17.6 | 0.00 | [-0.44, 0.44] |
| novelhopqa | bm25 | all | 10 | 908 | 22.0 | 21.9 | 21.8 | 0.11 | [-0.44, 0.66] |

## Affected queries only

| Dataset | Retriever | Subset | k | n | Live | Frozen | No memory | Live-Frozen (pp) | 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| loogle | bm25 | affected | 5 | 925 | 90.4 | 90.4 | 89.8 | 0.00 | [-0.65, 0.54] |
| loogle | bm25 | affected | 10 | 925 | 95.9 | 95.9 | 95.1 | 0.00 | [-0.32, 0.32] |
| musique_32k | bm25 | affected | 5 | 793 | 61.0 | 60.7 | 58.4 | 0.38 | [-0.38, 1.13] |
| musique_32k | bm25 | affected | 10 | 793 | 76.2 | 76.5 | 74.9 | -0.38 | [-1.01, 0.13] |
| qasper_64k | bm25 | affected | 5 | 1285 | 16.8 | 16.6 | 15.7 | 0.23 | [-0.23, 0.70] |
| qasper_64k | bm25 | affected | 10 | 1285 | 26.0 | 25.6 | 24.2 | 0.39 | [-0.23, 1.01] |
| novelhopqa | bm25 | affected | 5 | 903 | 17.8 | 17.8 | 17.4 | 0.00 | [-0.44, 0.44] |
| novelhopqa | bm25 | affected | 10 | 903 | 21.8 | 21.7 | 21.6 | 0.11 | [-0.44, 0.66] |

## Split by the position of the earliest evidence chunk

| Dataset | Retriever | Subset | k | n | Live | Frozen | No memory | Live-Frozen (pp) | 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| loogle | bm25 | first_third | 5 | 653 | 91.4 | 91.3 | 91.9 | 0.15 | [-0.31, 0.77] |
| loogle | bm25 | first_third | 10 | 653 | 96.3 | 96.5 | 95.6 | -0.15 | [-0.46, 0.00] |
| loogle | bm25 | middle_third | 5 | 614 | 91.7 | 91.9 | 91.5 | -0.16 | [-0.81, 0.49] |
| loogle | bm25 | middle_third | 10 | 614 | 96.1 | 95.9 | 95.9 | 0.16 | [0.00, 0.49] |
| loogle | bm25 | last_third | 5 | 565 | 94.0 | 94.0 | 92.9 | 0.00 | [0.00, 0.00] |
| loogle | bm25 | last_third | 10 | 565 | 97.0 | 97.2 | 96.8 | -0.18 | [-0.53, 0.00] |
| musique_32k | bm25 | first_third | 5 | 411 | 55.5 | 55.0 | 53.3 | 0.49 | [-0.73, 1.70] |
| musique_32k | bm25 | first_third | 10 | 411 | 72.5 | 73.0 | 71.8 | -0.49 | [-1.46, 0.49] |
| musique_32k | bm25 | middle_third | 5 | 285 | 69.8 | 69.5 | 65.3 | 0.35 | [0.00, 1.05] |
| musique_32k | bm25 | middle_third | 10 | 285 | 83.2 | 83.5 | 81.1 | -0.35 | [-1.05, 0.00] |
| musique_32k | bm25 | last_third | 5 | 204 | 67.6 | 67.6 | 64.2 | 0.00 | [-1.47, 1.47] |
| musique_32k | bm25 | last_third | 10 | 204 | 80.4 | 80.4 | 79.9 | 0.00 | [0.00, 0.00] |
| qasper_64k | bm25 | first_third | 5 | 687 | 15.7 | 15.9 | 15.3 | -0.15 | [-0.73, 0.44] |
| qasper_64k | bm25 | first_third | 10 | 687 | 26.1 | 25.6 | 24.5 | 0.44 | [-0.44, 1.31] |
| qasper_64k | bm25 | middle_third | 5 | 398 | 18.8 | 18.3 | 17.6 | 0.50 | [-0.50, 1.51] |
| qasper_64k | bm25 | middle_third | 10 | 398 | 26.9 | 27.4 | 24.4 | -0.50 | [-1.51, 0.50] |
| qasper_64k | bm25 | last_third | 5 | 248 | 17.3 | 16.9 | 15.7 | 0.40 | [0.00, 1.21] |
| qasper_64k | bm25 | last_third | 10 | 248 | 25.0 | 24.2 | 24.2 | 0.81 | [-0.81, 2.42] |
| novelhopqa | bm25 | first_third | 5 | 474 | 10.1 | 10.1 | 9.5 | 0.00 | [-0.63, 0.63] |
| novelhopqa | bm25 | first_third | 10 | 474 | 12.7 | 12.9 | 12.7 | -0.21 | [-1.05, 0.42] |
| novelhopqa | bm25 | middle_third | 5 | 229 | 23.1 | 23.1 | 22.7 | 0.00 | [-1.31, 1.31] |
| novelhopqa | bm25 | middle_third | 10 | 229 | 28.4 | 27.1 | 28.4 | 1.31 | [0.00, 3.06] |
| novelhopqa | bm25 | last_third | 5 | 205 | 30.7 | 30.7 | 30.7 | 0.00 | [0.00, 0.00] |
| novelhopqa | bm25 | last_third | 10 | 205 | 36.6 | 37.1 | 35.6 | -0.49 | [-1.46, 0.00] |

## Later-update statistics per dataset

| Dataset | Chunks | Share with a later update | Mean later aliases per chunk | Mean later descriptions per chunk |
| --- | --- | --- | --- | --- |
| loogle | 3346 | 48.1% | 0.32 | 4.72 |
| musique_32k | 17102 | 44.7% | 0.23 | 1.78 |
| novelhopqa | 7011 | 96.9% | 9.79 | 193.20 |
| qasper_64k | 17159 | 74.7% | 1.38 | 7.22 |

## Example queries (live hits at 5, frozen misses at 5)

Taken in saved query order, not selected by effect size.

- **loogle / 173** - What happened to Tunisia in the quarter-finals of the tournament?
  - evidence chunk step 11: entity "2015 Africa Cup of Nations" gained the alias "Tunisia vs. Equatorial Guinea refereeing at 2015 Afcon" at step 21
  - evidence chunk step 11: entity "2015 Africa Cup of Nations" gained the description "Controversial quarter-final defeat to Equatorial Guinea on 31 January 2015" at step 21
  - evidence chunk step 11: entity "2015 Africa Cup of Nations" gained the description "CAF banned referee Rajindraparsad Seechurn for six months" at step 21
- **loogle / 522** - What was the priority for aeromedical evacuation?
  - evidence chunk step 9: entity "57th Medical Detachment" gained the description "Provided aeromedical evacuation for both American and Vietnamese combatants and noncombatants in the Republic of Vietnam" at step 12
  - evidence chunk step 9: entity "57th Medical Detachment" gained the description "Split into three operational sections to cover a greater area, resulting in reduced coverage in Saigon and areas further South" at step 12
  - evidence chunk step 9: entity "57th Medical Detachment" gained the description "Provided coverage for armed and troop transport helicopters during combat heliborne assaults, U. S. Air Force defoliation missions, training parachute jumps, convoys, and transport of key medical personnel and emergency medical material" at step 12
- **loogle / 1204** - Who was voted as Arsenal's Player of the Month for August?
  - evidence chunk step 4: entity "Saka" gained the description "scored equaliser against West Ham United" at step 8
  - evidence chunk step 4: entity "Saka" gained the description "scored against Brighton & Hove Albion" at step 8
  - evidence chunk step 4: entity "Saka" gained the description "scored against Tottenham" at step 8
- **loogle / 1907** - What did the woman say about festivalgoers at the event?
  - evidence chunk step 14: entity "drug detection dog" gained the description "a dog trained to detect drugs, used in police operations at music festivals" at step 21
  - evidence chunk step 14: entity "strip search" gained the description "a full body search conducted when suspicion of drug possession exists" at step 21
  - evidence chunk step 14: entity "Law Enforcement Conduct Commission" gained the alias "LECC" at step 15
- loogle/bm25: 4 queries live-hit/frozen-miss at 5; 4 go the other way.
- **musique_32k / 11** - What is a notable work written by the author of The Timothy files?
  - evidence chunk step 245: entity "DVD Flick" gained the description "capable of importing audio tracks, video files and subtitles" at step 246
  - evidence chunk step 245: entity "DVD Flick" gained the description "composing a DVD-Video movie and burning it to a disc" at step 246
  - evidence chunk step 245: entity "DVD Flick" gained the description "creating an ISO image for later burning" at step 246
- **musique_32k / 161** - Who developed the eponymous character from the series that contains Mickey's Safari in Letterland?
  - evidence chunk step 55: entity "Mickey Mouse" gained the alias "Mickey" at step 66
  - evidence chunk step 55: entity "Mickey Mouse" gained the description "first cartoon character to have a star on the Hollywood Walk of Fame" at step 66
  - evidence chunk step 55: entity "Mickey Mouse" gained the description "appeared in over 130 films, including The Band Concert (1935), Brave Little Tailor (1938), and Fantasia (1940)" at step 66
- **musique_32k / 1183** - Who is the president of the newly declared independent country that is part of the Commission of Truth and Friendship with the country that Tony Gunawan is from?
  - evidence chunk step 124: entity "Indonesia–Timor Leste Commission on Truth and Friendship" gained the description "notable for being the first modern truth commission to be bilateral" at step 125
  - evidence chunk step 351: entity "Declaration of Independence" gained the alias "the Declaration of Independence" at step 358
  - evidence chunk step 351: entity "Declaration of Independence" gained the description "proclaimed that the former Thirteen Colonies were now a sovereign, independent nation and no longer part of the British Empire" at step 358
- **musique_32k / 1374** - When did the majority party in the House of Representatives take control of the determiner of rules of the US House and US Senate?
  - evidence chunk step 243: entity "Senate" gained the alias "Senate of Pakistan" at step 282
  - evidence chunk step 243: entity "Senate" gained the description "Consisted of 49 Democrats, 49 Republicans, and two Independents who caucused with the Democrats, giving the Democratic caucus the slightest 51-49 majority" at step 279
  - evidence chunk step 243: entity "Senate" gained the description "Of the seats up for election in 2008, 23 were held by Republicans and 12 by Democrats" at step 279
- **musique_32k / 1958** - In what German state is the district where the man, who wanted the religion underlying Daraga Church to reform, preached a sermon on Marian devotion?
  - evidence chunk step 19: entity "Wittenberg" gained the description "location of Martin Luther's final sermon on Marian devotion" at step 38
  - evidence chunk step 19: entity "Wittenberg" gained the description "district in eastern Germany" at step 38
  - evidence chunk step 27: entity "Martin Luther" gained the alias "Young Man Luther" at step 177
- musique_32k/bm25: 6 queries live-hit/frozen-miss at 5; 3 go the other way.
- **qasper_64k / 3** - What are the evaluation metrics and criteria used to evaluate the model performance?
  - evidence chunk step 613: entity "MISSA-con" gained the alias "MISSA-con (MISSA-con)" at step 627
  - evidence chunk step 613: entity "MISSA-con" gained the description "a variant of MISSA that conditions on system intent, showing better performance on RIP and ERIP compared to other variants on the AntiScam dataset" at step 627
  - evidence chunk step 613: entity "perplexity" gained the alias "PPL" at step 734
- **qasper_64k / 25** - What are strong baselines authors used?
  - evidence chunk step 87: entity "BIBREF2" gained the alias "attention layer BIBREF2" at step 216
  - evidence chunk step 87: entity "BIBREF2" gained the description "a citation for a study on identifying Arabic varieties in blogs and social fora" at step 120
  - evidence chunk step 87: entity "BIBREF2" gained the description "applies the principle to detect offensive language" at step 346
- **qasper_64k / 604** - What sources of less sensitive data are available?
  - evidence chunk step 5: entity "languages" gained the alias "languages" at step 746
  - evidence chunk step 5: entity "languages" gained the description "natural human languages used in multilingual dialogue systems" at step 746
  - evidence chunk step 5: entity "languages" gained the description "the set of six languages used in the experiment for user-model interaction and response generation" at step 746
- **qasper_64k / 99** - Which datasets do they evaluate on?
  - evidence chunk step 726: entity "FIGER (GOLD)" gained the alias "the dataset" at step 733
  - evidence chunk step 726: entity "FIGER (GOLD)" gained the description "dataset with 113 tag sets, allowing mentions to have multiple type paths" at step 727
  - evidence chunk step 726: entity "FIGER (GOLD)" gained the description "used for training with weakly labeled samples generated from Wikipedia anchor links" at step 727
- **qasper_64k / 782** - Did they use the same dataset as Skip-gram to train?
  - evidence chunk step 593: entity "SubGram" gained the alias "SubGram model" at step 604
  - evidence chunk step 593: entity "SubGram" gained the alias "our SubGram model" at step 606
  - evidence chunk step 593: entity "SubGram" gained the description "a modified version of word2vec based on the skip-gram model" at step 602
- qasper_64k/bm25: 6 queries live-hit/frozen-miss at 5; 4 go the other way.
- **novelhopqa / hop_1:828** - What indicates that Stepan Arkadyevitch avoided joining the grown-up people for tea?
  - evidence chunk step 64: entity "Anna Karenina" gained the alias "Anna Arkadyevna" at step 93
  - evidence chunk step 64: entity "Anna Karenina" gained the alias "the woman" at step 167
  - evidence chunk step 64: entity "Anna Karenina" gained the alias "she" at step 254
- **novelhopqa / hop_2:984** - What characteristic of Estella is highlighted by her interaction with others and her demeanor by the fire?
  - evidence chunk step 239: entity "Pip" gained the alias "Philip Pip" at step 288
  - evidence chunk step 239: entity "Pip" gained the alias "my Lord" at step 362
  - evidence chunk step 239: entity "Pip" gained the description "the narrator who is addressing Estella at the Assembly Ball" at step 244
- novelhopqa/bm25: 2 queries live-hit/frozen-miss at 5; 2 go the other way.

## Reproduction gate (LIVE vs saved reference)

| Dataset | Retriever | Queries | Hits@5 (mine/expected) | Hits@10 (mine/expected) | Rankings equal to saved |
| --- | --- | --- | --- | --- | --- |
| loogle | bm25 | 1832 | 1691/1691 | 1767/1767 | 1832/1832 |
| musique_32k | bm25 | 900 | 565/565 | 699/699 | 900/900 |
| qasper_64k | bm25 | 1333 | 226/226 | 348/348 | 1333/1333 |
| novelhopqa | bm25 | 908 | 164/164 | 200/200 | 908/908 |
