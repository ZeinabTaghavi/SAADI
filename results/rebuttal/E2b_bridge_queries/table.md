# E2b. Bridge queries: why frozen matched live

Tokenization: the repository's BM25 tokenizer, stopwords dropped. A token is rare in a document when it appears in fewer than 5% of that document's chunks. A memory query has an evidence chunk that matches a query token only through its linked records; a bridge query needs a field that arrived at a LATER step than the chunk.

## 1-2. Lexical overlap and query counts

| Dataset | Queries | Mean Q-in-text overlap | (rare tokens) | Memory queries | (rare) | Bridge queries | (rare) | Alias-bridge | (rare) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| loogle | 1832 | 86.7% | 60.8% | 128 (7.0%) | 23 (1.3%) | 29 (1.6%) | 2 (0.1%) | 0 (0.0%) | 0 (0.0%) |
| musique_32k | 900 | 31.1% | 24.5% | 770 (85.6%) | 621 (69.0%) | 378 (42.0%) | 269 (29.9%) | 4 (0.4%) | 0 (0.0%) |
| qasper_64k | 1333 | 31.9% | 22.4% | 993 (74.5%) | 530 (39.8%) | 464 (34.8%) | 218 (16.4%) | 18 (1.4%) | 9 (0.7%) |
| novelhopqa | 908 | 32.1% | 9.5% | 841 (92.6%) | 698 (76.9%) | 599 (66.0%) | 437 (48.1%) | 19 (2.1%) | 8 (0.9%) |

## 3. Retrieval on bridge queries only

HR in percent; paired live - frozen in pp with 95% bootstrap CI (10,000 resamples, seed 42). Groups under 30 queries are too small to conclude.

| Dataset | k | n | Live | Frozen | No memory | Live-Frozen (pp) | 95% CI | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| loogle | 5 | 29 | 82.8 | 82.8 | 86.2 | 0.00 | [0.00, 0.00] | too small to conclude |
| loogle | 10 | 29 | 89.7 | 89.7 | 89.7 | 0.00 | [0.00, 0.00] | too small to conclude |
| musique_32k | 5 | 378 | 57.9 | 56.9 | 54.8 | 1.06 | [-0.26, 2.65] |  |
| musique_32k | 10 | 378 | 73.5 | 73.5 | 71.7 | 0.00 | [-0.79, 0.79] |  |
| qasper_64k | 5 | 464 | 16.4 | 16.4 | 15.1 | 0.00 | [-0.86, 0.86] |  |
| qasper_64k | 10 | 464 | 26.5 | 25.9 | 24.4 | 0.65 | [-0.22, 1.72] |  |
| novelhopqa | 5 | 599 | 10.0 | 9.8 | 9.7 | 0.17 | [-0.33, 0.83] |  |
| novelhopqa | 10 | 599 | 13.2 | 12.9 | 12.9 | 0.33 | [0.00, 0.83] |  |

## 4-5. Record sharing and memory-channel ranks on bridge queries

Sharing: chunks of the document linked to the record carrying each bridge token (all of them receive the same boost). Rank: best memory-channel-only rank of a bridge evidence chunk.

| Dataset | Bridge tokens | Sharing median | Sharing p90 | Memory rank live (median) | Memory rank frozen (median) |
| --- | --- | --- | --- | --- | --- |
| loogle | 32 | 9 | 19 | 4 | 5 |
| musique_32k | 929 | 4 | 12 | 16 | 30 |
| qasper_64k | 1012 | 9 | 32 | 90 | 176 |
| novelhopqa | 3966 | 108 | 251 | 58 | 85 |

## 6. Example bridge queries (random, seed 42)

**loogle** (29 bridge queries)

- `611`: How many matches did BFC Dynamo win in the first half of the season?
  - bridge token "win" from the description "Scored a goal in the 2–0 overtime win in the 1987–88 FDGB-Pokal final" of "Michael Schulz" (chunk step 18, arrived step 32); hit@10 live=True, frozen=True
- `1074`: How many fatalities were caused by the blizzard in west-central New York?
  - bridge token "new" from the description "struck southwestern Ontario and west-central New York on January 29" of "second blizzard" (chunk step 20, arrived step 21); hit@10 live=False, frozen=False
- `1038`: What did Youngkin do regarding the Virginia Freedom of Information Act?
  - bridge token "regard" from the description "Reported that Youngkin’s actions to loosen standards regarding who can become a teacher mi" of "Washington Post" (chunk step 10, arrived step 22); hit@10 live=True, frozen=True
- `707`: What did Botafogo do in 1976 when it was facing a financial crisis?
  - bridge token "fac" from the description "Faced debts with players and businessmen, no place to train, no sponsors, and no stadium t" of "Botafogo de Futebol e Regatas" (chunk step 7, arrived step 11); hit@10 live=True, frozen=True
- `1507`: What was ABC's plan for a special Sunday afternoon telecast?
  - bridge token "plan" from the description "Reintroduced over-the-air broadcast rights in 1994 under a six-year plan with revenue shar" of "Major League Baseball" (chunk step 15, arrived step 19); hit@10 live=True, frozen=True

**musique_32k** (378 bridge queries)

- `1437`: When did the rx 350 model of the luxury division of the company that manufactures Scion Fuse change body style?
  - bridge token "manufactur" from the description "Premier touring motorcycle manufactured in two forms by Yamaha from 1983 to 1993 and from " of "Yamaha Royal Star Venture" (chunk step 110, arrived step 111); hit@10 live=True, frozen=True
- `1412`: When did the party that hold the majority in the House of Reps take control of the branch that approves members of the American cabinet?
  - bridge token "hold" from the description "holds elections concurrently with U.S. Senate elections" of "United States House of Representatives" (chunk step 167, arrived step 185); hit@10 live=False, frozen=False
- `1310`: What does seal stand for in the branch of the U.S. military that operates a list of destroyer classes as well as the USS Tarawa?
  - bridge token "destroyer" from the description "former commander of Destroyer Squadron 26, Destroyer Squadron 20, USS "Semmes" (DDG-18), a" of "John Walfrid Nyquist" (chunk step 329, arrived step 331); hit@10 live=True, frozen=True
- `88`: Where is the country with ISO code ISO 3166-2:CV located?
  - bridge token "locat" from the description "located 570 kilometres (350 mi) west of the Cape Verde Peninsula off the coast of Northwes" of "Cape Verde" (chunk step 284, arrived step 285); hit@10 live=False, frozen=False
- `1266`: When was the last time the team that Malcolm Briggs was a member of beat the 1894-95 FA cup winner?
  - bridge token "winner" from the description "winners qualify for the following season's UEFA Europa League (formerly named the UEFA Cup" of "FA Cup" (chunk step 294, arrived step 302); hit@10 live=False, frozen=False

**qasper_64k** (464 bridge queries)

- `685`: Which websites were used in the web crawl?
  - bridge token "used" from the description "not used in this work; instead, the system learns them during inference through multiple r" of "hand-crafted features" (chunk step 326, arrived step 673); hit@10 live=True, frozen=True
- `769`: What parts of their multitask model are shared?
  - bridge token "shar" from the description "enables learning across shared representations" of "multi-task learning" (chunk step 458, arrived step 472); hit@10 live=False, frozen=False
- `970`: what text embedding methods were used in their approach?
  - bridge token "method" from the description "a logistic regression classifier used as a baseline method" of "LR" (chunk step 154, arrived step 657); hit@10 live=False, frozen=False
- `510`: What were the usual logical approaches of the time period?
  - bridge token "time" from the description "enables fast development of linguistic applications with similar accuracy and a reasonable" of "MERF" (chunk step 198, arrived step 202); hit@10 live=False, frozen=False
- `1141`: What are the modifications made to post-trained BERT?
  - bridge token "post" from the description "post-trained BERT used in aspect extraction and aspect sentiment classification tasks" of "BERT" (chunk step 393, arrived step 414); hit@10 live=True, frozen=True

**novelhopqa** (599 bridge queries)

- `hop_3:749`: What contradiction exists in Sviazhsky's approach to the Russian peasantry and his beliefs about their development?
  - bridge token "approach" from the description "Is being approached by Stepan Arkadyevitch to discuss Anna’s emotional state and their mar" of "Alexey Alexandrovitch Karenin" (chunk step 254, arrived step 385); hit@10 live=False, frozen=False
- `hop_1:220`: What does Mr. Jaggers ask Pip to mention to Magwitch or Provis in his communication?
  - bridge token "communication" from the description "the intermediary in the chain of communication between Magwitch and Pip" of "Mr. Wemmick" (chunk step 134, arrived step 252); hit@10 live=False, frozen=False
- `hop_1:216`: What action does Miss Havisham take to indicate to Pip where he should sit?
  - bridge token "action" from the description "the one who advised Pip to take action on Wednesday or early in the week" of "Wemmick" (chunk step 161, arrived step 329); hit@10 live=False, frozen=False
- `hop_1:461`: What characteristic does Fauchelevent share with cats, according to the paragraph?
  - bridge token "accord" from the description "is the father of the bride according to the fishwife" of "Jean Valjean" (chunk step 443, arrived step 1085); hit@10 live=False, frozen=False
- `hop_1:972`: What event finally occurred on the third day that had a significant impact on the character's emotional state?
  - bridge token "state" from the description "had returned to Mansfield in a less assured state" of "Edmund" (chunk step 242, arrived step 292); hit@10 live=False, frozen=False

