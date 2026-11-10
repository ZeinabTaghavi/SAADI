# C1 Contextual Retrieval: MuSiQue-32K

- Written 2026-11-09 20:24:44; commit 99762a63e590; documents 45, chunks 17102, queries 900 (expected 900 on the full dataset).
- Status: generation done; jobs BM25 DONE, Contriever DONE, BGE-M3 DONE, Jina DONE, Qwen DONE.

## 1. Correctness check (empty context reproduces Vanilla)

CR with every context empty indexes exactly the chunk text, so it must give Vanilla (SAADI's text channel alone) and the paper's saved results. BM25: identical rankings; dense: identical top 10 for at least 99% of queries and HR within 0.1 points (C2's Jina drift rule applies). The pool itself is checked by reproducing SAADI's saved rankings with the published weights.

| Retriever | Gate | SAADI top-10 = saved | Empty = Vanilla top-10 | HR@5 / HR@10 empty vs Vanilla | = saved text-only file | Table 1 Vanilla |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 | pass | 900/900 | 900/900 (top-150 900) | 57.56 / 74.00 vs 57.56 / 74.00 | no file | 57.6 / 74.0 |
| Contriever | pass | 900/900 | 900/900 (top-150 900) | 59.89 / 76.11 vs 59.89 / 76.11 | no file | 59.9 / 76.1 |
| BGE-M3 | pass | 900/900 | 900/900 (top-150 900) | 73.22 / 87.44 vs 73.22 / 87.44 | no file | 73.2 / 87.4 |
| Jina | pass | 705/900 | 900/900 (top-150 900) | 71.67 / 85.78 vs 71.67 / 85.78 | no file | 71.6 / 85.8 |
| Qwen | pass | 899/900 | 900/900 (top-150 900) | 75.33 / 89.00 vs 75.33 / 89.00 | no file | 75.3 / 89.0 |
- BM25: Vanilla reference for the dense check = C2 saved streams (independent run); memory and summary channels from {'c2_streams': 45, 'published_path': 0} documents.
- Contriever: Vanilla reference for the dense check = C2 saved streams (independent run); memory and summary channels from {'c2_streams': 45, 'published_path': 0} documents.
- BGE-M3: Vanilla reference for the dense check = C2 saved streams (independent run); memory and summary channels from {'c2_streams': 45, 'published_path': 0} documents.
- Jina: Vanilla reference for the dense check = C2 saved streams (independent run); memory and summary channels from {'c2_streams': 45, 'published_path': 0} documents.
- Qwen: Vanilla reference for the dense check = C2 saved streams (independent run); memory and summary channels from {'c2_streams': 45, 'published_path': 0} documents.

## 2. Token census (Qwen3 tokenizer)

- Window 262144 tokens (model config, rope_scaling None); prompt template 70 tokens; output budget 1024.
- Document tokens: min 42K, median 49K, max 52K; chunk tokens median 127, max 429.
- Documents over the window: 0 of 45 (0 chunks); segments per document: {'1': 45}.
- Split vs joint tokenization of the prompt identical on the checked samples: [True, True, True].

## 3. GPU memory check (embedding models)

| Model | GPUs | Vanilla batch | Chosen batch | Peak MiB | Total MiB | Fits (10% margin) |
| --- | --- | --- | --- | --- | --- | --- |
| Contriever | 1 | 128 | 128 | [3306] | [48541] | yes |
| BGE-M3 | 1 | 16 | 16 | [3724] | [48541] | yes |
| Jina | 1 | 64 | 64 | [2082] | [48541] | yes |
| Qwen | 2 | 64 | 64 | [27106, 28446] | [48541, 48541] | yes |
- Longest plain chunk of this group (1294 characters) with 1024 placeholder tokens in front, float32 as Vanilla.

## 4. Long documents (segment rule)

- No document exceeds the window; every chunk saw its whole document.

## 5. Contexts

- 17102 contexts (0 missing), generator ['vllm']; empty 0, failed 0, cut at 1024 tokens 0.
- Context length (tokens): min 26, p25 78, median 93, mean 95.0, p75 110, p95 140, max 372.
- Cached tokens: ['vllm num_cached_tokens'].

## 6. Cost next to SAADI

| Quantity | CR | SAADI (memory + hierarchy calls) |
| --- | --- | --- |
| LLM calls (total) | 17102 | 21294 |
| LLM calls per chunk | 1.00 | 1.25 |
| LLM calls per document | 380.0 | 473.2 |
| Input tokens per chunk, nominal (full prompt) | 49K | 2697 (no prefix cache) |
| Input tokens per chunk, computed (after prefix-cache hits) | 349 | same as nominal |
| Input tokens total, nominal / computed | 839.0M / 6.0M | 46.1M |
| Generated tokens per chunk, mean / median | 95.0 / 93.0 | 697.2 |
| Generated tokens total | 1.6M | 11.9M |
| Contexts cut at 1024 tokens | 0 (0.0%) | — |
| Generation wall clock / GPU-hours | 2.1 h / 8.23 | not in the paper's tables |
| Indexing + scoring, BM25 (wall / GPU-hours) | 3 min / 0.00 | — |
| Indexing + scoring, Contriever (wall / GPU-hours) | 5 min / 0.08 | — |
| Indexing + scoring, BGE-M3 (wall / GPU-hours) | 9 min / 0.15 | — |
| Indexing + scoring, Jina (wall / GPU-hours) | 5 min / 0.08 | — |
| Indexing + scoring, Qwen (wall / GPU-hours) | 2.2 h / 4.39 | — |
- Ratios per chunk. SAADI's builds had no prefix caching, so its input is compared with both CR columns: CR nominal / SAADI =18.19x, CR computed / SAADI = 0.13x; calls 0.80x; generated tokens 0.14x. SAADI chunks in the cost table: 17102; C1 chunks: 17102.

## 7. Retrieval: per retriever

HR@5 / HR@10 (%). *Complementarity check, not a proposed method.

| Retriever | Vanilla (text channel) | Table 1 Vanilla | CR | SAADI (published) | SAADI on CR text* | LQCA (Table 1) | SAADI held-out (C2) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BM25 | 57.6 / 74.0 | 57.6 / 74.0 | 65.0 / 82.2 | 62.8 / 77.7 | 68.3 / 83.4 | 83.6 / 91.8 | 64.7 / 79.9 |
| Contriever | 59.9 / 76.1 | 59.9 / 76.1 | 69.3 / 86.1 | 68.8 / 83.7 | 73.6 / 87.9 | 80.8 / 89.0 | 68.8 / 84.9 |
| BGE-M3 | 73.2 / 87.4 | 73.2 / 87.4 | 77.4 / 89.7 | 76.9 / 88.3 | 79.4 / 90.6 | 76.7 / 90.4 | 76.1 / 90.8 |
| Jina | 71.7 / 85.8 | 71.6 / 85.8 | 77.1 / 89.7 | 73.6 / 88.1 | 76.7 / 90.0 | 79.4 / 89.0 | — / — |
| Qwen | 75.3 / 89.0 | 75.3 / 89.0 | 75.3 / 88.6 | 76.4 / 90.0 | 75.6 / 89.8 | 82.2 / 90.4 | — / — |

Paired differences (points) with 95% bootstrap intervals (10,000 resamples over queries, seed 42):

| Retriever | CR − Vanilla @5 | @10 | SAADI − CR @5 | @10 | SAADI-on-CR − CR @5 | @10 |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 | +7.4 [+5.1, +9.8] | +8.2 [+6.0, +10.4] | -2.2 [-4.7, +0.1] | -4.6 [-6.8, -2.3] | +3.3 [+1.3, +5.3] | +1.2 [-0.4, +2.9] |
| Contriever | +9.4 [+6.8, +12.2] | +10.0 [+7.6, +12.4] | -0.6 [-3.0, +2.0] | -2.4 [-4.6, -0.3] | +4.2 [+2.0, +6.6] | +1.8 [+0.0, +3.4] |
| BGE-M3 | +4.2 [+1.9, +6.6] | +2.2 [+0.6, +4.0] | -0.6 [-3.0, +1.8] | -1.3 [-3.1, +0.4] | +2.0 [+0.0, +4.1] | +0.9 [-0.6, +2.3] |
| Jina | +5.4 [+3.1, +7.8] | +3.9 [+2.2, +5.7] | -3.6 [-5.9, -1.2] | -1.6 [-3.3, +0.2] | -0.4 [-2.3, +1.3] | +0.3 [-1.1, +1.8] |
| Qwen | +0.0 [-2.8, +2.7] | -0.4 [-2.4, +1.6] | +1.1 [-1.6, +3.9] | +1.4 [-0.6, +3.4] | +0.2 [-1.9, +2.3] | +1.2 [-0.2, +2.7] |

## 8. Retrieval: hybrid (dense + BM25, cookbook fusion)

| Hybrid | Vanilla @5 / @10 | CR @5 / @10 | SAADI @5 / @10 | CR − Vanilla @10 | SAADI − CR @10 | SAADI − CR @5 |
| --- | --- | --- | --- | --- | --- | --- |
| Contriever + BM25 | 61.3 / 78.9 | 72.0 / 88.0 | 69.8 / 83.9 | +9.1 [+6.9, +11.3] | -4.1 [-6.1, -2.1] | -2.2 [-4.8, +0.3] |
| BGE-M3 + BM25 | 71.7 / 87.4 | 77.6 / 90.4 | 75.1 / 87.7 | +3.0 [+1.4, +4.7] | -2.8 [-4.6, -1.1] | -2.4 [-4.9, +0.0] |
| Jina + BM25 | 71.9 / 85.4 | 76.7 / 89.9 | 73.8 / 87.7 | +4.4 [+2.7, +6.3] | -2.2 [-4.0, -0.4] | -2.9 [-5.3, -0.6] |
| Qwen + BM25 | 73.3 / 88.9 | 76.7 / 89.7 | 75.0 / 89.7 | +0.8 [-1.2, +2.7] | +0.0 [-2.0, +2.0] | -1.7 [-4.2, +0.9] |
- Top 150 of each list, score 0.8 / rank_dense + 0.2 / rank_bm25 (1-based), the same merge for Vanilla, CR and SAADI (SAADI = SAADI with d + SAADI with BM25, published weights). No reranker.

## 9. Dense truncation of the contextualized chunks

| Retriever | Max input | CR chunks over max | CR tokens cut (end of chunk) | Plain chunks over max |
| --- | --- | --- | --- | --- |
| Contriever | 512 | 1/17102 | 38 | 0/17102 |
| BGE-M3 | 8192 | 0/17102 | 0 | 0/17102 |
| Jina | 512 | 1/17102 | 96 | 0/17102 |
| Qwen | 512 | 1/17102 | 107 | 0/17102 |

## 10. Sanity sample (10 random contexts, seed 42)

- `group_010:33` (segment 1/1, 92 tokens, stop): **Kevin Drew, a co-founder of Broken Social Scene, was also part of the lesser-known band KC Accidental with Charles Spearin. Jalanan, a 2013 Indonesian documentary directed by Daniel Ziv, follows Jakarta street musicians Bambang "Ho" Mulyono, Titi Juwariyah, and Boni Putera as they perform on commuter buses, face legal challenges, and navigate the political landscape of modern-day Indonesia.**
  - chunk: He was also part of the lesser-known KC Accidental, which consisted of Drew and Charles Spearin, another current member of Broken Social Scene. Jalanan (“Streetside” in Bahasa Indonesia) is an award-winning 2013 feature-length Indonesian documentary film directed and produced by Daniel Ziv and featu…
- `group_003:17` (segment 1/1, 69 tokens, stop): **The passage discusses the achievements of the Los Angeles Dodgers, highlighting their record number of NL MVP and Cy Young Award winners, as well as Rookie of the Year honorees. This is followed by a biographical profile of Kevin Durant, an NBA player for the Golden State Warriors, indicating a shift in focus to basketball and individual athlete accomplishments.**
  - chunk: 11 NL MVP award winners have played for the Dodgers, winning a total of 13 MVP Awards, Eight Cy Young Award winners have pitched for the Dodgers, winning a total of twelve Cy Young Awards. The team has also produced 18 Rookie of the Year Award winners, twice as many as the next closest team, includi…
- `group_024:346` (segment 1/1, 54 tokens, stop): **Jess Harnell voices Tim the Bear from season 3 episode 11 onwards in the animated series "The Bear in the Big Blue House," while Mike Myers, Eddie Murphy, Cameron Diaz, and other voice actors portray characters in the film "Shrek."**
  - chunk: Jess Harnell voices Tim from season 3 episode 11 onwards. Mike Myers as Shrek Eddie Murphy as Donkey Cameron Diaz as Princess Fiona John Lithgow as Lord Farquaad Vincent Cassel as ``Monsieur ''Robin Hood Conrad Vernon as Gingerbread Man Chris Miller as Geppetto / Magic Mirror Cody Cameron as Pinocch…
- `group_022:151` (segment 1/1, 98 tokens, stop): **The chunk discusses the Moroccan Communist Party's role in the struggle for independence from French colonial rule, highlighting its advocacy for a united national front, armed resistance, and post-independence goals such as nationalization, agrarian reform, and sovereignty. This context is part of a broader document covering political movements, historical events, and social developments across various countries and regions, including the partition of India, the rise of Pan-Somalism, and the history of the Moroccan Communist Party's activities.**
  - chunk: In the manifesto "For the unification and independence of Morocco", issued in August 1946, the party the need to create a united national front. Communists actively participated in armed struggles against the French colonial authorities in the period 1953-1956. After the proclamation of the sovereig…
- `group_020:18` (segment 1/1, 152 tokens, stop): **Darial is a town in the Islamabad Capital Territory of Pakistan, located at 33° 23' 35N 73° 19' 45E with an altitude of 494 metres (1624 feet). The Grove, also known as Loretto Rest, is a historic house located on Grove Court in Cold Spring, New York, United States, built in the mid-19th century as the estate of Frederick Lente, a surgeon at the nearby West Point Foundry and later a founder of the American Academy of Medicine. Designed in the Italian-villa style by architect Richard Upjohn, it was listed on the National Register of Historic Places in 2008.**
  - chunk: It is located at 33° 23' 35N 73° 19' 45E with an altitude of 494 metres (1624 feet). The Grove, also known as Loretto Rest, is a historic house located on Grove Court in Cold Spring, New York, United States. It was built as the estate of Frederick Lente, surgeon at the nearby West Point Foundry and …
- `group_013:127` (segment 1/1, 85 tokens, stop): **The chunk discusses the Declaration of Delhi, a 1959 statement by the International Commission of Jurists emphasizing the rule of law, including rights, an independent judiciary, and conditions for human dignity, while noting it did not mandate judicial review of legislative power. This appears in a broader context of political, legal, and historical documents covering topics such as national constitutions, governance, human rights, and international agreements.**
  - chunk: In 1959, an international gathering of over 185 judges, lawyers, and law professors from 53 countries, meeting in New Delhi and speaking as the International Commission of Jurists, made a declaration as to the fundamental principle of the rule of law. This was the Declaration of Delhi. They declared…
- `group_009:7` (segment 1/1, 119 tokens, stop): **The Jonga was a Nissan-designed vehicle used by the Indian Army, an acronym for Jabalpur Ordnance aNd Guncarriage Assembly. The Scion bbX, a concept car by Toyota's Scion brand, debuted at the 2003 New York International Auto Show and shared a similar design with the Scion xB, which was sold from 2003 to 2015. The Nissan Rogue, a compact crossover SUV, was introduced by Nissan in October 2007 for the 2008 model year.**
  - chunk: Jonga was an acronym for Jabalpur Ordnance aNd Guncarriage Assembly. The Scion bbX was first concept car to be revealed by Toyota for its daughter company, Scion. The concept was first shown at the New York International Auto Show in 2003. As shown at the motor show, the bbX is very similar to the d…
- `group_008:295` (segment 1/1, 113 tokens, stop): **The chunk discusses the USS Macdonough (DD-351), a "Farragut"-class destroyer in the U.S. Navy during World War II, and introduces the Navy SEALs as the U.S. Navy's primary special operations force, emphasizing their role in maritime operations from rivers, oceans, swamps, deltas, or coastlines. This information is part of a broader document detailing military history, naval vessels, and special operations units, including references to other destroyers, the Navy SEALs' origins, and their operational capabilities.**
  - chunk: The third USS "Macdonough" (DD-351) was a "Farragut"-class destroyer in the United States Navy during World War II. She was named for Thomas Macdonough. The United States Navy's ``Sea, Air, and Land ''Teams, commonly abbreviated as the Navy SEALs, are the U. S. Navy's primary special operations forc…
- `group_037:199` (segment 1/1, 49 tokens, stop): **The revival of the Hebrew language, which transitioned from a sacred Jewish language to a daily spoken and written language in Israel, began in the 19th century as Jews migrated to Palestine and used Hebrew as a common means of communication.**
  - chunk: The revival of the Hebrew language took place in Europe and Israel toward the end of the 19th century and into the 20th century, through which the language's usage changed from the sacred language of Judaism to a spoken and written language used for daily life in Israel. The process began as Jews st…
- `group_003:37` (segment 1/1, 51 tokens, stop): **The chunk is part of a document listing various individuals, places, and facts across diverse topics, including biographical details of political figures like Quinn from Green Bay, Wisconsin, and geographical information about Sinzongo, a village in the Central African Republic.**
  - chunk: Born in Green Bay, Wisconsin, Quinn was a realtor and served on the Green Bay Common Council, the Brown County, Wisconsin Board of Supervisors, the local Board of Education, and the Wisconsin State Assembly from 1955 until 1973. He was a Republican. Sinzongo is a village in the Lobaye region in the …

## 11. Estimates and timings

- Generation estimate 3.0 h for 17102 calls (per call 9.3e-06 s x visible document tokens + 0.15 s, plus 420 s engine start (B2's measured rate on the same model and runtime)); measured 2.1 h.
- BM25: estimate 3 min on CPU (C2's CPU rate x 3 (published streams, empty-context pass, CR pass)); measured 3 min.
- Contriever: estimate 3 min on 1 GPU(s) (C2 streams reused: CR chunks + plain chunks (empty-context check) encoded, 2 query encodings per query); measured 5 min.
- BGE-M3: estimate 14 min on 1 GPU(s) (C2 streams reused: CR chunks + plain chunks (empty-context check) encoded, 2 query encodings per query); measured 9 min.
- Jina: estimate 15 min on 1 GPU(s) (C2 streams reused: CR chunks + plain chunks (empty-context check) encoded, 2 query encodings per query); measured 5 min.
- Qwen: estimate 3.0 h on 2 GPU(s) (C2 streams reused: CR chunks + plain chunks (empty-context check) encoded, 2 query encodings per query); measured 2.2 h.

