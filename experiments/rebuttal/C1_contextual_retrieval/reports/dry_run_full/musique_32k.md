# C1 Contextual Retrieval: MuSiQue-32K (dry run full: stub contexts, BM25 only, placeholder numbers)

- Written 2026-11-09 14:59:55; commit e2dbaa178fa0; documents 45, chunks 17102, queries 900 (expected 900 on the full dataset).
- Status: generation done; jobs BM25 DONE.

## 1. Correctness check (empty context reproduces Vanilla)

CR with every context empty indexes exactly the chunk text, so it must give Vanilla (SAADI's text channel alone) and the paper's saved results. BM25: identical rankings; dense: identical top 10 for at least 99% of queries and HR within 0.1 points (C2's Jina drift rule applies). The pool itself is checked by reproducing SAADI's saved rankings with the published weights.

| Retriever | Gate | SAADI top-10 = saved | Empty = Vanilla top-10 | HR@5 / HR@10 empty vs Vanilla | = saved text-only file | Table 1 Vanilla |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 | pass | 900/900 | 900/900 (top-150 900) | 57.56 / 74.00 vs 57.56 / 74.00 | no file | 57.6 / 74.0 |
- BM25: Vanilla reference for the dense check = published path in this process (shared encoding cache); memory and summary channels from {'c2_streams': 0, 'published_path': 45} documents.

## 2. Token census (Qwen3 tokenizer)

- Window 262144 tokens (model config, rope_scaling None); prompt template 70 tokens; output budget 1024.
- Document tokens: min 42K, median 49K, max 52K; chunk tokens median 127, max 429.
- Documents over the window: 0 of 45 (0 chunks); segments per document: {'1': 45}.
- Split vs joint tokenization of the prompt identical on the checked samples: [True, True, True].

## 3. GPU memory check (embedding models)

- Skipped (CPU dry run).

## 4. Long documents (segment rule)

- No document exceeds the window; every chunk saw its whole document.

## 5. Contexts

- 17102 contexts (0 missing), generator ['stub']; empty 0, failed 0, cut at 1024 tokens 0.
- Context length (tokens): min 8, p25 40, median 44, mean 46.3, p75 50, p95 65, max 153.
- Cached tokens: ['simulated (stub)'].

## 6. Cost next to SAADI

| Quantity | CR | SAADI (memory + hierarchy calls) |
| --- | --- | --- |
| LLM calls (total) | 17102 | 21294 |
| LLM calls per chunk | 1.00 | 1.25 |
| LLM calls per document | 380.0 | 473.2 |
| Input tokens per chunk, nominal (full prompt) | 49K | 2697 (no prefix cache) |
| Input tokens per chunk, computed (after prefix-cache hits) | 326 | same as nominal |
| Input tokens total, nominal / computed | 839.0M / 5.6M | 46.1M |
| Generated tokens per chunk, mean / median | 46.3 / 44.0 | 697.2 |
| Generated tokens total | 792K | 11.9M |
| Contexts cut at 1024 tokens | 0 (0.0%) | — |
| Generation wall clock / GPU-hours | 30s / 0.00 | not in the paper's tables |
| Indexing + scoring, BM25 (wall / GPU-hours) | 51s / 0.00 | — |
- Ratios per chunk. SAADI's builds had no prefix caching, so its input is compared with both CR columns: CR nominal / SAADI =18.19x, CR computed / SAADI = 0.12x; calls 0.80x; generated tokens 0.07x. SAADI chunks in the cost table: 17102; C1 chunks: 17102.

## 7. Retrieval: per retriever

HR@5 / HR@10 (%). *Complementarity check, not a proposed method.

| Retriever | Vanilla (text channel) | Table 1 Vanilla | CR | SAADI (published) | SAADI on CR text* | LQCA (Table 1) |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 | 57.6 / 74.0 | 57.6 / 74.0 | 57.3 / 76.4 | 62.8 / 77.7 | 63.4 / 80.2 | 83.6 / 91.8 |
- SAADI held-out column left out: results/rebuttal/C2_heldout_fusion/REPORT.md does not exist yet.

Paired differences (points) with 95% bootstrap intervals (10,000 resamples over queries, seed 42):

| Retriever | CR − Vanilla @5 | @10 | SAADI − CR @5 | @10 | SAADI-on-CR − CR @5 | @10 |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 | -0.2 [-2.8, +2.3] | +2.4 [+0.1, +4.7] | +5.4 [+2.8, +8.2] | +1.2 [-1.1, +3.6] | +6.1 [+3.8, +8.4] | +3.8 [+1.9, +5.7] |

## 8. Retrieval: hybrid (dense + BM25, cookbook fusion)

| Hybrid | Vanilla @5 / @10 | CR @5 / @10 | SAADI @5 / @10 | CR − Vanilla @10 | SAADI − CR @10 | SAADI − CR @5 |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 + BM25 (dry-run stand-in for a dense retriever) | 57.6 / 74.0 | 57.3 / 76.4 | 62.8 / 77.7 | +2.4 [+0.1, +4.7] | +1.2 [-1.1, +3.6] | +5.4 [+2.8, +8.2] |
- Top 150 of each list, score 0.8 / rank_dense + 0.2 / rank_bm25 (1-based), the same merge for Vanilla, CR and SAADI (SAADI = SAADI with d + SAADI with BM25, published weights). No reranker.

## 9. Dense truncation of the contextualized chunks

- No dense retriever in this run.

## 10. Sanity sample (10 random contexts, seed 42)

- `group_010:33` (segment 1/1, 44 tokens, stop): **Stub context: In 2006, the Philadelphia Greyhound Terminal was the second busiest Greyhound terminal in the United States, after the Port Authority Bus Terminal in New York. Besides Greyhound, six other bus**
  - chunk: He was also part of the lesser-known KC Accidental, which consisted of Drew and Charles Spearin, another current member of Broken Social Scene. Jalanan (“Streetside” in Bahasa Indonesia) is an award-winning 2013 feature-length Indonesian documentary film directed and produced by Daniel Ziv and featu…
- `group_003:17` (segment 1/1, 60 tokens, stop): **Stub context: NBA All - Rookie First Team (2004) NBA scoring champion (2009) 2 × NBA Skills Challenge champion (2006, 2007) Consensus first - team All - American (2003) Third - team**
  - chunk: 11 NL MVP award winners have played for the Dodgers, winning a total of 13 MVP Awards, Eight Cy Young Award winners have pitched for the Dodgers, winning a total of twelve Cy Young Awards. The team has also produced 18 Rookie of the Year Award winners, twice as many as the next closest team, includi…
- `group_024:346` (segment 1/1, 51 tokens, stop): **Stub context: Frank Baum Portrayed by Ray Bolger Voiced by Paul Scheer (Once Upon a Time) Information Aliases Socrates Strawman Chang Wang Woe Fiyero Tigelaar Species Scarecrow Gender Male Occupation Ruler of**
  - chunk: Jess Harnell voices Tim from season 3 episode 11 onwards. Mike Myers as Shrek Eddie Murphy as Donkey Cameron Diaz as Princess Fiona John Lithgow as Lord Farquaad Vincent Cassel as ``Monsieur ''Robin Hood Conrad Vernon as Gingerbread Man Chris Miller as Geppetto / Magic Mirror Cody Cameron as Pinocch…
- `group_022:151` (segment 1/1, 48 tokens, stop): **Stub context: Since 1967, the unitary authority of Plymouth has included the, once independent, towns of Plympton and Plymstock which lie along the east of the River Plym. The River Tamar forms**
  - chunk: In the manifesto "For the unification and independence of Morocco", issued in August 1946, the party the need to create a united national front. Communists actively participated in armed struggles against the French colonial authorities in the period 1953-1956. After the proclamation of the sovereig…
- `group_020:18` (segment 1/1, 45 tokens, stop): **Stub context: Hickory Grove Estates is approximately south-southwest of Clayton Village. Risdon Park High School was located on Senate Rd, between Hannan St, Creasy St, and Ferme St in the suburb of**
  - chunk: It is located at 33° 23' 35N 73° 19' 45E with an altitude of 494 metres (1624 feet). The Grove, also known as Loretto Rest, is a historic house located on Grove Court in Cold Spring, New York, United States. It was built as the estate of Frederick Lente, surgeon at the nearby West Point Foundry and …
- `group_013:127` (segment 1/1, 49 tokens, stop): **Stub context: On 10 July 2006, he was sworn in as the First Deputy Prime Minister of Timor Leste. He was sworn in as the acting Prime Minister on 19 May 2007**
  - chunk: In 1959, an international gathering of over 185 judges, lawyers, and law professors from 53 countries, meeting in New Delhi and speaking as the International Commission of Jurists, made a declaration as to the fundamental principle of the rule of law. This was the Declaration of Delhi. They declared…
- `group_009:7` (segment 1/1, 47 tokens, stop): **Stub context: The company's credit union for employees, BECU, remains based in the Seattle area, though it is now open to all residents of Washington. The 1933 Datsun Type 12 was a**
  - chunk: Jonga was an acronym for Jabalpur Ordnance aNd Guncarriage Assembly. The Scion bbX was first concept car to be revealed by Toyota for its daughter company, Scion. The concept was first shown at the New York International Auto Show in 2003. As shown at the motor show, the bbX is very similar to the d…
- `group_008:295` (segment 1/1, 37 tokens, stop): **Stub context: He also worked in the Office of the Chief of Naval Operations as Assistant Deputy Chief of Naval Operations, Surface Warfare, and Assistant Chief of Naval Operations, Surface Warfare. He**
  - chunk: The third USS "Macdonough" (DD-351) was a "Farragut"-class destroyer in the United States Navy during World War II. She was named for Thomas Macdonough. The United States Navy's ``Sea, Air, and Land ''Teams, commonly abbreviated as the Navy SEALs, are the U. S. Navy's primary special operations forc…
- `group_037:199` (segment 1/1, 43 tokens, stop): **Stub context: Kosovo, a former autonomous province, made its Olympic debut as an independent national team at the 2016 Summer Olympics. There are other language groupings in Southern Europe. Albanian is spoken**
  - chunk: The revival of the Hebrew language took place in Europe and Israel toward the end of the 19th century and into the 20th century, through which the language's usage changed from the sacred language of Judaism to a spoken and written language used for daily life in Israel. The process began as Jews st…
- `group_003:37` (segment 1/1, 42 tokens, stop): **Stub context: Sleen has been inhabited for centuries. Much ancient history can be found in the area, particularly in the forests (which are planted, though). At birth and during the Republiek der**
  - chunk: Born in Green Bay, Wisconsin, Quinn was a realtor and served on the Green Bay Common Council, the Brown County, Wisconsin Board of Supervisors, the local Board of Education, and the Wisconsin State Assembly from 1955 until 1973. He was a Republican. Sinzongo is a village in the Lobaye region in the …

## 11. Estimates and timings

- Generation estimate 3.0 h for 17102 calls (per call 9.3e-06 s x visible document tokens + 0.15 s, plus 420 s engine start (B2's measured rate on the same model and runtime)); measured 30s.
- BM25: estimate 3 min on CPU (C2's CPU rate x 3 (published streams, empty-context pass, CR pass)); measured 51s.

