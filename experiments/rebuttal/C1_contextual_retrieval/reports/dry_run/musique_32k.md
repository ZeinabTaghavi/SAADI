# C1 Contextual Retrieval: MuSiQue-32K (dry run: stub contexts, BM25 only, placeholder numbers)

- Written 2026-11-09 15:02:19; commit e2dbaa178fa0; documents 2, chunks 771, queries 42 (expected 900 on the full dataset).
- Status: generation done; jobs BM25 DONE.

## 1. Correctness check (empty context reproduces Vanilla)

CR with every context empty indexes exactly the chunk text, so it must give Vanilla (SAADI's text channel alone) and the paper's saved results. BM25: identical rankings; dense: identical top 10 for at least 99% of queries and HR within 0.1 points (C2's Jina drift rule applies). The pool itself is checked by reproducing SAADI's saved rankings with the published weights.

| Retriever | Gate | SAADI top-10 = saved | Empty = Vanilla top-10 | HR@5 / HR@10 empty vs Vanilla | = saved text-only file | Table 1 Vanilla |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 | pass | 42/42 | 42/42 (top-150 42) | 64.29 / 80.95 vs 64.29 / 80.95 | no file | 57.6 / 74.0 |
- BM25: Vanilla reference for the dense check = published path in this process (shared encoding cache); memory and summary channels from {'c2_streams': 0, 'published_path': 2} documents.

## 2. Token census (Qwen3 tokenizer)

- Window 262144 tokens (model config, rope_scaling None); prompt template 70 tokens; output budget 1024.
- Document tokens: min 49K, median 49K, max 50K; chunk tokens median 127, max 261.
- Documents over the window: 0 of 2 (0 chunks); segments per document: {'1': 2}.
- Split vs joint tokenization of the prompt identical on the checked samples: [True, True].

## 3. GPU memory check (embedding models)

- Skipped (CPU dry run).

## 4. Long documents (segment rule)

- No document exceeds the window; every chunk saw its whole document.

## 5. Contexts

- 771 contexts (0 missing), generator ['stub']; empty 0, failed 0, cut at 1024 tokens 0.
- Context length (tokens): min 18, p25 40, median 44, mean 46.4, p75 50, p95 65, max 102.
- Cached tokens: ['simulated (stub)'].

## 6. Cost next to SAADI

| Quantity | CR | SAADI (memory + hierarchy calls) |
| --- | --- | --- |
| LLM calls (total) | 771 | 21294 |
| LLM calls per chunk | 1.00 | 1.25 |
| LLM calls per document | 385.5 | 473.2 |
| Input tokens per chunk, nominal (full prompt) | 50K | 2697 (no prefix cache) |
| Input tokens per chunk, computed (after prefix-cache hits) | 329 | same as nominal |
| Input tokens total, nominal / computed | 38.2M / 254K | 46.1M |
| Generated tokens per chunk, mean / median | 46.4 / 44.0 | 697.2 |
| Generated tokens total | 36K | 11.9M |
| Contexts cut at 1024 tokens | 0 (0.0%) | — |
| Generation wall clock / GPU-hours | 3s / 0.00 | not in the paper's tables |
| Indexing + scoring, BM25 (wall / GPU-hours) | 2s / 0.00 | — |
- Ratios per chunk. SAADI's builds had no prefix caching, so its input is compared with both CR columns: CR nominal / SAADI =18.39x, CR computed / SAADI = 0.12x; calls 0.80x; generated tokens 0.07x. SAADI chunks in the cost table: 17102; C1 chunks: 771.

## 7. Retrieval: per retriever

HR@5 / HR@10 (%). *Complementarity check, not a proposed method.

| Retriever | Vanilla (text channel) | Table 1 Vanilla | CR | SAADI (published) | SAADI on CR text* | LQCA (Table 1) |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 | 64.3 / 81.0 | 57.6 / 74.0 | 66.7 / 76.2 | 69.0 / 85.7 | 76.2 / 85.7 | 83.6 / 91.8 |
- SAADI held-out column left out: results/rebuttal/C2_heldout_fusion/REPORT.md does not exist yet.

Paired differences (points) with 95% bootstrap intervals (10,000 resamples over queries, seed 42):

| Retriever | CR − Vanilla @5 | @10 | SAADI − CR @5 | @10 | SAADI-on-CR − CR @5 | @10 |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 | +2.4 [-7.1, +14.3] | -4.8 [-11.9, +0.0] | +2.4 [-7.1, +11.9] | +9.5 [+0.0, +21.4] | +9.5 [-2.4, +21.4] | +9.5 [+2.4, +19.0] |

## 8. Retrieval: hybrid (dense + BM25, cookbook fusion)

| Hybrid | Vanilla @5 / @10 | CR @5 / @10 | SAADI @5 / @10 | CR − Vanilla @10 | SAADI − CR @10 | SAADI − CR @5 |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 + BM25 (dry-run stand-in for a dense retriever) | 64.3 / 81.0 | 66.7 / 76.2 | 69.0 / 85.7 | -4.8 [-11.9, +0.0] | +9.5 [+0.0, +21.4] | +2.4 [-7.1, +11.9] |
- Top 150 of each list, score 0.8 / rank_dense + 0.2 / rank_bm25 (1-based), the same merge for Vanilla, CR and SAADI (SAADI = SAADI with d + SAADI with BM25, published weights). No reranker.

## 9. Dense truncation of the contextualized chunks

- No dense retriever in this run.

## 10. Sanity sample (10 random contexts, seed 42)

- `group_036:349` (segment 1/1, 46 tokens, stop): **Stub context: Ascalon was a contentious issue as it threatened communication between Saladin's dominions in Egypt and Syria; it was eventually agreed that Ascalon, with its defences demolished, be returned to Saladin's**
  - chunk: . The political situation in England rapidly began to deteriorate. Longchamp refused to work with Puiset and became unpopular with the English nobility and clergy. John exploited this unpopularity to set himself up as an alternative ruler with his own royal court, complete with his own justiciar, ch…
- `group_026:200` (segment 1/1, 48 tokens, stop): **Stub context: His positions on birth control, promulgated most famously in the 1968 encyclical Humanae vitae, and other political issues, were often controversial, especially in Western Europe and North America. While at**
  - chunk: Although there had been significant earlier attempts to reform the Catholic Church before Luther -- such as those of Jan Hus, Peter Waldo, and John Wycliffe -- Martin Luther is widely acknowledged to have started the Reformation with the Ninety - five Theses. Luther began by criticising the sale of …
- `group_026:120` (segment 1/1, 38 tokens, stop): **Stub context: The district shares a border with Kampong Thom and Kampong Cham provinces to the east. Most of the district is low lying floodplain and the Tonlé Sap river runs through**
  - chunk: Ottoman cuisine refers to the cuisine of the capital, Istanbul, and the regional capital cities, where the melting pot of cultures created a common cuisine that most of the population regardless of ethnicity shared. This diverse cuisine was honed in the Imperial Palace's kitchens by chefs brought fr…
- `group_036:89` (segment 1/1, 41 tokens, stop): **Stub context: Smith Island is an island on the Chesapeake Bay, on the border of Maryland and Virginia territorial waters in the United States. The province of Kanije was established in 1600**
  - chunk: It was formally ceded to Habsburg Monarchy by the Treaty of Karlowitz in 1699. Pakistan is bordered by India to the east, Afghanistan to the west and Iran to the southwest while China borders the country in the northeast. The nation is geopolitically placed within some of the most controversial regi…
- `group_026:351` (segment 1/1, 45 tokens, stop): **Stub context: In February 1955, Eisenhower dispatched the first American soldiers to Vietnam as military advisors to Diem's army. After Diem announced the formation of the Republic of Vietnam (RVN, commonly known**
  - chunk: Among the reforms he introduced, which came into operation on 1 October 1945, were to make education free of charge for all students, to ensure that every student was provided with instruction in the religion of his / her parents, to prevent teachers from been exploited by managers of schools by hav…
- `group_026:323` (segment 1/1, 39 tokens, stop): **Stub context: The Millrose Games is an annual track and field meet whose featured event is the Wanamaker Mile. Boxing is also a prominent part of the city's sporting scene, with events**
  - chunk: Open golf tournament and the 1930 and 1939 PGA Championships, and has been host city for both events several times, most notably for nearby Winged Foot Golf Club. Charles Mingus was born in Nogales, Arizona. His father, Charles Mingus Sr. , was a sergeant in the U. S. Army. Mingus was largely raised…
- `group_026:303` (segment 1/1, 48 tokens, stop): **Stub context: Connolly (born May 7, 1981) is an American former professional ice hockey player who played eleven seasons in the National Hockey League (NHL) for the New York Islanders, Buffalo Sabres,**
  - chunk: The 2005–06 Southern Professional Hockey League season was the second season of the Southern Professional Hockey League. The regular season began October 21, 2005, and ended April 12, 2006, after a 56-game regular season and a six-team playoff. The Knoxville Ice Bears won their first SPHL championsh…
- `group_026:226` (segment 1/1, 35 tokens, stop): **Stub context: The single marked a change in the band that could be seen after the year-long break they took from touring and recording, this time working with the record label Sony**
  - chunk: The album was re-released in 2008 with three bonus tracks, one of which — a cover of Michael Bublé's "Home" — was released as a single, becoming Shelton's fourth Number One country hit. Of the eleven tracks, Shelton co-wrote three. The album has been certified Gold by RIAA. Candy Coburn is an Americ…
- `group_036:84` (segment 1/1, 50 tokens, stop): **Stub context: Stockton and Kearny went on to recover Los Angeles and force the capitulation of Alta California with the "Treaty of Cahuenga" on January 13, 1847. As a result of the**
  - chunk: The Mexican negotiators of that treaty tried to retain San Diego as part of Mexico, but the Americans insisted that San Diego was "for every commercial purpose of nearly equal importance to us with that of San Francisco," and the Mexican-American border was eventually established to be one league so…
- `group_026:192` (segment 1/1, 40 tokens, stop): **Stub context: Abdul Rehman Makki, in many of his public speeches and sermons, has stated that the war on terror started by America has caused the death of thousands of innocents in**
  - chunk: He was elected to the Académie française on 26 June 1975, replacing Jean Cardinal Daniélou, a post he held until his death on 15 January 2004 at Ancourt, in France. Both the main painting, ""Ansidei Madonna"", and the predella ""Saint John the Baptist Preaching"", are located at the National Gallery…

## 11. Estimates and timings

- Generation estimate 15 min for 771 calls (per call 9.3e-06 s x visible document tokens + 0.15 s, plus 420 s engine start (B2's measured rate on the same model and runtime)); measured 3s.
- BM25: estimate 7s on CPU (C2's CPU rate x 3 (published streams, empty-context pass, CR pass)); measured 2s.

