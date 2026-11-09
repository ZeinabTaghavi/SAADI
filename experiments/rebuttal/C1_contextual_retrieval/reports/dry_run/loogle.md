# C1 Contextual Retrieval: LooGLE (dry run: stub contexts, BM25 only, placeholder numbers)

- Written 2026-11-09 15:02:19; commit e2dbaa178fa0; documents 2, chunks 56, queries 21 (expected 1832 on the full dataset).
- Status: generation done; jobs BM25 DONE.

## 1. Correctness check (empty context reproduces Vanilla)

CR with every context empty indexes exactly the chunk text, so it must give Vanilla (SAADI's text channel alone) and the paper's saved results. BM25: identical rankings; dense: identical top 10 for at least 99% of queries and HR within 0.1 points (C2's Jina drift rule applies). The pool itself is checked by reproducing SAADI's saved rankings with the published weights.

| Retriever | Gate | SAADI top-10 = saved | Empty = Vanilla top-10 | HR@5 / HR@10 empty vs Vanilla | = saved text-only file | Table 1 Vanilla |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 | pass | 21/21 | 21/21 (top-150 21) | 85.71 / 90.48 vs 85.71 / 90.48 | 21/21 | 91.5 / 95.4 |
- BM25: Vanilla reference for the dense check = published path in this process (shared encoding cache); memory and summary channels from {'c2_streams': 0, 'published_path': 2} documents.
- On LooGLE, Table 1's Vanilla is not the text channel of the saved chunks (C2's server run: they differ by up to 13 points); C1's Vanilla is the text channel, and the gate uses the saved per-query text-only file where it exists.

## 2. Token census (Qwen3 tokenizer)

- Window 262144 tokens (model config, rope_scaling None); prompt template 70 tokens; output budget 1024.
- Document tokens: min 15K, median 17K, max 19K; chunk tokens median 620, max 727.
- Documents over the window: 0 of 2 (0 chunks); segments per document: {'1': 2}.
- Split vs joint tokenization of the prompt identical on the checked samples: [True, True].

## 3. GPU memory check (embedding models)

- Skipped (CPU dry run).

## 4. Long documents (segment rule)

- No document exceeds the window; every chunk saw its whole document.

## 5. Contexts

- 56 contexts (0 missing), generator ['stub']; empty 0, failed 0, cut at 1024 tokens 0.
- Context length (tokens): min 34, p25 39, median 42, mean 42.3, p75 44, p95 56, max 57.
- Cached tokens: ['simulated (stub)'].

## 6. Cost next to SAADI

| Quantity | CR | SAADI (memory + hierarchy calls) |
| --- | --- | --- |
| LLM calls (total) | 56 | 4046 |
| LLM calls per chunk | 1.00 | 1.21 |
| LLM calls per document | 28.0 | 38.5 |
| Input tokens per chunk, nominal (full prompt) | 18K | 4608 (no prefix cache) |
| Input tokens per chunk, computed (after prefix-cache hits) | 1282 | same as nominal |
| Input tokens total, nominal / computed | 1.0M / 72K | 15.4M |
| Generated tokens per chunk, mean / median | 42.3 / 41.5 | 1755.7 |
| Generated tokens total | 2370 | 5.9M |
| Contexts cut at 1024 tokens | 0 (0.0%) | — |
| Generation wall clock / GPU-hours | 2s / 0.00 | not in the paper's tables |
| Indexing + scoring, BM25 (wall / GPU-hours) | 0s / 0.00 | — |
- Ratios per chunk. SAADI's builds had no prefix caching, so its input is compared with both CR columns: CR nominal / SAADI =3.91x, CR computed / SAADI = 0.28x; calls 0.83x; generated tokens 0.02x. SAADI chunks in the cost table: 3346; C1 chunks: 56.

## 7. Retrieval: per retriever

HR@5 / HR@10 (%). *Complementarity check, not a proposed method.

| Retriever | Vanilla (text channel) | Table 1 Vanilla | CR | SAADI (published) | SAADI on CR text* | LQCA (Table 1) |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 | 85.7 / 90.5 | 91.5 / 95.4 | 81.0 / 90.5 | 90.5 / 95.2 | 90.5 / 90.5 | 88.8 / 94.2 |
- SAADI held-out column left out: results/rebuttal/C2_heldout_fusion/REPORT.md does not exist yet.

Paired differences (points) with 95% bootstrap intervals (10,000 resamples over queries, seed 42):

| Retriever | CR − Vanilla @5 | @10 | SAADI − CR @5 | @10 | SAADI-on-CR − CR @5 | @10 |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 | -4.8 [-14.3, +0.0] | +0.0 [+0.0, +0.0] | +9.5 [+0.0, +23.8] | +4.8 [+0.0, +14.3] | +9.5 [+0.0, +23.8] | +0.0 [+0.0, +0.0] |

## 8. Retrieval: hybrid (dense + BM25, cookbook fusion)

| Hybrid | Vanilla @5 / @10 | CR @5 / @10 | SAADI @5 / @10 | CR − Vanilla @10 | SAADI − CR @10 | SAADI − CR @5 |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 + BM25 (dry-run stand-in for a dense retriever) | 85.7 / 90.5 | 81.0 / 90.5 | 90.5 / 95.2 | +0.0 [+0.0, +0.0] | +4.8 [+0.0, +14.3] | +9.5 [+0.0, +23.8] |
- Top 150 of each list, score 0.8 / rank_dense + 0.2 / rank_bm25 (1-based), the same merge for Vanilla, CR and SAADI (SAADI = SAADI with d + SAADI with BM25, published weights). No reranker.

## 9. Dense truncation of the contextualized chunks

- No dense retriever in this run.

## 10. Sanity sample (10 random contexts, seed 42)

- `5984f759-6d02-4c94-858c-b4ca561ca5e5:23` (segment 1/1, 45 tokens, stop): **Stub context: The Jamiyyat al Ulama-i-Hind regarded the Muslim league's ‘hostage‘ theory that is the minority on both sides of the border, not only as political none sense but also as contrary**
  - chunk: .If an effective principle of cooperation is discovered in India, it will bring peace and mutual good will to this ancient land which has suffered so long, more because of her situation in historic space than because of any inherent incapacity of her people. And it will at the same time solve the en…
- `03c1b3e8-8aba-4a59-8a56-e24c4a2b88ec:15` (segment 1/1, 45 tokens, stop): **Stub context: In 1964 Picardo was involved, with the Ministry of Information and Tourism, in investigating old buildings for conversion into a new Parador in the Province of Guadalajara. Possible locations were**
  - chunk: The towers, too, were further raised, including the twin towers of the fortified gateway, the barbican, which over the centuries had been restyled almost as belfries with sloping conical roofs. . In search of a military external image Picardo removed all the large windows, balconies and other wide o…
- `03c1b3e8-8aba-4a59-8a56-e24c4a2b88ec:1` (segment 1/1, 56 tokens, stop): **Stub context: Early life. Picardo was born in Jerez de la Frontera, in the Province of Cádiz in Andalucía, Spain on 18 June 1919. His father was Alvaro Picardo de Celis and**
  - chunk: Particularly noted in his post-student days were illustrations portraying Madrid in the 1950s and 1960s, the Spanish protectorate in Morocco, and sketches of the Canarias (Canary Islands) in 1953. He also showed a growing interest in historic architecture, in particular its preservation and restorat…
- `5984f759-6d02-4c94-858c-b4ca561ca5e5:3` (segment 1/1, 41 tokens, stop): **Stub context: Iqbal's condemnation of concentration of wealth in few hands, exploitation of workers by capitalist class, his welfare feeling for the Punjab's peasants and the landlords’ unjust treatment of peasants, are**
  - chunk: In Iqbal's words an individual becomes a ‘dead matter’ if he ceases to know the importance of sprit within him. Ideas alone are not sufficient there must be action, movement, restlessness, love, and courageous sense of the importance of the self. Iqbal writes in the preface of his book which book th…
- `03c1b3e8-8aba-4a59-8a56-e24c4a2b88ec:3` (segment 1/1, 39 tokens, stop): **Stub context: He also moved on to a series of building commissions for the Spanish Ministerio de Información y Turismo (Ministry of Information and Tourism) which laid the foundation for his notable**
  - chunk: Around 1960 Picardo was rewarded by the Dirección General de Arquitectura (DGA) for the many illustrations he had provided for the DGA's Boletín since he was a student with the publication of a small book, Dibujos de José Luis Picardo (Drawings of José Luis Picardo). More than 60 drawings appear in …
- `03c1b3e8-8aba-4a59-8a56-e24c4a2b88ec:22` (segment 1/1, 39 tokens, stop): **Stub context: Externally most of the structure was coloured in a deep ochre, representative of the land and soil of the region while the infilling of the façades copied the stark white**
  - chunk: Guernica was installed in September 1981 and the room opened to the public on 25 October that year, Picasso's centenary. Within a year, over one million people had seen Guernica in its new Picardo/de Paredes setting. Opinions of the method of display differed. The artist's daughter, Paloma Picasso, …
- `03c1b3e8-8aba-4a59-8a56-e24c4a2b88ec:21` (segment 1/1, 44 tokens, stop): **Stub context: For the interior of the building Picardo designed several assembly halls, auditoria for concerts, theatre, cinema and conferences, along with numerous exhibition and gallery spaces, libraries, offices, Council rooms, conveniences**
  - chunk: Externally most of the structure was coloured in a deep ochre, representative of the land and soil of the region while the infilling of the façades copied the stark white of traditional Andalucían village homes. Rows of relief pillars were the perceived support for the immense crowning hip roof, wit…
- `03c1b3e8-8aba-4a59-8a56-e24c4a2b88ec:16` (segment 1/1, 46 tokens, stop): **Stub context: The towers, too, were further raised, including the twin towers of the fortified gateway, the barbican, which over the centuries had been restyled almost as belfries with sloping conical roofs.**
  - chunk: Wide wooden staircases on this eastern side led to the first and second floor bedrooms, a few of which were in the northeast tower with windows looking over the town, and some towards the south, but most looking into the courtyard with those on the upper floor benefitting from Picardo's typical  bal…
- `5984f759-6d02-4c94-858c-b4ca561ca5e5:6` (segment 1/1, 38 tokens, stop): **Stub context: . The elector had the right to demand the deposition of the Caliph, or the dismissal of his officials if their behavior was in contravention to the laws of the**
  - chunk: However, for Iqbal the establishment of a Muslim State in the Indian subcontinent was not an end by itself, but it was a means to achieve a higher goal-consolidation of the world-millat. Thus, the contradiction between Iqbal's theory of the Islami millat and his proposal for the establishment of a c…
- `03c1b3e8-8aba-4a59-8a56-e24c4a2b88ec:14` (segment 1/1, 45 tokens, stop): **Stub context: . As in previous Paradores built by Picardo he had control of the interior decoration down to the smallest detail, in Carmona achieving a Hispanic-Arabic ambience of a palatial Mudéjar**
  - chunk: In 1964 Picardo was involved, with the Ministry of Information and Tourism, in investigating old buildings for conversion into a new Parador in the Province of Guadalajara. Possible locations were the castle at Atienza and the Casa del Cordón, an old inn in the same town, the castle at Molina de Ara…

## 11. Estimates and timings

- Generation estimate 7 min for 56 calls (per call 9.3e-06 s x visible document tokens + 0.15 s, plus 420 s engine start (B2's measured rate on the same model and runtime)); measured 2s.
- BM25: estimate 0s on CPU (C2's CPU rate x 3 (published streams, empty-context pass, CR pass)); measured 0s.

