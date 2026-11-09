# C1 Contextual Retrieval: LooGLE (dry run full: stub contexts, BM25 only, placeholder numbers)

- Written 2026-11-09 14:59:53; commit e2dbaa178fa0; documents 105, chunks 3346, queries 1832 (expected 1832 on the full dataset).
- Status: generation done; jobs BM25 DONE.

## 1. Correctness check (empty context reproduces Vanilla)

CR with every context empty indexes exactly the chunk text, so it must give Vanilla (SAADI's text channel alone) and the paper's saved results. BM25: identical rankings; dense: identical top 10 for at least 99% of queries and HR within 0.1 points (C2's Jina drift rule applies). The pool itself is checked by reproducing SAADI's saved rankings with the published weights.

| Retriever | Gate | SAADI top-10 = saved | Empty = Vanilla top-10 | HR@5 / HR@10 empty vs Vanilla | = saved text-only file | Table 1 Vanilla |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 | pass | 1832/1832 | 1832/1832 (top-150 1832) | 91.87 / 95.74 vs 91.87 / 95.74 | 1832/1832 | 91.5 / 95.4 |
- BM25 note: empty-context HR@5 91.87 vs Table 1 Vanilla 91.5 (tolerance 0.05) - not gated: on this dataset Table 1 Vanilla is not the text channel of the saved chunks
- BM25 note: empty-context HR@10 95.74 vs Table 1 Vanilla 95.4 (tolerance 0.05) - not gated: on this dataset Table 1 Vanilla is not the text channel of the saved chunks
- BM25: Vanilla reference for the dense check = published path in this process (shared encoding cache); memory and summary channels from {'c2_streams': 0, 'published_path': 105} documents.
- On LooGLE, Table 1's Vanilla is not the text channel of the saved chunks (C2's server run: they differ by up to 13 points); C1's Vanilla is the text channel, and the gate uses the saved per-query text-only file where it exists.

## 2. Token census (Qwen3 tokenizer)

- Window 262144 tokens (model config, rope_scaling None); prompt template 70 tokens; output budget 1024.
- Document tokens: min 14K, median 19K, max 60K; chunk tokens median 666, max 1409.
- Documents over the window: 0 of 105 (0 chunks); segments per document: {'1': 105}.
- Split vs joint tokenization of the prompt identical on the checked samples: [True, True, True].

## 3. GPU memory check (embedding models)

- Skipped (CPU dry run).

## 4. Long documents (segment rule)

- No document exceeds the window; every chunk saw its whole document.

## 5. Contexts

- 3346 contexts (0 missing), generator ['stub']; empty 0, failed 0, cut at 1024 tokens 0.
- Context length (tokens): min 33, p25 39, median 43, mean 45.5, p75 49, p95 64, max 98.
- Cached tokens: ['simulated (stub)'].

## 6. Cost next to SAADI

| Quantity | CR | SAADI (memory + hierarchy calls) |
| --- | --- | --- |
| LLM calls (total) | 3346 | 4046 |
| LLM calls per chunk | 1.00 | 1.21 |
| LLM calls per document | 31.9 | 38.5 |
| Input tokens per chunk, nominal (full prompt) | 26K | 4608 (no prefix cache) |
| Input tokens per chunk, computed (after prefix-cache hits) | 1453 | same as nominal |
| Input tokens total, nominal / computed | 86.0M / 4.9M | 15.4M |
| Generated tokens per chunk, mean / median | 45.5 / 43.0 | 1755.7 |
| Generated tokens total | 152K | 5.9M |
| Contexts cut at 1024 tokens | 0 (0.0%) | — |
| Generation wall clock / GPU-hours | 9s / 0.00 | not in the paper's tables |
| Indexing + scoring, BM25 (wall / GPU-hours) | 8s / 0.00 | — |
- Ratios per chunk. SAADI's builds had no prefix caching, so its input is compared with both CR columns: CR nominal / SAADI =5.58x, CR computed / SAADI = 0.32x; calls 0.83x; generated tokens 0.03x. SAADI chunks in the cost table: 3346; C1 chunks: 3346.

## 7. Retrieval: per retriever

HR@5 / HR@10 (%). *Complementarity check, not a proposed method.

| Retriever | Vanilla (text channel) | Table 1 Vanilla | CR | SAADI (published) | SAADI on CR text* | LQCA (Table 1) |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 | 91.9 / 95.7 | 91.5 / 95.4 | 91.3 / 95.7 | 92.3 / 96.5 | 91.8 / 96.0 | 88.8 / 94.2 |
- SAADI held-out column left out: results/rebuttal/C2_heldout_fusion/REPORT.md does not exist yet.

Paired differences (points) with 95% bootstrap intervals (10,000 resamples over queries, seed 42):

| Retriever | CR − Vanilla @5 | @10 | SAADI − CR @5 | @10 | SAADI-on-CR − CR @5 | @10 |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 | -0.5 [-1.1, -0.1] | -0.1 [-0.4, +0.3] | +1.0 [+0.2, +1.9] | +0.8 [+0.0, +1.5] | +0.5 [-0.3, +1.3] | +0.3 [-0.4, +1.1] |

## 8. Retrieval: hybrid (dense + BM25, cookbook fusion)

| Hybrid | Vanilla @5 / @10 | CR @5 / @10 | SAADI @5 / @10 | CR − Vanilla @10 | SAADI − CR @10 | SAADI − CR @5 |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 + BM25 (dry-run stand-in for a dense retriever) | 91.9 / 95.7 | 91.3 / 95.7 | 92.3 / 96.5 | -0.1 [-0.4, +0.3] | +0.8 [+0.0, +1.5] | +1.0 [+0.2, +1.9] |
- Top 150 of each list, score 0.8 / rank_dense + 0.2 / rank_bm25 (1-based), the same merge for Vanilla, CR and SAADI (SAADI = SAADI with d + SAADI with BM25, published weights). No reranker.

## 9. Dense truncation of the contextualized chunks

- No dense retriever in this run.

## 10. Sanity sample (10 random contexts, seed 42)

- `c1ef894c-1533-40b4-819b-c64c75a77c7e:13` (segment 1/1, 40 tokens, stop): **Stub context: In contrast to Clawson, who struggled mightily to connect Gest with existing outlaw ballads, Holt's study indicated that none of the sources have survived, that the tales were not necessarily**
  - chunk: 47  to be used as a plot device to delay the Knight, thus preparing for the tale of Robin Hood and the Monk in the Fourth Fytte. . Third Fytte. This episode probably consists of three or four now-lost tales. The light-hearted opening scene at the archery shoot: lines 577–600  could have been borrowe…
- `2f354723-b5d4-4ddc-aad4-bd5b806e86f8:43` (segment 1/1, 40 tokens, stop): **Stub context: Through Funes, Sucre addressed the issue of an alliance between Bolivia, Argentina and Chile, which was supposedly to stop Brazilian expansionism, but in practice he was looking for support against**
  - chunk: Later, during the Gran Colombian-Peruvian War, Sucre's anti-Peruvian attitudes were denounced, who, to commemorate the triumph of the Gran Columbian army over the Peruvian advance, ordered the construction of a commemorative obelisk on the site where the battle occurred, with an inscription increasi…
- `1014f6b2-bf2f-4591-b1f4-6c380d183280:12` (segment 1/1, 38 tokens, stop): **Stub context: He had asked for a local trucker to store the bags for him, only for the trucker to be robbed at gunpoint by four members of the Rock Machine. The**
  - chunk: Tony Duguay, the Rock Machine member who was convicted of killing Hamel in 2006, was acquitted in 2016 when it was established that the eyewitness who gave the testimony that had convicted him had been fed information incriminating him by Detective Benoît Roberge, the senior anti-biker detective wit…
- `e7b76e86-d43b-4f40-8ecf-d51b11e0b08e:6` (segment 1/1, 37 tokens, stop): **Stub context: When Duplessis entered, they leapt to their feet and remained on their feet while the Prime Minister slowly made his way the length of the room to the head of**
  - chunk: a great measure of personalization of the electoral communications, usage of marketing professionals (in this case, recruited from the party officials themselves), data-guided decisionmaking, usage of all media outlets available, and lavish spending. In pre-1948 elections, the main barrier for the U…
- `65ab618b-5ec6-4a30-a59c-db6fc24cfd29:5` (segment 1/1, 67 tokens, stop): **Stub context: UOI (2017) and Shafin Jahan v Ashokan K. M. (2018), Laxmibai Chandaragi B. v. State of Karnataka (2021). Freedom of conscience and religion. Article 25 of the Indian Constitution guarantees**
  - chunk: The Law Commission published a report on the Prevention of Interference with the freedom of Matrimonial Alliances that recommended the removal of the notice and objection provisions of the secular marriage laws to prevent “high-handed or unwarranted interference” in marriages. Violations of decision…
- `58700aa8-a869-4184-b2cf-80f0d9f123e4:4` (segment 1/1, 47 tokens, stop): **Stub context: Seattle's ensuing possession ended when Charles Omenihu forced a fumble from Smith that Nick Bosa recovered for the 49ers at their 30-yard line. San Francisco then drove 70 yards in**
  - chunk: . Jacksonville was quickly forced to punt and the Chargers drove back for more points, this time going 67 yards in 13 plays, including Herbert's 23-yard completion to tight end Gerald Everett to convert a 3rd and 11. On the last play, he hit Everett again for a 9-yard score, increasing the team's le…
- `53125ece-f520-41f7-94bf-d3e1967f6498:15` (segment 1/1, 44 tokens, stop): **Stub context: . In a collaborative team effort, Stephen Curry scored 32 points, Klay Thompson added 26, and Jordan Poole dropped 22 points as the Warriors tied the series at two games**
  - chunk: Although they trailed at halftime, the Warriors opened the second half with a 22–8 run and held Sacramento to 42 points on 33% shooting after the break. . This was the first playoff meeting between these two teams. (4) Phoenix Suns vs. (5) Los Angeles Clippers. In the absence of Paul George, Kawhi L…
- `3924a3aa-e4eb-4b8a-9d20-96fae3ced0a3:10` (segment 1/1, 39 tokens, stop): **Stub context: Due to the increase in mail and the gradual need to manage the growing volume of mail, in 1792 the architect Juan Pedro Arnal designed what would become the Real**
  - chunk: Finally Charles III agreed, and Esquilache left the city with his family, ending the revolt. May 2, 1808. The presence of French troops in Spain, by virtue of the Treaty of Fontainebleau, had become threatening as they occupied various Spanish localities, without any support from the treaty. The tot…
- `e17debb6-5ed4-40f7-a345-dddca265cc46:7` (segment 1/1, 42 tokens, stop): **Stub context: On 26 May Nasser declared, "The battle will be a general one and our basic objective will be to destroy Israel". Israel considered the Straits of Tiran closure a Casus**
  - chunk: Today there are about 9,000 Jews living in Arab states, of whom 75% live in Morocco and 15% in Tunisia. Vast assets, approximately $150 billion worth of goods and property (before inflation) were left behind in these countries. After 1967 the Soviet block (except Romania) broke off relations with Is…
- `2f354723-b5d4-4ddc-aad4-bd5b806e86f8:1` (segment 1/1, 38 tokens, stop): **Stub context: By country. Argentina. In the midst of the Argentine War of Independence and the Auxiliary Expeditions to Upper Peru, there was a climate of tension between Peru, loyal to the**
  - chunk: The Board orders that they be harquebused Dn. Santiago Liniers, Don Juan Gutiérrez de la Concha, the Bishop of Córdoba, Dn. Victorino Rodríguez, Colonel Allende and the Royal Official Dn. Joaquin Moreno. At the moment that each or every one of them is caught, whatever the circumstances, this resolut…

## 11. Estimates and timings

- Generation estimate 28 min for 3346 calls (per call 9.3e-06 s x visible document tokens + 0.15 s, plus 420 s engine start (B2's measured rate on the same model and runtime)); measured 9s.
- BM25: estimate 28s on CPU (C2's CPU rate x 3 (published streams, empty-context pass, CR pass)); measured 8s.

