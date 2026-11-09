# C1 Contextual Retrieval: LooGLE

- Written 2026-11-09 17:32:47; commit d2ab689101df; documents 105, chunks 3346, queries 1832 (expected 1832 on the full dataset).
- Status: generation done; jobs BM25 DONE, Contriever DONE, BGE-M3 DONE, Jina DONE, Qwen FAILED.
- PROBLEM: qwen: FAILED

## 1. Correctness check (empty context reproduces Vanilla)

CR with every context empty indexes exactly the chunk text, so it must give Vanilla (SAADI's text channel alone) and the paper's saved results. BM25: identical rankings; dense: identical top 10 for at least 99% of queries and HR within 0.1 points (C2's Jina drift rule applies). The pool itself is checked by reproducing SAADI's saved rankings with the published weights.

| Retriever | Gate | SAADI top-10 = saved | Empty = Vanilla top-10 | HR@5 / HR@10 empty vs Vanilla | = saved text-only file | Table 1 Vanilla |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 | pass | 1832/1832 | 1832/1832 (top-150 1832) | 91.87 / 95.74 vs 91.87 / 95.74 | 1832/1832 | 91.5 / 95.4 |
| Contriever | pass | 1832/1832 | 1832/1832 (top-150 1832) | 75.16 / 86.03 vs 75.16 / 86.03 | no file | 72.4 / 73.3 |
| BGE-M3 | pass | 1832/1832 | 1832/1832 (top-150 1832) | 82.48 / 90.72 vs 82.48 / 90.72 | no file | 80.6 / 89.8 |
| Jina | pass | 1519/1832 | 1832/1832 (top-150 1832) | 69.92 / 80.68 vs 69.92 / 80.68 | no file | 70.1 / 80.1 |
| Qwen | FAIL | 733/1832 | 1832/1832 (top-150 1832) | 77.89 / 86.19 vs 77.89 / 86.19 | no file | 76.9 / 85.9 |
- BM25 note: empty-context HR@5 91.87 vs Table 1 Vanilla 91.5 (tolerance 0.05) - not gated: on this dataset Table 1 Vanilla is not the text channel of the saved chunks
- BM25 note: empty-context HR@10 95.74 vs Table 1 Vanilla 95.4 (tolerance 0.05) - not gated: on this dataset Table 1 Vanilla is not the text channel of the saved chunks
- BM25: Vanilla reference for the dense check = C2 saved streams (independent run); memory and summary channels from {'c2_streams': 105, 'published_path': 0} documents.
- Contriever note: empty-context HR@5 75.16 vs Table 1 Vanilla 72.4 (tolerance 0.15) - not gated: on this dataset Table 1 Vanilla is not the text channel of the saved chunks
- Contriever note: empty-context HR@10 86.03 vs Table 1 Vanilla 73.3 (tolerance 0.15) - not gated: on this dataset Table 1 Vanilla is not the text channel of the saved chunks
- Contriever: Vanilla reference for the dense check = C2 saved streams (independent run); memory and summary channels from {'c2_streams': 105, 'published_path': 0} documents.
- BGE-M3 note: empty-context HR@5 82.48 vs Table 1 Vanilla 80.6 (tolerance 0.15) - not gated: on this dataset Table 1 Vanilla is not the text channel of the saved chunks
- BGE-M3 note: empty-context HR@10 90.72 vs Table 1 Vanilla 89.8 (tolerance 0.15) - not gated: on this dataset Table 1 Vanilla is not the text channel of the saved chunks
- BGE-M3: Vanilla reference for the dense check = C2 saved streams (independent run); memory and summary channels from {'c2_streams': 105, 'published_path': 0} documents.
- Jina note: empty-context HR@10 80.68 vs Table 1 Vanilla 80.1 (tolerance 0.50) - not gated: on this dataset Table 1 Vanilla is not the text channel of the saved chunks
- Jina: Vanilla reference for the dense check = C2 saved streams (independent run); memory and summary channels from {'c2_streams': 105, 'published_path': 0} documents.
- Qwen PROBLEM: identical top-10 for 733/1832 queries (< 99%)
- Qwen PROBLEM: HR@5 77.13 vs saved 77.02 (tolerance 0.1 points)
- Qwen PROBLEM: HR@10 86.41 vs saved 86.68 (tolerance 0.1 points)
- Qwen PROBLEM: HR@5 77.13 vs expected 77.02
- Qwen PROBLEM: HR@10 86.41 vs expected 86.68
- Qwen note: empty-context HR@5 77.89 vs Table 1 Vanilla 76.9 (tolerance 0.15) - not gated: on this dataset Table 1 Vanilla is not the text channel of the saved chunks
- Qwen note: empty-context HR@10 86.19 vs Table 1 Vanilla 85.9 (tolerance 0.15) - not gated: on this dataset Table 1 Vanilla is not the text channel of the saved chunks
- Qwen: Vanilla reference for the dense check = published path in this process (shared encoding cache); memory and summary channels from {'c2_streams': 0, 'published_path': 105} documents.
- On LooGLE, Table 1's Vanilla is not the text channel of the saved chunks (C2's server run: they differ by up to 13 points); C1's Vanilla is the text channel, and the gate uses the saved per-query text-only file where it exists.

## 2. Token census (Qwen3 tokenizer)

- Window 262144 tokens (model config, rope_scaling None); prompt template 70 tokens; output budget 1024.
- Document tokens: min 14K, median 19K, max 60K; chunk tokens median 666, max 1409.
- Documents over the window: 0 of 105 (0 chunks); segments per document: {'1': 105}.
- Split vs joint tokenization of the prompt identical on the checked samples: [True, True, True].

## 3. GPU memory check (embedding models)

| Model | GPUs | Vanilla batch | Chosen batch | Peak MiB | Total MiB | Fits (10% margin) |
| --- | --- | --- | --- | --- | --- | --- |
| Contriever | 1 | 128 | 128 | [3306] | [48541] | yes |
| BGE-M3 | 1 | 16 | 16 | [4004] | [48541] | yes |
| Jina | 1 | 64 | 64 | [2082] | [48541] | yes |
| Qwen | 2 | 64 | 64 | [27106, 28446] | [48541, 48541] | yes |
- Longest plain chunk of this group (3907 characters) with 1024 placeholder tokens in front, float32 as Vanilla.

## 4. Long documents (segment rule)

- No document exceeds the window; every chunk saw its whole document.

## 5. Contexts

- 3346 contexts (0 missing), generator ['vllm']; empty 0, failed 0, cut at 1024 tokens 0.
- Context length (tokens): min 40, p25 103, median 127, mean 130.1, p75 154, p95 199, max 324.
- Cached tokens: ['vllm num_cached_tokens'].

## 6. Cost next to SAADI

| Quantity | CR | SAADI (memory + hierarchy calls) |
| --- | --- | --- |
| LLM calls (total) | 3346 | 4046 |
| LLM calls per chunk | 1.00 | 1.21 |
| LLM calls per document | 31.9 | 38.5 |
| Input tokens per chunk, nominal (full prompt) | 26K | 4608 (no prefix cache) |
| Input tokens per chunk, computed (after prefix-cache hits) | 1478 | same as nominal |
| Input tokens total, nominal / computed | 86.0M / 4.9M | 15.4M |
| Generated tokens per chunk, mean / median | 130.1 / 127.0 | 1755.7 |
| Generated tokens total | 435K | 5.9M |
| Contexts cut at 1024 tokens | 0 (0.0%) | — |
| Generation wall clock / GPU-hours | 37 min / 2.45 | not in the paper's tables |
| Indexing + scoring, BM25 (wall / GPU-hours) | 21s / 0.00 | — |
| Indexing + scoring, Contriever (wall / GPU-hours) | 2 min / 0.03 | — |
| Indexing + scoring, BGE-M3 (wall / GPU-hours) | 8 min / 0.13 | — |
| Indexing + scoring, Jina (wall / GPU-hours) | 2 min / 0.04 | — |
- Ratios per chunk. SAADI's builds had no prefix caching, so its input is compared with both CR columns: CR nominal / SAADI =5.58x, CR computed / SAADI = 0.32x; calls 0.83x; generated tokens 0.07x. SAADI chunks in the cost table: 3346; C1 chunks: 3346.

## 7. Retrieval: per retriever

HR@5 / HR@10 (%). *Complementarity check, not a proposed method.

| Retriever | Vanilla (text channel) | Table 1 Vanilla | CR | SAADI (published) | SAADI on CR text* | LQCA (Table 1) | SAADI held-out (C2) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BM25 | 91.9 / 95.7 | 91.5 / 95.4 | 92.0 / 96.0 | 92.3 / 96.5 | 92.4 / 96.1 | 88.8 / 94.2 | 92.3 / 96.2 |
| Contriever | 75.2 / 86.0 | 72.4 / 73.3 | 72.5 / 84.3 | 75.4 / 85.7 | 72.4 / 84.2 | 73.3 / 84.5 | 76.3 / 86.2 |
| BGE-M3 | 82.5 / 90.7 | 80.6 / 89.8 | 82.2 / 90.1 | 81.1 / 90.1 | 79.9 / 89.6 | 74.8 / 84.1 | 82.3 / 90.6 |
| Jina | 69.9 / 80.7 | 70.1 / 80.1 | 65.8 / 79.7 | 70.2 / 82.0 | 67.5 / 80.6 | 68.6 / 79.1 | — / — |

Paired differences (points) with 95% bootstrap intervals (10,000 resamples over queries, seed 42):

| Retriever | CR − Vanilla @5 | @10 | SAADI − CR @5 | @10 | SAADI-on-CR − CR @5 | @10 |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 | +0.2 [-0.6, +0.9] | +0.2 [-0.3, +0.8] | +0.3 [-0.6, +1.2] | +0.5 [-0.2, +1.2] | +0.3 [-0.5, +1.2] | +0.2 [-0.4, +0.8] |
| Contriever | -2.7 [-4.5, -0.9] | -1.7 [-3.2, -0.2] | +2.9 [+1.1, +4.8] | +1.4 [-0.2, +2.9] | -0.1 [-1.5, +1.3] | -0.1 [-1.3, +1.0] |
| BGE-M3 | -0.3 [-1.9, +1.2] | -0.7 [-2.0, +0.7] | -1.1 [-2.7, +0.5] | +0.1 [-1.3, +1.4] | -2.2 [-3.6, -1.0] | -0.4 [-1.4, +0.5] |
| Jina | -4.1 [-6.1, -2.3] | -1.0 [-2.7, +0.7] | +4.4 [+2.5, +6.4] | +2.3 [+0.6, +4.0] | +1.7 [+0.3, +3.1] | +0.9 [-0.3, +2.1] |

## 8. Retrieval: hybrid (dense + BM25, cookbook fusion)

| Hybrid | Vanilla @5 / @10 | CR @5 / @10 | SAADI @5 / @10 | CR − Vanilla @10 | SAADI − CR @10 | SAADI − CR @5 |
| --- | --- | --- | --- | --- | --- | --- |
| Contriever + BM25 | 89.3 / 95.6 | 88.3 / 95.6 | 88.3 / 95.2 | +0.0 [-0.8, +0.8] | -0.4 [-1.3, +0.5] | +0.1 [-1.3, +1.4] |
| BGE-M3 + BM25 | 91.4 / 96.1 | 91.6 / 96.5 | 90.4 / 96.0 | +0.4 [-0.3, +1.1] | -0.5 [-1.3, +0.3] | -1.1 [-2.3, +0.1] |
| Jina + BM25 | 88.2 / 94.7 | 87.4 / 94.9 | 87.1 / 94.8 | +0.2 [-0.6, +1.0] | -0.1 [-1.0, +0.8] | -0.3 [-1.6, +1.1] |
- Top 150 of each list, score 0.8 / rank_dense + 0.2 / rank_bm25 (1-based), the same merge for Vanilla, CR and SAADI (SAADI = SAADI with d + SAADI with BM25, published weights). No reranker.

## 9. Dense truncation of the contextualized chunks

| Retriever | Max input | CR chunks over max | CR tokens cut (end of chunk) | Plain chunks over max |
| --- | --- | --- | --- | --- |
| Contriever | 512 | 3293/3346 | 875K | 3268/3346 |
| BGE-M3 | 8192 | 0/3346 | 0 | 0/3346 |
| Jina | 512 | 3302/3346 | 1.2M | 3279/3346 |

## 10. Sanity sample (10 random contexts, seed 42)

- `c1ef894c-1533-40b4-819b-c64c75a77c7e:13` (segment 1/1, 201 tokens, stop): **This chunk discusses the literary analysis of the third and fourth fyttes of *A Gest of Robyn Hood*, focusing on the composition and narrative structure of the poem. It identifies the third fytte as likely composed of multiple now-lost tales, including a light-hearted archery scene, a comedic episode involving Little John’s revenge on the Sheriff, and a somber ambush that foreshadows later events. The fourth fytte is analyzed as introducing the tale of Robin and the Monk, which concludes the "Miracle of the Virgin Mary" narrative, with the Monk serving as the Virgin’s messenger who repays Robi**
  - chunk: 47  to be used as a plot device to delay the Knight, thus preparing for the tale of Robin Hood and the Monk in the Fourth Fytte. . Third Fytte. This episode probably consists of three or four now-lost tales. The light-hearted opening scene at the archery shoot: lines 577–600  could have been borrowe…
- `2f354723-b5d4-4ddc-aad4-bd5b806e86f8:43` (segment 1/1, 135 tokens, stop): **The chunk describes Sucre's anti-Peruvian actions during the Gran Colombian-Peruvian War, specifically his construction of a triumphalist obelisk that exaggerated the size of the Peruvian army and minimized the Colombian forces, along with the desecration of Peruvian dead, which led to protests from Peruvian leader José de La Mar and the eventual suspension of the Girón Agreement, reigniting hostilities. This episode is part of a broader narrative in the document detailing historical anti-Peruvian sentiment across Latin America, particularly from figures like Bolívar and Sucre, who are portray**
  - chunk: Later, during the Gran Colombian-Peruvian War, Sucre's anti-Peruvian attitudes were denounced, who, to commemorate the triumph of the Gran Columbian army over the Peruvian advance, ordered the construction of a commemorative obelisk on the site where the battle occurred, with an inscription increasi…
- `1014f6b2-bf2f-4591-b1f4-6c380d183280:12` (segment 1/1, 130 tokens, stop): **This chunk details key events during the Quebec Biker War (1994–2002), focusing on the assassination of Hells Angels associate Normand Hamel, the wrongful conviction and subsequent acquittal of Rock Machine member Tony Duguay due to police corruption, the murder of loan shark Robert "Bob" Savard, and the role of hitman Gerald Gallant. It also covers the major police crackdown Operation Amigo in 2002, which led to the arrest of 63 Rock Machine members, effectively ending the conflict and marking a turning point in the war between the Rock Machine and the Hells Angels.**
  - chunk: Tony Duguay, the Rock Machine member who was convicted of killing Hamel in 2006, was acquitted in 2016 when it was established that the eyewitness who gave the testimony that had convicted him had been fed information incriminating him by Detective Benoît Roberge, the senior anti-biker detective wit…
- `e7b76e86-d43b-4f40-8ecf-d51b11e0b08e:6` (segment 1/1, 112 tokens, stop): **The chunk describes the Union Nationale's highly personalized, media-savvy, and financially extravagant electoral strategies during Maurice Duplessis's second government (1944–1959), highlighting the party's use of marketing professionals, data-driven decisionmaking, and massive spending—fueled by post-war prosperity and systemic corruption through clientelism and pork-barrel projects. This section is part of a broader discussion on Duplessis's authoritarian governance, electoral dominance, and the innovative yet controversial campaign techniques that solidified his long rule.**
  - chunk: a great measure of personalization of the electoral communications, usage of marketing professionals (in this case, recruited from the party officials themselves), data-guided decisionmaking, usage of all media outlets available, and lavish spending. In pre-1948 elections, the main barrier for the U…
- `65ab618b-5ec6-4a30-a59c-db6fc24cfd29:5` (segment 1/1, 145 tokens, stop): **The chunk discusses the constitutional challenges to the notice and objection provisions in India's secular marriage laws (Special Marriage Act and Foreign Marriage Act), arguing they violate Articles 14, 15, 19, and 21 by infringing on privacy, decisional autonomy, and equality. It cites recommendations from the Law Commission to abolish these provisions to prevent unwarranted interference in marriages, and highlights judicial precedents from Delhi and Allahabad High Courts that have questioned their constitutionality. The chunk also extends the argument to the broader issue of legal recognit**
  - chunk: The Law Commission published a report on the Prevention of Interference with the freedom of Matrimonial Alliances that recommended the removal of the notice and objection provisions of the secular marriage laws to prevent “high-handed or unwarranted interference” in marriages. Violations of decision…
- `58700aa8-a869-4184-b2cf-80f0d9f123e4:4` (segment 1/1, 72 tokens, stop): **This chunk describes the pivotal second-half comeback by the Jacksonville Jaguars against the Los Angeles Chargers in the AFC Wild Card Playoff game on January 14, 2023, where the Jaguars overcame a 27–0 deficit to force overtime and eventually won on a walk-off field goal, marking the third-largest comeback in NFL postseason history.**
  - chunk: . Jacksonville was quickly forced to punt and the Chargers drove back for more points, this time going 67 yards in 13 plays, including Herbert's 23-yard completion to tight end Gerald Everett to convert a 3rd and 11. On the last play, he hit Everett again for a 9-yard score, increasing the team's le…
- `53125ece-f520-41f7-94bf-d3e1967f6498:15` (segment 1/1, 125 tokens, stop): **This chunk covers the Western Conference First Round series between the Phoenix Suns and Los Angeles Clippers, highlighting key performances, pivotal moments, and the Suns' eventual 4–1 series victory. It details the Clippers' loss of Paul George, Kawhi Leonard's standout play in Game 1, and the Suns' dominant offensive output led by Devin Booker and Kevin Durant, including their 110-point scoring runs in Games 2 and 3. The section also notes the first playoff meeting between these two teams and the significance of Chris Paul ending his 13-game playoff losing streak when officiated by Scott Fo**
  - chunk: Although they trailed at halftime, the Warriors opened the second half with a 22–8 run and held Sacramento to 42 points on 33% shooting after the break. . This was the first playoff meeting between these two teams. (4) Phoenix Suns vs. (5) Los Angeles Clippers. In the absence of Paul George, Kawhi L…
- `3924a3aa-e4eb-4b8a-9d20-96fae3ced0a3:10` (segment 1/1, 139 tokens, stop): **The chunk describes two pivotal historical events in Madrid's history: the Esquilache Riots of 1766, which ended with the banishment of Minister Esquilache, and the May 2, 1808 uprising against French occupation during the War of Independence. The latter event, centered on the Puerta del Sol, marked a turning point in Spanish history, leading to widespread resistance against Napoleon's forces and inspiring Goya's famous painting "The Charge of the Mamluks." The document situates these events within the broader narrative of Madrid's urban and political evolution, highlighting the Puerta del Sol**
  - chunk: Finally Charles III agreed, and Esquilache left the city with his family, ending the revolt. May 2, 1808. The presence of French troops in Spain, by virtue of the Treaty of Fontainebleau, had become threatening as they occupied various Spanish localities, without any support from the treaty. The tot…
- `e17debb6-5ed4-40f7-a345-dddca265cc46:7` (segment 1/1, 104 tokens, stop): **This chunk falls within the late 1960s section of the document, following Israel's victory in the Six-Day War (1967), and covers key developments including the post-war status of Jews in Arab states, the expansion of Jewish access to holy sites in Jerusalem, Hebron, and Bethlehem, the establishment of the first religious settlement at Kiryat Arba, educational reforms, the Karameh attack, and Golda Meir's rise to power as Israel's first female Prime Minister.**
  - chunk: Today there are about 9,000 Jews living in Arab states, of whom 75% live in Morocco and 15% in Tunisia. Vast assets, approximately $150 billion worth of goods and property (before inflation) were left behind in these countries. After 1967 the Soviet block (except Romania) broke off relations with Is…
- `2f354723-b5d4-4ddc-aad4-bd5b806e86f8:1` (segment 1/1, 188 tokens, stop): **The chunk describes a pivotal moment in early Argentine revolutionary history, highlighting the Junta's harsh execution order against counterrevolutionary leaders from Córdoba, framed as a warning to Peruvian elites. It then shifts to the controversial "Plan del Inca" proposed by Manuel Belgrano at the Congress of Tucumán, which sought to establish a constitutional Inca monarchy with a Peruvian sovereign—likely Juan Bautista Túpac Amaru—to legitimize Argentine independence through international recognition. The proposal was met with fierce rejection by Buenos Aires delegates, driven by anti-Pe**
  - chunk: The Board orders that they be harquebused Dn. Santiago Liniers, Don Juan Gutiérrez de la Concha, the Bishop of Córdoba, Dn. Victorino Rodríguez, Colonel Allende and the Royal Official Dn. Joaquin Moreno. At the moment that each or every one of them is caught, whatever the circumstances, this resolut…

## 11. Estimates and timings

- Generation estimate 28 min for 3346 calls (per call 9.3e-06 s x visible document tokens + 0.15 s, plus 420 s engine start (B2's measured rate on the same model and runtime)); measured 37 min.
- BM25: estimate 28s on CPU (C2's CPU rate x 3 (published streams, empty-context pass, CR pass)); measured 21s.
- Contriever: estimate 2 min on 1 GPU(s) (C2 streams reused: CR chunks + plain chunks (empty-context check) encoded, 2 query encodings per query); measured 2 min.
- BGE-M3: estimate 11 min on 1 GPU(s) (C2 streams reused: CR chunks + plain chunks (empty-context check) encoded, 2 query encodings per query); measured 8 min.
- Jina: estimate 9 min on 1 GPU(s) (C2 streams reused: CR chunks + plain chunks (empty-context check) encoded, 2 query encodings per query); measured 2 min.
- Qwen: estimate 2.1 h on 2 GPU(s) (C2's estimate for the published streams + CR chunks encoded); measured 56 min.

