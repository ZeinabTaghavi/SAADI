# C1 Contextual Retrieval: NovelHopQA (dry run full: stub contexts, BM25 only, placeholder numbers)

- Written 2026-11-09 15:00:39; commit e2dbaa178fa0; documents 16, chunks 7011, queries 908 (expected 908 on the full dataset).
- Status: generation done; jobs BM25 DONE.

## 1. Correctness check (empty context reproduces Vanilla)

CR with every context empty indexes exactly the chunk text, so it must give Vanilla (SAADI's text channel alone) and the paper's saved results. BM25: identical rankings; dense: identical top 10 for at least 99% of queries and HR within 0.1 points (C2's Jina drift rule applies). The pool itself is checked by reproducing SAADI's saved rankings with the published weights.

| Retriever | Gate | SAADI top-10 = saved | Empty = Vanilla top-10 | HR@5 / HR@10 empty vs Vanilla | = saved text-only file | Table 1 Vanilla |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 | pass | 908/908 | 908/908 (top-150 908) | 17.51 / 20.70 vs 17.51 / 20.70 | 908/908 | 16.9 / 20.0 |
- BM25 note: empty-context HR@5 17.51 vs Table 1 Vanilla 16.9 (tolerance 0.05) - not gated: on this dataset Table 1 Vanilla is not the text channel of the saved chunks
- BM25 note: empty-context HR@10 20.70 vs Table 1 Vanilla 20.0 (tolerance 0.05) - not gated: on this dataset Table 1 Vanilla is not the text channel of the saved chunks
- BM25: Vanilla reference for the dense check = published path in this process (shared encoding cache); memory and summary channels from {'c2_streams': 0, 'published_path': 16} documents.
- On NovelHopQA, Table 1's Vanilla is not the text channel of the saved chunks (C2's server run: they differ by up to 13 points); C1's Vanilla is the text channel, and the gate uses the saved per-query text-only file where it exists.

## 2. Token census (Qwen3 tokenizer)

- Window 262144 tokens (model config, rope_scaling None); prompt template 70 tokens; output budget 1024.
- Document tokens: min 134K, median 234K, max 763K; chunk tokens median 660, max 1033.
- Documents over the window: 7 of 16 (4250 chunks); segments per document: {'1': 9, '2': 6, '3': 1}.
- Split vs joint tokenization of the prompt identical on the checked samples: [True, True, True].

## 3. GPU memory check (embedding models)

- Skipped (CPU dry run).

## 4. Long documents (segment rule)

| Document | Tokens | Chunks | Segments k |
| --- | --- | --- | --- |
| B62 | 463K | 729 | 2 |
| B26 | 269K | 383 | 2 |
| B42 | 763K | 1165 | 3 |
| B54 | 441K | 660 | 2 |
| B44 | 268K | 384 | 2 |
| B15 | 295K | 421 | 2 |
| B05 | 345K | 508 | 2 |
- These chunks saw only their own segment of the document, not the whole document: a limit of CR on book-length text.

## 5. Contexts

- 7011 contexts (0 missing), generator ['stub']; empty 0, failed 0, cut at 1024 tokens 0.
- Context length (tokens): min 33, p25 38, median 41, mean 41.8, p75 44, p95 51, max 112.
- Cached tokens: ['simulated (stub)'].

## 6. Cost next to SAADI

| Quantity | CR | SAADI (memory + hierarchy calls) |
| --- | --- | --- |
| LLM calls (total) | 7011 | 9634 |
| LLM calls per chunk | 1.00 | 1.25 |
| LLM calls per document | 438.2 | 535.2 |
| Input tokens per chunk, nominal (full prompt) | 204K | 4797 (no prefix cache) |
| Input tokens per chunk, computed (after prefix-cache hits) | 1392 | same as nominal |
| Input tokens total, nominal / computed | 1.43B / 9.8M | 37.1M |
| Generated tokens per chunk, mean / median | 41.8 / 41.0 | 1199.0 |
| Generated tokens total | 293K | 9.3M |
| Contexts cut at 1024 tokens | 0 (0.0%) | — |
| Generation wall clock / GPU-hours | 26s / 0.00 | not in the paper's tables |
| Indexing + scoring, BM25 (wall / GPU-hours) | 60s / 0.00 | — |
- Ratios per chunk. SAADI's builds had no prefix caching, so its input is compared with both CR columns: CR nominal / SAADI =42.63x, CR computed / SAADI = 0.29x; calls 0.80x; generated tokens 0.03x. SAADI chunks in the cost table: 7736; C1 chunks: 7011.

## 7. Retrieval: per retriever

HR@5 / HR@10 (%). *Complementarity check, not a proposed method.

| Retriever | Vanilla (text channel) | Table 1 Vanilla | CR | SAADI (published) | SAADI on CR text* | LQCA (Table 1) |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 | 17.5 / 20.7 | 16.9 / 20.0 | 16.7 / 20.7 | 18.1 / 22.0 | 17.5 / 21.9 | 1.4 / 1.9 |
- SAADI held-out column left out: results/rebuttal/C2_heldout_fusion/REPORT.md does not exist yet.

Paired differences (points) with 95% bootstrap intervals (10,000 resamples over queries, seed 42):

| Retriever | CR − Vanilla @5 | @10 | SAADI − CR @5 | @10 | SAADI-on-CR − CR @5 | @10 |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 | -0.8 [-1.5, -0.1] | +0.0 [-0.6, +0.6] | +1.3 [+0.2, +2.4] | +1.3 [+0.1, +2.5] | +0.8 [-0.2, +1.8] | +1.2 [+0.1, +2.4] |

## 8. Retrieval: hybrid (dense + BM25, cookbook fusion)

| Hybrid | Vanilla @5 / @10 | CR @5 / @10 | SAADI @5 / @10 | CR − Vanilla @10 | SAADI − CR @10 | SAADI − CR @5 |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 + BM25 (dry-run stand-in for a dense retriever) | 17.5 / 20.7 | 16.7 / 20.7 | 18.1 / 22.0 | +0.0 [-0.6, +0.6] | +1.3 [+0.1, +2.5] | +1.3 [+0.2, +2.4] |
- Top 150 of each list, score 0.8 / rank_dense + 0.2 / rank_bm25 (1-based), the same merge for Vanilla, CR and SAADI (SAADI = SAADI with d + SAADI with BM25, published weights). No reranker.

## 9. Dense truncation of the contextualized chunks

- No dense retriever in this run.

## 10. Sanity sample (10 random contexts, seed 42)

- `B44:73` (segment 1/2, 39 tokens, stop): **Stub context: a private fear had haunted me, that in thus acting for myself, and by my own guidance, I ran the risk of getting into some scrape; and, above all things,**
  - chunk: Fairfax, and got that lady"s reply, stating that she was satisfied, and fixing that day fortnight as the period for my assuming the post of governess in her house. I now busied myself in preparations: the fortnight passed rapidly. I had not a very large wardrobe, though it was adequate to my wants;…
- `B12:174` (segment 1/1, 44 tokens, stop): **Stub context: “Your servant, Sir,” said Joe, “which I hope as you and Pip”—here his eye fell on the Avenger, who was putting some toast on table, and so plainly denoted an**
  - chunk: Indeed, it demanded from him a constant attention, and a quickness of eye and hand, very like that exacted by wicket-keeping. He made extraordinary play with it, and showed the greatest skill; now, rushing at it and catching it neatly as it dropped; now, merely stopping it midway, beating it up, and…
- `B04:282` (segment 1/1, 42 tokens, stop): **Stub context: At length he went back again, and took the road which leads from Hatfield to St. Albans. He went on doggedly; but as he left the town behind him, and**
  - chunk: The figure was behind him. He re-entered the shed, and shrunk down once more. The eyes were there, before he had laid himself along. And here he remained in such terror as none but he can know, trembling in every limb, and the cold sweat starting from every pore, when suddenly there arose upon the n…
- `B61:230` (segment 1/1, 36 tokens, stop): **Stub context: “ And it is pretty sure to be so indeed! And if some friend, not of too low or too high a grade in the service, should chance to pass**
  - chunk: milestones fly by, merchants on the front seats of their tilt-carts fly to meet one, the forest flies by on both sides with dark rows of firs and pines, with the ring of the axe and caw of the crows; the whole road flies into the unknown retreating distance; and there is something terrible in this r…
- `B26:184` (segment 1/2, 45 tokens, stop): **Stub context: ” Miss Ophelia took out a couple of dozen. “Thar’s some tickets in that ar old cracked jug on the top shelf,” said Dinah. “You, Jake, climb up and get**
  - chunk: ” “O, you couldn’t be so cruel, now! I was just dying to know whether you would appear in your pink tarletane,” said Adolph. “What is it? ” said Rosa, a bright, piquant little quadroon who came skipping down stairs at this moment. “Why, Mr. St. Clare’s so impudent! ” “On my honor,” said Adolph, “I’l…
- `B24:153` (segment 1/1, 39 tokens, stop): **Stub context: Allan the next day, “and it seemed like years while the flat was drifting down to the bridge and the water rising in it every moment. I prayed, Mrs. Allan,**
  - chunk: Gilbert glanced up and, much to his amazement, beheld a little white scornful face looking down upon him with big, frightened but also scornful gray eyes. “Anne Shirley! How on earth did you get there? ” he exclaimed. Without waiting for an answer he pulled close to the pile and extended his hand. T…
- `B18:273` (segment 1/1, 35 tokens, stop): **Stub context: Hence as Tess stirred the clods and sang her foolish little songs with scarce now a hope that Clare would ever hear them, she did not for a long time**
  - chunk: ” “But I like doing it—it is for my father. ” “Your engagement at the other place is ended? ” “Yes. ” “Where are you going to next? To join your dear husband? ” She could not bear the humiliating reminder. “O—I don’t know! ” she said bitterly. “I have no husband! ” “It is quite true—in the sense you…
- `B12:382` (segment 1/1, 41 tokens, stop): **Stub context: For eleven years, I had not seen Joe nor Biddy with my bodily eyes,—though they had both been often before my fancy in the East,—when, upon an evening in December,**
  - chunk: I had heard of her as leading a most unhappy life, and as being separated from her husband, who had used her with great cruelty, and who had become quite renowned as a compound of pride, avarice, brutality, and meanness. And I had heard of the death of her husband, from an accident consequent on his…
- `B61:194` (segment 1/1, 35 tokens, stop): **Stub context: A drowning man will catch even at a straw, they say, and he has not the sense at the moment to reflect that a fly could scarcely save itself on**
  - chunk: how it was found out that there were in his house counterfeit notes for two million roubles, a seal was put on the house, and two soldiers were set to keep guard at every door, and how Tchitchikov changed all the notes in a single night so that when the seal was taken off the next day, the notes wer…
- `B12:108` (segment 1/1, 39 tokens, stop): **Stub context: It began with the strange gentleman’s sitting down at the table, drawing the candle to him, and looking over some entries in his pocket-book. He then put up the pocket-book**
  - chunk: Miss Havisham was going to make my fortune on a grand scale. “Now, Mr. Pip,” pursued the lawyer, “I address the rest of what I have to say, to you. You are to understand, first, that it is the request of the person from whom I take my instructions that you always bear the name of Pip. You will have …

## 11. Estimates and timings

- Generation estimate 4.1 h for 7011 calls (per call 9.3e-06 s x visible document tokens + 0.15 s, plus 420 s engine start (B2's measured rate on the same model and runtime)); measured 26s.
- BM25: estimate 3 min on CPU (C2's CPU rate x 3 (published streams, empty-context pass, CR pass)); measured 60s.

