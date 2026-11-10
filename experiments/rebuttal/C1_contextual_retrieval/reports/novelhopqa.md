# C1 Contextual Retrieval: NovelHopQA

- Written 2026-11-10 12:45:55; commit 0df9d38e4fd5; documents 16, chunks 7011, queries 908 (expected 908 on the full dataset).
- Status: generation done; jobs BM25 DONE, Contriever DONE, BGE-M3 DONE, Jina DONE, Qwen DONE.

## 1. Correctness check (empty context reproduces Vanilla)

CR with every context empty indexes exactly the chunk text, so it must give Vanilla (SAADI's text channel alone) and the paper's saved results. BM25: identical rankings; dense: identical top 10 for at least 99% of queries and HR within 0.1 points (C2's Jina drift rule applies). The pool itself is checked by reproducing SAADI's saved rankings with the published weights.

| Retriever | Gate | SAADI top-10 = saved | Empty = Vanilla top-10 | HR@5 / HR@10 empty vs Vanilla | = saved text-only file | Table 1 Vanilla |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 | pass | 908/908 | 908/908 (top-150 908) | 17.51 / 20.70 vs 17.51 / 20.70 | 908/908 | 16.9 / 20.0 |
| Contriever | pass | 908/908 | 908/908 (top-150 908) | 12.44 / 17.84 vs 12.44 / 17.84 | no file | 11.8 / 11.8 |
| BGE-M3 | pass | 908/908 | 908/908 (top-150 908) | 17.29 / 21.92 vs 17.29 / 21.92 | no file | 14.1 / 20.7 |
| Jina | pass | 624/908 | 908/908 (top-150 908) | 15.97 / 20.15 vs 15.97 / 20.15 | no file | 14.8 / 19.1 |
| Qwen | pass | 908/908 | 908/908 (top-150 908) | 16.74 / 20.48 vs 16.74 / 20.48 | no file | 14.6 / 19.8 |
- BM25 note: empty-context HR@5 17.51 vs Table 1 Vanilla 16.9 (tolerance 0.05) - not gated: on this dataset Table 1 Vanilla is not the text channel of the saved chunks
- BM25 note: empty-context HR@10 20.70 vs Table 1 Vanilla 20.0 (tolerance 0.05) - not gated: on this dataset Table 1 Vanilla is not the text channel of the saved chunks
- BM25: Vanilla reference for the dense check = C2 saved streams (independent run); memory and summary channels from {'c2_streams': 16, 'published_path': 0} documents.
- Contriever note: empty-context HR@5 12.44 vs Table 1 Vanilla 11.8 (tolerance 0.15) - not gated: on this dataset Table 1 Vanilla is not the text channel of the saved chunks
- Contriever note: empty-context HR@10 17.84 vs Table 1 Vanilla 11.8 (tolerance 0.15) - not gated: on this dataset Table 1 Vanilla is not the text channel of the saved chunks
- Contriever: Vanilla reference for the dense check = C2 saved streams (independent run); memory and summary channels from {'c2_streams': 16, 'published_path': 0} documents.
- BGE-M3 note: empty-context HR@5 17.29 vs Table 1 Vanilla 14.1 (tolerance 0.15) - not gated: on this dataset Table 1 Vanilla is not the text channel of the saved chunks
- BGE-M3 note: empty-context HR@10 21.92 vs Table 1 Vanilla 20.7 (tolerance 0.15) - not gated: on this dataset Table 1 Vanilla is not the text channel of the saved chunks
- BGE-M3: Vanilla reference for the dense check = C2 saved streams (independent run); memory and summary channels from {'c2_streams': 16, 'published_path': 0} documents.
- Jina note: empty-context HR@5 15.97 vs Table 1 Vanilla 14.8 (tolerance 0.50) - not gated: on this dataset Table 1 Vanilla is not the text channel of the saved chunks
- Jina note: empty-context HR@10 20.15 vs Table 1 Vanilla 19.1 (tolerance 0.50) - not gated: on this dataset Table 1 Vanilla is not the text channel of the saved chunks
- Jina: Vanilla reference for the dense check = C2 saved streams (independent run); memory and summary channels from {'c2_streams': 16, 'published_path': 0} documents.
- Qwen note: empty-context HR@5 16.74 vs Table 1 Vanilla 14.6 (tolerance 0.50) - not gated: on this dataset Table 1 Vanilla is not the text channel of the saved chunks
- Qwen note: empty-context HR@10 20.48 vs Table 1 Vanilla 19.8 (tolerance 0.50) - not gated: on this dataset Table 1 Vanilla is not the text channel of the saved chunks
- Qwen: Vanilla reference for the dense check = C2 saved streams (independent run); memory and summary channels from {'c2_streams': 16, 'published_path': 0} documents.
- On NovelHopQA, Table 1's Vanilla is not the text channel of the saved chunks (C2's server run: they differ by up to 13 points); C1's Vanilla is the text channel, and the gate uses the saved per-query text-only file where it exists.

## 2. Token census (Qwen3 tokenizer)

- Window 262144 tokens (model config, rope_scaling None); prompt template 70 tokens; output budget 1024.
- Document tokens: min 134K, median 234K, max 763K; chunk tokens median 660, max 1033.
- Documents over the window: 7 of 16 (4250 chunks); segments per document: {'1': 9, '2': 6, '3': 1}.
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

- 7011 contexts (0 missing), generator ['vllm']; empty 0, failed 0, cut at 1024 tokens 0.
- Context length (tokens): min 31, p25 112, median 137, mean 138.0, p75 162, p95 198, max 312.
- Cached tokens: ['vllm num_cached_tokens'].

## 6. Cost next to SAADI

| Quantity | CR | SAADI (memory + hierarchy calls) |
| --- | --- | --- |
| LLM calls (total) | 7011 | 9634 |
| LLM calls per chunk | 1.00 | 1.25 |
| LLM calls per document | 438.2 | 535.2 |
| Input tokens per chunk, nominal (full prompt) | 204K | 4797 (no prefix cache) |
| Input tokens per chunk, computed (after prefix-cache hits) | 1427 | same as nominal |
| Input tokens total, nominal / computed | 1.43B / 10.0M | 37.1M |
| Generated tokens per chunk, mean / median | 138.0 / 137.0 | 1199.0 |
| Generated tokens total | 967K | 9.3M |
| Contexts cut at 1024 tokens | 0 (0.0%) | — |
| Generation wall clock / GPU-hours | 41 min / 2.73 | not in the paper's tables |
| Indexing + scoring, BM25 (wall / GPU-hours) | 2 min / 0.00 | — |
| Indexing + scoring, Contriever (wall / GPU-hours) | 7 min / 0.11 | — |
| Indexing + scoring, BGE-M3 (wall / GPU-hours) | 16 min / 0.27 | — |
| Indexing + scoring, Jina (wall / GPU-hours) | 9 min / 0.15 | — |
| Indexing + scoring, Qwen (wall / GPU-hours) | 1.7 h / 3.32 | — |
- Ratios per chunk. SAADI's builds had no prefix caching, so its input is compared with both CR columns: CR nominal / SAADI =42.63x, CR computed / SAADI = 0.30x; calls 0.80x; generated tokens 0.12x. SAADI chunks in the cost table: 7736; C1 chunks: 7011.

## 7. Retrieval: per retriever

HR@5 / HR@10 (%). *Complementarity check, not a proposed method.

| Retriever | Vanilla (text channel) | Table 1 Vanilla | CR | SAADI (published) | SAADI on CR text* | LQCA (Table 1) | SAADI held-out (C2) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BM25 | 17.5 / 20.7 | 16.9 / 20.0 | 18.4 / 21.7 | 18.1 / 22.0 | 18.4 / 23.2 | 1.4 / 1.9 | 17.6 / 21.1 |
| Contriever | 12.4 / 17.8 | 11.8 / 11.8 | 14.1 / 18.5 | 13.5 / 18.2 | 14.6 / 18.7 | 0.9 / 1.7 | 13.2 / 18.1 |
| BGE-M3 | 17.3 / 21.9 | 14.1 / 20.7 | 15.5 / 21.4 | 16.5 / 22.1 | 15.1 / 19.8 | 1.4 / 2.2 | 17.1 / 22.7 |
| Jina | 16.0 / 20.2 | 14.8 / 19.1 | 15.0 / 19.4 | 14.8 / 20.9 | 14.0 / 19.9 | 2.1 / 2.2 | — / — |
| Qwen | 16.7 / 20.5 | 14.6 / 19.8 | 11.8 / 16.7 | 15.2 / 20.6 | 12.7 / 18.2 | 2.2 / 2.8 | — / — |

Paired differences (points) with 95% bootstrap intervals (10,000 resamples over queries, seed 42):

| Retriever | CR − Vanilla @5 | @10 | SAADI − CR @5 | @10 | SAADI-on-CR − CR @5 | @10 |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 | +0.9 [-0.7, +2.4] | +1.0 [-0.4, +2.4] | -0.3 [-1.8, +1.2] | +0.3 [-1.0, +1.8] | +0.0 [-0.9, +0.9] | +1.5 [+0.4, +2.8] |
| Contriever | +1.7 [-0.2, +3.5] | +0.7 [-1.2, +2.6] | -0.6 [-2.3, +1.2] | -0.3 [-2.1, +1.4] | +0.6 [-0.8, +1.9] | +0.2 [-1.0, +1.4] |
| BGE-M3 | -1.8 [-3.6, +0.1] | -0.6 [-2.6, +1.4] | +1.0 [-0.9, +2.9] | +0.8 [-1.2, +2.8] | -0.4 [-1.9, +0.9] | -1.5 [-3.0, +0.0] |
| Jina | -1.0 [-3.0, +1.0] | -0.8 [-2.9, +1.3] | -0.2 [-2.2, +1.8] | +1.5 [-0.4, +3.5] | -1.0 [-2.4, +0.4] | +0.6 [-0.7, +1.9] |
| Qwen | -5.0 [-7.2, -2.8] | -3.7 [-6.1, -1.4] | +3.4 [+1.3, +5.5] | +3.9 [+1.7, +6.2] | +0.9 [-0.4, +2.2] | +1.4 [+0.0, +3.0] |

## 8. Retrieval: hybrid (dense + BM25, cookbook fusion)

| Hybrid | Vanilla @5 / @10 | CR @5 / @10 | SAADI @5 / @10 | CR − Vanilla @10 | SAADI − CR @10 | SAADI − CR @5 |
| --- | --- | --- | --- | --- | --- | --- |
| Contriever + BM25 | 16.6 / 22.1 | 18.1 / 22.9 | 16.9 / 22.6 | +0.8 [-1.1, +2.6] | -0.3 [-2.1, +1.4] | -1.2 [-2.9, +0.4] |
| BGE-M3 + BM25 | 19.9 / 25.2 | 18.5 / 24.4 | 19.6 / 24.9 | -0.8 [-2.5, +1.0] | +0.4 [-1.3, +2.3] | +1.1 [-0.7, +2.9] |
| Jina + BM25 | 19.4 / 24.3 | 19.1 / 24.4 | 19.1 / 25.2 | +0.1 [-1.9, +2.1] | +0.8 [-1.2, +2.8] | +0.0 [-1.8, +1.8] |
| Qwen + BM25 | 20.3 / 23.9 | 16.9 / 23.1 | 18.6 / 25.3 | -0.8 [-2.8, +1.2] | +2.2 [+0.2, +4.2] | +1.8 [-0.2, +3.7] |
- Top 150 of each list, score 0.8 / rank_dense + 0.2 / rank_bm25 (1-based), the same merge for Vanilla, CR and SAADI (SAADI = SAADI with d + SAADI with BM25, published weights). No reranker.

## 9. Dense truncation of the contextualized chunks

| Retriever | Max input | CR chunks over max | CR tokens cut (end of chunk) | Plain chunks over max | Extra chunk tokens cut per chunk (CR vs plain) |
| --- | --- | --- | --- | --- | --- |
| Contriever | 512 | 7002/7011 | 1.9M | 6988/7011 | 139 |
| BGE-M3 | 8192 | 0/7011 | 0 | 0/7011 | 0 |
| Jina | 512 | 7004/7011 | 2.6M | 6998/7011 | 164 |
| Qwen | 512 | 7003/7011 | 2.0M | 6992/7011 | 137 |
- Read the CR rows of these retrievers with care: Contriever (100% of contextualized chunks over 512 tokens, 139 more chunk tokens cut per chunk than plain); Jina (100% of contextualized chunks over 512 tokens, 164 more chunk tokens cut per chunk than plain); Qwen (100% of contextualized chunks over 512 tokens, 137 more chunk tokens cut per chunk than plain). The context comes first, so it pushes the end of the chunk out of the retriever's input; Vanilla loses less. Retrievers without truncation here (BGE-M3, 8192 tokens; BM25) are the comparison without this effect.

## 10. Sanity sample (10 random contexts, seed 42)

- `B44:73` (segment 1/2, 108 tokens, stop): **Jane Eyre is preparing to leave Lowood School for her new position as governess at Thornfield Hall, where she will care for Mr. Rochester's ward, Adèle. After receiving approval from Mrs. Fairfax and finalizing her arrangements, she is about to depart when Bessie, her former nurse from Gateshead, unexpectedly visits her at the school. This emotional reunion occurs just before Jane's journey to Thornfield, marking a poignant moment of connection with her past as she embarks on a new chapter in her life.**
  - chunk: Fairfax, and got that lady"s reply, stating that she was satisfied, and fixing that day fortnight as the period for my assuming the post of governess in her house. I now busied myself in preparations: the fortnight passed rapidly. I had not a very large wardrobe, though it was adequate to my wants;…
- `B12:174` (segment 1/1, 91 tokens, stop): **Joe visits Pip in London to deliver a message from Miss Havisham: Estella has returned and wishes to see him. Pip is initially impatient and uncomfortable with Joe's awkward, formal demeanor and outdated clothing, which highlights the growing social and emotional distance between them. This moment occurs during Pip's visit to Joe's home in the country, shortly after Pip has learned the truth about his benefactor and is grappling with the consequences of his changed status.**
  - chunk: Indeed, it demanded from him a constant attention, and a quickness of eye and hand, very like that exacted by wicket-keeping. He made extraordinary play with it, and showed the greatest skill; now, rushing at it and catching it neatly as it dropped; now, merely stopping it midway, beating it up, and…
- `B04:282` (segment 1/1, 129 tokens, stop): **This chunk occurs during the climactic pursuit of Bill Sikes, the murderer of Nancy, after he has fled London following her death. After a night of tormenting hallucinations and fear, Sikes is driven to seek refuge in a remote field, where he encounters a massive fire. The sudden noise and chaos of the fire provide him with a temporary escape from his psychological torment, allowing him to lose himself in the frenzied efforts to extinguish the blaze. This moment of temporary distraction and physical exertion marks a turning point in his descent into madness and isolation, as he momentarily for**
  - chunk: The figure was behind him. He re-entered the shed, and shrunk down once more. The eyes were there, before he had laid himself along. And here he remained in such terror as none but he can know, trembling in every limb, and the cold sweat starting from every pore, when suddenly there arose upon the n…
- `B61:230` (segment 1/1, 193 tokens, stop): **This chunk occurs at the end of Part I of *Dead Souls*, during Tchitchikov's departure from the provincial town of N. The passage is a lyrical, almost ecstatic meditation on the Russian landscape and the symbolic power of the troika (a three-horse sleigh), which becomes a metaphor for the restless, unstoppable forward motion of Russia itself. The narrator reflects on the nation's vastness, its unique spirit, and its mysterious, almost divine momentum, contrasting the fleeting, chaotic journey with the enduring, indifferent landscape. This moment of poetic transcendence is immediately followed **
  - chunk: milestones fly by, merchants on the front seats of their tilt-carts fly to meet one, the forest flies by on both sides with dark rows of firs and pines, with the ring of the axe and caw of the crows; the whole road flies into the unknown retreating distance; and there is something terrible in this r…
- `B26:184` (segment 1/2, 165 tokens, stop): **This chunk occurs in Volume I, Chapter XVIII, "Miss Ophelia’s Experiences and Opinions," during a scene in the kitchen of Augustine St. Clare’s New Orleans mansion. It captures a moment of idle, flirtatious banter among the household servants—Adolph, Jane, and Rosa—highlighting their superficiality and racial self-awareness, contrasted with the deeper moral and spiritual concerns of Tom, the enslaved man, who later compassionately engages with an old, suffering rusk-woman named Prue. The scene underscores the novel’s critique of the dehumanizing effects of slavery, juxtaposing the servants’ tr**
  - chunk: ” “O, you couldn’t be so cruel, now! I was just dying to know whether you would appear in your pink tarletane,” said Adolph. “What is it? ” said Rosa, a bright, piquant little quadroon who came skipping down stairs at this moment. “Why, Mr. St. Clare’s so impudent! ” “On my honor,” said Adolph, “I’l…
- `B24:153` (segment 1/1, 106 tokens, stop): **In the chapter "An Unfortunate Lily Maid," Anne Shirley and her friends stage a dramatic reenactment of Tennyson's "Elaine" on the pond, with Anne playing the role of the drowned lily maid. While floating in a leaking flat, she is forced to climb onto a bridge pile for safety. Gilbert Blythe rescues her, offering friendship and apologizing for past teasing, but Anne, still holding onto her old grudge, refuses to reconcile, despite a fleeting moment of hesitation.**
  - chunk: Gilbert glanced up and, much to his amazement, beheld a little white scornful face looking down upon him with big, frightened but also scornful gray eyes. “Anne Shirley! How on earth did you get there? ” he exclaimed. Without waiting for an answer he pulled close to the pile and extended his hand. T…
- `B18:273` (segment 1/1, 131 tokens, stop): **Tess, having returned to Marlott after her time at Flintcomb-Ash, is working in the allotment garden when Alec d’Urberville, disguised as a labourer, appears and attempts to offer her financial help. She refuses his aid, emphasizing her independence and her lack of a husband. He persists, claiming he wants to help her and her family, particularly her younger siblings, but she remains resolute. After a tense exchange, he leaves in anger, having revealed his identity. Tess then returns home to learn that her father has died, a devastating blow that underscores her isolation and the fragility of **
  - chunk: ” “But I like doing it—it is for my father. ” “Your engagement at the other place is ended? ” “Yes. ” “Where are you going to next? To join your dear husband? ” She could not bear the humiliating reminder. “O—I don’t know! ” she said bitterly. “I have no husband! ” “It is quite true—in the sense you…
- `B12:382` (segment 1/1, 125 tokens, stop): **After years of separation and personal transformation, Pip returns to his childhood home in Kent, now a ruined and overgrown site, following a visit with his dear friends Joe and Biddy. Having learned of Estella's unhappy marriage and her husband's death, Pip is drawn to revisit the place where their fated relationship began. As he walks through the desolate garden, he unexpectedly encounters Estella, who has also come to say farewell to the remnants of her past. Their emotional reunion, marked by mutual recognition and a quiet reconciliation, takes place under the rising moon, symbolizing a p**
  - chunk: I had heard of her as leading a most unhappy life, and as being separated from her husband, who had used her with great cruelty, and who had become quite renowned as a compound of pride, avarice, brutality, and meanness. And I had heard of the death of her husband, from an accident consequent on his…
- `B61:194` (segment 1/1, 234 tokens, stop): **This chunk occurs in Chapter 10 of Part II of *Dead Souls*, during a pivotal scene in which the town officials, alarmed by rumors about Tchitchikov’s mysterious activities, interrogate Nozdryov—Tchitchikov’s flamboyant and unreliable acquaintance—to uncover the truth. Nozdryov, known for his wild lies and exaggerations, fabricates a fantastical tale claiming Tchitchikov is a forger of counterfeit notes, a spy, and a conspirator planning to elope with the governor’s daughter. He details a fictional wedding in the village of Truhmatchevka, complete with bribed priests, stolen carriages, and stag**
  - chunk: how it was found out that there were in his house counterfeit notes for two million roubles, a seal was put on the house, and two soldiers were set to keep guard at every door, and how Tchitchikov changed all the notes in a single night so that when the seal was taken off the next day, the notes wer…
- `B12:108` (segment 1/1, 136 tokens, stop): **Pip has just learned from Mr. Jaggers that he is to receive a large fortune from an anonymous benefactor, but must keep the identity of the donor a secret. This revelation occurs after Pip's guardian, Mr. Jaggers, has informed him that he has "great expectations" and will be removed from his current life to be raised as a gentleman. The chunk captures the pivotal moment when Jaggers outlines the conditions of Pip's inheritance, including the requirement to keep his benefactor's identity secret and to never inquire about it. This sets up the central mystery of the novel and establishes the foun**
  - chunk: Miss Havisham was going to make my fortune on a grand scale. “Now, Mr. Pip,” pursued the lawyer, “I address the rest of what I have to say, to you. You are to understand, first, that it is the request of the person from whom I take my instructions that you always bear the name of Pip. You will have …

## 11. Estimates and timings

- Generation estimate 4.1 h for 7011 calls (per call 9.3e-06 s x visible document tokens + 0.15 s, plus 420 s engine start (B2's measured rate on the same model and runtime)); measured 41 min.
- BM25: estimate 3 min on CPU (C2's CPU rate x 3 (published streams, empty-context pass, CR pass)); measured 2 min.
- Contriever: estimate 3 min on 1 GPU(s) (C2 streams reused: CR chunks + plain chunks (empty-context check) encoded, 2 query encodings per query); measured 7 min.
- BGE-M3: estimate 21 min on 1 GPU(s) (C2 streams reused: CR chunks + plain chunks (empty-context check) encoded, 2 query encodings per query); measured 16 min.
- Jina: estimate 15 min on 1 GPU(s) (C2 streams reused: CR chunks + plain chunks (empty-context check) encoded, 2 query encodings per query); measured 9 min.
- Qwen: estimate 3.1 h on 2 GPU(s) (C2 streams reused: CR chunks + plain chunks (empty-context check) encoded, 2 query encodings per query); measured 74s.

