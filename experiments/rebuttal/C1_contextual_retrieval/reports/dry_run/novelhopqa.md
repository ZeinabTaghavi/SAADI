# C1 Contextual Retrieval: NovelHopQA (dry run: stub contexts, BM25 only, placeholder numbers)

- Written 2026-11-09 15:02:22; commit e2dbaa178fa0; documents 2, chunks 1114, queries 145 (expected 908 on the full dataset).
- Status: generation done; jobs BM25 DONE.

## 1. Correctness check (empty context reproduces Vanilla)

CR with every context empty indexes exactly the chunk text, so it must give Vanilla (SAADI's text channel alone) and the paper's saved results. BM25: identical rankings; dense: identical top 10 for at least 99% of queries and HR within 0.1 points (C2's Jina drift rule applies). The pool itself is checked by reproducing SAADI's saved rankings with the published weights.

| Retriever | Gate | SAADI top-10 = saved | Empty = Vanilla top-10 | HR@5 / HR@10 empty vs Vanilla | = saved text-only file | Table 1 Vanilla |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 | pass | 145/145 | 145/145 (top-150 145) | 21.38 / 25.52 vs 21.38 / 25.52 | 145/145 | 16.9 / 20.0 |
- BM25: Vanilla reference for the dense check = published path in this process (shared encoding cache); memory and summary channels from {'c2_streams': 0, 'published_path': 2} documents.
- On NovelHopQA, Table 1's Vanilla is not the text channel of the saved chunks (C2's server run: they differ by up to 13 points); C1's Vanilla is the text channel, and the gate uses the saved per-query text-only file where it exists.

## 2. Token census (Qwen3 tokenizer)

- Window 262144 tokens (model config, rope_scaling None); prompt template 70 tokens; output budget 1024.
- Document tokens: min 242K, median 352K, max 463K; chunk tokens median 632, max 760.
- Documents over the window: 1 of 2 (729 chunks); segments per document: {'1': 1, '2': 1}.
- Split vs joint tokenization of the prompt identical on the checked samples: [True, True].

## 3. GPU memory check (embedding models)

- Skipped (CPU dry run).

## 4. Long documents (segment rule)

| Document | Tokens | Chunks | Segments k |
| --- | --- | --- | --- |
| B62 | 463K | 729 | 2 |
- These chunks saw only their own segment of the document, not the whole document: a limit of CR on book-length text.

## 5. Contexts

- 1114 contexts (0 missing), generator ['stub']; empty 0, failed 0, cut at 1024 tokens 0.
- Context length (tokens): min 34, p25 38, median 40, mean 41.2, p75 44, p95 49, max 60.
- Cached tokens: ['simulated (stub)'].

## 6. Cost next to SAADI

| Quantity | CR | SAADI (memory + hierarchy calls) |
| --- | --- | --- |
| LLM calls (total) | 1114 | 9634 |
| LLM calls per chunk | 1.00 | 1.25 |
| LLM calls per document | 557.0 | 535.2 |
| Input tokens per chunk, nominal (full prompt) | 236K | 4797 (no prefix cache) |
| Input tokens per chunk, computed (after prefix-cache hits) | 1548 | same as nominal |
| Input tokens total, nominal / computed | 262.5M / 1.7M | 37.1M |
| Generated tokens per chunk, mean / median | 41.2 / 40.0 | 1199.0 |
| Generated tokens total | 46K | 9.3M |
| Contexts cut at 1024 tokens | 0 (0.0%) | — |
| Generation wall clock / GPU-hours | 11s / 0.00 | not in the paper's tables |
| Indexing + scoring, BM25 (wall / GPU-hours) | 12s / 0.00 | — |
- Ratios per chunk. SAADI's builds had no prefix caching, so its input is compared with both CR columns: CR nominal / SAADI =49.13x, CR computed / SAADI = 0.32x; calls 0.80x; generated tokens 0.03x. SAADI chunks in the cost table: 7736; C1 chunks: 1114.

## 7. Retrieval: per retriever

HR@5 / HR@10 (%). *Complementarity check, not a proposed method.

| Retriever | Vanilla (text channel) | Table 1 Vanilla | CR | SAADI (published) | SAADI on CR text* | LQCA (Table 1) |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 | 21.4 / 25.5 | 16.9 / 20.0 | 20.7 / 25.5 | 22.8 / 25.5 | 22.8 / 24.8 | 1.4 / 1.9 |
- SAADI held-out column left out: results/rebuttal/C2_heldout_fusion/REPORT.md does not exist yet.

Paired differences (points) with 95% bootstrap intervals (10,000 resamples over queries, seed 42):

| Retriever | CR − Vanilla @5 | @10 | SAADI − CR @5 | @10 | SAADI-on-CR − CR @5 | @10 |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 | -0.7 [-2.1, +0.0] | +0.0 [+0.0, +0.0] | +2.1 [-0.7, +5.5] | +0.0 [-2.1, +2.1] | +2.1 [+0.0, +4.8] | -0.7 [-2.1, +0.0] |

## 8. Retrieval: hybrid (dense + BM25, cookbook fusion)

| Hybrid | Vanilla @5 / @10 | CR @5 / @10 | SAADI @5 / @10 | CR − Vanilla @10 | SAADI − CR @10 | SAADI − CR @5 |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 + BM25 (dry-run stand-in for a dense retriever) | 21.4 / 25.5 | 20.7 / 25.5 | 22.8 / 25.5 | +0.0 [+0.0, +0.0] | +0.0 [-2.1, +2.1] | +2.1 [-0.7, +5.5] |
- Top 150 of each list, score 0.8 / rank_dense + 0.2 / rank_bm25 (1-based), the same merge for Vanilla, CR and SAADI (SAADI = SAADI with d + SAADI with BM25, published weights). No reranker.

## 9. Dense truncation of the contextualized chunks

- No dense retriever in this run.

## 10. Sanity sample (10 random contexts, seed 42)

- `B12:303` (segment 1/1, 44 tokens, stop): **Stub context: The boatswain, unmanned for the first time, respectfully dried his eyes on the Jack, and then cheering up, and addressing Mr. Wopsle as Your Honour, solicited permission to take him**
  - chunk: for these mysterious words gave me a chill. “Oh! He can’t be in sight,” said Mr. Wopsle. “He went out before I went off. I saw him go. ” Having the reason that I had for being suspicious, I even suspected this poor actor. I mistrusted a design to entrap me into some admission. Therefore I glanced at…
- `B12:144` (segment 1/1, 37 tokens, stop): **Stub context: But again there came upon me, for my relief, that odd impression that Herbert Pocket would never be very successful or rich. “I shall not rest satisfied with merely employing**
  - chunk: It was evident that he had nothing around him but the simplest necessaries, for everything that I remarked upon turned out to have been sent in on my account from the coffee-house or somewhere else. Yet, having already made his fortune in his own mind, he was so unassuming with it that I felt quite …
- `B62:259` (segment 1/2, 37 tokens, stop): **Stub context: there would be no indefiniteness or falsehood about it. The pain she had caused herself and her husband in uttering those words would be rewarded now by everything being made**
  - chunk: Betsy reminded her that Liza Merkalova and Baroness Shtoltz were coming to play croquet with her that morning with their adorers, Kaluzhsky and old Stremov. “Come, if only as a study in morals. I shall expect you,” she finished. Anna read the note and heaved a deep sigh. “Nothing, I need nothing,” s…
- `B62:202` (segment 1/2, 37 tokens, stop): **Stub context: Her daughter made her no reply, only in her heart she thought that one could not talk about exaggeration where Christianity was concerned. What exaggeration could there be in the**
  - chunk: She recalled the efforts she had made at first to overcome the repugnance she felt for him, as for all consumptive people, and the pains it had cost her to think of things to say to him. She recalled the timid, softened look with which he gazed at her, and the strange feeling of compassion and awkwa…
- `B62:163` (segment 1/2, 44 tokens, stop): **Stub context: Only when he stood by her head, she was suddenly quieter, while the muscles quivered under her soft, delicate coat. Vronsky patted her strong neck, straightened over her sharp withers**
  - chunk: And why do they worry me so? Just because they see that this is something they can’t understand. If it were a common, vulgar, worldly intrigue, they would have left me alone. They feel that this is something different, that this is not a mere pastime, that this woman is dearer to me than life. And t…
- `B12:355` (segment 1/1, 36 tokens, stop): **Stub context: It was a good thing that he had touched this point, for it put into my mind what I might not otherwise have thought of until too late,—that he need**
  - chunk: After three days’ delay, during which the crown prosecution stood over for the production of the witness from the prison-ship, the witness came, and completed the easy case. He was committed to take his trial at the next Sessions, which would come on in a month. It was at this dark time of my life t…
- `B12:287` (segment 1/1, 40 tokens, stop): **Stub context: ” “O Estella! ” I answered, as my bitter tears fell fast on her hand, do what I would to restrain them; “even if I remained in England and could**
  - chunk: Pursuing the narrow intricacies of the streets which at that time tended westward near the Middlesex shore of the river, my readiest access to the Temple was close by the river-side, through Whitefriars. I was not expected till to-morrow; but I had my keys, and, if Herbert were gone to bed, could ge…
- `B12:259` (segment 1/1, 41 tokens, stop): **Stub context: At last, the old woman and the niece came in,—the latter with a head not easily distinguishable from her dusty broom,—and testified surprise at sight of me and the fire.**
  - chunk: Some of his teeth had failed him since I saw him eat on the marshes, and as he turned his food in his mouth, and turned his head sideways to bring his strongest fangs to bear upon it, he looked terribly like a hungry old dog. If I had begun with any appetite, he would have taken it away, and I shoul…
- `B62:53` (segment 1/2, 58 tokens, stop): **Stub context: .” “Yes, that’s true, it is so,” said Stepan Arkadyevitch, laughing good-humoredly. “Will the train soon be in? ” Vronsky asked a railway official. “The train’s signaled,” answered the man.**
  - chunk: Vronsky, standing beside Oblonsky, watched the carriages and the passengers, totally oblivious of his mother. What he had just heard about Kitty excited and delighted him. Unconsciously he arched his chest, and his eyes flashed. He felt himself a conqueror. “Countess Vronskaya is in that compartment…
- `B12:157` (segment 1/1, 43 tokens, stop): **Stub context: In a back room, a high-shouldered man with a face-ache tied up in dirty flannel, who was dressed in old black clothes that bore the appearance of having been waxed,**
  - chunk: ” While he was putting up the other cast and coming down from the chair, the thought crossed my mind that all his personal jewelry was derived from like sources. As he had shown no diffidence on the subject, I ventured on the liberty of asking him the question, when he stood before me, dusting his h…

## 11. Estimates and timings

- Generation estimate 50 min for 1114 calls (per call 9.3e-06 s x visible document tokens + 0.15 s, plus 420 s engine start (B2's measured rate on the same model and runtime)); measured 11s.
- BM25: estimate 38s on CPU (C2's CPU rate x 3 (published streams, empty-context pass, CR pass)); measured 12s.

