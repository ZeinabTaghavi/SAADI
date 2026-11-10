# C1 Contextual Retrieval: QASPER-64K

- Written 2026-11-10 14:59:58; commit 0df9d38e4fd5; documents 23, chunks 17159, queries 1333 (expected 1333 on the full dataset).
- Status: generation done; jobs BM25 DONE, Contriever DONE, BGE-M3 DONE, Jina DONE, Qwen DONE.

## 1. Correctness check (empty context reproduces Vanilla)

CR with every context empty indexes exactly the chunk text, so it must give Vanilla (SAADI's text channel alone) and the paper's saved results. BM25: identical rankings; dense: identical top 10 for at least 99% of queries and HR within 0.1 points (C2's Jina drift rule applies). The pool itself is checked by reproducing SAADI's saved rankings with the published weights.

| Retriever | Gate | SAADI top-10 = saved | Empty = Vanilla top-10 | HR@5 / HR@10 empty vs Vanilla | = saved text-only file | Table 1 Vanilla |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 | pass | 1333/1333 | 1333/1333 (top-150 1333) | 14.40 / 22.06 vs 14.40 / 22.06 | no file | 14.4 / 22.1 |
| Contriever | pass | 1333/1333 | 1333/1333 (top-150 1333) | 15.15 / 23.33 vs 15.15 / 23.33 | no file | 15.2 / 23.3 |
| BGE-M3 | pass | 1333/1333 | 1333/1333 (top-150 1333) | 21.01 / 29.33 vs 21.01 / 29.33 | no file | 21.0 / 29.3 |
| Jina | pass | 927/1333 | 1333/1333 (top-150 1333) | 15.98 / 24.98 vs 15.98 / 24.98 | no file | 15.7 / 24.8 |
| Qwen | pass | 1333/1333 | 1333/1333 (top-150 1333) | 22.58 / 32.56 vs 22.58 / 32.56 | no file | 22.6 / 32.6 |
- BM25: Vanilla reference for the dense check = C2 saved streams (independent run); memory and summary channels from {'c2_streams': 23, 'published_path': 0} documents.
- Contriever: Vanilla reference for the dense check = C2 saved streams (independent run); memory and summary channels from {'c2_streams': 23, 'published_path': 0} documents.
- BGE-M3: Vanilla reference for the dense check = C2 saved streams (independent run); memory and summary channels from {'c2_streams': 23, 'published_path': 0} documents.
- Jina: Vanilla reference for the dense check = C2 saved streams (independent run); memory and summary channels from {'c2_streams': 23, 'published_path': 0} documents.
- Qwen: Vanilla reference for the dense check = C2 saved streams (independent run); memory and summary channels from {'c2_streams': 23, 'published_path': 0} documents.

## 2. Token census (Qwen3 tokenizer)

- Window 262144 tokens (model config, rope_scaling None); prompt template 70 tokens; output budget 1024.
- Document tokens: min 76K, median 92K, max 99K; chunk tokens median 121, max 469.
- Documents over the window: 0 of 23 (0 chunks); segments per document: {'1': 23}.
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

- 17159 contexts (0 missing), generator ['vllm']; empty 0, failed 0, cut at 1024 tokens 0.
- Context length (tokens): min 18, p25 81, median 98, mean 98.8, p75 115, p95 143, max 720.
- Cached tokens: ['vllm num_cached_tokens'].

## 6. Cost next to SAADI

| Quantity | CR | SAADI (memory + hierarchy calls) |
| --- | --- | --- |
| LLM calls (total) | 17159 | 21404 |
| LLM calls per chunk | 1.00 | 1.25 |
| LLM calls per document | 746.0 | 930.6 |
| Input tokens per chunk, nominal (full prompt) | 92K | 2530 (no prefix cache) |
| Input tokens per chunk, computed (after prefix-cache hits) | 351 | same as nominal |
| Input tokens total, nominal / computed | 1.59B / 6.0M | 43.4M |
| Generated tokens per chunk, mean / median | 98.8 / 98.0 | 633.4 |
| Generated tokens total | 1.7M | 10.9M |
| Contexts cut at 1024 tokens | 0 (0.0%) | — |
| Generation wall clock / GPU-hours | 2.9 h / 11.67 | not in the paper's tables |
| Indexing + scoring, BM25 (wall / GPU-hours) | 4 min / 0.00 | — |
| Indexing + scoring, Contriever (wall / GPU-hours) | 7 min / 0.12 | — |
| Indexing + scoring, BGE-M3 (wall / GPU-hours) | 12 min / 0.19 | — |
| Indexing + scoring, Jina (wall / GPU-hours) | 7 min / 0.11 | — |
| Indexing + scoring, Qwen (wall / GPU-hours) | 2.0 h / 4.08 | — |
- Ratios per chunk. SAADI's builds had no prefix caching, so its input is compared with both CR columns: CR nominal / SAADI =36.55x, CR computed / SAADI = 0.14x; calls 0.80x; generated tokens 0.16x. SAADI chunks in the cost table: 17159; C1 chunks: 17159.

## 7. Retrieval: per retriever

HR@5 / HR@10 (%). *Complementarity check, not a proposed method.

| Retriever | Vanilla (text channel) | Table 1 Vanilla | CR | SAADI (published) | SAADI on CR text* | LQCA (Table 1) | SAADI held-out (C2) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BM25 | 14.4 / 22.1 | 14.4 / 22.1 | 17.6 / 26.9 | 17.0 / 26.1 | 18.7 / 29.4 | 36.9 / 47.0 | 17.3 / 27.3 |
| Contriever | 15.2 / 23.3 | 15.2 / 23.3 | 16.7 / 26.0 | 16.2 / 25.8 | 17.6 / 27.6 | 34.4 / 46.0 | 15.8 / 25.2 |
| BGE-M3 | 21.0 / 29.3 | 21.0 / 29.3 | 23.9 / 33.9 | 20.9 / 30.9 | 22.6 / 33.4 | 32.9 / 45.8 | 22.1 / 31.9 |
| Jina | 16.0 / 25.0 | 15.7 / 24.8 | 17.9 / 28.1 | 18.8 / 27.1 | 18.1 / 27.8 | 31.7 / 41.8 | — / — |
| Qwen | 22.6 / 32.6 | 22.6 / 32.6 | 15.8 / 25.5 | 23.3 / 34.1 | 20.9 / 31.4 | 39.7 / 53.6 | — / — |

Paired differences (points) with 95% bootstrap intervals (10,000 resamples over queries, seed 42):

| Retriever | CR − Vanilla @5 | @10 | SAADI − CR @5 | @10 | SAADI-on-CR − CR @5 | @10 |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 | +3.2 [+1.7, +4.8] | +4.9 [+3.2, +6.6] | -0.7 [-2.3, +1.0] | -0.8 [-2.7, +1.1] | +1.1 [-0.3, +2.4] | +2.5 [+0.9, +4.1] |
| Contriever | +1.6 [+0.0, +3.2] | +2.7 [+0.9, +4.5] | -0.5 [-2.2, +1.2] | -0.2 [-2.1, +1.7] | +0.9 [-0.7, +2.4] | +1.6 [-0.1, +3.2] |
| BGE-M3 | +2.9 [+1.0, +4.7] | +4.6 [+2.7, +6.5] | -2.9 [-4.9, -1.0] | -3.0 [-5.0, -1.1] | -1.3 [-3.0, +0.4] | -0.5 [-2.2, +1.1] |
| Jina | +1.9 [+0.2, +3.6] | +3.1 [+1.2, +5.0] | +1.0 [-0.9, +2.9] | -1.0 [-2.9, +1.0] | +0.2 [-1.4, +1.9] | -0.2 [-1.9, +1.4] |
| Qwen | -6.8 [-9.0, -4.5] | -7.1 [-9.2, -4.8] | +7.5 [+5.2, +9.8] | +8.6 [+6.4, +11.0] | +5.0 [+3.3, +6.8] | +5.9 [+4.2, +7.7] |

## 8. Retrieval: hybrid (dense + BM25, cookbook fusion)

| Hybrid | Vanilla @5 / @10 | CR @5 / @10 | SAADI @5 / @10 | CR − Vanilla @10 | SAADI − CR @10 | SAADI − CR @5 |
| --- | --- | --- | --- | --- | --- | --- |
| Contriever + BM25 | 15.8 / 25.3 | 17.6 / 28.3 | 17.3 / 26.9 | +3.0 [+1.3, +4.7] | -1.4 [-3.2, +0.6] | -0.2 [-1.9, +1.5] |
| BGE-M3 + BM25 | 21.4 / 29.4 | 24.4 / 34.1 | 21.4 / 30.6 | +4.7 [+2.8, +6.8] | -3.5 [-5.5, -1.6] | -3.0 [-4.9, -1.1] |
| Jina + BM25 | 17.1 / 26.8 | 18.8 / 29.9 | 19.3 / 28.4 | +3.2 [+1.2, +5.1] | -1.5 [-3.5, +0.5] | +0.5 [-1.4, +2.4] |
| Qwen + BM25 | 22.4 / 32.5 | 17.3 / 29.2 | 23.1 / 34.5 | -3.3 [-5.5, -1.1] | +5.3 [+3.2, +7.7] | +5.9 [+3.7, +8.1] |
- Top 150 of each list, score 0.8 / rank_dense + 0.2 / rank_bm25 (1-based), the same merge for Vanilla, CR and SAADI (SAADI = SAADI with d + SAADI with BM25, published weights). No reranker.

## 9. Dense truncation of the contextualized chunks

| Retriever | Max input | CR chunks over max | CR tokens cut (end of chunk) | Plain chunks over max | Extra chunk tokens cut per chunk (CR vs plain) |
| --- | --- | --- | --- | --- | --- |
| Contriever | 512 | 23/17159 | 2291 | 5/17159 | 0 |
| BGE-M3 | 8192 | 0/17159 | 0 | 0/17159 | 0 |
| Jina | 512 | 38/17159 | 3390 | 5/17159 | 0 |
| Qwen | 512 | 7/17159 | 703 | 0/17159 | 0 |

## 10. Sanity sample (10 random contexts, seed 42)

- `group_005:7` (segment 1/1, 78 tokens, stop): **The chunk discusses a two-stage translation approach inspired by prior work that uses a draft translation from NMT followed by error correction. It highlights the importance of right context for translation regularization, contrasting the error-correction focus of previous methods with the current approach's complete re-translation, which avoids co-correction issues. The drafting-and-refinement concept is also noted as applicable to other tasks.**
  - chunk: Similar to our two-stage approach, they got a draft translation using NMT, and then designed a `word correction' model that can correct the potential errors in the draft translation. The author raised the a similar argument that the right context is important to regularize the translation; the diffe…
- `group_002:172` (segment 1/1, 99 tokens, stop): **This chunk discusses the challenges and objectives of detecting bots in health-related social media data, particularly on Twitter. It highlights that while existing bot detection methods—such as analyzing Honeypot profiles, social proximity, content and graph-based features, and tweet timing—have been used, bots have evolved to evade these techniques. The study aims to evaluate an existing bot detection system (Botometer) on health-related user data and extend it with new features to improve detection accuracy in the health domain.**
  - chunk: Some studies have used Honeypot profiles on Twitter to identify and analyze bots BIBREF12, while other studies have analyzed social proximity BIBREF13 or both social and content proximities BIBREF10, tweet timing intervals BIBREF14, or user-level content-based and graph-based features BIBREF15. Howe…
- `group_012:161` (segment 1/1, 108 tokens, stop): **This chunk is from a paper proposing a unified computational framework for generating major types of Chinese classical poetry, including Jueju and Lvshi (forms of SHI) and 121 major forms (Cipai) of CI, using a single GPT-2-based model. The framework enhances form control through a form-stressed weighting method and a unified training format, with preliminary results showing high-quality generation. The model has been integrated into Jiuge, a leading Chinese classical poetry generation system developed by Tsinghua University.**
  - chunk: In this paper, we propose a uniformed computational framework that tries to generate major types of Chinese classical poems with two major forms of SHI, Jueju, and Lvshi, as well as 121 major forms (Cipai) of CI using a single model. Preliminary experimental results validate the effectiveness of the…
- `group_010:59` (segment 1/1, 154 tokens, stop): **The chunk discusses the concept of "fortuitous data" — data that is readily available and easy to use, often as an unintended byproduct of other processes. It emphasizes two key dimensions: availability and ease-of-use. The author argues for leveraging such data to address challenges in natural language processing (NLP), particularly in handling non-canonical or diverse text domains. The discussion is framed within a broader critique of the lack of consensus on what constitutes a "domain" in NLP, referencing prior work that attributes domain differences to vocabulary, sociolinguistic factors, **
  - chunk: Availability and ease-of-use (or readiness) are therefore two important dimensions that define fortuitous data. Fortuitous data is the unintended yield of a process, a promising by-product or side benefit. In the following I will outline potential sources of fortuitous data. An overview is given in …
- `group_009:606` (segment 1/1, 120 tokens, stop): **This chunk is from Document 20/22, which proposes a joint model for zero anaphora (ZP) prediction and translation in machine translation, particularly for pro-drop languages like Chinese and Japanese. The section describes the experimental setup for training the model on Chinese-to-English and Japanese-to-English translation tasks, including the use of an RNN-based NMT model with dropout and improved attention, vocabulary limits, sentence length constraints, and mini-batch settings. The context is part of a larger evaluation of the model's performance against baselines and external ZP predicti**
  - chunk: AAAI, we also implemented our approach on top of the RNN-based NMT model, which incorporates dropout BIBREF21 on the output layer and improves the attention model by feeding the most recently generated word. For training the models, we limited the source and target vocabularies to the most frequent …
- `group_006__variant_ab84194533:173` (segment 1/1, 141 tokens, stop): **This chunk is part of a research paper that proposes a sub-sampled unigram distribution for negative sampling in Word2Vec, aiming to improve the quality of learned word vectors. The section discusses the theoretical foundation of the method, including the use of Zipf's law to derive a sub-sampling rate that balances the representation of frequent and infrequent words. The authors argue that their approach, which adapts to different corpora, outperforms the standard smoothed unigram distribution in word similarity, synonym selection, and word analogy tasks. The chunk specifically highlights the**
  - chunk: In semantics quantification, the modeling of word distribution is not limited to zipf's law. We adopt it because of its popularity and conciseness. There could be other choices BIBREF28, BIBREF29, and the expression of $t_c$ needs modification accordingly. Besides, one can either use the chosen law …
- `group_005:438` (segment 1/1, 88 tokens, stop): **The chunk describes the post-voting phase of the MagiCoder algorithm, where a sub-dictionary of MedDRA terms "voted" by words in a narrative ADR description is constructed, along with auxiliary data structures tracking word positions in both the description and the MedDRA terms, and preparing for term sorting based on predefined criteria. This step occurs after the word-by-word scan and voting task in the algorithm's pipeline.**
  - chunk: At the end of the scan, the procedure has built a sub-dictionary containing only terms “voted” at least by one word. We call INLINEFORM5 the sub-dictionary of voted terms. Each voted term INLINEFORM0 is equipped with two auxiliary data structures, containing, respectively: the positions of the votin…
- `group_004:668` (segment 1/1, 80 tokens, stop): **The chunk discusses early neural network approaches for question answering tasks, specifically highlighting the Memory Network framework and the Impatient/Attentive Reader models, which utilize attention mechanisms to access external memory and process textual information for accurate answers. This context is part of a broader exploration of neural architectures for reasoning over facts and documents, particularly in the domain of conversational recommender systems and open-domain question answering.**
  - chunk: One of the earlier approaches used to solve these tasks is given by the general Memory Network BIBREF21 , BIBREF22 framework which is one of the first neural network models able to access external memories to extract relevant information through an attention mechanism and to use them to provide the …
- `group_018__variant_62176dfa97:402` (segment 1/1, 86 tokens, stop): **The chunk discusses the use of discourse connectives as a strong feature for classifying explicit discourse relations and introduces a method for identifying non-explicit relations between adjacent sentences by leveraging shared words like "it" and "them," which are indicative of relational connections. This is part of a broader framework for shallow discourse parsing that integrates connective identification, argument detection, sense classification, and non-explicit relation recognition in a pipeline architecture.**
  - chunk: The connective itself is a very good feature because discourse connective almost determine senses. So we train an explicit classifier using simple but effective features. The non-explicit relation is the relation between adjacent sentences in same para-graph. So we just check adjacent sentences whic…
- `group_002:372` (segment 1/1, 152 tokens, stop): **In BIBREF3 (BIBREF3)'s work, commonly appearing words remain unchanged, while others are segmented into several subword units, which are from a fixed set. Both source and target side sentences can be changed into subword sequences. More specifically, some rare words are split into and represent as some more frequent units, based on a data compression technique, namely Byte Pair Encoding (BPE). The vocabulary built on common words and these frequent subword units can successfully improve the coverage of training data. This approach is part of a broader effort to address the out-of-vocabulary (O**
  - chunk: In BIBREF3 ( BIBREF3 )'s work, commonly appearing words remain unchanged, while others are segmented into several subword units, which are from a fixed set. Both source and target side sentences can be changed into subword sequences. More specifically, some rare words are split into and represent as…

## 11. Estimates and timings

- Generation estimate 4.9 h for 17159 calls (per call 9.3e-06 s x visible document tokens + 0.15 s, plus 420 s engine start (B2's measured rate on the same model and runtime)); measured 2.9 h.
- BM25: estimate 7 min on CPU (C2's CPU rate x 3 (published streams, empty-context pass, CR pass)); measured 4 min.
- Contriever: estimate 3 min on 1 GPU(s) (C2 streams reused: CR chunks + plain chunks (empty-context check) encoded, 2 query encodings per query); measured 7 min.
- BGE-M3: estimate 14 min on 1 GPU(s) (C2 streams reused: CR chunks + plain chunks (empty-context check) encoded, 2 query encodings per query); measured 12 min.
- Jina: estimate 15 min on 1 GPU(s) (C2 streams reused: CR chunks + plain chunks (empty-context check) encoded, 2 query encodings per query); measured 7 min.
- Qwen: estimate 2.9 h on 2 GPU(s) (C2 streams reused: CR chunks + plain chunks (empty-context check) encoded, 2 query encodings per query); measured 3 min.

