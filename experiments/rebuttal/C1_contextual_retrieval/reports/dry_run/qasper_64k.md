# C1 Contextual Retrieval: QASPER-64K (dry run: stub contexts, BM25 only, placeholder numbers)

- Written 2026-11-09 15:02:20; commit e2dbaa178fa0; documents 2, chunks 1488, queries 129 (expected 1333 on the full dataset).
- Status: generation done; jobs BM25 DONE.

## 1. Correctness check (empty context reproduces Vanilla)

CR with every context empty indexes exactly the chunk text, so it must give Vanilla (SAADI's text channel alone) and the paper's saved results. BM25: identical rankings; dense: identical top 10 for at least 99% of queries and HR within 0.1 points (C2's Jina drift rule applies). The pool itself is checked by reproducing SAADI's saved rankings with the published weights.

| Retriever | Gate | SAADI top-10 = saved | Empty = Vanilla top-10 | HR@5 / HR@10 empty vs Vanilla | = saved text-only file | Table 1 Vanilla |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 | pass | 129/129 | 129/129 (top-150 129) | 14.73 / 20.16 vs 14.73 / 20.16 | no file | 14.4 / 22.1 |
- BM25: Vanilla reference for the dense check = published path in this process (shared encoding cache); memory and summary channels from {'c2_streams': 0, 'published_path': 2} documents.

## 2. Token census (Qwen3 tokenizer)

- Window 262144 tokens (model config, rope_scaling None); prompt template 70 tokens; output budget 1024.
- Document tokens: min 89K, median 89K, max 89K; chunk tokens median 119, max 237.
- Documents over the window: 0 of 2 (0 chunks); segments per document: {'1': 2}.
- Split vs joint tokenization of the prompt identical on the checked samples: [True, True].

## 3. GPU memory check (embedding models)

- Skipped (CPU dry run).

## 4. Long documents (segment rule)

- No document exceeds the window; every chunk saw its whole document.

## 5. Contexts

- 1488 contexts (0 missing), generator ['stub']; empty 0, failed 0, cut at 1024 tokens 0.
- Context length (tokens): min 33, p25 38, median 41, mean 42.9, p75 46, p95 57, max 104.
- Cached tokens: ['simulated (stub)'].

## 6. Cost next to SAADI

| Quantity | CR | SAADI (memory + hierarchy calls) |
| --- | --- | --- |
| LLM calls (total) | 1488 | 21404 |
| LLM calls per chunk | 1.00 | 1.25 |
| LLM calls per document | 744.0 | 930.6 |
| Input tokens per chunk, nominal (full prompt) | 90K | 2530 (no prefix cache) |
| Input tokens per chunk, computed (after prefix-cache hits) | 310 | same as nominal |
| Input tokens total, nominal / computed | 133.2M / 462K | 43.4M |
| Generated tokens per chunk, mean / median | 42.9 / 41.0 | 633.4 |
| Generated tokens total | 64K | 10.9M |
| Contexts cut at 1024 tokens | 0 (0.0%) | — |
| Generation wall clock / GPU-hours | 5s / 0.00 | not in the paper's tables |
| Indexing + scoring, BM25 (wall / GPU-hours) | 7s / 0.00 | — |
- Ratios per chunk. SAADI's builds had no prefix caching, so its input is compared with both CR columns: CR nominal / SAADI =35.39x, CR computed / SAADI = 0.12x; calls 0.80x; generated tokens 0.07x. SAADI chunks in the cost table: 17159; C1 chunks: 1488.

## 7. Retrieval: per retriever

HR@5 / HR@10 (%). *Complementarity check, not a proposed method.

| Retriever | Vanilla (text channel) | Table 1 Vanilla | CR | SAADI (published) | SAADI on CR text* | LQCA (Table 1) |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 | 14.7 / 20.2 | 14.4 / 22.1 | 16.3 / 24.0 | 20.9 / 25.6 | 18.6 / 30.2 | 36.9 / 47.0 |
- SAADI held-out column left out: results/rebuttal/C2_heldout_fusion/REPORT.md does not exist yet.

Paired differences (points) with 95% bootstrap intervals (10,000 resamples over queries, seed 42):

| Retriever | CR − Vanilla @5 | @10 | SAADI − CR @5 | @10 | SAADI-on-CR − CR @5 | @10 |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 | +1.6 [-2.3, +5.4] | +3.9 [-0.8, +8.5] | +4.7 [+0.0, +10.1] | +1.6 [-4.7, +7.8] | +2.3 [-2.3, +7.8] | +6.2 [+0.8, +11.6] |

## 8. Retrieval: hybrid (dense + BM25, cookbook fusion)

| Hybrid | Vanilla @5 / @10 | CR @5 / @10 | SAADI @5 / @10 | CR − Vanilla @10 | SAADI − CR @10 | SAADI − CR @5 |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 + BM25 (dry-run stand-in for a dense retriever) | 14.7 / 20.2 | 16.3 / 24.0 | 20.9 / 25.6 | +3.9 [-0.8, +8.5] | +1.6 [-4.7, +7.8] | +4.7 [+0.0, +10.1] |
- Top 150 of each list, score 0.8 / rank_dense + 0.2 / rank_bm25 (1-based), the same merge for Vanilla, CR and SAADI (SAADI = SAADI with d + SAADI with BM25, published weights). No reranker.

## 9. Dense truncation of the contextualized chunks

- No dense retriever in this run.

## 10. Sanity sample (10 random contexts, seed 42)

- `group_014:608` (segment 1/1, 41 tokens, stop): **Stub context: For example, if we tokenized the sentence "go to binghamton university" and process the further steps without adding bigrams of them, the model is likely to yield a lower confidence**
  - chunk: TFIDF, as known as Term frequency inversed document frequency, is a document representation that takes account of the importance of each word by its frequency in the whole set of documents and its frequency in particular sets of documents. Specifically, let $D = \lbrace d_1, \dots , d_n\rbrace $ be …
- `group_013:303` (segment 1/1, 41 tokens, stop): **Stub context: To learn the model parameters, we minimize the cross-entropy loss as the training objective using the Adam Optimization algorithm BIBREF9 . It is given by DISPLAYFORM0 where p is the**
  - chunk: Given a pair of tweets, the goal is to predict their semantic equivalence (i. e. , if they express the same or very similar meaning), through a binary yes/no judgement. The dataset provided for this task contains 18K tweet pairs for training and 1K pairs for testing, with INLINEFORM0 of these pairs …
- `group_013:144` (segment 1/1, 40 tokens, stop): **Stub context: According to our survey BIBREF6 , the obvious opportunity of using SWT for MT has already been studied by a number of approaches, especially w. r. t. the issue of**
  - chunk: Recently, the idea of using KG into MT systems has gained renewed attention. Du et al. BIBREF9 created an approach to address the problem of OOV words by using BabelNet BIBREF10 . Their approach applies different methods of using BabelNet. In summary, they create additional training data and apply a…
- `group_013:605` (segment 1/1, 37 tokens, stop): **Stub context: The intent classifier and semantic slot classifier for human utterances capture the semantic and syntactic meaning of human utterances, providing information to select the appropriate response among response candidates while**
  - chunk: $h^l_{t-1}$ is the hidden states at the end of last sentence in turn $t-1$, $h^i_{t}$ is the last hidden states at the end of $i$-th sentence in turn $t$. $W_{2h}$ are weights learned during training. MISSA is able to classify multiple intents and multiple semantic slots in a single utterance with t…
- `group_013:55` (segment 1/1, 34 tokens, stop): **Stub context: We later show how we extend a few of these works into ours. This dataset consists of tweets with sexist tweets collected from Twitter by searching for tweets that contain**
  - chunk: Recently, BIBREF4 has published a large scale crowdsourced abusive tweet dataset with 60K tweets. Their work incrementally and iteratively investigated methods such as boosted sampling and exploratory rounds, to effectively annotate tweets through crowdsourcing. Through such systematic processes, th…
- `group_013:51` (segment 1/1, 36 tokens, stop): **Stub context: They further defined this model bias as unintended, “a model contains unintended bias if it performs better for comments containing some particular identity terms than for comments containing others. ”**
  - chunk: In this work, we address model biases specific to gender identities (gender bias) existing in abusive language datasets by measuring them with a generated unbiased test set and propose three reduction methods: (1) debiased word embedding, (2) gender swap data augmentation, (3) fine-tuning with a lar…
- `group_013:355` (segment 1/1, 50 tokens, stop): **Stub context: Thus, lower parser scores indicate less grammatically-correct utterances. In future work, we aim to use specifically designed grammar-scoring functions, e. g. BIBREF26 , once they become publicly available. To collect**
  - chunk: How do you judge the overall quality of the utterance in terms of its grammatical correctness and fluency? Each system output (see Table 1 ) was scored by 3 different crowdworkers. To reduce participants' bias, the order of appearance of utterances produced by each system was randomised and crowdwor…
- `group_013:287` (segment 1/1, 40 tokens, stop): **Stub context: In this section, we describe the CNN-LSTM encoder-decoder model that operates at the character level and generates vector representation of tweets. The encoder consists of convolutional layers to extract features**
  - chunk: In this model, we perform temporal convolutional and temporal max-pooling operations, which computes one-dimensional convolution and pooling functions, respectively, between input and output. Given a discrete input function INLINEFORM0 , a discrete kernel function INLINEFORM1 and stride INLINEFORM2 …
- `group_014:677` (segment 1/1, 35 tokens, stop): **Stub context: If a reader is able to guess the dropped information and uses it to solve the question, we cannot say that the question does not require the corresponding skill. For**
  - chunk: ablated features are ($\alpha $) unreconstructable and unnecessary, ($\beta $) unreconstructable and necessary, ($\gamma $) reconstructable and unnecessary, and ($\delta $) reconstructable and necessary. To verify that ablation methods work, we need to confirm that there are few questions of Case $\…
- `group_014:434` (segment 1/1, 45 tokens, stop): **Stub context: This initial FST is augmented by composition (denoted with the INLINEFORM1 -operator) with various other FSTs to cover different error types. Composition is a widely used standard operation on FSTs**
  - chunk: We selected INLINEFORM4 by looking up all tokens which have been deleted in the dev set more than five times and manually filtered that list slightly. We did not use the full list of dev set deletions to avoid under-estimating INLINEFORM5 in tuning. In a next step, we compose the transducer from ste…

## 11. Estimates and timings

- Generation estimate 31 min for 1488 calls (per call 9.3e-06 s x visible document tokens + 0.15 s, plus 420 s engine start (B2's measured rate on the same model and runtime)); measured 5s.
- BM25: estimate 43s on CPU (C2's CPU rate x 3 (published streams, empty-context pass, CR pass)); measured 7s.

