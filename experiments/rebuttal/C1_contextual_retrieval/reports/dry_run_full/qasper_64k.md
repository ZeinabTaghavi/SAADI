# C1 Contextual Retrieval: QASPER-64K (dry run full: stub contexts, BM25 only, placeholder numbers)

- Written 2026-11-09 15:00:13; commit e2dbaa178fa0; documents 23, chunks 17159, queries 1333 (expected 1333 on the full dataset).
- Status: generation done; jobs BM25 DONE.

## 1. Correctness check (empty context reproduces Vanilla)

CR with every context empty indexes exactly the chunk text, so it must give Vanilla (SAADI's text channel alone) and the paper's saved results. BM25: identical rankings; dense: identical top 10 for at least 99% of queries and HR within 0.1 points (C2's Jina drift rule applies). The pool itself is checked by reproducing SAADI's saved rankings with the published weights.

| Retriever | Gate | SAADI top-10 = saved | Empty = Vanilla top-10 | HR@5 / HR@10 empty vs Vanilla | = saved text-only file | Table 1 Vanilla |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 | pass | 1333/1333 | 1333/1333 (top-150 1333) | 14.40 / 22.06 vs 14.40 / 22.06 | no file | 14.4 / 22.1 |
- BM25: Vanilla reference for the dense check = published path in this process (shared encoding cache); memory and summary channels from {'c2_streams': 0, 'published_path': 23} documents.

## 2. Token census (Qwen3 tokenizer)

- Window 262144 tokens (model config, rope_scaling None); prompt template 70 tokens; output budget 1024.
- Document tokens: min 76K, median 92K, max 99K; chunk tokens median 121, max 469.
- Documents over the window: 0 of 23 (0 chunks); segments per document: {'1': 23}.
- Split vs joint tokenization of the prompt identical on the checked samples: [True, True, True].

## 3. GPU memory check (embedding models)

- Skipped (CPU dry run).

## 4. Long documents (segment rule)

- No document exceeds the window; every chunk saw its whole document.

## 5. Contexts

- 17159 contexts (0 missing), generator ['stub']; empty 0, failed 0, cut at 1024 tokens 0.
- Context length (tokens): min 28, p25 38, median 42, mean 43.9, p75 47, p95 59, max 235.
- Cached tokens: ['simulated (stub)'].

## 6. Cost next to SAADI

| Quantity | CR | SAADI (memory + hierarchy calls) |
| --- | --- | --- |
| LLM calls (total) | 17159 | 21404 |
| LLM calls per chunk | 1.00 | 1.25 |
| LLM calls per document | 746.0 | 930.6 |
| Input tokens per chunk, nominal (full prompt) | 92K | 2530 (no prefix cache) |
| Input tokens per chunk, computed (after prefix-cache hits) | 316 | same as nominal |
| Input tokens total, nominal / computed | 1.59B / 5.4M | 43.4M |
| Generated tokens per chunk, mean / median | 43.9 / 42.0 | 633.4 |
| Generated tokens total | 752K | 10.9M |
| Contexts cut at 1024 tokens | 0 (0.0%) | — |
| Generation wall clock / GPU-hours | 45s / 0.00 | not in the paper's tables |
| Indexing + scoring, BM25 (wall / GPU-hours) | 2 min / 0.00 | — |
- Ratios per chunk. SAADI's builds had no prefix caching, so its input is compared with both CR columns: CR nominal / SAADI =36.55x, CR computed / SAADI = 0.12x; calls 0.80x; generated tokens 0.07x. SAADI chunks in the cost table: 17159; C1 chunks: 17159.

## 7. Retrieval: per retriever

HR@5 / HR@10 (%). *Complementarity check, not a proposed method.

| Retriever | Vanilla (text channel) | Table 1 Vanilla | CR | SAADI (published) | SAADI on CR text* | LQCA (Table 1) |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 | 14.4 / 22.1 | 14.4 / 22.1 | 14.6 / 23.0 | 17.0 / 26.1 | 17.1 / 27.2 | 36.9 / 47.0 |
- SAADI held-out column left out: results/rebuttal/C2_heldout_fusion/REPORT.md does not exist yet.

Paired differences (points) with 95% bootstrap intervals (10,000 resamples over queries, seed 42):

| Retriever | CR − Vanilla @5 | @10 | SAADI − CR @5 | @10 | SAADI-on-CR − CR @5 | @10 |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 | +0.2 [-1.1, +1.5] | +0.9 [-0.6, +2.5] | +2.4 [+0.8, +4.0] | +3.2 [+1.4, +4.9] | +2.6 [+1.1, +4.1] | +4.2 [+2.6, +5.8] |

## 8. Retrieval: hybrid (dense + BM25, cookbook fusion)

| Hybrid | Vanilla @5 / @10 | CR @5 / @10 | SAADI @5 / @10 | CR − Vanilla @10 | SAADI − CR @10 | SAADI − CR @5 |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 + BM25 (dry-run stand-in for a dense retriever) | 14.4 / 22.1 | 14.6 / 23.0 | 17.0 / 26.1 | +0.9 [-0.6, +2.5] | +3.2 [+1.4, +4.9] | +2.4 [+0.8, +4.0] |
- Top 150 of each list, score 0.8 / rank_dense + 0.2 / rank_bm25 (1-based), the same merge for Vanilla, CR and SAADI (SAADI = SAADI with d + SAADI with BM25, published weights). No reranker.

## 9. Dense truncation of the contextualized chunks

- No dense retriever in this run.

## 10. Sanity sample (10 random contexts, seed 42)

- `group_005:7` (segment 1/1, 43 tokens, stop): **Stub context: found that his sequence-to-sequence model achieved a promising improvement when reversing the source sentence “ INLINEFORM0 ” to “ INLINEFORM1 ” BIBREF8 . They argued that reversing the input may**
  - chunk: Similar to our two-stage approach, they got a draft translation using NMT, and then designed a `word correction' model that can correct the potential errors in the draft translation. The author raised the a similar argument that the right context is important to regularize the translation; the diffe…
- `group_002:172` (segment 1/1, 49 tokens, stop): **Stub context: Bots may spread false information by automatically retweeting posts without a human verifying the facts or to influence public opinions on particular topics on purpose BIBREF5, BIBREF7, BIBREF8. For example,**
  - chunk: Some studies have used Honeypot profiles on Twitter to identify and analyze bots BIBREF12, while other studies have analyzed social proximity BIBREF13 or both social and content proximities BIBREF10, tweet timing intervals BIBREF14, or user-level content-based and graph-based features BIBREF15. Howe…
- `group_012:161` (segment 1/1, 44 tokens, stop): **Stub context: Most of the work so far mainly targeted at automatic generation of Jueju (including Wuyan Jueju and Qiyan Jueju), for an obvious reason that it is much easier for an**
  - chunk: In this paper, we propose a uniformed computational framework that tries to generate major types of Chinese classical poems with two major forms of SHI, Jueju, and Lvshi, as well as 121 major forms (Cipai) of CI using a single model. Preliminary experimental results validate the effectiveness of the…
- `group_010:59` (segment 1/1, 40 tokens, stop): **Stub context: What we need are models that are more robust, work better on unexpected input and can be trained from semi-automatically or weakly annotated data, from a variety of sources. In**
  - chunk: Availability and ease-of-use (or readiness) are therefore two important dimensions that define fortuitous data. Fortuitous data is the unintended yield of a process, a promising by-product or side benefit. In the following I will outline potential sources of fortuitous data. An overview is given in …
- `group_009:606` (segment 1/1, 49 tokens, stop): **Stub context: 15K sentence pairs, respectively. In the training data, there are 27% of Chinese pronouns are ZPs, which poses difficulties for NMT models. For Japanese $\Rightarrow $ English translation task, we**
  - chunk: AAAI, we also implemented our approach on top of the RNN-based NMT model, which incorporates dropout BIBREF21 on the output layer and improves the attention model by feeding the most recently generated word. For training the models, we limited the source and target vocabularies to the most frequent …
- `group_006__variant_ab84194533:173` (segment 1/1, 88 tokens, stop): **Stub context: The estimated constants are where $\left\langle x \right\rangle $ denotes the weighted average of $x$ such that $\left\langle x \right\rangle = \sum _{r=1}^{|V|}\frac{x}{r} / \sum _{r=1}^{|V|}\frac{1}{r}$. (2) wLSE-2. We use**
  - chunk: In semantics quantification, the modeling of word distribution is not limited to zipf's law. We adopt it because of its popularity and conciseness. There could be other choices BIBREF28, BIBREF29, and the expression of $t_c$ needs modification accordingly. Besides, one can either use the chosen law …
- `group_005:438` (segment 1/1, 41 tokens, stop): **Stub context: Moreover, it keeps track of the position where the INLINEFORM5 -th word occurs in INLINEFORM6 . INLINEFORM0 tries to find a word match both for the exact and the stemmed**
  - chunk: At the end of the scan, the procedure has built a sub-dictionary containing only terms “voted” at least by one word. We call INLINEFORM5 the sub-dictionary of voted terms. Each voted term INLINEFORM0 is equipped with two auxiliary data structures, containing, respectively: the positions of the votin…
- `group_004:668` (segment 1/1, 50 tokens, stop): **Stub context: MCTest BIBREF19 requires machines to answer multiple-choice reading comprehension questions about fictional stories, directly tackling the high-level goal of open-domain machine comprehension. Finally, SQuAD BIBREF20 consists in a set of**
  - chunk: One of the earlier approaches used to solve these tasks is given by the general Memory Network BIBREF21 , BIBREF22 framework which is one of the first neural network models able to access external memories to extract relevant information through an attention mechanism and to use them to provide the …
- `group_018__variant_62176dfa97:402` (segment 1/1, 38 tokens, stop): **Stub context: Second, we need to choose the best candidate as the head of Arg1 and Arg2. In the end, we need to obtain the arguments span according head and end of**
  - chunk: The connective itself is a very good feature because discourse connective almost determine senses. So we train an explicit classifier using simple but effective features. The non-explicit relation is the relation between adjacent sentences in same para-graph. So we just check adjacent sentences whic…
- `group_002:372` (segment 1/1, 35 tokens, stop): **Stub context: Results show that our method gives significant improvement on the English to Russian translation task on two different domains and two popular NMT architectures. We also verify our method on**
  - chunk: In BIBREF3 ( BIBREF3 )'s work, commonly appearing words remain unchanged, while others are segmented into several subword units, which are from a fixed set. Both source and target side sentences can be changed into subword sequences. More specifically, some rare words are split into and represent as…

## 11. Estimates and timings

- Generation estimate 4.9 h for 17159 calls (per call 9.3e-06 s x visible document tokens + 0.15 s, plus 420 s engine start (B2's measured rate on the same model and runtime)); measured 45s.
- BM25: estimate 7 min on CPU (C2's CPU rate x 3 (published streams, empty-context pass, CR pass)); measured 2 min.

