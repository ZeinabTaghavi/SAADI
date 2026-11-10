# C1: Contextual Retrieval on SAADI's four benchmarks

Rebuttal experiment for reviewers zXpi (Q2, "please compare against Contextual Retrieval") and iVU8
(novelty: SAADI "in the same family as Contextual Retrieval"). Contextual Retrieval (CR, Anthropic 2024)
writes one short context per chunk from the whole document and indexes `context + "\n\n" + chunk`.
C1 runs it under SAADI's conditions: the paper's saved chunks, queries and evidence labels, SAADI's
construction LLM (Qwen3-30B-A3B-Instruct-2507, greedy), the five Table 1 retrievers, the paper's HR.
Results are reported as they come out, also where CR is better.

## Start

From the repository root on the GPU server, one script per GPU group (they can run at the same time):

```bash
nohup bash experiments/rebuttal/C1_contextual_retrieval/run_gpus_0123.sh > /dev/null 2>&1 &   # group A: LooGLE, then NovelHopQA
nohup bash experiments/rebuttal/C1_contextual_retrieval/run_gpus_4567.sh > /dev/null 2>&1 &   # group B: MuSiQue-32K, then QASPER-64K
```

To finish whatever is missing on all four datasets (after a crash, a stop, or a failed job), and to see what
is done:

```bash
bash experiments/rebuttal/C1_contextual_retrieval/finish.sh    # starts only the groups that are incomplete and not running
bash experiments/rebuttal/C1_contextual_retrieval/status.sh    # per dataset: contexts n/expected, each retriever, report; running processes
```

A group never runs twice at once (lock file `outputs/rebuttal/C1_contextual_retrieval/group_<A|B>.lock`; a
second start exits with code 3), and its preflight refuses GPUs that another job already uses.

Each script tees its log to `logs/group_<A|B>_<timestamp>.log`; per-step logs (`<dataset>__generation.log`,
`<dataset>__<retriever>.log`, `memory_check_*.log`) are next to it. Rerunning the same command resumes.
`python scripts/rebuttal/c1_contextual_retrieval.py --report-only` rebuilds every report from what exists.
`--dry-run` (2 documents per dataset) and `--dry-run-full` (all documents) run on CPU with BM25 and stub contexts.

Exit codes: 0 everything done; 1 a step failed (its report is still written); 2 preflight failed
(nothing started; for example a GPU of the group already uses more than 2 GB, which means C2 or
another job is there: C1 never shares GPUs).

## Steps per group

1. Preflight: Python env and commit, modules (torch, transformers, vLLM), model weights in the HF cache
   (offline mode, nothing is downloaded mid-run; `C1_HF_HUB_OFFLINE=0` to allow it), free disk, the
   group's four GPUs free, the paper's input files, B2's prompt equal to Anthropic's text.
2. Token census per dataset with the Qwen tokenizer: document lengths, documents over the window, the
   segment plan, exact nominal prompt tokens. Then the expected run time per dataset (written to
   `reports/group_<X>.md` before generation starts).
3. Memory check of every embedding model on the group's GPUs: one batch of the longest plain chunk with
   1,024 placeholder tokens in front (an upper bound for any context), float32 as Vanilla; the batch size
   starts at Vanilla's and is halved until the peak fits with a 10% margin; Qwen3-Embedding-8B gets two
   GPUs (C2 found it does not fit on one A6000).
4. Per dataset: contexts with vLLM (TP 4 on the group's GPUs, the engine exits afterwards), then the
   retrieval jobs (dense encoders in parallel, one GPU each and two for Qwen; BM25 on CPU), then the
   hybrid and the complementarity check (CPU), then `reports/<dataset>.md` and `.json`.
5. The final report (`results/rebuttal/C1_contextual_retrieval/REPORT.md`, `REPORT_FOR_GDOC.md`,
   `tables/*.csv`) is written by whichever group finishes last (file lock).

## Fixed before running (nothing is tuned on evaluated queries)

- **Prompt**: Anthropic's published prompt, verbatim, as one user message with the document first;
  imported from B2 (`experiments/probe_baselines/systems/b2_contextual/build.py`, `prompt_for`) and
  checked against the published text in preflight.
- **LLM**: Qwen3-30B-A3B-Instruct-2507, temperature 0 (top_p 1, top_k -1, seed 42), `max_tokens` 1024
  (B2 used 200 and cut 4.4% of its contexts), vLLM offline engine with SAADI's saved runtime (bf16, TP 4,
  memory utilization 0.8, max_num_seqs 16, max_num_batched_tokens 16384, custom all-reduce off, prefix
  caching on) and `max_model_len` = the model's native 262,144 (config, no rope scaling).
- **Prefix caching**: segments are processed in waves whose prefixes fit in half the KV cache; for each
  segment one request runs first (it computes the document prefix), then all other chunks of that
  segment. Prompt token ids are built as prefix + suffix so that every request of a segment starts with
  identical ids (checked equal to joint tokenization). Nominal (full prompt) and cached tokens come from
  vLLM per request (`num_cached_tokens`).
- **Document text**: the saved chunks of the document, in order, joined by a single space. No separator
  restores the source exactly (the chunker's sentence splitter changed whitespace inside chunks, e.g.
  `1912. 02866`); the rebuilt text equals the source up to whitespace and is the text SAADI read.
- **Long documents**: if document + prompt + longest chunk + 1,024 output tokens exceed the window, the
  document is split at chunk boundaries into the smallest number k of contiguous segments of near-equal
  token length that each fit; each chunk gets its own segment as `{{WHOLE_DOCUMENT}}`. Expected on
  NovelHopQA only, reported as a limit of CR on book-length text.
- **Vanilla** (decided with the user, 2026-11-09): SAADI's text channel alone on the saved chunks
  (lambda_M = lambda_S = 0), so that only the indexed text differs between Vanilla and CR. On MuSiQue-32K
  and QASPER-64K it equals Table 1's Vanilla to the rounding; on LooGLE and NovelHopQA Table 1's Vanilla
  came from another run (C2's server run: up to 13 points apart, e.g. LooGLE Contriever HR@10 86.0 vs
  73.3), so both are shown and the gate there uses the saved per-query text-only files.
- **Retrieval** (all from the same contexts, no extra LLM calls): (a) contextual BM25 with SAADI's BM25
  on the contextualized text as one field; (b) contextual dense with the four Table 1 dense retrievers,
  the same adapters, truncation and query formatting as Vanilla (truncated chunks are counted);
  (c) hybrid: top 150 of contextual dense and of contextual BM25, `0.8/rank_dense + 0.2/rank_bm25`
  (1-based; the cookbook's `retrieve_advanced`), ties as the cookbook sorts them; the same merge for
  Vanilla and for SAADI. No reranker.
- **Scores** go through the published code: the text channel's own function (`mbr._score_pool`) on the
  contextualized texts, and `combine_mbr_variant_chunk_scores` for every ranking. Memory and summary
  channels are reused from C2's saved streams when they exist (`outputs/rebuttal/C2_heldout_fusion/streams/`,
  DONE and passed), otherwise recomputed with the published path; the report says which.
- **Correctness gate** per (dataset, retriever), before real contexts are used: SAADI's saved rankings
  reproduced with the published weights (the pool and queries are the paper's); CR with empty contexts
  equal to Vanilla (BM25: identical top-150 rankings; dense: identical top 10 for at least 99% of queries
  and HR within 0.1 points, with C2's drift rule for Jina, extended to Qwen with the user on 2026-11-10: the
  recomputed Qwen3-Embedding-8B scores on LooGLE reorder near-ties against the paper's run in C2 and C1 alike,
  733/1832 identical top-10, HR@10 86.41 vs 86.68; accepted while HR stays within 0.5 points, and every system
  of the pair uses the same recomputed scores); equal to the saved per-query text-only files where
  they exist; equal to Table 1 Vanilla on MuSiQue-32K and QASPER-64K. A failing gate stops that retriever.
- **Complementarity check** (not a proposed method): SAADI's published fusion with the text channel on
  the contextualized chunks, memory and summary channels unchanged.
- **Statistics**: paired bootstrap, 10,000 resamples over queries, seed 42; counts over like-for-like
  pairs only.

## Expected cost of the GPU run (from the full token census, 2026-11-09)

Estimates, not measurements. Generation uses B2's measured rate on the same model and runtime
(9.3e-6 s per visible document token per call, plus 0.15 s per call and about 7 min engine start);
retrieval uses C2's embedding cost model. The range is "C2's streams reused" to "recomputed".

| Group | Dataset | Docs | Chunks = LLM calls | Doc tokens median (max) | Docs over the window | Generation | Retrieval | Dataset |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | LooGLE | 105 | 3,346 | 19K (60K) | 0 | 0.5 h | 1.6-2.1 h | 2.1-2.6 h |
| A | NovelHopQA | 16 | 7,011 | 234K (763K) | 7 (4,250 chunks; k = 2 for 6, 3 for 1) | 4.1 h | 3.1-3.7 h | 7.3-7.9 h |
| B | MuSiQue-32K | 45 | 17,102 | 49K (52K) | 0 | 3.0 h | 3.0-4.2 h | 6.0-7.3 h |
| B | QASPER-64K | 23 | 17,159 | 92K (99K) | 0 | 4.9 h | 2.9-4.2 h | 7.9-9.2 h |

- Wall clock: group A about 9.5-10.5 h, group B about 14-16.5 h; about 73-81 GPU-hours in all
  (generation about 50 GPU-h on 4 GPUs per group; Qwen3-Embedding-8B on two GPUs is most of the rest).
- LLM tokens: 44,618 calls; nominal input 3.95B tokens (the whole document in every prompt; 28x
  SAADI's 139M at SAADI's per-chunk rate), computed input about 26M with ideal prefix caching (0.18x
  SAADI); generated about 6M at B2's median of 135 tokens per context (SAADI: 37M).

## Files

- Code: `src/saadi/rebuttal/contextual_retrieval.py` (library), `scripts/rebuttal/c1_contextual_retrieval.py` (entry point).
- Large outputs (not committed): `outputs/rebuttal/C1_contextual_retrieval/<dataset>/` with `census.json`,
  `contexts.jsonl` (chunk id, context, nominal prompt tokens, cached tokens, generated tokens, finish
  reason, segment), `generation.json`, `retrieval/<retriever>/rankings.pkl.gz` (top 150 per system and
  query) and `markers/`.
- Reports: `reports/group_<A|B>.md`, `reports/<dataset>.md` and `.json`; final: `results/rebuttal/C1_contextual_retrieval/`.
