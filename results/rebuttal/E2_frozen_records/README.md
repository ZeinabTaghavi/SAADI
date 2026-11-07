# E2. Frozen records against live records

**Command.** `.venv/bin/python scripts/rebuttal/e2_frozen_records.py --retrievers bm25` (from the repository root; no LLM calls, no GPU, no network).

**Settings.** Published MBR2 scoring: weights (lambda_t, lambda_n, lambda_h) = (1.0, 0.5, 0.3),
field weights (1, 1, 1), max aggregation, z-score normalization recomputed per variant,
retrieve_k = 10, HR computed with `src/saadi/evaluation/hr.py`'s implementation
(`evaluate_retrieval_results`). Query sets, chunks and evidence labels are read from the
paper's saved per-document rankings, so they are identical to the paper.

**Queries.** {"loogle": 1832, "musique_32k": 900, "qasper_64k": 1333, "novelhopqa": 908}

**Answers to Step 0.**
- Per-chunk construction records: `outputs/<dataset>_ablation/<variant>/<chunk>/llm_cache/.../AEWord/<doc>/steps_index.jsonl`
  with `L_0.jsonl`, `L_1.jsonl`, `memory.json`, indexed by `index.json` (LooGLE and NovelHopQA 500-token,
  MuSiQue 32K-expansion 100-token, QASPER 64K-expansion 100-token namespaces).
- `state_snapshot["active_entities"]` holds only the recalled subset (max 5 records), with the state
  AFTER that step's update. The full per-step memory is rebuilt by replaying the merges, exactly as the
  repository's loader does; replaying to step N reproduces `memory.json` exactly (checked per document).
- Chunk-to-name links: the names in the chunk's own step row, kept when they exist in the final memory
  (`chunk_to_names` in `build_mbr_document_memory_index`).
- Memory channel: unique name/alias/description pools from the final records of the linked names, each pool
  scored against the query (BM25 native scoring); per chunk, max over linked items per field; each field
  channel z-scored over the document's chunks; memory = equal-weight mean of the available fields; final
  score = renormalized 1.0*text + 0.5*memory + 0.3*summary.

**Result.** Live and frozen are statistically indistinguishable on every dataset: the paired
live-minus-frozen difference at k=10 is loogle -0.05 pp [-0.27, 0.11]; musique_32k -0.33 pp [-0.89, 0.11]; qasper_64k 0.23 pp [-0.38, 0.83]; novelhopqa 0.11 pp [-0.44, 0.66], and every 95% CI crosses zero
(the same holds at k=5 and on the affected-queries subset). Later record updates are common
(48-97% of chunks have at least one, see chunk_stats.parquet), but under the published scoring
they do not measurably change the retrieval of earlier chunks. The memory channel itself does
help (live against no-memory, e.g. MuSiQue-32K HR@5 62.8 against 59.6), and that gain is already
carried by the records as they stood at each chunk's own step. Reported as they come; the
difference here is zero-to-small, not positive.
