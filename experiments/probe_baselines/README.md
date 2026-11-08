# Probe baselines

Four baselines and one oracle on the persona probe (`experiments/probe_bridge`: the 10 modified
NovelHopQA books, frozen spec, 240 personas, 8 cells; Google Doc "SAADI_updates", tab 2), scored
with the probe's own HR so that every number sits next to SAADI's run 0, run 1 and run 1b. One
script, `run_all.sh`, runs everything on GPUs 4 to 7 overnight; the report for the Doc's new
sub-tab "2.1 Probe baselines" is `results/probe_baselines/REPORT_FOR_GDOC.md`.

## The five systems

**B0 Vanilla BM25.** The probe's own Vanilla column, recomputed through the shared ranking
format: the same function as the probe's step 7 (BM25 over the raw chunks, ties by position).
It reproduces `results/probe_bridge/table.csv` exactly in all 8 cells (checked locally).

**B1 Oracle.** Form 2 (the query's name) appended to the gold chunk of every persona whose form 2
is revealed in the book, then the same BM25. The ceiling for any bridging method. Cell 5 has no
reveal and stays unchanged, so its control keeps its meaning. Locally: HR@10 100% in every cell
with a reveal, median gold rank 1 to 2; cell 5 at 0%.

**B2 Contextual Retrieval, whole document.** One call per chunk with the prompt of Anthropic's
Contextual Retrieval post writes a context that places the chunk in the book; BM25 over
"context + chunk". The vLLM server runs Qwen3-30B-A3B-Instruct-2507 at its native 262,144-token
window (decided with the user; SAADI's builds use 32,768). With the Qwen tokenizer the books
are 166,636 to 268,771 tokens: 8 of 10 fit whole; B26 and B44 see chunks 0-370 and 0-371 of
384 and 385. Every persona records whether its reveal was visible, and whether the written
context contains form 2.

**B3 HippoRAG v2.** The official package (`hipporag` 2.0.0a4) in its own venv, package defaults
(decided with the user; the paper's Appendix E.6 was not available): OpenIE with the
construction LLM through the local server, NV-Embed-v2, its graph and PPR retrieval, one index
per book, passages mapped back to chunk ids by text. Counts triples connecting form 1 and form 2.

**B4 LQCA.** The official code (OceannTwT/LQCA at commit 2d05899, cloned at run time: no license
file) with the Maverick coreference model, called unchanged on consecutive segments of 8 chunks
(decided with the user: the code's mention matrix cannot take a whole book), BM25 over the
rewritten chunks. No LLM. Counts rewritten gold chunks that contain form 2.

## How to run

```
nohup bash experiments/probe_baselines/run_all.sh > experiments/probe_baselines/logs/run_all_nohup.out 2>&1 &
```

`--from <stage>` restarts at a stage, `--force` reruns finished stages from there on. Stages:
00_check, 01_b0_vanilla, 02_b1_oracle, 03_b2_contextual, 04_b3_hipporag, 05_b4_lqca, 06_score,
07_report. The script first verifies the frozen spec (a changed spec stops everything), starts
the vLLM server in the background, installs the two third-party venvs and the LQCA clone while
the model loads, runs the stages, stops the server before B4 (which needs the GPUs), and prints
the path of `reports/FINAL.md` as its last line. A failing system stage writes its report with
the traceback and the last 50 log lines and the script continues; 06_score and 07_report run on
whatever finished. B2 resumes per chunk, B3 from HippoRAG's own caches, B4 per segment.

## Layout

```
experiments/probe_baselines/
  env.sh, run_all.sh, README.md
  common/load_probe.py    the probe (spec hash, books, chunks, personas) via the probe's own code; BM25 as in step 7
  common/chunk_map.py     shared ranking format; units -> chunk ids; vLLM counters (pure stdlib, any venv)
  common/scorer.py        the one scorer; SAADI runs from their per-persona files; self-test
  common/report_utils.py  stage runner, markers, reports in the fixed format
  common/check.py         00_check (spec, export, server and window, scorer self-test, third-party imports)
  common/final_report.py  07_report: FINAL.md and REPORT_FOR_GDOC.md
  systems/<b0..b4>/       build.py, retrieve.py, config.yaml, README.md
  third_party/            venvs and the LQCA clone, created at run time (git-ignored)
  reports/, logs/
outputs/probe_baselines/  inputs/, rankings/<system>/<book>.jsonl, <system>/..., markers/
results/probe_baselines/  scores.json, table.csv, REPORT_FOR_GDOC.md
```

`check.py` and `final_report.py` are two files beyond the requested `common/` list: the stage
checks and the final report did not belong in the four named modules.

## Shared ranking format

One line per persona in `outputs/probe_baselines/rankings/<system>/<book_id>.jsonl`:
`persona_id, book_id, cell, ranked_chunk_ids` (top 50), `model_calls, input_tokens,
generated_tokens`, plus two fields added to the requested ones: `gold_rank` (from the full
ranking) and `n_chunks`. The median gold rank on this probe is 110 to 200, far below a top-50
list, so it cannot be read from the list. SAADI's runs saved only their top 3 and the gold rank;
they are read the same way.

## Interpretations recorded

1. The B1 oracle edits only personas with a reveal; cell 5's gold chunks stay unchanged.
2. B2's document is the book's chunks joined by blank lines; when it does not fit, the longest
   prefix of whole chunks that fits beside the book's longest chunk and the output budget
   (200 tokens). The document part is identical for every chunk of a book, so vLLM's prefix
   cache computes it once per book.
3. Server settings are SAADI's saved runtime except the window (262,144) and
   `--generation-config vllm` (the model's own defaults are temperature 0.7, top_k 20; every
   request sends temperature 0). SAADI's builds use vLLM's offline engine; this run uses its
   OpenAI-compatible server, because HippoRAG needs an endpoint.
4. HippoRAG retrieves every chunk of a book (`num_to_retrieve`), only so the gold rank is
   exact; its default depth of 200 would cut the larger books. Query-time LLM cost is the
   book's server counters divided evenly over its queries.
5. LQCA: segments of 8 chunks; the Maverick checkpoint from Hugging Face on a GPU (the code
   hard-codes the author's path and the CPU); a failing segment keeps its text and is counted.
   The B4 venv pins torch 2.5.1 and transformers 4.45.2 (maverick-coref's checkpoint does not
   load under torch 2.6 or later). LQCA's requirements.txt names `maverick`, which on PyPI is a
   poker library; the coreference package is `maverick-coref`. The official replacement step
   sometimes glues a replaced mention onto the previous word ("But I" became "Butme" locally);
   that output is kept as is.
6. Link formation is counted by case-insensitive string match, as specified: B2 context holds
   form 2, B3 a triple holds both names, B4 the rewritten gold chunk holds form 2.
7. SAADI's numbers come from each run's per-persona file (CIs and paired tests); when that file
   is absent, from its `results/method_update/*/table.csv` (aggregates only), and the report says so.
