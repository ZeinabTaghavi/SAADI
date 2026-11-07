# Bridge probe

A controlled experiment for the SAADI rebuttal: invented people are inserted
into ten NovelHopQA books so that a query can reach its gold chunk **only**
through a record field that arrives at a later step. It measures whether
SAADI's record updates make early chunks retrievable (the mechanism E2 could
not observe on the standard benchmarks, and E2b showed they barely contain).

The full design is in the note "Bridge Probe: Implementation Note". This folder
is self-contained: it reads the repository's code, data and caches, and writes
only here, to `outputs/probe_bridge/` and to `results/probe_bridge/` — plus one
declared config file, `configs/experiments/novelhopqa_probe_retrieval_ablation.yaml`
(the runner cannot read configs anywhere else; written by step 5).

## How to run (one step at a time, from the repository root)

```
bash experiments/probe_bridge/sh/01_select_books.sh
bash experiments/probe_bridge/sh/02_generate_personas.sh
bash experiments/probe_bridge/sh/03_check_and_place.sh
bash experiments/probe_bridge/sh/04_insert_and_rechunk.sh        # freezes the spec
bash experiments/probe_bridge/sh/05_index.sh [--tensor-parallel-size N] [--gpus 0,1,2,3]   # the GPU step
bash experiments/probe_bridge/sh/06_memory_check.sh
bash experiments/probe_bridge/sh/07_retrieve_and_score.sh
bash experiments/probe_bridge/sh/08_report.sh
```

Each script prints, as its **last line**, the path of its report
(`reports/step_NN.md`). Paste that report back before running the next step
(steps 1, 3, 4, 5, 6, 7 and 8 are hand-offs). A finished step is skipped on
rerun unless `--force` is given; every script refuses to run while the previous
step's marker (`outputs/probe_bridge/markers/`) is missing. Step 5 is resumable:
rerun it and construction continues from its cache (`--dry-run` resolves the
configuration without a GPU). A failing script still writes its report, with the
traceback and the last 50 log lines, and exits non-zero.

Requirements on the server: the study virtual environment (`.venv`), the
NovelHopQA book corpus at `$NOVELHOPQA_BOOKS_ROOT` (see
docs/data_and_artifacts.md), the reference bundle at `outputs/saadi/reference/`
(needed by `--profile published`), and GPUs only for step 5.

## Layout

```
experiments/probe_bridge/
  banks.py          fixed name banks, templates, phrasings (written in advance)
  common.py         markers, reports, tokenizer+stopwords, chunking, shared scoring
  steps/stepNN_*.py the eight steps
  sh/NN_*.sh        their wrappers (log + report + marker handling)
  spec/             books.json, templates.json, personas.jsonl, spec_frozen.json
  data/             per-book original/modified text, insertions.json, documents.json, qa.json
  reports/          step_NN.md + step_NN.json (what you paste back)
  logs/
outputs/probe_bridge/   caches, construction records, per-persona results, markers
results/probe_bridge/   README.md, table.md, table.csv, figures (step 8)
```

## Known deviations from the note (each is also flagged in the step reports)

1. **Config location.** The runner only accepts
   `configs/experiments/<dataset>_retrieval_ablation.yaml`, so the declared file
   is `novelhopqa_probe_retrieval_ablation.yaml` there, not
   `configs/probe_bridge/novelhopqa_probe.yaml`.
2. **Stopwords.** The repository's BM25 tokenizer has no stopword list; the
   probe uses the fixed list in `common.py` (the same as E2b).
3. **Family D reveals are kind-free** ("Mitra owned the Blue Heron."): the
   note's example reveal carries the query's kind word ("boat") and would fail
   its own check 5. Family C phrasings avoid "living"/"lived" for the same
   reason (check 4 on stems).
4. **Cell 8 rotates the six object kinds** instead of the five fact types; the
   note's family D defines only the Home-shaped gold.
5. **Inserted sentences are padded to 8-18 words** (the note's own example
   phrasings are shorter than its style rule).
6. **Geometry v2 (decided at the step-1 hand-off).** The 10 shortest books
   (213-383 chunks) cannot hold the note's gold window [0.10N, 0.30N] with 24
   sentences three chunks apart. The gold window is therefore [0.05N, 0.45N];
   middle-reveal cells (2/6/7/8) keep their gold above 0.20N so the reveal
   clears the window, cell 3 stays at or below 0.35N so the far reveal stays
   under 0.95N, and cell-4 reveals take slots in (0.25N, 0.35N] so their golds
   land in (0.45N, 0.55N] - all inside the note's cell-4 windows. Books under
   240 chunks are skipped for the next-shortest feasible ones (B24 -> B44).
   Placement verified by simulation at the real book sizes over five seeds.
