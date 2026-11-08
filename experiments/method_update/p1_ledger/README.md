# p1_ledger: candidate-first ledger (method update tracker, run 1)

Fix for Issue 1 of the SAADI method update tracker (minor entities never recorded,
cut by the 5-record budget, or not recalled), evaluated on the persona probe of
`experiments/probe_bridge`. Reference: Google Doc "SAADI_updates", tab
"3. Method update tracker" > "3.1 Issue 1: Solution" (the specification) and
"3.1 Issue 1: Report" (where the final text goes).

The design in one paragraph: before each LLM call a small open-type NER model
(the candidate net, GLiNER) lists every named span of the chunk; every record or
stub whose name or alias matches a candidate (after title stripping) is recalled
into the prompt in a second group outside the 5 overlap slots (cap 5, cap hits
logged); the LLM must give one decision per candidate (`NEW` | `SAME AS <id>
(<reason>)` | `NOT AN ENTITY`) and then describes at most 5 records as today; a
NEW candidate that is not described becomes a stub (name, chunk, one sentence);
a stub matched by SAME AS is promoted (its sentence becomes its first
description); chunks are linked to every NEW or SAME AS name. Query time is
unchanged. Code: `src/saadi/ledger/`; profile: `configs/profiles/p1_ledger.yaml`;
the hierarchical strategy branches to the ledger only when `strategy_kwargs.ledger`
is set (`tests/ledger/test_published_unchanged.py` proves the published path is
byte-identical).

## How to run (on the GPU server, one step at a time, from anywhere)

```
bash experiments/method_update/p1_ledger/sh/01_gate.sh          # CPU: candidate-net gate, prompt text -> HAND-OFF
bash experiments/method_update/p1_ledger/sh/02_build.sh         # GPUs 0-3: index the 10 books (resumable; rerun to continue) -> HAND-OFF
bash experiments/method_update/p1_ledger/sh/03_memory_check.sh  # CPU -> HAND-OFF (gate for step 4)
bash experiments/method_update/p1_ledger/sh/04_retrieve.sh      # CPU -> HAND-OFF
bash experiments/method_update/p1_ledger/sh/05_report.sh        # CPU: results/ + REPORT_FOR_GDOC.md -> HAND-OFF
```

Each script prints the path of its report as its last line, refuses to run while
the previous step's marker (`outputs/method_update/p1_ledger/markers/`) is
missing, skips a finished step unless `--force`, logs to
`logs/<step>_<timestamp>.log`, and writes its report with the traceback and the
last 50 log lines when it fails (exit code non-zero). No script asks for input.
Step 2 is nohup-safe and resumable per book; `--dry-run` resolves the
configuration without a GPU.

Useful flags: step 1 `--model <hf id>`, `--threshold 0.5`, `--device cpu|cuda`,
`--compare-models a,b`, `--no-second-threshold`, `--candidate-net regex --limit N`
(dry run without a model); step 2 `--dry-run`, `--tensor-parallel-size 4`,
`--gpus 0,1,2,3`.

## Isolation

- GPUs 0-3 only (`CUDA_VISIBLE_DEVICES=0,1,2,3`, tensor parallel 4), own vLLM port
  (`VLLM_PORT`, env.sh), own output root `outputs/method_update/p1_ledger/`
  (study runner output, ledger event logs, markers, gate data), own results root
  `results/method_update/p1_ledger/`. Run 0 under `outputs/probe_bridge/` is read only.
- The study venv is never modified: `sh/01_gate.sh` installs `gliner` with
  `--no-deps` into `outputs/method_update/p1_ledger/pydeps` (on PYTHONPATH via env.sh).
  torch, transformers, sentencepiece and protobuf come from the venv.
- Hugging Face cache: `PROBE_HF_HOME` (default `/mnt/cache/taghavi`, as in the
  probe). Step 1 downloads the candidate net there; steps 2-5 run offline.
- The frozen probe spec is checked (SHA-256) by steps 1 and 2.

## Layout

```
experiments/method_update/p1_ledger/
  env.sh               paths, seeds, GPU ids, vLLM port, venv, model names
  sh/0N_*.sh           wrappers (log + report + marker handling)
  scripts/common.py    paths, markers, StepReport, probe modules loaded by import
  scripts/report_utils.py  bootstrap / McNemar / table helpers
  scripts/step0N_*.py  one file per step
  reports/step_0N.md + .json   what you paste back
  logs/
outputs/method_update/p1_ledger/
  novelhopqa_probe/indexes/<fp>/llm_cache/.../index.json   construction records (run 1)
  ledger/<book>/events.jsonl, ledger.json                  one event per chunk: candidates, recalled groups,
                                                           decisions, ops, prompt text; the ledger snapshot
  ledger/system_prompt.txt, candidate_net.json
  gate/                                                     step 1 candidates and the example prompt
  personas_memory.jsonl, per_persona.jsonl, same_as_decisions.jsonl, title_conflicts.jsonl
results/method_update/p1_ledger/   README.md, table.md, table.csv, REPORT_FOR_GDOC.md
```

## Interpretations recorded for the hand-off (where the spec and the code differ)

1. "then the chunk summary as now" (spec E): the current entity call has no
   per-chunk summary (summaries are the separate fold calls, unchanged); the
   output is the decision block, the extraction JSON, then `[END_JSON]`.
2. Prompt order follows spec E (chunk, history, two recalled groups, candidates),
   not the published order; the system prompt is new text (step 1 prints it).
3. The description cap (5, replace the weakest = fewest content tokens, ties to
   the oldest) lives in the ledger; the construction cache rebuilds memory by
   merging step rows (uncapped), so steps 3-4 apply the ledger's replacements to
   the loaded index before scoring, and step 2 reports any drift in memory.json.
4. A candidate marked NEW whose name is already a record or stub is linked to it
   (a name-keyed memory cannot hold two entries of one name); counted as
   `new_but_exists`. A described record named by a stored alias is kept under
   the stored name (counted as `described_name_was_alias`).
5. Decision-block omissions (a candidate with no line) are treated as NEW and
   counted; an unparseable block makes every candidate a stub (spec C).
6. The overlap group keeps the published display caps (5 records, 2 aliases,
   3 descriptions); the ledger group shows a stub's name and sentence, and a
   record's name, all aliases and up to 3 descriptions ranked by overlap.
7. "False candidates per chunk" has no gold labels for the books' own names;
   step 1 reports a conservative proxy (common words, bare titles, no letters)
   and the raw count per chunk.
8. Ids in `SAME AS <id>` are ledger ids (E0012); a name instead of an id is
   resolved against stored names and aliases; an unresolvable target counts as
   `same_as_unresolved` and the candidate is treated as NEW.
