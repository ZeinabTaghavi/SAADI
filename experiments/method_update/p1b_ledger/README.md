# p1b_ledger: run 1b of the method update tracker

`p1b_ledger` = `p1_ledger` plus exactly three changes, aimed at the three causes run 1
exposed (Google Doc "SAADI_updates", sub-tab "3.1 Issue 1: Report"):

1. **Join direction, enforced in code.** An accepted SAME AS attaches the candidate as an
   alias of the target and never creates a record. A JSON record whose title-stripped name
   equals a SAME AS candidate is redirected into the target (descriptions merged, no
   record created; counted "redirected"). A NEW candidate whose title-stripped name equals
   the name or an alias of a ledger entry shown in the prompt, with no title or type
   conflict, becomes SAME AS that entry (counted and logged "auto-join").
2. **Recall at the reveal.** Ledger recall matches every ledger name and alias (title
   stripped, tokens of 3+ characters, no stopwords) against the chunk text, independent of
   the candidate net; the net now runs at threshold 0.3. Cap 8 instead of 5; priority full
   name, then alias, then single token; ties by the most recent mention. Every cap hit and
   the entries left out are logged.
3. **SAME AS acceptance rule (alias hygiene).** Accepted only with a shared name token or
   both names in one sentence of the chunk; rejected on a title conflict on the same
   stripped name or a person against place/organization type conflict (candidate-net labels
   stored per entry). A rejected SAME AS becomes NEW and is logged with its reason. Aliases
   stay uncapped, but every alias must be a candidate-net span or its title-stripped form.

Code: `src/saadi/ledger/p1b.py` (rules), hooks in `promotion.py` and `runtime.py` that only
run when the p1b keys are set. Profile: `configs/profiles/p1b_ledger.yaml`. The construction
LLM (Qwen3-30B-A3B-Instruct-2507, temperature 0, seed 42, same server settings, one call
per chunk), the candidate model (urchade/gliner_large-v2.1), the prompt text, the budget
and the fusion weights (1.0, 0.5, 0.3) are unchanged.

## Open question before the run (json_alias_join)

Run 1's 25 wrong-direction joins (cells 1-3) are, by the probe's own classification,
cases where form 2 did **not** become an alias of the form-1 record: the model answered
NEW for form 2 and wrote a separate record holding form 1 (as alias, description or in the
name). Change 1 as specified acts only when form 2 was answered SAME AS (redirect) or when
a NEW name equals a shown name (auto-join), so it does not reach that case; better recall
(change 2) can still help by showing the form-1 record more often.

`ledger.json_alias_join` (off in the profile, so the run is the specification as written)
adds one sub-rule to change 1: a NEW candidate whose own JSON record lists a shown entry's
name or alias as an alias is joined into that entry, if rule 3 accepts it (the probe's
reveal "Maysunah's family name was Sinchipuma" passes by the same-sentence rule). It does
not cover form 1 written as a description or inside a full name. Before starting, the
read-only check below shows how run 1's wrong joins split:

```
bash -c 'source experiments/method_update/p1b_ledger/env.sh && python experiments/method_update/p1b_ledger/scripts/check_run1_joins.py'
```

To switch it on, set `json_alias_join: true` in `configs/profiles/p1b_ledger.yaml` (the
profile test then needs the same value in `P1B_KEYS`).

## How to run (on the GPU server, one command, unattended)

```
nohup bash experiments/method_update/p1b_ledger/run_all.sh > experiments/method_update/p1b_ledger/logs/run_all_nohup.out 2>&1 &
```

Stages, each with a report `reports/step_0N.md` (+ `.json`) and a done marker in
`outputs/method_update/p1b_ledger/markers/`:

| stage | where | what |
| --- | --- | --- |
| 01_gate | CPU | candidate net at 0.3 on the gold and reveal chunks: form-1/form-2 recall per family, false candidates per chunk, next to run 1's 0.5; the prompt with one example. Reports only, never blocks. |
| 02_build | GPUs 0-3 | the 10 books with the p1b profile (resumable; retried once automatically); calls per book against run 1, tokens, records, stubs, cap hits, wall clock, GLiNER time, the counts of the three changes |
| 03_memory_check | CPU | per cell, run 1b / run 1 / run 0; why the earlier record was missing; join direction; form 2 at the reveal; merge audit |
| 04_retrieve | CPU | HR@5, HR@10, median rank per cell for Vanilla, No memory, Frozen, Live; live minus frozen with bootstrap interval and McNemar; controls; fusion landscape (information only) |
| 05_report | CPU | `reports/FINAL.md`, `results/method_update/p1b_ledger/REPORT_FOR_GDOC.md`, `TRACKER_ROW_1b.md`, `table.md`, `table.csv` |

`--from <stage>` restarts at a stage (earlier stages must have finished); `--force`
reruns finished stages from there on. A failing stage writes its report with the
traceback and the last 50 log lines, later stages do not run, and `reports/FINAL.md`
says where it stopped. The last line printed is always the path of `reports/FINAL.md`.
Log: `logs/run_all_<timestamp>.log`. Stage 02 alone can be re-reported without
starting vLLM: `python scripts/stage02_build.py --force --report-only` (after `source env.sh`).

## Isolation

- GPUs 0-3 only (`CUDA_VISIBLE_DEVICES=0,1,2,3`, tensor parallel 4). vLLM port 29537
  (run 1 used 29517); set `P1B_VLLM_PORT` if the build on GPUs 4-7 uses 29537.
- Writes only to `experiments/method_update/p1b_ledger/{reports,logs}`,
  `outputs/method_update/p1b_ledger/`, `results/method_update/p1b_ledger/`;
  `scripts/common_p1b.py` refuses to start if any output path points into run 1 or run 0.
- Run 1 (`experiments|outputs|results/method_update/p1_ledger`) and run 0
  (`outputs/probe_bridge`) are read only. Run 1's measurement code is imported unchanged
  (its `common.py` takes its paths from the `P1_*` variables that `env.sh` points here).
- `gliner` comes from run 1's pydeps folder (read only) or, if missing, is installed into
  `outputs/method_update/p1b_ledger/pydeps` with `--no-deps`. The venv is never modified.
  The candidate net loads offline from the shared cache when it is there (same weights as run 1).
- The frozen probe spec is checked (SHA-256) by stages 01 and 02.

## Interpretations recorded (where the prompt left room)

1. The study runner accepts only `base: published`, so the profile repeats p1's ledger
   keys; `tests/ledger/test_p1b_rules.py` asserts it differs from `p1_ledger.yaml` only in
   the seven keys of the three changes (and `store_dir`, which resolves to its own root).
2. The construction records keep the strategy name `hierarchical_summary_p1_ledger`: the
   name is set by the hook in `context_strategies/hierarchical_summary.py`, outside
   `src/saadi/ledger/`. Output root and cache fingerprint are run 1b's own.
3. The system prompt is byte-identical to run 1's; it still describes the second recalled
   group as entries "whose stored name or alias matches a candidate", now matched by chunk
   tokens.
4. Recall classes: "full name" = every recall token of the stored name occurs in the
   chunk (as a token set, not a contiguous phrase); "alias" = the same for an alias;
   "single token" = some but not all tokens. "Most recent mention" = the entry's last
   linked, described or aliased step.
5. Rule 3 (a) uses title-stripped, stopword-free name tokens; (b) needs one sentence of
   the chunk holding every stripped token of the candidate and of one stored name or alias.
   Conflicts are checked first and reject even with a shared token. Title conflict: one
   stripped name contains the other, both carry titles, and the titles differ (Dr./Doctor,
   Capt./Captain and similar count as one). Type conflict: person against place or
   organization only, as specified (person against named object is not a conflict).
   Labels come from the candidate span that created or first matched an entry; records the
   LLM added on its own carry no label and never type-conflict.
6. A rejected SAME AS becomes NEW, then goes through the auto-join check (the rejected
   target excluded) and the normal NEW path (stub, or the described record).
7. Auto-join applies to explicit NEW decisions and to rejected or unresolved SAME AS, not
   to omitted or parse-failure candidates (the parse-failure policy stays: all stubs).
   Two matching shown entries: no join, counted "ambiguous".
8. In run 1, a NEW candidate could be folded into a described JSON record that is an
   existing entry, which joins it without a SAME AS; p1b applies the title and type
   conflict check there too.
9. Alias hygiene covers every alias written to memory in the step (JSON aliases, the old
   name of a redirected record, candidate aliases): a span of the chunk or its
   title-stripped form, with no title conflict against the record's names and no type
   conflict against its label. Ledger aliases are candidate texts only, as in run 1.
10. Stage 01 never blocks: neither its numbers nor a failure stop the build.
11. "Wrong SAME AS in the audit sample" is a reading of the 30 sampled decisions; the code
    lists them, it cannot judge them.
12. Run 0 columns use the proxies of run 1's step 3; run 0's token counts are reused from
    run 1's step 2 report rather than rebuilt.
