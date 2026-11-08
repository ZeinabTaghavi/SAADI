# Bridge probe - step 06: Memory check

Status: **ok** - 2026-11-08 09:40:54

## What was done

- `bash experiments/probe_bridge/sh/06_*.sh --force`
- `inspect /mnt/data/zeinab/SAADI/outputs/probe_bridge/novelhopqa_probe/indexes/59bb6acf00d222d8/llm_cache/novelhopqa/hierarchical/branching_factor_5/chunk_500/AE_5_2_3/Qwen3-30B-A3B-Instruct-2507/AEWord/index.json with the E2 replay (saadi.rebuttal.frozen_records)`
- Versions: python=3.12.3, platform=Linux-6.17.0-23-generic-x86_64-with-glibc2.39, numpy=2.3.5, pandas=3.0.1, yaml=6.0.3, saadi=0.1.0, git_commit=15baade03770a80c68a61e02b2e705caaaf35b20

## Numbers

### Memory outcomes per cell

| cell | n | form1 record | form2 in final record | gold chunk linked | form2 in gold FROZEN record | merge directions |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 30 | 18/30 (60%) | 6/30 (20%) | 12/30 (40%) | 0/30 (0%) | {'form1_into_form2': 6, 'form2_into_form1': 6, 'never_joined': 7, 'no_record': 11} |
| 2 | 30 | 19/30 (63%) | 3/30 (10%) | 15/30 (50%) | 0/30 (0%) | {'form1_into_form2': 7, 'form2_into_form1': 3, 'never_joined': 13, 'no_record': 7} |
| 3 | 30 | 14/30 (47%) | 4/30 (13%) | 10/30 (33%) | 0/30 (0%) | {'form1_into_form2': 2, 'form2_into_form1': 4, 'never_joined': 9, 'no_record': 15} |
| 4 | 30 | 14/30 (47%) | 5/30 (17%) | 9/30 (30%) | 0/30 (0%) | {'form1_into_form2': 1, 'form2_into_form1': 5, 'never_joined': 10, 'no_record': 14} |
| 5 | 30 | 15/30 (50%) | - | 15/30 (50%) | 0/30 (0%) | {'never_joined': 15, 'no_record': 15} |
| 6 | 30 | 9/30 (30%) | 3/30 (10%) | 6/30 (20%) | 0/30 (0%) | {'form1_into_form2': 5, 'form2_into_form1': 3, 'never_joined': 6, 'no_record': 16} |
| 7 | 30 | 23/30 (77%) | 15/30 (50%) | 13/30 (43%) | 0/30 (0%) | {'form2_into_form1': 15, 'never_joined': 8, 'no_record': 7} |
| 8 | 30 | 10/30 (33%) | 3/30 (10%) | 6/30 (20%) | 0/30 (0%) | {'form1_into_form2': 8, 'form2_into_form1': 3, 'never_joined': 5, 'no_record': 14} |

### Where form 2 sits in the form-1 record

| cell | name | alias | description | none |
| --- | --- | --- | --- | --- |
| 1 | 1 | 4 | 1 | 24 |
| 2 | 0 | 3 | 0 | 27 |
| 3 | 0 | 4 | 0 | 26 |
| 4 | 2 | 3 | 0 | 25 |
| 5 | 0 | 0 | 0 | 0 |
| 6 | 2 | 1 | 0 | 27 |
| 7 | 0 | 1 | 14 | 15 |
| 8 | 0 | 0 | 3 | 27 |

### Gate

The bridge word reached the gold chunk's record in 1 of 90 personas of cells 1-3 (1%). **That is below half: the weak point is extraction, and the retrieval numbers of step 7 must be read with that in mind.**

### Parser status of chunks holding inserted sentences

{'fallback': 447, 'ok': 3}

### Chunk records per book (growth check is in step 4's table)

| book | chunk records |
| --- | --- |
| B31 | 244 |
| B43 | 246 |
| B37 | 262 |
| B18 | 312 |
| B04 | 321 |
| B34 | 335 |
| B27 | 338 |
| B61 | 357 |
| B26 | 384 |
| B44 | 385 |

### Five personas in full (seeded draw)

- **c8_B31_000** (cell 8): form1 record=None, form2 field=none, merge=form1_into_form2, arrival step=None, gold step=98, gold linked=False, frozen has form2=False
- **c3_B04_013** (cell 3): form1 record='Mehrparvar', form2 field=alias, merge=form2_into_form1, arrival step=261, gold step=85, gold linked=False, frozen has form2=False
- **c3_B04_014** (cell 3): form1 record=None, form2 field=none, merge=no_record, arrival step=None, gold step=92, gold linked=False, frozen has form2=False
- **c4_B27_020** (cell 4): form1 record=None, form2 field=none, merge=no_record, arrival step=None, gold step=183, gold linked=False, frozen has form2=False
- **c6_B61_021** (cell 6): form1 record=None, form2 field=none, merge=form1_into_form2, arrival step=None, gold step=137, gold linked=False, frozen has form2=False

## Anything unexpected

Nothing unexpected.

## Files written

- `/mnt/data/zeinab/SAADI/outputs/probe_bridge/personas_memory.jsonl`
