# Bridge probe - step 06: Memory check

Status: **ok** - 2026-11-07 23:45:07

## What was done

- `bash experiments/probe_bridge/sh/06_*.sh`
- `inspect /mnt/data/zeinab/SAADI/outputs/probe_bridge/novelhopqa_probe/indexes/7f585bd63886659e/llm_cache/novelhopqa/hierarchical/branching_factor_5/chunk_500/AE_5_2_3/Qwen3-30B-A3B-Instruct-2507/AEWord/index.json with the E2 replay (saadi.rebuttal.frozen_records)`
- Versions: python=3.12.3, platform=Linux-6.17.0-23-generic-x86_64-with-glibc2.39, numpy=2.3.5, pandas=3.0.1, yaml=6.0.3, saadi=0.1.0, git_commit=4f54773d8ccdd8aeb378b12a4d70e7255d6af8ba

## Numbers

### Memory outcomes per cell

| cell | n | form1 record | form2 in final record | gold chunk linked | form2 in gold FROZEN record | merge directions |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 30 | 4/30 (13%) | 3/30 (10%) | 1/30 (3%) | 0/30 (0%) | {'form1_into_form2': 2, 'form2_into_form1': 3, 'never_joined': 1, 'no_record': 24} |
| 2 | 30 | 5/30 (17%) | 2/30 (7%) | 3/30 (10%) | 0/30 (0%) | {'form1_into_form2': 2, 'form2_into_form1': 2, 'never_joined': 3, 'no_record': 23} |
| 3 | 30 | 4/30 (13%) | 1/30 (3%) | 3/30 (10%) | 0/30 (0%) | {'form2_into_form1': 1, 'never_joined': 3, 'no_record': 26} |
| 4 | 30 | 4/30 (13%) | 3/30 (10%) | 1/30 (3%) | 0/30 (0%) | {'form2_into_form1': 3, 'never_joined': 1, 'no_record': 26} |
| 5 | 30 | 1/30 (3%) | - | 1/30 (3%) | 0/30 (0%) | {'never_joined': 1, 'no_record': 29} |
| 6 | 30 | 6/30 (20%) | 2/30 (7%) | 4/30 (13%) | 0/30 (0%) | {'form1_into_form2': 1, 'form2_into_form1': 2, 'never_joined': 4, 'no_record': 23} |
| 7 | 30 | 8/30 (27%) | 5/30 (17%) | 3/30 (10%) | 0/30 (0%) | {'form2_into_form1': 5, 'never_joined': 3, 'no_record': 22} |
| 8 | 30 | 4/30 (13%) | 2/30 (7%) | 2/30 (7%) | 0/30 (0%) | {'form1_into_form2': 2, 'form2_into_form1': 2, 'never_joined': 2, 'no_record': 24} |

### Where form 2 sits in the form-1 record

| cell | name | alias | description | none |
| --- | --- | --- | --- | --- |
| 1 | 1 | 1 | 1 | 27 |
| 2 | 1 | 1 | 0 | 28 |
| 3 | 0 | 1 | 0 | 29 |
| 4 | 1 | 2 | 0 | 27 |
| 5 | 0 | 0 | 0 | 0 |
| 6 | 2 | 0 | 0 | 28 |
| 7 | 0 | 0 | 5 | 25 |
| 8 | 1 | 0 | 1 | 28 |

### Gate

The bridge word reached the gold chunk's record in 0 of 90 personas of cells 1-3 (0%). **That is below half: the weak point is extraction, and the retrieval numbers of step 7 must be read with that in mind.**

### Parser status of chunks holding inserted sentences

{'fallback': 446, 'ok': 4}

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

- **c8_B31_000** (cell 8): form1 record=None, form2 field=none, merge=no_record, arrival step=None, gold step=98, gold linked=False, frozen has form2=False
- **c3_B04_013** (cell 3): form1 record='Madame Mehrparvar', form2 field=none, merge=never_joined, arrival step=None, gold step=85, gold linked=True, frozen has form2=False
- **c3_B04_014** (cell 3): form1 record=None, form2 field=none, merge=no_record, arrival step=None, gold step=92, gold linked=False, frozen has form2=False
- **c4_B27_020** (cell 4): form1 record=None, form2 field=none, merge=no_record, arrival step=None, gold step=183, gold linked=False, frozen has form2=False
- **c6_B61_021** (cell 6): form1 record=None, form2 field=none, merge=no_record, arrival step=None, gold step=137, gold linked=False, frozen has form2=False

## Anything unexpected

Nothing unexpected.

## Files written

- `/mnt/data/zeinab/SAADI/outputs/probe_bridge/personas_memory.jsonl`
