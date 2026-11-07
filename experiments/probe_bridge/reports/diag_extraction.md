# Bridge probe - extraction diagnostic (after step 6)

Read-only. Cache: `/mnt/data/zeinab/SAADI/outputs/probe_bridge/novelhopqa_probe/indexes/7f585bd63886659e/llm_cache/novelhopqa/hierarchical/branching_factor_5/chunk_500/AE_5_2_3/Qwen3-30B-A3B-Instruct-2507/AEWord/index.json`. Word match, case-insensitive. Budget = 5 records per chunk.

## Form 1 at the gold chunk

| cell | n | in model's raw reply | kept as a record | rejected by anchor check | budget full (5) | anywhere in final memory |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 30 | 23/30 (77%) | 2/30 (7%) | 13/30 (43%) | 0/30 (0%) | 7/30 (23%) |
| 2 | 30 | 26/30 (87%) | 5/30 (17%) | 15/30 (50%) | 0/30 (0%) | 9/30 (30%) |
| 3 | 30 | 23/30 (77%) | 7/30 (23%) | 11/30 (37%) | 0/30 (0%) | 7/30 (23%) |
| 4 | 30 | 22/30 (73%) | 3/30 (10%) | 11/30 (37%) | 0/30 (0%) | 6/30 (20%) |
| 5 | 30 | 24/30 (80%) | 4/30 (13%) | 17/30 (57%) | 0/30 (0%) | 4/30 (13%) |
| 6 | 30 | 19/30 (63%) | 3/30 (10%) | 4/30 (13%) | 0/30 (0%) | 5/30 (17%) |
| 7 | 30 | 20/30 (67%) | 4/30 (13%) | 9/30 (30%) | 0/30 (0%) | 8/30 (27%) |
| 8 | 30 | 19/30 (63%) | 0/30 (0%) | 5/30 (17%) | 1/30 (3%) | 4/30 (13%) |

## Form 2 at the reveal chunk

| cell | n | in model's raw reply | kept as a record | rejected by anchor check | budget full (5) | anywhere in final memory |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 30 | 23/30 (77%) | 5/30 (17%) | 6/30 (20%) | 0/30 (0%) | 5/30 (17%) |
| 2 | 30 | 19/30 (63%) | 4/30 (13%) | 5/30 (17%) | 0/30 (0%) | 4/30 (13%) |
| 3 | 30 | 17/30 (57%) | 1/30 (3%) | 7/30 (23%) | 0/30 (0%) | 1/30 (3%) |
| 4 | 30 | 17/30 (57%) | 3/30 (10%) | 5/30 (17%) | 0/30 (0%) | 3/30 (10%) |
| 5 | 0 | - | - | - | - | - |
| 6 | 30 | 25/30 (83%) | 3/30 (10%) | 5/30 (17%) | 0/30 (0%) | 3/30 (10%) |
| 7 | 30 | 25/30 (83%) | 5/30 (17%) | 15/30 (50%) | 0/30 (0%) | 5/30 (17%) |
| 8 | 30 | 21/30 (70%) | 4/30 (13%) | 6/30 (20%) | 0/30 (0%) | 4/30 (13%) |

## Reading

- Gold chunks where the model did not mention form 1 and the 5 slots were full: 1/240.
- 'in raw reply' but not 'kept' means the pipeline dropped it after the model wrote it (anchor check or budget trim).
