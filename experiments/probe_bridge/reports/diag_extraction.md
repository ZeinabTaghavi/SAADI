# Bridge probe - extraction diagnostic (after step 6)

Read-only. Cache: `/mnt/data/zeinab/SAADI/outputs/probe_bridge/novelhopqa_probe/indexes/59bb6acf00d222d8/llm_cache/novelhopqa/hierarchical/branching_factor_5/chunk_500/AE_5_2_3/Qwen3-30B-A3B-Instruct-2507/AEWord/index.json`. Word match, case-insensitive. Budget = 5 records per chunk.

## Form 1 at the gold chunk

| cell | n | in model's raw reply | kept as a record | anchor not found | budget full (5) | anywhere in final memory |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 30 | 22/30 (73%) | 15/30 (50%) | 12/30 (40%) | 20/30 (67%) | 20/30 (67%) |
| 2 | 30 | 25/30 (83%) | 18/30 (60%) | 15/30 (50%) | 16/30 (53%) | 21/30 (70%) |
| 3 | 30 | 22/30 (73%) | 14/30 (47%) | 13/30 (43%) | 20/30 (67%) | 16/30 (53%) |
| 4 | 30 | 22/30 (73%) | 12/30 (40%) | 12/30 (40%) | 18/30 (60%) | 15/30 (50%) |
| 5 | 30 | 24/30 (80%) | 15/30 (50%) | 13/30 (43%) | 19/30 (63%) | 15/30 (50%) |
| 6 | 30 | 17/30 (57%) | 6/30 (20%) | 5/30 (17%) | 23/30 (77%) | 12/30 (40%) |
| 7 | 30 | 21/30 (70%) | 13/30 (43%) | 12/30 (40%) | 16/30 (53%) | 23/30 (77%) |
| 8 | 30 | 17/30 (57%) | 4/30 (13%) | 3/30 (10%) | 16/30 (53%) | 11/30 (37%) |

## Form 2 at the reveal chunk

| cell | n | in model's raw reply | kept as a record | anchor not found | budget full (5) | anywhere in final memory |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 30 | 22/30 (73%) | 12/30 (40%) | 8/30 (27%) | 19/30 (63%) | 12/30 (40%) |
| 2 | 30 | 22/30 (73%) | 12/30 (40%) | 6/30 (20%) | 15/30 (50%) | 12/30 (40%) |
| 3 | 30 | 17/30 (57%) | 6/30 (20%) | 3/30 (10%) | 19/30 (63%) | 6/30 (20%) |
| 4 | 30 | 15/30 (50%) | 7/30 (23%) | 6/30 (20%) | 18/30 (60%) | 7/30 (23%) |
| 5 | 0 | - | - | - | - | - |
| 6 | 30 | 23/30 (77%) | 8/30 (27%) | 6/30 (20%) | 18/30 (60%) | 8/30 (27%) |
| 7 | 30 | 23/30 (77%) | 15/30 (50%) | 13/30 (43%) | 14/30 (47%) | 15/30 (50%) |
| 8 | 30 | 20/30 (67%) | 11/30 (37%) | 8/30 (27%) | 16/30 (53%) | 11/30 (37%) |

## Reading

- Gold chunks where the model did not mention form 1 and the 5 slots were full: 35/240.
- 'in raw reply' but not 'kept' means the pipeline dropped it after the model wrote it (anchor not found under drop, or budget trim).
