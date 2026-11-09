# Run 1b (profile p1b_ledger): final report

Written 2026-11-09 00:43 by stage 05 from the stage reports in this folder. Commit `28eed3a1ee31487f3bf23dcb3470aa0e164e8cfd`. Stage status: {'01_gate': 'ok', '02_build': 'ok', '03_memory_check': 'ok', '04_retrieve': 'ok'}.
Every number is run 1b first, then run 1, then run 0 where it exists. Nothing was tuned; fusion weights (1.0, 0.5, 0.3) fixed.
Optional JSON-alias join: **off**.

## Expected against observed (expectations written before the run)

| Quantity | Expected (written before the run) | Observed | Verdict |
| --- | --- | --- | --- |
| Bridge formed, cells 1-3 | from 14 of 90 to above 40 of 90 | 24/90 (run 1: 14/90; run 0: 1/90) | not held |
| Join into the gold record (cells 1-3) | from 24 of 49 joins to most joins | 34 of 52 joins into the gold record, 18 into a new form-2 record (run 1: 24 of 49) | held |
| Earlier record in the prompt at the reveal | from 82 of 159 to above 130 of 159 | 121/174 (run 1: 82/159) | not held |
| Wrong SAME AS in the audit sample of 30 | from 7-8 of 30 to under 2 of 30 | to be read: the 30 sampled decisions are listed below; the code cannot judge a merge | to be read |
| Retrieval (live HR@10, cells 1-3) | probably still flat at fixed weights; the fusion landscape shows whether the bridges would be retrievable at a higher memory weight | live 6% against frozen 3% (run 1: 6% / 3%); live minus frozen +2.2 pp [-2.2, +6.7], McNemar p=0.625 | flat |
| Cell 5 (no reveal) | stays at zero | zero | held |
| Calls per book | unchanged (equal to run 1) | equal | held |

## Memory check per cell (run 1b / run 1 / run 0)

| cell | n | form 1 after gold | form 2 after reveal | both present (final) | earlier record in prompt at the reveal | joined when both present | bridge formed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 (A, reveal 3 chunks after the gold) | 30 | 27/30 (90%) / 26/30 (87%) / 18/30 (60%) | 24/30 (80%) / 23/30 (77%) / 6/30 (20%) | 24/30 (80%) / 22/30 (73%) / 19/30 (63%) | 27/30 (90%) / 16/30 (53%) / 0/30 (0%) | 20/30 (67%) / 17/30 (57%) / 12/30 (40%) | 13/30 (43%) / 4/30 (13%) / 1/30 (3%) |
| 2 (A, reveal 25% of the book after) | 30 | 27/30 (90%) / 27/30 (90%) / 19/30 (63%) | 24/30 (80%) / 23/30 (77%) / 3/30 (10%) | 24/30 (80%) / 21/30 (70%) / 23/30 (77%) | 24/30 (80%) / 15/30 (50%) / 1/30 (3%) | 17/30 (57%) / 14/30 (47%) / 10/30 (33%) | 8/30 (27%) / 5/30 (17%) / 0/30 (0%) |
| 3 (A, reveal 55% of the book after) | 30 | 29/30 (97%) / 28/30 (93%) / 14/30 (47%) | 24/30 (80%) / 23/30 (77%) / 4/30 (13%) | 24/30 (80%) / 22/30 (73%) / 15/30 (50%) | 17/30 (57%) / 10/30 (33%) / 1/30 (3%) | 14/30 (47%) / 15/30 (50%) / 6/30 (20%) | 3/30 (10%) / 5/30 (17%) / 0/30 (0%) |
| 4 (A, reveal 20% before the gold (control)) | 30 | 28/30 (93%) / 26/30 (87%) / 14/30 (47%) | 21/30 (70%) / 21/30 (70%) / 5/30 (17%) | 21/30 (70%) / 19/30 (63%) / 16/30 (53%) | 0/30 (0%) / 3/30 (10%) / 2/30 (7%) | 13/30 (43%) / 16/30 (53%) / 6/30 (20%) | 8/30 (27%) / 1/30 (3%) / 0/30 (0%) |
| 5 (A, no reveal (control)) | 30 | 27/30 (90%) / 25/30 (83%) / 15/30 (50%) | - | - | - | - | - |
| 6 (B, nickname, reveal 25% after) | 30 | 23/30 (77%) / 16/30 (53%) / 9/30 (30%) | 26/30 (87%) / 22/30 (73%) / 3/30 (10%) | 22/30 (73%) / 16/30 (53%) / 14/30 (47%) | 18/30 (60%) / 9/30 (30%) / 0/30 (0%) | 18/30 (60%) / 13/30 (43%) / 8/30 (27%) | 7/30 (23%) / 4/30 (13%) / 0/30 (0%) |
| 7 (C, description, reveal 25% after) | 30 | 25/30 (83%) / 25/30 (83%) / 23/30 (77%) | 0/30 (0%) / 0/30 (0%) / 15/30 (50%) | 0/30 (0%) / 0/30 (0%) / 23/30 (77%) | 20/30 (67%) / 18/30 (60%) / 2/30 (7%) | 0/30 (0%) / 0/30 (0%) / 15/30 (50%) | 17/30 (57%) / 20/30 (67%) / 5/30 (17%) |
| 8 (D, object, reveal 25% after) | 30 | 22/30 (73%) / 16/30 (53%) / 10/30 (33%) | 26/30 (87%) / 24/30 (80%) / 3/30 (10%) | 24/30 (80%) / 21/30 (70%) / 16/30 (53%) | 15/30 (50%) / 11/30 (37%) / 2/30 (7%) | 24/30 (80%) / 15/30 (50%) / 11/30 (37%) | 11/30 (37%) / 3/30 (10%) / 0/30 (0%) |

### Join direction

| group | into the gold record | into a new form-2 record | never joined | no record |
| --- | --- | --- | --- | --- |
| cells 1-3 | 34 / 24 / 13 | 18 / 25 / 15 | 38 / 39 / 29 | 0 / 2 / 33 |
| cell 1 | 16 / 8 / 6 | 4 / 9 / 6 | 10 / 12 / 7 | 0 / 1 / 11 |
| cell 2 | 11 / 8 / 3 | 6 / 8 / 7 | 13 / 14 / 13 | 0 / 0 / 7 |
| cell 3 | 7 / 8 / 4 | 8 / 8 / 2 | 15 / 13 / 9 | 0 / 1 / 15 |
| cell 4 | 11 / 12 / 5 | 3 / 6 / 1 | 14 / 10 / 10 | 2 / 2 / 14 |
| cell 6 | 9 / 7 / 3 | 10 / 9 / 5 | 10 / 11 / 6 | 1 / 3 / 16 |
| cell 7 | 21 / 24 / 15 | 0 / 0 / 0 | 8 / 6 / 8 | 1 / 0 / 7 |
| cell 8 | 18 / 12 / 3 | 8 / 7 / 8 | 4 / 10 / 5 | 0 / 1 / 14 |

### Earlier record at the link step

| outcome | run 1b | run 1 | run 0 |
| --- | --- | --- | --- |
| in the prompt | 121 | 82 | 8 |
| cut by the cap | 32 | 27 | 0 |
| not matched by recall | 21 | 50 | 71 |
| no earlier record | 36 | 51 | 131 |

### Form 2 at the reveal: decision and outcome

| outcome | run 1b | run 1 |
| --- | --- | --- |
| model NEW -> record | 84 | 80 |
| model NOT_AN_ENTITY -> not_an_entity | 23 | 29 |
| model SAME_AS -> joined (model) into the form-1 record | 17 | 18 |
| model NEW -> stub | 5 | 7 |
| model SAME_AS rejected (no_shared_token_or_sentence) -> record | 11 | 0 |
| form 2 not a candidate | 3 | 7 |
| model SAME_AS -> joined (model) into another record | 0 | 9 |
| model SAME_AS rejected (no_shared_token_or_sentence) -> stub | 4 | 0 |
| model SAME_AS rejected (type_conflict) -> record | 2 | 0 |
| model SAME_AS rejected (type_conflict) -> stub | 1 | 0 |

### Wrong direction: where form 1 sits

| field | run 1b | run 1 |
| --- | --- | --- |
| name | 0 | 0 |
| alias | 5 | 5 |
| description | 34 | 42 |
| none | 0 | 0 |
| record not found | 0 | 0 |

### The three changes at work

| measure | value |
| --- | --- |
| JSON records redirected into a SAME AS target | 2991 |
| auto-join (NEW equal to a shown entry) / blocked by a conflict / ambiguous | 633 / 1099 / 95 |
| JSON-alias join (optional switch) | 0 |
| model SAME AS rejected, by reason | {'no_shared_token_or_sentence': 1288, 'type_conflict': 559, 'title_conflict': 345} |
| model SAME AS accepted, by evidence | {'shared_token': 16164, 'same_sentence': 302} |
| NEW folded into a described record, rejected for a conflict | 292 |
| aliases rejected, by reason | {'not_a_span': 9811, 'title_conflict': 322, 'type_conflict': 36, 'descriptive': 7} |

## Retrieval per cell

### HR@10 (%) (run 1b / run 1 / run 0)

| cell | n | Vanilla | No memory | Frozen | Live |
| --- | --- | --- | --- | --- | --- |
| 1 (A, reveal 3 chunks after the gold) | 30 | 3 / 3 / 3 | 0 / 0 / 0 | 3 / 0 / 0 | 10 / 3 / 3 |
| 2 (A, reveal 25% of the book after) | 30 | 3 / 3 / 3 | 3 / 3 / 3 | 3 / 3 / 3 | 3 / 7 / 3 |
| 3 (A, reveal 55% of the book after) | 30 | 3 / 3 / 3 | 3 / 3 / 3 | 3 / 7 / 3 | 3 / 7 / 3 |
| 4 (A, reveal 20% before the gold (control)) | 30 | 7 / 7 / 7 | 3 / 3 / 3 | 7 / 7 / 3 | 3 / 3 / 3 |
| 5 (A, no reveal (control)) | 30 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| 6 (B, nickname, reveal 25% after) | 30 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 3 / 0 / 0 |
| 7 (C, description, reveal 25% after) | 30 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 3 / 0 | 0 / 3 / 0 |
| 8 (D, object, reveal 25% after) | 30 | 3 / 3 / 3 | 7 / 7 / 3 | 10 / 7 / 3 | 13 / 7 / 7 |

### HR@5 (%)

| cell | n | Vanilla | No memory | Frozen | Live |
| --- | --- | --- | --- | --- | --- |
| 1 (A, reveal 3 chunks after the gold) | 30 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 3 / 3 / 0 |
| 2 (A, reveal 25% of the book after) | 30 | 3 / 3 / 3 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| 3 (A, reveal 55% of the book after) | 30 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| 4 (A, reveal 20% before the gold (control)) | 30 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| 5 (A, no reveal (control)) | 30 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| 6 (B, nickname, reveal 25% after) | 30 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| 7 (C, description, reveal 25% after) | 30 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 3 / 0 |
| 8 (D, object, reveal 25% after) | 30 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 3 | 3 / 0 / 3 |

### Median gold rank

| cell | n | Vanilla | No memory | Frozen | Live |
| --- | --- | --- | --- | --- | --- |
| 1 | 30 | 144 / 144 / 144 | 148 / 149 / 140 | 184 / 156 / 172 | 68 / 115 / 172 |
| 2 | 30 | 124 / 124 / 124 | 138 / 134 / 142 | 155 / 134 / 148 | 82 / 116 / 142 |
| 3 | 30 | 162 / 162 / 162 | 162 / 168 / 175 | 160 / 174 / 216 | 148 / 146 / 200 |
| 4 | 30 | 118 / 118 / 118 | 130 / 104 / 121 | 98 / 118 / 108 | 112 / 120 / 111 |
| 5 | 30 | 152 / 152 / 152 | 142 / 134 / 136 | 134 / 116 / 134 | 122 / 116 / 128 |
| 6 | 30 | 168 / 168 / 168 | 147 / 166 / 137 | 143 / 153 / 156 | 118 / 122 / 146 |
| 7 | 30 | 174 / 174 / 174 | 174 / 197 / 178 | 203 / 182 / 171 | 130 / 111 / 166 |
| 8 | 30 | 128 / 128 / 128 | 113 / 143 / 124 | 98 / 119 / 135 | 95 / 114 / 116 |

### Live minus Frozen @10

| group | n | run 1b | run 1 |
| --- | --- | --- | --- |
| cells 1-3 pooled | 90 | +2.2 pp [-2.2, +6.7], McNemar p=0.625 (3 vs 1) | +2.2 pp [+0.0, +5.6], McNemar p=0.500 (2 vs 0) |
| cell 1 | 30 | +6.7 pp [-6.7, +20.0], McNemar p=0.625 (3 vs 1) | +3.3 pp [+0.0, +10.0], McNemar p=1.000 (1 vs 0) |
| cell 2 | 30 | +0.0 pp [+0.0, +0.0], McNemar p=1.000 (0 vs 0) | +3.3 pp [+0.0, +10.0], McNemar p=1.000 (1 vs 0) |
| cell 3 | 30 | +0.0 pp [+0.0, +0.0], McNemar p=1.000 (0 vs 0) | +0.0 pp [+0.0, +0.0], McNemar p=1.000 (0 vs 0) |
| cell 4 | 30 | -3.3 pp [-10.0, +0.0], McNemar p=1.000 (0 vs 1) | -3.3 pp [-10.0, +0.0], McNemar p=1.000 (0 vs 1) |
| cell 5 | 30 | +0.0 pp [+0.0, +0.0], McNemar p=1.000 (0 vs 0) | +0.0 pp [+0.0, +0.0], McNemar p=1.000 (0 vs 0) |
| cell 6 | 30 | +3.3 pp [+0.0, +10.0], McNemar p=1.000 (1 vs 0) | +0.0 pp [+0.0, +0.0], McNemar p=1.000 (0 vs 0) |
| cell 7 | 30 | +0.0 pp [+0.0, +0.0], McNemar p=1.000 (0 vs 0) | +0.0 pp [+0.0, +0.0], McNemar p=1.000 (0 vs 0) |
| cell 8 | 30 | +3.3 pp [+0.0, +10.0], McNemar p=1.000 (1 vs 0) | +0.0 pp [+0.0, +0.0], McNemar p=1.000 (0 vs 0) |

### Controls

| check | result |
| --- | --- |
| cell 5 (no reveal) at zero: live HR@10 = HR@5 = 0 | holds (live HR@10 0%) |
| cell 4 (reveal before gold) not below cells 1-3 at live HR@10 | holds (cell 4 3%, lowest of cells 1-3 3%) |

## Fusion landscape (information only, not used anywhere else)

| cell | n | lambda_n 0.5 (published) | lambda_n 1.0 | lambda_n 2.0 | memory channel only |
| --- | --- | --- | --- | --- | --- |
| 1 (A, reveal 3 chunks after the gold) | 30 | 10 / 3 / 3 | 23 / 17 / 3 | 30 / 20 / 3 | 37 / 20 / 3 |
| 2 (A, reveal 25% of the book after) | 30 | 3 / 7 / 3 | 17 / 10 / 3 | 20 / 17 / 0 | 27 / 20 / 3 |
| 3 (A, reveal 55% of the book after) | 30 | 3 / 7 / 3 | 7 / 10 / 3 | 7 / 17 / 0 | 13 / 17 / 3 |
| 4 (A, reveal 20% before the gold (control)) | 30 | 3 / 3 / 3 | 0 / 0 / 7 | 3 / 7 / 3 | 13 / 7 / 3 |
| 5 (A, no reveal (control)) | 30 | 0 / 0 / 0 | 0 / 3 / 0 | 3 / 7 / 3 | 7 / 3 / 7 |
| 6 (B, nickname, reveal 25% after) | 30 | 3 / 0 / 0 | 10 / 7 / 0 | 17 / 10 / 0 | 27 / 17 / 0 |
| 7 (C, description, reveal 25% after) | 30 | 0 / 3 / 0 | 0 / 10 / 0 | 7 / 20 / 7 | 27 / 47 / 17 |
| 8 (D, object, reveal 25% after) | 30 | 13 / 7 / 7 | 17 / 10 / 7 | 20 / 7 / 7 | 20 / 7 / 7 |

| persona | cell | rank at 0.5 | smallest lambda_n for top 10 | memory-channel-only rank |
| --- | --- | --- | --- | --- |
| c1_B04_013 | 1 | 26 | 0.75 | 1 |
| c1_B18_009 | 1 | 73 | 1 | 2 |
| c1_B18_011 | 1 | 57 | 1.5 | 1 |
| c1_B26_026 | 1 | 58 | >100 | 22 |
| c1_B27_018 | 1 | 6 | 0.5 | 1 |
| c1_B31_002 | 1 | 6 | 0.5 | 1 |
| c1_B34_015 | 1 | 11 | 0.75 | 1 |
| c1_B34_017 | 1 | 146 | 15 | 8 |
| c1_B43_005 | 1 | 41 | 1.5 | 6 |
| c1_B44_028 | 1 | 151 | 5 | 2 |
| c1_B44_029 | 1 | 38 | 0.75 | 1 |
| c1_B61_022 | 1 | 141 | >100 | 53 |
| c1_B61_023 | 1 | 2 | 0.5 | 2 |
| c2_B04_013 | 2 | 16 | 0.75 | 2 |
| c2_B04_014 | 2 | 77 | 5 | 7 |
| c2_B18_009 | 2 | 24 | 0.75 | 1 |
| c2_B26_025 | 2 | 24 | 1 | 3 |
| c2_B26_026 | 2 | 44 | 1 | 1 |
| c2_B27_019 | 2 | 95 | 5 | 8 |
| c2_B44_029 | 2 | 30 | 0.75 | 2 |
| c2_B61_021 | 2 | 51 | >100 | 48 |
| c3_B18_009 | 3 | 12 | 0.75 | 1 |
| c3_B37_007 | 3 | 201 | 10 | 2 |
| c3_B61_022 | 3 | 151 | >100 | 53 |
| c4_B04_014 | 4 | 24 | 1.5 | 7 |
| c4_B18_009 | 4 | 94 | 5 | 6 |
| c4_B18_011 | 4 | 56 | 3 | 2 |
| c4_B26_026 | 4 | 110 | >100 | 31 |
| c4_B27_018 | 4 | 120 | 5 | 2 |
| c4_B31_001 | 4 | 26 | >100 | 25 |
| c4_B31_002 | 4 | 37 | >100 | 18 |
| c4_B61_022 | 4 | 100 | >100 | 54 |
| c6_B04_014 | 6 | 16 | 0.75 | 2 |
| c6_B27_018 | 6 | 6 | 0.5 | 1 |
| c6_B27_019 | 6 | 62 | 5 | 8 |
| c6_B31_000 | 6 | 51 | 1.5 | 1 |
| c6_B44_027 | 6 | 42 | 1.5 | 3 |
| c6_B61_022 | 6 | 69 | >100 | 53 |
| c6_B61_023 | 6 | 135 | 1 | 2 |
| c7_B04_012 | 7 | 201 | >100 | 186 |
| c7_B04_014 | 7 | 29 | >100 | 15 |
| c7_B18_009 | 7 | 210 | 7.5 | 4 |
| c7_B26_024 | 7 | 134 | >100 | 68 |
| c7_B26_026 | 7 | 48 | 3 | 6 |
| c7_B27_018 | 7 | 126 | 2 | 1 |
| c7_B27_019 | 7 | 80 | >100 | 41 |
| c7_B31_002 | 7 | 144 | >100 | 45 |
| c7_B34_015 | 7 | 122 | 2 | 1 |
| c7_B34_016 | 7 | 142 | >100 | 18 |
| c7_B37_006 | 7 | 59 | >100 | 11 |
| c7_B37_008 | 7 | 142 | 10 | 6 |
| c7_B43_003 | 7 | 113 | 3 | 2 |
| c7_B43_005 | 7 | 68 | >100 | 13 |
| c7_B61_021 | 7 | 75 | 5 | 8 |
| c7_B61_022 | 7 | 23 | >100 | 26 |
| c7_B61_023 | 7 | 159 | >100 | 55 |
| c8_B18_011 | 8 | 207 | >100 | 144 |
| c8_B26_024 | 8 | 48 | >100 | 35 |
| c8_B26_026 | 8 | 20 | 2 | 16 |
| c8_B27_019 | 8 | 138 | >100 | 25 |
| c8_B31_000 | 8 | 125 | 7.5 | 7 |
| c8_B31_002 | 8 | 128 | 10 | 7 |
| c8_B34_017 | 8 | 82 | >100 | 21 |
| c8_B43_003 | 8 | 38 | 10 | 8 |
| c8_B61_021 | 8 | 34 | >100 | 14 |
| c8_B61_022 | 8 | 2 | 0.5 | 2 |
| c8_B61_023 | 8 | 13 | 0.75 | 2 |

## Merge audit

30 of 17099 final SAME AS decisions, seed 42 (to be read; run 1's sample was read as 7-8 wrong of 30):

1. B04 step 304: 'Oliver' -> 'Oliver Twist' (model, shared_token): the chunk refers to him as the boy who is being addressed and whose tears fall, consistent with prior records
2. B04 step 316: 'Oliver' -> 'Oliver Twist' (model, shared_token): the chunk refers to him as "Oliver" and he is the boy who was brought to the prison cell
3. B18 step 18: 'Tringham' -> 'Parson Tringham' (model, shared_token): the antiquary who previously mentioned the d’Urbervilles, referred to as "Pa’son Tringham" in the chunk and previously recorded
4. B18 step 143: 'The Gilded Antler' -> 'The Gilded Antler' (model, shared_token): the chunk states "The Gilded Antler belonged to Wamanpuma", and the record E0145 identifies it as a public house on the high road at Kelmgate, which matches the context
5. B26 step 30: 'Eliza' -> 'Eliza' (model, shared_token): the chunk confirms she is the mother of Harry and is fleeing with him
6. B26 step 89: 'George' -> 'George Harris' (auto-join): auto-join: the title-stripped name equals a shown record's name or alias
7. B27 step 198: 'London' -> 'London' (model, shared_token): the chunk mentions London as the destination for Fanny's departure, matching the record's description
8. B27 step 283: 'Crown' -> 'Crown' (model, shared_token): the chunk mentions Mr. Crawford meeting an acquaintance at the Crown, matching the recalled record
9. B31 step 72: 'free States' -> 'The river' (model, same_sentence): the free States are referenced as the destination on the river, consistent with the river's role as a path to freedom
10. B31 step 108: 'Jim' -> 'Jim' (model, shared_token): the chunk refers to Jim as "Jim" and describes his actions and relationship to Huck, matching the record
11. B31 step 139: 'Peter' -> 'Peter' (model, shared_token): the chunk confirms Peter was the dying brother, left a letter, and had a family connection to Harvey and William
12. B31 step 176: 'Jim' -> 'Jim' (model, shared_token): the chunk uses "Jim" as a direct reference to the same character previously established
13. B34 step 92: 'Mr. Elton' -> 'Mr. Elton' (model, shared_token): the chunk describes his recent confession and current isolation, matching his prior record
14. B34 step 121: 'Mr. Elton' -> 'Mr. Elton' (model, shared_token): the chunk confirms Mr. Elton’s engagement to Miss Hawkins and his changed demeanor, matching his prior role as the one who wrote the charade
15. B34 step 149: 'Mrs. Weston' -> 'Mrs. Weston' (model, shared_token): the chunk shows her speaking about the carriage and forming suspicions, matching her role as a thoughtful observer
16. B34 step 188: 'Miss Woodhouse' -> 'Emma' (model, shared_token): the chunk refers to Emma as "Miss Woodhouse" and describes her reactions, matching the record
17. B37 step 75: 'Darcy' -> 'Darcy' (model, shared_token): recalled record with same name and description of having silent contempt for the Bennet family and being the subject of Mr. Collins's intended introduction
18. B37 step 94: 'Jane' -> 'Jane' (model, shared_token): the chunk refers to Jane as the sister whose happiness is at stake, and she is already recorded
19. B37 step 187: 'Jane' -> 'Jane' (model, shared_token): the chunk mentions Jane’s letter and her role in conveying the news about Lydia, consistent with her prior record
20. B37 step 230: 'Elizabeth' -> 'Elizabeth' (model, shared_token): the chunk refers to Elizabeth as the one hoping to be joined by Darcy, consistent with her role in the history
21. B43 step 21: 'Hareton Earnshaw' -> 'Hareton Earnshaw' (model, shared_token): he is the young man at Wuthering Heights and the last of the Earnshaws
22. B43 step 81: 'Cathy' -> 'Miss Cathy' (model, shared_token): the chunk uses "Cathy" as a nickname for Miss Cathy, and she is described as Catherine's daughter, matching E0019's record
23. B43 step 115: 'Catherine' -> 'Catherine Earnshaw' (model, shared_token): Catherine Earnshaw is the central figure in the emotional reunion, and her name is used interchangeably with "Cathy" and "Catherine Linton"
24. B43 step 139: 'Hareton' -> 'Hareton Earnshaw' (model, shared_token): the chunk refers to Hareton as a strong lad of eighteen, matching the record's description
25. B43 step 149: 'Mr. Edgar' -> 'Master Linton' (model, shared_token): the chunk refers to Mr. Edgar as the father of Cathy and the one who says "if I can get him" about Linton's return
26. B44 step 6: 'Bessie' -> 'Bessie' (model, shared_token): the chunk uses "Bessie" and the record E0019 identifies her as the nurse
27. B44 step 74: 'John Reed' -> 'John' (model, shared_token): John Reed is the cousin of Jane Eyre, previously mentioned in the history and now described in detail
28. B44 step 147: 'AdÃ¨le' -> 'AdÃ¨le' (model, shared_token): the chunk refers to her as Mr. Rochester’s ward and describes her interaction with Henry Lynn, matching the record
29. B61 step 15: 'Nozdryov' -> 'Nozdryov' (model, shared_token): the chunk describes him as a flamboyant, carefree man who exposes Tchitchikov, matching the prior record's description
30. B61 step 274: 'Alexasha' -> 'Alexasha' (model, shared_token): The chunk mentions Alexasha as the one who announces Platon Mihailovitch’s arrival, matching the stub

- B04 step 300: 'The man' -> 'Bill Sikes' rejected: no_shared_token_or_sentence ; model's reason: auto-join: the title-stripped name equals a shown record's name or alias
- B04 step 308: 'Monks' -> 'Monks' rejected: type_conflict person against organization; model's reason: auto-join: the title-stripped name equals a shown record's name or alias
- B18 step 2: 'T. H' -> 'Thomas Hardy' rejected: no_shared_token_or_sentence ; model's reason: the chunk identifies T. H as the author, matching the recalled record
- B26 step 17: 'General Washington' -> 'Uncle Tom' rejected: no_shared_token_or_sentence ; model's reason: the chunk describes a portrait of General Washington, and he is known from prior context as a historical figure; no new record needed
- B27 step 267: 'Mr. Campbell' -> 'Turner' rejected: type_conflict person against place; model's reason: the chunk identifies him as the surgeon of the Thrush, and he is not previously recorded but matches the context of William's friend and the ship's surgeon
- B27 step 273: 'H. C' -> 'Henry Crawford' rejected: no_shared_token_or_sentence ; model's reason: the chunk identifies H. C. as Henry Crawford, who is described in the history as the suitor and brother of Miss Crawford
- B37 step 140: 'Pemberley' -> 'Miss Darcy' rejected: type_conflict place against person; model's reason: auto-join: the title-stripped name equals a shown record's name or alias
- B43 step 51: 'Ellen' -> 'Nelly' rejected: no_shared_token_or_sentence ; model's reason: auto-join: the title-stripped name equals a shown record's name or alias
- B43 step 222: 'Catherine' -> 'Miss Cathy' rejected: no_shared_token_or_sentence ; model's reason: the chunk refers to her as "Cathy" and describes her emotional state and actions, matching the record's description of Miss Cathy
- B44 step 305: 'Marseilles' -> 'Lowood school' rejected: no_shared_token_or_sentence ; model's reason: the chunk contrasts Marseilles with England, and E0065 is a place in a different context but not relevant; however, Marseilles is not in the recalled records, and no prior mention exists—this is a new location

Mr./Mrs. type joins on the books' real characters: 6 (run 1: 120). 

Aliases in total / most on one record / mean per record: run 1b 1846 / 22 / 0.47; run 1 3955 / 80 / 1.44. The ten records with the most aliases in run 1b (for a reader to judge pollution):

- B61 'NIKOLAI GOGOL': ['Gogol', 'Nikolai Vasilyevich Gogol', 'Nikolai', 'Ivan Turgenev', 'Poe', 'Simon Karlinsky', 'Vsevolod Setchkarev', 'Joseph Frank', 'Victor Erlich', 'Robert Maguire', 'James Woodward', 'George Gibian', 'Diary of a Madman', 'Caesar', 'Dante', 'Mark Chagall', 'Chagall', 'Robert A. Maguire', 'Setchkarev, Vsevolod', 'Nikolai Gogol', 'Virgil', 'Nikolai Karamzin']
- B18 'd’Urbervilles': ['d’Urberville family', 'Urbervilles', 'Stoke-d’Urbervilles', 'The d’Urbervilles', 'Mr d’Urberville', 'old d’Urberville family', 'd’Urberville', 'knightly d’Urbervilles', 'Purbeck-marble and alabaster people', 'Knight of the Conqueror', 'ancient Norman family', 'd’Urberville dames', 'mansion of the d’Urbervilles', 'D’Urberville', 'd’Urberville mansion', 'Egdon Heath', 'd’Urberville Aisle', 'd’Urberville Window', 'd’Urberville knights and dames']
- B18 'Tess of the d’Urbervilles': ['Tess Durbeyfield', 'Tess', 'Simple Tess Durbeyfield', 'Queen of Spades', 'Queen of Diamonds', 'Car', 'Tess d’Urberville', 'otherwise d’Urberville', 'The girl-mother', 'Tessy', 'Durbeyfield household', 'Nazareth', 'Mrs Angel Clare', 'Jesus Christ', 'Tess’s father and mother', 'T', 'Durbeyfield', 'Miss Durbeyfield']
- B04 'Oliver Twist': ['Oliver', 'young Oliver Twist', 'The boy', 'The Artful Dodger', 'Dodger', 'the Dodger', 'The Dodger', 'Brownlow', 'Mr. Brownlow', 'Little Oliver Twist', 'Little Oliver', 'Nolly', 'Ned', 'Hampton', 'Young Twist', 'young Oliver', 'Master Oliver']
- B26 'George Harris': ['George', 'Mr. Harris', 'young Mas’r Tom', 'Mas’r', 'Tom', 'young Master George', 'Mas’r George', 'Mas’r Georgey', 'Master George', 'Missis', 'Mr. Wilson', 'The old gentleman', 'This young man', 'Jim', 'George Harris', 'Harris']
- B43 'Master Linton': ['Edgar', 'Master Linton', 'Edgar Linton', 'Mr. Edgar', 'Linton', 'Maister Linton', 'Linton Heathcliff', 'Pretty Linton', 'Heathcliff', 'Master Heathcliff', 'Young master', 'My master', 'His nephew', 'Papa']
- B43 'Catherine Earnshaw': ['Catherine', 'Catherine Linton', 'Miss Catherine', 'Edgar', 'Edgar Linton', 'Linton', 'Miss Earnshaw', 'Miss Cathy', 'Cathy', 'Mr. Edgar', 'Catherine Heathcliff', 'My father', 'Catherine Earnshaw']
- B61 'Tchitchikov': ['Chichikov', 'Pavel Ivanovitch Tchitchikov', 'Pavel Ivanovitch', 'Madame Manilov', 'Manilov', 'Our hero', 'Gremin', 'Zvonsky', 'Lidin', 'Tchitchikov’s father', 'Tchitchikov’s horses', 'Tchitchikov', 'Platonov']
- B04 'Jack Dawkins': ['The Artful Dodger', 'Mr. Dawkins', 'Mr. John Dawkins', 'Artful Dodger', 'Dodger', 'Artful', 'The Artful', 'the Dodger', 'Dawkins', 'Jack', 'Jack Dawkins']
- B04 'Fagin': ['the Jew', 'the Dodger', 'Fagin', 'The Jew', 'Jew', 'Mr. Fagin', 'the old Jew', 'Faguey', 'old Fagin', 'young Jew', 'the crafty Jew']

## Cost

| Measure | run 1b | run 1 | run 0 |
| --- | --- | --- | --- |
| calls per book | equal to run 1 | equal to run 0 | - |
| input tokens per call (mean of book means) | 5159 | 4991 | 4065 |
| output tokens per call | 1134 | 1128 | 1079 |
| replies at the output cap with no JSON | 2 | 3 | - |
| records / stubs (10 books) | 2938 / 1005 | 2323 / 428 | records only |
| recall cap hits (chunks) | 2443 (77%, cap 8) | 1292 (cap 5) | - |
| wall clock on 4 GPUs | 3.84 h | 3.69 h | 3.36 h |
| candidate-net seconds per chunk (GPU, in the build) | 0.083 | 0.082 | - |
| candidate net, threshold / candidates per chunk | 0.3 / 7.6 | 0.5 / 7.0 | - |

## Conclusion

The link formed in 24 of 90 characters of cells 1-3 (run 1: 14/90, run 0: 1/90); 34 of 52 joins went into the gold record, and the earlier record was in the prompt at the reveal in 121 of 174 cases where it existed (run 1: 82 of 159). Retrieval did not clearly follow: live HR@10 in cells 1-3 is 6% against frozen 3% (live minus frozen +2.2 pp, 95% CI [-2.2, +6.7], McNemar p=0.625); run 1 live was 6%. It cost 5159 input and 1134 output tokens per call (run 1: 4991 and 1128), 3.84 h on 4 GPUs (run 1: 3.69 h), and the same number of calls.

## Open problems

- 18 joins in cells 1-3 still put form 1 into a separate form-2 record; form 1 sits there as {'description': 34, 'alias': 5}. The optional JSON-alias join was off.
- Earlier record not in the prompt at the link step: 32 cut by the cap of 8, 21 not matched by recall.
- SAME AS rejected by rule 3: {'no_shared_token_or_sentence': 1288, 'type_conflict': 559, 'title_conflict': 345}; the ten-row sample in stage 03 shows whether the rule cut true joins.
- 6 records still join one stripped name under different titles (run 1: 120).
- 2 replies hit the output cap with no JSON (more candidates at 0.3 lengthen the decision block).
- Fusion: the memory channel's ranking does not reach the top 10 at the published weight; the landscape above is information only (Issue 6).

## Commands and commit

- `nohup bash experiments/method_update/p1b_ledger/run_all.sh > experiments/method_update/p1b_ledger/logs/run_all_nohup.out 2>&1 &`
- Build: SADI_OUTPUT_DIR=/mnt/data/zeinab/SAADI/outputs/method_update/p1b_ledger CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_PORT=29537 /mnt/data/zeinab/raptor/.venv/bin/python3 /mnt/data/zeinab/SAADI/scripts/run_study.py --suite probe_bridge --profile p1b_ledger --datasets novelhopqa_probe --retrievers bm25 --stage ingest --tensor-parallel-size 4
- Commit `28eed3a1ee31487f3bf23dcb3470aa0e164e8cfd`; stage reports `experiments/method_update/p1b_ledger/reports/step_01..05.md`; run 1's own step code on run 1b in `step_03_run1_code.md`, `step_04_run1_code.md`.
