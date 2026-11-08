# p1_ledger tables

| Cell | n | Vanilla HR@10 / median rank | No memory HR@10 / median rank | Frozen HR@10 / median rank | Live HR@10 / median rank | Live - Frozen @10 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 (A, reveal 3 chunks after the gold) | 30 | 3 / 144 (run 0: 3 / 144) | 0 / 149 (run 0: 0 / 140) | 0 / 156 (run 0: 0 / 172) | 3 / 115 (run 0: 3 / 172) | +3.3 pp [+0.0, +10.0], McNemar p=1.000 (1 vs 0) |
| 2 (A, reveal 25% of the book after) | 30 | 3 / 124 (run 0: 3 / 124) | 3 / 134 (run 0: 3 / 142) | 3 / 134 (run 0: 3 / 148) | 7 / 116 (run 0: 3 / 142) | +3.3 pp [+0.0, +10.0], McNemar p=1.000 (1 vs 0) |
| 3 (A, reveal 55% of the book after) | 30 | 3 / 162 (run 0: 3 / 162) | 3 / 168 (run 0: 3 / 175) | 7 / 174 (run 0: 3 / 216) | 7 / 146 (run 0: 3 / 200) | +0.0 pp [+0.0, +0.0], McNemar p=1.000 (0 vs 0) |
| 4 (A, reveal 20% before the gold (control)) | 30 | 7 / 118 (run 0: 7 / 118) | 3 / 104 (run 0: 3 / 121) | 7 / 118 (run 0: 3 / 108) | 3 / 120 (run 0: 3 / 111) | -3.3 pp [-10.0, +0.0], McNemar p=1.000 (0 vs 1) |
| 5 (A, no reveal (control)) | 30 | 0 / 152 (run 0: 0 / 152) | 0 / 134 (run 0: 0 / 136) | 0 / 116 (run 0: 0 / 134) | 0 / 116 (run 0: 0 / 128) | +0.0 pp [+0.0, +0.0], McNemar p=1.000 (0 vs 0) |
| 6 (B, nickname, reveal 25% after) | 30 | 0 / 168 (run 0: 0 / 168) | 0 / 166 (run 0: 0 / 137) | 0 / 153 (run 0: 0 / 156) | 0 / 122 (run 0: 0 / 146) | +0.0 pp [+0.0, +0.0], McNemar p=1.000 (0 vs 0) |
| 7 (C, description, reveal 25% after) | 30 | 0 / 174 (run 0: 0 / 174) | 0 / 197 (run 0: 0 / 178) | 3 / 182 (run 0: 0 / 171) | 3 / 111 (run 0: 0 / 166) | +0.0 pp [+0.0, +0.0], McNemar p=1.000 (0 vs 0) |
| 8 (D, object, reveal 25% after) | 30 | 3 / 128 (run 0: 3 / 128) | 7 / 143 (run 0: 3 / 124) | 7 / 119 (run 0: 3 / 135) | 7 / 114 (run 0: 7 / 116) | +0.0 pp [+0.0, +0.0], McNemar p=1.000 (0 vs 0) |

| Cell | n | both names present | earlier record in prompt at the reveal | joined when both present | bridge formed | form 1 present after gold |
| --- | --- | --- | --- | --- | --- | --- |
| 1 (A, reveal 3 chunks after the gold) | 30 | 22/30 (73%) (run 0: 19/30 (63%)) | 16/30 (53%) (run 0: 0/30 (0%)) | 17/30 (57%) (run 0: 12/30 (40%)) | 4/30 (13%) (run 0: 1/30 (3%)) | 26/30 (87%) (run 0: 18/30 (60%)) |
| 2 (A, reveal 25% of the book after) | 30 | 21/30 (70%) (run 0: 23/30 (77%)) | 15/30 (50%) (run 0: 1/30 (3%)) | 14/30 (47%) (run 0: 10/30 (33%)) | 5/30 (17%) (run 0: 0/30 (0%)) | 27/30 (90%) (run 0: 19/30 (63%)) |
| 3 (A, reveal 55% of the book after) | 30 | 22/30 (73%) (run 0: 15/30 (50%)) | 10/30 (33%) (run 0: 1/30 (3%)) | 15/30 (50%) (run 0: 6/30 (20%)) | 5/30 (17%) (run 0: 0/30 (0%)) | 28/30 (93%) (run 0: 14/30 (47%)) |
| 4 (A, reveal 20% before the gold (control)) | 30 | 19/30 (63%) (run 0: 16/30 (53%)) | 3/30 (10%) (run 0: 2/30 (7%)) | 16/30 (53%) (run 0: 6/30 (20%)) | 1/30 (3%) (run 0: 0/30 (0%)) | 26/30 (87%) (run 0: 14/30 (47%)) |
| 5 (A, no reveal (control)) | 30 | 25/30 (83%) (run 0: 15/30 (50%)) | - | - | - | - |
| 6 (B, nickname, reveal 25% after) | 30 | 16/30 (53%) (run 0: 14/30 (47%)) | 9/30 (30%) (run 0: 0/30 (0%)) | 13/30 (43%) (run 0: 8/30 (27%)) | 4/30 (13%) (run 0: 0/30 (0%)) | 16/30 (53%) (run 0: 9/30 (30%)) |
| 7 (C, description, reveal 25% after) | 30 | 0/30 (0%) (run 0: 23/30 (77%)) | 18/30 (60%) (run 0: 2/30 (7%)) | 0/30 (0%) (run 0: 15/30 (50%)) | 20/30 (67%) (run 0: 5/30 (17%)) | 25/30 (83%) (run 0: 23/30 (77%)) |
| 8 (D, object, reveal 25% after) | 30 | 21/30 (70%) (run 0: 16/30 (53%)) | 11/30 (37%) (run 0: 2/30 (7%)) | 15/30 (50%) (run 0: 11/30 (37%)) | 3/30 (10%) (run 0: 0/30 (0%)) | 16/30 (53%) (run 0: 10/30 (33%)) |

| Quantity | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| both names present (cells with a reveal) | from ~30% to >80% | 121/210 (58%); run 0 126/210 (60%) | not held |
| earlier record in the prompt at the reveal (when it existed) | from 6 of 72 to most | 82/159 (52%) | held |
| bridge formed, cells 1-3 | from 1 of 90 to a clear majority, if the join works | 14/90 (16%) | not held |
| cell 5 stays at zero (live HR@10) | 0 | 0% | held |
| calls per book unchanged | equal to run 0 | equal | held |
