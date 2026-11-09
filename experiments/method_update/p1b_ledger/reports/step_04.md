# p1b_ledger - step 04: Retrieve and score, fusion landscape (run 1b / run 1 / run 0)

Status: **ok** - 2026-11-09 00:43:19

## What was done

- `bash experiments/method_update/p1b_ledger/run_all.sh  (stage 04_retrieve; python stage04_retrieve.py )`
- `git commit 28eed3a1ee31487f3bf23dcb3470aa0e164e8cfd`
- Versions: python=3.12.3, platform=Linux-6.17.0-23-generic-x86_64-with-glibc2.39, numpy=2.3.5, pandas=3.0.1, yaml=6.0.3, saadi=0.1.0, git_commit=28eed3a1ee31487f3bf23dcb3470aa0e164e8cfd, torch=2.11.0+cu129, transformers=4.57.6, gliner=0.2.30, vllm=0.23.0

## Numbers

### HR@10 (%) per cell: run 1b / run 1 / run 0

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

### HR@5 (%) per cell: run 1b / run 1 / run 0

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

### Median gold rank per cell: run 1b / run 1 / run 0

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

### Live minus Frozen @10: mean difference in pp, 95% bootstrap interval (10,000 resamples, seed 42), exact McNemar (discordant pairs)

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

### Bridge-reached personas only: live HR@10

| group | n (1b / 1) | live HR@10 (1b / 1) |
| --- | --- | --- |
| cells 1-3 | 24 / 14 | 12 / 14 |
| all cells | 67 / 42 | 7 / 7 |

### Fusion landscape, information only: Live HR@10 (%) by memory weight lambda_n (run 1b / run 1 / run 0); text 1.0 and history 0.3 fixed

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

### Bridged personas of run 1b (67): rank at the published weight and the smallest lambda_n on the grid [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0] that puts the gold chunk in the top 10

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

## Anything unexpected

Nothing unexpected.

## Files written

- `/mnt/data/zeinab/SAADI/experiments/method_update/p1b_ledger/reports/step_04_run1_code.md`
- `/mnt/data/zeinab/SAADI/outputs/method_update/p1b_ledger/per_persona.jsonl`
