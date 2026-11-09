# p1b_ledger - step 04: Retrieve and score (run 1's step 4 code on run 1b)

Status: **ok** - 2026-11-09 00:41:59

## What was done

- `score_mbr_variant_document_pools + combine (published settings) per persona; frozen via saadi.rebuttal.frozen_records with saadi.ledger.measure annotations; vanilla = the text channel alone`
- Versions: python=3.12.3, platform=Linux-6.17.0-23-generic-x86_64-with-glibc2.39, numpy=2.3.5, pandas=3.0.1, yaml=6.0.3, saadi=0.1.0, git_commit=28eed3a1ee31487f3bf23dcb3470aa0e164e8cfd, torch=2.11.0+cu129, transformers=4.57.6, gliner=0.2.30, vllm=0.23.0

## Numbers

### How to read this file

Produced by run 1's own step code (experiments/method_update/p1_ledger/scripts, imported unchanged) on run 1b's build. In its tables **'run 1' means run 1b** and 'run 0' is run 0. The run 1b report with three columns is step_04.md.

### Stubs in the frozen view

| book | chunks | chunks linked to at least one stub at their own step | linked names per chunk (mean) |
| --- | --- | --- | --- |
| B04 | 321 | 320 | 4.73 |
| B18 | 312 | 312 | 5.59 |
| B26 | 384 | 382 | 5.94 |
| B27 | 338 | 336 | 7.70 |
| B31 | 244 | 237 | 4.30 |
| B34 | 335 | 334 | 7.59 |
| B37 | 262 | 262 | 8.43 |
| B43 | 246 | 246 | 6.25 |
| B44 | 385 | 378 | 5.21 |
| B61 | 357 | 353 | 5.79 |

### HR@10 (%) / median gold rank per cell, run 1 (run 0 in parentheses)

| cell | n | Vanilla | No memory | Frozen | Live | Live - Frozen @10 | Live - Vanilla @10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 (A, reveal 3 chunks after the gold) | 30 | 3 / 144 (run 0: 3 / 144) | 0 / 148 (run 0: 0 / 140) | 3 / 184 (run 0: 0 / 172) | 10 / 68 (run 0: 3 / 172) | +6.7 pp [-6.7, +20.0], McNemar p=0.625 (3 vs 1) | +6.7 pp [-6.7, +20.0], McNemar p=0.625 (3 vs 1) |
| 2 (A, reveal 25% of the book after) | 30 | 3 / 124 (run 0: 3 / 124) | 3 / 138 (run 0: 3 / 142) | 3 / 155 (run 0: 3 / 148) | 3 / 82 (run 0: 3 / 142) | +0.0 pp [+0.0, +0.0], McNemar p=1.000 (0 vs 0) | +0.0 pp [+0.0, +0.0], McNemar p=1.000 (0 vs 0) |
| 3 (A, reveal 55% of the book after) | 30 | 3 / 162 (run 0: 3 / 162) | 3 / 162 (run 0: 3 / 175) | 3 / 160 (run 0: 3 / 216) | 3 / 148 (run 0: 3 / 200) | +0.0 pp [+0.0, +0.0], McNemar p=1.000 (0 vs 0) | +0.0 pp [+0.0, +0.0], McNemar p=1.000 (0 vs 0) |
| 4 (A, reveal 20% before the gold (control)) | 30 | 7 / 118 (run 0: 7 / 118) | 3 / 130 (run 0: 3 / 121) | 7 / 98 (run 0: 3 / 108) | 3 / 112 (run 0: 3 / 111) | -3.3 pp [-10.0, +0.0], McNemar p=1.000 (0 vs 1) | -3.3 pp [-10.0, +0.0], McNemar p=1.000 (0 vs 1) |
| 5 (A, no reveal (control)) | 30 | 0 / 152 (run 0: 0 / 152) | 0 / 142 (run 0: 0 / 136) | 0 / 134 (run 0: 0 / 134) | 0 / 122 (run 0: 0 / 128) | +0.0 pp [+0.0, +0.0], McNemar p=1.000 (0 vs 0) | +0.0 pp [+0.0, +0.0], McNemar p=1.000 (0 vs 0) |
| 6 (B, nickname, reveal 25% after) | 30 | 0 / 168 (run 0: 0 / 168) | 0 / 147 (run 0: 0 / 137) | 0 / 143 (run 0: 0 / 156) | 3 / 118 (run 0: 0 / 146) | +3.3 pp [+0.0, +10.0], McNemar p=1.000 (1 vs 0) | +3.3 pp [+0.0, +10.0], McNemar p=1.000 (1 vs 0) |
| 7 (C, description, reveal 25% after) | 30 | 0 / 174 (run 0: 0 / 174) | 0 / 174 (run 0: 0 / 178) | 0 / 203 (run 0: 0 / 171) | 0 / 130 (run 0: 0 / 166) | +0.0 pp [+0.0, +0.0], McNemar p=1.000 (0 vs 0) | +0.0 pp [+0.0, +0.0], McNemar p=1.000 (0 vs 0) |
| 8 (D, object, reveal 25% after) | 30 | 3 / 128 (run 0: 3 / 128) | 7 / 113 (run 0: 3 / 124) | 10 / 98 (run 0: 3 / 135) | 13 / 95 (run 0: 7 / 116) | +3.3 pp [+0.0, +10.0], McNemar p=1.000 (1 vs 0) | +10.0 pp [+0.0, +23.3], McNemar p=0.250 (3 vs 0) |

### HR@5 (%) and the memory-channel-only median rank of the gold chunk

| cell | n | Vanilla | No memory | Frozen | Live | mem-rank live | mem-rank frozen |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 30 | 0 (run 0: 0) | 0 (run 0: 0) | 0 (run 0: 0) | 3 (run 0: 0) | 48 | 145 |
| 2 | 30 | 3 (run 0: 3) | 0 (run 0: 0) | 0 (run 0: 0) | 0 (run 0: 0) | 116 | 121 |
| 3 | 30 | 0 (run 0: 0) | 0 (run 0: 0) | 0 (run 0: 0) | 0 (run 0: 0) | 82 | 153 |
| 4 | 30 | 0 (run 0: 0) | 0 (run 0: 0) | 0 (run 0: 0) | 0 (run 0: 0) | 79 | 76 |
| 5 | 30 | 0 (run 0: 0) | 0 (run 0: 0) | 0 (run 0: 0) | 0 (run 0: 0) | 140 | 150 |
| 6 | 30 | 0 (run 0: 0) | 0 (run 0: 0) | 0 (run 0: 0) | 0 (run 0: 0) | 64 | 162 |
| 7 | 30 | 0 (run 0: 0) | 0 (run 0: 0) | 0 (run 0: 0) | 0 (run 0: 0) | 50 | 193 |
| 8 | 30 | 0 (run 0: 0) | 0 (run 0: 0) | 0 (run 0: 3) | 3 (run 0: 3) | 36 | 116 |

### HR@5 (%) with 95% bootstrap interval over personas (10,000 resamples, seed 42)

| cell | n | vanilla | no_memory | frozen | live |
| --- | --- | --- | --- | --- | --- |
| 1 | 30 | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 3 [0, 10] |
| 2 | 30 | 3 [0, 10] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |
| 3 | 30 | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |
| 4 | 30 | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |
| 5 | 30 | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |
| 6 | 30 | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |
| 7 | 30 | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |
| 8 | 30 | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 3 [0, 10] |

### HR@10 (%) with 95% bootstrap interval over personas (10,000 resamples, seed 42)

| cell | n | vanilla | no_memory | frozen | live |
| --- | --- | --- | --- | --- | --- |
| 1 | 30 | 3 [0, 10] | 0 [0, 0] | 3 [0, 10] | 10 [0, 23] |
| 2 | 30 | 3 [0, 10] | 3 [0, 10] | 3 [0, 10] | 3 [0, 10] |
| 3 | 30 | 3 [0, 10] | 3 [0, 10] | 3 [0, 10] | 3 [0, 10] |
| 4 | 30 | 7 [0, 17] | 3 [0, 10] | 7 [0, 17] | 3 [0, 10] |
| 5 | 30 | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |
| 6 | 30 | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 3 [0, 10] |
| 7 | 30 | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |
| 8 | 30 | 3 [0, 10] | 7 [0, 17] | 10 [0, 23] | 13 [3, 27] |

### Share of personas with the gold chunk at rank 1 (%)

| cell | n | vanilla | no_memory | frozen | live |
| --- | --- | --- | --- | --- | --- |
| 1 | 30 | 0 | 0 | 0 | 0 |
| 2 | 30 | 0 | 0 | 0 | 0 |
| 3 | 30 | 0 | 0 | 0 | 0 |
| 4 | 30 | 0 | 0 | 0 | 0 |
| 5 | 30 | 0 | 0 | 0 | 0 |
| 6 | 30 | 0 | 0 | 0 | 0 |
| 7 | 30 | 0 | 0 | 0 | 0 |
| 8 | 30 | 0 | 0 | 0 | 0 |

### Bridge-reached personas only: HR@10 (%)

| cell | n | Vanilla | No memory | Frozen | Live | Live - Frozen @10 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 13 | 0 | 0 | 0 | 23 | +23.1 pp [+0.0, +46.2], McNemar p=0.250 (3 vs 0) |
| 2 | 8 | 0 | 0 | 0 | 0 | +0.0 pp [+0.0, +0.0], McNemar p=1.000 (0 vs 0) |
| 3 | 3 | 0 | 0 | 0 | 0 | +0.0 pp [+0.0, +0.0], McNemar p=1.000 (0 vs 0) |
| 4 | 8 | 0 | 0 | 0 | 0 | +0.0 pp [+0.0, +0.0], McNemar p=1.000 (0 vs 0) |
| 5 | 0 | - | - | - | - | - |
| 6 | 7 | 0 | 0 | 0 | 14 | +14.3 pp [+0.0, +42.9], McNemar p=1.000 (1 vs 0) |
| 7 | 17 | 0 | 0 | 0 | 0 | +0.0 pp [+0.0, +0.0], McNemar p=1.000 (0 vs 0) |
| 8 | 11 | 0 | 0 | 0 | 9 | +9.1 pp [+0.0, +27.3], McNemar p=1.000 (1 vs 0) |

### Checks

| check | result |
| --- | --- |
| cell 5 (no reveal) stays at zero: live HR@10 = 0 and HR@5 = 0 | holds (live HR@10 0%, HR@5 0%) |
| cell 4 (reveal before gold) not below cells 1-3 at live HR@10 | holds (cell 4 3%, cells 1-3 [10, 3, 3]) |
| cells 1-3 pooled: live minus frozen @10 | +2.2 pp [-2.2, +6.7], McNemar p=0.625 |
| cells 1-3 pooled: live HR@10 (run 0) | 6% (3%) |

### Vanilla leak check

A cell where Vanilla succeeds means a query word reached the gold chunk through the text; expected to fail everywhere (HR near 0), as in run 0.

### Hand-off

Paste this report back. Step 5 writes the final tables and the Google Doc text.

## Anything unexpected

Nothing unexpected.

## Files written

- `/mnt/data/zeinab/SAADI/outputs/method_update/p1b_ledger/per_persona.jsonl`
