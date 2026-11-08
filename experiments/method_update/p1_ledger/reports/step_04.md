# p1_ledger - step 04: Retrieve and score

Status: **ok** - 2026-11-08 18:27:20

## What was done

- `bash experiments/method_update/p1_ledger/sh/04_*.sh `
- `score_mbr_variant_document_pools + combine (published settings) per persona; frozen via saadi.rebuttal.frozen_records with saadi.ledger.measure annotations; vanilla = the text channel alone`
- Versions: python=3.12.3, platform=Linux-6.17.0-23-generic-x86_64-with-glibc2.39, numpy=2.3.5, pandas=3.0.1, yaml=6.0.3, saadi=0.1.0, git_commit=7d6f9b6214931734473c528f1f283498d2cabc5b, torch=2.11.0+cu129, transformers=4.57.6, gliner=0.2.30, vllm=0.23.0

## Numbers

### Stubs in the frozen view

| book | chunks | chunks linked to at least one stub at their own step | linked names per chunk (mean) |
| --- | --- | --- | --- |
| B04 | 321 | 319 | 4.34 |
| B18 | 312 | 312 | 4.84 |
| B26 | 384 | 380 | 5.11 |
| B27 | 338 | 338 | 7.04 |
| B31 | 244 | 237 | 4.20 |
| B34 | 335 | 335 | 7.03 |
| B37 | 262 | 262 | 7.41 |
| B43 | 246 | 246 | 5.76 |
| B44 | 385 | 364 | 4.39 |
| B61 | 357 | 352 | 4.78 |

### HR@10 (%) / median gold rank per cell, run 1 (run 0 in parentheses)

| cell | n | Vanilla | No memory | Frozen | Live | Live - Frozen @10 | Live - Vanilla @10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 (A, reveal 3 chunks after the gold) | 30 | 3 / 144 (run 0: 3 / 144) | 0 / 149 (run 0: 0 / 140) | 0 / 156 (run 0: 0 / 172) | 3 / 115 (run 0: 3 / 172) | +3.3 pp [+0.0, +10.0], McNemar p=1.000 (1 vs 0) | +0.0 pp [-10.0, +10.0], McNemar p=1.000 (1 vs 1) |
| 2 (A, reveal 25% of the book after) | 30 | 3 / 124 (run 0: 3 / 124) | 3 / 134 (run 0: 3 / 142) | 3 / 134 (run 0: 3 / 148) | 7 / 116 (run 0: 3 / 142) | +3.3 pp [+0.0, +10.0], McNemar p=1.000 (1 vs 0) | +3.3 pp [+0.0, +10.0], McNemar p=1.000 (1 vs 0) |
| 3 (A, reveal 55% of the book after) | 30 | 3 / 162 (run 0: 3 / 162) | 3 / 168 (run 0: 3 / 175) | 7 / 174 (run 0: 3 / 216) | 7 / 146 (run 0: 3 / 200) | +0.0 pp [+0.0, +0.0], McNemar p=1.000 (0 vs 0) | +3.3 pp [+0.0, +10.0], McNemar p=1.000 (1 vs 0) |
| 4 (A, reveal 20% before the gold (control)) | 30 | 7 / 118 (run 0: 7 / 118) | 3 / 104 (run 0: 3 / 121) | 7 / 118 (run 0: 3 / 108) | 3 / 120 (run 0: 3 / 111) | -3.3 pp [-10.0, +0.0], McNemar p=1.000 (0 vs 1) | -3.3 pp [-10.0, +0.0], McNemar p=1.000 (0 vs 1) |
| 5 (A, no reveal (control)) | 30 | 0 / 152 (run 0: 0 / 152) | 0 / 134 (run 0: 0 / 136) | 0 / 116 (run 0: 0 / 134) | 0 / 116 (run 0: 0 / 128) | +0.0 pp [+0.0, +0.0], McNemar p=1.000 (0 vs 0) | +0.0 pp [+0.0, +0.0], McNemar p=1.000 (0 vs 0) |
| 6 (B, nickname, reveal 25% after) | 30 | 0 / 168 (run 0: 0 / 168) | 0 / 166 (run 0: 0 / 137) | 0 / 153 (run 0: 0 / 156) | 0 / 122 (run 0: 0 / 146) | +0.0 pp [+0.0, +0.0], McNemar p=1.000 (0 vs 0) | +0.0 pp [+0.0, +0.0], McNemar p=1.000 (0 vs 0) |
| 7 (C, description, reveal 25% after) | 30 | 0 / 174 (run 0: 0 / 174) | 0 / 197 (run 0: 0 / 178) | 3 / 182 (run 0: 0 / 171) | 3 / 111 (run 0: 0 / 166) | +0.0 pp [+0.0, +0.0], McNemar p=1.000 (0 vs 0) | +3.3 pp [+0.0, +10.0], McNemar p=1.000 (1 vs 0) |
| 8 (D, object, reveal 25% after) | 30 | 3 / 128 (run 0: 3 / 128) | 7 / 143 (run 0: 3 / 124) | 7 / 119 (run 0: 3 / 135) | 7 / 114 (run 0: 7 / 116) | +0.0 pp [+0.0, +0.0], McNemar p=1.000 (0 vs 0) | +3.3 pp [+0.0, +10.0], McNemar p=1.000 (1 vs 0) |

### HR@5 (%) and the memory-channel-only median rank of the gold chunk

| cell | n | Vanilla | No memory | Frozen | Live | mem-rank live | mem-rank frozen |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 30 | 0 (run 0: 0) | 0 (run 0: 0) | 0 (run 0: 0) | 3 (run 0: 0) | 142 | 151 |
| 2 | 30 | 3 (run 0: 3) | 0 (run 0: 0) | 0 (run 0: 0) | 0 (run 0: 0) | 93 | 121 |
| 3 | 30 | 0 (run 0: 0) | 0 (run 0: 0) | 0 (run 0: 0) | 0 (run 0: 0) | 70 | 126 |
| 4 | 30 | 0 (run 0: 0) | 0 (run 0: 0) | 0 (run 0: 0) | 0 (run 0: 0) | 126 | 98 |
| 5 | 30 | 0 (run 0: 0) | 0 (run 0: 0) | 0 (run 0: 0) | 0 (run 0: 0) | 92 | 98 |
| 6 | 30 | 0 (run 0: 0) | 0 (run 0: 0) | 0 (run 0: 0) | 0 (run 0: 0) | 93 | 120 |
| 7 | 30 | 0 (run 0: 0) | 0 (run 0: 0) | 0 (run 0: 0) | 3 (run 0: 0) | 12 | 142 |
| 8 | 30 | 0 (run 0: 0) | 0 (run 0: 0) | 0 (run 0: 3) | 0 (run 0: 3) | 74 | 90 |

### HR@5 (%) with 95% bootstrap interval over personas (10,000 resamples, seed 42)

| cell | n | vanilla | no_memory | frozen | live |
| --- | --- | --- | --- | --- | --- |
| 1 | 30 | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 3 [0, 10] |
| 2 | 30 | 3 [0, 10] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |
| 3 | 30 | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |
| 4 | 30 | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |
| 5 | 30 | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |
| 6 | 30 | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |
| 7 | 30 | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 3 [0, 10] |
| 8 | 30 | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |

### HR@10 (%) with 95% bootstrap interval over personas (10,000 resamples, seed 42)

| cell | n | vanilla | no_memory | frozen | live |
| --- | --- | --- | --- | --- | --- |
| 1 | 30 | 3 [0, 10] | 0 [0, 0] | 0 [0, 0] | 3 [0, 10] |
| 2 | 30 | 3 [0, 10] | 3 [0, 10] | 3 [0, 10] | 7 [0, 17] |
| 3 | 30 | 3 [0, 10] | 3 [0, 10] | 7 [0, 17] | 7 [0, 17] |
| 4 | 30 | 7 [0, 17] | 3 [0, 10] | 7 [0, 17] | 3 [0, 10] |
| 5 | 30 | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |
| 6 | 30 | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |
| 7 | 30 | 0 [0, 0] | 0 [0, 0] | 3 [0, 10] | 3 [0, 10] |
| 8 | 30 | 3 [0, 10] | 7 [0, 17] | 7 [0, 17] | 7 [0, 17] |

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
| 1 | 4 | 0 | 0 | 0 | 25 | +25.0 pp [+0.0, +75.0], McNemar p=1.000 (1 vs 0) |
| 2 | 5 | 0 | 0 | 0 | 20 | +20.0 pp [+0.0, +60.0], McNemar p=1.000 (1 vs 0) |
| 3 | 5 | 0 | 0 | 0 | 0 | +0.0 pp [+0.0, +0.0], McNemar p=1.000 (0 vs 0) |
| 4 | 1 | 0 | 0 | 0 | 0 | +0.0 pp [+0.0, +0.0], McNemar p=1.000 (0 vs 0) |
| 5 | 0 | - | - | - | - | - |
| 6 | 4 | 0 | 0 | 0 | 0 | +0.0 pp [+0.0, +0.0], McNemar p=1.000 (0 vs 0) |
| 7 | 20 | 0 | 0 | 5 | 5 | +0.0 pp [+0.0, +0.0], McNemar p=1.000 (0 vs 0) |
| 8 | 3 | 0 | 0 | 0 | 0 | +0.0 pp [+0.0, +0.0], McNemar p=1.000 (0 vs 0) |

### Checks

| check | result |
| --- | --- |
| cell 5 (no reveal) stays at zero: live HR@10 = 0 and HR@5 = 0 | holds (live HR@10 0%, HR@5 0%) |
| cell 4 (reveal before gold) not below cells 1-3 at live HR@10 | holds (cell 4 3%, cells 1-3 [3, 7, 7]) |
| cells 1-3 pooled: live minus frozen @10 | +2.2 pp [+0.0, +5.6], McNemar p=0.500 |
| cells 1-3 pooled: live HR@10 (run 0) | 6% (3%) |

### Vanilla leak check

A cell where Vanilla succeeds means a query word reached the gold chunk through the text; expected to fail everywhere (HR near 0), as in run 0.

### Hand-off

Paste this report back. Step 5 writes the final tables and the Google Doc text.

## Anything unexpected

Nothing unexpected.

## Files written

- `/home/iataghav/data/SAADI/outputs/method_update/p1_ledger/per_persona.jsonl`
