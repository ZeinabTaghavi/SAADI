# Bridge probe - step 07: Retrieve and score

Status: **ok** - 2026-11-08 09:43:22

## What was done

- `bash experiments/probe_bridge/sh/07_*.sh`
- `score_mbr_variant_document_pools + combine (published settings) per persona; frozen via saadi.rebuttal.frozen_records; vanilla = the text channel alone`
- Versions: python=3.12.3, platform=Linux-6.17.0-23-generic-x86_64-with-glibc2.39, numpy=2.3.5, pandas=3.0.1, yaml=6.0.3, saadi=0.1.0, git_commit=2ae7fd656b63eff028448b60c3e9133eeecd2cae

## Numbers

### HR@10 (%) / median gold rank, per cell

| cell | n | Vanilla | No memory | Frozen | Live | Live-Frozen @10 | Live-Vanilla @10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 30 | 3 / 144 | 0 / 140 | 0 / 172 | 3 / 172 | +3.3 pp [+0.0, +10.0], McNemar p=1.000 (1 vs 0) | +0.0 pp [-10.0, +10.0], McNemar p=1.000 (1 vs 1) |
| 2 | 30 | 3 / 124 | 3 / 142 | 3 / 148 | 3 / 142 | +0.0 pp [+0.0, +0.0], McNemar p=1.000 (0 vs 0) | +0.0 pp [+0.0, +0.0], McNemar p=1.000 (0 vs 0) |
| 3 | 30 | 3 / 162 | 3 / 175 | 3 / 216 | 3 / 200 | +0.0 pp [+0.0, +0.0], McNemar p=1.000 (0 vs 0) | +0.0 pp [+0.0, +0.0], McNemar p=1.000 (0 vs 0) |
| 4 | 30 | 7 / 118 | 3 / 121 | 3 / 108 | 3 / 111 | +0.0 pp [+0.0, +0.0], McNemar p=1.000 (0 vs 0) | -3.3 pp [-10.0, +0.0], McNemar p=1.000 (0 vs 1) |
| 5 | 30 | 0 / 152 | 0 / 136 | 0 / 134 | 0 / 128 | +0.0 pp [+0.0, +0.0], McNemar p=1.000 (0 vs 0) | +0.0 pp [+0.0, +0.0], McNemar p=1.000 (0 vs 0) |
| 6 | 30 | 0 / 168 | 0 / 137 | 0 / 156 | 0 / 146 | +0.0 pp [+0.0, +0.0], McNemar p=1.000 (0 vs 0) | +0.0 pp [+0.0, +0.0], McNemar p=1.000 (0 vs 0) |
| 7 | 30 | 0 / 174 | 0 / 178 | 0 / 171 | 0 / 166 | +0.0 pp [+0.0, +0.0], McNemar p=1.000 (0 vs 0) | +0.0 pp [+0.0, +0.0], McNemar p=1.000 (0 vs 0) |
| 8 | 30 | 3 / 128 | 3 / 124 | 3 / 135 | 7 / 116 | +3.3 pp [+0.0, +10.0], McNemar p=1.000 (1 vs 0) | +3.3 pp [-6.7, +13.3], McNemar p=1.000 (2 vs 1) |

### HR@5 (%) and the memory-channel-only median rank of the gold chunk

| cell | n | Vanilla | No memory | Frozen | Live | mem-rank live | mem-rank frozen |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 30 | 0 | 0 | 0 | 0 | 136 | 193 |
| 2 | 30 | 3 | 0 | 0 | 0 | 200 | 170 |
| 3 | 30 | 0 | 0 | 0 | 0 | 174 | 190 |
| 4 | 30 | 0 | 0 | 0 | 0 | 120 | 124 |
| 5 | 30 | 0 | 0 | 0 | 0 | 175 | 192 |
| 6 | 30 | 0 | 0 | 0 | 0 | 128 | 138 |
| 7 | 30 | 0 | 0 | 0 | 0 | 75 | 120 |
| 8 | 30 | 0 | 0 | 3 | 3 | 120 | 214 |

### HR@5 (%) with 95% bootstrap interval over personas

| cell | n | vanilla | no_memory | frozen | live |
| --- | --- | --- | --- | --- | --- |
| 1 | 30 | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |
| 2 | 30 | 3 [0, 10] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |
| 3 | 30 | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |
| 4 | 30 | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |
| 5 | 30 | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |
| 6 | 30 | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |
| 7 | 30 | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |
| 8 | 30 | 0 [0, 0] | 0 [0, 0] | 3 [0, 10] | 3 [0, 10] |

### HR@10 (%) with 95% bootstrap interval over personas

| cell | n | vanilla | no_memory | frozen | live |
| --- | --- | --- | --- | --- | --- |
| 1 | 30 | 3 [0, 10] | 0 [0, 0] | 0 [0, 0] | 3 [0, 10] |
| 2 | 30 | 3 [0, 10] | 3 [0, 10] | 3 [0, 10] | 3 [0, 10] |
| 3 | 30 | 3 [0, 10] | 3 [0, 10] | 3 [0, 10] | 3 [0, 10] |
| 4 | 30 | 7 [0, 17] | 3 [0, 10] | 3 [0, 10] | 3 [0, 10] |
| 5 | 30 | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |
| 6 | 30 | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |
| 7 | 30 | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |
| 8 | 30 | 3 [0, 10] | 3 [0, 10] | 3 [0, 10] | 7 [0, 17] |

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

| cell | n | Vanilla | No memory | Frozen | Live | Live-Frozen @10 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0 | 0 | 0 | 100 | +100.0 pp [+100.0, +100.0], McNemar p=1.000 (1 vs 0) |
| 2 | 0 | - | - | - | - | - |
| 3 | 0 | - | - | - | - | - |
| 4 | 0 | - | - | - | - | - |
| 5 | 0 | - | - | - | - | - |
| 6 | 0 | - | - | - | - | - |
| 7 | 5 | 0 | 0 | 0 | 0 | +0.0 pp [+0.0, +0.0], McNemar p=1.000 (0 vs 0) |
| 8 | 0 | - | - | - | - | - |

### Vanilla leak check

A cell where Vanilla succeeds means a query word reached the gold chunk through the text; see the Vanilla column (expected: fails everywhere, HR near 0).

## Anything unexpected

Nothing unexpected.

## Files written

- `/mnt/data/zeinab/SAADI/outputs/probe_bridge/per_persona.jsonl`
