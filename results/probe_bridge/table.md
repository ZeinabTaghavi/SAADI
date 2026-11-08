# Bridge probe tables

| Cell | n | Vanilla | No memory | Frozen | Live | Live - Frozen @10 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 (A, reveal 3 chunks after the gold) | 30 | 3 / 144 | 0 / 140 | 0 / 172 | 3 / 172 | +3.3 [+0.0, +10.0] pp, McNemar p=1.000 |
| 2 (A, reveal 25% of the book after) | 30 | 3 / 124 | 3 / 142 | 3 / 148 | 3 / 142 | +0.0 [+0.0, +0.0] pp, McNemar p=1.000 |
| 3 (A, reveal 55% of the book after) | 30 | 3 / 162 | 3 / 175 | 3 / 216 | 3 / 200 | +0.0 [+0.0, +0.0] pp, McNemar p=1.000 |
| 4 (A, reveal 20% before the gold (control)) | 30 | 7 / 118 | 3 / 121 | 3 / 108 | 3 / 111 | +0.0 [+0.0, +0.0] pp, McNemar p=1.000 |
| 5 (A, no reveal (control)) | 30 | 0 / 152 | 0 / 136 | 0 / 134 | 0 / 128 | +0.0 [+0.0, +0.0] pp, McNemar p=1.000 |
| 6 (B, nickname, reveal 25% after) | 30 | 0 / 168 | 0 / 137 | 0 / 156 | 0 / 146 | +0.0 [+0.0, +0.0] pp, McNemar p=1.000 |
| 7 (C, description, reveal 25% after) | 30 | 0 / 174 | 0 / 178 | 0 / 171 | 0 / 166 | +0.0 [+0.0, +0.0] pp, McNemar p=1.000 |
| 8 (D, object, reveal 25% after) | 30 | 3 / 128 | 3 / 124 | 3 / 135 | 7 / 116 | +3.3 [+0.0, +10.0] pp, McNemar p=1.000 |

| Cells | Vanilla (expected/observed) | Frozen | Live | Pattern |
| --- | --- | --- | --- | --- |
| 1-3, 6-8 (reveal after gold) | fails / obs fails (2%) | fails / obs fails (2%) | succeeds, if the mechanism works / obs fails (3%) | DID NOT HOLD |
| 4 (reveal before gold) | fails / obs fails (7%) | succeeds / obs fails (3%) | succeeds / obs fails (3%) | DID NOT HOLD |
| 5 (no reveal) | fails / obs fails (0%) | fails / obs fails (0%) | fails / obs fails (0%) | held |

| Cell | n | no record | record without bridge word | bridge reached | reached and live hits@10 |
| --- | --- | --- | --- | --- | --- |
| 1 | 30 | 11 | 12 | 1 | 1/1 |
| 2 | 30 | 7 | 16 | 0 | - |
| 3 | 30 | 15 | 10 | 0 | - |
| 4 | 30 | 14 | 9 | 0 | - |
| 5 | 30 | 15 | 15 | 0 | - |
| 6 | 30 | 16 | 6 | 0 | - |
| 7 | 30 | 7 | 8 | 5 | 0/5 |
| 8 | 30 | 14 | 7 | 0 | - |

| Cell | placement | mean |reveal-gold| (chunks) | Vanilla | No memory | Frozen | Live |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | A, reveal 3 chunks after the gold | 3 | 3 | 0 | 0 | 3 |
| 2 | A, reveal 25% of the book after | 80 | 3 | 3 | 3 | 3 |
| 3 | A, reveal 55% of the book after | 177 | 3 | 3 | 3 | 3 |
| 4 | A, reveal 20% before the gold (control) | 64 | 7 | 3 | 3 | 3 |

| Cell | family | Vanilla | No memory | Frozen | Live | mem-rank live/frozen |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | A, reveal 25% of the book after | 3 | 3 | 3 | 3 | 200 / 170 |
| 6 | B, nickname, reveal 25% after | 0 | 0 | 0 | 0 | 128 / 138 |
| 7 | C, description, reveal 25% after | 0 | 0 | 0 | 0 | 75 / 120 |
| 8 | D, object, reveal 25% after | 3 | 3 | 3 | 7 | 120 / 214 |
