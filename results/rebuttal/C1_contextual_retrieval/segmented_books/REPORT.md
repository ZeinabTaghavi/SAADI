# C1 x C5: NovelHopQA books Contextual Retrieval had to cut vs books it saw whole

Written 2026-11-11 20:47:18 by `scripts/rebuttal/c1_segmented_books.py`, from C5's saved answers (BM25 top 10, one reader) and C1's census. Inputs from origin/fix/notebook-key-and-install @ 145afe5. The split is CR's own window rule (262,144 tokens), fixed before any result. Cut books: B62 (462K tokens, 2 segments), B26 (268K tokens, 2 segments), B42 (763K tokens, 3 segments), B54 (440K tokens, 2 segments), B44 (267K tokens, 2 segments), B15 (294K tokens, 2 segments), B05 (345K tokens, 2 segments).

## Means per group (%)

| Group | Books | Questions | Measure | SAADI | CR | Vanilla BM25 |
| --- | --- | --- | --- | --- | --- | --- |
| Books CR had to cut (over the 262K window) | 7 | 432 | F1 | 31.4 | 30.3 | 30.7 |
| Books CR had to cut (over the 262K window) | 7 | 432 | EM | 7.2 | 7.6 | 6.9 |
| Books CR had to cut (over the 262K window) | 7 | 432 | Containment | 21.1 | 21.3 | 20.6 |
| Books CR had to cut (over the 262K window) | 7 | 432 | HR@10 (paper labels) | 20.8 | 20.4 | 19.4 |
| Books CR saw whole | 9 | 476 | F1 | 31.6 | 29.4 | 30.9 |
| Books CR saw whole | 9 | 476 | EM | 8.0 | 6.7 | 8.4 |
| Books CR saw whole | 9 | 476 | Containment | 23.9 | 22.5 | 23.1 |
| Books CR saw whole | 9 | 476 | HR@10 (paper labels) | 23.1 | 22.9 | 21.8 |

## Paired differences (points, 95% bootstrap over questions; **bold** = excludes 0)

| Measure | Pair | Cut books | Whole books | Cut minus whole (interaction) | Cut books, book-level | Whole books, book-level |
| --- | --- | --- | --- | --- | --- | --- |
| F1 | SAADI − CR | +1.0 [-0.6, +2.7] | **+2.2 [+0.5, +4.0]** | -1.2 [-3.6, +1.2] | +1.0 [-1.1, +3.1] | **+2.2 [+0.2, +4.2]** |
| F1 | SAADI − Vanilla BM25 | +0.6 [-1.0, +2.2] | +0.7 [-0.9, +2.4] | -0.1 [-2.4, +2.2] | +0.6 [-1.2, +2.4] | +0.7 [-0.8, +2.5] |
| F1 | CR − Vanilla BM25 | -0.4 [-2.1, +1.2] | -1.5 [-3.4, +0.5] | +1.0 [-1.6, +3.7] | -0.4 [-3.1, +1.6] | -1.5 [-4.1, +0.9] |
| EM | SAADI − CR | -0.5 [-1.9, +0.7] | +1.3 [-0.2, +2.9] | -1.7 [-3.7, +0.2] | -0.5 [-1.5, +0.6] | +1.3 [+0.0, +2.7] |
| EM | SAADI − Vanilla BM25 | +0.2 [-0.9, +1.4] | -0.4 [-1.9, +1.1] | +0.7 [-1.3, +2.6] | +0.2 [-0.8, +1.1] | -0.4 [-2.1, +1.2] |
| EM | CR − Vanilla BM25 | +0.7 [-0.7, +2.1] | **-1.7 [-3.4, -0.2]** | **+2.4 [+0.4, +4.5]** | +0.7 [-0.3, +1.3] | -1.7 [-3.3, +0.0] |
| Containment | SAADI − CR | -0.2 [-3.0, +2.5] | +1.5 [-0.8, +3.8] | -1.7 [-5.4, +2.0] | -0.2 [-3.0, +2.4] | +1.5 [-0.4, +3.5] |
| Containment | SAADI − Vanilla BM25 | +0.5 [-1.9, +2.8] | +0.8 [-1.1, +2.7] | -0.4 [-3.4, +2.5] | +0.5 [-2.0, +2.4] | +0.8 [-0.8, +2.6] |
| Containment | CR − Vanilla BM25 | +0.7 [-1.6, +3.0] | -0.6 [-3.2, +1.9] | +1.3 [-2.0, +4.9] | +0.7 [-1.0, +1.9] | -0.6 [-3.4, +2.0] |
| HR@10 (paper labels) | SAADI − CR | +0.5 [-1.4, +2.3] | +0.2 [-1.9, +2.3] | +0.3 [-2.5, +3.0] | +0.5 [-1.0, +2.2] | +0.2 [-2.6, +2.3] |
| HR@10 (paper labels) | SAADI − Vanilla BM25 | +1.4 [-0.2, +3.2] | +1.3 [-0.2, +2.7] | +0.1 [-2.1, +2.4] | **+1.4 [+0.6, +2.4]** | **+1.3 [+0.2, +2.3]** |
| HR@10 (paper labels) | CR − Vanilla BM25 | +0.9 [-0.9, +2.8] | +1.1 [-1.1, +3.2] | -0.1 [-3.0, +2.7] | +0.9 [-0.3, +2.3] | +1.1 [-0.5, +2.9] |

## Per book (F1 %)

| Book | Group | Segments | Questions | SAADI | CR | Vanilla BM25 | SAADI − CR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B05 | cut | 2 | 22 | 34.4 | 28.3 | 34.2 | +6.1 |
| B15 | cut | 2 | 75 | 24.0 | 27.1 | 25.7 | -3.1 |
| B26 | cut | 2 | 50 | 33.5 | 35.4 | 35.4 | -1.9 |
| B42 | cut | 3 | 70 | 38.0 | 33.9 | 39.1 | +4.1 |
| B44 | cut | 2 | 57 | 29.1 | 26.0 | 28.6 | +3.1 |
| B54 | cut | 2 | 74 | 25.2 | 25.8 | 22.6 | -0.7 |
| B62 | cut | 2 | 84 | 37.3 | 34.6 | 33.3 | +2.7 |
| B04 | whole | 1 | 45 | 39.8 | 37.8 | 35.5 | +2.1 |
| B12 | whole | 1 | 61 | 28.6 | 25.0 | 23.3 | +3.6 |
| B18 | whole | 1 | 44 | 34.3 | 34.0 | 32.8 | +0.4 |
| B24 | whole | 1 | 65 | 35.1 | 29.5 | 35.4 | +5.6 |
| B27 | whole | 1 | 60 | 29.4 | 31.6 | 31.7 | -2.2 |
| B34 | whole | 1 | 65 | 28.3 | 22.1 | 29.4 | +6.2 |
| B37 | whole | 1 | 62 | 34.3 | 33.4 | 34.3 | +0.9 |
| B43 | whole | 1 | 60 | 27.2 | 27.9 | 26.1 | -0.7 |
| B61 | whole | 1 | 14 | 26.4 | 21.2 | 31.9 | +5.2 |

SAADI's F1 is above CR's in 4 of 7 cut books and 7 of 9 whole books.

Notes: HR@10 uses the paper's NovelHopQA evidence labels (label issue still open); F1, EM and containment do not depend on them. Questions cluster in books (7 cut, 9 whole), so the book-level bootstrap is the conservative interval.
