# Bridge probe - step 01: Select books

Status: **ok** - 2026-11-07 16:58:03

## What was done

- `bash experiments/probe_bridge/sh/01_*.sh`
- `load_novelhopqa_documents(config_name='all', books_root='/home/iataghav/data/SAADI/data/novelhopqa/book-corpus-root'); chunk_text(tokenizer=None, 500, 0)`
- Versions: python=3.12.3, platform=Linux-6.17.0-23-generic-x86_64-with-glibc2.39, numpy=2.3.5, pandas=3.0.1, yaml=6.0.3, saadi=0.1.0, git_commit=d3efbe7711c4546f232eef39d4e64d3d36c54763

## Numbers

### The 10 selected books (shortest by chunk count)

| doc_id | chunks | words | paper cache chunks | position windows feasible? |
| --- | --- | --- | --- | --- |
| B24 | 213 | 102908 | None | False |
| B31 | 244 | 117186 | None | False |
| B43 | 245 | 118850 | None | False |
| B37 | 261 | 126007 | None | False |
| B18 | 311 | 149887 | None | False |
| B04 | 320 | 156676 | None | False |
| B34 | 334 | 160345 | None | False |
| B27 | 337 | 162438 | None | False |
| B61 | 355 | 170013 | None | False |
| B26 | 383 | 183955 | None | False |

### Totals

Books loaded: 18. Selected: 10. Total chunks to index: **3003**. QA entries in the question set: 1841.

## Anything unexpected

These books have fewer than 450 chunks: ['B24', 'B31', 'B43', 'B37', 'B18', 'B04', 'B34', 'B27', 'B61', 'B26']. The note's position windows (21 golds in [0.10N, 0.30N], 3 chunks apart) cannot all fit there. STOP: decide at this hand-off whether to widen the windows or relax the spacing before step 2.

## Files written

- `/mnt/data/zeinab/SAADI/experiments/probe_bridge/spec/books.json`
- `/mnt/data/zeinab/SAADI/experiments/probe_bridge/spec/question_set.txt`
