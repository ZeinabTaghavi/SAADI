# Bridge probe - step 01: Select books

Status: **ok** - 2026-11-07 17:15:02

## What was done

- `bash experiments/probe_bridge/sh/01_*.sh --force`
- `load_novelhopqa_documents(config_name='all', books_root='/home/iataghav/data/SAADI/data/novelhopqa/book-corpus-root'); chunk_text(tokenizer=None, 500, 0)`
- Versions: python=3.12.3, platform=Linux-6.17.0-23-generic-x86_64-with-glibc2.39, numpy=2.3.5, pandas=3.0.1, yaml=6.0.3, saadi=0.1.0, git_commit=60cc1286d242b67608b6cadf3678c24856da74df

## Numbers

### The 10 selected books (shortest by chunk count)

| doc_id | chunks | words | paper cache chunks | position windows feasible? |
| --- | --- | --- | --- | --- |
| B31 | 244 | 117186 | None | True |
| B43 | 245 | 118850 | None | True |
| B37 | 261 | 126007 | None | True |
| B18 | 311 | 149887 | None | True |
| B04 | 320 | 156676 | None | True |
| B34 | 334 | 160345 | None | True |
| B27 | 337 | 162438 | None | True |
| B61 | 355 | 170013 | None | True |
| B26 | 383 | 183955 | None | True |
| B44 | 384 | 185356 | None | True |

### Totals

Books loaded: 18. Selected: 10. Total chunks to index: **3174**. QA entries in the question set: 1841.

### Skipped books

Too short for geometry v2 (under 240 chunks): ['B24']. The next-shortest feasible books take their places.

## Anything unexpected

Nothing unexpected.

## Files written

- `/mnt/data/zeinab/SAADI/experiments/probe_bridge/spec/books.json`
- `/mnt/data/zeinab/SAADI/experiments/probe_bridge/spec/question_set.txt`
