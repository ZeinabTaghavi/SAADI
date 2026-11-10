# C4 sub-document control - step 03: BM25 ranks inside the gold sub-document (FAILED)

Command (from the repository root on the server):

```bash
bash experiments/rebuttal/c4_subdoc_control/sh/03_bm25.sh
```

Seed 42. CPU only. Run time 0 s.
Pairs measured: 8. Main query set: R (set with `C4_MAIN_SET`; both sets are always computed). Vanilla = the text channel alone; SAADI = the published fusion (lambda_M=0.5, lambda_S=0.3, mu=(1.0, 1.0, 1.0), from `configs/methods/sadi.yaml`).

## 1. Query sets

| Setting | Retriever | Eligible (paper) | Set S | Set R | S with > 10 chunks | R with > 10 chunks | S: inside relabel != paper labels | Chunk-count mismatches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |


## 2a. Set R: full ranking (A), sub-document (C), inside, keep scores (B)

| Setting | Retriever | Method | n | Full HR@5 | Full HR@10 | Full median rank | Sub-doc hit@10 | Top-10 share from gold sub-doc | In HR@1 | In HR@3 | In HR@5 | In HR@10 | In HR@10 (>10 chunks) | In MRR | In median rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |


## 2b. Set S: full ranking (A), sub-document (C), inside, keep scores (B)

| Setting | Retriever | Method | n | Full HR@5 | Full HR@10 | Full median rank | Sub-doc hit@10 | Top-10 share from gold sub-doc | In HR@1 | In HR@3 | In HR@5 | In HR@10 | In HR@10 (>10 chunks) | In MRR | In median rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |


## 3. Set R: HR@k on queries whose gold sub-document has more than k chunks (keep / renorm)

| Setting | Retriever | Method | HR@1 (>1) | HR@3 (>3) | HR@5 (>5) | HR@10 (>10) | MRR |
| --- | --- | --- | --- | --- | --- | --- | --- |

keep = full-document scores, chunks outside the gold sub-document dropped (main). renorm = every channel z-scored again over the gold sub-document's chunks only. Vanilla's ranking is the same under both (z-scoring one channel does not change its order). All values in percent except ranks (1-based). The full-ranking columns use the paper's labels on the whole document.

## Traceback (musique_32k__bm25)

```
Traceback (most recent call last):
  File "/mnt/data/zeinab/SAADI/experiments/rebuttal/c4_subdoc_control/scripts/step03_measures.py", line 44, in _job
    return compute_pair(ALL_SETTINGS[key], retriever, cfg, log=lambda m: print(m, flush=True))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/SAADI/experiments/rebuttal/c4_subdoc_control/scripts/c4_measures.py", line 117, in compute_pair
    queries = load_query_arrays(sp, qp, retriever)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/SAADI/src/saadi/rebuttal/heldout_fusion.py", line 769, in load_query_arrays
    frame = pd.read_parquet(stream_path)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/pandas/io/parquet.py", line 671, in read_parquet
    return impl.read(
           ^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/pandas/io/parquet.py", line 253, in read
    path_or_handle, handles, filesystem = _get_path_or_handle(
                                          ^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/pandas/io/parquet.py", line 141, in _get_path_or_handle
    handles = get_handle(
              ^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/pandas/io/common.py", line 935, in get_handle
    handle = open(handle, ioargs.mode)
             ^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/mnt/data/zeinab/SAADI/experiments/rebuttal/c4_subdoc_control/work/streams/musique_32k__bm25.parquet'
```

## Traceback (musique_64k__bm25)

```
Traceback (most recent call last):
  File "/mnt/data/zeinab/SAADI/experiments/rebuttal/c4_subdoc_control/scripts/step03_measures.py", line 44, in _job
    return compute_pair(ALL_SETTINGS[key], retriever, cfg, log=lambda m: print(m, flush=True))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/SAADI/experiments/rebuttal/c4_subdoc_control/scripts/c4_measures.py", line 117, in compute_pair
    queries = load_query_arrays(sp, qp, retriever)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/SAADI/src/saadi/rebuttal/heldout_fusion.py", line 769, in load_query_arrays
    frame = pd.read_parquet(stream_path)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/pandas/io/parquet.py", line 671, in read_parquet
    return impl.read(
           ^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/pandas/io/parquet.py", line 253, in read
    path_or_handle, handles, filesystem = _get_path_or_handle(
                                          ^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/pandas/io/parquet.py", line 141, in _get_path_or_handle
    handles = get_handle(
              ^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/pandas/io/common.py", line 935, in get_handle
    handle = open(handle, ioargs.mode)
             ^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/mnt/data/zeinab/SAADI/experiments/rebuttal/c4_subdoc_control/work/streams/musique_64k__bm25.parquet'
```

## Traceback (musique_128k__bm25)

```
Traceback (most recent call last):
  File "/mnt/data/zeinab/SAADI/experiments/rebuttal/c4_subdoc_control/scripts/step03_measures.py", line 44, in _job
    return compute_pair(ALL_SETTINGS[key], retriever, cfg, log=lambda m: print(m, flush=True))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/SAADI/experiments/rebuttal/c4_subdoc_control/scripts/c4_measures.py", line 117, in compute_pair
    queries = load_query_arrays(sp, qp, retriever)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/SAADI/src/saadi/rebuttal/heldout_fusion.py", line 769, in load_query_arrays
    frame = pd.read_parquet(stream_path)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/pandas/io/parquet.py", line 671, in read_parquet
    return impl.read(
           ^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/pandas/io/parquet.py", line 253, in read
    path_or_handle, handles, filesystem = _get_path_or_handle(
                                          ^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/pandas/io/parquet.py", line 141, in _get_path_or_handle
    handles = get_handle(
              ^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/pandas/io/common.py", line 935, in get_handle
    handle = open(handle, ioargs.mode)
             ^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/mnt/data/zeinab/SAADI/experiments/rebuttal/c4_subdoc_control/work/streams/musique_128k__bm25.parquet'
```

## Traceback (musique_256k__bm25)

```
Traceback (most recent call last):
  File "/mnt/data/zeinab/SAADI/experiments/rebuttal/c4_subdoc_control/scripts/step03_measures.py", line 44, in _job
    return compute_pair(ALL_SETTINGS[key], retriever, cfg, log=lambda m: print(m, flush=True))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/SAADI/experiments/rebuttal/c4_subdoc_control/scripts/c4_measures.py", line 117, in compute_pair
    queries = load_query_arrays(sp, qp, retriever)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/SAADI/src/saadi/rebuttal/heldout_fusion.py", line 769, in load_query_arrays
    frame = pd.read_parquet(stream_path)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/pandas/io/parquet.py", line 671, in read_parquet
    return impl.read(
           ^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/pandas/io/parquet.py", line 253, in read
    path_or_handle, handles, filesystem = _get_path_or_handle(
                                          ^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/pandas/io/parquet.py", line 141, in _get_path_or_handle
    handles = get_handle(
              ^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/pandas/io/common.py", line 935, in get_handle
    handle = open(handle, ioargs.mode)
             ^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/mnt/data/zeinab/SAADI/experiments/rebuttal/c4_subdoc_control/work/streams/musique_256k__bm25.parquet'
```

## Traceback (qasper_32k__bm25)

```
Traceback (most recent call last):
  File "/mnt/data/zeinab/SAADI/experiments/rebuttal/c4_subdoc_control/scripts/step03_measures.py", line 44, in _job
    return compute_pair(ALL_SETTINGS[key], retriever, cfg, log=lambda m: print(m, flush=True))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/SAADI/experiments/rebuttal/c4_subdoc_control/scripts/c4_measures.py", line 117, in compute_pair
    queries = load_query_arrays(sp, qp, retriever)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/SAADI/src/saadi/rebuttal/heldout_fusion.py", line 769, in load_query_arrays
    frame = pd.read_parquet(stream_path)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/pandas/io/parquet.py", line 671, in read_parquet
    return impl.read(
           ^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/pandas/io/parquet.py", line 253, in read
    path_or_handle, handles, filesystem = _get_path_or_handle(
                                          ^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/pandas/io/parquet.py", line 141, in _get_path_or_handle
    handles = get_handle(
              ^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/pandas/io/common.py", line 935, in get_handle
    handle = open(handle, ioargs.mode)
             ^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/mnt/data/zeinab/SAADI/experiments/rebuttal/c4_subdoc_control/work/streams/qasper_32k__bm25.parquet'
```

## Traceback (qasper_64k__bm25)

```
Traceback (most recent call last):
  File "/mnt/data/zeinab/SAADI/experiments/rebuttal/c4_subdoc_control/scripts/step03_measures.py", line 44, in _job
    return compute_pair(ALL_SETTINGS[key], retriever, cfg, log=lambda m: print(m, flush=True))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/SAADI/experiments/rebuttal/c4_subdoc_control/scripts/c4_measures.py", line 117, in compute_pair
    queries = load_query_arrays(sp, qp, retriever)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/SAADI/src/saadi/rebuttal/heldout_fusion.py", line 769, in load_query_arrays
    frame = pd.read_parquet(stream_path)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/pandas/io/parquet.py", line 671, in read_parquet
    return impl.read(
           ^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/pandas/io/parquet.py", line 253, in read
    path_or_handle, handles, filesystem = _get_path_or_handle(
                                          ^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/pandas/io/parquet.py", line 141, in _get_path_or_handle
    handles = get_handle(
              ^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/pandas/io/common.py", line 935, in get_handle
    handle = open(handle, ioargs.mode)
             ^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/mnt/data/zeinab/SAADI/experiments/rebuttal/c4_subdoc_control/work/streams/qasper_64k__bm25.parquet'
```

## Traceback (qasper_128k__bm25)

```
Traceback (most recent call last):
  File "/mnt/data/zeinab/SAADI/experiments/rebuttal/c4_subdoc_control/scripts/step03_measures.py", line 44, in _job
    return compute_pair(ALL_SETTINGS[key], retriever, cfg, log=lambda m: print(m, flush=True))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/SAADI/experiments/rebuttal/c4_subdoc_control/scripts/c4_measures.py", line 117, in compute_pair
    queries = load_query_arrays(sp, qp, retriever)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/SAADI/src/saadi/rebuttal/heldout_fusion.py", line 769, in load_query_arrays
    frame = pd.read_parquet(stream_path)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/pandas/io/parquet.py", line 671, in read_parquet
    return impl.read(
           ^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/pandas/io/parquet.py", line 253, in read
    path_or_handle, handles, filesystem = _get_path_or_handle(
                                          ^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/pandas/io/parquet.py", line 141, in _get_path_or_handle
    handles = get_handle(
              ^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/pandas/io/common.py", line 935, in get_handle
    handle = open(handle, ioargs.mode)
             ^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/mnt/data/zeinab/SAADI/experiments/rebuttal/c4_subdoc_control/work/streams/qasper_128k__bm25.parquet'
```

## Traceback (qasper_256k__bm25)

```
Traceback (most recent call last):
  File "/mnt/data/zeinab/SAADI/experiments/rebuttal/c4_subdoc_control/scripts/step03_measures.py", line 44, in _job
    return compute_pair(ALL_SETTINGS[key], retriever, cfg, log=lambda m: print(m, flush=True))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/SAADI/experiments/rebuttal/c4_subdoc_control/scripts/c4_measures.py", line 117, in compute_pair
    queries = load_query_arrays(sp, qp, retriever)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/SAADI/src/saadi/rebuttal/heldout_fusion.py", line 769, in load_query_arrays
    frame = pd.read_parquet(stream_path)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/pandas/io/parquet.py", line 671, in read_parquet
    return impl.read(
           ^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/pandas/io/parquet.py", line 253, in read
    path_or_handle, handles, filesystem = _get_path_or_handle(
                                          ^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/pandas/io/parquet.py", line 141, in _get_path_or_handle
    handles = get_handle(
              ^^^^^^^^^^^
  File "/mnt/data/zeinab/raptor/.venv/lib/python3.12/site-packages/pandas/io/common.py", line 935, in get_handle
    handle = open(handle, ioargs.mode)
             ^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/mnt/data/zeinab/SAADI/experiments/rebuttal/c4_subdoc_control/work/streams/qasper_256k__bm25.parquet'
```

## Problems (the step failed)

- musique_32k__bm25: FileNotFoundError: [Errno 2] No such file or directory: '/mnt/data/zeinab/SAADI/experiments/rebuttal/c4_subdoc_control/work/streams/musique_32k__bm25.parquet'
- musique_64k__bm25: FileNotFoundError: [Errno 2] No such file or directory: '/mnt/data/zeinab/SAADI/experiments/rebuttal/c4_subdoc_control/work/streams/musique_64k__bm25.parquet'
- musique_128k__bm25: FileNotFoundError: [Errno 2] No such file or directory: '/mnt/data/zeinab/SAADI/experiments/rebuttal/c4_subdoc_control/work/streams/musique_128k__bm25.parquet'
- musique_256k__bm25: FileNotFoundError: [Errno 2] No such file or directory: '/mnt/data/zeinab/SAADI/experiments/rebuttal/c4_subdoc_control/work/streams/musique_256k__bm25.parquet'
- qasper_32k__bm25: FileNotFoundError: [Errno 2] No such file or directory: '/mnt/data/zeinab/SAADI/experiments/rebuttal/c4_subdoc_control/work/streams/qasper_32k__bm25.parquet'
- qasper_64k__bm25: FileNotFoundError: [Errno 2] No such file or directory: '/mnt/data/zeinab/SAADI/experiments/rebuttal/c4_subdoc_control/work/streams/qasper_64k__bm25.parquet'
- qasper_128k__bm25: FileNotFoundError: [Errno 2] No such file or directory: '/mnt/data/zeinab/SAADI/experiments/rebuttal/c4_subdoc_control/work/streams/qasper_128k__bm25.parquet'
- qasper_256k__bm25: FileNotFoundError: [Errno 2] No such file or directory: '/mnt/data/zeinab/SAADI/experiments/rebuttal/c4_subdoc_control/work/streams/qasper_256k__bm25.parquet'

## Files written

- `experiments/rebuttal/c4_subdoc_control/reports/step_03.json`
- `experiments/rebuttal/c4_subdoc_control/reports/step_03.md`
