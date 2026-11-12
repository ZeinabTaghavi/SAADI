# C8 step 04 - NovelHopQA, every book read whole with dual chunk attention: FAILED

Written 2026-11-11 22:37:59; commit b0842282f51d; host eee-s132-infsys-gpu03; 0 s.

## Error

```text
Traceback (most recent call last):
  File "/home/iataghav/data/SAADI/experiments/rebuttal/c8_long_context/scripts/step04_dca.py", line 184, in do_score
    raise RuntimeError("no full-document answers with dual chunk attention yet (dca/shard_*/answers/novelhopqa/full_doc.jsonl)")
RuntimeError: no full-document answers with dual chunk attention yet (dca/shard_*/answers/novelhopqa/full_doc.jsonl)
```

## Files written

- `experiments/rebuttal/c8_long_context/reports/step_04_dca_score.md`

## Push

```
pushed 1842cbd to origin/fix/notebook-key-and-install: C8 novelhopqa 04_dca_score: FAILED: RuntimeError: no full-document answers with dual chunk attention yet (dca/shard_*/answers/novelhopqa/full_doc.jsonl)
```
