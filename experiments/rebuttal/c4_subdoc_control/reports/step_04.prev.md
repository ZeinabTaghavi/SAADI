# C4 sub-document control - step 04: dense-retriever ranks inside the gold sub-document (FAILED)

Command (from the repository root on the server):

```bash
bash experiments/rebuttal/c4_subdoc_control/sh/04_dense.sh
```

Seed 42. CPU only. Run time 0 s.

## Traceback

```
Traceback (most recent call last):
  File "/Users/uem-stag-lap-ic/Desktop/SAADI/experiments/rebuttal/c4_subdoc_control/scripts/c4_common.py", line 425, in run_step
    body(report)
    ~~~~^^^^^^^^
  File "/Users/uem-stag-lap-ic/Desktop/SAADI/experiments/rebuttal/c4_subdoc_control/scripts/step03_measures.py", line 205, in <lambda>
    return run_step(report, lambda r: body(r, args, args.family))
                                      ~~~~^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/uem-stag-lap-ic/Desktop/SAADI/experiments/rebuttal/c4_subdoc_control/scripts/step03_measures.py", line 51, in body
    s02 = require_step_ok(2)
  File "/Users/uem-stag-lap-ic/Desktop/SAADI/experiments/rebuttal/c4_subdoc_control/scripts/c4_common.py", line 442, in require_step_ok
    raise RuntimeError(f"step {step:02d} has not run ({rel(path)} missing)")
RuntimeError: step 02 has not run (experiments/rebuttal/c4_subdoc_control/reports/step_02.json missing)
```

## Problems (the step failed)

- unexpected error: RuntimeError: step 02 has not run (experiments/rebuttal/c4_subdoc_control/reports/step_02.json missing)

## Files written

- `experiments/rebuttal/c4_subdoc_control/reports/step_04.json`
- `experiments/rebuttal/c4_subdoc_control/reports/step_04.md`
