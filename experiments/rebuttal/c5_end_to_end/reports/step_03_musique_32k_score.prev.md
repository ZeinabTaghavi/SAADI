# C5 end-to-end QA - step 03 scores, MuSiQue-32K: FAILED

Written 2026-11-10 19:59:32; commit dcaf0df6203c; host eee-s132-infsys-gpu03.

Scores in percent. EM and token F1 after SQuAD normalisation (best over the gold answers); Contain = a gold answer inside the output. Evidence yes/no = the paper's hit rule (a gold chunk or a whole silver group) applied to the chunks in the context. HR@10 = the condition's retrieval hit (RAPTOR: its upper-bound HR@10; A-RAG: hit over the chunks it read; none for the cut SAADI rows, whose retrieval measure is the evidence in the cut context).


## Error

```text
Traceback (most recent call last):
  File "/home/iataghav/data/SAADI/experiments/rebuttal/c5_end_to_end/scripts/step03_score.py", line 125, in main
    raise RuntimeError(f"no answers under {paths.results / 'answers' / ds}")
RuntimeError: no answers under /home/iataghav/data/SAADI/results/rebuttal/c5_end_to_end/answers/musique_32k
```

## Files written

- `experiments/rebuttal/c5_end_to_end/reports/step_03_musique_32k_score.md`

## Push

```
pushed f95ee27 to origin/fix/notebook-key-and-install: C5 musique_32k 03_score: FAILED: RuntimeError: no answers under /home/iataghav/data/SAADI/results/rebuttal/c5_end_to_end/answers/musique_32k
```
