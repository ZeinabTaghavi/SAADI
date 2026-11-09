# Issue 1 report, run 1b (profile p1b_ledger)

**Run 1b: bridge 24 of 90 in cells 1-3 (run 1: 14/90); retrieval flat at fixed weights.**

## R1. Candidate net (stage 01)

- Same model, urchade/gliner_large-v2.1; threshold 0.3 instead of 0.5.
- Names caught: form 1 97%, form 2 83% (run 1 at 0.5: 89%, 80%); candidates per chunk 10.0 (run 1: 9.0).
- Prompt text unchanged from run 1; recall no longer depends on the candidate list.

## R2. Build (stage 02)

- Calls per book: equal to run 1. Tokens per call: input 5159 (run 1: 4991), output 1134 (run 1: 1128).
- Wall clock 3.84 h (run 1: 3.69 h). Records 2938, stubs 1005. Recall cap 8 hit on 77% of chunks.
- Redirected JSON records 2991; auto-joins 633; SAME AS rejected 2192 ({'no_shared_token_or_sentence': 1288, 'type_conflict': 559, 'title_conflict': 345}); aliases rejected 10176.

## R3. Memory check (stage 03)

- Bridge formed, cells 1-3: 24/90 (run 1: 14/90, run 0: 1/90).
- Joins into the gold record: 34 of 52 (run 1: 24 of 49).
- Earlier record in the prompt at the reveal: 121/174 (run 1: 82/159).
- Title conflicts on the real characters: 6 (run 1: 120). Merge audit sample: in FINAL.md, to be read.

## R4. Retrieval (stage 04)

- Live HR@10 cells 1-3: 6% against frozen 3%; live minus frozen +2.2 pp [-2.2, +6.7], McNemar p=0.625.
- Cell 5 at zero: yes. Cell 4 not below cells 1-3: yes.
- Fusion landscape (information only): see FINAL.md.

## R5. Conclusion (stage 05)

- The link formed in 24 of 90 characters of cells 1-3 (run 1: 14/90, run 0: 1/90); 34 of 52 joins went into the gold record, and the earlier record was in the prompt at the reveal in 121 of 174 cases where it existed (run 1: 82 of 159).
- Retrieval did not clearly follow: live HR@10 in cells 1-3 is 6% against frozen 3% (live minus frozen +2.2 pp, 95% CI [-2.2, +6.7], McNemar p=0.625); run 1 live was 6%.
- It cost 5159 input and 1134 output tokens per call (run 1: 4991 and 1128), 3.84 h on 4 GPUs (run 1: 3.69 h), and the same number of calls.
- Open problems: 18 joins in cells 1-3 still put form 1 into a separate form-2 record; form 1 sits there as {'description': 34, 'alias': 5}. The optional JSON-alias join was off. Earlier record not in the prompt at the link step: 32 cut by the cap of 8, 21 not matched by recall. SAME AS rejected by rule 3: {'no_shared_token_or_sentence': 1288, 'type_conflict': 559, 'title_conflict': 345}; the ten-row sample in stage 03 shows whether the rule cut true joins. 6 records still join one stripped name under different titles (run 1: 120). 2 replies hit the output cap with no JSON (more candidates at 0.3 lengthen the decision block). Fusion: the memory channel's ranking does not reach the top 10 at the published weight; the landscape above is information only (Issue 6).
