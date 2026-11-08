# B4 LQCA

The official LQCA code (OceannTwT/LQCA, pinned commit, cloned at run time because the
repository has no license file) rewrites each book so that coreferent mentions take their
representative mention, using the Maverick coreference model; the index is BM25 over the
rewritten chunks, and results are returned as the original chunk ids.

The code builds an n-by-n matrix over every mention of its input and runs a Floyd pass over
it, which does not finish on a whole book. As decided with the user, `build.py` calls the
official functions unchanged on consecutive segments of 8 chunks and splices the rewrites
back; coreference across segments is lost, which is recorded as a deviation. Two more
departures are unavoidable: the Maverick checkpoint is loaded from Hugging Face (the code
hard-codes the author's local path) on a GPU, and a segment whose pipeline raises keeps its
original text and is counted (the official loop skips it silently). LQCA's rewrite uses no
LLM. `retrieve.py` also counts whether the rewritten gold chunk now contains form 2.

**Environment pins found in the local dry run (8 Nov 2026).** maverick-coref 1.0.7's checkpoint
does not load under torch 2.6 or later (`torch.load` now defaults to `weights_only=True`), and
current transformers requires torch 2.6, so the B4 venv pins `torch==2.5.1` and
`transformers==4.45.2` (maverick-coref asks for transformers >= 4.34). On one segment of B31
(8 chunks, 5,639 tokens, 976 mentions) the official pipeline took 71 s on a laptop CPU, almost
all of it Maverick; the Floyd pass took 1.8 s. All 8 chunks came back through the marker split.

**Observed behaviour of the official code, kept as is.** `replace_to_golden_form` joins the
representative mention's tokens without the leading space of the replaced span, so some
replacements glue onto the previous word ("But I" became "Butme" in that segment). This is
the code's own output; it is counted, not repaired, and it costs BM25 matches on those words.
