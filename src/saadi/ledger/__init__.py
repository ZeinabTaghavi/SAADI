"""Candidate-first ledger (construction profile ``p1_ledger``).

Fix for Issue 1 of the method update tracker: minor entities were never
recorded, cut by the 5-record budget, or not recalled. Four mechanisms, all
inside the existing one-LLM-call-per-chunk loop:

- :mod:`candidate_net`: a small open-type NER model lists every named span in
  the chunk before the LLM call (a floor, not a ceiling).
- :mod:`ledger`: the per-document ledger of records and stubs, name lookup
  (ledger recall) with title stripping, and the append-only event log that
  makes the build resumable and auditable.
- :mod:`accounting`: the forced-accounting parser (one decision per candidate:
  ``NEW`` | ``SAME AS <id> (<reason>)`` | ``NOT AN ENTITY``).
- :mod:`promotion`: decisions -> stubs, promotions, aliases, chunk links, and
  the description cap with logged replacements.
- :mod:`prompt`: the exact prompt text of the LLM call.
- :mod:`runtime`: the profile-gated hook used by the hierarchical strategy
  (inert when the profile is off).
- :mod:`measure`: the frozen-records extension for the probe measurement
  (stubs count as records; creation and promotion steps are known).
- :mod:`p1b`: profile ``p1b_ledger`` (run 1b): join direction enforced in
  code, recall by chunk tokens, the SAME AS acceptance rule and alias hygiene.
  Off unless its keys are set, so ``p1_ledger`` is unchanged.

Nothing in this package changes the published code path; the hierarchical
strategy only branches here when ``strategy_kwargs.ledger`` is configured.
"""

from .accounting import Decision, DecisionBlock, parse_decisions
from .candidate_net import Candidate, CandidateNet, RegexCandidateNet
from .ledger import DocumentLedger, LedgerEntry, LedgerStore, name_tokens, stripped_tokens
from .runtime import LedgerRuntime, ledger_config_from_strategy

__all__ = [
    "Candidate",
    "CandidateNet",
    "Decision",
    "DecisionBlock",
    "DocumentLedger",
    "LedgerEntry",
    "LedgerRuntime",
    "LedgerStore",
    "RegexCandidateNet",
    "ledger_config_from_strategy",
    "name_tokens",
    "parse_decisions",
    "stripped_tokens",
]
