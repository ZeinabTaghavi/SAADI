"""The profile-gated hook: what the hierarchical strategy delegates to when ``ledger`` is configured.

The strategy calls :meth:`LedgerRuntime.build_prompt` and
:meth:`LedgerRuntime.apply_output_deferred` (or :meth:`apply_output`) instead of
its own bodies. Everything else (hierarchy context, fold summaries, the
word-overlap recall, anchor materialization, the state update) is the
strategy's own code, called from here, so the profile differs from the
published build only where the specification says it does.

Configuration (``strategy_kwargs.ledger`` in the profile YAML)::

    store_dir: outputs/method_update/p1_ledger/ledger   # events.jsonl per document
    candidate_net: gliner | regex                        # regex only for tests and dry runs
    candidate_model: urchade/gliner_large-v2.1
    candidate_labels: [person, place, organization, named object, title]
    candidate_threshold: 0.5
    candidate_device: auto | cpu | cuda
    candidate_window_words: 200
    candidate_max_per_chunk: 40
    recall_cap: 5            # ledger-recall slots outside the overlap slots
    description_cap: 5       # per record; append until full, then replace the weakest
    extract_max_aliases: -1  # aliases are exact and uncapped
    log_prompts: true        # keep the user prompt of every call in the event log

Profile p1b_ledger adds (all absent = the p1 path; see ledger/p1b.py)::

    join_direction: enforced     # model (p1) | enforced: redirect + auto-join
    recall: chunk_tokens         # candidates (p1) | chunk_tokens: names and aliases against the chunk text
    same_as_rule: p1b            # none (p1) | p1b: shared token or same sentence; title and type conflicts reject
    aliases_from_spans: true     # every alias written to memory is a candidate span (or its title-stripped form)
    json_alias_join: false       # optional join of a NEW candidate via its own JSON record's aliases
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from ..core.canonicalization import parse_canonicalization_response
from ..core.parsers import safe_json_from_text
from ..core.state import begin_canonicalization_update, drive_rollup_generator
from ..core.types import CanonicalizationOutput, ChunkRecord, DiscourseState
from ..entities.active_selection import tokenize_active_entity_text
from ..llm.base import ChatPrompt, GenerateParams, LLMBackend
from .accounting import parse_decisions, split_reply
from .candidate_net import build_candidate_net
from .ledger import LedgerEntry, LedgerStore, name_key, stripped_tokens
from .p1b import P1bConfig, P1bStep, recall_by_chunk_tokens
from .promotion import StepPlan, apply_decisions, cap_descriptions, reconcile
from .prompt import LEDGER_SCHEMA, LEDGER_SYSTEM_PROMPT, build_ledger_user_prompt

logger = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    "recall_cap": 5,
    "description_cap": 5,
    "extract_max_aliases": -1,
    "log_prompts": True,
    "display_max_desc": 3,
}


def ledger_config_from_strategy(strategy: Any) -> dict[str, Any] | None:
    """The ``ledger`` mapping of a strategy, or None when the profile is off (published path)."""

    raw = getattr(strategy, "ledger", None)
    if raw is None or raw is False:
        return None
    if raw is True:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("strategy_kwargs.ledger must be a mapping (or omitted for the published profile)")
    if raw.get("enabled") is False:
        return None
    cfg = dict(DEFAULTS)
    cfg.update(raw)
    if not cfg.get("store_dir"):
        raise ValueError("strategy_kwargs.ledger.store_dir is required (where events.jsonl per document is written)")
    return cfg


def _rank_by_overlap(entries: list[str], chunk_text: str) -> list[str]:
    chunk_tokens = set(tokenize_active_entity_text(chunk_text))
    scored = []
    for idx, text in enumerate(entries):
        toks = set(tokenize_active_entity_text(text))
        score = len(toks & chunk_tokens) / max(1, len(toks)) if toks else 0.0
        scored.append((-score, idx, text))
    return [t for _, _, t in sorted(scored)]


class LedgerRuntime:
    """One instance per strategy; holds the candidate net, the store and the per-chunk plans."""

    def __init__(self, strategy: Any, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.store = LedgerStore(Path(str(cfg["store_dir"])))
        self.net = build_candidate_net(cfg)
        self.recall_cap = int(cfg["recall_cap"])
        self.description_cap = int(cfg["description_cap"])
        self.extract_max_aliases = int(cfg["extract_max_aliases"])
        self.display_max_desc = int(cfg["display_max_desc"])
        self.log_prompts = bool(cfg["log_prompts"])
        self.p1b = P1bConfig.from_cfg(cfg)
        self._plans: dict[tuple[str, str], StepPlan] = {}
        self._system_written = False
        self.system_prompt = LEDGER_SYSTEM_PROMPT

    # -- helpers ---------------------------------------------------------------
    def _write_system_prompt_once(self) -> None:
        if self._system_written:
            return
        try:
            self.store.root.mkdir(parents=True, exist_ok=True)
            (self.store.root / "system_prompt.txt").write_text(self.system_prompt + "\n", encoding="utf-8")
            import json

            (self.store.root / "candidate_net.json").write_text(json.dumps(self.net.describe(), indent=1), encoding="utf-8")
        except Exception as exc:  # pragma: no cover - disk problems only
            logger.warning("Could not write the ledger system prompt: %s", exc)
        self._system_written = True

    def _overlap_group(
        self, strategy: Any, chunk: ChunkRecord, state: DiscourseState, ledger
    ) -> tuple[list[dict[str, Any]], list[str], list[str]]:
        entries = strategy._select_entities(chunk, state)  # the existing 5 slots, unchanged
        out: list[dict[str, Any]] = []
        ids: list[str] = []
        names: list[str] = []
        for entry in entries:
            name = str(entry.get("name", ""))
            ledger_entry = ledger.by_name(name)
            row: dict[str, Any] = {"id": ledger_entry.id if ledger_entry else None, "name": name}
            if ledger_entry is not None and ledger_entry.kind == "stub":
                row["kind"] = "stub"
                row["sentence"] = ledger_entry.sentence
            else:
                row["description"] = list(entry.get("description", []) or [])
                row["aliases"] = list(entry.get("aliases", []) or [])
            out.append(row)
            if ledger_entry is not None:
                ids.append(ledger_entry.id)
            names.append(name)
        return out, ids, names

    def _ledger_group(self, entries: list[LedgerEntry], state: DiscourseState, chunk_text: str) -> list[dict[str, Any]]:
        memory = {name_key(e.name): e for e in state.entity_memory}
        out: list[dict[str, Any]] = []
        for entry in entries:
            if entry.kind == "stub":
                out.append({"id": entry.id, "name": entry.name, "kind": "stub", "sentence": entry.sentence})
                continue
            rec = memory.get(name_key(entry.name))
            descriptions = list(rec.description) if rec is not None else []
            aliases = list(rec.aliases) if rec is not None else list(entry.aliases)
            if self.display_max_desc >= 0:
                descriptions = _rank_by_overlap(descriptions, chunk_text)[: self.display_max_desc]
            out.append({"id": entry.id, "name": entry.name, "kind": "record", "description": descriptions, "aliases": aliases})
        return out

    # -- the hook ---------------------------------------------------------------
    def build_prompt(self, strategy: Any, chunk: ChunkRecord, state: DiscourseState) -> ChatPrompt:
        self._write_system_prompt_once()
        doc_id, chunk_id, step = str(chunk.doc_id), str(chunk.chunk_id), int(chunk.chunk_index)
        ledger = self.store.doc(doc_id)
        if ledger.last_step >= step:
            removed = self.store.truncate(doc_id, step)
            logger.info("Ledger doc=%s: regenerating from step %d, forgot %d later events", doc_id, step, removed)
        reconcile_ops, recovered, removed_desc = reconcile(ledger, state, step=step, chunk_id=chunk_id)
        if recovered:
            logger.warning("Ledger doc=%s step=%d: %d memory names unknown to the ledger were recovered", doc_id, step, len(recovered))

        run = self.net.run(chunk.raw_text)
        candidates = []
        dropped = list(run.dropped)
        for cand in run.candidates:
            if not stripped_tokens(cand.text):
                dropped.append({"text": cand.text, "label": cand.label, "score": round(cand.score, 3), "reason": "title or stopwords only"})
                continue
            candidates.append(cand)

        overlap_rows, overlap_ids, overlap_names = self._overlap_group(strategy, chunk, state, ledger)
        recall_matches: dict[str, dict[str, Any]] = {}
        if self.p1b.chunk_token_recall:
            # p1b change 2: every ledger name and alias against the chunk text, independent of the candidate net.
            recalled, left_out, cap_hit, recall_matches = recall_by_chunk_tokens(
                ledger, chunk.raw_text, exclude_keys={name_key(n) for n in overlap_names}, cap=self.recall_cap
            )
            matched = {entry_id: [m["via"]] for entry_id, m in recall_matches.items()}
        else:
            recalled, left_out, cap_hit, matched = ledger.recall(
                [c.text for c in candidates], exclude_keys={name_key(n) for n in overlap_names}, cap=self.recall_cap
            )
        ledger_rows = self._ledger_group(recalled, state, chunk.raw_text)
        hierarchy_context = strategy._hierarchy_context(state, step)
        user = build_ledger_user_prompt(
            chunk_text=chunk.raw_text,
            hierarchy_context=hierarchy_context,
            overlap_entries=overlap_rows,
            ledger_entries=ledger_rows,
            candidates=[c.to_dict() for c in candidates],
            schema=LEDGER_SCHEMA,
        )
        self._plans[(doc_id, chunk_id)] = StepPlan(
            doc_id=doc_id,
            chunk_id=chunk_id,
            step=step,
            candidates=candidates,
            dropped_candidates=dropped,
            candidate_raw_count=run.raw_count,
            candidate_seconds=run.seconds,
            candidate_cap_hit=run.cap_hit,
            candidate_device=run.device,
            overlap_ids=overlap_ids,
            overlap_names=overlap_names,
            ledger_ids=[e.id for e in recalled],
            left_out_ids=[e.id for e in left_out],
            cap_hit=cap_hit,
            matched_candidates=matched,
            reconcile_ops=reconcile_ops,
            recovered_names=recovered,
            dropped_descriptions_removed=removed_desc,
            prompt_system=self.system_prompt,
            prompt_user=user,
            p1b=P1bStep(config=self.p1b, chunk_text=chunk.raw_text, shown_ids=[*overlap_ids, *(e.id for e in recalled)])
            if self.p1b.any
            else None,
            recall_matches=recall_matches,
        )
        return ChatPrompt(system=self.system_prompt, user=user)

    def _plan_for(self, strategy: Any, chunk: ChunkRecord, state: DiscourseState) -> StepPlan:
        key = (str(chunk.doc_id), str(chunk.chunk_id))
        plan = self._plans.pop(key, None)
        if plan is None:
            # apply without a preceding build_prompt (tests, replays): rebuild deterministically.
            self.build_prompt(strategy, chunk, state)
            plan = self._plans.pop(key)
        return plan

    def _apply(self, strategy: Any, chunk: ChunkRecord, state: DiscourseState, raw_model_text: str):
        t0 = time.time()
        plan = self._plan_for(strategy, chunk, state)
        ledger = self.store.doc(plan.doc_id)
        block = parse_decisions(raw_model_text, [c.text for c in plan.candidates])
        _decisions_text, json_text = split_reply(raw_model_text)
        output = parse_canonicalization_response(
            json_text,
            fallback_chunk_text=chunk.raw_text,
            allow_linking_sentence=False,
            max_entities=strategy.max_entities,
            max_aliases=self.extract_max_aliases,
            max_desc=self.description_cap,
            anchor_matching=strategy.anchor_matching,
            ungrounded_entities=strategy.ungrounded_entities,
        )
        output = strategy._postprocess_output(chunk=chunk, output=output)
        parsed, _status = safe_json_from_text(json_text) if json_text else (None, "failed")
        proposed = []
        if isinstance(parsed, dict):
            raw_entities = parsed.get("active_entities")
            if isinstance(raw_entities, list):
                proposed = [str(x.get("name", "")).strip() for x in raw_entities if isinstance(x, dict) and str(x.get("name", "")).strip()]
        cut_names = proposed[int(strategy.max_entities) :]

        outcome = apply_decisions(ledger=ledger, plan=plan, block=block, described=list(output.active_entities), cut_names=cut_names)
        output = CanonicalizationOutput(
            updated_chunk_text=output.updated_chunk_text,
            active_entities=outcome.active_entities,
            raw_model_text=raw_model_text,
            parser_status=output.parser_status,
            linking_sentence=output.linking_sentence,
            insertions=list(output.insertions),
            rejected_entities=list(output.rejected_entities),
        )
        new_state, gen = begin_canonicalization_update(
            state,
            output,
            chunk_step=plan.step,
            hierarchy_l0_text=output.updated_chunk_text,
            hierarchy_branching_factor=(int(strategy.branching_factor) if strategy.stores_hierarchy else None),
            hierarchy_max_level=strategy._max_level_limit(),
            raw_chunk_text=chunk.raw_text,
        )
        cap_ops, replacements = cap_descriptions(ledger, new_state.entity_memory, cap=self.description_cap, step=plan.step)

        event: dict[str, Any] = {
            "step": plan.step,
            "chunk_id": plan.chunk_id,
            "doc_id": plan.doc_id,
            "ops": [*plan.reconcile_ops, *outcome.ops, *cap_ops],
            "counts": dict(outcome.counts),
            "candidates": [c.to_dict() for c in plan.candidates],
            "dropped_candidates": plan.dropped_candidates,
            "candidate_raw_count": plan.candidate_raw_count,
            "candidate_seconds": round(plan.candidate_seconds, 4),
            "candidate_cap_hit": plan.candidate_cap_hit,
            "candidate_device": plan.candidate_device,
            "recalled_overlap_ids": plan.overlap_ids,
            "recalled_overlap_names": plan.overlap_names,
            "recalled_ledger_ids": plan.ledger_ids,
            "ledger_left_out_ids": plan.left_out_ids,
            "ledger_cap_hit": plan.cap_hit,
            "ledger_matched_candidates": plan.matched_candidates,
            "recovered_names": plan.recovered_names,
            "dropped_descriptions_removed_on_reconcile": plan.dropped_descriptions_removed,
            "decision_status": block.status,
            "decision_omitted": block.omitted,
            "decision_unmatched_lines": block.unmatched_lines,
            "decisions": outcome.decisions,
            "new_entries": outcome.new_entries,
            "promotions": outcome.promotions,
            "aliases_added": outcome.aliases_added,
            "links": outcome.links,
            "llm_additions": outcome.llm_additions,
            "described_records": outcome.described,
            "proposed_records": proposed,
            "cut_by_budget": cut_names,
            "replacements": replacements,
            "parser_status": output.parser_status,
            "memory_size": len(new_state.entity_memory),
            "prompt_system_chars": len(plan.prompt_system),
            "prompt_user_chars": len(plan.prompt_user),
            "reply_chars": len(raw_model_text or ""),
            "apply_seconds": round(time.time() - t0, 4),
        }
        if plan.p1b is not None:
            event["p1b"] = {
                "recall": "chunk_tokens" if plan.p1b.config.chunk_token_recall else "candidates",
                "recall_matches": plan.recall_matches,
                "shown_ids": list(plan.p1b.shown_ids),
                "redirects": outcome.redirects,
                "alias_rejections": outcome.alias_rejections,
            }
        if self.log_prompts:
            event["prompt_user"] = plan.prompt_user
        self.store.append_event(plan.doc_id, event)
        return output, new_state, gen

    def apply_output_deferred(self, strategy: Any, chunk: ChunkRecord, state: DiscourseState, raw_model_text: str):
        return self._apply(strategy, chunk, state, raw_model_text)

    def apply_output(
        self,
        strategy: Any,
        chunk: ChunkRecord,
        state: DiscourseState,
        raw_model_text: str,
        *,
        backend: LLMBackend | None = None,
        generate: GenerateParams | None = None,
    ):
        output, new_state, gen = self._apply(strategy, chunk, state, raw_model_text)
        if gen is not None:

            def _rollup_fn(items: list[str], lower_cycle: int, next_cycle: int, step: int) -> str | None:
                summary = strategy._rollup_with_backend(
                    items=items, lower_cycle=lower_cycle, next_cycle=next_cycle, step=step, backend=backend, generate=generate
                )
                return summary or strategy._fallback_rollup(items)

            drive_rollup_generator(gen, _rollup_fn)
        return output, new_state


__all__ = ["DEFAULTS", "LedgerRuntime", "ledger_config_from_strategy"]
