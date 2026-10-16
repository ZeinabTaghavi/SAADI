"""Run one chunk through a prompt profile and confirm the outputs parse."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..context_strategies.hierarchical_summary import HierarchicalSummaryStrategy
from ..core.canonicalization import parse_canonicalization_response
from ..core.parsers import safe_json_from_text
from ..core.state import init_discourse_state
from ..core.types import ChunkRecord
from ..llm.base import GenerateParams
from ..llm.factory import resolve_backend
from .contract import validate_entity_guidance, validate_summary_guidance
from .profiles import PromptProfile, load_prompt_profile

SAMPLE_TEXT = (
    "The Aurora research team was founded in 2024 by Mira Chen, a limnologist who studies water quality "
    "in alpine lakes. Its first station opened beside Lake Alder. Mira appointed Leon Park to maintain the "
    "station; he records the water temperature every morning and sends the readings to the Aurora laboratory "
    "in Brennholz, where Sofia Marin checks them each month."
)
ENTITY_STOP = ["[END_JSON]", "<|im_end|>", "<|endoftext|>"]


@dataclass
class PromptCheckReport:
    profile: str
    directory: str
    content_hash: str
    backend: str
    entity_status: str = "failed"  # ok | fallback | failed
    entity_count: int = 0
    insertion_count: int = 0
    entity_names: list[str] = field(default_factory=list)
    summary_status: str = "failed"  # ok | failed
    summary_text: str = ""
    raw_entity_output: str = ""
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.entity_status != "failed" and self.summary_status == "ok" and not self.problems

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "directory": self.directory,
            "content_hash": self.content_hash,
            "backend": self.backend,
            "entity_status": self.entity_status,
            "entity_count": self.entity_count,
            "insertion_count": self.insertion_count,
            "entity_names": list(self.entity_names),
            "summary_status": self.summary_status,
            "summary_text": self.summary_text,
            "problems": list(self.problems),
            "notes": list(self.notes),
            "ok": self.ok,
        }


SAMPLE_FILE = "sample.txt"


def _profile_sample(profile: PromptProfile) -> str | None:
    """A profile may ship ``sample.txt`` so the check uses text from its own domain."""

    path = profile.directory / SAMPLE_FILE
    if path.is_file():
        content = path.read_text(encoding="utf-8").strip()
        return content or None
    return None


def check_prompt_profile(
    profile: str | PromptProfile,
    *,
    backend: Any = "mock",
    text: str | None = None,
    max_entities: int = 5,
    max_aliases: int = 2,
    max_desc: int = 3,
) -> PromptCheckReport:
    """Send one extraction prompt and one fold prompt built from ``profile`` and parse the answers.

    ``backend`` takes anything :func:`saadi.llm.resolve_backend` accepts. The
    mock backend proves the prompt and parser plumbing; a real model shows
    whether the guidance yields entities and a summary that follow the contract.
    The sample chunk is ``text``, else the profile's own ``sample.txt``, else a
    generic paragraph.
    """

    loaded = load_prompt_profile(profile)
    llm = resolve_backend(backend)
    backend_label = str(getattr(llm, "runtime_metadata", {}).get("backend") or type(llm).__name__)
    report = PromptCheckReport(
        profile=loaded.name,
        directory=str(loaded.directory),
        content_hash=loaded.content_hash,
        backend=backend_label,
    )
    for marker in validate_entity_guidance(loaded.entity_guidance):
        report.problems.append(f"entity guidance contains the contract phrase {marker!r}")
    for marker in validate_summary_guidance(loaded.summary_guidance):
        report.problems.append(f"summary guidance contains the contract phrase {marker!r}")

    sample = str(text or _profile_sample(loaded) or SAMPLE_TEXT).strip()
    strategy = HierarchicalSummaryStrategy(
        system_prompt=loaded.entity_system_prompt,
        rollup_system_prompt=loaded.summary_system_prompt,
        max_entities=int(max_entities),
        max_aliases=int(max_aliases),
        max_desc=int(max_desc),
    )
    chunk = ChunkRecord(doc_id="prompt_check", chunk_id="prompt_check:0", chunk_index=0, raw_text=sample)
    prompt = strategy.build_prompt(chunk, init_discourse_state())
    raw = llm.generate_chat_batch([prompt], max_tokens=1024, temperature=0.0, stop=ENTITY_STOP)[0]
    report.raw_entity_output = str(raw or "")
    _parsed, status = safe_json_from_text(report.raw_entity_output)
    report.entity_status = status
    if status == "failed":
        report.problems.append("the entity output is not a JSON object with active_entities and insertions")
    else:
        output = parse_canonicalization_response(
            report.raw_entity_output,
            fallback_chunk_text=sample,
            allow_linking_sentence=False,
            max_entities=int(max_entities),
            max_aliases=int(max_aliases),
            max_desc=int(max_desc),
        )
        report.entity_count = len(output.active_entities)
        report.insertion_count = len(output.insertions)
        report.entity_names = [str(entity.name) for entity in output.active_entities]
        if status == "fallback":
            report.notes.append("the JSON needed the lenient parser; the model added text around it")
        if report.entity_count == 0 and backend_label == "mock":
            report.notes.append("the mock backend returns no entities; pass --llm to test a real model")
        elif report.entity_count == 0:
            report.notes.append("no entities were extracted from the sample text")
        elif report.insertion_count < report.entity_count:
            report.notes.append(
                f"{report.entity_count - report.insertion_count} insertion(s) were dropped because their anchors "
                "were not exact substrings of the chunk"
            )

    rollup = strategy.build_rollup_prompt(items=[sample], lower_cycle=1, next_cycle=5, step=0)
    summaries = strategy.run_rollup_prompts(prompts=[rollup], backend=llm, generate=GenerateParams(max_tokens=256, temperature=0.0))
    summary = summaries[0] if summaries else None
    if summary:
        report.summary_status = "ok"
        report.summary_text = summary
    else:
        report.problems.append("the fold summary came back empty")
    return report


def format_report(report: PromptCheckReport) -> str:
    lines = [
        f"profile: {report.profile}",
        f"directory: {report.directory}",
        f"content hash: {report.content_hash}",
        f"backend: {report.backend}",
        f"entity extraction parse: {report.entity_status} ({report.entity_count} entities, {report.insertion_count} insertions)",
    ]
    if report.entity_names:
        lines.append("entities: " + ", ".join(report.entity_names))
    lines.append(f"fold summary: {report.summary_status}")
    if report.summary_text:
        lines.append(f"summary: {report.summary_text}")
    for note in report.notes:
        lines.append(f"note: {note}")
    for problem in report.problems:
        lines.append(f"PROBLEM: {problem}")
    lines.append("result: OK" if report.ok else "result: FAILED")
    return "\n".join(lines)


__all__ = ["PromptCheckReport", "SAMPLE_FILE", "SAMPLE_TEXT", "check_prompt_profile", "format_report"]
