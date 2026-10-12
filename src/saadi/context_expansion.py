"""Opt-in document-level context expansion for long-context experiments."""

from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .io.jsonl import write_jsonl

RANDOM_DOCUMENT_GROUPS_MODE = "random_document_groups"
SUPPORTED_CONTEXT_EXPANSION_DATASETS = {
    "loogle",
    "bigai-nlco/loogle",
    "bigainlco/loogle",
    "qasper",
    "quality",
    "musique",
    "dgslibisey/musique",
}


def parse_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    raw = str(value).strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off", ""}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def expansion_size_label(target_context_tokens: int) -> str:
    target = int(target_context_tokens)
    if target > 0 and target % 1000 == 0:
        return f"{target // 1000}k"
    return str(target)


def expansion_output_name(method_name: str, target_context_tokens: int) -> str:
    return f"{method_name}_expand{expansion_size_label(target_context_tokens)}"


@dataclass(frozen=True, slots=True)
class ContextExpansionConfig:
    expand_context: bool = False
    target_context_tokens: int | None = None
    tolerance_tokens: int = 4000
    expansion_seed: int = 42
    expansion_mode: str = RANDOM_DOCUMENT_GROUPS_MODE
    dataset_name: str = ""

    @property
    def enabled(self) -> bool:
        return bool(self.expand_context)

    @property
    def lower_bound_tokens(self) -> int:
        return max(0, int(self.target_context_tokens or 0) - int(self.tolerance_tokens))

    @property
    def upper_bound_tokens(self) -> int:
        return int(self.target_context_tokens or 0) + int(self.tolerance_tokens)

    @property
    def output_suffix(self) -> str:
        if self.target_context_tokens is None:
            return "expand"
        return f"expand{expansion_size_label(int(self.target_context_tokens))}"

    def validate(self) -> None:
        if not self.enabled:
            return
        dataset = str(self.dataset_name or "").strip().lower()
        if dataset and dataset not in SUPPORTED_CONTEXT_EXPANSION_DATASETS:
            raise ValueError(
                "Context expansion is currently implemented only for LooGLE, Qasper, Quality, and MuSiQue. "
                f"Got dataset={self.dataset_name!r}."
            )
        if self.target_context_tokens is None or int(self.target_context_tokens) <= 0:
            raise ValueError("--target_context_tokens must be > 0 when --expand_context is true.")
        if int(self.tolerance_tokens) < 0:
            raise ValueError("--tolerance_tokens must be >= 0.")
        if str(self.expansion_mode) != RANDOM_DOCUMENT_GROUPS_MODE:
            raise ValueError(
                f"Unsupported expansion mode {self.expansion_mode!r}. "
                f"Expected {RANDOM_DOCUMENT_GROUPS_MODE!r}."
            )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["enabled"] = self.enabled
        payload["lower_bound_tokens"] = self.lower_bound_tokens if self.enabled else None
        payload["upper_bound_tokens"] = self.upper_bound_tokens if self.enabled else None
        payload["output_suffix"] = self.output_suffix if self.enabled else None
        return payload


@dataclass(frozen=True, slots=True)
class ExpandedDocumentContext:
    original_doc_id: str
    expanded_group_id: str
    group_doc_order: list[str]
    target_context_tokens: int
    tolerance_tokens: int
    raw_group_tokens: int
    final_context_tokens: int
    target_doc_position: int
    was_trimmed: bool
    trim_side: str
    seed: int
    expansion_mode: str
    context_text: str

    def metadata(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("context_text", None)
        return payload

    def generation_metadata(self) -> dict[str, Any]:
        return {
            "expanded_group_id": self.expanded_group_id,
            "group_doc_order": list(self.group_doc_order),
            "target_context_tokens": int(self.target_context_tokens),
            "tolerance_tokens": int(self.tolerance_tokens),
            "raw_group_tokens": int(self.raw_group_tokens),
            "final_context_tokens": int(self.final_context_tokens),
            "target_doc_position": int(self.target_doc_position),
            "was_trimmed": bool(self.was_trimmed),
            "trim_side": self.trim_side,
            "expansion_seed": int(self.seed),
            "expansion_mode": self.expansion_mode,
        }


@dataclass(frozen=True, slots=True)
class ContextExpansionBundle:
    config: ContextExpansionConfig
    group_metadata_rows: list[dict[str, Any]]
    contexts_by_doc_id: dict[str, ExpandedDocumentContext]

    def context_for_doc(self, doc_id: str) -> ExpandedDocumentContext | None:
        return self.contexts_by_doc_id.get(str(doc_id))

    def expanded_documents_by_id(self) -> dict[str, str]:
        return {doc_id: item.context_text for doc_id, item in self.contexts_by_doc_id.items()}

    def metadata_for_doc(self, doc_id: str) -> dict[str, Any] | None:
        item = self.context_for_doc(doc_id)
        return item.metadata() if item is not None else None


def context_expansion_config_from_config(
    config: dict[str, Any],
    *,
    dataset_name: str | None = None,
) -> ContextExpansionConfig:
    raw: dict[str, Any] = {}
    top_level = config.get("context_expansion", {}) or {}
    if isinstance(top_level, dict):
        raw.update(top_level)
    generation = config.get("generation", {}) or {}
    if isinstance(generation, dict):
        nested = generation.get("context_expansion", {}) or {}
        if isinstance(nested, dict):
            raw.update(nested)

    inferred_dataset = dataset_name
    if inferred_dataset is None:
        dataset_cfg = config.get("dataset", {}) or {}
        if isinstance(dataset_cfg, dict):
            inferred_dataset = str(dataset_cfg.get("name", "") or "")

    return ContextExpansionConfig(
        expand_context=parse_bool(raw.get("expand_context", raw.get("enabled", False))),
        target_context_tokens=(
            None
            if raw.get("target_context_tokens") in (None, "", "null")
            else int(raw.get("target_context_tokens"))
        ),
        tolerance_tokens=int(raw.get("tolerance_tokens", 4000) or 0),
        expansion_seed=int(raw.get("expansion_seed", raw.get("seed", 42)) or 42),
        expansion_mode=str(raw.get("expansion_mode", RANDOM_DOCUMENT_GROUPS_MODE) or RANDOM_DOCUMENT_GROUPS_MODE),
        dataset_name=str(inferred_dataset or raw.get("dataset_name", "") or ""),
    )


def context_expansion_cli_overrides(args: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if getattr(args, "expand_context", None) is not None:
        out["expand_context"] = parse_bool(getattr(args, "expand_context"))
    for attr in ("target_context_tokens", "tolerance_tokens", "expansion_seed", "expansion_mode"):
        value = getattr(args, attr, None)
        if value is not None:
            out[attr] = value
    return out


def apply_context_expansion_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> None:
    if not overrides:
        return
    generation = config.setdefault("generation", {})
    if not isinstance(generation, dict):
        generation = {}
        config["generation"] = generation
    current = generation.get("context_expansion", {}) or {}
    if not isinstance(current, dict):
        current = {}
    current.update(overrides)
    generation["context_expansion"] = current


def _token_ids(text: str, tokenizer: Any | None) -> list[Any] | None:
    if tokenizer is None or not hasattr(tokenizer, "encode"):
        return None
    try:
        return list(tokenizer.encode(str(text or ""), add_special_tokens=False))
    except TypeError:
        try:
            return list(tokenizer.encode(str(text or "")))
        except Exception:
            return None
    except Exception:
        return None


def _decode_token_ids(token_ids: list[Any], tokenizer: Any | None) -> str:
    if tokenizer is None or not hasattr(tokenizer, "decode"):
        return " ".join(str(token) for token in token_ids)
    try:
        return str(tokenizer.decode(token_ids, skip_special_tokens=False))
    except TypeError:
        return str(tokenizer.decode(token_ids))


def count_tokens(text: str, tokenizer: Any | None = None) -> int:
    ids = _token_ids(text, tokenizer)
    if ids is not None:
        return len(ids)
    return len(str(text or "").split())


def trim_text_tokens(
    text: str,
    *,
    tokenizer: Any | None,
    max_tokens: int,
    trim_side: str,
) -> tuple[str, int, bool]:
    max_tokens = int(max_tokens)
    if max_tokens <= 0:
        return "", 0, bool(str(text or "").strip())
    ids = _token_ids(text, tokenizer)
    if ids is None:
        tokens = str(text or "").split()
        if len(tokens) <= max_tokens:
            return str(text or ""), len(tokens), False
        if trim_side == "beginning":
            kept = tokens[-max_tokens:]
        else:
            kept = tokens[:max_tokens]
        return " ".join(kept), len(kept), True
    if len(ids) <= max_tokens:
        return str(text or ""), len(ids), False
    if trim_side == "beginning":
        kept_ids = ids[-max_tokens:]
    else:
        kept_ids = ids[:max_tokens]
    return _decode_token_ids(kept_ids, tokenizer), len(kept_ids), True


@dataclass(frozen=True, slots=True)
class _DocumentInfo:
    doc_id: str
    text: str
    tokens: int


@dataclass(frozen=True, slots=True)
class _DocumentGroup:
    group_id: str
    docs: list[_DocumentInfo]
    raw_context: str
    raw_tokens: int


def _stable_random_choice(seed: int, *parts: str) -> str:
    material = "|".join([str(seed), *[str(part) for part in parts]])
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return "beginning" if int(digest[:8], 16) % 2 == 0 else "end"


def _trim_side_for_target(*, seed: int, group_id: str, doc_id: str, position: int, num_documents: int) -> str:
    if num_documents <= 1:
        return "end"
    if position <= 1:
        return "end"
    if position >= num_documents:
        return "beginning"
    relative = (position - 1) / max(1, num_documents - 1)
    if relative <= 0.33:
        return "end"
    if relative >= 0.67:
        return "beginning"
    return _stable_random_choice(seed, group_id, doc_id)


def _format_group_context(docs: list[_DocumentInfo]) -> str:
    total = len(docs)
    blocks: list[str] = []
    for index, doc in enumerate(docs, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[Document {index}/{total} | doc_id={doc.doc_id}]",
                    str(doc.text or "").strip(),
                ]
            ).strip()
        )
    return "\n\n".join(block for block in blocks if block).strip()


def _build_groups(
    documents: dict[str, str],
    *,
    cfg: ContextExpansionConfig,
    tokenizer: Any | None,
) -> list[_DocumentGroup]:
    infos = [
        _DocumentInfo(doc_id=str(doc_id), text=str(text or ""), tokens=count_tokens(str(text or ""), tokenizer))
        for doc_id, text in documents.items()
        if str(doc_id).strip() and str(text or "").strip()
    ]
    rng = random.Random(int(cfg.expansion_seed))
    rng.shuffle(infos)

    groups: list[list[_DocumentInfo]] = []
    current: list[_DocumentInfo] = []
    current_tokens = 0
    target = int(cfg.target_context_tokens or 0)

    def flush_current() -> None:
        nonlocal current, current_tokens
        if current:
            groups.append(current)
            current = []
            current_tokens = 0

    for info in infos:
        if info.tokens >= target:
            flush_current()
            groups.append([info])
            continue
        current.append(info)
        current_tokens += int(info.tokens)
        if current_tokens >= target:
            flush_current()
    flush_current()

    out: list[_DocumentGroup] = []
    for index, docs in enumerate(groups, start=1):
        raw_context = _format_group_context(docs)
        out.append(
            _DocumentGroup(
                group_id=f"group_{index:03d}",
                docs=docs,
                raw_context=raw_context,
                raw_tokens=count_tokens(raw_context, tokenizer),
            )
        )
    return out


def build_document_context_expansion(
    documents_by_id: dict[str, str],
    cfg: ContextExpansionConfig,
    *,
    tokenizer: Any | None = None,
) -> ContextExpansionBundle:
    cfg.validate()
    if not cfg.enabled:
        return ContextExpansionBundle(config=cfg, group_metadata_rows=[], contexts_by_doc_id={})

    groups = _build_groups(documents_by_id, cfg=cfg, tokenizer=tokenizer)
    contexts_by_doc_id: dict[str, ExpandedDocumentContext] = {}
    group_rows: list[dict[str, Any]] = []
    upper_bound = cfg.upper_bound_tokens

    for group in groups:
        doc_order = [doc.doc_id for doc in group.docs]
        group_rows.append(
            {
                "expanded_group_id": group.group_id,
                "group_doc_order": doc_order,
                "target_context_tokens": int(cfg.target_context_tokens or 0),
                "tolerance_tokens": int(cfg.tolerance_tokens),
                "lower_bound_tokens": cfg.lower_bound_tokens,
                "upper_bound_tokens": cfg.upper_bound_tokens,
                "raw_group_tokens": int(group.raw_tokens),
                "num_documents": len(group.docs),
                "seed": int(cfg.expansion_seed),
                "expansion_mode": cfg.expansion_mode,
            }
        )
        position_by_doc = {doc_id: index for index, doc_id in enumerate(doc_order, start=1)}
        for doc_id in doc_order:
            position = position_by_doc[doc_id]
            trim_side = "none"
            final_context = group.raw_context
            final_tokens = int(group.raw_tokens)
            was_trimmed = False
            if group.raw_tokens > upper_bound:
                trim_side = _trim_side_for_target(
                    seed=int(cfg.expansion_seed),
                    group_id=group.group_id,
                    doc_id=doc_id,
                    position=position,
                    num_documents=len(doc_order),
                )
                final_context, final_tokens, was_trimmed = trim_text_tokens(
                    group.raw_context,
                    tokenizer=tokenizer,
                    max_tokens=upper_bound,
                    trim_side=trim_side,
                )
            contexts_by_doc_id[doc_id] = ExpandedDocumentContext(
                original_doc_id=doc_id,
                expanded_group_id=group.group_id,
                group_doc_order=list(doc_order),
                target_context_tokens=int(cfg.target_context_tokens or 0),
                tolerance_tokens=int(cfg.tolerance_tokens),
                raw_group_tokens=int(group.raw_tokens),
                final_context_tokens=int(final_tokens),
                target_doc_position=position,
                was_trimmed=bool(was_trimmed),
                trim_side=trim_side,
                seed=int(cfg.expansion_seed),
                expansion_mode=cfg.expansion_mode,
                context_text=final_context,
            )

    return ContextExpansionBundle(
        config=cfg,
        group_metadata_rows=group_rows,
        contexts_by_doc_id=contexts_by_doc_id,
    )


def build_example_metadata_rows(
    qa_entries: Iterable[dict[str, Any]],
    bundle: ContextExpansionBundle,
    *,
    query_id_fn: Callable[[dict[str, Any], int], str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, entry in enumerate(qa_entries):
        doc_id = str(entry.get("document_id", "") or "").strip()
        context = bundle.context_for_doc(doc_id)
        if context is None:
            continue
        rows.append(
            {
                "example_id": str(query_id_fn(entry, index)),
                **context.metadata(),
            }
        )
    return rows


def write_context_expansion_metadata(
    output_dir: str | Path,
    bundle: ContextExpansionBundle,
    example_metadata_rows: Iterable[dict[str, Any]],
) -> dict[str, str]:
    out_dir = Path(output_dir)
    group_path = out_dir / "group_metadata.jsonl"
    example_path = out_dir / "example_metadata.jsonl"
    write_jsonl(group_path, bundle.group_metadata_rows)
    write_jsonl(example_path, example_metadata_rows)
    return {
        "group_metadata_jsonl": str(group_path),
        "example_metadata_jsonl": str(example_path),
    }
