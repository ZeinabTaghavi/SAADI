"""Shared helpers for static statistics over updated chunks."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Callable

from ....core.types import CanonicalizedChunkRecord


def _normalize_token_count_strategy(value: Any) -> str:
    raw = str(value or "whitespace").strip().lower()
    alias = {
        "default": "whitespace",
        "split": "whitespace",
        "whitespace_split": "whitespace",
        "space": "whitespace",
        "auto": "auto",
        "hf": "hf_tokenizer",
        "tokenizer": "hf_tokenizer",
        "huggingface": "hf_tokenizer",
        "experiment_model": "experiment_model",
    }
    out = alias.get(raw, raw)
    if out not in {"whitespace", "auto", "hf_tokenizer", "experiment_model"}:
        raise ValueError(
            "Invalid token_counting.strategy. Expected one of: whitespace, auto, hf_tokenizer, experiment_model "
            f"(got: {value!r})."
        )
    return out


def _infer_tokenizer_model_id(experiment_config: dict[str, Any] | None) -> str | None:
    model_cfg = (experiment_config or {}).get("model", {}) or {}
    if not isinstance(model_cfg, dict):
        return None
    for key in ("tokenizer_model_id", "model_id"):
        value = model_cfg.get(key)
        if value:
            return str(value)
    return None


def _count_whitespace_tokens(text: str) -> int:
    return len([tok for tok in str(text or "").split() if tok])


def build_token_counter(
    *,
    token_counting_config: dict[str, Any] | None = None,
    experiment_config: dict[str, Any] | None = None,
) -> tuple[Callable[[str], int], dict[str, Any]]:
    cfg = token_counting_config if isinstance(token_counting_config, dict) else {}
    strategy = _normalize_token_count_strategy(cfg.get("strategy", "whitespace"))
    tokenizer_model_id = str(
        cfg.get("tokenizer_model_id")
        or _infer_tokenizer_model_id(experiment_config)
        or ""
    ).strip()
    trust_remote_code = bool(cfg.get("trust_remote_code", True))
    cache_dir_raw = cfg.get("cache_dir")
    cache_dir = str(cache_dir_raw).strip() if cache_dir_raw is not None and str(cache_dir_raw).strip() else None

    if strategy == "whitespace":
        return _count_whitespace_tokens, {
            "requested_strategy": strategy,
            "resolved_strategy": "whitespace",
            "tokenizer_model_id": None,
            "fallback_used": False,
        }

    should_try_tokenizer = strategy in {"auto", "hf_tokenizer", "experiment_model"} and bool(tokenizer_model_id)
    if should_try_tokenizer:
        try:
            from ....llm.hf_tokenizers import load_hf_tokenizer

            tokenizer = load_hf_tokenizer(
                tokenizer_model_id,
                trust_remote_code=trust_remote_code,
                cache_dir=cache_dir,
            )

            def _count_with_tokenizer(text: str) -> int:
                encoded = tokenizer.encode(str(text or ""), add_special_tokens=False)
                return len(encoded)

            return _count_with_tokenizer, {
                "requested_strategy": strategy,
                "resolved_strategy": "hf_tokenizer",
                "tokenizer_model_id": tokenizer_model_id,
                "fallback_used": False,
                "trust_remote_code": trust_remote_code,
                "cache_dir": cache_dir,
            }
        except Exception as exc:
            return _count_whitespace_tokens, {
                "requested_strategy": strategy,
                "resolved_strategy": "whitespace",
                "tokenizer_model_id": tokenizer_model_id or None,
                "fallback_used": True,
                "fallback_reason": str(exc),
            }

    return _count_whitespace_tokens, {
        "requested_strategy": strategy,
        "resolved_strategy": "whitespace",
        "tokenizer_model_id": tokenizer_model_id or None,
        "fallback_used": strategy != "whitespace",
        "fallback_reason": "No tokenizer_model_id available; using whitespace token counting.",
    }


def safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def safe_median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    size = len(ordered)
    mid = size // 2
    if size % 2 == 1:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def compute_percentiles(values: list[float], percentiles: tuple[int, ...] = (5, 25, 50, 75, 95)) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {f"p{int(p)}": 0.0 for p in percentiles}
    if len(ordered) == 1:
        return {f"p{int(p)}": float(ordered[0]) for p in percentiles}

    out: dict[str, float] = {}
    last_index = len(ordered) - 1
    for percentile in percentiles:
        rank = (float(percentile) / 100.0) * last_index
        low = int(math.floor(rank))
        high = int(math.ceil(rank))
        if low == high:
            value = ordered[low]
        else:
            weight = rank - low
            value = ordered[low] + ((ordered[high] - ordered[low]) * weight)
        out[f"p{int(percentile)}"] = float(value)
    return out


def value_count_distribution(values: list[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(int(value))
        counts[key] = int(counts.get(key, 0)) + 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def normalize_active_entity(entity: Any) -> dict[str, Any] | None:
    if isinstance(entity, dict):
        name = str(entity.get("name", "") or "")
        aliases_raw = entity.get("aliases", []) or []
        descriptions_raw = entity.get("description", []) or []
    else:
        name = str(getattr(entity, "name", "") or "")
        aliases_raw = getattr(entity, "aliases", []) or []
        descriptions_raw = getattr(entity, "description", []) or []

    if not isinstance(aliases_raw, list):
        aliases_raw = []
    if not isinstance(descriptions_raw, list):
        descriptions_raw = []

    aliases = [str(value) for value in aliases_raw if str(value).strip()]
    descriptions = [str(value) for value in descriptions_raw if str(value).strip()]
    return {
        "name": name,
        "aliases": aliases,
        "description": descriptions,
    }


def active_entities_from_record(record: CanonicalizedChunkRecord) -> list[dict[str, Any]]:
    raw = (record.state_snapshot or {}).get("active_entities", []) or []
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        normalized = normalize_active_entity(item)
        if normalized is not None:
            out.append(normalized)
    return out


# Deprecated aliases for downstream scripts that still import the old names.
normalize_active_object = normalize_active_entity
active_objects_from_record = active_entities_from_record


def parse_l0aj_method_label(value: Any) -> int | None:
    text = str(value or "").strip().lower()
    match = re.fullmatch(r"l0aj_j([0-9]+)", text)
    if not match:
        return None
    try:
        out = int(match.group(1))
    except Exception:
        return None
    return out if out > 0 else None


def load_cached_report_index(llm_cache_index_json: str | Path) -> tuple[dict[str, Any] | None, Path]:
    index_path = Path(llm_cache_index_json).resolve()
    report_index_path = index_path.parent / "retrieval_reports" / "report_index.json"
    if not report_index_path.exists():
        return None, report_index_path
    try:
        import json

        with report_index_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return None, report_index_path
    return payload if isinstance(payload, dict) else None, report_index_path
