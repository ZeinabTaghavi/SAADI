"""Answer-generation metrics for Saadi."""

from __future__ import annotations

import json
import math
import re
import statistics
import string
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable

from ..io.jsonl import write_jsonl

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(text: Any) -> str:
    """SQuAD-style normalization for EM/F1 scoring."""

    value = str(text or "").lower()
    value = value.translate(_PUNCT_TABLE)
    value = _ARTICLES_RE.sub(" ", value)
    return " ".join(value.split())


def answer_tokens(text: Any) -> list[str]:
    normalized = normalize_answer(text)
    return normalized.split() if normalized else []


def exact_match(prediction: Any, reference: Any) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: Any, reference: Any) -> float:
    pred_tokens = answer_tokens(prediction)
    ref_tokens = answer_tokens(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for tok_a in a:
        cur = [0] * (len(b) + 1)
        for j, tok_b in enumerate(b, start=1):
            if tok_a == tok_b:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_l_f1(prediction: Any, reference: Any) -> float:
    pred_tokens = answer_tokens(prediction)
    ref_tokens = answer_tokens(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_len(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_containment(prediction: Any, references: Iterable[Any]) -> float:
    pred = normalize_answer(prediction)
    if not pred:
        return 0.0
    for ref in references:
        normalized_ref = normalize_answer(ref)
        if normalized_ref and normalized_ref in pred:
            return 1.0
    return 0.0


def extract_final_answer(text: Any) -> str:
    """Best-effort extraction for models that include labels or JSON wrappers."""

    raw = str(text or "").strip()
    if not raw:
        return ""
    if raw.startswith("```"):
        raw = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", raw).strip()
        raw = re.sub(r"\s*```$", "", raw).strip()
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = None
    if isinstance(parsed, dict):
        for key in ("answer", "final_answer", "generated_answer", "prediction", "response"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    patterns = (
        r"(?is)\bfinal\s+answer\s*:\s*(.+)$",
        r"(?is)\banswer\s*:\s*(.+)$",
    )
    for pattern in patterns:
        match = re.search(pattern, raw)
        if match:
            return match.group(1).strip()
    return raw


def best_over_references(metric_fn, prediction: Any, references: list[str]) -> float | None:
    refs = [str(ref) for ref in references if str(ref or "").strip()]
    if not refs:
        return None
    return max(float(metric_fn(prediction, ref)) for ref in refs)


def _mean(values: list[float]) -> float | None:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _percentile(values: list[float], percentile: float) -> float | None:
    vals = sorted(float(v) for v in values if v is not None and not math.isnan(float(v)))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    rank = (len(vals) - 1) * percentile
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return vals[lo]
    weight = rank - lo
    return vals[lo] * (1 - weight) + vals[hi] * weight


def distribution(values: Iterable[Any]) -> dict[str, float | None]:
    vals: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except Exception:
            continue
        if math.isnan(number):
            continue
        vals.append(number)
    if not vals:
        return {"mean": None, "median": None, "p95": None}
    return {
        "mean": float(sum(vals) / len(vals)),
        "median": float(statistics.median(vals)),
        "p95": _percentile(vals, 0.95),
    }


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None
    except Exception:
        return None


def package_versions() -> dict[str, str | None]:
    return {
        "bert-score": _package_version("bert-score"),
        "datasets": _package_version("datasets"),
        "numpy": _package_version("numpy"),
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "torch": _package_version("torch"),
        "transformers": _package_version("transformers"),
        "vllm": _package_version("vllm"),
    }


def _compute_bertscore(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    model_type: str,
    lang: str,
    batch_size: int,
    device: str | None,
    rescale_with_baseline: bool,
) -> tuple[dict[int, dict[str, float]], dict[str, Any], dict[str, str]]:
    normalized_mode = str(mode or "auto").strip().lower()
    config = {
        "enabled": normalized_mode not in {"0", "false", "no", "off", "disabled", "none"},
        "mode": normalized_mode,
        "model": model_type,
        "lang": lang,
        "batch_size": int(batch_size),
        "device": device,
        "rescale_with_baseline": bool(rescale_with_baseline),
        "best_over_references": True,
    }
    missing: dict[str, str] = {}
    if not config["enabled"]:
        missing["bertscore"] = "disabled"
        return {}, config, missing

    try:
        from bert_score import score as bert_score
    except Exception as exc:
        if normalized_mode in {"required", "require", "1", "true", "yes", "on"}:
            raise RuntimeError("BERTScore was required but `bert_score` is not installed.") from exc
        missing["bertscore"] = "bert_score package not installed"
        return {}, config, missing

    predictions: list[str] = []
    references: list[str] = []
    pair_to_row: list[int] = []
    for row_index, row in enumerate(rows):
        refs = [str(ref) for ref in row.get("answers", []) or [] if str(ref or "").strip()]
        if not refs:
            continue
        pred = str(row.get("generated_answer") or row.get("raw_output") or "")
        for ref in refs:
            predictions.append(pred)
            references.append(ref)
            pair_to_row.append(row_index)
    if not predictions:
        missing["bertscore"] = "no answer references available"
        return {}, config, missing

    kwargs: dict[str, Any] = {
        "model_type": model_type,
        "lang": lang,
        "batch_size": int(batch_size),
        "rescale_with_baseline": bool(rescale_with_baseline),
        "verbose": False,
    }
    if device:
        kwargs["device"] = device
    try:
        precision, recall, f1 = bert_score(predictions, references, **kwargs)
    except Exception as exc:
        if normalized_mode in {"required", "require", "1", "true", "yes", "on"}:
            raise
        missing["bertscore"] = f"bert_score execution failed: {exc}"
        return {}, config, missing
    by_row: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    for idx, p_val, r_val, f_val in zip(pair_to_row, precision, recall, f1):
        by_row[idx].append((float(p_val), float(r_val), float(f_val)))
    out: dict[int, dict[str, float]] = {}
    for idx, vals in by_row.items():
        best = max(vals, key=lambda item: item[2])
        out[idx] = {
            "bertscore_precision": best[0],
            "bertscore_recall": best[1],
            "bertscore_f1": best[2],
        }
    return out, config, missing


def _is_skipped_generation_row(row: dict[str, Any]) -> bool:
    status = str(row.get("status", "") or "").strip().lower()
    if status.startswith("skipped"):
        return True
    skip_reason = str(row.get("skip_reason", "") or "").strip()
    generated = str(row.get("generated_answer") or row.get("raw_output") or "").strip()
    return bool(skip_reason and not generated)


def _skip_accounting(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    evaluated_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    for row in rows:
        if _is_skipped_generation_row(row):
            skipped_rows.append(row)
        else:
            evaluated_rows.append(row)

    skipped_doc_ids = sorted(
        {
            str(row.get("doc_id", row.get("document_id", "")) or "")
            for row in skipped_rows
            if str(row.get("doc_id", row.get("document_id", "")) or "").strip()
        }
    )
    skipped_status_counts = Counter(str(row.get("status", "skipped") or "skipped") for row in skipped_rows)
    skipped_reason_counts = Counter(
        str(row.get("skip_reason", "") or "").strip()
        for row in skipped_rows
        if str(row.get("skip_reason", "") or "").strip()
    )
    accounting = {
        "n_total_rows": len(rows),
        "n_evaluated_examples": len(evaluated_rows),
        "n_skipped_examples": len(skipped_rows),
        "n_skipped_docs": len(skipped_doc_ids),
        "skipped_status_counts": dict(sorted(skipped_status_counts.items())),
        "skipped_reason_counts": dict(sorted(skipped_reason_counts.items())),
        "skipped_doc_ids_sample": skipped_doc_ids[:20],
    }
    return evaluated_rows, skipped_rows, accounting


def evaluate_generation_rows(
    rows: list[dict[str, Any]],
    *,
    dataset_name: str,
    split: str,
    method_name: str,
    run_name: str,
    generation_top_k: int | None = None,
    bert_score_mode: str = "auto",
    bert_score_model: str = "roberta-large",
    bert_score_lang: str = "en",
    bert_score_batch_size: int = 16,
    bert_score_device: str | None = None,
    bert_score_rescale_with_baseline: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return per-query rows, summary, leaderboard row, and evaluation manifest payload."""

    evaluated_rows, skipped_rows, skip_accounting = _skip_accounting(rows)
    bert_scores, bert_config, missing = _compute_bertscore(
        evaluated_rows,
        mode=bert_score_mode,
        model_type=bert_score_model,
        lang=bert_score_lang,
        batch_size=bert_score_batch_size,
        device=bert_score_device,
        rescale_with_baseline=bert_score_rescale_with_baseline,
    )

    per_query: list[dict[str, Any]] = []
    for row_index, row in enumerate(evaluated_rows):
        refs = [str(ref) for ref in row.get("answers", []) or [] if str(ref or "").strip()]
        raw_prediction = str(row.get("generated_answer") or row.get("raw_output") or "")
        extracted = extract_final_answer(raw_prediction)
        out = {
            "query_id": str(row.get("query_id", "")),
            "doc_id": str(row.get("doc_id", row.get("document_id", ""))),
            "question": str(row.get("question", "")),
            "exact_match": best_over_references(exact_match, raw_prediction, refs),
            "token_f1": best_over_references(token_f1, raw_prediction, refs),
            "rouge_l": best_over_references(rouge_l_f1, raw_prediction, refs),
            "exact_match_extracted": best_over_references(exact_match, extracted, refs),
            "token_f1_extracted": best_over_references(token_f1, extracted, refs),
            "answer_containment": answer_containment(raw_prediction, refs) if refs else None,
            "bertscore_precision": None,
            "bertscore_recall": None,
            "bertscore_f1": None,
            "retrieval_latency_ms": float(row.get("retrieval_latency_ms", 0.0) or 0.0),
            "generation_latency_ms": row.get("generation_latency_ms"),
            "total_latency_ms": row.get("total_latency_ms"),
            "context_tokens": row.get("context_tokens"),
            "prompt_tokens": row.get("prompt_tokens"),
            "answer_tokens": row.get("answer_tokens"),
            "tokens_per_second": row.get("tokens_per_second"),
            "peak_gpu_memory_mb": row.get("peak_gpu_memory_mb"),
            "status": row.get("status", "ok"),
        }
        out.update(bert_scores.get(row_index, {}))
        per_query.append(out)

    metric_keys = [
        "exact_match",
        "token_f1",
        "rouge_l",
        "exact_match_extracted",
        "token_f1_extracted",
        "answer_containment",
        "bertscore_precision",
        "bertscore_recall",
        "bertscore_f1",
    ]
    rag_metrics = {key: _mean([row.get(key) for row in per_query]) for key in metric_keys}
    efficiency_keys = [
        "retrieval_latency_ms",
        "generation_latency_ms",
        "total_latency_ms",
        "context_tokens",
        "prompt_tokens",
        "answer_tokens",
        "tokens_per_second",
        "peak_gpu_memory_mb",
    ]
    efficiency = {key: distribution(row.get(key) for row in per_query) for key in efficiency_keys}

    summary = {
        "method_name": method_name,
        "dataset_name": dataset_name,
        "split": split,
        "run_name": run_name,
        "n_queries": len(per_query),
        **skip_accounting,
        "generation_top_k": generation_top_k,
        "rag_metrics": rag_metrics,
        "efficiency": efficiency,
    }
    leaderboard = {
        "method_name": method_name,
        "dataset_name": dataset_name,
        "split": split,
        "run_name": run_name,
        "generation_top_k": generation_top_k,
        "n_total_rows": skip_accounting["n_total_rows"],
        "n_evaluated_examples": skip_accounting["n_evaluated_examples"],
        "n_skipped_examples": skip_accounting["n_skipped_examples"],
        "n_skipped_docs": skip_accounting["n_skipped_docs"],
    }
    leaderboard.update(rag_metrics)
    for key in ("retrieval_latency_ms", "generation_latency_ms", "total_latency_ms", "context_tokens", "answer_tokens", "tokens_per_second", "peak_gpu_memory_mb"):
        leaderboard[f"{key}_mean"] = efficiency[key]["mean"]

    assumptions = [
        "Retrieved-context generation uses retrieved chunks as the generation context.",
        "Retrieval latency is reported from the generation runner when available.",
        "Answer metrics are best-over-reference for QA entries with multiple accepted answers.",
        "Rows marked as skipped are excluded from answer metric and efficiency aggregation.",
    ]

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "assumptions": assumptions,
        "skip_accounting": {
            **skip_accounting,
            "skipped_query_ids_sample": [
                str(row.get("query_id", "") or "") for row in skipped_rows[:20] if str(row.get("query_id", "") or "").strip()
            ],
        },
        "bert_score_config": bert_config,
        "missing_metrics_and_reasons": missing,
        "package_versions": package_versions(),
    }
    return per_query, summary, leaderboard, manifest


def write_generation_evaluation(
    output_dir: str | Path,
    rows: list[dict[str, Any]],
    *,
    dataset_name: str,
    split: str,
    method_name: str,
    run_name: str,
    generation_top_k: int | None = None,
    bert_score_mode: str = "auto",
    bert_score_model: str = "roberta-large",
    bert_score_lang: str = "en",
    bert_score_batch_size: int = 16,
    bert_score_device: str | None = None,
    bert_score_rescale_with_baseline: bool = False,
    input_files_used: dict[str, str] | None = None,
    command_used: str | None = None,
) -> dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_query, summary, leaderboard, manifest = evaluate_generation_rows(
        rows,
        dataset_name=dataset_name,
        split=split,
        method_name=method_name,
        run_name=run_name,
        generation_top_k=generation_top_k,
        bert_score_mode=bert_score_mode,
        bert_score_model=bert_score_model,
        bert_score_lang=bert_score_lang,
        bert_score_batch_size=bert_score_batch_size,
        bert_score_device=bert_score_device,
        bert_score_rescale_with_baseline=bert_score_rescale_with_baseline,
    )
    paths = {
        "metrics_summary": str(out_dir / "metrics_summary.json"),
        "metrics_per_query": str(out_dir / "metrics_per_query.jsonl"),
        "leaderboard_row": str(out_dir / "leaderboard_row.json"),
        "evaluation_manifest": str(out_dir / "evaluation_manifest.json"),
    }
    manifest.update(
        {
            "command_used": command_used,
            "input_files_used": input_files_used or {},
            "output_files_written": paths,
        }
    )
    for path_key, payload in (
        ("metrics_summary", summary),
        ("leaderboard_row", leaderboard),
        ("evaluation_manifest", manifest),
    ):
        with Path(paths[path_key]).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    write_jsonl(paths["metrics_per_query"], per_query)
    return paths
