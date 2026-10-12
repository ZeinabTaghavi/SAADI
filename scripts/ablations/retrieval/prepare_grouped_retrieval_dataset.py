#!/usr/bin/env python3
"""Freeze a deterministic no-expansion or grouped-expansion retrieval dataset."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import time
from typing import Any

from saadi.config.loader import load_config
from saadi.context_expansion import ContextExpansionConfig, build_document_context_expansion, count_tokens
from saadi.indexing.dataset_loaders.prepared import load_dataset_from_config
from saadi.indexing.labeling import classify_overlap

MUSIQUE_HOPS = (2, 3, 4)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _paths(output_dir: Path) -> dict[str, Path]:
    return {
        "documents": output_dir / "documents.json",
        "qa": output_dir / "qa.json",
        "membership": output_dir / "group_membership.json",
        "metadata": output_dir / "context_expansion_metadata.json",
    }


def _select_documents(
    documents: dict[str, str],
    qa_entries: list[dict[str, Any]],
    *,
    max_documents: int | None,
    samples_per_hop: int | None = None,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    if samples_per_hop is not None:
        if max_documents != samples_per_hop * len(MUSIQUE_HOPS):
            raise RuntimeError(
                f"Stratified MuSiQue selection requires max_documents="
                f"{samples_per_hop * len(MUSIQUE_HOPS)}, got {max_documents}."
            )
        by_hop: dict[int, list[str]] = {hop: [] for hop in MUSIQUE_HOPS}
        for entry in qa_entries:
            doc_id = str(entry.get("document_id", ""))
            prefix = doc_id.split("__", 1)[0].lower()
            hop = next(
                (candidate for candidate in MUSIQUE_HOPS if prefix.startswith(f"{candidate}hop")),
                None,
            )
            if hop is not None and doc_id in documents and doc_id not in by_hop[hop]:
                by_hop[hop].append(doc_id)
        shortages = {
            hop: len(values)
            for hop, values in by_hop.items()
            if len(values) < samples_per_hop
        }
        if shortages:
            raise RuntimeError(
                f"MuSiQue cannot provide {samples_per_hop} documents per hop; available={shortages}."
            )
        selected_ids = [
            doc_id
            for hop in MUSIQUE_HOPS
            for doc_id in by_hop[hop][:samples_per_hop]
        ]
        selected = {doc_id: documents[doc_id] for doc_id in selected_ids}
    elif max_documents is None:
        selected = dict(documents)
    else:
        selected_ids: list[str] = []
        seen: set[str] = set()
        for entry in qa_entries:
            doc_id = str(entry.get("document_id", ""))
            if doc_id in documents and doc_id not in seen:
                seen.add(doc_id)
                selected_ids.append(doc_id)
                if len(selected_ids) >= max_documents:
                    break
        if len(selected_ids) < max_documents:
            for doc_id in documents:
                doc_id = str(doc_id)
                if doc_id not in seen:
                    seen.add(doc_id)
                    selected_ids.append(doc_id)
                    if len(selected_ids) >= max_documents:
                        break
        if len(selected_ids) < max_documents:
            raise RuntimeError(
                f"Requested {max_documents} source documents, but only {len(selected_ids)} are available."
            )
        selected = {doc_id: documents[doc_id] for doc_id in selected_ids}
    selected_ids_set = set(selected)
    selected_qa = [
        dict(entry)
        for entry in qa_entries
        if str(entry.get("document_id", "")) in selected_ids_set
    ]
    return selected, selected_qa


def _variant_id(group_id: str, context: str, variant_count: int) -> str:
    if variant_count == 1:
        return group_id
    digest = hashlib.sha256(context.encode("utf-8")).hexdigest()[:10]
    return f"{group_id}__variant_{digest}"


def _build_expanded(
    documents: dict[str, str],
    qa_entries: list[dict[str, Any]],
    *,
    dataset_name: str,
    target: int,
    tolerance: int,
    seed: int,
) -> tuple[dict[str, str], list[dict[str, Any]], dict[str, list[str]], dict[str, Any]]:
    expansion = build_document_context_expansion(
        documents,
        ContextExpansionConfig(
            expand_context=True,
            target_context_tokens=target,
            tolerance_tokens=tolerance,
            expansion_seed=seed,
            dataset_name=dataset_name,
        ),
        tokenizer=None,
    )

    source_to_output: dict[str, str] = {}
    output_documents: dict[str, str] = {}
    membership: dict[str, list[str]] = {}
    variant_groups = 0
    for row in expansion.group_metadata_rows:
        group_id = str(row["expanded_group_id"])
        members = [str(value) for value in row["group_doc_order"]]
        contexts_to_members: dict[str, list[str]] = {}
        for source_id in members:
            emitted = expansion.contexts_by_doc_id[source_id].context_text
            contexts_to_members.setdefault(emitted, []).append(source_id)
        if len(contexts_to_members) > 1:
            variant_groups += 1
        for context, owners in contexts_to_members.items():
            output_id = _variant_id(group_id, context, len(contexts_to_members))
            output_documents[output_id] = context
            membership[output_id] = list(members)
            for owner in owners:
                source_to_output[owner] = output_id

    remapped_qa: list[dict[str, Any]] = []
    for entry in qa_entries:
        source_id = str(entry["document_id"])
        output_id = source_to_output.get(source_id)
        if output_id is None:
            raise RuntimeError(f"No expanded context was emitted for source document {source_id!r}.")
        remapped_qa.append(
            {
                **entry,
                "source_document_id": source_id,
                "document_id": output_id,
            }
        )
    details = {
        "source_group_count": len(expansion.group_metadata_rows),
        "emitted_context_count": len(output_documents),
        "groups_with_trimmed_variants": variant_groups,
    }
    return output_documents, remapped_qa, membership, details


def _append_batch_prefix(*, previous_source_count: int, target_source_count: int) -> str:
    return f"append_{previous_source_count + 1:04d}_{target_source_count:04d}"


def _namespace_expanded_batch(
    documents: dict[str, str],
    qa_entries: list[dict[str, Any]],
    membership: dict[str, list[str]],
    *,
    prefix: str,
) -> tuple[dict[str, str], list[dict[str, Any]], dict[str, list[str]]]:
    renamed = {doc_id: f"{prefix}__{doc_id}" for doc_id in documents}
    return (
        {renamed[doc_id]: text for doc_id, text in documents.items()},
        [
            {
                **entry,
                "document_id": renamed[str(entry["document_id"])],
            }
            for entry in qa_entries
        ],
        {renamed[doc_id]: list(members) for doc_id, members in membership.items()},
    )


def _load_append_base(
    output_dir: Path,
    *,
    dataset_name: str,
    split: str,
    target: int,
    tolerance: int,
    seed: int,
    requested_max_documents: int | None,
) -> tuple[
    dict[str, str],
    list[dict[str, Any]],
    dict[str, list[str]],
    dict[str, Any],
] | None:
    if requested_max_documents is None:
        return None
    paths = _paths(output_dir)
    if any(not path.is_file() or path.stat().st_size == 0 for path in paths.values()):
        return None
    try:
        metadata = _read_json(paths["metadata"])
        documents = _read_json(paths["documents"])
        qa_entries = _read_json(paths["qa"])
        membership = _read_json(paths["membership"])
    except Exception:
        return None
    if (
        not isinstance(metadata, dict)
        or not isinstance(documents, dict)
        or not documents
        or not isinstance(qa_entries, list)
        or not qa_entries
        or not isinstance(membership, dict)
        or set(map(str, documents)) != set(map(str, membership))
        or metadata.get("dataset") != dataset_name
        or metadata.get("split") != split
        or int(metadata.get("target_context_tokens", -1)) != target
        or int(metadata.get("tolerance_tokens", -1)) != tolerance
        or int(metadata.get("expansion_seed", -1)) != seed
        or int(metadata.get("source_matchable_spans_lost", -1)) != 0
    ):
        return None
    previous_max = metadata.get("max_source_documents")
    try:
        previous_count = int(metadata.get("selected_source_documents", previous_max))
        previous_max_int = int(previous_max)
    except (TypeError, ValueError):
        return None
    if (
        previous_count <= 0
        or previous_max_int != previous_count
        or previous_count >= requested_max_documents
        or int(metadata.get("prepared_documents", -1)) != len(documents)
        or int(metadata.get("qa_entries", -1)) != len(qa_entries)
    ):
        return None
    normalized_membership = {
        str(group_id): [str(source_id) for source_id in (members or [])]
        for group_id, members in membership.items()
    }
    for entry in qa_entries:
        if not isinstance(entry, dict):
            return None
        document_id = str(entry.get("document_id", ""))
        source_id = str(entry.get("source_document_id", document_id))
        if source_id not in normalized_membership.get(document_id, []):
            return None
    return (
        {str(doc_id): str(text) for doc_id, text in documents.items()},
        [dict(entry) for entry in qa_entries],
        normalized_membership,
        metadata,
    )


def _validate(
    documents: dict[str, str],
    qa_entries: list[dict[str, Any]],
    membership: dict[str, list[str]],
    *,
    dataset_name: str,
    source_documents: dict[str, str] | None = None,
) -> dict[str, int]:
    if not documents:
        raise RuntimeError("Preparation produced no documents.")
    if not qa_entries:
        raise RuntimeError("Preparation produced no QA entries.")
    if set(documents) != set(membership):
        raise RuntimeError("Prepared documents and group-membership IDs do not match.")

    missing_documents = 0
    literal_span_misses = 0
    labeling_span_misses = 0
    source_matchable_spans = 0
    source_matchable_spans_lost = 0
    queries_with_labeling_match = 0
    queries_without_labeling_match = 0
    for entry in qa_entries:
        doc_id = str(entry.get("document_id", ""))
        text = documents.get(doc_id)
        if text is None:
            missing_documents += 1
            continue
        spans = [str(span) for span in (entry.get("retrieval_spans", []) or []) if str(span).strip()]
        query_has_match = False
        source_id = str(entry.get("source_document_id", entry.get("document_id", "")))
        source_text = (source_documents or {}).get(source_id)
        for span in spans:
            literal_match = span in text
            labeling_match = classify_overlap(text, span)[0] == "full"
            if not literal_match:
                literal_span_misses += 1
            if not labeling_match:
                labeling_span_misses += 1
            else:
                query_has_match = True
            if source_text is not None and classify_overlap(source_text, span)[0] == "full":
                source_matchable_spans += 1
                if not labeling_match:
                    source_matchable_spans_lost += 1
        if spans:
            if query_has_match:
                queries_with_labeling_match += 1
            else:
                queries_without_labeling_match += 1

    # MuSiQue's loader guarantees verbatim supporting paragraphs. Keep that
    # stronger invariant. QASPER evidence is evaluated by the established
    # normalized token matcher, and some annotations are not literal
    # substrings even in the original paper text.
    strict_literal_failure = dataset_name == "musique" and literal_span_misses > 0
    if missing_documents or strict_literal_failure or source_matchable_spans_lost:
        raise RuntimeError(
            "Prepared QA validation failed: "
            f"missing_documents={missing_documents}, "
            f"literal_span_misses={literal_span_misses}, "
            f"source_matchable_spans_lost={source_matchable_spans_lost}."
        )
    return {
        "qa_missing_documents": missing_documents,
        "spans_not_contained": literal_span_misses,
        "spans_not_matchable_by_labeling": labeling_span_misses,
        "source_matchable_spans": source_matchable_spans,
        "source_matchable_spans_lost": source_matchable_spans_lost,
        "queries_with_labeling_span_match": queries_with_labeling_match,
        "queries_without_labeling_span_match": queries_without_labeling_match,
    }


def _is_current(
    output_dir: Path,
    *,
    dataset_name: str,
    split: str,
    target: int,
    tolerance: int,
    seed: int,
    max_documents: int | None,
    samples_per_hop: int | None,
) -> bool:
    paths = _paths(output_dir)
    if any(not path.is_file() or path.stat().st_size == 0 for path in paths.values()):
        return False
    try:
        metadata = _read_json(paths["metadata"])
        documents = _read_json(paths["documents"])
        qa_entries = _read_json(paths["qa"])
        membership = _read_json(paths["membership"])
    except Exception:
        return False

    # Full span validation is performed before these files are atomically
    # frozen. Re-running the normalized QASPER matcher here made a cache hit
    # look hung for a long time. On resume, use structural integrity plus the
    # recorded validation result; the index binding separately verifies exact
    # documents.json and qa.json hashes before an index can be reused.
    if (
        not isinstance(metadata, dict)
        or not isinstance(documents, dict)
        or not documents
        or not isinstance(qa_entries, list)
        or not qa_entries
        or not isinstance(membership, dict)
        or set(map(str, documents)) != set(map(str, membership))
    ):
        return False
    document_ids = set(map(str, documents))
    normalized_membership = {
        str(group_id): {str(source_id) for source_id in (members or [])}
        for group_id, members in membership.items()
    }
    for entry in qa_entries:
        if not isinstance(entry, dict):
            return False
        document_id = str(entry.get("document_id", ""))
        source_id = str(entry.get("source_document_id", document_id))
        if (
            document_id not in document_ids
            or source_id not in normalized_membership.get(document_id, set())
        ):
            return False
    return bool(
        metadata.get("dataset") == dataset_name
        and metadata.get("split") == split
        and int(metadata.get("target_context_tokens", -1)) == target
        and int(metadata.get("tolerance_tokens", -1)) == tolerance
        and int(metadata.get("expansion_seed", -1)) == seed
        and metadata.get("max_source_documents") == max_documents
        and metadata.get("samples_per_hop") == samples_per_hop
        and int(metadata.get("prepared_documents", -1)) == len(documents)
        and int(metadata.get("qa_entries", -1)) == len(qa_entries)
        and int(metadata.get("source_matchable_spans_lost", -1)) == 0
    )


def prepare(
    *,
    config_path: Path,
    output_dir: Path,
    target: int,
    tolerance: int,
    seed: int,
    max_documents: int | None,
    samples_per_hop: int | None,
    force: bool,
) -> dict[str, Any]:
    config = load_config(config_path)
    dataset_cfg = dict(config.get("dataset", {}) or {})
    dataset_name = str(dataset_cfg.get("name", "")).strip().lower()
    split = str(dataset_cfg.get("split", "test"))
    if not dataset_name:
        raise ValueError("The base config must define dataset.name.")

    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".prepare.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        print(f"Checking frozen prepared dataset: {output_dir}", flush=True)
        lock_deadline = time.monotonic() + 120.0
        announced_wait = False
        while True:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if not announced_wait:
                    print(
                        f"Waiting for another preparation process holding: {lock_path}",
                        flush=True,
                    )
                    announced_wait = True
                if time.monotonic() >= lock_deadline:
                    raise RuntimeError(
                        "Timed out after 120 seconds waiting for the preparation lock. "
                        f"Another runner is still using {lock_path}. Check it with "
                        f"`lsof {lock_path}` or `fuser -v {lock_path}`."
                    )
                time.sleep(1.0)

        lock_handle.seek(0)
        lock_handle.truncate()
        lock_handle.write(f"pid={os.getpid()}\n")
        lock_handle.flush()
        if not force and _is_current(
            output_dir,
            dataset_name=dataset_name,
            split=split,
            target=target,
            tolerance=tolerance,
            seed=seed,
            max_documents=max_documents,
            samples_per_hop=samples_per_hop,
        ):
            metadata = _read_json(_paths(output_dir)["metadata"])
            print(f"Using frozen prepared dataset: {output_dir}", flush=True)
            print(json.dumps(metadata, indent=2))
            return metadata

        append_base = (
            None
            if force or samples_per_hop is not None
            else _load_append_base(
                output_dir,
                dataset_name=dataset_name,
                split=split,
                target=target,
                tolerance=tolerance,
                seed=seed,
                requested_max_documents=max_documents,
            )
        )
        print(f"Frozen preparation is absent or requires extension; loading {dataset_name}.", flush=True)
        documents, qa_entries, source_meta = load_dataset_from_config(
            dataset_cfg,
            base_dir=config_path.parent,
        )
        selected_documents, selected_qa = _select_documents(
            documents,
            qa_entries,
            max_documents=max_documents,
            samples_per_hop=samples_per_hop,
        )
        append_update: dict[str, Any] | None = None
        if append_base is not None:
            old_documents, old_qa, old_membership, old_metadata = append_base
            old_source_ids = {
                source_id
                for members in old_membership.values()
                for source_id in members
            }
            previous_count = int(old_metadata["selected_source_documents"])
            expected_old_ids = set(list(selected_documents)[:previous_count])
            if old_source_ids != expected_old_ids:
                raise RuntimeError(
                    "The existing frozen sample is not the deterministic prefix of the requested "
                    f"{max_documents}-document sample; refusing an unsafe in-place extension."
                )
            new_documents = {
                doc_id: text
                for doc_id, text in selected_documents.items()
                if doc_id not in old_source_ids
            }
            new_source_ids = set(new_documents)
            new_qa = [
                dict(entry)
                for entry in selected_qa
                if str(entry.get("document_id", "")) in new_source_ids
            ]
            if not new_documents:
                raise RuntimeError("In-place extension selected no new source documents.")

            if target > 0:
                batch_documents, batch_qa, batch_membership, batch_details = _build_expanded(
                    new_documents,
                    new_qa,
                    dataset_name=dataset_name,
                    target=target,
                    tolerance=tolerance,
                    seed=seed,
                )
                batch_documents, batch_qa, batch_membership = _namespace_expanded_batch(
                    batch_documents,
                    batch_qa,
                    batch_membership,
                    prefix=_append_batch_prefix(
                        previous_source_count=previous_count,
                        target_source_count=int(max_documents),
                    ),
                )
                mode = "random_document_groups"
            else:
                batch_documents = dict(new_documents)
                batch_qa = list(new_qa)
                batch_membership = {doc_id: [doc_id] for doc_id in batch_documents}
                batch_details = {
                    "source_group_count": len(batch_documents),
                    "emitted_context_count": len(batch_documents),
                    "groups_with_trimmed_variants": 0,
                }
                mode = "no_expansion"

            overlap = set(old_documents) & set(batch_documents)
            if overlap:
                raise RuntimeError(
                    f"Append-only preparation generated duplicate document IDs: {sorted(overlap)[:5]}"
                )
            prepared_documents = {**old_documents, **batch_documents}
            prepared_qa = [*old_qa, *batch_qa]
            membership = {**old_membership, **batch_membership}
            expansion_details = {
                "source_group_count": int(old_metadata.get("source_group_count", len(old_documents)))
                + int(batch_details.get("source_group_count", 0)),
                "emitted_context_count": len(prepared_documents),
                "groups_with_trimmed_variants": int(
                    old_metadata.get("groups_with_trimmed_variants", 0)
                )
                + int(batch_details.get("groups_with_trimmed_variants", 0)),
            }
            append_update = {
                "mode": "append_only",
                "previous_max_source_documents": previous_count,
                "new_max_source_documents": int(max_documents),
                "added_source_documents": len(new_documents),
                "preserved_prepared_document_ids": list(old_documents),
                "previous_documents_sha256": _sha256(_paths(output_dir)["documents"]),
                "previous_qa_sha256": _sha256(_paths(output_dir)["qa"]),
            }
            print(
                f"Extending frozen sample in place: {previous_count} -> {max_documents} "
                f"source documents; preserving {len(old_documents)} prepared contexts and "
                f"adding {len(batch_documents)} new contexts.",
                flush=True,
            )
        elif target > 0:
            prepared_documents, prepared_qa, membership, expansion_details = _build_expanded(
                selected_documents,
                selected_qa,
                dataset_name=dataset_name,
                target=target,
                tolerance=tolerance,
                seed=seed,
            )
            mode = "random_document_groups"
        else:
            prepared_documents = dict(selected_documents)
            prepared_qa = list(selected_qa)
            membership = {doc_id: [doc_id] for doc_id in prepared_documents}
            expansion_details = {
                "source_group_count": len(prepared_documents),
                "emitted_context_count": len(prepared_documents),
                "groups_with_trimmed_variants": 0,
            }
            mode = "no_expansion"

        validation = _validate(
            prepared_documents,
            prepared_qa,
            membership,
            dataset_name=dataset_name,
            source_documents=selected_documents,
        )
        lengths = [count_tokens(text, tokenizer=None) for text in prepared_documents.values()]
        metadata: dict[str, Any] = {
            "dataset": dataset_name,
            "split": split,
            "mode": mode,
            "target_context_tokens": target,
            "tolerance_tokens": tolerance,
            "expansion_seed": seed,
            "max_source_documents": max_documents,
            "samples_per_hop": samples_per_hop,
            "hop_query_counts": (
                {
                    str(hop): sum(
                        1
                        for entry in prepared_qa
                        if str(
                            entry.get("source_document_id", entry.get("document_id", ""))
                        ).lower().startswith(f"{hop}hop")
                    )
                    for hop in MUSIQUE_HOPS
                }
                if samples_per_hop is not None
                else None
            ),
            "available_source_documents": len(documents),
            "selected_source_documents": len(selected_documents),
            "available_qa_entries": len(qa_entries),
            "qa_entries": len(prepared_qa),
            "prepared_documents": len(prepared_documents),
            "document_id_mode": "source_document_id" if target == 0 else "expanded_group_id",
            "token_counter": "whitespace",
            "span_validation_mode": (
                "literal_verbatim" if dataset_name == "musique" else "existing_labeling_normalized_tokens"
            ),
            "prepared_token_lengths": {
                "min": min(lengths),
                "median": median(lengths),
                "max": max(lengths),
            },
            "source": source_meta,
            **expansion_details,
            **validation,
        }
        if append_update is not None:
            metadata["append_update"] = append_update
        paths = _paths(output_dir)
        _write_json_atomic(paths["documents"], prepared_documents)
        _write_json_atomic(paths["qa"], prepared_qa)
        _write_json_atomic(paths["membership"], membership)
        _write_json_atomic(paths["metadata"], metadata)
        print(f"Prepared and froze dataset: {output_dir}")
        print(json.dumps(metadata, indent=2))
        return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--target-context-tokens",
        type=int,
        default=0,
        help="0 preserves original documents; a positive value enables existing random grouping.",
    )
    parser.add_argument("--tolerance-tokens", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-documents",
        default="all",
        help="Number of original documents to select, or 'all'.",
    )
    parser.add_argument(
        "--samples-per-hop",
        type=int,
        default=None,
        help="For MuSiQue, select this many 2-hop, 3-hop, and 4-hop instances.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    raw_max = str(args.max_documents).strip().lower()
    max_documents = None if raw_max in {"all", "none", ""} else int(raw_max)
    if max_documents is not None and max_documents <= 0:
        raise ValueError("--max-documents must be positive or 'all'.")
    if args.samples_per_hop is not None and args.samples_per_hop <= 0:
        raise ValueError("--samples-per-hop must be positive.")
    if args.target_context_tokens < 0:
        raise ValueError("--target-context-tokens must be >= 0.")
    prepare(
        config_path=args.config.resolve(),
        output_dir=args.output_dir.resolve(),
        target=int(args.target_context_tokens),
        tolerance=int(args.tolerance_tokens),
        seed=int(args.seed),
        max_documents=max_documents,
        samples_per_hop=args.samples_per_hop,
        force=bool(args.force),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
