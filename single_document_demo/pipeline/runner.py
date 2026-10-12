from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from saadi.pipelines.single_document_pipeline import (
    build_backend,
    build_or_update_chunks_and_memory,
    generate_answer,
    load_questions_jsonl,
    load_single_document,
    prepare_context_for_generation,
    retrieve_for_query,
    run_multi_question_pipeline,
    save_demo_artifacts,
)

from single_document_demo.input_adapters import (
    ConversationCsvAdapter,
    PreparedDocument,
    TextAdapter,
)

from .artifacts import (
    attach_document_summary,
    attach_cleaned_outputs,
    write_json,
)
from .config import (
    DemoConfig,
    PromptedPipelineConfig,
    REPO_ROOT,
    WORKFLOW_THERAPY,
)
from .prompts import MemoryPromptProfile, load_prompt_profile
from .summaries import export_cleaned_outputs, generate_document_summary


def _safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip()).strip("._-")
    return cleaned or "document"


def _folder_key(value: str) -> str:
    """Validate, but otherwise preserve, an input folder name."""

    name = str(value or "").strip()
    if not name or name in {".", "..", "_batches"} or Path(name).name != name:
        raise ValueError(f"Input folder cannot be mirrored safely: {value!r}")
    return name


@dataclass(slots=True)
class DocumentRunRecord:
    document_id: str
    input_folder: str
    source_path: str
    input_format: str
    run_dir: str
    normalized_document: str
    status: str
    run_mode: str
    memory_summary: str | None = None
    cleaned_outputs_manifest: str | None = None
    document_summary: str | None = None
    run_manifest: str | None = None
    questions_summary: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DemoRunResult:
    batch_manifest_path: Path
    records: list[DocumentRunRecord]

    @property
    def failures(self) -> list[DocumentRunRecord]:
        return [record for record in self.records if record.status != "completed"]


def _adapter_for(config: DemoConfig):
    if config.workflow == WORKFLOW_THERAPY:
        return ConversationCsvAdapter(
            role_column=config.role_column,
            text_column=config.text_column,
            order_column=config.order_column,
            allowed_roles=config.allowed_roles,
            role_aliases=config.role_aliases,
            include_timestamps=config.include_timestamps,
        )
    return TextAdapter()


def discover_sources(config: DemoConfig) -> list[Path]:
    path = config.input_path
    expected_suffix = ".csv" if config.workflow == WORKFLOW_THERAPY else ".txt"
    if path.is_file():
        if path.suffix.lower() != expected_suffix:
            raise ValueError(
                f"The {config.workflow!r} workflow requires {expected_suffix} input: {path}"
            )
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Configured input does not exist: {path}")
    sources = sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() == expected_suffix
    )
    if not sources:
        raise ValueError(f"No {expected_suffix} input files found in: {path}")
    return sources


def _input_folder_name(config: DemoConfig, source: Path) -> str:
    """Return the folder key mirrored below ``output/``."""

    canonical_input_root = (config.demo_root / "input").resolve()
    resolved_source = source.resolve()
    try:
        relative = resolved_source.relative_to(canonical_input_root)
    except ValueError:
        relative = None
    if relative is not None:
        if len(relative.parts) >= 2:
            return _folder_key(relative.parts[0])
        return _folder_key(source.stem)

    configured = config.input_path.resolve()
    if configured.is_file():
        return _folder_key(source.parent.name or source.stem)
    try:
        relative = resolved_source.relative_to(configured)
    except ValueError:
        return _folder_key(source.parent.name or source.stem)
    if len(relative.parts) >= 2:
        return _folder_key(relative.parts[0])
    return _folder_key(configured.name or source.stem)


def prepare_documents(config: DemoConfig) -> list[PreparedDocument]:
    adapter = _adapter_for(config)
    documents = [adapter.prepare(source) for source in discover_sources(config)]
    seen: set[str] = set()
    seen_folders: set[str] = set()
    for document in documents:
        safe = _safe_id(document.document_id)
        if safe in seen:
            raise ValueError(f"Multiple source files resolve to the same document id: {safe}")
        document.document_id = safe
        seen.add(safe)
        input_folder = _input_folder_name(config, document.source_path)
        if input_folder in seen_folders:
            raise ValueError(
                "Each input folder must contain exactly one supported document; "
                f"multiple files map to {input_folder!r}"
            )
        document.metadata["input_folder"] = input_folder
        seen_folders.add(input_folder)
    return documents


def _resolve_questions_path(raw_path: str) -> Path | None:
    value = str(raw_path or "").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_run_mode(config: DemoConfig) -> tuple[str, list[str]]:
    """Infer memory/query mode from the public CLI options."""

    questions_path = _resolve_questions_path(config.questions_path)
    if config.query and questions_path is not None:
        raise ValueError("Use either a single query or a questions file, not both")
    if config.workflow == WORKFLOW_THERAPY:
        if config.query or questions_path is not None:
            raise ValueError(
                "The therapy workflow reproduces conversation profile and is memory-only; "
                "use the text workflow for query answering"
            )
        return "memory_only", []
    if questions_path is not None:
        questions = load_questions_jsonl(questions_path)
        if not questions:
            raise ValueError(f"Questions file contains no questions: {questions_path}")
        return "multi_question", questions
    if config.query:
        return "single_query", []
    return "memory_only", []


def _pipeline_config(
    demo: DemoConfig,
    document: PreparedDocument,
    profile: MemoryPromptProfile,
    *,
    normalized_path: Path,
    run_name: str,
    output_dir: Path,
    run_mode: str,
) -> PromptedPipelineConfig:
    source_hash = hashlib.sha256(document.text.encode("utf-8")).hexdigest()
    internal_id = _safe_id(
        f"{document.document_id}__{profile.name}_{profile.content_hash[:10]}"
        f"__source_{source_hash[:10]}"
    )
    values: dict[str, Any] = {
        "demo_root": str(demo.demo_root),
        "document_path": str(normalized_path),
        "document_id": internal_id,
        "run_name": run_name,
        "output_dir": str(output_dir),
        "query": demo.query,
        "run_mode": run_mode,
        "questions_path": demo.questions_path,
        "model_config": dict(demo.model_config),
        "memory_prompt_profile": profile.name,
        "memory_prompt_hash": profile.content_hash,
        "memory_entity_system_prompt": profile.entity_extraction_prompt,
        "memory_rollup_system_prompt": profile.hierarchy_summary_prompt,
    }
    saadi_options = dict(demo.saadi_options)
    aliases = {
        "max_objects": "max_entities",
        "active_object_selector": "active_entity_selector",
    }
    for legacy_key, canonical_key in aliases.items():
        if legacy_key not in saadi_options:
            continue
        if canonical_key in saadi_options and saadi_options[canonical_key] != saadi_options[legacy_key]:
            raise ValueError(
                f"Conflicting [saadi] settings: {canonical_key!r} and deprecated {legacy_key!r}."
            )
        saadi_options.setdefault(canonical_key, saadi_options[legacy_key])
        saadi_options.pop(legacy_key, None)
    valid_fields = set(PromptedPipelineConfig.__dataclass_fields__)
    unknown = sorted(set(saadi_options) - valid_fields)
    if unknown:
        raise ValueError("Unknown [saadi] configuration keys: " + ", ".join(unknown))
    values.update(saadi_options)
    return PromptedPipelineConfig(**values)


def _run_document(
    demo: DemoConfig,
    document: PreparedDocument,
    profile: MemoryPromptProfile,
    *,
    run_name: str,
    output_dir: Path,
    run_mode: str,
    questions: list[str],
    backend: Any | None,
) -> tuple[DocumentRunRecord, Any]:
    run_dir = output_dir
    source_dir = run_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = source_dir / "normalized_document.txt"
    normalized_path.write_text(document.text, encoding="utf-8")

    config = _pipeline_config(
        demo,
        document,
        profile,
        normalized_path=normalized_path,
        run_name=run_name,
        output_dir=run_dir,
        run_mode=run_mode,
    )
    if backend is None:
        backend = build_backend(config)
    loaded = load_single_document(config)
    ingest = build_or_update_chunks_and_memory(
        config, loaded.documents_by_id, backend=backend
    )
    document_summary_path = generate_document_summary(
        config,
        profile,
        ingest.canonicalized_chunks,
        backend=backend,
        run_dir=run_dir,
    )
    cleaned_outputs = export_cleaned_outputs(
        ingest,
        run_dir,
        memory_summary_path=ingest.memory_summary_path,
        document_summary_path=document_summary_path,
    )
    aliases_removed = cleaned_outputs.aliases_removed if cleaned_outputs else 0

    questions_summary: Path | None = None
    if run_mode == "memory_only":
        manifest = save_demo_artifacts(
            config,
            loaded_document=loaded,
            ingest=ingest,
            backend=backend,
        )
    elif run_mode == "single_query":
        retrieval = retrieve_for_query(
            config,
            ingest.raw_chunks,
            ingest.canonicalized_chunks,
            artifacts=ingest.artifacts,
        )
        context = prepare_context_for_generation(
            config,
            retrieval.retrieval_result,
            ingest.raw_chunks,
            ingest.canonicalized_chunks,
            artifacts=ingest.artifacts,
        )
        generation = generate_answer(config, context.context_text, backend=backend)
        manifest = save_demo_artifacts(
            config,
            loaded_document=loaded,
            ingest=ingest,
            retrieval=retrieval,
            context=context,
            generation=generation,
            backend=backend,
        )
    else:
        _, questions_summary = run_multi_question_pipeline(
            config, questions, backend=backend, loaded=loaded, ingest=ingest
        )
        manifest = save_demo_artifacts(
            config,
            loaded_document=loaded,
            ingest=ingest,
            backend=backend,
        )

    attach_document_summary(manifest, document_summary_path)
    attach_cleaned_outputs(
        manifest,
        cleaned_outputs.manifest_path if cleaned_outputs else None,
    )

    source_manifest = {
        "document_id": document.document_id,
        "internal_saadi_document_id": config.document_id,
        "source_path": str(document.source_path),
        "workflow": demo.workflow,
        "input_format": document.input_format,
        "normalized_document": str(normalized_path),
        "source_sha256": hashlib.sha256(document.text.encode("utf-8")).hexdigest(),
        "prompt_profile": profile.name,
        "prompt_directory": str(profile.directory),
        "prompt_content_hash": profile.content_hash,
        "cleaned_outputs_manifest": str(cleaned_outputs.manifest_path)
        if cleaned_outputs
        else None,
        "cleaned_memory_summary": (
            str(cleaned_outputs.memory_summary_path)
            if cleaned_outputs and cleaned_outputs.memory_summary_path
            else None
        ),
        "cleaned_document_summary": (
            str(cleaned_outputs.document_summary_path)
            if cleaned_outputs and cleaned_outputs.document_summary_path
            else None
        ),
        "forbidden_aliases_removed_from_memory_export": aliases_removed,
        "metadata": document.metadata,
    }
    write_json(source_dir / "source_manifest.json", source_manifest)
    return (
        DocumentRunRecord(
            document_id=document.document_id,
            input_folder=str(document.metadata["input_folder"]),
            source_path=str(document.source_path),
            input_format=document.input_format,
            run_dir=str(run_dir),
            normalized_document=str(normalized_path),
            status="completed",
            run_mode=run_mode,
            memory_summary=str(cleaned_outputs.memory_summary_path)
            if cleaned_outputs and cleaned_outputs.memory_summary_path
            else None,
            cleaned_outputs_manifest=str(cleaned_outputs.manifest_path)
            if cleaned_outputs
            else None,
            document_summary=str(cleaned_outputs.document_summary_path)
            if cleaned_outputs and cleaned_outputs.document_summary_path
            else None,
            run_manifest=str(manifest),
            questions_summary=str(questions_summary) if questions_summary else None,
            metadata=document.metadata,
        ),
        backend,
    )


def run_demo(config: DemoConfig, *, continue_on_error: bool | None = None) -> DemoRunResult:
    profile = load_prompt_profile(config.prompt_root, config.prompt_profile)
    run_mode, questions = resolve_run_mode(config)
    documents = prepare_documents(config)
    if continue_on_error is None:
        continue_on_error = config.input_path.is_dir() or len(documents) > 1

    records: list[DocumentRunRecord] = []
    backend: Any | None = None
    for document_index, document in enumerate(documents):
        input_folder = str(document.metadata["input_folder"])
        child_run_name = f"{config.run_name}/{input_folder}"
        document_output = config.output_root / input_folder
        try:
            record, backend = _run_document(
                config,
                document,
                profile,
                run_name=child_run_name,
                output_dir=document_output,
                run_mode=run_mode,
                questions=questions,
                backend=backend,
            )
            records.append(record)
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            failed_run_dir = document_output
            records.append(
                DocumentRunRecord(
                    document_id=document.document_id,
                    input_folder=input_folder,
                    source_path=str(document.source_path),
                    input_format=document.input_format,
                    run_dir=str(failed_run_dir),
                    normalized_document=str(
                        failed_run_dir / "source" / "normalized_document.txt"
                    ),
                    status="failed",
                    run_mode=run_mode,
                    error=error_text,
                    metadata=document.metadata,
                )
            )
            if not continue_on_error:
                raise
            if "Engine core initialization failed" in error_text:
                for skipped in documents[document_index + 1 :]:
                    skipped_input_folder = str(skipped.metadata["input_folder"])
                    skipped_run_dir = config.output_root / skipped_input_folder
                    records.append(
                        DocumentRunRecord(
                            document_id=skipped.document_id,
                            input_folder=skipped_input_folder,
                            source_path=str(skipped.source_path),
                            input_format=skipped.input_format,
                            run_dir=str(skipped_run_dir),
                            normalized_document=str(
                                skipped_run_dir / "source" / "normalized_document.txt"
                            ),
                            status="failed",
                            run_mode=run_mode,
                            error=(
                                "Skipped because the shared vLLM engine failed to "
                                "initialize for an earlier document."
                            ),
                            metadata=skipped.metadata,
                        )
                    )
                break

    batch_root = config.output_root / "_batches" / _safe_id(config.run_name)
    batch_manifest = write_json(
        batch_root / "batch_manifest.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "workflow": config.workflow,
            "run_mode": run_mode,
            "config_path": str(config.config_path),
            "input_path": str(config.input_path),
            "output_root": str(config.output_root),
            "prompt_profile": profile.name,
            "prompt_directory": str(profile.directory),
            "prompt_content_hash": profile.content_hash,
            "documents_total": len(documents),
            "documents_completed": sum(record.status == "completed" for record in records),
            "documents_failed": sum(record.status != "completed" for record in records),
            "documents": [asdict(record) for record in records],
        },
    )
    return DemoRunResult(batch_manifest_path=batch_manifest, records=records)
