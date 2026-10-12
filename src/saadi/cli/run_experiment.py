"""CLI command that constructs a Saadi index from dataset inputs."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from ..config.loader import load_config
from ..config.validation import validate_experiment
from ..indexing.dataset_loaders import load_dataset_from_config
from ..pipelines.experiment_runner import run_experiment

logger = logging.getLogger(__name__)


def _collect_cli_overrides(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {}
    for key in (
        "documents_json",
        "qa_json",
        "use_cache",
        "cache_save_every_steps",
        "cache_save_each_document",
        "schedule",
        "wavefront_batch_size",
        "max_documents",
        "max_qa_entries",
    ):
        value = getattr(args, key, None)
        if value is None:
            continue
        if isinstance(value, bool) and not value:
            continue
        overrides[key] = value
    return overrides


def _log_effective_experiment_config(*, config: dict, config_path: Path, args: argparse.Namespace) -> None:
    ingest_cfg = config.get("ingest", {}) or {}
    retrieval_cfg = config.get("retrieval", {}) or {}
    dataset_cfg = config.get("dataset", {}) or {}
    model_cfg = config.get("model", {}) or {}
    sample_cfg = config.get("sample", {}) or {}
    logger.info(
        "Experiment config summary: config=%s dataset=%s output_dir=%s strategy=%s chunk_size=%s use_cache=%s "
        "schedule=%s wavefront_batch_size=%s retriever=%s retrieve_k=%s retrieval_mode=%s model_backend=%s model_alias=%s "
        "max_documents=%s max_qa_entries=%s",
        config_path,
        dataset_cfg.get("name", dataset_cfg.get("source", "unknown")),
        config.get("output_dir", "outputs"),
        ingest_cfg.get("strategy", "hierarchical"),
        ingest_cfg.get("chunk_size", 500),
        ingest_cfg.get("use_cache", False),
        ingest_cfg.get("schedule", "document"),
        ingest_cfg.get("wavefront_batch_size", None),
        retrieval_cfg.get("retriever", "bm25"),
        retrieval_cfg.get("retrieve_k", 10),
        retrieval_cfg.get("mode", "both"),
        model_cfg.get("backend", "unknown"),
        model_cfg.get("alias", model_cfg.get("model_id", "unknown")),
        sample_cfg.get("max_documents", None),
        sample_cfg.get("max_qa_entries", None),
    )
    cli_overrides = _collect_cli_overrides(args)
    if cli_overrides:
        logger.info("Experiment CLI overrides:\n%s", json.dumps(cli_overrides, indent=2, ensure_ascii=False, default=str))
    logger.info("Effective experiment config:\n%s", json.dumps(config, indent=2, ensure_ascii=False, default=str))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Construct a Saadi index; use scripts/run_study.py for retrieval and evaluation")
    p.add_argument("--config", required=True)
    p.add_argument("--documents_json", required=False, help="Optional path to JSON object mapping doc_id->text")
    p.add_argument("--qa_json", required=False, help="Optional path to QA entries JSON list")
    p.add_argument("--use_cache", action="store_true", help="Reuse cached canonicalized chunk outputs when available")
    p.add_argument("--cache_save_every_steps", type=int, default=None, help="Checkpoint cache every N processed chunks")
    p.add_argument("--cache_save_each_document", action="store_true", help="Checkpoint cache at each document boundary")
    p.add_argument(
        "--schedule",
        choices=["document", "sequential", "wavefront"],
        default=None,
        help="Ingest scheduling mode override",
    )
    p.add_argument(
        "--wavefront_batch_size",
        type=int,
        default=None,
        help="Optional micro-batch size inside each wavefront step",
    )
    p.add_argument("--max_documents", type=int, default=None, help="Optional cap for quick experiments (first N documents)")
    p.add_argument(
        "--max_qa_entries",
        type=int,
        default=None,
        help="Optional cap for quick experiments after document filtering (first N QA entries)",
    )
    p.add_argument(
        "--ingest_only",
        action="store_true",
        help="Build/checkpoint the document index without running retrieval or evaluation",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    config = load_config(args.config)
    config_path = Path(args.config).resolve()
    logger.info("Loaded experiment config from %s", config_path)
    ingest_cfg = config.get("ingest", {}) or {}
    config["ingest"] = ingest_cfg
    if args.use_cache:
        ingest_cfg["use_cache"] = True
    if args.cache_save_every_steps is not None:
        ingest_cfg["cache_save_every_steps"] = int(args.cache_save_every_steps)
    if args.cache_save_each_document:
        ingest_cfg["cache_save_each_document"] = True
    if args.schedule is not None:
        ingest_cfg["schedule"] = str(args.schedule)
    if args.wavefront_batch_size is not None:
        ingest_cfg["wavefront_batch_size"] = int(args.wavefront_batch_size)
    sample_cfg = config.get("sample", {}) or {}
    config["sample"] = sample_cfg
    if args.max_documents is not None:
        sample_cfg["max_documents"] = int(args.max_documents)
    if args.max_qa_entries is not None:
        sample_cfg["max_qa_entries"] = int(args.max_qa_entries)
    if args.ingest_only:
        retrieval_cfg = config.get("retrieval", {}) or {}
        config["retrieval"] = retrieval_cfg
        retrieval_cfg["enabled"] = False
    validate_experiment(config)
    _log_effective_experiment_config(config=config, config_path=config_path, args=args)
    dataset_meta = {}
    if args.documents_json and args.qa_json:
        with open(args.documents_json, "r", encoding="utf-8") as f:
            docs = json.load(f)
        with open(args.qa_json, "r", encoding="utf-8") as f:
            qa = json.load(f)
        logger.info(
            "Loaded dataset from CLI files docs=%s qa=%s",
            Path(args.documents_json).resolve(),
            Path(args.qa_json).resolve(),
        )
        dataset_meta = {
            "dataset_source": "cli_files",
            "documents_json": str(Path(args.documents_json).resolve()),
            "qa_json": str(Path(args.qa_json).resolve()),
            "cached": False,
        }
    elif args.documents_json or args.qa_json:
        raise ValueError("Provide both --documents_json and --qa_json, or omit both and configure dataset in YAML.")
    else:
        docs, qa, dataset_meta = load_dataset_from_config(
            config.get("dataset", {}),
            base_dir=config_path.parent,
        )
        logger.info(
            "Loaded dataset from config source=%s docs=%d qa_entries=%d cached=%s",
            dataset_meta.get("dataset_source", "unknown"),
            len(docs),
            len(qa),
            dataset_meta.get("cached", False),
        )
    logger.info("Running experiment pipeline")
    out = run_experiment(config, documents_by_id=docs, qa_entries=qa)
    logger.info(
        "Experiment finished mode=%s output_dir=%s",
        out.get("retrieval_mode", "unknown"),
        out["artifacts"].root,
    )
    summary = {
        "retrieval_mode": out.get("retrieval_mode", "disabled"),
        "output_dir": str(out["artifacts"].root),
        "active_entity_config": out.get("active_entity_config", "AE_5_2_1"),
        "active_entity_selector": out.get("active_entity_selector", "bm25"),
        "active_entity_min_relevance": out.get("active_entity_min_relevance", 0.0),
        "rejected_entity_count": out.get("rejected_entity_count", 0),
        "active_entity_limits": out.get("active_entity_limits", {}),
        "sample_stats": out.get("sample_stats", {}),
        "dataset": dataset_meta,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
