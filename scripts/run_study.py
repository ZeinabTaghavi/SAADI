#!/usr/bin/env python3
"""Run Saadi construction, retrieval, and ablation configurations."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml
from saadi.config.loader import load_config
from saadi.config.validation import validate_experiment


def merge(base, update):
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def execute(command, dry_run):
    print(shlex.join(map(str, command)), flush=True)
    if not dry_run:
        subprocess.run(list(map(str, command)), cwd=ROOT, check=True)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def source_digest():
    digest = hashlib.sha256()
    for directory in (ROOT / "src", ROOT / "scripts"):
        for path in sorted(directory.rglob("*.py")):
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def published_config(config):
    """Restore recorded construction settings without renaming the saved results."""
    from saadi.reporting.tables import verify_bundle
    manifest = verify_bundle(ROOT / "outputs/sadi/reference")
    record = next((s for s in manifest["sources"] if s["study"] == "main"
                   and s["dataset"] == config["dataset"]["name"] and s["retriever"] == "bm25"), None)
    if not record or not record.get("scientific_config"):
        raise ValueError("No saved construction configuration for this dataset")
    saved = deepcopy(record["scientific_config"])
    strategy = saved["ingest"]["strategy_kwargs"]
    for old, new in (("active_object_selector", "active_entity_selector"), ("max_objects", "max_entities")):
        if old in strategy:
            strategy[new] = strategy.pop(old)
    # The saved constructions showed the model the anchored chunk copies as L0
    # history; new runs default to the original chunk text.
    strategy.setdefault("history_l0_text", "anchored")
    # They also located anchors with strict context matching.
    strategy.setdefault("anchor_matching", "strict")
    result = merge(config, {"ingest": saved["ingest"]})
    # Replace the model mapping: absent historical sampling fields must not inherit new defaults.
    result["model"] = saved["model"]
    return result


def sha256_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_hashes(directory):
    return {str(p.relative_to(directory)): sha256_file(p)
            for p in sorted(directory.rglob("*")) if p.is_file()
            and p.name not in {".stage.json", ".stage.tmp", ".run.lock"}
            and not {"__pycache__", "evaluation", "generation_pipeline"}.intersection(p.relative_to(directory).parts)}


@contextmanager
def stage_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process is writing {directory}") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def execute_stage(command, directory, *, dry_run, inputs):
    if dry_run:
        execute(command, True)
        return
    with stage_lock(directory):
        _execute_stage(command, directory, inputs=inputs)


def verified_stage(directory, *, inputs=None):
    """Require a completed, intact stage before its artifacts are consumed."""
    marker = directory / ".stage.json"
    if not marker.is_file():
        raise FileNotFoundError(f"No completed construction stage: {directory}. Run --stage ingest first.")
    state = json.loads(marker.read_text())
    if (inputs is not None and state.get("inputs") != inputs) or state.get("artifacts") != file_hashes(directory):
        raise ValueError(f"Completed stage changed: {directory}. Use a new output directory to rerun.")
    return state["artifacts"]


def _execute_stage(command, directory, *, inputs):
    marker = directory / ".stage.json"
    if marker.exists():
        verified_stage(directory, inputs=inputs)
        print(f"Verified completed stage: {directory}", flush=True)
        return
    execute(command, False)
    directory.mkdir(parents=True, exist_ok=True)
    state = {"inputs": inputs, "artifacts": file_hashes(directory)}
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, sort_keys=True, indent=2))
    temporary.replace(marker)


def save_config(path, config, dry_run):
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = yaml.safe_dump(config, sort_keys=False)
        if path.exists() and path.read_text() != content:
            raise ValueError(f"Refusing to change a resolved configuration: {path}")
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        temporary.replace(path)


def prepare_config(config, *, dry_run, execute_prepare):
    preparation = config.pop("preparation", None)
    if not preparation:
        return config
    dataset = config["dataset"]["name"]
    source_hashes = {key: sha256_file(Path(value)) for key in ("documents_json", "qa_json")
                     if (value := config["dataset"].get(key)) and Path(value).is_file()}
    fingerprint = hashlib.sha256(json.dumps(
        {"dataset": config["dataset"], "preparation": preparation, "code": source_digest(),
         "source_hashes": source_hashes}, sort_keys=True
    ).encode()).hexdigest()[:12]
    prepared = Path(os.environ["SADI_OUTPUT_DIR"]) / dataset / "prepared" / fingerprint
    source_path = prepared / "source_config.yaml"
    save_config(source_path, {"dataset": config["dataset"]}, dry_run)
    command = [sys.executable, ROOT / "scripts/ablations/retrieval/prepare_grouped_retrieval_dataset.py",
               "--config", source_path, "--output-dir", prepared,
               "--target-context-tokens", preparation.get("target_context_tokens", 0),
               "--max-documents", preparation.get("max_documents", "all"),
               "--seed", preparation.get("seed", 42),
               "--tolerance-tokens", preparation.get("tolerance_tokens", 4000)]
    if preparation.get("samples_per_hop"):
        command += ["--samples-per-hop", preparation["samples_per_hop"]]
    if execute_prepare:
        execute(command, dry_run)
    config["dataset"].update(source="files", documents_json=str(prepared / "documents.json"),
                              qa_json=str(prepared / "qa.json"))
    return config


def resolve_dataset_paths(config, base_dir):
    dataset = config["dataset"]
    if dataset.get("name") == "novelhopqa" and os.environ.get("NOVELHOPQA_BOOKS_ROOT"):
        dataset["books_root"] = os.environ["NOVELHOPQA_BOOKS_ROOT"]
    for key in ("documents_json", "qa_json", "books_root"):
        if dataset.get(key):
            path = Path(dataset[key]).expanduser()
            dataset[key] = str((path if path.is_absolute() else base_dir / path).resolve())
    return config


def run(args):
    suite_path = ROOT / "configs/suites" / f"{args.suite}.yaml"
    suite = load_config(suite_path)
    datasets = args.datasets.split(",") if args.datasets else suite["datasets"]
    retrievers = args.retrievers.split(",") if args.retrievers else suite.get("retrievers", ["bm25"])
    code_identity = source_digest()
    supported_datasets = {p.name.removesuffix("_retrieval_ablation.yaml") for p in (ROOT / "configs/experiments").glob("*.yaml")}
    supported_retrievers = {"bm25", "contriever", "bge_m3", "e5", "jina", "qwen"}
    datasets = [d.strip() for d in datasets]
    retrievers = [r.strip() for r in retrievers]
    if not set(datasets) <= supported_datasets or not set(retrievers) <= supported_retrievers:
        raise ValueError("Unknown dataset or retriever")
    for dataset in datasets:
        dataset = dataset.strip()
        experiment_path = ROOT / "configs/experiments" / f"{dataset}_retrieval_ablation.yaml"
        base = load_config(experiment_path)
        if args.profile == "published":
            base = published_config(base)
        for variant in suite.get("variants", [{"name": "default"}]):
            cfg = merge(base, variant.get("overrides", {}))
            if variant.get("model_config"):
                cfg["model"] = load_config(ROOT / variant["model_config"])
            resolve_dataset_paths(cfg, experiment_path.parent)
            if args.tensor_parallel_size is not None:
                cfg["model"]["tensor_parallel_size"] = args.tensor_parallel_size
            if args.max_documents is not None:
                cfg.setdefault("sample", {})["max_documents"] = args.max_documents
                if cfg.get("preparation"):
                    cfg["preparation"]["max_documents"] = args.max_documents
                    cfg["preparation"].pop("samples_per_hop", None)
            validate_experiment(cfg)
            output = Path(base["output_dir"])
            cfg = prepare_config(cfg, dry_run=args.dry_run, execute_prepare=args.stage in {"all", "prepare"})
            input_hashes = {}
            for key in ("documents_json", "qa_json"):
                value = cfg["dataset"].get(key)
                if value:
                    path = Path(value)
                    if not args.dry_run and not path.is_file():
                        raise FileNotFoundError(f"Run --stage prepare first: {path}")
                    input_hashes[key] = sha256_file(path) if path.is_file() else "planned"
            index_settings = {key: cfg.get(key) for key in ("dataset", "ingest", "model", "sample")}
            index_settings.update(code=code_identity, input_hashes=input_hashes, run_number=variant.get("run_number"))
            cfg["output_dir"] = str(output / "indexes" / fingerprint(index_settings))
            run_id = fingerprint({"config": cfg, "retrievers": retrievers, "profile": args.profile})
            run_root = output / "runs" / args.suite / variant["name"] / run_id
            config_path = run_root / "runtime_config.yaml"
            save_config(config_path, cfg, args.dry_run)
            print(f"Saadi: {dataset} / {args.suite} / {variant['name']}", flush=True)
            if args.stage in {"all", "ingest"}:
                execute_stage([sys.executable, "-m", "saadi.cli.main", "run-experiment", "--config", config_path,
                               "--ingest_only"], Path(cfg["output_dir"]), dry_run=args.dry_run, inputs=index_settings)
            index_artifacts = {}
            if not args.dry_run and args.stage in {"all", "retrieve", "generate"}:
                index_artifacts = verified_stage(Path(cfg["output_dir"]), inputs=index_settings)
            for retriever in retrievers:
                if args.stage in {"all", "retrieve"}:
                    execute_stage([sys.executable, "-m", "saadi.cli.main", "retrieve-cached", "--config", config_path,
                                   "--retriever", retriever, "--output_dir", run_root / retriever],
                                  run_root / retriever, dry_run=args.dry_run,
                                  inputs={"config": cfg, "retriever": retriever,
                                          "index": index_artifacts})
                    if args.suite == "multihop":
                        execute([sys.executable, ROOT / "scripts/evaluation/retrieval/evaluate_multihop.py",
                                 "--config", config_path, "--results-dir", run_root / retriever / "retrieval_reports",
                                 "--output-dir", run_root / "multihop" / retriever], args.dry_run)
                if suite.get("generation") and args.stage in {"all", "generate"}:
                    if not args.dry_run:
                        verified_stage(run_root / retriever, inputs={"config": cfg, "retriever": retriever,
                                                                    "index": index_artifacts})
                    from saadi.indexing.retrieval_methods.mbr import format_mbr_method_label
                    settings = cfg["cached_retrieval"]
                    label = format_mbr_method_label(**{key: settings[key] for key in (
                        "lambda_t", "lambda_n", "lambda_h", "normalization_method", "memory_name_weight",
                        "memory_alias_weight", "memory_description_weight")})
                    results = run_root / retriever / "retrieval_reports" / f"retrieval_results__{retriever}__{label}__per_document.json"
                    for experiment in suite["generation"]["experiments"]:
                        for top_k in suite["generation"]["top_k"]:
                            execute([sys.executable, "-m", "saadi.saadi_generation.run_saadi_generation",
                                     "--config", config_path, "--dataset", dataset, "--experiment", experiment,
                                     "--retriever", retriever, "--top_k", top_k, "--summary_k", 5,
                                     "--dataset_experiment_dir", run_root / "generation" / retriever,
                                     "--mbr_results_json", results, "--no_auto_mbr"], args.dry_run)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="main", choices=sorted(p.stem for p in (ROOT / "configs/suites").glob("*.yaml")))
    parser.add_argument("--profile", choices=["new", "published"], default="new", help="new: BM25 memory selection and raw-text L0 history; published: saved word-selector and anchored-history configuration")
    parser.add_argument("--datasets", help="Comma-separated dataset keys")
    parser.add_argument("--retrievers", help="Comma-separated retrievers")
    parser.add_argument("--stage", choices=["all", "prepare", "ingest", "retrieve", "generate"], default="all")
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ.setdefault("SADI_OUTPUT_DIR", str(ROOT / "outputs"))
    os.environ.setdefault("SADI_DATA_DIR", str(ROOT / "data"))
    os.environ.setdefault("SADI_CACHE_DIR", str(ROOT / ".cache"))
    for name in ("SADI_OUTPUT_DIR", "SADI_DATA_DIR", "SADI_CACHE_DIR"):
        os.environ[name] = str(Path(os.environ[name]).expanduser().resolve())
    os.environ.setdefault("NOVELHOPQA_REPORT_DIR", str(Path(os.environ["SADI_OUTPUT_DIR"]) / "novelhopqa/dataset_reports"))
    os.environ["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    if not args.dry_run:
        cache = Path(os.environ["SADI_CACHE_DIR"]).expanduser()
        if str(cache).startswith("/path/to/"):
            parser.error("Set SADI_CACHE_DIR to your cache directory; see README.md.")
        from saadi.runtime.hf_cache import configure_hf_cache_env
        configure_hf_cache_env(cache)
    if args.max_documents is not None and args.max_documents < 1:
        parser.error("--max-documents must be positive")
    if args.tensor_parallel_size is not None and args.tensor_parallel_size < 1:
        parser.error("--tensor-parallel-size must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
