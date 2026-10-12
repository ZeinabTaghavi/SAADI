"""Validate the public experiment contract before loading data or models."""
from __future__ import annotations
import math

FIELDS = {
    "": "output_dir dataset ingest retrieval cached_retrieval model evaluation sample preparation",
    "dataset": "name source display_name split config_name docs_config_name qa_config_name qa_n qa_selection_method books_root documents_json qa_json allowed_doc_ids table_methods",
    "ingest": "enabled use_cache strategy strategy_kwargs chunk_size chunk_overlap schedule wavefront_batch_size cache_save_every_steps cache_save_each_document write_raw_chunks",
    "ingest.strategy_kwargs": "active_entity_selector active_entity_min_relevance branching_factor max_entities max_aliases max_desc max_hierarchy_level memory_update_mode history_l0_text system_prompt rollup_system_prompt",
    "model": "backend model_id alias device dtype enforce_eager generate sampling load_chat_tokenizer gpu_memory_utilization max_model_len tensor_parallel_size vllm_kwargs revision tokenizer_cache_dir trust_remote_code chat_formatter_loader chat_formatter_kwargs cache_root",
    "model.generate": "max_tokens temperature stop",
    "model.sampling": "seed top_p top_k repetition_penalty",
    "retrieval": "enabled retriever retrieve_k eval_k mode scope label_source",
    "cached_retrieval": "method lambda_t lambda_n lambda_h normalization_method memory_name_weight memory_alias_weight memory_description_weight memory_aggregation augmentation_j max_aliases max_desc",
    "preparation": "max_documents samples_per_hop seed target_context_tokens tolerance_tokens",
    "sample": "max_documents max_qa_entries",
    "evaluation": "compute_per_query",
}


def validate_experiment(config: dict) -> None:
    def check(value, path):
        if not isinstance(value, dict):
            raise ValueError(f"{path or 'config'} must be a mapping")
        unknown = set(value) - set(FIELDS[path].split())
        if unknown:
            raise ValueError(f"Unknown configuration keys in {path or 'config'}: {sorted(unknown)}")
        for key, child in value.items():
            name = f"{path}.{key}" if path else key
            if name in FIELDS:
                check(child, name)
    check(config, "")
    for section in ("dataset", "ingest", "retrieval", "model"):
        if section not in config:
            raise ValueError(f"Missing {section} configuration")
    ingest, retrieval = config["ingest"], config["retrieval"]
    def positive(value, name, zero=False):
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if zero else 1):
            raise ValueError(f"{name} must be a {'nonnegative' if zero else 'positive'} integer")
    positive(ingest.get("chunk_size", 500), "chunk_size")
    positive(ingest.get("chunk_overlap", 0), "chunk_overlap", zero=True)
    if ingest.get("chunk_overlap", 0) >= ingest.get("chunk_size", 500):
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    if ingest.get("enabled") is False:
        raise ValueError("Saadi construction requires ingest.enabled=true")
    if ingest.get("strategy", "hierarchical") not in {"hierarchical", "limited_hierarchical_summary"}:
        raise ValueError("Unsupported Saadi construction strategy")
    if ingest.get("schedule", "document") not in {"document", "sequential", "wavefront"}:
        raise ValueError("Invalid ingestion schedule")
    strategy = ingest.get("strategy_kwargs", {})
    if strategy.get("active_entity_selector", "bm25") not in {"bm25", "word"}:
        raise ValueError("active_entity_selector must be bm25 or word")
    if strategy.get("history_l0_text", "raw") not in {"raw", "anchored"}:
        raise ValueError("history_l0_text must be raw or anchored")
    for key in ("branching_factor", "max_entities", "max_aliases", "max_desc"):
        if key in strategy:
            positive(strategy[key], key, zero=key != "branching_factor")
    for key in ("retrieve_k", "eval_k"):
        positive(retrieval.get(key, 10 if key == "retrieve_k" else 5), key)
    if retrieval.get("eval_k", 5) > retrieval.get("retrieve_k", 10):
        raise ValueError("eval_k exceeds retrieve_k")
    if retrieval.get("scope", "per_document") not in {"per_document", "global", "both"}:
        raise ValueError("Invalid retrieval scope")
    if retrieval.get("mode", "canonicalized") != "canonicalized":
        raise ValueError("The release supports Saadi retrieval only")
    method = config.get("cached_retrieval", {})
    if method.get("normalization_method", "none") not in {"none", "minmax", "zscore", "bert_rescoring"}:
        raise ValueError("Unsupported score normalization")
    if method.get("memory_aggregation", "max") not in {"max", "mean"}:
        raise ValueError("Unsupported memory aggregation")
    if method.get("method", "mbr") not in {"mbr", "mbr_linear", "mbr_pool", "l0", "l0aj"}:
        raise ValueError("Unsupported Saadi retrieval method")
    for key, value in method.items():
        if key.startswith("lambda_") or key.endswith("_weight"):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
    for key, value in config.get("sample", {}).items():
        if value is not None:
            positive(value, key)
    if config["model"].get("backend", "vllm") not in {"vllm", "mock"}:
        raise ValueError("model.backend must be vllm or mock")
