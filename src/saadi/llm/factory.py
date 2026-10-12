"""Builds LLM backends from model configuration, ``provider:model`` strings, or callables."""

from __future__ import annotations

import logging
import os
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from .anthropic_backend import DEFAULT_ANTHROPIC_MODEL, DEFAULT_MIN_MAX_TOKENS, AnthropicBackend
from .base import (
    DEFAULT_GENERATION_SEED,
    DEFAULT_GENERATION_TOP_K,
    DEFAULT_GENERATION_TOP_P,
    DEFAULT_REPETITION_PENALTY,
    LegacyStringBackend,
    LLMBackend,
)
from .callable_backend import CallableBackend
from .hf_tokenizers import load_hf_tokenizer
from .mock_backend import MockBackend
from .openai_backend import OLLAMA_DEFAULT_BASE_URL, OpenAICompatibleBackend
from .vllm_backend import VLLMBackend

logger = logging.getLogger(__name__)


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except Exception:
        return None


def _configured_model_revision(cfg: dict[str, Any]) -> str | None:
    for key in ("revision", "model_revision", "hf_revision"):
        value = cfg.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _commit_from_snapshot_path(value: object) -> str | None:
    parts = Path(str(value or "")).parts
    try:
        position = parts.index("snapshots")
    except ValueError:
        return None
    if position + 1 >= len(parts):
        return None
    commit = str(parts[position + 1]).strip()
    return commit or None


def _backend_runtime_metadata(
    *,
    model: object,
    tokenizer: object | None,
    cfg: dict[str, Any],
    model_id: str,
    dtype: Any,
    tensor_parallel_size: int | None,
    sampling_defaults: dict[str, object],
) -> dict[str, object]:
    """Collect resolved runtime details without relying on one vLLM layout."""

    engine = getattr(model, "llm_engine", None)
    vllm_config = getattr(engine, "vllm_config", None)
    model_config = getattr(engine, "model_config", None) or getattr(vllm_config, "model_config", None)
    parallel_config = getattr(engine, "parallel_config", None) or getattr(vllm_config, "parallel_config", None)
    hf_config = getattr(model_config, "hf_config", None)

    resolved_commit = getattr(hf_config, "_commit_hash", None)
    if not resolved_commit and tokenizer is not None:
        resolved_commit = getattr(tokenizer, "_commit_hash", None)
        init_kwargs = getattr(tokenizer, "init_kwargs", None)
        if not resolved_commit and isinstance(init_kwargs, dict):
            resolved_commit = init_kwargs.get("_commit_hash")
    if not resolved_commit:
        for source in (
            getattr(hf_config, "_name_or_path", None),
            getattr(model_config, "model", None),
            getattr(tokenizer, "name_or_path", None) if tokenizer is not None else None,
        ):
            resolved_commit = _commit_from_snapshot_path(source)
            if resolved_commit:
                break

    resolved_dtype = getattr(model_config, "dtype", None)
    if resolved_dtype is None:
        resolved_dtype = dtype
    resolved_dtype_text = str(resolved_dtype).replace("torch.", "") if resolved_dtype is not None else None

    resolved_tp = getattr(parallel_config, "tensor_parallel_size", None)
    if resolved_tp is None:
        resolved_tp = tensor_parallel_size

    effective_sampling: dict[str, object] = {
        "seed": DEFAULT_GENERATION_SEED,
        "top_p": DEFAULT_GENERATION_TOP_P,
        "top_k": DEFAULT_GENERATION_TOP_K,
        "repetition_penalty": DEFAULT_REPETITION_PENALTY,
    }
    effective_sampling.update(sampling_defaults)

    return {
        "backend": "vllm",
        "model_name": str(getattr(model_config, "model", None) or model_id),
        "hf_revision": str(
            getattr(model_config, "revision", None)
            or _configured_model_revision(cfg)
            or "main"
        ),
        "hf_commit_hash": (str(resolved_commit) if resolved_commit else None),
        "dtype": resolved_dtype_text,
        "tensor_parallel_size": (int(resolved_tp) if resolved_tp is not None else None),
        "vllm_version": _package_version("vllm"),
        "sampling": effective_sampling,
    }


def _infer_tensor_parallel_size(cfg: dict[str, Any]) -> int | None:
    explicit = cfg.get("tensor_parallel_size")
    if explicit is not None:
        return int(explicit)

    for env_name in ("SAADI_TENSOR_PARALLEL_SIZE", "TENSOR_PARALLEL_SIZE"):
        raw = str(os.getenv(env_name, "")).strip()
        if raw:
            return int(raw)

    cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES")
    if cuda_visible is None:
        return None
    raw = str(cuda_visible).strip()
    if not raw or raw == "-1":
        return None
    devs = [p.strip() for p in raw.split(",") if p.strip()]
    if len(devs) > 1:
        return len(devs)
    return None


BACKEND_ALIASES = {
    "mock": "mock",
    "dummy": "mock",
    "vllm": "vllm",
    "qwen": "vllm",
    "qwen_vllm": "vllm",
    "openai": "openai",
    "openai_compatible": "openai",
    "openai-compatible": "openai",
    "ollama": "ollama",
    "anthropic": "anthropic",
    "claude": "anthropic",
}
SUPPORTED_BACKENDS = ("mock", "vllm", "openai", "ollama", "anthropic")


def normalize_backend_name(name: Any) -> str:
    """Map a configured backend name or alias to its canonical name."""

    key = str(name if name is not None else "mock").strip().lower()
    try:
        return BACKEND_ALIASES[key]
    except KeyError:
        raise ValueError(
            f"Unsupported model backend: {name!r}. Expected one of {', '.join(SUPPORTED_BACKENDS)}."
        ) from None


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _required_model_id(cfg: dict[str, Any], provider: str) -> str:
    model = str(cfg.get("model_id") or cfg.get("model") or "").strip()
    if not model:
        raise ValueError(f"model.model_id is required for the {provider} backend")
    return model


def model_config_from_spec(spec: str) -> dict[str, Any]:
    """Turn a ``provider:model`` string into a model configuration mapping.

    Examples: ``mock``, ``vllm:Qwen/Qwen3.5-27B``, ``openai:gpt-4.1-mini``,
    ``ollama:qwen3:8b``, ``anthropic:claude-opus-5-5``, or just ``anthropic``.
    The model name is everything after the first colon, so Ollama tags keep theirs.
    """

    text = str(spec or "").strip()
    if not text:
        raise ValueError("Empty LLM spec; use provider:model, for example openai:gpt-4.1-mini")
    provider, _, model = text.partition(":")
    provider_key = provider.strip().lower()
    model = model.strip()
    if provider_key not in BACKEND_ALIASES:
        raise ValueError(
            f"Unknown LLM provider {provider!r} in {spec!r}. "
            f"Use one of {', '.join(SUPPORTED_BACKENDS)}, as in openai:gpt-4.1-mini."
        )
    backend = BACKEND_ALIASES[provider_key]
    config: dict[str, Any] = {"backend": backend}
    if backend == "mock":
        if model:
            raise ValueError("The mock backend takes no model name")
        return config
    if backend == "anthropic" and not model:
        model = DEFAULT_ANTHROPIC_MODEL
    if not model:
        raise ValueError(f"The {backend} backend needs a model name, as in {backend}:<model>")
    config["model_id"] = model
    return config


def backend_from_spec(spec: str) -> LLMBackend:
    """Build a backend from a ``provider:model`` string."""

    return build_backend_from_config(model_config_from_spec(spec))


def resolve_backend(llm: Any) -> LLMBackend:
    """Accept a backend object, a ``provider:model`` string, a config mapping, or a callable.

    Objects with ``generate_chat_batch`` are used as they are. Objects with only
    the older ``generate_batch(list[str])`` are wrapped. An object with a
    ``generate`` method, or a plain function ``(system, user, ...) -> str``,
    is wrapped in :class:`CallableBackend`.
    """

    if llm is None:
        raise ValueError("No language model given; pass a backend, a provider:model string, or a config mapping")
    if isinstance(llm, str):
        return backend_from_spec(llm)
    if isinstance(llm, dict):
        return build_backend_from_config(llm)
    if callable(getattr(llm, "generate_chat_batch", None)):
        return llm
    if callable(getattr(llm, "generate_batch", None)):
        return LegacyStringBackend(llm)
    generate = getattr(llm, "generate", None)
    if callable(generate):
        return CallableBackend(generate, name=type(llm).__name__)
    if callable(llm):
        return CallableBackend(llm, name=str(getattr(llm, "__name__", "callable")))
    raise TypeError(f"Cannot use an object of type {type(llm).__name__} as a language model backend")


def build_backend_from_config(model_cfg: dict[str, Any] | None) -> LLMBackend:
    """Build an LLM backend from a ``model`` configuration mapping.

    ``backend`` selects the implementation: ``mock``, ``vllm``, ``openai``
    (any OpenAI-compatible server), ``ollama``, or ``anthropic``. Each backend
    documents the keys it reads.
    """

    cfg = dict(model_cfg or {})
    backend = normalize_backend_name(cfg.get("backend", "mock"))
    if backend == "mock":
        return MockBackend()
    if backend == "vllm":
        return _build_vllm_backend(cfg)
    if backend in {"openai", "ollama"}:
        return _build_openai_compatible_backend(cfg, provider=backend)
    if backend == "anthropic":
        return _build_anthropic_backend(cfg)
    raise ValueError(f"Unsupported model backend: {backend}")  # pragma: no cover - normalize_backend_name guards this


def _build_openai_compatible_backend(cfg: dict[str, Any], *, provider: str) -> OpenAICompatibleBackend:
    model = _required_model_id(cfg, provider)
    base_url = cfg.get("base_url")
    api_key_env = cfg.get("api_key_env", "OPENAI_API_KEY")
    if provider == "ollama":
        base_url = base_url or os.environ.get("OLLAMA_BASE_URL") or OLLAMA_DEFAULT_BASE_URL
        api_key_env = cfg.get("api_key_env", "OLLAMA_API_KEY")
    sampling = cfg.get("sampling") or {}
    seed = sampling.get("seed") if isinstance(sampling, dict) else None
    # A local Ollama server answers one request at a time, so queue less and wait longer.
    default_workers, default_timeout = (2, 600.0) if provider == "ollama" else (8, 300.0)
    return OpenAICompatibleBackend(
        model,
        base_url=base_url,
        api_key=cfg.get("api_key"),
        api_key_env=api_key_env,
        max_workers=int(cfg.get("max_workers", default_workers)),
        max_retries=int(cfg.get("max_retries", 4)),
        timeout=float(cfg.get("timeout", default_timeout)),
        seed=(int(seed) if seed is not None else None),
        token_limit_param=str(cfg.get("token_limit_param", "auto")),
        send_temperature=bool(cfg.get("send_temperature", True)),
        send_stop=bool(cfg.get("send_stop", True)),
        request_kwargs=_mapping(cfg.get("request_kwargs"), "model.request_kwargs"),
        extra_body=_mapping(cfg.get("extra_body"), "model.extra_body"),
        provider_label=provider,
    )


def _build_anthropic_backend(cfg: dict[str, Any]) -> AnthropicBackend:
    model = str(cfg.get("model_id") or cfg.get("model") or DEFAULT_ANTHROPIC_MODEL).strip()
    api_key = cfg.get("api_key")
    api_key_env = cfg.get("api_key_env")
    if not api_key and api_key_env:
        api_key = os.environ.get(str(api_key_env)) or None
    return AnthropicBackend(
        model,
        api_key=api_key,
        base_url=cfg.get("base_url"),
        max_workers=int(cfg.get("max_workers", 8)),
        max_retries=int(cfg.get("max_retries", 4)),
        timeout=float(cfg.get("timeout", 600.0)),
        effort=cfg.get("effort"),
        thinking=_mapping(cfg.get("thinking"), "model.thinking") or None,
        send_temperature=bool(cfg.get("send_temperature", False)),
        min_max_tokens=int(cfg.get("min_max_tokens", DEFAULT_MIN_MAX_TOKENS)),
        fallbacks=cfg.get("fallbacks"),
        request_kwargs=_mapping(cfg.get("request_kwargs"), "model.request_kwargs"),
        extra_body=_mapping(cfg.get("extra_body"), "model.extra_body"),
    )


def _build_vllm_backend(cfg: dict[str, Any]) -> VLLMBackend:
    model_id = str(cfg.get("model_id", "Qwen/Qwen3.5-27B"))
    model_revision = _configured_model_revision(cfg)
    trust_remote_code = bool(cfg.get("trust_remote_code", True))
    dtype = cfg.get("dtype", "auto")
    tensor_parallel_size = _infer_tensor_parallel_size(cfg)
    gpu_memory_utilization = cfg.get("gpu_memory_utilization")
    max_model_len = cfg.get("max_model_len")
    enforce_eager = cfg.get("enforce_eager")
    extra_kwargs = cfg.get("vllm_kwargs", {}) or {}
    if not isinstance(extra_kwargs, dict):
        extra_kwargs = {}
    sampling_defaults = cfg.get("sampling", {}) or {}
    if not isinstance(sampling_defaults, dict):
        sampling_defaults = {}
    else:
        # Runtime call overrides these fields.
        sampling_defaults = {
            k: v
            for k, v in sampling_defaults.items()
            if k not in {"max_tokens", "temperature", "stop"}
        }
    load_chat_tokenizer = bool(cfg.get("load_chat_tokenizer", True))
    tokenizer_cache_dir = cfg.get("tokenizer_cache_dir")
    chat_formatter_loader = str(
        cfg.get("chat_formatter_loader", "tokenizer") or "tokenizer"
    ).strip().lower()
    chat_formatter_kwargs = cfg.get("chat_formatter_kwargs", {}) or {}
    if not isinstance(chat_formatter_kwargs, dict):
        raise ValueError("model.chat_formatter_kwargs must be a mapping")

    try:
        from vllm import LLM  # type: ignore
    except Exception as exc:
        raise RuntimeError("vLLM backend requested but `vllm` is not installed.") from exc

    llm_kwargs: dict[str, Any] = {
        "model": model_id,
        "trust_remote_code": trust_remote_code,
        "seed": int(sampling_defaults.get("seed", DEFAULT_GENERATION_SEED)),
    }
    if model_revision:
        llm_kwargs["revision"] = model_revision
    if dtype is not None:
        llm_kwargs["dtype"] = dtype
    if tensor_parallel_size is not None:
        llm_kwargs["tensor_parallel_size"] = int(tensor_parallel_size)
    if gpu_memory_utilization is not None:
        llm_kwargs["gpu_memory_utilization"] = float(gpu_memory_utilization)
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = int(max_model_len)
    if enforce_eager is not None:
        llm_kwargs["enforce_eager"] = bool(enforce_eager)
    llm_kwargs.update(extra_kwargs)

    try:
        model = LLM(**llm_kwargs)
    except TypeError as exc:
        msg = str(exc)
        if "unexpected keyword argument" in msg:
            raise RuntimeError(
                "vLLM rejected one or more model kwargs. Check model config fields under `model:` "
                "(especially dtype/tensor_parallel_size/gpu_memory_utilization/vllm_kwargs)."
            ) from exc
        raise
    chat_tokenizer = None
    if load_chat_tokenizer:
        try:
            chat_tokenizer = load_hf_tokenizer(
                model_id,
                trust_remote_code=trust_remote_code,
                cache_dir=str(tokenizer_cache_dir) if tokenizer_cache_dir else None,
                revision=model_revision,
                loader=chat_formatter_loader,
                extra_kwargs=chat_formatter_kwargs,
            )
        except Exception as exc:
            logger.warning(
                "Could not load chat tokenizer for model_id=%s; falling back to ChatML wrapper. error=%s",
                model_id,
                exc,
            )
    return VLLMBackend(
        model=model,
        sampling_defaults=sampling_defaults,
        chat_tokenizer=chat_tokenizer,
        runtime_metadata=_backend_runtime_metadata(
            model=model,
            tokenizer=chat_tokenizer,
            cfg=cfg,
            model_id=model_id,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            sampling_defaults=sampling_defaults,
        ),
    )
