"""Language-model backends: a shared chat contract and the built-in providers."""

from .anthropic_backend import AnthropicBackend
from .base import (
    BaseLLMBackend,
    ChatPrompt,
    GenerateParams,
    LegacyStringBackend,
    LLMBackend,
    RenderedStringBackend,
    describe_llm_provenance,
    package_version,
    parse_chatml,
    render_prompt,
    strip_reasoning_markup,
    truncate_at_stop,
)
from .callable_backend import CallableBackend
from .factory import (
    SUPPORTED_BACKENDS,
    backend_from_spec,
    build_backend_from_config,
    model_config_from_spec,
    normalize_backend_name,
    resolve_backend,
)
from .mock_backend import MockBackend
from .openai_backend import OpenAICompatibleBackend
from .vllm_backend import VLLMBackend

__all__ = [
    "AnthropicBackend",
    "BaseLLMBackend",
    "CallableBackend",
    "ChatPrompt",
    "GenerateParams",
    "LLMBackend",
    "LegacyStringBackend",
    "MockBackend",
    "OpenAICompatibleBackend",
    "RenderedStringBackend",
    "SUPPORTED_BACKENDS",
    "VLLMBackend",
    "backend_from_spec",
    "build_backend_from_config",
    "describe_llm_provenance",
    "model_config_from_spec",
    "normalize_backend_name",
    "package_version",
    "parse_chatml",
    "render_prompt",
    "resolve_backend",
    "strip_reasoning_markup",
    "truncate_at_stop",
]
