"""OpenAI-compatible chat backend.

One class covers OpenAI itself and every server that speaks its chat
completions protocol: Ollama, LM Studio, llama.cpp, Groq, Together, Mistral,
and ``vllm serve``. The prompts of a batch are sent concurrently. The SDK
retries rate limits, timeouts, and server errors with exponential backoff and
honours ``retry-after`` headers.

Servers differ in which request fields they accept. Reasoning models reject
``temperature`` and ``stop``, and newer OpenAI models want
``max_completion_tokens`` instead of ``max_tokens``. When a request is rejected
for one of those fields the backend drops or renames it, retries, and keeps the
adjustment for the rest of the session.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from typing import Any, Sequence
from urllib.parse import urlparse

from ._concurrency import map_concurrently
from .base import BaseLLMBackend, ChatPrompt, package_version, strip_reasoning_markup, truncate_at_stop

logger = logging.getLogger(__name__)

OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434/v1"
MAX_STOP_SEQUENCES = 4
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
_TOKEN_LIMIT_PARAMS = ("max_tokens", "max_completion_tokens")
_OPTIONAL_PARAMS = ("temperature", "stop", "seed")
_SPECIAL_TOKEN = re.compile(r"^<\|.*\|>$")


def is_local_url(url: str | None) -> bool:
    if not url:
        return False
    host = (urlparse(str(url)).hostname or "").lower()
    return host in _LOCAL_HOSTS or host.endswith(".localhost") or host.endswith(".local")


def resolve_api_key(
    explicit: str | None,
    env_name: str | None,
    *,
    base_url: str | None,
    provider_label: str,
) -> str:
    """Pick the API key: explicit value, then environment, then a placeholder for local servers."""

    if explicit and str(explicit).strip():
        return str(explicit).strip()
    if env_name:
        from_env = os.environ.get(env_name, "").strip()
        if from_env:
            return from_env
    if is_local_url(base_url):
        return "not-needed"
    hint = f"set {env_name} or pass api_key" if env_name else "pass api_key"
    raise RuntimeError(f"The {provider_label} needs an API key: {hint}.")


_PARAMETER_REJECTION_PHRASES = (
    "unsupported parameter",
    "unsupported value",
    "not supported",
    "does not support",
    "unknown parameter",
    "unrecognized",
    "unexpected keyword",
    "extra inputs",
    "additional properties",
    "not allowed",
    "invalid parameter",
    "use 'max_completion_tokens'",
)


def _rejected_param(exc: Exception, params: dict[str, Any]) -> str | None:
    """Name the request field a 400 response complained about, if we can tell.

    A structured ``param`` in the error body is trusted. Otherwise the message
    is consulted only when it reads like a parameter rejection, so a context
    length error that merely mentions ``max_tokens`` is left to propagate.
    """

    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict):
            param = error.get("param")
            if isinstance(param, str) and param in params:
                return param
    message = str(getattr(exc, "message", None) or exc).lower()
    if not any(phrase in message for phrase in _PARAMETER_REJECTION_PHRASES):
        return None
    for candidate in (*_TOKEN_LIMIT_PARAMS, *_OPTIONAL_PARAMS):
        if candidate in params and re.search(rf"\b{re.escape(candidate)}\b", message):
            return candidate
    return None


class OpenAICompatibleBackend(BaseLLMBackend):
    """Chat completions over the OpenAI protocol.

    Args:
        model: Model name as the server knows it, for example ``gpt-4.1-mini``
            or ``qwen3:8b``.
        base_url: Server root such as ``http://localhost:11434/v1``. Defaults to
            ``OPENAI_BASE_URL`` from the environment, then to OpenAI.
        api_key: Explicit key. Otherwise ``api_key_env`` is read; local servers
            need none.
        max_workers: Parallel requests per batch.
        max_retries: SDK retries for rate limits and server errors.
        timeout: Seconds per request.
        seed: Sent as ``seed`` when the server supports it.
        token_limit_param: ``auto``, ``max_tokens``, or ``max_completion_tokens``.
            ``auto`` picks ``max_completion_tokens`` for OpenAI and ``max_tokens``
            for other servers.
        request_kwargs: Extra fields for ``chat.completions.create``, for example
            ``{"reasoning_effort": "low"}``.
        extra_body: Non-standard JSON fields some servers accept.
        client: A pre-built ``openai.OpenAI`` client, mainly for tests.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = "OPENAI_API_KEY",
        max_workers: int = 8,
        max_retries: int = 4,
        timeout: float = 300.0,
        seed: int | None = None,
        token_limit_param: str = "auto",
        send_temperature: bool = True,
        send_stop: bool = True,
        request_kwargs: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
        provider_label: str = "openai",
        client: Any | None = None,
    ) -> None:
        self.model = str(model)
        self.base_url = str(base_url or os.environ.get("OPENAI_BASE_URL") or "").strip() or None
        self.api_key = api_key
        self.api_key_env = api_key_env
        self.max_workers = int(max_workers)
        self.max_retries = int(max_retries)
        self.timeout = float(timeout)
        self.seed = seed
        self.send_temperature = bool(send_temperature)
        self.send_stop = bool(send_stop)
        self.request_kwargs = dict(request_kwargs or {})
        self.extra_body = dict(extra_body or {})
        self.provider_label = str(provider_label)
        self._client = client
        self._lock = threading.Lock()
        self._disabled: set[str] = set()
        self._token_param_switched = False
        self._plain_stops_only = False

        normalized = str(token_limit_param or "auto").strip().lower()
        if normalized not in {"auto", *_TOKEN_LIMIT_PARAMS}:
            raise ValueError("token_limit_param must be auto, max_tokens, or max_completion_tokens")
        if normalized == "auto":
            normalized = "max_completion_tokens" if self.effective_base_url == OPENAI_DEFAULT_BASE_URL else "max_tokens"
        self._token_limit_param = normalized

        self.runtime_metadata: dict[str, Any] = {
            "backend": self.provider_label,
            "model_name": self.model,
            "base_url": self.effective_base_url,
            "openai_sdk_version": package_version("openai"),
            "max_workers": self.max_workers,
            "token_limit_param": self._token_limit_param,
            "sends_temperature": self.send_temperature,
            "sends_stop": self.send_stop,
            "sampling": {"seed": int(seed)} if seed is not None else {},
        }

    @property
    def effective_base_url(self) -> str:
        return (self.base_url or OPENAI_DEFAULT_BASE_URL).rstrip("/")

    def client(self) -> Any:
        if self._client is None:
            try:
                import openai  # type: ignore
            except ImportError as exc:  # pragma: no cover - depends on the environment
                raise RuntimeError(
                    "The OpenAI-compatible backend needs the `openai` package. "
                    "Install it with `pip install 'saadi[openai]'`."
                ) from exc
            self._client = openai.OpenAI(
                api_key=resolve_api_key(
                    self.api_key,
                    self.api_key_env,
                    base_url=self.effective_base_url,
                    provider_label=f"{self.provider_label} backend",
                ),
                base_url=self.effective_base_url,
                max_retries=self.max_retries,
                timeout=self.timeout,
            )
        return self._client

    def close(self) -> None:
        client = self._client
        self._client = None
        if client is not None and hasattr(client, "close"):
            client.close()

    def generate_chat_batch(
        self,
        prompts: Sequence[ChatPrompt],
        *,
        max_tokens: int,
        temperature: float,
        stop: list[str],
    ) -> list[str]:
        stops = [str(item) for item in (stop or []) if str(item)]

        def one(prompt: ChatPrompt) -> str:
            return self._generate_one(prompt, max_tokens=int(max_tokens), temperature=float(temperature), stop=stops)

        return map_concurrently(one, prompts, max_workers=self.max_workers)

    def build_request(self, prompt: ChatPrompt, *, max_tokens: int, temperature: float, stop: list[str]) -> dict[str, Any]:
        params: dict[str, Any] = {"model": self.model, "messages": prompt.messages()}
        params[self._token_limit_param] = int(max_tokens)
        if self.send_temperature and "temperature" not in self._disabled:
            params["temperature"] = float(temperature)
        if self.send_stop and stop and "stop" not in self._disabled:
            stops = [item for item in stop if not (self._plain_stops_only and _SPECIAL_TOKEN.match(item))]
            if stops:
                params["stop"] = stops[:MAX_STOP_SEQUENCES]
        if self.seed is not None and "seed" not in self._disabled:
            params["seed"] = int(self.seed)
        params.update(self.request_kwargs)
        if self.extra_body:
            params["extra_body"] = dict(self.extra_body)
        return params

    def _adapt_to_rejection(self, exc: Exception, params: dict[str, Any]) -> bool:
        """Drop or rename the field a server rejected. Returns True when a retry makes sense.

        Several requests of one batch can fail together. Whichever thread gets
        here first changes the shared settings; the others find the rejected
        field already gone from the current settings and simply retry, so a
        rejection only becomes an error when no further adjustment exists.
        """

        param = _rejected_param(exc, params)
        if param is None:
            return False
        with self._lock:
            if param in _TOKEN_LIMIT_PARAMS:
                if param != self._token_limit_param:
                    return True  # another request already switched the field name
                if self._token_param_switched:
                    return False  # both names were rejected
                other = "max_completion_tokens" if param == "max_tokens" else "max_tokens"
                self._token_limit_param = other
                self._token_param_switched = True
                logger.warning("%s rejected %s; using %s for model=%s", self.provider_label, param, other, self.model)
                return True
            if param in self._disabled:
                return True  # another request already dropped it
            if param == "stop":
                has_specials = any(_SPECIAL_TOKEN.match(item) for item in params.get("stop", []))
                if has_specials and not self._plain_stops_only:
                    # Tokenizer special tokens such as <|im_end|> mean nothing to other
                    # providers; keep the plain markers before giving up on stop entirely.
                    self._plain_stops_only = True
                    logger.warning(
                        "%s rejected special-token stop sequences; keeping plain ones for model=%s",
                        self.provider_label,
                        self.model,
                    )
                    return True
                if has_specials:
                    return True  # this request predates the switch to plain stops; retry
            self._disabled.add(param)
            logger.warning("%s rejected %s; omitting it for model=%s", self.provider_label, param, self.model)
            return True

    def _generate_one(self, prompt: ChatPrompt, *, max_tokens: int, temperature: float, stop: list[str]) -> str:
        client = self.client()  # raises a clear install hint when the SDK is missing
        import openai  # type: ignore

        attempts = len(_TOKEN_LIMIT_PARAMS) + len(_OPTIONAL_PARAMS) + 2
        response = None
        for _ in range(attempts):
            params = self.build_request(prompt, max_tokens=max_tokens, temperature=temperature, stop=stop)
            try:
                response = client.chat.completions.create(**params)
                break
            except openai.BadRequestError as exc:
                if self._adapt_to_rejection(exc, params):
                    continue
                raise
        if response is None:  # pragma: no cover - every adaptation path either breaks or raises
            raise RuntimeError("Request kept being rejected after adjusting its parameters")
        choices = getattr(response, "choices", None) or []
        text = ""
        if choices:
            message = getattr(choices[0], "message", None)
            text = str(getattr(message, "content", None) or "")
            if getattr(choices[0], "finish_reason", None) == "length":
                logger.warning("%s output hit the token limit (%d) for model=%s", self.provider_label, max_tokens, self.model)
        return truncate_at_stop(strip_reasoning_markup(text), stop)

