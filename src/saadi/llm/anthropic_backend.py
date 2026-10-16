"""Anthropic Claude backend built on the official ``anthropic`` SDK.

Each :class:`ChatPrompt` becomes one ``messages.create`` call with the system
instruction in ``system`` and the user text as the single user turn. Batches
are sent concurrently; the SDK retries rate limits and server errors.

Current Claude models reason before answering and reject sampling fields such
as ``temperature``, so the backend omits them unless asked. Reasoning tokens
count against ``max_tokens``, so the requested cap is raised to at least
``min_max_tokens`` to keep short extraction outputs from being cut off. Output
length is still bounded by the stop sequences.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Sequence

from ._concurrency import map_concurrently
from .base import BaseLLMBackend, ChatPrompt, package_version, strip_reasoning_markup, truncate_at_stop

logger = logging.getLogger(__name__)

DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"
DEFAULT_MIN_MAX_TOKENS = 8192
_FALLBACK_BETA_DEFAULT = "server-side-fallback-2026-07-01"
_FALLBACK_BETA_LIST = "server-side-fallback-2026-06-01"


class AnthropicBackend(BaseLLMBackend):
    """Claude through the Anthropic Messages API.

    Args:
        model: Claude model id, for example ``claude-opus-5-5``.
        api_key: Explicit key. When omitted the SDK resolves credentials itself
            from ``ANTHROPIC_API_KEY``, ``ANTHROPIC_AUTH_TOKEN``, or a profile
            created with ``ant auth login``.
        base_url: Alternative endpoint, mainly for tests and proxies.
        max_workers: Parallel requests per batch.
        max_retries: SDK retries for rate limits and server errors.
        timeout: Seconds per request.
        effort: ``low``, ``medium``, ``high``, ``xhigh``, or ``max``; omitted
            means the model default.
        thinking: Optional ``thinking`` request field, passed through as given.
        send_temperature: Put the pipeline temperature in the request body.
            Off by default because current models reject sampling fields;
            older models still accept it.
        min_max_tokens: Floor applied to ``max_tokens`` so reasoning does not
            consume the whole output budget. ``0`` disables the floor.
        fallbacks: Optional server-side refusal fallback. ``"default"`` lets
            the API route by refusal category; a list of ``{"model": ...}``
            entries names the fallback models. Uses the beta messages endpoint.
        request_kwargs: Extra fields for ``messages.create``.
        extra_body: Non-standard JSON fields.
        client: A pre-built ``anthropic.Anthropic`` client, mainly for tests.
    """

    def __init__(
        self,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        max_workers: int = 8,
        max_retries: int = 4,
        timeout: float = 600.0,
        effort: str | None = None,
        thinking: dict[str, Any] | None = None,
        send_temperature: bool = False,
        min_max_tokens: int = DEFAULT_MIN_MAX_TOKENS,
        fallbacks: Any = None,
        request_kwargs: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = str(model or DEFAULT_ANTHROPIC_MODEL)
        self.api_key = api_key
        self.base_url = str(base_url).strip() if base_url else None
        self.max_workers = int(max_workers)
        self.max_retries = int(max_retries)
        self.timeout = float(timeout)
        self.effort = str(effort).strip().lower() if effort else None
        self.thinking = dict(thinking) if thinking else None
        self.send_temperature = bool(send_temperature)
        self.min_max_tokens = max(0, int(min_max_tokens or 0))
        self.fallbacks = fallbacks
        self.request_kwargs = dict(request_kwargs or {})
        self.extra_body = dict(extra_body or {})
        self._client = client
        self._lock = threading.Lock()
        self._disabled: set[str] = set()
        self._floor_disabled = False
        self.runtime_metadata: dict[str, Any] = {
            "backend": "anthropic",
            "model_name": self.model,
            "base_url": self.base_url,
            "anthropic_sdk_version": package_version("anthropic"),
            "effort": self.effort,
            "max_workers": self.max_workers,
            "min_max_tokens": self.min_max_tokens,
            "sends_temperature": self.send_temperature,
            "sampling": {},
        }

    def client(self) -> Any:
        if self._client is None:
            try:
                import anthropic  # type: ignore
            except ImportError as exc:  # pragma: no cover - depends on the environment
                raise RuntimeError(
                    "The Anthropic backend needs the `anthropic` package. "
                    "Install it with `pip install 'saadi[anthropic]'`."
                ) from exc
            kwargs: dict[str, Any] = {"max_retries": self.max_retries, "timeout": self.timeout}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = anthropic.Anthropic(**kwargs)
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
        # The API rejects stop sequences that are empty or whitespace only.
        stops = [str(item) for item in (stop or []) if str(item).strip()]

        def one(prompt: ChatPrompt) -> str:
            return self._generate_one(prompt, max_tokens=int(max_tokens), temperature=float(temperature), stop=stops)

        return map_concurrently(one, prompts, max_workers=self.max_workers)

    def build_request(self, prompt: ChatPrompt, *, max_tokens: int, temperature: float, stop: list[str]) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": int(max_tokens) if self._floor_disabled else max(int(max_tokens), self.min_max_tokens),
            "messages": [{"role": "user", "content": prompt.user}],
        }
        if prompt.system:
            params["system"] = prompt.system
        if stop:
            params["stop_sequences"] = list(stop)
        if self.effort:
            params["output_config"] = {"effort": self.effort}
        if self.thinking:
            params["thinking"] = dict(self.thinking)
        params.update(self.request_kwargs)
        extra_body = dict(self.extra_body)
        if self.send_temperature and "temperature" not in self._disabled:
            # The SDK no longer exposes sampling fields; older models still accept them in the body.
            extra_body["temperature"] = float(temperature)
        if extra_body:
            params["extra_body"] = extra_body
        return params

    def _create(self, client: Any, params: dict[str, Any]) -> Any:
        if self.fallbacks is None:
            return client.messages.create(**params)
        beta = _FALLBACK_BETA_DEFAULT if isinstance(self.fallbacks, str) else _FALLBACK_BETA_LIST
        return client.beta.messages.create(betas=[beta], fallbacks=self.fallbacks, **params)

    def _generate_one(self, prompt: ChatPrompt, *, max_tokens: int, temperature: float, stop: list[str]) -> str:
        client = self.client()  # raises a clear install hint when the SDK is missing
        import anthropic  # type: ignore

        params = self.build_request(prompt, max_tokens=max_tokens, temperature=temperature, stop=stop)
        try:
            response = self._create(client, params)
        except anthropic.BadRequestError as exc:
            message = str(exc).lower()
            extra_body = dict(params.get("extra_body") or {})
            if "temperature" in extra_body and "temperature" in message:
                with self._lock:
                    self._disabled.add("temperature")
                logger.warning("Anthropic rejected temperature for model=%s; omitting it", self.model)
                extra_body.pop("temperature")
                if extra_body:
                    params["extra_body"] = extra_body
                else:
                    params.pop("extra_body", None)
                response = self._create(client, params)
            elif "max_tokens" in message and params["max_tokens"] > int(max_tokens):
                # The model's output cap is below the reasoning floor; use the requested cap.
                with self._lock:
                    self._floor_disabled = True
                logger.warning(
                    "Anthropic rejected max_tokens=%d for model=%s; using the requested %d from now on",
                    params["max_tokens"],
                    self.model,
                    int(max_tokens),
                )
                params["max_tokens"] = int(max_tokens)
                response = self._create(client, params)
            else:
                raise

        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            logger.warning(
                "Anthropic declined a request for model=%s (category=%s); returning empty output",
                self.model,
                getattr(details, "category", None),
            )
            return ""
        if stop_reason == "max_tokens":
            logger.warning("Anthropic output hit max_tokens=%d for model=%s", params["max_tokens"], self.model)
        text = "".join(
            str(getattr(block, "text", "") or "")
            for block in (getattr(response, "content", None) or [])
            if getattr(block, "type", None) == "text"
        )
        return truncate_at_stop(strip_reasoning_markup(text), stop)

