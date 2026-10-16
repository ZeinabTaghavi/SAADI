"""Backend contract tests plus fake-server tests for the hosted-API backends.

No real API keys or network access are used. The OpenAI and Anthropic SDKs are
pointed at a local HTTP server that speaks just enough of each protocol.
"""

from __future__ import annotations

import json
import re
import logging
import time
from pathlib import Path
from typing import Any

import pytest

from saadi.config.validation import validate_experiment
from saadi.context_strategies.hierarchical_summary import HIER_SYSTEM_PROMPT, HierarchicalSummaryStrategy
from saadi.core.prompts import chatml_wrap
from saadi.core.state import init_discourse_state
from saadi.core.types import ChunkRecord
from saadi.llm import (
    CallableBackend,
    ChatPrompt,
    LegacyStringBackend,
    MockBackend,
    build_backend_from_config,
    model_config_from_spec,
    render_prompt,
    resolve_backend,
    truncate_at_stop,
)
from saadi.llm.anthropic_backend import AnthropicBackend
from saadi.llm.openai_backend import OpenAICompatibleBackend

from _fake_api_server import FakeAPIServer, Response, anthropic_payload, last_user_text, openai_payload, saadi_like_responder


@pytest.fixture
def server():
    fake = FakeAPIServer()
    try:
        yield fake
    finally:
        fake.close()


def openai_backend(server: FakeAPIServer, **overrides: Any) -> OpenAICompatibleBackend:
    pytest.importorskip("openai")
    options: dict[str, Any] = {
        "base_url": server.url + "/v1",
        "api_key": "test-key",
        "max_workers": 4,
        "max_retries": 2,
        "timeout": 10.0,
    }
    options.update(overrides)
    return OpenAICompatibleBackend("test-model", **options)


def anthropic_backend(server: FakeAPIServer, **overrides: Any) -> AnthropicBackend:
    pytest.importorskip("anthropic")
    options: dict[str, Any] = {
        "base_url": server.url,
        "api_key": "test-key",
        "max_workers": 4,
        "max_retries": 2,
        "timeout": 10.0,
    }
    options.update(overrides)
    return AnthropicBackend("claude-test", **options)


# --- contract -----------------------------------------------------------------


def test_chat_prompt_messages_and_text() -> None:
    prompt = ChatPrompt(system="sys", user="usr")
    assert prompt.messages() == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]
    assert ChatPrompt(user="only").messages() == [{"role": "user", "content": "only"}]
    assert prompt.as_text() == chatml_wrap("sys", "usr")
    assert truncate_at_stop("abc[END_JSON]def<|im_end|>", ["<|im_end|>", "[END_JSON]"]) == "abc"
    assert truncate_at_stop("abc", ["[END_JSON]"]) == "abc"


def test_mock_backend_renders_chat_prompts_like_before() -> None:
    backend = MockBackend()
    prompt = ChatPrompt(system="sys", user="[CURRENT_CHUNK]\nhello world")
    out = backend.generate_chat_batch([prompt], max_tokens=16, temperature=0.0, stop=["[END_JSON]"])
    assert json.loads(out[0].split("[END_JSON]", 1)[0]) == {"active_entities": [], "insertions": []}
    assert backend.calls == [chatml_wrap("sys", "[CURRENT_CHUNK]\nhello world")]
    assert render_prompt(backend, prompt) == prompt.as_text()


def test_strategy_builds_chat_prompts() -> None:
    strategy = HierarchicalSummaryStrategy()
    chunk = ChunkRecord(doc_id="d", chunk_id="d-0", chunk_index=0, raw_text="Mira Chen founded the Aurora team.")
    prompt = strategy.build_prompt(chunk, init_discourse_state())
    assert isinstance(prompt, ChatPrompt)
    assert prompt.system == HIER_SYSTEM_PROMPT
    assert "[CURRENT_CHUNK]\nMira Chen founded the Aurora team." in prompt.user
    rollup = strategy.build_rollup_prompt(items=["a", "b"], lower_cycle=1, next_cycle=5, step=4)
    assert isinstance(rollup, ChatPrompt)
    assert "[ROLLUP_ITEMS_JSON]" in rollup.user


def test_resolve_backend_accepts_functions_objects_and_legacy_backends() -> None:
    def with_options(system: str, user: str, *, max_tokens: int, temperature: float, stop: list[str]) -> str:
        return f"{system}|{user}|{max_tokens}|{temperature}|{stop}"

    out = resolve_backend(with_options).generate_chat_batch(
        [ChatPrompt(system="s", user="u")], max_tokens=3, temperature=0.5, stop=["x"]
    )
    assert out == ["s|u|3|0.5|['x']"]

    def plain(system: str, user: str) -> str:
        return user.upper()

    assert resolve_backend(plain).generate_chat_batch([ChatPrompt(user="hi")], max_tokens=1, temperature=0.0, stop=[]) == ["HI"]

    class Client:
        def generate(self, system: str, user: str, **options: Any) -> str:
            return f"client:{user}:{options['max_tokens']}"

    client_backend = resolve_backend(Client())
    assert isinstance(client_backend, CallableBackend)
    assert client_backend.generate_chat_batch([ChatPrompt(user="q")], max_tokens=7, temperature=0.0, stop=[]) == ["client:q:7"]

    class Legacy:
        def generate_batch(self, prompts: list[str], *, max_tokens: int, temperature: float, stop: list[str]) -> list[str]:
            return [f"legacy:{p}" for p in prompts]

    legacy = resolve_backend(Legacy())
    assert isinstance(legacy, LegacyStringBackend)
    assert legacy.generate_chat_batch([ChatPrompt(system="s", user="u")], max_tokens=1, temperature=0.0, stop=[]) == [
        "legacy:" + chatml_wrap("s", "u")
    ]

    mock = MockBackend()
    assert resolve_backend(mock) is mock
    assert isinstance(resolve_backend({"backend": "mock"}), MockBackend)
    assert isinstance(resolve_backend("mock"), MockBackend)
    with pytest.raises(TypeError):
        resolve_backend(42)


def test_model_config_from_spec() -> None:
    assert model_config_from_spec("mock") == {"backend": "mock"}
    assert model_config_from_spec("ollama:qwen3:8b") == {"backend": "ollama", "model_id": "qwen3:8b"}
    assert model_config_from_spec("OpenAI:gpt-4.1-mini") == {"backend": "openai", "model_id": "gpt-4.1-mini"}
    assert model_config_from_spec("anthropic") == {"backend": "anthropic", "model_id": "claude-opus-5-5"}
    assert model_config_from_spec("vllm:Qwen/Qwen3.5-27B") == {"backend": "vllm", "model_id": "Qwen/Qwen3.5-27B"}
    for bad in ("", "gpt-4.1-mini", "openai", "mock:x", "nope:model"):
        with pytest.raises(ValueError):
            model_config_from_spec(bad)


def test_build_backend_from_config_reads_api_options(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

    ollama = build_backend_from_config({"backend": "ollama", "model_id": "qwen3:8b"})
    assert isinstance(ollama, OpenAICompatibleBackend)
    assert ollama.effective_base_url == "http://localhost:11434/v1"
    request = ollama.build_request(ChatPrompt(user="x"), max_tokens=5, temperature=0.0, stop=["[END_JSON]"])
    assert request["max_tokens"] == 5
    assert request["stop"] == ["[END_JSON]"]

    hosted = build_backend_from_config(
        {"backend": "openai-compatible", "model_id": "gpt-4.1-mini", "max_workers": 2, "sampling": {"seed": 7}}
    )
    request = hosted.build_request(ChatPrompt(user="x"), max_tokens=5, temperature=0.0, stop=[])
    assert request["max_completion_tokens"] == 5
    assert request["seed"] == 7
    assert hosted.max_workers == 2
    assert hosted.runtime_metadata["model_name"] == "gpt-4.1-mini"

    claude = build_backend_from_config({"backend": "claude", "effort": "low"})
    assert isinstance(claude, AnthropicBackend)
    assert claude.model == "claude-opus-5-5"
    assert claude.effort == "low"

    with pytest.raises(ValueError, match="Unsupported model backend"):
        build_backend_from_config({"backend": "nope"})
    with pytest.raises(ValueError, match="model_id"):
        build_backend_from_config({"backend": "openai"})


def test_validate_experiment_accepts_api_backends() -> None:
    config = {
        "dataset": {"name": "x"},
        "ingest": {"enabled": True},
        "retrieval": {},
        "model": {
            "backend": "openai",
            "model_id": "gpt-4.1-mini",
            "base_url": "http://localhost:8000/v1",
            "max_workers": 4,
        },
    }
    validate_experiment(config)
    with pytest.raises(ValueError, match="model.backend"):
        validate_experiment({**config, "model": {"backend": "nope"}})


# --- OpenAI-compatible backend ------------------------------------------------


def test_openai_backend_sends_messages_and_keeps_batch_order(server: FakeAPIServer) -> None:
    def slower_for_earlier_prompts(path: str, body: dict[str, Any]) -> Response | None:
        index = int(last_user_text(body).rsplit("-", 1)[1])
        time.sleep(0.02 * (6 - index))
        return None

    server.responder = slower_for_earlier_prompts
    backend = openai_backend(server)
    prompts = [ChatPrompt(system="sys", user=f"prompt-{i}") for i in range(6)]
    outputs = backend.generate_chat_batch(prompts, max_tokens=64, temperature=0.0, stop=["[END_JSON]"])
    assert outputs == [f"echo:prompt-{i}" for i in range(6)]
    assert len(server.requests) == 6
    body = server.requests[0]["body"]
    assert body["model"] == "test-model"
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert body["messages"][1]["role"] == "user"
    assert body["max_tokens"] == 64
    assert body["temperature"] == 0.0
    assert body["stop"] == ["[END_JSON]"]
    assert server.requests[0]["headers"]["authorization"] == "Bearer test-key"


def test_openai_backend_switches_token_limit_param_when_rejected(server: FakeAPIServer) -> None:
    def reject_max_tokens(path: str, body: dict[str, Any]) -> Response | None:
        if "max_tokens" in body:
            return 400, {}, {
                "error": {
                    "message": "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.",
                    "type": "invalid_request_error",
                    "param": "max_tokens",
                    "code": "unsupported_parameter",
                }
            }
        return None

    server.responder = reject_max_tokens
    backend = openai_backend(server, max_workers=1)
    assert backend.generate_chat_batch([ChatPrompt(user="a")], max_tokens=32, temperature=0.0, stop=[]) == ["echo:a"]
    assert "max_tokens" in server.requests[0]["body"]
    assert server.requests[1]["body"]["max_completion_tokens"] == 32
    backend.generate_chat_batch([ChatPrompt(user="b")], max_tokens=32, temperature=0.0, stop=[])
    assert len(server.requests) == 3
    assert "max_tokens" not in server.requests[2]["body"]


def test_openai_backend_drops_temperature_when_rejected(server: FakeAPIServer) -> None:
    def reject_temperature(path: str, body: dict[str, Any]) -> Response | None:
        if "temperature" in body:
            return 400, {}, {
                "error": {
                    "message": "Unsupported value: 'temperature' does not support 0.0 with this model.",
                    "type": "invalid_request_error",
                }
            }
        return None

    server.responder = reject_temperature
    backend = openai_backend(server, max_workers=1)
    assert backend.generate_chat_batch([ChatPrompt(user="a")], max_tokens=8, temperature=0.0, stop=[]) == ["echo:a"]
    assert "temperature" in server.requests[0]["body"]
    assert "temperature" not in server.requests[1]["body"]


def test_openai_backend_truncates_at_stop_sequence(server: FakeAPIServer) -> None:
    server.responder = lambda path, body: (200, {}, openai_payload(body, '{"a": 1}[END_JSON] trailing text'))
    backend = openai_backend(server)
    assert backend.generate_chat_batch([ChatPrompt(user="a")], max_tokens=8, temperature=0.0, stop=["[END_JSON]"]) == ['{"a": 1}']


def test_openai_backend_retries_rate_limits(server: FakeAPIServer) -> None:
    calls = {"count": 0}

    def rate_limit_once(path: str, body: dict[str, Any]) -> Response | None:
        calls["count"] += 1
        if calls["count"] == 1:
            return 429, {"retry-after-ms": "1"}, {"error": {"message": "slow down", "type": "rate_limit_error"}}
        return None

    server.responder = rate_limit_once
    backend = openai_backend(server)
    assert backend.generate_chat_batch([ChatPrompt(user="a")], max_tokens=8, temperature=0.0, stop=[]) == ["echo:a"]
    assert len(server.requests) == 2


def test_local_servers_need_no_api_key(server: FakeAPIServer, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    backend = build_backend_from_config(
        {"backend": "ollama", "model_id": "qwen3:8b", "base_url": server.url + "/v1", "timeout": 10}
    )
    assert backend.generate_chat_batch([ChatPrompt(user="hi")], max_tokens=8, temperature=0.0, stop=[]) == ["echo:hi"]
    assert server.requests[0]["headers"]["authorization"] == "Bearer not-needed"


def test_remote_hosts_require_an_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    backend = OpenAICompatibleBackend("gpt-4.1-mini", base_url="https://api.example.com/v1")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        backend.generate_chat_batch([ChatPrompt(user="hi")], max_tokens=8, temperature=0.0, stop=[])


# --- Anthropic backend --------------------------------------------------------


def test_anthropic_backend_request_shape(server: FakeAPIServer) -> None:
    backend = anthropic_backend(server)
    prompts = [ChatPrompt(system="sys", user=f"prompt-{i}") for i in range(3)]
    outputs = backend.generate_chat_batch(prompts, max_tokens=256, temperature=0.0, stop=["[END_JSON]"])
    assert outputs == [f"echo:prompt-{i}" for i in range(3)]
    body = server.requests[0]["body"]
    assert body["model"] == "claude-test"
    assert body["system"] == "sys"
    assert body["messages"] == [{"role": "user", "content": last_user_text(body)}]
    assert body["stop_sequences"] == ["[END_JSON]"]
    assert body["max_tokens"] == 8192
    assert "temperature" not in body
    assert "output_config" not in body
    assert server.requests[0]["headers"]["x-api-key"] == "test-key"


def test_anthropic_backend_options(server: FakeAPIServer) -> None:
    backend = anthropic_backend(server, effort="low", send_temperature=True, min_max_tokens=0)
    backend.generate_chat_batch([ChatPrompt(user="a")], max_tokens=256, temperature=0.5, stop=["[END_JSON]", "   "])
    body = server.requests[0]["body"]
    assert body["output_config"] == {"effort": "low"}
    assert body["temperature"] == 0.5
    assert body["max_tokens"] == 256
    assert body["stop_sequences"] == ["[END_JSON]"]


def test_anthropic_backend_reads_text_blocks_only_and_truncates(server: FakeAPIServer) -> None:
    server.responder = lambda path, body: (
        200,
        {},
        anthropic_payload(
            body,
            "",
            content=[
                {"type": "thinking", "thinking": "", "signature": "sig"},
                {"type": "text", "text": '{"a": 1}[END_JSON] junk'},
            ],
        ),
    )
    backend = anthropic_backend(server)
    assert backend.generate_chat_batch([ChatPrompt(user="a")], max_tokens=8, temperature=0.0, stop=["[END_JSON]"]) == ['{"a": 1}']


def test_anthropic_backend_refusal_returns_empty_output(server: FakeAPIServer, caplog: pytest.LogCaptureFixture) -> None:
    server.responder = lambda path, body: (
        200,
        {},
        anthropic_payload(
            body,
            "",
            stop_reason="refusal",
            content=[],
            stop_details={"type": "refusal", "category": "cyber", "explanation": "declined"},
        ),
    )
    backend = anthropic_backend(server)
    with caplog.at_level(logging.WARNING, logger="saadi.llm.anthropic_backend"):
        assert backend.generate_chat_batch([ChatPrompt(user="a")], max_tokens=8, temperature=0.0, stop=[]) == [""]
    assert "declined" in caplog.text


def test_anthropic_backend_drops_temperature_when_rejected(server: FakeAPIServer) -> None:
    def reject_temperature(path: str, body: dict[str, Any]) -> Response | None:
        if "temperature" in body:
            return 400, {}, {
                "type": "error",
                "error": {"type": "invalid_request_error", "message": "temperature: Extra inputs are not permitted"},
            }
        return None

    server.responder = reject_temperature
    backend = anthropic_backend(server, send_temperature=True, max_workers=1)
    assert backend.generate_chat_batch([ChatPrompt(user="a")], max_tokens=8, temperature=0.0, stop=[]) == ["echo:a"]
    assert "temperature" in server.requests[0]["body"]
    assert "temperature" not in server.requests[1]["body"]


# --- vLLM rendering -----------------------------------------------------------


def test_vllm_backend_renders_prompts_with_the_chat_template(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    from saadi.core.prompts import format_chat_prompt
    from saadi.llm import VLLMBackend

    class SamplingParams:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(SamplingParams=SamplingParams))

    class Tokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
            return "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages) + "<assistant>"

    class Model:
        def __init__(self) -> None:
            self.prompts: list[str] | None = None

        def generate(self, prompts, params):
            self.prompts = list(prompts)
            return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text=f"out:{i}")]) for i, _ in enumerate(prompts)]

    model, tokenizer = Model(), Tokenizer()
    backend = VLLMBackend(model=model, chat_tokenizer=tokenizer)
    prompt = ChatPrompt(system="sys", user="usr")
    assert backend.generate_chat_batch([prompt], max_tokens=4, temperature=0.0, stop=["[END_JSON]"]) == ["out:0"]
    # The backend renders exactly what the strategies used to render themselves.
    expected = format_chat_prompt(tokenizer=tokenizer, system_prompt="sys", user_prompt="usr")
    assert model.prompts == [expected]
    assert render_prompt(backend, prompt) == expected == "<system>sys</system><user>usr</user><assistant>"


# --- legacy raw-string path ---------------------------------------------------


def test_legacy_string_path_recovers_roles_for_hosted_backends(server: FakeAPIServer) -> None:
    from saadi.llm import parse_chatml

    assert parse_chatml(chatml_wrap("sys", "usr")) == ChatPrompt(system="sys", user="usr")
    assert parse_chatml("plain text") == ChatPrompt(user="plain text")

    backend = openai_backend(server, max_workers=1)
    outputs = backend.generate_batch([chatml_wrap("sys", "usr"), "plain"], max_tokens=8, temperature=0.0, stop=[])
    assert outputs == ["echo:usr", "echo:plain"]
    assert server.requests[0]["body"]["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]
    assert server.requests[1]["body"]["messages"] == [{"role": "user", "content": "plain"}]


# --- whole pipeline through a hosted backend ----------------------------------


def _run_demo_through(server: FakeAPIServer, tmp_path, model_config: dict[str, Any]):
    from single_document_demo.pipeline import load_demo_config, run_demo

    server.responder = saadi_like_responder
    config = load_demo_config("single_document_demo/configs/text_document.toml")
    config.demo_root = tmp_path / "demo"
    config.output_root = config.demo_root / "output"
    config.prompt_profile = str(ROOT / "single_document_demo" / "memory_prompts" / "default")
    config.run_name = "hosted"
    config.query = "Who leads the Aurora team?"
    config.model_config = model_config
    config.saadi_options.update({"use_cache": False, "chunk_size": 20})
    result = run_demo(config)
    assert not result.failures, [record.error for record in result.failures]
    run_dir = Path(result.records[0].run_dir)
    answer = (run_dir / "generation" / "answer.txt").read_text(encoding="utf-8")
    assert "Mira Chen" in answer
    kinds = {"chunk": 0, "rollup": 0, "answer": 0}
    for request in server.requests:
        user = last_user_text(request["body"])
        if "[ROLLUP_ITEMS_JSON]" in user:
            kinds["rollup"] += 1
        elif "[CURRENT_CHUNK]" in user:
            kinds["chunk"] += 1
        else:
            kinds["answer"] += 1
    assert kinds["chunk"] >= 5 and kinds["rollup"] >= 1 and kinds["answer"] == 1, kinds
    memory_text = (run_dir / "cleaned_outputs" / "manifest.json").read_text(encoding="utf-8")
    assert memory_text
    return run_dir


ROOT = Path(__file__).resolve().parents[1]


def test_single_document_pipeline_runs_through_openai_compatible_backend(server: FakeAPIServer, tmp_path) -> None:
    pytest.importorskip("openai")
    run_dir = _run_demo_through(
        server,
        tmp_path,
        {"backend": "openai", "model_id": "fake-model", "base_url": server.url + "/v1", "api_key": "test-key", "timeout": 10},
    )
    assert "<|system|>" in (run_dir / "generation" / "prompt.txt").read_text(encoding="utf-8")
    for request in server.requests:
        assert request["path"] == "/v1/chat/completions"
        assert request["body"]["messages"][0]["role"] == "system"


def test_single_document_pipeline_runs_through_anthropic_backend(server: FakeAPIServer, tmp_path) -> None:
    pytest.importorskip("anthropic")
    _run_demo_through(
        server,
        tmp_path,
        {"backend": "anthropic", "model_id": "claude-test", "base_url": server.url, "api_key": "test-key", "timeout": 10},
    )
    for request in server.requests:
        assert request["path"] == "/v1/messages"
        assert request["body"]["system"]
        assert "temperature" not in request["body"]


def test_hosted_backends_strip_closed_think_blocks(server: FakeAPIServer) -> None:
    server.responder = lambda path, body: (200, {}, openai_payload(body, "<think>reasoning</think>\n{\"a\": 1}[END_JSON]"))
    backend = openai_backend(server)
    assert backend.generate_chat_batch([ChatPrompt(user="a")], max_tokens=8, temperature=0.0, stop=["[END_JSON]"]) == ['{"a": 1}']


def test_openai_backend_keeps_plain_stops_before_dropping_stop(server: FakeAPIServer) -> None:
    def reject_special_tokens(path: str, body: dict[str, Any]) -> Response | None:
        if any(item.startswith("<|") for item in body.get("stop", [])):
            return 400, {}, {"error": {"message": "Invalid 'stop': special tokens are not allowed.", "type": "invalid_request_error", "param": "stop"}}
        return None

    server.responder = reject_special_tokens
    backend = openai_backend(server, max_workers=1)
    stops = ["[END_JSON]", "<|im_end|>", "<|endoftext|>"]
    assert backend.generate_chat_batch([ChatPrompt(user="a")], max_tokens=8, temperature=0.0, stop=stops) == ["echo:a"]
    assert server.requests[0]["body"]["stop"] == stops
    assert server.requests[1]["body"]["stop"] == ["[END_JSON]"]


def test_openai_backend_drops_stop_entirely_when_still_rejected(server: FakeAPIServer) -> None:
    def reject_any_stop(path: str, body: dict[str, Any]) -> Response | None:
        if "stop" in body:
            return 400, {}, {"error": {"message": "Unsupported parameter: 'stop' is not supported with this model.", "type": "invalid_request_error", "param": "stop"}}
        return 200, {}, openai_payload(body, "answer[END_JSON] more")

    server.responder = reject_any_stop
    backend = openai_backend(server, max_workers=1)
    stops = ["[END_JSON]", "<|im_end|>"]
    assert backend.generate_chat_batch([ChatPrompt(user="a")], max_tokens=8, temperature=0.0, stop=stops) == ["answer"]
    assert [("stop" in r["body"]) for r in server.requests] == [True, True, False]
    assert server.requests[1]["body"]["stop"] == ["[END_JSON]"]


# --- review fixes ---------------------------------------------------------------


def test_concurrent_requests_all_recover_when_a_param_is_rejected(server: FakeAPIServer) -> None:
    """Siblings in flight must retry with the adapted request instead of raising."""

    def reject_temperature_slowly(path: str, body: dict[str, Any]) -> Response | None:
        if "temperature" in body:
            time.sleep(0.05)  # keep every first-wave request in flight before any adaptation
            return 400, {}, {
                "error": {"message": "Unsupported parameter: 'temperature'", "type": "invalid_request_error", "param": "temperature"}
            }
        return None

    server.responder = reject_temperature_slowly
    backend = openai_backend(server, max_workers=4)
    prompts = [ChatPrompt(user=f"p{i}") for i in range(6)]
    outputs = backend.generate_chat_batch(prompts, max_tokens=8, temperature=0.0, stop=[])
    assert outputs == [f"echo:p{i}" for i in range(6)]
    rejected = [r for r in server.requests if "temperature" in r["body"]]
    assert 4 <= len(rejected) <= 6
    assert "temperature" not in server.requests[-1]["body"]


def test_unrelated_400_errors_are_not_misread_as_parameter_rejections(server: FakeAPIServer) -> None:
    openai = pytest.importorskip("openai")
    server.responder = lambda path, body: (
        400,
        {},
        {"error": {"message": "This model's maximum context length is 4096 tokens; max_tokens=2000 plus the prompt exceed it.", "type": "invalid_request_error"}},
    )
    backend = openai_backend(server, max_workers=1)
    with pytest.raises(openai.BadRequestError):
        backend.generate_chat_batch([ChatPrompt(user="a")], max_tokens=2000, temperature=0.0, stop=[])
    assert len(server.requests) == 1  # no rename, no retry; the real error surfaces


def test_build_backend_from_config_requires_backend_for_non_empty_configs() -> None:
    with pytest.raises(ValueError, match="model.backend is missing"):
        build_backend_from_config({"model_id": "gpt-4.1-mini"})
    with pytest.raises(ValueError, match="model.backend is missing"):
        resolve_backend({"model_id": "gpt-4.1-mini", "base_url": "http://localhost:8000/v1"})
    assert isinstance(build_backend_from_config({}), MockBackend)
    assert isinstance(build_backend_from_config(None), MockBackend)


def test_anthropic_backend_drops_the_token_floor_when_the_model_caps_lower(server: FakeAPIServer) -> None:
    def reject_large_max_tokens(path: str, body: dict[str, Any]) -> Response | None:
        if body.get("max_tokens", 0) > 4096:
            return 400, {}, {
                "type": "error",
                "error": {"type": "invalid_request_error", "message": "max_tokens: 8192 > 4096, which is the maximum allowed number of output tokens for claude-test"},
            }
        return None

    server.responder = reject_large_max_tokens
    backend = anthropic_backend(server, max_workers=1)
    assert backend.generate_chat_batch([ChatPrompt(user="a")], max_tokens=256, temperature=0.0, stop=[]) == ["echo:a"]
    assert [r["body"]["max_tokens"] for r in server.requests] == [8192, 256]
    backend.generate_chat_batch([ChatPrompt(user="b")], max_tokens=256, temperature=0.0, stop=[])
    assert server.requests[-1]["body"]["max_tokens"] == 256 and len(server.requests) == 3


def test_configured_api_key_env_must_be_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SAADI_TEST_MISSING_KEY", raising=False)
    with pytest.raises(RuntimeError, match="SAADI_TEST_MISSING_KEY"):
        build_backend_from_config({"backend": "anthropic", "api_key_env": "SAADI_TEST_MISSING_KEY"})


def test_strip_reasoning_markup_handles_both_tag_styles_and_truncation() -> None:
    from saadi.llm import strip_reasoning_markup

    assert strip_reasoning_markup("<think>plan</think>\n{\"a\": 1}") == '{"a": 1}'
    assert strip_reasoning_markup("x <|begin_of_thought|>y<|end_of_thought|> z").split() == ["x", "z"]
    assert strip_reasoning_markup("<think>still reasoning when cut off") == ""
    assert strip_reasoning_markup("answer <think>partial") == "answer"


def test_map_concurrently_cancels_pending_items_after_a_failure() -> None:
    from saadi.llm._concurrency import map_concurrently

    started: list[int] = []

    def work(item: int) -> int:
        started.append(item)
        if item == 0:
            raise RuntimeError("boom")
        time.sleep(0.2)
        return item

    with pytest.raises(RuntimeError, match="boom"):
        map_concurrently(work, [0, 1, 2, 3, 4, 5], max_workers=2)
    assert len(started) <= 3  # the first wave only; later items were cancelled


def test_legacy_backends_are_accepted_at_pipeline_entry_points(tmp_path) -> None:
    from saadi.pipelines.single_document_pipeline import SingleDocumentPipelineConfig, generate_answer
    from saadi.saadi_generation.generation.generator import GenerationConfig, GenerationRequest, SaadiAnswerGenerator

    class Legacy:
        def generate_batch(self, prompts: list[str], *, max_tokens: int, temperature: float, stop: list[str]) -> list[str]:
            return ["legacy answer" for _ in prompts]

    generator = SaadiAnswerGenerator(GenerationConfig(), backend=Legacy())
    rows = generator.generate_batch([GenerationRequest(example_id="q1", question="Who?", qa_entry={}, context="ctx", metadata={})])
    assert rows[0]["generated_answer"] == "legacy answer"

    config = SingleDocumentPipelineConfig(demo_root=str(tmp_path), document_id="doc", run_name="legacy", query="Who leads?")
    result = generate_answer(config, "Mira leads the team.", backend=Legacy(), out_dir=tmp_path / "out")
    assert result.generated_answer == "legacy answer"
    assert result.generation_record["status"] == "ok"


def test_empty_model_output_is_not_recorded_as_ok(tmp_path) -> None:
    from saadi.pipelines.single_document_pipeline import SingleDocumentPipelineConfig, generate_answer

    config = SingleDocumentPipelineConfig(demo_root=str(tmp_path), document_id="doc", run_name="empty", query="Who leads?")
    result = generate_answer(config, "context", backend=CallableBackend(lambda system, user: ""), out_dir=tmp_path / "out")
    assert result.generated_answer == ""
    assert result.generation_record["status"] == "empty_output"


def test_run_manifest_describes_non_vllm_backends_truthfully(tmp_path) -> None:
    from single_document_demo.pipeline import load_demo_config, run_demo

    config = load_demo_config("single_document_demo/configs/text_document.toml")
    config.demo_root = tmp_path / "demo"
    config.output_root = config.demo_root / "output"
    config.prompt_profile = str(ROOT / "single_document_demo" / "memory_prompts" / "default")
    config.run_name = "provenance"
    config.model_config = {"backend": "mock"}
    config.saadi_options.update({"use_cache": False, "chunk_size": 50})
    result = run_demo(config)
    assert not result.failures
    manifest = json.loads((Path(result.records[0].run_dir) / "run_manifest.json").read_text(encoding="utf-8"))
    reproducibility = manifest["reproducibility"]
    assert reproducibility["llm_backend"]["backend"] == "mock"
    assert set(reproducibility["sampling"]) == {"temperature"}  # no vLLM-only defaults for a backend that never applied them
