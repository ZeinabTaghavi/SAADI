"""A local HTTP server that speaks enough OpenAI and Anthropic protocol for tests.

Shared by the backend and retriever tests. No network access, no API keys.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

Response = tuple[int, dict[str, str], dict[str, Any]]
Responder = Callable[[str, dict[str, Any]], "Response | None"]


def last_user_text(body: dict[str, Any]) -> str:
    for message in reversed(body.get("messages", [])):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
            return "".join(part.get("text", "") for part in content or [])
    return ""


def openai_payload(body: dict[str, Any], text: str, *, finish_reason: str = "stop") -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": body.get("model", "test-model"),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def anthropic_payload(
    body: dict[str, Any],
    text: str,
    *,
    stop_reason: str = "end_turn",
    content: list[dict[str, Any]] | None = None,
    stop_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": body.get("model", "claude-test"),
        "content": content if content is not None else [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    if stop_details is not None:
        payload["stop_details"] = stop_details
    return payload


class FakeAPIServer:
    """Answers OpenAI chat-completion and Anthropic messages requests locally."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.responder: Responder | None = None
        self._lock = threading.Lock()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - http.server naming
                length = int(self.headers.get("content-length", "0") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                with server._lock:
                    server.requests.append(
                        {
                            "path": self.path,
                            "body": body,
                            "headers": {key.lower(): value for key, value in self.headers.items()},
                        }
                    )
                status, headers, payload = server.respond(self.path, body)
                data = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                for key, value in headers.items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args: Any) -> None:  # silence the test output
                return None

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def respond(self, path: str, body: dict[str, Any]) -> Response:
        if self.responder is not None:
            scripted = self.responder(path, body)
            if scripted is not None:
                return scripted
        text = "echo:" + last_user_text(body)
        if path.endswith("/chat/completions"):
            return 200, {}, openai_payload(body, text)
        if path.endswith("/messages"):
            return 200, {}, anthropic_payload(body, text)
        if path.endswith("/embeddings"):
            return 200, {}, embeddings_payload(body)
        return 404, {}, {"error": {"message": f"unexpected path {path}"}}

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


_EMBEDDING_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"


def fake_embedding(text: str) -> list[float]:
    """Deterministic vector from character counts, so similar texts score higher."""

    lowered = str(text or "").lower()
    return [1.0] + [float(lowered.count(ch)) for ch in _EMBEDDING_ALPHABET]


def embeddings_payload(body: dict[str, Any]) -> dict[str, Any]:
    inputs = body.get("input")
    if isinstance(inputs, str):
        inputs = [inputs]
    return {
        "object": "list",
        "data": [{"object": "embedding", "index": index, "embedding": fake_embedding(text)} for index, text in enumerate(inputs or [])],
        "model": body.get("model", "fake-embedding"),
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
    }


# Sentence-initial words that would otherwise look like the first half of a name.
_FUNCTION_WORDS = frozenset(
    "A An And At After Before But By Each Every For From He His Her In It Its On One Over She So That The Then They This To Under When With".split()
)


def saadi_like_reply(user: str) -> str:
    """Answer SAADI's three prompt kinds the way a cooperative model would."""

    if "[ROLLUP_ITEMS_JSON]" in user:
        text = "The Aurora team measures alpine lake water quality.\n[END_SUMMARY]"
    elif "[CURRENT_CHUNK]" in user:
        chunk = user.split("[CURRENT_CHUNK]\n", 1)[1].split("\n[", 1)[0]
        names: list[str] = []
        for match in re.finditer(r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b", chunk):
            name = match.group(1)
            if name.split()[0] in _FUNCTION_WORDS or name in names:
                continue
            names.append(name)
        payload = {
            "active_entities": [{"name": name, "aliases": [], "description": ["mentioned in the document"]} for name in names],
            "insertions": [{"name": name, "anchor_text": name, "left_context": "", "right_context": ""} for name in names],
        }
        text = "<think>plan</think>" + json.dumps(payload) + "\n[END_JSON]"
    else:
        text = "Mira Chen leads the Aurora team."
    return text


def saadi_like_responder(path: str, body: dict[str, Any]) -> Response | None:
    """HTTP form of :func:`saadi_like_reply` for the fake OpenAI and Anthropic endpoints."""

    text = saadi_like_reply(last_user_text(body))
    if path.endswith("/chat/completions"):
        return 200, {}, openai_payload(body, text)
    if path.endswith("/messages"):
        return 200, {}, anthropic_payload(body, text)
    return None  # embeddings and anything else: default handling
