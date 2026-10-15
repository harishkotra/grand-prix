"""A fake /v1/systemone server, so adapter behaviour is tested without any model.

It answers with whatever distribution the test tells it to, can be told to fail,
and runs on a thread in-process. It is deliberately strict about the request
shape, because a malformed question is rejected with a 4xx by the real servers
rather than guessed at.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class Handler(BaseHTTPRequestHandler):
    probabilities: dict[str, float] = {"hard_left": 0.1, "left": 0.2, "straight": 0.4, "right": 0.2, "hard_right": 0.1}
    fail_code: int | None = None
    fail_body: str = '{"error":"boom"}'
    require_model: bool = False
    last_request: dict[str, Any] = {}
    requests: list[dict[str, Any]] = []

    def log_message(self, *args: Any) -> None:  # keep the test output clean
        return

    def do_POST(self) -> None:  # noqa: N802 - http.server naming
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            self.send_error(400, "bad json")
            return
        Handler.last_request = body
        Handler.requests.append(body)
        if Handler.fail_code is not None:
            payload = Handler.fail_body.encode("utf-8")
            self.send_response(Handler.fail_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if Handler.require_model and "model" not in body:
            payload = b'{"error":{"message":"model is required"}}'
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        question_id = next(iter((body.get("questions") or {"steer": {}}).keys()))
        answer = {
            "model": body.get("model", "fake"),
            "answers": {
                question_id: {
                    "type": "choice",
                    "choice": max(Handler.probabilities.items(), key=lambda item: item[1])[0],
                    "probabilities": dict(Handler.probabilities),
                    "confidence": 0.5,
                }
            },
            "usage": {"output_tokens": 0},
            "warnings": [],
        }
        payload = json.dumps(answer).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        payload = b'{"status":"ok"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class FakeServer:
    def __init__(self, port: int = 18199) -> None:
        self.port = port
        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self) -> "FakeServer":
        Handler.fail_code = None
        Handler.require_model = False
        Handler.requests = []
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
