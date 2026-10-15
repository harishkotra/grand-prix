"""Zero-dependency HTTP gateway and race host.

Stdlib only, on purpose. The point of this piece is that it runs on one laptop
with nothing to install and no account to create, so the gateway is
`http.server` plus a dedicated thread running the asyncio race loop. FastAPI
would be a fine choice for a service under load; here it would be a dependency
to maintain for no gain.

Routes
    GET  /                    the dashboard
    GET  /static/<path>       assets (including the vendored PixiJS bundle)
    GET  /api/engines         every racer, probed live, with its status
    GET  /api/state           the latest snapshot, for the 30 Hz poll
    POST /api/race/start      {"racers": [...], "replay": false}
    POST /api/race/stop
    GET  /api/summary/<id>    the summary written for a finished race
    GET  /healthz
"""

from __future__ import annotations

import argparse
import asyncio
import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from race import RaceRunner

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
OUT_DIR = ROOT / "runs"


class RaceHost(threading.Thread):
    """Owns the asyncio loop that the race runs on, in its own thread."""

    def __init__(self, config: dict[str, Any], out_dir: Path) -> None:
        super().__init__(daemon=True, name="race-host")
        self.runner = RaceRunner(config, out_dir)
        self._loop = asyncio.new_event_loop()
        self._commands: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()
        self._ready = threading.Event()

    def run(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        finally:
            self._loop.close()

    async def _main(self) -> None:
        self._ready.set()
        while True:
            kind, payload = await self._commands.get()
            if kind == "start":
                try:
                    self.runner.start(
                        payload.get("racers"),
                        replay=bool(payload.get("replay", False)),
                        track=payload.get("track") or None,
                    )
                except ValueError as error:
                    print(f"[host] start refused: {error}")
            elif kind in {"stop", "shutdown"}:
                if self.runner.session is not None:
                    self.runner.session.stop()
                if kind == "shutdown":
                    return

    def submit(self, kind: str, payload: dict[str, Any] | None = None) -> None:
        self._ready.wait(timeout=5.0)
        self._loop.call_soon_threadsafe(self._commands.put_nowait, (kind, payload or {}))

    def session(self):
        return self.runner.session

    def state(self) -> dict[str, Any]:
        session = self.runner.session
        if session is None:
            return {"running": False, "finished": False, "race": None, "racers": [], "standings": [], "notes": []}
        return session.latest or session.snapshot()

    def catalogue(self) -> list[dict[str, Any]]:
        return self.runner.engine_catalogue()

    def tracks(self) -> list[dict[str, Any]]:
        return self.runner.track_catalogue()


class Handler(BaseHTTPRequestHandler):
    host: RaceHost
    server_version = "grand-prix/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        if "/api/state" in (args[0] if args else ""):
            return  # the UI polls this 30 times a second; do not log it
        print(f"[http] {fmt % args}")

    # -- helpers ------------------------------------------------------------
    def _json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path) -> None:
        if not path.is_file():
            self._json({"error": "not found", "path": str(path)}, 404)
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix == ".mjs":
            content_type = "text/javascript"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # No caching: this is a project under active change, and a stale app.js in
        # the browser cache looks exactly like a fix that did not work.
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    # -- routes -------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self._file(STATIC_DIR / "index.html")
        elif path.startswith("/static/"):
            relative = path[len("/static/") :]
            candidate = (STATIC_DIR / relative).resolve()
            if STATIC_DIR.resolve() not in candidate.parents and candidate != STATIC_DIR.resolve():
                self._json({"error": "forbidden"}, 403)
                return
            self._file(candidate)
        elif path == "/api/engines":
            self._json({"engines": self.host.catalogue()})
        elif path == "/api/tracks":
            self._json({"tracks": self.host.tracks()})
        elif path == "/api/state":
            self._json(self.host.state())
        elif path == "/healthz":
            session = self.host.session()
            self._json({"status": "ok", "race_id": session.race_id if session else None})
        elif path.startswith("/api/summary/"):
            race_id = path.rsplit("/", 1)[-1]
            self._file(OUT_DIR / f"{race_id}.summary.json")
        elif path.startswith("/api/telemetry/"):
            race_id = path.rsplit("/", 1)[-1]
            self._file(OUT_DIR / f"{race_id}.jsonl")
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/race/start":
            payload = self._body()
            self.host.submit("start", payload)
            self._json({"accepted": True, "requested": payload})
        elif path == "/api/race/stop":
            self.host.submit("stop")
            self._json({"stopped": True})
        else:
            self._json({"error": "not found"}, 404)


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Grand Prix: local decision models racing.")
    parser.add_argument("--port", type=int, default=8078)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    parser.add_argument("--out", default=str(OUT_DIR))
    parser.add_argument("--autostart", action="store_true", help="start a race as soon as the server is up")
    parser.add_argument("--replay", action="store_true", help="autostart in replay mode (synthetic racers)")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    host = RaceHost(config, Path(args.out))
    Handler.host = host
    host.start()

    if args.autostart:
        host.submit("start", {"replay": args.replay})

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Grand Prix dashboard on http://{args.host}:{args.port}/")
    print(f"Racers configured: {', '.join(spec['key'] for spec in config.get('racers', []))}")
    print("Start a race from the page, or POST /api/race/start")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        host.submit("shutdown")
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
