"""A tiny event server, so a browser can hear the brain.

Strudel runs in a web page, which cannot open a UDP socket, so the browser
target is served over HTTP instead: one Server-Sent Events stream carrying a
JSON frame per control tick, the player page, and the JavaScript helper that
turns the frames into Strudel signals.

Standard library only, one background thread, and a bounded queue per client:
a browser tab that stops reading must never be able to stall the flight loop.
"""

from __future__ import annotations

import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path

CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".mjs": "text/javascript; charset=utf-8",
                 ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8"}


def asset(name: str) -> str:
    """Read one of the packaged browser files (works from a wheel or a checkout)."""
    return resources.files("flydrones.music").joinpath("web", name).read_text(encoding="utf-8")


class EventServer:
    """Broadcasts JSON frames to every connected browser."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765, queue_size: int = 64):
        self.host, self.port = host, int(port)
        self.queue_size = int(queue_size)
        self._clients: list[queue.Queue] = []
        self._lock = threading.Lock()
        self.last: dict = {}
        self.published = 0
        self.dropped = 0
        self._httpd = _make_server(self)
        self.port = self._httpd.server_address[1]  # port 0 -> whatever we were given
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="flydrones-music", daemon=True)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def start(self) -> EventServer:
        self._thread.start()
        return self

    def publish(self, frame: dict) -> None:
        self.last = frame
        line = f"data: {json.dumps(frame, separators=(',', ':'))}\n\n".encode()
        with self._lock:
            clients = list(self._clients)
        for q in clients:
            try:
                q.put_nowait(line)
            except queue.Full:  # a tab that is not reading does not get to slow the fly down
                self.dropped += 1
        self.published += 1

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=self.queue_size)
        with self._lock:
            self._clients.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._clients:
                self._clients.remove(q)

    @property
    def clients(self) -> int:
        with self._lock:
            return len(self._clients)

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


def _make_server(hub: EventServer) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "flydrones-music"

        def log_message(self, *_args) -> None:  # the flight log is the interesting one
            pass

        def _head(self, status: int, ctype: str, length: int | None = None, stream: bool = False) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Access-Control-Allow-Origin", "*")  # so strudel.cc can listen too
            if stream:
                # No length and no chunking: the stream ends when the socket does,
                # which is what EventSource expects and what keeps this readable
                # by anything that can open a URL.
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.close_connection = True
            if length is not None:
                self.send_header("Content-Length", str(length))
            self.end_headers()

        def _send(self, body: str, ctype: str = "text/plain; charset=utf-8", status: int = 200) -> None:
            data = body.encode("utf-8")
            self._head(status, ctype, len(data))
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's spelling
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                return self._send(asset("index.html"), CONTENT_TYPES[".html"])
            if path == "/flybrain.mjs":
                return self._send(asset("flybrain.mjs"), CONTENT_TYPES[".mjs"])
            if path == "/state.json":
                return self._send(json.dumps(hub.last), "application/json")
            if path == "/events":
                return self._stream()
            return self._send(f"not found: {path}\n", status=404)

        def _stream(self) -> None:
            q = hub.subscribe()
            self._head(200, "text/event-stream", stream=True)
            try:
                self.wfile.write(b": flydrones\n\n")
                if hub.last:
                    self.wfile.write(f"data: {json.dumps(hub.last, separators=(',', ':'))}\n\n".encode())
                self.wfile.flush()
                while True:
                    try:
                        self.wfile.write(q.get(timeout=5.0))
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")  # through proxies and idle timeouts
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass  # the tab was closed
            finally:
                hub.unsubscribe(q)

        def do_OPTIONS(self) -> None:  # noqa: N802
            self._head(204, "text/plain")

    httpd = ThreadingHTTPServer((hub.host, hub.port), Handler)
    httpd.daemon_threads = True
    return httpd


def write_assets(directory: str | Path) -> list[Path]:
    """Copy the browser player next to a flight, for serving it from somewhere else."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name in ("index.html", "flybrain.mjs"):
        p = out / name
        p.write_text(asset(name), encoding="utf-8")
        written.append(p)
    return written
