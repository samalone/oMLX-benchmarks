"""Serve the HTML report over HTTP, rendered fresh on every request.

The page polls /version and reloads itself when the results change.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .db import DB
from .html_report import render
from .report import digest

log = logging.getLogger("omlxbench")


def parse_addr(text: str) -> tuple[str, int]:
    """'8765', ':8765' or 'host:8765' -> (host, port); the host defaults to localhost."""
    host, _, port = text.rpartition(":")
    return host or "127.0.0.1", int(port)


def _version(d: dict) -> str:
    """Changes exactly when the page's content would: a hash of the digest it renders."""
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _handler(db_path: Path) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?")[0]
            if path not in ("/", "/index.html", "/version"):
                self.send_error(404)
                return
            try:
                # A connection per request: sqlite3 connections belong to one thread.
                db = DB(db_path)
                try:
                    d = digest(db)
                finally:
                    db.conn.close()
                version = _version(d)
                if path == "/version":
                    body, ctype = version.encode(), "text/plain"
                else:
                    body, ctype = render(d, datetime.now(), version).encode(), "text/html"
            except Exception:
                log.exception("report page failed")
                self.send_error(500)
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            log.debug("http: " + fmt, *args)

    return Handler


def make_server(db_path: Path, addr: tuple[str, int]) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(addr, _handler(db_path))
    server.daemon_threads = True
    return server


def serve_in_background(db_path: Path, addr: tuple[str, int]) -> ThreadingHTTPServer | None:
    """Start the report server on a daemon thread. A busy port is logged, not fatal."""
    try:
        server = make_server(db_path, addr)
    except OSError as e:
        log.warning("report server not started on %s:%s: %s", *addr, e)
        return None
    threading.Thread(target=server.serve_forever, name="report-http", daemon=True).start()
    log.info("report at http://%s:%s/", *addr)
    return server
