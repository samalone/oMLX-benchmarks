"""Serve the HTML report over HTTP, rendered fresh on every request."""

from __future__ import annotations

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


def _handler(db_path: Path) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.split("?")[0] not in ("/", "/index.html"):
                self.send_error(404)
                return
            try:
                # A connection per request: sqlite3 connections belong to one thread.
                db = DB(db_path)
                try:
                    body = render(digest(db), datetime.now()).encode()
                finally:
                    db.conn.close()
            except Exception:
                log.exception("report page failed")
                self.send_error(500)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
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
