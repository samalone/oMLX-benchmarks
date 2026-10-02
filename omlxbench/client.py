"""Thin HTTP client for the oMLX server (verified against oMLX 0.7.0).

Two auth schemes are in play: `/v1/*` and `/api/status` take the API key as a
Bearer header, while `/admin/api/*` only accepts the `omlx_admin_session`
cookie that `POST /admin/api/login` sets.
"""

from __future__ import annotations

import json
import queue
import threading
from typing import Any, Iterator
from urllib.parse import quote

import httpx


class OmlxError(RuntimeError):
    def __init__(self, status: int, detail: str, path: str):
        super().__init__(f"{path}: HTTP {status}: {detail}")
        self.status = status


def _detail(resp: httpx.Response) -> str:
    try:
        body = resp.json()
    except ValueError:
        return resp.text[:500]
    if isinstance(body, dict) and "detail" in body:
        return str(body["detail"])
    return json.dumps(body)[:500]


# Everything a call to the server can raise that means "no usable answer
# right now". httpx transport errors are not OSError subclasses, and
# ValueError covers a non-JSON body.
NET_ERRORS = (OmlxError, OSError, httpx.HTTPError, ValueError)


class OmlxClient:
    def __init__(self, base_url: str, api_key: str, timeout: float = 60.0):
        self.base_url = base_url
        self.api_key = api_key
        self.http = httpx.Client(base_url=base_url, timeout=timeout)
        self._logged_in = False

    def close(self) -> None:
        self.http.close()

    # -- plumbing ---------------------------------------------------------

    def login(self) -> None:
        resp = self.http.post(
            "/admin/api/login", json={"api_key": self.api_key, "remember": True}
        )
        if resp.status_code != 200:
            raise OmlxError(resp.status_code, _detail(resp), "/admin/api/login")
        self._logged_in = True

    def _request(self, method: str, path: str, *, admin: bool, **kw: Any) -> Any:
        if admin and not self._logged_in:
            self.login()
        if not admin:
            kw.setdefault("headers", {})["Authorization"] = f"Bearer {self.api_key}"
        resp = self.http.request(method, path, **kw)
        if admin and resp.status_code == 401:
            # Session expired (24h/30d cookie) or server restarted with a new secret.
            self.login()
            resp = self.http.request(method, path, **kw)
        if resp.status_code >= 400:
            raise OmlxError(resp.status_code, _detail(resp), path)
        return resp.json() if resp.content else None

    def admin(self, method: str, path: str, **kw: Any) -> Any:
        return self._request(method, "/admin/api" + path, admin=True, **kw)

    def api(self, method: str, path: str, **kw: Any) -> Any:
        return self._request(method, path, admin=False, **kw)

    # -- server state -----------------------------------------------------

    def health(self) -> dict:
        resp = self.http.get("/health")
        return {"http_status": resp.status_code, **(resp.json() if resp.content else {})}

    def status(self) -> dict:
        return self.api("GET", "/api/status")

    def device_info(self) -> dict:
        return self.admin("GET", "/device-info")

    def stats(self) -> dict:
        return self.admin("GET", "/stats")

    def global_settings(self) -> dict:
        return self.admin("GET", "/global-settings")

    def admin_models(self) -> list[dict]:
        return self.admin("GET", "/models")["models"]

    def usage(self, range_: str = "30d") -> dict:
        return self.admin("GET", "/usage", params={"range": range_})

    def update_model_settings(self, model_id: str, settings: dict) -> Any:
        return self.admin("PUT", f"/models/{quote(model_id, safe='')}/settings", json=settings)

    # -- benchmarks -------------------------------------------------------

    def perf_active(self) -> dict:
        return self.admin("GET", "/bench/active")

    def context_active(self) -> dict:
        return self.admin("GET", "/bench/context/active")

    def accuracy_status(self) -> dict:
        return self.admin("GET", "/bench/accuracy/queue/status")

    def accuracy_results(self) -> dict:
        return self.admin("GET", "/bench/accuracy/results")

    def start_perf(self, request: dict) -> dict:
        return self.admin("POST", "/bench/start", json=request)

    def perf_results(self, bench_id: str) -> dict:
        return self.admin("GET", f"/bench/{bench_id}/results")

    def cancel_perf(self, bench_id: str) -> Any:
        return self.admin("POST", f"/bench/{bench_id}/cancel")

    def add_accuracy(self, request: dict) -> dict:
        return self.admin("POST", "/bench/accuracy/queue/add", json=request)

    def cancel_accuracy(self) -> Any:
        # Note: cancels the running accuracy job AND clears oMLX's queue.
        return self.admin("POST", "/bench/accuracy/cancel")

    def start_context(self, request: dict) -> dict:
        return self.admin("POST", "/bench/context/start", json=request)

    def context_results(self, bench_id: str) -> dict:
        return self.admin("GET", f"/bench/context/{bench_id}/results")

    def cancel_context(self, bench_id: str) -> Any:
        return self.admin("POST", f"/bench/context/{bench_id}/cancel")

    def results(self, kind: str, bench_id: str | None) -> dict:
        """Final/current state of a run. Accuracy has no per-run endpoint."""
        if kind == "perf":
            return self.perf_results(bench_id)
        if kind == "context":
            return self.context_results(bench_id)
        return self.accuracy_status()

    def cancel(self, kind: str, bench_id: str | None) -> None:
        """Cancel a run; a run that already ended (HTTP 400) is fine."""
        try:
            if kind == "perf":
                self.cancel_perf(bench_id)
            elif kind == "context":
                self.cancel_context(bench_id)
            else:
                self.cancel_accuracy()
        except OmlxError as e:
            if e.status != 400:
                raise

    @staticmethod
    def stream_path(kind: str, bench_id: str) -> str:
        return {
            "perf": f"/admin/api/bench/{bench_id}/stream",
            "accuracy": f"/admin/api/bench/accuracy/{bench_id}/stream",
            "context": f"/admin/api/bench/context/{bench_id}/stream",
        }[kind]

    def iter_events(self, path: str) -> Iterator[dict]:
        """Yield SSE `data:` payloads until the server closes the stream.

        oMLX replays the run's whole event log to every subscriber, then
        follows live, sending `: keepalive` comments every 60s of silence.
        """
        if not self._logged_in:
            self.login()
        timeout = httpx.Timeout(10.0, read=150.0)
        with httpx.Client(base_url=self.base_url, cookies=self.http.cookies, timeout=timeout) as c:
            with c.stream("GET", path, headers={"Accept": "text/event-stream"}) as resp:
                if resp.status_code >= 400:
                    resp.read()
                    raise OmlxError(resp.status_code, _detail(resp), path)
                for line in resp.iter_lines():
                    if line.startswith("data:"):
                        yield json.loads(line[5:].strip())


def in_flight(status: dict) -> int:
    """Requests in flight according to /api/status.

    `active_requests` counts every request with an output collector, which
    includes ones still queued; `waiting_requests` counts the queued ones a
    second time, so it must not be added.
    """
    return status.get("active_requests") or 0


def model_load(activity_entry: dict) -> int:
    """Requests on one model according to /admin/api/activity, which splits
    them into running (`active_requests`) and queued (`waiting_requests`)."""
    return (activity_entry.get("active_requests") or 0) + (activity_entry.get("waiting_requests") or 0)


class EventPump:
    """Read an SSE stream on a background thread so the caller can poll.

    Items on `.queue` are event dicts, then a final ("eof", exc_or_None).
    """

    EOF = "eof"

    def __init__(self, client: OmlxClient, path: str):
        self.queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, args=(client, path), daemon=True)
        self._thread.start()

    def _run(self, client: OmlxClient, path: str) -> None:
        try:
            for event in client.iter_events(path):
                self.queue.put(event)
        except Exception as exc:  # noqa: BLE001 - surfaced to the consumer
            self.queue.put((self.EOF, exc))
        else:
            self.queue.put((self.EOF, None))


def requests_on(client: OmlxClient, model_id: str, status: dict) -> int:
    """Requests in flight on `model_id`, including ones queued behind a prefill.

    /api/status counts every engine's requests (queued ones too) but not per
    model; /activity is per model but misses queued requests. So: the status
    total minus the other models' /activity counts.
    """
    total = in_flight(status)
    if not total or not any(m != model_id for m in status.get("loaded_models") or []):
        return total
    activity = client.admin("GET", "/activity")
    others = sum(model_load(m) for m in (activity.get("active_models") or {}).get("models") or []
                 if m.get("id") != model_id)
    return max(0, total - others)
