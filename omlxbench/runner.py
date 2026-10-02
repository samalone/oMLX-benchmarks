"""The background loop: wait for an idle server, run one unit, record, repeat.

Benchmarks are a low-priority guest on the server. The runner only starts a
run after the server has seen no API traffic for `quiet_minutes`, and cancels
its run as soon as it sees real use or a pause request. Results oMLX has
already emitted survive the cancel; only the test in progress is lost.

Detecting "real use" (oMLX 0.7.0):
- `/api/status.total_requests` counts completed HTTP API requests only; the
  built-in benchmarks call the engine directly and never move it.
- A long chat request only shows up there when it finishes, so during a run
  we also watch for models other than the benchmark's being loaded, and for
  more in-flight requests than the benchmark phase accounts for.
"""

from __future__ import annotations

import logging
import os
import queue
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from .client import EventPump, OmlxClient, OmlxError
from .db import DB, dumps, now
from .planner import WorkUnit, plan
from .snapshot import Environment, ModelSnapshot, capture_environment, snapshot_model
from .specs import Spec, Targets, accuracy, perf_batch, perf_single

log = logging.getLogger("omlxbench")

POLL_SECONDS = 2.0
IDLE_SLEEP_SECONDS = 15.0
NOTHING_TO_DO_SLEEP_SECONDS = 300.0
DRAIN_TIMEOUT_SECONDS = 120.0
USAGE_SNAPSHOT_INTERVAL = timedelta(hours=24)

# Benchmark phases during which only the benchmark's own model should be
# loaded and its own requests in flight. (Unload/load phases are excluded:
# other models legitimately appear there.)
_STEADY_PHASES = {"warmup", "single", "batch", "eval", "calibrate", "estimate", "verify"}


# -- pause control (shared with the CLI through the database) ---------------


def pause_state(db: DB) -> tuple[bool, str | None]:
    """(paused, description)."""
    until = db.get_control("paused_until")
    if not until:
        return False, None
    reason = db.get_control("pause_reason") or ""
    if until == "forever":
        return True, f"paused until resumed {reason}".strip()
    if datetime.fromisoformat(until) > datetime.now(timezone.utc):
        return True, f"paused until {until} {reason}".strip()
    db.set_control("paused_until", None)
    return False, None


def set_pause(db: DB, until: datetime | None, reason: str = "") -> None:
    db.set_control("paused_until", until.isoformat(timespec="seconds") if until else "forever")
    db.set_control("pause_reason", reason or None)


def clear_pause(db: DB) -> None:
    db.set_control("paused_until", None)
    db.set_control("pause_reason", None)


def cancel_bench(client: OmlxClient, kind: str, bench_id: str | None) -> None:
    try:
        if kind == "perf" and bench_id:
            client.cancel_perf(bench_id)
        elif kind == "context" and bench_id:
            client.cancel_context(bench_id)
        elif kind == "accuracy":
            client.cancel_accuracy()
    except OmlxError as e:
        # 400 = not running any more, which is what we wanted.
        if e.status != 400:
            raise


def daemon_alive(db: DB) -> int | None:
    pid = db.get_control("daemon_pid")
    if not pid:
        return None
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return None
    return int(pid)


# -- traffic observation ----------------------------------------------------


@dataclass
class TrafficWatch:
    """Tracks when the server last showed signs of real use."""

    last_total: int | None = None
    last_activity: float = field(default_factory=time.monotonic)

    def observe_idle(self, status: dict) -> None:
        total = status.get("total_requests")
        busy = (status.get("active_requests") or 0) + (status.get("waiting_requests") or 0)
        if total != self.last_total or busy or status.get("models_loading"):
            self.last_activity = time.monotonic()
        self.last_total = total

    def quiet_for(self) -> float:
        return time.monotonic() - self.last_activity


# -- the runner -------------------------------------------------------------


class _Shutdown(Exception):
    pass


class Runner:
    def __init__(self, client: OmlxClient, db: DB, targets_loader, quiet_minutes: float | None):
        self.client = client
        self.db = db
        self.load_targets = targets_loader
        self.quiet_minutes_override = quiet_minutes
        self.traffic = TrafficWatch()
        self._last_gate_msg: str | None = None

    # -- top level --------------------------------------------------------

    def loop(self, once: bool = False) -> None:
        other = daemon_alive(self.db)
        if other and other != os.getpid():
            raise SystemExit(f"another omlxbench runner is active (pid {other})")
        self.db.set_control("daemon_pid", str(os.getpid()))
        signal.signal(signal.SIGTERM, self._on_sigterm)
        try:
            self.reconcile_stale_runs()
            while True:
                self.db.set_control("daemon_heartbeat", now())
                targets: Targets = self.load_targets()
                quiet_min = (self.quiet_minutes_override
                             if self.quiet_minutes_override is not None else targets.quiet_minutes)
                ok, why = self.gate(quiet_min)
                if not ok:
                    self._say(why)
                    if once and why.startswith("paused"):
                        return
                    time.sleep(IDLE_SLEEP_SECONDS)
                    continue
                self.maybe_snapshot_usage()
                env, models = self.capture()
                units = plan(self.db, targets, models, env.hash)
                if not units:
                    self._say("nothing to do: every target has a result")
                    if once:
                        return
                    time.sleep(NOTHING_TO_DO_SLEEP_SECONDS)
                    continue
                self._last_gate_msg = None
                self.execute(units[0], env)
                if once:
                    return
        except (KeyboardInterrupt, _Shutdown):
            log.info("stopping")
        finally:
            if self.db.get_control("daemon_pid") == str(os.getpid()):
                self.db.set_control("daemon_pid", None)

    def _on_sigterm(self, *_):
        raise _Shutdown()

    def _say(self, msg: str) -> None:
        if msg != self._last_gate_msg:
            log.info(msg)
            self._last_gate_msg = msg

    def gate(self, quiet_minutes: float) -> tuple[bool, str]:
        paused, desc = pause_state(self.db)
        if paused:
            return False, desc or "paused"
        try:
            health = self.client.health()
            if health.get("http_status") != 200 or health.get("status") != "healthy":
                return False, f"server not ready: {health.get('status')}"
            status = self.client.status()
            busy = self.other_bench_running()
        except (OmlxError, OSError) as e:
            self.traffic.last_activity = time.monotonic()
            return False, f"server unreachable: {e}"
        self.traffic.observe_idle(status)
        if busy:
            self.traffic.last_activity = time.monotonic()
            return False, f"another benchmark is running ({busy})"
        quiet = self.traffic.quiet_for()
        if quiet < quiet_minutes * 60:
            left = quiet_minutes * 60 - quiet
            return False, f"waiting for {quiet_minutes:g} quiet minutes (~{left / 60:.0f} min to go)"
        return True, "ok"

    def other_bench_running(self) -> str | None:
        if self.client.perf_active().get("running"):
            return "performance"
        if self.client.context_active().get("running"):
            return "context"
        acc = self.client.accuracy_status()
        if acc.get("running") or acc.get("queue"):
            return "intelligence"
        return None

    def capture(self) -> tuple[Environment, list[ModelSnapshot]]:
        env = capture_environment(self.client)
        self.env_id = self.db.upsert_environment(env)
        models = [snapshot_model(e) for e in self.client.admin_models()]
        self.snapshot_ids = {m.model_id: self.db.upsert_model_snapshot(m) for m in models}
        return env, models

    def maybe_snapshot_usage(self) -> None:
        row = self.db.q1("SELECT max(captured_at) AS t FROM usage_snapshots")
        last = row["t"] if row else None
        if last and datetime.fromisoformat(last) > datetime.now(timezone.utc) - USAGE_SNAPSHOT_INTERVAL:
            return
        try:
            usage = self.client.usage("30d")
        except OmlxError as e:
            log.warning("usage snapshot failed: %s", e)
            return
        self.db.x("INSERT INTO usage_snapshots(captured_at, range, json) VALUES (?,?,?)",
                  now(), "30d", dumps(usage))

    # -- recovery ---------------------------------------------------------

    def reconcile_stale_runs(self) -> None:
        """Close out runs a previous runner left 'running' (crash, reboot)."""
        for run in self.db.q("SELECT * FROM runs WHERE status = 'running' AND source = 'runner'"):
            server_status = None
            try:
                if run["kind"] == "perf" and run["omlx_bench_id"]:
                    server_status = self.client.perf_results(run["omlx_bench_id"]).get("status")
                elif run["kind"] == "context" and run["omlx_bench_id"]:
                    server_status = self.client.context_results(run["omlx_bench_id"]).get("status")
                elif run["kind"] == "accuracy":
                    acc = self.client.accuracy_status()
                    if acc.get("current_bench_id") == run["omlx_bench_id"] and acc.get("running"):
                        server_status = "running"
            except OmlxError:
                pass
            if server_status == "running":
                log.info("cancelling orphaned run %s (%s)", run["id"], run["omlx_bench_id"])
                cancel_bench(self.client, run["kind"], run["omlx_bench_id"])
            self.db.finish_run(run["id"], status="interrupted",
                               error=f"runner exited mid-run (server status: {server_status})")

    # -- one run ----------------------------------------------------------

    def execute(self, unit: WorkUnit, env: Environment) -> str:
        log.info("starting %s", unit.describe())
        request = unit.request()
        run_id = self.db.create_run(
            kind=unit.kind, model_id=unit.model.model_id,
            model_snapshot_id=self.snapshot_ids.get(unit.model.model_id),
            environment_id=self.env_id, request=request, spec_keys=[s.key for s in unit.specs],
        )
        exec_ = _Execution(self, unit, env, run_id, request)
        try:
            return exec_.run()
        except (KeyboardInterrupt, _Shutdown):
            exec_.abort("shutdown")
            raise


class _Execution:
    def __init__(self, runner: Runner, unit: WorkUnit, env: Environment, run_id: int, request: dict):
        self.r = runner
        self.client = runner.client
        self.db = runner.db
        self.unit = unit
        self.env = env
        self.run_id = run_id
        self.request = request
        self.bench_id: str | None = None
        self.cancel_reason: str | None = None
        self.phase: str | None = None
        self.terminal: dict | None = None
        self.recorded: set[str] = set()
        self.upload_events: list[dict] = []
        self.prior_max_context: object = _UNSET
        self.pending_context: tuple[Spec, dict] | None = None

    # -- start ------------------------------------------------------------

    def start(self) -> None:
        kind = self.unit.kind
        if kind == "perf":
            self.bench_id = self.client.start_perf(self.request)["bench_id"]
        elif kind == "accuracy":
            status = self.client.add_accuracy(self.request)
            if status.get("current_model") != self.unit.model.model_id:
                raise RuntimeError(f"accuracy queue did not start our run: {status}")
            self.bench_id = status["current_bench_id"]
        else:
            self.prior_max_context = self.unit.model.settings.get("max_context_window")
            self.bench_id = self.client.start_context(self.request)["bench_id"]
        self.db.set_bench_id(self.run_id, self.bench_id)

    def run(self) -> str:
        try:
            self.start()
        except OmlxError as e:
            permanent = e.status in (400, 404, 422)
            self.db.finish_run(self.run_id, status="error", error=str(e))
            if permanent:
                self.fail_unrecorded(str(e))
            log.warning("could not start run: %s", e)
            return "error"
        self.baseline = self.client.status().get("total_requests")
        log.info("oMLX bench id %s", self.bench_id)
        try:
            self.consume()
        finally:
            self.restore_context_setting()
        return self.finish()

    # -- event loop ---------------------------------------------------------

    def consume(self) -> None:
        pump = EventPump(self.client, self.client.stream_path(self.unit.kind, self.bench_id))
        seq = 0
        next_check = time.monotonic() + POLL_SECONDS
        drain_deadline: float | None = None
        while True:
            try:
                item = pump.queue.get(timeout=POLL_SECONDS)
            except queue.Empty:
                item = None
            if isinstance(item, tuple) and item[0] == EventPump.EOF:
                if item[1] is not None and self.terminal is None:
                    log.warning("event stream ended with error: %s", item[1])
                return
            if item is not None:
                seq += 1
                self.db.add_event(self.run_id, seq, item)
                self.handle(item)
            if drain_deadline is not None:
                if time.monotonic() > drain_deadline:
                    log.warning("gave up waiting for the cancelled run to wind down")
                    return
            elif time.monotonic() >= next_check:
                next_check = time.monotonic() + POLL_SECONDS
                reason = self.should_yield()
                if reason:
                    self.abort(reason)
                    drain_deadline = time.monotonic() + DRAIN_TIMEOUT_SECONDS

    def handle(self, ev: dict) -> None:
        t = ev.get("type")
        if t == "progress":
            self.phase = ev.get("phase")
            msg = ev.get("message")
            if msg:
                log.info("  %s", msg)
        elif t == "result":
            self.record(ev.get("data") or {})
        elif t == "upload":
            self.upload_events.append(ev.get("data") or {})
        elif t in ("done", "error", "upload_done", "upload_skipped"):
            self.terminal = ev
            if t == "error":
                log.warning("  oMLX reported error: %s", ev.get("message"))

    def should_yield(self) -> str | None:
        paused, _ = pause_state(self.db)
        if paused:
            return "paused"
        try:
            status = self.client.status()
        except (OmlxError, OSError):
            return None
        if status.get("total_requests") != self.baseline:
            return "user_traffic"
        if self.phase in _STEADY_PHASES:
            ours = self.unit.model.model_id
            others = [m for m in status.get("loaded_models") or [] if m != ours]
            if others or status.get("models_loading"):
                return "user_traffic"
            in_flight = (status.get("active_requests") or 0) + (status.get("waiting_requests") or 0)
            if in_flight > self.expected_in_flight():
                return "user_traffic"
        return None

    def expected_in_flight(self) -> int:
        if self.unit.kind == "perf":
            return max([1, *self.request.get("batch_sizes", [])])
        if self.unit.kind == "accuracy":
            return self.request.get("batch_size", 1)
        return 1

    def abort(self, reason: str) -> None:
        if self.cancel_reason:
            return
        self.cancel_reason = reason
        log.info("yielding the server (%s): cancelling %s", reason, self.bench_id)
        try:
            cancel_bench(self.client, self.unit.kind, self.bench_id)
        except (OmlxError, OSError) as e:
            log.warning("cancel failed: %s", e)

    # -- recording ------------------------------------------------------------

    def spec_for(self, data: dict) -> Spec | None:
        kind = self.unit.kind
        if kind == "perf":
            p = self.unit.specs[0].p
            common = dict(tg=self.request["generation_length"], context_profile=p["context_profile"],
                          force_lm_engine=p["force_lm_engine"])
            if data.get("test_type") == "single":
                return perf_single(data.get("pp"), **common)
            if data.get("test_type") == "batch":
                return perf_batch(data.get("batch_size"), **common)
            return None
        if kind == "accuracy":
            p = self.unit.specs[0].p
            if data.get("benchmark") != p["suite"]:
                return None
            return accuracy(p["suite"], p["sample_size"], p["enable_thinking"],
                            p["sampling_profile"], p["batch_size"])
        return self.unit.specs[0]

    def record(self, data: dict) -> None:
        spec = self.spec_for(data)
        if spec is None:
            log.warning("  result did not match any spec; stored as event only")
            return
        if self.unit.kind == "perf":
            self.db.insert_perf_result(self.run_id, spec.key, data, spec.p["context_profile"])
            if data.get("test_type") == "single":
                log.info("  result pp%s: pp %.1f tok/s, tg %s tok/s, ttft %.0f ms",
                         data.get("pp"), data.get("processing_tps") or 0,
                         _fmt(data.get("gen_tps")), data.get("ttft_ms") or 0)
            else:
                log.info("  result batch%s: tg %s tok/s", data.get("batch_size"),
                         _fmt(data.get("tg_tps")))
        elif self.unit.kind == "accuracy":
            self.db.insert_accuracy_result(self.run_id, spec.key, spec.p["sample_size"], data)
            log.info("  result %s: %.1f%% (%s/%s)", data.get("benchmark"),
                     100 * (data.get("accuracy") or 0), data.get("correct"), data.get("total"))
        else:
            self.pending_context = (spec, data)  # recorded after the setting is restored
            log.info("  result context: verified %s tokens", data.get("verified_tokens"))
            return
        self.mark_ok(spec)

    def mark_ok(self, spec: Spec) -> None:
        self.recorded.add(spec.key)
        self.db.mark_spec(self.unit.model.model_id, spec.key, self.unit.model.settings_fingerprint,
                          ok=True, run_id=self.run_id)

    def restore_context_setting(self) -> None:
        if self.unit.kind != "context" or self.prior_max_context is _UNSET:
            return
        restored = None
        try:
            current = next((m for m in self.client.admin_models()
                            if m["id"] == self.unit.model.model_id), {})
            now_value = (current.get("settings") or {}).get("max_context_window")
            if now_value != self.prior_max_context:
                # null resets the field to its default, which is what "unset" was.
                self.client.update_model_settings(self.unit.model.model_id,
                                                  {"max_context_window": self.prior_max_context})
                restored = self.prior_max_context
                log.info("  restored max_context_window to %s (benchmark had set %s)",
                         self.prior_max_context, now_value)
        except (OmlxError, OSError) as e:
            log.warning("could not restore max_context_window: %s", e)
        if self.pending_context:
            spec, data = self.pending_context
            self.db.insert_context_result(self.run_id, spec.key, data, restored)
            self.mark_ok(spec)

    def fail_unrecorded(self, error: str) -> None:
        for spec in self.unit.specs:
            if spec.key not in self.recorded:
                self.db.mark_spec(self.unit.model.model_id, spec.key,
                                  self.unit.model.settings_fingerprint, ok=False,
                                  run_id=self.run_id, error=error,
                                  environment_hash=self.env.hash)

    # -- finish -------------------------------------------------------------

    def finish(self) -> str:
        final = None
        try:
            if self.unit.kind == "perf":
                final = self.client.perf_results(self.bench_id)
            elif self.unit.kind == "context":
                final = self.client.context_results(self.bench_id)
            else:
                final = self.client.accuracy_status()
        except (OmlxError, OSError) as e:
            log.warning("could not fetch final results: %s", e)

        server_status = (final or {}).get("status") or (final or {}).get("phase")
        terminal_type = (self.terminal or {}).get("type")
        if self.cancel_reason is None and server_status == "cancelled":
            # Cancelled by someone else: the pause command, or the web UI.
            self.cancel_reason = "paused" if pause_state(self.db)[0] else "external"
        if self.cancel_reason:
            status = "cancelled"
        elif terminal_type == "error" or server_status == "error":
            status = "error"
        else:
            status = "completed"

        error = None
        if status == "error":
            error = (self.terminal or {}).get("message") or (final or {}).get("error") or "unknown error"
            self.fail_unrecorded(error)
        elif status == "completed":
            # oMLX finished but some requested test produced nothing (e.g. a
            # batch test an engine silently skips): count it as a failed attempt.
            self.fail_unrecorded("run completed without producing this result")

        upload = (final or {}).get("upload_state") if self.unit.kind == "perf" else (
            self.upload_events or None)
        self.db.finish_run(self.run_id, status=status, cancel_reason=self.cancel_reason,
                           error=error, final=final, upload_state=upload)
        log.info("run %s %s%s (%d/%d results)", self.run_id, status,
                 f" ({self.cancel_reason})" if self.cancel_reason else "",
                 len(self.recorded), len(self.unit.specs))
        return status


_UNSET = object()


def _fmt(v) -> str:
    return "n/a" if v is None else f"{v:.1f}"
