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
  more requests in flight on the benchmark's model than the benchmark's own.

Cancelling safely: cancelling makes oMLX unload the benchmark's model with an
immediate abort, which strands any user request on that same model (it hangs
and never completes). So every cancel first waits until no user request is in
flight on the model under test (see `_Execution.cancel_when_clear`).
"""

from __future__ import annotations

import logging
import os
import queue
import re
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from .client import NET_ERRORS, EventPump, OmlxClient, OmlxError, in_flight, requests_on
from .control import daemon_alive, pause_state
from .db import DB, dumps, now
from .importer import ImportState, import_ui
from .planner import WorkUnit, plan
from .snapshot import State, capture_state
from .specs import Spec, Targets, perf_spec_from_result

log = logging.getLogger("omlxbench")

POLL_SECONDS = 2.0
FAST_POLL_SECONDS = 0.5           # while waiting for a user request to finish
GUEST_DEBOUNCE_SECONDS = 3.0      # a surplus request must persist this long to count
PHANTOM_GUEST_SECONDS = 15.0      # a "guest" that neither stays nor completes was noise
IDLE_SLEEP_SECONDS = 15.0
NOTHING_TO_DO_SLEEP_SECONDS = 300.0
DRAIN_TIMEOUT_SECONDS = 120.0
SHUTDOWN_GRACE_SECONDS = 15.0     # launchd sends SIGKILL 20s after SIGTERM
ORPHAN_GRACE_SECONDS = 120.0
USAGE_SNAPSHOT_INTERVAL = timedelta(hours=24)
IMPORT_INTERVAL_SECONDS = 600

# Requests the benchmark itself has in flight, by progress phase. Phases not
# listed (unload, load, cleanup, upload...) have none.
_OWN_REQUESTS = {"warmup": 1, "single": 1, "calibrate": 1, "estimate": 1, "verify": 1}


# -- observing the server ----------------------------------------------------


def cancel_orphan(client: OmlxClient, kind: str, bench_id: str | None, model_id: str,
                  grace: float = ORPHAN_GRACE_SECONDS) -> None:
    """Cancel a run we know nothing live about (no phase info), without
    stranding a user request: wait until at most one request (the benchmark's
    own) is on its model, or give up waiting after `grace` seconds."""
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        try:
            if requests_on(client, model_id, client.status()) <= 1:
                break
        except NET_ERRORS:
            pass
        time.sleep(FAST_POLL_SECONDS)
    client.cancel(kind, bench_id)


@dataclass
class TrafficWatch:
    """Tracks when the server last showed signs of real use (between runs)."""

    last_total: int | None = None
    last_activity: float = field(default_factory=time.monotonic)

    def observe(self, status: dict) -> None:
        total = status.get("total_requests")
        if total != self.last_total or in_flight(status) or status.get("models_loading"):
            self.touch()
        self.last_total = total

    def touch(self) -> None:
        self.last_activity = time.monotonic()

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
        self.imports = ImportState()
        self._last_gate_msg: str | None = None
        self._last_import = float("-inf")

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
                self.maybe_import_ui()
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
                try:
                    state = capture_state(self.client, self.db)
                except NET_ERRORS as e:
                    self._say(f"server unreachable: {e}")
                    time.sleep(IDLE_SLEEP_SECONDS)
                    continue
                units = plan(self.db, targets, state.models, state.env.hash)
                if not units:
                    self._say("nothing to do: every target has a result")
                    if once:
                        return
                    time.sleep(NOTHING_TO_DO_SLEEP_SECONDS)
                    continue
                self._last_gate_msg = None
                self.execute(units[0], state)
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
            self.traffic.observe(self.client.status())
            quiet = self.traffic.quiet_for()
            if quiet < quiet_minutes * 60:
                left = quiet_minutes * 60 - quiet
                return False, f"waiting for {quiet_minutes:g} quiet minutes (~{left / 60:.0f} min to go)"
            # Only worth asking once we'd otherwise start.
            health = self.client.health()
            if health.get("http_status") != 200 or health.get("status") != "healthy":
                return False, f"server not ready: {health.get('status')}"
            busy = self.other_bench_running()
        except NET_ERRORS as e:
            self.traffic.touch()
            return False, f"server unreachable: {e}"
        if busy:
            self.traffic.touch()
            return False, f"another benchmark is running ({busy})"
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

    def maybe_import_ui(self) -> None:
        """Keep results of runs started from the web UI (oMLX forgets them)."""
        if time.monotonic() - self._last_import < IMPORT_INTERVAL_SECONDS:
            return
        self._last_import = time.monotonic()
        try:
            import_ui(self.client, self.db, state=self.imports)
        except _Shutdown:
            raise
        except Exception as e:  # noqa: BLE001 - a failed import must not stop the runner
            log.warning("import of web-UI results failed: %s", e)

    def maybe_snapshot_usage(self) -> None:
        row = self.db.q1("SELECT max(captured_at) AS t FROM usage_snapshots")
        last = row["t"] if row else None
        if last and datetime.fromisoformat(last) > datetime.now(timezone.utc) - USAGE_SNAPSHOT_INTERVAL:
            return
        try:
            usage = self.client.usage("30d")
        except NET_ERRORS as e:
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
                result = self.client.results(run["kind"], run["omlx_bench_id"])
                if run["kind"] == "accuracy":
                    ours = result.get("current_bench_id") == run["omlx_bench_id"]
                    server_status = "running" if ours and result.get("running") else None
                else:
                    server_status = result.get("status")
                if server_status == "running":
                    log.info("cancelling orphaned run %s (%s)", run["id"], run["omlx_bench_id"])
                    cancel_orphan(self.client, run["kind"], run["omlx_bench_id"], run["model_id"])
            except NET_ERRORS:
                pass
            self.db.finish_run(run["id"], status="interrupted",
                               error=f"runner exited mid-run (server status: {server_status})")

    # -- one run ----------------------------------------------------------

    def execute(self, unit: WorkUnit, state: State) -> None:
        log.info("starting %s", unit.describe())
        request = unit.request()
        run_id = self.db.create_run(
            kind=unit.kind, model_id=unit.model.model_id,
            model_snapshot_id=state.snapshot_ids.get(unit.model.model_id),
            environment_id=state.env_id, request=request, spec_keys=[s.key for s in unit.specs],
        )
        if unit.kind == "tools":
            from .harness import ToolsExecution

            harness = ToolsExecution(self.client, self.db, unit, state.env.hash, run_id)
            try:
                harness.run()  # closing our connection on the way out cancels cleanly
            except (KeyboardInterrupt, _Shutdown):
                self.db.finish_run(run_id, status="cancelled", cancel_reason="shutdown")
                raise
            if harness.cancel_reason == "user_traffic":
                self.traffic.touch()
            return
        exec_ = _Execution(self.client, self.db, unit, state.env.hash, run_id, request)
        try:
            exec_.run()
            if exec_.cancel_reason == "user_traffic":
                self.traffic.touch()  # restart the quiet period from now
        except (KeyboardInterrupt, _Shutdown):
            exec_.cancel_when_clear("shutdown", grace=SHUTDOWN_GRACE_SECONDS)
            exec_.restore_context_setting()
            self.db.finish_run(run_id, status="cancelled", cancel_reason="shutdown")
            raise


@dataclass
class _Observation:
    paused: bool
    completed: int  # API requests completed since the run started
    foreign: bool   # another model loaded/loading while the benchmark should be alone
    guests: int     # user requests in flight on the benchmark's model


class _Execution:
    """One oMLX benchmark run: start, follow its events, yield, record."""

    def __init__(self, client: OmlxClient, db: DB, unit: WorkUnit, env_hash: str,
                 run_id: int, request: dict):
        self.client = client
        self.db = db
        self.unit = unit
        self.env_hash = env_hash
        self.run_id = run_id
        self.request = request
        self.bench_id: str | None = None
        self.baseline = 0
        self.phase: str | None = None
        self.current_batch = 0
        self.terminal: dict | None = None
        self.recorded: set[str] = set()
        self.upload_events: list[dict] = []
        self.prior_max_context: object = _UNSET
        self.pending_context: tuple[Spec, dict] | None = None
        # Yield state machine: running -> waiting (for a user request on our
        # model to finish) -> cancelling. See step().
        self.state = "running"
        self.cancel_reason: str | None = None
        self.surplus_since: float | None = None
        self.wait_reason: str | None = None
        self.wait_target = 0
        self.last_guest = 0.0

    # -- start ------------------------------------------------------------

    def start(self) -> None:
        kind = self.unit.kind
        if kind == "perf":
            self.bench_id = self.client.start_perf(self.request)["bench_id"]
        elif kind == "accuracy":
            status = self.client.add_accuracy(self.request)
            if status.get("current_model") != self.unit.model.model_id:
                # Someone else's job got there first; treat it like a busy server.
                raise OmlxError(409, f"accuracy queue did not start our run: {status}",
                                "/admin/api/bench/accuracy/queue/add")
            self.bench_id = status["current_bench_id"]
        else:
            self.prior_max_context = self.unit.model.settings.get("max_context_window")
            self.bench_id = self.client.start_context(self.request)["bench_id"]
        self.db.set_bench_id(self.run_id, self.bench_id)

    def run(self) -> None:
        try:
            self.baseline = self.client.status().get("total_requests") or 0
            self.start()
        except NET_ERRORS as e:
            self.db.finish_run(self.run_id, status="error", error=str(e))
            if isinstance(e, OmlxError) and e.status in (400, 404, 422):
                self.fail_unrecorded(str(e))
            log.warning("could not start run: %s", e)
            return
        log.info("oMLX bench id %s", self.bench_id)
        try:
            self.consume()
        finally:
            self.restore_context_setting()
        self.finish()

    # -- event loop ---------------------------------------------------------

    def consume(self) -> None:
        pump = EventPump(self.client, self.client.stream_path(self.unit.kind, self.bench_id))
        seq = 0
        next_check = time.monotonic() + POLL_SECONDS
        drain_deadline = 0.0
        while True:
            try:
                item = pump.queue.get(timeout=max(0.05, next_check - time.monotonic()))
            except queue.Empty:
                item = None
            if isinstance(item, tuple) and item[0] == EventPump.EOF:
                if item[1] is not None and self.terminal is None:
                    log.warning("event stream ended with error: %s", item[1])
                return
            if item is not None:
                seq += 1
                self.db.add_events(self.run_id, [(seq, item)])
                self.handle(item)
            if self.state == "cancelling":
                if time.monotonic() > drain_deadline:
                    log.warning("gave up waiting for the cancelled run to wind down")
                    return
            elif time.monotonic() >= next_check:
                self.step()
                if self.state == "cancelling":
                    drain_deadline = time.monotonic() + DRAIN_TIMEOUT_SECONDS
                fast = self.state == "waiting"
                next_check = time.monotonic() + (FAST_POLL_SECONDS if fast else POLL_SECONDS)

    def handle(self, ev: dict) -> None:
        t = ev.get("type")
        if t == "progress":
            self.phase = ev.get("phase")
            msg = ev.get("message") or ""
            if m := re.match(r"Batch (\d+)x", msg):  # oMLX reports the batch size only here
                self.current_batch = int(m.group(1))
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

    # -- yielding -----------------------------------------------------------

    def own_requests(self) -> int:
        """Requests the benchmark itself has in flight right now."""
        if self.phase == "batch":
            return self.current_batch
        if self.phase == "eval":
            return self.request.get("batch_size", 1)
        return _OWN_REQUESTS.get(self.phase, 0)

    def observe(self) -> _Observation | None:
        try:
            status = self.client.status()
            guests = requests_on(self.client, self.unit.model.model_id, status) - self.own_requests()
        except NET_ERRORS:
            return None
        steady = self.own_requests() > 0
        others = [m for m in status.get("loaded_models") or [] if m != self.unit.model.model_id]
        return _Observation(
            paused=pause_state(self.db)[0],
            completed=(status.get("total_requests") or 0) - self.baseline,
            foreign=steady and bool(others or status.get("models_loading")),
            guests=guests,
        )

    def step(self) -> None:
        """One poll of the yield state machine.

        running: yield on a pause, a completed API request, another model
          being loaded, or a surplus request on our model that persists
          (requests linger briefly at test boundaries). If a user request is
          on our model, go to waiting instead of cancelling.
        waiting: cancel once every user request seen has completed. If the
          surplus vanishes without anything completing, it was noise: go
          back to running and decide again.
        """
        o = self.observe()
        if o is None:
            return
        t = time.monotonic()
        if self.state == "running":
            if o.guests > 0:
                self.surplus_since = self.surplus_since or t
            else:
                self.surplus_since = None
            reason = ("paused" if o.paused else
                      "user_traffic" if o.completed > 0 or o.foreign else
                      "user_traffic" if self.surplus_since and t - self.surplus_since >= GUEST_DEBOUNCE_SECONDS
                      else None)
            if reason is None:
                return
            if o.guests > 0:
                log.info("user request on %s; cancelling once it finishes", self.unit.model.model_id)
                self.state, self.wait_reason = "waiting", reason
                self.wait_target, self.last_guest = o.completed + o.guests, t
            else:
                self.cancel(reason)
        elif self.state == "waiting":
            if o.guests > 0:
                self.wait_target = max(self.wait_target, o.completed + o.guests)
                self.last_guest = t
            elif o.completed >= self.wait_target:
                self.cancel(self.wait_reason or "user_traffic")
            elif t - self.last_guest > PHANTOM_GUEST_SECONDS:
                log.info("no user request after all; continuing")
                self.state, self.surplus_since = "running", None

    def cancel(self, reason: str) -> None:
        self.state, self.cancel_reason = "cancelling", reason
        log.info("yielding the server (%s): cancelling %s", reason, self.bench_id)
        try:
            self.client.cancel(self.unit.kind, self.bench_id)
        except NET_ERRORS as e:
            log.warning("cancel failed: %s", e)

    def cancel_when_clear(self, reason: str, grace: float) -> None:
        """Blocking cancel for shutdown: run the state machine (without
        reading events) until it cancels, or cancel anyway after `grace`."""
        if self.state == "cancelling" or self.bench_id is None:
            return
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            o = self.observe()
            if o is not None and o.guests <= 0 and (self.state != "waiting"
                                                    or o.completed >= self.wait_target):
                break
            if o is not None and o.guests > 0:
                self.state = "waiting"
                self.wait_target = max(self.wait_target, o.completed + o.guests)
            time.sleep(FAST_POLL_SECONDS)
        self.cancel(reason)

    # -- recording ------------------------------------------------------------

    def spec_for(self, data: dict) -> Spec | None:
        if self.unit.kind == "perf":
            p = self.unit.specs[0].p
            return perf_spec_from_result(data, tg=self.request["generation_length"],
                                         context_profile=p["context_profile"],
                                         force_lm_engine=p["force_lm_engine"])
        spec = self.unit.specs[0]
        if self.unit.kind == "accuracy" and data.get("benchmark") != spec.p["suite"]:
            return None
        return spec

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
            self.db.insert_accuracy_result(self.run_id, self.unit.model.model_id, spec.key,
                                           spec.p["sample_size"], data)
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
        except NET_ERRORS as e:
            log.warning("could not restore max_context_window: %s", e)
        self.prior_max_context = _UNSET  # restore at most once (also called on shutdown)
        if self.pending_context:
            spec, data = self.pending_context
            self.pending_context = None
            self.db.insert_context_result(self.run_id, spec.key, data, restored)
            self.mark_ok(spec)

    def fail_unrecorded(self, error: str) -> None:
        for spec in self.unit.specs:
            if spec.key not in self.recorded:
                self.db.mark_spec(self.unit.model.model_id, spec.key,
                                  self.unit.model.settings_fingerprint, ok=False,
                                  run_id=self.run_id, error=error,
                                  environment_hash=self.env_hash)

    # -- finish -------------------------------------------------------------

    def finish(self) -> None:
        final = None
        try:
            final = self.client.results(self.unit.kind, self.bench_id)
        except NET_ERRORS as e:
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


_UNSET = object()


def _fmt(v) -> str:
    return "n/a" if v is None else f"{v:.1f}"
