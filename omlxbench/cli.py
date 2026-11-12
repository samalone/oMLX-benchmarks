"""omlxbench command line."""

from __future__ import annotations

import argparse
import fnmatch
import json
import logging
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .client import NET_ERRORS, OmlxClient
from .config import load_config
from .db import DB
from .planner import HARNESS_KINDS, iter_targets, missing_by_model, plan, spec_state
from .control import clear_pause, daemon_alive, pause_state, set_pause
from .runner import Runner, cancel_orphan, engine_up
from .snapshot import capture_state
from .specs import load_targets


def _parse_duration(text: str) -> timedelta:
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([smhd]?)", text.strip().lower())
    if not m:
        raise SystemExit(f"bad duration {text!r}; use e.g. 30m, 2h, 1d")
    n, unit = float(m.group(1)), m.group(2) or "m"
    name = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}[unit]
    return timedelta(**{name: n})


def _version_tuple(v: str | None) -> tuple:
    return tuple(int(x) for x in re.findall(r"\d+", v or ""))


class App:
    def __init__(self):
        self.cfg = load_config()
        self.db = DB(self.cfg.db_path)
        self.clients: dict = {}
        if self.cfg.api_key:
            self.clients["omlx"] = OmlxClient(self.cfg.base_url, self.cfg.api_key)
        if self.cfg.tensorfold_api_key:
            from .engines import TensorFoldClient

            self.clients["tensorfold"] = TensorFoldClient(self.cfg.tensorfold_url,
                                                          self.cfg.tensorfold_api_key)
        self.client = self.clients.get("omlx")

    def targets(self):
        return load_targets(self.cfg.targets_path)

    def active_engine(self) -> str | None:
        """The engine that is up (the first, if several are)."""
        return next((name for name, c in self.clients.items() if engine_up(c)), None)

    def current_plan(self, engine: str):
        """(state, units) for `engine`, which must be up. Snapshots are recorded."""
        state = capture_state(self.clients[engine], self.db)
        return state, plan(self.db, self.targets(), state.models, state.env.hash, engine)

    # -- commands -----------------------------------------------------------

    def cmd_status(self, args) -> None:
        up = []
        for name, c in self.clients.items():
            if not engine_up(c):
                print(f"{name + ':':<11}down")
                continue
            up.append(name)
            try:
                st = c.status()
            except NET_ERRORS as e:
                print(f"{name + ':':<11}unreachable ({e})")
                continue
            loaded = ", ".join(st.get("loaded_models") or []) or "none"
            if name == "omlx":
                print(f"{'omlx:':<11}oMLX {st.get('version')}, {st.get('models_discovered')} "
                      f"models, loaded: {loaded}")
            else:
                print(f"{name + ':':<11}up at {c.base_url}, serving {loaded}")
        pid = daemon_alive(self.db)
        beat = self.db.get_control("daemon_heartbeat")
        print(f"runner:  {'running (pid %d, last check %s)' % (pid, beat) if pid else 'not running'}")
        paused, desc = pause_state(self.db)
        print(f"pause:   {desc if paused else 'not paused'}")

        for run in self.db.q("SELECT * FROM runs WHERE status='running' ORDER BY id"):
            ev = self.db.q1("SELECT json FROM run_events WHERE run_id=? AND type='progress'"
                            " ORDER BY seq DESC LIMIT 1", run["id"])
            msg = json.loads(ev["json"]).get("message") if ev else "starting"
            print(f"current: run {run['id']} {run['kind']} {run['model_id']} since "
                  f"{run['started_at']}: {msg}")

        if len(up) > 1:
            print(f"engines: {' and '.join(up)} are both up; the runner waits until only one is")
        for name in up:
            try:
                state, units = self.current_plan(name)
            except NET_ERRORS as e:
                print(f"missing: ({name} unreachable: {e})")
                continue
            n_specs = sum(len(u.specs) for u in units)
            print(f"missing: {n_specs} results in {len(units)} runs on {name}"
                  + (f"; next: {units[0].describe()}" if units else ""))
        for name in self.clients:
            if name == "omlx":
                continue
            # One model at a time: say which models still need this engine.
            served = set()
            if name in up:
                served = {m.model_id for m in capture_state(self.clients[name], self.db).models}
            others = {m: n for m, n in missing_by_model(self.db, self.targets(), name).items()
                      if m not in served}
            if others:
                print(f"{name} still needs: " + ", ".join(f"{m} ({n} results)"
                                                         for m, n in others.items())
                      + " (switch models in its app)")

        print("\nrecent runs:")
        for r in self.db.q("SELECT r.*, (SELECT count(*) FROM perf_results WHERE run_id=r.id)"
                           " + (SELECT count(*) FROM accuracy_results WHERE run_id=r.id)"
                           " + (SELECT count(*) FROM context_results WHERE run_id=r.id)"
                           " + (SELECT count(*) FROM tool_cases WHERE run_id=r.id)"
                           " + (SELECT count(*) FROM agent_turns WHERE run_id=r.id)"
                           " + (SELECT count(*) FROM pulse_memory_cases WHERE run_id=r.id) AS n"
                           " FROM runs r ORDER BY id DESC LIMIT ?", args.recent):
            reason = f" ({r['cancel_reason']})" if r["cancel_reason"] else ""
            print(f"  #{r['id']:<4} {r['started_at']}  {r['engine']:<10} {r['kind']:<12} "
                  f"{r['status']}{reason:<15} {r['n']} results  {r['model_id']}")

    def cmd_plan(self, args) -> None:
        engine = args.engine or self.active_engine()
        if engine is None:
            raise SystemExit("no engine is up; use --engine to see what one still needs")
        if engine not in self.clients:
            raise SystemExit(f"{engine} is not configured (no API key)")
        if not engine_up(self.clients[engine]):
            others = missing_by_model(self.db, self.targets(), engine)
            print(f"{engine} is down; results still missing on it, by model:")
            for m, n in others.items():
                print(f"  {m}: {n}")
            return
        state, units = self.current_plan(engine)
        print(f"engine: {engine}")
        if not units:
            print("nothing missing")
        for i, u in enumerate(units, 1):
            print(f"{i:3}. {u.describe()}")
        if engine != "omlx":
            served = {m.model_id for m in state.models}
            others = {m: n for m, n in missing_by_model(self.db, self.targets(), engine).items()
                      if m not in served}
            if others:
                print(f"\nnot being served, still missing on {engine}:")
                for m, n in others.items():
                    print(f"  {m}: {n}")
        if args.failed:
            targets = self.targets()
            print("\nfailed (skipped until environment or settings change):")
            for _, m, s in iter_targets(targets, state.models, engine):
                if spec_state(self.db, m.model_id, s, m.settings_fingerprint, state.env.hash,
                              engine) == "failed":
                    print(f"  {m.model_id}: {s.label()}")

    def cmd_run(self, args) -> None:
        if args.dry_run:
            return self.cmd_plan(argparse.Namespace(failed=False, engine=None))
        if self.cfg.http_addr:
            from .web import parse_addr, serve_in_background

            serve_in_background(self.cfg.db_path, parse_addr(self.cfg.http_addr))
        Runner(self.clients, self.db, self.targets, args.quiet_minutes).loop(once=args.once)

    def cmd_pause(self, args) -> None:
        until = None
        if args.duration:
            until = datetime.now(timezone.utc) + _parse_duration(args.duration)
        set_pause(self.db, until, args.reason or "")
        print(f"paused {'until ' + until.astimezone().strftime('%Y-%m-%d %H:%M') if until else 'until resumed'}")
        if daemon_alive(self.db):
            # The runner notices within seconds and cancels safely (it waits
            # for any request of yours on the model under test to finish).
            print("the runner will cancel its current run")
            return
        running = self.db.q("SELECT * FROM runs WHERE status='running' AND source='runner'")
        for run in running:
            if run["kind"] in HARNESS_KINDS or self.client is None:
                # omlxbench's own requests: nothing server-side to cancel.
                self.db.finish_run(run["id"], status="cancelled", cancel_reason="paused")
                continue
            try:
                cancel_orphan(self.client, run["kind"], run["omlx_bench_id"], run["model_id"])
                print(f"cancelled run {run['id']} ({run['kind']} on {run['model_id']})")
            except NET_ERRORS as e:
                print(f"could not cancel run {run['id']}: {e}", file=sys.stderr)
            self.db.finish_run(run["id"], status="cancelled", cancel_reason="paused")

    def cmd_resume(self, args) -> None:
        clear_pause(self.db)
        print("resumed" + ("" if daemon_alive(self.db) else " (note: no runner is running; "
                                                             "start one with `omlxbench run`)"))

    def cmd_report(self, args) -> None:
        from .report import digest, markdown

        d = digest(self.db)
        print(json.dumps(d, indent=2) if args.json else markdown(d))

    def cmd_html(self, args) -> None:
        import webbrowser

        from .html_report import render
        from .report import digest

        now = datetime.now()
        path = args.output or self.cfg.db_path.parent / "reports" / f"report-{now:%Y%m%d-%H%M%S}.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render(digest(self.db), now), encoding="utf-8")
        print(path)
        if not args.no_open:
            webbrowser.open(path.resolve().as_uri())

    def cmd_serve(self, args) -> None:
        from .web import make_server, parse_addr

        addr = parse_addr(args.addr or self.cfg.http_addr or "127.0.0.1:8765")
        server = make_server(self.cfg.db_path, addr)
        print(f"report at http://{addr[0]}:{addr[1]}/ (ctrl-c to stop)")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass

    def cmd_import_ui(self, args) -> None:
        from .importer import import_ui

        n = import_ui(self.client, self.db, args.bench_id or [])
        print(f"imported {n} runs")

    def cmd_requeue(self, args) -> None:
        rows = self.db.q("SELECT s.*, e.engine_version FROM spec_status s"
                         " LEFT JOIN runs r ON r.id = s.last_run_id"
                         " LEFT JOIN environments e ON e.id = r.environment_id")
        cutoff = _version_tuple(args.before_version) if args.before_version else None
        chosen = [
            r for r in rows
            if fnmatch.fnmatchcase(r["model_id"], args.model)
            and fnmatch.fnmatchcase(r["engine"], args.engine)
            and fnmatch.fnmatchcase(r["spec_key"], args.spec)
            and (not args.failed or r["state"] == "failed")
            and (cutoff is None or _version_tuple(r["engine_version"]) < cutoff)
        ]
        for r in chosen:
            print(f"  {r['state']:<6} {r['engine']:<10} {r['model_id']}  {r['spec_key']}")
        if not args.yes:
            print(f"{len(chosen)} entries would be re-queued; add --yes to do it "
                  "(stored results are kept either way)")
            return
        with self.db.tx():
            for r in chosen:
                self.db.x("DELETE FROM spec_status WHERE model_id=? AND engine=? AND spec_key=?"
                          " AND settings_fingerprint=?",
                          r["model_id"], r["engine"], r["spec_key"], r["settings_fingerprint"])
        print(f"re-queued {len(chosen)}")

    def cmd_snapshot(self, args) -> None:
        engine = self.active_engine()
        if engine is None:
            raise SystemExit("no engine is up")
        state = capture_state(self.clients[engine], self.db)
        env = state.env
        print(f"environment {env.hash}: {engine} {env.engine_version}, {env.chip} "
              f"{env.chip_variant}, {env.memory_gb} GB, {env.gpu_cores} GPU cores, "
              f"macOS {env.macos_version}")
        targets = self.targets()
        for m in state.models:
            wanted = "*" if targets.wants_model(m.admin_model) else " "
            bits = f"{m.quant_bits:g}bit" if m.quant_bits else "?bit"
            print(f" {wanted} {m.model_id:<70} {m.model_type or '?':<4} {m.arch or '?':<14} "
                  f"{bits:<6} {(m.size_bytes or 0) / 1e9:5.1f} GB  [{m.settings_fingerprint}]")
        print("\n* = selected by targets.toml")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="omlxbench", description=__doc__)
    ap.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("status", help="server, runner, pause state and recent runs")
    p.add_argument("--recent", type=int, default=8)

    p = sub.add_parser("plan", help="list the runs needed to fill in missing results")
    p.add_argument("--failed", action="store_true", help="also list specs given up on")
    p.add_argument("--engine", choices=["omlx", "tensorfold"],
                   help="engine to plan for (default: the one that is up)")

    p = sub.add_parser("run", help="run the background loop")
    p.add_argument("--once", action="store_true", help="run at most one unit, then exit")
    p.add_argument("--dry-run", action="store_true", help="show the plan and exit")
    p.add_argument("--quiet-minutes", type=float, default=None,
                   help="override targets.toml [runner] quiet_minutes")

    p = sub.add_parser("pause", help="stop benchmarking now (cancels the current run)")
    p.add_argument("duration", nargs="?", help="e.g. 30m, 2h, 1d (default: until resumed)")
    p.add_argument("--reason", default="")

    sub.add_parser("resume", help="allow benchmarking again")

    p = sub.add_parser("report", help="latest results per model")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("html", help="write a sortable HTML report and open it in the browser")
    p.add_argument("-o", "--output", type=Path, help="file to write (default: "
                   "data/reports/report-<timestamp>.html)")
    p.add_argument("--no-open", action="store_true", help="don't open it in the browser")

    p = sub.add_parser("serve", help="serve the HTML report over HTTP, fresh on every load "
                       "(the runner already does this)")
    p.add_argument("addr", nargs="?", help="[host:]port (default: OMLXBENCH_HTTP or "
                   "127.0.0.1:8765)")

    p = sub.add_parser("import-ui", help="import results of runs started in the oMLX web UI")
    p.add_argument("--bench-id", action="append", help="perf bench id (default: found in logs)")

    p = sub.add_parser("requeue", help="forget results so they are measured again")
    p.add_argument("--model", default="*", help="model id glob")
    p.add_argument("--engine", default="*", help="engine glob (omlx, tensorfold)")
    p.add_argument("--spec", default="*", help="spec key glob, e.g. 'perf.single:*' or '*mmlu*'")
    p.add_argument("--failed", action="store_true", help="only specs that were given up on")
    p.add_argument("--before-version", help="only results measured on an older engine version")
    p.add_argument("--yes", action="store_true")

    sub.add_parser("snapshot", help="show the environment and models as recorded")

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    app = App()
    try:
        getattr(app, "cmd_" + args.cmd.replace("-", "_"))(args)
    except NET_ERRORS as e:
        raise SystemExit(f"server error: {e}")
