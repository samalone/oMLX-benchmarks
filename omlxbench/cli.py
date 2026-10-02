"""omlxbench command line."""

from __future__ import annotations

import argparse
import fnmatch
import json
import logging
import re
import sys
from datetime import datetime, timedelta, timezone

from .client import NET_ERRORS, OmlxClient
from .config import load_config
from .db import DB
from .planner import iter_targets, plan, spec_state
from .control import clear_pause, daemon_alive, pause_state, set_pause
from .runner import Runner, cancel_orphan
from .snapshot import capture_environment, snapshot_model
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
        self.client = OmlxClient(self.cfg.base_url, self.cfg.api_key)

    def targets(self):
        return load_targets(self.cfg.targets_path)

    def current_plan(self):
        env = capture_environment(self.client)
        models = [snapshot_model(e) for e in self.client.admin_models()]
        return env, models, plan(self.db, self.targets(), models, env.hash)

    # -- commands -----------------------------------------------------------

    def cmd_status(self, args) -> None:
        try:
            st = self.client.status()
            print(f"server:  oMLX {st.get('version')}, {st.get('models_discovered')} models, "
                  f"loaded: {', '.join(st.get('loaded_models') or []) or 'none'}")
        except NET_ERRORS as e:
            print(f"server:  unreachable ({e})")
            st = None
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

        if st is not None:
            env, models, units = self.current_plan()
            n_specs = sum(len(u.specs) for u in units)
            print(f"missing: {n_specs} results in {len(units)} runs"
                  + (f"; next: {units[0].describe()}" if units else ""))

        print("\nrecent runs:")
        for r in self.db.q("SELECT r.*, (SELECT count(*) FROM perf_results WHERE run_id=r.id)"
                           " + (SELECT count(*) FROM accuracy_results WHERE run_id=r.id)"
                           " + (SELECT count(*) FROM context_results WHERE run_id=r.id) AS n"
                           " FROM runs r ORDER BY id DESC LIMIT ?", args.recent):
            reason = f" ({r['cancel_reason']})" if r["cancel_reason"] else ""
            print(f"  #{r['id']:<4} {r['started_at']}  {r['kind']:<8} {r['status']}{reason:<15} "
                  f"{r['n']} results  {r['model_id']}")

    def cmd_plan(self, args) -> None:
        env, models, units = self.current_plan()
        if not units:
            print("nothing missing")
        for i, u in enumerate(units, 1):
            print(f"{i:3}. {u.describe()}")
        if args.failed:
            targets = self.targets()
            print("\nfailed (skipped until environment or settings change):")
            for _, m, s in iter_targets(targets, models):
                if spec_state(self.db, m.model_id, s, m.settings_fingerprint, env.hash) == "failed":
                    print(f"  {m.model_id}: {s.label()}")

    def cmd_run(self, args) -> None:
        if args.dry_run:
            return self.cmd_plan(argparse.Namespace(failed=False))
        Runner(self.client, self.db, self.targets, args.quiet_minutes).loop(once=args.once)

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

    def cmd_import_ui(self, args) -> None:
        from .importer import import_ui

        n = import_ui(self.client, self.db, args.bench_id or [])
        print(f"imported {n} runs")

    def cmd_requeue(self, args) -> None:
        rows = self.db.q("SELECT s.*, e.omlx_version FROM spec_status s"
                         " LEFT JOIN runs r ON r.id = s.last_run_id"
                         " LEFT JOIN environments e ON e.id = r.environment_id")
        cutoff = _version_tuple(args.before_version) if args.before_version else None
        chosen = [
            r for r in rows
            if fnmatch.fnmatchcase(r["model_id"], args.model)
            and fnmatch.fnmatchcase(r["spec_key"], args.spec)
            and (not args.failed or r["state"] == "failed")
            and (cutoff is None or _version_tuple(r["omlx_version"]) < cutoff)
        ]
        for r in chosen:
            print(f"  {r['state']:<6} {r['model_id']}  {r['spec_key']}")
        if not args.yes:
            print(f"{len(chosen)} entries would be re-queued; add --yes to do it "
                  "(stored results are kept either way)")
            return
        with self.db.tx():
            for r in chosen:
                self.db.x("DELETE FROM spec_status WHERE model_id=? AND spec_key=?"
                          " AND settings_fingerprint=?",
                          r["model_id"], r["spec_key"], r["settings_fingerprint"])
        print(f"re-queued {len(chosen)}")

    def cmd_snapshot(self, args) -> None:
        env = capture_environment(self.client)
        print(f"environment {env.hash}: oMLX {env.omlx_version}, {env.chip} {env.chip_variant}, "
              f"{env.memory_gb} GB, {env.gpu_cores} GPU cores, macOS {env.macos_version}")
        targets = self.targets()
        for e in self.client.admin_models():
            m = snapshot_model(e)
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

    p = sub.add_parser("import-ui", help="import results of runs started in the oMLX web UI")
    p.add_argument("--bench-id", action="append", help="perf bench id (default: found in logs)")

    p = sub.add_parser("requeue", help="forget results so they are measured again")
    p.add_argument("--model", default="*", help="model id glob")
    p.add_argument("--spec", default="*", help="spec key glob, e.g. 'perf.single:*' or '*mmlu*'")
    p.add_argument("--failed", action="store_true", help="only specs that were given up on")
    p.add_argument("--before-version", help="only results measured on an older oMLX")
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
        raise SystemExit(f"oMLX error: {e}")
