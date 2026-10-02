"""Import results of benchmarks started from the oMLX web UI.

oMLX holds results in memory only, so this has to run before the server
restarts. Imported results are attributed to the model's *current* snapshot,
which is right unless the model's settings changed since the run.

- Performance: oMLX keeps the last 10 runs but has no endpoint listing them,
  so bench ids are found in the server log and the stream is replayed (it
  carries the model id, which the /results payload lacks).
- Intelligence: the accumulated results list (every suite since the last
  reset/restart). The original request isn't kept, so batch size is assumed
  to be 1 and "enable thinking" is taken from whether thinking was used.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from .client import OmlxClient, OmlxError
from .db import DB
from .snapshot import capture_environment, digest, snapshot_model
from .specs import accuracy, perf_batch, perf_single

log = logging.getLogger("omlxbench")

_BENCH_ID = re.compile(r"\bbench-[0-9a-f]{12}\b")
LOG_DIR = Path.home() / ".omlx" / "logs"


def _logged_bench_ids() -> list[str]:
    ids: list[str] = []
    for path in sorted(LOG_DIR.glob("server.log*")):
        try:
            ids += _BENCH_ID.findall(path.read_text(errors="replace"))
        except OSError:
            continue
    return list(dict.fromkeys(ids))


def import_ui(client: OmlxClient, db: DB, extra_ids: list[str] = ()) -> int:
    env = capture_environment(client)
    env_id = db.upsert_environment(env)
    snaps = {m.model_id: m for m in (snapshot_model(e) for e in client.admin_models())}
    snap_ids = {mid: db.upsert_model_snapshot(s) for mid, s in snaps.items()}
    imported = 0

    for bench_id in dict.fromkeys([*extra_ids, *_logged_bench_ids()]):
        if db.q1("SELECT 1 FROM runs WHERE kind='perf' AND omlx_bench_id=?", bench_id):
            continue
        try:
            final = client.perf_results(bench_id)
        except OmlxError as e:
            if e.status != 404:
                log.warning("%s: %s", bench_id, e)
            continue  # no longer held by the server
        if final.get("status") == "running" or not final.get("results"):
            continue
        events = list(client.iter_events(client.stream_path("perf", bench_id)))
        model_id = _perf_model_id(events, snaps)
        if model_id is None:
            log.warning("%s: could not determine the model; skipped", bench_id)
            continue
        snap = snaps[model_id]
        profile = final.get("context_profile") or "code_python"
        results = final["results"]
        specs = [
            (perf_single(r.get("pp"), r.get("tg", 128), profile) if r.get("test_type") == "single"
             else perf_batch(r.get("batch_size"), r.get("tg", 128), profile), r)
            for r in results
        ]
        run_id = db.create_run(
            kind="perf", source="import", omlx_bench_id=bench_id, model_id=model_id,
            model_snapshot_id=snap_ids[model_id], environment_id=env_id,
            request={"imported": True, "context_profile": profile},
            spec_keys=[s.key for s, _ in specs],
        )
        for seq, ev in enumerate(events, 1):
            db.add_event(run_id, seq, ev)
        for spec, r in specs:
            db.insert_perf_result(run_id, spec.key, r, profile)
            db.mark_spec(model_id, spec.key, snap.settings_fingerprint, ok=True, run_id=run_id)
        db.finish_run(run_id, status=final.get("status", "completed"), final=final,
                      upload_state=final.get("upload_state"))
        log.info("imported perf %s (%s, %d results)", bench_id, model_id, len(results))
        imported += 1

    for r in client.accuracy_results().get("results", []):
        model_id = r.get("model_id")
        if r.get("external") or model_id not in snaps:
            continue
        pseudo_id = "import-" + digest([model_id, r.get("benchmark"), r.get("total"),
                                        r.get("correct"), r.get("time_s")])
        if db.q1("SELECT 1 FROM runs WHERE kind='accuracy' AND omlx_bench_id=?", pseudo_id):
            continue
        # Suites the runner itself recorded are in oMLX's list too.
        if db.q1("SELECT 1 FROM accuracy_results a JOIN runs r ON r.id = a.run_id"
                 " WHERE r.model_id=? AND a.suite=? AND a.total IS ? AND a.correct IS ?"
                 " AND a.time_s IS ?", model_id, r.get("benchmark"), r.get("total"),
                 r.get("correct"), r.get("time_s")):
            continue
        total, dataset_total = r.get("total") or 0, r.get("dataset_total") or 0
        sample_size = 0 if dataset_total and total >= dataset_total else total
        spec = accuracy(r["benchmark"], sample_size, bool(r.get("thinking_used")),
                        r.get("sampling_profile") or "deterministic", 1)
        run_id = db.create_run(
            kind="accuracy", source="import", omlx_bench_id=pseudo_id, model_id=model_id,
            model_snapshot_id=snap_ids[model_id], environment_id=env_id,
            request={"imported": True}, spec_keys=[spec.key],
        )
        db.insert_accuracy_result(run_id, spec.key, sample_size, r)
        db.mark_spec(model_id, spec.key, snaps[model_id].settings_fingerprint, ok=True,
                     run_id=run_id)
        db.finish_run(run_id, status="completed", upload_state=r.get("upload"))
        log.info("imported accuracy %s on %s", r["benchmark"], model_id)
        imported += 1
    return imported


def _perf_model_id(events: list[dict], snaps: dict) -> str | None:
    for ev in events:
        if ev.get("type") == "done":
            mid = (ev.get("summary") or {}).get("model_id")
            if mid in snaps:
                return mid
    # Cancelled runs have no "done": look for the model named in the load
    # phase's progress text (the unload phase names *other* models).
    for ev in events:
        if ev.get("phase") != "load":
            continue
        msg = ev.get("message") or ""
        for mid in sorted(snaps, key=len, reverse=True):
            if mid in msg:
                return mid
    return None
