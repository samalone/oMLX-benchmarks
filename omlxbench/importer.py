"""Import results of benchmarks started from the oMLX web UI.

oMLX holds results in memory only, so this has to run before the server
restarts. Imported results are attributed to the model's *current* snapshot,
which is right unless the model's settings changed since the run.

- Performance: oMLX keeps the last 10 runs but has no endpoint listing them,
  so bench ids are found in the server log and the stream is replayed (it
  carries the model id, which the /results payload lacks). The force-LM-engine
  option isn't reported, so it is assumed off.
- Intelligence: the accumulated results list (every suite since the last
  reset/restart). The original request isn't kept, so batch size is assumed
  to be 1 and "enable thinking" is taken from whether thinking was used.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .client import OmlxClient, OmlxError
from .db import DB
from .snapshot import State, accuracy_identity, capture_state
from .specs import accuracy, perf_spec_from_result

log = logging.getLogger("omlxbench")

_BENCH_ID = re.compile(r"\bbench-[0-9a-f]{12}\b")
LOG_DIR = Path.home() / ".omlx" / "logs"


@dataclass
class ImportState:
    """What earlier imports in this process already looked at."""

    log_offsets: dict[str, tuple[int, int]] = field(default_factory=dict)  # path -> (inode, offset)
    bench_ids: list[str] = field(default_factory=list)  # every id seen in the logs, in order
    gone: set[str] = field(default_factory=set)  # ids the server no longer holds

    def scan_logs(self) -> None:
        """Pick up bench ids from new log bytes only (logs only grow or rotate)."""
        for path in sorted(LOG_DIR.glob("server.log*")):
            try:
                st = path.stat()
                inode, offset = self.log_offsets.get(str(path), (st.st_ino, 0))
                if inode != st.st_ino or st.st_size < offset:
                    offset = 0  # rotated or truncated
                if st.st_size == offset:
                    continue
                with path.open("rb") as f:
                    f.seek(offset)
                    text = f.read().decode(errors="replace")
                self.log_offsets[str(path)] = (st.st_ino, st.st_size)
            except OSError:
                continue
            for bench_id in _BENCH_ID.findall(text):
                if bench_id not in self.bench_ids:
                    self.bench_ids.append(bench_id)


def import_ui(client: OmlxClient, db: DB, extra_ids: list[str] = (),
              state: ImportState | None = None) -> int:
    state = state or ImportState()
    state.scan_logs()
    recorded: State | None = None  # captured lazily: most imports find nothing new

    def snapshots() -> State:
        nonlocal recorded
        recorded = recorded or capture_state(client, db)
        return recorded

    imported = 0
    for bench_id in dict.fromkeys([*extra_ids, *state.bench_ids]):
        if bench_id in state.gone or db.run_exists(bench_id):
            continue
        try:
            final = client.perf_results(bench_id)
        except OmlxError as e:
            if e.status == 404:
                state.gone.add(bench_id)
            else:
                log.warning("%s: %s", bench_id, e)
            continue
        if final.get("status") == "running" or not final.get("results"):
            continue
        events = list(client.iter_events(client.stream_path("perf", bench_id)))
        st = snapshots()
        model_id = _perf_model_id(events, {m.model_id for m in st.models})
        if model_id is None:
            log.warning("%s: could not determine the model; skipped", bench_id)
            state.gone.add(bench_id)
            continue
        snap = next(m for m in st.models if m.model_id == model_id)
        profile = final.get("context_profile") or "code_python"
        rows = [(perf_spec_from_result(r, tg=r.get("tg", 128), context_profile=profile), r)
                for r in final["results"]]
        rows = [(spec, r) for spec, r in rows if spec is not None]
        with db.tx():
            run_id = db.create_run(
                kind="perf", source="import", omlx_bench_id=bench_id, model_id=model_id,
                model_snapshot_id=st.snapshot_ids[model_id], environment_id=st.env_id,
                request={"imported": True, "context_profile": profile},
                spec_keys=[spec.key for spec, _ in rows],
            )
            db.add_events(run_id, list(enumerate(events, 1)))
            for spec, r in rows:
                db.insert_perf_result(run_id, spec.key, r, profile)
                db.mark_spec(model_id, spec.key, snap.settings_fingerprint, ok=True, run_id=run_id)
            db.finish_run(run_id, status=final.get("status", "completed"), final=final,
                          upload_state=final.get("upload_state"))
        log.info("imported perf %s (%s, %d results)", bench_id, model_id, len(rows))
        imported += 1

    for r in client.accuracy_results().get("results", []):
        model_id = r.get("model_id")
        if r.get("external") or not model_id:
            continue
        identity = accuracy_identity(model_id, r)
        if db.q1("SELECT 1 FROM accuracy_results WHERE identity = ?", identity):
            continue  # recorded by the runner, or imported before
        st = snapshots()
        snap = next((m for m in st.models if m.model_id == model_id), None)
        if snap is None:
            continue
        total, dataset_total = r.get("total") or 0, r.get("dataset_total") or 0
        sample_size = 0 if dataset_total and total >= dataset_total else total
        spec = accuracy(r["benchmark"], sample_size, bool(r.get("thinking_used")),
                        r.get("sampling_profile") or "deterministic", 1)
        with db.tx():
            run_id = db.create_run(
                kind="accuracy", source="import", omlx_bench_id=f"import-{identity}",
                model_id=model_id, model_snapshot_id=st.snapshot_ids[model_id],
                environment_id=st.env_id, request={"imported": True}, spec_keys=[spec.key],
            )
            db.insert_accuracy_result(run_id, model_id, spec.key, sample_size, r)
            db.mark_spec(model_id, spec.key, snap.settings_fingerprint, ok=True, run_id=run_id)
            db.finish_run(run_id, status="completed", upload_state=r.get("upload"))
        log.info("imported accuracy %s on %s", r["benchmark"], model_id)
        imported += 1
    return imported


def _perf_model_id(events: list[dict], model_ids: set[str]) -> str | None:
    for ev in events:
        if ev.get("type") == "done":
            mid = (ev.get("summary") or {}).get("model_id")
            if mid in model_ids:
                return mid
    # Cancelled runs have no "done": look for the model named in the load
    # phase's progress text (the unload phase names *other* models).
    for ev in events:
        if ev.get("phase") != "load":
            continue
        msg = ev.get("message") or ""
        for mid in sorted(model_ids, key=len, reverse=True):
            if mid in msg:
                return mid
    return None
