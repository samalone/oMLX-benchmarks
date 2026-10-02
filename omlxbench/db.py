"""SQLite storage.

Every table holding oMLX output keeps the verbatim JSON next to the extracted
columns, so fields oMLX adds later are never lost and can be queried with
SQLite's JSON functions. The schema is versioned with PRAGMA user_version;
append a script to MIGRATIONS to change it.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .snapshot import Environment, ModelSnapshot, accuracy_identity, canonical


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def dumps(obj: Any) -> str | None:
    return None if obj is None else canonical(obj)


_SCHEMA_V1 = """
CREATE TABLE environments (
    id INTEGER PRIMARY KEY,
    hash TEXT NOT NULL UNIQUE,
    omlx_version TEXT,
    engines_json TEXT,
    chip TEXT,
    chip_variant TEXT,
    memory_gb INTEGER,
    gpu_cores INTEGER,
    macos_version TEXT,
    perf_global_settings_json TEXT,
    global_settings_json TEXT,
    device_info_json TEXT,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL
);

CREATE TABLE model_snapshots (
    id INTEGER PRIMARY KEY,
    hash TEXT NOT NULL UNIQUE,
    model_id TEXT NOT NULL,
    model_path TEXT,
    source_repo_id TEXT,
    model_type TEXT,
    engine_type TEXT,
    arch TEXT,
    quant_bits REAL,
    quant_group_size INTEGER,
    quant_mode TEXT,
    size_bytes INTEGER,
    native_context INTEGER,
    settings_fingerprint TEXT NOT NULL,
    feature_flags_json TEXT,
    settings_json TEXT,
    config_json TEXT,
    admin_model_json TEXT,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL
);
CREATE INDEX model_snapshots_model ON model_snapshots(model_id);

CREATE TABLE runs (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,                -- perf | accuracy | context
    source TEXT NOT NULL DEFAULT 'runner',  -- runner | import
    omlx_bench_id TEXT,
    model_id TEXT NOT NULL,
    model_snapshot_id INTEGER REFERENCES model_snapshots(id),
    environment_id INTEGER REFERENCES environments(id),
    request_json TEXT,
    status TEXT NOT NULL,              -- running | completed | cancelled | error | interrupted
    cancel_reason TEXT,                -- paused | user_traffic | shutdown | ...
    error TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    final_json TEXT,                   -- oMLX's /results payload at the end
    upload_state_json TEXT             -- omlx.ai upload outcome, if any
);
CREATE INDEX runs_model ON runs(model_id);
CREATE UNIQUE INDEX runs_bench_id ON runs(kind, omlx_bench_id) WHERE omlx_bench_id IS NOT NULL;

CREATE TABLE run_specs (
    run_id INTEGER NOT NULL REFERENCES runs(id),
    spec_key TEXT NOT NULL,
    PRIMARY KEY (run_id, spec_key)
);

CREATE TABLE run_events (
    run_id INTEGER NOT NULL REFERENCES runs(id),
    seq INTEGER NOT NULL,
    ts TEXT NOT NULL,
    type TEXT,
    json TEXT NOT NULL,
    PRIMARY KEY (run_id, seq)
);

CREATE TABLE perf_results (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    spec_key TEXT NOT NULL,
    test_type TEXT NOT NULL,           -- single | batch
    pp INTEGER,
    tg INTEGER,
    batch_size INTEGER,
    context_profile TEXT,
    ttft_ms REAL,
    tpot_ms REAL,
    gen_tps REAL,                      -- single: tg tok/s; batch: aggregate tg_tps
    processing_tps REAL,               -- single: pp tok/s; batch: aggregate pp_tps
    avg_ttft_ms REAL,
    e2e_latency_s REAL,
    total_throughput REAL,
    peak_memory_bytes INTEGER,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    cached_tokens INTEGER,
    raw_json TEXT NOT NULL,            -- includes system_metrics
    recorded_at TEXT NOT NULL
);
CREATE INDEX perf_results_spec ON perf_results(spec_key);

CREATE TABLE accuracy_results (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    spec_key TEXT NOT NULL,
    suite TEXT NOT NULL,
    sample_size INTEGER,
    accuracy REAL,
    correct INTEGER,
    total INTEGER,
    dataset_total INTEGER,
    time_s REAL,
    thinking_used INTEGER,
    sampling_profile TEXT,
    finished_accuracy REAL,
    truncated_count INTEGER,
    category_scores_json TEXT,
    raw_json TEXT NOT NULL,            -- result minus question_results
    recorded_at TEXT NOT NULL
);
CREATE INDEX accuracy_results_spec ON accuracy_results(spec_key);

CREATE TABLE accuracy_questions (
    accuracy_result_id INTEGER NOT NULL REFERENCES accuracy_results(id),
    qid TEXT,
    category TEXT,
    correct INTEGER,
    expected TEXT,
    predicted TEXT,
    time_s REAL,
    finish_reason TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    question TEXT,
    raw_response TEXT
);
CREATE INDEX accuracy_questions_result ON accuracy_questions(accuracy_result_id);

CREATE TABLE context_results (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    spec_key TEXT NOT NULL,
    target_tokens INTEGER,
    native_context_length INTEGER,
    measured_tokens INTEGER,
    verified_tokens INTEGER,
    applied_tokens INTEGER,
    capped_by TEXT,
    prefill_tps REAL,
    duration_s REAL,
    restored_max_context_window INTEGER,
    raw_json TEXT NOT NULL,
    recorded_at TEXT NOT NULL
);

-- Planner bookkeeping: one row per (model, spec, settings) ever attempted.
CREATE TABLE spec_status (
    model_id TEXT NOT NULL,
    spec_key TEXT NOT NULL,
    settings_fingerprint TEXT NOT NULL,
    state TEXT NOT NULL,               -- ok | failed
    attempts INTEGER NOT NULL DEFAULT 0,
    failed_environment_hash TEXT,
    last_error TEXT,
    last_run_id INTEGER REFERENCES runs(id),
    updated_at TEXT NOT NULL,
    PRIMARY KEY (model_id, spec_key, settings_fingerprint)
);

CREATE TABLE control (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE usage_snapshots (
    id INTEGER PRIMARY KEY,
    captured_at TEXT NOT NULL,
    range TEXT NOT NULL,
    json TEXT NOT NULL
);

-- Convenience views for analysis. "latest" = newest row per
-- (model, spec, settings fingerprint), which is what the planner considers.
CREATE VIEW v_perf AS
SELECT r.model_id, m.settings_fingerprint, m.quant_bits, m.size_bytes, m.arch, m.model_type,
       e.omlx_version, e.chip, e.chip_variant, e.memory_gb,
       p.*, r.started_at AS run_started_at, r.status AS run_status
FROM perf_results p
JOIN runs r ON r.id = p.run_id
LEFT JOIN model_snapshots m ON m.id = r.model_snapshot_id
LEFT JOIN environments e ON e.id = r.environment_id;

CREATE VIEW v_latest_perf AS
SELECT * FROM v_perf v
WHERE v.id = (SELECT p2.id FROM v_perf p2
              WHERE p2.model_id = v.model_id AND p2.spec_key = v.spec_key
                AND p2.settings_fingerprint IS v.settings_fingerprint
              ORDER BY p2.recorded_at DESC, p2.id DESC LIMIT 1);

CREATE VIEW v_accuracy AS
SELECT r.model_id, m.settings_fingerprint, m.quant_bits, m.size_bytes, m.arch, m.model_type,
       e.omlx_version, a.*, r.started_at AS run_started_at
FROM accuracy_results a
JOIN runs r ON r.id = a.run_id
LEFT JOIN model_snapshots m ON m.id = r.model_snapshot_id
LEFT JOIN environments e ON e.id = r.environment_id;

CREATE VIEW v_latest_accuracy AS
SELECT * FROM v_accuracy v
WHERE v.id = (SELECT a2.id FROM v_accuracy a2
              WHERE a2.model_id = v.model_id AND a2.spec_key = v.spec_key
                AND a2.settings_fingerprint IS v.settings_fingerprint
              ORDER BY a2.recorded_at DESC, a2.id DESC LIMIT 1);

CREATE VIEW v_context AS
SELECT r.model_id, m.settings_fingerprint, m.size_bytes, e.omlx_version, c.*
FROM context_results c
JOIN runs r ON r.id = c.run_id
LEFT JOIN model_snapshots m ON m.id = r.model_snapshot_id
LEFT JOIN environments e ON e.id = r.environment_id;
"""


_SCHEMA_V2 = """
ALTER TABLE accuracy_results ADD COLUMN identity TEXT;
CREATE INDEX accuracy_results_identity ON accuracy_results(identity);
"""

_SCHEMA_V3 = """
-- Per-case results of harness suites (run by omlxbench itself through the
-- chat API, not by oMLX's built-in benchmarks). Cases are recorded one by
-- one so an interrupted suite resumes where it stopped.
CREATE TABLE tool_cases (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    model_id TEXT NOT NULL,
    spec_key TEXT NOT NULL,
    settings_fingerprint TEXT NOT NULL,
    category TEXT NOT NULL,
    case_id TEXT NOT NULL,
    correct INTEGER NOT NULL,
    reason TEXT,
    tool_calls_json TEXT,              -- parsed calls (BFCL names), NULL if unparseable
    content TEXT,                      -- assistant text, if any
    finish_reason TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    time_to_first_token REAL,          -- seconds, from oMLX usage
    total_time REAL,
    raw_json TEXT NOT NULL,            -- full response
    recorded_at TEXT NOT NULL,
    UNIQUE (model_id, spec_key, settings_fingerprint, case_id)
);

CREATE VIEW v_tools AS
SELECT t.model_id, t.settings_fingerprint, t.spec_key, t.category,
       count(*) AS n, sum(t.correct) AS correct, avg(t.correct) AS accuracy,
       avg(t.completion_tokens) AS avg_completion_tokens, avg(t.total_time) AS avg_time_s,
       max(t.recorded_at) AS recorded_at, m.quant_bits, m.size_bytes
FROM tool_cases t
LEFT JOIN model_snapshots m ON m.id = (SELECT max(id) FROM model_snapshots
                                       WHERE model_id = t.model_id)
GROUP BY t.model_id, t.settings_fingerprint, t.spec_key;
"""

# Each entry upgrades the schema by one version. Append; never edit old ones.
MIGRATIONS: list[str] = [_SCHEMA_V1, _SCHEMA_V2, _SCHEMA_V3]


class DB:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, timeout=30, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")  # safe with WAL; no fsync per commit
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    def _migrate(self) -> None:
        version = self.conn.execute("PRAGMA user_version").fetchone()[0]
        for i, script in enumerate(MIGRATIONS[version:], start=version + 1):
            # executescript() commits on its own, so the transaction lives in the script.
            self.conn.executescript(f"BEGIN;\n{script}\nPRAGMA user_version = {i};\nCOMMIT;")

    def tx(self):
        return _Tx(self.conn)

    def q(self, sql: str, *args: Any) -> list[sqlite3.Row]:
        return self.conn.execute(sql, args).fetchall()

    def q1(self, sql: str, *args: Any) -> sqlite3.Row | None:
        return self.conn.execute(sql, args).fetchone()

    def x(self, sql: str, *args: Any) -> int:
        return self.conn.execute(sql, args).lastrowid

    # -- control flags ----------------------------------------------------

    def get_control(self, key: str) -> str | None:
        row = self.q1("SELECT value FROM control WHERE key = ?", key)
        return row["value"] if row else None

    def set_control(self, key: str, value: str | None) -> None:
        if value is None:
            self.x("DELETE FROM control WHERE key = ?", key)
        else:
            self.x("INSERT INTO control(key, value) VALUES (?, ?) "
                   "ON CONFLICT(key) DO UPDATE SET value = excluded.value", key, value)

    # -- snapshots --------------------------------------------------------

    def upsert_environment(self, env: Environment) -> int:
        ts = now()
        row = self.q1("SELECT id FROM environments WHERE hash = ?", env.hash)
        if row:
            self.x("UPDATE environments SET last_seen = ? WHERE id = ?", ts, row["id"])
            return row["id"]
        return self.x(
            "INSERT INTO environments(hash, omlx_version, engines_json, chip, chip_variant,"
            " memory_gb, gpu_cores, macos_version, perf_global_settings_json,"
            " global_settings_json, device_info_json, first_seen, last_seen)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            env.hash, env.omlx_version, dumps(env.engines), env.chip, env.chip_variant,
            env.memory_gb, env.gpu_cores, env.macos_version, dumps(env.perf_global_settings),
            dumps(env.global_settings), dumps(env.device_info), ts, ts,
        )

    def upsert_model_snapshot(self, s: ModelSnapshot) -> int:
        ts = now()
        row = self.q1("SELECT id FROM model_snapshots WHERE hash = ?", s.hash)
        if row:
            self.x("UPDATE model_snapshots SET last_seen = ? WHERE id = ?", ts, row["id"])
            return row["id"]
        return self.x(
            "INSERT INTO model_snapshots(hash, model_id, model_path, source_repo_id, model_type,"
            " engine_type, arch, quant_bits, quant_group_size, quant_mode, size_bytes,"
            " native_context, settings_fingerprint, feature_flags_json, settings_json,"
            " config_json, admin_model_json, first_seen, last_seen)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            s.hash, s.model_id, s.model_path, s.source_repo_id, s.model_type, s.engine_type,
            s.arch, s.quant_bits, s.quant_group_size, s.quant_mode, s.size_bytes,
            s.native_context, s.settings_fingerprint, dumps(s.flags), dumps(s.settings),
            dumps(s.config), dumps(s.admin_model), ts, ts,
        )

    # -- runs -------------------------------------------------------------

    def create_run(self, *, kind: str, model_id: str, model_snapshot_id: int | None,
                   environment_id: int | None, request: dict, spec_keys: list[str],
                   omlx_bench_id: str | None = None, source: str = "runner") -> int:
        with self.tx():
            run_id = self.x(
                "INSERT INTO runs(kind, source, omlx_bench_id, model_id, model_snapshot_id,"
                " environment_id, request_json, status, started_at)"
                " VALUES (?,?,?,?,?,?,?,'running',?)",
                kind, source, omlx_bench_id, model_id, model_snapshot_id, environment_id,
                dumps(request), now(),
            )
            for key in spec_keys:
                self.x("INSERT OR IGNORE INTO run_specs(run_id, spec_key) VALUES (?, ?)",
                       run_id, key)
        return run_id

    def run_exists(self, omlx_bench_id: str) -> bool:
        return self.q1("SELECT 1 FROM runs WHERE omlx_bench_id = ?", omlx_bench_id) is not None

    def set_bench_id(self, run_id: int, bench_id: str) -> None:
        self.x("UPDATE runs SET omlx_bench_id = ? WHERE id = ?", bench_id, run_id)

    def add_events(self, run_id: int, events: list[tuple[int, dict]]) -> None:
        ts = now()
        self.conn.executemany(
            "INSERT OR IGNORE INTO run_events(run_id, seq, ts, type, json) VALUES (?,?,?,?,?)",
            [(run_id, seq, ts, ev.get("type"), dumps(ev)) for seq, ev in events])

    def finish_run(self, run_id: int, *, status: str, cancel_reason: str | None = None,
                   error: str | None = None, final: dict | None = None,
                   upload_state: dict | None = None) -> None:
        self.x(
            "UPDATE runs SET status = ?, cancel_reason = COALESCE(?, cancel_reason),"
            " error = COALESCE(?, error), ended_at = ?, final_json = COALESCE(?, final_json),"
            " upload_state_json = COALESCE(?, upload_state_json) WHERE id = ?",
            status, cancel_reason, error, now(), dumps(final), dumps(upload_state), run_id,
        )

    # -- results ----------------------------------------------------------

    def insert_perf_result(self, run_id: int, spec_key: str, r: dict,
                           context_profile: str | None) -> int:
        single = r.get("test_type") == "single"
        return self.x(
            "INSERT INTO perf_results(run_id, spec_key, test_type, pp, tg, batch_size,"
            " context_profile, ttft_ms, tpot_ms, gen_tps, processing_tps, avg_ttft_ms,"
            " e2e_latency_s, total_throughput, peak_memory_bytes, prompt_tokens,"
            " completion_tokens, cached_tokens, raw_json, recorded_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            run_id, spec_key, r.get("test_type"), r.get("pp"), r.get("tg"),
            1 if single else r.get("batch_size"), context_profile,
            r.get("ttft_ms"), r.get("tpot_ms"),
            r.get("gen_tps") if single else r.get("tg_tps"),
            r.get("processing_tps") if single else r.get("pp_tps"),
            r.get("avg_ttft_ms"), r.get("e2e_latency_s"), r.get("total_throughput"),
            r.get("peak_memory_bytes"), r.get("prompt_tokens"),
            r.get("completion_tokens") if single else r.get("total_gen_tokens"),
            r.get("cached_tokens"), dumps(r), now(),
        )

    def insert_accuracy_result(self, run_id: int, model_id: str, spec_key: str,
                               sample_size: int, r: dict) -> int:
        questions = r.get("question_results") or []
        summary = {k: v for k, v in r.items() if k != "question_results"}
        with self.tx():
            result_id = self.x(
                "INSERT INTO accuracy_results(run_id, spec_key, suite, sample_size, accuracy,"
                " correct, total, dataset_total, time_s, thinking_used, sampling_profile,"
                " finished_accuracy, truncated_count, category_scores_json, raw_json,"
                " recorded_at, identity) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                run_id, spec_key, r.get("benchmark"), sample_size, r.get("accuracy"),
                r.get("correct"), r.get("total"), r.get("dataset_total"), r.get("time_s"),
                int(bool(r.get("thinking_used"))), r.get("sampling_profile"),
                r.get("finished_accuracy"), r.get("truncated_count"),
                dumps(r.get("category_scores")), dumps(summary), now(),
                accuracy_identity(model_id, r),
            )
            self.conn.executemany(
                "INSERT INTO accuracy_questions(accuracy_result_id, qid, category, correct,"
                " expected, predicted, time_s, finish_reason, prompt_tokens, completion_tokens,"
                " question, raw_response) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (result_id, str(q.get("id")), q.get("category"),
                     None if q.get("correct") is None else int(bool(q.get("correct"))),
                     _text(q.get("expected")), _text(q.get("predicted")), q.get("time_s"),
                     q.get("finish_reason"), q.get("prompt_tokens"),
                     q.get("completion_tokens"), _text(q.get("question")),
                     _text(q.get("raw_response")))
                    for q in questions
                ],
            )
        return result_id

    def insert_context_result(self, run_id: int, spec_key: str, r: dict,
                              restored: int | None) -> int:
        return self.x(
            "INSERT INTO context_results(run_id, spec_key, target_tokens, native_context_length,"
            " measured_tokens, verified_tokens, applied_tokens, capped_by, prefill_tps,"
            " duration_s, restored_max_context_window, raw_json, recorded_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            run_id, spec_key, r.get("target_tokens"), r.get("native_context_length"),
            r.get("measured_tokens"), r.get("verified_tokens"), r.get("applied_tokens"),
            r.get("capped_by"), r.get("prefill_tps"), r.get("duration_s"), restored,
            dumps(r), now(),
        )

    # -- planner bookkeeping ---------------------------------------------

    def mark_spec(self, model_id: str, spec_key: str, fingerprint: str, *, ok: bool,
                  run_id: int | None, error: str | None = None,
                  environment_hash: str | None = None) -> None:
        ts = now()
        if ok:
            self.x(
                "INSERT INTO spec_status(model_id, spec_key, settings_fingerprint, state,"
                " attempts, last_run_id, updated_at) VALUES (?,?,?,'ok',1,?,?)"
                " ON CONFLICT DO UPDATE SET state='ok', attempts=attempts+1,"
                " last_error=NULL, failed_environment_hash=NULL, last_run_id=excluded.last_run_id,"
                " updated_at=excluded.updated_at",
                model_id, spec_key, fingerprint, run_id, ts,
            )
            return
        # A failure under a new environment restarts the attempt count, so a
        # spec that ran out of memory gets retried after, e.g., an oMLX upgrade.
        self.x(
            "INSERT INTO spec_status(model_id, spec_key, settings_fingerprint, state, attempts,"
            " failed_environment_hash, last_error, last_run_id, updated_at)"
            " VALUES (?,?,?,'failed',1,?,?,?,?)"
            " ON CONFLICT DO UPDATE SET"
            " attempts = CASE WHEN spec_status.state='failed' AND"
            "   spec_status.failed_environment_hash IS excluded.failed_environment_hash"
            "   THEN spec_status.attempts+1 ELSE 1 END,"
            " state = CASE WHEN spec_status.state='ok' THEN 'ok' ELSE 'failed' END,"
            " failed_environment_hash=excluded.failed_environment_hash,"
            " last_error=excluded.last_error, last_run_id=excluded.last_run_id,"
            " updated_at=excluded.updated_at",
            model_id, spec_key, fingerprint, environment_hash, error, run_id, ts,
        )


def _text(value: Any) -> str | None:
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value, default=str)


class _Tx:
    """BEGIN/COMMIT around a block; nested use is a no-op."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.outer = False

    def __enter__(self):
        if not self.conn.in_transaction:
            self.conn.execute("BEGIN IMMEDIATE")
            self.outer = True
        return self.conn

    def __exit__(self, exc_type, exc, tb):
        if self.outer:
            self.conn.execute("ROLLBACK" if exc_type else "COMMIT")
        return False
