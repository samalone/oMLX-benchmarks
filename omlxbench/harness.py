"""Suites omlxbench runs itself through the chat API (not oMLX built-ins):
BFCL tool calling, the Hermes-shaped agent-turns replay, Pulse memory and the
streamed speed test. They run on any engine (oMLX or TensorFold).

Unlike the built-in benchmarks, these own their requests, so yielding is
clean: dropping our connection makes oMLX abort just our request, and the
model stays loaded. No user request can be stranded, so the harness yields
immediately rather than waiting.

Every case is recorded as soon as it's scored, and a suite resumes from the
cases still missing, so a cancel loses at most the case in flight.

Telling our traffic from the user's: our requests also count in
/api/status `total_requests` once they complete, so the harness counts its
own completions and treats anything beyond those as the user. Likewise any
request in flight beyond our one.
"""

from __future__ import annotations

import json
import logging
import threading
import time

import httpx

from . import bfcl
from .client import NET_ERRORS, OmlxClient, RejectedRequests, in_flight
from .control import pause_state
from .db import DB, dumps, now
from .planner import WorkUnit

log = logging.getLogger("omlxbench")

POLL_SECONDS = 2.0
REQUEST_TIMEOUT_SECONDS = 900.0
MAX_CONSECUTIVE_ERRORS = 3
# Responses meaning oMLX could not serve the model at all (unknown model, a
# failed load, no memory to load it), as opposed to rejecting one request.
# Scoring these as wrong answers would record a model that never ran.
UNSERVABLE_STATUSES = (404, 409, 503, 507)
# oMLX applies a model's saved sampling settings when a request sends none
# (as Hermes's requests don't). Other engines get the same values sent
# explicitly, so engines are compared at the same sampling.
SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "presence_penalty")


def omlx_sampling(db: DB, model_id: str) -> dict:
    """The model's saved sampling settings in its newest oMLX snapshot."""
    row = db.q1("SELECT settings_json FROM model_snapshots WHERE model_id = ? AND engine = 'omlx'"
                " ORDER BY id DESC LIMIT 1", model_id)
    settings = json.loads(row["settings_json"]) if row and row["settings_json"] else {}
    return {k: settings[k] for k in SAMPLING_KEYS if settings.get(k) is not None}


def server_timing(data: dict) -> tuple[float | None, float | None]:
    """(time to first token, total time) as the server reports them: oMLX in
    `usage`, TensorFold in its own `tensorfold` block."""
    u = data.get("usage") or {}
    tf = data.get("tensorfold") or {}
    return (u.get("time_to_first_token") or tf.get("time_to_first_token"),
            u.get("total_time") or tf.get("seconds"))


class _Aborted(Exception):
    pass


class _Harness:
    """Shared plumbing: yielding, and requests that can be abandoned."""

    def __init__(self, client: OmlxClient, db: DB, unit: WorkUnit, env_hash: str, run_id: int):
        self.client = client
        self.db = db
        self.unit = unit
        self.model = unit.model
        self.spec = unit.specs[0]
        self.engine = unit.engine
        self.env_hash = env_hash
        self.run_id = run_id
        self.cancel_reason: str | None = None
        self.baseline = 0
        self.ours_completed = 0
        self.sampling = {} if self.engine == "omlx" else omlx_sampling(db, self.model.model_id)
        if self.engine == "omlx":
            self.rejected = RejectedRequests()
        else:
            from .engines import NoRejections

            self.rejected = NoRejections()
        self.rejected.new()  # start counting from here
        self.rejected_total = 0
        self.ours_rejected = 0  # our own non-2xx responses also appear in the log

    # -- requests -----------------------------------------------------------

    def adapt(self, body: dict) -> dict:
        """The request as this engine needs it: its model name, and for
        engines other than oMLX the thinking switch where they read it and
        oMLX's sampling for the model sent explicitly."""
        body = {**body, "model": self.model.request_model}
        if self.engine != "omlx":
            body = {**self.sampling, **body}
            if "enable_thinking" in body:
                body["chat_template_kwargs"] = {
                    **(body.get("chat_template_kwargs") or {}),
                    "enable_thinking": bool(body["enable_thinking"])}
        return body

    def mark(self, ok: bool, error: str | None = None, spec=None) -> None:
        spec = spec or self.spec
        self.db.mark_spec(self.model.model_id, spec.key, self.model.settings_fingerprint,
                          ok=ok, run_id=self.run_id, error=None if ok else error,
                          environment_hash=None if ok else self.env_hash, engine=self.engine)

    # -- yielding -----------------------------------------------------------

    def rebaseline(self) -> None:
        self.baseline = (self.client.status().get("total_requests") or 0) - self.ours_completed

    def should_yield(self, ours_in_flight: int) -> str | None:
        reason, why = self._yield_signal(ours_in_flight)
        if reason:
            log.info("  yield signal: %s", why)
        return reason

    def _yield_signal(self, ours_in_flight: int) -> tuple[str | None, str]:
        if pause_state(self.db)[0]:
            return "paused", "pause requested"
        try:
            status = self.client.status()
        except NET_ERRORS:
            return None, ""
        loaded = status.get("loaded_models") or []
        if self.engine != "omlx" and self.model.request_model not in loaded:
            # TensorFold serves one model; the user switched it in the app.
            return "model_switched", f"{self.engine} now serves {', '.join(loaded) or 'nothing'}"
        completed = (status.get("total_requests") or 0) - self.baseline
        if completed > self.ours_completed:
            return "user_traffic", f"{completed} API requests completed, {self.ours_completed} ours"
        self.rejected_total += self.rejected.new()
        if self.rejected_total > self.ours_rejected:
            return "user_traffic", (f"{self.rejected_total} API requests rejected,"
                                    f" {self.ours_rejected} ours")
        if in_flight(status) > ours_in_flight:
            return "user_traffic", f"{in_flight(status)} requests in flight, {ours_in_flight} ours"
        if status.get("models_loading") and self.model.model_id in loaded:
            return "user_traffic", "another model is loading"  # for someone else
        return None, ""

    # -- one request ----------------------------------------------------------

    def chat(self, body: dict) -> httpx.Response:
        """POST a chat completion (non-streaming)."""
        body = self.adapt(body)
        return self._abandonable(lambda http: http.post("/v1/chat/completions", json=body))

    def chat_first_token(self, body: dict) -> dict:
        """Stream a chat completion and time its first token client-side.

        Returns {"ttft": seconds or None, "t_last": seconds to the last
        token, "elapsed", "usage": dict or None, "server": the engine's own
        timing block or None, "finish_reason", "error": str or None}.
        Streaming keeps the timing even when oMLX ends the response with an
        error (e.g. `incomplete_tool_call` when a reply is cut off inside a
        tool call), which a non-streaming request loses.
        """
        body = {**self.adapt(body), "stream": True, "stream_options": {"include_usage": True}}

        def stream(http: httpx.Client) -> dict:
            out: dict = {"ttft": None, "t_last": None, "usage": None, "server": None,
                         "finish_reason": None, "error": None}
            t0 = time.monotonic()
            with http.stream("POST", "/v1/chat/completions", json=body) as resp:
                if resp.status_code >= 400:
                    resp.read()
                    out["error"] = f"HTTP {resp.status_code}: {resp.text[:300]}"
                    return out
                for line in resp.iter_lines():
                    if not line.startswith("data:") or line.strip() == "data: [DONE]":
                        continue
                    chunk = json.loads(line[5:])
                    if "error" in chunk:
                        out["error"] = json.dumps(chunk["error"])[:300]
                    out["usage"] = chunk.get("usage") or out["usage"]
                    out["server"] = chunk.get("tensorfold") or out["server"]
                    choice = (chunk.get("choices") or [{}])[0]
                    out["finish_reason"] = choice.get("finish_reason") or out["finish_reason"]
                    delta = choice.get("delta") or {}
                    if any(delta.get(k) for k in ("content", "reasoning_content", "reasoning",
                                                  "tool_calls")):
                        t = time.monotonic() - t0
                        if out["ttft"] is None:
                            out["ttft"] = t
                        out["t_last"] = t
            out["elapsed"] = time.monotonic() - t0
            return out

        return self._abandonable(stream)

    def _abandonable(self, request):
        """Run `request(http)` on a worker thread, polling for a reason to
        yield meanwhile. The worker has its own HTTP client; closing it drops
        the connection, which oMLX treats as a cancel of just our request."""
        http = httpx.Client(base_url=self.client.base_url, timeout=REQUEST_TIMEOUT_SECONDS,
                            headers={"Authorization": f"Bearer {self.client.api_key}"})
        box: dict = {}

        def worker():
            try:
                box["result"] = request(http)
            except Exception as e:  # noqa: BLE001 - surfaced below
                box["exc"] = e

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        try:
            while t.is_alive():
                t.join(POLL_SECONDS)
                if t.is_alive() and (reason := self.should_yield(ours_in_flight=1)):
                    self.cancel_reason = reason
                    raise _Aborted()
        finally:
            http.close()
        if "exc" in box:
            raise box["exc"]
        return box["result"]

    def finish(self, status: str, error: str | None, recorded: int, total: int) -> None:
        self.db.finish_run(self.run_id, status=status, cancel_reason=self.cancel_reason,
                           error=error)
        log.info("run %s %s%s (%d/%d recorded)", self.run_id, status,
                 f" ({self.cancel_reason})" if self.cancel_reason else "", recorded, total)


class _CaseSuite(_Harness):
    """A suite of independent chat requests, each recorded as it's scored,
    so a cancelled run resumes from the cases still missing."""

    what = "cases"

    def cases(self) -> list[tuple[str, object]]:
        """(key, case) for every case in the spec."""
        raise NotImplementedError

    def done(self) -> set[str]:
        """Keys of cases already recorded for this model, spec and settings."""
        raise NotImplementedError

    def body(self, case) -> dict:
        raise NotImplementedError

    def record(self, case, resp: httpx.Response, elapsed: float) -> None:
        raise NotImplementedError

    def tally(self) -> tuple[int, int]:
        """(recorded, passed)."""
        raise NotImplementedError

    def run(self) -> None:
        cases = self.cases()
        done = self.done()
        pending = [(k, c) for k, c in cases if k not in done]
        log.info("  %d/%d %s to go", len(pending), len(cases), self.what)
        status, error, errors = "completed", None, 0
        try:
            self.rebaseline()
            if pending:
                # Load the model before the first case, and stop here if it can't be.
                resp = self.chat({"model": self.model.model_id, "max_tokens": 1,
                                  "messages": [{"role": "user", "content": "Reply with OK."}]})
                if not resp.is_success:
                    self.ours_rejected += 1
                    raise OmlxStepError(f"warm-up: HTTP {resp.status_code}: {resp.text[:300]}")
                self.ours_completed += 1
            for i, (key, case) in enumerate(pending, 1):
                if reason := self.should_yield(ours_in_flight=0):
                    self.cancel_reason = reason
                    raise _Aborted()
                t0 = time.monotonic()
                try:
                    resp = self.chat(self.body(case))
                except NET_ERRORS as e:
                    errors += 1
                    log.warning("  %s: request failed: %s", key, e)
                    if errors >= MAX_CONSECUTIVE_ERRORS:
                        raise
                    self.rebaseline()
                    continue
                errors = 0
                if resp.status_code in UNSERVABLE_STATUSES:
                    self.ours_rejected += 1
                    raise OmlxStepError(f"{key}: HTTP {resp.status_code}: {resp.text[:300]}")
                self.record(case, resp, time.monotonic() - t0)
                if resp.is_success:
                    self.ours_completed += 1
                else:
                    self.ours_rejected += 1
                    self.rebaseline()  # unclear whether oMLX counted a rejected request
                if i % 10 == 0:
                    log.info("  %s: %d/%d done", self.what, len(done) + i, len(cases))
        except _Aborted:
            status = "cancelled"
            log.info("yielding the server (%s)", self.cancel_reason)
        except (OmlxStepError, *NET_ERRORS) as e:
            status, error = "error", str(e)
            log.warning("  %s run failed: %s", self.what, e)
        n, ok = self.tally()
        if n >= len(cases):
            self.mark(True)
            log.info("  result %s: %.1f%% (%s/%s)", self.what, 100 * ok / n if n else 0, ok, n)
        elif status == "error":
            self.mark(False, error)
        self.finish(status, error, n, len(cases))


class ToolsExecution(_CaseSuite):
    """Run one BFCL category against one model."""

    @property
    def what(self) -> str:
        return f"tools {self.spec.p['category']}"

    def cases(self) -> list[tuple[str, bfcl.Case]]:
        p = self.spec.p
        return [(c.id, c) for c in bfcl.sample(p["category"], p["sample_size"])]

    def done(self) -> set[str]:
        return {r["case_id"] for r in self.db.q(
            "SELECT case_id FROM tool_cases WHERE model_id=? AND engine=? AND spec_key=?"
            " AND settings_fingerprint=?",
            self.model.model_id, self.engine, self.spec.key, self.model.settings_fingerprint)}

    def body(self, case: bfcl.Case) -> dict:
        thinking = self.spec.p["enable_thinking"]
        body = {
            "model": self.model.model_id,
            "messages": case.messages,
            "tools": case.tools(),
            "tool_choice": "auto",
            # No sampling parameters, like Hermes: oMLX applies the
            # model's saved settings (each model's recommended values).
            "max_tokens": 8192 if thinking else 1024,
            "enable_thinking": thinking,
        }
        if thinking:
            body["reasoning_effort"] = "medium"  # what Hermes sends
        return body

    def tally(self) -> tuple[int, int]:
        r = self.db.q1(
            "SELECT count(*) AS n, sum(correct) AS ok FROM tool_cases"
            " WHERE model_id=? AND engine=? AND spec_key=? AND settings_fingerprint=?",
            self.model.model_id, self.engine, self.spec.key, self.model.settings_fingerprint)
        return r["n"], r["ok"] or 0

    def record(self, case: bfcl.Case, resp: httpx.Response, elapsed: float) -> None:
        try:
            data = resp.json()
        except ValueError:
            data = {"error": resp.text[:2000]}
        if resp.status_code >= 400 or not data.get("choices"):
            # e.g. the chat template rejects tools: a capability failure, not a glitch
            calls, message, finish = None, {}, None
            correct, reason = False, f"HTTP {resp.status_code}: {json.dumps(data)[:300]}"
        else:
            choice = data["choices"][0]
            message, finish = choice.get("message") or {}, choice.get("finish_reason")
            calls = bfcl.parse_calls(case, message)
            correct, reason = bfcl.score(case, calls)
            if not message.get("tool_calls") and _looks_like_tool_call(message.get("content")):
                reason += " (text resembles a tool call: parser did not extract it)"
        usage = data.get("usage") or {}
        ttft, total = server_timing(data)
        self.db.x(
            "INSERT OR REPLACE INTO tool_cases(run_id, model_id, engine, spec_key,"
            " settings_fingerprint, category, case_id, correct, reason, tool_calls_json, content,"
            " finish_reason, prompt_tokens, completion_tokens, time_to_first_token, total_time,"
            " raw_json, recorded_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            self.run_id, self.model.model_id, self.engine, self.spec.key,
            self.model.settings_fingerprint, case.category, case.id, int(correct), reason,
            dumps(calls), message.get("content"), finish, usage.get("prompt_tokens"),
            usage.get("completion_tokens"), ttft, total or elapsed, dumps(data), now(),
        )


def _looks_like_tool_call(text: str | None) -> bool:
    return bool(text) and any(m in text for m in ("<tool_call>", '"arguments"', "<function=",
                                                  "[TOOL_CALLS]"))


class AgentTurnsExecution(_Harness):
    """Replay the Hermes-shaped conversation and time every step.

    Unlike tool cases, steps depend on the cache state the previous ones
    left, so a cancelled replay is not resumed: the next run starts over
    with a fresh session marker (cold first step).
    """

    def run(self) -> None:
        from . import agent_turns

        p = self.spec.p
        thinking = p["enable_thinking"]
        status, error, recorded = "completed", None, 0
        try:
            tools, requests = agent_turns.session()
            self.rebaseline()
            # Load the model (and JIT) outside the measurement.
            if self.chat({"model": self.model.model_id, "max_tokens": 1,
                          "messages": [{"role": "user", "content": "Reply with OK."}]}).is_success:
                self.ours_completed += 1
            else:
                self.ours_rejected += 1
                self.rebaseline()  # unclear whether oMLX counted a rejected request
            for step, messages in enumerate(requests, 1):
                if reason := self.should_yield(ours_in_flight=0):
                    self.cancel_reason = reason
                    raise _Aborted()
                body = {"model": self.model.model_id, "messages": messages, "tools": tools,
                        "tool_choice": "auto", "max_tokens": agent_turns.MAX_TOKENS,
                        "enable_thinking": thinking}
                if thinking:
                    body["reasoning_effort"] = "medium"
                result = self.chat_first_token(body)
                if result["ttft"] is None:
                    self.ours_rejected += bool(result["error"])
                    raise OmlxStepError(f"step {step}: no tokens: {result['error']}")
                self.ours_completed += 1
                self.record(step, result)
                recorded += 1
        except _Aborted:
            status = "cancelled"
            log.info("yielding the server (%s)", self.cancel_reason)
        except (OmlxStepError, OSError, ValueError, *NET_ERRORS) as e:
            # OSError: corpus missing; ValueError: a non-JSON response body
            status, error = "error", str(e)
            log.warning("  agent turns failed: %s", e)
        if recorded == agent_turns.STEPS:
            self.mark(True)
        elif status == "error":
            self.mark(False, error)
        self.finish(status, error, recorded, agent_turns.STEPS)

    def record(self, step: int, result: dict) -> None:
        u = result["usage"] or {}
        tf = result.get("server") or {}
        # The server's own TTFT when it reported one (oMLX in usage,
        # TensorFold in its own block); ours otherwise (an error ending, e.g.
        # a reply cut off inside a tool call, has no usage).
        ttft = u.get("time_to_first_token") or tf.get("time_to_first_token") or result["ttft"]
        total = u.get("total_time") or tf.get("seconds")
        gen_tps = u.get("generation_tokens_per_second")
        if gen_tps is None and total and ttft and (u.get("completion_tokens") or 0) > 1 \
                and total > ttft:
            gen_tps = (u["completion_tokens"] - 1) / (total - ttft)
        prompt = u.get("prompt_tokens")
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        prompt_tps = u.get("prompt_tokens_per_second")
        if prompt_tps is None and tf.get("prefill_seconds") and prompt:
            # Like oMLX's figure: speed over the uncached tokens.
            prompt_tps = (prompt - cached) / tf["prefill_seconds"]
        log.info("  step %d: %s prompt tokens, ttft %.1fs%s", step, prompt, ttft,
                 f" ({result['error']})" if result["error"] else "")
        self.db.x(
            "INSERT INTO agent_turns(run_id, model_id, engine, spec_key, settings_fingerprint,"
            " step, prompt_tokens, cached_tokens, completion_tokens, time_to_first_token,"
            " prompt_tps, generation_tps, total_time, raw_json, recorded_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            self.run_id, self.model.model_id, self.engine, self.spec.key,
            self.model.settings_fingerprint, step, prompt,
            (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            u.get("completion_tokens"), ttft, prompt_tps, gen_tps, total,
            dumps({**result, "client_ttft": result["ttft"]}), now(),
        )


class PulseMemoryExecution(_CaseSuite):
    """Pulse's memory-extraction request, every case `repeats` times."""

    what = "pulse memory"

    def cases(self) -> list[tuple[str, tuple]]:
        from . import pulse_memory

        return [(f"{c.id}#{t}", (c, t)) for t in range(1, self.spec.p["repeats"] + 1)
                for c in pulse_memory.load()]

    def done(self) -> set[str]:
        return {f"{r['case_id']}#{r['trial']}" for r in self.db.q(
            "SELECT case_id, trial FROM pulse_memory_cases WHERE model_id=? AND engine=?"
            " AND spec_key=? AND settings_fingerprint=?",
            self.model.model_id, self.engine, self.spec.key, self.model.settings_fingerprint)}

    def body(self, case: tuple) -> dict:
        from . import pulse_memory

        return pulse_memory.body(self.model.model_id, *case)

    def tally(self) -> tuple[int, int]:
        r = self.db.q1(
            "SELECT count(*) AS n, sum(pulse_ok) AS ok FROM pulse_memory_cases"
            " WHERE model_id=? AND engine=? AND spec_key=? AND settings_fingerprint=?",
            self.model.model_id, self.engine, self.spec.key, self.model.settings_fingerprint)
        return r["n"], r["ok"] or 0

    def record(self, case: tuple, resp: httpx.Response, elapsed: float) -> None:
        from . import pulse_memory

        c, trial = case
        try:
            data = resp.json()
        except ValueError:
            data = {"error": resp.text[:2000]}
        usage = data.get("usage") or {}
        total = server_timing(data)[1] or elapsed
        if resp.status_code >= 400 or not data.get("choices"):
            content = finish = parsed = None
            pulse_ok = content_ok = False
            reason = f"HTTP {resp.status_code}: {json.dumps(data)[:300]}"
        else:
            choice = data["choices"][0]
            content = (choice.get("message") or {}).get("content")
            finish = choice.get("finish_reason")
            pulse_ok, content_ok, reason, parsed = pulse_memory.score(c, content, finish, total)
        if not pulse_ok:
            log.info("  %s#%d: %s", c.id, trial, reason)
        self.db.x(
            "INSERT OR REPLACE INTO pulse_memory_cases(run_id, model_id, engine, spec_key,"
            " settings_fingerprint, case_id, trial, pulse_ok, content_ok, reason, content,"
            " parsed_json, finish_reason, prompt_tokens, completion_tokens, total_time, raw_json,"
            " recorded_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            self.run_id, self.model.model_id, self.engine, self.spec.key,
            self.model.settings_fingerprint,
            c.id, trial, int(pulse_ok), int(content_ok), reason, content,
            None if parsed is None else dumps(parsed), finish, usage.get("prompt_tokens"),
            usage.get("completion_tokens"), total, dumps(data), now(),
        )


class StreamPerfExecution(_Harness):
    """Client-side streamed speed test, one spec (prompt length) per request,
    each recorded as it finishes. See stream_perf.py."""

    def _calibrate(self) -> tuple[float, int]:
        """(chars per token, prompt overhead in tokens) for this model's
        tokenizer, from two one-token requests. Also loads the model."""
        from . import stream_perf

        counts = []
        for n in (0, stream_perf.CALIBRATION_CHARS):
            resp = self.chat({"messages": stream_perf.messages(n), "max_tokens": 1,
                              "enable_thinking": False})
            if not resp.is_success:
                self.ours_rejected += 1
                raise OmlxStepError(f"calibration: HTTP {resp.status_code}: {resp.text[:300]}")
            self.ours_completed += 1
            counts.append((resp.json().get("usage") or {}).get("prompt_tokens"))
        base, full = counts
        if not base or not full or full <= base:
            raise OmlxStepError(f"calibration: no usable prompt token counts ({counts})")
        return stream_perf.CALIBRATION_CHARS / (full - base), base

    def _reset_peak(self) -> None:
        if self.engine == "tensorfold":
            try:
                self.client.http.get("/health", params={"reset_peak": 1})
            except NET_ERRORS:
                pass

    def _peak(self) -> int | None:
        if self.engine != "tensorfold":
            return None  # oMLX doesn't expose per-request peak memory
        try:
            m = self.client.metrics()
        except NET_ERRORS:
            return None
        v = m.get("device_memory_peak_bytes")
        return int(v) if v else None

    def run(self) -> None:
        from . import stream_perf

        status, error, recorded = "completed", None, 0
        try:
            self.rebaseline()
            cpt, overhead = self._calibrate()
            log.info("  %.2f chars/token, %d tokens of prompt overhead", cpt, overhead)
            for spec in self.unit.specs:
                if reason := self.should_yield(ours_in_flight=0):
                    self.cancel_reason = reason
                    raise _Aborted()
                pp, tg = spec.p["pp"], spec.p["tg"]
                body = {"messages": stream_perf.messages(stream_perf.chars_for(pp, cpt, overhead)),
                        "max_tokens": tg, "enable_thinking": False}
                self._reset_peak()
                timing = self.chat_first_token(body)
                timing["peak_memory_bytes"] = self._peak()
                if timing["ttft"] is None:
                    self.ours_rejected += bool(timing["error"])
                    raise OmlxStepError(f"pp{pp}: no tokens: {timing['error']}")
                self.ours_completed += 1
                r = stream_perf.result(pp, tg, timing)
                self.db.insert_perf_result(self.run_id, spec.key, r, None)
                if r["prompt_tokens"] and r["prompt_tokens"] > overhead:
                    # Code at scale tokenizes a little differently from the
                    # calibration sample: size the next prompt from this one.
                    cpt = stream_perf.chars_for(pp, cpt, overhead) / (r["prompt_tokens"] - overhead)
                self.mark(True, spec=spec)
                recorded += 1
                log.info("  result stream pp%s (%s tokens): pp %.1f tok/s, tg %s tok/s, ttft %.0f ms",
                         pp, r["prompt_tokens"], r["processing_tps"] or 0,
                         "n/a" if r["gen_tps"] is None else f"{r['gen_tps']:.1f}", r["ttft_ms"])
        except _Aborted:
            status = "cancelled"
            log.info("yielding the server (%s)", self.cancel_reason)
        except (OmlxStepError, OSError, ValueError, *NET_ERRORS) as e:
            status, error = "error", str(e)
            log.warning("  stream perf failed: %s", e)
            done = {s.key for s in self.unit.specs[:recorded]}
            for spec in self.unit.specs:
                if spec.key not in done:
                    self.mark(False, error, spec=spec)
        self.finish(status, error, recorded, len(self.unit.specs))


class OmlxStepError(RuntimeError):
    pass


EXECUTIONS = {"tools": ToolsExecution, "agent_turns": AgentTurnsExecution,
              "pulse_memory": PulseMemoryExecution, "stream_perf": StreamPerfExecution}
