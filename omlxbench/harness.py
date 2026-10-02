"""Suites omlxbench runs itself through oMLX's chat API (not oMLX built-ins):
BFCL tool calling and the Hermes-shaped agent-turns replay.

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
        self.env_hash = env_hash
        self.run_id = run_id
        self.cancel_reason: str | None = None
        self.baseline = 0
        self.ours_completed = 0
        self.rejected = RejectedRequests()
        self.rejected.new()  # start counting from here
        self.rejected_total = 0
        self.ours_rejected = 0  # our own non-2xx responses also appear in the log

    # -- yielding -----------------------------------------------------------

    def rebaseline(self) -> None:
        self.baseline = (self.client.status().get("total_requests") or 0) - self.ours_completed

    def should_yield(self, ours_in_flight: int) -> str | None:
        if pause_state(self.db)[0]:
            return "paused"
        try:
            status = self.client.status()
        except NET_ERRORS:
            return None
        loaded = status.get("loaded_models") or []
        if (status.get("total_requests") or 0) - self.baseline > self.ours_completed:
            return "user_traffic"
        self.rejected_total += self.rejected.new()
        if self.rejected_total > self.ours_rejected:
            return "user_traffic"  # oMLX turned someone else's request away
        if in_flight(status) > ours_in_flight:
            return "user_traffic"
        if status.get("models_loading") and self.model.model_id in loaded:
            return "user_traffic"  # loading something else, for someone else
        return None

    # -- one request ----------------------------------------------------------

    def chat(self, body: dict) -> httpx.Response:
        """POST a chat completion (non-streaming)."""
        return self._abandonable(lambda http: http.post("/v1/chat/completions", json=body))

    def chat_first_token(self, body: dict) -> dict:
        """Stream a chat completion and time its first token client-side.

        Returns {"ttft": seconds or None, "usage": dict or None, "error":
        str or None}. Streaming keeps the timing even when oMLX ends the
        response with an error (e.g. `incomplete_tool_call` when a reply is
        cut off inside a tool call), which a non-streaming request loses.
        """
        body = {**body, "stream": True, "stream_options": {"include_usage": True}}

        def stream(http: httpx.Client) -> dict:
            out: dict = {"ttft": None, "usage": None, "error": None}
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
                    delta = ((chunk.get("choices") or [{}])[0].get("delta") or {})
                    if out["ttft"] is None and any(delta.get(k) for k in
                                                   ("content", "reasoning_content", "reasoning",
                                                    "tool_calls")):
                        out["ttft"] = time.monotonic() - t0
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


class ToolsExecution(_Harness):
    """Run one BFCL category against one model."""

    def run(self) -> None:
        p = self.spec.p
        thinking = p["enable_thinking"]
        cases = bfcl.sample(p["category"], p["sample_size"])
        done = {r["case_id"] for r in self.db.q(
            "SELECT case_id FROM tool_cases WHERE model_id=? AND spec_key=? AND settings_fingerprint=?",
            self.model.model_id, self.spec.key, self.model.settings_fingerprint)}
        pending = [c for c in cases if c.id not in done]
        log.info("  %d/%d %s cases to go", len(pending), len(cases), p["category"])
        status, error, errors = "completed", None, 0
        try:
            self.rebaseline()
            for i, case in enumerate(pending, 1):
                if reason := self.should_yield(ours_in_flight=0):
                    self.cancel_reason = reason
                    raise _Aborted()
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
                try:
                    resp = self.chat(body)
                except NET_ERRORS as e:
                    errors += 1
                    log.warning("  %s: request failed: %s", case.id, e)
                    if errors >= MAX_CONSECUTIVE_ERRORS:
                        raise
                    self.rebaseline()
                    continue
                errors = 0
                self.record(case, resp)
                if resp.is_success:
                    self.ours_completed += 1
                else:
                    self.ours_rejected += 1
                    self.rebaseline()  # unclear whether oMLX counted a rejected request
                if i % 10 == 0:
                    log.info("  %s: %d/%d done", p["category"], len(done) + i, len(cases))
        except _Aborted:
            status = "cancelled"
            log.info("yielding the server (%s)", self.cancel_reason)
        except NET_ERRORS as e:
            status, error = "error", str(e)
        recorded = self.db.q1(
            "SELECT count(*) AS n, sum(correct) AS ok FROM tool_cases"
            " WHERE model_id=? AND spec_key=? AND settings_fingerprint=?",
            self.model.model_id, self.spec.key, self.model.settings_fingerprint)
        if recorded["n"] >= len(cases):
            self.db.mark_spec(self.model.model_id, self.spec.key, self.model.settings_fingerprint,
                              ok=True, run_id=self.run_id)
            log.info("  result tools %s: %.1f%% (%s/%s)", p["category"],
                     100 * (recorded["ok"] or 0) / recorded["n"], recorded["ok"], recorded["n"])
        elif status == "error":
            self.db.mark_spec(self.model.model_id, self.spec.key, self.model.settings_fingerprint,
                              ok=False, run_id=self.run_id, error=error,
                              environment_hash=self.env_hash)
        self.finish(status, error, recorded["n"], len(cases))

    def record(self, case: bfcl.Case, resp: httpx.Response) -> None:
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
        self.db.x(
            "INSERT OR REPLACE INTO tool_cases(run_id, model_id, spec_key, settings_fingerprint,"
            " category, case_id, correct, reason, tool_calls_json, content, finish_reason,"
            " prompt_tokens, completion_tokens, time_to_first_token, total_time, raw_json,"
            " recorded_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            self.run_id, self.model.model_id, self.spec.key, self.model.settings_fingerprint,
            case.category, case.id, int(correct), reason, dumps(calls), message.get("content"),
            finish, usage.get("prompt_tokens"), usage.get("completion_tokens"),
            usage.get("time_to_first_token"), usage.get("total_time"), dumps(data), now(),
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
            self.db.mark_spec(self.model.model_id, self.spec.key, self.model.settings_fingerprint,
                              ok=True, run_id=self.run_id)
        elif status == "error":
            self.db.mark_spec(self.model.model_id, self.spec.key, self.model.settings_fingerprint,
                              ok=False, run_id=self.run_id, error=error,
                              environment_hash=self.env_hash)
        self.finish(status, error, recorded, agent_turns.STEPS)

    def record(self, step: int, result: dict) -> None:
        u = result["usage"] or {}
        # oMLX's own TTFT when it reported usage; ours otherwise (an error
        # ending, e.g. a reply cut off inside a tool call, has no usage).
        ttft = u.get("time_to_first_token") or result["ttft"]
        prompt = u.get("prompt_tokens")
        log.info("  step %d: %s prompt tokens, ttft %.1fs%s", step, prompt, ttft,
                 f" ({result['error']})" if result["error"] else "")
        self.db.x(
            "INSERT INTO agent_turns(run_id, model_id, spec_key, settings_fingerprint, step,"
            " prompt_tokens, cached_tokens, completion_tokens, time_to_first_token,"
            " prompt_tps, generation_tps, total_time, raw_json, recorded_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            self.run_id, self.model.model_id, self.spec.key, self.model.settings_fingerprint,
            step, prompt, (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            u.get("completion_tokens"), ttft, u.get("prompt_tokens_per_second"),
            u.get("generation_tokens_per_second"), u.get("total_time"),
            dumps({**result, "client_ttft": result["ttft"]}), now(),
        )


class OmlxStepError(RuntimeError):
    pass
