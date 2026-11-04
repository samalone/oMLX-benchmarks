"""Pulse memory extraction: structured JSON output as Hermes's Pulse plugin asks for it.

Pulse (the conduit_push Hermes plugin) periodically sends a chunk of chat to
the model with `response_format: json_object` and a long instruction, and
expects back `{"items": [...], "closed": [...], "claims": [...]}`. Some
models return valid JSON in the wrong shape: a bare line number like `[1]`,
a single item not wrapped in `items`, or an echo of the input. Pulse rejects
those (BadReply) and the chunk's memories are lost. In November 2026
Ornith-1.5-35B-A3B-Abliterated failed about two calls in three this way.

Each case is sent exactly as Hermes's plugin LLM layer builds it
(agent/plugin_llm.py, `_build_structured_messages`): a system message with
the JSON-only directive, then one user message whose text parts are Pulse's
instructions (vendored verbatim from conduit_push 0.126.0) and the input. No
sampling parameters and no thinking flag, like Hermes, so oMLX applies the
model's saved settings. The failure is intermittent, so every case runs
`repeats` times, each with its own random-looking but fixed MARK token.

Two grades per reply:
- `pulse_ok`: Pulse would accept it: after stripping code fences (as Hermes
  does) it parses to an object whose "items" is a list, it wasn't cut off,
  and it came back within Pulse's 120 s timeout. This is the failure above.
- `content_ok`: also sensible: every item has a known kind and a text, and
  the case's expectations hold (nothing kept for a chat with nothing worth
  keeping; the expected kinds, closed numbers and claim actions present).
  Loose on purpose: it catches wrong-shaped and empty answers, not taste.

The chats are synthetic. Do not add fixtures from real conversations.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

SUITE = "pulse_memory_v1"
DATA_DIR = Path(__file__).parent / "data" / "pulse_memory"
SYSTEM = ("Respond with a single JSON object that matches the requested shape. "
          "Do not include prose or markdown fences.")
KINDS = ("commitment", "promise", "intention", "interest", "fact", "watch")
CLAIM_ACTIONS = ("reminder", "event", "job", "message")
PULSE_TIMEOUT_S = 120.0
# Pulse asks for 2000, but oMLX's log showed no max_tokens arriving from
# Hermes (so 32768 applied). This only bounds a runaway reply; a reply that
# long would miss Pulse's timeout anyway.
MAX_TOKENS = 8192

_FENCE = re.compile(r"```(?:json)?\s*(.+?)```", re.DOTALL | re.IGNORECASE)  # Hermes's


@dataclass(frozen=True)
class Case:
    id: str
    remembered: tuple[str, ...]
    chat: tuple[dict, ...]
    expect: dict


@cache
def instructions() -> str:
    return (DATA_DIR / "instructions.txt").read_text()


@cache
def load() -> tuple[Case, ...]:
    data = json.loads((DATA_DIR / "cases.json").read_text())
    return tuple(Case(c["id"], tuple(c["remembered"]), tuple(c["chat"]), c["expect"])
                 for c in data["cases"])


def render_chat(case: Case) -> str:
    """Numbered lines like Pulse's render(); tool outcomes unnumbered."""
    lines, n = [], 0
    for row in case.chat:
        if "tool" in row:
            lines.append(f"    (tool) {row['tool']}")
            continue
        n += 1
        text = re.sub(r"\s+", " ", row["text"]).strip()  # as Pulse's _text()
        lines.append(f"[{n}] {row['at']} {row['who']}: {text}")
    return "\n".join(lines)


def mark(case: Case, trial: int) -> str:
    return hashlib.sha256(f"{SUITE}:{case.id}:{trial}".encode()).hexdigest()[:12]


def messages(case: Case, trial: int) -> list[dict]:
    m = mark(case, trial)
    remembered = "\n".join(case.remembered) or "(nothing yet)"
    text = (f"<<<remembered {m}\n{remembered}\nremembered {m}>>>\n\n"
            f"<<<chat {m}\n{render_chat(case)}\nchat {m}>>>")
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [{"type": "text", "text": instructions().replace("MARK", m).strip()},
                                     {"type": "text", "text": text}]},
    ]


def body(model_id: str, case: Case, trial: int) -> dict:
    return {"model": model_id, "messages": messages(case, trial), "max_tokens": MAX_TOKENS,
            "response_format": {"type": "json_object"}}


def parse(content: str | None) -> tuple[Any, str | None]:
    """(parsed JSON, None), or (None, why not), the way Hermes parses it."""
    if not content or not content.strip():
        return None, "empty reply"
    match = _FENCE.search(content)
    text = match.group(1).strip() if match else content.strip()
    try:
        return json.loads(text), None
    except ValueError:
        return None, f"not JSON: {_snip(text)}"


def score(case: Case, content: str | None, finish_reason: str | None,
          total_time: float | None) -> tuple[bool, bool, str, Any]:
    """(pulse_ok, content_ok, reason, parsed)."""
    parsed, why = parse(content)
    if why is None:
        why = _shape_problem(parsed)
    if why is None and finish_reason == "length":
        why = "cut off at the token limit"
    if why is None and total_time is not None and total_time > PULSE_TIMEOUT_S:
        why = f"took {total_time:.0f} s, over Pulse's {PULSE_TIMEOUT_S:.0f} s timeout"
    if why is not None:
        return False, False, why, parsed
    why = _content_problem(case, parsed)
    return True, why is None, why or "ok", parsed


def _shape_problem(parsed: Any) -> str | None:
    if not isinstance(parsed, dict):
        return f"wrong shape: a JSON {type(parsed).__name__}, not an object: {_snip(json.dumps(parsed))}"
    if "items" not in parsed:
        return f"wrong shape: object without \"items\" (keys: {', '.join(list(parsed)[:6]) or 'none'})"
    if not isinstance(parsed["items"], list):
        return f"wrong shape: \"items\" is a {type(parsed['items']).__name__}, not a list"
    return None


def _content_problem(case: Case, parsed: dict) -> str | None:
    items = parsed["items"]
    closed = parsed.get("closed") if isinstance(parsed.get("closed"), list) else []
    claims = parsed.get("claims") if isinstance(parsed.get("claims"), list) else []
    for item in items:
        if not isinstance(item, dict) or item.get("kind") not in KINDS \
                or not isinstance(item.get("text"), str) or not item["text"].strip():
            return f"malformed item: {_snip(json.dumps(item))}"
    exp = case.expect
    if exp.get("empty"):
        if items:
            return f"kept {len(items)} item(s) from a chat with nothing new: " \
                   + ", ".join(str(i.get("kind")) for i in items)
        return None
    kinds = {i["kind"] for i in items}
    if missing := [k for k in exp.get("kinds", []) if k not in kinds]:
        return f"no {', '.join(missing)} item (got: {', '.join(sorted(kinds)) or 'none'})"
    numbers = {_int(c.get("number")) for c in closed if isinstance(c, dict)}
    if missing := [n for n in exp.get("closed", []) if n not in numbers]:
        return f"did not close #{', #'.join(map(str, missing))}"
    actions = {c.get("action") for c in claims if isinstance(c, dict)}
    if missing := [a for a in exp.get("claims", []) if a not in actions]:
        return f"no {', '.join(missing)} claim"
    return None


def _int(v: Any) -> int | None:
    try:
        return int(str(v).lstrip("#"))
    except (TypeError, ValueError):
        return None


def _snip(s: str, n: int = 80) -> str:
    s = re.sub(r"\s+", " ", s)
    return s if len(s) <= n else s[:n] + "…"
