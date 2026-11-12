"""Client-side speed test: one streamed chat request per prompt length.

oMLX's built-in perf benchmark runs inside oMLX, so it can't measure another
engine. This test measures any OpenAI-compatible server the same way, from
outside: a prompt of about `pp` tokens (Python source from the agent_turns
corpus), nothing cached, and a reply of `tg` tokens, timed on the client.

- TTFT: request sent -> first streamed token (content or reasoning).
- Prefill tok/s: prompt tokens / TTFT.
- Decode tok/s: (completion tokens - 1) / (last token - first token).

Each prompt starts with a fresh random marker, so no prefix cache can serve
it. Prompts are sized with a chars-per-token ratio measured at the start of
each run for the model's own tokenizer; the actual prompt length is recorded.
Thinking is off, so the reply is the rewrite itself, which runs far past `tg`.
"""

from __future__ import annotations

import uuid
from functools import cache

from .agent_turns import _corpus

SUITE = "stream_v1"
CALIBRATION_CHARS = 8_000
MIN_DECODE_TOKENS = 16  # as oMLX: below this, decode speed is not reported

_INSTRUCTION = ("Rewrite the following Python code, adding a one-line comment above every "
                "function and class. Output only the code.")


@cache
def _text(n_chars: int) -> str:
    corpus = _corpus()
    reps = n_chars // len(corpus) + 1
    return ("\n\n".join([corpus] * reps))[:n_chars]


def messages(n_chars: int) -> list[dict]:
    """One user message holding `n_chars` characters of code, uniquely marked."""
    return [{"role": "user", "content": f"[request {uuid.uuid4()}]\n{_INSTRUCTION}\n\n"
                                        f"```python\n{_text(n_chars)}\n```"}]


def chars_for(pp: int, chars_per_token: float, overhead_tokens: int) -> int:
    return max(0, int((pp - overhead_tokens) * chars_per_token))


def result(pp_target: int, tg: int, timing: dict) -> dict:
    """A perf_results row (test_type 'stream') from chat_first_token timing."""
    u = timing.get("usage") or {}
    prompt = u.get("prompt_tokens")
    completion = u.get("completion_tokens")
    ttft = timing.get("ttft")
    first, last = timing.get("ttft"), timing.get("t_last")
    decode_s = (last - first) if first is not None and last is not None else None
    gen_tps = ((completion - 1) / decode_s
               if completion and completion >= MIN_DECODE_TOKENS and decode_s else None)
    return {
        "test_type": "stream",
        "pp": pp_target,
        "tg": tg,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "ttft_ms": None if ttft is None else 1000 * ttft,
        "processing_tps": prompt / ttft if prompt and ttft else None,
        "gen_tps": gen_tps,
        "tpot_ms": 1000 / gen_tps if gen_tps else None,
        "e2e_latency_s": timing.get("elapsed"),
        "peak_memory_bytes": timing.get("peak_memory_bytes"),
        "finish_reason": timing.get("finish_reason"),
        "server_timing": timing.get("server"),
        "usage": u,
        "error": timing.get("error"),
    }
