"""A fixed, Hermes-shaped agent conversation for measuring real-use latency.

Hermes sessions are long and mostly prefix-cached: a large system prompt
with tool definitions, then a loop of model turn -> tool call -> tool result,
each request resending the whole history. oMLX's built-in benchmark defeats
the prefix cache on purpose, so it cannot show what such a session feels
like. This replays one: STEPS requests, each ending with a new tool result,
growing from ~20k to ~64k prompt tokens in ~4k-token steps.

The conversation is identical for every model and run (fixed text, fixed
assistant turns; the model's own replies are discarded), except for a
per-run marker at the very start so the first step is always cold and
nothing is reused from an earlier run.
"""

from __future__ import annotations

import json
import uuid
from functools import cache
from pathlib import Path

from . import bfcl

SUITE = "hermes_like_v1"
DATA_DIR = Path(__file__).parent / "data" / "agent_turns"
_CORPUS_FILES = ("argparse.py.txt", "http_client.py.txt", "asyncio_base_events.py.txt",
                 "json_decoder.py.txt", "json_encoder.py.txt", "textwrap.py.txt")

NOTES_CHARS = 20_000                      # "memory" section of the system prompt
# Characters of tool output added before each measured request. Python
# source runs ~4.75 chars/token on Qwen-family tokenizers and the tools and
# instructions take ~6k tokens, so the first request is ~20k tokens and each
# later one adds ~4k (about 94% cached, as in real Hermes sessions on this
# machine), ending near 64k. Actual counts are recorded per step.
TOOL_RESULT_CHARS = (46_000,) + (19_000,) * 11
STEPS = len(TOOL_RESULT_CHARS)
# Only time to first token is measured here (decode speed at long context is
# tier 2's job). 16 tokens stay inside the model's thinking, so a reply can't
# be cut off mid tool call, which oMLX rejects as `incomplete_tool_call`.
MAX_TOKENS = 16

_INSTRUCTIONS = """You are an autonomous software agent working in the user's repository.
You can read and write files and run shell commands through the provided tools.

Working rules:
- Investigate before changing anything: read the relevant files first.
- Prefer small, verifiable steps. After each change, run the tests.
- When a tool fails, read the error carefully and adapt; do not repeat the
  same failing call more than twice.
- Keep the user informed with short progress notes, and finish with a summary
  of what changed and what remains.
- Never run destructive commands (deleting files, force-pushing) without
  explicit confirmation.
- When you learn something reusable about this project, record it with the
  save_note tool so future sessions can use it.

The notes below were saved during earlier sessions."""

_OWN_TOOLS = [
    {"name": "read_file", "description": "Read a file from the repository, one page at a time.",
     "parameters": {"type": "dict", "properties": {
         "path": {"type": "string", "description": "Path relative to the repository root."},
         "page": {"type": "integer", "description": "Page number, starting at 1."}},
         "required": ["path"]}},
    {"name": "write_file", "description": "Create or overwrite a file in the repository.",
     "parameters": {"type": "dict", "properties": {
         "path": {"type": "string", "description": "Path relative to the repository root."},
         "content": {"type": "string", "description": "The full new file content."}},
         "required": ["path", "content"]}},
    {"name": "run_shell", "description": "Run a shell command in the repository and return its output.",
     "parameters": {"type": "dict", "properties": {
         "command": {"type": "string", "description": "The command line to run."},
         "timeout_s": {"type": "integer", "description": "Seconds before the command is killed."}},
         "required": ["command"]}},
    {"name": "save_note", "description": "Save a note for future sessions.",
     "parameters": {"type": "dict", "properties": {
         "text": {"type": "string", "description": "The note."}}, "required": ["text"]}},
]

_STEP_FILES = ("lib/cli/argparse.py", "lib/net/http_client.py", "lib/aio/base_events.py",
               "lib/serial/json_codec.py", "lib/text/wrap.py")
# Long files are read in pages, as agents do: ("path", page) per step.
_READS = [(_STEP_FILES[min(i // 3, len(_STEP_FILES) - 1)], i % 3 + 1)
          for i in range(len(TOOL_RESULT_CHARS))]


@cache
def _corpus() -> str:
    return "\n\n".join((DATA_DIR / f).read_text() for f in _CORPUS_FILES)


@cache
def tools() -> list[dict]:
    """Our four agent tools plus 36 unrelated BFCL functions, like the long
    tool lists real agents carry."""
    extra, seen = [], set()
    for case in bfcl.load("multiple"):
        for f in case.functions:
            if f["name"] not in seen and len(extra) < 36:
                seen.add(f["name"])
                extra.append(f)
    case = bfcl.Case(id="", category="", messages=[], functions=_OWN_TOOLS + extra, answers=None)
    return case.tools()


def session() -> tuple[list[dict], list[list[dict]]]:
    """(tools, message list for each of the STEPS requests), with a fresh
    run marker. The marker goes in both the first tool's description and the
    system prompt: chat templates differ in which they render first (Qwen3.5
    puts tools first), and the first cache block must be unique per run."""
    marker = f"[session {uuid.uuid4()}]"
    run_tools = json.loads(json.dumps(tools()))
    first = run_tools[0]["function"]
    first["description"] = f"{marker} {first['description']}"
    corpus = _corpus()
    assert NOTES_CHARS + sum(TOOL_RESULT_CHARS) <= len(corpus), "corpus too small"
    system = f"{marker}\n\n{_INSTRUCTIONS}\n\n## Notes\n\n{corpus[:NOTES_CHARS]}"
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": "Please review the modules under lib/ for bugs and "
                                    "risky error handling, one file at a time, and then "
                                    "summarize what you found."},
    ]
    requests = []
    offset = NOTES_CHARS
    for step, size in enumerate(TOOL_RESULT_CHARS):
        call_id = f"call_{step}"
        messages = messages + [
            {"role": "assistant", "content": f"Reading {_READS[step][0]}, page {_READS[step][1]}.",
             "tool_calls": [{"id": call_id, "type": "function", "function": {
                 "name": "read_file", "arguments": json.dumps({"path": _READS[step][0],
                                                               "page": _READS[step][1]})}}]},
            {"role": "tool", "tool_call_id": call_id, "content": corpus[offset:offset + size]},
        ]
        offset += size
        requests.append(messages)
    return run_tools, requests
