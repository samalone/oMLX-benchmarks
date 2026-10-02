"""Tool-calling cases from BFCL v3 and a scorer modeled on BFCL's AST checker.

Each case is sent to oMLX's OpenAI-compatible endpoint with the functions as
`tools`, so the model's chat template and oMLX's tool-call parser are tested
together, as an agent like Hermes would use them. A response passes when its
tool calls match one of the accepted answers:

- the right function names, the right number of calls (any order for
  parallel categories), no unknown parameters, every required one present;
- each argument equal to an accepted value: strings compared after BFCL's
  normalization (case, spaces and some punctuation ignored), ints accepted
  where floats are expected, lists element-wise, dicts recursively;
- for `irrelevance`, passing means calling no tool at all.

Simplified from BFCL's checker: no per-language (Java/JS) type rules.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent / "data" / "bfcl_v3"
CATEGORIES = ("simple", "multiple", "parallel", "parallel_multiple", "irrelevance")
SAMPLE_SEED = 42

# BFCL's Python-flavored schema types -> JSON Schema
_TYPE_MAP = {"dict": "object", "float": "number", "tuple": "array", "any": None}


@dataclass(frozen=True)
class Case:
    id: str
    category: str
    messages: list[dict]
    functions: list[dict]           # BFCL definitions (original names)
    answers: list[dict] | None      # [{func_name: {param: [accepted values]}}], None = no call

    def tools(self) -> list[dict]:
        return [{"type": "function", "function": {
            "name": api_name(f["name"]),
            "description": f.get("description", ""),
            "parameters": _schema(f.get("parameters") or {"type": "dict", "properties": {}}),
        }} for f in self.functions]


def api_name(name: str) -> str:
    """OpenAI-style tool names allow only [a-zA-Z0-9_-]; BFCL uses dots."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def _schema(node: Any) -> Any:
    if isinstance(node, list):
        return [_schema(v) for v in node]
    if not isinstance(node, dict):
        return node
    out = {}
    for k, v in node.items():
        if k == "type" and isinstance(v, str):
            mapped = _TYPE_MAP.get(v, v)
            if mapped is not None:
                out[k] = mapped
        elif k == "properties" and isinstance(v, dict):
            out[k] = {name: _schema(p) for name, p in v.items()}  # keys are param names
        else:
            out[k] = _schema(v)
    return out


@cache
def load(category: str) -> tuple[Case, ...]:
    questions = [json.loads(l) for l in (DATA_DIR / f"{category}.json").read_text().splitlines() if l]
    answers: dict[str, list] = {}
    answer_file = DATA_DIR / f"{category}_answers.json"
    if answer_file.exists():
        for line in answer_file.read_text().splitlines():
            if line:
                a = json.loads(line)
                answers[a["id"]] = a["ground_truth"]
    return tuple(
        Case(
            id=q["id"], category=category,
            messages=q["question"][0],  # single-turn categories have one turn
            functions=q["function"],
            answers=answers.get(q["id"]) if category != "irrelevance" else None,
        )
        for q in questions
    )


def sample(category: str, n: int) -> list[Case]:
    """Deterministic subset (n=0: all), stable across runs and models."""
    cases = list(load(category))
    if n and n < len(cases):
        cases = random.Random(f"{SAMPLE_SEED}:{category}").sample(cases, n)
    return cases


# -- scoring -------------------------------------------------------------------


def _norm_str(s: str) -> str:
    # BFCL standardize_string
    return re.sub(r"[ ,./\-_*^]", "", s).lower().replace("'", '"')


def _value_ok(value: Any, accepted: Any) -> bool:
    if isinstance(accepted, str):
        return isinstance(value, str) and _norm_str(value) == _norm_str(accepted)
    if isinstance(accepted, bool) or isinstance(value, bool):
        return value is accepted
    if isinstance(accepted, (int, float)):
        return isinstance(value, (int, float)) and float(value) == float(accepted)
    if isinstance(accepted, list):
        return (isinstance(value, (list, tuple)) and len(value) == len(accepted)
                and all(_value_ok(v, a) for v, a in zip(value, accepted)))
    if isinstance(accepted, dict):
        # nested dict: each key maps to a list of accepted values, "" = optional
        if not isinstance(value, dict) or set(value) - set(accepted):
            return False
        return all(
            (k in value and any(_value_ok(value[k], a) for a in opts if a != ""))
            or (k not in value and "" in opts)
            for k, opts in accepted.items()
        )
    return value == accepted


def _call_ok(call: dict, expected: dict, functions: dict[str, dict]) -> str | None:
    """None if `call` matches `expected` ({name: {param: [accepted]}}), else why not."""
    (name, params), = expected.items()
    if call["name"] != name:
        return f"wrong function {call['name']} (expected {name})"
    args = call["arguments"]
    schema = (functions.get(name, {}).get("parameters") or {})
    known = set((schema.get("properties") or {}).keys())
    if set(args) - known:
        return f"unknown parameter(s) {sorted(set(args) - known)}"
    for p in schema.get("required") or []:
        if p not in args:
            return f"missing required parameter {p}"
    for p, opts in params.items():
        if p not in args:
            if "" not in opts:
                return f"missing parameter {p}"
            continue
        if not any(_value_ok(args[p], a) for a in opts if a != ""):
            return f"wrong value for {p}: {args[p]!r}"
    return None


def score(case: Case, calls: list[dict] | None) -> tuple[bool, str]:
    """`calls` = [{"name": original name, "arguments": dict}], None if unparseable."""
    if calls is None:
        return False, "tool call arguments were not valid JSON"
    if case.answers is None:
        return (not calls), ("ok" if not calls else f"called {calls[0]['name']} for an irrelevant request")
    if not calls:
        return False, "no tool call"
    if len(calls) != len(case.answers):
        return False, f"{len(calls)} calls (expected {len(case.answers)})"
    functions = {f["name"]: f for f in case.functions}
    # Any order: find a matching for each expected call (sizes are small).
    remaining = list(calls)
    for expected in case.answers:
        reasons = [_call_ok(c, expected, functions) for c in remaining]
        hit = next((i for i, r in enumerate(reasons) if r is None), None)
        if hit is None:
            return False, reasons[0] if len(reasons) == 1 else f"no call matches {next(iter(expected))}"
        remaining.pop(hit)
    return True, "ok"


def parse_calls(case: Case, message: dict) -> list[dict] | None:
    """Tool calls from an OpenAI chat message, mapped back to BFCL names."""
    by_api = {api_name(f["name"]): f["name"] for f in case.functions}
    calls = []
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        raw = fn.get("arguments")
        try:
            args = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except ValueError:
            return None
        if not isinstance(args, dict):
            return None
        calls.append({"name": by_api.get(fn.get("name"), fn.get("name")), "arguments": args})
    return calls
