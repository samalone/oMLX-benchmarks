"""Benchmark specs: the unit of "have it / missing it".

A spec is the smallest independently recorded oMLX result: one single-request
perf test at one prompt length, one batch test, one accuracy suite at one
sample size, or one context probe. Specs are identified by `key`, a canonical
string of kind + params, so the same desired test always maps to the same key.
"""

from __future__ import annotations

import fnmatch
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .snapshot import canonical

# Allowed values, from oMLX 0.7.0 (admin/benchmark.py, accuracy_benchmark.py,
# context_benchmark.py). oMLX validates too; checking here catches typos in
# targets.toml before a run takes over the server.
PERF_PROMPT_LENGTHS = {1024, 4096, 8192, 16384, 32768, 65536, 131072, 200000}
PERF_BATCH_SIZES = {2, 4, 8}
CONTEXT_PROFILES = {"code_python", "code_mixed", "novel_ko", "novel_en", "novel_ja"}
ACCURACY_SUITES = {
    "mmlu", "mmlu_pro", "kmmlu", "cmmlu", "jmmlu", "hellaswag", "truthfulqa",
    "arc_challenge", "winogrande", "gsm8k", "mathqa", "humaneval", "mbpp",
    "livecodebench", "bbq", "safetybench",
}
ACCURACY_BATCH_SIZES = {1, 2, 4, 8, 16, 32}
CONTEXT_TARGETS = {16384, 32768, 65536, 131072, 262144, 524288}


@dataclass(frozen=True)
class Spec:
    kind: str  # perf.single | perf.batch | accuracy | context
    params: tuple  # sorted (name, value) pairs, so Spec is hashable

    @classmethod
    def make(cls, kind: str, **params: Any) -> "Spec":
        return cls(kind, tuple(sorted(params.items())))

    @property
    def p(self) -> dict:
        return dict(self.params)

    @property
    def key(self) -> str:
        return f"{self.kind}:{canonical(self.p)}"

    @property
    def run_kind(self) -> str:
        return self.kind.split(".")[0]

    def label(self) -> str:
        p = self.p
        if self.kind == "perf.single":
            return f"perf pp{p['pp']}/tg{p['tg']} [{p['context_profile']}]"
        if self.kind == "perf.batch":
            return f"perf batch{p['batch_size']} pp{p['pp']}/tg{p['tg']} [{p['context_profile']}]"
        if self.kind == "accuracy":
            n = p["sample_size"] or "full"
            think = " thinking" if p["enable_thinking"] else ""
            return f"{p['suite']} n={n}{think}"
        return f"context {p['target_tokens']}"


def perf_single(pp: int, tg: int = 128, context_profile: str = "code_python",
                force_lm_engine: bool = False) -> Spec:
    return Spec.make("perf.single", pp=pp, tg=tg, context_profile=context_profile,
                     force_lm_engine=force_lm_engine)


def perf_batch(batch_size: int, tg: int = 128, context_profile: str = "code_python",
               force_lm_engine: bool = False) -> Spec:
    # oMLX always runs batch tests at pp1024.
    return Spec.make("perf.batch", batch_size=batch_size, pp=1024, tg=tg,
                     context_profile=context_profile, force_lm_engine=force_lm_engine)


def perf_spec_from_result(r: dict, *, tg: int, context_profile: str,
                          force_lm_engine: bool = False) -> Spec | None:
    """The spec an oMLX perf `result` row satisfies."""
    if r.get("test_type") == "single":
        return perf_single(r.get("pp"), tg, context_profile, force_lm_engine)
    if r.get("test_type") == "batch":
        return perf_batch(r.get("batch_size"), tg, context_profile, force_lm_engine)
    return None


def accuracy(suite: str, sample_size: int, enable_thinking: bool = False,
             sampling_profile: str = "deterministic", batch_size: int = 1) -> Spec:
    return Spec.make("accuracy", suite=suite, sample_size=sample_size,
                     enable_thinking=enable_thinking, sampling_profile=sampling_profile,
                     batch_size=batch_size)


def context(target_tokens: int) -> Spec:
    return Spec.make("context", target_tokens=target_tokens)


# -- targets.toml ------------------------------------------------------------


@dataclass
class TargetGroup:
    tier: int
    specs: list[Spec]
    include: list[str] = field(default_factory=lambda: ["*"])
    exclude: list[str] = field(default_factory=list)

    def applies_to(self, model_id: str) -> bool:
        return _matches(model_id, self.include, self.exclude)


@dataclass
class Targets:
    include: list[str]
    exclude: list[str]
    model_types: list[str]
    groups: list[TargetGroup]
    quiet_minutes: float = 10.0

    def wants_model(self, entry: dict) -> bool:
        """`entry` is an /admin/api/models entry (or ModelSnapshot.admin_model)."""
        return (
            (entry.get("model_type") or "llm") in self.model_types
            and not entry.get("is_helper")
            and _matches(entry["id"], self.include, self.exclude)
        )


def _matches(name: str, include: list[str], exclude: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, p) for p in include) and not any(
        fnmatch.fnmatchcase(name, p) for p in exclude
    )


def _check(value: Any, allowed: set, what: str) -> None:
    if value not in allowed:
        raise ValueError(f"targets.toml: {what}={value!r} not in {sorted(allowed)}")


def _perf_specs(cfg: dict) -> list[Spec]:
    tg = int(cfg.get("generation_length", 128))
    profile = cfg.get("context_profile", "code_python")
    force_lm = bool(cfg.get("force_lm_engine", False))
    _check(profile, CONTEXT_PROFILES, "context_profile")
    specs = []
    for pp in cfg.get("prompt_lengths", []):
        _check(pp, PERF_PROMPT_LENGTHS, "prompt_lengths")
        specs.append(perf_single(pp, tg, profile, force_lm))
    for bs in cfg.get("batch_sizes", []):
        _check(bs, PERF_BATCH_SIZES, "batch_sizes")
        specs.append(perf_batch(bs, tg, profile, force_lm))
    return specs


def _accuracy_specs(cfg: dict) -> list[Spec]:
    thinking = bool(cfg.get("enable_thinking", False))
    profile = cfg.get("sampling_profile", "deterministic")
    _check(profile, {"deterministic", "model_settings"}, "sampling_profile")
    batch = int(cfg.get("batch_size", 1))
    _check(batch, ACCURACY_BATCH_SIZES, "batch_size")
    specs = []
    for suite, n in cfg.get("suites", {}).items():
        _check(suite, ACCURACY_SUITES, "suite")
        specs.append(accuracy(suite, int(n), thinking, profile, batch))
    return specs


def _context_specs(cfg: dict) -> list[Spec]:
    targets = cfg.get("target_tokens", [])
    if isinstance(targets, int):
        targets = [targets]
    for t in targets:
        _check(t, CONTEXT_TARGETS, "target_tokens")
    return [context(t) for t in targets]


def load_targets(path: Path) -> Targets:
    data = tomllib.loads(path.read_text())
    models = data.get("models", {})
    groups = []
    for i, b in enumerate(data.get("benchmarks", [])):
        specs: list[Spec] = []
        if "perf" in b:
            specs += _perf_specs(b["perf"])
        if "accuracy" in b:
            specs += _accuracy_specs(b["accuracy"])
        if "context" in b:
            specs += _context_specs(b["context"])
        if not specs:
            raise ValueError(f"targets.toml: [[benchmarks]] #{i + 1} defines no tests")
        groups.append(TargetGroup(
            tier=int(b.get("tier", 1)),
            specs=specs,
            include=list(b.get("models", ["*"])),
            exclude=list(b.get("exclude_models", [])),
        ))
    return Targets(
        include=list(models.get("include", ["*"])),
        exclude=list(models.get("exclude", [])),
        model_types=list(models.get("types", ["llm", "vlm"])),
        groups=groups,
        quiet_minutes=float(data.get("runner", {}).get("quiet_minutes", 10)),
    )
