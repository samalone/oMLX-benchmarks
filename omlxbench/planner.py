"""targets × models − what we already have → ordered list of oMLX runs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .db import DB
from .snapshot import ModelSnapshot
from .specs import Spec, Targets

MAX_ATTEMPTS = 2  # per (model, spec, settings, environment) before giving up


@dataclass
class WorkUnit:
    """One oMLX benchmark run covering one or more missing specs."""

    tier: int
    model: ModelSnapshot
    kind: str  # perf | accuracy | context
    specs: list[Spec]

    def request(self) -> dict:
        model_id = self.model.model_id
        if self.kind == "perf":
            p0 = self.specs[0].p
            pps = sorted({s.p["pp"] for s in self.specs if s.kind == "perf.single"})
            batches = sorted({s.p["batch_size"] for s in self.specs if s.kind == "perf.batch"})
            if not pps:
                # oMLX requires at least one prompt length; batch tests run at pp1024.
                pps = [1024]
            return {
                "model_id": model_id,
                "prompt_lengths": pps,
                "generation_length": p0["tg"],
                "batch_sizes": batches,
                "context_profile": p0["context_profile"],
                "force_lm_engine": p0["force_lm_engine"],
            }
        if self.kind == "accuracy":
            p = self.specs[0].p
            return {
                "model_id": model_id,
                "benchmarks": {p["suite"]: p["sample_size"]},
                "batch_size": p["batch_size"],
                "enable_thinking": p["enable_thinking"],
                "sampling_profile": p["sampling_profile"],
            }
        return {"model_id": model_id, "target_tokens": self.specs[0].p["target_tokens"]}

    def describe(self) -> str:
        return f"[tier {self.tier}] {self.model.model_id}: " + ", ".join(s.label() for s in self.specs)


def spec_state(db: DB, model_id: str, spec: Spec, fingerprint: str, env_hash: str) -> str:
    """'done', 'failed' (gave up under this environment) or 'missing'."""
    row = db.q1(
        "SELECT state, attempts, failed_environment_hash FROM spec_status"
        " WHERE model_id = ? AND spec_key = ? AND settings_fingerprint = ?",
        model_id, spec.key, fingerprint,
    )
    if row is None:
        return "missing"
    if row["state"] == "ok":
        return "done"
    if row["failed_environment_hash"] == env_hash and row["attempts"] >= MAX_ATTEMPTS:
        return "failed"
    return "missing"


def plan(db: DB, targets: Targets, models: list[ModelSnapshot], env_hash: str) -> list[WorkUnit]:
    wanted = [m for m in models if targets.wants_model(
        {"id": m.model_id, "model_type": m.model_type, "is_helper": m.admin_model.get("is_helper")}
    )]
    # Smallest model first within a tier: fastest to fill in, so a comparable
    # baseline across all models exists as early as possible.
    wanted.sort(key=lambda m: (m.size_bytes or 0, m.model_id))

    # tier -> model -> specs, de-duplicated across groups
    by_tier: dict[int, dict[str, list[Spec]]] = defaultdict(lambda: defaultdict(list))
    seen: set[tuple[str, str]] = set()
    for group in sorted(targets.groups, key=lambda g: g.tier):
        for m in wanted:
            if not group.applies_to(m.model_id):
                continue
            for spec in group.specs:
                if (m.model_id, spec.key) in seen:
                    continue
                seen.add((m.model_id, spec.key))
                if spec_state(db, m.model_id, spec, m.settings_fingerprint, env_hash) == "missing":
                    by_tier[group.tier][m.model_id].append(spec)

    units: list[WorkUnit] = []
    model_by_id = {m.model_id: m for m in wanted}
    for tier in sorted(by_tier):
        for m in wanted:
            specs = by_tier[tier].get(m.model_id)
            if not specs:
                continue
            units.extend(_group(tier, model_by_id[m.model_id], specs))
    return units


def _group(tier: int, model: ModelSnapshot, specs: list[Spec]) -> list[WorkUnit]:
    units: list[WorkUnit] = []
    # All perf specs sharing tg/profile/engine go in one run: one model load,
    # and partial results survive a cancel, so batching them loses nothing.
    perf: dict[tuple, list[Spec]] = defaultdict(list)
    for s in specs:
        if s.run_kind == "perf":
            p = s.p
            perf[(p["tg"], p["context_profile"], p["force_lm_engine"])].append(s)
    for group in perf.values():
        group.sort(key=lambda s: (s.kind != "perf.single", s.p.get("pp", 0), s.p.get("batch_size", 0)))
        units.append(WorkUnit(tier, model, "perf", group))
    # One accuracy suite per run: a cancel loses at most the suite in progress.
    for s in specs:
        if s.run_kind == "accuracy":
            units.append(WorkUnit(tier, model, "accuracy", [s]))
    for s in specs:
        if s.run_kind == "context":
            units.append(WorkUnit(tier, model, "context", [s]))
    return units
