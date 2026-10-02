"""Per-model digests of the latest results, for people and agents."""

from __future__ import annotations

from .db import DB


def digest(db: DB) -> dict:
    models: dict[str, dict] = {}

    def entry(model_id: str) -> dict:
        return models.setdefault(model_id, {"model_id": model_id, "perf": [], "accuracy": [],
                                            "tools": [], "agent_turns": [], "context": []})

    for m in db.q(
        "SELECT * FROM model_snapshots s WHERE s.id = (SELECT max(id) FROM model_snapshots"
        " WHERE model_id = s.model_id) ORDER BY size_bytes"
    ):
        models[m["model_id"]] = {
            "model_id": m["model_id"],
            "source_repo_id": m["source_repo_id"],
            "arch": m["arch"],
            "model_type": m["model_type"],
            "quant_bits": m["quant_bits"],
            "size_gb": round((m["size_bytes"] or 0) / 1e9, 2),
            "native_context": m["native_context"],
            "current_settings_fingerprint": m["settings_fingerprint"],
            "perf": [],
            "accuracy": [],
            "tools": [],
            "agent_turns": [],
            "context": [],
        }

    for p in db.q("SELECT * FROM v_latest_perf ORDER BY model_id, test_type DESC, pp, batch_size"):
        entry(p["model_id"])["perf"].append({
            "test": "single" if p["test_type"] == "single" else f"batch{p['batch_size']}",
            "pp": p["pp"], "tg": p["tg"], "context_profile": p["context_profile"],
            "pp_tps": _r(p["processing_tps"]), "tg_tps": _r(p["gen_tps"]),
            # batch tests report a mean time to first token instead
            "ttft_ms": _r(p["ttft_ms"] if p["ttft_ms"] is not None else p["avg_ttft_ms"], 0), "e2e_s": _r(p["e2e_latency_s"], 2),
            "peak_mem_gb": _r((p["peak_memory_bytes"] or 0) / 2**30, 2),
            "settings": p["settings_fingerprint"], "omlx": p["omlx_version"],
            "measured": p["recorded_at"],
        })

    for a in db.q("SELECT * FROM v_latest_accuracy ORDER BY model_id, suite"):
        entry(a["model_id"])["accuracy"].append({
            "suite": a["suite"], "n": a["total"], "accuracy": _r(a["accuracy"], 4),
            "thinking": bool(a["thinking_used"]), "sampling": a["sampling_profile"],
            "truncated": a["truncated_count"], "time_s": _r(a["time_s"], 0),
            "settings": a["settings_fingerprint"], "omlx": a["omlx_version"],
            "measured": a["recorded_at"],
        })

    for t in db.q("SELECT * FROM v_tools ORDER BY model_id, category"):
        entry(t["model_id"])["tools"].append({
            "category": t["category"], "n": t["n"], "accuracy": _r(t["accuracy"], 4),
            "avg_completion_tokens": _r(t["avg_completion_tokens"], 0),
            "avg_time_s": _r(t["avg_time_s"], 1), "settings": t["settings_fingerprint"],
            "measured": t["recorded_at"],
        })

    for a in db.q("SELECT * FROM v_agent_turns v WHERE run_id = (SELECT max(run_id) FROM"
                  " v_agent_turns w WHERE w.model_id = v.model_id"
                  " AND w.settings_fingerprint = v.settings_fingerprint)"):
        entry(a["model_id"])["agent_turns"].append({
            "first_prompt_tokens": a["first_prompt_tokens"], "cold_ttft_s": _r(a["cold_ttft_s"]),
            "cold_prefill_tps": _r(a["cold_prefill_tps"], 0),
            "warm_ttft_s": _r(a["warm_ttft_s"]), "warm_new_tokens": _r(a["warm_new_tokens"], 0),
            "warm_prefill_tps": _r(a["warm_prefill_tps"], 0),
            "last_prompt_tokens": a["last_prompt_tokens"], "last_ttft_s": _r(a["last_ttft_s"]),
            "settings": a["settings_fingerprint"], "measured": a["recorded_at"],
        })

    for c in db.q("SELECT * FROM v_context ORDER BY model_id, recorded_at"):
        entry(c["model_id"])["context"].append({
            "target_tokens": c["target_tokens"], "verified_tokens": c["verified_tokens"],
            "capped_by": c["capped_by"], "prefill_tps": _r(c["prefill_tps"]),
            "settings": c["settings_fingerprint"], "measured": c["recorded_at"],
        })

    env = db.q1("SELECT * FROM environments ORDER BY last_seen DESC LIMIT 1")
    return {
        "hardware": None if env is None else {
            "chip": f"{env['chip']} {env['chip_variant']}".strip(),
            "memory_gb": env["memory_gb"], "gpu_cores": env["gpu_cores"],
            "macos": env["macos_version"], "omlx": env["omlx_version"],
        },
        "models": list(models.values()),
    }


def _r(v, nd: int = 1):
    return None if v is None else round(v, nd)


def markdown(d: dict) -> str:
    out = []
    hw = d.get("hardware")
    if hw:
        out.append(f"# oMLX benchmarks — {hw['chip']}, {hw['memory_gb']} GB, macOS {hw['macos']}, "
                   f"oMLX {hw['omlx']}\n")
    for m in d["models"]:
        if not (m["perf"] or m["accuracy"] or m["tools"] or m["agent_turns"] or m["context"]):
            continue
        bits = m.get("quant_bits")
        out.append(f"## {m['model_id']}")
        out.append(f"{m.get('arch')} · {bits:g}-bit · {m.get('size_gb')} GB"
                   if bits else f"{m.get('arch')} · {m.get('size_gb')} GB")
        if m["perf"]:
            out.append("\n| test | pp | pp tok/s | tg tok/s | ttft ms | peak GB | settings |")
            out.append("|---|---|---|---|---|---|---|")
            for p in m["perf"]:
                out.append(f"| {p['test']} | {p['pp']} | {p['pp_tps']} | {p['tg_tps']} | "
                           f"{p['ttft_ms']} | {p['peak_mem_gb']} | {p['settings']} |")
        if m["accuracy"]:
            out.append("\n| suite | n | accuracy | thinking | settings |")
            out.append("|---|---|---|---|---|")
            for a in m["accuracy"]:
                out.append(f"| {a['suite']} | {a['n']} | {100 * (a['accuracy'] or 0):.1f}% | "
                           f"{a['thinking']} | {a['settings']} |")
        if m["tools"]:
            out.append("\n| tool calling | n | accuracy | avg s |")
            out.append("|---|---|---|---|")
            for t in m["tools"]:
                out.append(f"| {t['category']} | {t['n']} | {100 * (t['accuracy'] or 0):.1f}% | "
                           f"{t['avg_time_s']} |")
        for a in m["agent_turns"]:
            out.append(f"\nAgent turns ({a['first_prompt_tokens']}→{a['last_prompt_tokens']} tokens): "
                       f"session start {a['cold_ttft_s']}s to first token; each later step "
                       f"(~{a['warm_new_tokens']:.0f} new tokens) {a['warm_ttft_s']}s on average, "
                       f"{a['last_ttft_s']}s at the end")
        for c in m["context"]:
            out.append(f"\nMax verified context: {c['verified_tokens']} tokens "
                       f"(target {c['target_tokens']}, capped by {c['capped_by']})")
        out.append("")
    return "\n".join(out)


