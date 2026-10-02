"""Per-model digests of the latest results, for people and agents."""

from __future__ import annotations

import json

from .db import DB


def digest(db: DB) -> dict:
    models: dict[str, dict] = {}

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
            "context": [],
        }

    for p in db.q("SELECT * FROM v_latest_perf ORDER BY model_id, test_type DESC, pp, batch_size"):
        entry = models.setdefault(p["model_id"], {"model_id": p["model_id"], "perf": [],
                                                  "accuracy": [], "context": []})
        entry["perf"].append({
            "test": "single" if p["test_type"] == "single" else f"batch{p['batch_size']}",
            "pp": p["pp"], "tg": p["tg"], "context_profile": p["context_profile"],
            "pp_tps": _r(p["processing_tps"]), "tg_tps": _r(p["gen_tps"]),
            "ttft_ms": _r(p["ttft_ms"] if p["ttft_ms"] is not None else p["avg_ttft_ms"], 0), "e2e_s": _r(p["e2e_latency_s"], 2),
            "peak_mem_gb": _r((p["peak_memory_bytes"] or 0) / 2**30, 2),
            "settings": p["settings_fingerprint"], "omlx": p["omlx_version"],
            "measured": p["recorded_at"],
        })

    for a in db.q("SELECT * FROM v_latest_accuracy ORDER BY model_id, suite"):
        entry = models.setdefault(a["model_id"], {"model_id": a["model_id"], "perf": [],
                                                  "accuracy": [], "context": []})
        entry["accuracy"].append({
            "suite": a["suite"], "n": a["total"], "accuracy": _r(a["accuracy"], 4),
            "thinking": bool(a["thinking_used"]), "sampling": a["sampling_profile"],
            "truncated": a["truncated_count"], "time_s": _r(a["time_s"], 0),
            "settings": a["settings_fingerprint"], "omlx": a["omlx_version"],
            "measured": a["recorded_at"],
        })

    for c in db.q("SELECT * FROM v_context ORDER BY model_id, recorded_at"):
        entry = models.setdefault(c["model_id"], {"model_id": c["model_id"], "perf": [],
                                                  "accuracy": [], "context": []})
        entry["context"].append({
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
        if not (m["perf"] or m["accuracy"] or m["context"]):
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
        for c in m["context"]:
            out.append(f"\nMax verified context: {c['verified_tokens']} tokens "
                       f"(target {c['target_tokens']}, capped by {c['capped_by']})")
        out.append("")
    return "\n".join(out)


def as_json(d: dict) -> str:
    return json.dumps(d, indent=2)
