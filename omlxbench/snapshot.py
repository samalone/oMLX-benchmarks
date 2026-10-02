"""Capture what a result depends on: the environment and the model.

Snapshots are content-hashed so that identical states share one row; a change
anywhere in the hashed fields produces a new row, which is how the data keeps
track of, e.g., an oMLX upgrade or a model whose settings were changed.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .client import OmlxClient

_SECRET_MARKERS = ("api_key", "token", "password", "secret", "proxy", "sub_keys")


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def digest(obj: Any) -> str:
    return hashlib.sha256(canonical(obj).encode()).hexdigest()[:16]


def scrub(obj: Any) -> Any:
    """Drop anything credential-like before it is written to the database."""
    if isinstance(obj, dict):
        return {
            k: scrub(v)
            for k, v in obj.items()
            if not any(m in k.lower() for m in _SECRET_MARKERS)
        }
    if isinstance(obj, list):
        return [scrub(v) for v in obj]
    return obj


def macos_version() -> str:
    try:
        return subprocess.run(
            ["/usr/bin/sw_vers", "-productVersion"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return platform.mac_ver()[0]


# Global settings that plausibly change benchmark numbers. Only these feed the
# environment hash; the full (scrubbed) settings are stored alongside anyway.
_PERF_GLOBAL_SECTIONS = {
    "server": ("burst_decode_mode", "qwen4_gdn_decode_wide_proj", "gpu_keep_warm_interval",
               "preserve_mid_system_cache"),
    "memory": None,
    "scheduler": None,
    "cache": ("enabled", "hot_cache_only", "hot_cache_write_through", "hot_cache_max_size",
              "ane_compile_cache", "gdn_snapshot_storage", "gdn_ssd_split_enabled",
              "gdn_sidecar_precision", "initial_cache_blocks"),
}


@dataclass
class Environment:
    hash: str
    omlx_version: str
    engines: dict
    chip: str
    chip_variant: str
    memory_gb: int | None
    gpu_cores: int | None
    macos_version: str
    perf_global_settings: dict
    global_settings: dict
    device_info: dict


def capture_environment(client: OmlxClient) -> Environment:
    status = client.status()
    stats = client.stats()
    device = client.device_info()
    gs = client.global_settings()

    engines = {
        name: {"version": e.get("version"), "commit": e.get("commit")}
        for name, e in (stats.get("engines") or {}).items()
    }
    perf_gs: dict = {}
    for section, keys in _PERF_GLOBAL_SECTIONS.items():
        values = gs.get(section) or {}
        perf_gs[section] = values if keys is None else {k: values.get(k) for k in keys}

    identity = {
        "omlx_version": status.get("version"),
        "engines": engines,
        "chip": device.get("chip_name"),
        "chip_variant": device.get("chip_variant"),
        "memory_gb": device.get("memory_gb"),
        "gpu_cores": device.get("gpu_cores"),
        "macos_version": macos_version(),
        "perf_global_settings": perf_gs,
    }
    full_gs = scrub({k: v for k, v in gs.items() if k != "system"})
    return Environment(
        hash=digest(identity),
        omlx_version=identity["omlx_version"],
        engines=engines,
        chip=identity["chip"],
        chip_variant=identity["chip_variant"],
        memory_gb=identity["memory_gb"],
        gpu_cores=identity["gpu_cores"],
        macos_version=identity["macos_version"],
        perf_global_settings=perf_gs,
        global_settings=full_gs,
        device_info=scrub(device),
    )


# Mirrors oMLX's _FEATURE_FLAG_SPECS (admin/benchmark.py) plus the two other
# toggles that change throughput. (settings attr, flag key, detail attr)
_FEATURE_FLAGS = (
    ("dflash_enabled", "dflash", None),
    ("specprefill_enabled", "specprefill", None),
    ("turboquant_kv_enabled", "turboquant_kv", "turboquant_kv_bits"),
    ("mtp_enabled", "lightning_mtp", None),
    ("vlm_mtp_enabled", "vlm_mtp", None),
    ("qwen35_ane_prefill_enabled", "qwen35_ane_prefill", None),
    ("qwen35_oq_a8_enabled", "qwen35_oq_a8", None),
    ("moe_expert_offload_enabled", "moe_expert_offload", None),
)


def feature_flags(settings: dict) -> list[str]:
    """Active acceleration features, e.g. ["turboquant_kv_4bit"].

    This list is the settings fingerprint. It is deliberately coarse: models
    without saved settings and models whose saved settings are all defaults
    both yield [], and new oMLX settings fields do not invalidate old data.
    """
    flags = []
    for attr, key, detail in _FEATURE_FLAGS:
        if not settings.get(attr):
            continue
        if detail and settings.get(detail) is not None:
            key += f"_{float(settings[detail]):g}bit".replace(".", "_")
        flags.append(key)
    return flags


@dataclass
class ModelSnapshot:
    hash: str
    model_id: str
    model_path: str | None
    source_repo_id: str | None
    model_type: str | None
    engine_type: str | None
    arch: str | None
    quant_bits: float | None
    quant_group_size: int | None
    quant_mode: str | None
    size_bytes: int | None
    native_context: int | None
    flags: list[str]
    settings_fingerprint: str
    settings: dict
    config: dict | None
    admin_model: dict


def _read_config(model_path: str | None) -> dict | None:
    if not model_path:
        return None
    path = Path(model_path) / "config.json"
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _quantization(config: dict | None, model_id: str) -> tuple[float | None, int | None, str | None]:
    if config:
        q = config.get("quantization") or config.get("quantization_config") or {}
        if isinstance(q, dict) and q.get("bits") is not None:
            return q.get("bits"), q.get("group_size"), q.get("mode")
    m = re.search(r"(\d+(?:\.\d+)?)[-_]?bit", model_id, re.I)
    return (float(m.group(1)) if m else None), None, None


def snapshot_model(entry: dict) -> ModelSnapshot:
    settings = entry.get("settings") or {}
    config = _read_config(entry.get("model_path"))
    bits, group, mode = _quantization(config, entry["id"])
    flags = feature_flags(settings)
    # Only the model's own files feed the hash; volatile runtime fields
    # (loaded, last_access, actual_size...) live in admin_model but not here.
    identity = {
        "id": entry["id"],
        "path": entry.get("model_path"),
        "size": entry.get("estimated_size"),
        "config": config,
        "settings": settings,
    }
    volatile = {"loaded", "is_loading", "last_access", "actual_size", "actual_size_formatted",
                "pinned", "loading_started_at"}
    return ModelSnapshot(
        hash=digest(identity),
        model_id=entry["id"],
        model_path=entry.get("model_path"),
        source_repo_id=entry.get("source_repo_id") or entry.get("display_name"),
        model_type=entry.get("model_type"),
        engine_type=entry.get("engine_type"),
        arch=entry.get("config_model_type"),
        quant_bits=bits,
        quant_group_size=group,
        quant_mode=mode,
        size_bytes=entry.get("estimated_size"),
        native_context=entry.get("model_context_length"),
        flags=flags,
        settings_fingerprint=",".join(flags) or "baseline",
        settings=settings,
        config=config,
        admin_model={k: v for k, v in entry.items() if k not in volatile and k != "settings"},
    )
