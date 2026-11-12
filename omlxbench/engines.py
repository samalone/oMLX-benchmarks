"""TensorFold (tensorfold.dev), the second inference engine.

TensorFold serves one model per process: its app runs `tensorfold-native serve
<model dir> --port N --name <org>/<name>`, and the user switches models in the
app. There is no admin or benchmark API, only the OpenAI API, /health and
Prometheus /metrics, so only the suites omlxbench drives itself (harness.py)
can run on it.

`TensorFoldClient` offers the slice of `OmlxClient` that the runner and the
harness use, with `status()` shaped like oMLX's /api/status so the same
traffic and yield logic applies. Verified against tensorfold-native 1.2.0:
- `e2e_request_latency_seconds_count` counts finished inference requests
  only; polling /metrics, /health or /v1/models does not move it (unlike
  `requests_total`, which counts every HTTP reply).
- The "model" field of a chat request is not checked: whatever is loaded
  answers. So a model switch shows only in /v1/models.
- A top-level `enable_thinking` is ignored; `chat_template_kwargs:
  {"enable_thinking": ...}` works (as does `reasoning_effort: "none"`).
"""

from __future__ import annotations

import functools
import json
import platform
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx

from .client import OmlxError, _detail
from .snapshot import (Environment, ModelSnapshot, _quantization, _read_config, digest,
                       macos_version)

ENGINE = "tensorfold"

# Serve options that take no value (from `tensorfold-native serve --help`).
_BOOL_FLAGS = {
    "-h", "--help", "--metrics-open", "--dashboard", "--prompt-cache-over-cap", "--learn",
    "--learn-conversations", "--recall", "--recall-read-only", "--prompt-seed", "--thinking",
    "--no-thinking", "--loop-guard", "--loop-guard-stop", "--no-drafts", "--slide",
    "--cortheon", "--no-update-check",
}
# Options that say nothing about performance (or are secret).
_IRRELEVANT = {"--host", "--port", "--name", "--alias", "--allowed-host", "--allowed-origin",
               "--api-key", "--api-key-file", "--dashboard", "--metrics-open",
               "--no-update-check", "--record", "--chat-template"}
# Options that change speed: they define the settings fingerprint, as oMLX's
# acceleration flags do. Anything else perf-relevant feeds the environment hash.
_SPEED_FLAGS = ("--speed-up", "--no-drafts", "--drafter", "--drafter-bits", "--parallel",
                "--segments", "--slide", "--cortheon", "--backend")


class NoRejections:
    """TensorFold has no server log to count rejected requests from."""

    def new(self) -> int:
        return 0


@dataclass
class Served:
    """What the `tensorfold-native serve` process on our port is running."""

    name: str | None        # --name, what /v1/models reports
    model_path: str | None  # the positional model directory
    binary: str | None
    options: dict           # option -> value (True for flags), secrets removed


def parse_serve_argv(argv: list[str]) -> Served:
    try:
        i = argv.index("serve")
    except ValueError:
        return Served(None, None, argv[0] if argv else None, {})
    options: dict = {}
    model = None
    rest = argv[i + 1:]
    j = 0
    while j < len(rest):
        tok = rest[j]
        if tok.startswith("-"):
            if "=" in tok:
                key, _, val = tok.partition("=")
            elif tok in _BOOL_FLAGS:
                key, val = tok, True
            else:
                key, val = tok, rest[j + 1] if j + 1 < len(rest) else None
                j += 1
            options[key] = val
        elif model is None:
            model = tok
        j += 1
    name = options.get("--name")
    clean = {k: v for k, v in options.items() if "key" not in k}
    return Served(name if isinstance(name, str) else None, model, argv[0], clean)


def model_id_for(path: str | None, served_name: str | None) -> str:
    """The oMLX-style model id: the model directory's name, or the repo name
    for a Hugging Face cache path (models--org--name/snapshots/<hash>)."""
    if path:
        p = Path(path)
        for part in reversed(p.parts):
            if part.startswith("models--"):
                return part.split("--")[-1]
        if p.parent.name == "snapshots":
            return p.parent.parent.name.split("--")[-1]
        return p.name
    return (served_name or "unknown").split("/")[-1]


class TensorFoldClient:
    engine_name = ENGINE

    def __init__(self, base_url: str, api_key: str, timeout: float = 10.0):
        """An empty `base_url` means: wherever the serve process listens."""
        self.fixed_url = base_url or None
        self.api_key = api_key
        self.timeout = timeout
        self.base_url = ""
        self.http: httpx.Client | None = None
        self._connect(base_url)

    def _connect(self, base_url: str) -> None:
        if self.http is not None and base_url == self.base_url:
            return
        if self.http is not None:
            self.http.close()
        self.base_url = base_url
        self.port = urlparse(base_url).port if base_url else None
        self.http = httpx.Client(base_url=base_url or "http://127.0.0.1:1", timeout=self.timeout,
                                 headers={"Authorization": f"Bearer {self.api_key}"})

    def discover(self) -> bool:
        """Point at the serve process's port (unless TENSORFOLD_URL fixes it).
        False if no TensorFold server is running."""
        if self.fixed_url:
            self._connect(self.fixed_url)
            return True
        s = _find_serve(None)
        if s.binary is None:
            return False
        host = s.options.get("--host") or "127.0.0.1"
        port = s.options.get("--port")
        if not port:
            return False
        self._connect(f"http://{host}:{port}")
        return True

    def close(self) -> None:
        if self.http is not None:
            self.http.close()

    def _get(self, path: str) -> httpx.Response:
        resp = self.http.get(path)
        if resp.status_code >= 400:
            raise OmlxError(resp.status_code, _detail(resp), path)
        return resp

    def health(self) -> dict:
        if not self.discover():
            raise OSError("no tensorfold-native serve process is running")
        resp = self.http.get("/health")
        body = resp.json() if resp.content else {}
        ok = resp.status_code == 200 and body.get("status") == "ok"
        if ok:
            # /health needs no key; check ours. The app issues a new key when
            # it restarts the server, so a stale .env shows up here.
            auth = self.http.get("/v1/models")
            if auth.status_code in (401, 403):
                return {"http_status": auth.status_code,
                        "status": "API key rejected: update TENSORFOLD_API_KEY in .env "
                                  "(TensorFold issues a new key when it restarts)"}
        return {"http_status": resp.status_code, "status": "healthy" if ok else body.get("status")}

    def models(self) -> list[dict]:
        return self._get("/v1/models").json().get("data") or []

    def metrics(self) -> dict[str, float]:
        """Prometheus samples summed over labels, keyed without the prefix."""
        out: dict[str, float] = {}
        for line in self._get("/metrics").text.splitlines():
            if not line or line.startswith("#"):
                continue
            m = re.match(r"tensorfold:([A-Za-z0-9_]+)(\{[^}]*\})?\s+(\S+)", line)
            if m:
                try:
                    out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(3))
                except ValueError:
                    pass
        return out

    def status(self) -> dict:
        """oMLX /api/status shape: what TrafficWatch and the harness read."""
        if not self.discover():
            raise OSError("no tensorfold-native serve process is running")
        m = self.metrics()
        return {
            "total_requests": int(m.get("e2e_request_latency_seconds_count", 0)),
            "active_requests": int(m.get("requests_running", 0) + m.get("requests_waiting", 0)),
            "loaded_models": [d["id"] for d in self.models()],
            "models_loading": False,
            "process_footprint_peak_bytes": m.get("process_footprint_peak_bytes"),
            "process_footprint_bytes": m.get("process_footprint_bytes"),
        }

    def served(self) -> Served:
        """The serve process listening on our port, from `ps`."""
        return _find_serve(self.port)


def _find_serve(port: int | None) -> Served:
    """The `tensorfold-native serve` process (on `port`, if given)."""
    try:
        out = subprocess.run(["/bin/ps", "-axww", "-o", "command="], capture_output=True,
                             text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return Served(None, None, None, {})
    for line in out.splitlines():
        argv = line.split()
        if not argv or not argv[0].endswith("tensorfold-native") or "serve" not in argv:
            continue
        s = parse_serve_argv(argv)
        p = s.options.get("--port")
        if port is None or p is None or str(p) == str(port):
            return s
    return Served(None, None, None, {})


@functools.cache
def _binary_version(binary: str) -> str | None:
    try:
        out = subprocess.run([binary, "--version"], capture_output=True, text=True,
                             timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return out.split()[-1] if out else None


@functools.cache
def _hardware() -> dict:
    def sysctl(name: str) -> str:
        try:
            return subprocess.run(["/usr/sbin/sysctl", "-n", name], capture_output=True,
                                  text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    brand = sysctl("machdep.cpu.brand_string")  # e.g. "Apple M1 Max"
    m = re.match(r"Apple (M\d+)\s*(.*)", brand)
    chip, variant = (m.group(1), m.group(2)) if m else (brand or platform.machine(), "")
    mem = sysctl("hw.memsize")
    cores = None
    try:
        sp = json.loads(subprocess.run(["/usr/sbin/system_profiler", "SPDisplaysDataType",
                                        "-json"], capture_output=True, text=True,
                                       timeout=30).stdout)
        cores = int(sp["SPDisplaysDataType"][0].get("sppci_cores"))
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError, TypeError):
        pass
    return {"chip_name": chip, "chip_variant": variant,
            "memory_gb": round(int(mem) / 2**30) if mem.isdigit() else None, "gpu_cores": cores}


def _dir_size(path: str | None) -> int | None:
    if not path:
        return None
    try:
        return sum(f.stat().st_size for f in Path(path).iterdir()
                   if f.suffix in (".safetensors", ".bin", ".gguf")) or None
    except OSError:
        return None


def speed_fingerprint(options: dict) -> list[str]:
    """Non-default speed options, e.g. ["no_drafts", "speed_up_x"]."""
    flags = []
    for opt in _SPEED_FLAGS:
        if opt not in options:
            continue
        val = options[opt]
        name = opt.lstrip("-").replace("-", "_")
        flags.append(name if val is True else f"{name}_{re.sub(r'[^A-Za-z0-9.]+', '_', str(val))}")
    return flags


def capture_tensorfold(client: TensorFoldClient) -> tuple[Environment, list[ModelSnapshot]]:
    served = client.served()
    listed = client.models()
    hw = _hardware()
    version = _binary_version(served.binary) if served.binary else None
    perf_options = {k: v for k, v in served.options.items() if k not in _IRRELEVANT}
    identity = {
        "engine": ENGINE,
        "engine_version": version,
        "chip": hw["chip_name"], "chip_variant": hw["chip_variant"],
        "memory_gb": hw["memory_gb"], "gpu_cores": hw["gpu_cores"],
        "macos_version": macos_version(),
        "perf_global_settings": perf_options,
    }
    env = Environment(
        hash=digest(identity), omlx_version=None, engines={ENGINE: {"version": version}},
        chip=hw["chip_name"], chip_variant=hw["chip_variant"], memory_gb=hw["memory_gb"],
        gpu_cores=hw["gpu_cores"], macos_version=identity["macos_version"],
        perf_global_settings=perf_options, global_settings={"serve_options": served.options},
        device_info=hw, engine=ENGINE, engine_version=version,
    )
    models = []
    for entry in listed:
        name = entry.get("id")
        path = served.model_path if (served.name in (None, name)) else None
        model_id = model_id_for(path, name)
        config = _read_config(path)
        bits, group, mode = _quantization(config, model_id)
        flags = speed_fingerprint(served.options)
        settings = {"serve_options": {k: v for k, v in served.options.items()
                                      if k not in _IRRELEVANT}}
        size = _dir_size(path)
        admin = {**{k: v for k, v in entry.items() if k not in ("id", "created")},
                 "id": model_id, "model_type": "llm", "served_name": name}
        models.append(ModelSnapshot(
            hash=digest({"engine": ENGINE, "id": model_id, "path": path, "size": size,
                         "config": config, "settings": settings}),
            model_id=model_id, model_path=path, source_repo_id=name, model_type="llm",
            engine_type=ENGINE, arch=(config or {}).get("model_type"), quant_bits=bits,
            quant_group_size=group, quant_mode=mode, size_bytes=size,
            native_context=entry.get("context_length") or (config or {}).get(
                "max_position_embeddings"),
            flags=flags, settings_fingerprint=",".join(flags) or "baseline", settings=settings,
            config=config, admin_model=admin, engine=ENGINE, api_model=name,
        ))
    return env, models
