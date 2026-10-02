"""Paths and connection settings.

Everything can be overridden with environment variables; `.env` in the project
root is read for anything not already set in the environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values


@dataclass(frozen=True)
class Config:
    base_url: str
    api_key: str
    db_path: Path
    targets_path: Path


def load_config() -> Config:
    dotenv = _load_dotenv(PROJECT_ROOT / ".env")

    def get(name: str, default: str = "") -> str:
        return os.environ.get(name) or dotenv.get(name) or default

    api_key = get("OMLX_API_KEY")
    if not api_key:
        raise SystemExit("OMLX_API_KEY is not set (environment or .env)")
    return Config(
        base_url=get("OMLX_URL", "http://127.0.0.1:8000").rstrip("/"),
        api_key=api_key,
        db_path=Path(get("OMLXBENCH_DB", str(PROJECT_ROOT / "data" / "omlxbench.sqlite3"))),
        targets_path=Path(get("OMLXBENCH_TARGETS", str(PROJECT_ROOT / "targets.toml"))),
    )
