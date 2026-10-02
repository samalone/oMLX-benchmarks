"""Pause flag and runner liveness, shared by the runner and the CLI
through the database."""

from __future__ import annotations

import os
from datetime import datetime, timezone

from .db import DB


def pause_state(db: DB) -> tuple[bool, str | None]:
    """(paused, description)."""
    until = db.get_control("paused_until")
    if not until:
        return False, None
    reason = db.get_control("pause_reason") or ""
    if until == "forever":
        return True, f"paused until resumed {reason}".strip()
    if datetime.fromisoformat(until) > datetime.now(timezone.utc):
        return True, f"paused until {until} {reason}".strip()
    db.set_control("paused_until", None)
    return False, None


def set_pause(db: DB, until: datetime | None, reason: str = "") -> None:
    db.set_control("paused_until", until.isoformat(timespec="seconds") if until else "forever")
    db.set_control("pause_reason", reason or None)


def clear_pause(db: DB) -> None:
    db.set_control("paused_until", None)
    db.set_control("pause_reason", None)


def daemon_alive(db: DB) -> int | None:
    pid = db.get_control("daemon_pid")
    if not pid:
        return None
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return None
    return int(pid)
