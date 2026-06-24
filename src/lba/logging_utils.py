"""Logging helpers for LBA."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from time import strftime
from typing import Any


def default_log_dir(cwd: Path | None = None) -> Path:
    """Return the default LBA log directory."""

    if cwd is not None:
        return cwd / ".lba" / "logs"
    return Path.home() / ".lba" / "logs"


def create_run_logger(log_dir: str | Path | None = None) -> tuple[logging.Logger, Path]:
    """Create a per-run file logger."""

    directory = Path(log_dir) if log_dir is not None else default_log_dir()
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / f"lba-{strftime('%Y%m%d-%H%M%S')}-{os.getpid()}.log"

    logger = logging.getLogger(f"lba.{id(log_path)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return logger, log_path


def event_log_path_for(log_path: Path) -> Path:
    """Return the structured-event path next to a human log file."""

    return log_path.with_suffix(".jsonl")


class JsonlEventWriter:
    """Write structured LBA events as one JSON object per line."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, fields: Mapping[str, Any]) -> None:
        payload = {
            "time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "event": event,
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as file:
            json.dump(payload, file, sort_keys=True)
            file.write("\n")
