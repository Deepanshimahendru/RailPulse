"""Small shared helpers: safe backups, checksums and JSON conversion."""
from __future__ import annotations

import datetime as _dt
import hashlib
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import config


def ensure_dir(path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def backup_file(path, backup_dir=None) -> Path | None:
    """Copy ``path`` into the backup folder with a timestamp before it is replaced.

    Returns the backup path, or None if the file does not exist (nothing to back up).
    The original file is never modified or moved.
    """
    path = Path(path)
    if not path.is_file():
        return None
    backup_dir = ensure_dir(backup_dir or config.BACKUP_DIR)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    target = backup_dir / f"{path.stem}_{stamp}{path.suffix}"
    counter = 1
    while target.exists():  # never overwrite an earlier backup
        target = backup_dir / f"{path.stem}_{stamp}_{counter}{path.suffix}"
        counter += 1
    shutil.copy2(path, target)
    return target


def sha256_file(path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_jsonable(value):
    """Convert numpy / pandas / date objects (and NaN) into plain JSON types."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if math.isnan(value) or math.isinf(value) else float(value)
    if isinstance(value, (pd.Timestamp, _dt.datetime, _dt.date)):
        return value.isoformat()
    return value
