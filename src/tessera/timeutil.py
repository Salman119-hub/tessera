"""Time helpers.

All timestamps inside Tessera are integer seconds since the Unix epoch (UTC).
Integers make tile arithmetic exact and keep offline and online code paths
bit-identical; conversion happens only at the edges.
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

_DURATION_RE = re.compile(r"^\s*(\d+)\s*([smhdw])\s*$")
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86_400, "w": 604_800}


def parse_duration(value: str | int) -> int:
    """Parse ``"90s"``, ``"5m"``, ``"1h"``, ``"7d"`` or ``"2w"`` into seconds."""
    if isinstance(value, bool):
        raise ValueError(f"invalid duration: {value!r}")
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"duration must be positive, got {value}")
        return value
    match = _DURATION_RE.match(value)
    if not match:
        raise ValueError(f"invalid duration {value!r}; expected e.g. '30s', '5m', '1h', '7d'")
    seconds = int(match.group(1)) * _UNIT_SECONDS[match.group(2)]
    if seconds <= 0:
        raise ValueError(f"duration must be positive, got {value!r}")
    return seconds


def format_duration(seconds: int) -> str:
    """Inverse of :func:`parse_duration`, choosing the largest exact unit."""
    for unit in ("w", "d", "h", "m"):
        size = _UNIT_SECONDS[unit]
        if seconds % size == 0:
            return f"{seconds // size}{unit}"
    return f"{seconds}s"


def align(ts: int, hop: int) -> int:
    """Start of the tile of width ``hop`` that contains ``ts``."""
    return ts - (ts % hop)


def to_epoch_seconds(value: Any) -> int:
    """Convert a datetime, ISO string, pandas Timestamp or number to epoch seconds."""
    if isinstance(value, bool):
        raise TypeError("bool is not a timestamp")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return int(value)
    if isinstance(value, pd.Timestamp):
        stamp = value if value.tzinfo else value.tz_localize("UTC")
        return int(stamp.timestamp())
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    if isinstance(value, str):
        return to_epoch_seconds(pd.Timestamp(value))
    raise TypeError(f"cannot interpret {type(value).__name__} as a timestamp")


def series_to_epoch_seconds(series: pd.Series) -> np.ndarray:
    """Vectorised :func:`to_epoch_seconds` for a pandas column."""
    if pd.api.types.is_integer_dtype(series):
        return series.to_numpy(dtype=np.int64)
    if pd.api.types.is_float_dtype(series):
        return series.to_numpy(dtype=np.float64).astype(np.int64)
    # Resolution-independent: pandas may infer s, ms, us or ns units.
    parsed = pd.to_datetime(series, utc=True)
    delta = parsed - pd.Timestamp(0, tz="UTC")
    seconds: np.ndarray = (delta // pd.Timedelta(seconds=1)).to_numpy(dtype=np.int64)
    return seconds


def iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def now() -> int:
    return int(time.time())
