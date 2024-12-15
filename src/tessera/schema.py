"""Record and frame validation against a source schema.

The stream path validates one record at a time and dead-letters failures. The
batch path validates whole frames and fails loudly. Both coerce to the same
Python types, so the values reaching the aggregators are identical.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from tessera.definitions import BatchSource, Field
from tessera.errors import SchemaError
from tessera.timeutil import series_to_epoch_seconds, to_epoch_seconds

TS_COLUMN = "_ts"
_TRUE = {"true", "1", "yes", "y", "t"}
_FALSE = {"false", "0", "no", "n", "f"}


def _native(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return bool(value is pd.NA or value is pd.NaT)


def coerce_value(value: Any, dtype: str) -> Any:
    """Coerce ``value`` to ``dtype``. Missing values become ``None``."""
    value = _native(value)
    if is_missing(value):
        return None
    if dtype == "float":
        if isinstance(value, bool):
            return float(value)
        result = float(value)
        if math.isinf(result):
            raise ValueError("infinite value")
        return result
    if dtype == "int":
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, float):
            if not value.is_integer():
                raise ValueError(f"{value!r} is not an integer")
            return int(value)
        return int(value)
    if dtype == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
        raise ValueError(f"{value!r} is not a boolean")
    if dtype == "string":
        return str(value)
    raise ValueError(f"unknown dtype {dtype!r}")


def coerce_record(record: dict[str, Any], source: BatchSource) -> dict[str, Any]:
    """Validate one event. Returns a new dict with typed values and ``_ts``.

    Unknown keys are dropped; producers may add fields before consumers
    declare them, which keeps schema evolution additive.
    """
    errors: list[str] = []
    out: dict[str, Any] = {}
    for f in source.schema:
        try:
            out[f.name] = coerce_value(record.get(f.name), f.dtype)
        except (TypeError, ValueError) as exc:
            errors.append(f"{f.name}: {exc}")
    raw_ts = record.get(source.timestamp_field)
    if is_missing(raw_ts):
        errors.append(f"{source.timestamp_field}: missing timestamp")
    else:
        try:
            out[TS_COLUMN] = to_epoch_seconds(raw_ts)
        except (TypeError, ValueError) as exc:
            errors.append(f"{source.timestamp_field}: {exc}")
    if errors:
        raise SchemaError("; ".join(errors))
    if source.timestamp_field not in out:
        out[source.timestamp_field] = raw_ts
    return out


def _coerce_series(series: pd.Series, f: Field) -> pd.Series:
    if f.dtype == "float":
        return pd.to_numeric(series, errors="raise").astype("float64")
    if f.dtype == "int":
        numeric = pd.to_numeric(series, errors="raise")
        if numeric.isna().any():
            return numeric.astype("Int64")
        return numeric.astype("int64")
    if f.dtype == "bool":
        return series.map(lambda v: coerce_value(v, "bool")).astype("object")
    return series.map(lambda v: None if is_missing(v) else str(v)).astype("object")


def coerce_frame(df: pd.DataFrame, source: BatchSource) -> pd.DataFrame:
    """Validate and type a batch frame, adding the integer ``_ts`` column."""
    required = [*source.columns, source.timestamp_field]
    missing = [c for c in dict.fromkeys(required) if c not in df.columns]
    if missing:
        raise SchemaError(f"source {source.name!r} is missing columns {missing}")
    out: pd.DataFrame = df.copy()
    for f in source.schema:
        try:
            out[f.name] = _coerce_series(out[f.name], f)
        except (TypeError, ValueError) as exc:
            raise SchemaError(f"source {source.name!r}, column {f.name!r}: {exc}") from exc
    if out[source.timestamp_field].isna().any():
        raise SchemaError(f"source {source.name!r} has rows without a timestamp")
    out[TS_COLUMN] = series_to_epoch_seconds(out[source.timestamp_field])
    return out


def read_source_frame(source: BatchSource, path: str | None = None) -> pd.DataFrame:
    """Read a source's archive (CSV or Parquet) and coerce it."""
    location = path or source.path
    frame = pd.read_parquet(location) if location.endswith(".parquet") else pd.read_csv(location)
    return coerce_frame(frame, source)
