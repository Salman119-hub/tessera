"""Offline feature computation: point-in-time correct training sets.

Given a *spine* (entity keys, a timestamp per row and any request fields),
:func:`get_historical_features` returns the features each row would have
seen online at that instant, and nothing later:

* aggregate views replay events, in event-time order, through the same
  :class:`~tessera.tiles.TileEngine` the online path uses, reading the state
  just before each spine timestamp;
* snapshot views take the latest row strictly before the timestamp, dropped
  when older than the view's TTL (an as-of join);
* on-demand views run their function on each assembled row.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from tessera.definitions import (
    AggregateFeatureView,
    BatchSource,
    FeatureService,
    FeatureView,
    OnDemandFeatureView,
    SnapshotFeatureView,
    StreamSource,
)
from tessera.errors import DefinitionError, SchemaError
from tessera.quality import run_checks
from tessera.schema import TS_COLUMN, coerce_value, read_source_frame
from tessera.tiles import TileEngine
from tessera.timeutil import series_to_epoch_seconds


def records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """``to_dict('records')`` with NaN normalised to ``None``.

    Online code sees ``None`` for missing values. Offline must match, or a
    function like ``x or 0`` would behave differently for NaN, which is truthy.
    """
    out: list[dict[str, Any]] = [
        {str(k): v for k, v in row.items()} for row in df.to_dict("records")
    ]
    for row in out:
        for key, value in row.items():
            if (
                (isinstance(value, float) and math.isnan(value))
                or value is pd.NA
                or value is pd.NaT
            ):
                row[key] = None
            elif isinstance(value, np.generic):
                row[key] = value.item()
    return out


def prepare_events(view: AggregateFeatureView, events: pd.DataFrame) -> pd.DataFrame:
    """Apply the view's row transform and keep the columns its aggregates need."""
    key = view.entity.join_key
    if view.transform is not None:
        transformed = [view.transform(dict(row)) for row in records(events)]
        frame = pd.DataFrame.from_records(transformed, index=events.index)
        for col in (key, TS_COLUMN):
            if col not in frame.columns:
                frame[col] = events[col]
    else:
        frame = events
    missing = [c for c in view.input_columns if c not in frame.columns]
    if missing:
        raise DefinitionError(f"view {view.name!r}: transform did not produce {missing}")
    event_id = view.source.event_id_field if isinstance(view.source, StreamSource) else None
    columns = [key, TS_COLUMN, *view.input_columns]
    if event_id and event_id in frame.columns and event_id not in columns:
        columns.append(event_id)
    selected: pd.DataFrame = frame[columns]
    return selected


def compute_aggregate_features(
    view: AggregateFeatureView, events: pd.DataFrame, spine: pd.DataFrame
) -> pd.DataFrame:
    """Aggregate features for each spine row. ``spine`` needs the join key and ``_ts``."""
    engine = TileEngine(view)
    key = view.entity.join_key
    prepared = prepare_events(view, events)
    event_id_col = view.source.event_id_field if isinstance(view.source, StreamSource) else None
    by_key: dict[str, list[tuple[int, str | None, list[Any]]]] = defaultdict(list)
    for row in records(prepared):
        values = [row[c] for c in engine.input_columns]
        event_id = row.get(event_id_col) if event_id_col else None
        by_key[str(row[key])].append(
            (int(row[TS_COLUMN]), None if event_id is None else str(event_id), values)
        )
    for events_for_key in by_key.values():
        events_for_key.sort(key=lambda e: e[0])  # stable: ties keep archive order

    out: dict[int, dict[str, Any]] = {}
    empty = engine.empty_values()
    order = spine.assign(_key=spine[key].astype(str)).sort_values(
        ["_key", TS_COLUMN], kind="stable"
    )
    for entity_key, group in order.groupby("_key", sort=False):
        evs = by_key.get(str(entity_key))
        if not evs:
            for idx in group.index:
                out[idx] = dict(empty)
            continue
        state = engine.new_state()
        pointer = 0
        for idx, t in zip(group.index, group[TS_COLUMN].to_numpy()):
            t = int(t)
            while pointer < len(evs) and evs[pointer][0] < t:
                ts, event_id, values = evs[pointer]
                engine.apply(state, ts, values, event_id)
                pointer += 1
            out[idx] = engine.read(state, t)
    frame: pd.DataFrame = pd.DataFrame.from_dict(out, orient="index").reindex(spine.index)
    frame.columns = [view.qualified(str(c)) for c in frame.columns]
    return frame


def compute_snapshot_features(
    view: SnapshotFeatureView, table: pd.DataFrame, spine: pd.DataFrame
) -> pd.DataFrame:
    """As-of join: latest row with timestamp strictly before the spine time, within TTL."""
    key = view.entity.join_key
    fields = view.feature_names
    left = pd.DataFrame(
        {"_key": spine[key].astype(str).to_numpy(), "_t": spine[TS_COLUMN].to_numpy(np.int64)},
        index=spine.index,
    )
    left["_row"] = np.arange(len(left))
    right = table[[key, TS_COLUMN, *fields]].copy()
    right["_key"] = right[key].astype(str)
    right = right.rename(columns={TS_COLUMN: "_snap_t"}).drop(columns=[key])
    right["_snap_t"] = right["_snap_t"].astype(np.int64)
    merged = pd.merge_asof(
        left.sort_values("_t", kind="stable"),
        right.sort_values("_snap_t", kind="stable"),
        left_on="_t",
        right_on="_snap_t",
        by="_key",
        allow_exact_matches=False,
        direction="backward",
    ).sort_values("_row")
    age = merged["_t"] - merged["_snap_t"]
    stale = merged["_snap_t"].isna() | (age > view.ttl_seconds)
    result: pd.DataFrame = pd.DataFrame(index=spine.index)
    for f in fields:
        column = merged[f].astype(object).where(~stale.to_numpy(), None)
        result[view.qualified(f)] = column.to_numpy()
    return result


def compute_on_demand_features(view: OnDemandFeatureView, frame: pd.DataFrame) -> pd.DataFrame:
    rows = records(frame)
    outputs = [view.compute(row) for row in rows]
    result: pd.DataFrame = pd.DataFrame.from_records(
        outputs, index=frame.index, columns=view.feature_names
    )
    result.columns = [view.qualified(str(c)) for c in result.columns]
    return result


def _load_sources(
    views: list[FeatureView],
    sources: Mapping[str, pd.DataFrame] | None,
    run_quality: bool,
) -> dict[str, pd.DataFrame]:
    loaded: dict[str, pd.DataFrame] = {}
    for view in views:
        if not isinstance(view, (AggregateFeatureView, SnapshotFeatureView)):
            continue
        source: BatchSource = view.source
        if source.name in loaded:
            continue
        if sources is not None and source.name in sources:
            frame = sources[source.name]
            if TS_COLUMN not in frame.columns:
                from tessera.schema import coerce_frame

                frame = coerce_frame(frame, source)
        else:
            frame = read_source_frame(source)
        if run_quality:
            run_checks(frame, source).raise_for_errors()
        loaded[source.name] = frame
    return loaded


def prepare_spine(
    service: FeatureService, spine: pd.DataFrame, timestamp_column: str
) -> pd.DataFrame:
    if timestamp_column not in spine.columns:
        raise SchemaError(f"spine is missing timestamp column {timestamp_column!r}")
    missing = [e.join_key for e in service.entities if e.join_key not in spine.columns]
    missing += [f.name for f in service.request_fields if f.name not in spine.columns]
    if missing:
        raise SchemaError(f"spine for service {service.name!r} is missing columns {missing}")
    out: pd.DataFrame = spine.copy()
    out[TS_COLUMN] = series_to_epoch_seconds(out[timestamp_column])
    for f in service.request_fields:
        out[f.name] = [coerce_value(v, f.dtype) for v in out[f.name]]
    return out


def get_historical_features(
    service: FeatureService,
    spine: pd.DataFrame,
    *,
    sources: Mapping[str, pd.DataFrame] | None = None,
    timestamp_column: str = "timestamp",
    run_quality: bool = True,
) -> pd.DataFrame:
    """Point-in-time correct features for every spine row.

    Returns the spine's columns followed by ``service.output_columns``, in the
    spine's original row order.
    """
    original_index = spine.index
    spine = spine.reset_index(drop=True)
    prepared = prepare_spine(service, spine, timestamp_column)
    loaded = _load_sources(_required_views(service), sources, run_quality)
    computed: dict[str, pd.DataFrame] = {}
    for view in _required_views(service):
        if isinstance(view, AggregateFeatureView):
            computed[view.name] = compute_aggregate_features(
                view, loaded[view.source.name], prepared
            )
        elif isinstance(view, SnapshotFeatureView):
            computed[view.name] = compute_snapshot_features(
                view, loaded[view.source.name], prepared
            )
    for view in _required_views(service):
        if isinstance(view, OnDemandFeatureView):
            inputs = prepared[[f.name for f in view.request]]
            deps = [computed[d.name] for d in view.depends_on]
            frame = pd.concat([inputs, *deps], axis=1) if deps else inputs
            computed[view.name] = compute_on_demand_features(view, frame)
    parts = [spine]
    for selection in service.selections():
        cols = [selection.view.qualified(f) for f in selection.features]
        parts.append(computed[selection.view.name][cols])
    result = pd.concat(parts, axis=1)
    result.index = original_index
    return result


def _required_views(service: FeatureService) -> list[FeatureView]:
    """The service's views plus on-demand dependencies, in dependency order."""
    ordered: dict[str, FeatureView] = {}
    for view in service.views:
        if isinstance(view, OnDemandFeatureView):
            for dep in view.depends_on:
                ordered.setdefault(dep.name, dep)
        ordered.setdefault(view.name, view)
    return list(ordered.values())
