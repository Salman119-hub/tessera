"""Batch jobs shared by the CLI, Airflow and Spark entry points."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from tessera.definitions import AggregateFeatureView, SnapshotFeatureView, StreamSource
from tessera.offline import prepare_events, records
from tessera.online.store import OnlineStore
from tessera.online.writer import OnlineWriter
from tessera.quality import QualityReport, run_checks
from tessera.repository import Repository
from tessera.schema import TS_COLUMN, read_source_frame
from tessera.tiles import EntityState, TileEngine

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackfillResult:
    view: str
    namespace: str
    entities: int
    events: int

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def validate_sources(
    repo: Repository, frames: Mapping[str, pd.DataFrame] | None = None
) -> list[QualityReport]:
    reports = []
    for source in repo.sources.values():
        frame = (
            frames[source.name] if frames and source.name in frames else read_source_frame(source)
        )
        reports.append(run_checks(frame, source))
    return reports


def build_states(
    view: AggregateFeatureView, events: pd.DataFrame, *, until: int | None = None
) -> dict[str, EntityState]:
    """Replay an archive into fresh per-entity states (events with ``_ts < until``)."""
    engine = TileEngine(view)
    prepared = prepare_events(view, events)
    if until is not None:
        prepared = prepared[prepared[TS_COLUMN] < until]
    prepared = prepared.sort_values(TS_COLUMN, kind="stable")
    key = view.entity.join_key
    event_id_col = view.source.event_id_field if isinstance(view.source, StreamSource) else None
    states: dict[str, EntityState] = defaultdict(engine.new_state)
    for row in records(prepared):
        event_id = row.get(event_id_col) if event_id_col else None
        engine.apply(
            states[str(row[key])],
            int(row[TS_COLUMN]),
            [row[c] for c in engine.input_columns],
            None if event_id is None else str(event_id),
        )
    return dict(states)


def backfill_aggregate_view(
    view: AggregateFeatureView,
    events: pd.DataFrame,
    store: OnlineStore,
    *,
    until: int | None = None,
) -> BackfillResult:
    """Rebuild a view's online namespace from its archive.

    Run it before the stream processor starts writing the namespace (a new
    fingerprint has an empty namespace, so this is the normal rollout order),
    or with the processor paused. Existing keys are overwritten.
    """
    states = build_states(view, events, until=until)
    keys = list(states)
    for i in range(0, len(keys), 1000):
        chunk = keys[i : i + 1000]
        current = store.read(view.namespace, chunk)
        writes = {
            k: (states[k].to_json(), current[k].version if k in current else None) for k in chunk
        }
        committed = store.write_many(view.namespace, writes)
        if len(committed) != len(writes):
            lost = sorted(set(writes) - set(committed))[:5]
            raise RuntimeError(
                f"backfill of {view.name!r} raced a concurrent writer on keys {lost}; "
                "pause the stream processor and retry"
            )
    n_events = int((events[TS_COLUMN] < until).sum()) if until is not None else len(events)
    log.info("backfilled %s: %d entities from %d events", view.name, len(states), n_events)
    return BackfillResult(view.name, view.namespace, len(states), n_events)


def backfill(
    repo: Repository,
    store: OnlineStore,
    *,
    views: Sequence[str] | None = None,
    frames: Mapping[str, pd.DataFrame] | None = None,
    until: int | None = None,
) -> list[BackfillResult]:
    """Backfill aggregate views and materialise snapshot views into the online store."""
    selected = [repo.view(v) for v in views] if views else list(repo.views.values())
    cache: dict[str, pd.DataFrame] = dict(frames or {})
    results: list[BackfillResult] = []
    writer = OnlineWriter(repo, store)
    for view in selected:
        if not isinstance(view, (AggregateFeatureView, SnapshotFeatureView)):
            continue
        source = view.source
        if source.name not in cache:
            frame = read_source_frame(source)
            run_checks(frame, source).raise_for_errors()
            cache[source.name] = frame
        frame = cache[source.name]
        if isinstance(view, AggregateFeatureView):
            results.append(backfill_aggregate_view(view, frame, store, until=until))
        else:
            subset = frame if until is None else frame[frame[TS_COLUMN] < until]
            written = writer.materialize_snapshot(view, subset)
            results.append(BackfillResult(view.name, view.namespace, written, len(subset)))
    return results
