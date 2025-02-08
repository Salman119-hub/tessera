"""Writes events and snapshot rows into the online store."""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from tessera.definitions import AggregateFeatureView, SnapshotFeatureView, StreamSource
from tessera.errors import ConflictError, SchemaError
from tessera.offline import records
from tessera.online.store import OnlineStore
from tessera.repository import Repository
from tessera.schema import TS_COLUMN, coerce_record
from tessera.tiles import ApplyStatus, EntityState, TileEngine

log = logging.getLogger(__name__)


@dataclass
class IngestResult:
    received: int = 0
    invalid: list[tuple[dict[str, Any], str]] = field(default_factory=list)
    status_counts: Counter[str] = field(default_factory=Counter)
    conflicts_retried: int = 0
    max_event_ts: int | None = None

    @property
    def applied(self) -> int:
        return self.status_counts[ApplyStatus.APPLIED.value]

    @property
    def duplicates(self) -> int:
        return self.status_counts[ApplyStatus.DUPLICATE.value]

    @property
    def late(self) -> int:
        return self.status_counts[ApplyStatus.LATE.value]

    def merge(self, other: IngestResult) -> None:
        self.received += other.received
        self.invalid.extend(other.invalid)
        self.status_counts.update(other.status_counts)
        self.conflicts_retried += other.conflicts_retried
        if other.max_event_ts is not None:
            self.max_event_ts = max(self.max_event_ts or other.max_event_ts, other.max_event_ts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "received": self.received,
            "invalid": len(self.invalid),
            "applied": self.applied,
            "duplicates": self.duplicates,
            "late": self.late,
            "conflicts_retried": self.conflicts_retried,
        }


class OnlineWriter:
    """Applies events to aggregate state and materialises snapshot views.

    Writes are compare-and-set per entity key. On a conflict the writer
    re-reads the latest state and re-applies the batch's events for that key,
    up to ``max_retries`` times. Re-applying is safe because deduplication by
    event id makes ``apply`` idempotent.
    """

    def __init__(self, repo: Repository, store: OnlineStore, *, max_retries: int = 5) -> None:
        self.repo = repo
        self.store = store
        self.max_retries = max_retries
        self._engines: dict[str, TileEngine] = {}

    def engine(self, view: AggregateFeatureView) -> TileEngine:
        engine = self._engines.get(view.name)
        if engine is None:
            engine = TileEngine(view)
            self._engines[view.name] = engine
        return engine

    def ingest(self, source_name: str, raw_records: Iterable[dict[str, Any]]) -> IngestResult:
        """Validate records against the source schema and apply them to every
        aggregate view that reads the source."""
        source = self.repo.sources[source_name]
        result = IngestResult()
        valid: list[dict[str, Any]] = []
        for record in raw_records:
            result.received += 1
            try:
                valid.append(coerce_record(record, source))
            except SchemaError as exc:
                result.invalid.append((record, str(exc)))
        if valid:
            result.max_event_ts = max(r[TS_COLUMN] for r in valid)
        for view in self.repo.aggregate_views_for_source(source_name):
            self._apply_view(view, valid, result)
        return result

    def _apply_view(
        self, view: AggregateFeatureView, rows: list[dict[str, Any]], result: IngestResult
    ) -> None:
        engine = self.engine(view)
        key_col = view.entity.join_key
        event_id_col = view.source.event_id_field if isinstance(view.source, StreamSource) else None
        grouped: dict[str, list[tuple[int, str | None, list[Any]]]] = defaultdict(list)
        for row in rows:
            prepared = view.transform(dict(row)) if view.transform else row
            if prepared.get(key_col) is None:
                result.invalid.append((row, f"{view.name}: missing join key {key_col!r}"))
                continue
            values = [prepared.get(c) for c in engine.input_columns]
            event_id = row.get(event_id_col) if event_id_col else None
            grouped[str(prepared[key_col])].append(
                (int(row[TS_COLUMN]), None if event_id is None else str(event_id), values)
            )
        pending = dict(grouped)
        for attempt in range(self.max_retries + 1):
            if not pending:
                return
            current = self.store.read(view.namespace, pending.keys())
            writes: dict[str, tuple[bytes, int | None]] = {}
            statuses: dict[str, Counter[str]] = {}
            for key, events in pending.items():
                stored = current.get(key)
                state = EntityState.from_json(stored.value) if stored else engine.new_state()
                counts: Counter[str] = Counter()
                for ts, event_id, values in events:
                    counts[engine.apply(state, ts, values, event_id).value] += 1
                statuses[key] = counts
                if counts[ApplyStatus.APPLIED.value]:
                    writes[key] = (state.to_json(), stored.version if stored else None)
            committed = self.store.write_many(view.namespace, writes) if writes else {}
            retry: dict[str, list[tuple[int, str | None, list[Any]]]] = {}
            for key, counts in statuses.items():
                if key in writes and key not in committed:
                    retry[key] = pending[key]
                else:
                    result.status_counts.update(counts)
            if retry:
                result.conflicts_retried += len(retry)
                log.debug(
                    "view %s: %d key(s) conflicted (attempt %d)", view.name, len(retry), attempt
                )
            pending = retry
        raise ConflictError(
            f"view {view.name!r}: gave up on {len(pending)} key(s) after {self.max_retries} retries"
        )

    def materialize_snapshot(self, view: SnapshotFeatureView, frame: pd.DataFrame) -> int:
        """Write the latest row per entity. Rows older than what is stored are skipped,
        so replays and out-of-order loads never regress a value."""
        key_col = view.entity.join_key
        latest: dict[str, dict[str, Any]] = {}
        for row in records(frame[[key_col, TS_COLUMN, *view.feature_names]]):
            key = str(row[key_col])
            if key not in latest or row[TS_COLUMN] >= latest[key][TS_COLUMN]:
                latest[key] = row
        written = 0
        pending = latest
        for _ in range(self.max_retries + 1):
            if not pending:
                break
            current = self.store.read(view.namespace, pending.keys())
            writes: dict[str, tuple[bytes, int | None]] = {}
            for key, row in pending.items():
                stored = current.get(key)
                if stored is not None and json.loads(stored.value)["ts"] > row[TS_COLUMN]:
                    continue
                blob = {
                    "ts": int(row[TS_COLUMN]),
                    "values": {f: row[f] for f in view.feature_names},
                }
                writes[key] = (json.dumps(blob).encode(), stored.version if stored else None)
            committed = self.store.write_many(view.namespace, writes)
            written += len(committed)
            pending = {k: pending[k] for k in writes if k not in committed}
        return written
