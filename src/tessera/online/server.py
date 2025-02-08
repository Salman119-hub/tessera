"""Low-latency feature retrieval for online inference."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from tessera.definitions import (
    AggregateFeatureView,
    FeatureService,
    FeatureView,
    OnDemandFeatureView,
    SnapshotFeatureView,
)
from tessera.errors import SchemaError
from tessera.online.store import OnlineStore
from tessera.repository import Repository
from tessera.schema import coerce_value
from tessera.tiles import EntityState, TileEngine
from tessera.timeutil import now


@dataclass(frozen=True)
class ServedRow:
    """One served feature vector plus what is needed to audit it later."""

    service: str
    service_fingerprint: str
    request_time: int
    entities: dict[str, str]
    request: dict[str, Any]
    features: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "service": self.service,
            "service_fingerprint": self.service_fingerprint,
            "request_time": self.request_time,
            "entities": self.entities,
            "request": self.request,
            "features": self.features,
        }


class FeatureLogger(Protocol):
    def log(self, rows: Sequence[ServedRow]) -> None: ...


class MemoryFeatureLogger:
    def __init__(self) -> None:
        self.rows: list[ServedRow] = []
        self._lock = threading.Lock()

    def log(self, rows: Sequence[ServedRow]) -> None:
        with self._lock:
            self.rows.extend(rows)


class JsonlFeatureLogger:
    """Appends served vectors to a JSON-lines file, the input to the skew audit.

    In production this is usually a Kafka topic archived to the lake. The file
    variant keeps local runs and tests self-contained.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, rows: Sequence[ServedRow]) -> None:
        lines = "".join(json.dumps(r.to_dict(), default=str) + "\n" for r in rows)
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(lines)


def read_feature_log(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL feature log from a local path or an fsspec URL (``abfss://``, ``s3://``)."""
    if "://" in str(path):
        import fsspec  # installed with tessera[azure]

        with fsspec.open(str(path), "rt", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class FeatureServer:
    """Reads feature vectors for a service from the online store.

    One batched store read is issued per stateful view, regardless of how
    many rows are requested.
    """

    def __init__(
        self,
        repo: Repository,
        store: OnlineStore,
        *,
        logger: FeatureLogger | None = None,
        clock: Callable[[], int] = now,
    ) -> None:
        self.repo = repo
        self.store = store
        self.logger = logger
        self.clock = clock
        self._engines: dict[str, TileEngine] = {}

    def _engine(self, view: AggregateFeatureView) -> TileEngine:
        engine = self._engines.get(view.name)
        if engine is None:
            engine = TileEngine(view)
            self._engines[view.name] = engine
        return engine

    @staticmethod
    def _needed_views(service: FeatureService) -> list[FeatureView]:
        ordered: dict[str, FeatureView] = {}
        for view in service.views:
            if isinstance(view, OnDemandFeatureView):
                for dep in view.depends_on:
                    ordered.setdefault(dep.name, dep)
            ordered.setdefault(view.name, view)
        return list(ordered.values())

    def get_features(
        self,
        service: FeatureService | str,
        rows: Sequence[dict[str, Any]],
        *,
        request_time: int | None = None,
    ) -> list[dict[str, Any]]:
        """Feature vectors for ``rows``. Each row holds the service's entity join
        keys and request fields. Returns one ``{qualified_name: value}`` dict per row."""
        svc = self.repo.service(service) if isinstance(service, str) else service
        t = self.clock() if request_time is None else request_time
        entity_keys = [e.join_key for e in svc.entities]
        request_fields = svc.request_fields
        for i, row in enumerate(rows):
            missing = [k for k in entity_keys if row.get(k) is None]
            missing += [f.name for f in request_fields if f.name not in row]
            if missing:
                raise SchemaError(f"row {i}: missing {missing}")
        requests = [
            {f.name: coerce_value(row.get(f.name), f.dtype) for f in request_fields} for row in rows
        ]
        computed: list[dict[str, Any]] = [{} for _ in rows]
        for view in self._needed_views(svc):
            if isinstance(view, AggregateFeatureView):
                self._read_aggregate(view, rows, t, computed)
            elif isinstance(view, SnapshotFeatureView):
                self._read_snapshot(view, rows, t, computed)
            elif isinstance(view, OnDemandFeatureView):
                for req, out in zip(requests, computed):
                    inputs = {f.name: req.get(f.name) for f in view.request}
                    for dep in view.depends_on:
                        for name in dep.feature_names:
                            inputs[dep.qualified(name)] = out.get(dep.qualified(name))
                    for name, value in view.compute(inputs).items():
                        out[view.qualified(name)] = value
        columns = svc.output_columns
        vectors = [{c: out.get(c) for c in columns} for out in computed]
        if self.logger is not None:
            self.logger.log(
                [
                    ServedRow(
                        service=svc.name,
                        service_fingerprint=svc.fingerprint,
                        request_time=t,
                        entities={k: str(row[k]) for k in entity_keys},
                        request=req,
                        features=vec,
                    )
                    for row, req, vec in zip(rows, requests, vectors)
                ]
            )
        return vectors

    def _read_aggregate(
        self,
        view: AggregateFeatureView,
        rows: Sequence[dict[str, Any]],
        t: int,
        computed: list[dict[str, Any]],
    ) -> None:
        engine = self._engine(view)
        keys = [str(row[view.entity.join_key]) for row in rows]
        stored = self.store.read(view.namespace, keys)
        states: dict[str, EntityState] = {
            key: EntityState.from_json(value.value) for key, value in stored.items()
        }
        empty = engine.new_state()
        for key, out in zip(keys, computed):
            values = engine.read(states.get(key, empty), t)
            for name, value in values.items():
                out[view.qualified(name)] = value

    def _read_snapshot(
        self,
        view: SnapshotFeatureView,
        rows: Sequence[dict[str, Any]],
        t: int,
        computed: list[dict[str, Any]],
    ) -> None:
        keys = [str(row[view.entity.join_key]) for row in rows]
        stored = self.store.read(view.namespace, keys)
        for key, out in zip(keys, computed):
            item = stored.get(key)
            blob = json.loads(item.value) if item else None
            # Same rule as the offline as-of join: strictly before t, within TTL.
            fresh = blob is not None and blob["ts"] < t and t - blob["ts"] <= view.ttl_seconds
            for name in view.feature_names:
                out[view.qualified(name)] = blob["values"].get(name) if fresh and blob else None
