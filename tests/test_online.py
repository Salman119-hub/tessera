from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from tessera.errors import ConflictError, SchemaError
from tessera.jobs import backfill, build_states
from tessera.offline import get_historical_features, records
from tessera.online.server import (
    FeatureServer,
    JsonlFeatureLogger,
    MemoryFeatureLogger,
    read_feature_log,
)
from tessera.online.store import InMemoryStore, SQLiteStore, VersionedValue, build_store
from tessera.online.writer import OnlineWriter
from tessera.repository import Repository
from tessera.schema import coerce_frame

from .conftest import T0, click, clicks, profiles, service, user_clicks, user_profile


@pytest.fixture(params=["memory", "sqlite"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Any:
    s = InMemoryStore() if request.param == "memory" else SQLiteStore(tmp_path / "online.db")
    yield s
    s.close()


class TestStores:
    def test_compare_and_set(self, store: Any) -> None:
        assert store.write("ns", "k", b"v1", None) == 1
        with pytest.raises(ConflictError):
            store.write("ns", "k", b"again", None)
        assert store.write("ns", "k", b"v2", 1) == 2
        with pytest.raises(ConflictError):
            store.write("ns", "k", b"stale", 1)
        assert store.read("ns", ["k", "missing"]) == {"k": VersionedValue(2, b"v2")}
        assert store.read("other", ["k"]) == {}

    def test_write_many_partial_success(self, store: Any) -> None:
        store.write("ns", "a", b"1", None)
        committed = store.write_many("ns", {"a": (b"2", None), "b": (b"1", None)})
        assert committed == {"b": 1}
        assert store.count("ns") == 2

    def test_build_store(self, tmp_path: Path) -> None:
        assert isinstance(build_store("memory://"), InMemoryStore)
        sqlite = build_store(f"sqlite:///{tmp_path / 'x.db'}")
        assert isinstance(sqlite, SQLiteStore)
        sqlite.close()
        with pytest.raises(ValueError, match="unsupported"):
            build_store("cassandra://x")

    def test_sqlite_large_batched_read(self, tmp_path: Path) -> None:
        s = SQLiteStore(tmp_path / "big.db")
        s.write_many("ns", {f"k{i}": (b"x", None) for i in range(1200)})
        assert len(s.read("ns", [f"k{i}" for i in range(1200)])) == 1200
        s.close()


class TestWriter:
    def test_ingest_is_idempotent(
        self, repo: Repository, store: Any, click_events: list[dict[str, Any]]
    ) -> None:
        writer = OnlineWriter(repo, store)
        first = writer.ingest("clicks", click_events)
        assert first.applied == len(click_events)
        before = store.read(user_clicks.namespace, ["u1", "u2"])
        again = writer.ingest("clicks", click_events[-50:])
        # Recent redeliveries are caught by event id; older ones fall outside the
        # lateness horizon. Either way nothing is applied twice.
        assert again.applied == 0
        assert again.duplicates > 0
        assert again.duplicates + again.late == 50
        assert store.read(user_clicks.namespace, ["u1", "u2"]) == before

    def test_invalid_and_late_records(self, repo: Repository, store: Any) -> None:
        writer = OnlineWriter(repo, store)
        result = writer.ingest(
            "clicks",
            [
                click(1, "u1", T0 + 3600, 1.0),
                {"event_id": "bad", "user_id": "u1", "value": "NaNx", "ts": T0},
                click(2, "u1", T0, 1.0),  # an hour late; lateness is 5 minutes
                {"event_id": "nokey", "value": 1.0, "ts": T0 + 3600},
            ],
        )
        assert result.applied == 1
        assert result.late == 1
        assert len(result.invalid) == 2
        assert result.to_dict()["invalid"] == 2

    def test_retries_on_conflict(self, repo: Repository) -> None:
        class FlakyStore(InMemoryStore):
            """Loses the first write race for every key."""

            def __init__(self) -> None:
                super().__init__()
                self.raced: set[str] = set()

            def write_many(
                self, namespace: str, items: Mapping[str, tuple[bytes, int | None]]
            ) -> dict[str, int]:
                for key in items:
                    if key not in self.raced:
                        self.raced.add(key)
                        # a concurrent writer lands an event first
                        super().write_many(namespace, {key: (_state_with_event(key), None)})
                        return super().write_many(
                            namespace, {k: v for k, v in items.items() if k != key}
                        )
                return super().write_many(namespace, items)

        def _state_with_event(key: str) -> bytes:
            from tessera.tiles import TileEngine

            engine = TileEngine(user_clicks)
            state = engine.new_state()
            engine.apply(state, T0, [100.0, "z"], f"other-{key}")
            return state.to_json()

        flaky = FlakyStore()
        result = OnlineWriter(repo, flaky).ingest("clicks", [click(1, "u1", T0 + 10, 1.0)])
        assert result.conflicts_retried == 1
        server = FeatureServer(repo, flaky)
        values = server.get_features(
            service, [{"user_id": "u1", "price": 1.0}], request_time=T0 + 20
        )[0]
        assert values["user_clicks__value_sum_1h"] == 101.0  # both writes survived

    def test_gives_up_after_max_retries(self, repo: Repository) -> None:
        class AlwaysConflict(InMemoryStore):
            def write_many(
                self, namespace: str, items: Mapping[str, tuple[bytes, int | None]]
            ) -> dict[str, int]:
                return {}

        with pytest.raises(ConflictError, match="gave up"):
            OnlineWriter(repo, AlwaysConflict(), max_retries=2).ingest(
                "clicks", [click(1, "u1", T0, 1.0)]
            )

    def test_snapshot_never_regresses(
        self, repo: Repository, store: Any, profile_rows: pd.DataFrame
    ) -> None:
        writer = OnlineWriter(repo, store)
        frame = coerce_frame(profile_rows, profiles)
        assert writer.materialize_snapshot(user_profile, frame) == 2
        writer.materialize_snapshot(user_profile, frame.iloc[[0]])  # older row replayed
        stored = json.loads(store.read(user_profile.namespace, ["u1"])["u1"].value)
        assert stored["values"]["score"] == 0.7


class TestParity:
    """Online reads equal offline training values at the same instants."""

    def test_stream_ingest_then_serve_matches_offline(
        self,
        repo: Repository,
        store: Any,
        click_events: list[dict[str, Any]],
        profile_rows: pd.DataFrame,
    ) -> None:
        writer = OnlineWriter(repo, store)
        frames = {
            "clicks": coerce_frame(pd.DataFrame(click_events), clicks),
            "profiles": coerce_frame(profile_rows, profiles),
        }
        writer.materialize_snapshot(
            user_profile, frames["profiles"][frames["profiles"]["_ts"] < T0]
        )
        server = FeatureServer(repo, store)
        served, spine_rows = [], []
        for i, event in enumerate(click_events):
            if i % 7 == 0:
                for user_id in ("u1", "u2"):
                    row = {"user_id": user_id, "price": 12.5}
                    t = event["ts"]
                    served.append(server.get_features(service, [row], request_time=t)[0])
                    spine_rows.append({**row, "timestamp": t})
            writer.ingest("clicks", [event])
        # profiles updated during the run are only visible offline after T0 + 1d; restrict
        # the comparison to requests before the second profile update.
        offline = records(
            get_historical_features(service, pd.DataFrame(spine_rows), sources=frames)[
                service.output_columns
            ]
        )
        compared = 0
        for online_row, offline_row, spine_row in zip(served, offline, spine_rows):
            if spine_row["timestamp"] > T0 + 86_400:
                online_row = {
                    k: v for k, v in online_row.items() if not k.startswith("user_profile")
                }
            for key, value in online_row.items():
                compared += 1
                if isinstance(value, float):
                    assert value == pytest.approx(offline_row[key], rel=1e-9), key
                else:
                    assert value == offline_row[key], key
        assert compared > 500

    def test_backfill_equals_incremental_ingest(
        self, repo: Repository, click_events: list[dict[str, Any]]
    ) -> None:
        frame = coerce_frame(pd.DataFrame(click_events), clicks)
        incremental = InMemoryStore()
        OnlineWriter(repo, incremental).ingest("clicks", click_events)
        backfilled = InMemoryStore()
        results = backfill(repo, backfilled, views=["user_clicks"], frames={"clicks": frame})
        assert results[0].entities == 2
        t = max(e["ts"] for e in click_events) + 1
        row = [{"user_id": "u1", "price": 1.0}, {"user_id": "u2", "price": 1.0}]
        a = FeatureServer(repo, incremental).get_features(service, row, request_time=t)
        b = FeatureServer(repo, backfilled).get_features(service, row, request_time=t)
        assert a == b

    def test_build_states_until(self, click_events: list[dict[str, Any]]) -> None:
        frame = coerce_frame(pd.DataFrame(click_events), clicks)
        states = build_states(user_clicks, frame, until=T0 + 3600)
        assert all(s.max_ts is None or s.max_ts < T0 + 3600 for s in states.values())


class TestServer:
    def test_missing_entity_and_request_fields(self, repo: Repository) -> None:
        server = FeatureServer(repo, InMemoryStore())
        with pytest.raises(SchemaError, match="user_id"):
            server.get_features(service, [{"price": 1.0}])
        with pytest.raises(SchemaError, match="price"):
            server.get_features("recs_v1", [{"user_id": "u1"}])

    def test_unknown_entity_gets_defaults(self, repo: Repository) -> None:
        out = FeatureServer(repo, InMemoryStore(), clock=lambda: T0).get_features(
            service, [{"user_id": "nobody", "price": 3.0}]
        )[0]
        assert out["user_clicks__events_count_10m"] == 0
        assert out["user_profile__score"] is None
        assert out["click_context__has_history"] is False

    def test_snapshot_ttl_online(self, repo: Repository, profile_rows: pd.DataFrame) -> None:
        store = InMemoryStore()
        OnlineWriter(repo, store).materialize_snapshot(
            user_profile, coerce_frame(profile_rows, profiles)
        )
        server = FeatureServer(repo, store)
        fresh = server.get_features(
            service, [{"user_id": "u2", "price": 1.0}], request_time=T0 - 4 * 86_400
        )
        stale = server.get_features(service, [{"user_id": "u2", "price": 1.0}], request_time=T0)
        assert fresh[0]["user_profile__score"] == 0.2
        assert stale[0]["user_profile__score"] is None

    def test_feature_logging(self, repo: Repository, tmp_path: Path) -> None:
        memory = MemoryFeatureLogger()
        FeatureServer(repo, InMemoryStore(), logger=memory, clock=lambda: T0).get_features(
            service, [{"user_id": "u1", "price": 2.0}, {"user_id": "u2", "price": 3.0}]
        )
        assert [r.entities for r in memory.rows] == [{"user_id": "u1"}, {"user_id": "u2"}]
        assert memory.rows[0].service_fingerprint == service.fingerprint
        jsonl = JsonlFeatureLogger(tmp_path / "log" / "features.jsonl")
        jsonl.log(memory.rows)
        rows = read_feature_log(tmp_path / "log" / "features.jsonl")
        assert rows[1]["request"] == {"price": 3.0}
        assert rows[0]["request_time"] == T0

    def test_batched_reads(self, repo: Repository) -> None:
        class CountingStore(InMemoryStore):
            def __init__(self) -> None:
                super().__init__()
                self.reads = 0

            def read(self, namespace: str, keys: Iterable[str]) -> dict[str, VersionedValue]:
                self.reads += 1
                return super().read(namespace, keys)

        store = CountingStore()
        rows = [{"user_id": f"u{i}", "price": 1.0} for i in range(50)]
        FeatureServer(repo, store, clock=lambda: T0).get_features(service, rows)
        assert store.reads == 2  # one per stateful view, not per row
